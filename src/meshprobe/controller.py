"""Persistent Blender process controller."""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Self, cast

from PIL import Image

from meshprobe.artifacts import ArtifactCache, JsonValue
from meshprobe.camera import camera_diagnostics, orbit_camera
from meshprobe.contact_sheet import (
    compose_contact_sheet,
    compose_side_by_side_comparison,
    contact_sheet_staging_path,
)
from meshprobe.models import (
    Bounds,
    Camera,
    CameraDiagnostics,
    Component,
    ContactSheetCallout,
    ContactSheetManifest,
    ContactSheetPanel,
    ContactSheetPanelSpec,
    CustomIllumination,
    EnvironmentMap,
    GraphicsPlatform,
    Illumination,
    IlluminationPreset,
    NormalizedGeometryArtifact,
    OccluderRemovalStep,
    OcclusionBlocker,
    OcclusionEvidence,
    OcclusionQueryResult,
    OrthographicProjection,
    PerspectiveProjection,
    PresetIllumination,
    Projection,
    RenderComparisonManifest,
    RenderManifest,
    RenderStyle,
    SceneManifest,
    SessionSnapshot,
)
from meshprobe.protocol import (
    Command,
    CommandEffect,
    ComponentDisplayCommand,
    ComponentFindCommand,
    ComponentInspectCommand,
    ComponentMarkCommand,
    ComponentOcclusionCommand,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
    SessionResetCommand,
    ViewFrameCommand,
    ViewMoveCommand,
    ViewOrbitCommand,
    ViewRotateCommand,
    ViewSetCommand,
    command_payload,
)
from meshprobe.selectors import ComponentIndex
from meshprobe.sources import SourceSnapshot, sha256_file, snapshot_source

__all__ = [
    "DEFAULT_WORKER_TIMEOUT_SECONDS",
    "BlenderController",
    "BlenderWorkerCrashed",
    "BlenderWorkerError",
    "BlenderWorkerTimeout",
    "sha256_file",
]

DEFAULT_WORKER_TIMEOUT_SECONDS = 180.0


def _default_cache_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if os.name == "nt" and local_app_data:
        return Path(local_app_data) / "MeshProbe" / "cache"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "meshprobe"
    return Path.home() / ".cache" / "meshprobe"


class BlenderWorkerError(RuntimeError):
    """A structured Blender worker error."""


class BlenderWorkerCrashed(BlenderWorkerError):
    """The Blender process stopped before producing a response."""


class BlenderWorkerTimeout(BlenderWorkerError):
    """The Blender process exceeded an operation deadline."""


class BlenderController:
    """Open a 3D model in a Blender worker, drive it, and render images.

    This is the Python-API entry point for in-process scripting (no daemon):
    it owns one factory-clean Blender worker process, speaks its
    line-oriented protocol, and exposes :meth:`open_scene`,
    :meth:`execute` (camera/display/mark commands), :meth:`render_image`,
    and :meth:`render_contact_sheet`. Typical usage::

        from meshprobe import BlenderController

        with BlenderController() as controller:
            manifest = controller.open_scene("model.glb")
            ...

    For CLI-style session persistence across separate process invocations
    (what the ``meshprobe`` command line tool uses), see
    ``meshprobe.client.MeshProbeClient``, which talks to a project-local
    daemon that owns a ``BlenderController`` on your behalf.
    """

    def __init__(
        self,
        executable: str | Path | None = None,
        timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
        artifact_cache_root: str | Path | None = None,
    ) -> None:
        env_blender = os.environ.get("MESHPROBE_BLENDER")
        configured = str(executable or env_blender or "blender")
        self.executable = Path(configured)
        # True only when neither --blender nor MESHPROBE_BLENDER named an
        # executable, i.e. we are relying on the bare "blender" PATH lookup --
        # the case Windows auto-discovery (start()) should kick in for. An
        # explicit override that fails to resolve should surface as-is rather
        # than being silently replaced by a different discovered install.
        self._auto_discovery_eligible = executable is None and env_blender is None
        self.timeout_seconds = timeout_seconds
        configured_cache = artifact_cache_root or os.environ.get("MESHPROBE_CACHE")
        self._artifact_cache = ArtifactCache(
            Path(configured_cache) if configured_cache else _default_cache_root()
        )
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._logs: list[str] = []
        self._worker_path = Path(__file__).with_name("blender") / "worker.py"
        self.ready_event: dict[str, Any] | None = None
        self.graphics: GraphicsPlatform | None = None
        self._source_path: Path | None = None
        self._source_sha256: str | None = None
        self._aspect_ratio: float = 1.0
        self._unit_scale: float = 1.0
        self._manifest: SceneManifest | None = None
        self._source_snapshot: SourceSnapshot | None = None
        self._accepted_commands: list[tuple[str, dict[str, object]]] = []

    @property
    def logs(self) -> tuple[str, ...]:
        return tuple(self._logs)

    def start(self) -> dict[str, Any]:
        if self._process is not None:
            raise BlenderWorkerError("Blender worker is already started")
        resolved = shutil.which(str(self.executable))
        if resolved is None and self._auto_discovery_eligible and sys.platform == "win32":
            from meshprobe.blender_discovery import discover_windows_blender

            discovered = discover_windows_blender()
            if discovered is not None:
                resolved = str(discovered)
        if resolved is None:
            raise BlenderWorkerError(
                f"Blender executable not found: {self.executable} "
                "(pass --blender /path/to/blender, or set MESHPROBE_BLENDER=/path/to/blender, "
                "to override)"
            )
        self.executable = Path(resolved)
        output_queue: queue.Queue[str | None] = queue.Queue()
        self._lines = output_queue
        self._process = subprocess.Popen(
            [
                str(self.executable),
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python",
                str(self._worker_path),
                "--",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._worker_environment(),
        )
        process = self._process
        thread = threading.Thread(
            target=self._read_output,
            args=(process, output_queue),
            name="meshprobe-blender-output",
            daemon=True,
        )
        self._reader_thread = thread
        thread.start()
        try:
            event = self._wait_for(lambda payload: payload.get("event") == "ready")
        except (BlenderWorkerCrashed, BlenderWorkerTimeout):
            self.close()
            raise
        if event.get("protocol_version") != 2:
            self.close()
            raise BlenderWorkerError(
                f"unsupported worker protocol: {event.get('protocol_version')}"
            )
        try:
            self.graphics = GraphicsPlatform.model_validate(event.get("graphics"))
        except ValueError as error:
            self.close()
            raise BlenderWorkerError(
                f"worker reported invalid graphics platform: {error}"
            ) from error
        self.ready_event = event
        return event

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        environment = os.environ.copy()
        try:
            kernel_release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
        except OSError:
            return environment
        if "microsoft" not in kernel_release.casefold() or not Path("/dev/dxg").exists():
            return environment
        environment.setdefault("GALLIUM_DRIVER", "d3d12")
        if shutil.which("nvidia-smi") is not None:
            environment.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
        return environment

    def request(self, action: str, **arguments: object) -> dict[str, Any]:
        process = self._require_process()
        if process.stdin is None:
            raise BlenderWorkerCrashed("Blender worker stdin is unavailable")
        request_id = uuid.uuid4().hex
        command = {"request_id": request_id, "op": action, **arguments}
        try:
            process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise BlenderWorkerCrashed(self._crash_message()) from error
        response = self._wait_for(lambda payload: payload.get("request_id") == request_id)
        if not response.get("ok"):
            worker_error = response.get("error", {})
            raise BlenderWorkerError(
                f"{worker_error.get('code', 'worker.error')}: {worker_error.get('message', '')}"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise BlenderWorkerError("worker returned a non-object result")
        return result

    def _request_with_timeout(
        self, action: str, timeout_seconds: float, **arguments: object
    ) -> dict[str, Any]:
        previous_timeout = self.timeout_seconds
        self.timeout_seconds = timeout_seconds
        try:
            return self.request(action, **arguments)
        finally:
            self.timeout_seconds = previous_timeout

    def open_scene(
        self, source_path: str | Path, *, aspect_ratio: float = 1.0, unit_scale: float = 1.0
    ) -> SceneManifest:
        source = Path(source_path).expanduser().resolve(strict=True)
        before = snapshot_source(source)
        self._source_path = None
        self._source_sha256 = None
        self._aspect_ratio = aspect_ratio
        self._unit_scale = unit_scale
        self._manifest = None
        self._source_snapshot = None
        self._accepted_commands.clear()
        try:
            result = self.request(
                "scene.open",
                source_path=str(source),
                source_sha256=before.sha256,
                aspect_ratio=aspect_ratio,
                unit_scale=unit_scale,
            )
        except (BlenderWorkerCrashed, BlenderWorkerTimeout):
            self.close()
            self.start()
            result = self.request(
                "scene.open",
                source_path=str(source),
                source_sha256=before.sha256,
                aspect_ratio=aspect_ratio,
                unit_scale=unit_scale,
            )
        after_import = snapshot_source(source)
        if before != after_import:
            raise BlenderWorkerError("source asset bundle changed during read-only import")
        manifest = SceneManifest.model_validate(result)
        if manifest.source_sha256 != before.sha256:
            raise BlenderWorkerError("worker source hash does not match controller source hash")
        if self.ready_event is not None:
            manifest = self._attach_normalized_geometry(manifest, source, before.sha256)
        after_normalization = snapshot_source(source)
        if before != after_normalization:
            raise BlenderWorkerError("source asset bundle changed during geometry normalization")
        self._source_path = source
        self._source_sha256 = manifest.source_sha256
        self._manifest = manifest
        self._source_snapshot = before
        return manifest

    def _attach_normalized_geometry(
        self,
        manifest: SceneManifest,
        source: Path,
        source_sha256: str,
    ) -> SceneManifest:
        importer_version = self._normalizer_version()
        parameters: dict[str, JsonValue] = {
            "apply_modifiers": True,
            "coordinate_system": "gltf_y_up",
            "format": "glb",
            "include_cameras": False,
            "include_lights": False,
            "units": "meter",
            "unit_scale": self._unit_scale,
        }

        def write_outputs(payload: Path) -> None:
            output = payload / "scene.glb"
            try:
                self.request("scene.export_normalized", output_path=str(output))
            except (BlenderWorkerCrashed, BlenderWorkerTimeout):
                self.close()
                self.start()
                self.request(
                    "scene.open",
                    source_path=str(source),
                    source_sha256=source_sha256,
                    aspect_ratio=self._aspect_ratio,
                    unit_scale=self._unit_scale,
                )
                self.request("scene.export_normalized", output_path=str(output))

        entry = self._artifact_cache.publish(
            source_sha256,
            importer_version,
            parameters,
            write_outputs,
        )
        output = entry.outputs[0]
        normalized = NormalizedGeometryArtifact(
            cache_key=entry.key,
            path=str(entry.output_path(output.relative_path)),
            sha256=output.sha256,
            bytes=output.bytes,
            importer_version=entry.importer_version,
            normalization_parameters=entry.normalization_parameters,
        )
        return manifest.model_copy(update={"normalized_geometry": normalized})

    def _normalizer_version(self) -> str:
        if self.ready_event is None:
            raise BlenderWorkerError("worker is not ready")
        blender_version = self.ready_event.get("blender_version")
        if not isinstance(blender_version, str) or not blender_version:
            raise BlenderWorkerError("worker did not report its Blender version")
        worker_path = Path(__file__).with_name("blender") / "worker.py"
        return f"blender-{blender_version}+meshprobe-normalizer-v1-{sha256_file(worker_path)[:16]}"

    def execute(self, command: Command) -> object:
        if command.effect is CommandEffect.UNDECLARED:
            raise BlenderWorkerError(f"command has no declared history effect: {command.op}")
        if command.effect is CommandEffect.HISTORY:
            raise BlenderWorkerError(f"{command.op} requires a durable session manager")
        if isinstance(command, SceneOpenCommand):
            return self.open_scene(
                command.source_path,
                aspect_ratio=command.aspect_ratio,
                unit_scale=command.unit_scale,
            )
        if isinstance(command, (ComponentFindCommand, ComponentInspectCommand)):
            if self._manifest is None:
                raise BlenderWorkerError("cannot inspect components before a scene is open")
            index = ComponentIndex(self._manifest)
            if isinstance(command, ComponentFindCommand):
                return [
                    component.model_dump(mode="json") for component in index.find(command.selector)
                ]
            return index.by_id(command.component_id).model_dump(mode="json")
        if isinstance(command, ViewFrameCommand):
            return self.frame_view(command)
        if isinstance(command, RenderImageCommand):
            return self.render_image(command)
        if isinstance(command, RenderContactSheetCommand):
            return self.render_contact_sheet(command)
        if isinstance(command, ComponentOcclusionCommand):
            try:
                return self.query_occlusion(command)
            except (BlenderWorkerCrashed, BlenderWorkerTimeout):
                self._recover_session()
                return self.query_occlusion(command)
        operation = command.op
        arguments = command_payload(command, exclude={"request_id", "op"})
        if isinstance(command, IlluminationSetCommand):
            arguments["illumination"] = self._cache_environment_map(
                command.illumination
            ).model_dump(mode="json")
        try:
            result = self.request(operation, **arguments)
        except (BlenderWorkerCrashed, BlenderWorkerTimeout):
            self._recover_session()
            result = self.request(operation, **arguments)
        if isinstance(command, SessionResetCommand):
            if "aspect_ratio" in command.model_fields_set:
                self._aspect_ratio = command.aspect_ratio
            self._accepted_commands.clear()
            snapshot = SessionSnapshot.model_validate(result)
            return {"reset": True, "state_sha256": snapshot.state_sha256}
        if command.effect is CommandEffect.STATE_MUTATION:
            self._accepted_commands.append((operation, arguments))
            snapshot = SessionSnapshot.model_validate(result)
            return self._state_operation_result(command, snapshot)
        return result

    def query_occlusion(self, command: ComponentOcclusionCommand) -> OcclusionQueryResult:
        if self._manifest is None:
            raise BlenderWorkerError("cannot query occlusion before a scene is open")
        focus_ids = tuple(dict.fromkeys(command.component_ids))
        known_ids = {component.id for component in self._manifest.components}
        unknown = set(focus_ids) - known_ids
        if unknown:
            raise BlenderWorkerError(f"unknown component ids: {sorted(unknown)}")
        snapshot_result = self.request("session.snapshot")
        snapshot = SessionSnapshot.model_validate(snapshot_result["session"])
        ranking = self.request(
            "component.occluders",
            component_ids=list(focus_ids),
            max_samples_per_component=command.max_samples_per_component,
            aspect_ratio=snapshot.camera_diagnostics.aspect_ratio,
        )
        evaluated_aspect = float(ranking.get("aspect_ratio", 0))
        if not math.isclose(
            evaluated_aspect,
            snapshot.camera_diagnostics.aspect_ratio,
            abs_tol=1e-12,
        ):
            raise BlenderWorkerError("worker evaluated occlusion with the wrong camera aspect")
        if ranking.get("projection") != snapshot.camera.projection.mode:
            raise BlenderWorkerError("worker evaluated occlusion with the wrong projection")

        def sample_count_field(name: str) -> int:
            value = ranking.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise BlenderWorkerError("worker returned invalid occlusion sample counts")
            return value

        sample_count = sample_count_field("sample_count")
        visible_samples = sample_count_field("visible_rays")
        occluded_samples = sample_count_field("occluded_rays")
        unresolved_samples = sample_count_field("unresolved_rays")
        if visible_samples + occluded_samples + unresolved_samples != sample_count:
            raise BlenderWorkerError("worker returned inconsistent occlusion sample counts")
        raw_query_diagnostics = ranking.get("camera_diagnostics")
        if raw_query_diagnostics is None:
            raise BlenderWorkerError("worker omitted focus-specific camera diagnostics")
        query_diagnostics = CameraDiagnostics.model_validate(raw_query_diagnostics)
        blockers = tuple(self._occlusion_blocker(item) for item in ranking.get("occluders", []))
        if sum(blocker.ray_hit_count for blocker in blockers) != occluded_samples:
            raise BlenderWorkerError("worker blocker hit counts do not equal occluded rays")
        return OcclusionQueryResult(
            focus_component_ids=focus_ids,
            camera=snapshot.camera,
            camera_diagnostics=query_diagnostics,
            aspect_ratio=evaluated_aspect,
            projection_status="projected" if sample_count else "not_projected",
            sample_count=sample_count,
            visible_sample_count=visible_samples,
            occluded_sample_count=occluded_samples,
            unresolved_sample_count=unresolved_samples,
            visible_fraction=visible_samples / sample_count if sample_count else 0.0,
            blockers=blockers,
            state_sha256=snapshot.state_sha256,
        )

    def _occlusion_blocker(self, raw: object) -> OcclusionBlocker:
        if not isinstance(raw, dict):
            raise BlenderWorkerError("worker returned an invalid occlusion blocker")
        component = self._component(str(raw["component_id"]))
        ray_hit_count = raw.get("blocked_rays")
        if type(ray_hit_count) is not int or ray_hit_count <= 0:
            raise BlenderWorkerError("worker returned an invalid blocker hit count")
        return OcclusionBlocker(
            component_id=component.id,
            display_name=component.display_name,
            component_path=component.path,
            ray_hit_count=ray_hit_count,
        )

    def frame_view(self, command: ViewFrameCommand) -> object:
        if self._manifest is None:
            raise BlenderWorkerError("cannot frame the camera before a scene is open")
        focus_ids = tuple(dict.fromkeys(command.focus_component_ids))
        known_ids = {component.id for component in self._manifest.components}
        unknown = set(focus_ids) - known_ids
        if unknown:
            raise BlenderWorkerError(f"unknown component ids: {sorted(unknown)}")
        bounds = self._focus_bounds(focus_ids)
        center, _ = self._bounds_center_span(bounds)
        projection, distance = self._frame_camera(
            command.projection,
            bounds,
            center,
            command.azimuth_degrees,
            command.elevation_degrees,
            command.roll_degrees,
            command.aspect_ratio,
            command.margin,
        )
        return self.execute(
            ViewOrbitCommand(
                request_id=command.request_id,
                op="view.orbit",
                target_mm=center,
                azimuth_degrees=command.azimuth_degrees,
                elevation_degrees=command.elevation_degrees,
                roll_degrees=command.roll_degrees,
                distance_mm=distance,
                projection=projection,
                focus_component_ids=focus_ids,
                aspect_ratio=command.aspect_ratio,
            )
        )

    @staticmethod
    def _frame_camera(
        projection: Projection,
        bounds: Bounds,
        target_mm: tuple[float, float, float],
        azimuth_degrees: float,
        elevation_degrees: float,
        roll_degrees: float,
        aspect_ratio: float,
        margin: float,
    ) -> tuple[Projection, float]:
        _, span = BlenderController._bounds_center_span(bounds)
        half_span = span / 2
        if isinstance(projection, OrthographicProjection):
            # Orthographic framing is set by scale_mm, not distance, so the distance only has
            # to keep the camera outside the bounds and in front of its near plane (below).
            framing_distance = span
            # scale_mm is the vertical extent; for portrait aspect ratios the horizontal
            # extent shrinks to scale_mm * aspect_ratio, so divide it back out to keep the
            # bounds framed (matches the contact-sheet orthographic panels).
            projection = projection.model_copy(
                update={"scale_mm": span * margin / min(aspect_ratio, 1.0)}
            )
        else:
            camera = orbit_camera(
                target_mm=target_mm,
                azimuth_degrees=azimuth_degrees,
                elevation_degrees=elevation_degrees,
                roll_degrees=roll_degrees,
                distance_mm=1.0,
                projection=projection,
            )
            diagnostics = camera_diagnostics(
                camera,
                target_mm=target_mm,
                aspect_ratio=aspect_ratio,
            )
            horizontal_fov = diagnostics.horizontal_fov_degrees
            vertical_fov = diagnostics.vertical_fov_degrees
            if horizontal_fov is None or vertical_fov is None:
                raise BlenderWorkerError("perspective camera did not report a field of view")
            horizontal_half_tan = math.tan(math.radians(horizontal_fov / 2))
            vertical_half_tan = math.tan(math.radians(vertical_fov / 2))
            corners = BlenderController._bounds_corners(bounds)
            offsets = [
                cast(
                    tuple[float, float, float],
                    tuple(corner[axis] - target_mm[axis] for axis in range(3)),
                )
                for corner in corners
            ]
            depths = [BlenderController._dot(offset, diagnostics.forward) for offset in offsets]
            framing_distance = max(
                max(
                    abs(BlenderController._dot(offset, diagnostics.right))
                    * margin
                    / horizontal_half_tan
                    - depth,
                    abs(BlenderController._dot(offset, diagnostics.up)) * margin / vertical_half_tan
                    - depth,
                )
                for offset, depth in zip(offsets, depths, strict=True)
            )
            # Keep the near face in front of the camera and behind any caller-supplied near
            # clip. Unlike the old bounding-sphere floor, this only accounts for the actual
            # closest corner, so depth behind the target no longer adds needless padding.
            distance = max(framing_distance, projection.near_clip_mm - min(depths), 1.0)
            required_far_clip_mm = distance + max(depths)
        if isinstance(projection, OrthographicProjection):
            distance = max(framing_distance, span, projection.near_clip_mm + half_span, 1.0)
            required_far_clip_mm = distance + span
        if projection.far_clip_mm <= required_far_clip_mm:
            projection = projection.model_copy(update={"far_clip_mm": required_far_clip_mm * 1.1})
        return projection, distance

    @staticmethod
    def _bounds_corners(bounds: Bounds) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            (x, y, z)
            for x in (bounds.minimum_mm[0], bounds.maximum_mm[0])
            for y in (bounds.minimum_mm[1], bounds.maximum_mm[1])
            for z in (bounds.minimum_mm[2], bounds.maximum_mm[2])
        )

    @staticmethod
    def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
        return sum(first * second for first, second in zip(left, right, strict=True))

    @staticmethod
    def _state_operation_result(command: Command, snapshot: SessionSnapshot) -> dict[str, object]:
        result: dict[str, object] = {"state_sha256": snapshot.state_sha256}
        if isinstance(
            command, (ViewSetCommand, ViewOrbitCommand, ViewMoveCommand, ViewRotateCommand)
        ):
            result["camera"] = snapshot.camera.model_dump(mode="json")
            result["camera_diagnostics"] = snapshot.camera_diagnostics.model_dump(mode="json")
            if not isinstance(command, (ViewMoveCommand, ViewRotateCommand)):
                return result
            if snapshot.camera_operation is None:
                raise BlenderWorkerError(
                    f"{command.op} response omitted its camera operation receipt"
                )
            result["camera_operation"] = snapshot.camera_operation.model_dump(mode="json")
            return result
        if isinstance(command, IlluminationSetCommand):
            result["illumination"] = snapshot.illumination.model_dump(mode="json")
            return result
        if isinstance(command, (ComponentDisplayCommand, ComponentMarkCommand)):
            component_ids = (
                snapshot.components
                if isinstance(command, ComponentDisplayCommand) and command.mode.value == "isolated"
                else command.component_ids
            )
            result["components"] = {
                component_id: snapshot.components[component_id].model_dump(mode="json")
                for component_id in component_ids
            }
            return result
        return result

    @property
    def worker_pid(self) -> int | None:
        """Return the renderer process id while the worker is alive."""

        process = self._process
        if process is None or process.poll() is not None:
            return None
        return process.pid

    def kill(self) -> None:
        """Immediately terminate the renderer without a shutdown checkpoint."""

        process = self._process
        self._process = None
        self.ready_event = None
        if process is None:
            return
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        reader_thread = self._reader_thread
        self._reader_thread = None
        if reader_thread is not None and reader_thread is not threading.current_thread():
            reader_thread.join(timeout=2)

    def _cache_environment_map(
        self,
        illumination: Illumination,
    ) -> Illumination:
        if not isinstance(illumination, CustomIllumination):
            return illumination
        environment = illumination.environment_map
        if environment is None:
            return illumination
        source = Path(environment.path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise BlenderWorkerError(f"environment map is not a file: {source}")
        before = sha256_file(source)
        if before != environment.sha256:
            raise BlenderWorkerError("environment map hash does not match its declaration")
        suffix = source.suffix.lower()
        if suffix not in {".hdr", ".exr"}:
            raise BlenderWorkerError("environment map must be an HDR or EXR image")

        def write_output(payload: Path) -> None:
            shutil.copyfile(source, payload / f"environment{suffix}")

        entry = self._artifact_cache.publish(
            environment.sha256,
            "meshprobe-environment-v1",
            {"extension": suffix, "projection": environment.projection},
            write_output,
        )
        if sha256_file(source) != before:
            raise BlenderWorkerError("environment map changed while it was cached")
        output = entry.outputs[0]
        cached_environment = EnvironmentMap(
            path=str(entry.output_path(output.relative_path)),
            sha256=output.sha256,
            strength=environment.strength,
            rotation_degrees=environment.rotation_degrees,
            projection=environment.projection,
        )
        return illumination.model_copy(update={"environment_map": cached_environment})

    def render_image(
        self,
        command: RenderImageCommand,
        *,
        evaluator_output_dir: str | Path | None = None,
    ) -> RenderManifest:
        if self._source_snapshot is None:
            raise BlenderWorkerError("cannot render before a scene is open")
        output = Path(command.output_path).expanduser().resolve()
        comparison_output: Path | None = None
        reference_image: Path | None = None
        if command.comparison is not None:
            comparison_output = Path(command.comparison.output_path).expanduser().resolve()
            reference_image = Path(command.comparison.reference_image_path).expanduser().resolve()
            if comparison_output.suffix.lower() != ".png":
                raise BlenderWorkerError("comparison output_path must end in .png")
            if not reference_image.is_file():
                raise BlenderWorkerError(f"reference image is not a file: {reference_image}")
            comparison_staging = contact_sheet_staging_path(comparison_output)
            comparison_paths = (output, reference_image, comparison_output, comparison_staging)
            if len(set(comparison_paths)) != len(comparison_paths):
                raise BlenderWorkerError(
                    "render, reference, comparison output, and comparison staging paths must differ"
                )
            try:
                with Image.open(reference_image) as image:
                    image.verify()
            except (OSError, ValueError) as error:
                raise BlenderWorkerError(
                    f"reference image is not decodable: {reference_image}"
                ) from error
        source_paths = {asset.path for asset in self._source_snapshot.assets}
        output_paths = self._render_output_paths(output, evaluator_output_dir)
        if comparison_output is not None:
            output_paths.update({comparison_output, contact_sheet_staging_path(comparison_output)})
        if source_paths.intersection(output_paths):
            raise BlenderWorkerError("render output must not overwrite a source asset")
        arguments = command.model_dump(
            mode="json", exclude={"request_id", "op", "comparison", "timeout_seconds"}
        )
        arguments["output_path"] = str(output)
        if evaluator_output_dir is not None:
            arguments["evaluator_output_dir"] = str(
                Path(evaluator_output_dir).expanduser().resolve()
            )
        try:
            result = self._request_with_timeout(command.op, command.timeout_seconds, **arguments)
        except BlenderWorkerTimeout:
            self._recover_session()
            raise
        except BlenderWorkerCrashed:
            self._recover_session()
            try:
                result = self._request_with_timeout(
                    command.op, command.timeout_seconds, **arguments
                )
            except BlenderWorkerTimeout:
                self._recover_session()
                raise
        after = snapshot_source(self._source_snapshot.assets[0].path)
        if after != self._source_snapshot:
            raise BlenderWorkerError("source asset bundle changed during render")
        manifest = RenderManifest.model_validate(result)
        if manifest.source_sha256 != self._source_sha256:
            raise BlenderWorkerError("render manifest source hash does not match the open scene")
        self._verify_render_artifacts(manifest)
        if comparison_output is not None and reference_image is not None:
            try:
                artifact, reference, render_placement, reference_placement = (
                    compose_side_by_side_comparison(
                        Path(manifest.color.path), reference_image, comparison_output
                    )
                )
            except (OSError, ValueError) as error:
                raise BlenderWorkerError(
                    f"could not compose reference comparison: {error}"
                ) from error
            manifest = manifest.model_copy(
                update={
                    "comparison": RenderComparisonManifest(
                        mode=(
                            command.comparison.mode
                            if command.comparison is not None
                            else "side_by_side"
                        ),
                        artifact=artifact,
                        reference=reference,
                        render_placement=render_placement,
                        reference_placement=reference_placement,
                    )
                }
            )
        self.request(
            "render.style",
            style=manifest.style.value,
            shaded_edges=manifest.shaded_edges.model_dump(mode="json"),
        )
        return manifest

    @staticmethod
    def _verify_render_artifacts(manifest: RenderManifest) -> None:
        artifacts = [manifest.color]
        if manifest.evaluator is not None:
            artifacts.extend(
                (
                    manifest.evaluator.multilayer,
                    manifest.evaluator.component_ids,
                    manifest.evaluator.highlighted,
                )
            )
        for artifact_record in artifacts:
            path = Path(artifact_record.path)
            if not path.is_file():
                raise BlenderWorkerError(f"render artifact is missing: {path}")
            if (
                path.stat().st_size != artifact_record.bytes
                or sha256_file(path) != artifact_record.sha256
            ):
                raise BlenderWorkerError(f"render artifact digest mismatch: {path}")

    def render_contact_sheet(
        self,
        command: RenderContactSheetCommand,
        *,
        evaluator_output_dir: str | Path | None = None,
    ) -> ContactSheetManifest:
        if self._manifest is None or self._source_snapshot is None:
            raise BlenderWorkerError("cannot render before a scene is open")
        focus_ids = tuple(dict.fromkeys(command.focus_component_ids))
        known_ids = {component.id for component in self._manifest.components}
        unknown = set(focus_ids) - known_ids
        if unknown:
            raise BlenderWorkerError(f"unknown component ids: {sorted(unknown)}")
        output = Path(command.output_path).expanduser().resolve()
        if output.suffix.lower() != ".png":
            raise BlenderWorkerError("contact-sheet output_path must end in .png")
        panel_dir = output.parent / f"{output.stem}_panels"
        panel_count = 9
        if command.recipe == "orbit_sweep":
            if command.orbit_sweep is None:
                raise BlenderWorkerError("orbit sweep configuration is missing")
            panel_count = (
                len(command.orbit_sweep.azimuth_degrees)
                * len(command.orbit_sweep.elevation_degrees)
                * len(command.orbit_sweep.roll_degrees)
            )
        panel_paths = tuple(panel_dir / f"panel-{index}.png" for index in range(1, panel_count + 1))
        source_paths = {asset.path for asset in self._source_snapshot.assets}
        output_paths = {output, contact_sheet_staging_path(output), *panel_paths}
        evaluator_root = None
        if evaluator_output_dir is not None:
            evaluator_root = Path(evaluator_output_dir).expanduser().resolve()
            for index, panel_path in enumerate(panel_paths, start=1):
                output_paths.update(
                    self._render_output_paths(
                        panel_path,
                        evaluator_root / f"panel-{index}",
                    )
                )
        if source_paths.intersection(output_paths):
            raise BlenderWorkerError("contact-sheet output must not overwrite a source asset")
        if command.recipe == "custom_3x3":
            return self._render_custom_contact_sheet(
                command,
                output,
                panel_paths,
                evaluator_root,
                focus_ids,
            )
        if command.recipe == "orbit_sweep":
            return self._render_orbit_sweep(
                command,
                output,
                panel_paths,
                evaluator_root,
                focus_ids,
            )

        accepted_commands = list(self._accepted_commands)
        mark_colors = self._focus_mark_colors(accepted_commands, focus_ids)
        panels: list[ContactSheetPanel] = []
        removed_occluders: tuple[str, ...] = ()
        occlusion: OcclusionEvidence | None = None
        try:
            self.request("session.reset")
            self.request("illumination.set", illumination={"preset": "neutral_studio"})
            scene_center, scene_span = self._bounds_center_span(self._manifest.root_bounds)
            panel_aspect = command.panel_width / command.panel_height
            self._set_orbit(
                scene_center,
                scene_span,
                45,
                30,
                PerspectiveProjection(),
                panel_aspect,
                focus_ids,
            )
            panels.append(
                self._render_contact_panel(
                    command,
                    panel_paths[0],
                    1,
                    "Natural isometric",
                    evaluator_root,
                )
            )

            focus_bounds = self._focus_bounds(focus_ids)
            focus_center, focus_span = self._bounds_center_span(focus_bounds)
            focus_projection, focus_distance = self._focused_context_camera(
                self._manifest.root_bounds,
                focus_center,
                focus_span,
                panel_aspect,
            )
            self._set_orbit(
                focus_center,
                focus_span * 4.0,
                45,
                30,
                focus_projection,
                panel_aspect,
                focus_ids,
                minimum_distance_mm=focus_distance,
            )
            self._mark_focus_components(focus_ids, "highlighted", mark_colors)
            panels.append(
                self._render_contact_panel(
                    command,
                    panel_paths[1],
                    2,
                    "Focused in context",
                    evaluator_root,
                )
            )

            occlusion = self._remove_occluders(
                command,
                focus_ids,
                camera=panels[-1].render.session.camera,
            )
            removed_occluders = tuple(step.component_id for step in occlusion.steps)
            third_caption = self._removed_occluder_caption(occlusion.steps)
            if not removed_occluders:
                self._set_orbit(
                    focus_center,
                    focus_span * 4.0,
                    -45,
                    20,
                    focus_projection,
                    panel_aspect,
                    focus_ids,
                    minimum_distance_mm=focus_distance,
                )
                third_caption = self._empty_occlusion_caption(occlusion.stop_reason)
            panels.append(
                self._render_contact_panel(
                    command,
                    panel_paths[2],
                    3,
                    third_caption,
                    evaluator_root,
                )
            )

            self.request("illumination.set", illumination={"preset": "flat_diagnostic"})
            self.request("component.display", component_ids=list(focus_ids), mode="isolated")
            directions = (
                (0.0, 0.0, "+X"),
                (180.0, 0.0, "-X"),
                (90.0, 0.0, "+Y"),
                (-90.0, 0.0, "-Y"),
                (0.0, 90.0, "+Z"),
                (0.0, -90.0, "-Z"),
            )
            for panel_index, (azimuth, elevation, caption) in enumerate(directions, start=4):
                projection = OrthographicProjection(
                    scale_mm=focus_span * 1.2 / min(panel_aspect, 1.0)
                )
                self._set_orbit(
                    focus_center,
                    focus_span,
                    azimuth,
                    elevation,
                    projection,
                    panel_aspect,
                    focus_ids,
                )
                panels.append(
                    self._render_contact_panel(
                        command,
                        panel_paths[panel_index - 1],
                        panel_index,
                        caption,
                        evaluator_root,
                    )
                )
        finally:
            self.request("session.reset")
            for operation, arguments in accepted_commands:
                self.request(operation, **arguments)

        after = snapshot_source(self._source_snapshot.assets[0].path)
        if after != self._source_snapshot:
            raise BlenderWorkerError("source asset bundle changed during contact-sheet render")
        captions = tuple(
            (Path(panel.render.color.path), panel.caption, panel.callouts) for panel in panels
        )
        sheet = compose_contact_sheet(captions, output, command.panel_width, command.panel_height)
        if occlusion is None:
            raise BlenderWorkerError("contact-sheet occlusion analysis did not run")
        return ContactSheetManifest(
            recipe=command.recipe,
            focus_component_ids=focus_ids,
            removed_occluder_ids=removed_occluders,
            occlusion=occlusion,
            sheet=sheet,
            panels=tuple(panels),
        )

    def _render_orbit_sweep(
        self,
        command: RenderContactSheetCommand,
        output: Path,
        panel_paths: tuple[Path, ...],
        evaluator_root: Path | None,
        focus_ids: tuple[str, ...],
    ) -> ContactSheetManifest:
        if self._manifest is None or self._source_snapshot is None or command.orbit_sweep is None:
            raise BlenderWorkerError("cannot render orbit sweep before a scene is open")
        sweep = command.orbit_sweep
        bounds = self._manifest.root_bounds
        if sweep.target == "focus":
            bounds = self._focus_bounds(focus_ids)
        target, span = self._bounds_center_span(bounds)
        aspect_ratio = command.panel_width / command.panel_height
        accepted_commands = list(self._accepted_commands)
        panels: list[ContactSheetPanel] = []
        try:
            for index, (azimuth, elevation, roll) in enumerate(
                (
                    (azimuth, elevation, roll)
                    for azimuth in sweep.azimuth_degrees
                    for elevation in sweep.elevation_degrees
                    for roll in sweep.roll_degrees
                ),
                start=1,
            ):
                self._set_orbit(
                    target,
                    span,
                    azimuth,
                    elevation,
                    sweep.projection,
                    aspect_ratio,
                    focus_ids,
                    roll_degrees=roll,
                )
                projection = sweep.projection
                projection_label = (
                    f"perspective {projection.focal_length_mm:g}mm"
                    if isinstance(projection, PerspectiveProjection)
                    else f"orthographic {projection.scale_mm:g}mm"
                )
                panels.append(
                    self._render_contact_panel(
                        command,
                        panel_paths[index - 1],
                        index,
                        (
                            f"azimuth {azimuth:g}°, elevation {elevation:g}°, roll {roll:g}° "
                            f"[{projection_label}]"
                        ),
                        evaluator_root,
                    )
                )
        finally:
            self.request("session.reset")
            for operation, arguments in accepted_commands:
                self.request(operation, **arguments)
        after = snapshot_source(self._source_snapshot.assets[0].path)
        if after != self._source_snapshot:
            raise BlenderWorkerError("source asset bundle changed during orbit sweep render")
        composed_panels = tuple(
            (Path(panel.render.color.path), panel.caption, panel.callouts) for panel in panels
        )
        sheet = compose_contact_sheet(
            composed_panels,
            output,
            command.panel_width,
            command.panel_height,
        )
        return ContactSheetManifest(
            recipe="orbit_sweep",
            focus_component_ids=focus_ids,
            sheet=sheet,
            panels=tuple(panels),
        )

    def _render_custom_contact_sheet(
        self,
        command: RenderContactSheetCommand,
        output: Path,
        panel_paths: tuple[Path, ...],
        evaluator_root: Path | None,
        focus_ids: tuple[str, ...],
    ) -> ContactSheetManifest:
        if self._manifest is None or self._source_snapshot is None:
            raise BlenderWorkerError("cannot render before a scene is open")
        accepted_commands = list(self._accepted_commands)
        mark_colors = self._focus_mark_colors(accepted_commands, focus_ids)
        panels: list[ContactSheetPanel] = []
        aspect_ratio = command.panel_width / command.panel_height
        try:
            for index, (spec, panel_path) in enumerate(
                zip(command.panels, panel_paths, strict=True),
                start=1,
            ):
                self.request("session.reset")
                illumination = spec.illumination or PresetIllumination(
                    preset=IlluminationPreset.NEUTRAL_STUDIO
                )
                illumination = self._cache_environment_map(illumination)
                self.request(
                    "illumination.set",
                    illumination=illumination.model_dump(mode="json"),
                )
                if spec.display == "isolated":
                    self.request(
                        "component.display",
                        component_ids=list(focus_ids),
                        mode="isolated",
                    )
                if spec.mark.value != "unmarked":
                    self._mark_focus_components(focus_ids, spec.mark.value, mark_colors)
                if spec.camera is not None:
                    self.request(
                        "view.set",
                        camera=spec.camera.model_dump(mode="json"),
                        focus_component_ids=list(focus_ids),
                        aspect_ratio=aspect_ratio,
                    )
                else:
                    self._set_custom_orbit(spec, focus_ids, aspect_ratio)
                panels.append(
                    self._render_contact_panel(
                        command,
                        panel_path,
                        index,
                        self._custom_panel_caption(spec, illumination),
                        evaluator_root,
                        experiment=spec.experiment,
                    )
                )
        finally:
            self.request("session.reset")
            for operation, arguments in accepted_commands:
                self.request(operation, **arguments)

        after = snapshot_source(self._source_snapshot.assets[0].path)
        if after != self._source_snapshot:
            raise BlenderWorkerError("source asset bundle changed during contact-sheet render")
        composed_panels = tuple(
            (Path(panel.render.color.path), panel.caption, panel.callouts) for panel in panels
        )
        sheet = compose_contact_sheet(
            composed_panels,
            output,
            command.panel_width,
            command.panel_height,
        )
        return ContactSheetManifest(
            recipe=command.recipe,
            focus_component_ids=focus_ids,
            sheet=sheet,
            panels=tuple(panels),
        )

    @staticmethod
    def _focus_mark_colors(
        accepted_commands: list[tuple[str, dict[str, object]]],
        focus_ids: tuple[str, ...],
    ) -> dict[str, str | None]:
        colors = dict.fromkeys(focus_ids)
        for operation, arguments in accepted_commands:
            if operation != "component.mark":
                continue
            component_ids = arguments.get("component_ids")
            if not isinstance(component_ids, list | tuple):
                continue
            color = arguments.get("color")
            if color is not None and not isinstance(color, str):
                continue
            if arguments.get("mode") == "unmarked":
                color = None
            for component_id in component_ids:
                if isinstance(component_id, str) and component_id in colors:
                    colors[component_id] = color
        return colors

    def _mark_focus_components(
        self,
        focus_ids: tuple[str, ...],
        mode: str,
        colors: dict[str, str | None],
    ) -> None:
        groups: dict[str | None, list[str]] = {}
        for component_id in focus_ids:
            groups.setdefault(colors[component_id], []).append(component_id)
        for color, component_ids in groups.items():
            arguments: dict[str, object] = {
                "component_ids": component_ids,
                "mode": mode,
            }
            if color is not None:
                arguments["color"] = color
            self.request("component.mark", **arguments)

    def _set_custom_orbit(
        self,
        spec: ContactSheetPanelSpec,
        focus_ids: tuple[str, ...],
        aspect_ratio: float,
    ) -> None:
        if self._manifest is None or spec.orbit is None:
            raise BlenderWorkerError("custom orbit is not available")
        orbit = spec.orbit
        bounds = (
            self._focus_bounds(focus_ids) if orbit.target == "focus" else self._manifest.root_bounds
        )
        target, _ = self._bounds_center_span(bounds)
        distance = orbit.distance_mm
        if spec.experiment == "dolly_zoom":
            projection = orbit.projection
            if not isinstance(projection, PerspectiveProjection):
                raise BlenderWorkerError("dolly_zoom requires perspective projection")
            if orbit.reference_focal_length_mm is None or orbit.reference_distance_mm is None:
                raise BlenderWorkerError("dolly_zoom reference is incomplete")
            distance = (
                orbit.reference_distance_mm
                * projection.focal_length_mm
                / orbit.reference_focal_length_mm
            )
        self.request(
            "view.orbit",
            target_mm=list(target),
            azimuth_degrees=orbit.azimuth_degrees,
            elevation_degrees=orbit.elevation_degrees,
            roll_degrees=orbit.roll_degrees,
            distance_mm=distance,
            projection=orbit.projection.model_dump(mode="json"),
            focus_component_ids=list(focus_ids),
            aspect_ratio=aspect_ratio,
        )

    @staticmethod
    def _custom_panel_caption(
        spec: ContactSheetPanelSpec,
        illumination: Illumination,
    ) -> str:
        if spec.camera is not None:
            projection = spec.camera.projection
        elif spec.orbit is not None:
            projection = spec.orbit.projection
        else:
            raise BlenderWorkerError("custom panel has no declared view")
        projection_label = (
            f"perspective {projection.focal_length_mm:g}mm"
            if isinstance(projection, PerspectiveProjection)
            else f"orthographic {projection.scale_mm:g}mm"
        )
        illumination_label = illumination.preset
        properties = [projection_label, str(illumination_label)]
        if spec.experiment != "declared":
            properties.append(spec.experiment)
        return f"{spec.caption} [{'; '.join(properties)}]"[:256]

    def _remove_occluders(
        self,
        command: RenderContactSheetCommand,
        focus_ids: tuple[str, ...],
        *,
        camera: Camera,
    ) -> OcclusionEvidence:
        width, height = self._visibility_dimensions(command.panel_width, command.panel_height)

        def measure() -> dict[str, Any]:
            return self.request(
                "component.visibility",
                component_ids=list(focus_ids),
                width=width,
                height=height,
                graphics_policy=command.graphics_policy.value,
            )

        initial = measure()
        before = float(initial["visible_fraction"])
        after = before
        visible_pixels_before = int(initial["visible_pixels"])
        visible_pixels_after = visible_pixels_before
        isolated_pixels = int(initial["isolated_pixels"])
        steps: list[OccluderRemovalStep] = []
        sample_count = 0
        stop_reason: Literal[
            "threshold_met_initial",
            "threshold_met",
            "budget_exhausted",
            "no_blockers",
            "focus_not_projected",
        ]
        if not initial.get("projected"):
            stop_reason = "focus_not_projected"
        elif before >= command.visibility_threshold:
            stop_reason = "threshold_met_initial"
        else:
            stop_reason = "budget_exhausted"
            for _ in range(command.occluder_budget):
                ranking = self.request(
                    "component.occluders",
                    component_ids=list(focus_ids),
                    aspect_ratio=command.panel_width / command.panel_height,
                )
                sample_count = max(sample_count, int(ranking.get("sample_count", 0)))
                ranked = [
                    item
                    for item in ranking.get("occluders", [])
                    if item["component_id"] not in {step.component_id for step in steps}
                ]
                if not ranked:
                    stop_reason = "no_blockers"
                    break
                blocker = ranked[0]
                component_id = str(blocker["component_id"])
                self.request(
                    "component.display",
                    component_ids=[component_id],
                    mode="hidden",
                )
                measured = measure()
                after = float(measured["visible_fraction"])
                visible_pixels_after = int(measured["visible_pixels"])
                isolated_pixels = int(measured["isolated_pixels"])
                component = self._component(component_id)
                steps.append(
                    OccluderRemovalStep(
                        component_id=component_id,
                        display_name=component.display_name,
                        component_path=component.path,
                        ray_hit_count=int(blocker["blocked_rays"]),
                        visible_pixel_count_after=visible_pixels_after,
                        visible_fraction_after=after,
                    )
                )
                if after >= command.visibility_threshold:
                    stop_reason = "threshold_met"
                    break
        return OcclusionEvidence(
            camera=camera,
            camera_source="generated_focus_context",
            visibility_width_px=width,
            visibility_height_px=height,
            visibility_threshold=command.visibility_threshold,
            removal_budget=command.occluder_budget,
            sample_count=sample_count,
            visible_pixel_count_before=visible_pixels_before,
            visible_pixel_count_after=visible_pixels_after,
            isolated_pixel_count=isolated_pixels,
            visible_fraction_before=before,
            visible_fraction_after=after,
            steps=tuple(steps),
            stop_reason=stop_reason,
        )

    def _component(self, component_id: str) -> Component:
        if self._manifest is None:
            raise BlenderWorkerError("cannot resolve a component before a scene is open")
        return ComponentIndex(self._manifest).by_id(component_id)

    @staticmethod
    def _removed_occluder_caption(steps: tuple[OccluderRemovalStep, ...]) -> str:
        if not steps:
            return "No occluders found"
        return "Occluders removed:\n" + "\n".join(
            f"{index}. {step.display_name}" for index, step in enumerate(steps, 1)
        )

    @staticmethod
    def _empty_occlusion_caption(
        stop_reason: Literal[
            "threshold_met_initial",
            "threshold_met",
            "budget_exhausted",
            "no_blockers",
            "focus_not_projected",
        ],
    ) -> str:
        detail = {
            "threshold_met_initial": "visibility threshold already met",
            "threshold_met": "visibility threshold met",
            "budget_exhausted": "no occluders removed within budget",
            "no_blockers": "no blocking components ranked",
            "focus_not_projected": "focus not projected",
        }[stop_reason]
        return f"Alternate focused context; previous focused view: {detail}"

    @staticmethod
    def _focused_context_camera(
        scene_bounds: Bounds,
        focus_center: tuple[float, float, float],
        focus_span: float,
        aspect_ratio: float,
    ) -> tuple[PerspectiveProjection, float]:
        minimum = scene_bounds.minimum_mm
        maximum = scene_bounds.maximum_mm
        context_radius = math.sqrt(
            sum(
                max(abs(low - center), abs(high - center)) ** 2
                for low, high, center in zip(minimum, maximum, focus_center, strict=True)
            )
        )
        minimum_distance = max(context_radius * 1.05, 1.0)
        framed_span = focus_span * 4.0
        projection = PerspectiveProjection()
        framing_fov = min(
            projection.horizontal_fov_degrees(aspect_ratio),
            projection.vertical_fov_degrees(aspect_ratio),
        )
        framing_distance = (framed_span / 2) / math.tan(math.radians(framing_fov / 2)) * 1.25
        camera_distance = max(framing_distance, minimum_distance)
        if camera_distance > framing_distance:
            projection = projection.model_copy(
                update={
                    "focal_length_mm": (
                        projection.focal_length_mm * camera_distance / framing_distance
                    )
                }
            )
        required_far_clip_mm = camera_distance + context_radius * 1.05
        if projection.far_clip_mm <= required_far_clip_mm:
            projection = projection.model_copy(update={"far_clip_mm": required_far_clip_mm * 1.1})
        return projection, camera_distance

    @staticmethod
    def _visibility_dimensions(width: int, height: int) -> tuple[int, int]:
        scale = min(1.0, 256 / max(width, height))
        return max(4, round(width * scale)), max(4, round(height * scale))

    def _render_contact_panel(
        self,
        command: RenderContactSheetCommand,
        output_path: Path,
        index: int,
        caption: str,
        evaluator_root: Path | None,
        experiment: Literal["declared", "fixed_pose_focal_study", "dolly_zoom"] = "declared",
    ) -> ContactSheetPanel:
        evaluator_dir = None
        if evaluator_root is not None:
            evaluator_dir = str(evaluator_root / f"panel-{index}")
        result = self.request(
            "render.image",
            output_path=str(output_path),
            width=command.panel_width,
            height=command.panel_height,
            samples=command.samples,
            engine=command.engine.value,
            style=RenderStyle.SHADED.value,
            graphics_policy=command.graphics_policy.value,
            evaluator_output_dir=evaluator_dir,
        )
        render = RenderManifest.model_validate(result)
        self._verify_render_artifacts(render)
        return ContactSheetPanel(
            index=index,
            caption=caption,
            render=render,
            callouts=self._contact_panel_callouts(render, command.focus_component_ids),
            experiment=experiment,
        )

    def _contact_panel_callouts(
        self,
        render: RenderManifest,
        focus_component_ids: tuple[str, ...],
    ) -> tuple[ContactSheetCallout, ...]:
        if self._manifest is None:
            raise BlenderWorkerError("cannot build callouts before a scene is open")
        labels = {component.id: component.display_name for component in self._manifest.components}
        callouts: list[ContactSheetCallout] = []
        for number, component_id in enumerate(dict.fromkeys(focus_component_ids), start=1):
            bounds = render.session.camera_diagnostics.projected_bounds.get(component_id)
            if (
                bounds is None
                or bounds.projection_status != "in_front"
                or bounds.minimum_image_xy is None
                or bounds.maximum_image_xy is None
            ):
                continue
            center = (
                (bounds.minimum_image_xy[0] + bounds.maximum_image_xy[0]) / 2,
                (bounds.minimum_image_xy[1] + bounds.maximum_image_xy[1]) / 2,
            )
            if any(value < 0 or value > 1 for value in center):
                continue
            callouts.append(
                ContactSheetCallout(
                    number=number,
                    component_id=component_id,
                    label=labels[component_id],
                    image_xy=center,
                )
            )
        return tuple(callouts)

    @staticmethod
    def _render_output_paths(
        output: Path,
        evaluator_output_dir: str | Path | None,
    ) -> set[Path]:
        paths = {output}
        if evaluator_output_dir is None:
            return paths
        evaluator_dir = Path(evaluator_output_dir).expanduser().resolve()
        paths.update(
            {
                evaluator_dir / f"{output.stem}.passes.exr",
                evaluator_dir / f"{output.stem}.components.png",
                evaluator_dir / f"{output.stem}.highlighted.png",
            }
        )
        return paths

    def _set_orbit(
        self,
        target: tuple[float, float, float],
        span: float,
        azimuth: float,
        elevation: float,
        projection: Projection,
        aspect_ratio: float,
        focus_component_ids: tuple[str, ...] = (),
        minimum_distance_mm: float = 100.0,
        roll_degrees: float = 0.0,
    ) -> None:
        distance = max(span * 2, minimum_distance_mm)
        if isinstance(projection, PerspectiveProjection):
            framing_fov = min(
                projection.horizontal_fov_degrees(aspect_ratio),
                projection.vertical_fov_degrees(aspect_ratio),
            )
            distance = max(
                (span / 2) / math.tan(math.radians(framing_fov / 2)) * 1.25,
                minimum_distance_mm,
            )
            required_far_clip_mm = distance + span
            if projection.far_clip_mm <= required_far_clip_mm:
                projection = projection.model_copy(
                    update={"far_clip_mm": required_far_clip_mm * 1.1}
                )
        self.request(
            "view.orbit",
            target_mm=list(target),
            azimuth_degrees=azimuth,
            elevation_degrees=elevation,
            roll_degrees=roll_degrees,
            distance_mm=distance,
            projection=projection.model_dump(mode="json"),
            aspect_ratio=aspect_ratio,
            focus_component_ids=list(focus_component_ids),
        )

    def _focus_bounds(self, focus_ids: tuple[str, ...]) -> Bounds:
        if self._manifest is None:
            raise BlenderWorkerError("cannot inspect bounds before a scene is open")
        selected = [
            component.world_bounds
            for component in self._manifest.components
            if component.id in focus_ids
        ]
        minimum = (
            min(bounds.minimum_mm[0] for bounds in selected),
            min(bounds.minimum_mm[1] for bounds in selected),
            min(bounds.minimum_mm[2] for bounds in selected),
        )
        maximum = (
            max(bounds.maximum_mm[0] for bounds in selected),
            max(bounds.maximum_mm[1] for bounds in selected),
            max(bounds.maximum_mm[2] for bounds in selected),
        )
        return Bounds(minimum_mm=minimum, maximum_mm=maximum)

    @staticmethod
    def _bounds_center_span(bounds: Bounds) -> tuple[tuple[float, float, float], float]:
        minimum = bounds.minimum_mm
        maximum = bounds.maximum_mm
        center = (
            (minimum[0] + maximum[0]) / 2,
            (minimum[1] + maximum[1]) / 2,
            (minimum[2] + maximum[2]) / 2,
        )
        diagonal = math.sqrt(
            sum((high - low) ** 2 for low, high in zip(minimum, maximum, strict=True))
        )
        return center, max(diagonal, 1.0)

    def _recover_session(self) -> None:
        if self._source_path is None or self._source_sha256 is None:
            raise BlenderWorkerCrashed("cannot recover a worker before a scene is open")
        source_path = self._source_path
        source_sha256 = self._source_sha256
        aspect_ratio = self._aspect_ratio
        accepted_commands = list(self._accepted_commands)
        self.close()
        self.start()
        reopened = self.open_scene(
            source_path, aspect_ratio=aspect_ratio, unit_scale=self._unit_scale
        )
        if reopened.source_sha256 != source_sha256:
            self.close()
            self._source_path = None
            self._source_sha256 = None
            self._manifest = None
            self._source_snapshot = None
            raise BlenderWorkerError("source asset bundle changed since the session opened")
        self._accepted_commands = accepted_commands
        for operation, arguments in accepted_commands:
            self.request(operation, **arguments)

    def close(self) -> None:
        process = self._process
        self._process = None
        self.ready_event = None
        self.graphics = None
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(
                    json.dumps({"request_id": uuid.uuid4().hex, "op": "session.shutdown"}) + "\n"
                )
                process.stdin.flush()
                process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        reader_thread = self._reader_thread
        self._reader_thread = None
        if reader_thread is not None and reader_thread is not threading.current_thread():
            reader_thread.join(timeout=2)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _read_output(
        self,
        process: subprocess.Popen[str],
        output_queue: queue.Queue[str | None],
    ) -> None:
        if process.stdout is None:
            output_queue.put(None)
            return
        for line in process.stdout:
            output_queue.put(line.rstrip("\n"))
        output_queue.put(None)

    def _wait_for(self, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BlenderWorkerTimeout(
                    f"Blender worker did not respond within {self.timeout_seconds:g} seconds"
                )
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as error:
                raise BlenderWorkerTimeout(
                    f"Blender worker did not respond within {self.timeout_seconds:g} seconds"
                ) from error
            if line is None:
                raise BlenderWorkerCrashed(self._crash_message())
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._logs.append(line)
                continue
            if isinstance(payload, dict) and predicate(payload):
                return payload
            self._logs.append(line)

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise BlenderWorkerError("Blender worker is not started")
        if self._process.poll() is not None:
            raise BlenderWorkerCrashed(self._crash_message())
        return self._process

    def _crash_message(self) -> str:
        process = self._process
        return_code = None
        if process is not None:
            return_code = process.poll()
            if return_code is None:
                with suppress(subprocess.TimeoutExpired):
                    return_code = process.wait(timeout=1)
        recent_logs = "\n".join(self._logs[-20:])
        rendered_code = str(return_code)
        hint = ""
        if return_code is not None and return_code & 0xFFFFFFFF == 0xC0000409:
            rendered_code = f"{return_code} (0x{return_code & 0xFFFFFFFF:08X})"
            hint = (
                " Windows exception 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN). On NVIDIA, "
                "this can follow a driver reset/TDR; check nvlddmkm events, free VRAM or close "
                "other GPU clients, then retry."
            )
        return (
            f"Blender worker exited with code {rendered_code}.{hint} Recent output:\n{recent_logs}"
        )
