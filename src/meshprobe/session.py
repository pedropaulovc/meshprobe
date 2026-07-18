"""In-memory visual state with reset and replay-friendly hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from meshprobe.camera import camera_diagnostics, framed_default_camera
from meshprobe.models import (
    Camera,
    CameraPoseFrame,
    ComponentVisualState,
    DisplayMode,
    Illumination,
    MarkMode,
    SceneManifest,
    SessionSnapshot,
    SrgbHexColor,
)


class InspectionSession:
    """In-memory bookkeeping for one manifest's visual-override state.

    Given an already-open ``SceneManifest``, tracks camera, illumination, and
    per-component display/mark overrides and hashes them into a
    ``SessionSnapshot`` via :meth:`snapshot`. It never opens a file and never
    talks to Blender, so it is not the way to load or render a model.

    Most callers should not instantiate this directly. To open a model and
    render it from Python, use :class:`meshprobe.BlenderController` instead::

        from meshprobe import BlenderController
        from meshprobe.protocol import RenderImageCommand

        with BlenderController() as controller:
            manifest = controller.open_scene("model.glb")
            controller.render_image(
                RenderImageCommand(
                    request_id="r1", op="render.image", output_path="out.png"
                )
            )

    ``InspectionSession`` exists for tests and other code that needs to
    compute or replay session state against a manifest without a live
    Blender worker.
    """

    def __init__(self, manifest: SceneManifest, *, aspect_ratio: float = 1.0) -> None:
        self.manifest = manifest
        self._aspect_ratio = aspect_ratio
        self._known_ids = frozenset(component.id for component in manifest.components)
        self.reset()

    def reset(self) -> SessionSnapshot:
        self._camera = framed_default_camera(
            self.manifest.imported_camera,
            self.manifest.root_bounds,
            aspect_ratio=self._aspect_ratio,
        )
        self._illumination = self.manifest.imported_illumination
        self._components = {
            component_id: ComponentVisualState() for component_id in sorted(self._known_ids)
        }
        return self.snapshot()

    def set_camera(self, camera: Camera) -> SessionSnapshot:
        if camera.pose.frame is not CameraPoseFrame.WORLD:
            raise ValueError("in-memory sessions require a world-frame camera")
        self._camera = camera
        return self.snapshot()

    def set_illumination(self, illumination: Illumination) -> SessionSnapshot:
        self._illumination = illumination
        return self.snapshot()

    def display(self, component_ids: Iterable[str], mode: DisplayMode) -> SessionSnapshot:
        selected = self._validated_ids(component_ids)
        if mode is DisplayMode.ISOLATED:
            self._components = {
                component_id: state.model_copy(
                    update={
                        "display": (
                            DisplayMode.ISOLATED if component_id in selected else DisplayMode.HIDDEN
                        )
                    }
                )
                for component_id, state in self._components.items()
            }
            return self.snapshot()

        for component_id in selected:
            self._components[component_id] = self._components[component_id].model_copy(
                update={"display": mode}
            )
        return self.snapshot()

    def mark(
        self,
        component_ids: Iterable[str],
        mode: MarkMode,
        color: SrgbHexColor | None = None,
    ) -> SessionSnapshot:
        validated = ComponentVisualState(mark=mode, mark_color=color)
        selected = self._validated_ids(component_ids)
        for component_id in selected:
            self._components[component_id] = self._components[component_id].model_copy(
                update={"mark": validated.mark, "mark_color": validated.mark_color}
            )
        return self.snapshot()

    def snapshot(self) -> SessionSnapshot:
        payload = {
            "camera": self._camera.model_dump(mode="json"),
            "illumination": self._illumination.model_dump(mode="json"),
            "components": {
                component_id: state.model_dump(mode="json")
                for component_id, state in sorted(self._components.items())
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        minimum = self.manifest.root_bounds.minimum_mm
        maximum = self.manifest.root_bounds.maximum_mm
        target = (
            (minimum[0] + maximum[0]) / 2,
            (minimum[1] + maximum[1]) / 2,
            (minimum[2] + maximum[2]) / 2,
        )
        return SessionSnapshot(
            camera=self._camera,
            camera_diagnostics=camera_diagnostics(self._camera, target_mm=target),
            illumination=self._illumination,
            components=dict(self._components),
            state_sha256=hashlib.sha256(canonical).hexdigest(),
        )

    def _validated_ids(self, component_ids: Iterable[str]) -> frozenset[str]:
        selected = frozenset(component_ids)
        if not selected:
            raise ValueError("at least one component id is required")
        unknown = selected - self._known_ids
        if unknown:
            raise ValueError(f"unknown component ids: {sorted(unknown)}")
        return selected
