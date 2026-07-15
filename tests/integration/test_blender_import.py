from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest
from PIL import Image
from typer.testing import CliRunner

from meshprobe.cli import app
from meshprobe.controller import (
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    BlenderController,
    BlenderWorkerError,
)
from meshprobe.models import (
    AreaLight,
    Camera,
    ContactSheetManifest,
    CustomIllumination,
    DisplayMode,
    IlluminationPreset,
    MarkMode,
    OrthographicProjection,
    PerspectiveProjection,
    Pose,
    PresetIllumination,
    RenderEngine,
    RenderManifest,
    SessionSnapshot,
)
from meshprobe.protocol import (
    ComponentDisplayCommand,
    ComponentMarkCommand,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SessionResetCommand,
    ViewOrbitCommand,
    ViewSetCommand,
)
from meshprobe.selectors import ComponentIndex, ComponentSelector, SelectorKind
from meshprobe.sources import snapshot_source

pytestmark = pytest.mark.skipif(shutil.which("blender") is None, reason="Blender is not installed")
runner = CliRunner()


def build_glb(tmp_path: Path) -> Path:
    output = tmp_path / "fixture.glb"
    script = tmp_path / "build_fixture.py"
    script.write_text(
        f"""
import bpy
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)

source_material = bpy.data.materials.new('MeshProbeGhost')
source_material.diffuse_color = (0.8, 0.1, 0.05, 1.0)

def cube(name, location, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + '-mesh'
    obj.data.materials.append(source_material)
    obj.parent = parent
    return obj

cover = cube('cover', (0, 0, 0))
idler = cube('idler', (1.5, 0, 0), cover)
clip = cube('retaining-clip', (1.5, 0, 0.75), idler)

camera_data = bpy.data.cameras.new('inspection-camera')
camera = bpy.data.objects.new('inspection-camera', camera_data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (5, 4, 3)
camera.rotation_euler = ((Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y')).to_euler()
camera_data.lens = 85
bpy.context.scene.camera = camera

light_data = bpy.data.lights.new('key-light', 'POINT')
light_data.energy = 900
light = bpy.data.objects.new('key-light', light_data)
bpy.context.scene.collection.objects.link(light)
light.location = (3, 4, 5)

bpy.ops.export_scene.gltf(
    filepath={json.dumps(str(output))},
    export_format='GLB',
    export_cameras=True,
    export_lights=True,
)
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["blender", "--background", "--factory-startup", "--python", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return output


def build_flat_mesh(tmp_path: Path, source_format: str, cube_size: float = 1.0) -> Path:
    output = tmp_path / f"fixture.{source_format}"
    script = tmp_path / f"build_{source_format}.py"
    export_call = (
        f"bpy.ops.wm.obj_export(filepath={json.dumps(str(output))}, export_materials=False)"
        if source_format == "obj"
        else f"bpy.ops.wm.stl_export(filepath={json.dumps(str(output))})"
    )
    script.write_text(
        f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size={cube_size})
bpy.context.object.name = 'fixture-component'
{export_call}
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["blender", "--background", "--factory-startup", "--python", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return output


def build_external_gltf(tmp_path: Path) -> Path:
    output = tmp_path / "external.gltf"
    script = tmp_path / "build_external_gltf.py"
    script.write_text(
        f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0)
bpy.context.object.name = 'external-component'
bpy.ops.export_scene.gltf(
    filepath={json.dumps(str(output))},
    export_format='GLTF_SEPARATE',
)
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["blender", "--background", "--factory-startup", "--python", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return output


def build_duplicate_name_gltf(tmp_path: Path) -> Path:
    output = tmp_path / "duplicates.gltf"
    script = tmp_path / "build_duplicates.py"
    script.write_text(
        f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
for parent_name, x in [('left-group', -1.0), ('right-group', 1.0)]:
    parent = bpy.data.objects.new(parent_name, None)
    bpy.context.scene.collection.objects.link(parent)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, 0, 0))
    bpy.context.object.name = parent_name + '-bolt'
    bpy.context.object.parent = parent
bpy.ops.export_scene.gltf(
    filepath={json.dumps(str(output))},
    export_format='GLTF_SEPARATE',
)
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["blender", "--background", "--factory-startup", "--python", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    for node in document["nodes"]:
        if "mesh" in node:
            node["name"] = "bolt"
    output.write_text(json.dumps(document), encoding="utf-8")
    return output


def probe_active_camera(tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "active-camera.json"
    script = tmp_path / "probe_active_camera.py"
    worker_path = Path(__file__).parents[2] / "src" / "meshprobe" / "blender" / "worker.py"
    script.write_text(
        f"""
import bpy
import importlib.util
import json
bpy.ops.wm.read_factory_settings(use_empty=True)
for name, location in [('a-inactive', (1, 2, 3)), ('z-active', (8, 9, 10))]:
    data = bpy.data.cameras.new(name)
    camera = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = location
    if name == 'z-active':
        bpy.context.scene.camera = camera
bpy.context.view_layer.update()
spec = importlib.util.spec_from_file_location('meshprobe_worker', {json.dumps(str(worker_path))})
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
camera, warning = worker.imported_camera({{'minimum_mm': [0, 0, 0], 'maximum_mm': [1, 1, 1]}})
with open({json.dumps(str(output))}, 'w', encoding='utf-8') as destination:
    json.dump({{'camera': camera, 'warning': warning}}, destination)
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["blender", "--background", "--factory-startup", "--python", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return cast(dict[str, object], json.loads(output.read_text(encoding="utf-8")))


def build_zero_light_glb(tmp_path: Path) -> Path:
    output = tmp_path / "zero-light.glb"
    script = tmp_path / "build_zero_light.py"
    script.write_text(
        f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0)
data = bpy.data.lights.new('zero-light', 'POINT')
data.energy = 0
light = bpy.data.objects.new('zero-light', data)
bpy.context.scene.collection.objects.link(light)
bpy.ops.export_scene.gltf(
    filepath={json.dumps(str(output))},
    export_format='GLB',
    export_lights=True,
)
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["blender", "--background", "--factory-startup", "--python", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return output


def build_occluded_glb(tmp_path: Path) -> Path:
    output = tmp_path / "occluded.glb"
    script = tmp_path / "build_occluded.py"
    script.write_text(
        f"""
import bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
bpy.context.object.name = 'target'
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(2, 0, 0))
bpy.context.object.name = 'blocker'
data = bpy.data.cameras.new('inspection-camera')
camera = bpy.data.objects.new('inspection-camera', data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (6, 0, 0)
camera.rotation_euler = ((Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y')).to_euler()
bpy.context.scene.camera = camera
bpy.ops.export_scene.gltf(
    filepath={json.dumps(str(output))},
    export_format='GLB',
    export_cameras=True,
)
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["blender", "--background", "--factory-startup", "--python", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return output


def build_path_collision_glb(tmp_path: Path) -> Path:
    output = tmp_path / "path-collision.glb"
    script = tmp_path / "build_path_collision.py"
    script.write_text(
        f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-2, 0, 0))
bpy.context.object.name = 'a/b'
root = bpy.data.objects.new('a', None)
bpy.context.scene.collection.objects.link(root)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(2, 0, 0))
bpy.context.object.name = 'b'
bpy.context.object.parent = root
bpy.ops.export_scene.gltf(filepath={json.dumps(str(output))}, export_format='GLB')
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["blender", "--background", "--factory-startup", "--python", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return output


def test_persistent_worker_imports_glb_without_source_changes(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    before_stat = source.stat()

    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        second_manifest = controller.open_scene(source)

    assert manifest == second_manifest
    assert manifest.source_sha256 == before_hash
    assert manifest.source_format == "glb"
    assert manifest.capabilities.hierarchy == "preserved"
    assert manifest.capabilities.textures == "absent"
    assert len(manifest.components) == 3
    assert [component.display_name for component in manifest.components] == [
        "cover",
        "idler",
        "retaining-clip",
    ]
    assert manifest.components[1].parent_id == manifest.components[0].id
    assert manifest.components[2].parent_id == manifest.components[1].id
    assert manifest.imported_camera.projection.mode == "perspective"
    expected_vertical_fov = math.degrees(2 * math.atan((36 / (16 / 9)) / (2 * 85)))
    assert manifest.imported_camera.projection.vertical_fov_degrees(16 / 9) == pytest.approx(
        expected_vertical_fov, rel=1e-5
    )
    assert any(warning.code == "camera.lens_reconstructed" for warning in manifest.warnings)
    assert manifest.imported_illumination.preset == "custom"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_worker_rejects_unsupported_format(tmp_path: Path) -> None:
    source = tmp_path / "fixture.fbx"
    source.write_bytes(b"not an fbx")
    with (
        BlenderController(timeout_seconds=30) as controller,
        pytest.raises(BlenderWorkerError, match="unsupported source format"),
    ):
        controller.open_scene(source)


def test_worker_accepts_public_scene_open_shape(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        result = controller.request("scene.open", source_path=str(source))

    assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_worker_imports_external_gltf_as_one_immutable_bundle(tmp_path: Path) -> None:
    source = build_external_gltf(tmp_path)
    before = snapshot_source(source)
    assert len(before.assets) >= 2

    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)

    assert manifest.source_format == "gltf"
    assert manifest.source_sha256 == before.sha256
    assert snapshot_source(source) == before


def test_import_preserves_duplicate_source_names_and_reports_flattened_groups(
    tmp_path: Path,
) -> None:
    source = build_duplicate_name_gltf(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)

    matches = ComponentIndex(manifest).find(
        ComponentSelector(kind=SelectorKind.EXACT_NAME, pattern="bolt")
    )
    assert len(matches) == 2
    assert {component.display_name for component in matches} == {"bolt"}
    assert {component.path for component in matches} == {"left-group/bolt", "right-group/bolt"}
    assert manifest.capabilities.component_names == "source"
    assert manifest.capabilities.hierarchy == "flattened"
    assert any(warning.code == "hierarchy.flattened" for warning in manifest.warnings)


def test_controller_restarts_after_worker_crash(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        expected = controller.open_scene(source)
        assert controller._process is not None
        controller._process.kill()
        controller._process.wait(timeout=5)
        recovered = controller.open_scene(source)

    assert recovered == expected


def test_open_cli_emits_valid_manifest(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    result = runner.invoke(app, ["open", str(source)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["source_format"] == "glb"
    assert len(payload["components"]) == 3


def test_open_cli_reports_source_bundle_errors(tmp_path: Path) -> None:
    source = tmp_path / "invalid.gltf"
    source.write_text("{not-json", encoding="utf-8")

    result = runner.invoke(app, ["open", str(source)])

    assert result.exit_code == 2
    assert "invalid glTF JSON" in result.output


def test_run_cli_keeps_one_session_and_preserves_source(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    before = snapshot_source(source)
    commands = tmp_path / "inspection.jsonl"
    commands.write_text(
        "\n".join(
            (
                json.dumps({"request_id": "open", "op": "scene.open", "source_path": str(source)}),
                json.dumps({"request_id": "describe", "op": "scene.describe"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", str(commands)])

    assert result.exit_code == 0
    responses = json.loads(result.stdout)["results"]
    assert [response["op"] for response in responses] == ["scene.open", "scene.describe"]
    assert responses[0]["result"]["source_sha256"] == before.sha256
    assert responses[1]["result"]["session"]["state_sha256"]
    assert snapshot_source(source) == before


@pytest.mark.parametrize("source_format", ["obj", "stl"])
def test_worker_imports_flat_mesh_formats(tmp_path: Path, source_format: str) -> None:
    source = build_flat_mesh(tmp_path, source_format)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)

    assert manifest.source_format == source_format
    assert len(manifest.components) == 1
    assert any(warning.code == "units.assumed" for warning in manifest.warnings)
    expected_hierarchy = "flattened" if source_format == "stl" else "preserved"
    assert manifest.capabilities.hierarchy == expected_hierarchy


def test_generated_camera_far_clip_scales_with_large_scenes(tmp_path: Path) -> None:
    source = build_flat_mesh(tmp_path, "obj", cube_size=1_000.0)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)

    scene_span = max(
        high - low
        for low, high in zip(
            manifest.root_bounds.minimum_mm,
            manifest.root_bounds.maximum_mm,
            strict=True,
        )
    )
    assert manifest.imported_camera.projection.far_clip_mm > scene_span


def test_import_prefers_source_active_camera(tmp_path: Path) -> None:
    result = probe_active_camera(tmp_path)
    camera = result["camera"]
    assert isinstance(camera, dict)
    assert camera["pose"]["position_mm"] == pytest.approx((8_000, 9_000, 10_000))
    assert camera["projection"]["sensor_fit"] == "auto"
    assert result["warning"] is None


def test_zero_energy_lights_fall_back_to_valid_preset(tmp_path: Path) -> None:
    source = build_zero_light_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)

    assert manifest.imported_illumination.preset == "neutral_studio"
    assert any(warning.code == "illumination.generated" for warning in manifest.warnings)


def test_component_paths_escape_separator_characters(tmp_path: Path) -> None:
    source = build_path_collision_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)

    assert {component.path for component in manifest.components} == {"a%2Fb", "a/b"}


def test_worker_applies_visual_session_operations_and_reset(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        initial = SessionSnapshot.model_validate(controller.request("scene.describe")["session"])
        target = manifest.components[-1].id
        other = manifest.components[0].id
        initial_runtime = controller.request("session.runtime")["components"]

        orthographic = controller.execute(
            ViewSetCommand(
                request_id="view",
                op="view.set",
                camera=Camera(
                    pose=Pose(
                        position_mm=(5_000, 4_000, 3_000),
                        orientation_xyzw=manifest.imported_camera.pose.orientation_xyzw,
                    ),
                    projection=OrthographicProjection(scale_mm=2_500),
                ),
            )
        )
        assert isinstance(orthographic, SessionSnapshot)
        runtime = controller.request("session.runtime")
        assert runtime["camera"]["type"] == "ORTHO"
        assert runtime["camera"]["ortho_scale_mm"] == pytest.approx(2_500)

        hidden = controller.execute(
            ComponentDisplayCommand(
                request_id="hide",
                op="component.display",
                component_ids=(target,),
                mode=DisplayMode.HIDDEN,
            )
        )
        assert isinstance(hidden, SessionSnapshot)
        assert controller.request("session.runtime")["components"][target]["hide_render"]

        ghosted = controller.execute(
            ComponentDisplayCommand(
                request_id="ghost",
                op="component.display",
                component_ids=(target,),
                mode=DisplayMode.GHOSTED,
            )
        )
        assert isinstance(ghosted, SessionSnapshot)
        assert ghosted.components[target].display is DisplayMode.GHOSTED
        ghosted_runtime = controller.request("session.runtime")["components"]
        assert ghosted_runtime[target]["materials"] != ["MeshProbeGhost"]
        assert (
            ghosted_runtime[other]["material_colors"] == initial_runtime[other]["material_colors"]
        )

        marked = controller.execute(
            ComponentMarkCommand(
                request_id="mark",
                op="component.mark",
                component_ids=(target,),
                mode=MarkMode.HIGHLIGHTED,
            )
        )
        assert isinstance(marked, SessionSnapshot)
        assert marked.components[target].mark is MarkMode.HIGHLIGHTED
        marked_runtime = controller.request("session.runtime")["components"][target]
        assert marked_runtime["materials"] == ["MeshProbeMark-highlighted"]

        labeled = controller.execute(
            ComponentMarkCommand(
                request_id="label",
                op="component.mark",
                component_ids=(target,),
                mode=MarkMode.LABELED,
            )
        )
        assert isinstance(labeled, SessionSnapshot)
        labeled_runtime = controller.request("session.runtime")["components"][target]
        assert labeled_runtime["materials"] == ["MeshProbeMark-labeled"]
        assert labeled_runtime["label"].startswith("MeshProbeLabel-")
        initial_label_rotation = labeled_runtime["label_rotation_wxyz"]

        controller.execute(
            ViewSetCommand(
                request_id="perspective-auto",
                op="view.set",
                camera=Camera(
                    pose=Pose(
                        position_mm=(-4_000, 2_000, 1_000),
                        orientation_xyzw=(0, 0, 1, 0),
                    ),
                    projection=PerspectiveProjection(sensor_fit="auto"),
                ),
            )
        )
        camera_runtime = controller.request("session.runtime")
        assert camera_runtime["camera"]["sensor_fit"] == "AUTO"
        assert camera_runtime["components"][target]["label_rotation_wxyz"] != initial_label_rotation

        controller.execute(
            ComponentDisplayCommand(
                request_id="hide-label",
                op="component.display",
                component_ids=(target,),
                mode=DisplayMode.HIDDEN,
            )
        )
        assert controller.request("session.runtime")["components"][target]["label"] is None
        controller.execute(
            ComponentDisplayCommand(
                request_id="show-label",
                op="component.display",
                component_ids=(target,),
                mode=DisplayMode.SHOWN,
            )
        )
        assert controller.request("session.runtime")["components"][target]["label"].startswith(
            "MeshProbeLabel-"
        )

        lit = controller.execute(
            IlluminationSetCommand(
                request_id="light",
                op="illumination.set",
                illumination=PresetIllumination(preset=IlluminationPreset.RAKING_LEFT),
            )
        )
        assert isinstance(lit, SessionSnapshot)
        assert controller.request("session.runtime")["lights"] == ["MeshProbe-rake"]

        custom = controller.execute(
            IlluminationSetCommand(
                request_id="custom-light",
                op="illumination.set",
                illumination=CustomIllumination(
                    background_rgb=(0.01, 0.02, 0.03),
                    ambient_strength=0.1,
                    lights=(
                        AreaLight(
                            id="inspection",
                            position_mm=(1_000, 2_000, 3_000),
                            orientation_xyzw=(0, 0, 0, 1),
                            power_w=500,
                            size_mm=800,
                            color_temperature_k=5_200,
                        ),
                    ),
                ),
            )
        )
        assert isinstance(custom, SessionSnapshot)
        assert controller.request("session.runtime")["lights"] == ["MeshProbe-inspection"]

        before_invalid_mode = controller.request("scene.describe")["session"]
        with pytest.raises(BlenderWorkerError, match="unknown display mode"):
            controller.request(
                "component.display",
                component_ids=[target],
                mode="hide",
            )
        assert controller.request("scene.describe")["session"] == before_invalid_mode

        reset = controller.execute(SessionResetCommand(request_id="reset", op="session.reset"))
        assert reset == initial
        reset_runtime = controller.request("session.runtime")["components"]
        assert (
            reset_runtime[target]["material_colors"] == initial_runtime[target]["material_colors"]
        )


def test_worker_orbit_and_recovery_replay_state(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = manifest.components[-1].id
        orbit = controller.execute(
            ViewOrbitCommand(
                request_id="orbit",
                op="view.orbit",
                target_mm=(0, 0, 0),
                azimuth_degrees=0,
                elevation_degrees=0,
                distance_mm=2_000,
                projection=PerspectiveProjection(focal_length_mm=85),
            )
        )
        assert isinstance(orbit, SessionSnapshot)
        assert orbit.camera.pose.position_mm == pytest.approx((2_000, 0, 0))
        controller.execute(
            ComponentDisplayCommand(
                request_id="hide",
                op="component.display",
                component_ids=(target,),
                mode=DisplayMode.HIDDEN,
            )
        )
        assert controller._process is not None
        controller._process.kill()
        controller._process.wait(timeout=5)

        recovered = controller.execute(
            ComponentMarkCommand(
                request_id="mark",
                op="component.mark",
                component_ids=(target,),
                mode=MarkMode.SELECTED,
            )
        )
        runtime = controller.request("session.runtime")

    assert isinstance(recovered, SessionSnapshot)
    assert recovered.camera.pose.position_mm == pytest.approx((2_000, 0, 0))
    assert recovered.components[target].display is DisplayMode.HIDDEN
    assert recovered.components[target].mark is MarkMode.SELECTED
    assert runtime["components"][target]["hide_render"]


def test_worker_rejects_empty_component_selection_without_changing_scene(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = manifest.components[-1].id
        before = controller.request("session.runtime")
        with pytest.raises(BlenderWorkerError, match="at least one component"):
            controller.request("component.display", component_ids=[], mode="isolated")
        after = controller.request("session.runtime")

    assert after["components"][target] == before["components"][target]


def test_failed_open_clears_previous_worker_session(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    corrupt = tmp_path / "corrupt.glb"
    corrupt.write_bytes(b"not a glb")
    with BlenderController(timeout_seconds=30) as controller:
        controller.open_scene(source)
        with pytest.raises(BlenderWorkerError):
            controller.open_scene(corrupt)
        with pytest.raises(BlenderWorkerError, match="no scene is open"):
            controller.request("scene.describe")
        assert controller._source_path is None
        assert controller._source_sha256 is None


def test_worker_renders_color_and_private_evaluator_passes(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    before = snapshot_source(source)
    output = tmp_path / "evidence.png"
    evaluator_dir = tmp_path / "private"
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        target = manifest.components[-1].id
        controller.execute(
            ComponentMarkCommand(
                request_id="mark",
                op="component.mark",
                component_ids=(target,),
                mode=MarkMode.HIGHLIGHTED,
            )
        )
        command = RenderImageCommand(
            request_id="render",
            op="render.image",
            output_path=str(output),
            width=256,
            height=192,
            samples=1,
            engine=RenderEngine.EEVEE,
        )
        first = controller.render_image(command, evaluator_output_dir=evaluator_dir)
        controller.execute(
            ViewOrbitCommand(
                request_id="move",
                op="view.orbit",
                target_mm=(0, 0, 0),
                azimuth_degrees=120,
                elevation_degrees=25,
                distance_mm=5_000,
                projection=PerspectiveProjection(),
            )
        )
        rendered = controller.render_image(command, evaluator_output_dir=evaluator_dir)
        runtime = controller.request("session.runtime")

    assert isinstance(rendered, RenderManifest)
    assert rendered.width == 256
    assert rendered.height == 192
    assert rendered.evaluator is not None
    assert first.evaluator is not None
    assert first.evaluator.multilayer.sha256 != rendered.evaluator.multilayer.sha256
    assert runtime["render"]["eevee_samples"] == 1
    assert rendered.evaluator.component_colors.keys() == {
        component.id for component in manifest.components
    }
    assert Image.open(rendered.color.path).size == (256, 192)
    assert Image.open(rendered.evaluator.component_ids.path).size == (256, 192)
    highlighted = Image.open(rendered.evaluator.highlighted.path).convert("RGB")
    assert highlighted.getbbox() is not None
    pass_header = Path(rendered.evaluator.multilayer.path).read_bytes()[:4_096]
    assert b"Depth" in pass_header
    assert b"Normal" in pass_header
    assert snapshot_source(source) == before


@pytest.mark.skipif(
    os.name == "nt" and shutil.which("nvidia-smi") is None,
    reason="the Windows CI runner has no CUDA device",
)
def test_cycles_render_uses_cuda_device(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    output = tmp_path / "cycles.png"
    with BlenderController(timeout_seconds=120) as controller:
        controller.open_scene(source)
        rendered = controller.execute(
            RenderImageCommand(
                request_id="cycles",
                op="render.image",
                output_path=str(output),
                width=64,
                height=64,
                samples=1,
                engine=RenderEngine.CYCLES,
            )
        )

    assert isinstance(rendered, RenderManifest)
    assert rendered.engine is RenderEngine.CYCLES
    assert rendered.device == "cuda"
    assert rendered.evaluator is None
    assert rendered.blender_version
    assert Image.open(rendered.color.path).size == (64, 64)


def test_focused_contact_sheet_has_nine_manifested_panels_and_restores_state(
    tmp_path: Path,
) -> None:
    source = build_glb(tmp_path)
    output = tmp_path / "contact-sheet.png"
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = manifest.components[-1].id
        initial = controller.request("scene.describe")["session"]
        sheet = controller.execute(
            RenderContactSheetCommand(
                request_id="sheet",
                op="render.contact_sheet",
                output_path=str(output),
                focus_component_ids=(target,),
                panel_width=128,
                panel_height=128,
                samples=1,
            )
        )
        restored = controller.request("scene.describe")["session"]

    assert isinstance(sheet, ContactSheetManifest)
    assert len(sheet.panels) == 9
    assert [panel.index for panel in sheet.panels] == list(range(1, 10))
    assert len({panel.render.state_sha256 for panel in sheet.panels}) >= 8
    assert all(
        panel.render.session.camera.projection.mode == "orthographic" for panel in sheet.panels[3:]
    )
    assert Image.open(sheet.sheet.path).size == (384, 480)
    assert restored == initial


def test_worker_ranks_actual_line_of_sight_occluders(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])
        controller.request("component.display", component_ids=[by_name["blocker"]], mode="hidden")
        cleared = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert ranking["occluders"][0]["component_id"] == by_name["blocker"]
    assert ranking["occluders"][0]["blocked_rays"] > 0
    assert cleared["occluders"] == []
