from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from PIL import Image, ImageChops
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
    ContactSheetOrbit,
    ContactSheetPanelSpec,
    CustomIllumination,
    DepthOfField,
    DisplayMode,
    EnvironmentMap,
    GraphicsPolicy,
    IlluminationPreset,
    MarkMode,
    OrthographicProjection,
    PerspectiveProjection,
    Pose,
    PresetIllumination,
    RenderEngine,
    RenderManifest,
    RenderStyle,
    SceneManifest,
    SessionSnapshot,
    ShadedEdgesStyle,
    VisibleBackgroundMode,
)
from meshprobe.protocol import (
    ComponentDisplayCommand,
    ComponentMarkCommand,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SessionResetCommand,
    SessionSnapshotCommand,
    ViewMoveCommand,
    ViewOrbitCommand,
    ViewRotateCommand,
    ViewSetCommand,
)
from meshprobe.selectors import ComponentIndex, ComponentSelector, SelectorKind
from meshprobe.sources import snapshot_source
from meshprobe.workspace import SessionManager

pytestmark = pytest.mark.skipif(shutil.which("blender") is None, reason="Blender is not installed")
runner = CliRunner()


def build_obj_axes(tmp_path: Path) -> Path:
    output = tmp_path / "axes.obj"
    output.write_text(
        """o axes
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
l 1 2
l 1 3
l 1 4
""",
        encoding="utf-8",
    )
    return output


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


def build_edge_style_glb(tmp_path: Path) -> Path:
    output = tmp_path / "edge-style.glb"
    script = tmp_path / "build_edge_style.py"
    script.write_text(
        f"""
import bpy
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)

def material(name, color):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1.0)
    return value

panel_material = material('panel-material', (0.72, 0.78, 0.86))
cube_material = material('cube-material', (0.85, 0.32, 0.08))

mesh = bpy.data.meshes.new('triangulated-panel-mesh')
mesh.from_pydata(
    [(-1.5, -1.5, 0), (1.5, -1.5, 0), (1.5, 1.5, 0), (-1.5, 1.5, 0)],
    [],
    [(0, 1, 2), (0, 2, 3)],
)
panel = bpy.data.objects.new('triangulated-panel', mesh)
panel.location = (-1.8, 0, 0)
panel.data.materials.append(panel_material)
bpy.context.scene.collection.objects.link(panel)

bpy.ops.mesh.primitive_cube_add(size=1.5, location=(1.25, 0, 0.75))
cube = bpy.context.object
cube.name = 'creased-cube'
cube.data.materials.append(cube_material)

camera_data = bpy.data.cameras.new('edge-camera')
camera = bpy.data.objects.new('edge-camera', camera_data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (0, -9, 7)
direction = Vector((0, 0, 0.4)) - camera.location
camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
camera_data.lens = 55
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


def render_crash_worker(tmp_path: Path) -> tuple[Path, Path]:
    worker = Path(__file__).parents[2] / "src" / "meshprobe" / "blender" / "worker.py"
    patched_worker = tmp_path / "crash_once_worker.py"
    crash_marker = tmp_path / "render-crashed-once"
    source = worker.read_text(encoding="utf-8")
    needle = '    if operation == "render.image":\n        require_session()\n'
    replacement = (
        '    if operation == "render.image":\n'
        f"        crash_marker = Path({str(crash_marker)!r})\n"
        "        if not crash_marker.exists():\n"
        "            crash_marker.write_text('render crash', encoding='utf-8')\n"
        "            os._exit(86)\n"
        "        require_session()\n"
    )
    assert source.count(needle) == 1
    patched_worker.write_text(source.replace(needle, replacement), encoding="utf-8")

    return patched_worker, crash_marker


def build_environment_exr(tmp_path: Path) -> Path:
    output = tmp_path / "studio.exr"
    script = tmp_path / "build_environment.py"
    script.write_text(
        f"""
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
image = bpy.data.images.new('studio', width=8, height=4, float_buffer=True)
image.pixels = [0.8, 0.4, 0.2, 1.0] * (8 * 4)
bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
bpy.context.scene.render.image_settings.color_depth = '32'
image.save_render(filepath={json.dumps(str(output))}, scene=bpy.context.scene)
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


def build_unverified_capabilities_gltf(tmp_path: Path) -> Path:
    source = build_external_gltf(tmp_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    document["animations"] = [{"name": "inspection-motion", "channels": [], "samplers": []}]
    document["extensionsUsed"] = ["VENDOR_procedural_material"]
    source.write_text(json.dumps(document), encoding="utf-8")
    return source


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

    with BlenderController(
        timeout_seconds=30,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        manifest = controller.open_scene(source)
        assert manifest.normalized_geometry is not None
        normalized_path = Path(manifest.normalized_geometry.path)
        normalized_stat = normalized_path.stat()
        second_manifest = controller.open_scene(source)

    assert manifest == second_manifest
    assert manifest.source_sha256 == before_hash
    assert manifest.source_format == "glb"
    assert manifest.normalized_geometry is not None
    assert normalized_path.read_bytes()[:4] == b"glTF"
    assert manifest.normalized_geometry.bytes == normalized_path.stat().st_size
    assert (
        manifest.normalized_geometry.sha256
        == hashlib.sha256(normalized_path.read_bytes()).hexdigest()
    )
    assert normalized_path.stat().st_mtime_ns == normalized_stat.st_mtime_ns
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

    manifest = SceneManifest.model_validate(result)
    assert manifest.schema_version == 2
    assert result["schema_version"] == 2
    assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_exact_focus_distance_is_applied_and_changes_render(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    enabled_path = tmp_path / "depth-of-field-enabled.png"
    disabled_path = tmp_path / "depth-of-field-disabled.png"
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        projection = manifest.imported_camera.projection
        assert isinstance(projection, PerspectiveProjection)
        enabled_projection = projection.model_copy(
            update={
                "depth_of_field": DepthOfField(
                    mode="enabled",
                    aperture_fstop=0.5,
                    focus_distance_mm=1_000,
                )
            }
        )
        enabled_camera = manifest.imported_camera.model_copy(
            update={"projection": enabled_projection}
        )
        controller.execute(
            ViewSetCommand(
                request_id="enable-depth-of-field",
                op="view.set",
                camera=enabled_camera,
            )
        )
        runtime = controller.request("session.runtime")
        enabled = controller.render_image(
            RenderImageCommand(
                request_id="render-enabled",
                op="render.image",
                output_path=str(enabled_path),
                width=256,
                height=256,
                samples=1,
            )
        )
        disabled_camera = enabled_camera.model_copy(
            update={
                "projection": enabled_projection.model_copy(
                    update={"depth_of_field": DepthOfField()}
                )
            }
        )
        controller.execute(
            ViewSetCommand(
                request_id="disable-depth-of-field",
                op="view.set",
                camera=disabled_camera,
            )
        )
        disabled = controller.render_image(
            RenderImageCommand(
                request_id="render-disabled",
                op="render.image",
                output_path=str(disabled_path),
                width=256,
                height=256,
                samples=1,
            )
        )

    assert runtime["camera"]["depth_of_field"] == pytest.approx(
        {"aperture_fstop": 0.5, "focus_distance_mm": 1_000}
    )
    assert enabled.resolved_depth_of_field is not None
    assert enabled.resolved_depth_of_field.aperture_fstop == pytest.approx(0.5)
    assert enabled.resolved_depth_of_field.focus_distance_mm == pytest.approx(1_000)
    assert disabled.resolved_depth_of_field is None
    enabled_image = Image.open(enabled.color.path).convert("RGB")
    disabled_image = Image.open(disabled.color.path).convert("RGB")
    assert ImageChops.difference(enabled_image, disabled_image).getbbox() is not None


def test_relative_camera_move_combines_world_and_camera_basis(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        controller.open_scene(source)
        initial = SessionSnapshot.model_validate(controller.request("session.snapshot")["session"])
        moved = controller.execute(
            ViewMoveCommand(
                request_id="move",
                op="view.move",
                world_delta_mm=(0, 0, 100),
                camera_delta_mm=(-25, 0, 50),
            )
        )

    diagnostics = initial.camera_diagnostics
    expected_delta = tuple(
        world + diagnostics.right[axis] * -25 + diagnostics.forward[axis] * 50
        for axis, world in enumerate((0, 0, 100))
    )
    expected_position = tuple(
        initial.camera.pose.position_mm[axis] + expected_delta[axis] for axis in range(3)
    )
    assert isinstance(moved, SessionSnapshot)
    assert moved.camera.pose.position_mm == pytest.approx(expected_position)
    assert moved.camera.pose.orientation_xyzw == initial.camera.pose.orientation_xyzw
    assert moved.camera.projection == initial.camera.projection
    assert moved.camera_operation is not None
    assert moved.camera_operation.requested_world_delta_mm == (0, 0, 100)
    assert moved.camera_operation.requested_camera_delta_mm == (-25, 0, 50)
    assert moved.camera_operation.resolved_world_delta_mm == pytest.approx(expected_delta)


def test_rejected_camera_move_preserves_live_camera_state(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        controller.open_scene(source)
        before = SessionSnapshot.model_validate(controller.request("session.snapshot")["session"])
        with pytest.raises(BlenderWorkerError, match="unknown focus component ids"):
            controller.request(
                "view.move",
                world_delta_mm=[100, 0, 0],
                camera_delta_mm=[0, 0, 0],
                focus_component_ids=["missing"],
                aspect_ratio=1,
            )
        after = SessionSnapshot.model_validate(controller.request("session.snapshot")["session"])

    assert after.camera == before.camera
    assert after.camera_operation == before.camera_operation
    assert after.state_sha256 == before.state_sha256


def test_gltf_source_frame_rotation_maps_y_to_world_z(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        rotated = controller.execute(
            ViewRotateCommand(
                request_id="rotate-source-y",
                op="view.rotate",
                target_mm=(0, 0, 0),
                axis="y",
                degrees=90,
                frame="source",
            )
        )
        source_pose = controller.execute(
            ViewSetCommand(
                request_id="set-source-pose",
                op="view.set",
                camera=Camera(
                    pose=Pose(
                        position_mm=(1_000, 2_000, 3_000),
                        orientation_xyzw=(0, 0, 0, 1),
                        frame="source",
                    ),
                    projection=PerspectiveProjection(),
                ),
            )
        )

    assert manifest.coordinate_frames.source.name == "gltf_y_up"
    assert manifest.coordinate_frames.source_to_world == pytest.approx(
        (1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1)
    )
    assert manifest.root_bounds.frame == "world"
    assert all(component.local_bounds.frame == "component" for component in manifest.components)
    assert isinstance(rotated, SessionSnapshot)
    assert rotated.camera.pose.position_mm == pytest.approx((-4_000, 5_000, 3_000))
    assert rotated.camera_operation is not None
    assert rotated.camera_operation.frame == "source"
    assert rotated.camera_operation.axis_world == pytest.approx((0, 0, 1))
    assert rotated.camera_operation.resulting_pose == rotated.camera.pose
    assert isinstance(source_pose, SessionSnapshot)
    assert source_pose.camera.pose.frame == "world"
    assert source_pose.camera.pose.position_mm == pytest.approx((1_000, -3_000, 2_000))


def test_obj_source_frame_matches_default_importer_axis_conversion(tmp_path: Path) -> None:
    source = build_obj_axes(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        rotated = controller.request(
            "view.rotate",
            target_mm=(0, 0, 0),
            axis="z",
            degrees=15,
            frame="source",
        )

    assert manifest.coordinate_frames.source.name == "obj_y_up_assumed"
    assert manifest.coordinate_frames.source_to_world == pytest.approx(
        (1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1)
    )
    assert manifest.root_bounds.minimum_mm == pytest.approx((0, -1_000, 0), abs=1e-3)
    assert manifest.root_bounds.maximum_mm == pytest.approx((1_000, 0, 1_000), abs=1e-3)
    assert rotated["camera_operation"]["axis_world"] == pytest.approx((0, -1, 0))


def test_rejected_raw_rotation_preserves_camera_state(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = tuple(
            (minimum + maximum) / 2
            for minimum, maximum in zip(
                manifest.root_bounds.minimum_mm,
                manifest.root_bounds.maximum_mm,
                strict=True,
            )
        )
        controller.request(
            "view.rotate",
            target_mm=target,
            axis="z",
            degrees=15,
            frame="world",
        )
        before = controller.request("session.snapshot")["session"]

        with pytest.raises(BlenderWorkerError, match="unknown focus component ids"):
            controller.request(
                "view.rotate",
                target_mm=target,
                axis="z",
                degrees=30,
                frame="world",
                focus_component_ids=["missing-component"],
            )

        after = controller.request("session.snapshot")["session"]

    assert after["camera"] == before["camera"]
    assert after["camera_operation"] == before["camera_operation"]
    assert after["state_sha256"] == before["state_sha256"]


def test_rejected_raw_view_set_preserves_camera_operation(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = tuple(
            (minimum + maximum) / 2
            for minimum, maximum in zip(
                manifest.root_bounds.minimum_mm,
                manifest.root_bounds.maximum_mm,
                strict=True,
            )
        )
        controller.request(
            "view.rotate",
            target_mm=target,
            axis="z",
            degrees=15,
        )
        before = controller.request("session.snapshot")["session"]
        invalid_camera = deepcopy(before["camera"])
        invalid_camera["pose"]["frame"] = "camera"

        with pytest.raises(BlenderWorkerError, match="camera pose frame must be source or world"):
            controller.request("view.set", camera=invalid_camera)
        after = controller.request("session.snapshot")["session"]

    assert after["camera"] == before["camera"]
    assert after["camera_operation"] == before["camera_operation"]
    assert after["state_sha256"] == before["state_sha256"]


def test_raw_rotation_uses_world_frame_by_default(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = tuple(
            (minimum + maximum) / 2
            for minimum, maximum in zip(
                manifest.root_bounds.minimum_mm,
                manifest.root_bounds.maximum_mm,
                strict=True,
            )
        )
        rotated = controller.request(
            "view.rotate",
            target_mm=target,
            axis="z",
            degrees=15,
            projection={"mode": "perspective"},
        )

    assert rotated["camera_operation"]["frame"] == "world"
    assert rotated["camera_operation"]["target_mm"] == list(target)
    assert rotated["camera"]["projection"] == PerspectiveProjection().model_dump(mode="json")


@pytest.mark.parametrize(
    ("invalid_field", "invalid_value", "error"),
    [
        ("frame", "component", "camera rotation frame must be source or world"),
        ("degrees", math.nan, "degrees must be finite"),
        ("degrees", math.inf, "degrees must be finite"),
        ("target_mm", (math.nan, 0, 0), "target_mm must contain three finite numbers"),
        ("target_mm", (0, math.inf, 0), "target_mm must contain three finite numbers"),
    ],
)
def test_rejected_raw_rotation_contract_preserves_camera_state(
    tmp_path: Path,
    invalid_field: str,
    invalid_value: object,
    error: str,
) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = tuple(
            (minimum + maximum) / 2
            for minimum, maximum in zip(
                manifest.root_bounds.minimum_mm,
                manifest.root_bounds.maximum_mm,
                strict=True,
            )
        )
        before = controller.request("session.snapshot")["session"]
        command: dict[str, object] = {
            "target_mm": target,
            "axis": "z",
            "degrees": 30,
            "frame": "world",
        }
        command[invalid_field] = invalid_value

        with pytest.raises(BlenderWorkerError, match=error):
            controller.request("view.rotate", **command)
        after = controller.request("session.snapshot")["session"]

    assert after["camera"] == before["camera"]
    assert after["camera_operation"] == before["camera_operation"]
    assert after["state_sha256"] == before["state_sha256"]


def test_rejected_rotation_projection_preserves_camera_state(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = tuple(
            (minimum + maximum) / 2
            for minimum, maximum in zip(
                manifest.root_bounds.minimum_mm,
                manifest.root_bounds.maximum_mm,
                strict=True,
            )
        )
        before = controller.request("session.snapshot")["session"]
        projection = manifest.imported_camera.projection.model_dump(mode="json")
        unsupported_mode = {**projection, "mode": "fisheye"}
        bad_sensor_fit = {**projection, "sensor_fit": "diagonal"}
        missing_orthographic_scale = {"mode": "orthographic"}
        extra_projection_field = {**projection, "debug": 1}
        extra_depth_of_field = deepcopy(projection)
        extra_depth_of_field["depth_of_field"]["debug"] = 1

        for invalid_projection, error in (
            (unsupported_mode, "unknown projection mode"),
            (bad_sensor_fit, "unknown sensor fit"),
            (missing_orthographic_scale, "scale_mm"),
            (extra_projection_field, "unknown perspective projection fields"),
            (extra_depth_of_field, "unknown depth-of-field fields"),
        ):
            with pytest.raises(BlenderWorkerError, match=error):
                controller.request(
                    "view.rotate",
                    target_mm=target,
                    axis="z",
                    degrees=30,
                    projection=invalid_projection,
                )
            after = controller.request("session.snapshot")["session"]
            assert after["camera"] == before["camera"]
            assert after["camera_operation"] == before["camera_operation"]
            assert after["state_sha256"] == before["state_sha256"]


def test_rejected_rotation_depth_of_field_preserves_camera_state(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = tuple(
            (minimum + maximum) / 2
            for minimum, maximum in zip(
                manifest.root_bounds.minimum_mm,
                manifest.root_bounds.maximum_mm,
                strict=True,
            )
        )
        controller.request(
            "view.rotate",
            target_mm=target,
            axis="z",
            degrees=15,
            frame="world",
        )
        before = controller.request("session.snapshot")["session"]
        projection = manifest.imported_camera.projection.model_dump(mode="json")
        projection["depth_of_field"] = {
            "mode": "enabled",
            "aperture_fstop": 2.8,
            "focus_distance_mm": None,
            "focus": {"component_id": "missing-component"},
        }

        with pytest.raises(BlenderWorkerError, match="unknown depth-of-field component id"):
            controller.request(
                "view.rotate",
                target_mm=target,
                axis="z",
                degrees=30,
                frame="world",
                projection=projection,
            )

        after = controller.request("session.snapshot")["session"]

    assert after["camera"] == before["camera"]
    assert after["camera_operation"] == before["camera_operation"]
    assert after["state_sha256"] == before["state_sha256"]


def test_raw_world_camera_pose_has_one_canonical_hash(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        implicit_camera = manifest.imported_camera.model_dump(mode="json")
        implicit_camera["pose"].pop("frame")
        implicit = controller.request("view.set", camera=implicit_camera)

        explicit_camera = deepcopy(implicit_camera)
        explicit_camera["pose"]["frame"] = "world"
        explicit = controller.request("view.set", camera=explicit_camera)

    assert implicit["camera"]["pose"]["frame"] == "world"
    assert implicit["camera"] == explicit["camera"]
    assert implicit["state_sha256"] == explicit["state_sha256"]


def test_source_frame_rotation_survives_checkpoint_replay_and_reset(
    tmp_path: Path,
) -> None:
    source = build_glb(tmp_path)
    root = tmp_path / ".meshprobe"
    command = ViewRotateCommand(
        request_id="rotate-source-y",
        op="view.rotate",
        target_mm=(0, 0, 0),
        axis="y",
        degrees=90,
        frame="source",
    )
    manager = SessionManager(root, blender="blender", timeout_seconds=30)
    manager.open("review", source)
    rotated_receipt = manager.execute("review", command)
    rotated_envelope = manager.raw_result(rotated_receipt)
    assert isinstance(rotated_envelope, dict)
    rotated = SessionSnapshot.model_validate(rotated_envelope["result"])
    manager.close("review")

    recovered_manager = SessionManager(root, blender="blender", timeout_seconds=30)
    recovered_receipt = recovered_manager.execute(
        "review",
        SessionSnapshotCommand(request_id="recovered", op="session.snapshot"),
    )
    recovered_envelope = recovered_manager.raw_result(recovered_receipt)
    assert isinstance(recovered_envelope, dict)
    recovered = SessionSnapshot.model_validate(recovered_envelope["result"]["session"])

    assert recovered.camera.pose.position_mm == pytest.approx(rotated.camera.pose.position_mm)
    assert recovered.camera.pose.orientation_xyzw == pytest.approx(
        rotated.camera.pose.orientation_xyzw
    )
    assert recovered.camera.projection == rotated.camera.projection
    assert recovered.state_sha256 == rotated.state_sha256
    checkpoint = json.loads(
        (root / "sessions" / "review" / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["accepted_commands"] == [command.model_dump(mode="json")]

    recovered_manager.execute(
        "review",
        SessionResetCommand(request_id="reset", op="session.reset"),
    )
    reapplied_receipt = recovered_manager.execute(
        "review",
        command.model_copy(update={"request_id": "rotate-source-y-again"}),
    )
    reapplied_envelope = recovered_manager.raw_result(reapplied_receipt)
    assert isinstance(reapplied_envelope, dict)
    reapplied = SessionSnapshot.model_validate(reapplied_envelope["result"])

    assert reapplied.camera == rotated.camera
    assert reapplied.state_sha256 == rotated.state_sha256
    recovered_manager.shutdown()


def test_worker_imports_external_gltf_as_one_immutable_bundle(tmp_path: Path) -> None:
    source = build_external_gltf(tmp_path)
    before = snapshot_source(source)
    assert len(before.assets) >= 2

    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)

    assert manifest.source_format == "gltf"
    assert manifest.source_sha256 == before.sha256
    assert snapshot_source(source) == before


def test_import_reports_animation_and_procedural_extension_limits(tmp_path: Path) -> None:
    source = build_unverified_capabilities_gltf(tmp_path)

    with BlenderController(
        timeout_seconds=30,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        manifest = controller.open_scene(source)

    assert manifest.capabilities.animations == "static_pose"
    assert manifest.capabilities.procedural_materials == "unsupported"
    warning_codes = {warning.code for warning in manifest.warnings}
    assert "animation.static_pose" in warning_codes
    assert "extension.unverified" in warning_codes
    assert "material.procedural_unsupported" in warning_codes


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


def test_controller_recovers_from_real_blender_crash_during_render(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    before = snapshot_source(source)
    worker, crash_marker = render_crash_worker(tmp_path)
    output = tmp_path / "recovered-render.png"
    controller = BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
        artifact_cache_root=tmp_path / "cache",
    )
    controller._worker_path = worker
    with controller:
        manifest = controller.open_scene(source)
        first_pid = controller.ready_event["pid"] if controller.ready_event is not None else None
        target = manifest.components[-1].id
        controller.execute(
            ComponentMarkCommand(
                request_id="highlight",
                op="component.mark",
                component_ids=(target,),
                mode=MarkMode.HIGHLIGHTED,
            )
        )
        rendered = controller.render_image(
            RenderImageCommand(
                request_id="render",
                op="render.image",
                output_path=str(output),
                width=128,
                height=128,
                samples=1,
            )
        )
        second_pid = controller.ready_event["pid"] if controller.ready_event is not None else None
        runtime = controller.request("session.runtime")

    assert crash_marker.read_text(encoding="utf-8") == "render crash"
    assert first_pid != second_pid
    assert rendered.color.path == str(output)
    assert output.is_file()
    assert rendered.session.components[target].mark is MarkMode.HIGHLIGHTED
    assert runtime["components"][target]["materials"] == ["MeshProbeMark-highlighted"]
    assert snapshot_source(source) == before


def test_open_cli_emits_valid_manifest(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["--workspace", str(workspace), "--raw", "open", str(source)])
    stopped = runner.invoke(app, ["--workspace", str(workspace), "close", "--all"])

    assert result.exit_code == 0
    assert stopped.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["source_format"] == "glb"
    assert len(payload["components"]) == 3


def test_cli_resolves_exact_component_name_and_emits_readable_identity(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    workspace = tmp_path / "workspace"
    opened = runner.invoke(
        app,
        ["--workspace", str(workspace), "--session", "review", "open", str(source)],
    )
    displayed = runner.invoke(
        app,
        [
            "--workspace",
            str(workspace),
            "--session",
            "review",
            "display",
            "retaining-clip",
            "--mode",
            "isolated",
        ],
    )
    stopped = runner.invoke(app, ["--workspace", str(workspace), "close", "--all"])

    assert opened.exit_code == 0
    assert displayed.exit_code == 0
    assert stopped.exit_code == 0
    assert "component: c" in displayed.stdout
    assert "retaining-clip" in displayed.stdout
    assert "id=cmp_" in displayed.stdout


def test_open_cli_reports_source_bundle_errors(tmp_path: Path) -> None:
    source = tmp_path / "invalid.gltf"
    source.write_text("{not-json", encoding="utf-8")

    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["--workspace", str(workspace), "open", str(source)])
    stopped = runner.invoke(app, ["--workspace", str(workspace), "close", "--all"])

    assert result.exit_code == 2
    assert "invalid glTF JSON" in result.output
    assert stopped.exit_code == 0


def test_cli_recovers_killed_session_and_preserves_source(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    before = snapshot_source(source)
    workspace = tmp_path / "workspace"

    opened = runner.invoke(app, ["--workspace", str(workspace), "open", str(source)])
    hidden = runner.invoke(
        app,
        ["--workspace", str(workspace), "display", "c1", "--mode", "hidden"],
    )
    killed = runner.invoke(app, ["--workspace", str(workspace), "kill"])
    recovered = runner.invoke(
        app,
        ["--workspace", str(workspace), "--raw", "snapshot"],
    )
    stopped = runner.invoke(app, ["--workspace", str(workspace), "close", "--all"])

    assert opened.exit_code == 0
    assert hidden.exit_code == 0
    assert killed.exit_code == 0
    assert recovered.exit_code == 0
    assert stopped.exit_code == 0
    snapshot = json.loads(recovered.stdout)["session"]
    assert (
        sum(component["display"] == "hidden" for component in snapshot["components"].values()) == 1
    )
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
    environment_path = build_environment_exr(tmp_path)
    environment_hash = hashlib.sha256(environment_path.read_bytes()).hexdigest()
    with BlenderController(
        timeout_seconds=30,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        manifest = controller.open_scene(source)
        initial = SessionSnapshot.model_validate(controller.request("session.snapshot")["session"])
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

        custom_mark = controller.execute(
            ComponentMarkCommand(
                request_id="custom-mark",
                op="component.mark",
                component_ids=(target,),
                mode=MarkMode.HIGHLIGHTED,
                color="#ff00ff",
            )
        )
        custom_mark_runtime = controller.request("session.runtime")["components"][target]
        assert custom_mark.components[target].mark_color == "#ff00ff"
        assert custom_mark_runtime["materials"] == ["MeshProbeMark-highlighted-ff00ff"]
        assert custom_mark_runtime["material_colors"][0] == pytest.approx([1, 0, 1, 1])
        assert manifest.components[-1].materials.names != ("MeshProbeMark-highlighted-ff00ff",)

        uppercase_mark = SessionSnapshot.model_validate(
            controller.request(
                "component.mark",
                component_ids=[target],
                mode="highlighted",
                color="#FF00FF",
            )
        )
        lowercase_mark = SessionSnapshot.model_validate(
            controller.request(
                "component.mark",
                component_ids=[target],
                mode="highlighted",
                color="#ff00ff",
            )
        )
        assert uppercase_mark.components[target].mark_color == "#ff00ff"
        assert uppercase_mark.state_sha256 == lowercase_mark.state_sha256

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

        white_preset = controller.execute(
            IlluminationSetCommand(
                request_id="white-preset",
                op="illumination.set",
                illumination=PresetIllumination(
                    preset=IlluminationPreset.NEUTRAL_STUDIO,
                    background_rgb=(1, 1, 1),
                ),
            )
        )
        white_preset_runtime = controller.request("session.runtime")
        assert isinstance(white_preset.illumination, PresetIllumination)
        assert white_preset.illumination.background_rgb == (1, 1, 1)
        assert white_preset.illumination.background_strength == 1
        assert white_preset_runtime["lights"] == [
            "MeshProbe-fill",
            "MeshProbe-key",
            "MeshProbe-rim",
        ]
        assert white_preset_runtime["world"]["background_rgb"] == pytest.approx([1, 1, 1])
        assert white_preset_runtime["world"]["background_strength"] == pytest.approx(1)
        assert white_preset_runtime["world"]["ambient_rgb"] == pytest.approx([0.03, 0.03, 0.03])
        assert white_preset_runtime["world"]["ambient_strength"] == pytest.approx(0.15)

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
        custom_runtime = controller.request("session.runtime")
        assert isinstance(custom.illumination, CustomIllumination)
        assert custom.illumination.background_strength == pytest.approx(0.1)
        assert custom.illumination.ambient_rgb == pytest.approx((0.01, 0.02, 0.03))
        assert custom_runtime["lights"] == ["MeshProbe-inspection"]
        assert custom_runtime["world"]["background_rgb"] == pytest.approx([0.01, 0.02, 0.03])
        assert custom_runtime["world"]["background_strength"] == pytest.approx(0.1)
        assert custom_runtime["world"]["ambient_rgb"] == pytest.approx([0.01, 0.02, 0.03])
        assert custom_runtime["world"]["ambient_strength"] == pytest.approx(0.1)

        raw_custom = SessionSnapshot.model_validate(
            controller.request(
                "illumination.set",
                illumination={
                    "preset": "custom",
                    "background_rgb": [0.07, 0.08, 0.09],
                    "ambient_strength": 0.23,
                },
            )
        )
        raw_runtime = controller.request("session.runtime")["world"]
        assert isinstance(raw_custom.illumination, CustomIllumination)
        assert raw_custom.illumination.background_strength == pytest.approx(0.23)
        assert raw_custom.illumination.ambient_rgb == pytest.approx((0.07, 0.08, 0.09))
        assert raw_runtime["background_rgb"] == pytest.approx([0.07, 0.08, 0.09])
        assert raw_runtime["background_strength"] == pytest.approx(0.23)
        assert raw_runtime["ambient_rgb"] == pytest.approx([0.07, 0.08, 0.09])
        assert raw_runtime["ambient_strength"] == pytest.approx(0.23)

        separated = controller.execute(
            IlluminationSetCommand(
                request_id="separated-background",
                op="illumination.set",
                illumination=CustomIllumination(
                    background_rgb=(1, 1, 1),
                    background_strength=1,
                    ambient_rgb=(0.1, 0.2, 0.3),
                    ambient_strength=0.15,
                ),
            )
        )
        runtime_world = controller.request("session.runtime")["world"]
        assert isinstance(separated.illumination, CustomIllumination)
        assert separated.illumination.background_rgb == (1, 1, 1)
        assert separated.illumination.ambient_rgb == (0.1, 0.2, 0.3)
        assert runtime_world["background_rgb"] == pytest.approx([1, 1, 1])
        assert runtime_world["background_strength"] == pytest.approx(1)
        assert runtime_world["ambient_rgb"] == pytest.approx([0.1, 0.2, 0.3])
        assert runtime_world["ambient_strength"] == pytest.approx(0.15)

        environment_lit = controller.execute(
            IlluminationSetCommand(
                request_id="environment-light",
                op="illumination.set",
                illumination=CustomIllumination(
                    background_rgb=(0, 0, 0),
                    ambient_strength=0,
                    environment_map=EnvironmentMap(
                        path=str(environment_path),
                        sha256=environment_hash,
                        strength=1.25,
                        rotation_degrees=30,
                    ),
                ),
            )
        )
        assert isinstance(environment_lit, SessionSnapshot)
        assert isinstance(environment_lit.illumination, CustomIllumination)
        assert (
            environment_lit.illumination.visible_background_mode
            is VisibleBackgroundMode.ENVIRONMENT
        )
        cached_environment = environment_lit.illumination.environment_map
        assert cached_environment is not None
        assert cached_environment.path != str(environment_path)
        environment_runtime = controller.request("session.runtime")["environment_map"]
        environment_world = controller.request("session.runtime")["world"]
        assert environment_runtime["path"] == cached_environment.path
        assert environment_runtime["projection"] == "EQUIRECTANGULAR"
        assert environment_world["visible_background_mode"] == "environment"
        assert environment_world["camera_background_source"] == "MeshProbeEnvironmentLighting"

        raw_environment_spec = environment_lit.illumination.model_dump(mode="json")
        raw_environment_spec.pop("visible_background_mode")
        raw_environment = SessionSnapshot.model_validate(
            controller.request("illumination.set", illumination=raw_environment_spec)
        )
        raw_environment_world = controller.request("session.runtime")["world"]
        assert isinstance(raw_environment.illumination, CustomIllumination)
        assert (
            raw_environment.illumination.visible_background_mode
            is VisibleBackgroundMode.ENVIRONMENT
        )
        assert raw_environment_world["visible_background_mode"] == "environment"
        assert raw_environment_world["camera_background_source"] == "MeshProbeEnvironmentLighting"

        raw_environment_spec.update(
            {
                "background_rgb": [0.2, 0.3, 0.4],
                "background_strength": 0.6,
                "visible_background_mode": "color",
            }
        )
        color_backdrop = SessionSnapshot.model_validate(
            controller.request("illumination.set", illumination=raw_environment_spec)
        )
        color_world = controller.request("session.runtime")["world"]
        assert isinstance(color_backdrop.illumination, CustomIllumination)
        assert color_backdrop.illumination.visible_background_mode is VisibleBackgroundMode.COLOR
        assert color_world["visible_background_mode"] == "color"
        assert color_world["camera_background_source"] == "MeshProbeVisibleBackground"

        before_camera_only_session = controller.request("session.snapshot")["session"]
        before_camera_only_runtime = controller.request("session.runtime")
        with pytest.raises(BlenderWorkerError, match="non-zero light output"):
            controller.request(
                "illumination.set",
                illumination={
                    "preset": "custom",
                    "background_rgb": [1, 1, 1],
                    "background_strength": 1,
                    "ambient_rgb": [0, 0, 0],
                    "ambient_strength": 0,
                    "visible_background_mode": "color",
                    "lights": [],
                    "environment_map": None,
                },
            )
        assert controller.request("session.snapshot")["session"] == before_camera_only_session
        assert controller.request("session.runtime") == before_camera_only_runtime

        zero_strength_environment = cached_environment.model_dump(mode="json")
        zero_strength_environment["strength"] = 0
        zero_intensity_sources = (
            {
                "lights": [
                    {
                        "id": "zero-area",
                        "type": "area",
                        "position_mm": [1, 2, 3],
                        "orientation_xyzw": [0, 0, 0, 1],
                        "power_w": 0,
                        "size_mm": 50,
                        "linear_rgb": [1, 1, 1],
                    }
                ],
                "environment_map": None,
            },
            {
                "lights": [
                    {
                        "id": "zero-point",
                        "type": "point",
                        "position_mm": [1, 2, 3],
                        "power_w": 0,
                        "linear_rgb": [1, 1, 1],
                    }
                ],
                "environment_map": None,
            },
            {
                "lights": [
                    {
                        "id": "zero-spot",
                        "type": "spot",
                        "position_mm": [1, 2, 3],
                        "orientation_xyzw": [0, 0, 0, 1],
                        "power_w": 0,
                        "spot_size_degrees": 45,
                        "blend": 0.15,
                        "linear_rgb": [1, 1, 1],
                    }
                ],
                "environment_map": None,
            },
            {
                "lights": [
                    {
                        "id": "zero-sun",
                        "type": "sun",
                        "orientation_xyzw": [0, 0, 0, 1],
                        "strength": 0,
                        "angle_degrees": 0.526,
                        "linear_rgb": [1, 1, 1],
                    }
                ],
                "environment_map": None,
            },
            {"lights": [], "environment_map": zero_strength_environment},
        )
        for zero_intensity_source in zero_intensity_sources:
            with pytest.raises(BlenderWorkerError, match="non-zero light output"):
                controller.request(
                    "illumination.set",
                    illumination={
                        "preset": "custom",
                        "background_rgb": [1, 1, 1],
                        "background_strength": 1,
                        "ambient_rgb": [0, 0, 0],
                        "ambient_strength": 0,
                        "visible_background_mode": "color",
                        **zero_intensity_source,
                    },
                )
            assert controller.request("session.snapshot")["session"] == before_camera_only_session
            assert controller.request("session.runtime") == before_camera_only_runtime

        with pytest.raises(BlenderWorkerError, match="requires an environment map"):
            controller.request(
                "illumination.set",
                illumination={
                    "preset": "custom",
                    "background_rgb": [0.2, 0.3, 0.4],
                    "ambient_strength": 0.1,
                    "visible_background_mode": "environment",
                    "lights": [],
                    "environment_map": None,
                },
            )
        assert hashlib.sha256(environment_path.read_bytes()).hexdigest() == environment_hash
        original_path_spec = environment_lit.illumination.model_dump(mode="json")
        original_path_spec["environment_map"]["path"] = str(environment_path)
        same_content_state = SessionSnapshot.model_validate(
            controller.request("illumination.set", illumination=original_path_spec)
        )
        assert same_content_state.state_sha256 == environment_lit.state_sha256

        before_invalid_mode = controller.request("session.snapshot")["session"]
        with pytest.raises(BlenderWorkerError, match="unknown display mode"):
            controller.request(
                "component.display",
                component_ids=[target],
                mode="hide",
            )
        assert controller.request("session.snapshot")["session"] == before_invalid_mode

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
        camera_focus = manifest.components[0].id
        orbit = controller.execute(
            ViewOrbitCommand(
                request_id="orbit",
                op="view.orbit",
                target_mm=(0, 0, 0),
                azimuth_degrees=0,
                elevation_degrees=0,
                distance_mm=2_000,
                projection=PerspectiveProjection(focal_length_mm=85),
                focus_component_ids=(camera_focus,),
                aspect_ratio=16 / 9,
            )
        )
        assert isinstance(orbit, SessionSnapshot)
        assert orbit.camera.pose.position_mm == pytest.approx((2_000, 0, 0))
        diagnostics = orbit.camera_diagnostics
        assert diagnostics.aspect_ratio == pytest.approx(16 / 9)
        assert diagnostics.forward == pytest.approx((-1, 0, 0))
        assert diagnostics.target_depth_mm == pytest.approx(2_000)
        assert diagnostics.horizontal_fov_degrees == pytest.approx(
            PerspectiveProjection(focal_length_mm=85).horizontal_fov_degrees(16 / 9)
        )
        assert diagnostics.vertical_fov_degrees == pytest.approx(
            PerspectiveProjection(focal_length_mm=85).vertical_fov_degrees(16 / 9)
        )
        assert len(diagnostics.frustum_corners_mm) == 8
        projected = diagnostics.projected_bounds[camera_focus]
        assert projected.projection_status == "in_front"
        assert projected.minimum_image_xy is not None
        assert projected.maximum_image_xy is not None
        assert projected.minimum_depth_mm < projected.maximum_depth_mm
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


def test_worker_rejects_invalid_mark_without_mutation_and_still_renders(
    tmp_path: Path,
) -> None:
    source = build_glb(tmp_path)
    output = tmp_path / "after-invalid-mark.png"
    with BlenderController(timeout_seconds=30) as controller:
        controller.open_scene(source)
        before = controller.request("session.snapshot")["session"]
        before_runtime = controller.request("session.runtime")

        invalid_commands = (
            {"component_ids": list(before["components"]), "mode": "unmarked", "color": "#ff00ff"},
            {
                "component_ids": list(before["components"]),
                "mode": "highlighted",
                "color": "magenta",
            },
            {
                "component_ids": list(before["components"]),
                "mode": "highlighted",
                "color": [255, 0, 255],
            },
        )
        for arguments in invalid_commands:
            with pytest.raises(BlenderWorkerError, match="color"):
                controller.request("component.mark", **arguments)
            assert controller.request("session.snapshot")["session"] == before
            assert controller.request("session.runtime") == before_runtime

        rendered = controller.render_image(
            RenderImageCommand(
                request_id="render-after-invalid-mark",
                op="render.image",
                output_path=str(output),
                width=64,
                height=64,
                samples=1,
            )
        )

    assert rendered.session.state_sha256 == before["state_sha256"]
    assert output.is_file()


def test_failed_open_clears_previous_worker_session(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    corrupt = tmp_path / "corrupt.glb"
    corrupt.write_bytes(b"not a glb")
    with BlenderController(timeout_seconds=30) as controller:
        controller.open_scene(source)
        with pytest.raises(BlenderWorkerError):
            controller.open_scene(corrupt)
        with pytest.raises(BlenderWorkerError, match="no scene is open"):
            controller.request("session.snapshot")
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


def test_shaded_edges_draws_boundaries_and_creases_not_triangulation(tmp_path: Path) -> None:
    source = build_edge_style_glb(tmp_path)
    plain_output = tmp_path / "plain.png"
    edge_output = tmp_path / "shaded-edges.png"
    evaluator_dir = tmp_path / "plain-evaluator"
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        controller.execute(
            IlluminationSetCommand(
                request_id="edge-lighting",
                op="illumination.set",
                illumination=PresetIllumination(preset=IlluminationPreset.FLAT_DIAGNOSTIC),
            )
        )
        plain = controller.render_image(
            RenderImageCommand(
                request_id="render-plain",
                op="render.image",
                output_path=str(plain_output),
                width=320,
                height=240,
                samples=1,
            ),
            evaluator_output_dir=evaluator_dir,
        )
        blocked_evaluator = tmp_path / "blocked-evaluator"
        blocked_evaluator.write_text("not a directory", encoding="utf-8")
        with pytest.raises(BlenderWorkerError, match="File exists"):
            controller.render_image(
                RenderImageCommand(
                    request_id="render-edges-rejected",
                    op="render.image",
                    output_path=str(tmp_path / "rejected-edges.png"),
                    width=320,
                    height=240,
                    samples=1,
                    style=RenderStyle.SHADED_EDGES,
                ),
                evaluator_output_dir=blocked_evaluator,
            )
        rejected_runtime = controller.request("session.runtime")
        assert rejected_runtime["render"]["use_freestyle"] is False

        edged = controller.render_image(
            RenderImageCommand(
                request_id="render-edges",
                op="render.image",
                output_path=str(edge_output),
                width=320,
                height=240,
                samples=1,
                style=RenderStyle.SHADED_EDGES,
            )
        )
        edge_runtime = controller.request("session.runtime")

    assert plain.evaluator is not None
    assert plain.session.state_sha256 == edged.session.state_sha256
    assert plain.session.camera == edged.session.camera
    assert edged.style is RenderStyle.SHADED_EDGES
    assert edged.shaded_edges == ShadedEdgesStyle()
    assert edge_runtime["render"]["use_freestyle"] is True

    plain_image = Image.open(plain.color.path).convert("RGB")
    edge_image = Image.open(edged.color.path).convert("RGB")
    component_image = Image.open(plain.evaluator.component_ids.path).convert("RGB")
    changed = {
        (x, y)
        for y in range(plain_image.height)
        for x in range(plain_image.width)
        if max(
            abs(left - right)
            for left, right in zip(
                plain_image.getpixel((x, y)), edge_image.getpixel((x, y)), strict=True
            )
        )
        >= 24
    }
    ids_by_name = {component.display_name: component.id for component in manifest.components}
    colors = plain.evaluator.component_colors
    panel_color = colors[ids_by_name["triangulated-panel"]]
    cube_color = colors[ids_by_name["creased-cube"]]

    def pixels_of(color: tuple[int, int, int]) -> set[tuple[int, int]]:
        return {
            (x, y)
            for y in range(component_image.height)
            for x in range(component_image.width)
            if component_image.getpixel((x, y)) == color
        }

    def deep_pixels(pixels: set[tuple[int, int]], radius: int = 4) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x, y in pixels
            if all(
                (x + dx, y + dy) in pixels
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
            )
        }

    panel_pixels = pixels_of(panel_color)
    cube_pixels = pixels_of(cube_color)
    panel_boundary = panel_pixels - deep_pixels(panel_pixels, radius=2)
    cube_deep = deep_pixels(cube_pixels)
    panel_deep = deep_pixels(panel_pixels)

    assert len(changed & panel_boundary) >= 40
    assert len(changed & cube_deep) >= 10
    assert len(changed & panel_deep) <= 2


@pytest.mark.skipif(
    shutil.which("nvidia-smi") is None,
    reason="this host has no CUDA device",
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


@pytest.mark.skipif(
    not Path("/dev/dxg").exists() or shutil.which("nvidia-smi") is None,
    reason="this host is not WSL2 with an NVIDIA adapter",
)
def test_eevee_render_uses_wsl2_d3d12_hardware(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    output = tmp_path / "eevee-hardware.png"
    with BlenderController(timeout_seconds=120) as controller:
        assert controller.graphics is not None
        assert controller.graphics.device_class.value == "hardware"
        assert controller.graphics.backend == "OPENGL"
        assert "D3D12" in controller.graphics.renderer
        assert "NVIDIA" in controller.graphics.renderer
        graphics_renderer = controller.graphics.renderer
        controller.open_scene(source)
        rendered = controller.execute(
            RenderImageCommand(
                request_id="eevee-hardware",
                op="render.image",
                output_path=str(output),
                width=64,
                height=64,
                samples=1,
                engine=RenderEngine.EEVEE,
                graphics_policy=GraphicsPolicy.HARDWARE_REQUIRED,
            )
        )

    assert isinstance(rendered, RenderManifest)
    assert rendered.device == "graphics_hardware"
    assert rendered.graphics.renderer == graphics_renderer
    assert rendered.graphics.blender_device_type == "SOFTWARE"
    assert rendered.graphics.warnings
    assert Image.open(rendered.color.path).size == (64, 64)


def test_eevee_hardware_policy_rejects_software_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GALLIUM_DRIVER", "llvmpipe")
    source = build_glb(tmp_path)
    output = tmp_path / "eevee-software.png"
    with BlenderController(timeout_seconds=120) as controller:
        assert controller.graphics is not None
        assert controller.graphics.device_class.value == "software"
        assert "llvmpipe" in controller.graphics.renderer.casefold()
        controller.open_scene(source)
        with pytest.raises(BlenderWorkerError, match="hardware graphics required"):
            controller.execute(
                RenderImageCommand(
                    request_id="eevee-software",
                    op="render.image",
                    output_path=str(output),
                    width=64,
                    height=64,
                    samples=1,
                    engine=RenderEngine.EEVEE,
                    graphics_policy=GraphicsPolicy.HARDWARE_REQUIRED,
                )
            )

    assert not output.exists()


def test_focused_contact_sheet_has_nine_manifested_panels_and_restores_state(
    tmp_path: Path,
) -> None:
    source = build_glb(tmp_path)
    output = tmp_path / "contact-sheet.png"
    edged_output = tmp_path / "shaded-edges.png"
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        target = manifest.components[-1].id
        initial = controller.request("session.snapshot")["session"]
        controller.execute(
            RenderImageCommand(
                request_id="edges-before-sheet",
                op="render.image",
                output_path=str(edged_output),
                width=128,
                height=128,
                samples=1,
                style=RenderStyle.SHADED_EDGES,
            )
        )
        assert controller.request("session.runtime")["render"]["use_freestyle"] is True
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
        restored = controller.request("session.snapshot")["session"]
        render_runtime = controller.request("session.runtime")["render"]

    assert isinstance(sheet, ContactSheetManifest)
    assert len(sheet.panels) == 9
    assert [panel.index for panel in sheet.panels] == list(range(1, 10))
    assert len({panel.render.state_sha256 for panel in sheet.panels}) >= 8
    assert all(
        panel.render.session.camera.projection.mode == "orthographic" for panel in sheet.panels[3:]
    )
    assert all(panel.callouts for panel in sheet.panels)
    assert all(panel.render.style is RenderStyle.SHADED for panel in sheet.panels)
    assert render_runtime["use_freestyle"] is False
    assert all(panel.callouts[0].component_id == target for panel in sheet.panels)
    assert sheet.occlusion.visible_fraction_after >= sheet.occlusion.visible_fraction_before
    assert Image.open(sheet.sheet.path).size == (384, 528)
    assert restored == initial


def test_custom_contact_sheet_varies_projection_lighting_and_experiment(
    tmp_path: Path,
) -> None:
    source = build_glb(tmp_path)
    output = tmp_path / "custom-contact-sheet.png"
    with BlenderController(
        timeout_seconds=30,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        manifest = controller.open_scene(source)
        target = manifest.components[-1].id
        controller.execute(
            ComponentMarkCommand(
                request_id="custom-color",
                op="component.mark",
                component_ids=(target,),
                mode=MarkMode.HIGHLIGHTED,
                color="#ff00ff",
            )
        )
        fixed_pose = manifest.imported_camera.pose
        panels = (
            ContactSheetPanelSpec(
                caption="Perspective context",
                orbit=ContactSheetOrbit(
                    azimuth_degrees=45,
                    elevation_degrees=25,
                    distance_mm=4_000,
                    projection=PerspectiveProjection(focal_length_mm=50),
                ),
            ),
            ContactSheetPanelSpec(
                caption="Left rake",
                orbit=ContactSheetOrbit(
                    azimuth_degrees=45,
                    elevation_degrees=25,
                    distance_mm=4_000,
                    projection=OrthographicProjection(scale_mm=2_000),
                ),
                illumination=PresetIllumination(preset=IlluminationPreset.RAKING_LEFT),
                display="isolated",
            ),
            ContactSheetPanelSpec(
                caption="Right rake",
                orbit=ContactSheetOrbit(
                    azimuth_degrees=45,
                    elevation_degrees=25,
                    distance_mm=4_000,
                    projection=OrthographicProjection(scale_mm=2_000),
                ),
                illumination=PresetIllumination(preset=IlluminationPreset.RAKING_RIGHT),
                display="isolated",
            ),
            ContactSheetPanelSpec(
                caption="Fixed pose wide",
                camera=Camera(
                    pose=fixed_pose,
                    projection=PerspectiveProjection(focal_length_mm=35),
                ),
                experiment="fixed_pose_focal_study",
            ),
            ContactSheetPanelSpec(
                caption="Fixed pose tele",
                camera=Camera(
                    pose=fixed_pose,
                    projection=PerspectiveProjection(focal_length_mm=85),
                ),
                experiment="fixed_pose_focal_study",
            ),
            ContactSheetPanelSpec(
                caption="Dolly wide",
                orbit=ContactSheetOrbit(
                    azimuth_degrees=30,
                    elevation_degrees=20,
                    distance_mm=3_000,
                    projection=PerspectiveProjection(focal_length_mm=35),
                    reference_focal_length_mm=50,
                    reference_distance_mm=3_000,
                ),
                experiment="dolly_zoom",
            ),
            ContactSheetPanelSpec(
                caption="Dolly tele",
                orbit=ContactSheetOrbit(
                    azimuth_degrees=30,
                    elevation_degrees=20,
                    distance_mm=3_000,
                    projection=PerspectiveProjection(focal_length_mm=85),
                    reference_focal_length_mm=50,
                    reference_distance_mm=3_000,
                ),
                experiment="dolly_zoom",
            ),
            ContactSheetPanelSpec(
                caption="Backlit silhouette",
                orbit=ContactSheetOrbit(
                    azimuth_degrees=90,
                    elevation_degrees=0,
                    distance_mm=4_000,
                    projection=OrthographicProjection(scale_mm=2_000),
                ),
                illumination=PresetIllumination(preset=IlluminationPreset.BACKLIT),
            ),
            ContactSheetPanelSpec(
                caption="High key context",
                orbit=ContactSheetOrbit(
                    azimuth_degrees=-45,
                    elevation_degrees=30,
                    distance_mm=4_000,
                    projection=PerspectiveProjection(focal_length_mm=50),
                ),
                illumination=PresetIllumination(preset=IlluminationPreset.HIGH_KEY),
            ),
        )
        sheet = controller.execute(
            RenderContactSheetCommand(
                request_id="custom-sheet",
                op="render.contact_sheet",
                recipe="custom_3x3",
                panels=panels,
                output_path=str(output),
                focus_component_ids=(target,),
                panel_width=128,
                panel_height=128,
                samples=1,
            )
        )

    assert isinstance(sheet, ContactSheetManifest)
    assert sheet.recipe == "custom_3x3"
    assert sheet.occlusion is None
    assert sheet.panels[1].render.session.camera.projection.mode == "orthographic"
    assert sheet.panels[1].render.session.illumination.preset == "raking_left"
    assert sheet.panels[2].render.session.illumination.preset == "raking_right"
    assert sheet.panels[3].experiment == "fixed_pose_focal_study"
    assert sheet.panels[4].experiment == "fixed_pose_focal_study"
    assert sheet.panels[3].render.session.camera.pose == sheet.panels[4].render.session.camera.pose
    assert sheet.panels[5].experiment == "dolly_zoom"
    assert sheet.panels[6].experiment == "dolly_zoom"
    assert sheet.panels[5].render.session.camera_diagnostics.target_depth_mm == pytest.approx(2_100)
    assert sheet.panels[6].render.session.camera_diagnostics.target_depth_mm == pytest.approx(5_100)
    assert all(
        panel.render.session.components[target].mark_color == "#ff00ff" for panel in sheet.panels
    )
    assert "orthographic 2000mm" in sheet.panels[1].caption
    assert "dolly_zoom" in sheet.panels[5].caption
    assert Image.open(sheet.sheet.path).size == (384, 528)


def test_worker_ranks_actual_line_of_sight_occluders(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        visibility_before = controller.request(
            "component.visibility",
            component_ids=[by_name["target"]],
            width=128,
            height=128,
        )
        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])
        controller.request("component.display", component_ids=[by_name["blocker"]], mode="hidden")
        visibility_after = controller.request(
            "component.visibility",
            component_ids=[by_name["target"]],
            width=128,
            height=128,
        )
        cleared = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert ranking["occluders"][0]["component_id"] == by_name["blocker"]
    assert ranking["occluders"][0]["blocked_rays"] > 0
    assert visibility_before["visible_fraction"] < visibility_after["visible_fraction"]
    assert visibility_after["visible_fraction"] == pytest.approx(1)
    assert cleared["occluders"] == []
