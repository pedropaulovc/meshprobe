from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshprobe.cli import app
from meshprobe.controller import BlenderController, BlenderWorkerError
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

def cube(name, location, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + '-mesh'
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
    return json.loads(output.read_text(encoding="utf-8"))


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


def test_worker_imports_external_gltf_as_one_immutable_bundle(tmp_path: Path) -> None:
    source = build_external_gltf(tmp_path)
    before = snapshot_source(source)
    assert len(before.assets) >= 2

    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(source)

    assert manifest.source_format == "gltf"
    assert manifest.source_sha256 == before.sha256
    assert snapshot_source(source) == before


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
