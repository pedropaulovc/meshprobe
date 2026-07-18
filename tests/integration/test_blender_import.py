from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
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
    CameraDiagnostics,
    CameraRotationReceipt,
    CameraTranslationReceipt,
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
    OcclusionQueryResult,
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
    ComponentOcclusionCommand,
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


def build_rotation_control_glb(tmp_path: Path, *, name: str, model_rotation_degrees: float) -> Path:
    output = tmp_path / f"{name}.glb"
    script = tmp_path / f"build_{name}.py"
    script.write_text(
        f"""
import bpy
import math
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
assembly = bpy.data.objects.new('asymmetric-assembly', None)
bpy.context.scene.collection.objects.link(assembly)

def part(name, location, scale):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.object
    obj.name = name
    obj.parent = assembly
    obj.location = location
    obj.scale = scale

part('long-arm', (1.4, 0.3, 0.2), (1.7, 0.22, 0.31))
part('offset-block', (-0.35, 1.1, -0.4), (0.28, 0.55, 0.73))
assembly.rotation_euler[2] = math.radians({model_rotation_degrees})

camera_data = bpy.data.cameras.new('inspection-camera')
camera = bpy.data.objects.new('inspection-camera', camera_data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (5, 4, 3)
camera.rotation_euler = ((Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y')).to_euler()
camera_data.lens = 50
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


def build_occluded_glb(
    tmp_path: Path,
    *,
    blocker_alpha: float = 1.0,
    mixed_blocker_materials: bool = False,
    linked_blocker_alpha: bool = False,
    blocker_alpha_mode: str = "BLEND",
    blocker_alpha_cutoff: float | None = None,
) -> Path:
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
if {blocker_alpha!r} < 1 or {linked_blocker_alpha!r}:
    transparent = bpy.data.materials.new('clear-shell')
    transparent.diffuse_color = (0.2, 0.5, 0.8, {blocker_alpha!r})
    transparent.use_nodes = True
    principled = transparent.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (0.2, 0.5, 0.8, 1.0)
    if {linked_blocker_alpha!r}:
        image = bpy.data.images.new('opaque-alpha', width=1, height=1, alpha=True)
        image.pixels = [1.0, 1.0, 1.0, 1.0]
        image.pack()
        texture = transparent.node_tree.nodes.new('ShaderNodeTexImage')
        texture.image = image
        transparent.node_tree.links.new(texture.outputs['Alpha'], principled.inputs['Alpha'])
        if hasattr(transparent, 'surface_render_method'):
            transparent.surface_render_method = 'DITHERED'
    else:
        principled.inputs['Alpha'].default_value = {blocker_alpha!r}
    if {mixed_blocker_materials!r}:
        opaque = bpy.data.materials.new('opaque-frame')
        bpy.context.object.data.materials.append(opaque)
        bpy.context.object.data.materials.append(transparent)
        for polygon in bpy.context.object.data.polygons:
            polygon.material_index = 0 if polygon.normal.x > 0.5 else 1
    else:
        bpy.context.object.data.materials.append(transparent)
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
    if blocker_alpha_mode != "BLEND":
        rewrite_glb_alpha_mode(
            output,
            mode=blocker_alpha_mode,
            cutoff=blocker_alpha_cutoff,
        )
    return output


def rewrite_glb_alpha_mode(path: Path, *, mode: str, cutoff: float | None) -> None:
    payload = path.read_bytes()
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunks.append((chunk_type, payload[offset : offset + chunk_length]))
        offset += chunk_length
    document = json.loads(chunks[0][1].rstrip(b"\x00 \t\r\n").decode("utf-8"))
    material = document["materials"][0]
    material["alphaMode"] = mode
    if cutoff is not None:
        material["alphaCutoff"] = cutoff
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    rebuilt = bytearray(b"glTF" + struct.pack("<I", 2) + b"\x00\x00\x00\x00")
    rebuilt += struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    for chunk_type, chunk in chunks[1:]:
        rebuilt += struct.pack("<II", len(chunk), chunk_type) + chunk
    struct.pack_into("<I", rebuilt, 8, len(rebuilt))
    path.write_bytes(rebuilt)


def build_frustum_crossing_glb(tmp_path: Path) -> Path:
    output = tmp_path / "frustum-crossing.glb"
    script = tmp_path / "build_frustum_crossing.py"
    script.write_text(
        f"""
import bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new('crossing-panel')
mesh.from_pydata(
    [(-11, -10, 0), (12, -10, 0), (12, 10, 0), (-11, 10, 0)],
    [],
    [(0, 1, 2, 3)],
)
panel = bpy.data.objects.new('crossing-panel', mesh)
panel.location = (-9, 0, 0)
bpy.context.scene.collection.objects.link(panel)
data = bpy.data.cameras.new('inspection-camera')
camera = bpy.data.objects.new('inspection-camera', data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (0, 0, 6)
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


def build_label_occlusion_glb(tmp_path: Path) -> Path:
    output = tmp_path / "label-occlusion.glb"
    script = tmp_path / "build_label_occlusion.py"
    script.write_text(
        f"""
import bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
bpy.context.object.name = 'target'
bpy.ops.mesh.primitive_cube_add(size=0.1, location=(2, 1, -0.19))
bpy.context.object.name = 'WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW'
data = bpy.data.cameras.new('inspection-camera')
camera = bpy.data.objects.new('inspection-camera', data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (6, 0, 0)
camera.rotation_euler = (
    (Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y')
).to_euler()
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


def build_multi_surface_occlusion_glb(tmp_path: Path) -> Path:
    output = tmp_path / "multi-surface-occlusion.glb"
    script = tmp_path / "build_multi_surface_occlusion.py"
    script.write_text(
        f"""
import bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
bpy.context.object.name = 'target'
blockers = []
for x in (1.5, 2.5, 3.5):
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(x, 0, 0))
    blockers.append(bpy.context.object)
bpy.ops.object.select_all(action='DESELECT')
for blocker in blockers:
    blocker.select_set(True)
bpy.context.view_layer.objects.active = blockers[0]
bpy.ops.object.join()
bpy.context.object.name = 'multi-surface-blocker'
data = bpy.data.cameras.new('inspection-camera')
camera = bpy.data.objects.new('inspection-camera', data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (6, 0, 0)
camera.rotation_euler = (
    (Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y')
).to_euler()
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


def build_backface_culling_glb(tmp_path: Path, *, back_facing: bool) -> Path:
    output = tmp_path / f"backface-culling-{back_facing}.glb"
    script = tmp_path / f"build_backface_culling_{back_facing}.py"
    face = (0, 3, 2, 1) if back_facing else (0, 1, 2, 3)
    script.write_text(
        f"""
import bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
bpy.context.object.name = 'target'
mesh = bpy.data.meshes.new('single-sided-blocker')
mesh.from_pydata(
    [(2, -1, -1), (2, 1, -1), (2, 1, 1), (2, -1, 1)],
    [],
    [{face!r}],
)
blocker = bpy.data.objects.new('single-sided-blocker', mesh)
bpy.context.scene.collection.objects.link(blocker)
material = bpy.data.materials.new('single-sided')
material.use_backface_culling = True
mesh.materials.append(material)
data = bpy.data.cameras.new('inspection-camera')
camera = bpy.data.objects.new('inspection-camera', data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (6, 0, 0)
camera.rotation_euler = (
    (Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y')
).to_euler()
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


def build_mixed_culling_focus_glb(tmp_path: Path) -> Path:
    output = tmp_path / "mixed-culling-focus.glb"
    script = tmp_path / "build_mixed_culling_focus.py"
    script.write_text(
        f"""
import bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new('mixed-culling-focus')
mesh.from_pydata(
    [
        (2, -1, -1), (2, 0, -1), (2, 0, 1), (2, -1, 1),
        (2, 0, -1), (2, 1, -1), (2, 1, 1), (2, 0, 1),
    ],
    [],
    [(0, 3, 2, 1), (4, 5, 6, 7)],
)
focus = bpy.data.objects.new('mixed-culling-focus', mesh)
bpy.context.scene.collection.objects.link(focus)
material = bpy.data.materials.new('single-sided')
material.use_backface_culling = True
mesh.materials.append(material)
data = bpy.data.cameras.new('inspection-camera')
camera = bpy.data.objects.new('inspection-camera', data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (6, 0, 0)
camera.rotation_euler = (
    (Vector((2, 0, 0)) - camera.location).to_track_quat('-Z', 'Y')
).to_euler()
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


def build_dense_off_frame_glb(tmp_path: Path) -> Path:
    output = tmp_path / "dense-off-frame.glb"
    script = tmp_path / "build_dense_off_frame.py"
    script.write_text(
        f"""
import bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new('off-frame-grid')
vertices = []
faces = []
for row in range(20):
    for column in range(20):
        base = len(vertices)
        x = 100 + column
        y = row - 10
        vertices.extend(((x, y, 0), (x + 0.8, y, 0), (x, y + 0.8, 0)))
        faces.append((base, base + 1, base + 2))
mesh.from_pydata(vertices, [], faces)
panel = bpy.data.objects.new('off-frame-grid', mesh)
bpy.context.scene.collection.objects.link(panel)
data = bpy.data.cameras.new('inspection-camera')
camera = bpy.data.objects.new('inspection-camera', data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (0, 0, 6)
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
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
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
        BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller,
        pytest.raises(BlenderWorkerError, match="unsupported source format"),
    ):
        controller.open_scene(source)


def test_worker_accepts_public_scene_open_shape(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        result = controller.request("scene.open", source_path=str(source))

    manifest = SceneManifest.model_validate(result)
    assert manifest.schema_version == 2
    assert result["schema_version"] == 2
    assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_exact_focus_distance_is_applied_and_changes_render(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    enabled_path = tmp_path / "depth-of-field-enabled.png"
    disabled_path = tmp_path / "depth-of-field-disabled.png"
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
                style=RenderStyle.SHADED,
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
                style=RenderStyle.SHADED,
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
        snapshot = SessionSnapshot.model_validate(controller.request("session.snapshot")["session"])

    diagnostics = initial.camera_diagnostics
    expected_delta = tuple(
        world + diagnostics.right[axis] * -25 + diagnostics.forward[axis] * 50
        for axis, world in enumerate((0, 0, 100))
    )
    expected_position = tuple(
        initial.camera.pose.position_mm[axis] + expected_delta[axis] for axis in range(3)
    )
    moved_camera = Camera.model_validate(moved["camera"])
    move_receipt = CameraTranslationReceipt.model_validate(moved["camera_operation"])
    assert moved_camera.pose.position_mm == pytest.approx(expected_position)
    assert moved_camera.pose.orientation_xyzw == pytest.approx(initial.camera.pose.orientation_xyzw)
    assert moved_camera.projection == initial.camera.projection
    assert snapshot.camera_operation is not None
    assert move_receipt.requested_world_delta_mm == (0, 0, 100)
    assert move_receipt.requested_camera_delta_mm == (-25, 0, 50)
    assert move_receipt.resolved_world_delta_mm == pytest.approx(expected_delta)
    assert move_receipt.resulting_pose.position_mm == pytest.approx(
        snapshot.camera_operation.resulting_pose.position_mm
    )
    assert snapshot.camera_operation.requested_world_delta_mm == (0, 0, 100)
    assert snapshot.camera_operation.requested_camera_delta_mm == (-25, 0, 50)
    assert snapshot.camera_operation.resolved_world_delta_mm == pytest.approx(expected_delta)


def test_rejected_camera_move_preserves_live_camera_state(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    assert isinstance(rotated, dict)
    rotated_camera = Camera.model_validate(rotated["camera"])
    rotation_receipt = CameraRotationReceipt.model_validate(rotated["camera_operation"])
    assert rotated_camera.pose.position_mm == pytest.approx((4_000, -5_000, 3_000))
    assert rotation_receipt.frame == "source"
    assert rotation_receipt.axis_world == pytest.approx((0, 0, 1))
    assert rotation_receipt.requested_visual_degrees == 90
    assert rotation_receipt.applied_camera_orbit_degrees == -90
    assert rotation_receipt.resulting_pose == rotated_camera.pose
    assert isinstance(source_pose, dict)
    resolved_source_pose = Camera.model_validate(source_pose["camera"])
    assert resolved_source_pose.pose.frame == "world"
    assert resolved_source_pose.pose.position_mm == pytest.approx((1_000, -3_000, 2_000))


def test_obj_source_frame_matches_default_importer_axis_conversion(tmp_path: Path) -> None:
    source = build_obj_axes(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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


def test_raw_rotation_uses_source_frame_by_default(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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

    assert rotated["camera_operation"]["frame"] == "source"
    assert rotated["camera_operation"]["target_mm"] == list(target)
    assert rotated["camera"]["projection"] == PerspectiveProjection().model_dump(mode="json")


def test_camera_frame_rotation_axis_follows_current_facing(tmp_path: Path) -> None:
    from meshprobe.camera import camera_local_axis_world, camera_local_point_world

    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        controller.open_scene(source)
        # From opposite sides of the model, the camera-local axis a --frame camera
        # rotation acts on must track the camera's current facing — that is what keeps
        # the on-screen effect (and its sign) invariant, unlike a world-frame axis.
        for azimuth in (0.0, 180.0):
            controller.request(
                "view.orbit",
                target_mm=(0, 0, 0),
                azimuth_degrees=azimuth,
                elevation_degrees=10.0,
                distance_mm=500.0,
                projection={"mode": "perspective"},
            )
            before = controller.request("session.snapshot")["session"]
            orientation = tuple(before["camera"]["pose"]["orientation_xyzw"])
            position = tuple(before["camera"]["pose"]["position_mm"])
            rotated = controller.request(
                "view.rotate",
                target_mm=(0, 0, 0),
                axis="x",
                degrees=15,
                frame="camera",
                projection={"mode": "perspective"},
            )
            receipt = rotated["camera_operation"]
            assert receipt["frame"] == "camera"
            # Blender resolves the rotation in single precision, so compare against the
            # double-precision reference with a float32-scale tolerance, not machine eps.
            expected_axis = camera_local_axis_world(orientation, (1.0, 0.0, 0.0))
            assert receipt["axis_world"] == pytest.approx(list(expected_axis), abs=1e-5)
            # A camera-local target of (0, 0, 0) resolves to the camera position, so the
            # pivot is the camera itself and its position is unchanged by the rotation. A
            # wrong (world-origin) pivot would swing the camera ~130mm, far above 1e-3.
            expected_pivot = camera_local_point_world(orientation, position, (0.0, 0.0, 0.0))
            assert expected_pivot == pytest.approx(position, abs=1e-9)
            assert rotated["camera"]["pose"]["position_mm"] == pytest.approx(position, abs=1e-3)


@pytest.mark.parametrize(
    ("invalid_field", "invalid_value", "error"),
    [
        ("frame", "component", "camera rotation frame must be source, world, or camera"),
        ("degrees", math.nan, "degrees must be finite"),
        ("degrees", math.inf, "degrees must be finite"),
        ("target_mm", (math.nan, 0, 0), "target_mm must contain three finite numbers"),
        ("target_mm", (0, math.inf, 0), "target_mm must contain three finite numbers"),
        (
            "basis",
            {"x": [2, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
            "basis axes must have unit length",
        ),
    ],
)
def test_rejected_raw_rotation_contract_preserves_camera_state(
    tmp_path: Path,
    invalid_field: str,
    invalid_value: object,
    error: str,
) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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


def test_preset_background_srgb_absent_and_null_have_one_canonical_hash(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        controller.open_scene(source)
        absent = controller.request("illumination.set", illumination={"preset": "neutral_studio"})
        explicit_null = controller.request(
            "illumination.set",
            illumination={
                "preset": "neutral_studio",
                "background_rgb": None,
                "background_strength": None,
                "background_srgb": None,
            },
        )

    assert absent["state_sha256"] == explicit_null["state_sha256"]


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
    manager = SessionManager(
        root, blender="blender", timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS
    )
    manager.open("review", source)
    rotated_receipt = manager.execute("review", command)
    rotated_envelope = manager.raw_result(rotated_receipt)
    assert isinstance(rotated_envelope, dict)
    rotated_result = rotated_envelope["result"]
    assert isinstance(rotated_result, dict)
    rotated_camera = Camera.model_validate(rotated_result["camera"])
    rotated_state_sha256 = rotated_result["state_sha256"]
    manager.close("review")

    checkpoint_path = root / "sessions" / "review" / "checkpoint.json"
    legacy_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert legacy_checkpoint["schema_version"] == 2
    legacy_checkpoint["schema_version"] = 1
    legacy_checkpoint["accepted_commands"][0]["degrees"] = -command.degrees
    checkpoint_path.write_text(json.dumps(legacy_checkpoint), encoding="utf-8")

    recovered_manager = SessionManager(
        root, blender="blender", timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS
    )
    recovered_receipt = recovered_manager.execute(
        "review",
        SessionSnapshotCommand(request_id="recovered", op="session.snapshot"),
    )
    recovered_envelope = recovered_manager.raw_result(recovered_receipt)
    assert isinstance(recovered_envelope, dict)
    recovered = SessionSnapshot.model_validate(recovered_envelope["result"]["session"])

    assert recovered.camera.pose.position_mm == pytest.approx(rotated_camera.pose.position_mm)
    assert recovered.camera.pose.orientation_xyzw == pytest.approx(
        rotated_camera.pose.orientation_xyzw
    )
    assert recovered.camera.projection == rotated_camera.projection
    assert recovered.state_sha256 == rotated_state_sha256
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == 2
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
    reapplied = reapplied_envelope["result"]
    assert isinstance(reapplied, dict)

    assert Camera.model_validate(reapplied["camera"]) == rotated_camera
    assert reapplied["state_sha256"] == rotated_state_sha256
    recovered_manager.shutdown()


def test_positive_source_y_visual_rotation_matches_rotated_model(
    tmp_path: Path,
) -> None:
    degrees = 145.0
    source = build_rotation_control_glb(
        tmp_path,
        name="unrotated",
        model_rotation_degrees=0,
    )
    positive_control = build_rotation_control_glb(
        tmp_path,
        name="model-rotated-positive-145",
        model_rotation_degrees=degrees,
    )

    def highlighted_render(path: Path, *, rotate_view: bool) -> tuple[Image.Image, SessionSnapshot]:
        output = tmp_path / f"{path.stem}.png"
        evaluator = tmp_path / f"{path.stem}-evaluator"
        with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
            manifest = controller.open_scene(path)
            controller.execute(
                ComponentMarkCommand(
                    request_id="highlight-all",
                    op="component.mark",
                    component_ids=tuple(component.id for component in manifest.components),
                    mode=MarkMode.HIGHLIGHTED,
                )
            )
            if rotate_view:
                controller.execute(
                    ViewRotateCommand(
                        request_id="visual-positive-source-y-145",
                        op="view.rotate",
                        target_mm=(0, 0, 0),
                        axis="y",
                        degrees=degrees,
                        frame="source",
                    )
                )
            else:
                controller.execute(
                    SessionResetCommand(request_id="positive-control", op="session.reset")
                )
                controller.execute(
                    ComponentMarkCommand(
                        request_id="highlight-control",
                        op="component.mark",
                        component_ids=tuple(component.id for component in manifest.components),
                        mode=MarkMode.HIGHLIGHTED,
                    )
                )
            snapshot = SessionSnapshot.model_validate(
                controller.request("session.snapshot")["session"]
            )
            rendered = controller.render_image(
                RenderImageCommand(
                    request_id="render-control",
                    op="render.image",
                    output_path=str(output),
                    width=256,
                    height=256,
                    samples=1,
                    style=RenderStyle.SHADED,
                ),
                evaluator_output_dir=evaluator,
            )
        assert rendered.evaluator is not None
        image = Image.open(rendered.evaluator.highlighted.path).convert("RGB")
        image.load()
        return image, snapshot

    actual, rotated_snapshot = highlighted_render(source, rotate_view=True)
    expected, _ = highlighted_render(positive_control, rotate_view=False)

    assert rotated_snapshot.camera_operation is not None
    assert rotated_snapshot.camera_operation.requested_visual_degrees == degrees
    assert rotated_snapshot.camera_operation.applied_camera_orbit_degrees == -degrees
    difference = ImageChops.difference(actual, expected)
    assert difference.getbbox() is None


def test_worker_imports_external_gltf_as_one_immutable_bundle(tmp_path: Path) -> None:
    source = build_external_gltf(tmp_path)
    before = snapshot_source(source)
    assert len(before.assets) >= 2

    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)

    assert manifest.source_format == "gltf"
    assert manifest.source_sha256 == before.sha256
    assert snapshot_source(source) == before


def test_import_reports_animation_and_procedural_extension_limits(tmp_path: Path) -> None:
    source = build_unverified_capabilities_gltf(tmp_path)

    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
                style=RenderStyle.SHADED,
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)

    assert manifest.source_format == source_format
    assert len(manifest.components) == 1
    assert any(warning.code == "units.assumed" for warning in manifest.warnings)
    expected_hierarchy = "flattened" if source_format == "stl" else "preserved"
    assert manifest.capabilities.hierarchy == expected_hierarchy


def test_generated_camera_far_clip_scales_with_large_scenes(tmp_path: Path) -> None:
    source = build_flat_mesh(tmp_path, "obj", cube_size=1_000.0)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)

    assert manifest.imported_illumination.preset == "neutral_studio"
    assert any(warning.code == "illumination.generated" for warning in manifest.warnings)


def test_component_paths_escape_separator_characters(tmp_path: Path) -> None:
    source = build_path_collision_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)

    assert {component.path for component in manifest.components} == {"a%2Fb", "a/b"}


def test_worker_applies_visual_session_operations_and_reset(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    environment_path = build_environment_exr(tmp_path)
    environment_hash = hashlib.sha256(environment_path.read_bytes()).hexdigest()
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
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
        assert isinstance(orthographic, dict)
        assert orthographic["camera"]["projection"]["mode"] == "orthographic"
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
        assert isinstance(hidden, dict)
        assert hidden["components"][target]["display"] == "hidden"
        assert controller.request("session.runtime")["components"][target]["hide_render"]

        ghosted = controller.execute(
            ComponentDisplayCommand(
                request_id="ghost",
                op="component.display",
                component_ids=(target,),
                mode=DisplayMode.GHOSTED,
            )
        )
        assert isinstance(ghosted, dict)
        assert ghosted["components"][target]["display"] == "ghosted"
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
        assert isinstance(marked, dict)
        assert marked["components"][target]["mark"] == "highlighted"
        marked_runtime = controller.request("session.runtime")["components"][target]
        assert marked_runtime["materials"] == ["MeshProbeMark-highlighted"]
        assert marked_runtime["material_colors"][0] == pytest.approx(
            [1.0, 0.006995410187265387, 0.29177064981753587, 1.0]
        )

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
        assert custom_mark["components"][target]["mark_color"] == "#ff00ff"
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
        assert isinstance(labeled, dict)
        assert labeled["components"][target]["mark"] == "labeled"
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
        assert isinstance(lit, dict)
        assert lit["illumination"]["preset"] == "raking_left"
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
        white_illumination = PresetIllumination.model_validate(white_preset["illumination"])
        assert white_illumination.background_rgb == (1, 1, 1)
        assert white_illumination.background_strength == 1
        assert white_preset_runtime["lights"] == [
            "MeshProbe-fill",
            "MeshProbe-key",
            "MeshProbe-rim",
        ]
        assert white_preset_runtime["world"]["background_rgb"] == pytest.approx([1, 1, 1])
        assert white_preset_runtime["world"]["background_strength"] == pytest.approx(1)
        assert white_preset_runtime["world"]["ambient_rgb"] == pytest.approx([0.10, 0.10, 0.10])
        assert white_preset_runtime["world"]["ambient_strength"] == pytest.approx(0.5)

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
        assert isinstance(custom, dict)
        custom_illumination = CustomIllumination.model_validate(custom["illumination"])
        custom_runtime = controller.request("session.runtime")
        assert custom_illumination.background_strength == pytest.approx(0.1)
        assert custom_illumination.ambient_rgb == pytest.approx((0.01, 0.02, 0.03))
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
        separated_illumination = CustomIllumination.model_validate(separated["illumination"])
        assert separated_illumination.background_rgb == (1, 1, 1)
        assert separated_illumination.ambient_rgb == (0.1, 0.2, 0.3)
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
        assert isinstance(environment_lit, dict)
        environment_illumination = CustomIllumination.model_validate(
            environment_lit["illumination"]
        )
        assert environment_illumination.visible_background_mode is VisibleBackgroundMode.ENVIRONMENT
        cached_environment = environment_illumination.environment_map
        assert cached_environment is not None
        assert cached_environment.path != str(environment_path)
        environment_runtime = controller.request("session.runtime")["environment_map"]
        environment_world = controller.request("session.runtime")["world"]
        assert environment_runtime["path"] == cached_environment.path
        assert environment_runtime["projection"] == "EQUIRECTANGULAR"
        assert environment_world["visible_background_mode"] == "environment"
        assert environment_world["camera_background_source"] == "MeshProbeEnvironmentLighting"

        raw_environment_spec = environment_illumination.model_dump(mode="json")
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
        original_path_spec = environment_illumination.model_dump(mode="json")
        original_path_spec["environment_map"]["path"] = str(environment_path)
        same_content_state = SessionSnapshot.model_validate(
            controller.request("illumination.set", illumination=original_path_spec)
        )
        assert same_content_state.state_sha256 == environment_lit["state_sha256"]

        before_invalid_mode = controller.request("session.snapshot")["session"]
        with pytest.raises(BlenderWorkerError, match="unknown display mode"):
            controller.request(
                "component.display",
                component_ids=[target],
                mode="hide",
            )
        assert controller.request("session.snapshot")["session"] == before_invalid_mode

        controller.render_image(
            RenderImageCommand(
                request_id="render-edges-before-reset",
                op="render.image",
                output_path=str(tmp_path / "edges-before-reset.png"),
                width=128,
                height=128,
                samples=1,
                style=RenderStyle.SHADED_EDGES,
            )
        )
        assert controller.request("session.runtime")["render"]["use_freestyle"] is True

        reset = controller.execute(SessionResetCommand(request_id="reset", op="session.reset"))
        assert reset == {"reset": True, "state_sha256": initial.state_sha256}
        assert (
            SessionSnapshot.model_validate(controller.request("session.snapshot")["session"])
            == initial
        )
        reset_runtime = controller.request("session.runtime")
        assert reset_runtime["render"]["use_freestyle"] is False
        assert (
            reset_runtime["components"][target]["material_colors"]
            == initial_runtime[target]["material_colors"]
        )


def test_worker_orbit_and_recovery_replay_state(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
        assert isinstance(orbit, dict)
        orbit_camera_state = Camera.model_validate(orbit["camera"])
        assert orbit_camera_state.pose.position_mm == pytest.approx((2_000, 0, 0))
        diagnostics = CameraDiagnostics.model_validate(orbit["camera_diagnostics"])
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
        recovered_state = SessionSnapshot.model_validate(
            controller.request("session.snapshot")["session"]
        )

    assert isinstance(recovered, dict)
    assert recovered["components"][target]["mark"] == "selected"
    assert recovered_state.camera.pose.position_mm == pytest.approx((2_000, 0, 0))
    assert recovered_state.components[target].display is DisplayMode.HIDDEN
    assert recovered_state.components[target].mark is MarkMode.SELECTED
    assert runtime["components"][target]["hide_render"]


def test_worker_rejects_empty_component_selection_without_changing_scene(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
                style=RenderStyle.SHADED,
            )
        )

    assert rendered.session.state_sha256 == before["state_sha256"]
    assert output.is_file()


def test_failed_open_clears_previous_worker_session(tmp_path: Path) -> None:
    source = build_glb(tmp_path)
    corrupt = tmp_path / "corrupt.glb"
    corrupt.write_bytes(b"not a glb")
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
            style=RenderStyle.SHADED,
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


def test_shaded_edges_draws_boundaries_and_creases_not_triangulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_edge_style_glb(tmp_path)
    plain_output = tmp_path / "plain.png"
    edge_output = tmp_path / "shaded-edges.png"
    screen_edge_output = tmp_path / "screen-edges.png"
    colliding_screen_edge_sidecar = tmp_path / "screen-edges.screen-edges.exr"
    colliding_screen_edge_sidecar.write_bytes(b"source-sidecar-must-survive")
    evaluator_dir = tmp_path / "plain-evaluator"
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        default_style = controller.request("render.style")
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
                style=RenderStyle.SHADED,
            ),
            evaluator_output_dir=evaluator_dir,
        )
        blocked_evaluator = tmp_path / "blocked-evaluator"
        blocked_evaluator.write_text("not a directory", encoding="utf-8")
        with pytest.raises(BlenderWorkerError, match=r"worker\.FileExistsError"):
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

        def reject_render_artifacts(_manifest: RenderManifest) -> None:
            raise BlenderWorkerError("render artifact rejected")

        with monkeypatch.context() as artifact_rejection:
            artifact_rejection.setattr(
                BlenderController,
                "_verify_render_artifacts",
                staticmethod(reject_render_artifacts),
            )
            with pytest.raises(BlenderWorkerError, match="render artifact rejected"):
                controller.render_image(
                    RenderImageCommand(
                        request_id="render-edges-controller-rejected",
                        op="render.image",
                        output_path=str(tmp_path / "controller-rejected-edges.png"),
                        width=320,
                        height=240,
                        samples=1,
                        style=RenderStyle.SHADED_EDGES,
                    )
                )
        controller_rejected_runtime = controller.request("session.runtime")
        assert controller_rejected_runtime["render"]["use_freestyle"] is False

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
        screen_edged = controller.render_image(
            RenderImageCommand(
                request_id="render-screen-edges",
                op="render.image",
                output_path=str(screen_edge_output),
                width=320,
                height=240,
                samples=1,
            )
        )
        screen_edge_runtime = controller.request("session.runtime")
        raw_default_arguments = RenderImageCommand(
            request_id="raw-default-screen-edges",
            op="render.image",
            output_path=str(tmp_path / "screen[front].png"),
            width=320,
            height=240,
            samples=1,
        ).model_dump(mode="json", exclude={"request_id", "op", "style", "shaded_edges"})
        raw_default = RenderManifest.model_validate(
            controller.request("render.image", **raw_default_arguments)
        )
        failed_screen_command = RenderImageCommand(
            request_id="render-screen-edges-invalid-setup",
            op="render.image",
            output_path=str(tmp_path / "screen-edges-invalid-setup.png"),
            width=320,
            height=240,
            samples=1,
            style=RenderStyle.SCREEN_EDGES,
        ).model_dump(mode="json", exclude={"request_id", "op"})
        failed_screen_command["shaded_edges"]["line_color"] = "invalid"
        with pytest.raises(BlenderWorkerError, match="invalid literal for int"):
            controller.request("render.image", **failed_screen_command)
        recovered = controller.render_image(
            RenderImageCommand(
                request_id="render-after-screen-edge-setup-failure",
                op="render.image",
                output_path=str(tmp_path / "recovered-after-screen-edge-failure.png"),
                width=320,
                height=240,
                samples=1,
                style=RenderStyle.SHADED,
            )
        )
        recovered_runtime = controller.request("session.runtime")

    assert plain.evaluator is not None
    assert default_style == {"style": "screen_edges"}
    assert plain.session.state_sha256 == edged.session.state_sha256
    assert plain.session.state_sha256 == screen_edged.session.state_sha256
    assert plain.session.state_sha256 == recovered.session.state_sha256
    assert plain.session.camera == edged.session.camera
    assert plain.session.camera == screen_edged.session.camera
    assert edged.style is RenderStyle.SHADED_EDGES
    assert screen_edged.style is RenderStyle.SCREEN_EDGES
    assert raw_default.style is RenderStyle.SCREEN_EDGES
    assert Path(raw_default.color.path).is_file()
    assert edged.shaded_edges == ShadedEdgesStyle()
    assert edge_runtime["render"]["use_freestyle"] is True
    assert screen_edge_runtime["render"]["use_freestyle"] is False
    assert recovered_runtime["render"]["use_freestyle"] is False
    assert colliding_screen_edge_sidecar.read_bytes() == b"source-sidecar-must-survive"

    plain_image = Image.open(plain.color.path).convert("RGB")
    edge_image = Image.open(edged.color.path).convert("RGB")
    screen_edge_image = Image.open(screen_edged.color.path).convert("RGB")
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
    screen_changed = {
        (x, y)
        for y in range(plain_image.height)
        for x in range(plain_image.width)
        if max(
            abs(left - right)
            for left, right in zip(
                plain_image.getpixel((x, y)),
                screen_edge_image.getpixel((x, y)),
                strict=True,
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
    assert len(screen_changed & panel_boundary) >= 40
    assert screen_changed != changed


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
                style=RenderStyle.SHADED,
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
                style=RenderStyle.SHADED,
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
                    style=RenderStyle.SHADED,
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
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    assert len({panel.render.state_sha256 for panel in sheet.panels}) == 9
    assert all(
        panel.render.session.camera.projection.mode == "orthographic" for panel in sheet.panels[3:]
    )
    assert all(panel.callouts for panel in sheet.panels)
    assert all(panel.render.style is RenderStyle.SHADED for panel in sheet.panels)
    assert render_runtime["use_freestyle"] is False
    assert all(panel.callouts[0].component_id == target for panel in sheet.panels)
    focused_bounds = sheet.panels[1].render.session.camera_diagnostics.projected_bounds[target]
    deoccluded_bounds = sheet.panels[2].render.session.camera_diagnostics.projected_bounds[target]
    if sheet.removed_occluder_ids:
        assert focused_bounds == deoccluded_bounds
    else:
        assert focused_bounds != deoccluded_bounds
    assert sheet.occlusion.camera_source == "generated_focus_context"
    assert sheet.occlusion.camera == sheet.panels[1].render.session.camera
    assert sheet.occlusion.visibility_width_px == 128
    assert sheet.occlusion.visibility_height_px == 128
    assert focused_bounds.minimum_image_xy is not None
    assert focused_bounds.maximum_image_xy is not None
    focused_size = tuple(
        high - low
        for low, high in zip(
            focused_bounds.minimum_image_xy,
            focused_bounds.maximum_image_xy,
            strict=True,
        )
    )
    assert 0.1 <= max(focused_size) <= 0.3
    assert sheet.occlusion.visible_fraction_after >= sheet.occlusion.visible_fraction_before
    sheet_width, sheet_height = Image.open(sheet.sheet.path).size
    assert sheet_width == 384
    assert sheet_height >= 528
    assert restored == initial


def test_occlusion_query_uses_current_camera_and_leaves_no_artifact(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component for component in manifest.components if component.display_name == "target"
        )
        blocker = next(
            component for component in manifest.components if component.display_name == "blocker"
        )
        target_center = tuple(
            (low + high) / 2
            for low, high in zip(
                target.world_bounds.minimum_mm,
                target.world_bounds.maximum_mm,
                strict=True,
            )
        )
        comparisons = []
        projections = (
            PerspectiveProjection(focal_length_mm=50),
            OrthographicProjection(scale_mm=3_000),
        )
        for index, projection in enumerate(projections):
            controller.execute(
                ViewOrbitCommand(
                    request_id=f"occlusion-view-{index}",
                    op="view.orbit",
                    target_mm=cast(tuple[float, float, float], target_center),
                    azimuth_degrees=40,
                    elevation_degrees=0,
                    distance_mm=8_000,
                    projection=projection,
                    focus_component_ids=(blocker.id,),
                    aspect_ratio=1,
                )
            )
            before = controller.request("session.snapshot")["session"]
            files_before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
            result = controller.execute(
                ComponentOcclusionCommand(
                    request_id=f"occlusion-{index}",
                    op="component.occlusion",
                    component_ids=(target.id,),
                    max_samples_per_component=128,
                )
            )
            visibility = controller.request(
                "component.visibility",
                component_ids=[target.id],
                width=256,
                height=256,
                graphics_policy=GraphicsPolicy.SOFTWARE_ALLOWED.value,
            )
            after = controller.request("session.snapshot")["session"]
            files_after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
            comparisons.append((projection, before, result, visibility, after))
            assert files_after == files_before

    for projection, before, result, visibility, after in comparisons:
        assert isinstance(result, OcclusionQueryResult)
        assert result.camera_source == "current_session"
        assert result.camera == Camera.model_validate(before["camera"])
        assert set(result.camera_diagnostics.projected_bounds) == {target.id}
        before_diagnostics = CameraDiagnostics.model_validate(before["camera_diagnostics"])
        assert set(before_diagnostics.projected_bounds) == {blocker.id}
        assert result.aspect_ratio == 1
        assert result.camera.projection == projection
        assert result.state_sha256 == before["state_sha256"]
        assert after == before
        assert result.projection_status == "projected"
        assert result.sample_count <= 128
        assert result.visible_sample_count > 0
        assert result.occluded_sample_count > 0
        assert visibility["visible_pixels"] > 0
        assert visibility["visible_pixels"] < visibility["isolated_pixels"]
        assert result.visible_fraction == pytest.approx(visibility["visible_fraction"], abs=0.35)
        assert (
            result.visible_sample_count
            + result.occluded_sample_count
            + result.unresolved_sample_count
            == result.sample_count
        )
        blocker_result = next(item for item in result.blockers if item.component_id == blocker.id)
        assert blocker_result.display_name == blocker.display_name
        assert blocker_result.component_path == blocker.path


def test_occlusion_query_uses_recorded_aspect_for_off_frame_filtering(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS, artifact_cache_root=tmp_path / "cache"
    ) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component for component in manifest.components if component.display_name == "target"
        )
        results = []
        for index, aspect_ratio in enumerate((4.0, 0.25)):
            controller.execute(
                ViewOrbitCommand(
                    request_id=f"aspect-view-{index}",
                    op="view.orbit",
                    target_mm=(0, 1_000, 0),
                    azimuth_degrees=0,
                    elevation_degrees=0,
                    distance_mm=8_000,
                    projection=OrthographicProjection(scale_mm=3_000),
                    focus_component_ids=(target.id,),
                    aspect_ratio=aspect_ratio,
                )
            )
            results.append(
                controller.execute(
                    ComponentOcclusionCommand(
                        request_id=f"aspect-query-{index}",
                        op="component.occlusion",
                        component_ids=(target.id,),
                    )
                )
            )

    landscape, portrait = results
    assert landscape.aspect_ratio == landscape.camera_diagnostics.aspect_ratio == 4
    assert landscape.projection_status == "projected"
    assert landscape.sample_count > 0
    assert portrait.aspect_ratio == portrait.camera_diagnostics.aspect_ratio == 0.25
    assert portrait.projection_status == "not_projected"
    assert portrait.sample_count == 0


@pytest.mark.parametrize(
    "projection",
    [
        PerspectiveProjection(near_clip_mm=9_000, far_clip_mm=10_000),
        PerspectiveProjection(near_clip_mm=1, far_clip_mm=7_000),
    ],
)
def test_occlusion_query_rejects_focus_outside_camera_clip_range(
    tmp_path: Path,
    projection: PerspectiveProjection,
) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component for component in manifest.components if component.display_name == "target"
        )
        controller.execute(
            ViewOrbitCommand(
                request_id="clipped-view",
                op="view.orbit",
                target_mm=(0, 0, 0),
                azimuth_degrees=0,
                elevation_degrees=0,
                distance_mm=8_000,
                projection=projection,
                focus_component_ids=(target.id,),
                aspect_ratio=1,
            )
        )

        result = controller.execute(
            ComponentOcclusionCommand(
                request_id="clipped-query",
                op="component.occlusion",
                component_ids=(target.id,),
            )
        )

    assert result.projection_status == "not_projected"
    assert result.sample_count == 0
    assert result.visible_sample_count == 0
    assert result.occluded_sample_count == 0


def test_occlusion_query_ignores_blockers_before_near_clip(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component for component in manifest.components if component.display_name == "target"
        )
        controller.execute(
            ViewSetCommand(
                request_id="near-clipped-view",
                op="view.set",
                camera=Camera(
                    pose=manifest.imported_camera.pose,
                    projection=PerspectiveProjection(
                        focal_length_mm=50,
                        near_clip_mm=5_250,
                        far_clip_mm=10_000,
                    ),
                ),
                focus_component_ids=(target.id,),
                aspect_ratio=1,
            )
        )

        result = controller.execute(
            ComponentOcclusionCommand(
                request_id="near-clipped-query",
                op="component.occlusion",
                component_ids=(target.id,),
            )
        )
        visibility = controller.request(
            "component.visibility",
            component_ids=[target.id],
            width=128,
            height=128,
            graphics_policy=GraphicsPolicy.SOFTWARE_ALLOWED.value,
        )

    assert result.projection_status == "projected"
    assert result.visible_sample_count > 0
    assert result.occluded_sample_count == 0
    assert result.blockers == ()
    assert visibility["visible_pixels"] == visibility["isolated_pixels"]


@pytest.mark.parametrize(
    "projection",
    [
        PerspectiveProjection(
            focal_length_mm=50,
            near_clip_mm=5_500,
            far_clip_mm=10_000,
        ),
        OrthographicProjection(
            scale_mm=3_000,
            near_clip_mm=5_500,
            far_clip_mm=10_000,
        ),
    ],
)
def test_occlusion_query_excludes_zero_distance_near_plane_samples(
    tmp_path: Path,
    projection: PerspectiveProjection | OrthographicProjection,
) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component for component in manifest.components if component.display_name == "target"
        )
        controller.execute(
            ViewSetCommand(
                request_id="near-plane-view",
                op="view.set",
                camera=Camera(
                    pose=manifest.imported_camera.pose,
                    projection=projection,
                ),
                focus_component_ids=(target.id,),
                aspect_ratio=1,
            )
        )

        result = controller.execute(
            ComponentOcclusionCommand(
                request_id="near-plane-query",
                op="component.occlusion",
                component_ids=(target.id,),
            )
        )

    assert result.projection_status == "projected"
    assert result.sample_count > 0
    assert (
        result.visible_sample_count + result.occluded_sample_count + result.unresolved_sample_count
        == result.sample_count
    )


@pytest.mark.parametrize(
    "projection",
    [
        PerspectiveProjection(
            focal_length_mm=50,
            near_clip_mm=1,
            far_clip_mm=5_500,
        ),
        OrthographicProjection(
            scale_mm=3_000,
            near_clip_mm=1,
            far_clip_mm=5_500,
        ),
    ],
)
def test_occlusion_query_excludes_focus_on_far_clip_plane(
    tmp_path: Path,
    projection: PerspectiveProjection | OrthographicProjection,
) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        controller.request("component.display", component_ids=[by_name["blocker"]], mode="hidden")
        controller.execute(
            ViewSetCommand(
                request_id="far-plane-view",
                op="view.set",
                camera=Camera(
                    pose=manifest.imported_camera.pose,
                    projection=projection,
                ),
                focus_component_ids=(by_name["target"],),
                aspect_ratio=1,
            )
        )

        result = controller.execute(
            ComponentOcclusionCommand(
                request_id="far-plane-query",
                op="component.occlusion",
                component_ids=(by_name["target"],),
            )
        )

    assert result.projection_status == "not_projected"
    assert result.sample_count == 0


def test_occlusion_query_samples_focus_surface_crossing_frustum(tmp_path: Path) -> None:
    source = build_frustum_crossing_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component
            for component in manifest.components
            if component.display_name == "crossing-panel"
        )

        result = controller.execute(
            ComponentOcclusionCommand(
                request_id="crossing-query",
                op="component.occlusion",
                component_ids=(target.id,),
            )
        )
        visibility = controller.request(
            "component.visibility",
            component_ids=[target.id],
            width=128,
            height=128,
            graphics_policy=GraphicsPolicy.SOFTWARE_ALLOWED.value,
        )

    assert visibility["visible_pixels"] > 0
    assert result.projection_status == "projected"
    assert result.sample_count > 0
    assert result.visible_sample_count > 0


def test_occlusion_query_does_not_sample_hidden_focus(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component for component in manifest.components if component.display_name == "target"
        )
        controller.request("component.display", component_ids=[target.id], mode="hidden")

        result = controller.execute(
            ComponentOcclusionCommand(
                request_id="hidden-focus-query",
                op="component.occlusion",
                component_ids=(target.id,),
            )
        )

    assert result.projection_status == "not_projected"
    assert result.sample_count == 0
    assert result.blockers == ()


def test_occlusion_surface_scan_is_budgeted_and_ignores_in_frame_object_origin(
    tmp_path: Path,
) -> None:
    source = build_dense_off_frame_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component
            for component in manifest.components
            if component.display_name == "off-frame-grid"
        )

        ranking = controller.request(
            "component.occluders",
            component_ids=[target.id],
            max_samples_per_component=1,
        )

    assert ranking["sample_count"] == 0
    assert ranking["surface_triangles_scanned"] <= 8


def test_custom_contact_sheet_varies_projection_lighting_and_experiment(
    tmp_path: Path,
) -> None:
    source = build_glb(tmp_path)
    output = tmp_path / "custom-contact-sheet.png"
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
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
    sheet_width, sheet_height = Image.open(sheet.sheet.path).size
    assert sheet_width == 384
    assert sheet_height >= 528


def test_worker_ranks_actual_line_of_sight_occluders(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
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
    assert cleared["visible_rays"] == cleared["sample_count"]
    assert cleared["unresolved_rays"] == 0


def test_worker_traces_through_ghosted_occluders(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        controller.request("component.display", component_ids=[by_name["blocker"]], mode="ghosted")

        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert ranking["occluders"] == []
    assert ranking["visible_rays"] == ranking["sample_count"]
    assert ranking["unresolved_rays"] == 0


def test_worker_traces_through_alpha_blended_occluders(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path, blocker_alpha=0.2)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}

        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert ranking["occluders"] == []
    assert ranking["visible_rays"] == ranking["sample_count"]
    assert ranking["unresolved_rays"] == 0


def test_worker_counts_alpha_blended_focus_hits(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path, blocker_alpha=0.2)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}

        ranking = controller.request("component.occluders", component_ids=[by_name["blocker"]])

    assert ranking["visible_rays"] > 0
    assert ranking["occluded_rays"] == 0
    assert ranking["visible_rays"] + ranking["unresolved_rays"] == ranking["sample_count"]


def test_worker_excludes_fully_transparent_focus_until_marked(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path, blocker_alpha=0.0)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        blocker = by_name["blocker"]

        transparent = controller.request("component.occluders", component_ids=[blocker])
        controller.request("component.mark", component_ids=[blocker], mode="highlighted")
        marked = controller.request("component.occluders", component_ids=[blocker])

    assert transparent["sample_count"] == 0
    assert marked["sample_count"] > 0
    assert marked["visible_rays"] > 0


def test_worker_checks_alpha_blending_on_the_hit_face(tmp_path: Path) -> None:
    source = build_occluded_glb(
        tmp_path,
        blocker_alpha=0.2,
        mixed_blocker_materials=True,
    )
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}

        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert ranking["occluders"][0]["component_id"] == by_name["blocker"]
    assert ranking["occluded_rays"] > 0


def test_worker_treats_linked_alpha_as_opaque_without_a_hit_sample(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path, linked_blocker_alpha=True)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}

        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert ranking["occluders"][0]["component_id"] == by_name["blocker"]
    assert ranking["occluded_rays"] > 0


def test_worker_respects_opaque_gltf_alpha_mode(tmp_path: Path) -> None:
    source = build_occluded_glb(
        tmp_path,
        blocker_alpha=0.2,
        blocker_alpha_mode="OPAQUE",
    )
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}

        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert ranking["occluders"][0]["component_id"] == by_name["blocker"]
    assert ranking["occluded_rays"] > 0


@pytest.mark.parametrize(
    ("alpha", "opaque"),
    [
        (0.2, False),
        (0.5, True),
        (0.8, True),
    ],
)
def test_worker_respects_gltf_alpha_mask_cutoff(
    tmp_path: Path,
    alpha: float,
    opaque: bool,
) -> None:
    source = build_occluded_glb(
        tmp_path,
        blocker_alpha=alpha,
        blocker_alpha_mode="MASK",
        blocker_alpha_cutoff=0.5,
    )
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}

        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    if opaque:
        assert ranking["occluders"][0]["component_id"] == by_name["blocker"]
        assert ranking["occluded_rays"] > 0
        return
    assert ranking["occluders"] == []
    assert ranking["visible_rays"] == ranking["sample_count"]


def test_worker_ignores_component_label_geometry(tmp_path: Path) -> None:
    source = build_label_occlusion_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        target = by_name["target"]
        label_source = by_name["WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"]
        baseline = controller.request("component.occluders", component_ids=[target])
        controller.request("component.mark", component_ids=[label_source], mode="labeled")

        ranking = controller.request("component.occluders", component_ids=[target])

    assert ranking["occluders"] == []
    assert baseline["visible_rays"] == baseline["sample_count"]
    assert ranking["visible_rays"] == ranking["sample_count"]
    assert ranking["unresolved_rays"] == 0


def test_worker_traces_all_surfaces_in_a_ghosted_component(tmp_path: Path) -> None:
    source = build_multi_surface_occlusion_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        blocker = by_name["multi-surface-blocker"]
        controller.request("component.display", component_ids=[blocker], mode="ghosted")

        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert ranking["occluders"] == []
    assert ranking["visible_rays"] == ranking["sample_count"]
    assert ranking["unresolved_rays"] == 0


@pytest.mark.parametrize("back_facing", [False, True])
def test_worker_respects_backface_culling(tmp_path: Path, back_facing: bool) -> None:
    source = build_backface_culling_glb(tmp_path, back_facing=back_facing)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}

        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    if not back_facing:
        assert ranking["occluders"][0]["component_id"] == by_name["single-sided-blocker"]
        assert ranking["occluded_rays"] > 0
        return
    assert ranking["occluders"] == []
    assert ranking["visible_rays"] == ranking["sample_count"]
    assert ranking["unresolved_rays"] == 0


def test_worker_excludes_backface_culled_focus_samples(tmp_path: Path) -> None:
    source = build_backface_culling_glb(tmp_path, back_facing=True)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        target = by_name["target"]
        culled = by_name["single-sided-blocker"]

        rendered_only = controller.request("component.occluders", component_ids=[target])
        culled_only = controller.request("component.occluders", component_ids=[culled])
        mixed = controller.request("component.occluders", component_ids=[target, culled])

    assert culled_only["sample_count"] == 0
    assert culled_only["visible_rays"] == 0
    assert culled_only["occluded_rays"] == 0
    assert culled_only["unresolved_rays"] == 0
    assert mixed["sample_count"] == rendered_only["sample_count"]
    assert mixed["visible_rays"] == rendered_only["visible_rays"]
    assert mixed["unresolved_rays"] == 0


def test_worker_budgets_only_rendered_focus_candidates(tmp_path: Path) -> None:
    source = build_mixed_culling_focus_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        focus = manifest.components[0].id

        ranking = controller.request(
            "component.occluders",
            component_ids=[focus],
            max_samples_per_component=1,
        )

    assert ranking["sample_count"] == 1
    assert ranking["visible_rays"] == 1
    assert ranking["occluded_rays"] == 0
    assert ranking["unresolved_rays"] == 0


def test_worker_counts_ghosted_focus_hits(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        controller.request("component.display", component_ids=[by_name["target"]], mode="ghosted")
        controller.request("component.display", component_ids=[by_name["blocker"]], mode="hidden")

        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert ranking["occluders"] == []
    assert ranking["visible_rays"] == ranking["sample_count"]
    assert ranking["unresolved_rays"] == 0


def test_worker_counts_marked_ghosted_occluders(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        by_name = {component.display_name: component.id for component in manifest.components}
        blocker = by_name["blocker"]
        controller.request("component.display", component_ids=[blocker], mode="ghosted")
        controller.request("component.mark", component_ids=[blocker], mode="highlighted")

        runtime = controller.request("session.runtime")["components"][blocker]
        ranking = controller.request("component.occluders", component_ids=[by_name["target"]])

    assert runtime["materials"] == ["MeshProbeMark-highlighted"]
    assert ranking["occluders"][0]["component_id"] == blocker
    assert ranking["occluded_rays"] > 0


@pytest.mark.parametrize(("width", "height"), [(256, 8), (8, 256)])
def test_worker_visibility_accepts_aspect_preserving_small_axis(
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component for component in manifest.components if component.display_name == "target"
        )

        visibility = controller.request(
            "component.visibility",
            component_ids=[target.id],
            width=width,
            height=height,
            graphics_policy=GraphicsPolicy.SOFTWARE_ALLOWED.value,
        )
        with pytest.raises(BlenderWorkerError, match="between 4 and 1024"):
            controller.request(
                "component.visibility",
                component_ids=[target.id],
                width=3,
                height=height,
                graphics_policy=GraphicsPolicy.SOFTWARE_ALLOWED.value,
            )

    assert visibility["width"] == width
    assert visibility["height"] == height
    assert visibility["visible_pixels"] >= 0
    assert visibility["isolated_pixels"] >= 0


def test_occlusion_evidence_steps_match_strict_blender_visibility_gain(tmp_path: Path) -> None:
    source = build_occluded_glb(tmp_path)
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS, artifact_cache_root=tmp_path / "cache"
    ) as controller:
        manifest = controller.open_scene(source)
        target = next(
            component for component in manifest.components if component.display_name == "target"
        )
        controller.execute(
            ViewOrbitCommand(
                request_id="evidence-view",
                op="view.orbit",
                target_mm=(0, 0, 0),
                azimuth_degrees=0,
                elevation_degrees=0,
                distance_mm=8_000,
                projection=OrthographicProjection(scale_mm=3_000),
                focus_component_ids=(target.id,),
                aspect_ratio=1,
            )
        )
        snapshot = SessionSnapshot.model_validate(controller.request("session.snapshot")["session"])
        evidence = controller._remove_occluders(
            RenderContactSheetCommand(
                request_id="evidence",
                op="render.contact_sheet",
                output_path=str(tmp_path / "unused.png"),
                focus_component_ids=(target.id,),
                panel_width=128,
                panel_height=128,
                samples=1,
                visibility_threshold=0.9,
                occluder_budget=1,
                graphics_policy=GraphicsPolicy.SOFTWARE_ALLOWED,
            ),
            (target.id,),
            camera=snapshot.camera,
        )

    assert evidence.visible_pixel_count_after > evidence.visible_pixel_count_before
    assert evidence.visible_fraction_after > evidence.visible_fraction_before
    assert evidence.steps
    assert evidence.steps[-1].visible_pixel_count_after == evidence.visible_pixel_count_after
    assert evidence.steps[-1].visible_fraction_after == evidence.visible_fraction_after
