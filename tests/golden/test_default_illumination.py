from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from meshprobe.controller import DEFAULT_WORKER_TIMEOUT_SECONDS, BlenderController
from meshprobe.models import (
    IlluminationPreset,
    OrthographicProjection,
    PresetIllumination,
    RenderEngine,
    RenderStyle,
)
from meshprobe.protocol import (
    IlluminationSetCommand,
    RenderImageCommand,
    ViewOrbitCommand,
)

pytestmark = pytest.mark.skipif(shutil.which("blender") is None, reason="Blender is not installed")

# The auto-selected default for light-less sources must stay legible from any orbit,
# not near-black like the pre-fix neutral_studio (issue #53, which measured ~8-14).
MINIMUM_MEAN_LUMA = 35.0


def build_lightless_scene(tmp_path: Path) -> Path:
    output = tmp_path / "lightless.glb"
    script = tmp_path / "build-lightless.py"
    script.write_text(
        f"""
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

def material(name, color, roughness):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    shader = value.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1.0)
    shader.inputs['Roughness'].default_value = roughness
    return value

steel = material('steel', (0.32, 0.38, 0.46), 0.35)
orange = material('orange', (0.78, 0.12, 0.025), 0.42)

def cube(name, location, scale, value):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(value)
    return obj

cube('body', (0, 0, 0), (0.15, 0.1, 0.08), steel)
cube('knob', (0.06, -0.02, 0.09), (0.04, 0.04, 0.04), orange)

bpy.ops.export_scene.gltf(
    filepath={json.dumps(str(output))},
    export_format='GLB',
    export_cameras=False,
    export_lights=False,
)
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["blender", "--background", "--factory-startup", "--python", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if not output.is_file():
        raise AssertionError("light-less scene export failed")
    return output


def test_default_preset_is_legible_from_every_orbit(tmp_path: Path) -> None:
    source = build_lightless_scene(tmp_path)
    span_mm = 300.0
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        controller.open_scene(source)
        controller.execute(
            IlluminationSetCommand(
                request_id="light",
                op="illumination.set",
                illumination=PresetIllumination(preset=IlluminationPreset.NEUTRAL_STUDIO),
            )
        )
        luma_by_azimuth: dict[int, float] = {}
        for azimuth in (0, 90, 180, 270):
            controller.execute(
                ViewOrbitCommand(
                    request_id=f"view-{azimuth}",
                    op="view.orbit",
                    target_mm=(0, 0, 0),
                    azimuth_degrees=azimuth,
                    elevation_degrees=25,
                    distance_mm=span_mm * 3,
                    projection=OrthographicProjection(scale_mm=span_mm * 1.6),
                )
            )
            output = tmp_path / f"render-{azimuth}.png"
            controller.render_image(
                RenderImageCommand(
                    request_id=f"render-{azimuth}",
                    op="render.image",
                    output_path=str(output),
                    width=192,
                    height=192,
                    samples=8,
                    engine=RenderEngine.EEVEE,
                    style=RenderStyle.SHADED,
                ),
                evaluator_output_dir=tmp_path / "eval",
            )
            with Image.open(output) as rendered:
                luma_by_azimuth[azimuth] = ImageStat.Stat(rendered.convert("L")).mean[0]

    dark = {azimuth: luma for azimuth, luma in luma_by_azimuth.items() if luma < MINIMUM_MEAN_LUMA}
    assert not dark, f"neutral_studio renders too dark at {dark}; all luma={luma_by_azimuth}"
