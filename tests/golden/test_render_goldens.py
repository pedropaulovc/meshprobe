from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageFilter, ImageStat

from meshprobe.controller import DEFAULT_WORKER_TIMEOUT_SECONDS, BlenderController
from meshprobe.models import (
    Component,
    DisplayMode,
    IlluminationPreset,
    PerspectiveProjection,
    PresetIllumination,
    RenderEngine,
    RenderStyle,
)
from meshprobe.protocol import (
    ComponentDisplayCommand,
    IlluminationSetCommand,
    RenderImageCommand,
    ViewOrbitCommand,
)

pytestmark = pytest.mark.skipif(shutil.which("blender") is None, reason="Blender is not installed")

GOLDEN_ROOT = Path(__file__).parent
COLOR_GOLDEN = GOLDEN_ROOT / "inspection-color.png"
MASK_GOLDEN = GOLDEN_ROOT / "inspection-components.png"


def build_golden_scene(tmp_path: Path) -> Path:
    output = tmp_path / "golden-scene.glb"
    script = tmp_path / "build-golden-scene.py"
    script.write_text(
        f"""
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

def material(name, color, metallic, roughness):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1.0)
    value.metallic = metallic
    value.roughness = roughness
    value.use_nodes = True
    shader = value.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1.0)
    shader.inputs['Metallic'].default_value = metallic
    shader.inputs['Roughness'].default_value = roughness
    return value

steel = material('steel', (0.32, 0.38, 0.46), 0.75, 0.28)
orange = material('orange', (0.78, 0.12, 0.025), 0.1, 0.42)
dark = material('dark', (0.025, 0.03, 0.04), 0.0, 0.8)

def cube(name, location, scale, material_value):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material_value)
    return obj

base = cube('base', (0, 0, 0), (2.4, 1.7, 0.22), steel)
housing = cube('housing', (0, 0, 0.72), (1.85, 1.25, 0.5), dark)
idler = cube('idler', (0.35, -0.08, 1.42), (0.58, 0.58, 0.58), orange)
clip = cube('clip', (0.35, -0.78, 1.42), (0.12, 0.18, 0.7), steel)

bpy.ops.export_scene.gltf(
    filepath={json.dumps(str(output))},
    export_format='GLB',
    export_cameras=False,
    export_lights=False,
)
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "blender",
            "--background",
            "--factory-startup",
            "--python",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if not output.is_file():
        raise AssertionError(f"golden scene export failed:\n{completed.stdout}\n{completed.stderr}")
    return output


def semantic_component_mask(
    component_image: Path,
    component_colors: dict[str, tuple[int, int, int]],
    components: tuple[Component, ...],
) -> Image.Image:
    role_by_name = {"base": 64, "housing": 128, "idler": 192, "clip": 255}
    role_by_id = {component.id: role_by_name[component.source_name] for component in components}
    color_roles = {
        color: role_by_id[component_id] for component_id, color in component_colors.items()
    }
    with Image.open(component_image) as source:
        pixels = tuple(source.convert("RGB").get_flattened_data())
        size = source.size
    semantic = Image.new("L", size)
    semantic.putdata(tuple(color_roles.get(pixel, 0) for pixel in pixels))
    return semantic


def srgb_to_lab(pixel: tuple[int, int, int]) -> tuple[float, float, float]:
    channels = tuple(channel / 255 for channel in pixel)
    linear = tuple(
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    )
    x = (linear[0] * 0.4124564 + linear[1] * 0.3575761 + linear[2] * 0.1804375) / 0.95047
    y = linear[0] * 0.2126729 + linear[1] * 0.7151522 + linear[2] * 0.0721750
    z = (linear[0] * 0.0193339 + linear[1] * 0.1191920 + linear[2] * 0.9503041) / 1.08883
    delta = 6 / 29
    transformed = tuple(
        value ** (1 / 3) if value > delta**3 else value / (3 * delta**2) + 4 / 29
        for value in (x, y, z)
    )
    return (
        116 * transformed[1] - 16,
        500 * (transformed[0] - transformed[1]),
        200 * (transformed[1] - transformed[2]),
    )


def delta_e_values(observed: Image.Image, expected: Image.Image) -> list[float]:
    deltas = []
    for observed_pixel, expected_pixel in zip(
        observed.convert("RGB").get_flattened_data(),
        expected.convert("RGB").get_flattened_data(),
        strict=True,
    ):
        observed_lab = srgb_to_lab(observed_pixel)
        expected_lab = srgb_to_lab(expected_pixel)
        deltas.append(
            math.sqrt(
                sum(
                    (observed_value - expected_value) ** 2
                    for observed_value, expected_value in zip(
                        observed_lab,
                        expected_lab,
                        strict=True,
                    )
                )
            )
        )
    return deltas


def test_render_matches_perceptual_edge_and_exact_mask_goldens(tmp_path: Path) -> None:
    source = build_golden_scene(tmp_path)
    output = tmp_path / "inspection-color.png"
    evaluator_root = tmp_path / "evaluator"
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        manifest = controller.open_scene(source)
        controller.execute(
            ViewOrbitCommand(
                request_id="golden-view",
                op="view.orbit",
                target_mm=(0, 0, 800),
                azimuth_degrees=-55,
                elevation_degrees=28,
                distance_mm=5_200,
                projection=PerspectiveProjection(focal_length_mm=62),
            )
        )
        controller.execute(
            IlluminationSetCommand(
                request_id="golden-light",
                op="illumination.set",
                illumination=PresetIllumination(preset=IlluminationPreset.NEUTRAL_STUDIO),
            )
        )
        rendered = controller.render_image(
            RenderImageCommand(
                request_id="golden-render",
                op="render.image",
                output_path=str(output),
                width=192,
                height=192,
                samples=1,
                engine=RenderEngine.EEVEE,
                style=RenderStyle.SHADED,
            ),
            evaluator_output_dir=evaluator_root,
        )

    assert rendered.evaluator is not None
    semantic_mask = semantic_component_mask(
        Path(rendered.evaluator.component_ids.path),
        rendered.evaluator.component_colors,
        manifest.components,
    )
    if os.environ.get("MESHPROBE_UPDATE_GOLDENS") == "1":
        shutil.copyfile(rendered.color.path, COLOR_GOLDEN)
        semantic_mask.save(MASK_GOLDEN)

    observed = Image.open(rendered.color.path).convert("RGB")
    expected = Image.open(COLOR_GOLDEN).convert("RGB")
    assert observed.size == expected.size == (192, 192)
    delta_e = sorted(delta_e_values(observed, expected))
    assert sum(delta_e) / len(delta_e) <= 2.0
    assert delta_e[round(0.95 * (len(delta_e) - 1))] <= 8.0
    observed_edges = observed.convert("L").filter(ImageFilter.FIND_EDGES)
    expected_edges = expected.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_difference = ImageChops.difference(observed_edges, expected_edges)
    assert ImageStat.Stat(edge_difference).mean[0] / 255 <= 0.015

    expected_mask = Image.open(MASK_GOLDEN).convert("L")
    assert semantic_mask.size == expected_mask.size
    assert semantic_mask.tobytes() == expected_mask.tobytes()


def test_display_referred_background_renders_exact_srgb(tmp_path: Path) -> None:
    source = build_golden_scene(tmp_path)
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        controller.open_scene(source)
        controller.execute(
            ViewOrbitCommand(
                request_id="bg-view",
                op="view.orbit",
                target_mm=(0, 0, 800),
                azimuth_degrees=-55,
                elevation_degrees=28,
                distance_mm=5_200,
                projection=PerspectiveProjection(focal_length_mm=62),
            )
        )

        def render_backdrop_corners(
            color: tuple[float, float, float], name: str
        ) -> set[tuple[int, int, int]]:
            controller.execute(
                IlluminationSetCommand(
                    request_id=f"bg-light-{name}",
                    op="illumination.set",
                    illumination=PresetIllumination(
                        preset=IlluminationPreset.NEUTRAL_STUDIO,
                        background_srgb=color,
                    ),
                )
            )
            output = tmp_path / f"background-{name}.png"
            controller.render_image(
                RenderImageCommand(
                    request_id=f"bg-render-{name}",
                    op="render.image",
                    output_path=str(output),
                    width=96,
                    height=96,
                    samples=1,
                    engine=RenderEngine.EEVEE,
                    style=RenderStyle.SHADED,
                )
            )
            with Image.open(output) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                return {
                    rgb.getpixel(corner)
                    for corner in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))
                }

        # 1 1 1 must clip to pure white, which the AgX view transform alone never reaches.
        assert render_backdrop_corners((1.0, 1.0, 1.0), "white") == {(255, 255, 255)}
        gray_corners = render_backdrop_corners((0.5, 0.5, 0.5), "gray")
        assert all(all(abs(channel - 128) <= 1 for channel in corner) for corner in gray_corners)


def edge_magnitude(image: Image.Image) -> float:
    return ImageStat.Stat(image.convert("L").filter(ImageFilter.FIND_EDGES)).mean[0]


def test_shaded_edges_keeps_freestyle_off_the_evaluator_passes(tmp_path: Path) -> None:
    source = build_golden_scene(tmp_path)
    evaluator_root = tmp_path / "evaluator"
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        manifest = controller.open_scene(source)
        controller.execute(
            ViewOrbitCommand(
                request_id="edges-view",
                op="view.orbit",
                target_mm=(0, 0, 800),
                azimuth_degrees=-55,
                elevation_degrees=28,
                distance_mm=5_200,
                projection=PerspectiveProjection(focal_length_mm=62),
            )
        )
        controller.execute(
            IlluminationSetCommand(
                request_id="edges-light",
                op="illumination.set",
                illumination=PresetIllumination(preset=IlluminationPreset.NEUTRAL_STUDIO),
            )
        )
        rendered = controller.render_image(
            RenderImageCommand(
                request_id="edges-render",
                op="render.image",
                output_path=str(tmp_path / "edges-color.png"),
                width=192,
                height=192,
                samples=1,
                engine=RenderEngine.EEVEE,
                style=RenderStyle.SHADED_EDGES,
            ),
            evaluator_output_dir=evaluator_root,
        )

    # The evaluator passes render with Freestyle disabled (it never writes to Depth or
    # Normal), so the exact component mask still resolves under shaded_edges.
    assert rendered.evaluator is not None
    semantic_mask = semantic_component_mask(
        Path(rendered.evaluator.component_ids.path),
        rendered.evaluator.component_colors,
        manifest.components,
    )
    assert semantic_mask.tobytes() == Image.open(MASK_GOLDEN).convert("L").tobytes()

    # Freestyle stays on for the visible color pass: its outlines lift the edge content
    # well above the plain shaded golden.
    edges = edge_magnitude(Image.open(rendered.color.path).convert("RGB"))
    shaded_edges_baseline = edge_magnitude(Image.open(COLOR_GOLDEN).convert("RGB"))
    assert edges > shaded_edges_baseline


def test_render_flags_effectively_empty_frame_when_nothing_is_shown(tmp_path: Path) -> None:
    source = build_golden_scene(tmp_path)
    with BlenderController(
        timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
        artifact_cache_root=tmp_path / "cache",
    ) as controller:
        manifest = controller.open_scene(source)
        controller.execute(
            ViewOrbitCommand(
                request_id="framed-view",
                op="view.orbit",
                target_mm=(0, 0, 800),
                azimuth_degrees=-55,
                elevation_degrees=28,
                distance_mm=5_200,
                projection=PerspectiveProjection(focal_length_mm=62),
            )
        )
        framed = controller.render_image(
            RenderImageCommand(
                request_id="framed-render",
                op="render.image",
                output_path=str(tmp_path / "framed.png"),
                width=192,
                height=192,
                samples=1,
                engine=RenderEngine.EEVEE,
            )
        )
        controller.execute(
            ComponentDisplayCommand(
                request_id="hide-all",
                op="component.display",
                component_ids=tuple(component.id for component in manifest.components),
                mode=DisplayMode.HIDDEN,
            )
        )
        blank = controller.render_image(
            RenderImageCommand(
                request_id="blank-render",
                op="render.image",
                output_path=str(tmp_path / "blank.png"),
                width=192,
                height=192,
                samples=1,
                engine=RenderEngine.EEVEE,
            )
        )

    assert framed.foreground.visible_fraction > 0.01
    assert framed.warnings == ()

    assert blank.foreground.visible_fraction == 0.0
    assert any("effectively empty" in warning for warning in blank.warnings)
