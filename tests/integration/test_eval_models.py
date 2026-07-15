from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from meshprobe.controller import BlenderController
from meshprobe.evals.generators import GeneratorFamily, build_model, publish_model

pytestmark = pytest.mark.skipif(shutil.which("blender") is None, reason="Blender is not installed")


def test_blender_imports_every_procedural_family_with_exact_identity(tmp_path: Path) -> None:
    with BlenderController(timeout_seconds=30) as controller:
        for family in GeneratorFamily:
            published = publish_model(build_model(family, 0), tmp_path)
            manifest = controller.open_scene(published.path)

            assert manifest.source_sha256 == published.sha256
            assert {component.path for component in manifest.components} == set(
                published.component_paths.values()
            )
            assert {component.id for component in manifest.components} == set(
                published.component_ids.values()
            )


def test_missing_material_family_reports_partial_material_capability(tmp_path: Path) -> None:
    published = publish_model(build_model(GeneratorFamily.MISSING_MATERIAL, 0), tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(published.path)

    assert manifest.capabilities.materials == "partial"


def test_start_state_variant_imports_orthographic_camera_and_lights(tmp_path: Path) -> None:
    published = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 7), tmp_path)
    with BlenderController(timeout_seconds=30) as controller:
        manifest = controller.open_scene(published.path)

    assert manifest.imported_camera.projection.mode == "orthographic"
    assert manifest.imported_illumination.preset == "custom"
