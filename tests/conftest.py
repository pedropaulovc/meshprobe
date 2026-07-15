from __future__ import annotations

import hashlib

import pytest

from meshprobe.models import (
    Bounds,
    Camera,
    Component,
    PerspectiveProjection,
    Pose,
    PresetIllumination,
    SceneCapabilities,
    SceneManifest,
)
from meshprobe.selectors import stable_component_id

IDENTITY = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


@pytest.fixture
def scene_manifest() -> SceneManifest:
    source_hash = hashlib.sha256(b"fixture assembly").hexdigest()
    root_id = stable_component_id(source_hash, "assembly")
    clip_id = stable_component_id(source_hash, "assembly/cover/clip")
    idler_id = stable_component_id(source_hash, "assembly/drive/idler")
    bounds = Bounds(minimum_mm=(-10, -10, -10), maximum_mm=(10, 10, 10))
    return SceneManifest(
        source_sha256=source_hash,
        source_format="glb",
        units="millimeter",
        root_bounds=bounds,
        components=(
            Component(
                id=root_id,
                path="assembly",
                display_name="assembly",
                source_name="assembly",
                child_ids=(clip_id, idler_id),
                world_transform=IDENTITY,
                local_bounds=bounds,
                world_bounds=bounds,
            ),
            Component(
                id=clip_id,
                path="assembly/cover/clip",
                display_name="retaining clip",
                source_name="retaining clip",
                parent_id=root_id,
                world_transform=IDENTITY,
                local_bounds=bounds,
                world_bounds=bounds,
            ),
            Component(
                id=idler_id,
                path="assembly/drive/idler",
                display_name="idler",
                source_name="idler",
                parent_id=root_id,
                world_transform=IDENTITY,
                local_bounds=bounds,
                world_bounds=bounds,
            ),
        ),
        imported_camera=Camera(
            pose=Pose(position_mm=(100, 100, 100), orientation_xyzw=(0, 0, 0, 1)),
            projection=PerspectiveProjection(),
        ),
        imported_illumination=PresetIllumination(preset="neutral_studio"),
        capabilities=SceneCapabilities(
            hierarchy="preserved",
            component_names="source",
            materials="preserved",
            textures="preserved",
        ),
    )
