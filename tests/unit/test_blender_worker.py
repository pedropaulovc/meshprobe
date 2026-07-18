"""Unit tests for the pure (bpy/gpu-free) pieces of ``meshprobe.blender.worker``.

``worker.py`` runs inside Blender's bundled Python and imports ``bpy``/``gpu``
at module scope, so it cannot be imported directly in this pure-Python test
environment. We load it via ``importlib`` with lightweight stand-in modules
installed in ``sys.modules`` for ``bpy``, ``gpu``, ``bpy_extras.object_utils``
and ``mathutils`` -- just enough for the module-level imports to resolve.
Nothing under test here touches those stand-ins; ``require_supported_blender_
version`` is a pure function over a version tuple.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

WORKER_PATH = Path(__file__).resolve().parents[2] / "src" / "meshprobe" / "blender" / "worker.py"


def _install_stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


@pytest.fixture
def worker_module() -> Iterator[types.ModuleType]:
    """Import worker.py under a throwaway name with bpy/gpu stubbed out."""
    saved = {
        name: sys.modules.get(name)
        for name in ("bpy", "gpu", "bpy_extras", "bpy_extras.object_utils", "mathutils")
    }
    bpy_stub = _install_stub_module("bpy")
    bpy_stub.types = types.SimpleNamespace()  # type: ignore[attr-defined]
    _install_stub_module("gpu")
    bpy_extras_stub = _install_stub_module("bpy_extras")
    object_utils_stub = _install_stub_module("bpy_extras.object_utils")
    object_utils_stub.world_to_camera_view = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    bpy_extras_stub.object_utils = object_utils_stub  # type: ignore[attr-defined]
    mathutils_stub = _install_stub_module("mathutils")
    mathutils_stub.Matrix = object  # type: ignore[attr-defined]
    mathutils_stub.Quaternion = object  # type: ignore[attr-defined]
    mathutils_stub.Vector = object  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location("meshprobe_worker_under_test", WORKER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_minimum_blender_version_is_5_2(worker_module: types.ModuleType) -> None:
    assert worker_module.MINIMUM_BLENDER_VERSION == (5, 2)


@pytest.mark.parametrize(
    "version",
    [
        (4, 5, 10),
        (5, 0, 0),
        (5, 1, 2),
        (5, 1, 999),
    ],
)
def test_require_supported_blender_version_rejects_old_versions(
    worker_module: types.ModuleType, version: tuple[int, int, int]
) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        worker_module.require_supported_blender_version(version)
    message = str(excinfo.value)
    detected = ".".join(str(component) for component in version)
    assert detected in message
    assert "MeshProbe requires Blender >= 5.2" in message
    assert "5.2 LTS" in message


@pytest.mark.parametrize(
    "version",
    [
        (5, 2, 0),
        (5, 2, 1),
        (5, 3, 0),
        (6, 0, 0),
    ],
)
def test_require_supported_blender_version_accepts_supported_versions(
    worker_module: types.ModuleType, version: tuple[int, int, int]
) -> None:
    worker_module.require_supported_blender_version(version)  # must not raise


def test_initialize_graphics_platform_checks_version_before_gpu_init(
    worker_module: types.ModuleType,
) -> None:
    """The version check must run before any gpu.* call, so an old Blender

    fails with the clear RuntimeError instead of an AttributeError from
    inside gpu.init() (issue #93).
    """
    worker_module.bpy.app = types.SimpleNamespace(version=(5, 1, 2))
    with pytest.raises(RuntimeError, match=r"Blender 5\.1\.2 is unsupported"):
        worker_module.initialize_graphics_platform()
