from __future__ import annotations

from pathlib import Path

import pytest

from meshprobe.blender_discovery import (
    MIN_SUPPORTED_BLENDER_VERSION,
    _discover_windows_blender,
    parse_blender_version,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"), (5, 2, 0)),
        (Path("C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"), (4, 5, 0)),
        (Path("C:/Program Files/Blender Foundation/Blender 5.2.1/blender.exe"), (5, 2, 1)),
        (Path("C:/Program Files/Blender Foundation/Blender 5.1 LTS/blender.exe"), (5, 1, 0)),
        (Path("D:/custom/blender-5.2.3-windows-x64/blender.exe"), (5, 2, 3)),
        (Path("C:/tools/blender.exe"), None),
        (Path("blender.exe"), None),
    ],
)
def test_parse_blender_version(path: Path, expected: tuple[int, int, int] | None) -> None:
    assert parse_blender_version(path) == expected


def test_min_supported_blender_version_is_four_two() -> None:
    # Pinned to the worker compatibility floor: Blender < 4.2 is not
    # supported by the worker script, so discovery must never silently prefer
    # an older install.
    assert MIN_SUPPORTED_BLENDER_VERSION == (4, 2)


def test_discover_windows_blender_prefers_newest_supported() -> None:
    candidates = [
        Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"),
        Path("C:/Program Files/Blender Foundation/Blender 5.3/blender.exe"),
        Path("C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"),
    ]
    assert _discover_windows_blender(candidates) == Path(
        "C:/Program Files/Blender Foundation/Blender 5.3/blender.exe"
    )


def test_discover_windows_blender_skips_unsupported_versions() -> None:
    candidates = [
        Path("C:/Program Files/Blender Foundation/Blender 3.6/blender.exe"),
        Path("C:/Program Files/Blender Foundation/Blender 4.1/blender.exe"),
    ]
    assert _discover_windows_blender(candidates) is None


def test_discover_windows_blender_ignores_unparseable_candidates() -> None:
    candidates = [
        Path("C:/tools/blender/blender.exe"),
        Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"),
    ]
    assert _discover_windows_blender(candidates) == Path(
        "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
    )


def test_discover_windows_blender_empty_candidates_returns_none() -> None:
    assert _discover_windows_blender([]) is None


def test_discover_windows_blender_boundary_version_is_supported() -> None:
    candidates = [Path("C:/Program Files/Blender Foundation/Blender 4.2/blender.exe")]
    assert _discover_windows_blender(candidates) == candidates[0]


def test_discover_windows_blender_just_below_boundary_is_unsupported() -> None:
    candidates = [Path("C:/Program Files/Blender Foundation/Blender 4.1.9/blender.exe")]
    assert _discover_windows_blender(candidates) is None


def test_discover_windows_blender_registry_style_install_location(tmp_path: Path) -> None:
    # Mirrors a real Uninstall-registry InstallLocation: the versioned
    # directory itself (see winget-pkgs#95653), joined with blender.exe by
    # `_registry_install_hints`.
    install_location = tmp_path / "Blender Foundation" / "Blender 5.2"
    install_location.mkdir(parents=True)
    candidate = install_location / "blender.exe"
    assert _discover_windows_blender([candidate]) == candidate


def test_discover_windows_blender_non_windows_platform_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import meshprobe.blender_discovery as module

    monkeypatch.setattr("meshprobe.blender_discovery.sys.platform", "linux")
    assert module.discover_windows_blender() is None
