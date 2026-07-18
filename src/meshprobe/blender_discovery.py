"""Best-effort discovery of an installed Blender when it isn't on PATH.

Windows-only. Blender's official Windows installer (an MSI --
`blender-<version>-windows-x64.msi`, per winget-pkgs#95653) does not add
`blender.exe` to PATH, so `shutil.which("blender")` predictably fails even on a
machine with Blender properly installed (meshprobe issue #95). Before giving
up, this module probes two independent, documented sources for a real
install:

1. The standard per-version install directory Blender's installer uses --
   `%ProgramFiles%\\Blender Foundation\\Blender <X.Y>\\blender.exe`. The
   Blender manual documents the install root as `Blender Foundation\\Blender`;
   in practice each installed "medium" version (5.1, 5.2, ...) gets its own
   `Blender <X.Y>` subfolder (see winget-pkgs#95653, which documents this
   exact per-version layout as the cause of a duplicate-install bug report).
2. The standard Windows "Uninstall" registry key that every MSI installer --
   including Blender's -- registers an entry under for Add/Remove Programs:
   `HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\<ProductCode>`
   (and its `WOW6432Node` mirror on 64-bit Windows), carrying `DisplayName`
   and `InstallLocation` values. See
   https://learn.microsoft.com/en-us/windows/win32/msi/uninstall-registry-key.
   This is the general MSI convention, not Blender-specific folklore, so it
   stays correct even if a future installer build moves the "Blender
   Foundation" folder layout.

Among everything found, prefers the newest version >= MIN_SUPPORTED_BLENDER_VERSION
(5.2, the real minimum per issue #93's unsupported-version handling) rather than
the newest overall -- an older, unsupported Blender found this way would just
fail later with a more confusing error than a plain "not found".

NOTE ON VERIFICATION: `_discover_windows_blender` and `parse_blender_version`
are pure and unit-tested with fabricated paths (see
`tests/unit/test_blender_discovery.py`). The filesystem-globbing
(`_standard_install_globs`) and registry-reading (`_registry_install_hints`)
halves that feed them are Windows-only and are NOT exercised by this
(Linux/Mac) development environment -- they are exercised for real by this
repo's Windows CI job.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "MIN_SUPPORTED_BLENDER_VERSION",
    "BlenderVersion",
    "discover_windows_blender",
    "parse_blender_version",
]

# Kept in sync with issue #93's unsupported-Blender-version handling: Blender
# older than 5.2 is not supported by meshprobe's worker script, so
# auto-discovery must not silently pick an older install and fail later with a
# confusing downstream error.
MIN_SUPPORTED_BLENDER_VERSION: tuple[int, int] = (5, 2)

BlenderVersion = tuple[int, int, int]

_VERSION_PATTERN = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def parse_blender_version(path: Path) -> BlenderVersion | None:
    """Extract a (major, minor, patch) version from a Blender install path.

    Blender's Windows installer names both the per-version Program Files
    folder (`Blender Foundation\\Blender 5.2\\blender.exe`) and, per the
    standard MSI Uninstall registry convention, the registered
    `InstallLocation` after the same "Blender <X.Y>" pattern -- so scanning
    path components for the first `major.minor[.patch]` run covers both
    discovery sources. Returns None when no path component looks like a
    version.
    """

    for part in (path.name, path.parent.name, *(p.name for p in path.parents)):
        match = _VERSION_PATTERN.search(part)
        if match is not None:
            major, minor, patch = match.groups()
            return (int(major), int(minor), int(patch or 0))
    return None


def _discover_windows_blender(candidate_paths: list[Path]) -> Path | None:
    """Pick the newest *supported* Blender executable among candidate paths.

    Pure and platform-independent -- never touches the filesystem, PATH, or
    registry, so it is fully unit-testable with fabricated paths. Candidates
    whose version can't be parsed, or that parse below
    MIN_SUPPORTED_BLENDER_VERSION, are ignored: we would rather fall through
    to the ordinary "not found" error (which names --blender) than silently
    launch an unsupported Blender that then fails with a more confusing
    worker-side error (see issue #93).
    """

    supported = [
        (version, path)
        for path in candidate_paths
        if (version := parse_blender_version(path)) is not None
        and version[:2] >= MIN_SUPPORTED_BLENDER_VERSION
    ]
    if not supported:
        return None
    supported.sort(key=lambda item: item[0], reverse=True)
    return supported[0][1]


def _standard_install_globs() -> list[Path]:
    """Candidate `blender.exe` paths under the standard Windows install roots."""

    roots = []
    for env_var in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
        value = os.environ.get(env_var)
        if value:
            roots.append(Path(value) / "Blender Foundation")
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        candidates.extend(root.glob("Blender*/blender.exe"))
    return candidates


def _registry_install_hints() -> list[Path]:  # pragma: no cover - exercised on Windows CI only
    """`blender.exe` paths hinted by the Windows Uninstall registry key.

    Best-effort: any registry access failure yields no hints (rather than
    raising), since this is a supplement to the Program Files scan above, not
    the only discovery path.

    `winreg` is stdlib-but-Windows-only, so typeshed only declares its
    attributes when mypy's assumed platform is "win32" (it isn't, in this
    Linux/Mac dev environment) -- `winreg_module` sidesteps that by importing
    dynamically and typing the result as `Any`, rather than requiring a
    project-wide `platform = "win32"` mypy override that would affect every
    other module.
    """

    if sys.platform != "win32":
        return []

    winreg_module: Any = importlib.import_module("winreg")

    subkeys = (
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
    )
    hints: list[Path] = []
    for subkey in subkeys:
        try:
            uninstall_key = winreg_module.OpenKey(winreg_module.HKEY_LOCAL_MACHINE, subkey)
        except OSError:
            continue
        with uninstall_key:
            index = 0
            while True:
                try:
                    entry_name = winreg_module.EnumKey(uninstall_key, index)
                except OSError:
                    break
                index += 1
                hint = _registry_entry_blender_hint(winreg_module, uninstall_key, entry_name)
                if hint is not None:
                    hints.append(hint)
    return hints


def _registry_entry_blender_hint(
    winreg_module: Any, uninstall_key: Any, entry_name: str
) -> Path | None:  # pragma: no cover - exercised on Windows CI only
    try:
        with winreg_module.OpenKey(uninstall_key, entry_name) as entry_key:
            display_name = winreg_module.QueryValueEx(entry_key, "DisplayName")[0]
            if not isinstance(display_name, str) or "blender" not in display_name.lower():
                return None
            install_location = winreg_module.QueryValueEx(entry_key, "InstallLocation")[0]
    except OSError:
        return None
    if not isinstance(install_location, str) or not install_location:
        return None
    return Path(install_location) / "blender.exe"


def discover_windows_blender() -> Path | None:
    """Best-effort: find an installed, supported Blender executable on Windows.

    Returns None on any non-Windows platform, when no install is found, or
    when every install found is below MIN_SUPPORTED_BLENDER_VERSION --
    callers should fall back to the ordinary "not found" error in every such
    case. Never raises: filesystem/registry access failures are swallowed,
    since this is purely a convenience over the explicit --blender flag /
    MESHPROBE_BLENDER env var, not a required path.
    """

    if sys.platform != "win32":
        return None
    candidates = [
        path for path in (*_standard_install_globs(), *_registry_install_hints()) if path.is_file()
    ]
    return _discover_windows_blender(candidates)
