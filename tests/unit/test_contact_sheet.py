from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from meshprobe.contact_sheet import compose_contact_sheet


def test_compose_contact_sheet_builds_captioned_grid(tmp_path: Path) -> None:
    panels: list[tuple[Path, str]] = []
    for index in range(9):
        path = tmp_path / f"panel-{index}.png"
        Image.new("RGB", (40, 30), (index * 20, 40, 80)).save(path)
        panels.append((path, f"Panel {index + 1}"))
    output = tmp_path / "sheet.png"

    artifact = compose_contact_sheet(tuple(panels), output, 80, 60)

    assert Image.open(output).size == (240, 276)
    assert artifact.bytes == output.stat().st_size
    assert len(artifact.sha256) == 64


def test_compose_contact_sheet_requires_nine_panels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly nine"):
        compose_contact_sheet((), tmp_path / "sheet.png", 80, 60)
