from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "tools" / "compare_edge_renders.py"


def run_comparison(tmp_path: Path, screen_color: str) -> dict[str, object]:
    shaded = tmp_path / "shaded.png"
    freestyle = tmp_path / "freestyle.png"
    screen = tmp_path / "screen.png"
    output = tmp_path / "comparison"
    Image.new("RGB", (4, 3), "white").save(shaded)
    Image.new("RGB", (4, 3), "white").save(freestyle)
    Image.new("RGB", (4, 3), screen_color).save(screen)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--shaded",
            str(shaded),
            "--freestyle",
            str(freestyle),
            "--screen",
            str(screen),
            "--output-dir",
            str(output),
            "--threshold",
            "24",
            "--freestyle-seconds",
            "2",
            "--screen-seconds",
            "1",
        ],
        check=True,
    )
    assert (output / "side-by-side-crop.png").is_file()
    assert (output / "edge-difference-crop.png").is_file()
    return json.loads((output / "metrics.json").read_text(encoding="utf-8"))


def test_comparison_handles_no_changed_pixels(tmp_path: Path) -> None:
    metrics = run_comparison(tmp_path, "white")

    assert metrics["crop"] == [0, 0, 4, 3]
    assert metrics["intersection_over_union"] == 0
    assert metrics["freestyle_pixels_recalled_by_screen"] == 0


def test_comparison_handles_no_freestyle_pixels(tmp_path: Path) -> None:
    metrics = run_comparison(tmp_path, "black")

    assert metrics["screen_changed_pixels"] == 12
    assert metrics["freestyle_pixels_recalled_by_screen"] == 0
