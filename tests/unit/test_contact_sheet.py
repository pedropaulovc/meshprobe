from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from meshprobe.contact_sheet import compose_contact_sheet
from meshprobe.models import ContactSheetCallout
from meshprobe.protocol import RenderContactSheetCommand


def test_compose_contact_sheet_builds_captioned_grid(tmp_path: Path) -> None:
    panels: list[tuple[Path, str, tuple[ContactSheetCallout, ...]]] = []
    for index in range(9):
        path = tmp_path / f"panel-{index}.png"
        Image.new("RGB", (40, 30), (index * 20, 40, 80)).save(path)
        panels.append(
            (
                path,
                f"Panel {index + 1}",
                (
                    ContactSheetCallout(
                        number=1,
                        component_id="cmp_target",
                        label="target",
                        image_xy=(0.5, 0.5),
                    ),
                )
                if index == 0
                else (),
            )
        )
    output = tmp_path / "sheet.png"

    artifact = compose_contact_sheet(tuple(panels), output, 80, 60)

    assert Image.open(output).size == (240, 324)
    assert artifact.bytes == output.stat().st_size
    assert len(artifact.sha256) == 64


def test_compose_contact_sheet_requires_nine_panels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly nine"):
        compose_contact_sheet((), tmp_path / "sheet.png", 80, 60)


def test_compose_contact_sheet_expands_to_render_every_caption_line(tmp_path: Path) -> None:
    names = tuple(f"blocker-{index}-" + "x" * 80 for index in range(1, 10))
    caption = "Occluders removed:\n" + "\n".join(
        f"{index}. {name}" for index, name in enumerate(names, 1)
    )
    panels = []
    for index in range(9):
        path = tmp_path / f"long-panel-{index}.png"
        Image.new("RGB", (80, 60), "black").save(path)
        panels.append((path, caption if index == 2 else "Panel", ()))

    output = tmp_path / "long-sheet.png"
    compose_contact_sheet(tuple(panels), output, 80, 60)

    assert Image.open(output).height > 324
    assert all(name in caption for name in names)


def test_compose_contact_sheet_expands_for_full_callout_label(tmp_path: Path) -> None:
    label = "selected-component-" + "x" * 230
    callout = ContactSheetCallout(
        number=1,
        component_id="cmp_target",
        label=label,
        image_xy=(0.5, 0.5),
    )
    panels = []
    for index in range(9):
        path = tmp_path / f"callout-panel-{index}.png"
        Image.new("RGB", (80, 60), "black").save(path)
        panels.append((path, "Panel", (callout,) if index == 0 else ()))

    output = tmp_path / "callout-sheet.png"
    compose_contact_sheet(tuple(panels), output, 80, 60)

    assert Image.open(output).height > 324


def test_compose_contact_sheet_wraps_long_single_line_caption(tmp_path: Path) -> None:
    panels = []
    for index in range(9):
        path = tmp_path / f"caption-panel-{index}.png"
        Image.new("RGB", (80, 60), "black").save(path)
        panels.append((path, "x" * 256 if index == 0 else "Panel", ()))

    output = tmp_path / "caption-sheet.png"
    compose_contact_sheet(tuple(panels), output, 80, 60)

    assert Image.open(output).height > 324


def test_compose_contact_sheet_preserves_default_occluder_removal_caption(
    tmp_path: Path,
) -> None:
    default_budget = RenderContactSheetCommand.model_fields["occluder_budget"].default
    callout = ContactSheetCallout(
        number=1,
        component_id="cmp_target",
        label="target",
        image_xy=(0.5, 0.5),
    )

    def occluder_caption(count: int) -> str:
        names = tuple(f"blocker-{index}" for index in range(1, count + 1))
        return "Occluders removed:\n" + "\n".join(
            f"{index}. {name}" for index, name in enumerate(names, 1)
        )

    default_panel = RenderContactSheetCommand.model_fields["panel_width"].default

    def build_sheet(name: str, count: int) -> Image.Image:
        panels = []
        for index in range(9):
            path = tmp_path / f"{name}-panel-{index}.png"
            Image.new("RGB", (8, 8), "black").save(path)
            panels.append(
                (
                    path,
                    occluder_caption(count) if index == 2 else "Panel",
                    (callout,) if index == 2 else (),
                )
            )
        output = tmp_path / f"{name}.png"
        compose_contact_sheet(tuple(panels), output, default_panel, default_panel)
        return Image.open(output)

    # A legitimate default-sized "Occluders removed:" caption (header + one line
    # per removed blocker, up to occluder_budget's default) must survive the
    # per-band caption cap in full, uncapped by an unrelated callout legend
    # sharing the same panel: rendering one fewer removed blocker must still
    # shrink the composed sheet, proving the default budget's last line was not
    # already being silently dropped by the cap.
    at_default_budget = build_sheet("at-default", default_budget)
    one_below_default_budget = build_sheet("below-default", default_budget - 1)
    assert at_default_budget.height > one_below_default_budget.height


def test_compose_contact_sheet_caps_caption_growth_at_default_panel_size(tmp_path: Path) -> None:
    long_label = "selected-component-" + "x" * 230
    callouts = tuple(
        ContactSheetCallout(
            number=number,
            component_id=f"cmp_{number}",
            label=long_label,
            image_xy=(0.5, 0.5),
        )
        for number in range(1, 4)
    )
    panels = []
    default_panel = RenderContactSheetCommand.model_fields["panel_width"].default
    for index in range(9):
        path = tmp_path / f"budget-panel-{index}.png"
        Image.new("RGB", (8, 8), "black").save(path)
        panels.append((path, long_label if index == 0 else "Panel", callouts if index == 0 else ()))

    output = tmp_path / "budget-sheet.png"
    compose_contact_sheet(tuple(panels), output, default_panel, default_panel)

    # An arbitrarily long display name / callout legend must not grow the composed
    # sheet past the ~4K long-edge budget the default panel size was chosen for.
    assert Image.open(output).height <= 4_096


def test_compose_contact_sheet_bounds_font_for_tall_narrow_panels(tmp_path: Path) -> None:
    label = "selected-component-" + "x" * 230
    callout = ContactSheetCallout(
        number=1,
        component_id="cmp_target",
        label=label,
        image_xy=(0.5, 0.5),
    )
    panels = []
    for index in range(9):
        path = tmp_path / f"tall-panel-{index}.png"
        Image.new("RGB", (8, 8), "black").save(path)
        panels.append((path, label, (callout,) if index == 0 else ()))

    output = tmp_path / "tall-sheet.png"
    compose_contact_sheet(tuple(panels), output, 128, 4096)

    assert Image.open(output).height < 4 * 4096
