"""High-density contact-sheet composition."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from meshprobe.models import ContactSheetCallout, ImageArtifact
from meshprobe.sources import sha256_file


def contact_sheet_staging_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.part")


def compose_contact_sheet(
    panels: tuple[tuple[Path, str, tuple[ContactSheetCallout, ...]], ...],
    output_path: Path,
    panel_width: int,
    panel_height: int,
) -> ImageArtifact:
    if len(panels) != 9:
        raise ValueError("focused_3x3 requires exactly nine panels")
    caption_height = max(48, panel_height // 10)
    sheet = Image.new("RGB", (panel_width * 3, (panel_height + caption_height) * 3), "#17191d")
    font = ImageFont.load_default(size=max(14, caption_height // 2))
    for panel_index, (path, caption, callouts) in enumerate(panels):
        column = panel_index % 3
        row = panel_index // 3
        left = column * panel_width
        top = row * (panel_height + caption_height)
        with Image.open(path) as source:
            fitted = ImageOps.fit(source.convert("RGB"), (panel_width, panel_height))
        callout_draw = ImageDraw.Draw(fitted)
        radius = max(8, min(panel_width, panel_height) // 35)
        for callout in callouts:
            x = round(callout.image_xy[0] * (panel_width - 1))
            y = round((1 - callout.image_xy[1]) * (panel_height - 1))
            callout_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill="#ffd43b",
                outline="#111318",
                width=max(1, radius // 4),
            )
            callout_draw.text(
                (x, y),
                str(callout.number),
                fill="#111318",
                font=font,
                anchor="mm",
            )
        sheet.paste(fitted, (left, top))
        caption_layer = Image.new("RGB", (panel_width, caption_height), "#17191d")
        caption_draw = ImageDraw.Draw(caption_layer)
        caption_y = caption_height // 2 if not callouts else caption_height // 3
        caption_draw.text(
            (10, caption_y),
            f"{panel_index + 1}. {caption}",
            fill="#f2f4f8",
            font=font,
            anchor="lm",
        )
        if callouts:
            legend = " | ".join(
                f"{callout.number} {callout.label}" for callout in callouts
            )
            caption_draw.text(
                (10, caption_height * 2 // 3),
                legend,
                fill="#ffd43b",
                font=font,
                anchor="lm",
            )
        sheet.paste(caption_layer, (left, top + panel_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = contact_sheet_staging_path(output_path)
    sheet.save(staging, format="PNG", optimize=True)
    staging.replace(output_path)
    return ImageArtifact(
        path=str(output_path),
        media_type="image/png",
        sha256=sha256_file(output_path),
        bytes=output_path.stat().st_size,
    )
