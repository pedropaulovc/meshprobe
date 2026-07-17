"""High-density contact-sheet composition."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from meshprobe.models import ContactSheetCallout, ImageArtifact
from meshprobe.sources import sha256_file


def contact_sheet_staging_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.part")


_MAX_CAPTION_LINES = 4
_MAX_LEGEND_LINES = 2


def _cap_lines(lines: tuple[str, ...], limit: int) -> tuple[str, ...]:
    """Bound how many lines a caption band contributes, so an arbitrarily long
    component display name or callout legend cannot grow the composed sheet past
    its intended size."""
    if len(lines) <= limit:
        return lines
    truncated = list(lines[:limit])
    truncated[-1] = f"{truncated[-1].rstrip()}…"
    return tuple(truncated)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    width: int,
) -> tuple[str, ...]:
    """Wrap every character into the image instead of clipping long identifiers."""
    lines: list[str] = []
    for paragraph in text.splitlines() or ("",):
        remaining = paragraph
        if not remaining:
            lines.append("")
            continue
        while remaining:
            end = len(remaining)
            while end > 1 and draw.textlength(remaining[:end], font=font) > width:
                end -= 1
            if end < len(remaining):
                word_end = remaining.rfind(" ", 0, end + 1)
                if word_end > 0:
                    end = word_end
            lines.append(remaining[:end].rstrip())
            remaining = remaining[end:].lstrip()
    return tuple(lines)


def compose_contact_sheet(
    panels: tuple[tuple[Path, str, tuple[ContactSheetCallout, ...]], ...],
    output_path: Path,
    panel_width: int,
    panel_height: int,
) -> ImageArtifact:
    if len(panels) != 9:
        raise ValueError("focused_3x3 requires exactly nine panels")
    minimum_caption_height = max(48, panel_height // 10)
    font_size = max(14, min(minimum_caption_height // 4, panel_width // 12))
    font = ImageFont.load_default(size=font_size)
    measuring_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_box = measuring_draw.textbbox((0, 0), "Ag", font=font)
    line_height = int(line_box[3] - line_box[1] + 3)
    panel_lines: list[tuple[str, ...]] = []
    panel_caption_line_counts: list[int] = []
    for panel_index, (_path, caption, callouts) in enumerate(panels):
        numbered_caption = f"{panel_index + 1}. {caption}"
        caption_lines = _wrap_text(measuring_draw, numbered_caption, font, panel_width - 12)
        # Cap the caption and the legend separately so a legitimate default-sized
        # caption (e.g. the focused sheet's "Occluders removed:" header plus one
        # line per removed blocker, up to occluder_budget's default of 3) is never
        # truncated by an unrelated long callout legend; only each band's own
        # pathological growth (a very long display name, a very long legend) gets
        # truncated.
        capped_caption_lines = _cap_lines(caption_lines, _MAX_CAPTION_LINES)
        lines = list(capped_caption_lines)
        if callouts:
            legend = " | ".join(f"{callout.number} {callout.label}" for callout in callouts)
            legend_lines = _wrap_text(measuring_draw, legend, font, panel_width - 12)
            lines.extend(_cap_lines(legend_lines, _MAX_LEGEND_LINES))
        panel_caption_line_counts.append(len(capped_caption_lines))
        panel_lines.append(tuple(lines))
    row_caption_heights = tuple(
        max(
            minimum_caption_height,
            max(len(panel_lines[index]) for index in range(row * 3, row * 3 + 3)) * line_height
            + 12,
        )
        for row in range(3)
    )
    sheet = Image.new(
        "RGB",
        (panel_width * 3, panel_height * 3 + sum(row_caption_heights)),
        "#17191d",
    )
    for panel_index, (path, _caption, callouts) in enumerate(panels):
        column = panel_index % 3
        row = panel_index // 3
        left = column * panel_width
        top = row * panel_height + sum(row_caption_heights[:row])
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
        caption_height = row_caption_heights[row]
        caption_layer = Image.new("RGB", (panel_width, caption_height), "#17191d")
        caption_draw = ImageDraw.Draw(caption_layer)
        caption_line_count = panel_caption_line_counts[panel_index]
        for line_index, line in enumerate(panel_lines[panel_index]):
            caption_draw.text(
                (6, 6 + line_index * line_height),
                line,
                fill="#f2f4f8" if line_index < caption_line_count else "#ffd43b",
                font=font,
                anchor="la",
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
