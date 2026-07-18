"""High-density contact-sheet composition."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from meshprobe.models import ContactSheetCallout, ImageArtifact
from meshprobe.sources import sha256_file


def contact_sheet_staging_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.part")


_DEFAULT_PANEL_HEIGHT_PX = 1200  # RenderContactSheetCommand.panel_height's default
_MAX_CAPTION_LINES = 4
_MAX_LEGEND_LINES = 2
# Bound on caption + legend lines together, so a panel that maxes out BOTH bands
# at once (a long custom_3x3 caption and a long callout legend, all nine panels)
# cannot push every row to its own worst case simultaneously and bust the ~4K
# long-edge budget the default panel size was chosen for. The caption keeps its
# own full _MAX_CAPTION_LINES allowance always (preserving the default
# occluder-removal caption in full); only the legend gives up room when the
# caption is already at its cap.
_MAX_BAND_LINES = 5


def _caption_line_caps(panel_height: int) -> tuple[int, int, int]:
    """Scale the caption/legend/band line caps for a panel taller than the
    default the ~4K sheet-height budget above was calibrated for. A caller who
    explicitly opts into a larger panel_height (e.g. custom_3x3 with
    --panel-height 4096) has already opted into a bigger sheet, so holding
    them to the same fixed line budget as the 1200px default would silently
    drop occluder names or custom caption text the enlarged panel has room to
    show. Panels at or below the default keep today's exact caps."""
    scale = max(1.0, panel_height / _DEFAULT_PANEL_HEIGHT_PX)
    return (
        round(_MAX_CAPTION_LINES * scale),
        round(_MAX_LEGEND_LINES * scale),
        round(_MAX_BAND_LINES * scale),
    )


_CAPTION_TRUNCATION_NOTE = "(see render metadata for full text)"


def _cap_lines(lines: tuple[str, ...], limit: int) -> tuple[str, ...]:
    """Bound how many lines a caption band contributes, so an arbitrarily long
    component display name or callout legend cannot grow the composed sheet past
    its intended size."""
    if limit <= 0:
        return ()
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


def _prepare_caption_lines(
    draw: ImageDraw.ImageDraw,
    caption: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    width: int,
    max_lines: int,
) -> tuple[str, ...]:
    """Wrap `caption` for the caption band, capped at `max_lines`.

    Every paragraph (a `\\n`-separated entry, e.g. one line per removed
    occluder) is wrapped in full first. If the total physical-line count
    already fits `max_lines` — including a caller-scaled budget for a larger
    panel — nothing is truncated at all, so an enlarged sheet still gets the
    room its scaled cap grants it.

    Only when the full wrap doesn't fit does a fallback kick in: each
    paragraph is truncated to at most ONE physical line, so a single long
    entry can't consume the whole budget and crowd its siblings out of view
    entirely (the ellipsis on an entry still marks that entry as cut). If
    that STILL doesn't fit (more entries than `max_lines` even at one line
    each), trailing entries are dropped too. Whenever anything was cut this
    way, an extra note line is appended pointing at the untruncated data
    (`panel.caption` in the render result always holds every entry in full;
    only what's burned into the image can be capped) — added ON TOP of
    `max_lines`, never by displacing an entry that would otherwise have fit,
    so a default-budget entry is never silently dropped just to make room for
    the note. A single-paragraph caption with no `\\n` keeps the previous
    plain wrap-then-cap behavior, since there are no sibling entries it could
    crowd out.
    """
    paragraphs = caption.splitlines() or [""]
    if len(paragraphs) <= 1:
        return _cap_lines(_wrap_text(draw, caption, font, width), max_lines)
    wrapped_paragraphs = tuple(_wrap_text(draw, p, font, width) or ("",) for p in paragraphs)
    full_lines = tuple(line for wrapped in wrapped_paragraphs for line in wrapped)
    if len(full_lines) <= max_lines:
        return full_lines
    single_line_paragraphs = [
        f"{wrapped[0].rstrip()}…" if len(wrapped) > 1 else wrapped[0]
        for wrapped in wrapped_paragraphs
    ]
    count_truncated = len(single_line_paragraphs) > max_lines
    shown = single_line_paragraphs[:max_lines]
    if count_truncated and shown and not shown[-1].endswith("…"):
        shown = [*shown[:-1], f"{shown[-1].rstrip()}…"]
    return (*shown, _CAPTION_TRUNCATION_NOTE)


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
    max_caption_lines, max_legend_lines, max_band_lines = _caption_line_caps(panel_height)
    panel_lines: list[tuple[str, ...]] = []
    panel_caption_line_counts: list[int] = []
    for panel_index, (_path, caption, callouts) in enumerate(panels):
        numbered_caption = f"{panel_index + 1}. {caption}"
        # Cap the caption and the legend separately so a legitimate default-sized
        # caption (e.g. the focused sheet's "Occluders removed:" header plus one
        # line per removed blocker, up to occluder_budget's default of 3) is never
        # truncated by an unrelated long callout legend; only each band's own
        # pathological growth (a very long display name, a very long legend) gets
        # truncated.
        capped_caption_lines = _prepare_caption_lines(
            measuring_draw, numbered_caption, font, panel_width - 12, max_caption_lines
        )
        lines = list(capped_caption_lines)
        if callouts:
            legend = " | ".join(f"{callout.number} {callout.label}" for callout in callouts)
            legend_lines = _wrap_text(measuring_draw, legend, font, panel_width - 12)
            legend_limit = min(max_legend_lines, max_band_lines - len(capped_caption_lines))
            lines.extend(_cap_lines(legend_lines, legend_limit))
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
