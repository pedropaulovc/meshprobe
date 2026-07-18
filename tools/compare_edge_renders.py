from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

LABEL_HEIGHT = 52
FREESTYLE_ONLY = (255, 0, 180)
SCREEN_ONLY = (0, 220, 255)
SHARED = (255, 255, 255)


def changed_pixels(base: Image.Image, candidate: Image.Image, threshold: int) -> set[int]:
    return {
        index
        for index, (left, right) in enumerate(
            zip(base.get_flattened_data(), candidate.get_flattened_data(), strict=True)
        )
        if max(abs(a - b) for a, b in zip(left, right, strict=True)) >= threshold
    }


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def labeled_pair(
    left: Image.Image,
    right: Image.Image,
    left_label: str,
    right_label: str,
) -> Image.Image:
    canvas = Image.new("RGB", (left.width + right.width, left.height + LABEL_HEIGHT), "#181818")
    canvas.paste(left, (0, LABEL_HEIGHT))
    canvas.paste(right, (left.width, LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    label_font = font(18)
    draw.text((12, 14), left_label, fill="white", font=label_font)
    draw.text((left.width + 12, 14), right_label, fill="white", font=label_font)
    return canvas


def edge_bounds(
    indices: set[int],
    width: int,
    height: int,
    padding: int = 32,
) -> tuple[int, int, int, int]:
    xs = [index % width for index in indices]
    ys = [index // width for index in indices]
    return (
        max(0, min(xs) - padding),
        max(0, min(ys) - padding),
        min(width, max(xs) + padding + 1),
        min(height, max(ys) + padding + 1),
    )


def difference_overlay(
    base: Image.Image,
    freestyle: set[int],
    screen: set[int],
) -> Image.Image:
    dimmed = ImageOps.grayscale(base).point(lambda value: value // 4).convert("RGB")
    pixels = dimmed.load()
    width = dimmed.width
    for index in freestyle | screen:
        coordinate = index % width, index // width
        if index in freestyle and index in screen:
            pixels[coordinate] = SHARED
            continue
        if index in freestyle:
            pixels[coordinate] = FREESTYLE_ONLY
            continue
        pixels[coordinate] = SCREEN_ONLY
    return dimmed


def labeled_overlay(overlay: Image.Image) -> Image.Image:
    entries = (
        ("both", SHARED),
        ("Freestyle only", FREESTYLE_ONLY),
        ("screen only", SCREEN_ONLY),
    )
    compact = overlay.width < 500
    label_height = 82 if compact else LABEL_HEIGHT
    canvas = Image.new("RGB", (overlay.width, overlay.height + label_height), "#181818")
    canvas.paste(overlay, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    label_font = font(15)
    if compact:
        for row, (label, color) in enumerate(entries):
            y = 8 + row * 24
            draw.rectangle((12, y + 3, 26, y + 17), fill=color)
            draw.text((32, y), label, fill="white", font=label_font)
        return canvas

    x = 12
    for label, color in entries:
        draw.rectangle((x, 17, x + 14, 31), fill=color)
        x += 20
        draw.text((x, 14), label, fill="white", font=label_font)
        x += int(draw.textlength(label, font=label_font)) + 18
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shaded", type=Path, required=True)
    parser.add_argument("--freestyle", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=24)
    parser.add_argument("--freestyle-seconds", type=float, required=True)
    parser.add_argument("--screen-seconds", type=float, required=True)
    args = parser.parse_args()

    images = {
        "shaded": Image.open(args.shaded).convert("RGB"),
        "freestyle": Image.open(args.freestyle).convert("RGB"),
        "screen": Image.open(args.screen).convert("RGB"),
    }
    sizes = {image.size for image in images.values()}
    if len(sizes) != 1:
        raise ValueError(f"input dimensions differ: {sorted(sizes)}")

    freestyle = changed_pixels(images["shaded"], images["freestyle"], args.threshold)
    screen = changed_pixels(images["shaded"], images["screen"], args.threshold)
    shared = freestyle & screen
    union = freestyle | screen
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    left_label = f"Freestyle  {args.freestyle_seconds:.2f} s"
    right_label = f"Screen space  {args.screen_seconds:.2f} s"
    labeled_pair(images["freestyle"], images["screen"], left_label, right_label).save(
        output / "side-by-side.png"
    )
    crop = edge_bounds(union, images["shaded"].width, images["shaded"].height)
    labeled_pair(
        images["freestyle"].crop(crop),
        images["screen"].crop(crop),
        left_label,
        right_label,
    ).save(output / "side-by-side-crop.png")

    overlay = difference_overlay(images["shaded"], freestyle, screen)
    labeled_overlay(overlay).save(output / "edge-difference.png")
    labeled_overlay(overlay.crop(crop)).save(output / "edge-difference-crop.png")

    metrics = {
        "threshold": args.threshold,
        "total_pixels": images["shaded"].width * images["shaded"].height,
        "freestyle_changed_pixels": len(freestyle),
        "screen_changed_pixels": len(screen),
        "shared_changed_pixels": len(shared),
        "freestyle_only_pixels": len(freestyle - screen),
        "screen_only_pixels": len(screen - freestyle),
        "intersection_over_union": len(shared) / len(union),
        "freestyle_pixels_recalled_by_screen": len(shared) / len(freestyle),
        "freestyle_seconds": args.freestyle_seconds,
        "screen_seconds": args.screen_seconds,
        "speedup": args.freestyle_seconds / args.screen_seconds,
        "crop": crop,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
