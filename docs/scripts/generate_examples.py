"""Regenerate the README example assets from a microscope frame."""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SVG_TEMPLATE = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' role='img' aria-labelledby='title desc'>\n  <title id='title'>{title}</title>\n  <desc id='desc'>{desc}</desc>\n  <image href='data:image/png;base64,{data}' x='0' y='0' width='{w}' height='{h}' preserveAspectRatio='xMidYMid meet'/>\n</svg>\n"""


@dataclass
class OverlayResult:
    image: Image.Image
    count: int
    backend: str


def _load_font(size: int = 20) -> ImageFont.ImageFont:
    for font_path in (
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _threshold_overlay(image: Image.Image, *, threshold: float = 0.55, min_pixels: int = 12) -> OverlayResult:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    normalised = (arr - arr.min()) / max(arr.ptp(), 1.0)
    mask = normalised > threshold
    visited = np.zeros_like(mask, dtype=bool)
    boxes = []
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            pixels = 0
            while stack:
                cy, cx = stack.pop()
                pixels += 1
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if pixels >= min_pixels:
                pad = 4
                boxes.append(
                    (
                        max(min_x - pad, 0),
                        max(min_y - pad, 0),
                        min(max_x + pad, width - 1),
                        min(max_y + pad, height - 1),
                    )
                )

    resized = image.resize((640, 640), Image.LANCZOS)
    scale_x = 640 / image.width
    scale_y = 640 / image.height
    draw = ImageDraw.Draw(resized)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle(
            (
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            ),
            outline=(255, 0, 255),
            width=2,
        )

    draw.rectangle((10, 10, 210, 44), fill=(0, 0, 0))
    draw.text((18, 18), f"count: {len(boxes)}", fill=(255, 255, 255), font=_load_font())
    return OverlayResult(image=resized, count=len(boxes), backend="threshold fallback")


def _model_overlay(image_path: Path, *, device: str | None, image_size: int, weights: Optional[Path]) -> Optional[OverlayResult]:
    try:
        from cell_counting import load_model
        from cell_counting.inference import predict_image
    except Exception:  # pragma: no cover - optional dependency
        return None

    try:
        model = load_model(weights_path=weights, device=device, image_size=image_size)
        result = predict_image(
            image=image_path,
            model=model,
            device=device,
            image_size=image_size,
            draw=True,
            return_image=True,
        )
    except Exception:  # pragma: no cover - depends on torch availability
        return None

    if result.image is None:
        return None

    return OverlayResult(image=result.image, count=result.count, backend="cell_counting")


def _write_svg(png_path: Path, svg_path: Path, *, title: str, desc: str) -> None:
    payload = base64.b64encode(png_path.read_bytes()).decode("ascii")
    with Image.open(png_path) as im:
        width, height = im.size
    svg_path.write_text(
        SVG_TEMPLATE.format(
            w=width,
            h=height,
            title=title,
            desc=desc,
            data=payload,
        ),
        encoding="utf-8",
    )


def generate_examples(
    source: Path,
    out_dir: Path,
    *,
    device: Optional[str],
    image_size: int,
    weights: Optional[Path],
) -> OverlayResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_image = Image.open(source).convert("RGB")
    overlay = _model_overlay(source, device=device, image_size=image_size, weights=weights)
    if overlay is None:
        overlay = _threshold_overlay(base_image)

    raw_path = out_dir / "sample_input.png"
    annotated_path = out_dir / "sample_output.png"
    base_image.resize((640, 640), Image.LANCZOS).save(raw_path)
    overlay.image.save(annotated_path)

    _write_svg(
        raw_path,
        out_dir / "sample_input.svg",
        title="Example microscope input frame",
        desc="Grayscale microscope capture showing densely packed circular cells.",
    )
    _write_svg(
        annotated_path,
        out_dir / "sample_output.svg",
        title="Example detection overlay",
        desc="Microscope capture with magenta bounding boxes drawn around detected cells and a count label.",
    )
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Microscope image used as the documentation example")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Where to place the generated PNG and SVG assets (default: docs/assets)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("results/models/best.pt"),
        help="Path to the trained weights used by the detector (default: results/models/best.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string forwarded to cell_counting.load_model (default: cpu)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Image size forwarded to the detector (default: 640)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlay = generate_examples(
        args.source,
        args.out_dir,
        device=args.device,
        image_size=args.image_size,
        weights=args.weights,
    )
    print(f"Generated overlay using the {overlay.backend} backend (count={overlay.count}).")


if __name__ == "__main__":
    main()
