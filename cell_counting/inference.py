"""Inference helpers extracted from the original notebook."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING

import torch
from torch import nn
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw, ImageFont

from .model import PredictionResult

if TYPE_CHECKING:
    from .model import CellCountingModel

PathLike = Union[str, Path]
DeviceLike = Union[str, torch.device]


def _ensure_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _load_font() -> ImageFont.ImageFont:
    for font_path in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_path, 18)
        except OSError:
            continue
    return ImageFont.load_default()


def predict_image(
    image: Union[PathLike, Image.Image],
    model: Union[nn.Module, "CellCountingModel"],
    device: Optional[DeviceLike] = None,
    image_size: int = 640,
    conf: float = 0.15,
    nms_iou: float = 0.45,
    size_min: Optional[float] = 12.0 * 12.0,
    size_max: Optional[float] = 80.0 * 80.0,
    draw: bool = False,
    out_path: Optional[PathLike] = None,
    return_image: bool = False,
) -> PredictionResult:
    """Run a forward pass on a single image and optionally annotate it."""

    if hasattr(model, "model") and hasattr(model, "device"):
        inner_model = model.model  # type: ignore[attr-defined]
        device_obj = model.device  # type: ignore[attr-defined]
    else:
        inner_model = model
        device_obj = torch.device("cpu") if device is None else torch.device(device)

    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    resized = pil_image.resize((image_size, image_size), Image.BILINEAR)
    tensor = to_tensor(resized).to(device_obj)
    with torch.no_grad():
        prediction = inner_model([tensor])[0]

    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()

    keep = scores >= conf
    boxes = boxes[keep]
    scores = scores[keep]

    if boxes.numel():
        keep_idx = nms(boxes, scores, nms_iou)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

    if (size_min is not None) or (size_max is not None):
        if boxes.numel():
            wh = boxes[:, 2:4] - boxes[:, 0:2]
            area = wh[:, 0] * wh[:, 1]
            keep_area = torch.ones(len(area), dtype=torch.bool)
            if size_min is not None:
                keep_area &= area >= size_min
            if size_max is not None:
                keep_area &= area <= size_max
            boxes = boxes[keep_area]
            scores = scores[keep_area]

    annotated_image = None
    if draw or return_image or out_path is not None:
        annotated_image = resized.copy()
        draw_ctx = ImageDraw.Draw(annotated_image)
        font = _load_font()
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            draw_ctx.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=2)
        draw_ctx.rectangle([6, 6, 220, 32], fill=(0, 0, 0))
        draw_ctx.text((10, 10), f"count: {len(boxes)}", fill=(255, 255, 255), font=font)
        if out_path is not None:
            _ensure_path(out_path).parent.mkdir(parents=True, exist_ok=True)
            annotated_image.save(out_path)

    return PredictionResult(
        count=int(len(boxes)),
        boxes=boxes,
        scores=scores,
        image=annotated_image if (return_image or draw or out_path is not None) else None,
    )


def predict_folder(
    folder: PathLike,
    out_dir: PathLike,
    model: Union[nn.Module, "CellCountingModel"],
    device: Optional[DeviceLike] = None,
    image_size: int = 640,
    conf: float = 0.15,
    nms_iou: float = 0.45,
    size_min: Optional[float] = 12.0 * 12.0,
    size_max: Optional[float] = 80.0 * 80.0,
) -> List[PredictionResult]:
    """Run inference on every supported image file inside ``folder``."""

    folder_path = _ensure_path(folder)
    out_path = _ensure_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "viz").mkdir(exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = [p for p in folder_path.rglob("*") if p.suffix.lower() in exts]

    results: List[PredictionResult] = []
    csv_path = out_path / "predict_counts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "pred_count"])
        for path in sorted(image_paths):
            result = predict_image(
                image=path,
                model=model,
                device=device,
                image_size=image_size,
                conf=conf,
                nms_iou=nms_iou,
                size_min=size_min,
                size_max=size_max,
                draw=True,
                out_path=out_path / "viz" / f"{path.stem}_viz.png",
                return_image=False,
            )
            writer.writerow([str(path), result.count])
            results.append(result)

    return results


__all__ = ["predict_image", "predict_folder"]
