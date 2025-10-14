"""Inference helpers extracted from the original notebook."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv

from .model import PredictionResult

if TYPE_CHECKING:
    from .model import CellCountingModel

PathLike = Union[str, Path]
DeviceLike = Union[str, torch.device]

'''
def _preprocess_for_detector(image: Image.Image, size: int) -> Image.Image:
    """Replicate the preprocessing pipeline from the training notebook."""

    rgb = image.convert("RGB")
    np_image = np.array(rgb)
    gray = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (41, 41))
    background = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    gray = cv.subtract(gray, background)
    gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
    stacked = np.stack([gray, gray, gray], axis=2)
    resized = cv.resize(stacked, (size, size), interpolation=cv.INTER_LINEAR)
    return Image.fromarray(resized, mode="RGB")'''
def _preprocess_for_detector(pil_img, size, blank_image=None):
    im = np.array(pil_img)
    g  = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    if blank_image is not None:
        blank_arr = np.array(blank_image.convert("RGB"))
        blank_gray = cv.cvtColor(blank_arr, cv.COLOR_RGB2GRAY)
        if blank_gray.shape != g.shape:
            blank_gray = cv.resize(blank_gray, (g.shape[1], g.shape[0]), interpolation=cv.INTER_LINEAR)
        g = cv.subtract(g, blank_gray)
        g = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g  = clahe.apply(g)
    se_big = cv.getStructuringElement(cv.MORPH_RECT, (41,41))
    bg = cv.morphologyEx(g, cv.MORPH_OPEN, se_big)
    g  = cv.subtract(g, bg)
    # 정규화~리사이즈
    g  = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX)
    im3 = np.stack([g, g, g], axis=2)
    im3 = cv.resize(im3, (size, size), cv.INTER_LINEAR)
    return Image.fromarray(im3, mode="RGB")



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
    blank_image: Optional[Union[PathLike, Image.Image]] = None,
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
        pil_image = Image.open(image)
    else:
        pil_image = image

    pil_image = pil_image.convert("RGB")
    original_width, original_height = pil_image.size

    blank_pil: Optional[Image.Image] = None
    if blank_image is not None:
        if isinstance(blank_image, (str, Path)):
            blank_pil = Image.open(blank_image).convert("RGB")
        else:
            blank_pil = blank_image.convert("RGB")

    processed = _preprocess_for_detector(pil_image, image_size, blank_pil)
    tensor = to_tensor(processed).to(device_obj)
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

    scaled_boxes = boxes.clone()
    if scaled_boxes.numel():
        scale_x = original_width / float(image_size)
        scale_y = original_height / float(image_size)
        scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]] * scale_x
        scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]] * scale_y

    annotated_image = None
    if draw or return_image or out_path is not None:
        annotated_image = pil_image.copy()
        draw_ctx = ImageDraw.Draw(annotated_image)
        font = _load_font()
        for box in scaled_boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            draw_ctx.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=2)
        text = f"count: {len(scaled_boxes)}"
        if hasattr(draw_ctx, "textbbox"):
            left, top, right, bottom = draw_ctx.textbbox((0, 0), text, font=font)
            text_width = right - left
            text_height = bottom - top
        else:
            text_width, text_height = draw_ctx.textsize(text, font=font)
        overlay = [6, 6, 6 + text_width + 8, 6 + text_height + 8]
        draw_ctx.rectangle(overlay, fill=(0, 0, 0))
        draw_ctx.text((10, 10), text, fill=(255, 255, 255), font=font)
        if out_path is not None:
            _ensure_path(out_path).parent.mkdir(parents=True, exist_ok=True)
            annotated_image.save(out_path)

    return PredictionResult(
        count=int(len(scaled_boxes)),
        boxes=scaled_boxes,
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
    blank_image: Optional[Union[PathLike, Image.Image]] = None,
) -> List[PredictionResult]:
    """Run inference on every supported image file inside ``folder``."""

    folder_path = _ensure_path(folder)
    out_path = _ensure_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "viz").mkdir(exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = [p for p in folder_path.rglob("*") if p.suffix.lower() in exts]

    blank_ref: Optional[Image.Image] = None
    if blank_image is not None:
        if isinstance(blank_image, (str, Path)):
            blank_ref = Image.open(blank_image).convert("RGB")
        else:
            blank_ref = blank_image.convert("RGB")

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
                blank_image=blank_ref,
                draw=True,
                out_path=out_path / "viz" / f"{path.stem}_viz.png",
                return_image=False,
            )
            writer.writerow([str(path), result.count])
            results.append(result)

    return results


__all__ = ["predict_image", "predict_folder"]
