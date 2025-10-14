"""Model construction and loading helpers for cell counting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torchvision
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

DeviceLike = Union[str, torch.device]
PathLike = Union[str, Path]


def _resolve_device(device: Optional[DeviceLike]) -> torch.device:
    """Resolve a device specification into a :class:`torch.device`."""

    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def build_model(
    num_classes: int = 2,
    probe_size: int = 320,
    pretrained: bool = False,
    pretrained_backbone: bool = False,
) -> nn.Module:
    """Construct the SSDLite MobileNetV3 detection network.

    The logic mirrors the implementation from the ``hepatocytometer.ipynb``
    notebook while making it safe to import in CPU-only environments. The
    function automatically handles version differences in ``torchvision`` when
    replacing the classification head and avoids downloading pretrained
    weights unless explicitly requested via ``pretrained`` or
    ``pretrained_backbone``.

    """

    try:
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )

        try:
            from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
        except ImportError:  # pragma: no cover - older torchvision releases
            MobileNet_V3_Large_Weights = None  # type: ignore[assignment]

        weights = (
            SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if pretrained else None
        )
        backbone_weights = None
        if pretrained_backbone and MobileNet_V3_Large_Weights is not None:
            backbone_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1

        model = ssdlite320_mobilenet_v3_large(
            weights=weights,
            weights_backbone=backbone_weights,
        )
    except ImportError:
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(  # type: ignore[attr-defined]
            pretrained=pretrained
        )
        if pretrained_backbone:
            try:
                model.backbone.body.load_state_dict(torchvision.models.mobilenet_v3_large(  # type: ignore[attr-defined]
                    pretrained=True
                ).state_dict())
            except Exception:
                pass
    except Exception:
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(  # type: ignore[attr-defined]
            pretrained=False

        )

    try:
        in_channels: Sequence[int] = model.backbone.out_channels  # type: ignore[attr-defined]
        if isinstance(in_channels, int):
            in_channels = [in_channels]
    except Exception:
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, probe_size, probe_size)
            features = model.backbone(dummy)  # type: ignore[operator]
            in_channels = [tensor.shape[1] for tensor in features.values()]

    num_anchors = model.anchor_generator.num_anchors_per_location()  # type: ignore[attr-defined]

    try:
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )
    except TypeError:
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=nn.BatchNorm2d,
        )

    return model


@dataclass
class PredictionResult:
    """Container for predictions returned by :func:`predict_image`."""

    count: int
    boxes: Tensor
    scores: Tensor
    image: Optional["Image.Image"] = None


@dataclass
class CellCountingModel:
    """Light-weight wrapper around the detector used for cell counting."""

    model: nn.Module
    device: torch.device
    image_size: int = 640

    def __init__(
        self,
        weights_path: Optional[PathLike] = "results/models/best.pt",
        device: Optional[DeviceLike] = None,
        image_size: int = 640,
        num_classes: int = 2,
        pretrained_backbone: bool = False,
    ) -> None:
        self.device = _resolve_device(device)
        self.image_size = int(image_size)
        self.model = build_model(
            num_classes=num_classes,
            probe_size=self.image_size,
            pretrained=False,
            pretrained_backbone=pretrained_backbone,

        ).to(self.device)
        self.model.eval()

        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, weights_path: PathLike) -> None:
        """Load detector weights from ``results/models/best.pt`` or a custom path."""

        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def count_cells(
        self,
        image: Union[PathLike, "Image.Image"],
        conf: float = 0.15,
        nms_iou: float = 0.45,
        size_min: Optional[float] = 12.0 * 12.0,
        size_max: Optional[float] = 80.0 * 80.0,
        image_size: Optional[int] = None,
        return_image: bool = False,
        draw: bool = False,
        out_path: Optional[PathLike] = None,
    ) -> Union[Tuple[int, Tensor], Tuple[int, Tensor, "Image.Image"]]:
        """Count cells within a single image."""

        from .inference import predict_image  # Local import to avoid a cycle.

        effective_draw = draw or return_image or out_path is not None
        inference_size = int(image_size or self.image_size)
        result = predict_image(
            image=image,
            model=self.model,
            device=self.device,
            image_size=inference_size,
            conf=conf,
            nms_iou=nms_iou,
            size_min=size_min,
            size_max=size_max,
            draw=effective_draw,
            out_path=out_path,
            return_image=return_image,
        )

        if return_image:
            return result.count, result.boxes, result.image  # type: ignore[return-value]
        return result.count, result.boxes


__all__ = ["CellCountingModel", "PredictionResult", "build_model"]
