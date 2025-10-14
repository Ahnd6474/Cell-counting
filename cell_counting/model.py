"""Model construction and loading helpers for cell counting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torchvision

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
    """Construct the RetinaNet (ResNet-50 FPN) detector used for training."""

    try:
        from torchvision.models.detection import (
            retinanet_resnet50_fpn,
            RetinaNet_ResNet50_FPN_Weights,
        )
    except ImportError:  # pragma: no cover - compatibility with older torchvision
        retinanet_resnet50_fpn = torchvision.models.detection.retinanet_resnet50_fpn  # type: ignore[attr-defined]
        RetinaNet_ResNet50_FPN_Weights = None  # type: ignore[assignment]

    weights = None
    backbone_weights = None
    if pretrained and "RetinaNet_ResNet50_FPN_Weights" in locals() and RetinaNet_ResNet50_FPN_Weights is not None:
        weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
    if pretrained_backbone and "RetinaNet_ResNet50_FPN_Weights" in locals() and RetinaNet_ResNet50_FPN_Weights is not None:
        backbone_weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1

    try:
        model = retinanet_resnet50_fpn(
            weights=weights,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
        )
    except TypeError:
        # Older torchvision versions expose a legacy API without ``weights``.
        kwargs = {"pretrained": bool(pretrained)}
        model = torchvision.models.detection.retinanet_resnet50_fpn(**kwargs)  # type: ignore[attr-defined]
        try:
            from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        except Exception:  # pragma: no cover - fallback for very old versions
            RetinaNetClassificationHead = None  # type: ignore[assignment]

        if "RetinaNetClassificationHead" in locals() and RetinaNetClassificationHead is not None:
            head = model.head.classification_head
            num_anchors = getattr(head, "num_anchors", 9)
            try:
                in_channels = head.conv[0].in_channels  # type: ignore[index]
            except Exception:  # pragma: no cover - safeguard for older structures
                in_channels = 256
            model.head.classification_head = RetinaNetClassificationHead(
                in_channels,
                num_anchors,
                num_classes,
            )
        else:  # pragma: no cover - best effort fallback
            head = model.head.classification_head
            if hasattr(head, "num_classes"):
                head.num_classes = num_classes

    try:
        model.transform.min_size = (probe_size,)
        model.transform.max_size = probe_size
    except Exception:  # pragma: no cover - defensive programming
        pass

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
