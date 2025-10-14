"""Public package interface for cell counting utilities."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import torch

from .inference import predict_folder, predict_image
from .model import CellCountingModel

PathLike = Union[str, Path]
DeviceLike = Union[str, torch.device]


def _normalise_path(path: Optional[PathLike]) -> Optional[str]:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def _normalise_device(device: Optional[DeviceLike]) -> Optional[str]:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return str(device)
    return str(torch.device(device))


@lru_cache(maxsize=None)
def _cached_model(
    weights_path: Optional[str],
    device: Optional[str],
    image_size: int,
    num_classes: int,
    pretrained_backbone: bool,
) -> CellCountingModel:
    return CellCountingModel(
        weights_path=weights_path,
        device=device,
        image_size=image_size,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
    )


def load_model(
    weights_path: Optional[PathLike] = "results/models/best.pt",
    device: Optional[DeviceLike] = None,
    image_size: int = 640,
    num_classes: int = 2,
    pretrained_backbone: bool = False,
) -> CellCountingModel:
    """Load (and cache) a :class:`~cell_counting.model.CellCountingModel`."""

    key = (
        _normalise_path(weights_path),
        _normalise_device(device),
        int(image_size),
        int(num_classes),
        bool(pretrained_backbone),
    )
    return _cached_model(*key)


def count_cells(
    image: Union[PathLike, "Image.Image"],
    weights_path: Optional[PathLike] = "results/models/best.pt",
    device: Optional[DeviceLike] = None,
    image_size: int = 640,
    num_classes: int = 2,
    pretrained_backbone: bool = False,
    **kwargs,
):
    """Count cells in ``image`` using a cached :class:`CellCountingModel`."""

    model = load_model(
        weights_path=weights_path,
        device=device,
        image_size=image_size,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
    )
    return model.count_cells(image, **kwargs)


__all__ = [
    "CellCountingModel",
    "count_cells",
    "load_model",
    "predict_folder",
    "predict_image",
]
