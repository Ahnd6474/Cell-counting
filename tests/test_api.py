"""Smoke tests for the public cell counting API."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")
from PIL import Image  # type: ignore  # noqa: E402

from cell_counting import CellCountingModel, count_cells, load_model


def test_load_model_without_weights_uses_cpu() -> None:
    model = load_model(weights_path=None, device="cpu", image_size=64, pretrained_backbone=False)
    assert isinstance(model, CellCountingModel)
    assert str(model.device) == "cpu"


def test_count_cells_accepts_pil_image() -> None:
    image = Image.new("RGB", (64, 64), color="black")
    count, boxes = count_cells(
        image,
        weights_path=None,
        device="cpu",
        image_size=64,
        pretrained_backbone=False,
        conf=0.5,
    )
    assert isinstance(count, int)
    assert isinstance(boxes, torch.Tensor)
    assert boxes.dim() == 2 or boxes.numel() == 0
