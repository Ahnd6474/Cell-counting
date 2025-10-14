"""Streamlit interface for the cell counting demo."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image
import torch

from cell_counting import load_model
from cell_counting.inference import predict_image

DEFAULT_WEIGHTS = Path("results/models/best.pt")
SUPPORTED_TYPES = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")


def _resolve_device() -> str:
    """Pick the best available inference device."""

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


@st.cache_resource(show_spinner=False)
def get_model(device: str):
    """Load the cell counting model once per device."""

    return load_model(weights_path=DEFAULT_WEIGHTS, device=device)


def _to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="Cell Counting", page_icon="🧫", layout="centered")
    st.title("Cell Counting Demo")
    st.caption(
        "Upload a microscopy image to run the trained detector and count the detected cells."
    )

    device = _resolve_device()
    st.sidebar.header("Inference settings")
    st.sidebar.write(f"Using **{device.upper()}** for inference")
    confidence = st.sidebar.slider(
        "Confidence threshold", min_value=0.05, max_value=0.9, value=0.15, step=0.05
    )
    grid_blank_k = st.sidebar.slider(
        "Grid removal size (k)", min_value=5, max_value=51, value=25, step=2,
        help="Adjust the structuring element size used to blank the grid before inference."
    )

    uploaded = st.file_uploader(
        "Upload an image", type=SUPPORTED_TYPES, accept_multiple_files=False
    )

    if not uploaded:
        st.info("Upload an image to begin.")
        return

    try:
        original_image = Image.open(uploaded).convert("RGB")
    except Exception as exc:  # pragma: no cover - user input handling
        st.error(f"Failed to open the uploaded image: {exc}")
        return

    with st.spinner("Loading model and running inference..."):
        try:
            model = get_model(device)
        except FileNotFoundError:
            st.error(
                "Model weights not found. Place the trained weights at "
                f"`{DEFAULT_WEIGHTS.as_posix()}` to enable inference."
            )
            st.stop()
        except Exception as exc:  # pragma: no cover - defensive user feedback
            st.error(f"Failed to load the model: {exc}")
            st.stop()

        try:
            result = predict_image(
                image=original_image,
                model=model,
                image_size=getattr(model, "image_size", 640),
                conf=float(confidence),
                grid_blank_k=int(grid_blank_k),
                draw=True,
                return_image=True,
            )
        except Exception as exc:  # pragma: no cover - defensive user feedback
            st.error(f"Inference failed: {exc}")
            st.stop()

    st.success(f"Predicted cell count: {result.count}")

    if result.image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original image", use_column_width=True)
        with col2:
            st.image(result.image, caption="Annotated detections", use_column_width=True)
        st.download_button(
            "Download annotated image",
            data=_to_bytes(result.image),
            file_name="cell_counting_result.png",
            mime="image/png",
        )
    else:
        st.image(original_image, caption="Original image", use_column_width=True)


if __name__ == "__main__":
    main()
