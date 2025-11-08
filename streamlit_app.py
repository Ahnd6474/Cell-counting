"""Streamlit interface for the cell counting demo."""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
import torch

from cell_counting import load_model
from cell_counting.inference import predict_image

DEFAULT_WEIGHTS = Path("results/models/best.pt")
SUPPORTED_TYPES = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")
SAMPLE_FEED_DIR = Path("data/valid/images")
Box = Tuple[float, float, float, float]


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

def _list_sample_frames() -> List[Path]:
    if not SAMPLE_FEED_DIR.exists():
        return []
    return sorted(
        [p for p in SAMPLE_FEED_DIR.rglob("*") if p.suffix.lower().lstrip(".") in SUPPORTED_TYPES]
    )


def _load_uploaded_image(uploaded) -> Optional[Image.Image]:  # type: ignore[override]
    if uploaded is None:
        return None
    try:
        return Image.open(uploaded).convert("RGB")
    except Exception as exc:  # pragma: no cover - user input handling
        st.error(f"Failed to open the provided image: {exc}")
        return None


def _canvas_rectangles(canvas_json: Optional[Dict]) -> List[Dict[str, float]]:
    if not canvas_json:
        return []
    rectangles: List[Dict[str, float]] = []
    for obj in canvas_json.get("objects", []):
        if obj.get("type") != "rect":
            continue
        rectangles.append(
            {
                "left": float(obj.get("left", 0.0)),
                "top": float(obj.get("top", 0.0)),
                "width": float(obj.get("width", 0.0)),
                "height": float(obj.get("height", 0.0)),
            }
        )
    return rectangles


def _crop_roi(image: Image.Image, rect: Dict[str, float]) -> Tuple[Image.Image, Box]:
    width, height = image.size
    x1 = max(int(rect["left"]), 0)
    y1 = max(int(rect["top"]), 0)
    x2 = min(int(rect["left"] + rect["width"]), width)
    y2 = min(int(rect["top"] + rect["height"]), height)
    if x2 <= x1 or y2 <= y1:
        return image, (0.0, 0.0, float(width), float(height))
    return image.crop((x1, y1, x2, y2)), (float(x1), float(y1), float(x2), float(y2))


def _normalise_box(box: Box, width: int, height: int) -> Optional[Box]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _annotate_image(image: Image.Image, boxes: Iterable[Box]) -> Image.Image:
    annotated = image.copy()
    draw_ctx = ImageDraw.Draw(annotated)
    boxes_list = [
        (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for box in boxes
    ]
    for box in boxes_list:
        x1, y1, x2, y2 = box
        draw_ctx.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=6)
    count_text = f"count: {len(boxes_list)}"
    if hasattr(draw_ctx, "textbbox"):
        left, top, right, bottom = draw_ctx.textbbox((0, 0), count_text)
        text_width = right - left
        text_height = bottom - top
    else:  # pragma: no cover - Pillow < 8 compatibility
        text_width, text_height = draw_ctx.textsize(count_text)
    overlay = [6, 6, 6 + text_width + 12, 6 + text_height + 12]
    draw_ctx.rectangle(overlay, fill=(0, 0, 0))
    draw_ctx.text((12, 12), count_text, fill=(255, 255, 255))
    return annotated


def _prepare_box_editor_data(boxes: Sequence[Box]) -> List[Dict[str, object]]:
    return [
        {
            "id": idx + 1,
            "x1": round(float(box[0]), 2),
            "y1": round(float(box[1]), 2),
            "x2": round(float(box[2]), 2),
            "y2": round(float(box[3]), 2),
            "active": True,
            "source": "model",
        }
        for idx, box in enumerate(boxes)
    ]


def _extract_active_boxes(rows: Sequence[Dict[str, object]], width: int, height: int) -> List[Box]:
    active: List[Box] = []
    for row in rows:
        if not row:
            continue
        if not row.get("active", True):
            continue
        try:
            x1 = float(row.get("x1", 0))
            y1 = float(row.get("y1", 0))
            x2 = float(row.get("x2", 0))
            y2 = float(row.get("y2", 0))
        except (TypeError, ValueError):
            continue
        normalised = _normalise_box((x1, y1, x2, y2), width, height)
        if normalised is not None:
            active.append(normalised)
    return active


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
    blank_upload = st.sidebar.file_uploader(
        "Blanking image (optional)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        help="Upload a blank reference image to subtract before inference.",
    )

    # uploaded = st.file_uploader(
    #     "Upload an image", type=SUPPORTED_TYPES, accept_multiple_files=False
    st.subheader("Input selection")
    input_mode = st.radio(
        "Choose how to provide imagery",
        ("Upload image", "Live microscope feed"),
        horizontal=True,
    )

    # if not uploaded:
    #     st.info("Upload an image to begin.")
    #     return
    original_image: Optional[Image.Image] = None
    image_label = "uploaded"

    # try:
    #     original_image = Image.open(uploaded).convert("RGB")
    # except Exception as exc:  # pragma: no cover - user input handling
    #     st.error(f"Failed to open the uploaded image: {exc}")
    #     return
    if input_mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload an image", type=SUPPORTED_TYPES, accept_multiple_files=False
        )
        if uploaded is None:
            st.info("Upload an image to begin.")
            return
        original_image = _load_uploaded_image(uploaded)
        image_label = getattr(uploaded, "name", "uploaded")
        if original_image is None:
            return
    else:
        feed_source = st.selectbox(
            "Live input source",
            ("Webcam (camera input)", "Microscope adapter (sample feed)"),
        )
        if feed_source == "Webcam (camera input)":
            camera_capture = st.camera_input(
                "Microscope feed", key="microscope_camera", help="Capture a frame from the connected microscope."
            )
            if camera_capture is None:
                st.info("Capture a frame from the microscope feed to continue.")
                return
            original_image = _load_uploaded_image(camera_capture)
            image_label = "camera_capture"
            if original_image is None:
                return
        else:
            frames = _list_sample_frames()
            if not frames:
                st.warning(
                    "No sample frames were found in `data/valid/images`. Add reference imagery to use the simulated microscope adapter feed."
                )
                return
            index_key = "sample_feed_index"
            if index_key not in st.session_state:
                st.session_state[index_key] = 0
            cols = st.columns([1, 1, 2])
            with cols[0]:
                if st.button("Previous", use_container_width=True):
                    st.session_state[index_key] = (st.session_state[index_key] - 1) % len(frames)
            with cols[1]:
                if st.button("Next", use_container_width=True):
                    st.session_state[index_key] = (st.session_state[index_key] + 1) % len(frames)
            current_index = st.session_state[index_key]
            current_frame = frames[current_index]
            cols[2].markdown(f"**Frame:** `{current_frame.name}` ({current_index + 1}/{len(frames)})")
            original_image = Image.open(current_frame).convert("RGB")
            image_label = current_frame.name

        st.image(original_image, caption="Latest microscope frame", use_column_width=True)

    blank_image = None
    if blank_upload is not None:
        try:
            blank_image = Image.open(blank_upload).convert("RGB")
        except Exception as exc:  # pragma: no cover - user input handling
            st.error(f"Failed to open the blanking image: {exc}")
            return
    roi_info: Optional[Tuple[Image.Image, Box]] = None
    st.subheader("Region of interest")
    enable_roi = st.toggle(
        "Enable region selection",
        value=False,
        help="Draw a rectangle on the frame to analyse only a specific area.",
    )

    inference_image = original_image
    if enable_roi and original_image is not None:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 255, 0.2)",
            stroke_width=3,
            stroke_color="#ff00ff",
            background_image=original_image,
            height=original_image.height,
            width=original_image.width,
            drawing_mode="rect",
            key=f"roi_canvas_{image_label}",
            update_streamlit=True,
        )
        rectangles = _canvas_rectangles(canvas_result.json_data)
        if rectangles:
            cropped, roi_box = _crop_roi(original_image, rectangles[-1])
            inference_image = cropped
            roi_info = (cropped, roi_box)
            st.caption("Using the highlighted region for inference.")
            st.image(cropped, caption="Selected region", use_column_width=True)
        else:
            st.caption("No region selected; analysing the full frame.")

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
                image=inference_image,
                model=model,
                image_size=getattr(model, "image_size", 640),
                conf=float(confidence),
                blank_image=blank_image
                if not roi_info
                or blank_image is None
                or blank_image.size != original_image.size
                else blank_image.crop(tuple(int(v) for v in roi_info[1])),
                draw=True,
                return_image=True,
            )
        except Exception as exc:  # pragma: no cover - defensive user feedback
            st.error(f"Inference failed: {exc}")
            st.stop()

    st.success(f"Predicted cell count: {result.count}")

    if result.image is not None:
    annotated_source = result.image or inference_image
    image_digest = hashlib.md5(_to_bytes(annotated_source)).hexdigest()

    #     st.image(result.image, caption="Annotated detections")
    #     st.download_button(
    #         "Download annotated image",
    #         data=_to_bytes(result.image),
    #         file_name="cell_counting_result.png",
    #         mime="image/png",
    #     )
    # else:
    #     st.image(original_image, caption="Original image", use_column_width=True)
    st.subheader("Review detections")
    st.image(annotated_source, caption="Model detections", use_column_width=True)

    boxes_state = st.session_state.setdefault("boxes_editor", {})
    initial_boxes = [tuple(map(float, box)) for box in result.boxes.tolist()] if result.boxes is not None else []

    if boxes_state.get("digest") != image_digest:
        boxes_state.clear()
        boxes_state["digest"] = image_digest
        boxes_state["data"] = _prepare_box_editor_data(initial_boxes)

    st.markdown("Adjust the detections below. Toggle **active** to remove a box or append a new row to add one.")

    edited_rows = st.data_editor(
        boxes_state.get("data", []),
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "x1": st.column_config.NumberColumn("x1", help="Left coordinate (pixels)"),
            "y1": st.column_config.NumberColumn("y1", help="Top coordinate (pixels)"),
            "x2": st.column_config.NumberColumn("x2", help="Right coordinate (pixels)"),
            "y2": st.column_config.NumberColumn("y2", help="Bottom coordinate (pixels)"),
            "active": st.column_config.CheckboxColumn("Active"),
            "source": st.column_config.TextColumn("Source", help="Origin of the bounding box"),
        },
        key=f"editor_{image_digest}",
    )

    boxes_state["data"] = edited_rows

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Reset to model predictions", use_container_width=True):
            boxes_state["data"] = _prepare_box_editor_data(initial_boxes)
            st.experimental_rerun()

    adjusted_boxes = _extract_active_boxes(boxes_state["data"], inference_image.width, inference_image.height)

    adjusted_image = _annotate_image(inference_image, adjusted_boxes)

    st.metric("Adjusted cell count", len(adjusted_boxes), delta=len(adjusted_boxes) - result.count)
    st.image(adjusted_image, caption="Adjusted detections", use_column_width=True)

    download_name = f"cell_counting_{image_label.replace(' ', '_')}.png"
    st.download_button(
        "Download adjusted image",
        data=_to_bytes(adjusted_image),
        file_name=download_name,
        mime="image/png",
    )


if __name__ == "__main__":
    main()
