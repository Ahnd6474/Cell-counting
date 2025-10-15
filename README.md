# Cell-counting
 
Utilities and models for performing automated cell counting using a trained
SSDLite MobileNetV3 detector.
 
## Project overview

This repository packages the components used to detect and count cells in
hemocytometer images. It bundles the trained SSDLite MobileNetV3 detector,
Python APIs for batch and single-image inference, and an optional Streamlit
application for interactive experimentation. The code mirrors the original
`hepatocytometer.ipynb` workflow while making it easy to install and reuse in
other projects.

## Features

- Preconfigured SSDLite MobileNetV3 model wrapper with weight-loading helpers.
- Simple `load_model` and `count_cells` APIs for scripted inference on files or
  Pillow images.
- Batch prediction utilities that export both CSV summaries and annotated
  overlays.
- Streamlit demo that visualises predictions directly in the browser.

## Quickstart

1. Clone the repository and create a virtual environment (recommended).
2. Activate the environment and install the runtime dependencies pinned for the
   released weights:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install the package itself for importable helpers:

   ```bash
   pip install -e .
   ```

4. Obtain the pretrained detector weights (see below) and place them where you
   intend to load them from (the default is `results/models/best.pt`).
5. Run the Streamlit demo or call the Python API to verify everything is
   working.

### Installing the package locally

Once `pyproject.toml` is available, `pip` can build and install the package
straight from the repository root after the runtime dependencies are installed:

```bash
pip install .
```

For development you can prefer an editable install (after installing
`requirements.txt`):

```bash
pip install -e .
```

When you need the testing utilities, install the development extras:

```bash
pip install -r requirements-dev.txt
```

### Pretrained weights

The examples assume the best-performing weights are stored at
`results/models/best.pt`. If you are starting from a fresh clone:

1. Download the pretrained checkpoint that accompanies the project (for
   example via an internal artifact store) and save it as `best.pt` inside
   `results/models/`.
2. Alternatively, supply a custom path when calling `load_model` or
   `count_cells` if you keep the weights elsewhere.

## Python usage

### Loading the model once

```python
from cell_counting import load_model

model = load_model(
    weights_path="results/models/best.pt",
    device="cuda:0",  # or "cpu"
    image_size=640,
)

count, boxes = model.count_cells("docs/assets/sample_input.svg")
print(f"Detected {count} cells")
```

> **Note:** The packaged SVG is a lightweight illustration; replace the path
> with an actual microscope capture (for example
> `docs/assets/sample_input.jpg`) before running the snippet so that Pillow can
> decode the image.

### One-off predictions with caching

```python
from cell_counting import count_cells

count, boxes, annotated = count_cells(
    "docs/assets/sample_input.svg",
    weights_path="results/models/best.pt",
    device="cpu",
    return_image=True,
    draw=True,
)
annotated.save("prediction.jpg")
print(f"Predicted {count} cells with {len(boxes)} bounding boxes")
```

When experimenting locally, point the API to a raster image (JPG or PNG).
Providing the illustrative SVG is handy for documentation, but inference should
use the corresponding bitmap capture to mirror real workflows.

## Streamlit app

The same runtime requirements enable the interactive demo. After installing
`requirements.txt` and downloading the trained weights, start the app with:

```bash
streamlit run streamlit_app.py
```

The interface will prompt you for an image and download-ready annotated output
once inference finishes. The app lets you upload hemocytometer imagery, tweak
confidence thresholds, and inspect detections without writing code.

## Additional resources

- `hepatocytometer.ipynb` &mdash; the original exploratory notebook that contains
  the training and evaluation workflow.
- `docs/` &mdash; documentation assets, including sample images used throughout the
  README examples.
- `results/` &mdash; a suggested directory layout for storing trained models and
  experiment outputs.

Refer to the [tests](tests/) folder for basic smoke tests that validate the
package installations and the provided inference utilities.

## Examples

To stay within the “no binary artifacts in git” guideline, the example assets
are embedded as base64-encoded PNGs inside the SVG wrappers in
`docs/assets/`. The input frame below matches the microscope crop supplied for
this task and the accompanying output shows the resulting overlay. The helper
script `docs/scripts/generate_examples.py` (which falls back to a lightweight
threshold-based segmentation when PyTorch is unavailable) regenerates both
artifacts from the source PNG and keeps the repository reproducible.

| Sample hemocytometer input | Annotated output |
| --- | --- |
| ![Sample input](docs/assets/seq0432_jpg.rf.f16687b29f969b08fdc2900f51b3e5d3.jpg) | ![Annotated output](docs/assets/cell_counting_result.png) |

The repository inference helpers (`cell_counting.count_cells` or
`cell_counting.inference.predict_image`) will recreate the overlay when PyTorch
is available. In environments where the heavy dependencies cannot be installed,
the regeneration script resorts to intensity-based segmentation to draw
bounding boxes so that the documentation remains illustrative.


## Evaluation

Validation results bundled with the repository show the detector remains close
to ground-truth counts: across three validation frames the model attains a mean
absolute error of 1.33 cells (median 1, maximum 2) while predicting 18 cells
versus 16 labelled cells overall. See `results/report_val.csv` for the full
per-image breakdown.
