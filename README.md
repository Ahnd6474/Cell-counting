# Cell-counting
 
Utilities and models for automated cell counting with the RetinaNet
(ResNet-50 FPN) detector that powers the production notebooks.

<img src="docs/assets/cell_counting_result.png" alt="Annotated cell counting example" width="400">

## Project overview

This repository contains everything needed to detect and count cells in
hemocytometer images. It bundles the trained RetinaNet ResNet-50 FPN detector,
Python APIs for batch and single-image inference, and a Streamlit application
for interactive experimentation. The code mirrors the original
`hepatocytometer.ipynb` workflow while remaining easy to install and reuse in
other projects.

## Features

- Preconfigured RetinaNet (ResNet-50 FPN) model wrapper with weight-loading
  helpers.
- Simple `load_model` and `count_cells` APIs for scripted inference on files or
  Pillow images.
- Batch prediction utilities that export both CSV summaries and annotated
  overlays.
- Streamlit demo that visualises predictions directly in the browser.

## Quickstart

1. Clone the repository and move into the project directory.

   ```bash
   git clone https://github.com/<your-org>/Cell-counting.git
   cd Cell-counting
   ```

2. (Recommended) Create and activate a virtual environment.
3. Install the runtime dependencies pinned for the released weights.

   ```bash
   pip install -r requirements.txt
   ```

4. Download the pretrained detector weights (see below) and place them in
   `results/models/best.pt` or another preferred path.
5. (Optional) Capture a blank hemocytometer frame if you intend to use the
   background subtraction helper.
6. Run the Streamlit demo or call the Python API to confirm everything works.

## Python usage

### Loading the model

```python
from cell_counting import load_model

model = load_model(
    weights_path="results/models/best.pt",
    device="cuda:0",  # or "cpu"
    image_size=640,
)

count, boxes = model.count_cells(
    "docs/assets/seq0432_jpg.rf.f16687b29f969b08fdc2900f51b3e5d3.jpg",
    blank_image="path/to/blank_reference.jpg",  # optional user-provided frame
)
print(f"Detected {count} cells")
```

> **Note:** Use an actual microscope capture (JPG or PNG). Raster exemplars in
> `docs/assets/` match the samples referenced throughout this README.

### Inference helpers

```python
from cell_counting import count_cells

count, boxes, annotated = count_cells(
    "docs/assets/seq0432_jpg.rf.f16687b29f969b08fdc2900f51b3e5d3.jpg",
    weights_path="results/models/best.pt",
    device="cpu",
    blank_image="path/to/blank_reference.jpg",  # optional user-provided frame
    return_image=True,
    draw=True,
)
annotated.save("prediction.jpg")
print(f"Predicted {count} cells with {len(boxes)} bounding boxes")
```

When experimenting locally, supplying a matching blank reference frame helps
remove background artefacts prior to detection. Omit `blank_image` when the
subtraction step is unnecessary.

### Detailed workflow

1. **Prepare weights** &mdash; place the trained RetinaNet checkpoint at
   `results/models/best.pt`. The helper accepts alternative locations via the
   `weights_path` argument if you prefer a custom folder layout.
2. **Load the model** &mdash; call `load_model()` to build the detector and load the
   checkpoint. Pass `device="cuda:0"` when a GPU is available or leave it unset
   to default to CPU.
3. **(Optional) Calibrate with a blank frame** &mdash; capture an empty
   hemocytometer chamber image and provide it as `blank_image`. The preprocessing
   routine subtracts this reference to attenuate lighting artefacts.
4. **Run inference** &mdash; use `model.count_cells(...)` for repeated predictions on
   a long-lived model instance. For lightweight scripts, the functional helper
   `count_cells(...)` accepts the same arguments and handles model caching for
   you.
5. **Adjust detection thresholds** &mdash; tweak `conf`, `nms_iou`, `size_min`, and
   `size_max` to match the expected cell morphology. For example, increase
   `conf` to reduce false positives or tighten `size_max` when debris is being
   counted accidentally.
6. **Export results** &mdash; set `return_image=True` or `out_path="prediction.jpg"`
   to capture annotated overlays. The raw bounding boxes are returned in image
   coordinates so that you can post-process them downstream.
7. **Batch processing** &mdash; call `cell_counting.inference.predict_folder(...)` to
   sweep an entire directory, saving annotated images and a CSV summary in
   `out_dir`.

Refer to `cell_counting/inference.py` for additional options, including font
customisation for overlays and blank-frame preprocessing hooks.

## Streamlit app

The same runtime requirements enable the interactive demo. After installing the
dependencies and downloading the trained weights, start the app from the
project root:

```bash
streamlit run streamlit_app.py
```

The interface guides you through selecting the checkpoint, uploading microscope
imagery, and downloading annotated results. Within the sidebar you can adjust
confidence thresholds, toggle blank-frame subtraction, and inspect detection
counts without writing code.

## Additional resources

- `hepatocytometer.ipynb` &mdash; the original exploratory notebook that contains
  the training and evaluation workflow.
- `docs/` &mdash; documentation assets, including sample images used throughout the
  README examples.
- `results/` &mdash; a suggested directory layout for storing trained models and
  experiment outputs.

## Evaluation

Validation results bundled with the repository show that the detector predicts
counts very close to the ground truth. Across three validation frames, the
model recorded a mean absolute error of 1.33 cells (median 1, maximum 2). In
total it predicted 18 cells against 16 labelled cells. Per-image details are
available in `results/report_val.csv`.

## TODO

### UI interface _(assigned to S.Yeon)_

- [ ] Provide an entry point to choose between a live microscope feed and an
      imported image for cell counting.
- [ ] Integrate direct interpretation from a connected microscope adapter.
- [ ] Enable real-time region selection while viewing the microscope feed.
- [ ] Invoke the counting model and present a fixed annotated image along with
      the predicted count (no live overlay required).
- [ ] Allow users to add or remove bounding boxes and update the cell count
      interactively.
- [ ] Support exporting the analysed image, mirroring the workflow for imported
      images.

### Model architecture diagram

The repository ships a PlotNeuralNet-based script that renders a RetinaNet
overview diagram matching the detector configured in
`cell_counting/model.py`. Generate the `.tex` file (and optionally compile it to
PDF when `pdflatex` is installed) with:

```bash
python docs/scripts/render_model_diagram.py --compile
```

The TikZ source lands in `docs/assets/retinanet_architecture.tex`. If LaTeX is
available in your environment, the script also tries to create a standalone PDF
next to the TeX file.

Refer to the [tests](tests/) folder for basic smoke tests that validate the
package installations and the provided inference utilities.

## Examples

The example assets live in `docs/assets/` as conventional JPG and PNG files so
you can inspect them directly or reuse them in your own experiments. The input
frame below matches the microscope crop supplied for this task and the
accompanying output shows the resulting overlay. The helper script
`docs/scripts/generate_examples.py` (which falls back to a lightweight
threshold-based segmentation when PyTorch is unavailable) regenerates both
artifacts from the source imagery and keeps the repository reproducible.

| Sample hemocytometer input | Annotated output |
| --- | --- |
| ![Sample input](docs/assets/seq0432_jpg.rf.f16687b29f969b08fdc2900f51b3e5d3.jpg) | ![Annotated output](docs/assets/cell_counting_result.png) |

The repository inference helpers (`cell_counting.count_cells` or
`cell_counting.inference.predict_image`) will recreate the overlay when PyTorch
is available. In environments where the heavy dependencies cannot be installed,
the regeneration script resorts to intensity-based segmentation to draw
bounding boxes so that the documentation remains illustrative.

