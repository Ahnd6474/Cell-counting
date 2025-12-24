# Cell-counting / 셀 카운팅

Utilities and models for automated cell counting with the RetinaNet
(ResNet-50 FPN) detector that powers the production notebooks.

RetinaNet(ResNet-50 FPN) 검출기를 활용해 자동으로 세포를 세는 데 필요한
유틸리티와 모델을 제공합니다.

<img src="docs/assets/cell_counting_result.png" alt="Annotated cell counting example" width="400">

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
</p>

## TODO

### UI interface _(assigned to S.Yeon)_

- [x] Provide an entry point to choose between a live microscope feed and an
      imported image for cell counting.
- [x] Integrate direct interpretation from a connected microscope adapter.
- [x] Enable real-time region selection while viewing the microscope feed.
- [x] Invoke the counting model and present a fixed annotated image along with
      the predicted count (no live overlay required).
- [x] Allow users to add or remove bounding boxes and update the cell count
      interactively.
- [x] Support exporting the analysed image, mirroring the workflow for imported
      images.

## 📑 Table of Contents / 목차
- [TODO](#todo)
- [Project Overview](#project-overview)
- [Features](#features)
- [Quickstart](#quickstart)
- [Python Usage](#python-usage)
- [Streamlit App](#streamlit-app)
- [Additional Resources](#additional-resources)
- [Evaluation](#evaluation)
- [Examples](#examples)

## Project overview / 프로젝트 개요

This repository packages everything needed to detect and count cells in
hemocytometer images. It bundles trained RetinaNet weights, Python helpers for
single or batch inference, and a Streamlit app for interactive experimentation.
The implementation mirrors the original `hepatocytometer.ipynb` workflow while
remaining easy to install and extend.

이 저장소는 혈구계수기 이미지를 탐지하고 세기 위한 모든 구성 요소를
포함합니다. 학습된 RetinaNet 가중치, 단일·배치 추론을 위한 Python 헬퍼,
그리고 대화형 실험용 Streamlit 앱을 함께 제공하며, 원본
`hepatocytometer.ipynb` 워크플로를 반영하면서도 설치와 확장이 쉽도록
정리되어 있습니다.

## Features / 특징

- RetinaNet (ResNet-50 FPN) wrapper with convenient weight-loading helpers.  
  가중치 로딩이 간편한 RetinaNet(ResNet-50 FPN) 래퍼 제공
- `load_model` and `count_cells` APIs for scripted inference on files or Pillow
  images.  
  파일 또는 Pillow 이미지를 대상으로 스크립트형 추론을 수행하는
  `load_model`, `count_cells` API
- Batch prediction utilities that export CSV summaries and annotated overlays.  
  CSV 요약과 주석 이미지를 동시에 내보내는 배치 예측 유틸리티
- Streamlit demo that visualises predictions directly in the browser.  
  브라우저에서 예측을 시각화하는 Streamlit 데모

## Quickstart / 빠른 시작

1. Clone the repository and move into the project directory.  
   저장소를 클론하고 프로젝트 디렉터리로 이동합니다.

   ```bash
   git clone https://github.com/<your-org>/Cell-counting.git
   cd Cell-counting
   ```

2. (Optional) Create and activate a virtual environment.  
   (선택 사항) 가상 환경을 생성하고 활성화합니다.
3. Install runtime dependencies.  
   런타임 의존성을 설치합니다.

   ```bash
   pip install -r requirements.txt
   ```

4. Download the pretrained detector weights and place them in
   `results/models/best.pt` (or provide a custom path when loading the model).  
   사전 학습된 가중치를 `results/models/best.pt`에 두거나 모델 로딩 시
   원하는 경로를 지정합니다.
5. Run the Streamlit demo or use the Python API to confirm the setup.  
   Streamlit 데모를 실행하거나 Python API로 구성을 확인합니다.

## Python usage / Python 사용법

```python
from cell_counting import count_cells, load_model

model = load_model(
    weights_path="results/models/best.pt",
    device="cuda:0",  # or "cpu"
    image_size=640,
)

count, boxes, annotated = count_cells(
    "docs/assets/seq0432_jpg.rf.f16687b29f969b08fdc2900f51b3e5d3.jpg",
    weights_path="results/models/best.pt",
    blank_image="path/to/blank_reference.jpg",  # optional
    return_image=True,
    draw=True,
)
annotated.save("prediction.jpg")
print(f"Detected {count} cells across {len(boxes)} boxes")
```

Tips / 팁:

- Provide a blank reference frame to remove background artefacts when
  necessary.  
  필요하다면 빈 기준 프레임을 제공해 배경 아티팩트를 제거하세요.
- Adjust `conf`, `nms_iou`, `size_min`, and `size_max` to tailor detections to
  your imagery.  
  `conf`, `nms_iou`, `size_min`, `size_max` 값을 조정해 이미지 특성에 맞는
  탐지를 얻을 수 있습니다.
- Use `cell_counting.inference.predict_folder(...)` for batch processing with
  CSV summaries and annotated overlays.  
  CSV 요약과 주석 이미지를 함께 생성하는 배치 처리에는
  `cell_counting.inference.predict_folder(...)`를 사용하세요.

## Streamlit app / Streamlit 앱

After installing dependencies and downloading the trained weights, launch the
interactive demo from the project root:

```bash
streamlit run streamlit_app.py
```

Use the sidebar to select checkpoints, upload microscope imagery, adjust
thresholds, toggle blank-frame subtraction, and download annotated results
without writing code. The interface also lets you switch between live capture
and local files, crop a region of interest before running inference, tweak the
predicted bounding boxes, and immediately recalculate the resulting cell count.

사이드바에서 체크포인트 선택, 현미경 이미지 업로드, 임계값 조정, 빈 프레임
보정 토글, 주석 결과 다운로드까지 코드를 작성하지 않고 진행할 수
있습니다.

## Additional resources / 추가 자료

- `hepatocytometer.ipynb` &mdash; original exploratory notebook with the training
  and evaluation workflow.
- `docs/` &mdash; documentation assets, including sample images referenced in this
  README.
- `results/` &mdash; suggested directory layout for storing trained models and
  experiment outputs.

## Evaluation / 평가

Validation results bundled with the repository show that the detector predicts
counts very close to the ground truth. Across three validation frames, the
model recorded a mean absolute error of 1.33 cells (median 1, maximum 2). In
total it predicted 18 cells against 16 labelled cells. Per-image details are
available in `results/report_val.csv`.

저장소에 포함된 검증 결과에 따르면 모델은 정답에 근접한 셀 수를 예측하며,
세 개의 검증 프레임에서 평균 절대 오차 1.33개(중앙값 1, 최대 2)를
기록했습니다. 총 16개의 라벨 셀에 대해 18개를 예측했으며, 이미지별 상세
내역은 `results/report_val.csv`에서 확인할 수 있습니다.

## Model architecture diagram / 모델 구조 다이어그램

Render the RetinaNet overview diagram (matching `cell_counting/model.py`) with:

`cell_counting/model.py`와 일치하는 RetinaNet 개요 다이어그램은 다음 명령으로
렌더링할 수 있습니다.

```bash
python docs/scripts/render_model_diagram.py --compile
```

The TikZ source is saved to `docs/assets/retinanet_architecture.tex`; when
LaTeX is available, the script also produces a PDF.

TikZ 소스는 `docs/assets/retinanet_architecture.tex`에 저장되며, LaTeX이 설치된
환경에서는 PDF도 생성됩니다.

Refer to the [tests](tests/) folder for smoke tests that validate the package
installation and inference utilities.

## Examples / 예시

The example assets live in `docs/assets/` as conventional JPG and PNG files so
you can inspect them directly or reuse them in your own experiments. The input
frame below matches the microscope crop supplied for this task and the
accompanying output shows the resulting overlay. The helper script
`docs/scripts/generate_examples.py` (which falls back to a lightweight
threshold-based segmentation when PyTorch is unavailable) regenerates both
artifacts from the source imagery and keeps the repository reproducible.

예제 자산은 `docs/assets/` 폴더에 JPG/PNG 형태로 포함되어 있어 바로 확인하거나
실험에 재사용할 수 있습니다. 아래 입력 프레임은 제공된 현미경 크롭과 동일하며,
오른쪽 결과는 생성된 오버레이를 보여 줍니다.
PyTorch가 없는 환경에서는 경량화된 임계값 분할로 대체하는
`docs/scripts/generate_examples.py` 스크립트가 두 아티팩트를 재생성해 문서의
재현성을 유지합니다.

| Sample hemocytometer input | Annotated output |
| --- | --- |
| ![Sample input](docs/assets/seq0432_jpg.rf.f16687b29f969b08fdc2900f51b3e5d3.jpg) | ![Annotated output](docs/assets/cell_counting_result.png) |

The repository inference helpers (`cell_counting.count_cells` or
`cell_counting.inference.predict_image`) will recreate the overlay when PyTorch
is available. In environments where the heavy dependencies cannot be installed,
the regeneration script resorts to intensity-based segmentation to draw
bounding boxes so that the documentation remains illustrative.

