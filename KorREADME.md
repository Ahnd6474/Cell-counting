# 셀 카운팅

RetinaNet(ResNet-50 FPN) 검출기를 활용해 자동으로 세포를 세는 데 필요한 유틸리티와 모델을 제공합니다. 이 저장소는 운영 환경에서 사용되는 노트북에 전력을 공급하는 구성 요소를 정리한 것입니다.
<img src="docs/assets/cell_counting_result.png" alt="주석이 추가된 셀 카운팅 예시" width="400">


## 프로젝트 개요

이 저장소는 혈구계수기(hemocytometer) 이미지를 탐지하고 세기 위한 구성 요소를 패키징합니다. 학습된 RetinaNet ResNet-50 FPN 검출기, 배치 및 단일 이미지 추론을 위한 Python API, 그리고 대화형 실험을 위한 선택적 Streamlit 애플리케이션이 포함되어 있습니다. 코드는 원본 `hepatocytometer.ipynb` 워크플로를 반영하면서도 다른 프로젝트에서 쉽게 설치하고 재사용할 수 있도록 구성되어 있습니다.

## 특징

- 가중치 로딩 헬퍼가 사전 구성된 RetinaNet(ResNet-50 FPN) 모델 래퍼
- 파일 또는 Pillow 이미지를 대상으로 스크립트형 추론을 수행하는 간단한 `load_model`, `count_cells` API
- CSV 요약과 주석이 추가된 오버레이를 모두 내보내는 배치 예측 유틸리티
- 브라우저에서 직접 예측을 시각화하는 Streamlit 데모

## 빠른 시작

1. 저장소를 클론하고 프로젝트 디렉터리로 이동합니다.

   ```bash
   git clone https://github.com/<your-org>/Cell-counting.git
   cd Cell-counting
   ```

2. (권장) 독립적인 환경을 위해 가상 환경을 생성하고 활성화합니다.
3. 공개된 가중치에 맞춰 고정된 실행 시 의존성을 설치합니다.

   ```bash
   pip install -r requirements.txt
   ```

4. (아래 참고) 사전 학습된 검출기 가중치를 받아 기본 경로(`results/models/best.pt`) 또는 원하는 위치에 배치합니다.
5. 배경 제거 헬퍼를 사용할 계획이라면 빈 혈구계수기 챔버를 촬영한 기준 이미지를 다운로드합니다.
6. Streamlit 데모를 실행하거나 Python API를 호출해 모든 것이 정상 작동하는지 확인합니다.

## Python 사용법

### 모델 로딩

```python
from cell_counting import load_model

model = load_model(
    weights_path="results/models/best.pt",
    device="cuda:0",  # 또는 "cpu"
    image_size=640,
)

count, boxes = model.count_cells(
    "docs/assets/seq0432_jpg.rf.f16687b29f969b08fdc2900f51b3e5d3.jpg",
    blank_image="path/to/blank_reference.jpg",  # 사용자가 제공하는 선택적 프레임
)
print(f"Detected {count} cells")
```

> **참고:** 실제 현미경 촬영 이미지(JPG 또는 PNG)를 사용하세요. 저장소에는 `docs/assets/`에 래스터 예제가 포함되어 있어 README 전반에서 사용된 동일한 PNG/JPG 예제를 참조할 수 있습니다.

### 추론 헬퍼

```python
from cell_counting import count_cells

count, boxes, annotated = count_cells(
    "docs/assets/seq0432_jpg.rf.f16687b29f969b08fdc2900f51b3e5d3.jpg",
    weights_path="results/models/best.pt",
    device="cpu",
    blank_image="path/to/blank_reference.jpg",  # 사용자가 제공하는 선택적 프레임
    return_image=True,
    draw=True,
)
annotated.save("prediction.jpg")
print(f"Predicted {count} cells with {len(boxes)} bounding boxes")
```

로컬에서 실험할 때는 일치하는 빈 기준 프레임을 제공하면 탐지 이전에 배경 아티팩트를 제거하는 데 도움이 됩니다. `blank_image`가 필요 없는 경우 생략해도 됩니다.

### 상세 워크플로

1. **가중치 준비** — 학습된 RetinaNet 체크포인트를 `results/models/best.pt`에 배치합니다. 맞춤형 폴더 구성을 선호한다면 `weights_path` 인자를 통해 다른 위치도 지정할 수 있습니다.
2. **모델 로딩** — `load_model()`을 호출해 검출기를 구성하고 체크포인트를 로드합니다. GPU가 있다면 `device="cuda:0"`를 전달하고, 그렇지 않다면 기본값인 CPU를 사용합니다.
3. **(선택) 빈 프레임으로 보정** — 빈 혈구계수기 챔버 이미지를 촬영해 `blank_image`로 제공하면 전처리 루틴이 이 기준을 빼서 조명 아티팩트를 완화합니다.
4. **추론 실행** — 지속적으로 모델을 사용할 때는 `model.count_cells(...)`를, 간단한 스크립트에는 동일한 인자를 받으면서 모델 캐싱을 처리하는 함수형 헬퍼 `count_cells(...)`를 사용합니다.
5. **검출 임계값 조정** — 예상되는 세포 형태에 맞게 `conf`, `nms_iou`, `size_min`, `size_max`를 조정하세요. 예를 들어 거짓 양성을 줄이려면 `conf`를 높이고, 찌꺼기가 세포로 잘못 계산된다면 `size_max`를 줄입니다.
6. **결과 내보내기** — `return_image=True` 또는 `out_path="prediction.jpg"`를 설정해 주석이 포함된 오버레이를 저장합니다. 반환되는 원시 바운딩 박스는 이미지 좌표로 제공되므로 후처리를 자유롭게 진행할 수 있습니다.
7. **배치 처리** — `cell_counting.inference.predict_folder(...)`를 호출해 전체 디렉터리를 순차적으로 처리하고, 주석 이미지를 저장하며 CSV 요약을 `out_dir`에 생성합니다.

추가 옵션은 `cell_counting/inference.py`에서 확인할 수 있으며, 오버레이용 글꼴 사용자화와 빈 프레임 전처리 훅도 포함되어 있습니다.

## Streamlit 앱

동일한 실행 의존성으로 대화형 데모를 사용할 수 있습니다. 의존성을 설치하고 학습된 가중치를 다운로드한 후 프로젝트 루트에서 다음 명령으로 앱을 실행합니다.

```bash
streamlit run streamlit_app.py
```

인터페이스는 체크포인트 선택, 현미경 이미지 업로드, 주석이 추가된 결과 다운로드 과정을 안내합니다. 사이드바에서 신뢰도 임계값을 조정하고, 빈 프레임 보정을 전환하며, 코드를 작성하지 않고도 검출 수치를 확인할 수 있습니다.

## 추가 자료

- `hepatocytometer.ipynb` — 학습 및 평가 워크플로가 포함된 원본 탐색용 노트북
- `docs/` — README 예제 전반에서 사용된 샘플 이미지를 포함한 문서화 자산
- `results/` — 학습된 모델과 실험 결과를 저장하기 위한 권장 디렉터리 레이아웃

## 평가

저장소에 함께 포함된 검증 결과에 따르면 검출기는 정답에 매우 근접한 셀 수를 유지합니다.
세 개의 검증 프레임 전반에서 모델은 평균 절대 오차 1.33개(중앙값 1, 최대 2)를 기록했습니다.
전체적으로 16개의 라벨링된 셀에 대해 18개의 셀을 예측했습니다.
이미지별 세부 내역은 `results/report_val.csv`에서 확인할 수 있습니다.

## TODO

### UI 인터페이스 _(담당: S.Yeon)_

- [ ] 실시간 현미경 피드와 가져온 이미지를 선택할 수 있는 진입점을 제공합니다.
- [ ] 연결된 현미경 어댑터에서 직접 영상을 가져오는 기능을 통합합니다.
- [ ] 현미경 피드를 보면서 실시간으로 영역을 선택할 수 있도록 지원합니다.
- [ ] 카운팅 모델을 호출해 고정된 주석 이미지를 예측된 개수와 함께 표시합니다(실시간 오버레이는 필요 없음).
- [ ] 사용자가 바운딩 박스를 추가하거나 제거해 세포 수를 상호작용적으로 업데이트할 수 있도록 합니다.
- [ ] 분석된 이미지를 가져온 이미지 워크플로와 동일하게 내보낼 수 있도록 지원합니다.
