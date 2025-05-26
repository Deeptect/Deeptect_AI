# 🧠 Deeptect_AI - Deepfake Detection Inference

본 프로젝트는 학습된 Deepfake 탐지 모델을 기반으로 추론을 실행하는 코드와 구조를 포함하고 있으며, 학습 데이터로는 FaceForensics++(FF++) 데이터셋을 사용했습니다.

---

## 📦 1. Python 환경 및 의존성 설치

```bash
# Python 가상환경 권장
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 의존성 설치
pip install -r requirements.txt
```

> 필수 패키지 예시:  
> `torch`, `torchvision`, `gdown`, `opencv-python`, `numpy`, `pandas`

---

## ⬇️ 2. 학습된 모델 가중치 다운로드

Google Drive에 업로드된 모델 가중치를 자동으로 다운로드합니다.

```bash
pip install gdown
python download_weights.py
```

> 다운로드된 파일은 `weights/best_model.pth` 경로에 저장됩니다.  
> 내부적으로 Google Drive 공유 링크에서 추출한 ID를 사용합니다.

---

## 📁 3. FF++ Dataset 구성 안내

본 프로젝트는 공개 데이터셋인 **[FaceForensics++ (FF++)](https://github.com/ondyari/FaceForensics)**를 기반으로 학습되었습니다.

- 다운로드 링크: https://github.com/ondyari/FaceForensics#download
- 사용자는 약관에 동의하고 직접 다운로드해야 합니다.

데이터 디렉토리 구조는 아래와 같아야 합니다:

```
datasets/
├── FaceForensics++/
│   ├── original/             # Real 영상
│   ├── Deepfakes/            # Fake 영상
│   └── labels.csv            # real / fake 라벨
```

> `labels.csv`에는 영상 파일 이름과 라벨(`real`, `fake`)이 포함되어 있어야 합니다.

---

## ▶️ 4. 추론 코드 실행 예시

```bash
python inference/inference.py
```

예시 출력:
```
Input: sample.mp4
Prediction: FAKE
Confidence: 0.9812
```

> `--input_path` 옵션에 추론할 영상의 경로를 지정하세요.

---

## 📌 프로젝트 구조

```
Deeptect_AI/
├── inference/               # 추론 코드
├── train/                   # 학습 코드
├── datasets/                # (git에는 포함되지 않음)
├── weights/                 # 모델 가중치 저장 경로
├── download_weights.py      # 가중치 다운로드 스크립트
├── requirements.txt
└── README.md
```

---

## 📬 문의

질문이나 제안 사항은 [GitHub Issue](https://github.com/Deeptect/Deeptect_AI/issues) 또는 팀 내부 채널을 통해 문의해 주세요.
