# 손글씨 숫자 인식 웹앱 (Handwritten Digit Recognizer)

브라우저 캔버스에 0~9 숫자를 그리면, MNIST 데이터셋으로 학습한 신경망(MLP)이 어떤 숫자인지 추정해 주는 웹 서비스입니다.
Python/FastAPI 백엔드와 HTML/CSS/JavaScript 프론트엔드로 구성되어 있으며, 일반 노트북 CPU에서 동작합니다.

> 결과는 모델의 **추정치**이며 항상 정확하지는 않습니다.

## 구성

| 파일 | 설명 |
|---|---|
| `app.py` | FastAPI 서버 (`/` 화면, `/predict` 추론 API, `/health` 상태 확인) |
| `static/` | 프론트엔드 (캔버스 그리기, 결과·확률 막대 표시, 오류 처리) |
| `train.py` | MNIST 다운로드 → NumPy MLP 학습 → `model/mnist_mlp.npz` 저장 |
| `model/mnist_mlp.npz` | 학습된 가중치 (약 2MB, 테스트 정확도 약 98.7%) |
| `run.bat` | Windows에서 패키지 설치 후 서버 실행 (Docker 없이) |
| `evaluate.py` | 고정 평가셋(MNIST 테스트 1만 장 + 고정 시드 변형셋)으로 모델 비교 |
| `Dockerfile` | 실행 환경 + 모델을 포함한 이미지 (빌드 중 모델 로드 검증) |
| `.github/workflows/ci.yml` | push → 테스트 · 품질 게이트 → Docker Hub 업로드 |
| `.github/workflows/deploy.yml` | CI 성공 → Self-hosted Runner(내 PC)에서 무중단 배포 |
| `deploy/deploy.ps1` | blue/green 배포 + 1초 간격 실시간 모니터링 + 요약 리포트 |

- 모델: 784-512-256-10 다층 퍼셉트론, ReLU, Adam, 평행이동 데이터 증강
- 전처리: 캔버스 이미지 흑백 반전 → 숫자 영역 자르기 → 20×20 축소 → 28×28 중앙(질량 중심) 정렬 (MNIST 형식)
- 추론은 PyTorch/TensorFlow 없이 NumPy만 사용하므로 설치가 가볍습니다.

## 설치

Python 3.9 이상이 필요합니다. PowerShell에서:

```powershell
cd mnist_webapp
python -m pip install -r requirements.txt
```

## 모델 준비

학습된 모델(`model/mnist_mlp.npz`)이 저장소에 포함되어 있어 바로 실행할 수 있습니다.
모델을 다시 학습하려면 (MNIST 자동 다운로드, CPU로 수 분 소요):

```powershell
python train.py
```

## 실행

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

또는 `run.bat`을 더블클릭하세요. 브라우저에서 http://127.0.0.1:8000 에 접속해 숫자를 그리고 **인식하기**를 누르면 됩니다.

## API

```
POST /predict
Content-Type: application/json
{"image": "data:image/png;base64,...."}

→ {"digit": 7, "confidence": 0.99, "probabilities": [10개 확률]}
```

빈 캔버스나 잘못된 이미지는 `400` 오류와 한국어 메시지를 반환합니다.

## CI/CD 파이프라인 (HW3)

```
git push ─▶ [CI · GitHub-hosted]  pytest → 정확도 게이트(≥97%) → docker build → Docker Hub push (:latest, :<sha>)
                 │ 성공 시
                 ▼
          [CD · Self-hosted Windows runner]  docker pull → 새 색상 컨테이너 기동·헬스체크·추론 검증
                                              → nginx 트래픽 전환(graceful reload) → 기존 컨테이너 제거
```

- 사용자는 항상 **http://127.0.0.1:8000** (nginx 프록시) 로 접속하고, 뒤의 앱 컨테이너만 `mnist-blue` ↔ `mnist-green` 으로 교체됩니다.
- 배포 동안 1초 간격으로 실제 `/predict` 요청을 보내 성공/실패, 응답 모델 버전, 요청 간 최대 공백을 기록하고 GitHub Actions **Summary** 와 Artifact(`requests.csv`)로 남깁니다.
- 필요한 GitHub Secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

### 모델 업데이트 (MLOps)

```powershell
pip install -r requirements-train.txt
python train.py --version v2 --augment affine --hidden 800 400 --epochs 20 --dropout 0.2
python evaluate.py model/mnist_mlp.npz
git add model && git commit -m "Model v2" && git push   # → 자동 재배포
```
