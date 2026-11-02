"""손글씨 숫자 인식 웹앱 (FastAPI + NumPy MLP, MNIST 학습)"""
import base64
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "mnist_mlp.npz"

app = FastAPI(title="Handwritten Digit Recognizer")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"모델 파일이 없습니다: {MODEL_PATH} (python train.py 로 생성하세요)")
        d = np.load(MODEL_PATH)
        _model = [(d[f"W{i}"], d[f"b{i}"]) for i in range(3)]
    return _model


def preprocess(img: Image.Image) -> np.ndarray:
    """캔버스 이미지(흰 배경/검은 글씨)를 MNIST 형식(28x28, 중앙 정렬)으로 변환."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    g = np.asarray(img.convert("L"), dtype=np.float32)
    if g.mean() > 127:  # 흰 배경이면 반전
        g = 255 - g
    mask = g > 30
    if not mask.any():
        raise ValueError("빈 캔버스입니다. 숫자를 그려주세요.")
    ys, xs = np.where(mask)
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((nw, nh), Image.LANCZOS), dtype=np.float32)
    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = small
    # 질량 중심을 (14,14)로 이동
    tot = canvas.sum()
    if tot > 0:
        cy = (canvas.sum(1) * np.arange(28)).sum() / tot
        cx = (canvas.sum(0) * np.arange(28)).sum() / tot
        canvas = np.roll(np.roll(canvas, int(round(14 - cy)), 0), int(round(14 - cx)), 1)
    return (canvas / 255.0).reshape(1, 784)


def predict(x: np.ndarray) -> np.ndarray:
    h = x
    layers = load_model()
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    z = h - h.max()
    p = np.exp(z)
    return (p / p.sum())[0]


class PredictRequest(BaseModel):
    image: str  # data URL 또는 base64 PNG


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    load_model()
    return {"status": "ok"}


@app.post("/predict")
def predict_api(req: PredictRequest):
    try:
        data = req.image.split(",", 1)[1] if req.image.startswith("data:") else req.image
        img = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")
    try:
        x = preprocess(img)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")
    try:
        probs = predict(x)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    top = int(np.argmax(probs))
    return {
        "digit": top,
        "confidence": float(probs[top]),
        "probabilities": [float(p) for p in probs],
    }
