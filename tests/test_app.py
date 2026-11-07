import base64
import gzip
import io
import os
import sys

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["model_version"]


def test_model_info():
    assert "test_acc" in client.get("/api/model").json()


def test_predict_sample_digit():
    img = Image.open(os.path.join(os.path.dirname(__file__), "..", "deploy", "sample_digit.png"))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = client.post("/predict", json={"image": base64.b64encode(buf.getvalue()).decode()})
    assert r.status_code == 200
    assert r.json()["digit"] == 7


def test_empty_canvas_rejected():
    r = client.post("/predict", json={"image": png_b64(np.full((280, 280), 255, np.uint8))})
    assert r.status_code == 400


def test_bad_image_rejected():
    assert client.post("/predict", json={"image": "not-an-image"}).status_code == 400
