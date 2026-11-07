FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY static/ ./static/
COPY model/ ./model/

# 빌드 단계에서 모델이 정상 로드·추론되는지 검증 (실패 시 이미지 빌드 실패)
RUN python -c "import numpy as np, app; m=app.load_model(); p=app.predict(np.zeros((1,784),np.float32)); assert p.shape==(10,); print('model ok', app.model_info())"

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
