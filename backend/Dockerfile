
# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models DATA_DIR=/data

WORKDIR /srv

ARG INSTALL_OCR=true

# Tesseract lets VerityGate read scanned PDFs (OCR).
# Skip with: docker compose build --build-arg INSTALL_OCR=false backend
RUN if [ "$INSTALL_OCR" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*; \
    fi

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# CPU-only torch keeps the image far smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-ml.txt ./

RUN pip install -r requirements.txt -r requirements-ml.txt

RUN useradd --create-home --uid 10001 verigate && \
    mkdir -p /data /models && \
    chown -R verigate:verigate /data /models

FROM base AS runtime

COPY app ./app

USER verigate

EXPOSE 8000

# Behind nginx every request comes from the proxy.
ENV FORWARDED_ALLOW_IPS="*"

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--timeout-graceful-shutdown", "20"]

FROM base AS test

COPY requirements-dev.txt ./

RUN pip install -r requirements-dev.txt

COPY app ./app
COPY tests ./tests
COPY evaluation ./evaluation
COPY pyproject.toml ./

# Fix permissions so the non-root user can generate evaluation PDFs.
RUN chown -R verigate:verigate /srv/evaluation

USER verigate

# Fast tests with deterministic stand-in models.
# For real-model tests: pytest -m real_models
CMD ["pytest", "-q"]
