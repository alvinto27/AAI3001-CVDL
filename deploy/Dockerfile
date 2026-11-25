FROM docker.io/library/python:3.12-slim-bookworm

# Default CPU image: ONNX Runtime, no PyTorch. deploy/Dockerfile.torch keeps
# the PyTorch image (CPU or GPU wheels).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    CVDL_ENGINE=onnx
WORKDIR /app

COPY deploy/requirements-onnx.txt /tmp/requirements/
RUN python -m pip install --no-cache-dir -r /tmp/requirements/requirements-onnx.txt \
    && rm -rf /tmp/requirements \
    && groupadd --gid 10001 pipeline \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin pipeline

# Only the torch-free modules. bundle.py/export_onnx.py stay out: they need torch.
COPY cvdl/__init__.py cvdl/photo_inference.py cvdl/cell_metrics.py cvdl/photo_verdict.py /app/cvdl/
COPY cvdl/pipeline/__init__.py cvdl/pipeline/__main__.py cvdl/pipeline/contract.py cvdl/pipeline/inputs.py \
     cvdl/pipeline/onnx_engine.py cvdl/pipeline/serve.py /app/cvdl/pipeline/
COPY LSB-Dataset-Generator/lsb_train_grid.py LSB-Dataset-Generator/lsb_source.py LSB-Dataset-Generator/lsb_raw.py /app/LSB-Dataset-Generator/
COPY deploy/run13-rule-c-v1/deploy.json deploy/run13-rule-c-v1/model.onnx deploy/run13-rule-c-v1/onnx.json /app/deploy/run13-rule-c-v1/
COPY deploy/run18-rule-c-v1/deploy.json deploy/run18-rule-c-v1/model.onnx deploy/run18-rule-c-v1/onnx.json /app/deploy/run18-rule-c-v1/
COPY deploy/run22-rule-c-v1/deploy.json deploy/run22-rule-c-v1/model.onnx deploy/run22-rule-c-v1/onnx.json /app/deploy/run22-rule-c-v1/

USER 10001:10001
EXPOSE 8000
# Exec form forwards stop signals. Host/port remain set when CMD is overridden.
ENTRYPOINT ["python", "-m", "cvdl.pipeline.serve", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--engine", "onnx"]
