FROM docker.io/library/python:3.12-slim-bookworm

# CPU by default. GPU builds need their own compatible torch wheel index/host.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp
WORKDIR /app

COPY deploy/requirements-core.txt deploy/requirements-http.txt /tmp/requirements/
RUN python -m pip install --no-cache-dir torch==2.14.0 --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir -r /tmp/requirements/requirements-http.txt \
    && rm -rf /tmp/requirements \
    && groupadd --gid 10001 pipeline \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin pipeline

COPY cvdl/__init__.py cvdl/models.py cvdl/pixels.py cvdl/photo_inference.py cvdl/cell_diagnostics.py cvdl/cell_metrics.py cvdl/photo_verdict.py /app/cvdl/
COPY cvdl/pipeline/*.py /app/cvdl/pipeline/
COPY LSB-Dataset-Generator/lsb_train_grid.py LSB-Dataset-Generator/lsb_source.py LSB-Dataset-Generator/lsb_raw.py /app/LSB-Dataset-Generator/
COPY deploy/run13-rule-c-v1/ /app/deploy/run13-rule-c-v1/

USER 10001:10001
EXPOSE 8000
# Exec form forwards stop signals. Host/port remain set when CMD is overridden.
ENTRYPOINT ["python", "-m", "cvdl.pipeline.serve", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--device", "cpu"]
