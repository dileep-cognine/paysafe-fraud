# syntax=docker/dockerfile:1
# Python 3.13.13 matches the validated local runtime. The image digest pins the
# exact official slim Bookworm base used by both build and runtime stages.
ARG PYTHON_IMAGE=python:3.13.13-slim-bookworm@sha256:f576b530293e74140ea91d262232648d5c4f45640a95ec447757701bfcacf034

FROM ${PYTHON_IMAGE} AS builder

WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# The package metadata remains the single source of runtime dependencies.
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN python -m pip wheel --constraint requirements.lock --wheel-dir /wheels .

FROM ${PYTHON_IMAGE} AS runtime

RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=prod \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=1 \
    LOG_LEVEL=info

RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --home-dir /app --create-home appuser

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels fraud-scoring \
    && rm -rf /wheels

# Profiles are non-secret defaults. Runtime variables may override any value.
COPY --chown=appuser:appuser configs ./configs

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)"

# Do not use reload in the container. Environment variables supply host, port,
# worker count, log level, and MLflow connection details at runtime.
CMD ["sh", "-c", ": \"${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI to your reachable MLflow server}\"; exec uvicorn fraud_scoring.api:create_app_from_environment --factory --host \"${API_HOST:-0.0.0.0}\" --port \"${API_PORT:-8000}\" --workers \"${API_WORKERS:-1}\" --log-level \"${LOG_LEVEL:-info}\""]
