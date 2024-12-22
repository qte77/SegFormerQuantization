ARG APP_EP="app"
ARG APP_ROOT="/app"
ARG UV_VER="0.5.11"
ARG PYTHON_VERSION="3.11"
ARG USER="appuser"


# Stage 1: Builder Image
# FROM docker.io/library/python:${PYTHON_VERSION}-slim AS builder
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

LABEL site="https://qte77.github.io/SegFormerQuantization/"
LABEL author="qte77"

ARG UV_VER
ARG USER
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN set -xe \
    && useradd --create-home ${USER} \
    && apt-get update \
    && pip install --no-cache-dir uv==${UV_VER} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# USER ${USER}
COPY --chown=${USER}:${USER} ./pyproject.toml uv.lock ./
RUN set -xe \
    uv sync


# Stage 2: Runtime Image
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG APP_EP
ARG APP_ROOT
ARG PYTHON_VERSION
ARG USER
# ENV PYTHONDONTWRITEBYTECODE=1 \
#    PYTHONUNBUFFERED=1
#    PYTHONPATH=/app \
#    WANDB_KEY=${WANDB_KEY} \
#    WANDB_DISABLE_CODE=true \
#    PYTHONUNBUFFERED=1

USER ${USER}
WORKDIR ${APP_ROOT}
ENV PATH="${APP_ROOT}:/home/${USER}/.local/bin:${PATH}"
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages \
    /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY --from=builder /home/${USER}/.local/bin /home/${USER}/.local/bin
# COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=${USER}:${USER} ${APP_ROOT} .
# COPY --chown=${USER}:${USER} ${WANDB_KEYFILE} \
#   "${HOME}/${WANDB_KEYFILE}"

CMD ["uv", "run", "${APP_EP}"]

# Optional: health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#    CMD curl -f http://localhost:8000/health || exit 1
