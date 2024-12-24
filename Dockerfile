ARG APP_ROOT="/src"
ARG PYTHON_VERSION="3.11"
ARG USER="appuser"


# Stage 1: Builder Image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

LABEL site="https://qte77.github.io/SegFormerQuantization/"
LABEL author="qte77"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY uv.lock .
# https://docs.astral.sh/uv/concepts/projects/sync/
RUN set -xe \
    && pip install --no-cache-dir uv \
    && uv sync --frozen


# Stage 2: Runtime Image
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG APP_ROOT
ARG PYTHON_VERSION
ARG USER
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=${APP_ROOT}
#    WANDB_KEY=${WANDB_KEY} \
#    WANDB_DISABLE_CODE=true

RUN set -xe \
    && useradd --create-home ${USER}
USER ${USER}
WORKDIR ${APP_ROOT}
ENV PATH="${APP_ROOT}:${PATH}"

# TODO where are site-packages located?
# .venv/bin, .venv/lib, turn off uv venv?
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages \
    /usr/local/lib/python${PYTHON_VERSION}/site-packages

COPY --chown=${USER}:${USER} ${APP_ROOT} .

# https://docs.astral.sh/uv/concepts/projects/run/
CMD ["uv", "run", "--locked", "--no-sync", "python", "-m", "."]