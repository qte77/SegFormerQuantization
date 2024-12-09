ARG APP_EP="app"
ARG APP_ROOT="/app"
ARG POETRY_VER="1.8.4"
ARG PYTHON_VERSION="3.10"
ARG USER="appuser"


# Stage 1: Builder Image
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS builder

LABEL site="https://qte77.github.io/SegFormerQuantization/"
LABEL author="qte77"

ARG POETRY_VER
ARG USER
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN set -xe \
    && useradd --create-home ${USER} \
    && apt-get update \
    && pip install --no-cache-dir poetry==${POETRY_VER} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# USER ${USER}
COPY --chown=${USER}:${USER} ./pyproject.toml poetry.lock ./
RUN set -xe \
    && poetry config virtualenvs.create false \
    && poetry config installer.max-workers 10 \
    && poetry install --no-interaction --no-ansi


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

CMD ["poetry", "run", "python", "-m", "${APP_EP}"]

# Optional: health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#    CMD curl -f http://localhost:8000/health || exit 1
