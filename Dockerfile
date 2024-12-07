FROM docker.io/library/python:3.10-slim

LABEL site="https://qte77.github.io/SegFormerQuantization/"
LABEL author="qte77"

ARG USER="appuser"
ARG APP_ROOT="/app"
ARG APP_EP="app.py"
ARG REQS="./pyproject.toml"
# ARG WANDB_KEYFILE=".wandb/wandb.key"
# ARG WANDB_KEY="<token>"

ENV PYTHONDONTWRITEBYTECOD=1
ENV PYTHONUNBUFFERED=1
ENV PATH="${APP_ROOT}:/home/user/.local/bin:${PATH}"
# ENV WANDB_KEY=${WANDB_KEY}

# several RUN to produce separate pip layers ?
RUN set -xe \
    && useradd -m ${USER} \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && pip install --no-cache-dir poetry==1.8.4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${USER}
WORKDIR ${APP_ROOT}

COPY --chown=${USER}:${USER} ${REQS} .
COPY --chown=${USER}:${USER} ${APP_ROOT} .
# COPY --chown=${USER}:${USER} ${WANDB_KEYFILE} \
#   "${HOME}/${WANDB_KEYFILE}"

RUN set -xe \
    && poetry config virtualenvs.create true \
    && poetry install --no-interaction --no-ansi

# EXPOSE 8080

CMD ["poetry", "run", "python", "${APP_EP}"]
# CMD ["python", "-m", "${APP_EP}"]
# RUN chmod +x ${APP_EP}
# ENTRYPOINT ["${APP_EP}"]

# TODO FastAPI etc.
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "-k", \
#     "uvicorn.workers.UvicornWorker", "app.py:app"]
