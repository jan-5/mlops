FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface \
    WANDB_DIR=/app/experiments/wandb

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .

RUN uv sync --frozen

RUN mkdir -p /app/models/checkpoints /app/experiments/wandb

ENTRYPOINT ["uv", "run", "python", "main.py"]

CMD ["--checkpoint_dir","models/checkpoints","--learning_rate","2e-5"]