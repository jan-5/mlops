# ---- Base: slim Python, good for Playground/Codespaces (CPU only) ----
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface \
    WANDB_DIR=/app/experiments/wandb

# Minimal OS deps; keep it light
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (package manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

# ---- Resolve and cache deps first (fast rebuilds) ----
# If uv.lock exists, we use it for fully reproducible builds
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# ---- Add project files ----
COPY . .

# Install project (and any extras declared in pyproject)
RUN uv sync --frozen

# Create runtime dirs so we can optionally mount over them
RUN mkdir -p /app/models/checkpoints /app/experiments/wandb

# (Optional) If your torch in pyproject resolves to CUDA by default on some hosts,
# force CPU wheel with:
# RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch --upgrade

# Default entrypoint per assignment (Task 1/2/3)
ENTRYPOINT ["uv", "run", "python", "main.py"]

# Sensible defaults; you can override at `docker run ... CMD ...`
CMD ["--checkpoint_dir","models/checkpoints","--learning_rate","2e-5"]