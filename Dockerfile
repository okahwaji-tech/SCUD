FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY scud/ scud/
COPY train.py .
COPY configs/ configs/

# Install package
RUN uv pip install --system -e ".[protein,text]"

# Default command
CMD ["scud-train", "--help"]
