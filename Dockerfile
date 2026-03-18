FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# System deps for building wheels (kept minimal)
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install uv (single tool for venv + deps)
RUN python -m pip install --no-cache-dir uv

# Use a fixed virtualenv path for caching/layering
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (better caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source + data and install project
COPY src ./src
COPY players.csv ./players.csv
RUN uv sync --frozen --no-dev

# Run as non-root
RUN useradd -m appuser
USER appuser

ENTRYPOINT ["nba-draft-success"]
CMD ["train", "--model", "rf"]

