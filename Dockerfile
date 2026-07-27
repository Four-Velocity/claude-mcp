# syntax=docker/dockerfile:1

# Use the official uv image (Debian slim) with a pinned Python 3.13.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# uv configuration for reproducible, container-friendly installs.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install dependencies first (cached) using only the lockfile + manifest.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the application source (including pyproject.toml so the version is
# readable at runtime) and finalise the environment.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8080

# Bind to all interfaces on the Fly-provided port via the package entrypoint.
CMD ["python", "-m", "claude_mcp"]
