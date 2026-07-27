"""FastAPI application wiring the MCP server together with plain HTTP endpoints.

The MCP server is mounted as a streamable-HTTP sub-application (reachable at
``/mcp``), while ``/version`` and ``/health`` are served directly by FastAPI.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI

from claude_mcp.config import get_settings
from claude_mcp.openai_client import OpenAIImageClient
from claude_mcp.server import close_image_client, mcp, set_image_client
from claude_mcp.version import read_version


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        A configured :class:`fastapi.FastAPI` instance with the MCP server mounted
        at ``/mcp`` and ``/version`` / ``/health`` endpoints registered.
    """
    # Build the streamable-HTTP sub-app first: this lazily creates the MCP
    # session manager that the lifespan below needs to run.
    mcp_app = mcp.streamable_http_app()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage startup/shutdown: load version, OpenAI client, and MCP session manager."""
        settings = get_settings()
        app.state.version = read_version()
        set_image_client(
            OpenAIImageClient(
                settings.openai_api_key,
                base_url=settings.openai_base_url,
                model=settings.image_model,
                timeout_seconds=settings.request_timeout_seconds,
            )
        )
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            await close_image_client()

    app = FastAPI(
        title="Claude OpenAI Image MCP",
        version=read_version(),
        lifespan=lifespan,
    )

    @app.get("/version")
    async def version() -> dict[str, str]:
        """Return the running service version read from ``pyproject.toml`` at startup."""
        current: str = app.state.version
        return {"version": current}

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Return a simple liveness payload for platform health checks."""
        return {"status": "ok"}

    # Mount the MCP streamable-HTTP app last so the explicit routes above win.
    app.mount("/", mcp_app)
    return app


app = create_app()
