"""FastAPI application wiring the MCP server together with plain HTTP endpoints.

The MCP server is mounted as a streamable-HTTP sub-application (reachable at
``/mcp``), while ``/version``, ``/health``, ``/images/{id}`` and — when OAuth is
enabled — the ``/login`` pages are served directly by FastAPI.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from claude_mcp.auth import LoginError, LoginThrottledError, render_login_page
from claude_mcp.config import get_settings
from claude_mcp.openai_client import OpenAIImageClient
from claude_mcp.server import (
    auth_provider,
    close_image_client,
    get_image_store,
    mcp,
    set_image_client,
)
from claude_mcp.storage import content_type_for
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

    @app.api_route("/images/{filename}", methods=["GET", "HEAD"])
    async def get_image(filename: str) -> FileResponse:
        """Serve a previously generated image by its stored file name.

        Args:
            filename: The stored image file name from the generated Markdown link.

        Returns:
            The image file with an appropriate content type.

        Raises:
            HTTPException: 404 if the name is invalid or the file does not exist.
        """
        path = get_image_store().resolve(filename)
        if path is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(
            path,
            media_type=content_type_for(path),
            headers={"Cache-Control": "public, max-age=3600"},
        )

    if auth_provider is not None:
        provider = auth_provider

        @app.get("/login", response_class=HTMLResponse)
        async def login_form(rid: str) -> HTMLResponse:
            """Render the password prompt that authorizes a Claude connection.

            Args:
                rid: The pending authorization request id created by ``/authorize``.

            Returns:
                The login page. ``no-store`` prevents the form (and the request id
                in its query string) from being cached.
            """
            return HTMLResponse(render_login_page(rid), headers={"Cache-Control": "no-store"})

        # response_model=None: the return type is a union of Response subclasses,
        # which FastAPI must not try to turn into a Pydantic response model.
        @app.post("/login", response_model=None)
        async def login_submit(
            request: Request,
            rid: Annotated[str, Form()],
            password: Annotated[str, Form()],
        ) -> HTMLResponse | RedirectResponse:
            """Verify the password and hand an authorization code back to Claude.

            Args:
                request: The incoming request, used for the throttling key.
                rid: The pending authorization request id.
                password: The submitted password.

            Returns:
                A redirect to the OAuth client's callback on success, or the login
                page re-rendered with an error message on failure.
            """
            client_key = request.client.host if request.client else "unknown"
            try:
                redirect_url = await provider.complete_login(rid, password, client_key)
            except LoginThrottledError as exc:
                return HTMLResponse(
                    render_login_page(rid, str(exc)),
                    status_code=429,
                    headers={"Cache-Control": "no-store"},
                )
            except LoginError as exc:
                return HTMLResponse(
                    render_login_page(rid, str(exc)),
                    status_code=401,
                    headers={"Cache-Control": "no-store"},
                )
            return RedirectResponse(
                redirect_url, status_code=302, headers={"Cache-Control": "no-store"}
            )

    # Mount the MCP streamable-HTTP app last so the explicit routes above win.
    app.mount("/", mcp_app)
    return app


app = create_app()
