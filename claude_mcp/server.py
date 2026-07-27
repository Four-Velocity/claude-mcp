"""MCP server definition and the ``generate_image`` tool.

The server is created with :class:`FastMCP` and served over streamable HTTP so it
can be added to Claude as a custom connector. It exposes a single tool that
generates an image with OpenAI's ``gpt-image-2`` model.

To stay within the ~1 MB MCP tool-response limit enforced by the Claude desktop
and mobile apps, the tool does **not** return the (multi-megabyte) image inline.
Instead it saves the image locally, serves it over HTTP, and returns a small
Markdown image link that Claude renders — the image bytes are then loaded
directly over HTTP rather than through the MCP channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from mcp.server.auth.provider import ProviderTokenVerifier
from mcp.server.auth.settings import (
    AuthSettings,
    ClientRegistrationOptions,
    RevocationOptions,
)
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl, Field
from starlette.requests import Request

from claude_mcp.auth import MCP_SCOPE, SingleUserOAuthProvider
from claude_mcp.auth_store import AuthStore
from claude_mcp.config import Settings, get_settings
from claude_mcp.openai_client import OpenAIImageClient
from claude_mcp.storage import ImageStore, default_auth_storage_path, default_storage_dir


def _build_transport_security(settings: Settings) -> TransportSecuritySettings:
    """Build MCP transport-security settings from application settings.

    FastMCP enables DNS-rebinding protection by default and only allows localhost
    hosts, which would reject the ``*.fly.dev`` (or custom-domain) ``Host`` header
    and break the connector. When no allow-list is configured we therefore disable
    the protection so the public HTTPS endpoint responds on any host; when hosts
    are configured we enable it and honour them.

    Args:
        settings: The application settings providing any configured allow-lists.

    Returns:
        Transport-security settings appropriate for the deployment.
    """
    if settings.allowed_hosts:
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=settings.allowed_hosts,
            allowed_origins=settings.allowed_origins,
        )
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


def _build_auth_provider(settings: Settings) -> SingleUserOAuthProvider | None:
    """Build the OAuth provider when a password is configured.

    Args:
        settings: The application settings.

    Returns:
        The provider, or ``None`` when no password is set (auth disabled).
    """
    if not settings.auth_enabled:
        return None
    assert settings.mcp_auth_password is not None  # guaranteed by auth_enabled
    path = (
        Path(settings.auth_storage_path)
        if settings.auth_storage_path
        else default_auth_storage_path()
    )
    return SingleUserOAuthProvider(
        AuthStore(path),
        password=settings.mcp_auth_password,
        base_url=settings.issuer_url,
        resource_url=settings.resource_url,
        access_token_ttl=settings.access_token_ttl_minutes * 60.0,
        refresh_token_ttl=settings.refresh_token_ttl_days * 86400.0,
    )


def _build_auth_settings(settings: Settings) -> AuthSettings | None:
    """Build MCP ``AuthSettings`` describing this server as both AS and RS.

    Dynamic Client Registration is enabled so Claude can connect without any
    manually configured client id or secret.

    Args:
        settings: The application settings.

    Returns:
        The auth settings, or ``None`` when auth is disabled.
    """
    if not settings.auth_enabled:
        return None
    return AuthSettings(
        issuer_url=AnyHttpUrl(settings.issuer_url),
        resource_server_url=AnyHttpUrl(settings.resource_url),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=[MCP_SCOPE],
            default_scopes=[MCP_SCOPE],
        ),
        revocation_options=RevocationOptions(enabled=True),
        # No required scopes: access is all-or-nothing for a single owner, and
        # demanding a scope only adds a way for the handshake to fail.
        required_scopes=None,
    )


_settings = get_settings()
auth_provider: SingleUserOAuthProvider | None = _build_auth_provider(_settings)

mcp: FastMCP = FastMCP(
    name="openai-image-generator",
    instructions=(
        "Provides image generation via OpenAI's gpt-image-2 model. "
        "Call `generate_image` with a descriptive prompt to receive a rendered image."
    ),
    stateless_http=True,
    transport_security=_build_transport_security(_settings),
    auth_server_provider=auth_provider,
    token_verifier=ProviderTokenVerifier(auth_provider) if auth_provider else None,
    auth=_build_auth_settings(_settings),
)

_image_client: OpenAIImageClient | None = None
_image_store: ImageStore | None = None


def set_image_client(client: OpenAIImageClient | None) -> None:
    """Install the shared OpenAI image client used by the tool.

    Args:
        client: The client to use, or ``None`` to reset the shared instance.
    """
    global _image_client
    _image_client = client


def get_image_client() -> OpenAIImageClient:
    """Return the shared OpenAI image client, creating it lazily if needed.

    Returns:
        The process-wide :class:`OpenAIImageClient`. If one was not installed via
        :func:`set_image_client` (for example in a test), it is built from the
        current settings on first use.
    """
    global _image_client
    if _image_client is None:
        settings = get_settings()
        _image_client = OpenAIImageClient(
            settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.image_model,
            timeout_seconds=settings.request_timeout_seconds,
        )
    return _image_client


async def close_image_client() -> None:
    """Close and clear the shared OpenAI image client, if one exists."""
    global _image_client
    if _image_client is not None:
        await _image_client.aclose()
        _image_client = None


def set_image_store(store: ImageStore | None) -> None:
    """Install the shared image store used by the tool and the image route.

    Args:
        store: The store to use, or ``None`` to reset the shared instance.
    """
    global _image_store
    _image_store = store


def get_image_store() -> ImageStore:
    """Return the shared image store, creating it lazily from settings if needed.

    Returns:
        The process-wide :class:`~claude_mcp.storage.ImageStore`.
    """
    global _image_store
    if _image_store is None:
        settings = get_settings()
        directory = (
            Path(settings.image_storage_dir)
            if settings.image_storage_dir
            else default_storage_dir()
        )
        _image_store = ImageStore(directory, settings.image_retention_minutes * 60.0)
    return _image_store


def _resolve_base_url(settings: Settings, request: Request | None) -> str:
    """Determine the absolute base URL used to build image links.

    Preference order: an explicit ``PUBLIC_BASE_URL`` setting, then the incoming
    request (honouring ``X-Forwarded-Proto`` so links are ``https`` behind the
    Fly.io proxy), then a localhost fallback.

    Args:
        settings: Application settings (for the explicit override and port).
        request: The incoming Starlette request, if available.

    Returns:
        A base URL without a trailing slash, e.g. ``https://myapp.fly.dev``.
    """
    if settings.public_base_url:
        return settings.public_base_url.rstrip("/")
    if request is not None:
        host = request.headers.get("host")
        if host:
            proto = request.headers.get("x-forwarded-proto") or request.url.scheme
            return f"{proto}://{host}"
    return f"http://localhost:{settings.port}"


def _markdown_alt(text: str) -> str:
    """Sanitise text for use inside a Markdown image alt (``![alt](url)``).

    Args:
        text: Raw prompt or revised-prompt text.

    Returns:
        A single-line, bracket-free string capped at 200 characters.
    """
    cleaned = " ".join(text.split()).replace("[", "").replace("]", "")
    return cleaned[:200] or "generated image"


@mcp.tool(
    title="Generate image",
    description="Generate an image from a text prompt using OpenAI's gpt-image-2 model.",
    structured_output=False,
)
async def generate_image(
    ctx: Context[Any, Any, Any],
    prompt: Annotated[
        str,
        Field(description="A detailed text description of the image to generate."),
    ],
    size: Annotated[
        str,
        Field(
            description=("Output size such as '1024x1024', '1536x1024', '1024x1536', or 'auto'."),
        ),
    ] = "auto",
    quality: Annotated[
        str,
        Field(description="Rendering quality: 'low', 'medium', 'high', or 'auto'."),
    ] = "auto",
) -> str:
    """Generate an image with ``gpt-image-2`` and return it as a Markdown link.

    The image is saved to the server's local storage and served over HTTP; the
    returned Markdown references it by URL so the response stays well under the
    MCP tool-response size limit.

    Args:
        ctx: The MCP request context (injected), used to derive the server URL.
        prompt: A detailed description of the desired image.
        size: Requested output resolution, or ``"auto"`` to let the model choose.
        quality: Rendering quality tier, or ``"auto"``.

    Returns:
        Markdown containing the rendered image and a direct link to it.

    Raises:
        OpenAIImageError: If the OpenAI Image API request fails or returns no image.
    """
    result = await get_image_client().generate_image(prompt=prompt, size=size, quality=quality)
    name = await get_image_store().save(result.data, result.image_format)

    try:
        raw_request = ctx.request_context.request
    except (ValueError, AttributeError):
        # No HTTP request context (e.g. a direct in-process invocation).
        raw_request = None
    request = raw_request if isinstance(raw_request, Request) else None
    base_url = _resolve_base_url(get_settings(), request)
    url = f"{base_url}/images/{name}"

    alt = _markdown_alt(result.revised_prompt or prompt)
    return f"![{alt}]({url})\n\n[Open image]({url})"
