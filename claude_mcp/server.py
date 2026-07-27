"""MCP server definition and the ``generate_image`` tool.

The server is created with :class:`FastMCP` and served over streamable HTTP so it
can be added to Claude as a custom connector. It intentionally exposes a single
tool for now: generating an image with OpenAI's ``gpt-image-2`` model.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

from claude_mcp.config import Settings, get_settings
from claude_mcp.openai_client import OpenAIImageClient


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


mcp: FastMCP = FastMCP(
    name="openai-image-generator",
    instructions=(
        "Provides image generation via OpenAI's gpt-image-2 model. "
        "Call `generate_image` with a descriptive prompt to receive a rendered image."
    ),
    stateless_http=True,
    transport_security=_build_transport_security(get_settings()),
)

_image_client: OpenAIImageClient | None = None


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


@mcp.tool(
    title="Generate image",
    description="Generate an image from a text prompt using OpenAI's gpt-image-2 model.",
    structured_output=False,
)
async def generate_image(
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
) -> Image:
    """Generate an image with ``gpt-image-2`` and return it to Claude.

    Args:
        prompt: A detailed description of the desired image.
        size: Requested output resolution, or ``"auto"`` to let the model choose.
        quality: Rendering quality tier, or ``"auto"``.

    Returns:
        An :class:`~mcp.server.fastmcp.Image` wrapping the PNG bytes, which the MCP
        runtime converts into image content for Claude to display.

    Raises:
        OpenAIImageError: If the OpenAI Image API request fails or returns no image.
    """
    client = get_image_client()
    result = await client.generate_image(prompt=prompt, size=size, quality=quality)
    return Image(data=result.data, format=result.image_format)
