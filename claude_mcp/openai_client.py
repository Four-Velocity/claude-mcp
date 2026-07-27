"""Async client for OpenAI's Image generation API.

This wraps a single :class:`httpx.AsyncClient` and exposes one coroutine that
calls ``POST /v1/images/generations`` (the Image API, not the Responses API) and
returns the decoded image bytes.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from types import TracebackType

import httpx

_IMAGES_ENDPOINT = "/v1/images/generations"


class OpenAIImageError(RuntimeError):
    """Raised when the OpenAI Image API cannot fulfil a generation request."""


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """A single generated image returned by the OpenAI Image API.

    Attributes:
        data: The raw (already base64-decoded) image bytes.
        image_format: The image format, e.g. ``"png"``, matching ``output_format``.
        revised_prompt: The prompt as revised by the model, when provided.
    """

    data: bytes
    image_format: str
    revised_prompt: str | None


class OpenAIImageClient:
    """Thin async wrapper around the OpenAI Image generation endpoint."""

    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = "https://api.openai.com",
        model: str = "gpt-image-2",
        timeout_seconds: float = 180.0,
    ) -> None:
        """Initialise the client and its underlying async HTTP connection pool.

        Args:
            api_key: OpenAI API key. May be ``None``; a generation call then
                fails fast with :class:`OpenAIImageError`.
            base_url: Base URL of the OpenAI REST API.
            model: Image model to request (defaults to ``gpt-image-2``).
            timeout_seconds: Total request timeout, in seconds. Image generation
                can take well over a minute, so this should be generous.
        """
        self._api_key = api_key
        self._model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        )

    async def generate_image(
        self,
        *,
        prompt: str,
        size: str = "auto",
        quality: str = "auto",
        output_format: str = "png",
    ) -> GeneratedImage:
        """Generate a single image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            size: Requested image size (e.g. ``"1024x1024"``) or ``"auto"``.
            quality: Rendering quality: ``"low"``, ``"medium"``, ``"high"`` or ``"auto"``.
            output_format: Encoding of the returned image: ``"png"``, ``"jpeg"`` or ``"webp"``.

        Returns:
            The decoded image together with its format and any revised prompt.

        Raises:
            OpenAIImageError: If no API key is configured, the API returns an
                error status, or the response body is missing image data.
        """
        if not self._api_key:
            raise OpenAIImageError(
                "OPENAI_API_KEY is not configured; set it as a credential/secret."
            )

        payload = {
            "model": self._model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": quality,
            "output_format": output_format,
        }

        try:
            response = await self._client.post(_IMAGES_ENDPOINT, json=payload)
        except httpx.HTTPError as exc:  # network/timeout level failures
            raise OpenAIImageError(f"Request to OpenAI failed: {exc}") from exc

        if response.status_code >= 400:
            raise OpenAIImageError(_format_api_error(response))

        body = response.json()
        items = body.get("data") or []
        if not items or "b64_json" not in items[0]:
            raise OpenAIImageError("OpenAI response did not contain image data.")

        image_bytes = base64.b64decode(items[0]["b64_json"])
        return GeneratedImage(
            data=image_bytes,
            image_format=output_format,
            revised_prompt=items[0].get("revised_prompt"),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> OpenAIImageClient:
        """Enter an async context manager, returning ``self``."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client when leaving an async context manager."""
        await self.aclose()


def _format_api_error(response: httpx.Response) -> str:
    """Build a readable error message from an OpenAI error response.

    Args:
        response: The failed HTTP response.

    Returns:
        A single-line description including the status code and, when present,
        the structured error message from the response body.
    """
    detail = f"HTTP {response.status_code}"
    try:
        data = response.json()
    except ValueError:
        return f"OpenAI API error ({detail}): {response.text[:500]}"

    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return f"OpenAI API error ({detail}): {error['message']}"
    return f"OpenAI API error ({detail}): {data}"
