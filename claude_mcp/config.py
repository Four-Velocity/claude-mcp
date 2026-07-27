"""Application configuration loaded from the environment (credentials)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings sourced from environment variables and an optional ``.env``.

    On Fly.io the ``OPENAI_API_KEY`` credential is provided via ``fly secrets set``,
    which exposes it as an environment variable that this class reads.

    Attributes:
        openai_api_key: Secret API key used to authenticate against OpenAI. It is
            optional at load time so the HTTP service (including ``/version`` and
            ``/health``) can still boot when the credential is absent; the image
            tool itself fails with a clear error if the key is missing.
        openai_base_url: Base URL for the OpenAI REST API.
        image_model: The OpenAI image model to call.
        request_timeout_seconds: Total timeout for an image-generation request.
        host: Interface the HTTP server binds to.
        port: TCP port the HTTP server listens on.
        mcp_allowed_hosts: Comma-separated ``Host`` header values permitted by the
            MCP transport's DNS-rebinding protection. When empty (the default),
            protection is disabled so the server responds on any host (required
            for the varying ``*.fly.dev`` hostname). Set it (e.g.
            ``"myapp.fly.dev"``) to lock the server to specific hosts.
        mcp_allowed_origins: Comma-separated ``Origin`` header values permitted
            when DNS-rebinding protection is enabled.
        public_base_url: Absolute base URL (e.g. ``https://myapp.fly.dev``) used
            to build image links returned by the tool. When empty the URL is
            derived from the incoming request (honouring ``X-Forwarded-Proto`` /
            ``Host``), falling back to ``http://localhost:<port>``.
        image_storage_dir: Directory for generated image files. Defaults to a
            ``claude-mcp-images`` folder inside the system temp directory.
        image_retention_minutes: Age after which stored images are pruned. Set to
            ``0`` to disable pruning.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None)
    openai_base_url: str = Field(default="https://api.openai.com")
    image_model: str = Field(default="gpt-image-2")
    request_timeout_seconds: float = Field(default=180.0, gt=0)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    mcp_allowed_hosts: str = Field(default="")
    mcp_allowed_origins: str = Field(default="")
    public_base_url: str = Field(default="")
    image_storage_dir: str = Field(default="")
    image_retention_minutes: float = Field(default=60.0, ge=0)

    @property
    def allowed_hosts(self) -> list[str]:
        """Return ``mcp_allowed_hosts`` parsed into a list of trimmed values."""
        return [item.strip() for item in self.mcp_allowed_hosts.split(",") if item.strip()]

    @property
    def allowed_origins(self) -> list[str]:
        """Return ``mcp_allowed_origins`` parsed into a list of trimmed values."""
        return [item.strip() for item in self.mcp_allowed_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Caching ensures the environment is parsed once per process while still
    allowing tests to clear the cache via ``get_settings.cache_clear()``.

    Returns:
        The process-wide settings object.
    """
    return Settings()
