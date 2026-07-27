"""Local filesystem storage for generated images.

Images are written to a temporary directory and served back over HTTP so the
MCP tool response only needs to carry a small URL rather than multi-megabyte
base64 data (which would exceed the ~1 MB MCP tool-response limit in the Claude
desktop and mobile apps).
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import time
import uuid
from pathlib import Path

_EXTENSION_BY_FORMAT = {
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "webp": "webp",
}

_CONTENT_TYPE_BY_EXTENSION = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}

# Stored file names are ``<32 hex chars>.<ext>`` — used to reject path traversal.
_SAFE_NAME = re.compile(r"^[0-9a-f]{32}\.(png|jpg|webp)$")


def default_storage_dir() -> Path:
    """Return the default image storage directory inside the system temp dir."""
    return Path(tempfile.gettempdir()) / "claude-mcp-images"


def default_auth_storage_path() -> Path:
    """Return the default OAuth state file path inside the system temp dir."""
    return Path(tempfile.gettempdir()) / "claude-mcp-oauth.json"


class ImageStore:
    """Persist generated images to a directory and prune expired ones."""

    def __init__(self, directory: Path, retention_seconds: float) -> None:
        """Create the store, ensuring its backing directory exists.

        Args:
            directory: Directory in which image files are written.
            retention_seconds: Age after which files are eligible for deletion.
        """
        self._dir = directory
        self._retention_seconds = retention_seconds
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        """Return the directory backing this store."""
        return self._dir

    async def save(self, data: bytes, image_format: str) -> str:
        """Write image bytes to the store and return the generated file name.

        Args:
            data: Raw image bytes.
            image_format: Image format such as ``"png"`` or ``"jpeg"``.

        Returns:
            The generated file name (e.g. ``"<hex>.png"``), safe to embed in a URL.
        """
        extension = _EXTENSION_BY_FORMAT.get(image_format.lower(), "png")
        name = f"{uuid.uuid4().hex}.{extension}"
        path = self._dir / name
        await asyncio.to_thread(path.write_bytes, data)
        await asyncio.to_thread(self._prune)
        return name

    def resolve(self, name: str) -> Path | None:
        """Return the on-disk path for a stored file name, if it is valid and present.

        Args:
            name: The requested file name (from a URL path segment).

        Returns:
            The resolved path if ``name`` is a safe, existing file within the
            store, otherwise ``None``.
        """
        if not _SAFE_NAME.match(name):
            return None
        path = (self._dir / name).resolve()
        if path.parent != self._dir.resolve() or not path.is_file():
            return None
        return path

    def _prune(self) -> None:
        """Delete files older than the configured retention period."""
        if self._retention_seconds <= 0:
            return
        cutoff = time.time() - self._retention_seconds
        for entry in self._dir.iterdir():
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    entry.unlink(missing_ok=True)
            except OSError:
                # Best-effort cleanup; ignore files that vanish or can't be stat'd.
                continue


def content_type_for(path: Path) -> str:
    """Return the HTTP content type for a stored image path.

    Args:
        path: Path to a stored image file.

    Returns:
        The matching ``image/*`` MIME type, defaulting to ``application/octet-stream``.
    """
    return _CONTENT_TYPE_BY_EXTENSION.get(path.suffix.lstrip("."), "application/octet-stream")
