"""Console entrypoint: run the FastAPI/MCP server with uvicorn.

Usage:
    python -m claude_mcp
"""

from __future__ import annotations

import uvicorn

from claude_mcp.config import get_settings


def main() -> None:
    """Start the uvicorn server bound to the configured host and port."""
    settings = get_settings()
    uvicorn.run(
        "claude_mcp.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
