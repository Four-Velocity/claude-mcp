"""Claude MCP server package.

Exposes a FastAPI application that serves a Model Context Protocol (MCP) server
over streamable HTTP. The server provides a single tool that generates images
with OpenAI's ``gpt-image-2`` model and returns them to Claude as image content.
"""

from claude_mcp.version import read_version

__all__ = ["read_version"]
