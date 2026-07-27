# claude-mcp

A small [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that
exposes a single tool to Claude: **generate an image with OpenAI's `gpt-image-2`
model** and return it inline. Built with FastAPI, fully async, and deployable to
[Fly.io](https://fly.io).

## What it does

- Serves an MCP server over **streamable HTTP** at `/mcp` (add it to Claude as a custom connector).
- Exposes one tool, `generate_image(prompt, size, quality)`, which calls the OpenAI
  **Image API** (`POST /v1/images/generations`) via an async `httpx` client.
- **Stays under the ~1 MB MCP tool-response limit** (enforced by the Claude desktop and
  mobile apps): instead of returning multi-megabyte base64 image bytes inline, it saves
  the image to a local temp directory, serves it at `GET /images/{id}`, and returns a
  small **Markdown image link**. Claude renders the Markdown and the image bytes load
  directly over HTTP — never through the MCP channel. The image URL is auto-derived from
  the request (honouring `X-Forwarded-Proto`/`Host`), so it works on localhost and Fly
  without configuration; override with `PUBLIC_BASE_URL` if needed.
- Adds a `/version` endpoint that reports the version read from `pyproject.toml` at startup.
- Adds a `/health` endpoint for platform health checks.

## Layout

| Path | Purpose |
| --- | --- |
| `claude_mcp/config.py` | Environment-based settings (`OPENAI_API_KEY`, host/port, allowed hosts). |
| `claude_mcp/version.py` | Reads `[project].version` from `pyproject.toml`. |
| `claude_mcp/openai_client.py` | Async OpenAI Image API client. |
| `claude_mcp/storage.py` | Local temp-dir image storage + pruning. |
| `claude_mcp/server.py` | `FastMCP` server + the `generate_image` tool. |
| `claude_mcp/app.py` | FastAPI app: mounts MCP at `/mcp`, serves `/version`, `/health`, `/images/{id}`. |
| `claude_mcp/__main__.py` | `python -m claude_mcp` entrypoint (uvicorn). |

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key with access to `gpt-image-2` (organization verification may be required).

## Local development

```bash
uv sync
cp .env.example .env    # then edit .env and set OPENAI_API_KEY
uv run python -m claude_mcp
```

The server listens on `http://0.0.0.0:8080` by default. Verify it:

```bash
curl http://127.0.0.1:8080/version
curl http://127.0.0.1:8080/health
```

The MCP endpoint is `http://127.0.0.1:8080/mcp`.

## Quality gates

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
```

## Deploy to Fly.io

1. Install [flyctl](https://fly.io/docs/flyctl/install/) and log in (`fly auth login`).
2. Edit `app = "claude-mcp"` in `fly.toml` to a unique name (or run `fly launch` to generate one).
3. Set the OpenAI key as a secret (this is the "credentials" store — it is injected as an env var):

   ```bash
   fly secrets set OPENAI_API_KEY=sk-your-key-here
   ```

4. Deploy:

   ```bash
   fly deploy
   ```

Your MCP URL will be `https://<your-app>.fly.dev/mcp`.

## Connect it to Claude

Add a **custom connector** in Claude pointing at `https://<your-app>.fly.dev/mcp`.
Claude will then offer the `generate_image` tool.

## Security notes

- **The OpenAI key stays server-side.** It is read from `OPENAI_API_KEY` (a Fly.io
  secret in production) and never exposed to Claude or clients.
- **The MCP endpoint is unauthenticated** in this minimal setup — anyone who knows the
  URL can spend your OpenAI credits. For anything beyond personal use, put it behind
  auth (e.g. a reverse proxy, Fly.io access controls, or MCP OAuth).
- **DNS-rebinding protection** is disabled by default so the server responds on the
  varying `*.fly.dev` host. To lock it to specific hosts, set `MCP_ALLOWED_HOSTS`
  (comma-separated), e.g. `MCP_ALLOWED_HOSTS=your-app.fly.dev`.
- **Served image URLs are unauthenticated** and live in the (ephemeral) temp dir until
  pruned (`IMAGE_RETENTION_MINUTES`, default 60). Anyone with a URL can fetch that image
  during its lifetime.
- **Multi-machine note:** images are stored on the machine that generated them. If you
  scale to multiple Fly machines, a URL served by another machine will 404 — keep a
  single machine, or back the store with a shared volume/object store.
