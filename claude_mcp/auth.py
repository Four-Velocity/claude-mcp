"""Single-user OAuth 2.1 authorization server for the MCP endpoint.

The MCP Python SDK implements the OAuth *protocol* (metadata discovery, PKCE
verification, the ``/authorize`` → ``/token`` dance, dynamic client registration,
and the ``401`` + ``WWW-Authenticate`` challenge). This module supplies the parts
the SDK delegates to the application:

* **credential storage** — clients, codes, and tokens (see :mod:`claude_mcp.auth_store`);
* **the login gate** — a password page that authenticates the single human owner
  before an authorization code is issued.

Design notes:

* Only one user exists, so authorization is all-or-nothing: anyone who proves
  knowledge of ``MCP_AUTH_PASSWORD`` is the owner. Scopes are cosmetic.
* Dynamic Client Registration is enabled because Claude supports it out of the
  box, which means no client id/secret needs to be configured by hand. Being
  able to *register* grants nothing on its own — the password gate still applies.
* Access tokens are opaque random strings validated against our own store, so
  tokens minted for another audience can never be replayed here. Audience
  (RFC 8707) is additionally checked when the client supplies a ``resource``.
"""

from __future__ import annotations

import html
import secrets
import time
from dataclasses import dataclass, field
from urllib.parse import quote

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl

from claude_mcp.auth_store import AuthStore

#: Nominal scope granted to every token. Access is all-or-nothing for one user.
MCP_SCOPE = "mcp"

#: Entropy for generated codes and tokens, in bytes (256 bits).
_TOKEN_BYTES = 32


class LoginError(Exception):
    """Raised when a login attempt cannot be completed."""


class LoginThrottledError(LoginError):
    """Raised when too many failed login attempts came from one address."""


def _canonical_resource(value: str) -> str:
    """Normalise a resource indicator for comparison.

    Args:
        value: A resource URI.

    Returns:
        The URI lower-cased without a trailing slash.
    """
    return value.rstrip("/").lower()


@dataclass
class LoginThrottle:
    """In-memory failed-login throttle, keyed by client address.

    The login page is publicly reachable, so this raises the cost of guessing the
    password. State is per-process and resets on restart, which is acceptable for
    a single-user deployment.

    Attributes:
        max_failures: Failures allowed within ``window_seconds`` before lockout.
        window_seconds: Sliding window over which failures are counted.
    """

    max_failures: int = 5
    window_seconds: float = 300.0
    _failures: dict[str, list[float]] = field(default_factory=dict)

    def _recent(self, key: str, now: float) -> list[float]:
        """Return failure timestamps for ``key`` within the current window."""
        cutoff = now - self.window_seconds
        return [ts for ts in self._failures.get(key, []) if ts > cutoff]

    def check(self, key: str) -> None:
        """Raise if ``key`` is currently locked out.

        Args:
            key: Client identifier (typically the remote IP address).

        Raises:
            LoginThrottledError: If the failure count within the window is exhausted.
        """
        recent = self._recent(key, time.time())
        self._failures[key] = recent
        if len(recent) >= self.max_failures:
            raise LoginThrottledError("Too many failed attempts. Try again later.")

    def record_failure(self, key: str) -> None:
        """Record a failed attempt for ``key``."""
        now = time.time()
        self._failures[key] = [*self._recent(key, now), now]

    def reset(self, key: str) -> None:
        """Clear recorded failures for ``key`` after a success."""
        self._failures.pop(key, None)


class SingleUserOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """OAuth 2.1 authorization server gated by a single shared password."""

    def __init__(
        self,
        store: AuthStore,
        *,
        password: str,
        base_url: str,
        resource_url: str,
        access_token_ttl: float,
        refresh_token_ttl: float,
        code_ttl: float = 120.0,
        login_ttl: float = 600.0,
    ) -> None:
        """Configure the provider.

        Args:
            store: Backing store for clients, codes, and tokens.
            password: The owner's password; compared in constant time.
            base_url: Public base URL of this server (used to build the login URL).
            resource_url: Canonical resource identifier of the MCP endpoint, used
                for RFC 8707 audience validation.
            access_token_ttl: Access token lifetime in seconds.
            refresh_token_ttl: Refresh token lifetime in seconds.
            code_ttl: Authorization code lifetime in seconds.
            login_ttl: How long a pending login may sit unconfirmed, in seconds.
        """
        self._store = store
        self._password = password
        self._base_url = base_url.rstrip("/")
        self._resource_url = _canonical_resource(resource_url)
        self._access_token_ttl = access_token_ttl
        self._refresh_token_ttl = refresh_token_ttl
        self._code_ttl = code_ttl
        self._login_ttl = login_ttl
        self._throttle = LoginThrottle()

    # ---- dynamic client registration ----

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        """Return a previously registered client.

        Args:
            client_id: The client identifier presented by the caller.

        Returns:
            The client information, or ``None`` if it is not registered.
        """
        return await self._store.get_client(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        """Persist a dynamically registered client.

        Registration alone grants no access; the password gate still applies.

        Args:
            client_info: Client metadata assembled by the SDK's registration handler.
        """
        await self._store.put_client(client_info)

    # ---- authorization ----

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Begin authorization by redirecting the browser to the login page.

        The request is parked server-side under an unguessable id; the login form
        completes it once the password is verified.

        Args:
            client: The client requesting authorization.
            params: Parsed authorization request parameters.

        Returns:
            The URL of the login page to redirect the user agent to.

        Raises:
            AuthorizeError: If the request carries a resource indicator that does
                not identify this server.
        """
        if params.resource and _canonical_resource(params.resource) != self._resource_url:
            raise AuthorizeError(
                error="invalid_request",
                error_description="resource indicator does not match this server",
            )

        request_id = secrets.token_urlsafe(_TOKEN_BYTES)
        await self._store.put_pending(
            request_id,
            {
                "client_id": str(client.client_id),
                "redirect_uri": str(params.redirect_uri),
                "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
                "code_challenge": params.code_challenge,
                "state": params.state,
                "scopes": params.scopes or [MCP_SCOPE],
                "resource": params.resource,
                "expires_at": time.time() + self._login_ttl,
            },
        )
        return f"{self._base_url}/login?rid={quote(request_id, safe='')}"

    async def complete_login(self, request_id: str, password: str, client_key: str) -> str:
        """Verify the password and finish a parked authorization request.

        Args:
            request_id: The pending request id from the login form.
            password: The password submitted by the user.
            client_key: Caller identity for throttling (typically the remote IP).

        Returns:
            The redirect URL carrying the authorization ``code`` and ``state``.

        Raises:
            LoginThrottledError: If too many recent attempts failed.
            LoginError: If the password is wrong or the request expired.
        """
        self._throttle.check(client_key)

        # Look the request up before checking the password so that an expired
        # request reports accurately, then compare in constant time.
        pending = await self._store.get_pending(request_id)
        if not secrets.compare_digest(password, self._password):
            self._throttle.record_failure(client_key)
            raise LoginError("Incorrect password.")
        if pending is None:
            raise LoginError("This login request expired. Start the connection again.")

        self._throttle.reset(client_key)
        await self._store.delete_pending(request_id)

        code = secrets.token_urlsafe(_TOKEN_BYTES)
        await self._store.put_code(
            code,
            {
                "client_id": pending["client_id"],
                "redirect_uri": pending["redirect_uri"],
                "redirect_uri_provided_explicitly": pending["redirect_uri_provided_explicitly"],
                "code_challenge": pending["code_challenge"],
                "scopes": pending["scopes"],
                "resource": pending["resource"],
                "expires_at": time.time() + self._code_ttl,
            },
        )
        return construct_redirect_uri(pending["redirect_uri"], code=code, state=pending["state"])

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        """Load an authorization code issued to ``client``.

        Args:
            client: The client redeeming the code.
            authorization_code: The raw code value.

        Returns:
            The code object, or ``None`` if unknown, expired, or issued to a
            different client.
        """
        record = await self._store.get_code(authorization_code)
        if record is None or record["client_id"] != str(client.client_id):
            return None
        return AuthorizationCode(
            code=authorization_code,
            scopes=record["scopes"],
            expires_at=float(record["expires_at"]),
            client_id=record["client_id"],
            code_challenge=record["code_challenge"],
            redirect_uri=AnyUrl(record["redirect_uri"]),
            redirect_uri_provided_explicitly=record["redirect_uri_provided_explicitly"],
            resource=record["resource"],
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        """Exchange a validated authorization code for tokens.

        The SDK has already verified PKCE and the redirect URI at this point.

        Args:
            client: The client redeeming the code.
            authorization_code: The code being redeemed.

        Returns:
            A freshly issued access token and refresh token.

        Raises:
            TokenError: If the code was already redeemed.
        """
        if await self._store.get_code(authorization_code.code) is None:
            raise TokenError(error="invalid_grant", error_description="code is not valid")
        # Authorization codes are single-use.
        await self._store.delete_code(authorization_code.code)
        return await self._issue_tokens(
            client_id=str(client.client_id),
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
        )

    # ---- refresh ----

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        """Load a refresh token belonging to ``client``.

        Args:
            client: The client presenting the refresh token.
            refresh_token: The raw refresh token value.

        Returns:
            The refresh token object, or ``None`` if unknown, expired, or issued
            to a different client.
        """
        record = await self._store.get_refresh_token(refresh_token)
        if record is None or record["client_id"] != str(client.client_id):
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=record["client_id"],
            scopes=record["scopes"],
            expires_at=int(float(record["expires_at"])),
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """Rotate a refresh token and issue a new access token.

        Both tokens are rotated, as recommended for public clients.

        Args:
            client: The client presenting the refresh token.
            refresh_token: The refresh token being redeemed.
            scopes: Scopes requested for the new access token.

        Returns:
            A new access token and a new refresh token.

        Raises:
            TokenError: If the refresh token is no longer valid, or the request
                asks for scopes beyond those originally granted.
        """
        record = await self._store.get_refresh_token(refresh_token.token)
        if record is None:
            raise TokenError(error="invalid_grant", error_description="refresh token is not valid")
        granted = refresh_token.scopes
        requested = scopes or granted
        if not set(requested).issubset(set(granted)):
            raise TokenError(
                error="invalid_scope",
                error_description="requested scopes exceed the granted scopes",
            )
        await self._store.delete_refresh_token(refresh_token.token)
        return await self._issue_tokens(
            client_id=str(client.client_id),
            scopes=requested,
            resource=record.get("resource"),
        )

    async def _issue_tokens(
        self, *, client_id: str, scopes: list[str], resource: str | None
    ) -> OAuthToken:
        """Mint and persist a new access/refresh token pair.

        Args:
            client_id: Client the tokens are bound to.
            scopes: Scopes to grant.
            resource: Audience recorded for later validation, if any.

        Returns:
            The issued OAuth token response.
        """
        access_token = secrets.token_urlsafe(_TOKEN_BYTES)
        refresh_token = secrets.token_urlsafe(_TOKEN_BYTES)
        now = time.time()
        await self._store.put_access_token(
            access_token,
            {
                "client_id": client_id,
                "scopes": scopes,
                "resource": resource,
                "expires_at": now + self._access_token_ttl,
            },
        )
        await self._store.put_refresh_token(
            refresh_token,
            {
                "client_id": client_id,
                "scopes": scopes,
                "resource": resource,
                "expires_at": now + self._refresh_token_ttl,
            },
        )
        return OAuthToken(
            access_token=access_token,
            expires_in=int(self._access_token_ttl),
            scope=" ".join(scopes),
            refresh_token=refresh_token,
        )

    # ---- verification / revocation ----

    async def load_access_token(self, token: str) -> AccessToken | None:
        """Validate a bearer token presented to the MCP endpoint.

        In addition to existence and expiry, the recorded audience is checked so a
        token minted for a different resource is rejected (RFC 8707). Tokens
        without a recorded audience are accepted: they are opaque values that only
        exist in this server's own store, so they cannot originate elsewhere.

        Args:
            token: The raw bearer token.

        Returns:
            The access token record, or ``None`` if the token is not valid here.
        """
        record = await self._store.get_access_token(token)
        if record is None:
            return None
        resource = record.get("resource")
        if resource and _canonical_resource(resource) != self._resource_url:
            return None
        return AccessToken(
            token=token,
            client_id=record["client_id"],
            scopes=record["scopes"],
            expires_at=int(float(record["expires_at"])),
            resource=resource,
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        """Revoke an access or refresh token.

        Args:
            token: The token object to revoke; the matching stored record is removed.
        """
        if isinstance(token, RefreshToken):
            await self._store.delete_refresh_token(token.token)
        else:
            await self._store.delete_access_token(token.token)


def render_login_page(request_id: str, error: str | None = None) -> str:
    """Render the password login page.

    The markup is fully self-contained (no external assets) and adapts to the
    viewer's light or dark colour scheme.

    Args:
        request_id: The pending authorization request id, echoed as a hidden field.
        error: An optional message to display above the form.

    Returns:
        A complete HTML document.
    """
    error_block = f'<p class="error" role="alert">{html.escape(error)}</p>' if error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Sign in</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; display: grid; place-items: center;
    font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: #f6f7f9; color: #14161a; padding: 1.5rem;
  }}
  .card {{
    width: 100%; max-width: 22rem; background: #fff; padding: 2rem;
    border-radius: 14px; border: 1px solid #e3e6ea;
    box-shadow: 0 1px 3px rgb(0 0 0 / 0.06), 0 8px 24px rgb(0 0 0 / 0.06);
  }}
  h1 {{ margin: 0 0 .35rem; font-size: 1.2rem; }}
  p.sub {{ margin: 0 0 1.5rem; color: #5d646e; font-size: .9rem; }}
  label {{ display: block; font-size: .85rem; font-weight: 600; margin-bottom: .4rem; }}
  input {{
    width: 100%; padding: .6rem .7rem; font-size: 1rem; color: inherit;
    background: #fff; border: 1px solid #ccd1d8; border-radius: 8px;
  }}
  input:focus-visible {{ outline: 2px solid #4c7ef3; outline-offset: 1px; border-color: #4c7ef3; }}
  button {{
    width: 100%; margin-top: 1.1rem; padding: .65rem; font-size: 1rem; font-weight: 600;
    color: #fff; background: #1a1d23; border: 0; border-radius: 8px; cursor: pointer;
  }}
  button:hover {{ background: #33383f; }}
  .error {{
    margin: 0 0 1rem; padding: .6rem .7rem; font-size: .875rem;
    color: #8c1c13; background: #fdecea; border: 1px solid #f5c2bd; border-radius: 8px;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #15171a; color: #e8eaed; }}
    .card {{ background: #1e2126; border-color: #2f343b;
             box-shadow: 0 1px 3px rgb(0 0 0 / 0.4), 0 8px 24px rgb(0 0 0 / 0.35); }}
    p.sub {{ color: #9aa1ab; }}
    input {{ background: #15181c; border-color: #3a4048; }}
    button {{ background: #e8eaed; color: #15171a; }}
    button:hover {{ background: #c9ccd1; }}
    .error {{ color: #f5b7b1; background: #3b1a17; border-color: #6b2a24; }}
  }}
</style>
</head>
<body>
  <main class="card">
    <h1>Sign in</h1>
    <p class="sub">Authorize Claude to use this image generation server.</p>
    {error_block}
    <form method="post" action="/login">
      <input type="hidden" name="rid" value="{html.escape(request_id, quote=True)}">
      <label for="password">Password</label>
      <input id="password" name="password" type="password" required autofocus
             autocomplete="current-password">
      <button type="submit">Sign in</button>
    </form>
  </main>
</body>
</html>
"""
