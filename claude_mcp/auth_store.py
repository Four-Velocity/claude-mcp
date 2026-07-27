"""Persistent storage for OAuth clients, authorization codes, and tokens.

State is kept in a single JSON file so that registered clients and issued tokens
survive a process restart or redeploy (otherwise every deploy would force a
re-authorization in Claude).

Credentials (authorization codes, access tokens, refresh tokens, and pending
login request ids) are **never** stored in plaintext: only their SHA-256 digests
are persisted, and lookups hash the incoming value. A leaked state file
therefore does not yield usable credentials.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from mcp.shared.auth import OAuthClientInformationFull

# Top-level keys in the JSON state file.
_CLIENTS = "clients"
_PENDING = "pending"
_CODES = "codes"
_ACCESS = "access_tokens"
_REFRESH = "refresh_tokens"

_SECTIONS = (_CLIENTS, _PENDING, _CODES, _ACCESS, _REFRESH)


def hash_secret(value: str) -> str:
    """Return the hex SHA-256 digest of a credential.

    Used to key stored records so raw credentials never touch disk.

    Args:
        value: The raw credential (code, token, or request id).

    Returns:
        The hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(value.encode()).hexdigest()


class AuthStore:
    """A JSON-file-backed store for OAuth state, safe for concurrent async use."""

    def __init__(self, path: Path) -> None:
        """Load existing state from ``path``, creating its parent directory.

        Args:
            path: Location of the JSON state file.
        """
        self._path = path
        self._lock = asyncio.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, Any]] = self._read()

    def _read(self) -> dict[str, dict[str, Any]]:
        """Read and normalise the state file, tolerating absence or corruption.

        Returns:
            The parsed state with all expected sections present.
        """
        data: dict[str, dict[str, Any]] = {}
        if self._path.is_file():
            try:
                loaded = json.loads(self._path.read_text())
                if isinstance(loaded, dict):
                    data = {k: v for k, v in loaded.items() if isinstance(v, dict)}
            except (OSError, ValueError):
                # A corrupt or unreadable state file must not prevent startup;
                # clients simply re-register and re-authorize.
                data = {}
        for section in _SECTIONS:
            data.setdefault(section, {})
        return data

    def _write(self) -> None:
        """Atomically persist current state with owner-only permissions."""
        tmp = self._path.with_suffix(f"{self._path.suffix}.tmp")
        tmp.write_text(json.dumps(self._data))
        os.chmod(tmp, 0o600)
        os.replace(tmp, self._path)

    def _prune(self) -> None:
        """Drop expired short-lived records (pending logins, codes, tokens)."""
        now = time.time()
        for section in (_PENDING, _CODES, _ACCESS, _REFRESH):
            live = {}
            for key, record in self._data[section].items():
                expires_at = record.get("expires_at")
                if expires_at is None or float(expires_at) > now:
                    live[key] = record
            self._data[section] = live

    async def _commit(self) -> None:
        """Prune expired records and write state to disk."""
        self._prune()
        await asyncio.to_thread(self._write)

    # ---- clients (dynamic client registration) ----

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        """Return a registered client by id.

        Args:
            client_id: The OAuth client identifier.

        Returns:
            The stored client information, or ``None`` if unknown or unparsable.
        """
        async with self._lock:
            record = self._data[_CLIENTS].get(client_id)
        if record is None:
            return None
        try:
            return OAuthClientInformationFull.model_validate(record)
        except ValueError:
            return None

    async def put_client(self, client: OAuthClientInformationFull) -> None:
        """Persist a newly registered client.

        Args:
            client: The client information to store; its ``client_id`` is the key.
        """
        async with self._lock:
            self._data[_CLIENTS][str(client.client_id)] = client.model_dump(mode="json")
            await self._commit()

    # ---- generic record helpers ----

    async def put(self, section: str, key_hash: str, record: dict[str, Any]) -> None:
        """Store a record in a section under a pre-hashed key.

        Args:
            section: One of the module-level section names.
            key_hash: SHA-256 digest of the raw credential.
            record: JSON-serialisable record body.
        """
        async with self._lock:
            self._data[section][key_hash] = record
            await self._commit()

    async def get(self, section: str, key_hash: str) -> dict[str, Any] | None:
        """Return a live (non-expired) record, or ``None``.

        Args:
            section: One of the module-level section names.
            key_hash: SHA-256 digest of the raw credential.

        Returns:
            The record if present and unexpired, otherwise ``None``.
        """
        async with self._lock:
            record = self._data[section].get(key_hash)
            if record is None:
                return None
            expires_at = record.get("expires_at")
            if expires_at is not None and float(expires_at) <= time.time():
                del self._data[section][key_hash]
                await self._commit()
                return None
            return dict(record)

    async def delete(self, section: str, key_hash: str) -> None:
        """Delete a record if present.

        Args:
            section: One of the module-level section names.
            key_hash: SHA-256 digest of the raw credential.
        """
        async with self._lock:
            if self._data[section].pop(key_hash, None) is not None:
                await self._commit()

    # ---- named section accessors (readability at call sites) ----

    async def put_pending(self, request_id: str, record: dict[str, Any]) -> None:
        """Store a pending authorization awaiting the password login."""
        await self.put(_PENDING, hash_secret(request_id), record)

    async def get_pending(self, request_id: str) -> dict[str, Any] | None:
        """Return a pending authorization by its raw request id."""
        return await self.get(_PENDING, hash_secret(request_id))

    async def delete_pending(self, request_id: str) -> None:
        """Delete a pending authorization by its raw request id."""
        await self.delete(_PENDING, hash_secret(request_id))

    async def put_code(self, code: str, record: dict[str, Any]) -> None:
        """Store an issued authorization code."""
        await self.put(_CODES, hash_secret(code), record)

    async def get_code(self, code: str) -> dict[str, Any] | None:
        """Return an authorization code record by its raw code."""
        return await self.get(_CODES, hash_secret(code))

    async def delete_code(self, code: str) -> None:
        """Delete an authorization code (codes are single-use)."""
        await self.delete(_CODES, hash_secret(code))

    async def put_access_token(self, token: str, record: dict[str, Any]) -> None:
        """Store an issued access token."""
        await self.put(_ACCESS, hash_secret(token), record)

    async def get_access_token(self, token: str) -> dict[str, Any] | None:
        """Return an access token record by its raw token."""
        return await self.get(_ACCESS, hash_secret(token))

    async def delete_access_token(self, token: str) -> None:
        """Delete an access token."""
        await self.delete(_ACCESS, hash_secret(token))

    async def put_refresh_token(self, token: str, record: dict[str, Any]) -> None:
        """Store an issued refresh token."""
        await self.put(_REFRESH, hash_secret(token), record)

    async def get_refresh_token(self, token: str) -> dict[str, Any] | None:
        """Return a refresh token record by its raw token."""
        return await self.get(_REFRESH, hash_secret(token))

    async def delete_refresh_token(self, token: str) -> None:
        """Delete a refresh token."""
        await self.delete(_REFRESH, hash_secret(token))
