"""Read the project version from ``pyproject.toml`` at runtime."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path


def _find_pyproject() -> Path:
    """Locate the ``pyproject.toml`` for this project.

    Walks upward from this module's location until a ``pyproject.toml`` is found.
    This keeps version resolution working both when running from a source
    checkout and when the source tree is copied into a container image.

    Returns:
        The path to the discovered ``pyproject.toml``.

    Raises:
        FileNotFoundError: If no ``pyproject.toml`` can be found.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not locate pyproject.toml to read the project version.")


@lru_cache(maxsize=1)
def read_version() -> str:
    """Return the ``[project].version`` value from ``pyproject.toml``.

    The result is cached so the file is read only once per process, which makes
    this cheap to call on every request while still reflecting the version that
    was written into the TOML file at startup.

    Returns:
        The project version string (for example ``"0.1.0"``).

    Raises:
        FileNotFoundError: If ``pyproject.toml`` cannot be located.
        KeyError: If the ``[project].version`` key is missing.
    """
    pyproject = _find_pyproject()
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    return str(data["project"]["version"])
