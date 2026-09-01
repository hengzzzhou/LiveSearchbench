"""Credential and endpoint resolution.

Precedence for every setting is: explicit argument > environment variable >
``.env`` file in the repository root > built-in default. Missing required
values raise :class:`MissingCredential` with a message that names the variable
and the file to put it in, instead of failing later with an opaque HTTP error.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default Serper endpoint. The previous release shipped an empty string here,
#: which made ``scripts/eval/RAG.py`` unusable without reading Serper's docs.
DEFAULT_SERPER_ENDPOINT = "https://google.serper.dev/search"

#: Default Wikidata Query Service endpoint.
DEFAULT_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

#: Default Wikidata Action API endpoint.
DEFAULT_WIKIDATA_API = "https://www.wikidata.org/w/api.php"

#: Contact string embedded in the User-Agent. Wikimedia's User-Agent policy
#: asks for a way to reach the operator; override via ``LSB_CONTACT``.
DEFAULT_CONTACT = "https://github.com/hengzzzhou/LiveSearchbench; hengzzzhou@gmail.com"


class MissingCredential(RuntimeError):
    """Raised when a required credential cannot be resolved."""


_DOTENV_CACHE: Optional[dict] = None


def _load_dotenv() -> dict:
    """Parse ``.env`` in the repo root. Returns {} when absent.

    Deliberately tiny rather than depending on python-dotenv, so the library
    imports cleanly in a bare environment.
    """
    global _DOTENV_CACHE
    if _DOTENV_CACHE is not None:
        return _DOTENV_CACHE

    values: dict = {}
    dotenv = REPO_ROOT / ".env"
    if dotenv.is_file():
        for raw in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            values[key.strip()] = val
    _DOTENV_CACHE = values
    return values


def get(name: str, *, override: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    """Resolve a setting. Returns ``default`` when unset."""
    if override:
        return override
    env = os.environ.get(name)
    if env:
        return env
    dotenv = _load_dotenv().get(name)
    if dotenv:
        return dotenv
    return default


def require(name: str, *, override: Optional[str] = None, purpose: str = "") -> str:
    """Resolve a setting, raising :class:`MissingCredential` when absent."""
    value = get(name, override=override)
    if value:
        return value
    hint = f" ({purpose})" if purpose else ""
    raise MissingCredential(
        f"{name} is not set{hint}.\n"
        f"  Set it one of three ways:\n"
        f"    export {name}=...\n"
        f"    add '{name}=...' to {REPO_ROOT / '.env'}\n"
        f"    pass the corresponding --... flag on the command line\n"
        f"  See .env.example for the full list."
    )


def user_agent(component: str = "LiveSearchBench") -> str:
    """Build a Wikimedia-policy-compliant User-Agent string."""
    contact = get("LSB_CONTACT", default=DEFAULT_CONTACT)
    import requests  # local import so config stays importable without requests

    return f"{component}/1.1 ({contact}) python-requests/{requests.__version__}"


def openai_credentials(*, base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Return ``(base_url, api_key)`` for the OpenAI-compatible endpoint."""
    return (
        get("OPENAI_BASE_URL", override=base_url, default="https://api.openai.com/v1"),
        require("OPENAI_API_KEY", override=api_key, purpose="model inference"),
    )


def serper_credentials(*, endpoint: Optional[str] = None, api_key: Optional[str] = None):
    """Return ``(endpoint, api_key)`` for Serper web search."""
    return (
        get("SERPER_ENDPOINT", override=endpoint, default=DEFAULT_SERPER_ENDPOINT),
        require("SERPER_API_KEY", override=api_key, purpose="web search in RAG.py"),
    )
