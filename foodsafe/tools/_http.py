"""Shared HTTP session for the public scientific APIs this project reads.

NCBI and EMBL-EBI both ask callers to identify themselves and to back off rather
than hammer the endpoints, so every request goes through here.
"""

from __future__ import annotations

import time

import requests

USER_AGENT = (
    "foodsafety-agents/0.1 (open-source food safety evidence tool; "
    "https://github.com/Laxluther/foodsafety-agents)"
)

DEFAULT_TIMEOUT = 30
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.5


class ApiError(RuntimeError):
    """A public API was reachable but did not return usable data."""


_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return _session


def get_json(url: str, params: dict | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict | list:
    return _get(url, params, timeout).json()


def get_text(url: str, params: dict | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    return _get(url, params, timeout).text


def _get(url: str, params: dict | None, timeout: int) -> requests.Response:
    last: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = session().get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last = exc
        else:
            if response.status_code == 404:
                raise ApiError(f"not found: {response.url}")
            # 429/5xx are worth retrying; anything else is a real answer.
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                return response
            last = ApiError(f"HTTP {response.status_code} from {response.url}")

        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))

    raise ApiError(f"giving up on {url} after {_MAX_ATTEMPTS} attempts: {last}")
