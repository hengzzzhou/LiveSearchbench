"""Polite, bounded HTTP access to Wikimedia services.

The previous release retried failed Wikidata calls forever with no backoff and
omitted ``timeout=`` on its two highest-volume requests, so a transient
``internal_api_error_DBQueryTimeoutError`` -- which arrives as HTTP 200 with the
error inside the JSON body -- made the extractor spin indefinitely and write
nothing. Everything here is bounded: attempts are capped, backoff is
exponential, every request has a timeout, and exhaustion raises rather than
looping.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, Optional

import requests

from . import config

logger = logging.getLogger("livesearchbench.http")

#: (connect, read) timeout applied to every request.
DEFAULT_TIMEOUT = (10, 60)
#: Maximum attempts per request before giving up.
DEFAULT_MAX_ATTEMPTS = 5
#: Base seconds for exponential backoff.
DEFAULT_BACKOFF = 2.0

#: Wikidata Action API error codes that are worth retrying.
TRANSIENT_API_ERRORS = frozenset({
    "internal_api_error_DBQueryTimeoutError",
    "internal_api_error_DBConnectionError",
    "internal_api_error_ReadOnlyError",
    "maxlag",
    "ratelimited",
    "readonly",
    "servererror",
    "timeout",
})


class RequestFailed(RuntimeError):
    """Raised when a request still fails after the configured attempts."""


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a ``Retry-After`` header, which may be seconds or an HTTP-date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        from email.utils import parsedate_to_datetime
        import datetime as _dt

        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        return max(0.0, (when - _dt.datetime.now(_dt.timezone.utc)).total_seconds())
    except Exception:
        return None


class PoliteSession:
    """A ``requests.Session`` with a compliant UA, retries, and backoff."""

    def __init__(
        self,
        *,
        component: str = "LiveSearchBench",
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout=DEFAULT_TIMEOUT,
        backoff: float = DEFAULT_BACKOFF,
        min_interval: float = 0.0,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.user_agent(component),
            "Accept-Encoding": "gzip, deflate",
        })
        self.max_attempts = max(1, int(max_attempts))
        self.timeout = timeout
        self.backoff = float(backoff)
        self.min_interval = float(min_interval)
        self._last_request = 0.0

    # -- internals ---------------------------------------------------------

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _sleep(self, attempt: int, retry_after: Optional[float] = None) -> None:
        if retry_after is not None:
            delay = max(0.0, float(retry_after))
        else:
            delay = self.backoff * (2 ** (attempt - 1))
        delay += random.uniform(0, 0.25 * max(delay, 1.0))  # jitter
        time.sleep(min(delay, 60.0))

    # -- public API --------------------------------------------------------

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Issue a request with bounded retries. Raises :class:`RequestFailed`."""
        kwargs.setdefault("timeout", self.timeout)
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            self._throttle()
            try:
                response = self.session.request(method, url, **kwargs)
                self._last_request = time.monotonic()
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("%s %s failed (attempt %d/%d): %s",
                               method, url, attempt, self.max_attempts, last_error)
                if attempt < self.max_attempts:
                    self._sleep(attempt)
                continue

            if response.status_code in (429, 503):
                retry_after = response.headers.get("Retry-After")
                last_error = f"HTTP {response.status_code}"
                logger.warning("%s throttled (attempt %d/%d), Retry-After=%s",
                               url, attempt, self.max_attempts, retry_after)
                if attempt < self.max_attempts:
                    self._sleep(attempt, _parse_retry_after(retry_after))
                continue

            if 500 <= response.status_code < 600:
                last_error = f"HTTP {response.status_code}"
                logger.warning("%s server error %d (attempt %d/%d)",
                               url, response.status_code, attempt, self.max_attempts)
                if attempt < self.max_attempts:
                    self._sleep(attempt)
                continue

            return response

        raise RequestFailed(
            f"{method} {url} failed after {self.max_attempts} attempts. Last error: {last_error}"
        )

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request("POST", url, **kwargs)

    def wikidata_api(self, params: Dict[str, Any], *, endpoint: Optional[str] = None) -> Dict:
        """Call the Wikidata Action API, retrying transient in-body errors.

        Unlike the previous implementation this distinguishes transient errors
        (retried with backoff, then raised) from permanent ones (raised
        immediately), and never loops without a bound.
        """
        url = endpoint or config.get("WIKIDATA_API", default=config.DEFAULT_WIKIDATA_API)
        query = dict(params)
        query.setdefault("format", "json")
        query.setdefault("maxlag", "5")

        # ``get`` retries transport-level failures on its own. Nesting a second
        # full-length loop around it would allow max_attempts^2 requests, so the
        # transport budget is reduced to one attempt per outer iteration and the
        # outer loop owns the retry policy for in-body errors.
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            saved = self.max_attempts
            try:
                self.max_attempts = 1
                response = self.get(url, params=query)
            except RequestFailed as exc:
                self.max_attempts = saved
                last_error = str(exc)
                logger.warning("Wikidata API transport failure (attempt %d/%d): %s",
                               attempt, saved, last_error)
                if attempt < saved:
                    self._sleep(attempt)
                    continue
                raise RequestFailed(
                    f"Wikidata API failed after {saved} attempts. Last error: {last_error}"
                ) from exc
            finally:
                self.max_attempts = saved
            try:
                data = response.json()
            except ValueError:
                last_error = f"non-JSON response (HTTP {response.status_code})"
                logger.warning("Wikidata API returned non-JSON (attempt %d/%d); "
                               "this usually means the User-Agent was rejected",
                               attempt, self.max_attempts)
                if attempt < self.max_attempts:
                    self._sleep(attempt)
                continue

            error = data.get("error")
            if not error:
                return data

            code = str(error.get("code", ""))
            last_error = f"{code}: {error.get('info', '')}"
            if code in TRANSIENT_API_ERRORS:
                lag = error.get("lag")
                logger.warning("Wikidata API transient error '%s' (attempt %d/%d)",
                               code, attempt, self.max_attempts)
                if attempt < self.max_attempts:
                    self._sleep(attempt, float(lag) + 1.0 if lag else None)
                continue

            raise RequestFailed(f"Wikidata API rejected the request: {last_error}")

        raise RequestFailed(
            f"Wikidata API failed after {self.max_attempts} attempts. Last error: {last_error}\n"
            f"  If this is a DBQueryTimeoutError, lower --rc-limit; if the response was\n"
            f"  non-JSON, set LSB_CONTACT to a reachable URL or email."
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "PoliteSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
