"""
MIDAN Data Pipeline — Rate-Limited HTTP Client
Handles retries, rate limiting, and polite crawling.
"""

import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import HTTP_TIMEOUT, HTTP_MAX_RETRIES, HTTP_USER_AGENT, RATE_LIMIT_DELAY
from utils.logger import setup_logger

logger = setup_logger("http_client")

# Track last request time per domain
_last_request_time: dict[str, float] = {}


def _rate_limit(url: str, delay: float = RATE_LIMIT_DELAY):
    """Enforce minimum delay between requests to the same domain."""
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    now = time.time()
    last = _last_request_time.get(domain, 0)
    wait_time = delay - (now - last)
    if wait_time > 0:
        logger.debug(f"Rate limiting: waiting {wait_time:.1f}s for {domain}")
        time.sleep(wait_time)
    _last_request_time[domain] = time.time()


@retry(
    stop=stop_after_attempt(HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def fetch_url(url: str, delay: float = RATE_LIMIT_DELAY,
              headers: dict | None = None, timeout: int = HTTP_TIMEOUT) -> requests.Response | None:
    """
    Fetch a URL with rate limiting, retries, and error handling.
    Returns Response object or None on failure.
    """
    _rate_limit(url, delay)

    default_headers = {
        "User-Agent": HTTP_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    if headers:
        default_headers.update(headers)

    try:
        response = requests.get(url, headers=default_headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        logger.debug(f"Fetched {url} — {response.status_code} ({len(response.content)} bytes)")
        return response
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        if status == 403:
            logger.warning(f"Blocked (403) — {url}")
        elif status == 404:
            logger.warning(f"Not found (404) — {url}")
        elif status == 429:
            logger.warning(f"Rate limited (429) — {url} — backing off")
            time.sleep(10)
            raise  # trigger retry
        else:
            logger.warning(f"HTTP {status} — {url}")
        return None
    except requests.ConnectionError:
        logger.warning(f"Connection failed — {url}")
        raise  # trigger retry
    except requests.Timeout:
        logger.warning(f"Timeout — {url}")
        raise  # trigger retry
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return None


def fetch_json(url: str, params: dict | None = None,
               headers: dict | None = None, delay: float = RATE_LIMIT_DELAY) -> dict | None:
    """Fetch a URL and parse as JSON. Returns dict or None."""
    _rate_limit(url, delay)

    default_headers = {
        "User-Agent": HTTP_USER_AGENT,
        "Accept": "application/json",
    }
    if headers:
        default_headers.update(headers)

    try:
        response = requests.get(url, params=params, headers=default_headers,
                                timeout=HTTP_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"JSON fetch failed for {url}: {e}")
        return None
