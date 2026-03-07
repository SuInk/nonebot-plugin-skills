import httpx
from typing import Optional

_HTTP_CLIENT: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(follow_redirects=True)
    return _HTTP_CLIENT
