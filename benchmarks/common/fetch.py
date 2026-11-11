"""HTTP helpers the instance downloaders share."""

from __future__ import annotations

import urllib.request


def http_get(url: str, user_agent: str, timeout: int = 120) -> bytes:
    """The body at `url`, fetched with `user_agent`."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data: bytes = resp.read()
    return data


def looks_like_html(data: bytes) -> bool:
    """Whether a download is an HTML page (an error or login page served with
    200) rather than the file asked for."""
    head = data[:512].lstrip().lower()
    return head.startswith((b"<!doctype", b"<html"))
