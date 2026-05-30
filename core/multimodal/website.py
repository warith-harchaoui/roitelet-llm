"""Website-to-text extraction via Firecrawl.

Adds a fourth modality (website URLs) to Roitelet's
audio / image / PDF / website family. The contract is the same
as the other three: take an opaque source, return a plain string
the rest of the pipeline can splice into the user prompt.

Why Firecrawl
-------------
A normal user typing "summarise this article" attaches a URL the
same way they'd attach a PDF. We need a fetcher that:

* renders client-side JS (most articles today),
* strips chrome / ads / nav so the LLM sees clean prose,
* returns markdown (not raw HTML — chat models reason about
  markdown more reliably).

Firecrawl (`firecrawl-py`) does all three and exposes a cheap async
``AsyncFirecrawl().scrape(url)`` call that returns markdown.

Modes
-----
* **Hosted Firecrawl** — set ``FIRECRAWL_API_KEY``; the SDK talks
  to ``https://api.firecrawl.dev``. Default.
* **Self-hosted Firecrawl** — set ``FIRECRAWL_API_URL`` to your
  own instance (https://github.com/mendableai/firecrawl) and leave
  the key blank or set the local-only token. Lets the local-first
  privacy story extend to website ingestion when the user runs
  Firecrawl on their own machine.

Both modes are reached through the same Python SDK; we don't
maintain provider clients here.

Dependencies
------------
``firecrawl-py`` is in the ``[multimodal]`` extras. The module
import is lazy so the base install stays light; a clear
``ImportError`` reaches the API layer as a 503 when the dep is
missing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# How many characters of scraped markdown to keep at most. A long
# article can easily exceed 100k characters — most local Ollama
# models cap context around 8–32k tokens (roughly 32–128k chars).
# Truncation is a hard cap rather than a chunker; the user sees
# the cap in the audit so they know the model saw a slice, not
# the whole page.
_MAX_CHARS = 80_000


async def fetch_website(url: str) -> str:
    """Fetch ``url`` and return Firecrawl-rendered markdown.

    Parameters
    ----------
    url:
        Absolute http(s) URL. Validation is intentionally minimal —
        we trust Firecrawl to refuse or redirect anything weird.

    Returns
    -------
    str
        Markdown body of the page, possibly truncated to
        :data:`_MAX_CHARS`. The first line of the returned string
        is a labelled header (``# <title>``) when Firecrawl
        provided one — gives the LLM something to anchor on.

    Raises
    ------
    ImportError
        When the optional ``firecrawl-py`` extra is not installed.
        The API layer surfaces this as HTTP 503 with a
        ``pip install -e .[multimodal]`` hint.
    RuntimeError
        On any Firecrawl error or empty content. Surfaced to the
        user as part of the multimodal "skipped" note so the
        request still completes with whatever else they sent.
    """
    try:
        from firecrawl import AsyncFirecrawl  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            'firecrawl-py is not installed. `pip install -e .[multimodal]` '
            'to enable website attachments.'
        ) from exc

    api_key = os.environ.get('FIRECRAWL_API_KEY') or None
    api_url = os.environ.get('FIRECRAWL_API_URL') or None

    # The SDK accepts ``api_key`` and ``api_url`` keyword arguments;
    # passing ``None`` makes it use the hosted defaults.
    client_kwargs: dict = {}
    if api_key:
        client_kwargs['api_key'] = api_key
    if api_url:
        client_kwargs['api_url'] = api_url
    client = AsyncFirecrawl(**client_kwargs)

    try:
        result = await client.scrape(url=url, formats=['markdown'])
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f'Firecrawl scrape failed for {url}: {exc}') from exc

    markdown = getattr(result, 'markdown', None) or (
        result.get('markdown') if isinstance(result, dict) else None
    )
    if not markdown:
        raise RuntimeError(f'Firecrawl returned no markdown for {url}.')

    title = None
    metadata = getattr(result, 'metadata', None) or (
        result.get('metadata') if isinstance(result, dict) else None
    )
    if isinstance(metadata, dict):
        title = metadata.get('title') or metadata.get('ogTitle')

    body = markdown.strip()
    if len(body) > _MAX_CHARS:
        body = body[:_MAX_CHARS] + f'\n\n*[truncated at {_MAX_CHARS} chars]*'

    if title:
        return f'# {title}\n\n{body}'
    return body
