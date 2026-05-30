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

# Cap on the number of pages Firecrawl returns for a recursive
# crawl. The default is meant to be conservative: too many pages
# blow past the context budget of a small local judge. Override at
# call time when the user explicitly wants a deeper crawl.
_DEFAULT_RECURSIVE_LIMIT = 10


def _make_client():
    """Build an AsyncFirecrawl client honouring the environment overrides."""
    try:
        from firecrawl import AsyncFirecrawl  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            'firecrawl-py is not installed. `pip install -e .[multimodal]` '
            'to enable website attachments.'
        ) from exc
    api_key = os.environ.get('FIRECRAWL_API_KEY') or None
    api_url = os.environ.get('FIRECRAWL_API_URL') or None
    kwargs: dict = {}
    if api_key:
        kwargs['api_key'] = api_key
    if api_url:
        kwargs['api_url'] = api_url
    return AsyncFirecrawl(**kwargs)


def _extract_markdown_and_title(result) -> tuple[str, str | None]:
    """Pull (markdown, title) out of a Firecrawl scrape/crawl-page result.

    The SDK returns either an object with attributes or a plain dict,
    depending on version. Cover both shapes here so the rest of the
    module doesn't have to.
    """
    markdown = getattr(result, 'markdown', None) or (
        result.get('markdown') if isinstance(result, dict) else None
    )
    metadata = getattr(result, 'metadata', None) or (
        result.get('metadata') if isinstance(result, dict) else None
    )
    title: str | None = None
    source_url: str | None = None
    if isinstance(metadata, dict):
        title = metadata.get('title') or metadata.get('ogTitle')
        source_url = metadata.get('sourceURL') or metadata.get('url')
    return markdown or '', title or source_url


async def fetch_website(url: str, *, recursive: bool = False, limit: int | None = None) -> str:
    """Fetch ``url`` and return Firecrawl-rendered markdown.

    Parameters
    ----------
    url:
        Absolute http(s) URL. Validation is intentionally minimal —
        we trust Firecrawl to refuse or redirect anything weird.
    recursive:
        When ``True`` Firecrawl follows links from the page and
        returns up to ``limit`` (default :data:`_DEFAULT_RECURSIVE_LIMIT`)
        sub-pages. The pages are concatenated with ``---`` separators
        and a per-page ``# <title or url>`` header so the LLM can tell
        them apart. Off by default — a single-page scrape is cheaper
        and almost always what the user wants.
    limit:
        Hard cap on the number of pages returned by the recursive
        crawl. Ignored when ``recursive`` is ``False``.

    Returns
    -------
    str
        Markdown body of the page(s), possibly truncated to
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
    client = _make_client()

    if not recursive:
        try:
            result = await client.scrape(url=url, formats=['markdown'])
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f'Firecrawl scrape failed for {url}: {exc}') from exc
        markdown, title = _extract_markdown_and_title(result)
        if not markdown:
            raise RuntimeError(f'Firecrawl returned no markdown for {url}.')
        body = markdown.strip()
        if len(body) > _MAX_CHARS:
            body = body[:_MAX_CHARS] + f'\n\n*[truncated at {_MAX_CHARS} chars]*'
        return f'# {title}\n\n{body}' if title else body

    # Recursive path — crawl, then concatenate the returned pages.
    page_cap = max(1, limit or _DEFAULT_RECURSIVE_LIMIT)
    try:
        crawl_result = await client.crawl(
            url=url,
            limit=page_cap,
            scrape_options={'formats': ['markdown']},
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f'Firecrawl crawl failed for {url}: {exc}') from exc

    pages = getattr(crawl_result, 'data', None) or (
        crawl_result.get('data') if isinstance(crawl_result, dict) else None
    ) or []
    if not pages:
        raise RuntimeError(f'Firecrawl crawl returned no pages for {url}.')

    blocks: list[str] = []
    remaining = _MAX_CHARS
    for page in pages:
        markdown, title = _extract_markdown_and_title(page)
        if not markdown:
            continue
        body = markdown.strip()
        # Pro-rate the per-page budget so an early huge page doesn't
        # starve the rest. Each page gets remaining // pages-left.
        share = max(1024, remaining // max(1, page_cap))
        if len(body) > share:
            body = body[:share] + '\n\n*[truncated]*'
        header = f'## {title}' if title else '## (untitled page)'
        blocks.append(f'{header}\n\n{body}')
        remaining -= len(body) + len(header) + 4
        if remaining <= 0:
            break

    if not blocks:
        raise RuntimeError(f'Firecrawl crawl returned no usable markdown for {url}.')

    note = f'*[Firecrawl recursive crawl from {url} — {len(blocks)} page(s)]*'
    return f'{note}\n\n' + '\n\n---\n\n'.join(blocks)
