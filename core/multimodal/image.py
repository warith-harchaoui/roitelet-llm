"""One-shot image-to-text captioning via a local Ollama VLM.

Mirrors the contract of ``audio.py`` and ``pdf.py``: one async coroutine,
in = file path, out = a plain text description ready to be spliced into
the user prompt. The downstream router/judge/synthesis stays text-only.

The VLM model is configurable via ``local_vlm_model`` (default
``qwen2.5vl:7b``). The user must have it pulled in Ollama; we don't
auto-pull because that's a multi-gigabyte side effect.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

from ..config import get_settings
from ..storage import get_storage

logger = logging.getLogger(__name__)


_DESCRIBE_PROMPT = (
    'Describe this image in thorough detail so a text-only assistant could '
    'answer questions about it. Cover: subjects, layout, text/labels/OCR, '
    'colors, style, notable objects, and anything that looks unusual or '
    'noteworthy. Be specific and factual; avoid speculation.'
)


async def describe_image(path: Path) -> str:
    """Return a textual description of one image file.

    Parameters
    ----------
    path:
        Path to an image (jpg, png, webp, gif, bmp …).

    Returns
    -------
    str
        Textual description from the local VLM, or an empty string if the
        VLM is unavailable / returns nothing.
    """
    settings = get_settings()
    runtime = get_storage().load_app_settings()
    base_url = (runtime.ollama_base_url or settings.local_llm_base_url).rstrip('/')
    vlm_model = runtime.local_vlm_model or settings.local_vlm_model

    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    payload = {
        'model': vlm_model,
        'messages': [
            {
                'role': 'user',
                'content': _DESCRIBE_PROMPT,
                'images': [encoded],
            }
        ],
        'stream': False,
    }
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f'{base_url}/api/chat', json=payload)
            response.raise_for_status()
            data = response.json()
        return (data.get('message', {}).get('content', '') or '').strip()
    except Exception as exc:
        logger.warning('Local VLM (%s) failed on %s: %s', vlm_model, path.name, exc)
        return ''
