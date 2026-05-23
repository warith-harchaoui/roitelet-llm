"""One-shot PDF text extraction for Roitelet.

Uses `kreuzberg` (https://github.com/Goldziher/kreuzberg) which wraps
pdfium for text-layer PDFs and Tesseract for scanned / image-only PDFs.
Same contract as the audio module: one async coroutine, returns plain
text ready to splice into the user prompt.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def extract_pdf(path: Path) -> str:
    """Extract text from one PDF file.

    Parameters
    ----------
    path:
        Path to a PDF document.

    Returns
    -------
    str
        The extracted text, OCR'd transparently if the PDF has no text
        layer. Empty string if extraction yields nothing usable.
    """
    from kreuzberg import extract_file  # type: ignore[import-not-found]

    result = await extract_file(str(path))
    content = getattr(result, 'content', '') or ''
    return content.strip()
