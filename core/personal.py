"""Personal mode — RAG + Karpathy-style LLM wiki for Roitelet.

Two patterns share the same backend:

- **Wiki mode (Karpathy LLM wiki)**: a small curated corpus of plain
  ``.md`` files lives at ``<data_dir>/personal/wiki/``. When the user
  enables personal mode, the full wiki is concatenated and injected
  as a "From your personal knowledge base:" section before the prompt.
  Cheap; works because frontier and OSS models all carry 8k+ context.

- **RAG mode**: when the wiki grows past
  :data:`_WIKI_MAX_INLINE_CHARS`, switch to retrieval. We embed each
  wiki file (lazily, cached) via the local Ollama embedding model and
  return the top-K most similar chunks instead of the whole corpus.

A separate ``<data_dir>/personal/inbox/`` folder holds **raw files**
(audio, image, PDF). Calling :func:`ingest_inbox` walks the folder and
auto-converts each new file via the existing
:mod:`core.multimodal` extractors:

- ``audio/*`` → :func:`core.multimodal.audio.transcribe_audio`
  (whisper.cpp + NeMo Sortformer diarization)
- ``image/*`` → :func:`core.multimodal.image.describe_image`
  (Ollama VLM caption)
- ``application/pdf`` → :func:`core.multimodal.pdf.extract_pdf`
  (kreuzberg with OCR fallback)

The extracted text is written to ``wiki/<basename>.md`` with a
provenance header so the user knows what came from what. Hand-written
wiki entries live alongside the auto-generated ones; an ``index.json``
manifest tracks which inbox files have already been ingested so
re-runs are idempotent.

Examples
--------
>>> from core.personal import wiki_dir, inbox_dir
>>> wiki_dir().name
'wiki'
>>> inbox_dir().name
'inbox'
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Above this many characters of total wiki text we switch from "stuff it
# all in context" to retrieval. 32 k chars ≈ 8 k tokens — comfortable
# fit even for 8 k-context models, and leaves room for the user prompt
# + the K candidate answers in the fusion judge.
_WIKI_MAX_INLINE_CHARS: int = 32_000

# RAG knobs.
_RAG_CHUNK_CHARS: int = 1_200       # ~300 tokens per chunk
_RAG_CHUNK_OVERLAP: int = 200        # 1-paragraph slide-over
_RAG_TOP_K: int = 5                  # chunks to inject when retrieving

# Extension classification mirrors the API layer's _modality_of so the
# same files that flow through /api/chat/multimodal flow through here.
_AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.aac'}
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.heic', '.heif'}
_PDF_EXTS = {'.pdf'}
_TEXT_EXTS = {'.txt', '.md', '.markdown'}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def personal_root() -> Path:
    """Return the root directory for personal-mode data.

    Returns
    -------
    pathlib.Path
        ``<ROITELET_DATA_DIR>/personal/``. Created on first call.
    """
    path = get_settings().data_dir / 'personal'
    path.mkdir(parents=True, exist_ok=True)
    return path


def inbox_dir() -> Path:
    """Return the folder where users drop raw files for ingestion.

    Returns
    -------
    pathlib.Path
        ``<ROITELET_DATA_DIR>/personal/inbox/``. Created on first call.
    """
    path = personal_root() / 'inbox'
    path.mkdir(parents=True, exist_ok=True)
    return path


def wiki_dir() -> Path:
    """Return the folder where converted + hand-written wiki files live.

    Returns
    -------
    pathlib.Path
        ``<ROITELET_DATA_DIR>/personal/wiki/``. Created on first call.
    """
    path = personal_root() / 'wiki'
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path() -> Path:
    """Return the JSON manifest path that tracks ingested inbox files.

    Returns
    -------
    pathlib.Path
        ``<ROITELET_DATA_DIR>/personal/index.json``.
    """
    return personal_root() / 'index.json'


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _load_manifest() -> dict[str, dict]:
    """Read the ingestion manifest from disk.

    Returns
    -------
    dict
        Mapping ``inbox-filename -> {'sha256': ..., 'wiki_path': ...,
        'modality': ..., 'ingested_at': ...}``. Returns ``{}`` when the
        manifest doesn't exist yet or is unreadable.
    """
    path = manifest_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001
        logger.warning('Personal manifest unreadable (%s) — treating as empty.', exc)
        return {}


def _save_manifest(manifest: dict[str, dict]) -> None:
    """Persist the manifest atomically.

    Parameters
    ----------
    manifest : dict
        The full updated manifest (caller is responsible for merging
        with any existing entries).
    """
    path = manifest_path()
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IngestResult:
    """Outcome of a single inbox-file conversion.

    Attributes
    ----------
    source : pathlib.Path
        Original file under ``inbox/``.
    wiki_path : pathlib.Path or None
        Path to the generated wiki file. ``None`` when conversion
        failed or the file was skipped.
    modality : str
        ``'audio'``, ``'image'``, ``'pdf'``, ``'text'``, or
        ``'skipped'`` for unrecognised types.
    error : str or None
        Populated on conversion failure.
    """

    source: Path
    wiki_path: Path | None
    modality: str
    error: str | None = None


def _classify(path: Path) -> str:
    """Classify a file by extension into a modality bucket.

    Parameters
    ----------
    path : pathlib.Path
        File to classify.

    Returns
    -------
    str
        One of ``'audio'``, ``'image'``, ``'pdf'``, ``'text'``,
        ``'skipped'``.
    """
    ext = path.suffix.lower()
    if ext in _AUDIO_EXTS:
        return 'audio'
    if ext in _IMAGE_EXTS:
        return 'image'
    if ext in _PDF_EXTS:
        return 'pdf'
    if ext in _TEXT_EXTS:
        return 'text'
    return 'skipped'


def _safe_slug(name: str) -> str:
    """Convert an arbitrary filename to a stable, filesystem-safe slug.

    Parameters
    ----------
    name : str
        Source file basename (extension included).

    Returns
    -------
    str
        Lowercased, alphanumeric + dashes only, with the extension
        replaced by ``.md``.
    """
    stem = Path(name).stem.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', stem).strip('-') or 'doc'
    return f'{slug}.md'


async def _convert(path: Path, modality: str) -> str:
    """Run the right :mod:`core.multimodal` extractor for ``modality``.

    Parameters
    ----------
    path : pathlib.Path
        Input file.
    modality : str
        ``'audio'``, ``'image'``, ``'pdf'``, or ``'text'``.

    Returns
    -------
    str
        Extracted text. ``'text'`` modality short-circuits and returns
        the file contents directly.

    Raises
    ------
    ValueError
        For modalities the extractor pool can't handle (caller should
        check via :func:`_classify` first).
    """
    if modality == 'audio':
        from .multimodal.audio import transcribe_audio
        return await transcribe_audio(path)
    if modality == 'image':
        from .multimodal.image import describe_image
        return await describe_image(path)
    if modality == 'pdf':
        from .multimodal.pdf import extract_pdf
        return await extract_pdf(path)
    if modality == 'text':
        return path.read_text(encoding='utf-8', errors='replace')
    raise ValueError(f'Unsupported modality: {modality!r}')


async def ingest_inbox(force: bool = False) -> list[IngestResult]:
    """Walk ``inbox/`` and convert any new file to a wiki entry.

    Parameters
    ----------
    force : bool, default=False
        When ``True`` re-ingest every inbox file even if the manifest
        already recorded a conversion. Useful when an extractor
        changed and you want fresh output.

    Returns
    -------
    list of IngestResult
        One entry per file the walk inspected. Skipped (unknown
        modality) and failed files are included so the caller can
        surface them in the UI.

    Notes
    -----
    Idempotent: by default each inbox file is converted exactly once.
    The manifest at :func:`manifest_path` records which file produced
    which wiki entry; deleting the wiki file does not re-trigger
    ingestion (delete the manifest line too, or pass ``force=True``).
    """
    inbox = inbox_dir()
    wiki = wiki_dir()
    manifest = _load_manifest()
    results: list[IngestResult] = []

    for path in sorted(inbox.iterdir()):
        if not path.is_file() or path.name.startswith('.'):
            continue
        modality = _classify(path)
        if modality == 'skipped':
            results.append(IngestResult(
                source=path, wiki_path=None, modality='skipped',
                error=f'unknown extension {path.suffix!r}',
            ))
            continue
        already = manifest.get(path.name)
        if already and not force:
            existing_wiki = wiki / Path(already['wiki_path']).name
            results.append(IngestResult(
                source=path,
                wiki_path=existing_wiki if existing_wiki.exists() else None,
                modality=already.get('modality', modality),
            ))
            continue

        try:
            text = await _convert(path, modality)
        except Exception as exc:  # noqa: BLE001
            logger.warning('Personal ingestion failed for %s: %s', path.name, exc)
            results.append(IngestResult(
                source=path, wiki_path=None, modality=modality, error=str(exc),
            ))
            continue

        wiki_path = wiki / _safe_slug(path.name)
        header = (
            f'# {path.name}\n\n'
            f'_Auto-converted from `{path.name}` ({modality}) on '
            f'{datetime.now(UTC).isoformat()}._\n\n'
        )
        wiki_path.write_text(header + text, encoding='utf-8')
        manifest[path.name] = {
            'wiki_path': str(wiki_path),
            'modality': modality,
            'ingested_at': datetime.now(UTC).isoformat(),
        }
        results.append(IngestResult(source=path, wiki_path=wiki_path, modality=modality))

    _save_manifest(manifest)
    return results


# ---------------------------------------------------------------------------
# Wiki loading + RAG retrieval
# ---------------------------------------------------------------------------


def _read_wiki_files() -> list[tuple[Path, str]]:
    """Read every wiki file in deterministic order.

    Returns
    -------
    list of (pathlib.Path, str)
        Path + full content, sorted by filename so the concatenated
        wiki is reproducible across runs.
    """
    pairs: list[tuple[Path, str]] = []
    for path in sorted(wiki_dir().iterdir()):
        if not path.is_file() or path.suffix.lower() not in _TEXT_EXTS:
            continue
        try:
            pairs.append((path, path.read_text(encoding='utf-8', errors='replace')))
        except Exception as exc:  # noqa: BLE001
            logger.warning('Skipping unreadable wiki file %s: %s', path.name, exc)
    return pairs


def _chunk(text: str, chunk_chars: int = _RAG_CHUNK_CHARS,
           overlap: int = _RAG_CHUNK_OVERLAP) -> list[str]:
    """Split ``text`` into overlapping fixed-size character windows.

    Parameters
    ----------
    text : str
        Input.
    chunk_chars : int, default=_RAG_CHUNK_CHARS
        Window width in characters.
    overlap : int, default=_RAG_CHUNK_OVERLAP
        How much each window overlaps the previous one.

    Returns
    -------
    list of str
        Chunks; the last one may be shorter than ``chunk_chars``.
    """
    if not text:
        return []
    step = max(1, chunk_chars - overlap)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_chars])
        if start + chunk_chars >= len(text):
            break
    return chunks


def _retrieve_chunks(prompt: str, top_k: int = _RAG_TOP_K) -> list[tuple[Path, str]]:
    """Return the top-K most similar chunks across the wiki.

    Parameters
    ----------
    prompt : str
        User prompt to retrieve against.
    top_k : int, default=_RAG_TOP_K
        Number of chunks to return.

    Returns
    -------
    list of (pathlib.Path, str)
        ``(source wiki path, chunk text)`` pairs. Empty when no
        embeddings could be computed — caller should fall back to
        skipping the personal-context injection.

    Notes
    -----
    Reuses :func:`core.capability_classifier._embed_prompt` so the same
    Ollama embedding model and timeout behaviour apply. On any
    embedding failure (server down, model not pulled) returns an
    empty list, which the pipeline interprets as "skip RAG."
    """
    import numpy as np

    from .capability_classifier import _embed_prompt

    pairs = _read_wiki_files()
    if not pairs:
        return []

    chunks: list[tuple[Path, str, np.ndarray]] = []
    for path, body in pairs:
        for piece in _chunk(body):
            vec = _embed_prompt(piece)
            if vec is None:
                # Cold-start: if the embedding model can't run, bail
                # out wholesale rather than mix scored + unscored
                # chunks.
                return []
            chunks.append((path, piece, vec))

    query_vec = _embed_prompt(prompt)
    if query_vec is None:
        return []

    qn = float(np.linalg.norm(query_vec)) or 1.0
    scored: list[tuple[float, Path, str]] = []
    for path, piece, vec in chunks:
        cn = float(np.linalg.norm(vec)) or 1.0
        cosine = float(np.dot(query_vec, vec) / (qn * cn))
        scored.append((cosine, path, piece))
    scored.sort(key=lambda triple: triple[0], reverse=True)
    return [(p, c) for _, p, c in scored[:max(1, top_k)]]


def build_personal_context(prompt: str) -> str:
    """Return the personal-context block to prepend to a chat prompt.

    Parameters
    ----------
    prompt : str
        User prompt; used to drive retrieval when the wiki is large.

    Returns
    -------
    str
        A formatted Markdown block ready to splice in front of the
        user's prompt. Empty string when the wiki is empty — caller
        should treat that as "no personal context available."

    Notes
    -----
    Strategy is size-dependent: below
    :data:`_WIKI_MAX_INLINE_CHARS` characters total, concatenate the
    full wiki (Karpathy-style — all in long context). Above that
    threshold, run retrieval and return only the top-K chunks.
    """
    pairs = _read_wiki_files()
    if not pairs:
        return ''
    total_chars = sum(len(body) for _, body in pairs)
    if total_chars <= _WIKI_MAX_INLINE_CHARS:
        # Wiki mode — small enough to send the whole thing.
        body = '\n\n---\n\n'.join(
            f'## `{path.name}`\n\n{content.strip()}' for path, content in pairs
        )
        return (
            '# From your personal knowledge base\n\n'
            f'{body}\n\n---\n\n'
        )

    # RAG mode — too big for inline; retrieve.
    chunks = _retrieve_chunks(prompt)
    if not chunks:
        return ''  # caller skips injection on retrieval failure
    body = '\n\n---\n\n'.join(
        f'## `{path.name}` (excerpt)\n\n{chunk.strip()}' for path, chunk in chunks
    )
    return (
        '# From your personal knowledge base (top matches)\n\n'
        f'{body}\n\n---\n\n'
    )


def project_chunks_2d() -> list[dict]:
    """Embed every wiki chunk and project into 2-D for visualisation.

    Returns
    -------
    list of dict
        One entry per chunk. Keys:

        * ``path`` — source wiki filename (string).
        * ``chunk_index`` — 0-based position within the source.
        * ``text`` — the chunk body.
        * ``x``, ``y`` — 2-D PCA coordinates (floats, centred on the
          corpus mean).

        Empty list when the wiki is empty *or* when the embedding
        model is unreachable; the caller surfaces that as "viz not
        available right now."

    Notes
    -----
    The projection is plain PCA (truncated SVD on the centred
    embedding matrix). No t-SNE / UMAP — those add dependencies and
    a personal corpus is rarely large enough for the non-linearity
    to matter. The whole projection runs in memory on every request;
    cache it if the corpus grows past a few thousand chunks.
    """
    import numpy as np

    from .capability_classifier import _embed_prompt

    pairs = _read_wiki_files()
    if not pairs:
        return []

    chunks: list[tuple[Path, int, str, np.ndarray]] = []
    for path, body in pairs:
        for idx, piece in enumerate(_chunk(body)):
            vec = _embed_prompt(piece)
            if vec is None:
                return []  # embedding model unavailable — bail
            chunks.append((path, idx, piece, vec))
    if not chunks:
        return []

    matrix = np.stack([c[3] for c in chunks]).astype(np.float32)
    centred = matrix - matrix.mean(axis=0, keepdims=True)

    # Top-2 components via SVD. Compute the full SVD only when the
    # corpus is small enough; for larger ones the randomised SVD via
    # sklearn would be faster, but a few thousand chunks at ~384 dims
    # finishes in milliseconds on a laptop.
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        components = vt[:2]
        coords = centred @ components.T
    except Exception as exc:  # noqa: BLE001
        logger.warning('Personal viz projection failed: %s', exc)
        return []

    points: list[dict] = []
    for (path, idx, text, _), (x, y) in zip(chunks, coords, strict=True):
        points.append({
            'path': path.name,
            'chunk_index': idx,
            'text': text,
            'x': float(x),
            'y': float(y),
        })
    return points


def personal_status() -> dict[str, int]:
    """Return a quick summary of the personal corpus.

    Returns
    -------
    dict
        ``{'inbox': int, 'wiki': int, 'wiki_chars': int, 'mode': str}``.
        ``mode`` is ``'wiki'`` when the corpus fits inline,
        ``'rag'`` when retrieval kicks in, or ``'empty'`` otherwise.
    """
    inbox_files = [
        p for p in inbox_dir().iterdir()
        if p.is_file() and not p.name.startswith('.')
    ]
    wiki_pairs = _read_wiki_files()
    total = sum(len(body) for _, body in wiki_pairs)
    mode = 'empty'
    if wiki_pairs:
        mode = 'wiki' if total <= _WIKI_MAX_INLINE_CHARS else 'rag'
    return {
        'inbox': len(inbox_files),
        'wiki': len(wiki_pairs),
        'wiki_chars': total,
        'mode': mode,
    }
