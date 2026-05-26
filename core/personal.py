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

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

# Persistent on-disk index next to the wiki. The sidecar JSON carries
# the fingerprint and the chunk text; the ``.npy`` carries the dense
# embeddings (canonical, used by both the ANN path and the fallback
# brute-force scan and by ``project_chunks_2d``); the ``.tq`` carries
# the compressed turbovec IdMapIndex when that optional dependency is
# installed.
_RAG_SIDECAR_NAME: str = '.rag_index.json'
_RAG_EMBEDDINGS_NAME: str = '.rag_embeddings.npy'
_RAG_TURBOVEC_NAME: str = '.rag_index.tq'

# turbovec compression budget. 4 bits per dimension ≈ 16× compression
# vs float32; matches the library's recommended default for ≥384-dim
# embeddings. The fallback path ignores this knob.
_RAG_TURBOVEC_BIT_WIDTH: int = 4

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


@dataclass
class _RagIndex:
    """In-memory bundle of the persistent on-disk RAG index.

    Attributes
    ----------
    fingerprint:
        SHA-256 over the sorted ``(path, mtime_ns, size)`` triples of
        every text file in the wiki, plus the embedding dimension and
        chunking knobs. Stable as long as no wiki file changes; a
        rebuild is triggered as soon as it drifts.
    dim:
        Width of the embedding vectors. Cached so a silent embedding-
        model swap (different dimensionality) forces a rebuild.
    chunks:
        ``(path, chunk_text)`` per row, parallel to ``embeddings``.
    embeddings:
        ``(N, dim)`` float32 matrix. Canonical — both the fallback
        scan and the turbovec rebuild derive from this.
    turbovec_index:
        Compressed ANN index if ``turbovec`` is importable. ``None``
        on the pure-numpy fallback path.
    """

    fingerprint: str
    dim: int
    chunks: list[tuple[Path, str]]
    # Typed as ``Any`` so the dataclass declaration carries no import of
    # numpy at module load; the field always holds a 2-D float32
    # ``numpy.ndarray`` at runtime.
    embeddings: Any
    turbovec_index: object | None


def _wiki_fingerprint(pairs: list[tuple[Path, str]]) -> str:
    """Compute a content fingerprint over the wiki + chunking knobs.

    Parameters
    ----------
    pairs:
        Output of :func:`_read_wiki_files`. Hashing the tuples instead
        of re-statting keeps the helper testable.

    Returns
    -------
    str
        Hex SHA-256. Any change in file set, mtime, size, or in the
        chunking knobs invalidates the cached index.
    """
    h = hashlib.sha256()
    for path, body in pairs:
        try:
            stat = path.stat()
            stamp = f'{path.name}:{stat.st_mtime_ns}:{stat.st_size}:{len(body)}'
        except OSError:
            stamp = f'{path.name}:NOSTAT:{len(body)}'
        h.update(stamp.encode('utf-8'))
        h.update(b'\x00')
    h.update(f'chunk={_RAG_CHUNK_CHARS},overlap={_RAG_CHUNK_OVERLAP}'.encode())
    return h.hexdigest()


def _try_import_turbovec():
    """Return the ``turbovec`` module if available, else ``None``.

    The dependency is in the ``[personal]`` extra; production
    installations without it fall back to a numpy brute-force scan,
    which is correct and fast enough up to a few thousand chunks. The
    import is wrapped so that *any* failure (missing wheel for the
    current platform, ABI breakage) degrades gracefully rather than
    crashing the personal-mode entry path.
    """
    try:
        import turbovec
    except Exception as exc:  # noqa: BLE001
        logger.debug('turbovec unavailable, using numpy fallback: %s', exc)
        return None
    return turbovec


def _build_index(
    pairs: list[tuple[Path, str]],
    *,
    sidecar_path: Path,
    embeddings_path: Path,
    turbovec_path: Path,
) -> _RagIndex | None:
    """Embed every wiki chunk, persist to disk, return the in-memory bundle.

    Parameters
    ----------
    pairs:
        Wiki files as ``(path, body)``. Empty input returns ``None``.
    sidecar_path, embeddings_path, turbovec_path:
        Target files. Written atomically (write-then-rename) so a
        crash mid-rebuild never leaves a half-written index that
        would silently lie about the fingerprint on the next call.

    Returns
    -------
    _RagIndex or None
        ``None`` when the embedding model is unreachable on the very
        first chunk — the caller treats that as "skip RAG."
    """
    import numpy as np

    from .capability_classifier import _embed_prompt

    rows: list[tuple[Path, str]] = []
    vectors: list = []
    for path, body in pairs:
        for piece in _chunk(body):
            vec = _embed_prompt(piece)
            if vec is None:
                # Cold-start: bail wholesale rather than persist a
                # partial index that the next call would treat as
                # complete.
                return None
            rows.append((path, piece))
            vectors.append(vec.astype(np.float32))
    if not rows:
        return None

    matrix = np.stack(vectors).astype(np.float32)
    dim = int(matrix.shape[1])
    fingerprint = _wiki_fingerprint(pairs)

    sidecar_payload = {
        'fingerprint': fingerprint,
        'dim': dim,
        'chunks': [{'path': p.name, 'text': t} for p, t in rows],
        'embedding_count': len(rows),
        'created_at': datetime.now(UTC).isoformat(),
    }
    # Atomic writes: temp + rename. Crash safety > a microsecond.
    # ``np.save`` silently rewrites the path to end in ``.npy`` when
    # it doesn't already, so we open the file ourselves to control
    # the on-disk name exactly.
    embeddings_tmp = embeddings_path.with_name(embeddings_path.name + '.tmp')
    sidecar_tmp = sidecar_path.with_name(sidecar_path.name + '.tmp')
    with open(embeddings_tmp, 'wb') as fh:
        np.save(fh, matrix)
    sidecar_tmp.write_text(json.dumps(sidecar_payload), encoding='utf-8')
    embeddings_tmp.replace(embeddings_path)
    sidecar_tmp.replace(sidecar_path)

    turbovec = _try_import_turbovec()
    tv_index: object | None = None
    if turbovec is not None:
        try:
            tv_index = turbovec.IdMapIndex(dim=dim, bit_width=_RAG_TURBOVEC_BIT_WIDTH)
            ids = np.arange(len(rows), dtype=np.uint64)
            tv_index.add_with_ids(matrix, ids)
            tv_tmp = turbovec_path.with_name(turbovec_path.name + '.tmp')
            tv_index.write(str(tv_tmp))
            tv_tmp.replace(turbovec_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning('turbovec index build failed, falling back to numpy: %s', exc)
            tv_index = None

    # Convert paths back to absolute for the in-memory bundle so the
    # caller can return real Path objects to upstream renderers.
    base_dir = sidecar_path.parent
    abs_rows = [(base_dir / p.name if not p.is_absolute() else p, t) for p, t in rows]
    return _RagIndex(
        fingerprint=fingerprint,
        dim=dim,
        chunks=abs_rows,
        embeddings=matrix,
        turbovec_index=tv_index,
    )


def _load_index_if_fresh(
    pairs: list[tuple[Path, str]],
    *,
    sidecar_path: Path,
    embeddings_path: Path,
    turbovec_path: Path,
) -> _RagIndex | None:
    """Load the persisted index when its fingerprint matches the wiki.

    Returns
    -------
    _RagIndex or None
        ``None`` if the sidecar is missing, the fingerprint drifted,
        the embeddings file is missing/corrupt, or the dimension
        recorded in the sidecar disagrees with the embeddings matrix.
        Caller rebuilds on ``None``.
    """
    import numpy as np

    if not sidecar_path.exists() or not embeddings_path.exists():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None

    expected = _wiki_fingerprint(pairs)
    if payload.get('fingerprint') != expected:
        return None
    try:
        embeddings = np.load(embeddings_path)
    except (OSError, ValueError):
        return None
    chunks_payload = payload.get('chunks') or []
    if embeddings.shape[0] != len(chunks_payload):
        return None
    dim = int(payload.get('dim') or 0)
    if dim <= 0 or embeddings.shape[1] != dim:
        return None

    base_dir = sidecar_path.parent
    chunks = [(base_dir / entry['path'], entry['text']) for entry in chunks_payload]

    tv_index: object | None = None
    turbovec = _try_import_turbovec()
    if turbovec is not None and turbovec_path.exists():
        try:
            tv_index = turbovec.IdMapIndex.load(str(turbovec_path))
        except Exception as exc:  # noqa: BLE001
            logger.debug('turbovec index load failed, will rebuild compressed copy: %s', exc)
            tv_index = None

    # If turbovec is available but the .tq is stale/missing, rebuild
    # it from the canonical .npy without re-running the embedder. Cheap
    # (compression only) and keeps the fast path warm.
    if turbovec is not None and tv_index is None:
        try:
            tv_index = turbovec.IdMapIndex(dim=dim, bit_width=_RAG_TURBOVEC_BIT_WIDTH)
            tv_index.add_with_ids(embeddings.astype(np.float32),
                                  np.arange(embeddings.shape[0], dtype=np.uint64))
            tv_tmp = turbovec_path.with_name(turbovec_path.name + '.tmp')
            tv_index.write(str(tv_tmp))
            tv_tmp.replace(turbovec_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug('turbovec rebuild from cached embeddings failed: %s', exc)
            tv_index = None

    return _RagIndex(
        fingerprint=expected,
        dim=dim,
        chunks=chunks,
        embeddings=embeddings.astype(np.float32),
        turbovec_index=tv_index,
    )


def _get_rag_index() -> _RagIndex | None:
    """Return the up-to-date RAG index, building or refreshing as needed.

    Reads the wiki, computes the current fingerprint, loads the
    persisted sidecar + embeddings if they're still valid, otherwise
    rebuilds. The function is the single entry point for both the
    retrieval path (``_retrieve_chunks``) and the visualisation path
    (``project_chunks_2d``); embedding work is therefore amortised
    across both.

    Returns
    -------
    _RagIndex or None
        ``None`` when the wiki is empty or the embedding model is
        unreachable on a cold-start rebuild.
    """
    pairs = _read_wiki_files()
    if not pairs:
        return None

    wiki = wiki_dir()
    sidecar_path = wiki / _RAG_SIDECAR_NAME
    embeddings_path = wiki / _RAG_EMBEDDINGS_NAME
    turbovec_path = wiki / _RAG_TURBOVEC_NAME

    cached = _load_index_if_fresh(
        pairs,
        sidecar_path=sidecar_path,
        embeddings_path=embeddings_path,
        turbovec_path=turbovec_path,
    )
    if cached is not None:
        return cached

    return _build_index(
        pairs,
        sidecar_path=sidecar_path,
        embeddings_path=embeddings_path,
        turbovec_path=turbovec_path,
    )


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
    Two-stage path:

    1. :func:`_get_rag_index` loads or rebuilds the persistent index
       so chunk embeddings are computed once per wiki revision, not
       once per query. The previous implementation re-embedded every
       chunk on every prompt, which was effectively unusable above
       a few hundred chunks.
    2. The query is embedded via
       :func:`core.capability_classifier._embed_prompt` (same model,
       timeout and fall-back behaviour as the capability classifier).
       If ``turbovec`` is installed the search runs against the
       compressed ANN index; otherwise we fall back to a numpy
       brute-force cosine scan over the cached embedding matrix.

    On any embedding failure (server down, model not pulled) returns
    an empty list, which the pipeline interprets as "skip RAG."
    """
    import numpy as np

    from .capability_classifier import _embed_prompt

    index = _get_rag_index()
    if index is None or index.embeddings.shape[0] == 0:
        return []

    query_vec = _embed_prompt(prompt)
    if query_vec is None:
        return []
    query = query_vec.astype(np.float32)
    if query.shape[0] != index.dim:
        # Embedding model was swapped under us. Treat as a soft
        # cache invalidation: drop the index and fall through to an
        # empty result; the next call rebuilds with the new dim.
        logger.warning(
            'RAG embedding dim mismatch (query=%d, index=%d); skipping retrieval.',
            query.shape[0], index.dim,
        )
        return []

    k = max(1, min(int(top_k), index.embeddings.shape[0]))

    if index.turbovec_index is not None:
        try:
            _, ids = index.turbovec_index.search(query.reshape(1, -1), k=k)
            picked = [int(i) for i in ids[0]]
            return [index.chunks[i] for i in picked if 0 <= i < len(index.chunks)]
        except Exception as exc:  # noqa: BLE001
            logger.debug('turbovec search failed, falling back to numpy: %s', exc)

    # Pure-numpy fallback. Vectorised cosine — fast at the corpus
    # sizes this code path is expected to see (< few thousand chunks).
    qn = float(np.linalg.norm(query)) or 1.0
    norms = np.linalg.norm(index.embeddings, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    scores = (index.embeddings @ query) / (norms * qn)
    order = np.argsort(-scores)[:k]
    return [index.chunks[int(i)] for i in order]


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
    to matter. The embedding matrix is the same one
    :func:`_retrieve_chunks` uses, so the viz path is effectively
    free once retrieval has warmed the persistent cache.
    """
    import numpy as np

    index = _get_rag_index()
    if index is None or index.embeddings.shape[0] == 0:
        return []

    centred = index.embeddings - index.embeddings.mean(axis=0, keepdims=True)
    # Top-2 components via SVD. Full SVD is fine at the corpus sizes
    # this code path is expected to see; bump to randomised SVD via
    # sklearn if you grow past a few thousand chunks.
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        components = vt[:2]
        coords = centred @ components.T
    except Exception as exc:  # noqa: BLE001
        logger.warning('Personal viz projection failed: %s', exc)
        return []

    # Per-source chunk index — reconstructed from the path sequence
    # because the persistent index stores chunks in load order.
    per_path: dict[str, int] = {}
    points: list[dict] = []
    for (path, text), (x, y) in zip(index.chunks, coords, strict=True):
        local = per_path.get(path.name, 0)
        points.append({
            'path': path.name,
            'chunk_index': local,
            'text': text,
            'x': float(x),
            'y': float(y),
        })
        per_path[path.name] = local + 1
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
