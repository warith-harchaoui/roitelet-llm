r"""One-shot audio transcription + diarization for Roitelet.

Adapted from operator-assistance — its streaming/VAD/accumulator machinery
is stripped out because Roitelet only ever sees one complete file at a time.

Pipeline (single coroutine, ``transcribe_audio``):

1. ``audio_helper.load_audio`` → 16 kHz mono float32 numpy array.
2. whisper.cpp via ``pywhispercpp`` → list of timestamped segments.
3. NeMo ``SortformerEncLabelModel`` → list of ``(speaker, start, end)``
   diarization segments. Run concurrently with (2).
4. Overlap-based speaker assignment per whisper segment.
5. Collapse consecutive same-speaker lines, format as
   ``"[SPEAKER_00] text\n[SPEAKER_01] text"``.

The whisper and NeMo models are lazy-loaded and cached at module scope —
the first request after server boot pays ~1 GB of weights download plus
model init, every subsequent request reuses them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Defaults tuned for Apple Silicon laptops — good quality/speed tradeoff.
_WHISPER_MODEL = os.environ.get('ROITELET_WHISPER_MODEL', 'medium')
_WHISPER_THREADS = int(os.environ.get('ROITELET_WHISPER_THREADS', '4'))
_NEMO_MODEL = os.environ.get('ROITELET_NEMO_MODEL', 'nvidia/diar_sortformer_4spk-v1')
_NEMO_CHUNK_S = float(os.environ.get('ROITELET_NEMO_CHUNK_S', '300'))
_TARGET_SR = 16_000


# ── NumPy 2.x compatibility shim — NeMo references ``np.sctypes`` ──────────
if not hasattr(np, 'sctypes'):
    np.sctypes = {  # type: ignore[attr-defined]
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }


# ──────────────────────────────────────────────────────────────────────────
# Lazy model loaders
# ──────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _whisper_model() -> Any:
    """Load whisper.cpp once and cache. Auto-downloads ggml weights."""
    from pywhispercpp.model import Model  # type: ignore[import-not-found]
    logger.info('Loading whisper.cpp model: %s', _WHISPER_MODEL)
    return Model(
        model=_WHISPER_MODEL,
        n_threads=_WHISPER_THREADS,
        print_realtime=False,
        print_progress=False,
        print_timestamps=False,
    )


@lru_cache(maxsize=1)
def _nemo_model() -> Any:
    """Load NeMo Sortformer once and cache. Auto-downloads weights from HF."""
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel  # type: ignore[import-not-found]
        cls = SortformerEncLabelModel
    except ImportError:
        from nemo.collections.asr.models import EncDecDiarLabelModel  # type: ignore[import-not-found]
        cls = EncDecDiarLabelModel
    logger.info('Loading NeMo diarization model: %s (%s)', _NEMO_MODEL, cls.__name__)
    model = cls.from_pretrained(model_name=_NEMO_MODEL)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────────────────────────────────


def _load_audio(path: Path) -> np.ndarray:
    """Load any common audio format as 16 kHz mono float32 numpy."""
    import audio_helper as ah  # type: ignore[import-not-found]
    data, _ = ah.load_audio(str(path), target_sample_rate=_TARGET_SR, to_numpy=True)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return data


# ──────────────────────────────────────────────────────────────────────────
# Whisper.cpp transcription with timestamps
# ──────────────────────────────────────────────────────────────────────────


def _whisper_segments(samples: np.ndarray, language: str | None = None) -> list[tuple[float, float, str]]:
    """Transcribe with timestamps. Returns ``[(start_s, end_s, text), ...]``."""
    model = _whisper_model()
    params: dict[str, Any] = {
        'translate': False,
        'n_threads': _WHISPER_THREADS,
        'print_realtime': False,
        'print_progress': False,
        'print_timestamps': False,
        'single_segment': False,
    }
    if language:
        params['language'] = language
    segments = model.transcribe(samples, **params)
    # pywhispercpp ``Segment.t0`` / ``t1`` are in centiseconds.
    return [
        (seg.t0 / 100.0, seg.t1 / 100.0, seg.text.strip())
        for seg in (segments or [])
        if seg.text.strip()
    ]


# ──────────────────────────────────────────────────────────────────────────
# NeMo diarization (one-shot, with optional chunking for long files)
# ──────────────────────────────────────────────────────────────────────────


def _diarize(path: Path, audio: np.ndarray) -> list[dict]:
    """Return list of ``{"speaker", "start", "end"}`` for the whole file.

    Long files (> ``_NEMO_CHUNK_S``) are split to avoid GPU/MPS OOM. Each
    chunk is written to a temporary WAV — NeMo's diarize() expects file
    paths, not raw tensors.
    """
    import soundfile as sf  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    model = _nemo_model()
    total = len(audio)
    chunk = int(_NEMO_CHUNK_S * _TARGET_SR)
    segments: list[dict] = []

    # Fast path: file fits in one chunk → diarize directly from disk.
    if total <= chunk:
        with torch.no_grad():
            predicted = model.diarize(audio=str(path), batch_size=1)
        segments.extend(_parse_nemo_output(predicted, 0.0))
        return segments

    # Chunked path for long files.
    for start_idx in range(0, total, chunk):
        piece = audio[start_idx:start_idx + chunk]
        offset = start_idx / _TARGET_SR
        fd, tmp_path = tempfile.mkstemp(suffix='.wav')
        try:
            os.close(fd)
            sf.write(tmp_path, piece, _TARGET_SR)
            with torch.no_grad():
                predicted = model.diarize(audio=tmp_path, batch_size=1)
            segments.extend(_parse_nemo_output(predicted, offset))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return segments


def _normalize_speaker(raw: str) -> str:
    """Coerce any NeMo speaker label variant to ``SPEAKER_NN`` zero-padded."""
    import re
    match = re.search(r'(\d+)$', raw)
    if match:
        return f'SPEAKER_{int(match.group(1)):02d}'
    return raw


def _parse_nemo_output(predicted: Any, time_offset: float) -> list[dict]:
    """Normalize NeMo's varying return shapes into ``{speaker,start,end}`` dicts.

    NeMo's ``diarize()`` returns RTTM strings, dicts keyed by filename, lists
    of named tuples, or nested lists depending on version and config. This
    parser handles the union — borrowed directly from operator-assistance
    where it was developed against three NeMo releases.
    """
    if not predicted:
        return []
    if isinstance(predicted, dict):
        out: list[dict] = []
        for value in predicted.values():
            if isinstance(value, (list, tuple)):
                out.extend(_parse_items(value, time_offset))
        return out
    items: Any = predicted
    if isinstance(predicted, (list, tuple)) and predicted:
        if isinstance(predicted[0], (list, tuple)):
            items = predicted[0]
        elif isinstance(predicted[0], dict) and len(predicted) == 1:
            return _parse_nemo_output(predicted[0], time_offset)
    return _parse_items(items, time_offset)


def _parse_items(items: Any, time_offset: float) -> list[dict]:
    """Parse one flat batch of NeMo diarization items."""
    out: list[dict] = []
    for item in items:
        if isinstance(item, str):
            parts = item.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                # RTTM: SPEAKER <file> 1 <start> <dur> <NA> <NA> <speaker>
                start = float(parts[3]) + time_offset
                dur = float(parts[4])
                out.append({
                    'speaker': _normalize_speaker(parts[7]),
                    'start': round(start, 3),
                    'end': round(start + dur, 3),
                })
            elif len(parts) == 3:
                # Sortformer text form: "start end speaker".
                out.append({
                    'speaker': _normalize_speaker(parts[2]),
                    'start': round(float(parts[0]) + time_offset, 3),
                    'end': round(float(parts[1]) + time_offset, 3),
                })
        elif isinstance(item, dict):
            out.append({
                'speaker': _normalize_speaker(str(item.get('speaker', 'unk'))),
                'start': round(float(item.get('start', 0.0)) + time_offset, 3),
                'end': round(float(item.get('end', 0.0)) + time_offset, 3),
            })
        else:
            speaker = getattr(item, 'speaker', getattr(item, 'label', 'unk'))
            out.append({
                'speaker': _normalize_speaker(str(speaker)),
                'start': round(float(getattr(item, 'start', 0.0)) + time_offset, 3),
                'end': round(float(getattr(item, 'end', 0.0)) + time_offset, 3),
            })
    return out


# ──────────────────────────────────────────────────────────────────────────
# Alignment: whisper segment → dominant NeMo speaker
# ──────────────────────────────────────────────────────────────────────────


def _dominant_speaker(start: float, end: float, diar: list[dict]) -> str:
    """Pick the speaker with the most cumulative overlap on ``[start, end]``."""
    if not diar:
        return 'SPEAKER_00'
    totals: dict[str, float] = {}
    for seg in diar:
        overlap = max(0.0, min(end, seg['end']) - max(start, seg['start']))
        if overlap > 0:
            totals[seg['speaker']] = totals.get(seg['speaker'], 0.0) + overlap
    if not totals:
        return 'SPEAKER_00'
    return max(totals, key=totals.get)  # type: ignore[arg-type]


def _format_transcript(
    whisper_segs: list[tuple[float, float, str]],
    diar_segs: list[dict],
) -> str:
    """Join transcript lines, collapsing consecutive turns by same speaker."""
    if not whisper_segs:
        return ''
    lines: list[tuple[str, list[str]]] = []
    for start, end, text in whisper_segs:
        speaker = _dominant_speaker(start, end, diar_segs)
        if lines and lines[-1][0] == speaker:
            lines[-1][1].append(text)
        else:
            lines.append((speaker, [text]))
    return '\n'.join(f'[{speaker}] {" ".join(parts)}' for speaker, parts in lines)


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


async def transcribe_audio(path: Path, language: str | None = None) -> str:
    """Transcribe + diarize one audio file. Returns formatted transcript.

    Parameters
    ----------
    path:
        Path to any audio format ``audio_helper`` can read (WAV, MP3, M4A,
        FLAC, OGG …).
    language:
        Optional BCP-47 language hint forwarded to whisper.cpp. When
        ``None`` whisper auto-detects.

    Returns
    -------
    str
        Lines of the form ``"[SPEAKER_00] text"``, one per consecutive
        speaker turn. Empty string if the file is silent.
    """
    audio = await asyncio.to_thread(_load_audio, path)
    if audio.size == 0:
        return ''
    whisper_task = asyncio.to_thread(_whisper_segments, audio, language)
    diar_task = asyncio.to_thread(_diarize, path, audio)
    whisper_segs, diar_segs = await asyncio.gather(whisper_task, diar_task)
    return _format_transcript(whisper_segs, diar_segs)
