"""Hardware detection helpers.

Examples
--------
>>> from core.device import detect_best_accelerator
>>> detect_best_accelerator() in {'mps', 'cuda', 'cpu'}
True

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

from importlib import util


def detect_best_accelerator() -> str:
    """Detect the best available accelerator.

    Returns
    -------
    str
        `'mps'`, `'cuda'`, or `'cpu'`.
    """
    if util.find_spec('torch') is None:
        return 'cpu'
    import torch  # type: ignore

    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
