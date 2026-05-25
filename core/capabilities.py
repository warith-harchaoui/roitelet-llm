"""Prompt capability detection for Roitelet routing.

The first version intentionally favors transparency over sophistication.
It maps a prompt to multiple weighted capabilities using simple lexical cues,
which are then combined with benchmark priors and rolling Elo updates.

Examples
--------
>>> from core.capabilities import detect_capabilities
>>> scores = detect_capabilities("Write Python code for a fast API")
>>> scores['coding'] > 0
True

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List

CAPABILITY_KEYWORDS: Dict[str, List[str]] = {
    'coding': [
        'code', 'python', 'bug', 'debug', 'api', 'sql', 'javascript', 'algorithm',
        'function', 'class', 'typescript', 'rust', 'golang', 'bash', 'shell',
        'script', 'test', 'unittest', 'regex', 'refactor', 'optimize', 'implement',
        'docker', 'kubernetes', 'git', 'library', 'package', 'module', 'import',
    ],
    'math': [
        'math', 'proof', 'equation', 'integral', 'derivative', 'matrix', 'probability',
        'statistics', 'algebra', 'geometry', 'calculus', 'theorem', 'vector', 'tensor',
        'optimization', 'linear', 'quadratic', 'solve', 'compute', 'calculate', 'formula',
    ],
    'reasoning': [
        'why', 'compare', 'tradeoff', 'strategy', 'plan', 'reason', 'argue',
        'explain', 'analyze', 'evaluate', 'critique', 'debate', 'pros', 'cons',
        'decision', 'should', 'recommend', 'choose', 'best', 'tradeoffs',
    ],
    'writing': [
        'write', 'rewrite', 'email', 'story', 'tone', 'copywriting',
        'essay', 'blog', 'draft', 'edit', 'proofread', 'cover', 'letter',
        'narrative', 'summarize', 'paraphrase', 'article', 'report',
    ],
    'analysis': [
        'analyze', 'analysis', 'summarize', 'table', 'csv', 'data',
        'dataset', 'chart', 'trend', 'pattern', 'insight', 'statistic',
        'breakdown', 'compare', 'metrics', 'benchmark', 'evaluate',
    ],
    'vision': ['image', 'photo', 'diagram', 'screenshot', 'chart', 'figure', 'picture', 'visual'],
    # Reserved for the planned image-generation extension. Today no
    # model in the bootstrap exposes a non-zero prior on this
    # capability, so the router's existing scoring still ranks text
    # candidates first — the keywords just make image-y prompts
    # legible in telemetry so we can measure demand. See
    # ``.private/IMAGEGEN.md``.
    'image_gen': [
        'generate image', 'generate an image', 'draw', 'paint',
        'illustration', 'picture of', 'image of', 'render',
        'create artwork', 'design a logo', 'sketch', 'photorealistic',
    ],
    'multilingual': [
        'translate', 'french', 'english', 'spanish', 'german', 'japanese',
        'chinese', 'arabic', 'portuguese', 'italian', 'korean', 'russian',
        'language', 'traduction', 'übersetzung',
    ],
    'long_context': ['long document', 'full report', 'entire codebase', 'large context', 'book', 'paper'],
}


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word-like units."""
    return re.findall(r"[a-zA-Z0-9_\-']+", text.lower())


def detect_capabilities(prompt: str) -> Dict[str, float]:
    """Infer prompt capabilities from a user request.

    Parameters
    ----------
    prompt:
        Prompt or question supplied by the user.

    Returns
    -------
    dict of str to float
        Normalized capability weights summing to one.
    """
    tokens = _tokenize(prompt)
    joined = ' '.join(tokens)
    raw_scores: Dict[str, float] = defaultdict(float)

    # Start with a small prior on generic reasoning so that every prompt gets a
    # meaningful distribution even when lexical matching is sparse.
    raw_scores['reasoning'] = 0.25

    for capability, keywords in CAPABILITY_KEYWORDS.items():
        for keyword in keywords:
            if ' ' in keyword:
                if keyword in joined:
                    raw_scores[capability] += 1.0
            else:
                raw_scores[capability] += tokens.count(keyword)

    if prompt.count('```') >= 1 or 'def ' in prompt or 'class ' in prompt:
        raw_scores['coding'] += 2.0
    if any(symbol in prompt for symbol in ['∑', '∫', '√', 'λ', '=']):
        raw_scores['math'] += 1.0
    # 4000 characters ≈ 700 tokens — a meaningful long-context signal.
    if len(prompt) > 4000:
        raw_scores['long_context'] += 1.0

    total = sum(raw_scores.values()) or 1.0
    return {name: value / total for name, value in raw_scores.items() if value > 0}


def top_capabilities(capabilities: Dict[str, float], limit: int = 3) -> List[str]:
    """Return the dominant capabilities sorted by decreasing weight.

    Parameters
    ----------
    capabilities:
        Weighted capability distribution.
    limit:
        Maximum number of capability names returned.

    Returns
    -------
    list of str
        Sorted capability names.
    """
    return [name for name, _ in sorted(capabilities.items(), key=lambda item: item[1], reverse=True)[:limit]]
