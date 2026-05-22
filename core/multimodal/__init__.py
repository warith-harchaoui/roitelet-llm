"""Roitelet multimodal helpers.

One module per modality. Each exposes a single high-level coroutine that
takes a file path and returns a plain string ready to be spliced into the
user prompt — so the rest of the pipeline (router, candidates, judge)
remains text-only.
"""
