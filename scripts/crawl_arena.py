#!/usr/bin/env python3
"""Autonomous Elo / Capabilities Extractor for Roitelet LLM.

This script fetches raw leaderboard text (or HTML) from a given URL, uses the local
Ollama synthesis model to parse the unstructured data into a structured JSON
model capability map, normalizes Elo scores onto the Roitelet 0.0 -> 1.5 scale,
and merges the new scores back into `data/bootstrap/model_priors.json`.

Usage:
    python scripts/crawl_arena.py https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
    python scripts/crawl_arena.py dump.txt
"""

import argparse
import datetime
import html.parser
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants for normalization
ELO_MIN = 1000.0
ELO_MAX = 1300.0
ROITELET_MAX = 1.5

# System prompt for the local LLM
EXTRACTION_PROMPT = """You are an autonomous data extraction agent for the Roitelet LLM routing engine.
Your goal is to read the raw leaderboard text below and extract the Large Language Models and their Elo ratings.

Rules:
1. Extract the model identifier (e.g. "gpt-4o", "claude-3-opus", "llama-3-70b-instruct"). Try to use standard, recognizable ID formats.
2. Extract the overall Elo rating (usually a number between 1000 and 1300).
3. Ignore image generation models or irrelevant tabular data.
4. Output EXACTLY a valid JSON array of objects, with no markdown, no backticks, and no explanation text.

Format:
[
  {"model": "gpt-4o", "elo": 1287},
  {"model": "claude-3-opus", "elo": 1253}
]

Raw Text:
"""


class HTMLStripper(html.parser.HTMLParser):
    """Simple parser to strip HTML tags from a webpage."""

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text: List[str] = []

    def handle_data(self, d: str) -> None:
        stripped = d.strip()
        if stripped:
            self.text.append(stripped)

    def get_data(self) -> str:
        return ' '.join(self.text)


def fetch_content(source: str) -> str:
    """Read content from a URL or local file, stripping HTML if necessary."""
    if source.startswith("http://") or source.startswith("https://"):
        logger.info(f"Fetching URL: {source}")
        req = urllib.request.Request(source, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8")
        if "<html" in raw.lower() or "<body" in raw.lower():
            logger.info("Stripping HTML tags...")
            stripper = HTMLStripper()
            stripper.feed(raw)
            return stripper.get_data()
        return raw
    else:
        logger.info(f"Reading file: {source}")
        return Path(source).read_text(encoding="utf-8")


def call_local_llm(text: str, base_url: str = "http://localhost:11434", model: str = "qwen3:8b") -> List[Dict[str, Any]]:
    """Send the text to the local Ollama instance for JSON extraction."""
    logger.info(f"Sending raw text to local LLM ({model}) for extraction...")
    
    # We clip the text to prevent context limits (first 10k chars usually container the top leaderboard)
    clipped_text = text[:15000]

    payload = {
        "model": model,
        "prompt": EXTRACTION_PROMPT + clipped_text,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0
        }
    }

    try:
        resp = httpx.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        raw_output = resp.json().get("response", "[]").strip()
        
        # Guard against minor markdown wrapping if the LLM ignores instructions
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
            
        data = json.loads(raw_output)
        if not isinstance(data, list):
            raise ValueError("LLM did not return a JSON array.")
        return data
    except Exception as e:
        logger.error(f"Failed to extract JSON from LLM: {e}")
        sys.exit(1)


def normalize_elo(elo: float) -> float:
    """Map an Elo score (e.g. 1000 - 1300) onto the 0.0 -> 1.5 Roitelet scale."""
    if elo <= ELO_MIN:
        return 0.5
    if elo >= ELO_MAX:
        return ROITELET_MAX
    # Linear interpolation
    fraction = (elo - ELO_MIN) / (ELO_MAX - ELO_MIN)
    return 0.5 + (fraction * (ROITELET_MAX - 0.5))


def update_priors(extracted: List[Dict[str, Any]], source: Optional[str] = None) -> None:
    """Merge normalized Elo scores into model_priors.json with provenance.

    Each touched entry receives a ``_meta`` block recording the raw Elo,
    the source URL/file, and a UTC timestamp — so the next reader knows
    when the prior last moved and why.
    """
    priors_path = Path(__file__).parent.parent / "data" / "bootstrap" / "model_priors.json"
    if not priors_path.exists():
        logger.error(f"Priors file not found at {priors_path}")
        sys.exit(1)

    logger.info("Loading existing priors...")
    priors = json.loads(priors_path.read_text(encoding="utf-8"))

    update_count = 0
    # Create simple mapping of existing names to IDs for easier matching
    provider_map = {k.split("/")[-1].lower(): k for k in priors.keys()}
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for entry in extracted:
        model_name = str(entry.get("model", "")).lower()
        raw_elo = float(entry.get("elo", 0))
        if not model_name or raw_elo == 0:
            continue

        normalized = normalize_elo(raw_elo)

        # Try to match the exact model ID or the model suffix
        target_id = None
        for known_name, full_id in provider_map.items():
            if known_name in model_name or model_name in known_name:
                target_id = full_id
                break

        if target_id is not None:
            old_reasoning = priors[target_id]["capabilities"].get("reasoning", 0.0)
            # Smooth the update (50% old, 50% new leaderboard score)
            new_reasoning = round((old_reasoning + normalized) / 2.0, 2)

            priors[target_id]["capabilities"]["reasoning"] = new_reasoning
            priors[target_id]["capabilities"]["analysis"] = new_reasoning
            priors[target_id]["_meta"] = {
                "source": source or "manual",
                "elo_raw": raw_elo,
                "elo_normalized": normalized,
                "refreshed_at": now_iso,
            }

            logger.info(f"Updated {target_id}: reasoning {old_reasoning} -> {new_reasoning} (Raw Elo: {raw_elo})")
            update_count += 1
        else:
            logger.debug(f"Skipped {model_name} (Not registered in model_priors.json).")

    if update_count > 0:
        logger.info(f"Successfully updated {update_count} model priors. Writing to disk...")
        priors_path.write_text(json.dumps(priors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        logger.info("No matching models were updated. Check the extracted model names.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Elo Extractor")
    parser.add_argument("source", help="URL or local text file containing leaderboard data.")
    parser.add_argument("--model", default="qwen3:8b", help="Local Ollama model to use for extraction.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama Base URL.")
    args = parser.parse_args()

    raw_text = fetch_content(args.source)
    logger.info(f"Fetched {len(raw_text)} characters. Proceeding to extraction.")

    extracted_data = call_local_llm(raw_text, base_url=args.ollama_url, model=args.model)
    logger.info(f"LLM successfully extracted {len(extracted_data)} model scores.")

    update_priors(extracted_data, source=args.source)
    logger.info("Arena update cycle complete.")


if __name__ == "__main__":
    main()
