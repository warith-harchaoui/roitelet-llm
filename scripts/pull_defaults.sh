#!/usr/bin/env bash
# Pull the Roitelet OSS default model bundle into local Ollama.
#
# Roitelet doesn't pick a "best" local model — it queries K models in
# parallel and fuses their answers. Cross-family diversity therefore
# matters more than raw capability: a fusion of Qwen + Llama + Gemma +
# Phi outputs is reliably stronger than three flavours of one family.
# This script ensures one capable model from each major OSS family is
# available even before the user touches any paid API key.

set -euo pipefail

OLLAMA_HOST="${LOCAL_LLM_BASE_URL:-http://localhost:11434}"

# One model per major OSS family + one dedicated VLM.
MODELS=(
  # Alibaba — Roitelet's default synthesis judge (multilingual, instruction-tuned)
  "qwen3:8b-instruct"
  # Alibaba — lighter alternative for 8–16 GB machines
  "qwen3:4b-instruct"
  # Meta — small + fast generalist; the routing pool wants a Llama-family vote
  "llama3.2:3b-instruct"
  # Google — multimodal-native; covers text + light vision in one model
  "gemma3:4b"
  # Microsoft — reasoning-tuned small model; punches above its weight on math
  "phi4-mini:3.8b"
  # Alibaba VL — primary vision-language model for image prompts
  "qwen2.5-vl:7b"
)

echo "Roitelet — pulling the OSS default bundle from Ollama at ${OLLAMA_HOST}"
echo

# Probe Ollama. A wrong/unreachable host produces N opaque pull failures;
# checking once up-front gives a single clear error instead.
if ! curl -fsS "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
  echo "Error: Ollama is not reachable at ${OLLAMA_HOST}" >&2
  echo "Start it with:  ollama serve" >&2
  echo "Or set LOCAL_LLM_BASE_URL to the right host:port." >&2
  exit 1
fi

# Pull each model. `ollama pull` is idempotent and shows progress per layer.
for model in "${MODELS[@]}"; do
  echo "──────────────────────────────────────────────────────────"
  echo "Pulling: ${model}"
  echo "──────────────────────────────────────────────────────────"
  OLLAMA_HOST="${OLLAMA_HOST}" ollama pull "${model}"
  echo
done

echo "Done. Models now available locally:"
printf '  - %s\n' "${MODELS[@]}"
echo
echo "Roitelet will discover them automatically via /api/tags within 60 s."
echo "To verify:  curl ${OLLAMA_HOST}/api/tags | jq '.models[].name'"
