#!/usr/bin/env bash
# Pull a Roitelet OSS model bundle into local Ollama.
#
# Two profiles:
#
#   --minimal   One small judge + the embedding model (~3 GB total).
#               The fastest onboarding path: Roitelet works in
#               local-only mode and the embedding-based capability
#               detector + personal-mode RAG are also functional.
#               Fan-out has only one local candidate, so fusion adds
#               little — pair this profile with optional remote
#               candidates via an API key.
#
#   (default)   Full OSS bundle (~15 GB). One capable model from each
#               major OSS family (Qwen, Llama, Gemma, Phi) plus a
#               vision-language model. Designed for cross-family
#               fusion: K=3 over different model families typically
#               yields more diverse candidate answers than three
#               flavours of one family.
#
# See docs/INSTALL.md / README "Minimal mode" for guidance on which
# profile to pick.

set -euo pipefail

PROFILE="full"
for arg in "$@"; do
  case "$arg" in
    --minimal) PROFILE="minimal" ;;
    --full)    PROFILE="full" ;;
    -h|--help)
      cat <<EOF
Usage: scripts/pull_defaults.sh [--minimal | --full]

  --minimal   Pull only qwen3:4b (small synthesis judge) and
              nomic-embed-text (for the embedding-based detector +
              personal-mode RAG). ~3 GB total.
  --full      Pull the full cross-family OSS bundle (default). ~15 GB.

Env:
  LOCAL_LLM_BASE_URL    Ollama base URL (default http://localhost:11434).
EOF
      exit 0
      ;;
  esac
done

OLLAMA_HOST="${LOCAL_LLM_BASE_URL:-http://localhost:11434}"

if [[ "$PROFILE" == "minimal" ]]; then
  # Smallest path to a working local-only Roitelet:
  # - qwen3:4b is the synthesis judge and the only fan-out candidate.
  # - nomic-embed-text unlocks the embedding-based capability detector
  #   and personal-mode RAG retrieval.
  MODELS=(
    "qwen3:4b"
    "nomic-embed-text"
  )
  BUNDLE_LABEL="minimal (~3 GB)"
else
  # One model per major OSS family + one dedicated VLM.
  MODELS=(
    # Alibaba — default synthesis judge (multilingual, instruction-tuned)
    "qwen3:8b"
    # Alibaba — lighter alternative for 8–16 GB machines
    "qwen3:4b"
    # Meta — small + fast generalist; the routing pool wants a Llama vote
    "llama3.2:3b"
    # Google — multimodal-native; covers text + light vision in one model
    "gemma3:4b"
    # Microsoft — reasoning-tuned small model; strong on math
    "phi4-mini:3.8b"
    # Alibaba VL — primary vision-language model for image prompts
    "qwen2.5vl:7b"
    # Embedding model for capability detector + personal-mode RAG
    "nomic-embed-text"
  )
  BUNDLE_LABEL="full (~15 GB)"
fi

echo "Roitelet — pulling the ${BUNDLE_LABEL} bundle from Ollama at ${OLLAMA_HOST}"
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

if [[ "$PROFILE" == "minimal" ]]; then
  cat <<'EOF'

Next steps for the minimal profile:

  - Run start.sh and open http://localhost:8000/.
  - With one local model, fan-out is K=1 (no fusion benefit). Either:
      * stay minimal and use Roitelet as a local-only chat client, or
      * add an OPENROUTER_API_KEY / OPENAI_API_KEY to .env to bring
        remote candidates into the routing pool. Fusion then runs
        with the local qwen3:4b judge over remote candidate answers.
  - To upgrade to the full bundle later:  scripts/pull_defaults.sh --full
EOF
fi
