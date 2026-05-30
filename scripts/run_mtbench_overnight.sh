#!/usr/bin/env bash
#
# Overnight runner for the full MT-Bench eval.
#
# Pre-flights every dependency the run needs (Ollama, models, disk
# space, HuggingFace dataset cache, Python venv), then launches the
# benchmark in the background under ``nohup`` with a timestamped log.
#
# Designed for "kick it off, close the laptop, come back in the
# morning" — every checkpoint is on disk by the time the process
# exits, and partial progress survives a kill.
#
# Usage:
#
#   bash scripts/run_mtbench_overnight.sh           # full 80 prompts
#   MTBENCH_LIMIT=16 bash scripts/run_mtbench_overnight.sh  # smaller slice
#
# After launch the script prints the log path and the report path.
# Tail the log to watch progress:
#
#   tail -f .private/eval_runs/mtbench-<timestamp>.log
#
# Or inspect the partial JSON at any moment:
#
#   cat .private/eval_runs/mtbench-*.json | jq '.meta'

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPORTS_DIR=".private/eval_runs"
mkdir -p "$REPORTS_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$REPORTS_DIR/mtbench-overnight-$STAMP.log"
PIDFILE="$REPORTS_DIR/mtbench-overnight-$STAMP.pid"

LIMIT="${MTBENCH_LIMIT:-0}"  # 0 = full set; integer = first N prompts.

# ── Pre-flight checks ────────────────────────────────────────────────

echo "==> Pre-flight checks"

# 1. Ollama reachable + the judge model is pulled.
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
if ! curl -fsS --max-time 3 "$OLLAMA_URL/api/tags" > /dev/null; then
  echo "FAIL: Ollama is not reachable at $OLLAMA_URL"
  echo "      Start it with: ollama serve   (or 'brew services start ollama' on macOS)"
  exit 1
fi
JUDGE_MODEL="${LOCAL_LLM_MODEL:-qwen3:8b}"
if ! curl -fsS "$OLLAMA_URL/api/tags" | grep -q "\"$JUDGE_MODEL\""; then
  echo "FAIL: Ollama does not have the judge model '$JUDGE_MODEL' pulled."
  echo "      Pull it with: ollama pull $JUDGE_MODEL"
  exit 1
fi
echo "  OK  Ollama up; judge model '$JUDGE_MODEL' available"

# 2. Eval extras installed.
if ! python -c "import deepeval, datasets" > /dev/null 2>&1; then
  echo "FAIL: DeepEval / datasets not importable."
  echo "      Install with: pip install -e '.[eval]'"
  exit 1
fi
echo "  OK  DeepEval + datasets importable"

# 3. Disk space. The dataset cache + reports + qwen3:8b's working set
#    fit in <2 GB, but call out anything below 5 GB free as a risk.
AVAIL_KB=$(df -k "$ROOT" | tail -1 | awk '{print $4}')
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
if [ "$AVAIL_GB" -lt 5 ]; then
  echo "WARN: only ${AVAIL_GB} GB free in $ROOT — check before launching"
fi
echo "  OK  ${AVAIL_GB} GB free in $ROOT"

# 4. Pre-warm the HuggingFace dataset cache so the first prompt isn't
#    blocked on a ~30s download. Skipped silently if HF is unreachable
#    (the test will hit it itself, just slower).
echo "==> Pre-warming MT-Bench dataset cache"
if python - <<'PYEOF' 2>/dev/null
import datasets, sys
try:
    ds = datasets.load_dataset('lmsys/mt_bench_human_judgments', split='human')
    print(f'  OK  {len(ds)} rows cached')
except Exception as exc:
    print(f'  WARN dataset prewarm failed ({exc}); test will retry')
    sys.exit(0)
PYEOF
then :; fi

# ── Launch ───────────────────────────────────────────────────────────

echo
echo "==> Launching MT-Bench full run"
echo "    MTBENCH_LIMIT=$LIMIT (0 means all 80 first-turn prompts)"
echo "    log:     $LOG"
echo "    report:  $REPORTS_DIR/mtbench-<unix-timestamp>.json (written incrementally)"
echo

# stdbuf -oL forces line buffering so ``tail -f`` shows progress in
# real time. Without it, Python buffers ~4 KB which can hide hours of
# progress.
nohup env MTBENCH_LIMIT="$LIMIT" PYTHONUNBUFFERED=1 \
  python -m pytest -m eval tests/eval/bench_mtbench.py -q -s \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
echo "Launched. PID=$PID"
echo
echo "Watch progress:    tail -f $LOG"
echo "Inspect partial:   ls -lt $REPORTS_DIR/mtbench-*.json | head -1"
echo "                   cat \$(ls -t $REPORTS_DIR/mtbench-*.json | head -1) | jq '.meta'"
echo "Stop the run:      kill \$(cat $PIDFILE)"
echo
echo "Estimated wall-clock: 3-5 hours on CPU with $JUDGE_MODEL."
