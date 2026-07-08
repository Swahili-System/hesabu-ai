#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Create it and install dependencies first."
  exit 1
fi

OUT_DIR="${OUT_DIR:-outputs/gemma3-4b-hisabati-unsloth-full}"
BASE_MODEL="${BASE_MODEL:-unsloth/gemma-3-4b-it-unsloth-bnb-4bit}"
PROMPTS_FILE="${PROMPTS_FILE:-testing/prompts_smoke.jsonl}"
OFFLINE="${OFFLINE:-1}"
EPOCHS="${EPOCHS:-3}"
PERSIST="${PERSIST:-1}"
RUN_LOG="${RUN_LOG:-$OUT_DIR/run.log}"
PID_FILE="${PID_FILE:-$OUT_DIR/run.pid}"
export OUT_DIR

# Default mode is persistent background execution so the run
# survives terminal/session interruptions.
if [[ "${RUN_GEMMA_DETACHED:-0}" != "1" && "$PERSIST" == "1" ]]; then
  mkdir -p "$OUT_DIR"
  if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
      echo "Gemma run already active (pid: $OLD_PID)."
      echo "Log file: $RUN_LOG"
      exit 0
    fi
  fi

  nohup env RUN_GEMMA_DETACHED=1 PERSIST=0 OUT_DIR="$OUT_DIR" BASE_MODEL="$BASE_MODEL" \
    PROMPTS_FILE="$PROMPTS_FILE" OFFLINE="$OFFLINE" EPOCHS="$EPOCHS" RUN_LOG="$RUN_LOG" \
    PID_FILE="$PID_FILE" bash "$0" >"$RUN_LOG" 2>&1 &
  NEW_PID=$!
  echo "$NEW_PID" > "$PID_FILE"
  echo "Gemma run started in background (pid: $NEW_PID)."
  echo "Log file: $RUN_LOG"
  echo "Check status with: scripts/show_gemma_results.sh \"$OUT_DIR\""
  exit 0
fi

mkdir -p "$OUT_DIR"
echo "$$" > "$PID_FILE"
cleanup_pid() {
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup_pid EXIT

echo "Preparing SFT dataset..."
.venv/bin/python scripts/prepare_sft_dataset.py --out-dir data/sft

# Local header workaround for environments without system python3.12-dev.
LOCAL_INCLUDE_ROOT="$ROOT_DIR/.local-python-dev/usr/include"
if [[ -d "$LOCAL_INCLUDE_ROOT" ]]; then
  export C_INCLUDE_PATH="$LOCAL_INCLUDE_ROOT:$LOCAL_INCLUDE_ROOT/python3.12:$LOCAL_INCLUDE_ROOT/x86_64-linux-gnu/python3.12${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH="$LOCAL_INCLUDE_ROOT:$LOCAL_INCLUDE_ROOT/python3.12:$LOCAL_INCLUDE_ROOT/x86_64-linux-gnu/python3.12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi

if [[ "$OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

echo "Training Gemma model..."
.venv/bin/python scripts/train_gemma4_lora.py \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --eval-strategy steps \
  --eval-steps 50 \
  --save-strategy steps \
  --save-steps 50 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --resume-auto

echo "Summarizing metrics..."
.venv/bin/python scripts/summarize_training_run.py --run-dir "$OUT_DIR"

BEST_CKPT="$(python3 - <<'PY'
import json
import os
from pathlib import Path
run_dir = Path(os.environ["OUT_DIR"])
tracking = json.loads((run_dir / "tracking_summary.json").read_text(encoding="utf-8"))
print(tracking.get("best_model_checkpoint", str(run_dir / "checkpoint-193")))
PY
)"

echo "Running post-train test prompts on: $BEST_CKPT"
.venv/bin/python scripts/test_model_outputs.py \
  --model-path "$BEST_CKPT" \
  --engine unsloth \
  --prompts-file "$PROMPTS_FILE" \
  --offline \
  --merge-system-into-user \
  --temperature 0

echo "Gemma full run complete."
echo "Run directory: $OUT_DIR"
