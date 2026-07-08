#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Create it and install dependencies first."
  exit 1
fi

OUT_DIR="${OUT_DIR:-outputs/qwen25-1p5b-hisabati-hf-full}"
BASE_MODEL="${BASE_MODEL:-unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit}"
PROMPTS_FILE="${PROMPTS_FILE:-testing/prompts_smoke.jsonl}"
OFFLINE="${OFFLINE:-1}"
EPOCHS="${EPOCHS:-3}"

echo "Preparing SFT dataset..."
.venv/bin/python scripts/prepare_sft_dataset.py --out-dir data/sft

if [[ "$OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  EXTRA_ARGS=(--offline --local-files-only)
else
  EXTRA_ARGS=()
fi

echo "Training Qwen model..."
.venv/bin/python scripts/train_qwen_lora_hf.py \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --eval-strategy steps \
  --eval-steps 50 \
  --save-strategy steps \
  --save-steps 50 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 8 \
  "${EXTRA_ARGS[@]}"

echo "Summarizing metrics..."
.venv/bin/python scripts/summarize_training_run.py --run-dir "$OUT_DIR"

echo "Running post-train test prompts..."
.venv/bin/python scripts/test_model_outputs.py \
  --model-path "$OUT_DIR" \
  --engine hf \
  --prompts-file "$PROMPTS_FILE" \
  --temperature 0 \
  "${EXTRA_ARGS[@]}"

echo "Qwen full run complete."
echo "Run directory: $OUT_DIR"
