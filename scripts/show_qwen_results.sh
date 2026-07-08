#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${1:-outputs/qwen25-1p5b-hisabati-hf-full}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR"
  exit 1
fi

export RUN_DIR
python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
tracking = json.loads((run_dir / "tracking_summary.json").read_text(encoding="utf-8"))

print("=== Qwen Run Summary ===")
print(f"run_dir: {run_dir}")
print(f"train_loss: {summary['train_metrics'].get('train_loss')}")
print(f"eval_loss: {summary['eval_metrics'].get('eval_loss')}")
print(f"eval_mean_token_accuracy: {summary['eval_metrics'].get('eval_mean_token_accuracy')}")
print(f"best_metric: {tracking.get('best_metric')}")
print(f"best_checkpoint: {tracking.get('best_model_checkpoint')}")

metrics_csv = run_dir / "metrics_by_step.csv"
if metrics_csv.exists():
    with metrics_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows:
        last = rows[-1]
        print("\n=== Last Metrics Row ===")
        print(last)

results_dir = Path("testing/results")
if results_dir.exists():
    qwen_files = sorted(results_dir.glob("qwen*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if qwen_files:
        print(f"\nlatest_test_outputs: {qwen_files[0]}")
PY
