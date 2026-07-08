#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${1:-outputs/gemma3-4b-hisabati-unsloth-full}"
PID_FILE="$RUN_DIR/run.pid"
RUN_LOG="$RUN_DIR/run.log"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR"
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  RUN_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$RUN_PID" ]] && kill -0 "$RUN_PID" 2>/dev/null; then
    echo "Gemma training is still running (pid: $RUN_PID)."
    echo "Log file: $RUN_LOG"
    echo "Tip: tail -f \"$RUN_LOG\""
  fi
fi

export RUN_DIR
python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
summary_path = run_dir / "run_summary.json"
tracking_path = run_dir / "tracking_summary.json"

if not summary_path.exists() or not tracking_path.exists():
    print("Summary files are not ready yet.")
    raise SystemExit(0)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
tracking = json.loads(tracking_path.read_text(encoding="utf-8"))

print("=== Gemma Run Summary ===")
print(f"run_dir: {run_dir}")
print(f"train_loss: {summary['train_metrics'].get('train_loss')}")
print(f"eval_loss: {summary['eval_metrics'].get('eval_loss')}")
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
    gemma_files = sorted(results_dir.glob("checkpoint-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if gemma_files:
        print(f"\nlatest_test_outputs: {gemma_files[0]}")
PY
