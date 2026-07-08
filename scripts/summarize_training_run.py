#!/usr/bin/env python3
"""Create easy-to-track metric files from a trainer_state.json output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    trainer_state_path = run_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing {trainer_state_path}")

    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history = trainer_state.get("log_history", [])

    rows = []
    for entry in log_history:
        rows.append(
            {
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "loss": entry.get("loss"),
                "eval_loss": entry.get("eval_loss"),
                "learning_rate": entry.get("learning_rate"),
                "mean_token_accuracy": entry.get("mean_token_accuracy"),
                "eval_mean_token_accuracy": entry.get("eval_mean_token_accuracy"),
            }
        )

    csv_path = run_dir / "metrics_by_step.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "best_global_step": trainer_state.get("best_global_step"),
        "best_metric": trainer_state.get("best_metric"),
        "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
        "global_step": trainer_state.get("global_step"),
        "max_steps": trainer_state.get("max_steps"),
    }
    summary_path = run_dir / "tracking_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
