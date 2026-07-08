#!/usr/bin/env python3
"""Merge and split the Swahili math datasets into train/val JSONL for SFT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append({"messages": record["messages"]})
    return examples


def deduplicate(examples: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for example in examples:
        key = json.dumps(example["messages"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetune", type=Path, default=Path("data/finetune/all_finetune.jsonl"))
    parser.add_argument("--qa", type=Path, default=Path("data/qa/all_qa.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    examples = load_examples(args.finetune) + load_examples(args.qa)
    examples = deduplicate(examples)

    random.Random(args.seed).shuffle(examples)
    val_size = max(1, int(len(examples) * args.val_fraction))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    write_jsonl(args.out_dir / "train.jsonl", train_examples)
    write_jsonl(args.out_dir / "val.jsonl", val_examples)

    print(f"merged unique examples: {len(examples)}")
    print(f"train: {len(train_examples)} examples")
    print(f"val: {len(val_examples)} examples")


if __name__ == "__main__":
    main()
