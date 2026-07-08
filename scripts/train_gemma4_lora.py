#!/usr/bin/env python3
"""QLoRA fine-tune Gemma 4 on the Swahili primary math dataset (Unsloth + TRL).

Run on the GPU box after `prepare_sft_dataset.py` has produced
data/sft/train.jsonl and data/sft/val.jsonl.

Gemma's chat template has no "system" role, so the system prompt is merged
into the first user turn instead of passed as a separate message. Thinking
mode is left off (no `<|think|>` token) to match the dataset's direct,
non-chain-of-thought answers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="unsloth/gemma-4-12B-it")
    parser.add_argument("--train-file", default="data/sft/train.jsonl")
    parser.add_argument("--val-file", default="data/sft/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/sdt-flare-gm-12-hisabati")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--resume-auto", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def merge_system_into_user(messages: list[dict]) -> list[dict]:
    system_text = next((m["content"] for m in messages if m["role"] == "system"), None)
    merged = [m for m in messages if m["role"] != "system"]
    if system_text and merged and merged[0]["role"] == "user":
        merged[0] = {"role": "user", "content": f"{system_text}\n\n{merged[0]['content']}"}
    return merged


def render_chat_text(example: dict, tokenizer) -> dict:
    messages = merge_system_into_user(example["messages"])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def write_run_summary(output_dir: str, train_metrics: dict, eval_metrics: dict) -> None:
    summary_path = Path(output_dir) / "run_summary.json"
    payload = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training summary written to {summary_path}")


def find_latest_checkpoint(output_dir: str) -> str:
    base = Path(output_dir)
    if not base.exists():
        return ""
    checkpoints = []
    for path in base.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.replace("checkpoint-", "")
        if not suffix.isdigit():
            continue
        checkpoints.append((int(suffix), path))
    if not checkpoints:
        return ""
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    return str(checkpoints[0][1])


def main() -> None:
    args = parse_args()
    if "tensorboard" in str(args.report_to):
        os.environ["TENSORBOARD_LOGGING_DIR"] = f"{args.output_dir}/logs"
    model, tokenizer = load_model(args)

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    val_dataset = load_dataset("json", data_files=args.val_file, split="train")
    train_dataset = train_dataset.map(lambda example: render_chat_text(example, tokenizer))
    val_dataset = val_dataset.map(lambda example: render_chat_text(example, tokenizer))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            logging_dir=f"{args.output_dir}/logs",
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            bf16=True,
            optim="adamw_8bit",
            report_to=args.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataset_text_field="text",
            max_length=args.max_seq_length,
        ),
    )
    resume_checkpoint = args.resume_from_checkpoint.strip()
    if args.resume_auto and not resume_checkpoint:
        resume_checkpoint = find_latest_checkpoint(args.output_dir)
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()
    write_run_summary(args.output_dir, train_result.metrics, eval_metrics)

    merged_dir = f"{args.output_dir}/merged-16bit"
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"LoRA adapter + trainer state saved to {args.output_dir}")
    print(f"Merged 16-bit model saved to {merged_dir}")


if __name__ == "__main__":
    main()
