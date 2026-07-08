#!/usr/bin/env python3
"""QLoRA fine-tune Qwen models without Unsloth (Transformers + PEFT + TRL)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-file", default="data/sft/train.jsonl")
    parser.add_argument("--val-file", default="data/sft/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/smoke-qwen25-1p5b-hisabati-hf")
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
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--merge-system-into-user", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def merge_system_into_user(messages: list[dict]) -> list[dict]:
    system_text = next((m["content"] for m in messages if m.get("role") == "system"), None)
    merged = [m for m in messages if m.get("role") != "system"]
    if system_text and merged and merged[0].get("role") == "user":
        merged[0] = {"role": "user", "content": f"{system_text}\n\n{merged[0]['content']}"}
    return merged


def render_chat_text(example: dict, tokenizer, merge_system: bool) -> dict:
    messages = example["messages"]
    if merge_system:
        messages = merge_system_into_user(messages)
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


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        args.local_files_only = True
    if "tensorboard" in str(args.report_to):
        os.environ["TENSORBOARD_LOGGING_DIR"] = f"{args.output_dir}/logs"
    model, tokenizer = load_model_and_tokenizer(args)

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    val_dataset = load_dataset("json", data_files=args.val_file, split="train")
    train_dataset = train_dataset.map(
        lambda example: render_chat_text(example, tokenizer, args.merge_system_into_user)
    )
    val_dataset = val_dataset.map(
        lambda example: render_chat_text(example, tokenizer, args.merge_system_into_user)
    )

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

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()
    write_run_summary(args.output_dir, train_result.metrics, eval_metrics)

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter + tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
