#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3-14B on the Swahili primary math dataset (Unsloth + TRL).

Run on the GPU box after `prepare_sft_dataset.py` has produced
data/sft/train.jsonl and data/sft/val.jsonl.
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="unsloth/Qwen3-14B")
    parser.add_argument("--train-file", default="data/sft/train.jsonl")
    parser.add_argument("--val-file", default="data/sft/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/sdt-flare-qn-14-hisabati")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
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


def render_chat_text(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text": text}


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args)

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    val_dataset = load_dataset("json", data_files=args.val_file, split="train")
    train_dataset = train_dataset.map(lambda example: render_chat_text(example, tokenizer))
    val_dataset = val_dataset.map(lambda example: render_chat_text(example, tokenizer))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            bf16=True,
            optim="adamw_8bit",
            report_to="none",
        ),
    )
    trainer.train()

    merged_dir = f"{args.output_dir}/merged-16bit"
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"LoRA adapter + trainer state saved to {args.output_dir}")
    print(f"Merged 16-bit model saved to {merged_dir}")


if __name__ == "__main__":
    main()
