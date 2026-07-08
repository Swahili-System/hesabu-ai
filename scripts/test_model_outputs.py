#!/usr/bin/env python3
"""Run post-training inference tests and save outputs as JSONL."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Adapter/checkpoint/model directory")
    parser.add_argument("--engine", choices=["auto", "hf", "unsloth"], default="auto")
    parser.add_argument("--prompts-file", default="testing/prompts_smoke.jsonl")
    parser.add_argument("--out-file", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--merge-system-into-user", action="store_true")
    return parser.parse_args()


def maybe_enable_offline(offline: bool) -> None:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def maybe_set_local_python_headers() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    local_dev_root = repo_root / ".local-python-dev" / "usr" / "include"
    if not local_dev_root.exists():
        return

    include_paths = [
        str(local_dev_root),
        str(local_dev_root / "python3.12"),
        str(local_dev_root / "x86_64-linux-gnu" / "python3.12"),
    ]

    existing_c = os.environ.get("C_INCLUDE_PATH", "")
    existing_cpp = os.environ.get("CPLUS_INCLUDE_PATH", "")
    os.environ["C_INCLUDE_PATH"] = ":".join(include_paths + ([existing_c] if existing_c else []))
    os.environ["CPLUS_INCLUDE_PATH"] = ":".join(include_paths + ([existing_cpp] if existing_cpp else []))


def load_prompts(path: Path) -> list[dict]:
    prompts: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def merge_system_into_user(messages: list[dict]) -> list[dict]:
    system_text = next((m["content"] for m in messages if m.get("role") == "system"), None)
    merged = [m for m in messages if m.get("role") != "system"]
    if system_text and merged and merged[0].get("role") == "user":
        merged[0] = {"role": "user", "content": f"{system_text}\n\n{merged[0]['content']}"}
    return merged


def build_messages(prompt: dict) -> list[dict]:
    if "messages" in prompt and isinstance(prompt["messages"], list):
        return prompt["messages"]
    question = prompt.get("question") or prompt.get("prompt")
    if not question:
        raise ValueError("Prompt must have either `messages` or `question`/`prompt`.")
    return [{"role": "user", "content": question}]


def infer_engine(model_path: Path, requested_engine: str) -> str:
    if requested_engine != "auto":
        return requested_engine
    name = str(model_path).lower()
    if "gemma" in name:
        return "unsloth"
    return "hf"


def generate_with_hf(
    model_path: Path,
    prompts: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    merge_system: bool,
) -> list[dict]:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        peft_config = PeftConfig.from_pretrained(str(model_path))
        base_model_name = peft_config.base_model_name_or_path
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(model, str(model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=local_files_only,
        )

    model.eval()
    outputs: list[dict] = []
    for prompt in prompts:
        messages = build_messages(prompt)
        if merge_system:
            messages = merge_system_into_user(messages)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        generated = model.generate(**inputs, **generate_kwargs)
        prompt_tokens = inputs["input_ids"].shape[-1]
        decoded = tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True).strip()
        outputs.append(
            {
                "id": prompt.get("id"),
                "question": prompt.get("question", prompt.get("prompt")),
                "expected": prompt.get("expected"),
                "response": decoded,
            }
        )
    return outputs


def generate_with_unsloth(
    model_path: Path,
    prompts: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    merge_system: bool,
) -> list[dict]:
    maybe_set_local_python_headers()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    outputs: list[dict] = []
    for prompt in prompts:
        messages = build_messages(prompt)
        if merge_system:
            messages = merge_system_into_user(messages)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        generated = model.generate(**inputs, **generate_kwargs)
        prompt_tokens = inputs["input_ids"].shape[-1]
        decoded = tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True).strip()
        outputs.append(
            {
                "id": prompt.get("id"),
                "question": prompt.get("question", prompt.get("prompt")),
                "expected": prompt.get("expected"),
                "response": decoded,
            }
        )
    return outputs


def default_out_file(model_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = model_path.name.replace("/", "_")
    return Path("testing") / "results" / f"{safe_name}-{stamp}.jsonl"


def main() -> None:
    args = parse_args()
    maybe_enable_offline(args.offline)

    model_path = Path(args.model_path)
    prompts_file = Path(args.prompts_file)
    out_file = Path(args.out_file) if args.out_file else default_out_file(model_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_file)
    engine = infer_engine(model_path, args.engine)

    if engine == "unsloth":
        rows = generate_with_unsloth(
            model_path=model_path,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            merge_system=args.merge_system_into_user,
        )
    else:
        rows = generate_with_hf(
            model_path=model_path,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            merge_system=args.merge_system_into_user,
        )

    with out_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Engine: {engine}")
    print(f"Prompts: {len(rows)}")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
