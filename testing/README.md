# Post-Training Testing Workspace

Use this folder to run quick model checks after each training run.

## Files

- `prompts_smoke.jsonl` - starter Swahili math prompts with expected answers.
- `results/` - generated model outputs (JSONL) from test runs.

## Run tests

From repo root:

```bash
.venv/bin/python scripts/test_model_outputs.py \
  --model-path outputs/qwen25-1p5b-hisabati-hf-run1 \
  --engine hf \
  --prompts-file testing/prompts_smoke.jsonl \
  --offline
```

For Gemma Unsloth checkpoints:

```bash
.venv/bin/python scripts/test_model_outputs.py \
  --model-path outputs/gemma3-4b-hisabati-unsloth-run1/checkpoint-193 \
  --engine unsloth \
  --prompts-file testing/prompts_smoke.jsonl \
  --offline \
  --merge-system-into-user
```

The script prints the output path and writes JSONL rows with:

- `id`
- `question`
- `expected`
- `response`
