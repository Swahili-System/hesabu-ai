# Tanzanian Primary Mathematics Dataset Tools

This workspace contains a small scraper for FlipHTML5 books and a dataset builder for Swahili primary mathematics fine-tuning data.

## Quick Start

Scrape the configured book text:

```bash
python3 scripts/scrape_fliphtml5.py --manifest books.json --out-dir data
```

By default, the fine-tune JSONL skips front matter such as title pages, copyright, contents, acknowledgements, and introduction pages. Keep those pages in the fine-tune file if needed:

```bash
python3 scripts/scrape_fliphtml5.py --manifest books.json --out-dir data --include-front-matter
```

Also download page images listed in the FlipHTML5 config:

```bash
python3 scripts/scrape_fliphtml5.py --manifest books.json --out-dir data --download-images
```

Build computed question-answer examples for fine-tuning:

```bash
python3 scripts/build_qa_dataset.py --pages data/pages/all_pages.jsonl --out-dir data/qa
```

## Outputs

For each book id in `books.json`, the script writes:

- `data/raw/<book_id>.html` - downloaded FlipHTML5 HTML snapshot.
- `data/pages/<book_id>.jsonl` - one JSON object per extracted page.
- `data/finetune/<book_id>.jsonl` - chat-style training examples.
- `data/images/<book_id>/...` - optional page images when `--download-images` is used.
- `data/qa/<book_id>.jsonl` - computed Q&A examples from parsed exercises.

It also writes combined datasets:

- `data/pages/all_pages.jsonl`
- `data/finetune/all_finetune.jsonl`
- `data/qa/all_qa.jsonl`

For question-answering fine-tuning, prefer `data/qa/all_qa.jsonl`. The `data/finetune/all_finetune.jsonl` file is better for style and textbook-like teaching content.

The QA builder includes computed arithmetic, multiplication, division (with remainders), place-value, money, fractions, percentage, average, and word-problem examples spanning Darasa la Kwanza through Saba. These examples are tagged with `metadata.skill` (for example `multiplication_word_problem` or `division_remainder`) and `metadata.grade`, so they can be inspected or filtered before training. Addition and subtraction still dominate the set by volume, since Darasa 1-2 content is the most complete; treat upper-grade skills as a starting point rather than exhaustive coverage.

Some FlipHTML5 books, such as `hisabati_darasa_3` and `hisabati_darasa_4`, may not expose a text layer. For those books the scraper records one page object per page image with `text_source: "image_only"` and `image_url`. Use `--download-images` to save the images locally for OCR.

Run local macOS Vision OCR and merge the text back into the dataset:

```bash
CLANG_MODULE_CACHE_PATH=.swift_module_cache swift scripts/ocr_vision.swift data/images/hisabati_darasa_3_png data/ocr/hisabati_darasa_3.jsonl
python3 scripts/import_ocr_pages.py --book-id hisabati_darasa_3 --ocr-jsonl data/ocr/hisabati_darasa_3.jsonl --manifest books.json --out-dir data
CLANG_MODULE_CACHE_PATH=.swift_module_cache swift scripts/ocr_vision.swift data/images/hisabati_darasa_4_png data/ocr/hisabati_darasa_4.jsonl
python3 scripts/import_ocr_pages.py --book-id hisabati_darasa_4 --ocr-jsonl data/ocr/hisabati_darasa_4.jsonl --manifest books.json --out-dir data
```

If the source images are `.webp`, convert them to PNG first with `sips` so Vision can read them.

`CLANG_MODULE_CACHE_PATH=.swift_module_cache` writes compiler cache files under `.swift_module_cache/` — this directory is gitignored and safe to delete (`rm -rf .swift_module_cache`) any time; it is not part of the dataset.

## Adding Darasa la Pili mpaka Saba

Add each book to `books.json` with a stable `id`, `grade`, `subject`, `title`, and FlipHTML5 `url`, then rerun the scraper. The parser uses the visible FlipHTML5 page text, so books with the same `flip-basic-text` layout should work without OCR.

Before fine-tuning a model, verify that you have the right to use each book for model training.

## Fine-tuning candidates

Both candidates below run entirely on your own GPU, no cloud billing or exportable-weights tradeoff, in current priority order:

1. **Gemma 4 (12B) QLoRA** — first priority. Strong reasoning (configurable thinking mode), Apache 2.0, comfortably fits a single 32GB GPU via QLoRA. Google's model card lists Swahili only under the broad 140+ language pretraining tier, not the 35+ "out-of-the-box" tier, so its Swahili quality here is unverified until you eval it against Qwen3.
2. **Qwen3-14B QLoRA** — same GPU-local approach, with documented (research-backed) Swahili translation quality, as a fallback/comparison if Gemma 4 underperforms on Swahili.

Both read from the same merged dataset. Build it once:

```bash
python3 scripts/prepare_sft_dataset.py --out-dir data/sft
```

This dedupes across `data/finetune/all_finetune.jsonl` and `data/qa/all_qa.jsonl` and writes `data/sft/train.jsonl` and `data/sft/val.jsonl`.

Output models are named `sdt-flare-<base>-<size>-hisabati`, where `<base>` is `gm` for Gemma or `qn` for Qwen and `<size>` is the parameter count in billions.

### Gemma 4 QLoRA (first priority) — sdt-flare-gm-12-hisabati

Copy the repo to the GPU box, install `unsloth`, `trl`, `peft`, `bitsandbytes`, `datasets`, `accelerate`, then run:

```bash
python3 scripts/train_gemma4_lora.py --base-model unsloth/gemma-4-12B-it --output-dir outputs/sdt-flare-gm-12-hisabati
```

Verify the `--base-model` repo id on Hugging Face/Unsloth before running — Unsloth's naming for newly released models sometimes changes in the first weeks. The script merges the system prompt into the first user turn since Gemma's chat template has no `system` role, and leaves thinking mode off (no `<|think|>` token) to match the dataset's direct, non-chain-of-thought answers.

### Qwen3-14B QLoRA (fallback/comparison) — sdt-flare-qn-14-hisabati

```bash
python3 scripts/train_qwen3_lora.py --base-model unsloth/Qwen3-14B --output-dir outputs/sdt-flare-qn-14-hisabati
```

Both scripts save a LoRA adapter plus a merged 16-bit checkpoint under `--output-dir`. The current dataset (3,240 examples: 3,078 train / 162 val) is enough for style and format adaptation across Darasa 1-7; addition/subtraction still dominate the QA set by volume, so treat upper-grade results as a starting point, not a final answer.

## Tracking training results

Training scripts write trackable artifacts under `--output-dir`:

- `logs/` - TensorBoard logs when TensorBoard callback is active
- `train_results.json` and `eval_results.json` - final trainer metrics
- `trainer_state.json` - full step-by-step history (`log_history`)
- `run_summary.json` - compact train/eval snapshot
- `metrics_by_step.csv` and `tracking_summary.json` - generated by `scripts/summarize_training_run.py`

Open TensorBoard during/after training:

```bash
tensorboard --logdir outputs --port 6006
```

Generate CSV/JSON tracking files from any completed run:

```bash
python3 scripts/summarize_training_run.py --run-dir outputs/sdt-flare-gm-12-hisabati
```

If you cannot use Unsloth in your current environment (for example missing system headers or no internet to fetch new model repos), use the Transformers + PEFT fallback trainer:

```bash
python3 scripts/train_qwen_lora_hf.py \
  --base-model unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit \
  --output-dir outputs/smoke-qwen25-1p5b-hisabati-hf \
  --max-steps 20 \
  --eval-strategy steps \
  --eval-steps 10 \
  --save-strategy steps \
  --save-steps 10 \
  --offline
```

### Local Linux note (Unsloth + Triton headers)

On Linux, Unsloth kernels may fail with `Python.h` missing if `python3.12-dev` is not installed system-wide.

If you do not have sudo access, you can extract headers locally and export include paths:

```bash
apt download python3.12-dev libpython3.12-dev
mkdir -p .local-python-dev
dpkg-deb -x ./libpython3.12-dev_*_amd64.deb .local-python-dev
dpkg-deb -x ./python3.12-dev_*_amd64.deb .local-python-dev
export C_INCLUDE_PATH="$PWD/.local-python-dev/usr/include:$PWD/.local-python-dev/usr/include/python3.12:$PWD/.local-python-dev/usr/include/x86_64-linux-gnu/python3.12"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
```

Then run Gemma training in the same shell.

## Post-run model testing

Use the testing workspace in `testing/` to check model outputs after training.

Run Qwen/HF checkpoints:

```bash
python3 scripts/test_model_outputs.py \
  --model-path outputs/qwen25-1p5b-hisabati-hf-run1 \
  --engine hf \
  --prompts-file testing/prompts_smoke.jsonl \
  --offline
```

Run Gemma/Unsloth checkpoints:

```bash
python3 scripts/test_model_outputs.py \
  --model-path outputs/gemma3-4b-hisabati-unsloth-run1/checkpoint-193 \
  --engine unsloth \
  --prompts-file testing/prompts_smoke.jsonl \
  --offline \
  --merge-system-into-user
```

Outputs are written to `testing/results/*.jsonl` for easy comparison.

You can force deterministic greedy decoding during tests with:

```bash
python3 scripts/test_model_outputs.py \
  --model-path outputs/qwen25-1p5b-hisabati-hf-run1 \
  --engine hf \
  --prompts-file testing/prompts_smoke.jsonl \
  --offline \
  --temperature 0
```

## Automation scripts (full run + results)

Run full Qwen pipeline (prepare -> train -> summarize -> test):

```bash
scripts/run_qwen_full.sh
```

Run full Gemma pipeline (prepare -> train -> summarize -> test):

```bash
scripts/run_gemma_full.sh
```

`scripts/run_gemma_full.sh` runs in persistent background mode by default:

- starts detached and writes `outputs/gemma3-4b-hisabati-unsloth-full/run.log`
- stores PID in `outputs/gemma3-4b-hisabati-unsloth-full/run.pid`
- resumes automatically from the latest `checkpoint-*` when rerun after interruption

To force foreground execution:

```bash
PERSIST=0 scripts/run_gemma_full.sh
```

Show Qwen run summary:

```bash
scripts/show_qwen_results.sh outputs/qwen25-1p5b-hisabati-hf-run1
```

Show Gemma run summary:

```bash
scripts/show_gemma_results.sh outputs/gemma3-4b-hisabati-unsloth-run1
```

For quick smoke checks before a full multi-hour run, cap steps:

```bash
python3 scripts/train_gemma4_lora.py \
  --base-model unsloth/gemma-4-12B-it \
  --output-dir outputs/sdt-flare-gm-12-hisabati-smoke \
  --max-steps 20 \
  --eval-strategy steps \
  --eval-steps 10 \
  --save-strategy steps \
  --save-steps 10
```
