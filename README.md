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
