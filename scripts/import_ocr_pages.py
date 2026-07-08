#!/usr/bin/env python3
"""Merge OCR output into image-only page records and rebuild finetune JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scrape_fliphtml5 import finetune_records, load_manifest, normalize_text, write_jsonl


OCR_NOISE_LINES = {
    "FOR ONLINE USE ONLY",
    "DO NOT DUPLICATE",
    "ONLINE USE ONLY",
    "OR ONLINE USE ONLY",
    "LINE USE ONLY",
    "USE ONLY",
    "USE",
    "LINE USE",
    "LICATE",
    "CATE",
    "ONLY",
    "ONL",
    "FOR",
    "FOR ON",
    "FOR ONLII",
    "ONLINE",
    "LINE",
    "LINET",
    "LIU",
    "US",
    "SE",
    "SE ONLY",
    "NE",
    "DU",
    "NOT",
}

OCR_NOISE_PATTERNS = (
    re.compile(r"^(?:FOR|LINE|USE|US|SE|DO|NOT|DU|NE)(?:\s+[A-Z]{1,4}){0,2}\.?$", re.IGNORECASE),
    re.compile(r"^(?:FOR|LINE|USE|US|SE|DO|NOT|DU|NE)\b", re.IGNORECASE),
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_ocr_lines(lines: list[str]) -> list[str]:
    cleaned = []
    for line in lines:
        text = normalize_text(line)
        text = re.sub(r"\s*(?:DUPLICATE|UPLICATE|PLICATE|LICATE|CATE)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:30/07/2021|11:\d{2})\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        upper = text.upper()
        word_count = len(text.split())
        if not text:
            continue
        if upper in OCR_NOISE_LINES:
            continue
        if any(pattern.match(text) for pattern in OCR_NOISE_PATTERNS):
            continue
        if "INDD" in upper:
            continue
        if upper.startswith("STD III 30/ JULY/ 2021"):
            continue
        if upper in {"®", "•", "••"}:
            continue
        if word_count <= 2 and any(token in upper for token in ("FOR", "LINE", "USE", "DO", "NOT", "DU", "NE", "US", "SE")):
            continue
        cleaned.append(text)
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--ocr-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("books.json"), type=Path)
    parser.add_argument("--out-dir", default=Path("data"), type=Path)
    parser.add_argument("--include-front-matter", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    books = {book["id"]: book for book in manifest}
    if args.book_id not in books:
        raise ValueError(f"Unknown book id: {args.book_id}")

    page_path = args.out_dir / "pages" / f"{args.book_id}.jsonl"
    records = load_jsonl(page_path)
    ocr_rows = {row["page_index"]: row for row in load_jsonl(args.ocr_jsonl)}

    updated = 0
    for record in records:
        ocr = ocr_rows.get(record["page_index"])
        if not ocr:
            continue
        lines = clean_ocr_lines(ocr.get("text_lines", []))
        if not lines:
            continue
        record["text"] = "\n".join(lines)
        record["text_lines"] = lines
        record["text_source"] = "ocr_vision"
        updated += 1

    write_jsonl(page_path, records)
    examples = finetune_records(
        books[args.book_id],
        records,
        include_front_matter=args.include_front_matter,
    )
    write_jsonl(args.out_dir / "finetune" / f"{args.book_id}.jsonl", examples)

    all_pages: list[dict] = []
    all_examples: list[dict] = []
    for book in manifest:
        book_id = book["id"]
        book_pages_path = args.out_dir / "pages" / f"{book_id}.jsonl"
        book_finetune_path = args.out_dir / "finetune" / f"{book_id}.jsonl"
        if book_pages_path.exists():
            all_pages.extend(load_jsonl(book_pages_path))
        if book_finetune_path.exists():
            all_examples.extend(load_jsonl(book_finetune_path))

    write_jsonl(args.out_dir / "pages" / "all_pages.jsonl", all_pages)
    write_jsonl(args.out_dir / "finetune" / "all_finetune.jsonl", all_examples)
    print(f"{args.book_id}: updated_pages={updated} finetune_examples={len(examples)}")
    print(f"all_pages={len(all_pages)} all_finetune={len(all_examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
