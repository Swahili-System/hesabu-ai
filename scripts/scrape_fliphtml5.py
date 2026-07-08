#!/usr/bin/env python3
"""Scrape FlipHTML5 page text and build Swahili math fine-tuning JSONL."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; dataset-prep/1.0)"
WATERMARKS = {"FOR ONLINE USE ONLY", "DO NOT DUPLICATE"}
WATERMARK_RE = re.compile(r"(?:FOR\s+ONLINE\s+USE\s+ONLY|DO\s+NOT\s+DUPLICATE)", re.IGNORECASE)
EXPORT_FOOTER_RE = re.compile(
    r"(?:\d+\s+)?(?:Kuhesabu|Hisabati)\s+[^.\n]{0,80}\.indd\s+\d+",
    re.IGNORECASE,
)
EXPORT_DATE_RE = re.compile(r"\b(?:7/23/21|30/07/2021)\s*(?:5:\d{2}\s+PM|11:\d{2})?\b")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
OBFUSCATED_WATERMARK_RE = re.compile(
    r"\S*D\S*O\S*N\S*O\S*T\S*D\S*U\S*P\S*L\S*I\S*C\S*A\S*T\S*E\S*"
)
OBFUSCATED_ONLINE_RE = re.compile(r"\S*F\S*O\S*R\S*O\S*N\S*L\S*I\S*N\S*E\S*")
CORRUPTION_MARKERS = (
    "DUPLICATE",
    "ONLINEN",
    "USE ONLY",
    "indd",
    "DSOur",
    "D_O",
    "DfaO",
    "DfO",
    "DhO",
    "DmO",
    "DnO",
    "D_O",
)


@dataclass
class Page:
    visible_page: str | None
    text_lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines).strip()


class FlipBasicTextParser(HTMLParser):
    def __init__(self, keep_watermarks: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.keep_watermarks = keep_watermarks
        self.pages: list[Page] = []
        self.in_page = False
        self.page_depth = 0
        self.capture: str | None = None
        self.buffer: list[str] = []
        self.current_page_num: str | None = None
        self.current_lines: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {name: value or "" for name, value in attrs}
        classes = set(attrs_map.get("class", "").split())

        if tag == "div" and "flip-basic-text" in classes and "above-text" not in classes:
            self.in_page = True
            self.page_depth = 1
            self.current_page_num = None
            self.current_lines = []
            self.capture = None
            self.buffer = []
            return

        if not self.in_page:
            return

        if tag == "div":
            self.page_depth += 1
            if "flip-basic-num" in classes:
                self.capture = "page_num"
                self.buffer = []
        elif tag == "p":
            self.capture = "paragraph"
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.in_page and self.capture:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_page:
            return

        if tag == "p" and self.capture == "paragraph":
            text = normalize_text("".join(self.buffer))
            if text and (self.keep_watermarks or text not in WATERMARKS):
                self.current_lines.append(text)
            self.capture = None
            self.buffer = []
            return

        if tag == "div":
            if self.capture == "page_num":
                page_num = normalize_text("".join(self.buffer))
                match = re.search(r"P:\s*0*(\d+)", page_num)
                self.current_page_num = match.group(1) if match else page_num or None
                self.capture = None
                self.buffer = []

            self.page_depth -= 1
            if self.page_depth == 0:
                if self.current_lines:
                    self.pages.append(Page(self.current_page_num, self.current_lines))
                self.in_page = False


def normalize_text(value: str) -> str:
    value = unescape(value).replace("\ufeff", "").replace("\xa0", " ")
    value = CONTROL_CHAR_RE.sub(" ", value)
    value = OBFUSCATED_WATERMARK_RE.sub(" ", value)
    value = OBFUSCATED_ONLINE_RE.sub(" ", value)
    value = WATERMARK_RE.sub(" ", value)
    value = value.replace("DSOurNaOyTaDUKPwLaICnAzTaE", " ")
    value = re.sub(r"FOR\s+ONLINEN?", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"USE\s+ONLY", " ", value, flags=re.IGNORECASE)
    value = EXPORT_FOOTER_RE.sub(" ", value)
    value = EXPORT_DATE_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def fetch_text(url: str, retries: int = 3, pause: float = 1.0) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=45) as response:
                raw = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
                return raw.decode(charset, errors="replace")
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(pause * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def fetch_binary(url: str, retries: int = 3, pause: float = 1.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(pause * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def extract_pages(html: str, keep_watermarks: bool = False) -> list[Page]:
    parser = FlipBasicTextParser(keep_watermarks=keep_watermarks)
    parser.feed(html)
    return parser.pages


def extract_book_online_url(html: str, page_url: str) -> str | None:
    patterns = [
        r"bookOnlieUrl:\s*[\"']([^\"']+)",
        r"bookLink[\"']?\s*:\s*[\"']([^\"']+)",
        r"<meta\s+name=[\"']twitter:player[\"']\s+content=[\"']([^\"']+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return urljoin(page_url, match.group(1))
    return None


def extract_config_url(html: str, page_url: str) -> str | None:
    match = re.search(r'<script[^>]+src=["\']([^"\']*javascript/config\.js[^"\']*)["\']', html)
    return urljoin(page_url, match.group(1)) if match else None


def extract_image_urls(config_js: str, online_url: str) -> list[str]:
    urls: list[str] = []
    for name in re.findall(r'"n"\s*:\s*\[\s*"([^"]+)"', config_js):
        name = name.replace("\\/", "/")
        if name.startswith(("./", "/", "http://", "https://")):
            urls.append(urljoin(online_url, name))
        else:
            urls.append(urljoin(online_url, f"files/large/{name}"))
    return urls


def page_records(book: dict, pages: Iterable[Page]) -> list[dict]:
    records: list[dict] = []
    for index, page in enumerate(pages, start=1):
        records.append(
            {
                "book_id": book["id"],
                "title": book.get("title", ""),
                "subject": book.get("subject", ""),
                "grade": book.get("grade"),
                "source_url": book["url"],
                "page_index": index,
                "visible_page": page.visible_page,
                "text": page.text,
                "text_lines": page.text_lines,
            }
        )
    return records


def image_page_records(book: dict, image_urls: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for index, image_url in enumerate(image_urls, start=1):
        records.append(
            {
                "book_id": book["id"],
                "title": book.get("title", ""),
                "subject": book.get("subject", ""),
                "grade": book.get("grade"),
                "source_url": book["url"],
                "page_index": index,
                "visible_page": str(index),
                "text": "",
                "text_lines": [],
                "text_source": "image_only",
                "image_url": image_url,
            }
        )
    return records


def finetune_records(book: dict, records: Iterable[dict], include_front_matter: bool = False) -> list[dict]:
    examples: list[dict] = []
    seen_topics: set[str] = set()
    for record in records:
        text = record["text"].strip()
        if len(text) < 20:
            continue
        if not include_front_matter and is_front_matter(record):
            continue
        if is_corrupted_text(text):
            continue

        topic = infer_topic(record["text_lines"])
        page_label = f"ukurasa {record.get('visible_page') or record['page_index']}"
        if topic == "Hisabati" or topic.lower() in seen_topics:
            topic_label = page_label
        else:
            seen_topics.add(topic.lower())
            topic_label = topic
        system = (
            "Wewe ni mwalimu wa Hisabati wa shule ya msingi Tanzania. "
            "Jibu kwa Kiswahili rahisi na tumia mifano inayofaa darasa."
        )
        user = (
            f"Andaa maudhui ya kufundishia kwa {record['subject']} "
            f"Darasa la {record['grade']}. Mada: {topic_label}."
        )
        assistant = text
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {
                    "book_id": record["book_id"],
                    "page_index": record["page_index"],
                    "visible_page": record["visible_page"],
                },
            }
        )
    return examples


def is_front_matter(record: dict) -> bool:
    visible_page = record.get("visible_page")
    if visible_page and str(visible_page).isdigit() and int(visible_page) < 8:
        return True

    text = record["text"]
    front_markers = (
        "© Taasisi ya Elimu Tanzania",
        "Yaliyomo",
        "Shukurani",
        "Utangulizi",
        "Kitabu cha Mwanafunzi",
    )
    return any(marker in text for marker in front_markers)


def is_corrupted_text(text: str) -> bool:
    upper_text = text.upper()
    if any(marker.upper() in upper_text for marker in CORRUPTION_MARKERS):
        return True
    if EXPORT_DATE_RE.search(text):
        return True
    if CONTROL_CHAR_RE.search(text):
        return True

    compact = re.sub(r"[^A-Za-z]", "", text)
    if re.search(r"D[A-Za-z]{0,8}O[A-Za-z]{0,8}N[A-Za-z]{0,8}O[A-Za-z]{0,8}T", compact):
        return True
    if re.search(r"D[A-Za-z]{0,8}U[A-Za-z]{0,8}P[A-Za-z]{0,8}L[A-Za-z]{0,8}I[A-Za-z]{0,8}C", compact):
        return True
    return False


def infer_topic(lines: list[str]) -> str:
    for line in lines:
        if is_corrupted_text(line):
            continue
        if "FOR ONLINE" in line.upper() or "DUPLICATE" in line.upper():
            continue
        if re.match(r"^(Sura|Zoezi|Mfano|Jaribio)\b", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^[ivxlcdm\d\s.,:+\-–—=_ºª/]+$", line, flags=re.IGNORECASE):
            continue
        letters = re.sub(r"[^A-Za-zÀ-ÿ]", "", line)
        if len(letters) < 4 or not re.search(r"[a-zà-ÿ]", line):
            continue
        return line[:120]
    return "Hisabati"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scrape_book(
    book: dict,
    out_dir: Path,
    keep_watermarks: bool,
    download_images: bool,
    include_front_matter: bool,
) -> tuple[list[dict], list[dict]]:
    html = fetch_text(book["url"])
    raw_path = out_dir / "raw" / f"{book['id']}.html"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(html, encoding="utf-8")

    pages = extract_pages(html, keep_watermarks=keep_watermarks)
    records = page_records(book, pages)
    if not records:
        image_urls = extract_book_image_urls(book, html)
        if image_urls:
            print(
                f"[warn] {book['id']}: no text layer found; recorded {len(image_urls)} image-only pages",
                file=sys.stderr,
            )
            records = image_page_records(book, image_urls)
    examples = finetune_records(book, records, include_front_matter=include_front_matter)

    write_jsonl(out_dir / "pages" / f"{book['id']}.jsonl", records)
    write_jsonl(out_dir / "finetune" / f"{book['id']}.jsonl", examples)

    if download_images:
        download_book_images(book, html, out_dir)

    return records, examples


def download_book_images(book: dict, html: str, out_dir: Path) -> None:
    image_urls = extract_book_image_urls(book, html)
    if not image_urls:
        print(f"[warn] {book['id']}: no page images found; skipping images", file=sys.stderr)
        return

    image_dir = out_dir / "images" / book["id"]
    image_dir.mkdir(parents=True, exist_ok=True)

    for index, image_url in enumerate(image_urls, start=1):
        suffix = Path(image_url).suffix or ".webp"
        target = image_dir / f"page_{index:03d}{suffix}"
        if target.exists():
            continue
        try:
            target.write_bytes(fetch_binary(image_url))
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[warn] {book['id']}: page {index} image download failed: {exc}", file=sys.stderr)


def extract_book_image_urls(book: dict, html: str) -> list[str]:
    online_url = extract_book_online_url(html, book["url"])
    config_url = extract_config_url(html, book["url"])
    if not config_url and online_url:
        config_url = urljoin(online_url, "javascript/config.js")
    if not config_url:
        return []

    config_js = fetch_text(config_url)
    base_url = online_url or book["url"]
    return extract_image_urls(config_js, base_url)


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        books = json.load(handle)
    if not isinstance(books, list):
        raise ValueError("Manifest must be a JSON list")
    for book in books:
        for key in ("id", "url"):
            if key not in book:
                raise ValueError(f"Manifest book missing required key: {key}")
    return books


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="books.json", type=Path)
    parser.add_argument("--out-dir", default="data", type=Path)
    parser.add_argument("--keep-watermarks", action="store_true")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--include-front-matter", action="store_true")
    parser.add_argument("--book-id", action="append", help="Scrape only this book id. May be used more than once.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    books = load_manifest(args.manifest)
    if args.book_id:
        requested = set(args.book_id)
        books = [book for book in books if book["id"] in requested]
        missing = requested - {book["id"] for book in books}
        if missing:
            raise ValueError(f"Unknown book id(s): {', '.join(sorted(missing))}")

    all_pages: list[dict] = []
    all_examples: list[dict] = []
    for book in books:
        print(f"Scraping {book['id']} ...")
        pages, examples = scrape_book(
            book,
            args.out_dir,
            keep_watermarks=args.keep_watermarks,
            download_images=args.download_images,
            include_front_matter=args.include_front_matter,
        )
        print(f"  pages={len(pages)} finetune_examples={len(examples)}")
        all_pages.extend(pages)
        all_examples.extend(examples)

    write_jsonl(args.out_dir / "pages" / "all_pages.jsonl", all_pages)
    write_jsonl(args.out_dir / "finetune" / "all_finetune.jsonl", all_examples)
    print(f"Wrote {len(all_pages)} page records and {len(all_examples)} examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
