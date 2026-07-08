---
language:
- sw
license: unknown
task_categories:
- question-answering
- text-generation
pretty_name: Hesabu AI - Swahili Primary Mathematics
size_categories:
- 1K<n<10K
tags:
- swahili
- mathematics
- education
- tanzania
- primary-school
- sft
---

# Hesabu AI: Swahili Primary Mathematics (Darasa 1-7)

A supervised fine-tuning dataset for teaching language models Tanzanian primary-school mathematics (Hisabati/Kuhesabu) in Swahili, covering Darasa la Kwanza through Darasa la Saba (grades 1-7).

## Dataset Description

The dataset has two complementary sources, merged into a single chat-format split for training:

- **Textbook content** (`finetune`) — page-level teaching content extracted from official Tanzanian primary math textbooks, framed as "prepare a lesson on topic X" instructions. Good for style, explanation register, and topic coverage.
- **Computed Q&A** (`qa`) — arithmetic, word, and applied-math problems with verified answers, either parsed directly from textbook exercises or generated programmatically (and independently re-verified for correctness). Good for grounded, checkable problem-solving.

Every example is a 3-turn chat record: a system prompt establishing the "Tanzanian primary math teacher" persona, a user instruction or question in Swahili, and an assistant response in Swahili.

```json
{
  "messages": [
    {"role": "system", "content": "Wewe ni mwalimu wa Hisabati wa shule ya msingi Tanzania. Jibu kwa Kiswahili rahisi na onesha hatua fupi za kupata jibu."},
    {"role": "user", "content": "Zidisha 5 x 2."},
    {"role": "assistant", "content": "5 x 2 = 10. Kwa hiyo, jibu ni 10."}
  ],
  "metadata": {"source": "synthetic", "skill": "multiplication", "grade": 3}
}
```

## Dataset Structure

| split | examples | description |
|---|---|---|
| `train` | 3,078 | training split |
| `validation` | 162 | held-out split (5%) |
| **total** | **3,240** | deduplicated union of `finetune` + `qa` |

Source breakdown before merge/dedup:

| source | examples |
|---|---|
| textbook content (`finetune`) | 1,426 |
| computed Q&A (`qa`) | 1,814 |

### Grade coverage (`qa` source)

| grade (Darasa) | examples |
|---|---|
| 1 | 416 |
| 2 | 332 |
| 3 | 196 |
| 4 | 221 |
| 5 | 224 |
| 6 | 188 |
| 7 | 237 |

### Skill coverage (`qa` source, `metadata.skill`)

| skill | examples | skill | examples |
|---|---|---|---|
| addition | 432 | fractions | 39 |
| subtraction | 336 | percentage | 30 |
| multiplication | 257 | division_remainder | 24 |
| division | 257 | place_value | 21 |
| missing_addend | 67 | average | 21 |
| number_to_words | 57 | fraction_word_problem | 18 |
| words_to_number | 57 | percentage_word_problem | 18 |
| complex_word_problem | 49 | comparison | 17 |
| multiplication_word_problem | 40 | average_word_problem | 12 |
| division_word_problem | 40 | ordering / conversion | 6 each |
| | | money_addition / money_subtraction | 5 each |

Addition and subtraction dominate by volume because Darasa 1-2 source material is the most complete; treat grade 3-7 coverage (multiplication, division, fractions, percentage, average) as a solid starting point rather than exhaustive.

## Data Fields

- `messages`: list of `{role, content}` turns (`system`, `user`, `assistant`).
- `metadata.skill` *(qa only)*: fine-grained skill tag (see table above).
- `metadata.grade`: Darasa level, 1-7.
- `metadata.book_id` *(where applicable)*: source textbook identifier.
- `metadata.source` *(qa only)*: `parsed_page` (extracted from a real exercise) or `synthetic` (programmatically generated and verified).
- `metadata.page_index` / `metadata.visible_page` *(finetune, and parsed qa)*: source page reference within the book.

## Source Data

Textbook content was scraped from publicly hosted FlipHTML5 editions of official Tanzanian primary mathematics textbooks (Kuhesabu Darasa la 1-2, Hisabati Darasa la 3-7), with OCR fallback (Apple Vision) for image-only pages. See the companion `scripts/` in this repository for the full scrape → parse → QA-generation pipeline.

## Licensing and Usage Rights

**The copyright status of the source textbooks has not been independently verified.** They are believed to be Tanzanian government/institutional curriculum material accessed via a public FlipHTML5 hosting link, but no explicit redistribution license was confirmed before this dataset was built. Do not treat this repository as cleared for unrestricted redistribution or commercial use until that is checked. If you are the rights holder and have concerns about this dataset, please open a discussion on the repository.

## Known Limitations

- Skill and grade coverage is uneven (see tables above) — grades 1-2 arithmetic is overrepresented relative to grades 3-7.
- `finetune`-source examples reflect raw textbook page text, including occasional OCR noise on image-only source pages.
- Answers in `qa` were independently re-verified by recomputation at dataset-build time (multiplication, division, division-with-remainder, fractions, percentage, and average were checked programmatically); textbook-extracted `parsed_page` examples reflect the source exercise as printed.

## Citation

If you use this dataset, please credit the Hesabu AI project and the source textbook publishers.
