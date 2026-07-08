#!/usr/bin/env python3
"""Build computed Q&A fine-tuning data from scraped Swahili math books."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


SYSTEM_PROMPT = (
    "Wewe ni mwalimu wa Hisabati wa shule ya msingi Tanzania. "
    "Jibu kwa Kiswahili rahisi na onesha hatua fupi za kupata jibu."
)

ONES = {
    0: "sifuri",
    1: "moja",
    2: "mbili",
    3: "tatu",
    4: "nne",
    5: "tano",
    6: "sita",
    7: "saba",
    8: "nane",
    9: "tisa",
}
TENS = {
    10: "kumi",
    20: "ishirini",
    30: "thelathini",
    40: "arobaini",
    50: "hamsini",
    60: "sitini",
    70: "sabini",
    80: "themanini",
    90: "tisini",
}
HUNDREDS = {
    100: "mia moja",
    200: "mia mbili",
    300: "mia tatu",
    400: "mia nne",
    500: "mia tano",
    600: "mia sita",
    700: "mia saba",
    800: "mia nane",
    900: "mia tisa",
}
WORD_NUMBERS = {value: key for key, value in ONES.items()}
WORD_NUMBERS.update({value: key for key, value in TENS.items()})


def number_to_swahili(number: int) -> str:
    if number < 0 or number > 1000:
        raise ValueError(f"Unsupported number: {number}")
    if number < 10:
        return ONES[number]
    if number < 100:
        tens = number // 10 * 10
        ones = number % 10
        return TENS[tens] if ones == 0 else f"{TENS[tens]} na {ONES[ones]}"
    if number == 1000:
        return "elfu moja"
    hundreds = number // 100 * 100
    remainder = number % 100
    return HUNDREDS[hundreds] if remainder == 0 else f"{HUNDREDS[hundreds]} {number_to_swahili(remainder)}"


def swahili_to_number(text: str) -> int | None:
    value = text.strip().lower().replace("  ", " ")
    if value in WORD_NUMBERS:
        return WORD_NUMBERS[value]
    if value == "elfu moja":
        return 1000
    for base_text, base_value in HUNDREDS.items():
        if value == base_value:
            return base_text
        prefix = f"{base_value} "
        if value.startswith(prefix):
            remainder = swahili_to_number(value[len(prefix) :])
            return None if remainder is None else base_text + remainder
    for tens_text, tens_value in WORD_NUMBERS.items():
        if tens_value >= 10 and value.startswith(f"{tens_text} na "):
            one = WORD_NUMBERS.get(value.replace(f"{tens_text} na ", ""))
            return None if one is None else tens_value + one
    return None


def qa_record(user: str, assistant: str, metadata: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
    }


def arithmetic_answer(numbers: list[int], op: str) -> tuple[int, str]:
    if op == "+":
        return sum(numbers), " + ".join(str(n) for n in numbers)
    if op == "-":
        result = numbers[0]
        for number in numbers[1:]:
            result -= number
        return result, " - ".join(str(n) for n in numbers)
    raise ValueError(op)


def arithmetic_explanation(numbers: list[int], op: str, unit: str = "") -> str:
    result, expression = arithmetic_answer(numbers, op)
    label = f" {unit}" if unit else ""
    if op == "+":
        return f"{expression} = {result}. Kwa hiyo, jibu ni{label} {result}."
    return f"{expression} = {result}. Kwa hiyo, jibu ni{label} {result}."


def build_synthetic_examples() -> list[dict]:
    examples: list[dict] = []

    grade_ranges = {1: range(0, 101), 2: range(0, 1001)}
    for grade, numbers in grade_ranges.items():
        step = 5 if grade == 1 else 25
        for number in list(numbers)[::step]:
            examples.append(
                qa_record(
                    f"Andika namba {number} kwa maneno.",
                    f"Namba {number} kwa maneno ni {number_to_swahili(number)}.",
                    {"source": "synthetic", "skill": "number_to_words", "grade": grade},
                )
            )
            examples.append(
                qa_record(
                    f"Andika '{number_to_swahili(number)}' kwa tarakimu.",
                    f"'{number_to_swahili(number)}' kwa tarakimu ni {number}.",
                    {"source": "synthetic", "skill": "words_to_number", "grade": grade},
                )
            )

    for a in range(0, 10):
        for b in range(0, 10):
            if a + b <= 9:
                examples.append(
                    qa_record(
                        f"Jumlisha {a} + {b}.",
                        arithmetic_explanation([a, b], "+"),
                        {"source": "synthetic", "skill": "addition", "grade": 1},
                    )
                )
            if a >= b:
                examples.append(
                    qa_record(
                        f"Toa {a} - {b}.",
                        arithmetic_explanation([a, b], "-"),
                        {"source": "synthetic", "skill": "subtraction", "grade": 1},
                    )
                )

    grade2_pairs = [
        (247, 123),
        (299, 133),
        (597, 50),
        (795, 275),
        (980, 440),
        (635, 213),
        (528, 417),
        (446, 358),
        (590, 348),
        (516, 378),
        (625, 286),
        (900, 800),
        (1000, 650),
    ]
    for a, b in grade2_pairs:
        examples.append(
            qa_record(
                f"Jumlisha {a} + {b}.",
                arithmetic_explanation([a, b], "+"),
                {"source": "synthetic", "skill": "addition", "grade": 2},
            )
        )
        if a >= b:
            examples.append(
                qa_record(
                    f"Toa {a} - {b}.",
                    arithmetic_explanation([a, b], "-"),
                    {"source": "synthetic", "skill": "subtraction", "grade": 2},
                )
            )

    for number in [36, 47, 69, 105, 248, 500, 999, 1000]:
        examples.append(place_value_example(number, grade=2))

    money_pairs = [(600, 400), (100, 900), (350, 400), (550, 435), (750, 250)]
    for a, b in money_pairs:
        examples.append(
            qa_record(
                f"Shilingi {a} ongeza shilingi {b}. Jumla ni shilingi ngapi?",
                arithmetic_explanation([a, b], "+", unit="shilingi"),
                {"source": "synthetic", "skill": "money_addition", "grade": 2},
            )
        )
    for a, b in [(900, 800), (450, 100), (1000, 650), (700, 250), (800, 750)]:
        examples.append(
            qa_record(
                f"Shilingi {a} punguza shilingi {b}. Baki shilingi ngapi?",
                arithmetic_explanation([a, b], "-", unit="shilingi"),
                {"source": "synthetic", "skill": "money_subtraction", "grade": 2},
            )
        )

    examples.extend(word_problem_examples())
    examples.extend(grade_enrichment_examples())
    examples.extend(advanced_word_problems())
    return examples


def place_value_example(number: int, grade: int) -> dict:
    thousands = number // 1000
    hundreds = number % 1000 // 100
    tens = number % 100 // 10
    ones = number % 10
    parts = []
    if thousands:
        parts.append(f"maelfu {thousands}")
    parts.extend([f"mamia {hundreds}", f"makumi {tens}", f"mamoja {ones}"])
    return qa_record(
        f"Andika thamani ya nafasi za tarakimu katika namba {number}.",
        f"Namba {number} ina {', '.join(parts)}.",
        {"source": "synthetic", "skill": "place_value", "grade": grade},
    )


def word_problem_examples() -> list[dict]:
    templates = [
        (
            "Kijiji kimoja kina wanawake 446 na wanaume 358. Kijiji hicho kina jumla ya watu wangapi?",
            [446, 358],
            "+",
            "watu",
            2,
        ),
        (
            "Shule ina wanafunzi 590 na shule nyingine ina wanafunzi 348. Shule hizo zina jumla ya wanafunzi wangapi?",
            [590, 348],
            "+",
            "wanafunzi",
            2,
        ),
        (
            "Kaka alikuwa na mayai 649, akauza mayai 415. Alibaki na mayai mangapi?",
            [649, 415],
            "-",
            "mayai",
            2,
        ),
        (
            "Basi lilibeba wanafunzi 874. Wanafunzi 360 wakashuka njiani. Walibaki wanafunzi wangapi ndani ya basi?",
            [874, 360],
            "-",
            "wanafunzi",
            2,
        ),
        (
            "Asha alinunua maandazi ya shilingi 250. Alitoa noti ya shilingi 1000. Alirudishiwa shilingi ngapi?",
            [1000, 250],
            "-",
            "shilingi",
            2,
        ),
        (
            "Maria alipewa shilingi 300 na mama yake. Akaongezewa shilingi 600 na baba yake. Maria alipata jumla ya shilingi ngapi?",
            [300, 600],
            "+",
            "shilingi",
            2,
        ),
        (
            "Kuku walitaga mayai 32. Mayai 17 yalivunjika. Yalibaki mayai mangapi?",
            [32, 17],
            "-",
            "mayai",
            1,
        ),
        (
            "Bakari alikuwa na maembe 31. Alitupa maembe 12 yaliyooza. Alibaki na maembe mangapi?",
            [31, 12],
            "-",
            "maembe",
            1,
        ),
        (
            "Boni alipanda miche 46. Miche 26 ilinyauka. Ilibaki miche mingapi?",
            [46, 26],
            "-",
            "miche",
            1,
        ),
        (
            "Neema alikuwa na vikombe 51. Alitupa vikombe 15. Alibaki na vikombe vingapi?",
            [51, 15],
            "-",
            "vikombe",
            1,
        ),
        (
            "Doto alikuwa na chupa za soda 90. Aliuza chupa 61. Alibaki na chupa ngapi?",
            [90, 61],
            "-",
            "chupa",
            1,
        ),
        (
            "Watoto walikuwa na chupa 96 zenye juisi. Walikunywa chupa 47. Zilibaki chupa ngapi?",
            [96, 47],
            "-",
            "chupa",
            1,
        ),
        (
            "Saida alikuwa na machungwa 66. Aliwapatia wadogo zake machungwa 37. Alibaki na machungwa mangapi?",
            [66, 37],
            "-",
            "machungwa",
            1,
        ),
        (
            "Kulwa alikuwa na kalamu 75. Alimpatia rafiki yake kalamu 25. Alibaki na kalamu ngapi?",
            [75, 25],
            "-",
            "kalamu",
            1,
        ),
        (
            "Baba alikuwa na vitabu 88. Vitabu 11 vilipotea. Vilivyobaki ni vitabu vingapi?",
            [88, 11],
            "-",
            "vitabu",
            1,
        ),
        (
            "Amina ana miaka 6 na Yohana ana miaka 7. Jumla ya umri wao ni miaka mingapi?",
            [6, 7],
            "+",
            "miaka",
            1,
        ),
        (
            "Neema ana mayai 15. Mayai 8 yamevunjika. Yatabaki mayai mangapi?",
            [15, 8],
            "-",
            "mayai",
            1,
        ),
        (
            "Mama alikuwa na kuku 90. Aliuza kuku 25. Walibaki kuku wangapi?",
            [90, 25],
            "-",
            "kuku",
            1,
        ),
        (
            "Ashura alikuwa na machungwa 87. Amenunua machungwa mengine 13. Ana jumla ya machungwa mangapi?",
            [87, 13],
            "+",
            "machungwa",
            1,
        ),
        (
            "Familia moja ilikuwa na mbuzi 130. Familia nyingine ilikuwa na mbuzi 110. Jumla ya mbuzi ni wangapi?",
            [130, 110],
            "+",
            "mbuzi",
            2,
        ),
        (
            "Mkulima alipanda miche 403 ya mipapai na miche 592 ya michungwa. Alipanda jumla ya miche mingapi?",
            [403, 592],
            "+",
            "miche",
            2,
        ),
        (
            "Shule moja ina wanafunzi 372 na shule nyingine ina wanafunzi 527. Shule hizo zina jumla ya wanafunzi wangapi?",
            [372, 527],
            "+",
            "wanafunzi",
            2,
        ),
        (
            "Fundi alishona sare 601 za wasichana na sare 326 za wavulana. Alishona jumla ya sare ngapi?",
            [601, 326],
            "+",
            "sare",
            2,
        ),
        (
            "Shule ina wasichana 294 na wavulana 312. Shule ina jumla ya wanafunzi wangapi?",
            [294, 312],
            "+",
            "wanafunzi",
            2,
        ),
        (
            "Darasa moja lina madawati 333 na darasa jingine lina madawati 426. Jumla ya madawati ni mangapi?",
            [333, 426],
            "+",
            "madawati",
            2,
        ),
        (
            "Juma ana shilingi 350 na Asha ana shilingi 520. Wana jumla ya shilingi ngapi?",
            [350, 520],
            "+",
            "shilingi",
            2,
        ),
        (
            "Mwalimu alikuwa na madaftari 150. Akanunua madaftari mengine 216. Ana jumla ya madaftari mangapi?",
            [150, 216],
            "+",
            "madaftari",
            2,
        ),
        (
            "Mkulima alikuwa na ng'ombe 247. Akanunua ng'ombe wengine 264. Ana jumla ya ng'ombe wangapi?",
            [247, 264],
            "+",
            "ng'ombe",
            2,
        ),
        (
            "Duka lilikuwa na mayai 516. Likaongezewa mayai 378. Dukani kuna jumla ya mayai mangapi?",
            [516, 378],
            "+",
            "mayai",
            2,
        ),
        (
            "Muuza soda alikuwa na soda 464. Akanunua soda nyingine 385. Ana jumla ya soda ngapi?",
            [464, 385],
            "+",
            "soda",
            2,
        ),
        (
            "Duka lilikuwa na kalamu 625. Likaongezewa kalamu 286. Dukani kuna jumla ya kalamu ngapi?",
            [625, 286],
            "+",
            "kalamu",
            2,
        ),
        (
            "Kijiji kilipokea vyandarua 766. Kikaongezewa vyandarua 186. Kilipokea jumla ya vyandarua vingapi?",
            [766, 186],
            "+",
            "vyandarua",
            2,
        ),
        (
            "Mkulima alivuna mapapai 187. Akavuna mapapai mengine 216. Alivuna jumla ya mapapai mangapi?",
            [187, 216],
            "+",
            "mapapai",
            2,
        ),
        (
            "Mwalimu alikuwa na vitabu 596. Akawagawia wanafunzi vitabu 421. Alibaki na vitabu vingapi?",
            [596, 421],
            "-",
            "vitabu",
            2,
        ),
        (
            "Roza alikuwa na machungwa 365. Akauza machungwa yote 365. Alibaki na machungwa mangapi?",
            [365, 365],
            "-",
            "machungwa",
            2,
        ),
        (
            "Dukani kulikuwa na kalamu 635. Kalamu 412 zikauzwa. Zilibaki kalamu ngapi?",
            [635, 412],
            "-",
            "kalamu",
            2,
        ),
        (
            "Muuza mitumba alikuwa na midoli 888. Akauza midoli 573. Alibaki na midoli mingapi?",
            [888, 573],
            "-",
            "midoli",
            2,
        ),
        (
            "Mfanyabiashara alinunua chupa za juisi 760. Akauza chupa 650. Alibaki na chupa ngapi za juisi?",
            [760, 650],
            "-",
            "chupa",
            2,
        ),
        (
            "Ali alinunua sahani kwa shilingi 450 na kikombe kwa shilingi 540. Alitumia jumla ya shilingi ngapi?",
            [450, 540],
            "+",
            "shilingi",
            2,
        ),
        (
            "Mama alinunua maembe kwa shilingi 550 na mapera kwa shilingi 400. Alitumia jumla ya shilingi ngapi?",
            [550, 400],
            "+",
            "shilingi",
            2,
        ),
        (
            "Asha alinunua rula kwa shilingi 250 na kalamu kwa shilingi 200. Alitumia jumla ya shilingi ngapi?",
            [250, 200],
            "+",
            "shilingi",
            2,
        ),
        (
            "Bibi alikuwa na shilingi 900. Akaongezewa shilingi 50. Ana jumla ya shilingi ngapi?",
            [900, 50],
            "+",
            "shilingi",
            2,
        ),
        (
            "Asha alikuwa na shilingi 650. Alinunua chungwa kwa shilingi 300. Alibaki na shilingi ngapi?",
            [650, 300],
            "-",
            "shilingi",
            2,
        ),
        (
            "Roza alikuwa na shilingi 900. Alitumia shilingi 500. Alibaki na shilingi ngapi?",
            [900, 500],
            "-",
            "shilingi",
            2,
        ),
        (
            "Puto ziliuzwa kwa shilingi 800. Mteja alilipa shilingi 150 kwanza. Alibaki kudaiwa shilingi ngapi?",
            [800, 150],
            "-",
            "shilingi",
            2,
        ),
        (
            "Ali alikuwa na shilingi 600. Alinunua kitu cha shilingi 450. Alibaki na shilingi ngapi?",
            [600, 450],
            "-",
            "shilingi",
            2,
        ),
    ]
    examples = []
    for question, numbers, op, unit, grade in templates:
        examples.append(
            qa_record(
                question,
                arithmetic_explanation(numbers, op, unit=unit),
                {"source": "synthetic", "skill": "complex_word_problem", "grade": grade},
            )
        )
    examples.append(
        qa_record(
            "Yohana alinunua daftari la shilingi 500 na kalamu 2. Kila kalamu ni shilingi 200. Alitumia jumla ya shilingi ngapi?",
            "Kwanza tafuta bei ya kalamu 2: 200 + 200 = 400. Kisha jumlisha na daftari: 500 + 400 = 900. Kwa hiyo, alitumia shilingi 900.",
            {"source": "synthetic", "skill": "complex_word_problem", "grade": 2},
        )
    )
    examples.append(
        qa_record(
            "Asha alikuwa na shilingi 900. Alinunua daftari kwa shilingi 500 na kalamu kwa shilingi 200. Alibaki na shilingi ngapi?",
            "Kwanza jumlisha alizotumia: 500 + 200 = 700. Kisha toa kwenye 900: 900 - 700 = 200. Kwa hiyo, Asha alibaki na shilingi 200.",
            {"source": "synthetic", "skill": "complex_word_problem", "grade": 2},
        )
    )
    examples.append(
        qa_record(
            "Juma ana noti ya shilingi 1000. Roza ana shilingi 500 na shilingi 200. Nani ana fedha nyingi zaidi na tofauti ni shilingi ngapi?",
            "Roza ana 500 + 200 = 700. Juma ana 1000. Tofauti ni 1000 - 700 = 300. Kwa hiyo, Juma ana fedha nyingi zaidi kwa shilingi 300.",
            {"source": "synthetic", "skill": "complex_word_problem", "grade": 2},
        )
    )
    return examples


def comparison_answer(a: int, b: int) -> str:
    if a > b:
        return f"{a} ni kubwa kuliko {b} kwa tofauti ya {a - b}."
    if b > a:
        return f"{b} ni kubwa kuliko {a} kwa tofauti ya {b - a}."
    return f"{a} na {b} ni sawa."


def ordering_answer(numbers: list[int]) -> str:
    ordered = ", ".join(str(number) for number in sorted(numbers))
    return f"Mpangilio kuanzia ndogo hadi kubwa ni {ordered}."


def multiplication_answer(a: int, b: int) -> str:
    return f"{a} x {b} = {a * b}. Kwa hiyo, jibu ni {a * b}."


def division_answer(dividend: int, divisor: int) -> str:
    return f"{dividend} ÷ {divisor} = {dividend // divisor}. Kwa hiyo, jibu ni {dividend // divisor}."


def fraction_compare_answer(n1: int, d1: int, n2: int, d2: int) -> str:
    left = n1 / d1
    right = n2 / d2
    if left > right:
        return f"{n1}/{d1} ni kubwa kuliko {n2}/{d2}."
    if right > left:
        return f"{n2}/{d2} ni kubwa kuliko {n1}/{d1}."
    return f"{n1}/{d1} na {n2}/{d2} ni sawa."


def average_answer(numbers: list[int]) -> str:
    total = sum(numbers)
    count = len(numbers)
    average = total // count
    return f"Jumla ni {total}. Tukigawanya kwa {count}, tunapata {average}. Kwa hiyo, wastani ni {average}."


def percentage_answer(percent: int, number: int) -> str:
    value = number * percent // 100
    return f"{percent}% ya {number} ni {value}."


def division_remainder_answer(dividend: int, divisor: int) -> str:
    quotient, remainder = divmod(dividend, divisor)
    if remainder == 0:
        return f"{dividend} ÷ {divisor} = {quotient}. Kwa hiyo, jibu ni {quotient}."
    return (
        f"{dividend} ÷ {divisor} = {quotient} na mabaki ya {remainder}. "
        f"Kwa hiyo, jibu ni {quotient} na mabaki {remainder}."
    )


def fraction_addition_answer(n1: int, n2: int, d: int) -> str:
    total = n1 + n2
    return f"{n1}/{d} + {n2}/{d} = {total}/{d}."


def multiplication_division_facts(pairs: list[tuple[int, int]], grade: int) -> list[dict]:
    examples: list[dict] = []
    for a, b in pairs:
        examples.append(
            qa_record(
                f"Zidisha {a} x {b}.",
                multiplication_answer(a, b),
                {"source": "synthetic", "skill": "multiplication", "grade": grade},
            )
        )
        dividend = a * b
        examples.append(
            qa_record(
                f"Gawa {dividend} kwa {a}.",
                division_answer(dividend, a),
                {"source": "synthetic", "skill": "division", "grade": grade},
            )
        )
    return examples


NAMES = [
    "Juma", "Asha", "Neema", "Bakari", "Kulwa", "Doto",
    "Amina", "Yohana", "Fatuma", "Baraka", "Salma", "Rehema",
]

COUNTABLE_NOUNS = [
    "mayai", "machungwa", "madaftari", "vitabu", "kalamu",
    "chupa", "miche", "sare", "madawati", "ng'ombe",
]


def multiplication_word_problems(pairs: list[tuple[int, int]], grade: int) -> list[dict]:
    examples: list[dict] = []
    for index, (groups, per_group) in enumerate(pairs):
        name = NAMES[index % len(NAMES)]
        noun = COUNTABLE_NOUNS[index % len(COUNTABLE_NOUNS)]
        total = groups * per_group
        examples.append(
            qa_record(
                f"{name} ana makundi {groups} ya {noun}, kila kundi lina {noun} {per_group}. "
                f"Ana jumla ya {noun} wangapi?",
                f"{groups} x {per_group} = {total}. Kwa hiyo, {name} ana jumla ya {noun} {total}.",
                {"source": "synthetic", "skill": "multiplication_word_problem", "grade": grade},
            )
        )
    return examples


def division_word_problems(pairs: list[tuple[int, int]], grade: int) -> list[dict]:
    examples: list[dict] = []
    for index, (total, people) in enumerate(pairs):
        name = NAMES[index % len(NAMES)]
        noun = COUNTABLE_NOUNS[index % len(COUNTABLE_NOUNS)]
        share = total // people
        examples.append(
            qa_record(
                f"{name} ana {noun} {total}, anataka kuzigawa sawa kwa watu {people}. "
                f"Kila mtu atapata {noun} ngapi?",
                f"{total} ÷ {people} = {share}. Kwa hiyo, kila mtu atapata {noun} {share}.",
                {"source": "synthetic", "skill": "division_word_problem", "grade": grade},
            )
        )
    return examples


def price_word_problems(pairs: list[tuple[int, int]], grade: int) -> list[dict]:
    examples: list[dict] = []
    for index, (price, quantity) in enumerate(pairs):
        name = NAMES[index % len(NAMES)]
        noun = COUNTABLE_NOUNS[index % len(COUNTABLE_NOUNS)]
        total = price * quantity
        examples.append(
            qa_record(
                f"Bei ya {noun} moja ni shilingi {price}. {name} ananunua {noun} {quantity}. "
                f"Atalipa shilingi ngapi?",
                f"{price} x {quantity} = {total}. Kwa hiyo, {name} atalipa shilingi {total}.",
                {"source": "synthetic", "skill": "multiplication_word_problem", "grade": grade},
            )
        )
    return examples


def money_share_word_problems(pairs: list[tuple[int, int]], grade: int) -> list[dict]:
    examples: list[dict] = []
    for index, (total, people) in enumerate(pairs):
        name = NAMES[index % len(NAMES)]
        share = total // people
        examples.append(
            qa_record(
                f"{name} ana shilingi {total}. Anataka kugawanya kwa watu {people} kwa usawa. "
                f"Kila mtu atapata shilingi ngapi?",
                f"{total} ÷ {people} = {share}. Kwa hiyo, kila mtu atapata shilingi {share}.",
                {"source": "synthetic", "skill": "division_word_problem", "grade": grade},
            )
        )
    return examples


def fraction_word_problems(triples: list[tuple[int, int, int]], grade: int) -> list[dict]:
    examples: list[dict] = []
    for index, (total, numerator, denominator) in enumerate(triples):
        name = NAMES[index % len(NAMES)]
        planted = total * numerator // denominator
        examples.append(
            qa_record(
                f"Shamba la {name} lina ekari {total}. Alipanda mahindi kwenye sehemu ya "
                f"{numerator}/{denominator} ya shamba. Ni ekari ngapi zenye mahindi?",
                f"{total} x {numerator}/{denominator} = {planted}. "
                f"Kwa hiyo, ekari zenye mahindi ni {planted}.",
                {"source": "synthetic", "skill": "fraction_word_problem", "grade": grade},
            )
        )
    return examples


def percentage_word_problems(pairs: list[tuple[int, int]], grade: int) -> list[dict]:
    examples: list[dict] = []
    for total, percent in pairs:
        girls = total * percent // 100
        examples.append(
            qa_record(
                f"Darasa lina wanafunzi {total}. Asilimia {percent} ya wanafunzi ni wasichana. "
                f"Wasichana ni wangapi?",
                f"{percent}% ya {total} ni {girls}. Kwa hiyo, wasichana ni wanafunzi {girls}.",
                {"source": "synthetic", "skill": "percentage_word_problem", "grade": grade},
            )
        )
    return examples


def average_word_problems(score_sets: list[list[int]], grade: int) -> list[dict]:
    examples: list[dict] = []
    for index, scores in enumerate(score_sets):
        name = NAMES[index % len(NAMES)]
        scores_text = ", ".join(str(score) for score in scores)
        examples.append(
            qa_record(
                f"{name} alipata alama zifuatazo katika mitihani {len(scores)}: {scores_text}. "
                f"Wastani wa alama zake ni ngapi?",
                average_answer(scores),
                {"source": "synthetic", "skill": "average_word_problem", "grade": grade},
            )
        )
    return examples


def advanced_word_problems() -> list[dict]:
    examples: list[dict] = []

    examples.extend(multiplication_word_problems(
        [(4, 6), (5, 7), (3, 9), (6, 4), (7, 3), (8, 2), (9, 5), (2, 8)], grade=3,
    ))
    examples.extend(division_word_problems(
        [(24, 4), (36, 6), (45, 5), (18, 3), (56, 7), (63, 9), (72, 8), (28, 4)], grade=3,
    ))

    examples.extend(price_word_problems(
        [(150, 4), (200, 3), (120, 5), (250, 4), (180, 3), (300, 2), (90, 6), (175, 4)], grade=4,
    ))
    examples.extend(money_share_word_problems(
        [(480, 4), (360, 6), (420, 7), (540, 9), (640, 8), (450, 5), (540, 6), (720, 8)], grade=4,
    ))

    examples.extend(price_word_problems(
        [(1200, 4), (1500, 3), (850, 6), (950, 5), (2000, 3), (1750, 4), (600, 8), (450, 9)], grade=5,
    ))
    examples.extend(money_share_word_problems(
        [(4500, 5), (6400, 8), (7200, 9), (3600, 6), (5600, 7), (8100, 9), (4900, 7), (3000, 4)], grade=5,
    ))
    examples.extend(fraction_word_problems(
        [(12, 1, 3), (20, 1, 4), (15, 2, 5), (18, 1, 6), (24, 3, 4), (30, 2, 5)], grade=5,
    ))
    examples.extend(percentage_word_problems(
        [(40, 25), (60, 50), (80, 75), (50, 10), (200, 15), (120, 25)], grade=5,
    ))
    examples.extend(average_word_problems(
        [[65, 75, 85, 95], [60, 70, 80, 90], [55, 65, 75, 85], [88, 92, 84, 96]], grade=5,
    ))

    examples.extend(price_word_problems(
        [(2500, 4), (3200, 3), (1800, 6), (2100, 5), (4000, 3), (3500, 4), (1200, 8), (900, 9)], grade=6,
    ))
    examples.extend(money_share_word_problems(
        [(9600, 8), (12000, 10), (8400, 7), (10800, 9), (7200, 6), (13500, 9), (9800, 7), (6000, 4)], grade=6,
    ))
    examples.extend(fraction_word_problems(
        [(36, 1, 3), (48, 1, 4), (45, 2, 5), (54, 1, 6), (72, 3, 4), (60, 2, 5)], grade=6,
    ))
    examples.extend(percentage_word_problems(
        [(80, 25), (120, 50), (160, 75), (100, 10), (240, 15), (200, 35)], grade=6,
    ))
    examples.extend(average_word_problems(
        [[150, 170, 190, 210], [220, 240, 260, 280], [140, 160, 180, 200], [300, 320, 340, 360]], grade=6,
    ))

    examples.extend(price_word_problems(
        [(3500, 6), (4200, 5), (2800, 7), (3100, 8), (5000, 4), (4500, 6), (1600, 9), (2700, 7)], grade=7,
    ))
    examples.extend(money_share_word_problems(
        [(18000, 12), (24000, 15), (16800, 14), (21000, 10), (19500, 13), (27000, 18), (15400, 11), (23400, 13)],
        grade=7,
    ))
    examples.extend(fraction_word_problems(
        [(84, 1, 3), (96, 1, 4), (90, 2, 5), (108, 1, 6), (120, 3, 4), (105, 2, 5)], grade=7,
    ))
    examples.extend(percentage_word_problems(
        [(160, 25), (240, 50), (320, 75), (200, 10), (360, 15), (280, 35)], grade=7,
    ))
    examples.extend(average_word_problems(
        [[240, 260, 280, 300], [310, 330, 350, 370], [400, 420, 440, 460], [180, 200, 220, 240]], grade=7,
    ))

    return examples


def grade_enrichment_examples() -> list[dict]:
    examples: list[dict] = []

    grade1_pairs = [(4, 3), (7, 5), (8, 1), (9, 2), (10, 6), (11, 4), (12, 7), (13, 8)]
    for a, b in grade1_pairs:
        examples.append(
            qa_record(
                f"Jumlisha {a} + {b}.",
                arithmetic_explanation([a, b], "+"),
                {"source": "synthetic", "skill": "addition", "grade": 1},
            )
        )
        if a >= b:
            examples.append(
                qa_record(
                    f"Toa {a} - {b}.",
                    arithmetic_explanation([a, b], "-"),
                    {"source": "synthetic", "skill": "subtraction", "grade": 1},
                )
            )
    for a, b in [(5, 2), (6, 6), (9, 4), (7, 8), (3, 1)]:
        examples.append(
            qa_record(
                f"Namba gani kubwa kati ya {a} na {b}?",
                comparison_answer(a, b),
                {"source": "synthetic", "skill": "comparison", "grade": 1},
            )
        )

    grade2_pairs = [(124, 35), (248, 57), (312, 89), (405, 126), (560, 240), (732, 155)]
    for a, b in grade2_pairs:
        examples.append(
            qa_record(
                f"Jumlisha {a} + {b}.",
                arithmetic_explanation([a, b], "+"),
                {"source": "synthetic", "skill": "addition", "grade": 2},
            )
        )
        if a >= b:
            examples.append(
                qa_record(
                    f"Toa {a} - {b}.",
                    arithmetic_explanation([a, b], "-"),
                    {"source": "synthetic", "skill": "subtraction", "grade": 2},
                )
            )
        examples.append(
            qa_record(
                f"Ni namba gani kubwa kati ya {a} na {b}?",
                comparison_answer(a, b),
                {"source": "synthetic", "skill": "comparison", "grade": 2},
            )
        )
    for number in [134, 240, 365, 480, 507, 690, 812, 999]:
        examples.append(place_value_example(number, grade=2))
    for numbers in ([124, 248, 365], [312, 405, 560], [732, 155, 418]):
        examples.append(
            qa_record(
                f"Panga namba {', '.join(str(number) for number in numbers)} kuanzia ndogo hadi kubwa.",
                ordering_answer(list(numbers)),
                {"source": "synthetic", "skill": "ordering", "grade": 2},
            )
        )

    grade3_multiples = [(a, b) for a in range(2, 10) for b in range(2, 10)]
    examples.extend(multiplication_division_facts(grade3_multiples, grade=3))
    for number in [1243, 2365, 4087, 5120, 6789, 7004]:
        examples.append(
            qa_record(
                f"Katika namba {number}, tarakimu ya makumi ni ipi na ina thamani gani?",
                f"Tarakimu ya makumi ni {str(number)[-2]}. Thamani yake ni {int(str(number)[-2]) * 10}.",
                {"source": "synthetic", "skill": "place_value", "grade": 3},
            )
        )
    for numbers in ([243, 324, 234], [1789, 1879, 8179], [4062, 4602, 6402]):
        examples.append(
            qa_record(
                f"Panga namba {', '.join(str(number) for number in numbers)} kuanzia ndogo hadi kubwa.",
                ordering_answer(list(numbers)),
                {"source": "synthetic", "skill": "ordering", "grade": 3},
            )
        )

    grade4_pairs = [(2345, 678), (4567, 890), (3204, 785), (4890, 1205), (5678, 2344), (7125, 1600)]
    for a, b in grade4_pairs:
        examples.append(
            qa_record(
                f"Jumlisha {a} + {b}.",
                arithmetic_explanation([a, b], "+"),
                {"source": "synthetic", "skill": "addition", "grade": 4},
            )
        )
        if a >= b:
            examples.append(
                qa_record(
                    f"Toa {a} - {b}.",
                    arithmetic_explanation([a, b], "-"),
                    {"source": "synthetic", "skill": "subtraction", "grade": 4},
                )
            )
        examples.append(
            qa_record(
                f"Ni namba gani kubwa kati ya {a} na {b}?",
                comparison_answer(a, b),
                {"source": "synthetic", "skill": "comparison", "grade": 4},
            )
        )
    grade4_multiples = [(a, b) for a in range(12, 49, 6) for b in range(2, 10)]
    examples.extend(multiplication_division_facts(grade4_multiples, grade=4))
    for m, cm in [(3, 2), (5, 4), (7, 3)]:
        examples.append(
            qa_record(
                f"Mita {m} ni sentimita ngapi?",
                f"Mita {m} ni sentimita {m * 100}.",
                {"source": "synthetic", "skill": "conversion", "grade": 4},
            )
        )
        examples.append(
            qa_record(
                f"Sentimita {cm * 100} ni mita ngapi?",
                f"Sentimita {cm * 100} ni mita {cm}.",
                {"source": "synthetic", "skill": "conversion", "grade": 4},
            )
        )

    grade5_multiples = [(a, b) for a in range(11, 30, 3) for b in range(11, 30, 3)]
    examples.extend(multiplication_division_facts(grade5_multiples, grade=5))
    for dividend, divisor in [(53, 4), (67, 5), (86, 7), (94, 6), (77, 8), (103, 9), (58, 3), (121, 10)]:
        examples.append(
            qa_record(
                f"Gawa {dividend} kwa {divisor}.",
                division_remainder_answer(dividend, divisor),
                {"source": "synthetic", "skill": "division_remainder", "grade": 5},
            )
        )
    for numbers in (
        [45, 55, 60, 50], [72, 68, 80, 76], [120, 130, 125, 115],
        [90, 100, 110, 100], [64, 68, 72, 76], [200, 210, 190, 200],
        [55, 65, 75, 65],
    ):
        examples.append(
            qa_record(
                f"Tafuta wastani wa namba {', '.join(str(number) for number in numbers)}.",
                average_answer(list(numbers)),
                {"source": "synthetic", "skill": "average", "grade": 5},
            )
        )
    for fractions in [
        ((1, 2), (1, 3)), ((2, 5), (1, 2)), ((3, 4), (2, 3)), ((1, 3), (1, 4)),
        ((3, 5), (1, 2)), ((2, 3), (3, 5)), ((5, 6), (3, 4)), ((1, 6), (1, 3)),
    ]:
        (n1, d1), (n2, d2) = fractions
        examples.append(
            qa_record(
                f"Ni sehemu gani kubwa kati ya {n1}/{d1} na {n2}/{d2}?",
                fraction_compare_answer(n1, d1, n2, d2),
                {"source": "synthetic", "skill": "fractions", "grade": 5},
            )
        )
    for n1, n2, d in [(1, 2, 5), (2, 3, 7), (1, 4, 8), (3, 2, 9), (2, 2, 6)]:
        examples.append(
            qa_record(
                f"Jumlisha {n1}/{d} + {n2}/{d}.",
                fraction_addition_answer(n1, n2, d),
                {"source": "synthetic", "skill": "fractions", "grade": 5},
            )
        )
    for percent, number in [
        (10, 200), (25, 400), (50, 300), (75, 200), (5, 400),
        (20, 250), (40, 150), (80, 250), (15, 400), (90, 200),
    ]:
        examples.append(
            qa_record(
                f"Tafuta {percent}% ya {number}.",
                percentage_answer(percent, number),
                {"source": "synthetic", "skill": "percentage", "grade": 5},
            )
        )

    grade6_multiples = [(a, b) for a in range(15, 60, 5) for b in range(6, 15, 2)]
    examples.extend(multiplication_division_facts(grade6_multiples, grade=6))
    for dividend, divisor in [(145, 6), (238, 9), (317, 8), (176, 7), (289, 11), (405, 12), (163, 5), (524, 9)]:
        examples.append(
            qa_record(
                f"Gawa {dividend} kwa {divisor}.",
                division_remainder_answer(dividend, divisor),
                {"source": "synthetic", "skill": "division_remainder", "grade": 6},
            )
        )
    for numbers in (
        [150, 180, 210, 240], [320, 340, 300, 360], [75, 125, 100, 150],
        [220, 240, 260, 280], [410, 390, 400, 400], [155, 165, 175, 185],
        [500, 520, 480, 500],
    ):
        examples.append(
            qa_record(
                f"Tafuta wastani wa namba {', '.join(str(number) for number in numbers)}.",
                average_answer(list(numbers)),
                {"source": "synthetic", "skill": "average", "grade": 6},
            )
        )
    for fractions in [
        ((1, 4), (1, 2)), ((3, 8), (1, 2)), ((5, 6), (2, 3)), ((2, 7), (1, 3)),
        ((4, 9), (1, 2)), ((5, 12), (1, 3)), ((7, 8), (3, 4)), ((2, 9), (1, 4)),
    ]:
        (n1, d1), (n2, d2) = fractions
        examples.append(
            qa_record(
                f"Ni sehemu gani kubwa kati ya {n1}/{d1} na {n2}/{d2}?",
                fraction_compare_answer(n1, d1, n2, d2),
                {"source": "synthetic", "skill": "fractions", "grade": 6},
            )
        )
    for n1, n2, d in [(2, 3, 8), (3, 4, 10), (1, 5, 9), (4, 3, 11), (5, 2, 12)]:
        examples.append(
            qa_record(
                f"Jumlisha {n1}/{d} + {n2}/{d}.",
                fraction_addition_answer(n1, n2, d),
                {"source": "synthetic", "skill": "fractions", "grade": 6},
            )
        )
    for percent, number in [
        (5, 200), (12, 250), (20, 150), (30, 500), (45, 200),
        (10, 350), (65, 200), (35, 400), (55, 400), (85, 200),
    ]:
        examples.append(
            qa_record(
                f"Tafuta {percent}% ya {number}.",
                percentage_answer(percent, number),
                {"source": "synthetic", "skill": "percentage", "grade": 6},
            )
        )

    grade7_multiples = [(a, b) for a in range(20, 100, 10) for b in range(11, 30, 3)]
    examples.extend(multiplication_division_facts(grade7_multiples, grade=7))
    for dividend, divisor in [(437, 12), (592, 15), (721, 13), (836, 17), (654, 14), (913, 16), (348, 11), (777, 19)]:
        examples.append(
            qa_record(
                f"Gawa {dividend} kwa {divisor}.",
                division_remainder_answer(dividend, divisor),
                {"source": "synthetic", "skill": "division_remainder", "grade": 7},
            )
        )
    for numbers in (
        [260, 280, 300, 320], [480, 520, 500, 540], [95, 105, 115, 125],
        [610, 590, 600, 600], [340, 360, 380, 400], [225, 235, 245, 255],
        [710, 730, 690, 710],
    ):
        examples.append(
            qa_record(
                f"Tafuta wastani wa namba {', '.join(str(number) for number in numbers)}.",
                average_answer(list(numbers)),
                {"source": "synthetic", "skill": "average", "grade": 7},
            )
        )
    for fractions in [
        ((2, 3), (3, 4)), ((5, 8), (2, 5)), ((7, 10), (3, 5)), ((3, 7), (2, 5)),
        ((5, 9), (1, 2)), ((7, 12), (5, 8)), ((4, 11), (1, 3)), ((9, 10), (5, 6)),
    ]:
        (n1, d1), (n2, d2) = fractions
        examples.append(
            qa_record(
                f"Ni sehemu gani kubwa kati ya {n1}/{d1} na {n2}/{d2}?",
                fraction_compare_answer(n1, d1, n2, d2),
                {"source": "synthetic", "skill": "fractions", "grade": 7},
            )
        )
    for n1, n2, d in [(3, 5, 13), (4, 6, 15), (2, 7, 14), (5, 4, 16), (6, 3, 17)]:
        examples.append(
            qa_record(
                f"Jumlisha {n1}/{d} + {n2}/{d}.",
                fraction_addition_answer(n1, n2, d),
                {"source": "synthetic", "skill": "fractions", "grade": 7},
            )
        )
    for percent, number in [
        (15, 200), (20, 350), (40, 250), (60, 500), (8, 400),
        (24, 250), (36, 200), (48, 400), (72, 200), (95, 400),
    ]:
        examples.append(
            qa_record(
                f"Tafuta {percent}% ya {number}.",
                percentage_answer(percent, number),
                {"source": "synthetic", "skill": "percentage", "grade": 7},
            )
        )

    return examples


def build_parsed_examples(records: Iterable[dict]) -> list[dict]:
    examples: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        text = normalize_math_text(record["text"])
        if has_bad_artifacts(text):
            continue
        context_op = infer_context_op(text)
        for match in re.finditer(r"(?<!\d)(\d{1,4})\s*([+−–-])\s*(\d{1,4})(?:\s*\+\s*(\d{1,4}))?\s*=", text):
            a = int(match.group(1))
            op = "+" if match.group(2) == "+" else "-"
            b = int(match.group(3))
            c = int(match.group(4)) if match.group(4) else None
            numbers = [a, b] if c is None else [a, b, c]
            if op == "-" and c is not None:
                continue
            if any(number > 1000 for number in numbers):
                continue
            result, expression = arithmetic_answer(numbers, op)
            if result < 0 or result > 2000:
                continue
            add_example(examples, seen, record, expression, numbers, op)

        for match in re.finditer(r"(?<!\d)(\d{1,4})\s+_\s+(\d{1,4})\s*=", text):
            if context_op != "-":
                continue
            a, b = int(match.group(1)), int(match.group(2))
            if a < b or a > 1000 or b > 1000:
                continue
            add_example(examples, seen, record, f"{a} - {b}", [a, b], "-")

        for match in re.finditer(r"(?<!\d)(\d{1,4})\s*\+\s*=\s*(\d{1,4})", text):
            a, total = int(match.group(1)), int(match.group(2))
            missing = total - a
            if 0 <= missing <= 1000:
                add_missing_addend_example(examples, seen, record, f"{a} + __ = {total}", missing)

        for match in re.finditer(r"(?<!\d\s)\+\s*(\d{1,4})\s*=\s*(\d{1,4})", text):
            b, total = int(match.group(1)), int(match.group(2))
            missing = total - b
            if 0 <= missing <= 1000:
                add_missing_addend_example(examples, seen, record, f"__ + {b} = {total}", missing)

    return examples


def normalize_math_text(text: str) -> str:
    return text.replace("–", "-").replace("−", "-")


def infer_context_op(text: str) -> str | None:
    lowered = text.lower()
    if any(word in lowered for word in ("toa", "kutoa", "punguza", "baki")):
        return "-"
    if any(word in lowered for word in ("jumlisha", "ongeza", "jumla")):
        return "+"
    return None


def has_bad_artifacts(text: str) -> bool:
    bad = ("DUPLICATE", "FOR ONLINE", "indd", "D_O", "DfaO", "DfO", "DhO", "DmO", "DnO")
    return any(marker.lower() in text.lower() for marker in bad)


def add_example(
    examples: list[dict],
    seen: set[tuple[str, str]],
    record: dict,
    expression: str,
    numbers: list[int],
    op: str,
) -> None:
    user_verb = "Jumlisha" if op == "+" else "Toa"
    user = f"{user_verb} {expression}."
    assistant = arithmetic_explanation(numbers, op)
    key = (user, assistant)
    if key in seen:
        return
    seen.add(key)
    examples.append(
        qa_record(
            user,
            assistant,
            {
                "source": "parsed_page",
                "skill": "addition" if op == "+" else "subtraction",
                "book_id": record["book_id"],
                "grade": record["grade"],
                "page_index": record["page_index"],
                "visible_page": record["visible_page"],
            },
        )
    )


def add_missing_addend_example(
    examples: list[dict],
    seen: set[tuple[str, str]],
    record: dict,
    expression: str,
    missing: int,
) -> None:
    user = f"Tafuta namba inayokosekana: {expression}."
    assistant = f"Namba inayokosekana ni {missing}, kwa sababu {expression.replace('__', str(missing))}."
    key = (user, assistant)
    if key in seen:
        return
    seen.add(key)
    examples.append(
        qa_record(
            user,
            assistant,
            {
                "source": "parsed_page",
                "skill": "missing_addend",
                "book_id": record["book_id"],
                "grade": record["grade"],
                "page_index": record["page_index"],
                "visible_page": record["visible_page"],
            },
        )
    )


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe(rows: Iterable[dict]) -> list[dict]:
    output = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        user = row["messages"][1]["content"]
        assistant = row["messages"][2]["content"]
        key = (user, assistant)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", default=Path("data/pages/all_pages.jsonl"), type=Path)
    parser.add_argument("--out-dir", default=Path("data/qa"), type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.pages)
    parsed = build_parsed_examples(records)
    synthetic = build_synthetic_examples()
    all_examples = dedupe([*parsed, *synthetic])

    by_book: dict[str, list[dict]] = {}
    for row in all_examples:
        book_id = row["metadata"].get("book_id") or f"synthetic_grade_{row['metadata'].get('grade', 'all')}"
        by_book.setdefault(book_id, []).append(row)

    for book_id, rows in sorted(by_book.items()):
        write_jsonl(args.out_dir / f"{book_id}.jsonl", rows)
        print(f"{book_id}: {len(rows)}")
    write_jsonl(args.out_dir / "all_qa.jsonl", all_examples)
    print(f"all_qa: {len(all_examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
