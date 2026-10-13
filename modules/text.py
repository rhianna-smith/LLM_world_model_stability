import re
import string
import pandas
from typing import Optional

PUNCT_TO_REMOVE = string.punctuation.replace("-", "")


def normalize_text(text: object) -> Optional[str]:
    if text is None or pandas.isna(text):
        return None

    normalized = str(text).lower().strip()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = normalized.translate(str.maketrans("", "", PUNCT_TO_REMOVE))
    normalized = re.sub(r"\b(a|an|the)\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def check_if_same(text_1: object, text_2: object) -> bool:
    return normalize_text(text_1) == normalize_text(text_2)


def map_answer_to_option(
    final_answer: object,
    option_1: object,
    option_2: object
) -> Optional[str]:
    answer_norm = normalize_text(final_answer)
    option_1_norm = normalize_text(option_1)
    option_2_norm = normalize_text(option_2)

    if answer_norm is None:
        return None

    if answer_norm == option_1_norm:
        return "option_1"

    if answer_norm == option_2_norm:
        return "option_2"

    if option_1_norm in answer_norm or answer_norm in option_1_norm:
        return "option_1"

    if option_2_norm in answer_norm or answer_norm in option_2_norm:
        return "option_2"

    return "unclear"


def map_answer_to_reverse_option(
    final_answer: object,
    option_1: object,
    option_2: object
) -> Optional[str]:
    mapped_option = map_answer_to_option(
        final_answer,
        option_1,
        option_2,
    )
    if mapped_option == "option_1":
        return "reverse_option_1"

    if mapped_option == "option_2":
        return "reverse_option_2"

    return mapped_option
