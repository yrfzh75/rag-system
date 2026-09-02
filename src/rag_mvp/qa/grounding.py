from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d+)\]")
_IGNORED = {"the", "a", "an", "is", "are", "of", "to", "and", "有", "是", "的"}


def content_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.casefold()))
    return tokens - _IGNORED


def lexical_support(answer: str, source_texts: Sequence[str]) -> float:
    answer_tokens = content_tokens(_CITATION.sub("", answer))
    context_tokens = content_tokens(" ".join(source_texts))
    return len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 1.0


@dataclass(frozen=True)
class GroundingCheck:
    valid: bool
    support_score: float
    reason: str | None


def validate_grounding(
    answer: str, source_texts: Sequence[str], *, min_support: float
) -> GroundingCheck:
    citations = [int(value) for value in _CITATION.findall(answer)]
    if not citations:
        return GroundingCheck(False, 0.0, "missing_citation")
    if any(value < 1 or value > len(source_texts) for value in citations):
        return GroundingCheck(False, 0.0, "invalid_citation")
    score = lexical_support(answer, source_texts)
    if score < min_support:
        return GroundingCheck(False, score, "insufficient_context_support")
    return GroundingCheck(True, score, None)
