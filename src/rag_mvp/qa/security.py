from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redacted: bool


_PII_PATTERNS = (
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
)

_INJECTION_PATTERNS = (
    re.compile(r"ignore (?:all |any )?(?:previous|prior|system) instructions?", re.IGNORECASE),
    re.compile(r"(?:reveal|show|print|repeat) (?:the )?(?:system|developer) prompt", re.IGNORECASE),
    re.compile(r"(?:override|bypass|disable) (?:the )?(?:safety|guardrails?|rules?)", re.IGNORECASE),
    re.compile(r"忽略(?:之前|以上|系统).{0,8}(?:指令|提示|规则)"),
    re.compile(r"(?:显示|泄露|输出).{0,8}(?:系统提示|开发者提示)"),
)


def redact_pii(text: str) -> RedactionResult:
    result = text
    for pattern, replacement in _PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return RedactionResult(text=result, redacted=result != text)


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
