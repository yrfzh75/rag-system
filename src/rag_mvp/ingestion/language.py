import re

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Return a lightweight language label suitable for metadata filtering."""
    cjk_count = len(_CJK_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    total = cjk_count + latin_count
    if total == 0:
        return "unknown"
    cjk_ratio = cjk_count / total
    if cjk_ratio >= 0.70:
        return "zh"
    if cjk_ratio <= 0.10:
        return "en"
    return "mixed"
