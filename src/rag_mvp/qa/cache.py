from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import monotonic

from rag_mvp.qa.models import QAResponse


@dataclass(frozen=True)
class _CacheEntry:
    value: QAResponse
    expires_at: float


class TTLAnswerCache:
    """Small process-local TTL/LRU cache suitable for a single-instance MVP."""

    def __init__(self, *, max_entries: int = 256, ttl_seconds: float = 300) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> QAResponse | None:
        now = monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.value.model_copy(deep=True)

    def set(self, key: str, value: QAResponse) -> None:
        with self._lock:
            self._entries[key] = _CacheEntry(
                value=value.model_copy(deep=True), expires_at=monotonic() + self.ttl_seconds
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
