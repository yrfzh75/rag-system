from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Protocol


class EventSink(Protocol):
    def emit(self, event: Mapping[str, object]) -> None: ...


class JsonLineEventSink:
    """Emit structured events to stdout and, optionally, an append-only JSONL file."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = Lock()

    def emit(self, event: Mapping[str, object]) -> None:
        line = json.dumps(dict(event), ensure_ascii=False, separators=(",", ":"))
        print(line, flush=True)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as output:
                output.write(line + "\n")


class NullEventSink:
    def emit(self, event: Mapping[str, object]) -> None:
        return None
