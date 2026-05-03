"""Append-only JSONL trade journal for manual trading actions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_JOURNAL_FILE = "trade_journal.jsonl"


def append_journal_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    path: str | os.PathLike = DEFAULT_JOURNAL_FILE,
) -> dict[str, Any]:
    """Append one immutable journal event and return the written record."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with path_obj.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return record
