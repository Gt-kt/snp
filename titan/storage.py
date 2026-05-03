"""Shared JSON storage helpers for config and local trading state."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json_object(path: str | os.PathLike, *, strict: bool = False, logger=None) -> dict:
    """Read a JSON object from disk.

    Missing files return `{}`. Invalid files raise when `strict=True`; otherwise
    they log and return `{}`.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    try:
        with path_obj.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        raise ValueError(f"{path_obj} must contain a JSON object")
    except Exception as exc:
        if strict:
            raise ValueError(f"Failed to read {path_obj}: {exc}") from exc
        if logger:
            logger.warning(f"Failed to read {path_obj}: {exc}")
        return {}


def write_json_atomic(path: str | os.PathLike, data: Any, *, indent: int = 2) -> None:
    """Atomically write JSON data to disk."""
    path_obj = Path(path)
    parent = path_obj.parent if str(path_obj.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        os.replace(tmp_path, path_obj)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
