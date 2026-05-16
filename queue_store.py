"""File-backed review queue.

Each item is one parsed contact awaiting review:
  { id, created_at, source, fields, crm, raw_text, preview, rotation,
    qr_codes, status: "pending"|"approved"|"deleted", push_result?, choices? }
"""
from __future__ import annotations
import json
import os
import time
import threading
import uuid
from datetime import datetime, timezone

_QUEUE_FILE = os.path.join(os.path.dirname(__file__), "queue.json")
_lock = threading.Lock()


def _load() -> list:
    if not os.path.exists(_QUEUE_FILE):
        return []
    try:
        with open(_QUEUE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(items: list) -> None:
    tmp = _QUEUE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _QUEUE_FILE)


def list_items(status: str | None = None) -> list:
    with _lock:
        items = _load()
    if status:
        items = [it for it in items if it.get("status") == status]
    return items


def counts() -> dict:
    items = list_items()
    out = {"all": len(items), "pending": 0, "approved": 0, "deleted": 0}
    for it in items:
        s = it.get("status", "pending")
        out[s] = out.get(s, 0) + 1
    return out


def add_items(new: list) -> list:
    """Each entry must already have at least `fields`. Returns added items
    with `id` and `created_at` filled in."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _lock:
        items = _load()
        for item in new:
            item.setdefault("id", uuid.uuid4().hex[:12])
            item.setdefault("status", "pending")
            item.setdefault("created_at", now)
        items.extend(new)
        _save(items)
    return new


def update_item(item_id: str, patch: dict) -> dict | None:
    with _lock:
        items = _load()
        for it in items:
            if it.get("id") == item_id:
                it.update(patch)
                _save(items)
                return it
    return None


def delete_item(item_id: str) -> bool:
    with _lock:
        items = _load()
        before = len(items)
        items = [it for it in items if it.get("id") != item_id]
        if len(items) != before:
            _save(items)
            return True
    return False


def get_item(item_id: str) -> dict | None:
    for it in list_items():
        if it.get("id") == item_id:
            return it
    return None


def purge_all() -> None:
    with _lock:
        _save([])
