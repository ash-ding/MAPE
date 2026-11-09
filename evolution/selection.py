"""Frontier record store.

`Frontier` records every evaluated harness with its objectives (per-dataset
`accuracy_<ds>`, `score_max_<ds>`, the cost axes) and the `parent_ids` the proposer
drew from. It is a persisted population, NOT a selection algorithm.

There is deliberately NO algorithmic parent selection (Meta-Harness paper
alignment): the search loop only `add()`s + `save()`s, and the proposer
reads frontier.json directly to pick which prior(s) to build on. Each
dataset is recorded independently; there is no cross-benchmark mean.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Entry:
    id: str
    objectives: Dict[str, float]
    parent_ids: List[str] = field(default_factory=list)
    # v5 fields — None for entries written by older versions.
    content_hash: Optional[str] = None
    created_at: Optional[str] = None


class Frontier:
    """Persisted population of evaluated harnesses (a record store)."""

    def __init__(self, entries: Optional[List[Entry]] = None):
        self._entries: List[Entry] = list(entries or [])

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def add(self, entry: Entry) -> None:
        for i, e in enumerate(self._entries):
            if e.id == entry.id:
                self._entries[i] = entry
                return
        self._entries.append(entry)

    def get(self, id: str) -> Optional[Entry]:
        for e in self._entries:
            if e.id == id:
                return e
        return None

    def all_entries(self) -> List[Entry]:
        return list(self._entries)

    def remove_by_ids(self, ids) -> int:
        """Drop entries whose id is in `ids` (set or any container).

        Returns the count removed. Used by the orchestrator's resume cleanup.
        """
        ids = set(ids)
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id not in ids]
        return before - len(self._entries)

    # ------------------------------------------------------------------
    # Persistence (with backward-compat for v3 frontier.json)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        out_entries = []
        for e in self._entries:
            d: Dict[str, object] = {
                "id": e.id,
                "objectives": e.objectives,
                "parent_ids": list(e.parent_ids),
            }
            if e.content_hash is not None:
                d["content_hash"] = e.content_hash
            if e.created_at is not None:
                d["created_at"] = e.created_at
            out_entries.append(d)
        return {"entries": out_entries}

    @classmethod
    def from_dict(cls, data: dict) -> "Frontier":
        entries: List[Entry] = []
        for e in data.get("entries", []):
            parent_ids = list(e.get("parent_ids") or [])
            entries.append(Entry(
                id=e["id"],
                objectives=dict(e["objectives"]),
                parent_ids=parent_ids,
                content_hash=e.get("content_hash"),
                created_at=e.get("created_at"),
            ))
        return cls(entries=entries)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Frontier":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
