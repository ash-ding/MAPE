"""Compose transplant: a hand-built baseline's memory, MAPE-A_LME's compose.

Memory and retrieval are the baseline's own, replayed byte for byte from its
reported LongMemEval held-out run (`replay.json`). `adapters.py` reshapes each
recorded retrieval into the session/notes structure MAPE-A_LME's compose reads,
by one mechanical rule shared by all baselines (see its docstring). The compose
stage is MAPE-A_LME's (7723a2cc870a), inherited unchanged from `mape_a.py`,
which is that harness's `memo.py` under another name, together with its `src/`.

No model calls in build or retrieve; every compose and answer call goes to the
run's QA model, as for MAPE-A_LME itself. A question with no recorded retrieval
raises, so a gap in the replay table fails loudly.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import to_memorph_retrieval  # noqa: E402
from mape_a import AtomicLedgerMemo  # noqa: E402

_TABLE: Dict[str, Any] = {}


def _table() -> Dict[str, Any]:
    if not _TABLE:
        _TABLE.update(json.loads((Path(__file__).resolve().parent / "replay.json").read_text()))
    return _TABLE


class ComposeTransplantMemo(AtomicLedgerMemo):
    def __init__(self, config=None):
        super().__init__(config)
        self._served: Dict[str, int] = defaultdict(int)

    async def build_memory_from_data(self, recorder) -> None:
        return None

    async def retrieve_memory_for_query(self, recorder) -> Dict[str, Any]:
        table = _table()
        init = getattr(recorder, "init", None) or {}
        query = str(init.get("query", "") or "")
        qdate = str(init.get("question_date", "") or "")
        key = json.dumps([query, qdate])
        entries = table["entries"].get(key)
        if not entries:
            raise KeyError(f"no recorded retrieval for {key[:200]}")
        i = min(self._served[key], len(entries) - 1)
        self._served[key] += 1
        return to_memorph_retrieval(table["baseline"], entries[i], query, qdate)
