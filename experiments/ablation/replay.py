"""Compose ablation of MAPE-A_LME (7723a2cc870a): replay its recorded retrieval,
render it with the benchmark's default context.

What this measures. The reported held-out runs of 7723a2cc870a recorded, for
every test question, exactly what its `retrieve_memory_for_query` returned.
This harness returns that same object for the same question and defines NO
`compose_memory_context`, so the host renders it the way it renders every
harness baseline's retrieval (`BaseWorkflow.default_context`: the retrieved
dict as JSON) and the run's QA model answers from it. Memory and retrieval are
therefore byte-identical to the reported run; only the compose stage is
removed. Run it under `contract.compose: false`.

`replay.json` (built by `build_replay.py` from the reported run's traces):
    {"benchmark": "longmemeval" | "locomo",
     "source": "<trace dir the entries came from>",
     "entries": {"<key>": [<retrieved_memory>, ...]}}
Keys: LongMemEval `json.dumps([query, question_date])`; LoCoMo
`json.dumps([speaker_a, speaker_b, query])`. A key that occurs more than once
(LoCoMo repeats a few questions within a conversation) holds one entry per
occurrence, served in order.

No model calls: build stores nothing and retrieve is a lookup, so this run's
build/retrieve cost is zero; add the reported run's build and retrieve cost
when comparing money. A question with no recorded retrieval raises, so a gap in
the replay table fails loudly instead of scoring as an empty-memory answer.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from evolution.memo_class import MemoClass

_REPLAY_PATH = Path(__file__).resolve().parent / "replay.json"
_TABLE: Dict[str, Any] = {}


def _table() -> Dict[str, Any]:
    if not _TABLE:
        _TABLE.update(json.loads(_REPLAY_PATH.read_text()))
    return _TABLE


def replay_key(benchmark: str, init: Dict[str, Any]) -> str:
    query = str(init.get("query", "") or "")
    if benchmark == "locomo":
        conv = init.get("conversation") or {}
        return json.dumps([conv.get("speaker_a", ""), conv.get("speaker_b", ""), query])
    return json.dumps([query, str(init.get("question_date", "") or "")])


class ReplayNoComposeMemo(MemoClass):
    def __init__(self, config=None):
        super().__init__(config)
        self._served: Dict[str, int] = defaultdict(int)

    async def build_memory_from_data(self, recorder) -> None:
        return None

    async def retrieve_memory_for_query(self, recorder) -> Dict[str, Any]:
        table = _table()
        init = getattr(recorder, "init", None) or {}
        key = replay_key(table["benchmark"], init)
        entries = table["entries"].get(key)
        if not entries:
            raise KeyError(f"no recorded retrieval for {key[:200]}")
        i = min(self._served[key], len(entries) - 1)
        self._served[key] += 1
        return entries[i]
