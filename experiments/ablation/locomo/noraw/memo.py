"""Retrieval ablation of MAPE-A_LME (7723a2cc870a) on LoCoMo, through the same input
adapter as its LoCoMo transfer run.

`ablation_lme.py` is the LongMemEval ablation harness (results/longmemeval_ablation_qwen),
unchanged; `mape_a.py` and `src/` are MAPE-A_LME's code, byte-identical; and
`locomo_adapter.py` is the hand-written LoCoMo -> LongMemEval input adapter of the
transfer run (role_mode="user_assistant"), unchanged. This file only converts
LoCoMo's recorder.init into the LongMemEval shape before each hook, exactly as the
transfer run's memo.py does, and points the memory cache at its own directory.

Arms (`variant.txt`): `full` (MAPE-A_LME unchanged; builds and caches the memory),
`noraw` (same sessions and notes, the five raw rounds not attached), and
`noraw_nocompose` (the `noraw` retrieval with no compose stage; the host renders it
with the benchmark default). See ablation_lme.py for each arm's exact definition.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ablation_lme as A  # noqa: E402
from locomo_adapter import to_lme_init  # noqa: E402

A.CACHE = Path("/hf-cache/mape_ablation_7723a2_locomo")
assert A.VARIANT in ("full", "noraw", "noraw_nocompose"), A.VARIANT


class LoCoMoRetrievalAblationMemo(A.RetrievalGranularityMemo):
    async def build_memory_from_data(self, recorder) -> None:
        orig = getattr(recorder, "init", None)
        recorder.init = to_lme_init(orig or {}, role_mode="user_assistant")
        try:
            return await super().build_memory_from_data(recorder)
        finally:
            recorder.init = orig

    async def retrieve_memory_for_query(self, recorder):
        orig = getattr(recorder, "init", None)
        recorder.init = to_lme_init(orig or {}, role_mode="user_assistant")
        try:
            return await super().retrieve_memory_for_query(recorder)
        finally:
            recorder.init = orig
