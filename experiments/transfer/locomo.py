"""AtomicLedgerMemo, evolved on LongMemEval (lme_qwen_acc step 20, 7723a2cc870a), run on
LoCoMo through a hand-written input adapter.

The evolved harness is memo_lme.py + src/, unchanged. This file only converts
LoCoMo's recorder.init into the LongMemEval shape before each hook (locomo_adapter.py, role_mode="user_assistant":
speaker_a -> user, speaker_b -> assistant, names kept in the text, so the harness's
round splitting at each user message still works).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from locomo_adapter import to_lme_init  # noqa: E402
from memo_lme import AtomicLedgerMemo as _Evolved  # noqa: E402


class LoCoMoAdaptedAtomicLedgerMemo(_Evolved):
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
