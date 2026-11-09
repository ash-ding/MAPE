"""no_memory baseline — the calibration floor.

BUILD stores nothing. RETRIEVE returns `{}`, so the shared QA agent answers
from the question alone. Every score above this is what a memory bought; a
benchmark where a real memory system cannot beat it is telling you the
questions are answerable without memory (LoCoMo's category-5 "not mentioned"
questions are free points here, so its floor is well above zero).

It is a REAL baseline, not a stub: it runs through the same
`baselines.eval_harness` path, the same workflows and the same judge
as every other harness, so its numbers sit on the same axis.

    uv run python -m baselines.eval_harness \\
        --config baselines/config.example.yaml     # harness: no_memory

This baseline needs no dependencies of its own — no vendored source, no
models, no API calls of its own — so the repo-root venv runs it (there is no
`--project` to pass, unlike the vendored baselines).

`harness.py` next to this file is the SAME method written against the evolution loop's
contract (`evolution/memo_class.py`), for use as a evolution seed. Two entry shapes
for one trivial method: `memo.py` is what `eval_harness` evaluates, and
`harness.py` is what a evolution workspace copies in and runs in-container.
"""
from typing import Dict

from core.memo_class import MemoClass

# No method knobs: there is no method. Declared anyway so `--describe` and
# `resolve_memo_config` behave uniformly across the registry.
CONFIG_DEFAULTS = {}

# `arm: unified` has nothing to switch here — this baseline calls no model.
UNIFIED_MODEL_KEYS = {}


class NoMemoryMemo(MemoClass):
    async def build_memory_from_data(self, recorder) -> None:
        return None

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        return {}
