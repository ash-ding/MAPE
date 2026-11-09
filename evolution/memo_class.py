"""the evolution loop's memo contract — the base class every EVOLVED harness inherits.

This is the evolution loop's own surface, split from `core.memo_class` so
the evolution contract can gain documentation / optional hooks without
touching the contract the baselines are written against.

`MemoClass` here SUBCLASSES `core.memo_class.MemoClass`, so
everything downstream that type-checks against the common ABC
(`evolution/launch.py::_load_harness_class`,
`core/workflow.py`) accepts evolution harnesses unchanged, including ones
that import `core.memo_class` directly.

The contract (unchanged from common):

  Phase 1 — `build_memory_from_data(recorder)` is called ONCE per build call with
  `recorder.init` holding the data newly visible for that call (see the
  per-dataset shapes in the proposer system prompt). Build/extend your memory;
  choose your own ingestion granularity (write your own loop if you want to
  ingest in pieces).

  Phase 2 — `retrieve_memory_for_query(recorder)` is called once per query with
  `recorder.init` holding the query (+ benchmark-specific context). Return a
  Dict of retrieved MEMORY for the answer step — evidence, never the answer
  itself (a top-level key named like `answer` is rejected). MUST be read-only
  with respect to memory state (queries must not pollute memory). Tip: return
  `{"inline_memory_blocks": [str, ...]}` to control exactly how retrieved
  memory is rendered into the benchmark's prompt (blocks are joined verbatim);
  any other dict shape is serialized as one JSON block.

  Phase 2, optional — `compose_memory_context(recorder, retrieved)` organises
  what you retrieved into the CONTEXT the benchmark's QA prompt is rendered
  with. `recorder` is the one retrieve just saw (`recorder.init["query"]`,
  and `question_date` on LongMemEval); `retrieved` is what retrieve returned. You do not answer: the host renders that prompt
  around what you return, sends ONE call to the run's QA model, and the answer
  goes to the judge unchanged. Return `None` for the benchmark's own default
  rendering, a `str` for one block, or a `list` of blocks.

  What you return is measured as `context_tokens` and its size is paid for by
  every QA call, so a longer context is a real cost on the frontier — whether
  you spend it on memory or on telling the model how to use it. Model calls
  you make INSIDE this hook (re-ranking, compressing, a second retrieval
  pass) are charged to the `compose` phase and count as memory-system cost;
  use `Agent()` for them as build and retrieve do.

A fresh MemoClass instance is created per user/sample — no cross-user
state is possible.
"""

from __future__ import annotations

from core.recorder import Basic_Recorder  # noqa: F401  (re-export)
from core.memo_class import MemoClass as _CommonMemoClass


class MemoClass(_CommonMemoClass):
    """Evolution-target base class for evolution harnesses.

    Inherit from this and implement build (`build_memory_from_data`) and retrieve
    (`retrieve_memory_for_query`); optionally override compose
    (`compose_memory_context`):

        class MyHarness(MemoClass):
            async def build_memory_from_data(self, recorder) -> None: ...
            async def retrieve_memory_for_query(self, recorder) -> dict: ...
            async def compose_memory_context(self, recorder, retrieved):
                return None            # the default: the benchmark renders `retrieved`
    """
