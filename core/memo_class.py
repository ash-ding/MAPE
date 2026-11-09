"""The memory-system contract every benchmark evaluates through.

A memory program is a `MemoClass` subclass with three lifecycle hooks:

  BUILD     `build_memory_from_data(recorder)`     required
  RETRIEVE  `retrieve_memory_for_query(recorder)`  required
  COMPOSE   `compose_memory_context(recorder, retrieved)`  optional

BUILD turns the conversation history into memory. RETRIEVE selects evidence
for one query and returns it as a dict. COMPOSE organises that evidence into
the context the benchmark's QA prompt is rendered with; its default returns
None, which means "render `retrieved` the way this benchmark always did".
The program never answers: the host renders the benchmark's own prompt around
the context and makes one call to the run's QA model, identical for every
program, and that answer goes to the judge unchanged.

Every hook receives a `Basic_Recorder` (`core/recorder.py`) whose `init` dict
holds the data for that call. For BUILD it is the newly visible history; for
RETRIEVE and COMPOSE it is the same dict, holding the query and, on
LongMemEval, the `question_date`. See `benchmarks/<name>/workflow.py` for the
exact field names.

What a program may not see: the benchmark's labels (LoCoMo `category`,
LongMemEval `question_type`) and gold evidence never reach any hook. The
benchmark's prompt still dispatches on them, host-side, and is sealed
(`core/qa.py`), so a design that wants question-type-specific behaviour has to
infer the type from the query itself.

The two required hooks are abstract, so a subclass that misses one (a typo'd
name included) fails at instantiation instead of silently evaluating as a
no-memory system. A fresh instance is created per user/sample by the
framework, so no cross-user state is possible.

Configuration: the framework may pass a `config` dict to the constructor and
each instance keeps its own copy at `self.config`. The class holds no defaults;
whoever constructs the program supplies the complete config (for the bundled
baselines, `baselines/eval_harness.py`). Evolved programs are never handed a
config, so a plain `def __init__(self)` override also works there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from core.recorder import Basic_Recorder


class MemoClass(ABC):
    #: Declared here for readers and type checkers; assigned per instance in
    #: __init__ — a class-level dict would be shared across users.
    config: Dict[str, Any]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Per-instance copy — instances must never share mutable config state
        # (the fresh-instance-per-user guarantee extends to configuration).
        self.config = dict(config) if config else {}

    # -------- Lifecycle hooks --------

    @abstractmethod
    async def build_memory_from_data(self, recorder: Basic_Recorder) -> None:
        """BUILD (Phase 1, REQUIRED). `recorder.init` holds the data newly
        visible for THIS call; accumulate state across calls and choose your
        own ingestion granularity. A system that stores nothing implements
        this as `return None`."""

    @abstractmethod
    async def retrieve_memory_for_query(self, recorder: Basic_Recorder) -> Dict:
        """RETRIEVE (Phase 2, REQUIRED). `recorder.init` holds the query (+
        benchmark-specific context). Return retrieved MEMORY — evidence for
        the question, never an answer to it (a top-level key named like one
        is rejected); `{"inline_memory_blocks": [str,...]}` controls inline
        rendering. MUST be read-only w.r.t. memory."""

    async def compose_memory_context(self, recorder: Basic_Recorder, retrieved: Dict
                                     ) -> Union[str, List[str], None]:
        """COMPOSE (Phase 2, OPTIONAL). Organise what `retrieve_memory_for_query`
        returned into the CONTEXT this benchmark's QA prompt is rendered with.

        `recorder` is the one RETRIEVE just saw: `recorder.init["query"]` is
        the question and, on LongMemEval, `recorder.init["question_date"]` is
        its date. `retrieved` is what RETRIEVE returned.

        You do not answer the question. The host renders the benchmark's own
        prompt around whatever you return and sends it to the run's QA model;
        that answer goes to the judge unchanged.

        Return:
          * `None`  — use the benchmark's default rendering of `retrieved`.
          * `str`   — one context block.
          * `list`  — several blocks, joined by the benchmark with a blank line.

        Whatever you return is measured as `context_tokens` and priced into
        the `answer` phase, because it is what the QA call carries. Model
        calls you make HERE — re-ranking, compressing, a second retrieval
        pass — are charged to the `compose` phase and count as memory-system
        cost. Use `Agent()` for them, as build and retrieve do; the model is
        the run's and the policy refuses any other.
        """
        return None
