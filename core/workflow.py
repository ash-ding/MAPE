"""Benchmark-agnostic per-user two-phase execution scaffold.

Subclass `BaseWorkflow` for each benchmark; implement the abstract hooks.
The base class handles:

  - concurrency over users (semaphore) + per-user Exception capture
  - QA progress tracking (`_QAProgressTracker`)
  - Phase 1 dispatch: single build_memory_from_data call per visible-data batch
  - Phase 2 QA loop with per-QA failure isolation (a retrieve/answer error
    records a score=0 step with a `[Phase2_*_ERROR]` marker and continues;
    the per-user error tally lands in `recorder.failure_info`, which the
    orchestrator's sanity gate inspects)
  - `save_full_traces` (one JSON per user under traces/)
  - per-user isolation (fresh MemoClass instance per user)
  - LLM cost accounting: each hook call runs inside a `core.tokens.phase`
    block (build / retrieve / compose / answer / judge) so tokens are
    attributed to what they were spent on, and the composed context is
    measured directly (see `_measure_context_tokens`)

To add a new benchmark, subclass `BaseWorkflow` and override the hooks listed
under "subclass hooks" below. See `LoCoMoWorkflow` in
`benchmarks/locomo/workflow.py` for a concrete implementation.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from core import pricing, tokens
from core.memo_class import MemoClass
from core.qa import QAContext
from core.logger import get_logger
from core.recorder import Basic_Recorder

log = get_logger("main")


class _MemoryEncoder(json.JSONEncoder):
    """JSON encoder for common non-serializable types harnesses put into
    `steps[].retrieved_memory` (sets, Counters, datetimes, numpy scalars...).
    Used by `save_full_traces` — anything still unknown degrades to repr()."""

    def default(self, obj):
        if isinstance(obj, set):
            return sorted(obj) if all(isinstance(x, str) for x in obj) else list(obj)
        if isinstance(obj, (Counter, defaultdict)):
            return dict(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, '__float__'):
            return float(obj)
        if hasattr(obj, '__int__'):
            return int(obj)
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


_JSON_KEY_NATIVE = (str, int, float, bool, type(None))


def _json_safe_keys(obj):
    """Recursively coerce dict keys that aren't JSON-native (str/int/float/bool/None)
    to their `str(...)` form. Returns a new structure; leaves values alone.

    Why this can't live in `_MemoryEncoder.default()`: Python's `json.dump`
    enforces the dict-key type constraint BEFORE calling the encoder's
    `default()` hook, so a custom encoder can't intercept e.g. tuple keys —
    they must be coerced via pre-processing. Harnesses commonly use
    tuple-keyed inverted indices (`Dict[Tuple[int, int], ...]` for
    (year, month), (app, api), etc.) in retrieved_memory; without this,
    every such harness's per-step retrieval record would be lost.
    """
    if isinstance(obj, dict):
        return {
            (k if isinstance(k, _JSON_KEY_NATIVE) else str(k)): _json_safe_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe_keys(x) for x in obj]
    return obj


class _QAProgressTracker:
    """Shared counter for QA progress across concurrent users."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self._lock = asyncio.Lock()
        self._last_pct = -1

    async def increment(self) -> None:
        async with self._lock:
            self.completed += 1
            pct = int(self.completed / self.total * 100) if self.total > 0 else 100
            if pct // 10 > self._last_pct // 10:
                self._last_pct = pct
                log.info(f"[Phase 2] QA Progress: {self.completed}/{self.total} ({pct}%)")


# ---------------------------------------------------------------------------
# BaseWorkflow
# ---------------------------------------------------------------------------

class BaseWorkflow(ABC):
    """Per-user two-phase scheduler. Subclass per benchmark."""

    # ---- Subclass overrides (class attributes) ----

    recorder_class: Type[Basic_Recorder]
    """Recorder class used for both the accumulator recorder and per-QA
    retrieve recorders. Must be set by subclass."""

    _phase1_item_label: str = "items"
    """Human-readable label for init_data elements, used in Phase 1 log
    messages (LoCoMo uses "sessions")."""

    _default_qa_per_user_hint: int = 178
    """Progress-bar hint when eval_n_qa is None (purely cosmetic)."""

    judge_score_max: int = 10
    """Maximum score the judge can produce for this benchmark. Single
    source of truth for the score range — `_make_judge` reads it, and the
    host orchestrator reads it (via score.json::score_max) to normalize
    each benchmark's accuracy to [0, 1] before averaging across benchmarks.
    Both benchmarks override it to 1 (0/1 binary); the class default of 10
    only applies to the generic Judge prompt."""

    def _qa_per_user_estimate(self, sample_spec: Optional[Dict]) -> int:
        """Expected number of QAs per user/sample — drives the Phase-2 progress-tracker total.

        Default honours the sample spec's `n_qa` (falling back to the
        `eval_n_qa` ctor param). Benchmarks where the QA count per sample is
        fixed (LongMemEval, always 1) override this."""
        n_qa = (sample_spec or {}).get("n_qa")
        if n_qa is not None:
            return int(n_qa)
        return self.eval_n_qa or self._default_qa_per_user_hint

    def _new_memo(self):
        """The ONE way a memo instance is constructed — with `memo_config`
        when the caller supplied one (the parameterized harness baselines),
        zero-arg otherwise (evolution-evolved harnesses)."""
        if self.memo_config is not None:
            return self.memo_class(config=self.memo_config)
        return self.memo_class()

    def _phase1_item_count(self, init_data) -> "int | str":
        """Return the count of "items" in Phase 1's init_data — used purely
        for the end-of-Phase-1 log message. Default: `len(init_data)` when
        sized (works for list-shaped init). Subclasses whose
        init_data is a container of sub-items should override (e.g.
        LoCoMo returns the session count, not the dict key count)."""
        try:
            return len(init_data)  # type: ignore[arg-type]
        except TypeError:
            return "?"

    # ---- Constructor ----

    def __init__(
        self,
        memo_class: Type[MemoClass],
        model: str,
        max_logs: Optional[int] = None,
        eval_n_qa: Optional[int] = None,
        judge_model: str = "gpt-5-mini",
        memo_config: Optional[Dict] = None,
    ):
        self.memo_class = memo_class
        # Optional constructor config for the per-user memo instances. None →
        # zero-arg instantiation (evolution-evolved harnesses; their settings are
        # baked into the generated code). A dict → passed as memo_class(config=...)
        # (the parameterized harness baselines).
        self.memo_config = memo_config
        self.memo_sha = ""                  # set by caller (for logging only)
        self.status = "search"
        self.max_logs = max_logs
        self.eval_n_qa = eval_n_qa
        self.judge_model = judge_model
        # Output directory — set by launch.py before run_all_users. Full
        # traces go to {output_run_dir}/traces/.
        self.output_run_dir: Optional[Path] = None
        # Lazy-constructed Judge (per core.metric.Judge). Subclasses can
        # override `_make_judge` to customize prompt / score range.
        self._judge_instance = None

        # Parse "model/reasoning_effort" format (e.g. "gpt-4o-mini/low")
        if "/" in model:
            self.model, self.reasoning_effort = model.split("/", 1)
        else:
            self.model = model
            self.reasoning_effort = None

    # ---- Subclass hooks (must implement) ----

    @abstractmethod
    async def load_user_data(
        self, user_dir: str, eval_n_qa: Optional[int], sample_seed: Optional[str] = None
    ) -> Tuple[Any, List[Dict]]:
        """Load raw data for one user.

        `sample_seed` (sample_spec["sample_seed"], via
        core.sampling.combine_seed) seeds the per-user QA subset for
        benchmarks that sample one. None (default) seeds on user_dir alone.

        Returns (init_data, qa_pairs):
          init_data: opaque payload passed back to `phase1_log_init` and
                     `build_query_recorder_init`. For LongMemEval, the
                     session list.
          qa_pairs:  list of QA dicts. Each is expected to have at least
                     `query`, `reference`, and an optional `metadata` dict.
        """

    @abstractmethod
    async def phase1_log_init(self, recorder: Basic_Recorder, chunk: Any) -> None:
        """Populate `recorder.init` with a Phase 1 chunk of init_data.

        Typically `await recorder.log_init(chunk)`.
        """

    @abstractmethod
    def build_query_recorder_init(self, init_data: Any, qa: Dict) -> Dict:
        """Build the `recorder.init` dict for a single Phase 2 retrieve call.

        For LongMemEval: {"sessions": init_data, "query": ..., "question_date": ...}
        """

    @abstractmethod
    def build_qa_prompt(
        self,
        query: str,
        context: Any,
        qa_metadata: Dict,
        reference: str = "",
    ) -> List[Dict]:
        """Two-message prompt (system + user) for the QA agent.

        `context` is what the memo's `compose_memory_context` returned (or
        this benchmark's `default_context(retrieved)` when it returned None):
        one block, or a list of blocks this benchmark knows how to join. It
        is already the text the prompt carries — do not re-serialize a dict.

        `reference` is the gold answer. No current benchmark feeds it into
        the prompt (gold must not leak to the QA agent); the parameter is
        kept on the hook signature for benchmarks whose official protocol
        requires reference-derived prompt content.

        `qa_metadata` is the dict returned by `build_qa_metadata(qa)` for this
        step — benchmarks that need extra context (e.g. LongMemEval uses the
        question_date for temporal reasoning) can pull fields from here.
        Simple benchmarks can ignore it.
        """

    @abstractmethod
    def extract_relevant_context(self, qa: Dict, init_data: Any) -> Any:
        """Ground-truth context relevant to this QA (logged in the trace).

        For LoCoMo: the evidence turns listed in qa['metadata'].
        """

    @abstractmethod
    def build_qa_metadata(self, qa: Dict) -> Dict:
        """Per-step metadata to persist in the trace and to drive the
        benchmark's own prompt (host-side). None of it reaches the memo's
        hooks: labels (LoCoMo category, LongMemEval question_type) and gold
        evidence stay with the host. What the hooks see is `recorder.init`,
        built by `build_query_recorder_init`. See core/qa.py."""

    @abstractmethod
    async def log_qa_step(
        self,
        recorder: Basic_Recorder,
        query: str,
        predicted: str,
        reference: str,
        score: int,
        judge_reason: str,
        qa_metadata: Dict,
        retrieved_memory: Dict,
        relevant_context: Any,
    ) -> None:
        """Call `recorder.log_step(...)` with the benchmark-specific field
        names (LoCoMo stores `relevant_context` as `relevant_turns`)."""

    def aggregate_extra_metrics(self, recorder_list: List[Any]) -> Dict[str, Any]:
        """Benchmark-specific aggregate metrics, merged into score.json under
        `extra_metrics`. Default: none.

        This is the extension point for anything a single benchmark wants to
        report that is NOT the promotion signal — the shared score-summary
        builder (`core.evaluate._build_score_json`) stays benchmark-agnostic,
        so per-benchmark reporting never turns into a special case inside it.

        Contract for overrides:
        - REPORTING ONLY. `benchmark_eval_score` is the promotion signal and is
          computed from the judge; nothing returned here may feed back into it.
        - Must never raise. A reporting metric that crashes must not take down
          an otherwise-valid eval — return {} instead (the caller also guards).
        - Return a JSON-serializable dict keyed by a namespace you own, e.g.
          {"locomo_lexical": {...}}.
        """
        return {}

    # ---- Memory tokens (how much the memory added to the QA prompt) ----
    #
    # Measured DIFFERENTIALLY: tokens(prompt with the retrieved memory) minus
    # tokens(the same prompt built with nothing retrieved). That is exactly
    # what the memory contributed as it was actually rendered, and it needs no
    # special case per benchmark — each prompt module joins and templates its
    # payload differently, so tokenizing the raw `retrieved` payload would
    # drift from what really landed in the prompt. Costs one extra prompt
    # build per query: pure string work, no API call.

    _context_token_warned = False

    def default_context(self, retrieved: Dict) -> Any:
        """This benchmark's own rendering of `retrieved`, used when the memo's
        `compose_memory_context` returns None.

        The default is what LoCoMo and LongMemEval did inline before the hook
        existed.
        """
        if not retrieved:
            return ""
        return json.dumps(retrieved, indent=2, ensure_ascii=False, default=str)

    def context_to_text(self, context: Any) -> str:
        """Flatten a composed context to the text the prompt carries.

        A benchmark that joins blocks differently overrides this; the generic
        join is a blank line.
        """
        if context is None:
            return ""
        if isinstance(context, str):
            return context
        if isinstance(context, (list, tuple)):
            return "\n\n".join(str(b) for b in context if str(b).strip())
        return str(context)

    def _measure_context_tokens(self, recorder: Basic_Recorder, context: Any) -> Optional[int]:
        """Tally this query's context tokens onto `recorder`; returns the count.

        Counted on the composed context itself, in the QA model's encoding —
        no differential, because the block is explicit now. It covers
        EVERYTHING the memo put there, memory and any instructions alike:
        both are carried by every QA call and both are paid for.

        Never raises: an unmeasurable context degrades to an unmeasured step
        (logged once), not a failed evaluation.
        """
        try:
            n = tokens.count_text_tokens(self.context_to_text(context), self.model)
        except Exception as exc:
            if not type(self)._context_token_warned:
                type(self)._context_token_warned = True
                log.warning(
                    f"context-token measurement unavailable for "
                    f"{type(self).__name__} ({type(exc).__name__}: {exc}); "
                    f"context_tokens will be reported as 0"
                )
            return None
        recorder.context_tokens_total = getattr(recorder, "context_tokens_total", 0) + n
        recorder.context_tokens_n = getattr(recorder, "context_tokens_n", 0) + 1
        return n

    @staticmethod
    def _stamp_context_tokens(recorder: Basic_Recorder, context_tokens: Optional[int]) -> None:
        """Record this step's context size in the trace."""
        if context_tokens is None or not recorder.steps:
            return
        recorder.steps[-1]["context_tokens"] = context_tokens

    def _make_judge(self):
        """Construct the judge for this workflow. Override to customize
        prompt template / score range / model — e.g. for benchmark-specific
        judges (LoCoMo binary, LongMemEval per-question-type, ...)."""
        from core.metric import Judge
        # timeout / max_retries deliberately NOT overridden here — the Judge
        # defaults in core/metric.py are the single source of truth.
        return Judge(model=self.judge_model)

    async def judge(
        self,
        query: str,
        predicted: str,
        reference: str,
        qa_metadata: Optional[Dict] = None,
    ) -> Tuple[int, str]:
        """Default impl uses self._make_judge() lazily. Subclasses with a
        single benchmark-wide prompt only need to override `_make_judge`.

        `qa_metadata` carries per-question fields (question_type, etc.) and
        is unused by the default impl; LongMemEval overrides this method to
        dispatch among multiple prompts based on question_type.
        """
        if self._judge_instance is None:
            self._judge_instance = self._make_judge()
        return await self._judge_instance.score(query, predicted, reference)

    # ---- Main entry point: run all users ----

    async def run_all_users(
        self,
        task_list: List[str],
        *,
        sample_spec: Optional[Dict] = None,
        max_sample_concurrent: int = 6,
    ) -> Tuple[List[Any], int]:
        """Run Phase 1 + Phase 2 for every user in task_list.

        `sample_spec` carries the benchmark-native size fields (e.g. n_qa)
        and the sample seed.

        task_list is expected to be pre-capped by the caller (a deterministic
        prefix of the split — no resampling here).

        Returns (results_list, results_len). Failed users appear as Exception
        objects (with `.user_id` attribute) in the list. The returned length
        equals len(results_list); downstream filters out Exceptions itself.
        """
        qa_per_user = self._qa_per_user_estimate(sample_spec)
        total_qa = len(task_list) * qa_per_user
        qa_tracker = _QAProgressTracker(total_qa)
        log.info(
            f"Starting evaluation: {len(task_list)} users × ~{qa_per_user} QA = ~{total_qa} total"
        )

        semaphore = asyncio.Semaphore(max_sample_concurrent)

        async def run_one(user_dir: str):
            async with semaphore:
                try:
                    # One cost scope per user. Set inside this task, so it
                    # covers exactly this user's LLM calls even though users
                    # overlap (each asyncio task carries its own context, and
                    # both `to_thread` and the patched ThreadPoolExecutor copy
                    # it into worker threads).
                    with tokens.tally_calls() as tally:
                        recorder = await self.run_single_user(
                            user_dir, sample_spec=sample_spec, qa_tracker=qa_tracker
                        )
                    self._stamp_cost(recorder, tally)
                    return recorder
                except Exception as exc:
                    user_tag = user_dir[-15:]
                    log.error(f"[ERROR] User {user_tag} failed: {exc}\n{traceback.format_exc()}")
                    try:
                        exc.user_id = user_tag
                    except Exception:
                        pass
                    return exc

        t0 = time.time()
        results = await asyncio.gather(*[run_one(u) for u in task_list])
        elapsed = time.time() - t0
        valid_count = sum(1 for r in results if not isinstance(r, Exception))
        log.info(f"Evaluation complete: {valid_count}/{len(task_list)} users succeeded in {elapsed:.1f}s")
        self._raise_on_judge_outage(results)
        return list(results), len(results)

    # Fraction of judged steps whose judge failed at transport level above
    # which the whole eval aborts instead of publishing a fake near-0 score.
    JUDGE_OUTAGE_FRACTION = 0.5

    def _raise_on_judge_outage(self, results: List[Any]) -> None:
        """Abort the eval when the judge itself was down for most steps.

        Isolated judge failures score 0 per item. Without this guard, an
        OpenAI outage during Phase 2 produces a legitimate-looking near-0
        score.json that would enter the search as a real score. Raising here
        means no score.json
        is written — the orchestrator records an eval failure (retryable)
        rather than a score.

        Counts only steps the judge actually ran on (Phase2_* error steps
        are harness-side, judge never invoked). Needs >= 2 judged steps so
        a single flaky call in a 1-QA sanity run doesn't abort.
        """
        judged = 0
        failed = 0
        for r in results:
            if isinstance(r, Exception):
                continue
            for s in getattr(r, "steps", []):
                reason = str(s.get("judge_reason", ""))
                if reason.startswith("[Phase2_"):
                    continue
                judged += 1
                # "Judge error:" = core.metric.Judge transport exhaustion;
                # "JUDGE_ERROR:" = a benchmark judge's own failure marker.
                if reason.startswith("Judge error:") or "JUDGE_ERROR:" in reason:
                    failed += 1
        if judged >= 2 and failed / judged > self.JUDGE_OUTAGE_FRACTION:
            raise RuntimeError(
                f"judge outage: {failed}/{judged} judged steps failed at the "
                f"transport level — aborting eval instead of publishing a "
                f"fake near-0 score (official protocol aborts on judge failure)"
            )

    #: A RETRIEVE result may carry memory in any shape, but not the answer:
    #: a top-level key named like one is how an answerer hid inside retrieve.
    _ANSWER_LIKE_KEY = re.compile(r"answer", re.IGNORECASE)

    @classmethod
    def check_retrieved(cls, retrieved: Any) -> None:
        """Raise if `retrieved` looks like an answer rather than memory."""
        if not isinstance(retrieved, dict):
            return
        bad = [k for k in retrieved if isinstance(k, str) and cls._ANSWER_LIKE_KEY.search(k)]
        if bad:
            raise ValueError(
                f"retrieve_memory_for_query returned answer-like key(s) {bad}: RETRIEVE "
                f"returns memory (evidence for the question), never an answer to it. "
                f"The host does the answering, from the context "
                f"compose_memory_context builds."
            )

    @staticmethod
    def overrides_compose(memo: MemoClass) -> bool:
        """True when this memo composes its own context."""
        return (type(memo).compose_memory_context
                is not MemoClass.compose_memory_context)

    @staticmethod
    def _answer_snapshot() -> Optional[Dict[str, int]]:
        """The ambient per-user tally's answer-phase counters right now, or
        None outside a tally scope. Taken before and after the answer hook,
        the difference is what THIS query's answering cost."""
        tally = tokens.current_tally()
        return tally.phase(tokens.ANSWER) if tally is not None else None

    @staticmethod
    def _answer_stats(before: Optional[Dict[str, int]], after: Optional[Dict[str, int]],
                      seconds: float) -> Optional[Dict[str, Any]]:
        if before is None or after is None:
            return None
        return {
            "answer_calls": int(after["calls"] - before["calls"]),
            "answer_prompt_tokens": int(after["prompt_tokens"] - before["prompt_tokens"]),
            "answer_completion_tokens": int(after["completion_tokens"] - before["completion_tokens"]),
            "answer_seconds": round(float(seconds), 3),
        }

    @staticmethod
    def _stamp_answer_stats(recorder: Basic_Recorder, stats: Optional[Dict[str, Any]]) -> None:
        """Record what the host's QA call cost on the step dict. One call
        per question now, so `answer_prompt_tokens` is comparable across
        memos: it is the benchmark's template plus steps[].context_tokens
        plus the query."""
        if stats is None or not recorder.steps:
            return
        recorder.steps[-1].update(stats)

    def _stamp_cost(self, recorder: Basic_Recorder, tally: "tokens.CallTally") -> None:
        """Record what THIS user's memory system cost, on their recorder.

        Two readings, both per user:

        * `cost_<phase>_<counter>` for build / retrieve / answer — LLM tokens
          only, the counters the older token metrics are defined on
          (`cost_tokens_per_query` charges build + retrieve + the memory's
          share of the QA prompt).
        * `cost_usd_<phase>` / `cost_usd_total` — every call in build,
          retrieve, compose and answer priced by model (core.pricing),
          embeddings included. This is what the memo cost to run, in money:
          `compose` is the memo organising its context, and `answer` is the
          host's single QA call, whose size the memo chose.
          `cost_embedding_tokens` reports the embedding share separately.

        `judge` is evaluation overhead and in neither. A model the table
        cannot price (only possible with the model policy off) is priced at
        the run's unified LLM rate and named in `cost_unpriced_models`.
        A user who raised before returning a recorder is never stamped; those
        users are dropped from the cost metric by `core.evaluate` exactly as
        they are dropped from the score.
        """
        for phase_name in (tokens.BUILD, tokens.RETRIEVE, tokens.COMPOSE, tokens.ANSWER):
            counters = tally.phase(phase_name, exclude_embeddings=True)
            for counter, value in counters.items():
                setattr(recorder, f"cost_{phase_name}_{counter}", int(value))
        from core import model_policy
        fallback = model_policy.unified_llm() or self.model
        usd = {tokens.BUILD: 0.0, tokens.RETRIEVE: 0.0,
               tokens.COMPOSE: 0.0, tokens.ANSWER: 0.0}
        embedding_tokens = 0
        unpriced = set()
        for (ph, model), c in tally.snapshot_by_model().items():
            if ph not in usd:
                continue
            if tokens.is_embedding_model(model):
                embedding_tokens += int(c["prompt_tokens"])
            amount = pricing.usd(model, c["prompt_tokens"], c["completion_tokens"])
            if amount is None:
                unpriced.add(model)
                amount = pricing.usd(fallback, c["prompt_tokens"], c["completion_tokens"]) or 0.0
            usd[ph] += amount
        for ph, amount in usd.items():
            setattr(recorder, f"cost_usd_{ph}", float(amount))
        recorder.cost_usd_total = float(sum(usd.values()))
        recorder.cost_embedding_tokens = int(embedding_tokens)
        recorder.cost_unpriced_models = sorted(unpriced)

    # ---- Single-user execution ----

    async def run_single_user(
        self,
        user_dir: str,
        *,
        sample_spec: Optional[Dict] = None,
        qa_tracker: Optional[_QAProgressTracker] = None,
    ) -> Basic_Recorder:
        """Run Phase 1 (update) then Phase 2 (retrieve + QA) for one user."""
        # Agent is lazy-imported so module load stays cheap & avoids cycles.
        from core.llm import Agent

        user_tag = user_dir[-15:]
        # Before BUILD, not after: build's extraction calls are the first
        # thing a harness does, and they are exactly the ones an attribution
        # needs. Setting this later filed all 20 of them under _unscoped.
        tokens.set_user(user_tag)

        qa_size = (sample_spec or {}).get("n_qa", self.eval_n_qa)
        sample_seed = (sample_spec or {}).get("sample_seed")
        init_data, qa_pairs = await self.load_user_data(user_dir, qa_size, sample_seed=sample_seed)

        memo = self._new_memo()

        t1 = time.time()
        # Wrapped at the CALL SITE, not inside `_phase1_update`: benchmarks
        # override that method (LoCoMo's init_data is a conversation dict,
        # not a list) and an override does not inherit a phase block living
        # in the base implementation. A real simplemem/LoCoMo run filed
        # 151k build tokens under `other` before this moved out here.
        # Everything spent inside is the cost of HAVING the memory.
        with tokens.phase(tokens.BUILD):
            await self._phase1_update(memo, init_data)
        t1_elapsed = time.time() - t1
        item_count = self._phase1_item_count(init_data)
        log.info(
            f"[Phase 1] User {user_tag} update complete "
            f"({item_count} {self._phase1_item_label}, {t1_elapsed:.1f}s)"
        )

        t2 = time.time()
        recorder = self.recorder_class()
        recorder.user_id = user_tag
        await self.phase1_log_init(recorder, init_data)

        # timeout / max_retries deliberately NOT overridden — Agent defaults
        # in core/llm.py are the single source of truth.
        agent = Agent(system_prompt="", model=self.model)

        # Phase-2 error tally → recorder.failure_info. The sanity gate
        # (evolution/orchestrator.py::_collect_sanity_errors) fails a harness
        # whose per_user entries carry a non-empty failure_info, so a
        # broken retrieve_memory_for_query is caught at sanity size instead of
        # silently burning a full evaluation on all-zero steps.
        n_retrieve_err = 0
        n_answer_err = 0
        first_err = ""

        for qa in qa_pairs:
            # The index this question's step will get in recorder.steps, so a
            # level-2 call record joins to the right question (core.trace).
            tokens.set_query(len(recorder.steps))
            retrieve_recorder = self.recorder_class()
            retrieve_recorder.init = self.build_query_recorder_init(init_data, qa)

            try:
                with tokens.phase(tokens.RETRIEVE):
                    retrieved = await memo.retrieve_memory_for_query(retrieve_recorder)
                self.check_retrieved(retrieved)
            except Exception as exc:
                # Per-QA failure isolation: log a score=0 step and continue
                # to the next QA — a single bad query must not silently
                # truncate the rest of the user's QA loop (which would
                # under-report the harness's real capability).
                n_retrieve_err += 1
                first_err = first_err or f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning(
                    f"retrieve_memory_for_query failed for {user_tag} on QA "
                    f"{len(recorder.steps)+1}/{len(qa_pairs)} "
                    f"({type(exc).__name__}: {str(exc)[:120]}) "
                    f"— recording score=0 and continuing"
                )
                await self.log_qa_step(
                    recorder=recorder,
                    query=qa["query"],
                    predicted="",
                    reference=qa.get("reference", ""),
                    score=0,
                    judge_reason=(
                        f"[Phase2_Retrieve_ERROR] {type(exc).__name__}: "
                        f"{str(exc)[:200]}"
                    ),
                    qa_metadata=self.build_qa_metadata(qa),
                    retrieved_memory={},
                    relevant_context=self.extract_relevant_context(qa, init_data),
                )
                if qa_tracker:
                    await qa_tracker.increment()
                continue

            relevant_context = self.extract_relevant_context(qa, init_data)
            qa_metadata = self.build_qa_metadata(qa)
            reference = qa.get("reference", "")
            # COMPOSE — the memo turns what it retrieved into the context this
            # benchmark's prompt is rendered with. Returning None means "use
            # the benchmark's own rendering", which is what every benchmark
            # did before this hook existed, so a memo that does not override
            # it gets a byte-identical prompt. Model calls made in here are
            # the memo's, charged to the `compose` phase.
            try:
                with tokens.phase(tokens.COMPOSE):
                    context = await memo.compose_memory_context(retrieve_recorder, retrieved)
            except Exception as exc:
                n_answer_err += 1
                first_err = first_err or f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning(f"compose_memory_context failed for {user_tag}: {exc}")
                await self.log_qa_step(
                    recorder=recorder,
                    query=qa["query"],
                    predicted="",
                    reference=qa.get("reference", ""),
                    score=0,
                    judge_reason=(
                        f"[Phase2_Compose_ERROR] {type(exc).__name__}: "
                        f"{str(exc)[:200]}"
                    ),
                    qa_metadata=qa_metadata,
                    retrieved_memory=retrieved,
                    relevant_context=relevant_context,
                )
                if qa_tracker:
                    await qa_tracker.increment()
                continue
            if context is None:
                context = self.default_context(retrieved)
            # What the QA call will carry, for every memo alike.
            context_tokens = self._measure_context_tokens(recorder, context)

            answer_before, answer_t0 = self._answer_snapshot(), time.monotonic()
            try:
                with tokens.phase(tokens.ANSWER):
                    # The HOST answers: this benchmark's standard prompt
                    # rendered around the memo's context, one call, the run's
                    # QA model. `reference` is bound here and never reaches
                    # the memo — it is the gold answer.
                    prompt = self.build_qa_prompt(
                        qa["query"], context, qa_metadata, reference=reference)
                    answer = await QAContext(
                        agent=agent, reasoning_effort=self.reasoning_effort,
                    ).ask(prompt)
                    answer = "" if answer is None else str(answer)
            except Exception as exc:
                # QA-agent transport failure (retries already exhausted in
                # core.llm). Mirror the retrieve-error path: record a
                # marked score=0 step instead of judging an empty answer —
                # otherwise an API outage is indistinguishable in traces
                # from a genuinely wrong answer.
                n_answer_err += 1
                first_err = first_err or f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning(f"Agent ask failed for {user_tag}: {exc}")
                await self.log_qa_step(
                    recorder=recorder,
                    query=qa["query"],
                    predicted="",
                    reference=qa.get("reference", ""),
                    score=0,
                    judge_reason=(
                        f"[Phase2_Answer_ERROR] {type(exc).__name__}: "
                        f"{str(exc)[:200]}"
                    ),
                    qa_metadata=qa_metadata,
                    retrieved_memory=retrieved,
                    relevant_context=relevant_context,
                )
                self._stamp_context_tokens(recorder, context_tokens)
                self._stamp_answer_stats(recorder, self._answer_stats(
                    answer_before, self._answer_snapshot(), time.monotonic() - answer_t0))
                if qa_tracker:
                    await qa_tracker.increment()
                continue

            answer_stats = self._answer_stats(
                answer_before, self._answer_snapshot(), time.monotonic() - answer_t0)
            recorder.context_tokens_answered_n = getattr(
                recorder, "context_tokens_answered_n", 0) + (context_tokens is not None)
            # Queries this user was actually answered on — the cost denominator.
            # Independent of the memory-token measurement, which a memo with its
            # own answer hook does not get.
            recorder.queries_answered_n = getattr(recorder, "queries_answered_n", 0) + 1

            # Wrapped at the CALL SITE, not inside core.metric.Judge:
            # benchmarks override `judge` (LongMemEval dispatches per
            # question_type), and the call site is the one
            # place all of them pass through.
            with tokens.phase(tokens.JUDGE):
                score, judge_reason = await self.judge(
                    qa["query"], answer, qa.get("reference", ""),
                    qa_metadata=qa_metadata,
                )

            await self.log_qa_step(
                recorder=recorder,
                query=qa["query"],
                predicted=answer,
                reference=qa.get("reference", ""),
                score=score,
                judge_reason=judge_reason,
                qa_metadata=qa_metadata,
                retrieved_memory=retrieved,
                relevant_context=relevant_context,
            )
            self._stamp_context_tokens(recorder, context_tokens)
            self._stamp_answer_stats(recorder, answer_stats)

            if qa_tracker:
                await qa_tracker.increment()

        recorder.compose_hook = "custom" if self.overrides_compose(memo) else "default"
        if n_retrieve_err or n_answer_err:
            recorder.failure_info = (
                f"phase2_errors: retrieve={n_retrieve_err}/{len(qa_pairs)} "
                f"answer={n_answer_err}/{len(qa_pairs)}; first: {first_err}"
            )

        t2_elapsed = time.time() - t2
        scores = [s["score"] for s in recorder.steps]
        avg = sum(scores) / len(scores) if scores else 0.0
        await recorder.set_reward(avg)
        log.info(f"[Phase 2] User {user_tag} QA complete: reward={avg:.2f}/{self.judge_score_max} ({len(scores)} QA, {t2_elapsed:.1f}s)")
        return recorder

    # ---- Phase 1 dispatch ----

    async def _phase1_update(self, memo: MemoClass, init_data: Any) -> None:
        """Hand the whole visible `init_data` to the memo in ONE build_memory_from_data
        call. Ingestion granularity is the memo's own choice; the workflow
        does not chunk."""
        if not isinstance(init_data, list):
            raise TypeError(
                f"{type(self).__name__}: default _phase1_update expects init_data to be a list; "
                f"got {type(init_data).__name__}. Override _phase1_update for non-list init."
            )
        # NOTE: max_logs trims THIS call's init_data (defaults to None, unused in evolution).
        items = init_data[-self.max_logs:] if self.max_logs else init_data
        log.info(f"[Phase 1] build_memory_from_data ({len(items)} {self._phase1_item_label})")
        r = self.recorder_class()
        await self.phase1_log_init(r, items)
        try:
            await memo.build_memory_from_data(r)
        except Exception as exc:
            log.warning(f"build_memory_from_data failed: {exc}")
            raise RuntimeError(f"[Phase1_Update] {type(exc).__name__}: {exc}") from exc

    # ---- Full-trace persistence (no sampling) ----

    @staticmethod
    def _trace_payload(rec: Any) -> Dict[str, Any]:
        """One user's trace: reward, the per-step records, and everything the
        evaluation charged THIS user — the `cost_*` fields `_stamp_cost` put on
        the recorder (LLM tokens per phase, dollars per phase, embedding
        tokens, unpriced models) under `cost`, with the query denominators
        beside them. metrics.json only keeps the mean over users; users differ
        in history length, so the per-user figure is only recoverable here."""
        steps = getattr(rec, "steps", [])
        cost = {k: v for k, v in vars(rec).items() if k.startswith("cost_")} if hasattr(rec, "__dict__") else {}
        return {
            "user_id": getattr(rec, "user_id", "") or "unknown",
            "reward": float(getattr(rec, "reward", 0.0)),
            "failure_info": getattr(rec, "failure_info", None),
            "n_qa": len(steps),
            "queries_answered_n": getattr(rec, "queries_answered_n", 0),
            # What every QA call carried; per-step values live in
            # steps[].context_tokens. Measured for EVERY memo, unlike the
            # differential it replaces.
            "context_tokens_total": getattr(rec, "context_tokens_total", 0),
            "context_tokens_n_queries": getattr(rec, "context_tokens_n", 0),
            "compose_hook": getattr(rec, "compose_hook", None),
            "cost": cost,
            "steps": steps,
        }

    def save_full_traces(self, recorder_list: List[Any]) -> None:
        """Write every user's full QA trajectory to output_run_dir/traces/<user_id>.json.

        Failed users (Exception objects) are skipped; partial recorders are
        serialized with their failure_info preserved.
        """
        if self.output_run_dir is None:
            log.warning("save_full_traces: output_run_dir not set, skipping")
            return

        traces_dir = self.output_run_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        for rec in recorder_list:
            if isinstance(rec, Exception):
                continue
            user_id = getattr(rec, "user_id", "") or "unknown"
            payload = self._trace_payload(rec)
            safe = user_id.replace("/", "_").replace("\\", "_") or "unknown"
            file_path = traces_dir / f"{safe}.json"
            try:
                # Serialize FIRST, write only on success — a serialization
                # error must never leave a truncated/0-byte trace file.
                text = json.dumps(
                    _json_safe_keys(payload),
                    indent=2, ensure_ascii=False, cls=_MemoryEncoder,
                )
                file_path.write_text(text, encoding="utf-8")
            except Exception as e:
                log.warning(f"Failed to save trace for {user_id}: {e}")

            # Level 2+: the same steps, flattened next to the call records so
            # a call can be joined to the question it served. Derived from
            # this very payload, so the two can never disagree.
            from core import trace as _trace
            for i, step in enumerate(payload.get("steps") or []):
                _trace.record_qa(user=user_id, query_index=i, step=step)
