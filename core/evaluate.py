"""Shared evaluation layer, used by the evolution loop AND the baselines (which
must not import evolution). The sizing schema lives here, and `evaluate_memo`,
the single, execution-independent evaluator, runs one pass over a benchmark
split in the current process (isolation is the caller's wrapper: the evolution
loop wraps it in a Singularity container, the baselines and
scripts/run_method.py call it directly)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Plain getLogger (not core.logger.get_logger): this module is imported
# in-process by the baselines, and get_logger opens a log file on import.
log = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Sizing — per-benchmark size schema
#
# Each benchmark's `datasets.<ds>` block carries two size blocks written in
# the benchmark's NATIVE fields:
#
#   size          the evaluation itself. A null field means the whole split
#                 for that dimension; LoCoMo's n_qa is per conversation.
#   sanity_check  the tiny pre-evaluation pass the search loop runs on every
#                 candidate (also what --smoke-test runs). Optional; defaults
#                 below.
# ---------------------------------------------------------------------------

# Benchmark-family field vocabulary: <family> -> (sample-count field, extras)
_FAMILY_FIELDS = {
    "locomo": ("n_conversations", ("n_qa",)),
    "longmemeval": ("n_questions", ()),
}

DEFAULT_SANITY_CHECK: Dict[str, Dict[str, Any]] = {
    "locomo": {"n_conversations": 1, "n_qa": 3},
    "longmemeval": {"n_questions": 2},
}



def _benchmark_family(ds: str) -> str:
    """Map a dataset name to its size-schema family."""
    if ds.startswith("longmemeval"):
        return "longmemeval"
    if ds not in _FAMILY_FIELDS:
        raise ValueError(
            f"Unknown benchmark {ds!r} — no size schema. Known families: "
            f"{sorted(_FAMILY_FIELDS)} (longmemeval_s maps to the 'longmemeval' family)."
        )
    return ds


def size_fields(ds: str) -> List[str]:
    """The benchmark's native size fields, sample-count field first."""
    sample_field, extras = _FAMILY_FIELDS[_benchmark_family(ds)]
    return [sample_field, *extras]


def _wire_size(v: Any) -> Optional[int]:
    """Size field → wire value. None (a `null` / `full` / `all` field) passes
    through as None = no cap (whole split for that dimension); anything else
    is coerced to int."""
    if isinstance(v, str) and v.strip().lower() in ("full", "all"):
        return None
    return None if v is None else int(v)


def _check_size_block(ds: str, block: Any, *, name: str) -> Dict[str, Any]:
    """Validate one size block: a mapping with only this benchmark's native
    fields. Returns a copy with `full`/`all` normalised to None."""
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ValueError(f"datasets.{ds}.{name} must be a mapping")
    allowed = set(size_fields(ds))
    unknown = set(block) - allowed
    if unknown:
        raise ValueError(
            f"datasets.{ds}.{name}: unknown field(s) {sorted(unknown)} "
            f"for family {_benchmark_family(ds)!r}; allowed: {sorted(allowed)}"
        )
    return {k: _wire_size(v) for k, v in block.items()}


def size_wire_spec(ds: str, block: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a size block to the wire spec the evaluator consumes:
    {"n_samples": int|None, ...family extras}. A missing or None field means
    the whole split for that dimension."""
    sample_field, extras = _FAMILY_FIELDS[_benchmark_family(ds)]
    spec: Dict[str, Any] = {"n_samples": _wire_size(block.get(sample_field))}
    for f in extras:
        spec[f] = _wire_size(block.get(f))
    return spec


def full_wire_spec(ds: str) -> Dict[str, Any]:
    """Wire spec with every size field None = the whole split."""
    return size_wire_spec(ds, {})


def resolve_size_spec(ds: str, size: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate the `size` block and return its wire spec. An ABSENT block
    (None) is an error — no silent whole-split evaluation; an empty block
    ({}) or all-null fields is the whole split, stated."""
    if size is None:
        sample_field = size_fields(ds)[0]
        raise ValueError(
            f"datasets.{ds}: a `size` block is required (null fields = the whole "
            f"split), e.g. size: {{{sample_field}: null}}"
        )
    return size_wire_spec(ds, _check_size_block(ds, size, name="size"))


def resolve_sanity_spec(ds: str, sanity_check: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Wire spec for the sanity-check pass: the block's fields over the
    family defaults."""
    block = dict(DEFAULT_SANITY_CHECK[_benchmark_family(ds)])
    block.update(_check_size_block(ds, sanity_check, name="sanity_check"))
    return size_wire_spec(ds, block)


def missing_sizing_config(dataset: str, params: Dict[str, Any],
                          path_prefix: str = "") -> List[str]:
    """Strict-config leaf check: the `size` block must be present and list
    every native size field (a null value counts as listed). Returns the
    missing paths."""
    pre = f"{path_prefix}." if path_prefix else ""
    size = (params or {}).get("size")
    if not isinstance(size, dict):
        return [f"{pre}size"]
    return [f"{pre}size.{f}" for f in size_fields(dataset) if f not in size]


def resolve_dataset_sizes(ds: str, params: Dict[str, Any]) -> None:
    """Validate a dataset's size blocks in place: check `size` when present (its presence is enforced by resolve_size_spec at
    evaluation time, or by the strict-config check), and fill `sanity_check`
    from the family defaults."""
    _benchmark_family(ds)
    if params.get("size") is not None:
        params["size"] = _check_size_block(ds, params["size"], name="size")
    block = dict(DEFAULT_SANITY_CHECK[_benchmark_family(ds)])
    block.update(_check_size_block(ds, params.get("sanity_check"), name="sanity_check"))
    params["sanity_check"] = block


# ---------------------------------------------------------------------------
# evaluate_memo — the single, execution-independent evaluation entry point.
# ---------------------------------------------------------------------------

def _build_score_json(recorder_list: List[Any]) -> Dict[str, Any]:
    """Summarize a recorder list into the canonical score.json dict (mean per-user
    reward + SE, per_user details, invalid_users) — the ONE score-summary builder."""
    rewards: List[float] = []
    per_user: Dict[str, Any] = {}
    invalid_users: List[Dict[str, Any]] = []

    for rec in recorder_list:
        if isinstance(rec, Exception):
            uid = getattr(rec, "user_id", "unknown")
            invalid_users.append({"user_id": uid, "error": repr(rec)})
            continue
        try:
            uid = getattr(rec, "user_id", "") or "unknown"
            reward = float(getattr(rec, "reward", 0.0))
            steps = getattr(rec, "steps", [])
            fi = getattr(rec, "failure_info", None)
            rewards.append(reward)
            per_user[uid] = {"reward": reward, "n_qa": len(steps), "failure_info": fi}
        except Exception as exc:
            invalid_users.append({"user_id": "unknown", "error": f"record access error: {exc!r}"})

    if rewards:
        mean = sum(rewards) / len(rewards)
        if len(rewards) > 1:
            var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
            se = (var ** 0.5) / (len(rewards) ** 0.5)
        else:
            se = 0.0
    else:
        mean, se = 0.0, 0.0

    return {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": mean,
            "benchmark_overall_eval_standard_deviation": se,
        },
        "per_user": per_user,
        "invalid_users": invalid_users,
    }


def _per_user_stddev(score: Dict[str, Any]) -> Optional[float]:
    """Population stddev over per_user rewards (None with <2 users)."""
    rewards = [float(u.get("reward", 0.0)) for u in score.get("per_user", {}).values()]
    if len(rewards) < 2:
        return None
    mean = sum(rewards) / len(rewards)
    return (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5


def _warn_on_cost_attribution_gap(cost: Dict[str, Any],
                                  stage_usage: Dict[str, Any]) -> None:
    """Shout when the two token counters disagree about the same work.

    `tokens_*` comes from the global tracker; `cost_*_per_query` comes from
    each user's own tally, stamped on their recorder. They measure the same
    calls from two directions, so the memory system cannot have burned build
    tokens on an LLM while every user reports a build cost of zero. Checked for
    build, retrieve and compose — the three phases that are memo code — plus a
    separate check that nothing was spent outside the priced phases at all.

    That exact disagreement shipped: zep recorded 3,059,840 build tokens and
    `cost_build_per_query: 0.0`, which put the most expensive baseline at the
    TOP of the efficiency axis. Nothing failed and no field was missing — the
    number was just wrong in the flattering direction, which is the one shape
    of bug a metric must never keep quiet about.

    EMBEDDINGS ARE EXCLUDED on purpose, on both sides of the comparison: the
    per-user tally never counts them (`core.tokens.CallTally`), so a phase
    whose only tokens are embeddings — every vector-search retrieval, which is
    most of them — is not a gap. Comparing raw `tokens_retrieve` here made the
    warning fire on mem0, LightMem and zep alike for 41 embedding tokens.
    """
    from core.tokens import OTHER, is_embedding_model

    by_model_phase = (stage_usage or {}).get("by_model_phase") or {}

    def measured_in(phase: str) -> int:
        return sum(
            int((phases.get(phase) or {}).get("total_tokens", 0) or 0)
            for model, phases in by_model_phase.items()
            if not is_embedding_model(model)
        )

    # `compose` is memo code making model calls, the same risk shape as
    # build and retrieve.
    for phase in ("build", "retrieve", "compose"):
        measured = measured_in(phase)
        attributed = float(cost.get(f"cost_{phase}_per_query", 0.0) or 0.0)
        if measured > 0 and attributed == 0.0 and int(cost.get("cost_n_users", 0) or 0) > 0:
            log.warning(
                f"[cost] {phase} LLM tokens were measured ({measured}) but attributed "
                f"to no user (cost_{phase}_per_query=0 over {cost.get('cost_n_users')} "
                f"user(s)). The efficiency metric for this run understates the "
                f"memory system's cost — treat cost_tokens_per_query as invalid."
            )

    # A call that reached no phase context lands in `other`, and the priced
    # total sums only build+retrieve+compose+answer — so those tokens are
    # spent and never charged. Same failure direction as the gap above: the
    # candidate looks cheaper than it is, which is the one direction that
    # must never pass quietly.
    stray = measured_in(OTHER)
    if stray > 0:
        log.warning(
            f"[cost] {stray} LLM tokens were spent outside every priced phase "
            f"(phase={OTHER!r}). The priced total covers build+retrieve+compose+"
            f"answer only, so this run's cost_usd_per_1k_queries understates what "
            f"the memory system cost. A harness reaches this by calling a model "
            f"outside the hooks — e.g. from a thread or task that does not inherit "
            f"the phase contextvar."
        )


def _usage_cost_metrics(stage_usage: Dict[str, Any]) -> Dict[str, Any]:
    """Cost block for the metrics dict, from the run's usage delta.

    `tokens` stays the ALL-PHASE total (unchanged semantics — the evolution loop's
    `tokens_total` frontier objective and every frontier entry recorded before
    the phase dimension existed mean exactly this). The phase split arrives as
    ADDITIONAL keys rather than by redefining the old one:

      tokens_build     the cost of HAVING the memory — the issue's headline
      tokens_retrieve  memory-system work at query time
      tokens_memory    build + retrieve = the full cost of operating the memory
      tokens_answer    the QA agent (cost of answering, not of the memory)
      tokens_judge     evaluation overhead — never part of a cost claim
      llm_calls        request count, all phases
    """
    from core.tokens import (ANSWER, BUILD, JUDGE, MEMORY_PHASES, RETRIEVE,
                               phase_tokens, total_tokens)
    return {
        "tokens": total_tokens(stage_usage),
        "tokens_build": phase_tokens(stage_usage, BUILD),
        "tokens_retrieve": phase_tokens(stage_usage, RETRIEVE),
        "tokens_memory": phase_tokens(stage_usage, *MEMORY_PHASES),
        "tokens_answer": phase_tokens(stage_usage, ANSWER),
        "tokens_judge": phase_tokens(stage_usage, JUDGE),
        "llm_calls": int(stage_usage.get("totals", {}).get("calls", 0)),
        "phase_seconds": dict(stage_usage.get("phase_seconds", {})),
    }


#: Stated on every run record. Token counts cover API calls ONLY; local
#: compute (embedders, rerankers, the LLMlingua-2 compressor) produces no
#: usage object and can never enter them. Any cost comparison across
#: baselines has to say so, because the gap is largest exactly where the
#: local compute is heaviest (zep runs bge-m3 AND a cross-encoder).
COST_COVERAGE_CAVEAT = (
    "Token counts cover API calls only. Local models listed under "
    "`local_models` ran on this machine and produce no usage object, so their "
    "compute is NOT in any token number here — `phase_seconds` is the only "
    "figure that covers both."
)


def _run_record(stage_usage: Dict[str, Any],
                local: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What this run actually spent, including the parts tokens can't say."""
    return {
        "local_models": local,
        "phase_seconds": dict(stage_usage.get("phase_seconds", {})),
        "llm_calls_by_phase": {
            ph: c.get("calls", 0) for ph, c in stage_usage.get("by_phase", {}).items()
        },
        "coverage_caveat": COST_COVERAGE_CAVEAT,
    }


def _context_token_metrics(recorder_list: List[Any]) -> Dict[str, Any]:
    """Context tokens for one run, summed over its users.

    The context is what `compose_memory_context` produced (or the benchmark's
    own rendering of `retrieved` when it produced None) — everything the QA
    call carried, memory and any instructions alike. Measured for EVERY memo,
    unlike the differential it replaces, which only worked when the memo left
    the answer step alone.

    `context_tokens_per_query` divides by the queries actually COMPOSED. A
    query whose QA call then failed still had a context built, so its cost
    was genuinely incurred. `context_tokens_n_answered` carries the stricter
    denominator so the other reading stays recoverable.
    """
    total = 0
    n_composed = 0
    n_answered = 0
    for rec in recorder_list:
        if isinstance(rec, Exception):
            continue
        total += int(getattr(rec, "context_tokens_total", 0))
        n_composed += int(getattr(rec, "context_tokens_n", 0))
        n_answered += int(getattr(rec, "context_tokens_answered_n", 0))
    return {
        "context_tokens_total": total,
        "context_tokens_per_query": (total / n_composed) if n_composed else 0.0,
        "context_tokens_n_queries": n_composed,
        "context_tokens_n_answered": n_answered,
    }


def _cost_per_query_metrics(recorder_list: List[Any]) -> Dict[str, Any]:
    """What operating the memory cost, per query, averaged over users.

    Per user: (BUILD + RETRIEVE tokens spent on that user) + (tokens the
    memory added to that user's QA prompts), divided by the queries that user
    was actually PROMPTED with. The mean over users is the headline
    `cost_tokens_per_query`.

    Why this shape:

    * Amortized. Building is a one-off whose cost only makes sense spread over
      the queries it serves — and without amortization a system could look
      cheap by precomputing everything at build time.
    * Per user first, then averaged. Users differ in history length and query
      count; pooling the sums would let the longest conversation dominate.
    * ANSWER contributes only the memory's own share. The QA agent is the same
      for every system (prompt template, question, its own output), so
      charging its full prompt would add a constant that hides the difference
      between memories.
    * JUDGE, the evolving agent's own calls and embeddings are not in here at
      all (see `core.tokens.CallTally`).

    Denominators and exclusions: a user who raised (no recorder) or answered
    nothing contributes no value — the same users the score drops. The count
    of those is reported as `cost_users_excluded` so a shrinking denominator
    can never pass unnoticed.

    Comparability: the number depends on how many queries were sampled per
    user, so it is only comparable across runs of the SAME sizing, and never
    across benchmarks (LongMemEval has exactly one query per user, so nothing
    amortizes there; LoCoMo has dozens). `cost_n_queries` records the
    denominator actually used.
    """
    per_user: List[float] = []
    build_pq: List[float] = []
    retrieve_pq: List[float] = []
    compose_pq: List[float] = []
    context_pq: List[float] = []
    prompt_pq: List[float] = []
    completion_pq: List[float] = []
    # Money, per 1K queries: build + retrieve + compose + answer, every model
    # priced (core.pricing), embeddings included. `compose` is the memo
    # organising its context; `answer` is the host's single QA call, whose
    # size the memo chose, so it is charged to the memo whole.
    usd_pk: List[float] = []
    usd_build_pk: List[float] = []
    usd_retrieve_pk: List[float] = []
    usd_compose_pk: List[float] = []
    usd_answer_pk: List[float] = []
    answer_pq: List[float] = []
    embedding_pq: List[float] = []
    unpriced: set = set()
    n_queries = 0
    excluded = 0
    for rec in recorder_list:
        if isinstance(rec, Exception):
            excluded += 1
            continue
        # Queries answered; the older memory-token denominator is the fallback
        # for artifacts written before `queries_answered_n` existed.
        n = int(getattr(rec, "queries_answered_n", 0)) or int(getattr(rec, "context_tokens_n", 0))
        if n <= 0:                      # answered nothing: no per-query cost
            excluded += 1
            continue
        g = lambda name: int(getattr(rec, name, 0))   # noqa: E731
        build = g("cost_build_prompt_tokens") + g("cost_build_completion_tokens")
        retrieve = g("cost_retrieve_prompt_tokens") + g("cost_retrieve_completion_tokens")
        compose = g("cost_compose_prompt_tokens") + g("cost_compose_completion_tokens")
        context = g("context_tokens_total")
        n_queries += n
        per_user.append((build + retrieve + context) / n)
        build_pq.append(build / n)
        retrieve_pq.append(retrieve / n)
        compose_pq.append(compose / n)
        context_pq.append(context / n)
        # Input/output split, kept separate because their prices differ by
        # several times; whether to weight them is a reporting choice.
        prompt_pq.append(
            (g("cost_build_prompt_tokens") + g("cost_retrieve_prompt_tokens") + context) / n)
        completion_pq.append(
            (g("cost_build_completion_tokens") + g("cost_retrieve_completion_tokens")) / n)
        gf = lambda name: float(getattr(rec, name, 0.0) or 0.0)   # noqa: E731
        usd_pk.append(gf("cost_usd_total") / n * 1000.0)
        usd_build_pk.append(gf("cost_usd_build") / n * 1000.0)
        usd_retrieve_pk.append(gf("cost_usd_retrieve") / n * 1000.0)
        usd_compose_pk.append(gf("cost_usd_compose") / n * 1000.0)
        usd_answer_pk.append(gf("cost_usd_answer") / n * 1000.0)
        answer_pq.append((g("cost_answer_prompt_tokens") + g("cost_answer_completion_tokens")) / n)
        embedding_pq.append(g("cost_embedding_tokens") / n)
        unpriced.update(getattr(rec, "cost_unpriced_models", []) or [])
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0    # noqa: E731
    from core.pricing import COST_DEFINITION
    return {
        "cost_tokens_per_query": mean(per_user),
        "cost_build_per_query": mean(build_pq),
        "cost_retrieve_per_query": mean(retrieve_pq),
        "cost_compose_per_query": mean(compose_pq),
        "cost_context_tokens_per_query": mean(context_pq),
        "cost_prompt_tokens_per_query": mean(prompt_pq),
        "cost_completion_tokens_per_query": mean(completion_pq),
        "cost_answer_tokens_per_query": mean(answer_pq),
        "cost_embedding_tokens_per_query": mean(embedding_pq),
        "cost_usd_per_1k_queries": mean(usd_pk),
        "cost_usd_build_per_1k_queries": mean(usd_build_pk),
        "cost_usd_retrieve_per_1k_queries": mean(usd_retrieve_pk),
        "cost_usd_compose_per_1k_queries": mean(usd_compose_pk),
        "cost_usd_answer_per_1k_queries": mean(usd_answer_pk),
        "cost_unpriced_models": sorted(unpriced),
        "cost_definition": COST_DEFINITION,
        "cost_n_users": len(per_user),
        "cost_n_queries": n_queries,
        "cost_users_excluded": excluded,
    }




async def evaluate_memo(
    *,
    memo_class: type,
    dataset: str,
    split: str,
    out_dir: Path,
    qa_model: str,
    judge_model: str,
    size: Optional[Dict[str, Any]] = None,
    sanity_check: Optional[Dict[str, Any]] = None,
    smoke: bool = False,
    max_sample_concurrent: int = 3,
    sample_seed: Optional[str] = None,
    max_logs: Optional[int] = None,
    memo_sha: str = "",
    memo_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one memo on one dataset/split in ONE pass — the single,
    execution-independent entry point. It runs in the CURRENT process;
    isolation is the caller's wrapper (the evolution loop's container runs
    this inside `singularity exec`, the baselines and scripts/run_method.py
    call it directly).

    Sizing: ``size`` (required unless ``smoke``) in the benchmark's native
    fields, a null field meaning the whole split; ``smoke=True`` runs the
    ``sanity_check`` sizes instead (family defaults when None) — the search
    loop's sanity gate and --smoke-test.

    Writes ``out_dir/{score.json, token_usage.json, run_record.json, traces/}``.
    A score.json is guaranteed even when the pass crashes (error score), so
    callers can always read one.

    Returns the metrics dict:

      raw_score, score_max, per_user_stddev
      failed                     True when the pass crashed (`error` says why)
      sample_spec                the wire spec that sized this pass
      tokens                     ALL-PHASE token total
      tokens_build               cost of BUILDING the memory
      tokens_retrieve            memory-system work at query time
      tokens_memory              build + retrieve
      tokens_answer/_judge       QA agent / evaluation overhead
      llm_calls                  request count
      phase_seconds              wall-clock per phase — the only cost signal
                                 that also covers local models
      context_tokens_total       tokens the composed context carried
      cost_usd_per_1k_queries    THE efficiency metric: build + retrieve +
                                 compose + answer, every model priced
                                 (core/pricing.py, embeddings included), per
                                 1K queries, meaned over users
      cost_usd_{build,retrieve,compose,answer}_per_1k_queries   its phases
      cost_unpriced_models / cost_definition   what could not be priced / how
      cost_tokens_per_query      the older token reading: (build + retrieve +
                                 injected memory tokens) per query
      cost_build/retrieve/context_tokens_per_query   its three components
      cost_prompt/completion_tokens_per_query       its input/output split
      cost_n_users / cost_n_queries / cost_users_excluded
                                 what the mean was taken over
      context_tokens_per_query   what every QA call carried
      context_tokens_n_queries   denominator: queries actually composed
      context_tokens_n_answered  stricter denominator (answer succeeded)
      local_models               local (non-API) models that ran

    ``sample_seed`` selects which subset of the split is drawn (None = the
    split's fixed prefix). ``max_logs`` / ``memo_sha`` are forwarded to the
    workflow (the evolution loop passes memo_sha=harness id)."""
    import json as _json
    from benchmarks.registry import resolve as _resolve_dataset
    from core.tokens import diff_summary as _diff_summary, init_global_tracker

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = dict(resolve_sanity_spec(dataset, sanity_check) if smoke
                else resolve_size_spec(dataset, size))
    if sample_seed is not None:
        spec["sample_seed"] = sample_seed

    workflow_cls, env_module, _rec = _resolve_dataset(dataset)
    tracker = init_global_tracker()

    # Local models (embedders, rerankers, the LLMlingua-2 compressor) are real
    # compute that can NEVER appear in token counts — they are not API calls.
    # Installed here, before any memo is constructed, so every run record names
    # what ran and no cost figure is read as complete when it is not.
    from core import local_models as _local_models
    _local_models.install()
    # Vendored systems fan LLM calls out across their own thread pools;
    # `ThreadPoolExecutor.submit` drops contextvars, which would file that
    # work under `other` instead of the phase that started it.
    from core.tokens import install_thread_context_propagation
    install_thread_context_propagation()

    n = spec.get("n_samples")
    task_list = env_module.get_task_list(
        status=split,
        eval_n_samples=None if n is None else int(n),
        seed=spec.get("sample_seed"),
    )

    workflow = workflow_cls(
        memo_class=memo_class, model=qa_model, max_logs=max_logs,
        judge_model=judge_model, memo_config=memo_config,
    )
    workflow.memo_sha = memo_sha
    workflow.status = split
    workflow.output_run_dir = out_dir

    # The tracker is process-global and cumulative; token_usage.json must
    # describe this pass alone — snapshot before, subtract after.
    usage_before = tracker.summary()
    raw_score = 0.0
    stddev: Optional[float] = None
    crashed: Optional[Exception] = None
    usage: Dict[str, Any] = {}
    records_ok: List[Any] = []
    try:
        records, rlen = await workflow.run_all_users(
            task_list, sample_spec=spec, max_sample_concurrent=max_sample_concurrent,
        )
        records_ok = records[:rlen]
        workflow.save_full_traces(records_ok)
        score = _build_score_json(records_ok)
        # Benchmark-specific REPORTING metrics (LoCoMo's paper-comparable
        # token-F1 / BLEU-1, ...). Guarded: a reporting metric must never fail
        # an otherwise-valid run, and it never touches raw_score.
        try:
            extra = workflow.aggregate_extra_metrics(records_ok)
            if extra:
                score["extra_metrics"] = extra
        except Exception as exc:
            log.warning(f"aggregate_extra_metrics failed (reporting only, ignored): {exc!r}")
        raw_score = float(score["benchmark_eval_score"]["benchmark_overall_eval_score"])
        stddev = _per_user_stddev(score)
        usage = _diff_summary(tracker.summary(), usage_before)
        with (out_dir / "score.json").open("w", encoding="utf-8") as f:
            _json.dump(score, f, indent=2, ensure_ascii=False)
        with (out_dir / "token_usage.json").open("w", encoding="utf-8") as f:
            _json.dump(usage, f, indent=2, ensure_ascii=False)
        with (out_dir / "run_record.json").open("w", encoding="utf-8") as f:
            _json.dump(_run_record(usage, _local_models.summary()), f, indent=2, ensure_ascii=False)
    except Exception as exc:
        crashed = exc
    # A crashed pass still burned tokens before it died — count them.
    if not usage:
        usage = _diff_summary(tracker.summary(), usage_before)

    cost = {**_usage_cost_metrics(usage),
            **_context_token_metrics(records_ok),
            **_cost_per_query_metrics(records_ok)}
    _warn_on_cost_attribution_gap(cost, usage)
    m: Dict[str, Any] = {
        "raw_score": raw_score,
        "score_max": workflow.judge_score_max,
        "per_user_stddev": stddev,
        "failed": crashed is not None,
        "sample_spec": spec,
        **cost,
        # Carried in the metrics so every consumer of evaluate_memo sees which
        # compute is missing from the token numbers.
        "local_models": _local_models.summary(),
    }
    if crashed is not None:
        m["error"] = repr(crashed)

    # Guarantee a score.json even if the pass crashed before writing one
    # (the evolution loop's sanity gate and the host always read one).
    if not (out_dir / "score.json").exists():
        with (out_dir / "score.json").open("w", encoding="utf-8") as f:
            _json.dump({
                "benchmark_eval_score": {
                    "benchmark_overall_eval_score": 0.0,
                    "benchmark_overall_eval_standard_deviation": 0.0,
                },
                "per_user": {},
                "invalid_users": [{"user_id": "eval_crashed",
                                   "error": str(crashed) if crashed else "no score produced"}],
            }, f, indent=2, ensure_ascii=False)

    return m
