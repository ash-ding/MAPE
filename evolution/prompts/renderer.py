"""Prompt rendering: turn a run's settings (sanity, datasets, metrics,
contract) into the final system / task / fix / analyst prompt strings.

All the prompt text lives in `evolution/prompts/parts/` — this module only
does sentinel substitution, dataset filtering and fix-trace truncation.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .loader import load_prompt_parts


def _normalize_active_datasets(
    active_datasets: Optional[Iterable[str]],
    dataset_info: dict,
    dataset_render_order: List[str],
) -> List[str]:
    """Map dataset registration names → canonical shape group keys, in render order.

    longmemeval_s uses the "longmemeval" shape doc.
    Unknown names are silently dropped (defensive against typos / future names).
    None / empty → all groups in canonical order (the full prompt).
    """
    if not active_datasets:
        return list(dataset_render_order)
    seen = set()
    for ds in active_datasets:
        key = "longmemeval" if ds.startswith("longmemeval") else ds
        if key in dataset_info:
            seen.add(key)
    return [k for k in dataset_render_order if k in seen]


def _primary_names(
    active_datasets: Optional[Iterable[str]],
    active: List[str],
    dataset_render_order: List[str],
) -> tuple:
    """The dataset the prompt writes its concrete example paths against.

    Two names, because they are not the same string. A harness's results live
    at `harnesses/<id>/<ds>/` under the REGISTRATION name from cfg["datasets"]
    (`longmemeval_s`), while the reference source lives at
    `/app/benchmarks/<bench>/` under the shape group (`longmemeval`). An
    example path that mixes them up sends the proposer to a directory that
    is not there.
    """
    group = active[0] if active else dataset_render_order[0]
    registered = group
    for ds in active_datasets or ():
        if ("longmemeval" if ds.startswith("longmemeval") else ds) == group:
            registered = ds
            break
    return registered, group


def _build_dataset_subs(
    active: List[str],
    dataset_info: dict,
    primary_ds: str,
    primary_bench: str,
) -> dict:
    """Compute the dataset-conditional sentinel substitutions."""
    n = len(active)
    names_csv = ", ".join(dataset_info[k]["display_name"] for k in active)

    if n == 0:
        # Defensive: shouldn't happen since _normalize falls back to all three,
        # but if it does, leave intro vague.
        eval_intro = (
            "Your memory system is evaluated on long-context QA benchmarks. "
            "It must handle the `recorder.init` shape provided at runtime."
        )
    elif n == 1:
        eval_intro = (
            f"Your memory system is evaluated on the {names_csv} benchmark. "
            f"See the dataset shape and dispatch pattern below."
        )
    else:
        eval_intro = (
            f"Your memory system is evaluated on MULTIPLE benchmarks "
            f"(currently {names_csv}). Each benchmark exposes a different "
            f"`recorder.init` shape, so it must handle all of them — either "
            f"with dataset-agnostic logic, or by dispatching on the keys of "
            f"`recorder.init`."
        )

    if n == 1:
        accuracy_bullet = (
            "        \u2022 accuracy_<dataset>  \u2014 this benchmark's score, native scale\n"
            "                                (HIGH = good; currently 0-1)"
        )
        shapes_header = "# Dataset shape"
    else:
        accuracy_bullet = (
            "        \u2022 accuracy_<dataset>  \u2014 per-benchmark score, native scale (HIGH = good;\n"
            "                                both 0-1: LoCoMo / LongMemEval binary). There is NO\n"
            "                                cross-benchmark mean \u2014 each benchmark is recorded\n"
            "                                on its own axis (a mean would reward imbalanced\n"
            "                                harnesses and blend non-comparable scores)"
        )
        shapes_header = "# Dataset shapes (the ONLY things differing across benchmarks)"

    qa_meta = "\n".join(dataset_info[k]["qa_metadata"] for k in active)
    relevant = "\n".join(dataset_info[k]["relevant"] for k in active)
    shapes = "\n".join(dataset_info[k]["shape"] for k in active)
    raw_data = "\n".join(dataset_info[k]["raw_data"] for k in active)
    build_calls = "\n".join(dataset_info[k]["build_call"] for k in active)
    protocol_extra = "".join(dataset_info[k].get("protocol_extra", "") for k in active)
    readonly_extra = "".join(dataset_info[k].get("readonly_extra", "") for k in active)

    if n == 1:
        # One benchmark: there is nothing to dispatch on, and an `if` with a
        # single arm invites the proposer to key on a dataset name.
        info = dataset_info[active[0]]
        # A comment written for the multi-dataset case indents its own
        # continuation lines to sit inside an `if` branch. With no branch to
        # sit in, that indent is 4 spaces too deep — and this block is a code
        # example the proposer may copy.
        def _dedent(s):
            return s.replace("\n        ", "\n    ")
        update_lines = [f"    {_dedent(info['dispatch_update_comment'])}\n    ..."]
        retrieve_lines = [f"    {_dedent(info['dispatch_retrieve_comment'])}\n    ..."]
        dispatch_header = "# Phase entry points"
    else:
        update_lines = []
        retrieve_lines = []
        for i, k in enumerate(active):
            kw = "if" if i == 0 else "elif"
            info = dataset_info[k]
            update_lines.append(
                f"    {kw} {info['dispatch_check']}:\n"
                f"        {info['dispatch_update_comment']}\n"
                f"        ..."
            )
            retrieve_lines.append(
                f"    {kw} {info['dispatch_check']}:\n"
                f"        {info['dispatch_retrieve_comment']}\n"
                f"        ..."
            )
        dispatch_header = "# Recommended dispatch pattern"

    return {
        "<<EVAL_INTRO_BLOCK>>": eval_intro,
        "<<ACCURACY_AXIS_BULLET>>": accuracy_bullet,
        "<<TRACE_QA_METADATA_BLOCK>>": qa_meta,
        "<<TRACE_RELEVANT_BLOCK>>": relevant,
        "<<DATASET_SHAPES_HEADER>>": shapes_header,
        "<<DATASET_SHAPES_BLOCK>>": shapes,
        "<<RAW_DATA_FILES>>": raw_data,
        "<<BUILD_CALL_SHAPES>>": build_calls,
        "<<PROTOCOL_EXTRA_BLOCK>>": protocol_extra,
        "<<READONLY_EXTRA_BLOCK>>": readonly_extra,
        "<<DISPATCH_SECTION_HEADER>>": dispatch_header,
        "<<DISPATCH_UPDATE_BRANCHES>>": "\n".join(update_lines),
        "<<DISPATCH_RETRIEVE_BRANCHES>>": "\n".join(retrieve_lines),
        # primary_bench, not active[0]: the defensive n==0 path still has to
        # name a real file rather than leave the paragraph pointing at nothing.
        "<<PRIMARY_QA_PROMPTS>>": dataset_info[primary_bench]["prompts_path"],
        "<<PRIMARY_DATASET>>": primary_ds,
        "<<PRIMARY_BENCH_DIR>>": primary_bench,
    }


def build_proposer_system(
    *,
    sanity_enabled: bool = True,
    active_datasets: Optional[Iterable[str]] = None,
    metrics: Optional[Iterable[str]] = None,
    compose_enabled: bool = True,
) -> str:
    """Render the proposer SYSTEM prompt.

    Args:
      sanity_enabled  — False when this run will NOT run the sanity layer
                        (cfg.sanity.enabled=false, or cfg.status=devtest).
      metrics         — what this run optimizes (cfg["metrics"]), which
                        decides which objective axes the frontier carries and
                        therefore what the prompt may describe. None → the
                        accuracy-only text.
      active_datasets — iterable of dataset registration names from
                        cfg["datasets"] keys (e.g. ["locomo", "longmemeval_s"]). None / empty → render all known
                        shape groups.
      compose_enabled — cfg.contract.compose. False → the text describes a
                        two-hook contract (build + retrieve) and says the
                        third hook is refused.
    """
    mod = load_prompt_parts()

    active = _normalize_active_datasets(
        active_datasets, mod.DATASET_INFO, mod.DATASET_RENDER_ORDER
    )
    primary_ds, primary_bench = _primary_names(
        active_datasets, active, mod.DATASET_RENDER_ORDER
    )
    subs = {
        **(mod.SANITY_ON_SUBS if sanity_enabled else mod.SANITY_OFF_SUBS),
        **_build_dataset_subs(active, mod.DATASET_INFO, primary_ds, primary_bench),
        "<<OBJECTIVE_AXES_BLOCK>>": _objective_axes_block(mod, metrics),
        **_compose_subs(mod, compose_enabled),
    }
    out = mod.SYSTEM_TEMPLATE
    for sentinel, replacement in subs.items():
        out = out.replace(sentinel, replacement)
    return out


def _compose_subs(mod, compose_enabled: bool) -> dict:
    """The <<COMPOSE_*>> sentinel texts for this run's `contract.compose`."""
    return mod.COMPOSE_SUBS["on" if compose_enabled else "off"]


def _objective_axes_block(mod, metrics: Optional[Iterable[str]]) -> str:
    """The objective-axes text for this run's `metrics:` setting.

    The frontier only carries the axes the run records, so describing token
    axes to a run that optimizes accuracy alone would point the proposer at
    keys that are not there. An unknown combination falls back to the
    accuracy-only text rather than failing the run — the prompt stays true to
    the smaller scoreboard either way.
    """
    key = tuple(sorted(set(metrics or ("accuracy",))))
    block = mod.OBJECTIVE_AXES_SUBS.get(key)
    if block is None:
        block = mod.OBJECTIVE_AXES_SUBS[("accuracy",)]
    return block


def proposer_task_prompt(
    new_dir_rel: str,
    *,
    active_datasets: Optional[Iterable[str]] = None,
    analyst_enabled: bool = False,
) -> str:
    """Per-iteration TASK prompt. All paths are workspace-relative.

    `active_datasets` conditions the same way it does for the system prompt:
    the task prompt tells the proposer how to read the frontier and what
    "general" means for this run, and both of those read differently when
    the run targets one benchmark instead of three.
    """
    mod = load_prompt_parts()
    active = _normalize_active_datasets(
        active_datasets, mod.DATASET_INFO, mod.DATASET_RENDER_ORDER
    )
    out = mod.TASK_PROMPT_TEMPLATE.format(new_dir_rel=new_dir_rel)
    for sentinel, replacement in _build_task_subs(active, mod).items():
        out = out.replace(sentinel, replacement)
    # Who writes the post-evaluation analysis.
    block = mod.CLOSE_THE_LOOP_SUBS["on" if analyst_enabled else "off"]
    block = block.replace("<<ANALYST_EVIDENCE>>", mod.ANALYST_EVIDENCE)
    out = out.replace("<<CLOSE_THE_LOOP_BLOCK>>", block)
    return out


def _build_task_subs(active: List[str], mod) -> dict:
    """The task prompt's dataset-conditional blocks.

    A run on one benchmark and a run on several disagree about what the
    frontier means and what "general" asks for, so the arity picks the text.
    Nothing is defaulted away: an absent block would reach the proposer as a
    literal `<<SENTINEL>>`, and the loader makes TASK_DATASET_SUBS a required
    export so that fails at load instead.
    """
    return mod.TASK_DATASET_SUBS["single" if len(active) == 1 else "multi"]


def analyst_prompt(
    harness_id: str,
    *,
    datasets: Iterable[str],
    parent_ids: Optional[Iterable[str]] = None,
) -> Tuple[str, str]:
    """(system, task) for the analyst: read ONE candidate that has just been
    evaluated and write `diagnostics/<id>.md` + `.json`.

    Separate from the proposer prompts because the job is different: the
    analyst explains a result it cannot change, and its output is evidence
    for whoever proposes next."""
    mod = load_prompt_parts()
    parents = [p for p in (parent_ids or []) if p]
    if parents:
        parent_block = (
            f"harnesses/{parents[0]}/          its parent: the same files one step earlier"
            + (f"\n    (also drew on: {', '.join(parents[1:])})" if len(parents) > 1 else "")
        )
    else:
        parent_block = "(no parent — this candidate was written from scratch)"
    system = (
        mod.ANALYST_TEMPLATE
        .replace("<<ANALYST_ID>>", harness_id)
        .replace("<<ANALYST_PARENT_BLOCK>>", parent_block)
    )
    names = ", ".join(datasets) or "this run's benchmark"
    task = (
        f"Analyse harness `{harness_id}` on {names}. It has just finished "
        f"evaluation; everything it produced is under "
        f"`harnesses/{harness_id}/<dataset>/`.\n\n"
        f"Write exactly two files and nothing else, anywhere:\n"
        f"  diagnostics/{harness_id}.md\n"
        f"  diagnostics/{harness_id}.json\n"
    )
    return system, task


def proposer_fix_prompt(
    new_dir_rel: str,
    error_trace: str,
    *,
    compose_enabled: bool = True,
) -> str:
    """Prompt issued after a sanity-check failure: Read → diagnose → Edit.

    `compose_enabled` selects the <<COMPOSE_*>> texts of the fix half (what to
    do with an answer-like key: move the answering into compose, or keep the
    answer out of memory altogether)."""
    lines = error_trace.splitlines()
    if len(lines) > 90:
        error_trace = (
            "\n".join(lines[:60])
            + "\n... [truncated] ...\n"
            + "\n".join(lines[-30:])
        )
    mod = load_prompt_parts()
    out = mod.FIX_PROMPT_TEMPLATE
    for sentinel, replacement in _compose_subs(mod, compose_enabled).items():
        out = out.replace(sentinel, replacement)
    return out.format(new_dir_rel=new_dir_rel, error_trace=error_trace)
