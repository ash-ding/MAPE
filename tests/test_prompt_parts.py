"""The prompt split: evolving / design / task / fix / fragments.

Locks in the two properties the split has to keep true:
  1. assembly is exact — SYSTEM_TEMPLATE is the evolving half with the design
     half spliced into its sentinel, nothing else;
  2. rendering leaves no sentinel behind, on every switch the renderer has.

Run:  uv run pytest -q tests/test_prompt_parts.py
"""

import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolution.prompts import (  # noqa: E402
    build_proposer_system,
    load_prompt_parts,
    proposer_fix_prompt,
    proposer_task_prompt,
)
from evolution.prompts.loader import DESIGN_SENTINEL, _PARTS_DIR  # noqa: E402

DATASETS = ["locomo", "longmemeval_s"]
METRIC_SETS = [None, ("accuracy",), ("accuracy", "efficiency")]


def _dataset_combos():
    for r in range(len(DATASETS) + 1):
        for c in itertools.combinations(DATASETS, r):
            yield list(c) if c else None


def test_the_prompt_set_has_all_its_parts():
    """All six files must exist. Five must have content — `design` may be
    empty, which is how a run is given no design priors; the file stays so
    the slot, and the splice that fills it, keep working."""
    for kind, ext in (("evolving", "md"), ("design", "md"), ("task", "md"),
                      ("fix", "md"), ("analyst", "md"), ("fragments", "py")):
        path = _PARTS_DIR / f"{kind}.{ext}"
        assert path.is_file(), f"missing {path}"
        if kind != "design":
            assert path.stat().st_size > 0, f"empty {path}"


def test_evolving_half_carries_exactly_one_design_sentinel():
    evolving = (_PARTS_DIR / "evolving.md").read_text(encoding="utf-8")
    n = evolving.count(DESIGN_SENTINEL)
    assert n == 1, f"expected 1 design sentinel in evolving.md, found {n}"


def test_system_template_is_exactly_evolving_plus_design():
    evolving = (_PARTS_DIR / "evolving.md").read_text(encoding="utf-8")
    design = (_PARTS_DIR / "design.md").read_text(encoding="utf-8")
    parts = load_prompt_parts()
    assert parts.SYSTEM_TEMPLATE == evolving.replace(DESIGN_SENTINEL, design)


def test_design_half_survives_into_every_rendered_system_prompt():
    """A rendering path that drops the design half would silently ship a
    proposer with no search direction.

    The design half is currently EMPTY on purpose — the
    hand-written taxonomy was stripped so the search gets no design priors.
    So the guard cannot pin a probe string; it pins the splice instead, and
    comes back to life the moment the file has content again.
    """
    design = (_PARTS_DIR / "design.md").read_text(encoding="utf-8")
    probe = max(design.splitlines(), key=len).strip() if design.strip() else None
    for sanity, ds, metrics in itertools.product(
        (True, False), _dataset_combos(), METRIC_SETS
    ):
        out = build_proposer_system(
            sanity_enabled=sanity, active_datasets=ds, metrics=metrics
        )
        assert DESIGN_SENTINEL.strip() not in out, \
            f"design sentinel unspliced: sanity={sanity} ds={ds} m={metrics}"
        if probe:
            assert probe in out, \
                f"design block missing: sanity={sanity} ds={ds} m={metrics}"


def test_no_design_priors_reach_the_proposer():
    """The stripped taxonomy must not survive as a dangling reference.

    Emptying design_ alone would leave `the 12 MISSION axes (A1..C4 — see
    "Search dimensions" above)` in the harness contract, pointing at a
    section that is gone — and telling the proposer there IS a taxonomy it
    is not being shown, which is a worse prior than showing it.
    """
    leaked = ("MISSION", "Search dimensions", "axes_addressed",
              "biological memory", "12 axes")
    for ds in _dataset_combos():
        rendered = build_proposer_system(active_datasets=ds) + \
            proposer_task_prompt("cand_0001", active_datasets=ds)
        for phrase in leaked:
            assert phrase not in rendered, f"{phrase!r} survived: ds={ds}"


def test_no_sentinel_survives_rendering():
    """Any `<<NAME>>` left in a rendered prompt is an unsubstituted hole."""
    leftover = re.compile(r"<<[A-Z_]+>>")
    for sanity, ds, metrics in itertools.product(
        (True, False), _dataset_combos(), METRIC_SETS
    ):
        out = build_proposer_system(
            sanity_enabled=sanity, active_datasets=ds, metrics=metrics
        )
        found = leftover.findall(out)
        assert not found, f"unsubstituted {found}: sanity={sanity} ds={ds} m={metrics}"
    for ds in _dataset_combos():
        text = proposer_task_prompt("cand_0001", active_datasets=ds)
        assert not leftover.findall(text), f"unsubstituted in task: ds={ds}"
    text = proposer_fix_prompt("cand_0001", "boom")
    assert not leftover.findall(text), leftover.findall(text)


def test_task_prompt_drops_the_multi_benchmark_prose_for_a_one_dataset_run():
    """A run that targets one benchmark has no cross-benchmark mean to warn
    about and no `recorder.init` shapes to dispatch between. Saying otherwise
    sends the proposer looking for a frontier axis and a branch that are not
    there — which is the whole reason the task prompt is conditioned at all."""
    one = proposer_task_prompt("cand_0001", active_datasets=["locomo"])
    three = proposer_task_prompt("cand_0001", active_datasets=["locomo", "longmemeval_s"])
    for phrase in ("cross-benchmark mean", "dataset-aware DISPATCH"):
        assert phrase not in one, f"{phrase!r} survived a single-dataset render"
        assert phrase in three, f"{phrase!r} missing from the multi-dataset render"
    # No version of the prompt may name an axis the frontier stopped carrying.
    for text in (one, three):
        assert "robustness_" not in text


def test_fragments_carry_the_required_exports():
    parts = load_prompt_parts()
    assert parts.DATASET_RENDER_ORDER, "DATASET_RENDER_ORDER is empty"
    for key in parts.DATASET_RENDER_ORDER:
        assert key in parts.DATASET_INFO, f"{key} missing from DATASET_INFO"
    assert ("accuracy",) in parts.OBJECTIVE_AXES_SUBS, "accuracy-only axes block missing"
    assert set(parts.SANITY_ON_SUBS) == set(parts.SANITY_OFF_SUBS), \
        "sanity on/off must cover the same sentinels"


def test_task_and_fix_keep_their_format_slots():
    parts = load_prompt_parts()
    assert "{new_dir_rel}" in parts.TASK_PROMPT_TEMPLATE
    assert "{new_dir_rel}" in parts.FIX_PROMPT_TEMPLATE
    assert "{error_trace}" in parts.FIX_PROMPT_TEMPLATE
    assert "cand_0001" in proposer_task_prompt("cand_0001")
    assert "kaboom" in proposer_fix_prompt("cand_0001", "kaboom")


def test_task_prompt_target_is_absolute_container_path():
    """The proposer reads orchestrator.log, which names the HOST workspace; a
    relative `harnesses/<id>` target once got "translated" into
    /workspace/<run_name>/harnesses/<id>. _render_and_stage_prompts renders the absolute
    container path so there is nothing to resolve."""
    import tempfile
    from pathlib import Path
    import evolution.proposer as P

    with tempfile.TemporaryDirectory() as td:
        class _Paths:
            harnesses_dir = Path(td) / "harnesses"
            workspace = Path(td)
        orig = P.paths
        P.paths = _Paths()
        try:
            P._render_and_stage_prompts(
                new_id="pending_deadbeef", mode="propose",
                sanity_enabled=True, active_datasets=["locomo"], metrics=["accuracy"],
            )
        finally:
            P.paths = orig
        task = (Path(td) / "harnesses" / "pending_deadbeef" / ".prompt_task.txt").read_text()
    assert "`/workspace/harnesses/pending_deadbeef/`" in task
    assert "`harnesses/pending_deadbeef/`" not in task


def test_stray_sentinel_is_named_in_the_failure():
    import tempfile
    from pathlib import Path
    import evolution.proposer as P

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        new_dir = ws / "harnesses" / "pending_1"
        new_dir.mkdir(parents=True)
        stray = ws / "run_x" / "harnesses" / "pending_1"
        stray.mkdir(parents=True)
        (stray / "PROPOSAL_READY").touch()
        class _Paths:
            workspace = ws
            harnesses_dir = ws / "harnesses"
        orig = P.paths
        P.paths = _Paths()
        try:
            note = P._stray_sentinel_note(new_dir)
            (new_dir / "PROPOSAL_READY").touch()
            note_own = P._stray_sentinel_note(new_dir)
            (stray / "PROPOSAL_READY").unlink()
            note_none = P._stray_sentinel_note(new_dir)
        finally:
            P.paths = orig
    assert str(stray / "PROPOSAL_READY") in note and "wrongly" in note
    assert str(stray / "PROPOSAL_READY") in note_own       # own sentinel never listed, stray still is
    assert str(new_dir) not in note_own
    assert note_none == ""


# --------------------------------------------------- parent_ids ordering

def test_the_proposer_is_told_which_parent_goes_first():
    """`parent_ids[0]` is the base. The analyst renders it as "its parent"
    and aligns traces against it, and until now nothing asked the proposer to
    make that true — the order was whatever it happened to write. Measured
    over 49 candidates that declared 2+ parents, [0] was the closest code
    relative 42 times; this closes the other 7."""
    from evolution.prompts.renderer import build_proposer_system, proposer_task_prompt
    system = build_proposer_system(active_datasets=["locomo"])
    task = proposer_task_prompt("/workspace/harnesses/x", active_datasets=["locomo"])
    assert "ORDER MATTERS" in system
    assert "Position 0 is the BASE" in system
    assert "BASE FIRST" in task or "BASE" in task
    # and it still asks for all of them, which is what makes the order useful
    assert "ALL harness dirs you drew from" in system
