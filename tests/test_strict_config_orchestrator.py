"""Tests for strict-config wiring into the evolution loop's nested config resolver
(evolution/orchestrator.py::REQUIRED_SCHEMA + _strict_validate +
_provided_tree, gated into _resolve_config via core.config.strict_on).

Run in the root project environment:
    uv run pytest -q tests/test_strict_config_orchestrator.py

Kept separate from tests/test_strict_config.py, which covers the baseline
entrypoint (baselines/eval_harness.py) and skips baselines whose dependencies
are not installed; evolution cases here need only the root venv.

Each test loads evolution/orchestrator.py as a standalone module (via importlib,
custom module name — never registered under its real dotted name, and
`__name__ != "__main__"` so no code under an `if __name__ == "__main__":`
guard ever fires) so it can call the internal `_strict_validate`
directly without building a full argparse Namespace / going through the
whole CLI.
"""
import sys

import pytest
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, PROJECT_ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m   # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Hand-built complete fixture — a full evolution YAML dict satisfying
# REQUIRED_SCHEMA + the dynamic `datasets` sizing loop. Kept in-file for
# the synthetic missing-key cases below (they need to mutate
# individual leaves in isolation); the real-file anchors
# (test_search_example_passes_strict / test_test_example_passes_strict)
# load configs/evolution/example.yaml + configs/evaluation/example.yaml directly.
# ---------------------------------------------------------------------------

def _min_cfg():
    cfg = {
        "steps": 5, "smoke_test": False,
        "model": "gpt-5-mini", "judge_model": "gpt-5-mini",
        "metrics": ["accuracy"],
        "max_sample_concurrent": 3,
        "adopt_orphans": True,
        "llm": {
            "routes": {},
        },
        "proposer": {
            "model": "claude-opus-4-8", "effort": "xhigh",
            "disallowed_tools": ["mcp__*"], "timeout_s": 2700,
        },
        "propose": {"k_per_step": 2},
        "sanity": {"enabled": True, "max_retries": 2},
        # Only  is unconditional: the rest is Cond(enabled).
        "analyst": {"enabled": False},
        "seed": {"enabled": False},
        "gpu": {"enabled": False},
        "trace": {"max_value_chars": 20000},
        "contract": {"compose": True},
        "datasets": {
            "locomo": {
                "size": {"n_conversations": 2, "n_qa": 20},
                "sanity_check": {"n_conversations": 1, "n_qa": 3},
            },
        },
    }
    return cfg


def test_search_example_passes_strict():
    """configs/evolution/example.yaml is the real, checked-in exhaustive evolution
    search template — it must pass REQUIRED_SCHEMA as shipped."""
    orch = _load("_orch1", "evolution/orchestrator.py")
    from core.config import load_config_file
    fc = load_config_file(str(PROJECT_ROOT / "configs" / "evolution" / "example.yaml"))
    orch._strict_validate(fc, resolved_cfg=fc)  # no raise


def test_test_example_passes_strict():
    """configs/evaluation/example.yaml is the real, checked-in exhaustive held-out
    template — it must pass the smaller HELDOUT_REQUIRED_SCHEMA as shipped."""
    orch = _load("_orch1b", "evolution/orchestrator.py")
    from core.config import load_config_file
    fc = load_config_file(str(PROJECT_ROOT / "configs" / "evaluation" / "example.yaml"))
    orch._strict_validate(fc, resolved_cfg=fc,
                                schema=orch.HELDOUT_REQUIRED_SCHEMA)  # no raise


def test_missing_top_key_raises():
    orch = _load("_orch2", "evolution/orchestrator.py")
    from core.config import ConfigCompletenessError
    fc = {"steps": 1}  # nearly everything missing
    raised = False
    try:
        orch._strict_validate(fc, resolved_cfg=fc)
    except ConfigCompletenessError as e:
        raised = "judge_model" in str(e) and "datasets" in str(e)
    assert raised


def test_complete_cfg_passes_and_a_missing_proposer_leaf_raises():
    orch = _load("_orch4", "evolution/orchestrator.py")
    from core.config import ConfigCompletenessError
    fc = _min_cfg()
    orch._strict_validate(fc, resolved_cfg=fc)  # no raise
    del fc["proposer"]["model"]
    with pytest.raises(ConfigCompletenessError, match="proposer.model"):
        orch._strict_validate(fc, resolved_cfg=fc)


def test_strict_off_via_config_flag():
    """--no-strict-config / strict_config: false must bypass the gate inside
    _resolve_config end-to-end (not just the internal validator), even for a
    badly incomplete --config file."""
    import tempfile
    import os
    import yaml
    orch = _load("_orch5", "evolution/orchestrator.py")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.yaml")
        with open(p, "w") as f:
            yaml.safe_dump({"datasets": {"locomo": {}}, "strict_config": False}, f)
        cfg = orch._resolve_config(
            orch.build_arg_parser().parse_args(["--config", p]))
        assert cfg["strict_config"] is False


def test_strict_on_via_config_flag_raises_end_to_end():
    """The full _resolve_config path (not just _strict_validate in
    isolation) raises ConfigCompletenessError for an incomplete --config file
    when strict_config is left at its default (True)."""
    import tempfile
    import os
    import yaml
    orch = _load("_orch6", "evolution/orchestrator.py")
    from core.config import ConfigCompletenessError
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.yaml")
        with open(p, "w") as f:
            yaml.safe_dump({"datasets": {"locomo": {}}}, f)
        raised = False
        try:
            orch._resolve_config(orch.build_arg_parser().parse_args(["--config", p]))
        except ConfigCompletenessError:
            raised = True
        assert raised


def test_no_config_file_never_trips_strict():
    """Strict mode is gated on args.config being non-None — an in-process
    config (no --config file at all) must never trip it, however incomplete."""
    orch = _load("_orch7", "evolution/orchestrator.py")
    args = orch.build_arg_parser().parse_args(["--datasets", "locomo"])
    assert args.config is None
    cfg = orch._resolve_config(args)  # must not raise
    assert "locomo" in cfg["datasets"]


# ---------------------------------------------------------------------------
# evolution.heldout's smaller config surface: evolution.heldout also drives
# _resolve_config, but never reads the search-loop-only fields
# (steps/smoke_test/adopt_orphans/proposer.*/propose.*/sanity.*/seed.*).
# HELDOUT_REQUIRED_SCHEMA is the smaller schema evolution.heldout passes via
# _resolve_config's `required_schema` kwarg. These two synthetic cases use an
# in-file fixture for isolated leaf-deletion; the real-file anchor
# (test_test_example_passes_strict, above) covers
# configs/evaluation/example.yaml directly.
# ---------------------------------------------------------------------------

def _heldout_min_cfg():
    return {
        "model": "gpt-5-mini", "judge_model": "gpt-5-mini",
        "max_sample_concurrent": 3,
        "gpu": {"enabled": False},
        "trace": {"max_value_chars": 20000},
        "llm": {
            "routes": {},
        },
        "datasets": {
            "locomo": {
                "size": {"n_conversations": None, "n_qa": None},
            },
        },
    }


def test_heldout_schema_omits_search_fields():
    """A cfg complete for HELDOUT_REQUIRED_SCHEMA but with NO proposer/sanity/
    seed (search-loop-only fields heldout never reads) must NOT raise."""
    orch = _load("_orch8", "evolution/orchestrator.py")
    fc = _heldout_min_cfg()
    assert "proposer" not in fc and "sanity" not in fc and "seed" not in fc
    orch._strict_validate(fc, resolved_cfg=fc,
                                schema=orch.HELDOUT_REQUIRED_SCHEMA)  # no raise


def test_heldout_schema_missing_model_raises():
    orch = _load("_orch9", "evolution/orchestrator.py")
    from core.config import ConfigCompletenessError
    fc = _heldout_min_cfg()
    del fc["model"]
    raised = False
    try:
        orch._strict_validate(fc, resolved_cfg=fc,
                                    schema=orch.HELDOUT_REQUIRED_SCHEMA)
    except ConfigCompletenessError as e:
        # Quoted-repr match (the message renders sorted(missing) as a list
        # repr) — avoids false-matching "judge_model", which also contains
        # the substring "model".
        raised = "'model'" in str(e)
    assert raised


def test_schema_covers_every_nonexempt_default_key():
    # Guards against a new top-level DEFAULT_CONFIG key silently escaping
    # strict-config.
    orch = _load("_orch_cov", "evolution/orchestrator.py")
    META_EXEMPT = {"run_name", "coverage", "strict_config"}
    top = set(orch.DEFAULT_CONFIG.keys())
    required = set(orch.REQUIRED_SCHEMA.keys())
    missing = top - required - META_EXEMPT
    assert not missing, (
        f"top-level DEFAULT_CONFIG keys missing from REQUIRED_SCHEMA "
        f"(add to schema or META_EXEMPT): {sorted(missing)}"
    )
