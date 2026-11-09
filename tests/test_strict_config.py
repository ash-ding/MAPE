"""Config-completeness tests for the baseline entrypoints.

The harness baselines (amem, mem0, ...) share ONE entrypoint
  (baselines/eval_harness.py) and ONE frame config, validated by
  `core.config.validate_exact_config` against eval_harness.FRAME_KEYS
  (missing keys AND unknown keys both abort; sizing checked to the leaf).
  Method knobs are each memo.py's module-level CONFIG_DEFAULTS, resolved by
  eval_harness.resolve_memo_config; `memo:` overrides are validated.

Runs in the repo-root venv (or any venv): a baseline whose isolated
per-baseline-venv deps aren't installed in the CURRENT venv (e.g. amem's
sentence-transformers in the root venv) is SKIPPED, not failed — its own venv
verifies it.
    uv run pytest -q tests/test_strict_config.py
"""
import sys, importlib.util
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    ConfigCompletenessError, load_config_file, validate_exact_config,
)

HARNESS_EXAMPLE = PROJECT_ROOT / "baselines" / "config.example.yaml"


class _SkipTest(Exception):
    """This baseline's deps aren't installed in the current venv — skip it here
    (its own per-baseline venv verifies its config wiring)."""


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, PROJECT_ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ImportError as e:  # ModuleNotFoundError included — baseline deps absent
        raise _SkipTest(f"{rel}: deps unavailable in this venv ({e})")
    return m


# ---------------------------------------------------------------------------
# Harness baselines — one shared frame config, exact validation
# ---------------------------------------------------------------------------

def _frame_cfg(**changes):
    cfg = dict(load_config_file(HARNESS_EXAMPLE))
    cfg.update(changes)
    return cfg


def test_shared_example_passes_exactly():
    from baselines.eval_harness import load_frame_config
    load_frame_config(HARNESS_EXAMPLE)   # no raise


def test_missing_key_raises():
    from baselines.eval_harness import FRAME_KEYS
    cfg = _frame_cfg(); del cfg["llm_model"]; del cfg["run_name"]
    try:
        validate_exact_config(cfg, FRAME_KEYS, "harness")
    except ConfigCompletenessError as e:
        assert "llm_model" in str(e)
    else:
        raise AssertionError("expected ConfigCompletenessError")


def test_unknown_key_raises():
    # Typo protection: an unknown key must abort, not silently ride along —
    # and a METHOD knob at the top level is an unknown key now (it belongs
    # under `memo:`).
    from baselines.eval_harness import FRAME_KEYS
    for bad in ("llm_modle", "retrieve_k"):
        cfg = _frame_cfg(**{bad: 1}); del cfg["run_name"]
        try:
            validate_exact_config(cfg, FRAME_KEYS, "harness")
        except ConfigCompletenessError as e:
            assert bad in str(e)
        else:
            raise AssertionError("expected ConfigCompletenessError")


def test_missing_sizing_leaf_raises():
    # Second check layer: every top-level key present, but the `size` block
    # misses one native sizing leaf — must still raise, naming the sizing path.
    from baselines.eval_harness import FRAME_KEYS, OPTIONAL_KEYS
    cfg = _frame_cfg(dataset="locomo", size={"n_conversations": 2})   # n_qa leaf missing
    for k in OPTIONAL_KEYS:                                  # what load_frame_config strips
        cfg.pop(k, None)
    try:
        validate_exact_config(cfg, FRAME_KEYS, "harness")
    except ConfigCompletenessError as e:
        assert "size" in str(e)
    else:
        raise AssertionError("expected ConfigCompletenessError")


def test_memo_override_of_unknown_key_raises():
    # The `memo:` block is validated against CONFIG_DEFAULTS: a typo must
    # abort rather than silently become a no-op.
    from baselines.eval_harness import resolve_memo_config

    defaults, model_keys = {"top_k": 5, "llm": "m"}, {"llm": ("llm",)}
    assert resolve_memo_config(defaults, model_keys, arm="faithful", overrides={"top_k": 9}) == {"top_k": 9, "llm": "m"}
    try:
        resolve_memo_config(defaults, model_keys, arm="faithful", overrides={"topk": 9})
    except KeyError as e:
        assert "topk" in str(e)
    else:
        raise AssertionError("expected KeyError")
