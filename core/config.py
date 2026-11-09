"""Config loading and strict completeness checks, shared by
baselines/eval_harness.py and the evolution loop. Imports only stdlib + PyYAML,
never evolution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """In-place recursive merge of overlay into base (dicts only)."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Strict-config completeness validation — used by baselines/eval_harness.py
# and the evolution loop's _resolve_config to reject configs that omit required
# keys instead of silently falling back to a default. A `null` YAML value
# still counts as "provided" (the operator explicitly opted into the
# default) — only an ABSENT key is a completeness violation.
# ---------------------------------------------------------------------------


class ConfigCompletenessError(ValueError):
    """Raised in strict-config mode when the config file omits a required key."""


def load_config_file(path):
    """safe_load a YAML config to a mapping."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config file must be a mapping at top level: {p}")
    return cfg


def strict_on(config_path, resolved_cfg):
    """Strict completeness enforcement is active iff a config file was given AND
    strict_config (default True) was not turned off."""
    return config_path is not None and bool(resolved_cfg.get("strict_config", True))


def _completeness_msg(context, missing):
    return (
        f"{context}: config is missing required key(s): {sorted(missing)}. Every "
        f"parameter must be listed explicitly (a null value is allowed) — this is "
        f"strict-config mode. Set strict_config: false (or pass --no-strict-config) "
        f"to disable."
    )


REQUIRED = object()   # schema sentinel: a required leaf key
_MISSING = object()   # walk sentinel: key absent from the provided tree


class Cond:
    """Schema node applied only when predicate(resolved_cfg) is truthy."""
    def __init__(self, predicate, subschema):
        self.predicate = predicate
        self.subschema = subschema


def _walk_schema(node, provided_node, resolved_cfg, path, missing):
    if isinstance(node, Cond):
        if node.predicate(resolved_cfg):
            _walk_schema(node.subschema, provided_node, resolved_cfg, path, missing)
        return
    if node is REQUIRED:
        if provided_node is _MISSING:
            missing.append(".".join(path))
        return
    if isinstance(node, dict):
        pv = provided_node if isinstance(provided_node, dict) else {}
        for k, sub in node.items():
            _walk_schema(sub, pv.get(k, _MISSING), resolved_cfg, path + [k], missing)


def missing_schema_paths(provided_tree, schema, resolved_cfg):
    """Validate a nested `provided_tree` (raw YAML ∪ CLI-provided paths) against
    `schema` (nested dict of REQUIRED / Cond / sub-dicts) and return every
    missing path. `resolved_cfg` drives Cond predicates."""
    missing = []
    _walk_schema(schema, provided_tree, resolved_cfg, [], missing)
    return missing


def raise_completeness(context, missing):
    """Raise ConfigCompletenessError with the standard message for `missing`
    paths (public entry so callers needn't touch the private formatter)."""
    if missing:
        raise ConfigCompletenessError(_completeness_msg(context, missing))


def validate_exact_config(cfg, required, context):
    """Config-file-ONLY validation (harness baselines): the YAML must
    list EVERY required key (a null value counts as listed) and NOTHING else —
    there are no CLI parameter overrides and no built-in defaults anymore, so a
    missing key has nothing to fall back to and an unknown key is a typo or a
    stale setting. Also checks sizing to the leaf (`size` native fields). Returns cfg unchanged on success; raises ConfigCompletenessError
    naming every problem at once."""
    if not isinstance(cfg, dict):
        raise ConfigCompletenessError(f"{context}: the config file must be a YAML mapping")
    required = set(required)
    missing = sorted(required - set(cfg))
    unknown = sorted(set(cfg) - required)
    problems = []
    if missing:
        problems.append(f"missing required key(s): {missing}")
    if unknown:
        problems.append(f"unknown key(s): {unknown}")
    if problems:
        raise ConfigCompletenessError(
            f"{context}: " + "; ".join(problems) +
            ". Every parameter lives in the config file (a null value counts as "
            "listed); there are no CLI overrides and no built-in defaults — copy "
            "config.example.yaml and edit."
        )
    from core.evaluate import missing_sizing_config  # lazy: keep import surface light
    miss = missing_sizing_config(cfg["dataset"], cfg, path_prefix="")
    if miss:
        raise ConfigCompletenessError(f"{context}: missing sizing leaf(s): {sorted(miss)}")
    return cfg
