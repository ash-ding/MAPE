"""Which models a memory system may call, and which one it should.

The evaluation's contract fixes the QA model and the judge, but nothing fixed
what a harness calls *inside* its own build / retrieve code. An accuracy-only
search found that gap within two steps: it put a gpt-5-mini reasoner behind
retrieve, then escalated it to gpt-5 at high effort ("cost is free"), and
every later candidate inherited it. The score compared reasoners, not
memories, and the evaluation bill went from $0.2 to $30 a step.

Two environment variables carry the run's policy into the eval container:

  MAPE_ALLOWED_MODELS   comma-separated model names a request may name.
                           Unset/empty → no enforcement (host-side tests,
                           baselines run outside evolution).
  MAPE_UNIFIED_LLM      the LLM a harness should use when it does not
                           say (Agent(model=None)); the QA model of the run.
  MAPE_UNIFIED_EMBEDDING  likewise for Embedding(model=None).

Enforcement sits where a request is made, not where a config is read: the
OpenAI SDK shim (`core.openai_usage`) checks the `model=` of every create()
before it is sent, and `core.llm.Agent` / `Embedding` check at construction
so a harness fails at import-level self-validation rather than mid-eval. A
"/effort" suffix is not part of the model name and is stripped first.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Set

from core.logger import get_logger

log = get_logger("main")

ALLOWED_ENV = "MAPE_ALLOWED_MODELS"
UNIFIED_LLM_ENV = "MAPE_UNIFIED_LLM"
UNIFIED_EMBEDDING_ENV = "MAPE_UNIFIED_EMBEDDING"

#: The backbone settings this project reports. A backbone is the run's
#: unified LLM: the QA model that
#: answers from memory AND the model behind every agentic call inside the
#: memory system. `model:` stays a config value — these names are what our
#: runs use, and a run naming anything else is warned about, not refused.
#:
#:   gpt-4.1-mini                  evolution (the search loop) and held-out test
#:   qwen3-30b-a3b-instruct-2507   held-out test only, served locally with
#:                                 vLLM (scripts/vllm/) and reached by name
#:                                 through MAPE_MODEL_ROUTES. A harness
#:                                 is evolved under gpt-4.1-mini and re-run on Qwen with
#:                                 only the task-time backbone swapped — a
#:                                 model-transfer measurement, never a second
#:                                 evolution.
BACKBONES = ("gpt-4.1-mini", "qwen3-30b-a3b-instruct-2507")
#: LLM-as-judge in every setting (configurable via `judge_model:`, never
#: varied in reported runs).
JUDGE = "gpt-5-mini"

_BACKBONE_WARNED: Set[str] = set()


def warn_if_unusual_backbone(model: Optional[str], *, where: str = "") -> None:
    """Log once when a run's backbone is not one of BACKBONES. A warning, not
    a refusal: the backbone is configurable; the reported settings are two."""
    base = base_name(model)
    if not base or base in BACKBONES or base in _BACKBONE_WARNED:
        return
    _BACKBONE_WARNED.add(base)
    log.warning(f"backbone {model!r}{' (' + where + ')' if where else ''} is not one of the "
                f"project's reported settings {list(BACKBONES)}; its numbers will not sit on "
                f"the same axis as the README's")


#: Every refusal this process made, in order. A harness may catch the
#: exception and carry on (generated code often wraps its calls in a broad
#: except); the host reads this list back from metrics.json and fails the
#: harness anyway — a swallowed violation must not become a silent downgrade.
_VIOLATIONS: list = []


def violations() -> list:
    return [dict(v) for v in _VIOLATIONS]


def reset() -> None:
    _VIOLATIONS.clear()


class ModelPolicyViolation(RuntimeError):
    """A request named a model the run does not allow. Raised BEFORE the
    request is sent, so the violation costs nothing and the trace names the
    call site."""


def base_name(model: Optional[str]) -> str:
    """The model name without the repo's "/effort" suffix, lower-cased."""
    m = str(model or "").strip()
    if "/" in m:
        m = m.split("/", 1)[0]
    return m.lower()


def allowed_models() -> Optional[Set[str]]:
    """The allowlist from the environment, or None when policy is off."""
    raw = os.environ.get(ALLOWED_ENV, "")
    names = {base_name(n) for n in raw.split(",") if n.strip()}
    return names or None


def unified_llm() -> Optional[str]:
    return os.environ.get(UNIFIED_LLM_ENV, "").strip() or None


def unified_embedding() -> Optional[str]:
    return os.environ.get(UNIFIED_EMBEDDING_ENV, "").strip() or None


def check(model: Optional[str], *, where: str = "") -> None:
    """Raise ModelPolicyViolation if `model` is outside the allowlist.

    A request with no model at all is left to the SDK, which rejects it on
    its own terms; the policy only judges names that were actually chosen.
    """
    allowed = allowed_models()
    if allowed is None or not model:
        return
    name = base_name(model)
    if name in allowed:
        return
    _VIOLATIONS.append({"model": name, "where": where})
    raise ModelPolicyViolation(
        f"model {model!r} is not allowed in this run"
        + (f" (at {where})" if where else "")
        + f"; allowed: {', '.join(sorted(allowed))}. The memory system must use "
        f"the run's unified model — {unified_llm() or 'see MAPE_UNIFIED_LLM'} "
        f"for LLM calls, {unified_embedding() or 'see MAPE_UNIFIED_EMBEDDING'} "
        f"for embeddings. Leave `model` unset on core.llm.Agent / Embedding "
        f"to get them."
    )


def env_for_container(*, llm: str, judge: str, embedding: str,
                      extra: Iterable[str] = ()) -> dict:
    """The environment a host injects to put this policy into a container.
    The judge is allowed because the shared QA workflow calls it in the same
    process; nothing stops a harness from calling it too — the price table
    and the trace do."""
    names = [base_name(llm), base_name(judge), base_name(embedding)] + [base_name(e) for e in extra]
    seen, ordered = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n); ordered.append(n)
    return {
        ALLOWED_ENV: ",".join(ordered),
        UNIFIED_LLM_ENV: base_name(llm),
        UNIFIED_EMBEDDING_ENV: base_name(embedding),
    }


def describe() -> str:
    """One line for a log: what this process enforces."""
    allowed = allowed_models()
    if allowed is None:
        return "model policy: off (MAPE_ALLOWED_MODELS unset)"
    return (f"model policy: allowed={sorted(allowed)} unified_llm={unified_llm()} "
            f"unified_embedding={unified_embedding()}")
