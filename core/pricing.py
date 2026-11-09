"""What a token costs, so tokens of different models can be added.

Raw token counts are not additive across models: a gpt-4.1-mini output token
costs 80× an embedding token and 4× one of its own input tokens. The cost a
search should trade against accuracy is money, and with the run's models
fixed by the model policy (core.model_policy) that is three constants.

List prices, USD per 1M tokens, (input, output). Cached-input discounts are
deliberately ignored: whether a prefix hits the cache depends on timing and
concurrency, so two evaluations of the same harness would price differently.
Full list price is deterministic and conservative. Reasoning tokens are a
subset of completion tokens on OpenAI's usage objects and are billed as
output — they are not added again here.

`COST_DEFINITION` names this table and its rules; it is part of the
evaluation fingerprint a resumed run is checked against
(evolution.orchestrator._eval_inputs), so a price change stops a resume
instead of silently mixing two currencies on one frontier.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from core.model_policy import base_name

PRICES_DATE = "2026-09-21"

#: USD per 1M tokens: (input, output).
PRICES_USD_PER_1M: Dict[str, Tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    # The second backbone (core.model_policy.BACKBONES). Served locally
    # (scripts/vllm/), but charged at Alibaba Cloud Model Studio's list price
    # for the same model (2026-09: $0.20 in / $0.80 out per 1M tokens) so a
    # local backbone is not "free" on the cost axis. A route's own
    # price_usd_per_1m does NOT override a listed name (price() below).
    "qwen3-30b-a3b-instruct-2507": (0.20, 0.80),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-ada-002": (0.10, 0.0),
}

COST_DEFINITION = (f"usd-v1/list-price-{PRICES_DATE}/no-cache-discount/"
                   f"phases=build+retrieve+compose+answer")


def price(model: Optional[str]) -> Optional[Tuple[float, float]]:
    """(input, output) USD per 1M tokens for `model`, or None if unpriced.
    Exact match on the base name (no "/effort" suffix, case-insensitive); a
    dated snapshot is not the listed model and prices as unknown. A locally
    served model is priced by its route (`price_usd_per_1m` in
    MAPE_MODEL_ROUTES) — the run says what it charges its own backbone."""
    name = base_name(model)
    listed = PRICES_USD_PER_1M.get(name)
    if listed is not None:
        return listed
    route = _route_price(name)
    return route


def _route_price(name: str) -> Optional[Tuple[float, float]]:
    import json, os
    raw = os.environ.get("MAPE_MODEL_ROUTES", "").strip()
    if not raw:
        return None
    try:
        entry = {str(k).lower(): v for k, v in (json.loads(raw) or {}).items()}.get(name) or {}
        pair = entry.get("price_usd_per_1m")
    except Exception:
        return None
    if not pair or len(pair) != 2:
        return None
    return float(pair[0]), float(pair[1])


def usd(model: Optional[str], prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """What `prompt_tokens` in and `completion_tokens` out of `model` cost,
    or None when the model is unpriced (the caller decides what that means)."""
    p = price(model)
    if p is None:
        return None
    return prompt_tokens / 1e6 * p[0] + completion_tokens / 1e6 * p[1]
