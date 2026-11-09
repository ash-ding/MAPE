"""The price table: tokens of different models become money.

    uv run --with pytest pytest tests/test_pricing.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import pricing as P   # noqa: E402


def test_input_and_output_are_priced_differently():
    # 1M in + 1M out of gpt-4.1-mini: $0.40 + $1.60
    assert abs(P.usd("gpt-4.1-mini", 1_000_000, 1_000_000) - 2.00) < 1e-9
    assert abs(P.usd("gpt-4.1-mini", 1_000_000, 0) - 0.40) < 1e-9
    assert abs(P.usd("gpt-4.1-mini", 0, 1_000_000) - 1.60) < 1e-9


def test_embeddings_are_cheap_but_not_free():
    assert abs(P.usd("text-embedding-3-small", 1_000_000, 0) - 0.02) < 1e-9
    assert P.usd("text-embedding-3-small", 0, 0) == 0.0


def test_effort_suffix_and_case_do_not_change_the_price():
    assert P.price("GPT-4.1-Mini/low") == P.price("gpt-4.1-mini")


def test_unknown_models_are_unpriced_not_free():
    assert P.usd("gpt-4.1-mini-2025-04-14", 1000, 0) is None   # a dated snapshot is not the listed model
    assert P.usd("some-model", 1000, 0) is None
    assert P.usd(None, 1000, 0) is None


def test_the_definition_names_the_table_and_the_rules():
    """The phase list is in the fingerprint on purpose: it changed when the
    memo stopped answering and started composing, and a cached or resumed
    result priced under the old list must not land on the new frontier."""
    assert P.PRICES_DATE in P.COST_DEFINITION
    assert "no-cache-discount" in P.COST_DEFINITION
    assert "build+retrieve+compose+answer" in P.COST_DEFINITION
    from core import tokens as T
    for phase in T.MEMORY_PHASES + (T.ANSWER,):
        assert phase in P.COST_DEFINITION, phase
    assert T.JUDGE not in P.COST_DEFINITION, "judge is evaluation overhead"
