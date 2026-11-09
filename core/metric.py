"""Evaluation metrics — every way this repo turns a prediction into a number.

Currently implemented:
- `Judge` — LLM-as-judge: single-shot scoring with a per-benchmark prompt
  template + score range. Used by LoCoMo (binary, one prompt) and LongMemEval
  (binary, four per-question_type prompts).

- `token_f1` / `bleu1` — the lexical metrics the LoCoMo-derived memory papers
  actually report. Pure functions, stdlib-only, no LLM call. LoCoMo computes
  them per QA step (benchmarks/locomo/workflow.py::log_qa_step) and aggregates
  them per category into score.json's `extra_metrics`. They are reported
  ALONGSIDE the judge, never instead of it — the judge remains the promotion
  signal.

`Judge` requests share `core.llm`'s per-event-loop client, global
concurrency gate, and unified retry kernel (429/5xx/timeout/empty-response
are all retried with jittered backoff — see core.llm._call_with_retries).
Token usage is reported to `core.tokens.GLOBAL_TOKEN_TRACKER` if it has
been initialized (e.g. by `init_global_tracker()` in the host launch
script). Same lazy-import pattern as `Agent.ask`, no separate "injection"
mechanism is needed.
"""
from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from typing import Optional, Tuple

# Module reference (not from-import) so the shared client/kernel stay
# late-bound — monkeypatching core.llm internals (tests) or any future
# swap of the kernel takes effect here too.
from core import llm as _llm
from core.llm import _split_model_effort

log = logging.getLogger("main")


class Judge:
    """LLM-as-judge wrapper. Single-shot, prompt-templated, score+reason output."""

    #: A judge that hits the completion cap with nothing to show has spent the
    #: whole budget on reasoning: its output is a one-line JSON, so the cap is
    #: never the answer's fault. `LengthLimitError` is globally non-retryable
    #: (core.llm) because replaying an identical request is deterministic —
    #: so this retry does not repeat the call, it raises the cap. Once. Every
    #: other judge failure keeps the behaviour it had: score_min plus a
    #: "Judge error:" reason, which the workflow's outage guard counts and the
    #: analyst reads.
    RETRY_MAX_TOKENS = 32_768

    DEFAULT_PROMPT_TEMPLATE = """You are an expert evaluator scoring an AI-generated answer against a reference answer.

Question: {query}
Reference: {reference}
Prediction: {prediction}

Score from 0 to 10 based on how well the prediction matches the reference's key information. Be lenient with phrasing differences and extra non-contradicting detail; deduct for missing key facts, factual errors, or contradictions.

Output ONLY a JSON object:
{{"reason": "<one sentence explanation>", "score": <integer 0-10>}}
"""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        prompt_template: Optional[str] = None,
        score_min: int = 0,
        score_max: int = 10,
        timeout: float = 300.0,
        max_retries: int = 5,
    ):
        # Honour the repo-wide "model/effort" suffix convention (same parsing
        # as Agent) — "gpt-5-mini/low" must NOT be sent verbatim to the API
        # (deterministic 404 → every judge call would silently score 0).
        self.model, effort = _split_model_effort(model)
        # Explicit suffix wins; otherwise judges default to low effort
        # (scoring is a short structured task).
        self.reasoning_effort = effort or "low"
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.score_min = score_min
        self.score_max = score_max
        self.timeout = timeout
        self.max_retries = max_retries

    async def _chat(self, user_msg: str) -> Tuple[str, dict]:
        """One judge call, retried once — and only — on a length-limit failure.

        Raises whatever the transport raises; `score` turns that into
        score_min + a "Judge error:" reason, exactly as before.
        """
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": user_msg}],
            timeout=self.timeout,
            max_retries=self.max_retries,
            json_mode=True,
            effort=self.reasoning_effort,
            what="Judge",
        )
        try:
            return await _llm._chat_completion(**kwargs)
        except _llm.LengthLimitError as exc:
            log.warning(
                f"Judge produced no content within the completion cap "
                f"({self.model}, effort={self.reasoning_effort}); retrying once at "
                f"max_tokens={self.RETRY_MAX_TOKENS}. Original: {exc}"
            )
            return await _llm._chat_completion(**kwargs, max_tokens=self.RETRY_MAX_TOKENS)

    async def score(
        self, query: str, predicted: str, reference: str
    ) -> Tuple[int, str]:
        """Score (predicted vs reference) for a given query. Returns (score, reason).

        Never raises: on retry exhaustion or any non-retryable exception,
        returns (score_min, error_message). A malformed-JSON judge reply is
        retried once before giving up (a whole QA step's signal is worth one
        extra call).
        """
        try:
            user_msg = self.prompt_template.format(
                query=query, reference=reference, prediction=predicted
            )
        except (KeyError, IndexError, ValueError) as exc:
            # A template with unescaped braces must degrade like every other
            # judge failure (this method's contract is "never raises").
            log.warning(f"Judge prompt template failed to render: {exc!r}")
            return self.score_min, f"Judge error: bad prompt template ({exc!r})"

        try:
            for parse_attempt in range(2):  # 1 call + 1 retry on malformed JSON
                # Provider-agnostic call (OpenAI / Anthropic API, routed by
                # model name inside the kernel). json_mode
                # maps to OpenAI's response_format; Claude models rely on the
                # prompt's JSON demand + this parse-retry loop. Per-provider
                # effort gating also lives in the kernel.
                raw, usage = await self._chat(user_msg)
                # Lazy-import to match Agent.ask pattern; tracker may be None.
                # Judge calls are forced to the `judge` phase rather than
                # inheriting the ambient one: judging is evaluation overhead
                # and must never land in a cost claim, even when a caller
                # scores inside a build/answer block.
                from core.tokens import GLOBAL_TOKEN_TRACKER, JUDGE
                if GLOBAL_TOKEN_TRACKER is not None:
                    try:
                        GLOBAL_TOKEN_TRACKER.update(
                            model_name=self.model, usage=usage, phase_name=JUDGE
                        )
                    except Exception as tracker_exc:
                        log.debug(f"token tracker update failed: {tracker_exc}")
                raw = raw or ""
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    if parse_attempt == 0:
                        log.warning(f"Judge returned non-JSON, retrying once: {raw[:120]}")
                        continue
                    return self.score_min, f"Judge returned non-JSON: {raw[:200]}"

                try:
                    # int(float(...)) so a "8.5"/"8" string or 8.5 float from
                    # the judge coerces instead of falling into the broad
                    # except below (which would skip the parse retry).
                    score = int(float(result.get("score", self.score_min)))
                except (TypeError, ValueError, AttributeError):
                    if parse_attempt == 0:
                        log.warning(f"Judge returned malformed score, retrying once: {raw[:120]}")
                        continue
                    return self.score_min, f"Judge returned malformed score: {raw[:200]}"
                score = max(self.score_min, min(self.score_max, score))
                reason = str(result.get("reason", ""))
                return score, reason
        except Exception as exc:
            log.warning(f"Judge failed: {repr(exc)}")
            return self.score_min, f"Judge error: {exc}"

        return self.score_min, "Judge exhausted retries"


# ---------------------------------------------------------------------------
# Lexical metrics — token-F1 / BLEU-1, the metrics the LoCoMo-derived memory
# papers (Mem0 arXiv 2504.19413, MemoryOS arXiv 2506.06326) actually report.
#
# Reported ALONGSIDE the LLM judge, never instead of it: the judge stays the
# promotion signal (the evolution loop's `accuracy_locomo` axis). These are stdlib-only and
# deterministic, so they cost nothing to compute inline per QA step.
#
# ---------------------------------------------------------------------------

def _normalize(text: object) -> str:
    """SQuAD-style: lowercase, drop punctuation and articles, collapse spaces."""
    s = "".join(c for c in str(text).lower() if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def token_f1(prediction: object, reference: object) -> float:
    """Token-level F1 between prediction and reference (LoCoMo paper metric).

    Returns a fraction in [0, 1] (callers that report percentages scale by 100).
    Two empty strings score 1.0; one empty side scores 0.0."""
    p, g = _normalize(prediction).split(), _normalize(reference).split()
    if not p or not g:
        return float(p == g)
    hit = sum((Counter(p) & Counter(g)).values())
    if not hit:
        return 0.0
    precision, recall = hit / len(p), hit / len(g)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: object, reference: object) -> float:
    """BLEU-1 between prediction and reference (LoCoMo paper metric).

    Unigram precision WITHOUT the brevity penalty — deliberate, and NOT a bug to
    "fix": it is what the LoCoMo evaluation scripts these papers build on
    actually compute, so adding a brevity penalty would silently stop our
    numbers being comparable to the published ones."""
    p, g = _normalize(prediction).split(), _normalize(reference).split()
    if not p or not g:
        return float(p == g)
    return sum((Counter(p) & Counter(g)).values()) / len(p)
