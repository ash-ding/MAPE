import asyncio
import json
import os
import random
import weakref
from typing import Any, Dict, List, Optional, Tuple

import httpx
import jsonschema
import openai
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.spatial.distance import cosine

from core.logger import get_logger

# Guarded import: the anthropic SDK is only needed when a claude-* model is
# requested. Environments without it (e.g. a venv predating the multi-provider
# kernel) keep the OpenAI path fully functional.
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

log = get_logger("main")
load_dotenv()


# OpenAI rejects `reasoning_effort` for non-reasoning models. Keep a
# narrow allowlist by name prefix; update as new families appear.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _supports_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES)


# ---------------- Provider routing (OpenAI vs Anthropic) ----------------
#
# Model-name based: "claude-*" → anthropic (the Anthropic API, keyed by
# ANTHROPIC_API_KEY), everything else → openai.

def _provider_for_model(model: str) -> str:
    return "anthropic" if (model or "").startswith("claude") else "openai"


# ---------------- Model routes (locally served backbones) ----------------
#
# A run may serve its unified LLM itself (scripts/vllm/serve.sh) instead of
# calling the OpenAI API. The route table says where a model name lives:
#
#   MAPE_MODEL_ROUTES = {"qwen3-8b": {"base_urls": ["http://127.0.0.1:8300/v1", ...],
#                                        "api_key": "EMPTY",
#                                        "chat_template_kwargs": {"enable_thinking": false},
#                                        "price_usd_per_1m": [0.10, 0.30]}}
#
# Routing is BY MODEL NAME, not by a global OPENAI_BASE_URL: the judge and the
# embedding model keep going to OpenAI while the unified LLM goes to the
# local server. Several base_urls round-robin (one vLLM server per GPU).
# Only chat calls route; embeddings are OpenAI-only (see Embedding).
MODEL_ROUTES_ENV = "MAPE_MODEL_ROUTES"
_ROUTE_RR: Dict[str, int] = {}


def _model_routes() -> Dict[str, Dict[str, Any]]:
    raw = os.environ.get(MODEL_ROUTES_ENV, "").strip()
    if not raw:
        return {}
    try:
        table = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{MODEL_ROUTES_ENV} is not JSON: {exc}") from exc
    return {str(k).lower(): dict(v) for k, v in (table or {}).items()}


def _route_for(model: str) -> Optional[Dict[str, Any]]:
    """The route entry for `model` (its base name, case-insensitive), or None
    when the model is not served locally and goes to the provider's API."""
    base, _ = _split_model_effort(model or "")
    return _model_routes().get(base.lower())


def _route_base_url(route: Dict[str, Any], model: str) -> str:
    """Next base_url for this model — round-robin over the route's servers."""
    urls = route.get("base_urls") or ([route["base_url"]] if route.get("base_url") else [])
    if not urls:
        raise ValueError(f"route for {model!r} has no base_urls")
    i = _ROUTE_RR.get(model, 0)
    _ROUTE_RR[model] = i + 1
    return str(urls[i % len(urls)])


# Anthropic models accepting `output_config.effort` (prefix allowlist —
# sending effort to e.g. claude-haiku-4-5 is a 400). Mirrors the
# _supports_reasoning pattern on the OpenAI side.
_ANTHROPIC_EFFORT_PREFIXES = (
    "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-4-6", "claude-sonnet-5", "claude-fable", "claude-mythos",
)


def _supports_anthropic_effort(model: str) -> bool:
    return any(model.startswith(p) for p in _ANTHROPIC_EFFORT_PREFIXES)


def _require_anthropic_sdk() -> None:
    if anthropic is None:
        raise RuntimeError(
            "model routes to the Anthropic provider but the `anthropic` SDK is "
            "not installed in this environment. Install `anthropic` "
            "(see requirements.txt) or use an OpenAI model."
        )


def _split_model_effort(model: str) -> Tuple[str, Optional[str]]:
    """Parse "model/effort" (e.g. "gpt-5-mini/low") into (model, effort).

    The suffix is a repo-wide convention (config `model:` values, harness
    constants) for pinning a default reasoning effort. No suffix → effort
    None. The effort is only *sent* when the base model supports reasoning
    (see `_supports_reasoning`), so "gpt-4.1/low" degrades gracefully.
    """
    if model and "/" in model:
        base, effort = model.split("/", 1)
        return base, effort
    return model, None


# ---------------- Shared client + global concurrency gate ----------------
#
# Both registries are keyed by the RUNNING event loop (weakly, so dead loops
# don't pin entries):
#   - AsyncOpenAI/httpx pools are loop-bound; reusing one client across loops
#     (e.g. `Embedding.__call__`'s asyncio.run in a worker thread) is a known "Event loop is closed" footgun.
#   - One client per loop means every Agent/Embedding/Judge in the eval shares
#     one connection pool instead of each instance opening its own.
#   - The semaphore is the GLOBAL LLM concurrency gate. Phase 1 harnesses may
#     fan out hundreds of concurrent calls; unbounded concurrency causes
#     mass timeouts (server-side queueing pushes response times past the
#     client timeout). Tune via
#     MAPE_LLM_MAX_CONCURRENT (default 48).

DEFAULT_MAX_CONCURRENT = 48

# loop → {"openai": client, "anthropic": client, "openai@<base_url>": client}
_ASYNC_CLIENTS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[str, Any]]" = (
    weakref.WeakKeyDictionary()
)
_LLM_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _build_anthropic_client():
    """Construct the async Anthropic API client."""
    _require_anthropic_sdk()
    return anthropic.AsyncAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=httpx.Timeout(600.0, connect=10.0),
        max_retries=0,
    )


def _get_async_client(provider: str = "openai", *, base_url: Optional[str] = None,
                      api_key: Optional[str] = None):
    """One shared async client per (running event loop, provider).

    `max_retries=0` because retries live in `_call_with_retries` (the SDK's
    built-in retry would double-retry and ignore our backoff policy).
    Per-request timeouts are passed at call time, so one client serves
    callers with different timeout needs.
    """
    loop = asyncio.get_running_loop()
    per_loop = _ASYNC_CLIENTS.get(loop)
    if per_loop is None:
        per_loop = {}
        _ASYNC_CLIENTS[loop] = per_loop

    key = provider
    if provider == "openai" and base_url:
        key = f"openai@{base_url}"       # one client per local server

    client = per_loop.get(key)
    if client is None:
        if provider == "openai":
            client = AsyncOpenAI(
                api_key=(api_key if base_url else os.getenv("OPENAI_API_KEY")) or "EMPTY",
                base_url=base_url or None,
                timeout=httpx.Timeout(600.0, connect=10.0),
                max_retries=0,
            )
            # Usage on THIS client is reported by the call sites below; the
            # SDK-boundary shim (core/openai_usage.py) that captures the
            # vendored baselines' own clients must skip it or every call
            # would be counted twice.
            from core.openai_usage import OWNED_CLIENT_ATTR
            setattr(client, OWNED_CLIENT_ATTR, True)
        elif provider == "anthropic":
            client = _build_anthropic_client()
        else:
            raise ValueError(f"unknown LLM provider: {provider!r}")
        per_loop[key] = client
    return client


async def _close_loop_client() -> None:
    """Close and drop the current loop's shared clients (if any).

    Called by sync entry points that spin up a throwaway event loop
    (`Embedding.__call__`): the loop dies right after, so its pooled
    connections must be closed inside the loop to avoid httpx destructor
    noise ("Event loop is closed")."""
    loop = asyncio.get_running_loop()
    per_loop = _ASYNC_CLIENTS.pop(loop, None)
    if per_loop:
        for client in per_loop.values():
            await client.close()


def _get_llm_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _LLM_SEMAPHORES.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(
            int(os.getenv("MAPE_LLM_MAX_CONCURRENT", str(DEFAULT_MAX_CONCURRENT)))
        )
        _LLM_SEMAPHORES[loop] = sem
    return sem


# ---------------- Unified retry kernel ----------------


class EmptyResponseError(Exception):
    """API returned no choices / empty content — transient, retryable."""


class LengthLimitError(Exception):
    """Generation hit max_completion_tokens with no visible content —
    deterministic, NOT retryable (reasoning tokens likely ate the budget)."""


class RefusalError(Exception):
    """Anthropic safety classifiers declined the request (stop_reason=refusal)
    — deterministic for a given prompt, NOT retryable."""


_RETRYABLE = [
    openai.APITimeoutError,      # subclass of APIConnectionError; listed for clarity
    openai.APIConnectionError,
    openai.InternalServerError,  # any 5xx
    openai.RateLimitError,       # 429
    asyncio.TimeoutError,
    EmptyResponseError,
]
_RATE_LIMIT_EXCEPTIONS = [openai.RateLimitError]
if anthropic is not None:
    # Same taxonomy as openai's SDK: connection/timeout, ≥500 (incl. 529
    # overloaded), and 429 are transient; other 4xx classes fast-fail.
    _RETRYABLE += [
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
        anthropic.RateLimitError,
    ]
    _RATE_LIMIT_EXCEPTIONS.append(anthropic.RateLimitError)

RETRYABLE_EXCEPTIONS = tuple(_RETRYABLE)
_RATE_LIMIT_EXCEPTIONS = tuple(_RATE_LIMIT_EXCEPTIONS)


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Extract a Retry-After header (seconds) from an API error, if present."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    val = headers.get("retry-after")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _retry_delay(
    exc: BaseException, attempt: int, base_delay: float = 2.0, cap: float = 60.0
) -> float:
    """Jittered exponential backoff: min(cap, base·2^(attempt+1))·U(0.5,1.5).

    Rate limits back off twice as hard; a server-provided Retry-After is a
    floor (never sleep less than the server asked)."""
    delay = min(cap, base_delay * (2 ** (attempt + 1)))
    if isinstance(exc, _RATE_LIMIT_EXCEPTIONS):
        delay = min(cap, delay * 2)
    delay *= random.uniform(0.5, 1.5)
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return delay


async def _call_with_retries(
    op, *, what: str, max_retries: int, base_delay: float = 2.0, cap: float = 60.0
):
    """Run one-attempt coroutine factory `op` under the global concurrency
    gate, retrying transient failures (RETRYABLE_EXCEPTIONS) with jittered
    exponential backoff. Total attempts = max_retries + 1.

    The semaphore is held only while a request is in flight — never across
    backoff sleeps — so a retrying call doesn't starve others of a slot.
    Non-retryable exceptions (4xx other than 429, schema errors, ...)
    propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            async with _get_llm_semaphore():
                return await op()
        except RETRYABLE_EXCEPTIONS as e:
            if attempt == max_retries:
                raise
            delay = _retry_delay(e, attempt, base_delay, cap)
            log.warning(
                f"[{what}] Retryable error (attempt {attempt + 1}/{max_retries}): "
                f"{e!r}, retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)


# ---------------- Shared provider-agnostic chat call ----------------


def _split_system_messages(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """Split chat-format messages into (system_text, non_system_messages).

    Anthropic's Messages API takes the system prompt as a separate `system`
    parameter. Multiple system entries (possible via Agent's with_full_msg
    escape hatch) are joined with blank lines; their relative order and the
    order of the remaining messages are preserved.
    """
    system_parts: List[str] = []
    rest: List[Dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(str(m.get("content", "")))
        else:
            rest.append(m)
    return "\n\n".join(p for p in system_parts if p), rest


async def _chat_completion(
    *,
    model: str,
    messages: List[Dict],
    timeout: float,
    max_retries: int,
    json_mode: bool = False,
    effort: Optional[str] = None,
    max_tokens: Optional[int] = 16384,
    temperature: Optional[float] = None,
    what: str = "Agent",
) -> Tuple[str, Dict[str, int]]:
    """One chat call, provider-routed by model name, through the retry kernel.

    Returns (text, usage) where usage is normalized to OpenAI-style keys
    ({prompt,completion,total}_tokens [+ completion_tokens_details for
    OpenAI reasoning models]) — feed it straight to TokenTracker.update.

    Provider notes:
      openai    — behavior identical to the pre-refactor Agent/Judge payloads
                  (json_mode → response_format json_object; effort gated by
                  _supports_reasoning; max_tokens → max_completion_tokens).
      anthropic — system messages split into the `system` param; `max_tokens`
                  is REQUIRED by the API (None falls back to 16384); effort is
                  sent via extra_body.output_config for models on the
                  _ANTHROPIC_EFFORT_PREFIXES allowlist (extra_body keeps
                  compatibility with older SDKs in prebuilt images); no
                  response_format equivalent — json_mode relies on the prompt
                  (callers already parse-and-retry); temperature is never sent
                  (rejected with 400 on current Claude models);
                  stop_reason refusal → RefusalError (non-retryable),
                  max_tokens with no text → LengthLimitError, empty text →
                  EmptyResponseError (retryable).
    """
    provider = _provider_for_model(model)
    request_timeout = httpx.Timeout(timeout, connect=10.0)

    if provider == "openai":
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if effort and _supports_reasoning(model):
            kwargs["reasoning_effort"] = effort

        # A locally served model: its own client (base_url), and whatever the
        # route asks the chat template for (Qwen3: no thinking preamble).
        route = _route_for(model)
        if route is not None:
            client = _get_async_client("openai", base_url=_route_base_url(route, model),
                                       api_key=str(route.get("api_key") or "EMPTY"))
            if route.get("chat_template_kwargs"):
                kwargs["extra_body"] = {"chat_template_kwargs": dict(route["chat_template_kwargs"])}
        else:
            client = _get_async_client("openai")

        async def _attempt():
            resp = await client.chat.completions.create(**kwargs, timeout=request_timeout)
            if not resp.choices:
                raise EmptyResponseError("API returned no choices")
            choice = resp.choices[0]
            content = choice.message.content
            if content is None or not content.strip():
                if getattr(choice, "finish_reason", None) == "length":
                    raise LengthLimitError(
                        f"generation hit max_completion_tokens={max_tokens} "
                        f"with no visible content (model={model}; reasoning tokens "
                        f"likely consumed the budget) — raise max_completion_tokens "
                        f"or lower the reasoning effort"
                    )
                raise EmptyResponseError("API returned empty content")
            return resp

        resp = await _call_with_retries(_attempt, what=what, max_retries=max_retries)
        usage = getattr(resp, "usage", None)
        usage_dict: Dict[str, int] = {}
        if usage is not None:
            def _get(obj, key):
                return getattr(obj, key, 0) or 0
            usage_dict = {
                "prompt_tokens": _get(usage, "prompt_tokens"),
                "completion_tokens": _get(usage, "completion_tokens"),
                "total_tokens": _get(usage, "total_tokens"),
            }
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                usage_dict["completion_tokens_details"] = {
                    "reasoning_tokens": _get(details, "reasoning_tokens"),
                }
        return resp.choices[0].message.content, usage_dict

    # ---- anthropic (the Anthropic API) ----
    _require_anthropic_sdk()
    system_text, chat_messages = _split_system_messages(messages)
    a_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        # Required by the Messages API; mirror the OpenAI default when the
        # caller disabled the cap.
        "max_tokens": max_tokens if max_tokens is not None else 16384,
    }
    if system_text:
        a_kwargs["system"] = system_text
    if effort and _supports_anthropic_effort(model):
        a_kwargs["extra_body"] = {"output_config": {"effort": effort}}

    a_client = _get_async_client("anthropic")

    async def _a_attempt():
        resp = await a_client.messages.create(**a_kwargs, timeout=request_timeout)
        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason == "refusal":
            raise RefusalError(
                f"Anthropic safety classifiers declined the request "
                f"(model={model}) — deterministic, not retried"
            )
        text = "".join(
            block.text for block in (resp.content or [])
            if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            if stop_reason == "max_tokens":
                raise LengthLimitError(
                    f"generation hit max_tokens={a_kwargs['max_tokens']} with no "
                    f"visible text (model={model}) — raise max_tokens or lower "
                    f"the effort"
                )
            raise EmptyResponseError("API returned empty content")
        return resp, text

    resp, text = await _call_with_retries(_a_attempt, what=what, max_retries=max_retries)
    usage = getattr(resp, "usage", None)
    usage_dict = {}
    if usage is not None:
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        usage_dict = {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }
    return text, usage_dict


# ---------------- Hire Agent ----------------
class Agent:
    """
    Asynchronous chat wrapper, provider-routed by model name: OpenAI models
    (gpt-*, o*) call the OpenAI API; claude-* models call the Anthropic API.
    Allows custom system prompt, user prompt, and optional JSON schema validation.

    All requests go through a per-event-loop shared client and a GLOBAL
    concurrency gate (MAPE_LLM_MAX_CONCURRENT, default 48) — you can fan
    out as many concurrent `ask()` calls as you like without your own
    throttling; excess calls queue on the semaphore.

    `model` accepts an optional "/effort" suffix (e.g. "gpt-5-mini/low") to
    pin a default reasoning effort; it is ignored for non-reasoning models.

    NOTE on `max_retries`: OpenAI's API has noticeable jitter — under
    moderate concurrency, transient timeouts / 5xx / 429s are routine. We
    DEFAULT to `max_retries=5` for a reason: it converts almost all
    transient failures into eventual success. Harnesses that override to a
    smaller value (e.g. `max_retries=2`) silently drop work when the API is
    flaky — Phase 1 fact extraction returns empty, memory is incomplete,
    retrieval scores degrade. **Do not set `max_retries < 5` unless you
    have a specific reason and explicitly handle the failure path.**
    """
    def __init__(
        self,
        system_prompt: str,
        output_schema: Optional[Dict] = None,
        model: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 5,  # see class docstring — keep ≥5 to absorb API jitter
        max_completion_tokens: Optional[int] = 16384,  # None disables the cap
    ):
        # No model → the run's unified LLM (MAPE_UNIFIED_LLM, set by the
        # evaluator), else gpt-4.1. Checked against the run's
        # allowlist here so a harness fails at import-level self-validation,
        # not on the first call mid-evaluation.
        from core import model_policy
        chosen = model or model_policy.unified_llm() or 'gpt-4.1'
        self.model, self._default_reasoning_effort = _split_model_effort(chosen)
        model_policy.check(self.model, where="core.llm.Agent")
        self.messages: List[Dict] = [{'role': 'system', 'content': system_prompt + (f"""ONLY output a valid JSON object conforming to the schema. The json schema is given below: {json.dumps(output_schema, ensure_ascii=False, indent=1)}""" if output_schema else "")}]
        self.output_schema = output_schema or None
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_completion_tokens = max_completion_tokens

    async def ask(self, user_input, with_full_msg: bool = False, with_history: bool = False, temperature=None, reasoning_effort=None) -> Any:
        """
        Send a user message asynchronously and return the model's response.
        If an output schema is defined, the response will be parsed and validated.

        Raises on failure (retry exhaustion, LengthLimitError, non-retryable
        API errors) — an empty/placeholder answer is never fabricated.
        """
        self.messages.append({'role': 'user', 'content': user_input})
        if not with_history:
            chat_messages = [self.messages[0], self.messages[-1]]
        else:
            chat_messages = [self.messages[0]] + self.messages[1:]

        if with_full_msg:
            chat_messages = user_input

        # `if temperature:` (not `is not None`) preserved from the pre-refactor
        # behavior — temperature=0 was never sent. Anthropic models ignore
        # temperature entirely (rejected by current Claude models).
        answer, usage = await _chat_completion(
            model=self.model,
            messages=chat_messages,
            timeout=self.timeout,
            max_retries=self.max_retries,
            json_mode=bool(self.output_schema),
            effort=reasoning_effort or self._default_reasoning_effort,
            max_tokens=self.max_completion_tokens,
            temperature=temperature if temperature else None,
            what="Agent",
        )

        # A completed request counts even when the server returned no usage
        # block — the call was made and paid for. The phase comes from the
        # ambient ContextVar (core.tokens.phase, set by core/workflow.py).
        from core.tokens import GLOBAL_TOKEN_TRACKER
        if GLOBAL_TOKEN_TRACKER is not None:
            GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=usage)

        # Level 2+: what this call actually sent and got back. Before the
        # answer is appended, so `self.messages` is the prompt as sent.
        try:
            from core import trace as _trace
            _trace.record_call(model=self.model, prompt=list(self.messages),
                               response=answer, usage=usage, op="chat")
        except Exception:
            pass

        self.messages.append({'role': 'assistant', 'content': answer})

        # --- Parse according to output schema if provided ---
        if self.output_schema:
            try:
                parsed = json.loads(answer)
                jsonschema.validate(instance=parsed, schema=self.output_schema)
                return parsed
            except json.JSONDecodeError:
                log.warning(f"fail to parse: {answer}")
                return {"error": "Model output is not valid JSON.", "raw_output": answer}
            except jsonschema.ValidationError as e:
                log.warning(f"fail to parse: {answer}")
                return {"error": f"Output does not match schema: {e.message}", "raw_output": answer}

        return answer


class Embedding:
    """
    Async embedding manager for computing single or batch embeddings,
    with optional similarity calculation. Supports token tracking.

    Requests share the per-event-loop client and global concurrency gate
    with `Agent` (see module docstring on _get_async_client).

    Batch chunking respects ALL three OpenAI embedding limits:
      • per-request items   <= 2048
      • per-request tokens  <= 290_000  (cap is 300k; we leave 10k margin)
      • per-input tokens    <= 8192     (text-embedding-3-*)
    Long single inputs are SILENTLY TRUNCATED to the per-input cap with a
    WARNING log line — core.llm.Embedding is the right place to absorb
    this rather than make every harness handle it.
    """

    # Per-process tiktoken encoder cache (cl100k_base is shared by all
    # text-embedding-3-* models, so populated lazily once per encoding).
    _encoder_cache: Dict[str, Any] = {}

    # Class constants for OpenAI embedding API limits.
    MAX_ITEMS_PER_REQUEST = 2048
    MAX_TOKENS_PER_REQUEST = 290_000   # 300k cap; 10k margin for tokenizer/server drift
    MAX_TOKENS_PER_INPUT = 8192        # text-embedding-3-* per-input cap

    def __init__(self, model: Optional[str] = None, retries: int = 4, retry_delay: float = 1.0, timeout: float = 60.0):
        from core import model_policy
        model = model or model_policy.unified_embedding() or "text-embedding-3-small"
        model_policy.check(model, where="core.llm.Embedding")
        if _provider_for_model(model) != "openai":
            raise ValueError(
                f"Embedding model {model!r}: Anthropic has no embeddings API — "
                f"use an OpenAI embedding model (e.g. text-embedding-3-small)."
            )
        self.model = model
        self.retries = retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def _get_encoder(self):
        """Lazy-load + cache a tiktoken encoder matched to the model."""
        if self.model not in self._encoder_cache:
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback: cl100k_base covers all current text-embedding-3-* +
                # gpt-3.5/4 families; a safe default for unknown model strings.
                enc = tiktoken.get_encoding("cl100k_base")
            self._encoder_cache[self.model] = enc
        return self._encoder_cache[self.model]

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Safe synchronous entry point that works both inside and outside async loops."""
        if not texts:
            return []

        async def _run():
            try:
                return await self.get_batch_embeddings(texts)
            finally:
                # This throwaway loop dies right after asyncio.run returns —
                # close its shared client so pooled connections don't outlive
                # the loop (httpx "Event loop is closed" destructor noise).
                await _close_loop_client()

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())

        # `ThreadPoolExecutor.submit` does NOT carry contextvars into the
        # worker (unlike `asyncio.to_thread`), so the ambient LLM phase
        # (core.tokens.phase) would be lost and these embeddings would be
        # filed under "other". Copy the calling context explicitly.
        import concurrent.futures
        import contextvars
        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: ctx.run(lambda: asyncio.run(_run())))
            return future.result()

    async def get_embedding(self, text: str) -> List[float]:
        """Compute embedding for a single text string asynchronously."""
        assert isinstance(text, str)

        client = _get_async_client()
        request_timeout = httpx.Timeout(self.timeout, connect=10.0)

        async def _attempt():
            resp = await client.embeddings.create(
                model=self.model, input=text, timeout=request_timeout
            )
            from core.tokens import GLOBAL_TOKEN_TRACKER
            if GLOBAL_TOKEN_TRACKER is not None:
                GLOBAL_TOKEN_TRACKER.update(
                    model_name=self.model, usage=getattr(resp, "usage", None)
                )
            try:
                from core import trace as _trace
                # The vector itself is not evidence; WHAT was embedded is.
                _trace.record_call(model=self.model, prompt=text, response=None,
                                   usage=getattr(resp, "usage", None), op="embed",
                                   n_inputs=1)
            except Exception:
                pass
            return resp.data[0].embedding

        try:
            return await _call_with_retries(
                _attempt, what="Embedding", max_retries=self.retries,
                base_delay=self.retry_delay,
            )
        except Exception as e:
            # RuntimeError wrap kept for backward compatibility with harness
            # `except RuntimeError` paths. Non-retryable errors (e.g. 400)
            # land here immediately — no pointless retries on deterministic
            # failures.
            log.error(f"[Embedding] Failed (model={self.model}): {e!r}")
            raise RuntimeError(f"Failed to get embedding, with error: {e!r}") from e

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch, chunked by BOTH item count AND token budget.

        Why both: the OpenAI embedding API caps a single request at 2048 items
        AND ~300k tokens. Chunking only by item count fails on long inputs
        (e.g. 2000 dialogue turns can easily exceed 300k tokens). Per-input
        texts longer than 8192 tokens are silently truncated.
        """
        if not texts:
            return []

        enc = self._get_encoder()

        # Pre-tokenize once. Token IDs are reused for the actual API request
        # (sending pre-tokenized inputs guarantees the server sees exactly the
        # token count we budgeted against — no tokenizer/server drift).
        encoded: List[List[int]] = []
        truncated = 0
        for t in texts:
            # disallowed_special=(): a literal "<|endoftext|>" in the data (one
            # LongMemEval-S test haystack has one) is ordinary text, not a control token.
            ids = enc.encode(t or " ", disallowed_special=())  # empty string -> single space (safe; some servers reject [])
            if len(ids) > self.MAX_TOKENS_PER_INPUT:
                ids = ids[: self.MAX_TOKENS_PER_INPUT]
                truncated += 1
            encoded.append(ids)
        if truncated:
            log.warning(
                f"Embedding: truncated {truncated}/{len(texts)} input(s) to "
                f"{self.MAX_TOKENS_PER_INPUT} tokens (per-input cap)."
            )

        # Pack into chunks respecting both caps.
        chunks: List[List[int]] = []  # list of lists of indices into `texts`
        cur: List[int] = []
        cur_tok = 0
        for i, ids in enumerate(encoded):
            n = len(ids)
            if cur and (
                len(cur) >= self.MAX_ITEMS_PER_REQUEST
                or cur_tok + n > self.MAX_TOKENS_PER_REQUEST
            ):
                chunks.append(cur)
                cur = []
                cur_tok = 0
            cur.append(i)
            cur_tok += n
        if cur:
            chunks.append(cur)

        client = _get_async_client()
        request_timeout = httpx.Timeout(self.timeout, connect=10.0)

        # Send chunks; preserve original order in output via index mapping.
        all_embeddings: List[Optional[List[float]]] = [None] * len(texts)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_input = [encoded[i] for i in chunk]
            chunk_tok = sum(len(encoded[i]) for i in chunk)

            async def _attempt(chunk=chunk, chunk_input=chunk_input):
                resp = await client.embeddings.create(
                    model=self.model, input=chunk_input, timeout=request_timeout
                )
                from core.tokens import GLOBAL_TOKEN_TRACKER
                if GLOBAL_TOKEN_TRACKER is not None:
                    GLOBAL_TOKEN_TRACKER.update(
                        model_name=self.model, usage=getattr(resp, "usage", None)
                    )
                try:
                    from core import trace as _trace
                    # The TEXTS, not `chunk_input` — the request carries
                    # pre-tokenized ids, and a list of integers answers no
                    # question anyone would ask of this record. A couple of
                    # examples plus the count: the texts themselves are the
                    # memory, which the level-1 trace already carries.
                    _trace.record_call(model=self.model,
                                       prompt=[texts[i] for i in chunk[:2]],
                                       response=None, usage=getattr(resp, "usage", None),
                                       op="embed", n_inputs=len(chunk_input))
                except Exception:
                    pass
                for j, item in zip(chunk, resp.data):
                    all_embeddings[j] = item.embedding

            try:
                await _call_with_retries(
                    _attempt, what="Embedding", max_retries=self.retries,
                    base_delay=self.retry_delay,
                )
            except Exception as e:
                log.error(
                    f"Embedding: chunk {chunk_idx + 1}/{len(chunks)} "
                    f"({len(chunk)} items, ~{chunk_tok} tokens) failed. Error: {e!r}"
                )
                raise RuntimeError(
                    f"Failed to get batch embeddings. With error: {e!r}"
                ) from e

        return all_embeddings  # type: ignore[return-value]

    @staticmethod
    async def compute_similarity(emb1: List[float], emb2: List[float], metric: str = "cosine") -> float:
        """Asynchronously compute cosine similarity between two embeddings."""
        if metric == "cosine":
            return await asyncio.to_thread(lambda: 1 - cosine(emb1, emb2))
        else:
            log.error(f"Unsupported similarity metric: {metric}")
            raise ValueError(f"Unsupported similarity metric: {metric}")

