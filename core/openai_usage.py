"""Capture the OpenAI traffic of vendored baseline code, without editing it.

Every harness baseline drives its own LLM calls through the `openai` Python
SDK from inside `baselines/*/src/` — zep (7 files), lightmem (5),
simplemem, amem (a lazy import inside `OpenAIController.__init__`),
and mem0 through venv-installed packages. None of that traffic
reached `core.tokens`, which is most of what a memory system actually costs:
amem alone spends 2 gpt-4o-mini calls per ingested note, and build is the
dominant phase.

Why a shim rather than rewriting the call sites
-----------------------------------------------
The method source under `baselines/*/src/` is vendored BYTE-IDENTICAL
to upstream, and each README ships a `diff -r` command asserting it. Editing
`src/graphiti_core/`, `src/simplemem/`, `src/lightmem/` or
`src/memory_layer.py` to swap in our client would break that guarantee and
every faithfulness claim built on it. So we patch at the SDK boundary — the
same pattern the repo already uses in the `_st_shim.py` files — and install it
from `memo.py`, which is integration code, not vendored source.

Why the RESOURCE METHODS, not the client constructors
-----------------------------------------------------
Patching `openai.OpenAI` / `openai.AsyncOpenAI` only works if it happens
before the vendored module runs `from openai import OpenAI`, since that binds
the name at import time — and most of these modules do exactly that at module
level. Patching `Completions.create` (and friends) on the resource CLASSES is
import-order independent, catches clients that were already constructed, and
covers the Azure client classes too (they share these resource classes).

Double counting is avoided by tagging the clients `core.llm` builds: usage
from those is reported by `core.llm` itself, so the shim skips them.

Model routes (a locally served backbone)
----------------------------------------
When MAPE_MODEL_ROUTES names the model a vendored client asks for (the
Qwen backbone served by scripts/vllm/serve.sh), the wrappers send that chat
call to the route's server instead: the ORIGINAL create()/parse() is invoked
on the same resource of a client built for the route's base_url (one per
server, round-robin across `base_urls`, `chat_template_kwargs` added as
`extra_body`). The vendored code keeps its `OpenAI(api_key=...)` and its
model name; nothing under `src/` changes, and OPENAI_BASE_URL stays refused
(`eval_harness.check_environment`) — routing is by model NAME, so the
embedding model and the judge still go to OpenAI. Only chat completions
route: a routed model asked through the Responses API (graphiti's
`responses.parse`) has no local equivalent and is refused with a clear error
rather than a 404 from OpenAI.

Model policy
------------
The same wrappers enforce `core.model_policy`: when MAPE_ALLOWED_MODELS
is set (the eval container), a create() naming any other model raises
ModelPolicyViolation before the request is sent. See that module for why.

Not captured
------------
Streamed responses carry no usage block unless the caller asks for one
(`stream_options={"include_usage": True}`), so a streaming call is counted as
a CALL with zero tokens and warned about once. Local models (LLMlingua-2, the
bge cross-encoder, the local embedders) are not API calls at all and can never
appear here — see `core/tokens.py`.
"""
from __future__ import annotations

import functools
from typing import Any, Optional

from core.logger import get_logger

log = get_logger("main")

_installed = False
_stream_warned = False

#: Attribute stamped on clients built by `core.llm`, whose usage is already
#: reported there. Read by the wrappers to skip double counting.
OWNED_CLIENT_ATTR = "_MAPE_reports_usage"


def _is_ours(resource: Any) -> bool:
    """True when this resource belongs to a client core.llm already tracks."""
    return bool(getattr(getattr(resource, "_client", None), OWNED_CLIENT_ATTR, False))


def _record(resource: Any, response: Any, requested_model: Optional[str],
            streaming: bool) -> None:
    """Report one completed SDK call to the global tracker. Never raises —
    accounting must not be able to fail a baseline's evaluation."""
    try:
        from core.tokens import GLOBAL_TOKEN_TRACKER
        if GLOBAL_TOKEN_TRACKER is None:
            return
        if streaming:
            global _stream_warned
            if not _stream_warned:
                _stream_warned = True
                log.warning(
                    "[openai-usage] a streamed call was made without "
                    "stream_options={'include_usage': True}; it is counted as a "
                    "call with 0 tokens (the API returns no usage block)"
                )
            usage = None
        else:
            usage = getattr(response, "usage", None)
        # The server's echoed model is the ground truth (it resolves aliases
        # and deployment names); fall back to what the caller asked for.
        model = getattr(response, "model", None) or requested_model or "unknown"
        GLOBAL_TOKEN_TRACKER.update(model_name=str(model), usage=usage)
    except Exception as exc:      # pragma: no cover - defensive
        log.debug(f"[openai-usage] failed to record usage: {exc!r}")


# ---------------- Model routes ----------------
_ROUTE_CLIENTS: dict = {}


def _route_client(base_url: str, api_key: str, is_async: bool):
    """One SDK client per (server, sync/async[, event loop]) — the counterpart
    of core.llm's per-server clients, for calls that did not go through
    core.llm."""
    import openai
    loop_key = None
    if is_async:
        import asyncio
        try:
            loop_key = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_key = None
    key = (base_url, is_async, loop_key)
    client = _ROUTE_CLIENTS.get(key)
    if client is None:
        cls = openai.AsyncOpenAI if is_async else openai.OpenAI
        client = cls(api_key=api_key or "EMPTY", base_url=base_url, max_retries=2)
        _ROUTE_CLIENTS[key] = client
    return client


def _same_resource_on(client: Any, resource: Any) -> Any:
    """The resource `resource` is an instance of, on another client:
    chat.completions, or beta.chat.completions (structured outputs)."""
    if ".beta." in type(resource).__module__:
        return client.beta.chat.completions
    return client.chat.completions


def _reroute(resource: Any, kwargs: dict, *, is_async: bool, routable: bool):
    """(resource, kwargs) to actually call. A chat call naming a routed model
    moves to a client for the route's server unless it is already on one
    (core.llm's own route clients pass through). Never raises for an
    unrouted model."""
    model = kwargs.get("model")
    if not model:
        return resource, kwargs
    from core import llm as _llm
    route = _llm._route_for(str(model))
    if route is None:
        return resource, kwargs
    if not routable:
        raise RuntimeError(
            f"model {model!r} is served locally (MAPE_MODEL_ROUTES) but "
            f"{type(resource).__module__}.{type(resource).__name__} has no local "
            f"equivalent — vLLM's OpenAI-compatible server implements chat completions "
            f"only; this baseline cannot run on that backbone")
    urls = [str(u) for u in (route.get("base_urls") or ([route["base_url"]] if route.get("base_url") else []))]
    current = str(getattr(getattr(resource, "_client", None), "base_url", "") or "").rstrip("/")
    if current and any(current == u.rstrip("/") for u in urls):
        return resource, kwargs
    base_url = _llm._route_base_url(route, str(model))
    target = _same_resource_on(
        _route_client(base_url, str(route.get("api_key") or "EMPTY"), is_async), resource)
    if route.get("chat_template_kwargs"):
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("chat_template_kwargs", dict(route["chat_template_kwargs"]))
        kwargs = {**kwargs, "extra_body": extra}
    return target, kwargs


def _record_trace(kwargs: dict, response: Any) -> None:
    """Level 2+: the prompt and response of a VENDORED client's call
    (core.trace). Our own clients are recorded by core.llm, which sees
    the same call one layer up. Never raises."""
    try:
        from core import trace as _trace
        if not _trace.active():
            return
        messages = kwargs.get("messages")
        if messages is not None:
            text = None
            try:
                choices = getattr(response, "choices", None) or []
                text = getattr(getattr(choices[0], "message", None), "content", None)
            except Exception:
                text = None
            _trace.record_call(model=kwargs.get("model"), prompt=messages,
                               response=text, usage=getattr(response, "usage", None),
                               op="chat", client="vendored")
            return
        payload = kwargs.get("input")
        if payload is not None:
            items = payload if isinstance(payload, (list, tuple)) else [payload]
            _trace.record_call(model=kwargs.get("model"), prompt=list(items[:2]),
                               response=None, usage=getattr(response, "usage", None),
                               op="embed", n_inputs=len(items), client="vendored")
    except Exception:
        pass


def _check_policy(resource: Any, method, kwargs: dict) -> None:
    # Before the request, for every client — the ones core.llm builds too.
    # A refused call is never sent, so a violation costs nothing and the
    # trace names the resource and method that made it.
    from core import model_policy
    model_policy.check(kwargs.get("model"),
                       where=f"{type(resource).__name__}.{getattr(method, '__name__', 'create')}")


def _wrap_sync(method, routable: bool = False):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        _check_policy(self, method, kwargs)
        target, kwargs = _reroute(self, kwargs, is_async=False, routable=routable)
        response = method(target, *args, **kwargs)
        if not _is_ours(self):
            _record(self, response, kwargs.get("model"), bool(kwargs.get("stream")))
            _record_trace(kwargs, response)
        return response
    wrapper.__wrapped_by_mape__ = True    # type: ignore[attr-defined]
    return wrapper


def _wrap_async(method, routable: bool = False):
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        _check_policy(self, method, kwargs)
        target, kwargs = _reroute(self, kwargs, is_async=True, routable=routable)
        response = await method(target, *args, **kwargs)
        if not _is_ours(self):
            _record(self, response, kwargs.get("model"), bool(kwargs.get("stream")))
            _record_trace(kwargs, response)
        return response
    wrapper.__wrapped_by_mape__ = True    # type: ignore[attr-defined]
    return wrapper


#: Every entrypoint that issues a billable request. `create` is the obvious
#: one; `parse` is the structured-output variant, and it is NOT a wrapper
#: around `create` — it posts on its own, so hooking `create` alone misses it
#: entirely. zep measured the cost of that gap: graphiti calls
#: `client.responses.parse` for every entity and edge extraction, so a build
#: that made thousands of LLM calls recorded 2223 embeddings and zero chat
#: tokens, and the baseline came out looking 20x CHEAPER than the rest —
#: silently, since the install line still said it was tracking Responses.
_TRACKED_METHODS = ("create", "parse")


def _patch(cls, is_async: bool, routable: bool = False) -> bool:
    patched_any = False
    for name in _TRACKED_METHODS:
        method = getattr(cls, name, None)
        if method is None or getattr(method, "__wrapped_by_mape__", False):
            continue
        setattr(cls, name, _wrap_async(method, routable) if is_async else _wrap_sync(method, routable))
        patched_any = True
    return patched_any


def install() -> bool:
    """Patch the OpenAI SDK's create() methods to report usage. Idempotent.

    Call from a baseline's `memo.py` BEFORE the vendored system is
    constructed. Returns True if anything was patched (False when already
    installed, or when the `openai` SDK is not importable — a baseline running
    on a non-OpenAI stack is not an error).
    """
    global _installed
    if _installed:
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        log.debug("[openai-usage] openai SDK not installed; nothing to patch")
        return False

    patched = []
    # Chat completions + embeddings are what the vendored systems use; the
    # Responses API is patched too so a future upstream bump is covered.
    # Each entry is optional: SDK versions move these around, and a missing
    # one must degrade to "not captured", never to an import error.
    # The last field: whether a call naming a locally served model can be
    # sent to its server (chat completions — what vLLM implements).
    targets = (
        ("openai.resources.chat.completions", "Completions", False, True),
        ("openai.resources.chat.completions", "AsyncCompletions", True, True),
        ("openai.resources.embeddings", "Embeddings", False, False),
        ("openai.resources.embeddings", "AsyncEmbeddings", True, False),
        ("openai.resources.responses", "Responses", False, False),
        ("openai.resources.responses", "AsyncResponses", True, False),
        # The beta namespace is where structured outputs lived before they
        # graduated; graphiti's Azure path still calls it.
        ("openai.resources.beta.chat.completions", "Completions", False, True),
        ("openai.resources.beta.chat.completions", "AsyncCompletions", True, True),
    )
    for module_path, cls_name, is_async, routable in targets:
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError):
            continue
        if _patch(cls, is_async, routable):
            patched.append(f"{cls_name}")

    _installed = True
    if patched:
        log.info(f"[openai-usage] tracking SDK calls via {', '.join(patched)}")
    else:
        log.warning(
            "[openai-usage] found no OpenAI resource classes to patch — this "
            "baseline's internal LLM calls will NOT appear in token_usage.json"
        )
    return bool(patched)
