"""What an evaluation records. Every evaluation records all three levels below.

**Level 1** is the evaluation's own artifacts, and needs no code here: per
question, `traces/<user>.json` holds the query, the gold answer, the gold
evidence, the prediction, the retrieved memory and the judge's reasoning. It
says WHAT went wrong and never WHERE. A fact missing from an answer could have
been dropped by the extractor, missed by retrieval, or ignored by the answer
model, and those are three different bugs with three different fixes. Every
"why" in a retro written so far is an inference from category-level movement,
never an observation: one says sentence fusion "diluted turn ranking, pushing
the turns cat-1 aggregation depends on out of the top-k" without ever having
opened that ranking.

**Level 2** adds one record per LLM and embedding call, with the prompt and
the response, keyed so it joins to the question it belongs to.

**Level 3** adds the harness's own function calls: `instrument()` wraps the
module-level functions and class methods defined under the harness directory,
so each call writes an `op` record with its arguments, its return value and
its wall time. It is not a profiler — nested functions, comprehensions and
property accessors are not wrapped — and it answers the question level 2
cannot: what the harness's own code did with what the model gave it. The
date resolver, the ranking function and the dedupe make no LLM call at all,
so at level 2 they are invisible.

Layout, next to the level-1 traces so the pair is obvious::

    <out_dir>/traces/<user_id>.json    level 1 — the QA trajectory
    <out_dir>/calls/<user_id>.jsonl    level 2+ — one record per line

Three kinds of record::

    {"kind": "call", "user", "phase", "query_index", "op", "model",
     "prompt", "response", "usage"}
    {"kind": "qa",   "user", "query_index", "query", "reference", "evidence",
     "predicted", "retrieved", "score", "judge_reason"}
    {"kind": "op",   "user", "query_index", "fn", "args", "ret", "seconds"}

`qa` is a flat copy of the level-1 record, written from the same recorder
steps that produce `traces/<user>.json`, so the two can never disagree. It is
what a `call` joins against.

**Plain text, one file per user, deliberately.** The reader is an agent with
`grep`, `jq` and `head`. Compression would save ~4x on disk and cost that
agent a turn every time it forgets `zcat` — and the measured sizes do not
justify the trade: a frontier-shaped candidate writes 1-2 MB per user, and
only a 5-way answer ensemble reaches ~100 MB. Per-user files mean the reader
opens exactly the user it is asking about, and the byte budget below protects
each file's greppability on its own.

**BUILD-phase calls carry `query_index: null`.** They run before any question
and belong to the whole user, so one bad extraction shows up in every later
question for that user. Any analysis joining calls to questions has to treat
build as a per-user prefix, not as part of one question.

Every record is flushed as it is written, so a trace can be read while the
evaluation is still running and a killed container loses nothing.

Nothing here can change an evaluation's result. The sink only observes, every
write is wrapped, and any failure degrades to "not recorded" with one warning.
Hitting the per-user byte budget writes an explicit `truncated` record rather
than silently dropping the rest.
"""
from __future__ import annotations

import json
import functools
import inspect
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

from core.logger import get_logger

log = get_logger("main")

#: Carried into the eval container by evolution/evaluator.py.
MAX_VALUE_ENV = "MAPE_TRACE_MAX_VALUE"
MAX_BYTES_ENV = "MAPE_TRACE_MAX_BYTES"

DEFAULT_MAX_VALUE = 20_000
#: Per USER, not per candidate. The heaviest harness measured (a 5-way answer
#: ensemble, 50 M tokens over two users) writes ~100 MB per user, so this
#: keeps that intact and still catches a runaway.
DEFAULT_MAX_BYTES = 256_000_000

_SINK: Optional["_Sink"] = None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _safe_name(user: Optional[str]) -> str:
    name = (user or "_unscoped").replace("/", "_").replace("\\", "_").strip()
    return name or "_unscoped"


class _Sink:
    """One open `calls/<user>.jsonl` per user, created on that user's first
    record. Thread-safe: harness hooks run on worker threads
    (`asyncio.to_thread`), so several users may write at once."""

    def __init__(self, calls_dir: Path, *, max_value: int, max_bytes: int):
        self.dir = calls_dir
        self.max_value = max_value
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._files: Dict[str, TextIO] = {}
        self._written: Dict[str, int] = {}
        self._stopped: Dict[str, bool] = {}
        self._warned_unscoped = False
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, user: Optional[str], record: Dict[str, Any]) -> None:
        name = _safe_name(user)
        if user is None and not self._warned_unscoped:
            # A call nobody can attribute is a placement bug, not a data
            # point: it means the scope was set after the work started.
            self._warned_unscoped = True
            log.warning("[trace] a call was recorded with no user in scope; it lands in "
                        "_unscoped.jsonl and joins to nothing — see core.tokens.set_user")
        line = json.dumps(record, ensure_ascii=False, default=_fallback)
        with self._lock:
            if self._stopped.get(name):
                return
            fh = self._files.get(name)
            if fh is None:
                fh = open(self.dir / f"{name}.jsonl", "a", encoding="utf-8")
                self._files[name] = fh
                self._written[name] = 0
            if self._written[name] + len(line) > self.max_bytes:
                self._stopped[name] = True
                fh.write(json.dumps({"kind": "truncated", "user": name, "reason": "max_bytes",
                                     "bytes_written": self._written[name],
                                     "max_bytes": self.max_bytes}) + "\n")
                fh.flush()
                log.warning(f"[trace] {name}: byte budget reached ({self.max_bytes}); "
                            f"no further records for this user")
                return
            fh.write(line + "\n")
            fh.flush()          # a still-running evaluation's trace is readable
            self._written[name] += len(line) + 1

    def close(self) -> None:
        with self._lock:
            for fh in self._files.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._files.clear()


def _fallback(obj: Any) -> str:
    return repr(obj)[:200]


def install(out_dir: Path) -> None:
    """Open the call/op sink for this evaluation. Called once by
    `evolution/launch.py` (and `scripts/run_method.py`) before the harness runs."""
    global _SINK
    close()
    try:
        _SINK = _Sink(Path(out_dir) / "calls",
                      max_value=_int_env(MAX_VALUE_ENV, DEFAULT_MAX_VALUE),
                      max_bytes=_int_env(MAX_BYTES_ENV, DEFAULT_MAX_BYTES))
        log.info(f"[trace] recording calls under {_SINK.dir}")
    except Exception as exc:
        _SINK = None
        log.warning(f"[trace] the sink could not be opened: {exc!r}")


def close() -> None:
    global _SINK
    if _SINK is not None:
        _SINK.close()
        _SINK = None


def active() -> bool:
    return _SINK is not None


def clip(value: Any, limit: Optional[int] = None) -> Any:
    """Shorten long strings anywhere inside `value`, marking what was cut.

    A prompt is worth keeping in full up to the cap; past it the tail is
    almost never what an attribution turns on, and an unbounded copy of a
    200 K-token context would dominate the file."""
    lim = limit if limit is not None else (_SINK.max_value if _SINK else DEFAULT_MAX_VALUE)
    if isinstance(value, str):
        if len(value) <= lim:
            return value
        return value[:lim] + f"…[+{len(value) - lim} chars]"
    if isinstance(value, dict):
        return {k: clip(v, lim) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clip(v, lim) for v in value]
    return value


def _scope() -> Dict[str, Any]:
    from core import tokens
    return {"user": tokens.current_user(), "phase": tokens.current_phase(),
            "query_index": tokens.current_query()}


def record_call(*, model: str, prompt: Any, response: Any,
                usage: Any = None, op: str = "chat", **extra: Any) -> None:
    """One completed LLM or embedding call. Never raises."""
    if _SINK is None:
        return
    try:
        scope = _scope()
        rec = {"kind": "call", **scope, "op": op, "model": str(model),
               "prompt": clip(prompt), "response": clip(response)}
        if usage is not None:
            rec["usage"] = _usage_dict(usage)
        rec.update({k: clip(v) for k, v in extra.items()})
        _SINK.write(scope["user"], rec)
    except Exception as exc:
        log.debug(f"[trace] record_call failed: {exc!r}")


def record_qa(*, user: str, query_index: int, step: Dict[str, Any]) -> None:
    """One question's level-1 record, flattened so calls can join to it.

    Written from the same recorder step that produces `traces/<user>.json`,
    so the two cannot drift apart."""
    if _SINK is None:
        return
    try:
        _SINK.write(user, {
            "kind": "qa", "user": user, "query_index": query_index,
            "query": clip(step.get("query")),
            "reference": clip(step.get("reference")),
            "evidence": clip(step.get("relevant_turns")),
            "predicted": clip(step.get("predicted")),
            "retrieved": clip(step.get("retrieved_memory")),
            "score": step.get("score"),
            "judge_reason": clip(step.get("judge_reason")),
            "qa_metadata": step.get("qa_metadata"),
        })
    except Exception as exc:
        log.debug(f"[trace] record_qa failed: {exc!r}")


def record_op(*, fn: str, args: Any = None, ret: Any = None,
              seconds: Optional[float] = None) -> None:
    """One harness function call (level 3). Never raises."""
    if _SINK is None:
        return
    try:
        scope = _scope()
        rec = {"kind": "op", **scope, "fn": fn}
        if args is not None:
            rec["args"] = clip(args, 2000)
        if ret is not None:
            rec["ret"] = clip(ret, 2000)
        if seconds is not None:
            rec["seconds"] = round(seconds, 4)
        _SINK.write(scope["user"], rec)
    except Exception as exc:
        log.debug(f"[trace] record_op failed: {exc!r}")


#: Level-3 op records are short by construction: one per call, and a harness
#: helper can run thousands of times per user. Values are summarised rather
#: than copied — the full prompt of any LLM call is already in the level-2
#: record next to it, and the full retrieved memory is in the level-1 trace.
OP_VALUE_CHARS = 300
_OP_MARK = "_MAPE_traced"


def _brief(value: Any, depth: int = 0) -> Any:
    """A bounded summary of one argument or return value.

    Not `clip`: that shortens strings but copies a list of 400 turns whole,
    and at level 3 that list is an argument to a function called once per
    query. Containers keep their length and their first two elements, which
    is what an attribution actually reads."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= OP_VALUE_CHARS else value[:OP_VALUE_CHARS] + f"…[+{len(value) - OP_VALUE_CHARS} chars]"
    if depth >= 2:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        out: Dict[str, Any] = {"__len__": len(value)}
        for k, v in list(value.items())[:2]:
            out[str(k)[:60]] = _brief(v, depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        return {"__type__": type(value).__name__, "__len__": len(seq),
                "head": [_brief(v, depth + 1) for v in seq[:2]]}
    try:
        return repr(value)[:OP_VALUE_CHARS]
    except Exception:
        return f"<{type(value).__name__}>"


def _wrap(fn: Callable, label: str, *, drop_self: bool) -> Callable:
    """Wrap one harness callable so each call writes an `op` record.

    A raising call is recorded too, with `error`, and then re-raised: a
    harness that swallows its own exception is exactly the case where the
    level-1 trace shows a wrong answer and no reason for it."""
    def _args(a: tuple, kw: dict) -> Dict[str, Any]:
        pos = a[1:] if drop_self else a
        rec: Dict[str, Any] = {}
        if pos:
            rec["args"] = [_brief(v) for v in pos]
        if kw:
            rec["kwargs"] = {str(k)[:60]: _brief(v) for k, v in kw.items()}
        return rec

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def awrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                out = await fn(*a, **kw)
            except BaseException as exc:
                record_op(fn=label, args=_args(a, kw) or None,
                          ret={"error": f"{type(exc).__name__}: {exc}"[:OP_VALUE_CHARS]},
                          seconds=time.perf_counter() - t0)
                raise
            record_op(fn=label, args=_args(a, kw) or None, ret=_brief(out),
                      seconds=time.perf_counter() - t0)
            return out
        wrapper: Callable = awrapper
    else:
        @functools.wraps(fn)
        def swrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                out = fn(*a, **kw)
            except BaseException as exc:
                record_op(fn=label, args=_args(a, kw) or None,
                          ret={"error": f"{type(exc).__name__}: {exc}"[:OP_VALUE_CHARS]},
                          seconds=time.perf_counter() - t0)
                raise
            record_op(fn=label, args=_args(a, kw) or None, ret=_brief(out),
                      seconds=time.perf_counter() - t0)
            return out
        wrapper = swrapper
    setattr(wrapper, _OP_MARK, True)
    return wrapper


def _own_modules(root: Path) -> List[Any]:
    """Every already-imported module whose source file lives under `root`."""
    out = []
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            if Path(f).resolve().is_relative_to(root):
                out.append(mod)
        except (OSError, ValueError):
            continue
    return out


def instrument(harness_dir: Path) -> int:
    """LEVEL 3: wrap the harness's own functions so every call writes an `op`.

    Call AFTER the harness has been imported (`evolution/launch.py`), so its
    modules are in `sys.modules` and their classes exist.

    This is deliberately NOT a profiler. `sys.setprofile` would fire on every
    frame in the process — library internals, the event loop, the retry
    kernel — and cost far more than it tells anyone. What a diagnosis reads is
    the harness's OWN call boundaries: the ranking function, the dedupe, the
    date resolver. So: module-level functions and class methods defined in
    files under `harness_dir`, and nothing else.

    What that leaves out, and should be said in the prompt rather than
    discovered: functions nested inside other functions, comprehensions,
    lambdas, and `property` accessors are not wrapped; a method called
    through a reference captured before this ran keeps the original.

    Returns how many callables were wrapped.
    """
    try:
        root = Path(harness_dir).resolve()
    except OSError:
        return 0
    n = 0
    for mod in _own_modules(root):
        mod_name = getattr(mod, "__name__", "?")
        try:
            members = list(vars(mod).items())
        except Exception:
            continue
        for name, obj in members:
            if name.startswith("__"):
                continue
            try:
                if inspect.isfunction(obj) and not getattr(obj, _OP_MARK, False):
                    if _defined_under(obj, root):
                        setattr(mod, name, _wrap(obj, f"{mod_name}.{name}", drop_self=False))
                        n += 1
                elif inspect.isclass(obj) and _defined_under(obj, root):
                    n += _wrap_class(obj, root)
            except Exception as exc:                       # never break a harness
                log.debug(f"[trace] could not wrap {mod_name}.{name}: {exc!r}")
    log.info(f"[trace] level 3: instrumented {n} harness callables under {root.name}")
    return n


def _defined_under(obj: Any, root: Path) -> bool:
    try:
        f = inspect.getfile(obj)
    except (TypeError, OSError):
        return False
    try:
        return Path(f).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _wrap_class(cls: type, root: Path) -> int:
    n = 0
    for name, raw in list(vars(cls).items()):
        if name.startswith("__"):
            continue
        label = f"{cls.__module__}.{cls.__name__}.{name}"
        try:
            if isinstance(raw, staticmethod):
                inner = raw.__func__
                if getattr(inner, _OP_MARK, False) or not _defined_under(inner, root):
                    continue
                setattr(cls, name, staticmethod(_wrap(inner, label, drop_self=False)))
            elif isinstance(raw, classmethod):
                inner = raw.__func__
                if getattr(inner, _OP_MARK, False) or not _defined_under(inner, root):
                    continue
                setattr(cls, name, classmethod(_wrap(inner, label, drop_self=True)))
            elif inspect.isfunction(raw):
                if getattr(raw, _OP_MARK, False) or not _defined_under(raw, root):
                    continue
                setattr(cls, name, _wrap(raw, label, drop_self=True))
            else:
                continue                                    # property, descriptor, data
            n += 1
        except (AttributeError, TypeError) as exc:          # slots, frozen, C types
            log.debug(f"[trace] could not wrap {label}: {exc!r}")
    return n


def _usage_dict(usage: Any) -> Dict[str, Any]:
    if isinstance(usage, dict):
        src: Dict[str, Any] = usage
    else:
        src = {k: getattr(usage, k, None) for k in
               ("prompt_tokens", "completion_tokens", "total_tokens",
                "input_tokens", "output_tokens")}
    return {k: v for k, v in src.items() if isinstance(v, (int, float))}


def read(path: Path) -> List[Dict[str, Any]]:
    """Every record in one `calls/<user>.jsonl`, skipping a torn final line.

    A file cut short by a killed container ends mid-line, which is not an
    error worth raising."""
    out: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except OSError:
        pass
    return out
