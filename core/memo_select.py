"""Pick the memory system a loaded harness file defines.

Shared by every loader of generated / candidate harness files (the evolution loop's
container launch + host-side contract check). Kept out of
core/memo_class.py on purpose: that file is the contract itself, and the
proposer reads its source verbatim.
"""
from __future__ import annotations

import inspect
from typing import Iterable

from core.memo_class import MemoClass


def select_memo_class(candidates: Iterable[type], source: str) -> type:
    """Return the first concrete MemoClass subclass among `candidates`.

    Abstract bases (MemoClass itself, evolution.memo_class.MemoClass, a harness's
    own helper base) are skipped. When every candidate is abstract, the error
    names the hooks left unimplemented — BUILD and RETRIEVE are required, so a
    missing or misspelled hook fails here, legibly, instead of at instantiation.
    """
    classes = [c for c in candidates if inspect.isclass(c) and issubclass(c, MemoClass)]
    for cls in classes:
        if not inspect.isabstract(cls):
            return cls
    if classes:
        cls = classes[0]
        missing = ", ".join(sorted(cls.__abstractmethods__))
        raise TypeError(
            f"`{cls.__name__}` in {source} does not implement: {missing}. A memory system "
            f"must define both `build_memory_from_data` and `retrieve_memory_for_query`."
        )
    raise TypeError(f"no MemoClass subclass found in {source}")
