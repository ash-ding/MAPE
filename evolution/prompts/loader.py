"""Prompt part loading.

The prompt set lives under `evolution/prompts/parts/`:

    evolving.md     HOW TO EVOLVE: the loop, the workspace layout, the
                    harness contract, the rules. Carries the
                    `<<DESIGN_KNOWLEDGE_BLOCK>>` sentinel.
    design.md       HOW TO DESIGN A MEMORY SYSTEM: the search direction.
                    Substituted into that sentinel (empty in this release).
    task.md         the per-iteration task prompt
    fix.md          the post-sanity-failure fix prompt
    analyst.md      the prompt for the post-evaluation diagnosis agent
    fragments.py    keyed substitution data (see that file's docstring)

The two prose halves are separate files on purpose: `evolving.md` is
mechanical and follows the loop's own file organization, while `design.md`
is the search direction. Swapping `design.md` alone is a one-file change.
The parts are loaded once per process and cached.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class PromptPartsError(RuntimeError):
    """Raised when a prompt part is missing or malformed. Always actionable."""


_PARTS_DIR = Path(__file__).resolve().parent / "parts"

# Where the design half is spliced into the evolving half.
DESIGN_SENTINEL = "<<DESIGN_KNOWLEDGE_BLOCK>>\n"

# Exports fragments.py must provide — checked at load time so a malformed file
# fails fast with a clear message instead of an AttributeError in the renderer.
_REQUIRED_FRAGMENTS = (
    "OBJECTIVE_AXES_SUBS",
    "SANITY_ON_SUBS",
    "SANITY_OFF_SUBS",
    "DATASET_INFO",
    "DATASET_RENDER_ORDER",
    "TASK_DATASET_SUBS",
    "CLOSE_THE_LOOP_SUBS",
    "ANALYST_EVIDENCE",
    "COMPOSE_SUBS",
)


@dataclass(frozen=True)
class PromptParts:
    """The prompt set, assembled. Attribute names match what the renderer reads."""

    SYSTEM_TEMPLATE: str
    TASK_PROMPT_TEMPLATE: str
    FIX_PROMPT_TEMPLATE: str
    ANALYST_TEMPLATE: str
    OBJECTIVE_AXES_SUBS: Dict[Any, str]
    SANITY_ON_SUBS: Dict[str, str]
    SANITY_OFF_SUBS: Dict[str, str]
    DATASET_INFO: Dict[str, Dict[str, str]]
    DATASET_RENDER_ORDER: List[str]
    TASK_DATASET_SUBS: Dict[str, Dict[str, str]]
    #: {"on": ..., "off": ...} for <<CLOSE_THE_LOOP_BLOCK>> in the task prompt:
    #: who writes the post-evaluation analysis.
    CLOSE_THE_LOOP_SUBS: Dict[str, str]
    #: The phrase for <<ANALYST_EVIDENCE>> inside the analyst-on block.
    ANALYST_EVIDENCE: str
    #: {"on": {sentinel: text}, "off": {sentinel: text}} for the <<COMPOSE_*>>
    #: sentinels: what the proposer is told about the third hook
    #: (cfg.contract.compose).
    COMPOSE_SUBS: Dict[str, Dict[str, str]]


_PARTS_CACHE: Optional[PromptParts] = None


def _read_part(kind: str) -> str:
    path = _PARTS_DIR / f"{kind}.md"
    if not path.exists():
        raise PromptPartsError(f"prompts: {kind} part not found at {path}")
    return path.read_text(encoding="utf-8")


def load_prompt_parts() -> PromptParts:
    """Assemble the prompt set (cached after the first call).

    Raises PromptPartsError if a part file is missing, the evolving half does
    not carry exactly one design sentinel, or fragments.py lacks any of
    `_REQUIRED_FRAGMENTS`.
    """
    global _PARTS_CACHE
    if _PARTS_CACHE is not None:
        return _PARTS_CACHE

    evolving = _read_part("evolving")
    design = _read_part("design")
    n = evolving.count(DESIGN_SENTINEL)
    if n != 1:
        raise PromptPartsError(
            f"prompts: evolving.md carries the design sentinel "
            f"{DESIGN_SENTINEL.strip()} {n} times; expected exactly 1."
        )

    fragments = importlib.import_module("evolution.prompts.parts.fragments")
    missing = [name for name in _REQUIRED_FRAGMENTS if not hasattr(fragments, name)]
    if missing:
        raise PromptPartsError(f"prompts: fragments.py is missing required exports: {missing}.")

    _PARTS_CACHE = PromptParts(
        SYSTEM_TEMPLATE=evolving.replace(DESIGN_SENTINEL, design),
        TASK_PROMPT_TEMPLATE=_read_part("task"),
        FIX_PROMPT_TEMPLATE=_read_part("fix"),
        ANALYST_TEMPLATE=_read_part("analyst"),
        **{name: getattr(fragments, name) for name in _REQUIRED_FRAGMENTS},
    )
    return _PARTS_CACHE
