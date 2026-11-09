"""Prompt parts for the proposer and the diagnosis agent.

The prompt set is four Markdown files plus one Python fragments module under
`evolution/prompts/parts/`. The two prose halves are deliberately separate:

  - `evolving.md` — how to run the search: workspace layout, the two-phase
    protocol, the harness contract, the rules.
  - `design.md`   — how to design a good memory system: the search
    direction. Spliced into the evolving half at `<<DESIGN_KNOWLEDGE_BLOCK>>`.

Splitting them means the design half can be replaced on its own — by hand, or
eventually by something derived from execution evidence — without touching the
mechanical half.

Public API:
  - build_proposer_system(*, sanity_enabled, active_datasets, metrics, compose_enabled)
  - proposer_task_prompt(new_dir_rel, *, active_datasets, analyst_enabled)
  - proposer_fix_prompt(new_dir_rel, error_trace, *, compose_enabled)
  - load_prompt_parts()           — the assembled set (tests / introspection)
  - PromptParts, PromptPartsError
"""

from .loader import (
    PromptParts,
    PromptPartsError,
    load_prompt_parts,
)
from .renderer import (
    build_proposer_system,
    proposer_fix_prompt,
    proposer_task_prompt,
)

__all__ = [
    "PromptParts",
    "PromptPartsError",
    "build_proposer_system",
    "load_prompt_parts",
    "proposer_fix_prompt",
    "proposer_task_prompt",
]
