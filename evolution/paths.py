"""Shared paths for evolution modules — runtime-configurable per run.

Evolution isolates each orchestrator invocation under `workspace/<run_id>/`. The
run_id is set once at startup by `paths.set_run_id(...)`; afterwards
`paths.workspace`, `paths.harnesses_dir`, etc. resolve to the per-run
locations.

Why a singleton instead of plain module constants: callers do
`from evolution.paths import paths` and access attributes (e.g.
`paths.harnesses_dir`). If we exported plain constants, modules would
capture a snapshot of the value at import time and miss the run_id
update. The singleton hands out fresh values via property access.

Bare-import constants (PROJECT_ROOT, EVOLUTION_ROOT,
WORKSPACE_BASE, IMAGES_DIR, EVAL_BASE_SIF, ...) are immutable
across runs and remain plain module-level paths.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: The file that defines a harness's `MemoClass` subclass — the ONE interface,
#: the same for a packaged baseline and for an evolved candidate. Whatever else
#: the harness needs goes under `IMPL_DIR` beside it, organised however it
#: likes: evolution only ever imports this file.
ENTRY_FILE = "memo.py"

#: Where a harness keeps its implementation. Fixed name, free contents — a
#: single module, a package tree, a vendored library.
IMPL_DIR = "src"

def entry_file(harness_dir: Path) -> Optional[Path]:
    """The harness's interface file, or None if it has none."""
    candidate = harness_dir / ENTRY_FILE
    return candidate if candidate.is_file() else None

EVOLUTION_ROOT: Path = PROJECT_ROOT / "evolution"

# Per-run workspaces live under WORKSPACE_BASE/<run_id>/. The base dir itself
# holds nothing; only its children matter.
WORKSPACE_BASE: Path = PROJECT_ROOT / "workspace"

CONTAINERS_DIR: Path = PROJECT_ROOT / "containers"

# SIFs live on scratch (too big for project tree). Shared cache across runs.
# Where the Singularity images live: MAPE_IMAGES_DIR in the environment (an
# absolute directory), else .cache/images under the repository.
IMAGES_DIR: Path = Path(
    os.environ.get("MAPE_IMAGES_DIR") or str(PROJECT_ROOT / ".cache" / "images")
)
EVAL_BASE_SIF: Path = IMAGES_DIR / "eval-base.sif"
PROPOSER_BASE_SIF: Path = IMAGES_DIR / "proposer-base.sif"

# proot is required by `singularity build` for unprivileged delta builds.
PROOT_BIN_DIR: Path = Path.home() / ".local" / "bin"

LOGS_DIR: Path = EVOLUTION_ROOT / "logs"


class _Paths:
    """Run-scoped path resolver. Use the module-level `paths` instance."""

    def __init__(self) -> None:
        self._run_id: Optional[str] = None

    def set_run_id(self, run_id: str) -> None:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise ValueError(f"invalid run_id: {run_id!r}")
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        if self._run_id is None:
            raise RuntimeError(
                "paths.run_id accessed before paths.set_run_id() was called. "
                "The orchestrator must call set_run_id during startup."
            )
        return self._run_id

    @property
    def workspace(self) -> Path:
        return WORKSPACE_BASE / self.run_id

    @property
    def harnesses_dir(self) -> Path:
        return self.workspace / "harnesses"

    @property
    def runs_dir(self) -> Path:
        return self.workspace / "runs"

    @property
    def frontier_path(self) -> Path:
        return self.workspace / "frontier.json"

    @property
    def history_path(self) -> Path:
        """Ordered log of what each search step produced.

        Harness dirs are named by the hash of their code, which says WHAT a
        harness is but not WHEN it arrived; this file is where the order
        lives. Appended as each candidate settles, so an interrupted run
        keeps the steps it finished.
        """
        return self.workspace / "history.json"


paths = _Paths()


def ensure_dirs() -> None:
    """Create all directories needed by the current run.

    Must be called AFTER `paths.set_run_id(...)`.

    Notes:
      - `paths.runs_dir` not pre-created — `run_evaluation` auto-mkdirs the
        per-eval subdir on demand, and `_publish_run_artifacts` rmtrees both
        the subdir and the empty `runs/` parent. So `runs/` only exists while
        at least one eval is in flight.
    """
    for d in (paths.workspace, paths.harnesses_dir,
              IMAGES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
