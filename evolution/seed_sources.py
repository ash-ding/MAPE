"""What a seed may be, and how a registered name becomes runnable code.

A seed names one of the baselines registered in
`baselines.eval_harness.MEMOS` — `mem0`, `zep`, `no_memory`, …. Nothing else.

The rule exists because the old one was "any directory containing a memo.py",
which accepted a directory holding two unrelated lines, copied it into the
workspace, paid for an image build, and only failed inside the container. The
one spelling it did NOT accept was the registered name itself.

Packaging is not the caller's problem. A registered name is turned into a
self-contained harness by `scripts/package_baseline.py` on first use and cached
under `packaged/`, keyed by the settings that change the code it emits. The
packager has to run in that baseline's own venv — it imports the vendored
method to resolve its config — so this shells out to `uv run`, which is also
the only way the seven baselines with their own `pyproject.toml` can be
imported at all.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from core.logger import get_logger
from evolution.paths import ENTRY_FILE, PROJECT_ROOT, entry_file

log = get_logger("main")

#: Where `uv run python -m scripts.package_baseline` leaves its output. One
#: directory per (baseline, arm, models); gitignored, safe to delete.
PACKAGED_DIR: Path = PROJECT_ROOT / "packaged"

#: A seed may also be written as the path to its baseline, which is how
#: configs named `no_memory` before the registry was the spelling. It resolves
#: through the registry all the same — the prefix is an alias, not a path
#: lookup, so `baselines/typo` still fails as an unknown name.
_BASELINE_PREFIX = "baselines/"


class SeedSourceError(Exception):
    """A seed that cannot be resolved. Always fatal: a run seeded with the
    wrong thing costs the same as one seeded with the right thing."""


def registered_names() -> List[str]:
    """The baselines a seed may name, in the order a human would list them."""
    from baselines.eval_harness import MEMOS
    return sorted(MEMOS)


def _cache_key(name: str, arm: str, unified_models: Optional[Dict[str, str]]) -> str:
    """Settings that change the packaged code, as a short stable digest.

    The packager bakes the resolved config into the entry file, so a baseline
    packaged for one model is different code from the same baseline packaged
    for another — and must not share a directory with it.
    """
    payload = json.dumps({"name": name, "arm": arm,
                          "unified_models": unified_models or {}},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _package_command(name: str, arm: str, unified_models: Optional[Dict[str, str]],
                     out: Path) -> List[str]:
    """`uv run` in the baseline's own project when it has one.

    `icl` and `no_memory` have no `pyproject.toml` — they call no model and
    ship no vendored source, so they run from the repo-root venv.
    """
    project = PROJECT_ROOT / "baselines" / name
    cmd = ["uv", "run"]
    if (project / "pyproject.toml").exists():
        cmd += ["--project", str(project)]
    cmd += ["python", "-m", "scripts.package_baseline", name, "--arm", arm,
            "--out", str(out)]
    models = unified_models or {}
    if arm == "unified":
        if models.get("llm"):
            cmd += ["--unified-llm", str(models["llm"])]
        if models.get("embedding"):
            cmd += ["--unified-embedding", str(models["embedding"])]
    return cmd


def ensure_packaged(name: str, *, arm: str = "unified",
                    unified_models: Optional[Dict[str, str]] = None,
                    cache_dir: Optional[Path] = None) -> Path:
    """Package `name` if it is not already cached, and return the directory.

    Cached by the settings that produce the code, so repeated runs (and
    repeated seeds within one run) package once.
    """
    root = cache_dir or PACKAGED_DIR
    out = root / f"{name}_{arm}_{_cache_key(name, arm, unified_models)}"
    if entry_file(out) is not None:
        log.info(f"seed {name!r}: using packaged baseline at {out}")
        return out

    cmd = _package_command(name, arm, unified_models, out)
    log.info(f"seed {name!r}: packaging ({' '.join(cmd[:4])} …) → {out}")
    root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                            capture_output=True, text=True)
    if result.returncode != 0 or entry_file(out) is None:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-12:]
        raise SeedSourceError(
            f"could not package baseline {name!r} (exit {result.returncode}).\n"
            f"  command: {' '.join(cmd)}\n"
            f"  Its dependencies live in its own venv — `uv sync --project "
            f"baselines/{name}` if that venv was never built.\n"
            + "\n".join(f"  | {line}" for line in tail)
        )
    return out


def resolve(source: str, *, arm: str = "unified",
            unified_models: Optional[Dict[str, str]] = None,
            cache_dir: Optional[Path] = None) -> Path:
    """The directory holding this seed's code, packaging it if need be.

    Accepted: a registered baseline name (`mem0`), or the `baselines/mem0`
    spelling of one — packaged on demand. Anything else raises. A path that
    merely contains a `memo.py` is NOT a seed: a typo'd path would otherwise
    fail only inside the container, after an image build.
    """
    name = source[len(_BASELINE_PREFIX):] if source.startswith(_BASELINE_PREFIX) else source
    name = name.rstrip("/")
    if name in registered_names():
        return ensure_packaged(name, arm=arm, unified_models=unified_models,
                               cache_dir=cache_dir)


    raise SeedSourceError(
        f"seed {source!r} is not a baseline this repo registers.\n"
        f"  registered baselines: {', '.join(registered_names())}\n"
        f"  A seed must be one of those. A directory that merely holds a "
        f"{ENTRY_FILE} is not a seed."
    )


def describe(source: str) -> str:
    """One line for a log: what kind of seed this is."""
    name = source[len(_BASELINE_PREFIX):] if source.startswith(_BASELINE_PREFIX) else source
    return f"registered baseline {name.rstrip('/')!r}"
