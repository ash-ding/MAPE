"""Query a search run's history: lineage, ranking, one harness, a diff, a bill.

A run's history is complete on disk but spread across three files that have to
be joined by hand — `history.json` has the chronology, `frontier.json` has the
scores, and each `harnesses/<id>/meta.json` has that harness's parents. Doing
that join by reading files costs the proposer tokens and invites mistakes:
harness directories are 12-hex content hashes, which carry no order and look
alike. A short CLI beats walking a tree, and querying one is closer to what
coding agents are trained on.

    python /app/search_history.py tree            # who came from whom
    python /app/search_history.py pareto          # the trade-offs worth keeping
    python /app/search_history.py best            # record holder per axis
    python /app/search_history.py top -k 10       # ranked by an objective
    python /app/search_history.py show <id>       # one harness, in full
    python /app/search_history.py diff <a> <b>    # what changed between two
    python /app/search_history.py cost            # what the SEARCH itself cost

Runs against the current workspace by default (the proposer's cwd is the run's
workspace); `--workspace PATH` picks another. Read-only: it never writes.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[Any]:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class Run:
    """One run's workspace, joined once so every command can just read it."""

    def __init__(self, workspace: Path):
        self.root = workspace
        if not (self.root / "frontier.json").exists() and (self.root / "harnesses").exists() is False:
            raise SystemExit(
                f"{self.root} does not look like a run workspace "
                f"(no frontier.json, no harnesses/). Pass --workspace.")
        frontier = _read_json(self.root / "frontier.json") or {}
        history = _read_json(self.root / "history.json") or {}

        #: id -> {objectives, parent_ids, created_at, ...}
        self.entries: Dict[str, Dict[str, Any]] = {
            e["id"]: e for e in frontier.get("entries", []) if e.get("id")
        }
        self.steps: List[Dict[str, Any]] = history.get("steps", [])
        self.run_id: str = history.get("run_id") or self.root.name

        #: id -> step number it first appeared in, and its sanity verdict.
        self.step_of: Dict[str, int] = {}
        self.sanity: Dict[str, str] = {}
        for step in self.steps:
            for h in step.get("harnesses", []):
                hid = h.get("id")
                if hid and hid not in self.step_of:
                    self.step_of[hid] = step.get("step", -1)
                    self.sanity[hid] = h.get("sanity", "")
                # history carries parents too; frontier is authoritative when
                # both exist, but a harness that never reached the frontier
                # only appears here.
                if hid and hid not in self.entries:
                    # `metrics` carries whatever the run was searching on,
                    # already named the way the frontier names it. Runs
                    # written before that carried `scores`, keyed by dataset.
                    objectives = dict(h.get("metrics") or {})
                    if not objectives:
                        objectives = {f"accuracy_{ds}": v
                                      for ds, v in (h.get("scores") or {}).items()}
                    self.entries[hid] = {
                        "id": hid, "objectives": objectives,
                        "parent_ids": h.get("parent_ids") or [], "created_at": h.get("at", ""),
                        "_off_frontier": True,
                    }

        #: id -> [child ids], derived from parent_ids.
        self.children: Dict[str, List[str]] = {hid: [] for hid in self.entries}
        for hid, e in self.entries.items():
            for parent in e.get("parent_ids") or []:
                self.children.setdefault(parent, []).append(hid)

    # -- helpers -----------------------------------------------------------

    def dir_of(self, hid: str) -> Path:
        return self.root / "harnesses" / hid

    def resolve(self, prefix: str) -> str:
        """Accept a unique prefix, the way git does with commits."""
        if prefix in self.entries:
            return prefix
        hits = [h for h in self.entries if h.startswith(prefix)]
        if not hits:
            raise SystemExit(f"no harness matches {prefix!r} in {self.run_id}")
        if len(hits) > 1:
            raise SystemExit(f"{prefix!r} is ambiguous: {', '.join(sorted(hits))}")
        return hits[0]

    def objectives(self, hid: str) -> Dict[str, Any]:
        return (self.entries.get(hid) or {}).get("objectives") or {}

    def optimised_axes(self) -> List[str]:
        """The axes a search actually optimises: accuracy up, cost down.

        Everything else in `objectives` is there to read those two correctly
        (`score_max_<ds>` gives the scale), so a Pareto front over all of
        them would be nonsense.
        """
        axes = {k for e in self.entries.values() for k in (e.get("objectives") or {})
                if k.startswith(("accuracy_", "cost_usd_per_1k_queries_"))}
        return sorted(axes)

    def produced_nothing(self, hid: str) -> bool:
        """True when no user of this harness yielded a usable result.

        A run that crashed costs nothing, and "nothing" is the best possible
        score on a cost axis — so a failed harness lands on the Pareto front
        and holds the cheapest record while having answered no question at
        all. Measured on a real zep failure: accuracy 0.0, cost 0, and
        `cost_n_users: 0` with `cost_users_excluded: 1`.

        The discriminator is the denominator, not the cost: a memoryless
        baseline legitimately costs ~0 per query while still answering, and
        it reports `cost_n_users` > 0.
        """
        d = self.dir_of(hid)
        if not d.exists():
            return False
        ran = [m for m in ((_read_json(ds / "metrics.json") or {})
                           for ds in d.iterdir() if ds.is_dir()) if m]
        return bool(ran) and all(int(m.get("cost_n_users", 0) or 0) == 0 for m in ran)

    def accuracy_axes(self) -> List[str]:
        axes = {k for e in self.entries.values()
                for k in (e.get("objectives") or {}) if k.startswith("accuracy_")}
        return sorted(axes)

    def label(self, hid: str) -> str:
        """One line describing a harness: where it came from, what it scored."""
        meta = _read_json(self.dir_of(hid) / "meta.json") or {}
        desc = meta.get("description") or ""
        scores = " ".join(f"{k.removeprefix('accuracy_')}={v:.3g}"
                          for k, v in sorted(self.objectives(hid).items())
                          if k.startswith("accuracy_"))
        # Cost axes carry the dataset in the name and are only recorded when
        # the run asked for `efficiency`: dollars per 1K queries.
        costs = " ".join(
            f"{k.removeprefix('cost_usd_per_1k_queries_')}:${v:,.2f}/1k"
            for k, v in sorted(self.objectives(hid).items())
            if k.startswith("cost_usd_per_1k_queries_"))
        bits = [b for b in (scores, costs) if b]
        step = self.step_of.get(hid)
        if step is not None:
            bits.append(f"step {step}")
        if self.sanity.get(hid) not in ("", "passed", None):
            bits.append(f"sanity={self.sanity[hid]}")
        tail = "  ".join(bits)
        return f"{hid}  {tail}" + (f"   {desc[:60]}" if desc else "")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_tree(run: Run, args) -> None:
    """The lineage, as a tree. Roots are seeds (no parents)."""
    roots = sorted(hid for hid, e in run.entries.items() if not (e.get("parent_ids") or []))
    if not roots:
        print("(no harnesses yet)")
        return
    seen: set = set()

    def walk(hid: str, prefix: str, last: bool, top: bool) -> None:
        connector = "" if top else ("└── " if last else "├── ")
        if hid in seen:                       # a harness with two parents
            print(f"{prefix}{connector}{hid}  (shown above)")
            return
        seen.add(hid)
        print(f"{prefix}{connector}{run.label(hid)}")
        kids = sorted(run.children.get(hid, []))
        extend = "" if top else ("    " if last else "│   ")
        for i, kid in enumerate(kids):
            walk(kid, prefix + extend, i == len(kids) - 1, top=False)

    for root in roots:
        walk(root, "", last=True, top=True)
    off = [h for h, e in run.entries.items() if e.get("_off_frontier")]
    if off:
        print(f"\n({len(off)} harness(es) shown from history.json only — "
              f"never reached the frontier)")


def cmd_top(run: Run, args) -> None:
    """Rank by one objective. Accuracy sorts high-first, cost low-first."""
    axis = args.by or (run.accuracy_axes() or ["accuracy"])[0]
    lower_is_better = not axis.startswith(("accuracy", "score"))
    rows = [(hid, o[axis]) for hid, o in
            ((h, run.objectives(h)) for h in run.entries)
            if axis in o and not run.produced_nothing(hid)]
    if not rows:
        axes = sorted({k for h in run.entries for k in run.objectives(h)})
        raise SystemExit(f"no harness has objective {axis!r}. Available: {', '.join(axes)}")
    rows.sort(key=lambda r: r[1], reverse=not lower_is_better)
    print(f"ranked by {axis} ({'lower' if lower_is_better else 'higher'} first)")
    for rank, (hid, value) in enumerate(rows[:args.k], 1):
        print(f"  {rank:2d}. {run.label(hid)}")


def cmd_show(run: Run, args) -> None:
    """One harness in full: objectives, lineage, files."""
    hid = run.resolve(args.harness)
    entry = run.entries[hid]
    meta = _read_json(run.dir_of(hid) / "meta.json") or {}
    print(f"harness {hid}")
    if meta.get("description"):
        print(f"  {meta['description']}")
    print(f"  step       {run.step_of.get(hid, '?')}"
          f"   sanity {run.sanity.get(hid) or '?'}"
          f"   created {entry.get('created_at', '?')}")
    parents = entry.get("parent_ids") or []
    print(f"  parents    {', '.join(parents) if parents else '(seed)'}")
    kids = sorted(run.children.get(hid, []))
    print(f"  children   {', '.join(kids) if kids else '(none yet)'}")
    print("  objectives")
    for k, v in sorted(entry.get("objectives", {}).items()):
        print(f"    {k:34s} {v:,.4g}" if isinstance(v, (int, float)) else f"    {k:34s} {v}")
    code = sorted(p for p in run.dir_of(hid).rglob("*.py")
                  if "__pycache__" not in p.parts)
    print(f"  code       {len(code)} .py file(s), "
          f"{sum(p.stat().st_size for p in code):,} bytes")
    for p in code[:8]:
        print(f"    {p.relative_to(run.dir_of(hid))}")
    if len(code) > 8:
        print(f"    ... and {len(code) - 8} more")
    datasets = sorted(d.name for d in run.dir_of(hid).iterdir()
                      if d.is_dir() and (d / "metrics.json").exists()) \
        if run.dir_of(hid).exists() else []
    for ds in datasets:
        m = _read_json(run.dir_of(hid) / ds / "metrics.json") or {}
        # Token totals and per-user spread are deliberately NOT frontier axes
        # (see evolution/orchestrator.py::_build_objectives); they are read here,
        # where they actually live, so `show` stays the one place that has
        # everything about a harness.
        print(f"  {ds}")
        print(f"    tokens     {m.get('tokens', 0):,} "
              f"(build {m.get('tokens_build', 0):,}, "
              f"answer {m.get('tokens_answer', 0):,}, "
              f"judge {m.get('tokens_judge', 0):,})")
        print(f"    llm_calls  {m.get('llm_calls', 0):,}")
        spread = m.get("per_user_stddev")
        if spread is not None:
            print(f"    per-user spread  {spread:.4g}")
        local = m.get("local_models") or []
        if local:
            print("    local models  " + ", ".join(
                f"{d.get('role')}@{d.get('device')}" for d in local))


def cmd_diff(run: Run, args) -> None:
    """What changed between two harnesses, file by file."""
    a, b = run.resolve(args.a), run.resolve(args.b)
    da, db = run.dir_of(a), run.dir_of(b)

    def code_files(d: Path) -> Dict[str, Path]:
        return {str(p.relative_to(d)): p for p in sorted(d.rglob("*"))
                if p.is_file() and p.suffix == ".py" and "__pycache__" not in p.parts}

    fa, fb = code_files(da), code_files(db)
    names = sorted(set(fa) | set(fb))
    changed = 0
    for name in names:
        left = fa[name].read_text(encoding="utf-8", errors="replace").splitlines(True) \
            if name in fa else []
        right = fb[name].read_text(encoding="utf-8", errors="replace").splitlines(True) \
            if name in fb else []
        if left == right:
            continue
        changed += 1
        sys.stdout.writelines(difflib.unified_diff(
            left, right, fromfile=f"{a}/{name}", tofile=f"{b}/{name}", n=args.context))
    if not changed:
        print(f"{a} and {b} have identical code "
              f"(they would share a content hash — check you passed two different ids)")

    print(f"\nobjectives  {a} → {b}")
    keys = sorted(set(run.objectives(a)) | set(run.objectives(b)))
    for k in keys:
        va, vb = run.objectives(a).get(k), run.objectives(b).get(k)
        mark = "" if va == vb else "  <-- changed"
        print(f"  {k:34s} {va!s:>12} → {vb!s:>12}{mark}")


def _better(axis: str, x: float, y: float) -> bool:
    """Is x better than y on this axis? Cost is the one that sorts downward."""
    return x < y if axis.startswith("cost_") else x > y


def cmd_best(run: Run, args) -> None:
    """The record holder on each optimised axis — one line per axis."""
    axes = run.optimised_axes()
    if not axes:
        print("nothing evaluated yet")
        return
    for axis in axes:
        rows = [(hid, o[axis]) for hid, o in
                ((h, run.objectives(h)) for h in run.entries)
                if axis in o and not run.produced_nothing(hid)]
        if not rows:
            print(f"{axis}  (no harness produced a result)")
            continue
        hid, value = rows[0]
        for cand, v in rows[1:]:
            if _better(axis, v, value):
                hid, value = cand, v
        direction = "lowest" if axis.startswith("cost_") else "highest"
        print(f"{axis}  ({direction})")
        print(f"    {value:,.6g}   {run.label(hid)}")


def cmd_pareto(run: Run, args) -> None:
    """The non-dominated set: every trade-off worth building on.

    NOT the same as `best`. A harness that scores 0.78 at a tenth of the cost
    holds no record and is still on the front — and it is exactly the kind of
    prior a search should notice.
    """
    axes = run.optimised_axes()
    if len(axes) < 2:
        print(f"only one optimised axis ({', '.join(axes) or 'none'}) — "
              f"`top` already ranks it; a front needs a trade-off")
        return
    scored = [h for h in run.entries
              if all(a in run.objectives(h) for a in axes)
              and not run.produced_nothing(h)]
    skipped = [h for h in run.entries
               if all(a in run.objectives(h) for a in axes) and run.produced_nothing(h)]
    front = []
    for hid in scored:
        o = run.objectives(hid)
        dominated_by = None
        for other in scored:
            if other == hid:
                continue
            oo = run.objectives(other)
            if (all(not _better(a, o[a], oo[a]) for a in axes)
                    and any(_better(a, oo[a], o[a]) for a in axes)):
                dominated_by = other
                break
        if dominated_by is None:
            front.append(hid)
        elif args.verbose:
            print(f"  (dominated: {hid} by {dominated_by})")
    print(f"pareto front over {', '.join(axes)} — {len(front)} of {len(scored)}")
    key = axes[0]
    for hid in sorted(front, key=lambda h: run.objectives(h)[key],
                      reverse=not key.startswith("cost_")):
        print(f"    {run.label(hid)}")
    if not front:
        print("    (nothing scored on every axis yet)")
    if skipped:
        print(f"\n  excluded, answered nothing: {', '.join(sorted(skipped))}\n"
              f"  (a crashed run costs nothing, which is the best possible cost — "
              f"it is not a trade-off)")


def cmd_cost(run: Run, args) -> None:
    """What the SEARCH cost — every evaluation this run paid for.

    Not the same as a harness's `cost_tokens_per_query`, which is what that
    memory system costs to USE. This is the bill for finding it.
    """
    totals = {"tokens": 0, "tokens_build": 0, "tokens_answer": 0, "tokens_judge": 0,
              "llm_calls": 0}
    seconds = 0.0
    evaluated = 0
    for hid in sorted(run.entries):
        d = run.dir_of(hid)
        if not d.exists():
            continue
        for ds_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            m = _read_json(ds_dir / "metrics.json")
            if not m:
                continue
            evaluated += 1
            for k in totals:
                totals[k] += int(m.get(k, 0) or 0)
            seconds += sum(float(v) for v in (m.get("phase_seconds") or {}).values())
    if not evaluated:
        print("nothing evaluated yet")
        return
    print(f"run {run.run_id}: {evaluated} evaluation(s) across "
          f"{len(run.entries)} harness(es)")
    for k, v in totals.items():
        print(f"  {k:16s} {v:>14,}")
    print(f"  {'wall seconds':16s} {seconds:>14,.0f}")
    print("\n(tokens here include the judge and the answering agent — this is the "
          "cost of SEARCHING, not the cost of using any one memory system)")


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", type=Path, default=Path("."),
                   help="run workspace (default: cwd, which is the proposer's)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tree", help="lineage: who came from whom").set_defaults(fn=cmd_tree)

    top = sub.add_parser("top", help="rank by an objective")
    top.add_argument("-k", type=int, default=10)
    top.add_argument("--by", help="objective name (default: the first accuracy axis)")
    top.set_defaults(fn=cmd_top)

    show = sub.add_parser("show", help="one harness in full")
    show.add_argument("harness", help="id or unique prefix")
    show.set_defaults(fn=cmd_show)

    diff = sub.add_parser("diff", help="code + objectives between two harnesses")
    diff.add_argument("a")
    diff.add_argument("b")
    diff.add_argument("--context", type=int, default=3)
    diff.set_defaults(fn=cmd_diff)

    sub.add_parser("best", help="record holder per optimised axis").set_defaults(fn=cmd_best)

    par = sub.add_parser("pareto", help="the non-dominated set (trade-offs worth keeping)")
    par.add_argument("-v", "--verbose", action="store_true",
                     help="also name who dominated each excluded harness")
    par.set_defaults(fn=cmd_pareto)

    sub.add_parser("cost", help="what this search has cost so far").set_defaults(fn=cmd_cost)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.fn(Run(args.workspace.resolve()), args)


if __name__ == "__main__":
    main()
