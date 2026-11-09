# Iteration target

Propose a new MemoClass harness, written to `{new_dir_rel}/`.

# Prior experience

You have read access to the entire per-run workspace (your cwd). Concretely:

  - `frontier.json`              population with scores + telemetry
<<TASK_FRONTIER_AXES>>
  - `harnesses/<hash>/`          every prior candidate: code, per-question
                                  traces, scores and token breakdown

You decide which priors to study and how deeply. The general spirit:

  - Learn from what works: study high-scoring harnesses to absorb
    successful patterns (storage choice, retrieval, dispatch, chunking).
  - Learn from what fails: skim low-scoring or crashed candidates to
    avoid repeating their pitfalls.
<<TASK_DIAGNOSE_BLOCK>>

<<CLOSE_THE_LOOP_BLOCK>>

# Target

Write to `{new_dir_rel}/`:

  1. memo.py               new MemoClass subclass (+ src/ if it needs one).
  2. meta.json             {{"parent_ids":     [...],
                             "description":    "...",
                             "design_choices": [{{"choice": "...",
                                                  "how":    "..."}}, ...]}}
                            parent_ids: ALL harness dirs you drew from,
                            BASE FIRST (position 0 = the harness this
                            modifies; everything else consulted after)
                            (full `<int>_<hash>` form, e.g.
                            `["a1b2c3d4e5f6", "5e6f78901234"]`). Empty list if
                            you started from scratch.
                            description: one short line on what's new + why.
                            design_choices: the decisions this harness
                            materially rests on (strict — concrete code,
                            not docstring claims). Each
                            entry's `how` covers BOTH mapping (code
                            mechanism) AND rationale (why this choice over
                            alternatives + what failure mode it targets).
                            See PROPOSER_SYSTEM "Harness contract" →
                            meta.json for the full schema + example.
  3. requirements.txt      OPTIONAL. Only if you need packages beyond the base image.
  4. <helper>.py           OPTIONAL helper modules.
  5. PROPOSAL_READY        Empty file — write LAST as the done-sentinel.

# What makes a good proposal

The BAR — what counts as an improvement on this run's axes — is stated
once, in PROPOSER_SYSTEM under `frontier.json`. Read it before you decide
what to build; everything below is about HOW to clear it, not what it is.

- BOTH stances are valid; pick whichever is most likely to clear the
  bar:
    • INCREMENTAL — find specific weakness in prior candidates
      (failure patterns in traces, weak QA category, brittle retrieval,
      missing dispatch, etc.) and target it with a focused change.
    • EXPLORATORY — try a fundamentally different design philosophy,
      possibly one inspired by a research line you investigated via
      WebSearch/WebFetch. The structure can share little with prior
      best candidates.
  **Bias toward EXPLORATORY when the prior frontier looks saturated** —
  and it saturates fast. The search loop is much more likely to break
  through via bolder structural rethinks than via cycle-after-cycle of
  small tweaks.
  `parent_ids = []` is fine (and often EXPECTED) when you genuinely
  depart from the prior line. The only requirement is that whatever
  you produce shows you LEARNED from prior history — whether by
  extending it or by deliberately departing from it.

- ALWAYS produce a candidate. "The frontier is already optimal", "no
  further improvement is available", "the remaining ideas are not worth
  trying" are not outputs — a saturated frontier is the reason to depart
  from it, not the reason to stop. The step costs the same whether you
  spend it on a bold design or on nothing, and a step spent on nothing
  leaves the next iteration exactly where this one started.

- Substantive over cosmetic. Whichever stance you take, the change
  should be real — not a prompt rewording, constant tweak, or rename.
  Small surface-level changes have too low signal-to-noise to drive
  search; the search loop wastes a cycle and you don't learn anything
  the next iteration can build on.

<<TASK_GENERALITY_BLOCK>>

- DO NOT hardcode shortcuts that game search-set scores.
  Concretely, AVOID:
    • dataset name strings keying special-case branches with no
      generalizable principle behind them
    • magic constants tuned to specific user_ids / sample_ids
    • hardcoded answer lookups (e.g. dict from query string → reference)
    • dia_id / session_id whitelists derived from search-set traces
    • string-matching that exploits search-set quirks (e.g. "if query
      contains 'birthday' then ...")
  After search ends, the top harnesses are re-evaluated on a HELD-OUT
  TEST SPLIT (different users, different samples). Anything that
  overfits to search-set details collapses there. A search-set 0.90
  that drops to 0.40 at test time is worse than a search-set 0.75 that
  holds. Generality is the goal; search-set rank is an instrument
  toward it, not the goal itself.
