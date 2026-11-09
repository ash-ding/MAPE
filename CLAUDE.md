# Repository maintenance

- Keep framework code in `evolution/`, shared execution in `core/`, and the released final program in `programs/mape_a/`.
- Assemble experiments with `scripts/prepare_experiments.py`; do not commit generated `workspace/` copies, results, data, credentials or model caches.
- `evolution/prompts/parts/` is the single prompt set; keep `fragments.py` consistent with the Markdown parts. `docs/reproduction.md` records where it differs from the prompts of the paper's runs.
- `programs/mape_a/` and `experiments/` are pinned by `experiments/provenance.json`; update the hashes there when you change either.
- Do not edit vendored baseline `src/` implementations to adapt interfaces or track costs. Use the surrounding adapter.
- Claude Code and claude-* models authenticate with `ANTHROPIC_API_KEY` only.
- After import/path changes, run `uv run pytest -q` with benchmark data downloaded.
