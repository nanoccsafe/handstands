# Worker prompt: #99 split mixed-shape holds into per-shape segments

You are a worker on the handstand form-analysis repo. Your worktree is
`/mnt/sharedOs/wt-i99-segments` (branch `i99-segments`); work ONLY inside it.

1. Read `AGENTS.md` and follow it.
2. The spec is chainlink issue #99:
   `CHAINLINK_DB=/mnt/sharedOs/handstand-workspace/.chainlink chainlink issue show 99`.
   Implement exactly that spec (phase 1 only: the module, tests, CLI, docs). Read
   `pipeline/handstand/features.py` (`hold_rows`, `_spread`, `_rounded_scalar`,
   `ClipFeatures`, the parquet loader), `pipeline/handstand/hold_shapes.py`
   (`predict_shape`, `UNMEASURED`, `load_hold_shapes`) and `pipeline/handstand/view.py`
   (CLI, report-path and atomic-write conventions) before writing code, and reuse them.
3. Tests are synthetic only (tmp_path). Never read, copy or commit anything from the
   main repo's `data/` or `videos/`. No clip ids, IPs or personal info in committed files.
4. Do not change existing behaviour or outputs: no edits to hold ids, hold_summary.csv,
   `predict_shape`, features.py logic.
5. Before finishing, run in `pipeline/`: `uv run pytest -q`, `uv run ruff check .`,
   `uv run ruff format --check .` — all must pass.
6. Commit on the branch with a message `Split holds into per-shape segments (#99)` ending with
   the lines
   `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Do not push, merge, close the
   issue or comment on it.
7. Report back: files changed, test counts, anything in the spec you could not follow and why.
