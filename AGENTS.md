# Worker rules

You are a worker implementing ONE chainlink issue in your own git worktree and branch. The task
prompt names the issue and its acceptance criteria. Stay inside that scope.

## Paths
- Code: this worktree (you are already in it). Do not touch other worktrees or the main checkout's code.
- Raw videos (read-only): `/mnt/sharedOs/handstand-workspace/videos/`. Never copy, modify or commit them.
- Generated data: `/mnt/sharedOs/handstand-workspace/data/` (shared, git-ignored). Read the path from the
  `HANDSTAND_DATA` env var with that directory as the default. Never commit data files.
- Never `cd` into directories outside your worktree (data, videos, the main checkout): pass absolute paths to
  commands instead (`ls /mnt/sharedOs/handstand-workspace/data/...`, `ffprobe <abs path>`) or use Python with
  `handstand.paths`. A `cd` outside the worktree triggers a permission request nobody can answer and blocks you.
- Do not read or modify `.chainlink/` and do not run `chainlink`; the lead tracks issues.
- `CLAUDE.md` is the lead's instructions, not yours: do not read or follow it.

## Layout
- `pipeline/` Python package `handstand` (uv, src at `pipeline/handstand/`, tests in `pipeline/tests/`)
- `tools/` scripts, `docs/` notes, `swift/` HandstandCore package, `ios/` app

## Python
- `cd pipeline && uv sync`, `uv run pytest`, `uv run ruff check`. Add dependencies with `uv add`, never pip.
- Type hints on public functions, small pure functions, no notebooks.
- Every task adds or updates tests for its logic. Tests must not depend on the real videos;
  use tiny synthetic inputs (e.g. numpy arrays, a 10-frame generated video).

## Swift
- You cannot build Swift here (Linux). Write the code and tests; the lead builds and tests on the Mac.

## Finishing
1. Run tests and lint; fix failures.
2. `git add` only the files you created or changed, then commit on your branch with a clear message.
   Use the repo's configured git identity: never pass `-c user.name`/`user.email` or `--author`.
3. End with a short report: what you did, files changed, how to run it, anything unfinished or uncertain.
Do not merge, push, or switch branches. Never use `git stash` (the stash is shared by every worktree).

You run non-interactively: nobody can answer questions. If you are blocked or unsure, stop and explain it
in your final report instead of asking.
