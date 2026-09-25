# Handstand form analysis: lead instructions

You are the lead. Plan, delegate implementation to OpenCode workers, review, and keep chainlink up to date.
Workers follow `AGENTS.md`. The plan lives in chainlink (`.chainlink/`, local only, not in git).

## Picking work
- `chainlink issue ready` is the queue. The blocking links make it sequential: work only on ready issues.
- Owner labels: `worker` = delegate to OpenCode, `human` = the user does it (ask them), `lead` = you do it
  (decisions, docs, reviews), `mac` = needs the Mac mini to build or verify.
- Before delegating, run `chainlink issue show <id>`. If the description lacks concrete paths, interfaces or
  acceptance criteria, write them into the issue (`chainlink issue update <id> -d ...`) first. Workers are
  small models: they need an exact spec, not a goal.

## Workers
- Models: default `opencode/space-bunny-free`; alternative `opencode/mimo-v2.6-flash-free` or
  `opencode-go/mimo-v2.6-flash`. Use a different model on retry if a worker fails.
- Prompt file: write to `/tmp/prompt-<id>.md`, containing the issue title and description, acceptance
  criteria, relevant existing files to read, and "Follow AGENTS.md." Copy it to `docs/prompts/` so prompts
  survive reboots and can be reused on retry.
- Spawn: `tools/worker/spawn.sh <id> <slug> /tmp/prompt-<id>.md [model]`
  (worktree `../wt-i<id>-<slug>`, branch `i<id>-<slug>`, tmux window in session `workers`, state and log in
  `/tmp/handstand-workers/<name>.{env,log}`). It runs `opencode run --standalone`, so each worker has a
  private server that dies with its window; never drop `--standalone` (the shared service keeps runs going
  after the client is gone). It refuses to start if any window, worktree, branch or process of that name is
  left over. Then `chainlink issue comment <id> "worker: <branch>, <model>"`.
- Status: `tools/worker/status.sh` shows RUNNING / DONE / STOPPED / ORPHANED, commits, uncommitted files and
  processes still inside the worktree. ORPHANED means it ended without a marker: run `stop.sh` on it.
  Watch live: `tmux attach -t workers`.
- Stop: `tools/worker/stop.sh <name>` closes the window, TERMs then KILLs the process tree and anything whose
  cwd is in the worktree, and verifies nothing is left. `--clean` also deletes the worktree, branch, state
  and the worker's OpenCode sessions (only when its work is not wanted). `stop.sh --all [--clean]` for all.
- Shutdown: before ending a lead session, or when the user says stop, run `tools/worker/stop.sh --all` and
  confirm `status.sh` shows no RUNNING workers and `procs_in_worktree=0`.
- Run at most 2-3 workers at once, and only on issues that don't touch the same files.

## Review and merge
- `git -C ../wt-<name> log --oneline main..` and `git -C ../wt-<name> diff main...`. Run the tests yourself
  in the worktree. For `mac` issues: `rsync -a --delete --exclude .git ../wt-<name>/ macmini:~/wt/<name>/`
  then `ssh macmini 'cd ~/wt/<name>/swift/HandstandCore && swift test'` (or `xcodebuild` for the app).
- Summarize the diff and test results for the user. Merge only after the user approves, and only with
  `tools/worker/merge.sh <name>`: it commits chainlink's CHANGELOG edits, merges, runs the tests (undoing
  the merge if they fail), pushes, and only then runs `stop.sh --clean` and closes the issue. Never chain
  merge and cleanup commands by hand: a failed merge followed by `--clean` deletes unmerged work.
- If rejected: comment the reasons on the issue, remove the worktree, and respawn with a better prompt.

## Machines
- Linux (here): Python pipeline, workers, chainlink.
- Mac mini: `ssh macmini` (Xcode 27, Swift 6.4, repo at `~/GitRepo/handstands`). Commit identity on both:
  Roger <333816131+nanoccsafe@users.noreply.github.com>.

## Constraints
- The GitHub repo is public: never commit videos, data, keypoints, IPs or personal info.
