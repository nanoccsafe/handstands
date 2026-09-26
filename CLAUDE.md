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
The default flow is one issue at a time through the opencode-loop-plugin's single-issue Chainlink loop:
1. Make the issue description the complete spec (paths, interfaces to reuse, API, CLI, tests, acceptance):
   the plugin feeds the issue itself to its worker and reviewer, and they are small models.
2. `tools/worker/spawn.sh <id> <slug> --chainlink [model]` runs the plugin's deterministic
   `chainlink-loop --task <id> --no-close` (installed at `~/.config/opencode/plugins/opencode-loop-plugin/bin/`)
   in the issue's worktree (`CHAINLINK_DB` points at the main `.chainlink`). One-shot `opencode run` steps:
   build-agent worker, plan-agent reviewer that cannot edit, findings fed back to the worker session, until the
   reviewer approves; the issue stays open. Long batches: `LOOP_ARGS="--worker-timeout 10800"`. Don't comment on
   an issue right before spawning (the loop treats comments as earlier work and reviews first).
3. When the loop finishes, review it yourself (see below) against the spec. If it falls short, put the findings
   in `chainlink issue comment <id> ...` and rerun: `spawn.sh <id> <slug> --chainlink --rerun` (same worktree).
   The plugin is meant to work on its own: if the loop fails (no reviewer verdict, a timeout, the orchestrator
   doing the task itself), say so plainly, write the plugin problem up in `docs/prompts/plugin-*.md` for the user
   to route, and never patch over it silently.
4. When it is good, summarize for the user and ask to merge. Merge only with `merge.sh` after approval.

Fallback when the plugin is unavailable: `spawn.sh <id> <slug> <prompt-file> [model]` (one-shot worker; keep the
prompt in `docs/prompts/`).

- Models: default `opencode/space-bunny-free`; alternatives `opencode/mimo-v2.6-flash-free`,
  `opencode-go/mimo-v2.6-flash`. Switch model on a rerun if a model keeps failing.
- Every worker: worktree `../wt-i<id>-<slug>`, branch `i<id>-<slug>`, tmux window in session `workers`, state and
  log in `/tmp/handstand-workers/<name>.{env,log}`. Runs use `opencode run --standalone`, so a worker has a private
  server that dies with its window; never drop `--standalone` (the shared service keeps runs going after the
  client is gone). Spawn refuses if a window, worktree, branch or process of that name is left over.
- Status: `tools/worker/status.sh` shows RUNNING / DONE / STOPPED / ORPHANED, commits, uncommitted files and
  processes still inside the worktree. ORPHANED means it ended without a marker: run `stop.sh` on it.
  Watch live: `tmux attach -t workers`.
- Stop: `tools/worker/stop.sh <name>` closes the window, TERMs then KILLs the process tree and anything whose
  cwd is in the worktree, and verifies nothing is left. `--clean` also deletes the worktree, branch, state
  and the worker's OpenCode sessions (only when its work is not wanted). `stop.sh --all [--clean]` for all.
- After a crash/reboot: `/tmp` is wiped (worker state, logs, tmux). Check `git worktree list`, worker commits,
  and processes whose cwd is in a worktree. The shared OpenCode service can resume an interrupted worker session
  on its own when any `opencode` starts; delete such orphan sessions (`opencode session delete <id>`) before
  restarting the loop with `--rerun`, or two workers will edit the same worktree.
- Shutdown: before ending a lead session, or when the user says stop, run `tools/worker/stop.sh --all` and
  confirm `status.sh` shows no RUNNING workers and `procs_in_worktree=0`.
- One issue at a time (the user's choice).

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
