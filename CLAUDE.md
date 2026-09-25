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
  criteria, relevant existing files to read, and "Follow AGENTS.md."
- Spawn: `tools/worker/spawn.sh <id> <slug> /tmp/prompt-<id>.md [model]`
  (creates worktree `../wt-i<id>-<slug>` on branch `i<id>-<slug>`, tmux window in session `workers`,
  log `/tmp/wt-i<id>-<slug>.log`). Then `chainlink issue comment <id> "worker: <branch>, <model>"`.
- Status: `tools/worker/status.sh` (a log ending in `__DONE__` is finished). Watch live: `tmux attach -t workers`.
- Run at most 2-3 workers at once, and only on issues that don't touch the same files.

## Review and merge
- `git -C ../wt-<name> log --oneline main..` and `git -C ../wt-<name> diff main...`. Run the tests yourself
  in the worktree. For `mac` issues: `rsync -a --delete --exclude .git ../wt-<name>/ macmini:~/wt/<name>/`
  then `ssh macmini 'cd ~/wt/<name>/swift/HandstandCore && swift test'` (or `xcodebuild` for the app).
- Summarize the diff and test results for the user. Merge only after the user approves:
  `git merge --no-ff <branch>`, `git push`, `git worktree remove ../wt-<name>`, `git branch -d <branch>`,
  then `chainlink issue close <id>`.
- If rejected: comment the reasons on the issue, remove the worktree, and respawn with a better prompt.

## Machines
- Linux (here): Python pipeline, workers, chainlink.
- Mac mini: `ssh macmini` (Xcode 27, Swift 6.4, repo at `~/GitRepo/handstands`). Commit identity on both:
  Roger <333816131+nanoccsafe@users.noreply.github.com>.

## Constraints
- The GitHub repo is public: never commit videos, data, keypoints, IPs or personal info.
