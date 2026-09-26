# Prompt: opencode-loop-plugin — behaviour after a machine crash (2026-09-26)

Setup: `chainlink-loop --task 69 --no-close --worker-timeout 10800` from `tools/worker/spawn.sh`, worker step running
a long background batch. The machine crashed and rebooted at 15:40.

What happened:
1. `/tmp` was wiped, including the plugin working copy `/tmp/opencode/opencode-loop-plugin` (uncommitted work; the
   installed copy in `~/.config/opencode/plugins/opencode-loop-plugin`, synced 2026-09-26 12:58, survived). Keep the
   working copy outside `/tmp` and commit/push it.
2. The `chainlink-loop` process died with the machine, but its worker session `Chainlink 69 worker`
   (ses_f20edfe6…) was **resumed by the shared service**: at 15:45 the user started an unrelated interactive
   `opencode` in another project, which launched `opencode serve --service`, and one minute later that service
   continued the worker session (it restarted the batch and started polling). The step had been launched with
   `opencode run --standalone`, so it was never meant to run in the shared service. Nothing would ever have
   reviewed its result, and a restarted loop would have edited the same worktree concurrently.

Needed:
- Sessions created by `chainlink-loop` must not be resumable by another server: mark them (metadata/title) and make
  the plugin refuse to continue them in any process other than their owning `chainlink-loop` (owner pid gone ->
  mark the workflow `interrupted`, do not resume the session).
- `chainlink-loop` on start should detect an interrupted workflow for the same task/worktree and report it (and the
  last worker session id) instead of silently starting over; `--review-first auto` should then review what exists.
- Document the recovery procedure after a crash.
