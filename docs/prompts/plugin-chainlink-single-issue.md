# Prompt: opencode-loop-plugin — Chainlink single-issue mode and worktree-safe workflow

Working copy: `/tmp/opencode/opencode-loop-plugin` (uncommitted Chainlink work on top of `a7c3bed`).
Installed copy: `~/.config/opencode/plugins/opencode-loop-plugin`.

## Context
A project (handstand, `/mnt/sharedOs/handstand-workspace`) wants to drive its Chainlink backlog with this plugin in
two modes:
1. **Queue mode** (exists): `/chainlink` loops over `chainlink issue next` until nothing is ready.
2. **Single-issue mode** (needed): `/chainlink #24` runs the worker/reviewer loop for exactly that issue, then stops.
The project works one issue at a time, and the user approves every merge.

The in-progress `taskIds` change (`fetchTaskById`, `requestedTaskIDs`) already covers most of mode 2. Please finish
it with the points below.

## Required changes

1. **Explicit task selection: validate before running.**
   `fetchTaskById` must refuse, with a clear error and without creating a workflow, if the issue is closed, is an
   epic/parent (`is_epic` or `subissue_count > 0`), or has open blockers (read them from `chainlink issue show --json`).
   Check that the id exists (a non-zero exit or empty show output is an error, not a fallback title).

2. **Deterministic argument parsing for the command.**
   Parse `/chainlink` arguments in code, not by asking the model to interpret them:
   - no args: queue mode
   - `#24`, `task 24`, `24` (a bare integer), or a list like `#24 #25`: single/explicit mode with those ids
   - `--attempts N`: max attempts for this run (the README says `/chainlink 20` sets attempts, but the tool input
     is empty, so that number is currently ignored. Either wire it up or remove it from the README; a bare
     integer should mean a task id)
   - `--no-close`: see 3
   Pass the parsed values to `run_chainlink_outer` as tool input (`task_ids`, `max_attempts`, `close_on_approval`),
   and make the command template tell the model to call the tool with exactly those values.

3. **Do not close on approval when asked.**
   Add a per-run `close_on_approval` (tool input, defaulting to the `chainlink_close_completed_tasks` option).
   With `false`, reviewer approval finishes the workflow as `completed` (or a new status such as `approved`) and
   leaves the issue open, because the project merges the branch after human approval and closes the issue then.
   Report the final state clearly so the caller knows the issue is approved but still open.

4. **Run in a git worktree without losing the Chainlink database.**
   The project runs each worker in its own worktree (`../wt-i<id>-<slug>`). `.chainlink/` is untracked, so it is
   absent from the worktree. Make the Chainlink runner honour a configurable database path: pass
   `CHAINLINK_DB` (or `--db`) from a new option `chainlink_db_path` and/or inherit `CHAINLINK_DB` from the
   environment, and document it. The worker and reviewer child sessions must use the same directory as the parent
   session (the worktree), which `childLocation` already does; add a test that confirms it.

5. **Model per role.**
   Allow choosing the model for the worker and the reviewer (options `chainlink_worker_model`,
   `chainlink_reviewer_model`, plus per-run tool input). The project wants small models for workers
   (`opencode/space-bunny-free`, `opencode-go/mimo-v2.6-flash`), and possibly a different one for review.
   If the V2 session API cannot set a model on a child session, say so in the README instead of faking it.

6. **Clean cancellation.**
   When the run is aborted (the parent `opencode run --standalone` process is killed, or the session is
   interrupted), abort the in-flight worker/reviewer child sessions too and mark the workflow `cancelled`.
   The project found that a plain `opencode run` (without `--standalone`) keeps executing in the shared
   background service after its client is killed. Document that `--standalone` is required for a kill to stop
   the run, and verify that the plugin does not leave child sessions running after cancellation.

7. **Non-interactive safety.**
   In `opencode run`, the question tool fails ("The user dismissed this question") and ends the session.
   Tell the worker in its prompt that it runs unattended: it must not ask questions, and should end with a report
   of any blocker instead. Also tell the worker not to use `git stash` (the stash is shared by all worktrees) and
   not to override the git identity (`-c user.name`, `--author`).

8. **Loading.**
   OpenCode currently loads the npm package `@prevalentware/opencode-loop-plugin@latest` (0.1.8), which has **no**
   Chainlink support, so `/chainlink` is unavailable right now. After building, describe the exact config change
   that makes OpenCode load the local build (for example a `file://` entry in `~/.config/opencode/opencode.json`
   `plugins`), and note that it applies to every project.

## Tests
Extend `test/chainlink.test.ts` with a fake runner covering: single id, multiple ids in order, a closed issue,
a blocked issue, an epic, an unknown id, `close_on_approval=false` leaving the issue open, queue mode unchanged,
argument parsing for every form in 2, and cancellation marking the workflow `cancelled`.
`bun test`, `bun run typecheck` and `bun run lint` must pass, then `bun run build`.

## How the project will call it
From the issue's worktree:
`CHAINLINK_DB=/mnt/sharedOs/handstand-workspace/.chainlink opencode run --standalone --auto -m <model> "/chainlink #24 --no-close"`
It expects one workflow, a worker that commits on the worktree's branch, a reviewer verdict, the issue left open,
and a final report that includes the workflow status.
