# Prompt: opencode-loop-plugin — issues from the #67 run (2026-09-25, after the round-2 fixes)

Run: `CHAINLINK_DB=… opencode run --standalone --auto -m opencode/space-bunny-free "/chainlink #67 --no-close"`.
The round-2 fixes are in the loaded bundle; the stall detector fired correctly. Timeline from the orchestrator
session `i67-multipose` and the workflow records:

| time | what happened |
|---|---|
| 16:45 | orchestrator calls `run_chainlink_outer {"task_ids":["#67"]}` → `failed: invalid Chainlink task id "#67"` |
| 16:45–16:58 | orchestrator calls again with `"67"`; worker session …ZEJBu does the whole task (122 tool calls, commits `806b42f`, idle outcome `succeeded`) |
| 16:46:42 | that workflow `chainlink_c3aok` is marked `interrupted` / "OpenCode restarted before the Chainlink workflow finished" while its worker is still running |
| 16:58 | tool returns `failed: Chainlink workflow "chainlink_c3aok" is interrupted`; no reviewer ever ran on the finished work |
| 16:58–17:09 | orchestrator calls the tool a 3rd time; new worker …Hmjs stalls after a quick `ls` → `failed: stalled for 300s` |
| 17:09 | orchestrator ignores `orchestrator_instruction: "Do not continue… end the turn"`, says "Let me retry with a longer stall budget", calls the tool a 4th time; worker …kFV stalls again right after an `ls` at 17:10 |

## Problems

1. **The command adapter lets the model re-type the tool input.** The template says to pass the exact
   deterministic JSON, but the model passed `"#67"`. Either have the command adapter call the tool itself
   (no model in between), or make the tool accept `#67` and normalise ids.

2. **Live workflows are still marked interrupted.** `chainlink_c3aok` was interrupted 36 s after it started,
   while its worker ran for 13 more minutes and finished. The `ownerPid` check did not protect it. Find what
   called `interruptActiveChainlinkWorkflows` (or another path setting `interrupted`) at 16:46:42 and log the
   caller pid/reason. Also: when the worker finished successfully, the loop should continue to review instead
   of failing on the stale `interrupted` flag.

3. **The orchestrator retries after failure.** It called the tool 4 times and changed parameters on its own.
   Enforce "call exactly once" in code: after the first `run_chainlink_outer` call in a command turn, further
   calls from the same session return an error without starting a workflow.

4. **Each retry starts a fresh worker and discards finished work from the reviewer's view.** The work from
   attempt 2 (commit `806b42f`) is in the worktree, but no reviewer saw it. When a task already has commits on
   the branch, start with a reviewer pass instead of a new worker.

5. **Root cause of every "stall" (the most important item): unanswered permission requests in child sessions.**
   All three stalls (#24 at 11:58, #67 at 16:59 and 17:04) were the same kind of tool call: a shell command that
   `cd`s into a directory outside the worktree (`/mnt/sharedOs/handstand-workspace/data/...`). Each step started
   streaming within 3-4 s, emitted the tool call, then waited until aborted; the tool part ends as
   `aborted: Tool execution interrupted`. The model never stalled.
   Reproduced outside the plugin in a scratch dir with `cd /mnt/sharedOs/handstand-workspace/data/keypoints && ls`:
   - `opencode run --standalone --auto`: permission auto-approved, done in 5 s;
   - `opencode run --standalone` (no `--auto`): auto-declined immediately ("The user declined this tool call");
   - plugin child session: neither. `--auto` only covers the parent session, no client is attached to the child,
     so the `external_directory` permission request waits forever.
   Needed:
   - child worker/reviewer sessions inherit the parent run's permission mode (`--auto`), or the plugin answers
     their permission requests by a configurable policy (`chainlink_child_permissions`: `inherit` | `deny` | `allow`);
   - a permission request that nobody can answer must never block: deny it after a short timeout and tell the
     model why, so it can use another route;
   - the stall detector should say *what* is pending ("waiting on permission external_directory for <path>"),
     not just "no activity";
   - the stall retry prompt ("continue; your last response stalled") is misleading for this case: the model
     retried the same command and blocked again.

## Tests
Adapter or tool normalises `#67`; a second tool call in the same turn is refused; a running workflow owned by a
live process is never marked interrupted by another instance starting; a finished worker plus stale interrupt
proceeds to review; a child session's permission request is answered by the configured policy (never left pending); a pending permission is reported by the stall detector.
