# Prompt: opencode-loop-plugin — issues from the first single-issue run (#24, 2026-09-25)

Run: `CHAINLINK_DB=… opencode run --standalone --auto -m opencode/space-bunny-free "/chainlink #24 --no-close"`.
Parsing worked (`run_chainlink_outer {"task_ids":["24"],"close_on_approval":false}`), the right issue was
selected and the issue was left open. Three problems:

1. **A stalled model stream blocks the loop for the full worker timeout.**
   Worker session `Chainlink 24 worker` (agent build, model opencode-go/space-bunny-free) finished the work by
   11:58, started its next assistant message (streamed at 11:58:30), then received nothing for 51 minutes until the
   3600 s `waitForSession` timeout interrupted it at 12:49:49. Add an inactivity/stall timeout (no new message
   parts for N seconds, configurable, e.g. 300 s) that aborts the stuck turn and re-prompts the same worker session
   once ("continue; your last response stalled"), counting it as an attempt, before failing the workflow.

2. **After the tool failed, the orchestrator session did the task itself.**
   `run_chainlink_outer` returned `{"status":"failed","error":"Chainlink child session … timed out"}`; the parent
   (the `/chainlink` command turn) then implemented and committed the task in place, with no reviewer pass. The
   command template says not to perform the task in the command turn, but that is not enforced for the
   failure path. On failure the orchestrator must only report the status and error and stop. Consider saying so
   in the tool result itself ("do not continue the task yourself"), since small models read the latest message.

3. **Workflow status is mislabeled.**
   The workflow ended as `interrupted` / "OpenCode restarted before the Chainlink workflow finished" with
   `attemptsUsed: 0` and `lastError: null`, while the real cause was the child timeout above. Record `failed`
   with the timeout error. Also check that startup's `interruptActiveChainlinkWorkflows` only touches workflows
   owned by this process/server: other `opencode` processes (e.g. `opencode session list` from another shell)
   start while a standalone run is active and may mark its live workflow interrupted.

Tests: a fake session that stops producing parts triggers the stall path (re-prompt, then fail); a failing
child makes the tool result instruct the caller to stop; status/error recorded as failed; a second plugin
instance starting does not interrupt another instance's running workflow.
