# chainlink-loop: the review-first path ends in "workflow … is in phase worker"

Observed on 2026-10-02 in the handstand repo (chainlink #86), with `chainlink-loop --task 86 --no-close
--worker-model opencode-go/mimo-v2.6-flash --reviewer-model opencode-go/mimo-v2.6-flash --worker-timeout 14400`,
started by `tools/worker/spawn.sh 86 ingest --chainlink`. It was a fresh worktree with no commits on the branch.

## What happened

```
[chainlink] task 86: existing work detected (the issue already has 5 comment(s) recording earlier work); reviewing before dispatching a worker
[chainlink] $ opencode run plan Chainlink 86 reviewer 1
[chainlink] status: failed after 1 task(s)
[chainlink]   #86 failed (0 attempt(s)) Chainlink workflow "chainlink_32prs" is in phase worker
[chainlink] error: Chainlink workflow "chainlink_32prs" is in phase worker
```

The loop exited with code 1. No worker step ran, and the branch stayed empty. The state file
(`OPENCODE_LOOP_STATE_PATH`) holds the workflow `chainlink_32prs` for task 86. After the review-first reviewer step it
was left in phase `worker`, and the loop then refused to proceed from that same phase.

## Root cause (filed as plugin chainlink #6, 2026-10-02)

runReviewer() moves the workflow to phase "reviewer" only `if (result.sessionID)` (src/chainlink-process.ts). The
reviewer step here produced NO OpenCode session (there's no "Chainlink 86 reviewer 1" session in opencode.db), so
the phase stayed "worker", and recordChainlinkReview() (src/state.ts) threw the phase-mismatch error. The real
failure is that the reviewer step didn't start, and the loop doesn't log why.

## Why the review-first path was taken

The issue's comments were the lead's scope notes (spec additions), not earlier work. The loop treats any comment
as "earlier work", so it reviewed an empty branch first.

## Expected

- After the review-first reviewer step, the loop dispatches the worker with the review's findings (or with the
  spec when there's nothing to review), and the phase transition reviewer → worker succeeds.
- Possibly: "existing work" should require commits on the branch (or a worktree diff), not just issue comments.
  Lead comments that refine the spec aren't work to review.

## Repro

1. Create a chainlink issue with a description and a few comments, and no commits on its branch.
2. Run `chainlink-loop --task <id> --no-close` in a fresh worktree.
3. It reviews first, then fails with `workflow "<id>" is in phase worker`.

## Workaround used (not a fix)

The lead moved the stale state file aside (kept as `i86-ingest.loop-state.failed-review-first.json` in
`/tmp/handstand-workers/`) and reran with the plugin's documented `--review-first never`. The bug in the review-first
path remains: any issue with comments and an empty branch hits it under the default `--review-first auto`. The plugin
itself was not modified.
