#!/usr/bin/env bash
# Show every worker: state, branch commits, uncommitted files, and the last log lines.
# States: RUNNING (window alive), DONE, STOPPED, ORPHANED (no window, no end marker: it died or was killed
# outside stop.sh; run stop.sh <name> to make sure nothing is left).
# Usage: tools/worker/status.sh [lines]
set -uo pipefail
source "$(dirname "$0")/lib.sh"
lines="${1:-5}"

names="$( { ls "$WORKER_STATE" 2>/dev/null | sed -n 's/\.env$//p';
            tmux list-windows -t "$TMUX_SESSION" -F '#W' 2>/dev/null | grep '^i[0-9]' || true; } | sort -u)"
[[ -z "$names" ]] && { echo "no workers"; exit 0; }

for name in $names; do
  wt="$(wt_path "$name")"
  commits="-"; dirty="-"
  if [[ -d "$wt" ]]; then
    commits="$(git -C "$wt" rev-list --count main..HEAD 2>/dev/null || echo ?)"
    dirty="$(git -C "$wt" status --porcelain 2>/dev/null | grep -vc 'worker-prompt' || true)"
  fi
  procs="$(pids_in_worktree "$name" | wc -l)"
  echo "== $name [$(worker_state "$name")] commits=$commits uncommitted=$dirty procs_in_worktree=$procs"
  if [[ "$lines" -gt 0 ]]; then
    grep -vE '^__(DONE|STOPPED)__' "$(log_path "$name")" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -n "$lines"
  fi
done
exit 0
