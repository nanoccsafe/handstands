#!/usr/bin/env bash
# Stop a worker and verify nothing of it keeps running. Optionally delete everything it left behind.
# Usage: tools/worker/stop.sh <name> [--clean]      e.g. tools/worker/stop.sh i8-catalogue --clean
#        tools/worker/stop.sh --all [--clean]
#
# Stop: collect the worker's process tree, close its tmux window, TERM then KILL any survivors (including
# any process whose cwd is inside the worktree), then check that none remain.
# --clean: also remove the worktree, the branch, the log/env state and the worker's OpenCode sessions.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

stop_one() {
  local name="$1" clean="$2"
  local wt; wt="$(wt_path "$name")"

  local pids=""
  if window_exists "$name"; then
    local pane; pane="$(tmux list-panes -t "$TMUX_SESSION:$name" -F '#{pane_pid}' | head -n 1)"
    pids="$pane $(descendants "$pane" | tr '\n' ' ')"
    tmux kill-window -t "$TMUX_SESSION:$name"
    echo "$name: closed tmux window"
  fi
  pids="$pids $(pids_in_worktree "$name" | tr '\n' ' ')"
  pids="$(echo $pids | tr ' ' '\n' | sort -un | tr '\n' ' ')"

  local survivors; survivors="$(terminate_pids "$pids" 10)"
  if [[ -n "$survivors" ]]; then
    echo "$name: WARNING processes still alive after KILL: $survivors" >&2
    return 1
  fi
  echo "$name: no processes left"

  local log; log="$(log_path "$name")"
  if [[ -f "$log" ]] && ! tail -n 1 "$log" | grep -qE '^__(DONE|STOPPED)__'; then
    echo "__STOPPED__ $(date -Is)" >> "$log"
  fi

  if [[ "$clean" == "--clean" ]]; then
    [[ -e "$wt" ]] && git -C "$REPO" worktree remove --force "$wt" && echo "$name: removed worktree"
    git -C "$REPO" worktree prune
    git -C "$REPO" show-ref --quiet "refs/heads/$name" && git -C "$REPO" branch -q -D "$name" \
      && echo "$name: deleted branch"
    # OpenCode keeps every session; delete the ones titled with this worker's name.
    opencode session list 2>/dev/null | awk -v n="$name" -F '\t' '$2 == n {print $1}' | while read -r sid; do
      opencode session delete "$sid" >/dev/null 2>&1 && echo "$name: deleted opencode session $sid"
    done
    rm -f "$log" "$(env_path "$name")" "$WORKER_STATE/$name.loop-state.json"
  fi
}

target="${1:?usage: stop.sh <name>|--all [--clean]}"; clean="${2:-}"
if [[ "$target" == "--all" ]]; then
  names="$( { tmux list-windows -t "$TMUX_SESSION" -F '#W' 2>/dev/null | grep '^i[0-9]' || true;
              ls "$WORKER_STATE" 2>/dev/null | sed -n 's/\.env$//p'; } | sort -u)"
  rc=0; for n in $names; do stop_one "$n" "$clean" || rc=1; done; exit $rc
fi
stop_one "$target" "$clean"
