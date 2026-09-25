#!/usr/bin/env bash
# Stop a worker (close its tmux window, which ends its opencode run). Optionally delete its worktree + branch.
# Usage: tools/worker/stop.sh <name> [--clean]     e.g. tools/worker/stop.sh i8-catalogue --clean
set -euo pipefail

name="$1"; clean="${2:-}"
repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
wt="$(dirname "$repo")/wt-${name}"
log="/tmp/wt-${name}.log"

if tmux kill-window -t "workers:${name}" 2>/dev/null; then
  echo "__STOPPED__" >> "$log"
  echo "stopped $name"
else
  echo "no running window for $name"
fi

if [[ "$clean" == "--clean" ]]; then
  git -C "$repo" worktree remove --force "$wt" 2>/dev/null && echo "removed worktree $wt"
  git -C "$repo" branch -D "$name" 2>/dev/null && echo "deleted branch $name"
  rm -f "$log"
fi
