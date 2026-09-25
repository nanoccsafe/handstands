#!/usr/bin/env bash
# Spawn an OpenCode worker for one chainlink issue in its own worktree + tmux window.
# Usage: tools/worker/spawn.sh <issue-id> <slug> <prompt-file> [model]
set -euo pipefail

issue="$1"; slug="$2"; prompt_file="$3"
model="${4:-opencode/space-bunny-free}"
name="i${issue}-${slug}"
repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
wt="$(dirname "$repo")/wt-${name}"
log="/tmp/wt-${name}.log"

git -C "$repo" worktree add -q "$wt" -b "$name" main
cp "$prompt_file" "$wt/.worker-prompt.md"

tmux has-session -t workers 2>/dev/null || tmux new-session -d -s workers -n idle
tmux new-window -d -t workers -n "$name" -c "$wt" \
  "opencode run --auto -m '$model' --title '$name' \"\$(cat .worker-prompt.md)\" 2>&1 | tee '$log'; echo __DONE__ >> '$log'"

echo "spawned $name (model $model) in $wt, log $log"
