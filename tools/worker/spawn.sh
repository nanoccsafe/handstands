#!/usr/bin/env bash
# Spawn an OpenCode worker for one chainlink issue in its own worktree + tmux window.
# Usage: tools/worker/spawn.sh <issue-id> <slug> <prompt-file> [model]
#
# Uses `opencode run --standalone` so the worker has a private server that dies with the tmux window.
# (Without --standalone the run lives in the shared background service and keeps going after the window
# is closed.)
set -euo pipefail
source "$(dirname "$0")/lib.sh"

issue="$1"; slug="$2"; prompt_file="$3"
model="${4:-opencode/space-bunny-free}"
name="i${issue}-${slug}"
wt="$(wt_path "$name")"; log="$(log_path "$name")"; env="$(env_path "$name")"

# Refuse to reuse anything left over from a previous worker with the same name.
leftover=""
window_exists "$name" && leftover+=" tmux-window"
[[ -e "$wt" ]] && leftover+=" worktree"
git -C "$REPO" show-ref --quiet "refs/heads/$name" && leftover+=" branch"
[[ -n "$(pids_in_worktree "$name")" ]] && leftover+=" processes"
if [[ -n "$leftover" ]]; then
  echo "refusing: leftover$leftover for $name. Run: tools/worker/stop.sh $name --clean" >&2
  exit 1
fi

git -C "$REPO" worktree add -q "$wt" -b "$name" main
cp "$prompt_file" "$wt/.worker-prompt.md"
: > "$log"
cat > "$env" <<EOF
NAME=$name
ISSUE=$issue
MODEL=$model
WORKTREE=$wt
STARTED=$(date -Is)
EOF

tmux has-session -t "$TMUX_SESSION" 2>/dev/null || tmux new-session -d -s "$TMUX_SESSION" -n idle
tmux new-window -d -t "$TMUX_SESSION" -n "$name" -c "$wt" \
  "opencode run --standalone --auto -m '$model' --title '$name' \"\$(cat .worker-prompt.md)\" 2>&1 | tee -a '$log';
   echo \"__DONE__ exit=\${PIPESTATUS[0]} \$(date -Is)\" >> '$log'"

echo "spawned $name (model $model) in $wt, log $log"
