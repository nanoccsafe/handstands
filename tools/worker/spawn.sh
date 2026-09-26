#!/usr/bin/env bash
# Spawn an OpenCode worker for one chainlink issue in its own worktree + tmux window.
#
# Usage:
#   tools/worker/spawn.sh <issue-id> <slug> --chainlink [worker-model] [--rerun]
#       Runs the opencode-loop-plugin's deterministic `chainlink-loop --task <id> --no-close` (no LLM in the
#       control path; one-shot `opencode run --standalone --auto` steps: build-agent worker, plan-agent reviewer
#       that cannot edit, findings fed back to the same worker session). Existing work on the branch or issue
#       comments are reviewed first. The issue stays open for the lead's review and the user's merge approval.
#       The issue description is the spec. Env: CHAINLINK_LOOP (path to the CLI), REVIEWER_MODEL,
#       LOOP_ARGS (extra chainlink-loop flags, e.g. "--worker-timeout 10800").
#   tools/worker/spawn.sh <issue-id> <slug> <prompt-file> [model] [--rerun]
#       Plain one-shot worker with a prompt file (fallback when the plugin is unavailable).
#
# --rerun reuses the existing worktree and branch (after the lead's review asked for changes); it still refuses
# if a window or process of that worker is alive.
#
# Uses `opencode run --standalone` so the worker has a private server that dies with the tmux window.
# (Without --standalone the run lives in the shared background service and keeps going after the window
# is closed.)
set -euo pipefail
source "$(dirname "$0")/lib.sh"

rerun=""
args=()
for a in "$@"; do [[ "$a" == "--rerun" ]] && rerun=1 || args+=("$a"); done
set -- "${args[@]}"

issue="$1"; slug="$2"; mode="$3"
model="${4:-opencode/space-bunny-free}"
name="i${issue}-${slug}"
wt="$(wt_path "$name")"; log="$(log_path "$name")"; env="$(env_path "$name")"

leftover=""
window_exists "$name" && leftover+=" tmux-window"
[[ -n "$(pids_in_worktree "$name")" ]] && leftover+=" processes"
if [[ -z "$rerun" ]]; then
  [[ -e "$wt" ]] && leftover+=" worktree"
  git -C "$REPO" show-ref --quiet "refs/heads/$name" && leftover+=" branch"
fi
if [[ -n "$leftover" ]]; then
  echo "refusing: leftover$leftover for $name. Stop it (tools/worker/stop.sh $name [--clean]) or use --rerun." >&2
  exit 1
fi

if [[ -n "$rerun" ]]; then
  [[ -d "$wt" ]] || { echo "--rerun: no worktree $wt" >&2; exit 1; }
else
  git -C "$REPO" worktree add -q "$wt" -b "$name" main
fi

loop_cli="${CHAINLINK_LOOP:-$HOME/.config/opencode/plugins/opencode-loop-plugin/bin/chainlink-loop}"
reviewer_model="${REVIEWER_MODEL:-opencode-go/mimo-v2.6-flash}"
if [[ "$mode" == "--chainlink" ]]; then
  [[ -x "$loop_cli" ]] || { echo "chainlink-loop not found at $loop_cli (set CHAINLINK_LOOP)" >&2; exit 1; }
  run_cmd="'$loop_cli' --task $issue --no-close --worker-model '$model' --reviewer-model '$reviewer_model' ${LOOP_ARGS:-}"
else
  cp "$mode" "$wt/.worker-prompt.md"
  run_cmd="opencode run --standalone --auto -m '$model' --title '$name' \"\$(cat .worker-prompt.md)\""
fi

[[ -n "$rerun" ]] && echo "__RERUN__ $(date -Is)" >> "$log" || : > "$log"
cat > "$env" <<EOF
NAME=$name
ISSUE=$issue
MODE=$([[ "$mode" == "--chainlink" ]] && echo chainlink || echo prompt)
MODEL=$model
REVIEWER_MODEL=$([[ "$mode" == "--chainlink" ]] && echo "$reviewer_model" || echo -)
WORKTREE=$wt
STARTED=$(date -Is)
EOF

tmux has-session -t "$TMUX_SESSION" 2>/dev/null || tmux new-session -d -s "$TMUX_SESSION" -n idle
tmux new-window -d -t "$TMUX_SESSION" -n "$name" -c "$wt" \
  -e "CHAINLINK_DB=$REPO/.chainlink" \
  -e "OPENCODE_LOOP_STATE_PATH=$WORKER_STATE/$name.loop-state.json" \
  "$run_cmd 2>&1 | tee -a '$log';
   echo \"__DONE__ exit=\${PIPESTATUS[0]} \$(date -Is)\" >> '$log'"

echo "spawned $name ($([[ "$mode" == "--chainlink" ]] && echo "chainlink-loop, worker $model, reviewer $reviewer_model" || echo "prompt file, model $model")) in $wt, log $log"
