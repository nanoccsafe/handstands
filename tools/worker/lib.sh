# Shared helpers for the worker scripts. Source it; do not run it.
# State per worker lives in $WORKER_STATE/<name>.{log,env}; the worktree is ../wt-<name> next to the repo.

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
WORKER_STATE="${HANDSTAND_WORKERS:-/tmp/handstand-workers}"
TMUX_SESSION="workers"
mkdir -p "$WORKER_STATE"

wt_path()  { echo "$(dirname "$REPO")/wt-$1"; }
log_path() { echo "$WORKER_STATE/$1.log"; }
env_path() { echo "$WORKER_STATE/$1.env"; }

window_exists() { tmux list-windows -t "$TMUX_SESSION" -F '#W' 2>/dev/null | grep -qx "$1"; }

# All descendants of a pid (not including it).
descendants() {
  local kids; kids="$(pgrep -P "$1" 2>/dev/null || true)"
  for k in $kids; do echo "$k"; descendants "$k"; done
}

# Pids whose working directory is inside the worker's worktree (catches anything that escaped the tree).
# Never includes this script or its ancestors (e.g. the lead's shell sitting in the worktree).
pids_in_worktree() {
  local wt; wt="$(realpath -m "$(wt_path "$1")")"
  local self=" " p=$$
  while [[ "$p" -gt 1 ]]; do self+="$p "; p="$(ps -o ppid= -p "$p" | tr -d ' ')"; done
  for d in /proc/[0-9]*; do
    local pid="${d#/proc/}"
    [[ "$self" == *" $pid "* ]] && continue
    local cwd; cwd="$(readlink "$d/cwd" 2>/dev/null)" || continue
    [[ "$cwd" == "$wt" || "$cwd" == "$wt"/* ]] && echo "$pid"
  done
}

# Last marker line of the log: DONE / STOPPED / RUNNING / ORPHANED.
worker_state() {
  local name="$1" log; log="$(log_path "$name")"
  local last; last="$(tail -n 1 "$log" 2>/dev/null)"
  case "$last" in
    __DONE__*)    echo DONE ;;
    __STOPPED__*) echo STOPPED ;;
    *) if window_exists "$name"; then echo RUNNING; else echo ORPHANED; fi ;;
  esac
}

# Terminate: TERM, wait up to $2 seconds, then KILL. Prints pids still alive at the end (should be none).
terminate_pids() {
  local pids="$1" wait_s="${2:-10}"
  [[ -z "$pids" ]] && return 0
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 "$wait_s"); do
    local alive=""; for p in $pids; do kill -0 "$p" 2>/dev/null && alive+="$p "; done
    [[ -z "$alive" ]] && return 0
    sleep 1
  done
  kill -KILL $pids 2>/dev/null || true
  sleep 1
  for p in $pids; do kill -0 "$p" 2>/dev/null && echo "$p"; done
}
