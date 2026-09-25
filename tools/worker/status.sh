#!/usr/bin/env bash
# Show state of every worker log: running / done, plus the last lines.
# Usage: tools/worker/status.sh [lines]
lines="${1:-5}"
shopt -s nullglob
for log in /tmp/wt-i*.log; do
  name="$(basename "$log" .log)"
  last="$(tail -n 1 "$log")"
  case "$last" in __DONE__) state=DONE ;; __STOPPED__) state=STOPPED ;; *) state=RUNNING ;; esac
  echo "== ${name#wt-} [$state]"
  grep -vE "__DONE__|__STOPPED__" "$log" | tail -n "$lines"
done
