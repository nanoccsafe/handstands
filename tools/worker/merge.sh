#!/usr/bin/env bash
# Merge an approved worker branch, test, push, and only then clean up and close the issue.
# Usage: tools/worker/merge.sh <name>          e.g. tools/worker/merge.sh i24-overlay
#
# Every step must succeed before the next one runs; nothing is deleted unless the merge landed on main,
# the tests passed and the push succeeded.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

name="$1"
issue="${name#i}"; issue="${issue%%-*}"
cd "$REPO"

[[ "$(git branch --show-current)" == "main" ]] || { echo "not on main" >&2; exit 1; }
git show-ref --quiet "refs/heads/$name" || { echo "no branch $name" >&2; exit 1; }
[[ "$(worker_state "$name")" != RUNNING ]] || { echo "$name is still running" >&2; exit 1; }

# chainlink appends to CHANGELOG.md when issues close; commit that first so it cannot block the merge.
if ! git diff --quiet -- CHANGELOG.md; then
  git add CHANGELOG.md
  git commit -q -m "CHANGELOG: chainlink entries"
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "main has uncommitted changes; commit or discard them first:" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

before="$(git rev-parse HEAD)"
if ! git merge -q --no-ff "$name" -m "Merge $name (#$issue)"; then
  echo "merge failed; resolve it by hand. Nothing was cleaned up." >&2
  exit 1
fi
git merge-base --is-ancestor "$name" HEAD || { echo "branch not contained in main after merge" >&2; exit 1; }

if [[ -d pipeline ]]; then
  if ! (cd pipeline && uv sync -q && uv run pytest -q && uv run ruff check); then
    echo "tests failed after merge; undoing the merge. Nothing was cleaned up." >&2
    git reset -q --hard "$before"
    exit 1
  fi
fi

git push -q
"$(dirname "$0")/stop.sh" "$name" --clean
chainlink -q issue close "$issue" || echo "note: could not close chainlink issue #$issue" >&2
echo "merged $name into main, pushed, cleaned up, closed #$issue"
