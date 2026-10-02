#!/usr/bin/env bash
# Build the iOS app and run its unit tests on the Mac mini, from Linux.
#
#   tools/mac/ios_build.sh [remote-dir]
#
# remote-dir  Directory on the Mac that receives ios/ and swift/
#             (default wt/i44)
#
# What it does:
#   1. fetch the MediaPipe model (tools/ios/fetch_mediapipe_model.sh) when
#      ios/LocalResources/models/pose_landmarker_full.task is missing, so the
#      remote copy carries it — the model is never committed (chainlink #45);
#   2. rsync the repo's ios/ and swift/ to $host:$remote-dir/, excluding
#      .build, .swiftpm, DerivedData, *.xcodeproj, Pods/ and the generated
#      *.xcworkspace — the project, Pods and the DerivedData cache live on
#      the Mac, only the sources (plus Podfile/Podfile.lock) travel;
#   3. on the Mac: `xcodegen generate` then `pod install` in
#      $remote-dir/ios — always in that order, CocoaPods hooks into the
#      generated project; `pod install` runs with --no-repo-update when
#      Pods/ is already installed, so a re-run needs no network;
#   4. `xcodebuild build` of the generated .xcworkspace for
#      "generic/platform=iOS Simulator";
#   5. pick an iPhone from `xcrun simctl list devices available` (no
#      hard-coded UUID) and run `xcodebuild test` on it;
#   6. print the result lines (** BUILD SUCCEEDED **, ** TEST SUCCEEDED **,
#      "Executed … tests") and exit non-zero if any step failed.
#
# Examples:
#   tools/mac/ios_build.sh
#   tools/mac/ios_build.sh wt/i44
#
# Env: HANDSTAND_MAC_HOST (default macmini).
set -euo pipefail

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

die() { echo "ios_build: $*" >&2; exit 1; }

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

remote_dir="${1:-wt/i44}"
host="${HANDSTAND_MAC_HOST:-macmini}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -d "$repo_root/ios" && -d "$repo_root/swift/HandstandCore" ]] \
    || die "repo layout not found under $repo_root"
[[ "$remote_dir" =~ ^[A-Za-z0-9._/-]+$ && "$remote_dir" != /* && "$remote_dir" != *..* ]] \
    || die "remote dir must be a safe relative path on $host: '$remote_dir'"

# The .xcodeproj is generated on the Mac (xcodegen), the workspace on the Mac
# (pod install), Pods/ lives only there, DerivedData is a build cache that
# must survive the sync, and .build/.swiftpm belong to swift test. The
# excludes also protect those directories from --delete.
exclude=(
    --exclude '.build' --exclude '.swiftpm' --exclude 'DerivedData'
    --exclude '*.xcodeproj' --exclude 'Pods' --exclude '*.xcworkspace'
)

# The model the default backend loads (chainlink #45): not committed, so it
# is fetched into ios/LocalResources/models/ before the rsync when missing —
# a copy from pipeline/models or a download, see the script itself.
if [[ ! -f "$repo_root/ios/LocalResources/models/pose_landmarker_full.task" ]]; then
    echo "== MediaPipe model missing; running tools/ios/fetch_mediapipe_model.sh"
    "$repo_root/tools/ios/fetch_mediapipe_model.sh"
fi

echo "== rsync ios/ and swift/ to $host:$remote_dir/ (excluding .build, DerivedData, *.xcodeproj, Pods, *.xcworkspace)"
rsync -a --delete "${exclude[@]}" "$repo_root/ios/" "$host:$remote_dir/ios/"
rsync -a --delete "${exclude[@]}" "$repo_root/swift/" "$host:$remote_dir/swift/"

tmp_dir="${TMPDIR:-/tmp/opencode}"
mkdir -p "$tmp_dir"
log="$(mktemp "$tmp_dir/ios_build.XXXXXX")"
trap 'rm -f "$log"' EXIT

echo "== xcodegen + xcodebuild on $host: $remote_dir/ios"
# The remote script arrives on stdin; REMOTE_DIR is set in the remote command
# string because ssh does not forward the caller's environment.
if ! ssh "$host" "REMOTE_DIR='$remote_dir' bash -s" 2>&1 <<'REMOTE' | tee "$log"
set -euo pipefail
export PATH="/opt/homebrew/bin:${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"

[[ -n "${REMOTE_DIR:-}" && -d "$REMOTE_DIR/ios" ]] || {
    echo "ios_build: no $REMOTE_DIR/ios on the Mac" >&2
    exit 1
}
cd "$REMOTE_DIR/ios"

echo "== xcodegen generate"
xcodegen generate

echo "== pod install (CocoaPods: MediaPipeTasksVision, chainlink #45)"
# `pod install` always runs AFTER `xcodegen generate` — it hooks into the
# generated project, and a regenerated project has to be hooked in again.
# With Pods/ already installed --no-repo-update keeps it offline; a first
# install (no Pods/ yet) needs the network once to fetch the pod.
if [[ -d Pods ]]; then
    pod install --no-repo-update
else
    pod install
fi

echo "== xcodebuild build (generic/platform=iOS Simulator)"
xcodebuild -workspace Handstand.xcworkspace -scheme HandstandApp \
    -destination 'generic/platform=iOS Simulator' \
    -derivedDataPath DerivedData build

echo "== pick an available iPhone simulator"
iphone_line="$(xcrun simctl list devices available \
    | grep -E '^[[:space:]]+iPhone ' | head -n 1 || true)"
[[ -n "$iphone_line" ]] || {
    echo "ios_build: no available iPhone simulator on the Mac" >&2
    exit 1
}
udid="$(sed -E 's/.*\(([0-9A-Fa-f-]{36})\).*/\1/' <<<"$iphone_line")"
name="$(sed -E 's/^[[:space:]]*//; s/[[:space:]]*\([0-9A-Fa-f-]{36}\).*//' <<<"$iphone_line")"
[[ "$udid" =~ ^[0-9A-Fa-f-]{36}$ ]] || {
    echo "ios_build: could not parse a simulator id from: $iphone_line" >&2
    exit 1
}
echo "simulator: $name ($udid)"

echo "== xcodebuild test ($name)"
xcodebuild -workspace Handstand.xcworkspace -scheme HandstandApp \
    -destination "platform=iOS Simulator,id=$udid" \
    -derivedDataPath DerivedData test
REMOTE
then
    die "xcodegen / build / test FAILED on $host in $remote_dir/ios"
fi

# The lines the report quotes: the xcodebuild result banners and XCTest's
# per-suite counts.
echo "== summary"
summary="$(grep -E '\*\* (BUILD|TEST) (SUCCEEDED|FAILED) \*\*|Executed [0-9]+ tests|Test Suite .All tests.' "$log" || true)"
if [[ -z "$summary" ]]; then
    echo "  (no result line found in the output above)"
else
    while IFS= read -r line; do echo "  $line"; done <<< "$summary"
fi
