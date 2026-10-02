#!/usr/bin/env bash
# Run the Apple Vision body-pose keypoints for a set of clips, on the Mac mini,
# from the Linux workstation.
#
# Apple's body-pose model is macOS/iOS only, so the Swift runner
# (swift/VisionPose) cannot run here. This script is the whole round trip:
# copy the clips across, copy the package across, build it release, run it,
# copy the CSVs back into `<data_dir>/keypoints/vision_raw/`. Everything after
# that (the parquets) is `handstand.vision_import` on this side.
#
#   tools/mac/run_vision.sh --rotate auto
#   tools/mac/run_vision.sh --rotate none --rotate auto --clips 6508f9b355bd 64184de33f84
#   tools/mac/run_vision.sh --all --rotate auto
#
# --rotate may be repeated. --clips takes catalogue clip ids (or file names);
# --all takes every clip in the catalogue. With neither, every clip whose
# file is present in the videos directory is used.
#
# The clips are copied with `rsync --ignore-existing`, so a second run over the
# same clips transfers nothing but the keypoints back. A clip's frame count is
# compared against the MediaPipe run of the *same* rotation mode, which is what
# the `--expect-frames` warning in the runner is for: the two have to line up
# frame for frame before the bake-off (chainlink #16) can compare them.
#
# Env: HANDSTAND_DATA, HANDSTAND_VIDEOS (defaults as in handstand.paths).
# Flags: --host, --remote-root, --max-frames, --skip-build.
set -euo pipefail

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
}

die() { echo "run_vision: $*" >&2; exit 1; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_dir="${HANDSTAND_DATA:-/mnt/sharedOs/handstand-workspace/data}"
videos_dir="${HANDSTAND_VIDEOS:-/mnt/sharedOs/handstand-workspace/videos}"
catalogue="$data_dir/catalogue.csv"
package="$repo_root/swift/VisionPose"

host="macmini"
remote_root="wt/i15"
remote_videos="handstand-data/videos"
remote_vision="handstand-data/vision"
max_frames=""
skip_build=0
rotate=()
clips=()
all=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rotate) rotate+=("${2:?--rotate needs a value}"); shift 2 ;;
        --clips) shift; while [[ $# -gt 0 && "$1" != --* ]]; do clips+=("$1"); shift; done ;;
        --all) all=1; shift ;;
        --max-frames) max_frames="${2:?--max-frames needs a value}"; shift 2 ;;
        --host) host="${2:?--host needs a value}"; shift 2 ;;
        --remote-root) remote_root="${2:?--remote-root needs a value}"; shift 2 ;;
        --skip-build) skip_build=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

[[ ${#rotate[@]} -gt 0 ]] || rotate=(none)
for mode in "${rotate[@]}"; do
    case "$mode" in
        none|180|auto) ;;
        *) die "--rotate must be none, 180 or auto, got '$mode'" ;;
    esac
done
[[ -d $package ]] || die "missing $package"
[[ -f $catalogue ]] || die "missing $catalogue ($catalogue)"

# clip id -> file name, from the catalogue (the pipeline's clip id definition).
# Rows with `missing=true` are skipped: their video file is gone (chainlink #49),
# so --all must not try to copy or run them.
declare -A filename_by_id=()
declare -A id_by_name=()
while IFS=, read -r clip_id filename _rest; do
    [[ $clip_id == "clip_id" || -z $clip_id ]] && continue
    filename="${filename%$'\r'}"
    missing="${_rest##*,}"
    [[ "${missing%$'\r'}" =~ ^(true|1|yes)$ ]] && continue
    filename_by_id["$clip_id"]="$filename"
    id_by_name["$filename"]="$clip_id"
done < <(tail -n +2 "$catalogue")
[[ ${#filename_by_id[@]} -gt 0 ]] || die "no clips in $catalogue"

# Which clips to run: --clips, --all, or every file present locally.
selected=()
if [[ ${#clips[@]} -gt 0 ]]; then
    for wanted in "${clips[@]}"; do
        if [[ -n ${filename_by_id[$wanted]:-} ]]; then
            selected+=("$wanted")
        elif [[ -n ${id_by_name[$wanted]:-} ]]; then
            selected+=("${id_by_name[$wanted]}")
        else
            die "clip '$wanted' is not in $catalogue"
        fi
    done
elif [[ $all -eq 1 ]]; then
    selected=("${!filename_by_id[@]}")
else
    for path in "$videos_dir"/*.mp4; do
        [[ -e $path ]] || continue
        clip_id="${id_by_name[$(basename "$path")]:-}"
        [[ -n $clip_id ]] && selected+=("$clip_id")
    done
    [[ ${#selected[@]} -gt 0 ]] || die "no *.mp4 in $videos_dir"
fi

echo "clips=${#selected[@]} rotate=${rotate[*]} host=$host data=$data_dir"

# --------------------------------------------------------------------------- #
# 1. The clips, once. --ignore-existing keeps a second run free.
# --------------------------------------------------------------------------- #
files=()
for clip_id in "${selected[@]}"; do
    path="$videos_dir/${filename_by_id[$clip_id]}"
    [[ -f $path ]] || die "missing video for $clip_id: $path"
    files+=("$path")
done
echo "== copying videos to $host:$remote_videos (skipping any already there)"
rsync -a --ignore-existing --include='*/' --include='*.mp4' --exclude='*' \
    "${files[@]}" "$host:$remote_videos/"

# --------------------------------------------------------------------------- #
# 2. The package, and a release build of it.
# --------------------------------------------------------------------------- #
# VisionPose depends on swift/HandstandCore by relative path (see its
# Package.swift), so both packages have to land on the Mac — a fresh remote
# root without HandstandCore fails the build with "the package at
# .../HandstandCore cannot be accessed" (chainlink #16).
echo "== copying swift/HandstandCore to $host:$remote_root/swift/HandstandCore"
rsync -a --delete --exclude .build "$repo_root/swift/HandstandCore/" \
    "$host:$remote_root/swift/HandstandCore/"
echo "== copying swift/VisionPose to $host:$remote_root/swift/VisionPose"
rsync -a --delete --exclude .build "$package/" "$host:$remote_root/swift/VisionPose/"

binary="$host:$remote_root/swift/VisionPose/.build/release/vision-pose"
if [[ $skip_build -eq 0 ]]; then
    echo "== swift build -c release on $host"
    ssh "$host" "cd $remote_root/swift/VisionPose && swift build -c release"
fi

# --------------------------------------------------------------------------- #
# 3. Run, per clip and per rotation mode.
# --------------------------------------------------------------------------- #
failures=0
for mode in "${rotate[@]}"; do
    ssh "$host" "mkdir -p '$remote_vision/$mode'"
    for clip_id in "${selected[@]}"; do
        filename="${filename_by_id[$clip_id]}"
        # The MediaPipe run of the same mode, if there is one, is the reference
        # frame count: it has to be identical or the two keypoint sets do not
        # line up.
        reference="$data_dir/keypoints/mediapipe/$mode/$clip_id.json"
        expect=()
        if [[ -f $reference ]]; then
            count="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frame_count"])' \
                "$reference" 2>/dev/null || true)"
            [[ -n $count ]] && expect=(--expect-frames "$count")
        fi
        limit=()
        [[ -n $max_frames ]] && limit=(--max-frames "$max_frames")

        echo "== vision-pose $clip_id rotate=$mode${expect[*]:+ expect=${expect[*]}}"
        if ! ssh "$host" "$remote_root/swift/VisionPose/.build/release/vision-pose \
                --video '$remote_videos/$filename' \
                --rotate '$mode' \
                --out '$remote_vision/$mode/$clip_id.csv' \
                --clip-id '$clip_id' \
                ${expect[*]:+${expect[*]}} ${limit[*]:+${limit[*]}}"; then
            echo "run_vision: FAILED $clip_id rotate=$mode" >&2
            failures=$((failures + 1))
        fi
    done
done

# --------------------------------------------------------------------------- #
# 4. The CSVs (and their run manifests) back.
# --------------------------------------------------------------------------- #
for mode in "${rotate[@]}"; do
    target="$data_dir/keypoints/vision_raw/$mode"
    mkdir -p "$target"
    echo "== copying $mode keypoints back to $target"
    rsync -a "$host:$remote_vision/$mode/" "$target/"
done

if [[ $failures -gt 0 ]]; then
    die "$failures clip/mode run(s) failed"
fi
echo "== done: run 'cd pipeline && uv run python -m handstand.vision_import --rotate <mode>' to import"
