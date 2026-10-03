#!/usr/bin/env bash
# Put the MediaPipe pose landmarker model where the iOS app can bundle it.
#
#   tools/ios/fetch_mediapipe_model.sh
#
# Chainlink #45: the app runs MediaPipe exactly the way the pipeline does,
# with `pose_landmarker_full.task` (chainlink #16's bake-off picked MediaPipe,
# docs/bakeoff.md). The model is a build artifact — `*.task` and
# `ios/LocalResources/` are both git-ignored — so this script produces
# `ios/LocalResources/models/pose_landmarker_full.task` from, in order:
#
#   1. nothing, if it is already there;
#   2. `pipeline/models/pose_landmarker_full.task`, when that exists — the
#      exact file `pipeline/scripts/download_models.py` fetched, copied
#      instead of downloaded (works offline);
#   3. `MODEL_URL` as written in `pipeline/handstand/pose_mediapipe.py`,
#      read out of the module with python's ast so the pipeline and the app
#      can never download different files (the constant is a concatenated
#      string literal, so grep would not do).
#
# `tools/mac/ios_build.sh` runs this before its rsync whenever the file is
# missing. Exits non-zero when the model could not be produced.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
filename="pose_landmarker_full.task"
dest="$repo_root/ios/LocalResources/models/$filename"

if [[ -f "$dest" ]]; then
    echo "fetch_mediapipe_model: already present: $dest ($(du -h "$dest" | cut -f1))"
    exit 0
fi

mkdir -p "$(dirname "$dest")"

src="$repo_root/pipeline/models/$filename"
if [[ -f "$src" ]]; then
    cp "$src" "$dest"
    echo "fetch_mediapipe_model: copied from pipeline/models/"
    exit 0
fi

url="$(python3 - "$repo_root/pipeline/handstand/pose_mediapipe.py" <<'PY' 2>/dev/null || true
import ast
import pathlib
import sys

tree = ast.parse(pathlib.Path(sys.argv[1]).read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "MODEL_URL":
                print(ast.literal_eval(node.value))
                raise SystemExit(0)
raise SystemExit(1)
PY
)"
# Fallback: the same URL, spelled out, if python (or the file) is missing.
url="${url:-https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task}"

echo "fetch_mediapipe_model: downloading $url"
partial="$dest.part"
trap 'rm -f "$partial"' EXIT
curl -fL --retry 3 --connect-timeout 20 -o "$partial" "$url"
[[ -s "$partial" ]] || { echo "fetch_mediapipe_model: empty download" >&2; exit 1; }
mv "$partial" "$dest"
trap - EXIT
echo "fetch_mediapipe_model: saved: $dest ($(du -h "$dest" | cut -f1))"
