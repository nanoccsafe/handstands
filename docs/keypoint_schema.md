# Keypoint schema

Output of the MediaPipe pose runner (`pipeline/handstand/pose_mediapipe.py`).
The Apple Vision runner (`ios/`, later issue) must write byte-compatible files:
same columns, same types, same coordinate conventions.

## Files

```
<data_dir>/keypoints/mediapipe/<rotate>/<clip_id>.parquet         # the keypoints
<data_dir>/keypoints/mediapipe/<rotate>/<clip_id>.json             # sidecar metadata
<data_dir>/keypoints/mediapipe_multi/<rotate>/<clip_id>.parquet    # --num-poses > 1
<data_dir>/keypoints/mediapipe_multi/<rotate>/<clip_id>.json        # sidecar metadata
```

* `<data_dir>` = `handstand.paths.data_dir()` (default
  `/mnt/sharedOs/handstand-workspace/data`, override with `$HANDSTAND_DATA`).
* `<rotate>` = the rotation mode the file was produced with: `none`, `180` or
  `auto`. The three modes are independent outputs of the same clip, never
  merged — compare them side by side.
* `mediapipe_multi` only exists for `--num-poses > 1`. A multi-person run
  writes to its own root, so it can never overwrite the single-person
  parquets the other stages read.
* `<clip_id>` = the `clip_id` column of `<data_dir>/catalogue.csv`
  (`clip_id,filename`) when that file exists, otherwise the first 12 hex
  characters of the SHA-1 of the video file's bytes (same definition as the
  catalogue).

Generate the model first with `cd pipeline && uv run python scripts/download_models.py`
(model lands in `pipeline/models/`, git-ignored), then run:

```sh
cd pipeline
uv run python -m handstand.pose_mediapipe --rotate none --limit 3
uv run python -m handstand.pose_mediapipe --rotate auto --limit 3
uv run python -m handstand.pose_mediapipe --rotate auto --num-poses 3
```

## Parquet: one row per (frame, joint)

Long format — every frame contributes exactly 33 rows, one per landmark,
even when nothing was detected. (`--num-poses > 1` keeps this layout and adds
a block per person; see [Several people per frame](#several-people-per-frame---num-poses-25).)

| column | type | meaning |
|---|---|---|
| `frame_idx` | int64 | 0-based index of the decoded frame, in decode order |
| `t_ms` | int64 | frame timestamp in milliseconds from `CAP_PROP_POS_MSEC`, strictly increasing (MediaPipe VIDEO mode bumps a duplicate by 1 ms) |
| `joint` | string | one of the 33 landmark names below |
| `x` | float64 | horizontal pixel in the **display** frame, `0 <= x <= width - 1` |
| `y` | float64 | vertical pixel in the **display** frame, `0 <= y <= height - 1`; `y` grows downwards |
| `z` | float64 | MediaPipe depth estimate, same units/scale as the model outputs; *not* affected by the frame rotation |
| `visibility` | float64 | MediaPipe visibility score, or NaN when the model did not report one |
| `presence` | float64 | MediaPipe presence score, or NaN when the model did not report one |
| `rotated` | bool | was this frame rotated 180° before inference? |
| `detected` | bool | did the model find a pose on this frame? |

Rows are ordered by `frame_idx`, then by landmark index (the order of the
table below).

### Coordinates are always display-frame pixels

"Display orientation" means upright, as a person would view the video: the
runner resolves the container's rotation metadata (these clips are 1024x576
with rotation `-90`, i.e. 576x1024 displayed) and asserts every decoded frame
matches the display size. Keypoints inferred on a rotated frame are mapped
back before being written, so **no rotated coordinates ever leave the
runner** — `x`/`y` can be fed straight to a renderer or compared across the
`none`/`180`/`auto` outputs.

MediaPipe reports normalised coordinates in `[0, 1]`. The runner converts them
with

```
x_px = x_norm * (width - 1)        y_px = y_norm * (height - 1)
```

so the frame edges land on pixel indices `0` and `width - 1` (the same
indexing the rotation formula uses), and maps a 180°-rotated detection back
with

```
x = width - 1 - x_rotated          y = height - 1 - y_rotated
```

(`handstand.rotation.rotate_points` / `inverse_rotate_points`, which also
implement the 90°/270° cases for later use). Because both use `size - 1`, a
point and its mirrored normalised coordinates map to the same display pixel:
the `none` and `auto` outputs of a clip agree to the model's own accuracy
rather than being offset by a pixel.

`z` is depth along the camera axis, which an in-plane 180° rotation does not
touch, so it passes through unchanged.

### No detection

Frames where the model found nothing still get their 33 rows with
`detected = false` and NaN in `x`, `y`, `z`, `visibility`, `presence`. A
consumer can therefore assume `len(frame) == 33` for every frame and filter on
`detected` instead of re-indexing.

## Several people per frame (`--num-poses 2..5`)

Clips often contain the trainer next to the athlete, and a single-pose
detector happily snaps the athlete's joints onto whoever is closest to the
camera. `--num-poses N` asks MediaPipe for up to N people and keeps **all** of
them; the runner never picks the athlete — that is a later step, reading the
`person_idx` column described here.

```
uv run python -m handstand.pose_mediapipe --rotate auto --num-poses 3
```

* Output root: `keypoints/mediapipe_multi/<rotate>/<clip_id>.{parquet,json}`.
  Single-person output is left byte-for-byte alone: same paths, same columns,
  no `person_idx`, and a sidecar with exactly the keys it always had.
* Columns: the single-person schema above **plus** one column.

| column | type | meaning |
|---|---|---|
| `person_idx` | int64 | which person of that frame the row is about, `0 .. P-1`, in the order MediaPipe returned them (most confident first) |

* One **block of 33 rows per person**, ordered by `frame_idx`, then
  `person_idx`, then landmark index. A frame with `P` people therefore has
  `33 * P` rows, and `P` varies from frame to frame.
* A frame with nobody in it keeps **one** block of 33 rows with
  `person_idx = -1`, `detected = false` and NaN coordinates — the same
  placeholder the single-person schema uses, so a frame is never missing from
  the file. `person_idx = -1` therefore means "no person", never "person -1".
* `frame_idx`, `t_ms` and `rotated` describe the *frame*, so they repeat
  across the blocks of that frame.
* Coordinates are display-frame pixels, as above; `z`, `visibility` and
  `presence` are that person's own scores.

```python
import pandas as pd

df = pd.read_parquet(".../keypoints/mediapipe_multi/auto/1a2b3c4d5e6f.parquet")
people = df[df["detected"]]  # drops the person_idx = -1 placeholder blocks
for person_idx, block in people[people["frame_idx"] == 0].groupby("person_idx"):
    print(person_idx, block.set_index("joint").loc["nose", ["x", "y"]])
```

## Joint names

The 33 MediaPipe pose landmarks, snake_case, in model order (this is the
index order used everywhere):

| # | name | # | name | # | name |
|---|---|---|---|---|---|
| 0 | `nose` | 11 | `left_shoulder` | 22 | `right_thumb` |
| 1 | `left_eye_inner` | 12 | `right_shoulder` | 23 | `left_hip` |
| 2 | `left_eye` | 13 | `left_elbow` | 24 | `right_hip` |
| 3 | `left_eye_outer` | 14 | `right_elbow` | 25 | `left_knee` |
| 4 | `right_eye_inner` | 15 | `left_wrist` | 26 | `right_knee` |
| 5 | `right_eye` | 16 | `right_wrist` | 27 | `left_ankle` |
| 6 | `right_eye_outer` | 17 | `left_pinky` | 28 | `right_ankle` |
| 7 | `left_ear` | 18 | `right_pinky` | 29 | `left_heel` |
| 8 | `right_ear` | 19 | `left_index` | 30 | `right_heel` |
| 9 | `mouth_left` | 20 | `right_index` | 31 | `left_foot_index` |
| 10 | `mouth_right` | 21 | `left_thumb` | 32 | `right_foot_index` |

The authoritative list in code is `handstand.pose_mediapipe.JOINT_NAMES`.

## Sidecar JSON

`<clip_id>.json` describes how the parquet was produced:

```json
{
  "clip_id": "1a2b3c4d5e6f",
  "source_file": "WhatsApp Video 2026-07-06 at 7.09.23 PM.mp4",
  "model": "pose_landmarker_full.task",
  "rotate": "auto",
  "display_width": 576,
  "display_height": 1024,
  "frame_count": 244,
  "detected_frame_count": 241,
  "rotated_frame_count": 96,
  "mediapipe_version": "1.0.1",
  "runtime_seconds": 12.345
}
```

| key | meaning |
|---|---|
| `clip_id` | id used in the file names |
| `source_file` | basename of the input video |
| `model` | model file name (not the full path) |
| `rotate` | `none`, `180` or `auto` |
| `display_width`, `display_height` | size of the frames the coordinates refer to |
| `frame_count` | decoded frames (rows / 33) |
| `detected_frame_count` | frames with at least one pose |
| `rotated_frame_count` | frames fed to the model rotated 180°: `frame_count` for mode `180`, `0` for mode `none`, and in `auto` the number of frames the previous frame's judgement switched on |
| `mediapipe_version` | version used for the run |
| `runtime_seconds` | wall-clock seconds for the whole clip, including writing the parquet |

A sidecar under `mediapipe_multi` (i.e. `--num-poses > 1`) adds two keys:

| key | meaning |
|---|---|
| `num_poses` | people per frame the model was asked for (2..5) |
| `frames_by_people` | how many frames had how many people, e.g. `{"0": 3, "1": 120, "2": 98, "3": 23}`; the values add up to `frame_count` |

A single-person sidecar does **not** carry them, so `--num-poses 1` output is
unchanged.

## Rotation modes

| mode | frames fed to the model | `rotated` column |
|---|---|---|
| `none` | as displayed | all false |
| `180` | every frame rotated 180° | all true |
| `auto` | 180° when the *previous* frame's result was inverted — mean y of both wrists greater (lower in the image) than mean y of both ankles, computed in display coordinates; the first frame is never rotated; a frame with no detection keeps the previous decision | true on rotated frames |

`auto` owns two landmarker instances (one that only ever sees upright frames,
one that only ever sees rotated frames) so each MediaPipe VIDEO-mode tracker
observes a consistent orientation.

With `--num-poses > 1` the `auto` judgement is taken from **the person whose
wrists are lowest in the image** of the previous frame — the largest mean wrist
`y` — because that is the one most likely to be the athlete on their hands.
People whose wrists are not both visible cannot be ranked and are skipped; if
no person can be ranked the first person is judged instead, which for a
single-person frame is exactly the rule above. A frame where nobody was
detected at all still keeps the previous decision.

## Reading the file

```python
import pandas as pd

df = pd.read_parquet(".../keypoints/mediapipe/auto/1a2b3c4d5e6f.parquet")
good = df[df["detected"]]
frame0 = good[good["frame_idx"] == 0].set_index("joint")
frame0.loc["left_wrist", ["x", "y"]]   # display-frame pixels
```
