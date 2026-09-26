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
<data_dir>/keypoints/mediapipe_athlete/<rotate>/<clip_id>.parquet  # athlete selection
<data_dir>/keypoints/mediapipe_athlete/<rotate>/<clip_id>.json      # sidecar metadata
```

* `<data_dir>` = `handstand.paths.data_dir()` (default
  `/mnt/sharedOs/handstand-workspace/data`, override with `$HANDSTAND_DATA`).
* `<rotate>` = the rotation mode the file was produced with: `none`, `180` or
  `auto`. The three modes are independent outputs of the same clip, never
  merged — compare them side by side.
* `mediapipe_multi` only exists for `--num-poses > 1`. A multi-person run
  writes to its own root, so it can never overwrite the single-person
  parquets the other stages read.
* `mediapipe_athlete` is the athlete selection
  ([below](#athlete-selection--trainer-contact)) — the single-person schema
  again, holding only the athlete's body.
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

# the recommended multi-person setting, then the athlete selection on top of it
uv run python -m handstand.pose_mediapipe --rotate auto --num-poses 3 \
    --running-mode image --min-detection 0.2 --min-presence 0.2
uv run python -m handstand.athlete --rotate auto
uv run python -m handstand.overlay --clip 6508f9b355bd --source athlete
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

## Detector settings (`--running-mode`, `--min-*`)

MediaPipe's own default is VIDEO mode (the detector only re-runs when tracking
is lost) with every score threshold at `0.5`. That is the default here too, so a
run without these flags is byte-identical to what the runner has always written.

| flag | default | meaning |
|---|---|---|
| `--running-mode {video,image}` | `video` | `video` tracks between frames and needs timestamps; `image` runs the detector on **every** frame and needs none |
| `--min-detection` | `0.5` | `min_pose_detection_confidence`: a detection below this is dropped |
| `--min-presence` | `0.5` | `min_pose_presence_confidence`: a landmark below this is not "present" |
| `--min-tracking` | `0.5` | `min_tracking_confidence`: how well a landmark has to follow the previous frame to be associated with it |

VIDEO mode with `0.5` reports **one** person in 98–100 % of frames even when a
trainer is standing right there, because the second detection is suppressed.
IMAGE mode with `--min-detection 0.2 --min-presence 0.2` is therefore the
recommended multi-person setting: the detector sees the trainer on every frame.
Going below `0.2` (e.g. `0.05`) only adds phantom people.

The four settings are recorded in the sidecar whenever they differ from the
default, so a later reader can reproduce the run (see
[Sidecar JSON](#sidecar-json)).

## Athlete selection + trainer contact

`handstand.athlete` reads `keypoints/mediapipe_multi/<rotate>/<clip_id>.parquet`
and writes `keypoints/mediapipe_athlete/<rotate>/<clip_id>.parquet`: the
**single-person schema** above, holding the athlete's body only, so every later
stage keeps working unchanged.

```sh
cd pipeline
uv run python -m handstand.athlete --rotate auto
uv run python -m handstand.athlete --rotate auto --clips 6508f9b355bd
```

| column | type | meaning |
|---|---|---|
| `athlete_score` | float64 | the winner's score, 0..1 (the weights below add up to 1); NaN when the frame is attributed to nobody |
| `n_people` | int64 | how many people were in the frame **after** dedup |
| `trainer_contact` | bool | the trainer overlaps or touches the athlete: the keypoints are the athlete's, but the frame must not be scored |
| `contact_reason` | string | which rule set that flag: `""`, `"box_iou"`, `"mixed_skeleton"` or `"bone_length"` — the first one that fired. Downstream code filters on `trainer_contact`, this is for explaining a flag |

The rules, applied to each frame in this order (the constants are
`handstand.athlete`'s module-level ones; see its docstring for the code):

1. **Deduplicate.** Two detections of one body are one person: hip midpoints
   within `0.15` body lengths *and* a median joint-to-joint distance below `0.1`
   body lengths (over at least 4 joints both reported). The more visible one
   survives. A *body length* is the person's own shoulder-midpoint to
   ankle-midpoint span, so every distance below is scale-free.
2. **Score and choose.** Each remaining person is scored 0..1 from
   * **inversion (0.4)** — mean wrist y greater than mean ankle y, i.e. on their
     hands;
   * **support (0.2)** — wrists near the lowest wrist/foot point in the frame
     (the mat), measured in that person's own shoulder-to-ankle lengths;
   * **continuity (0.3)** — hip midpoint close to the previously chosen athlete,
     in that person's own body lengths;
   * **visibility (0.1)** — mean visibility of the 12 main joints (shoulders,
     elbows, wrists, hips, knees, ankles).

   The clip is swept **forward and backward**, each sweep measuring continuity
   against the athlete *that* sweep chose in its previous frame, and the
   higher-scoring sweep wins per frame. That is what covers the kick-up before
   the first inverted frame, where nothing looks like a handstand yet.
3. **Flag contact.** The chosen person is `trainer_contact`, with the reason in
   `contact_reason`, when any of these fires:

   | reason | what it means |
   |---|---|
   | `box_iou` | another person's bounding box (of its visible joints, visibility ≥ 0.5) overlaps the athlete's by IoU > `0.3` |
   | `mixed_skeleton` | one of the athlete's limb joints is closer to the other person's joints than to the athlete's own centre line (their feet to their head) — the two skeletons overlap so much that the model stitched them into one |
   | `bone_length` | one of the athlete's own bones is the wrong length: more than `BONE_LENGTH_TOLERANCE` (0.35) off the median of the same bone over the clip, or off the same bone on the other side of the body |

   The first two need a *second detection*, so contamination inside a single
   reported skeleton — the athlete's upper body with the trainer's foot on the
   floor, which is what MediaPipe reports when the trainer stands right behind
   them — slips past them. `bone_length` is what catches that: the bones are
   measured over the five pairs of the athlete's own (upper arm, forearm,
   thigh, shin, and the side of the torso, left and right), each pair's median
   over the frames where both its joints were seen at visibility ≥ 0.5, and a
   bone outside the ±35 % band is somebody else's limb.

   The left/right half of that rule only runs for a clip in which MediaPipe
   reported 2+ people in at least one frame (`ASYMMETRY_MIN_PEOPLE`). An athlete
   straddling their legs in a handstand has one leg pointing at the camera and
   one away, and MediaPipe reports that as a 35–50 % left/right difference for
   whole stretches of a clip with nobody else in it — those frames are the
   athlete's own and must stay scorable. The per-clip median needs no second
   body, so it always runs.

   What no rule can catch: contamination that *is* the clip's norm. In
   6508f9b355bd the trainer's leg is stitched onto the athlete's left hip in
   every frame the model reports one body, so the clip's own median is the
   contaminated length and only the worst frames stand out (16 of 244).
4. **Give up honestly.** A frame whose best score is below `0.35`, or where the
   two best candidates are within `0.05` of each other, is written as *not
   detected*: 33 NaN rows with `detected = false`, exactly as a frame the model
   missed. `rotated` and `t_ms` still describe the frame, and `n_people` still
   says how many people were in it.

```python
import pandas as pd

df = pd.read_parquet(".../keypoints/mediapipe_athlete/auto/1a2b3c4d5e6f.parquet")
scorable = df[df["detected"] & ~df["trainer_contact"]]
# why a frame was dropped:
print(df[df["trainer_contact"]]["contact_reason"].value_counts())
```

### Watching one

`handstand.overlay --source athlete` draws these keypoints and marks every
flagged frame with a red border and the caption `trainer contact`, so the frames
that must not be scored are obvious while watching:

```sh
cd pipeline
uv run python -m handstand.overlay --clip 6508f9b355bd --source athlete
# -> <data_dir>/overlays/6508f9b355bd_athlete_auto.mp4
```

The source is part of the file name (only for a non-default source), so the
plain `--source mediapipe` render of the same rotation mode is left alone.

## Trainer report

`handstand.trainer_report` turns the two frame-level columns above —
`n_people` and `trainer_contact` — into the dataset-level question: **how much of
the data is usable as it stands, how much has to go through the athlete-selection
path, and how much is lost to trainer contact**. It reads every clip in
`catalogue.csv` and writes two files:

```
<data_dir>/reports/trainer_report.csv   # one row per catalogue clip
<data_dir>/reports/trainer_report.md    # the summary, in markdown
```

Neither is committed (both are under the git-ignored `data/`): they are a
rendering of the parquets and are rebuilt by re-running the report.

```sh
cd pipeline
uv run python -m handstand.trainer_report                 # report on what exists
uv run python -m handstand.trainer_report --generate      # generate what is missing, then report
uv run python -m handstand.trainer_report --limit 5       # a first look
```

`--generate` is separate because it is the expensive half (a multi-person run
over the whole dataset takes tens of minutes) and because the report has to be
re-runnable on its own. It runs exactly the two commands above over every
catalogue clip, one clip at a time, skipping the clips whose parquet already
exists, so an interrupted run is resumed by running it again. A clip whose
keypoints were made with *other* detector settings is re-run rather than skipped:
the default VIDEO-mode run reports one person per frame even with the trainer
standing there, and would measure that clip as "no trainer, ever".

| column | meaning |
|---|---|
| `frames`, `fps` | decoded frames (rows / 33), and the frame rate as the **median gap between the `t_ms` timestamps** — the clips are variable frame rate, so `1/fps_avg` is not it |
| `second_person_frames`, `pct_frames_second_person` | frames with `n_people >= 2` (after dedup), and their share of the clip |
| `second_person_seconds` | how long a second person was there **in total**, timed from the timestamps |
| `trainer_present` | `second_person_seconds >= 0.5`. The threshold is what ignores a one-frame phantom detection, which happens in almost every clip; a trainer who walks past twice counts as one |
| `contact_frames`, `pct_trainer_contact` | frames flagged `trainer_contact`, and their share of the clip |
| `longest_contact_run_s` | the longest **unbroken** stretch of contact, in seconds — a quick spotter and a hands-on session have the same percentage and very different runs |
| `athlete_frames`, `pct_athlete`, `dropped_frames`, `pct_dropped` | frames an athlete was attributed to, and the frames the selection left unattributed |
| `contact_box_iou_frames`, `contact_mixed_skeleton_frames`, `contact_bone_length_frames` | the flagged frames per rule, so the report says *how* the trainer is in a clip |
| `generate_seconds` | the runtimes the two stages recorded in their sidecars, added up |
| `notes` | the catalogue's own `notes` column, carried through untouched, so the report can be lined up against the hand annotation of the same clips |
| `error` | why a clip has no numbers (it failed to generate, or its parquet is unreadable). Every catalogue clip gets a row, and a failure never stops the run |

The markdown summary answers the dataset question directly: how many clips have
a trainer in them, the frame-level shares (second person, contact, dropped,
scorable), the ten most affected clips, and a histogram of `pct_trainer_contact`
per clip in 10 % buckets.

```sh
cd pipeline
uv run python -m handstand.trainer_report | head -40
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

Any run whose detector settings differ from MediaPipe's own defaults (see
[Detector settings](#detector-settings---running-mode---min-)) adds four more:

| key | meaning |
|---|---|
| `running_mode` | `video` or `image` |
| `min_detection` | `--min-detection` |
| `min_presence` | `--min-presence` |
| `min_tracking` | `--min-tracking` |

A run on the defaults records none of them, so a historical sidecar stays
byte-identical.

A sidecar under `mediapipe_athlete` (written by `handstand.athlete`) describes
the selection instead:

| key | meaning |
|---|---|
| `clip_id`, `rotate` | as above |
| `source_parquet` | the multi-person parquet it read, e.g. `mediapipe_multi/auto/1a2b3c4d5e6f.parquet` |
| `frame_count` | frames in the clip (rows / 33) |
| `athlete_frame_count` | frames a person was attributed to |
| `contact_frame_count` | of those, the frames flagged as trainer contact |
| `dropped_frame_count` | frames written as not detected (`frame_count - athlete_frame_count`) |
| `frames_by_people` | frames per number of people, **after** dedup |
| `contact_frames_by_reason` | flagged frames per rule, e.g. `{"box_iou": 20, "mixed_skeleton": 1, "bone_length": 16}`; the three values add up to `contact_frame_count` |
| `mean_athlete_score` | mean `athlete_score` over the attributed frames, or `null` when none was |
| `runtime_seconds` | wall-clock seconds for the whole clip |

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
