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
<data_dir>/keypoints/vision_raw/<rotate>/<clip_id>.csv             # Apple Vision, from the Mac
<data_dir>/keypoints/vision_raw/<rotate>/<clip_id>.json             # its run manifest
<data_dir>/keypoints/vision_multi/<rotate>/<clip_id>.parquet       # every person Vision saw
<data_dir>/keypoints/vision_multi/<rotate>/<clip_id>.json           # sidecar metadata
<data_dir>/keypoints/vision/<rotate>/<clip_id>.parquet             # the lowest-wrist person
<data_dir>/keypoints/vision/<rotate>/<clip_id>.json                 # sidecar metadata
<data_dir>/keypoints/vision_athlete/<rotate>/<clip_id>.parquet     # athlete selection
<data_dir>/keypoints/vision_athlete/<rotate>/<clip_id>.json         # sidecar metadata
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
* `vision*` is [Apple Vision](#apple-vision), whose model is macOS/iOS only:
  the CSV is produced on the Mac mini (`tools/mac/run_vision.sh`) and imported
  here, into the same two schemas, so a Vision run and a MediaPipe run of the
  same clip and mode differ only in which directory they are in.
* `vision_athlete` is the athlete selection of the Vision keypoints —
  `handstand.athlete --source vision`, with the same rules, the same four extra
  columns and the same 33 rows per frame as `mediapipe_athlete`
  ([below](#athlete-selection--trainer-contact)).
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

# the same selection over the Apple Vision keypoints (chainlink #16's bake-off)
uv run python -m handstand.athlete --source vision --rotate auto
uv run python -m handstand.overlay --clip 6508f9b355bd --source vision_athlete
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
uv run python -m handstand.athlete --source vision --rotate auto   # Apple Vision
```

`--source {mediapipe,vision}` (default `mediapipe`, whose output is unchanged)
picks the pair of directories to work on and nothing else: the same rules, the
same constants, the same four extra columns. `vision` reads `vision_multi/` and
writes `vision_athlete/`, so the bake-off compares the two models *after* the
same selection rather than one of them before it. What a Vision block looks like
to these rules is under [Apple Vision](#apple-vision).

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
   | `bone_length` | one of the athlete's own bones is the wrong length: more than `BONE_LENGTH_TOLERANCE` (0.35) off the 90th percentile of the same bone over the clip (`BONE_LENGTH_PERCENTILE`), or off the same bone on the other side of the body |

   The first two need a *second detection*, so contamination inside a single
   reported skeleton — the athlete's upper body with the trainer's foot on the
   floor, which is what MediaPipe reports when the trainer stands right behind
   them — slips past them. `bone_length` is what catches that: the bones are
   measured over the five pairs of the athlete's own (upper arm, forearm,
   thigh, shin, and the side of the torso, left and right), and each is compared
   against the **90th percentile** of its own length over the frames where both
   its joints were seen at visibility ≥ 0.5 — the length the bone is seen at
   when it lies in the picture plane, i.e. at full length. A bone outside the
   ±35 % band is somebody else's limb. The percentile, rather than the median,
   is what leaves room for a bone that is foreshortened in most of a clip while
   still being measured against the frames in which it is not.

   **That half only ever fires on a bone that is too _long_.** Foreshortening can
   only make a projected bone *shorter*: a leg swung out of the camera plane in
   a split, a stag or a straddle comes out shorter and the model is right.
   Contamination by another body goes the other way — the limb has been finished
   at the trainer's foot, so it comes out longer. So against the clip's own
   history a short bone is the athlete's own and is never flagged, and a
   two-sided version of that half was throwing away the frames of a solo split
   clip for nothing: in `a72f0c886e1c` and `a9153273dab1` MediaPipe reported one
   person in every frame, and 85 % and 81 % of them were flagged, none of them a
   trainer. Both are now 0 %.

   The left/right half is **two-sided**, and has to be: a trainer standing
   *behind* the athlete shortens a bone just as surely as one standing beside
   them lengthens it (in `5f71d966c49a` the right thigh and shin come out at
   roughly half the left ones in nearly every frame, which is 71 % of the clip
   still flagged). What that half cannot do is tell a straddle from a swapped
   leg — to it a leg pointing at the camera and a leg that was never seen whole
   look the same — which is why it only runs for a clip in which MediaPipe
   reported 2+ people in at least one frame (`ASYMMETRY_MIN_PEOPLE`). The
   per-clip full length needs no second body, so it always runs.

   What no rule can catch: contamination that *is* the clip's norm. In
   6508f9b355bd the trainer's leg is stitched onto the athlete's left hip in
   every frame the model reports one body, so the clip's own reference is the
   contaminated length: the left leg measures 158.9 px against the right one's
   105.5 in frame 121, and very nearly that in almost every other frame, so
   there is no outlier left for the per-clip half to find. The left/right half
   still catches 14 of the 244 frames, and the frames the model does report two
   people in are caught by `box_iou` (20 of 244) and `mixed_skeleton` (1).

   Net over all 180 clips: `bone_length` falls from 22.3 % of frames to 7.3 %,
   and from 17.6 % to 2.7 % in the 107 clips with no trainer detected, with no
   clip in either group flagged *more* than before.
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

# the same selection over the Apple Vision keypoints
uv run python -m handstand.overlay --clip 6508f9b355bd --source vision_athlete
# -> <data_dir>/overlays/6508f9b355bd_vision_athlete_auto.mp4
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
per clip in 10 % buckets. Its runtime section keeps the two halves of a run
apart — what the keypoints cost (as recorded in the sidecars, however many
batches produced them), what *this* run spent in `--generate`, and what reading
the parquets took, which is under a second for the whole dataset.

```sh
cd pipeline
uv run python -m handstand.trainer_report | head -40
```

## Processed keypoints

`handstand.postprocess` turns the per-frame guesses of the athlete selection into
the trajectory every later stage measures: clean, smooth and in the athlete's own
units. It reads the schema above (`keypoints/mediapipe_athlete/`, or
`--source vision` for `vision_athlete/`) and writes the **same long schema with
the same rows**, so `handstand.overlay` draws it unchanged, plus four columns:

```
<data_dir>/processed/<source>/<clip_id>.parquet   # input schema + 4 columns
<data_dir>/processed/<source>/<clip_id>.json      # what was done to the clip
```

```sh
cd pipeline
uv run python -m handstand.postprocess --all                          # mediapipe, auto
uv run python -m handstand.postprocess --source vision --all
uv run python -m handstand.postprocess --clips 6508f9b355bd --overwrite
```

Note that unlike `keypoints/` there is no rotation-mode directory: one clip has
one processed trajectory, the recommended `auto` mode, and re-running a clip in
another mode needs `--overwrite`.

| column | type | meaning |
|---|---|---|
| `x`, `y` | float64 | the **processed** position in display pixels, NaN wherever there is none |
| `x_raw`, `y_raw` | float64 | what the model said, before any of this: the input's own `x`/`y`, NaN where the input had none |
| `valid` | bool | does this sample have a position at all — false exactly where `x`/`y` are NaN, so one filter drops every unusable sample |
| `filled` | bool | was this position interpolated across a short gap rather than measured. A filled sample is valid, but it is a bridge, not an observation |

Every other column (`z`, `visibility`, `presence`, `rotated`, `detected`,
`athlete_score`, `n_people`, `trainer_contact`, `contact_reason`) is passed
through untouched.

### What is done, in order

1. **Gating.** Every joint of a frame is invalid when the frame is
   `detected = false` or `trainer_contact = true`, and so is any joint whose
   `visibility` is below `0.5` — or is NaN, which is what the renderer refuses to
   draw as well. A contact frame's keypoints are two bodies' keypoints stitched
   together; carrying them into a trajectory is how a centre of mass ends up on
   the wrong person.
2. **Body length `L`.** Torso (shoulder midpoint to hip midpoint) + thigh + shin,
   each the **90th percentile** of that span over the clip's valid frames, in
   display pixels. The percentile makes it a scale rather than a pose: a split, a
   stag or a leg out of the picture plane can only make a projected span *shorter*,
   so the frames where the limb lies in the picture plane set the yardstick (the
   same idea as the bone-length percentile of the athlete selection). The thigh
   and the shin are measured on each leg and the **longer** leg of a frame wins.
   A clip with fewer than 10 valid frames for a part has no `L`, and is written
   with every joint invalid and `usable: false` in its sidecar.
3. **Outliers.** A joint that covers more than `8` body lengths in a second
   between two consecutive valid samples is invalid in the later of the two: the
   model lost it, it did not travel that fast. The earlier sample stays the
   reference, so a single spike is dropped and the next sample is judged against
   the last good one rather than against the spike.
4. **Gap fill.** A run of invalid samples is interpolated linearly **in time**
   between the valid samples on either side of it when those two are at most
   `0.2 s` apart, and left NaN when they are further apart. Filled samples are
   marked as such, so a later stage can refuse the bridges.
5. **Smoothing.** A One-Euro filter per joint coordinate (`min_cutoff = 1.0`,
   `beta = 0.3`, `d_cutoff = 1.0`), with the real `dt` from `t_ms` because these
   clips are variable frame rate. The filter runs on positions **divided by
   `L`**, so a cutoff means the same thing for a small athlete and a large one,
   and it is restarted after every invalid run so it never smooths across a gap
   it knows nothing about.

### The body frame

`handstand.bodyframe` is where a later stage turns a processed position into the
athlete's own coordinates: the origin is the **wrist midpoint**, `u` runs to the
right, `v` runs **up** (display `y` with the sign flipped, because the athlete is
upside down), and both are divided by `L`.

```python
from handstand.bodyframe import body_frame_points, to_body_frame

u, v = to_body_frame(x, y, wrist_mid_x, wrist_mid_y, L)         # one point
uv = body_frame_points(points, (wrist_mid_x, wrist_mid_y), L)  # (..., 2) points
```

`handstand.prelabel` already measures model disagreement in the same unit (body
lengths), so the three agree.

### The processed sidecar

| key | meaning |
|---|---|
| `clip_id`, `source`, `rotate` | as the athlete selection's sidecar |
| `source_parquet` | the athlete parquet it read, e.g. `keypoints/mediapipe_athlete/auto/1a2b3c4d5e6f.parquet` |
| `frame_count`, `joint_count` | frames in the clip and joints in its schema (33 for both models) |
| `usable`, `unusable_reason` | whether the clip has a body length, and why not when it has none |
| `body_length_px` | `L` in display pixels, or `null` |
| `body_length_parts_px` | the torso / thigh / shin percentiles it is the sum of |
| `body_length_frames` | how many valid frames each part was measured on |
| `valid_sample_count`, `measured_sample_count`, `filled_sample_count`, `unfilled_sample_count` | the four kinds the rows fall into: a position from the model, a bridge, a position, and none. `valid == measured + filled`, and the four add up to `total_sample_count` |
| `gated_out_sample_count`, `outlier_sample_count` | the samples lost to the gate and to the speed rule; the two add up to `unfilled_sample_count`, because a sample a rule removed that the gap fill then bridged is a bridge, not a loss |
| `total_sample_count`, `pct_valid_samples` | rows in the parquet, and the share of them with a position |
| `tracked_valid_sample_count`, `tracked_total_sample_count`, `pct_valid_tracked_samples` | the same over the 15 joints of the shared schema only (a face or a finger is not evidence about a hold) |
| `hold_like_frame_count`, `pct_valid_tracked_samples_hold_like` | frames where the wrists are below the ankles (the `auto` rotation rule, on the gated coordinates), and the valid share of their tracked samples |
| `jitter_l_raw`, `jitter_l_processed`, `pct_jitter_reduction` | median frame-to-frame displacement of the two wrists and two ankles in body lengths, before and after, and the fraction removed |
| `parameters` | the thresholds the run used, so a later run can be compared against this one |
| `runtime_seconds` | wall-clock seconds for the clip |

## Phases

`handstand.phases` labels **every frame** of a clip with the phase it belongs to
and numbers the holds, so scoring reads only the `hold` frames and hand steps
(#71) and faults (#32) are events *inside* one. It reads the processed parquets
above and the `body_length_px` of their sidecars, and writes:

```
<data_dir>/phases/<source>/<clip_id>.parquet   # one row per frame
<data_dir>/phases/<source>/segments.csv        # one row per phase run
```

```sh
cd pipeline
uv run python -m handstand.phases --all                     # every processed clip
uv run python -m handstand.phases --source vision --all
uv run python -m handstand.phases --clips 057c9e6c96af --render 057c9e6c96af
```

`--render` writes `<data_dir>/overlays/<clip_id>_phases.mp4` with the phase and
the hold number on every frame, which is the only way to see whether the
boundaries landed where they should. A clip with no body length is written with
every frame `unknown`, as the processed stage wrote it unusable.

The parquet is **one row per frame**, not per (frame, joint):

| column | type | meaning |
|---|---|---|
| `frame_idx`, `t_ms` | int64 | the frame and its timestamp, as in the keypoint schema |
| `phase` | str | `pre`, `kickup`, `hold`, `exit`, `post` or `unknown` |
| `hold_id` | int64 | the hold this frame is inside, numbered from 0 per clip, `-1` outside one |
| `known` | bool | the frame has the wrist, ankle and hip midpoints a phase is read from, and no trainer contact |
| `unknown_reason` | str | why not: `trainer_contact`, `no_visible_wrist`, `no_visible_ankle`, `no_visible_hip`; empty when known |
| `v_ankle_mid`, `u_ankle_mid` | float64 | the ankle midpoint in the body frame (origin at the wrist midpoint, `u` right, `v` up, in body lengths) |
| `v_hip_mid` | float64 | the hip midpoint in the body frame, the same units |
| `body_angle_deg` | float64 | the angle of the wrist→ankle vector from straight up, signed by which way it leans |
| `inverted` | bool | `v_ankle_mid > 0.6`: a body standing on its hands, with a margin |
| `hands_low` | bool | the hip midpoint is at least `0.1` body lengths above the wrist midpoint: the hands are the support |
| `wrist_speed_l_per_s` | float64 | the fastest visible wrist's speed over the previous `0.2 s`, NaN where there is no earlier frame to measure against |
| `wrist_step_l` | float64 | the largest displacement of a visible wrist over that same `0.2 s` window |
| `hands_down` | bool | `hands_low` and the hands are not moving: no visible wrist faster than `0.3` body lengths per second. A wrist whose speed could not be measured is not evidence of movement, so a frame just after a gap still counts |
| `hand_step` | bool | a wrist moved more than `0.1` body lengths over that window: the hands went down somewhere else, and this is what ends a hold |
| `legs_velocity_l_per_s` | float64 | the ankle midpoint's vertical velocity over a centred `0.1 s` window |
| `legs_rising`, `legs_falling` | bool | that velocity above or below `0.1` body lengths per second: the feet going away from the floor, or coming back to it |

Every boolean is `False` on a frame with `known = false`, so `known` is the
column to filter on.

`segments.csv` is the same answer as intervals, one row per run of frames that
share a phase, for every clip of the source:

| column | meaning |
|---|---|
| `clip_id` | the clip the row is about |
| `phase` | as the parquet |
| `hold_id` | the hold, or `-1` outside one. Two holds separated by a hand step are two rows |
| `start_frame`, `end_frame` | the first and last frame of the run, inclusive |
| `start_ms`, `end_ms` | the same in milliseconds, from `t_ms` |
| `duration_s` | `(end_ms - start_ms) / 1000`, so the four time columns of a row always agree |

The file is per source, not per run: a run over a few clips replaces its own rows
and keeps the rest, so `--all` is what writes the whole dataset.

### The rules

A hold is `inverted` **and** `|body_angle| <= 35°` **and** `hands_down` **and** no
hand step, sustained for at least `0.3 s`. A stretch of frames that fails the
condition for less than `0.3 s` does not end a hold — a trainer walking in front
of the camera, or a held handstand swaying a thousandth of a body length under
the margin — but a hand step always does. The phase itself comes from a state
machine over that: `pre` before the hands are down, `kickup` while they are down
and the legs are rising, `hold` while a run qualifies, `exit` from the frame a
hold stops qualifying until the feet land, and `post` once the body is no longer
above its hands — until the hands are down and the legs rising again, which is a
second attempt.

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

A sidecar under `mediapipe_athlete` (written by `handstand.athlete`) — or under
`vision_athlete`, which is the same thing for `--source vision` — describes the
selection instead:

| key | meaning |
|---|---|
| `clip_id`, `rotate` | as above |
| `source_parquet` | the multi-person parquet it read, e.g. `mediapipe_multi/auto/1a2b3c4d5e6f.parquet` (or `vision_multi/...` for `--source vision`) |
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

## Apple Vision

`swift/VisionPose` is the second pose model, Apple's `VNDetectHumanBodyPoseRequest`
— no dependency to install, but macOS/iOS only, so it runs on the Mac mini and
everything after it runs here. It exists so the two models can be compared on the
same clips, frame by frame (the bake-off, chainlink #16), which only means
anything if they write the same schema. They do, with one difference of joint set
and two of column content, both below.

```sh
# on this machine: copy the clips across, build, run, copy the CSVs back
tools/mac/run_vision.sh --rotate none --rotate auto --clips 6508f9b355bd
tools/mac/run_vision.sh --all --rotate auto

# then import them (Linux)
cd pipeline
uv run python -m handstand.vision_import --rotate auto
uv run python -m handstand.vision_import --rotate none --rotate auto --clips 6508f9b355bd
```

`run_vision.sh` rsyncs the videos to the Mac mini
(`--ignore-existing`, so a second run transfers no video), copies the package,
builds it release, runs it, and rsyncs the CSVs back. Each run also writes a
**run manifest** (`<clip_id>.json` next to the CSV) with the things only the Mac
knows — the macOS version, the Vision revision, the runtime — which
`handstand.vision_import` folds into the parquet's sidecar.

### The CSV

One row per (frame, person, joint), `x`/`y` already in display pixels:

| column | type | meaning |
|---|---|---|
| `frame_idx` | int | 0-based index of the decoded frame, in decode order |
| `t_ms` | int | frame presentation timestamp in ms, rounded |
| `person_idx` | int | which person of the frame, `0 .. P-1` in the order Vision returned them; `-1` on a frame with nobody |
| `joint` | string | one of the 19 [Vision joints](#vision-joint-names) |
| `x`, `y` | float | display-frame pixels, 4 decimals; **empty** on a `person_idx = -1` row |
| `confidence` | float | Vision's per-joint confidence, 6 decimals; empty on a `person_idx = -1` row |
| `rotated` | bool | was this frame fed to the model 180° turned? |
| `detected` | bool | did Vision find a person on this frame? |

A frame nobody was detected in still gets one row per joint, with
`person_idx = -1`, empty `x`/`y`/`confidence` and `detected = false` — the same
placeholder the MediaPipe schema uses, so a frame is never missing.

### Coordinates

Identical conventions to MediaPipe, and that is the point: `x_px = x * (width - 1)`,
`y_px = (1 - y) * (height - 1)`. The only difference is where the normalized
origin is — **Vision puts it at the bottom left**, so its `y` is flipped where
MediaPipe's is not. The `size - 1` indexing and the 180° map-back
`x = width - 1 - x_rotated` are the same formulas, so `--rotate 180` and
`--rotate auto` produce display pixels in both runners.

Frames are decoded in display orientation too: the Swift runner applies the
track's `preferredTransform` itself, so a 1024x576 clip with rotation `-90` is
decoded as 576x1024 and the frame counts match the MediaPipe parquets frame for
frame (it warns on stderr if they ever do not, and `run_vision.sh` passes
MediaPipe's own count as the reference).

### Rotation modes

Identical to [MediaPipe's](#rotation-modes), including the rule: `auto` turns the
next frame when the **previous** frame's body had its wrists below its ankles,
judged from the person whose wrists are lowest, with the first frame never
turned and a frame with nobody in it keeping the previous decision. Both
implementations have to agree or the two runs are not comparable at all.

### Vision joint names

19 joints, in the runner's joint order (which is the CSV row order and, after
import, the row order inside a person's block):

| # | name | # | name | # | name |
|---|---|---|---|---|---|
| 0 | `nose` | 7 | `neck` † | 14 | `right_hip` |
| 1 | `left_eye` | 8 | `left_elbow` | 15 | `left_knee` |
| 2 | `right_eye` | 9 | `right_elbow` | 16 | `right_knee` |
| 3 | `left_ear` | 10 | `left_wrist` | 17 | `left_ankle` |
| 4 | `right_ear` | 11 | `right_wrist` | 18 | `right_ankle` |
| 5 | `left_shoulder` | 12 | `root` † | | |
| 6 | `right_shoulder` | 13 | `left_hip` | | |

The authoritative list in code is `VisionJoint.columnNames` (Swift) and
`handstand.vision_import.JOINT_NAMES` (Python).

**What is different from MediaPipe's 33:**

* **17 shared joints, spelled MediaPipe's way** — `left_shoulder`, `right_ankle`
  and so on, so a comparison is a join on the joint name.
* **2 Vision-only joints, `neck` and `root`** († above) — Vision's shoulder and
  hip midpoints. MediaPipe spells those out with several landmarks each and has
  no single name for them, so a join drops them and a stage that indexes joints
  by MediaPipe's names never sees them.
* **16 MediaPipe joints are missing**, and nothing fills them in: the four
  eye-inner/eye-outer points, `mouth_left`/`mouth_right`, and — the ones that
  matter for a handstand — **`left_pinky`/`right_pinky`, `left_index`/
  `right_index`, `left_thumb`/`right_thumb` and `left_heel`/`right_heel`/
  `left_foot_index`/`right_foot_index`**. Vision stops at the wrist and the
  ankle: there is no hand or foot detail, so the wrist and the ankle *are* the
  extremities. Any analysis that used the foot index to measure how flat a
  handstand is has to stop at the ankle.

### The parquets

`handstand.vision_import` writes the two files above, with the same columns, the
same order, the same dtypes and the same ordering rules as the MediaPipe runner:

* `vision_multi/<rotate>/<clip_id>.parquet` — every person Vision reported
  (`person_idx` `0 .. P-1`, one 19-row block per person, the `person_idx = -1`
  placeholder for a frame with nobody). Vision returns every person it finds, so
  this is written unconditionally rather than behind a `--num-poses` flag.
* `vision/<rotate>/<clip_id>.parquet` — the **single-person schema**, no
  `person_idx`, holding the person whose **wrists are lowest** in each frame. The
  same rule `pose_mediapipe.lowest_wrist_pose` applies to a MediaPipe run, so
  "which body is this frame about" is decided identically for both. Choosing the
  athlete out of several people is still `handstand.athlete`'s job — which is
  what `handstand.athlete --source vision` does, writing `vision_athlete/`.

Three columns do not survive the crossing:

| column | MediaPipe | Vision |
|---|---|---|
| `visibility` | visibility score | the model's per-joint **confidence** |
| `presence` | presence score | **NaN** — Vision reports no separate presence |
| `z` | depth estimate | **NaN** — Vision's body-pose model reports no depth |

Everything else — `frame_idx`, `t_ms`, `joint`, `x`, `y`, `rotated`, `detected`
(and `person_idx` in the multi file) — is identical.

### The athlete selection over Vision keypoints

`handstand.athlete --source vision` runs the rules of
[Athlete selection + trainer contact](#athlete-selection--trainer-contact)
unchanged over `vision_multi/` and writes `vision_athlete/`: the same four extra
columns, and the same **33 rows per frame in MediaPipe's joint order**, so the
two athlete files of one clip join on `(frame_idx, joint)` row for row. The 16
landmarks Vision has no name for are NaN in that file, which is the schema's own
way of saying "this model did not report this joint"; Vision's own `neck` and
`root` are not MediaPipe rows and are left out entirely.

Every rule therefore either reads joints both models have, or skips the ones that
are not there:

* the visibility term (the 12 main joints) and all ten measured bones —
  upper arm, forearm, thigh, shin and the side of the torso — are shoulders,
  elbows, wrists, hips, knees and ankles, all of which Vision reports;
* the mat line and the centre line's foot are the lowest wrist or **ankle**
  rather than of a toe, and the mixed-skeleton check asks about the eight limb
  joints a Vision skeleton has;
* the head end of the centre line is the mean of the face joints Vision did
  report (no mouth), falling back to the shoulders when it reported none;
* a bone one of whose ends is missing is not measured, and a frame with nothing
  to place a centre line on is simply not checked for a stitch.

One number does not carry over, and it is worth knowing before reading a
comparison: a Vision parquet's `visibility` is a **confidence** on a lower scale
than MediaPipe's visibility, while `CONTACT_MIN_VISIBILITY` is MediaPipe's `0.5`.
A Vision bone is therefore only measurable when both of its ends clear `0.5` and
is skipped otherwise, so `bone_length` can see less of a Vision clip than of a
MediaPipe one. The *selection* does not read that threshold as a filter (only as
the mean of the 12 main scores, a score), so which athlete is chosen is decided
the same way for both models whatever it is.

### The Vision sidecar

```json
{
  "clip_id": "1a2b3c4d5e6f",
  "source_file": "WhatsApp Video 2026-07-06 at 7.09.23 PM.mp4",
  "source_csv": "1a2b3c4d5e6f.csv",
  "model": "apple-vision-body-pose",
  "vision_revision": 1,
  "macos_version": "26.7.0",
  "rotate": "auto",
  "display_width": 576,
  "display_height": 1024,
  "frame_count": 244,
  "detected_frame_count": 200,
  "rotated_frame_count": 239,
  "frames_by_people": {"0": 44, "1": 200},
  "runtime_seconds": 2.679
}
```

`model`, `macos_version`, `vision_revision`, `runtime_seconds`, `display_width`
and `display_height` come from the Mac's run manifest, not from the CSV, which is
nothing but keypoints. The sidecar next to the `vision/` (single-person) parquet
adds two more:

| key | meaning |
|---|---|
| `selection` | always `"lowest_wrist"` — which person the single-person table kept |
| `selected_frame_count` | frames a person was attributed to |

## Reading the file

```python
import pandas as pd

df = pd.read_parquet(".../keypoints/mediapipe/auto/1a2b3c4d5e6f.parquet")
good = df[df["detected"]]
frame0 = good[good["frame_idx"] == 0].set_index("joint")
frame0.loc["left_wrist", ["x", "y"]]   # display-frame pixels
```

The two models side by side — same columns, so it is a join on
`(frame_idx, joint)`:

```python
import numpy as np, pandas as pd

mp = pd.read_parquet(".../keypoints/mediapipe/auto/1a2b3c4d5e6f.parquet")
vi = pd.read_parquet(".../keypoints/vision/auto/1a2b3c4d5e6f.parquet")
joints = ["left_wrist", "right_wrist", "left_shoulder", "left_ankle"]
both = mp[mp["detected"] & mp["joint"].isin(joints)].merge(
    vi[vi["detected"] & vi["joint"].isin(joints)], on=["frame_idx", "joint"], suffixes=("_mp", "_vi")
)
both.assign(d=np.hypot(both.x_mp - both.x_vi, both.y_mp - both.y_vi)).groupby("joint")["d"].median()
```

### Comparing the two athlete selections

The same comparison over the athlete files — the pair the bake-off (#16) is
actually about — is a join on `(frame_idx, joint)` of
`mediapipe_athlete/<rotate>/<clip_id>.parquet` and
`vision_athlete/<rotate>/<clip_id>.parquet`, both of which hold the same 33 rows
per frame in the same joint order:

```python
import numpy as np, pandas as pd

JOINTS = ["left_wrist", "right_wrist", "left_shoulder", "right_shoulder",
          "left_hip", "right_hip", "left_ankle", "right_ankle"]

def read(root, clip):
    t = pd.read_parquet(f".../keypoints/{root}/auto/{clip}.parquet")
    t = t[t["detected"] & t["joint"].isin(JOINTS)]
    return t[["frame_idx", "joint", "x", "y"]].rename(columns={"x": f"x_{root}", "y": f"y_{root}"})

both = read("mediapipe_athlete", CLIP).merge(read("vision_athlete", CLIP), on=["frame_idx", "joint"])
both = both.dropna()          # only frames both models attributed, with finite pixels
both["d"] = np.hypot(both.x_mediapipe_athlete - both.x_vision_athlete,
                     both.y_mediapipe_athlete - both.y_vision_athlete)
per_joint = both.groupby("joint").d.median()
per_joint.median()             # A: median of the eight per-joint medians
both.d.median()                # B: pooled median over every row
```

**Say which of the two you quote.** They are not the same number, because one bad
joint is one of eight rows pooled and one of eight medians, and per clip they
differ by more than 5 px (`6508f9b355bd`: A = 12.9 px, B = 18.1 px, because its
left ankle is ~280 px out in every frame). The numbers the issue for #76 asked
for, per clip, as of that run (`--rotate auto`, 3 clips with Vision keypoints):

| clip | A: median of per-joint medians | B: pooled median | worst joint (its median) |
|---|---|---|---|
| `057c9e6c96af` | 12.6 px | 12.0 px | `left_ankle` 160 px |
| `64184de33f84` | 14.9 px | 13.6 px | `left_hip` 22 px, but `left_wrist` 240+ px in 51 of 275 frames (max 266) |
| `6508f9b355bd` | 12.9 px | 18.1 px | `left_ankle` 282 px (>100 px in 195 of 195 rows) |

The aggregate hides the shape of the disagreement: it is concentrated in **one
leg**. In `057c9e6c96af` and `6508f9b355bd` the athlete's left ankle is far out
while the other seven joints agree to ~10 px, and in `64184de33f84` the left
wrist is 240+ px out in 51 frames (a further 47 frames are over 100 px) while the
body agrees. That is the trainer's leg and the trainer's wrist being read as the
athlete's — the trainer standing right behind them, inside **one** skeleton,
which is why the contact rules (which need a *second* detection) cannot see it.
