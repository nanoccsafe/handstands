# Keypoint labelling (Label Studio)

The pose-model bake-off (chainlink #16)
needs ground truth: frames a person has marked the athlete's joints on by hand,
so MediaPipe — and later Apple Vision and YOLO-pose — can be scored against a
human instead of against another model. This page is the whole loop: pick the
frames, label them, export, convert.

Nothing in this loop adds a dependency to the pipeline. Label Studio runs in its
own environment; the pipeline only reads the images and the JSON.

```
handstand.frame_sampler  ->  data/label_frames/*.jpg + manifest.csv
Label Studio (separate)   ->  data/label_studio_export.json
handstand.labels import   ->  data/labels/keypoints.csv
```

## 1. Pick the frames

```bash
cd pipeline
uv run python -m handstand.frame_sampler --n 300
```

Writes `data/label_frames/<clip_id>_<frame_idx>.jpg` and
`data/label_frames/manifest.csv`:

| column | meaning |
| --- | --- |
| `image` | file name of the JPEG |
| `clip_id`, `frame_idx`, `t_ms` | which frame of which clip |
| `stratum` | `clean_no_trainer`, `clean_trainer` or `trainer_contact` |
| `skill` | the catalogue's skill for that clip |
| `trainer_present`, `trainer_contact` | per clip / per frame flags |
| `display_width`, `display_height` | the size of the JPEG, in pixels |

The JPEGs are decoded in **display orientation** (upright, as a person sees
them) with the same `frame_idx` the keypoint parquets use, so a hand-placed
point and a MediaPipe landmark of the same frame are in the same pixel
coordinates.

The sampler spreads its picks over the dataset on purpose:

* **Skill filter.** Only clips whose `skill` is one of `--skills` (default
  `line`) are used, and `walk` is never used. While the `skill` column of
  `catalogue.csv` is still empty (#64) *every* clip is used and the manifest
  says so: the `skill` column then reads `unlabelled`. Once you fill the column
  in, re-run and the filter switches itself on.
* **Strata.** 60 % clean frames from clips with no trainer, 25 % clean frames
  from clips with a trainer, 15 % frames the trainer is actually touching the
  athlete in. The last group is the hard case and the one the models are worst
  at, so it is a fifth of the sample.
* **Spread.** Clips are taken round-robin over session dates, and inside a clip
  the frames come from equal-width time bins, at most 4 per clip and never
  closer than 0.5 s.
* **Holds.** At least 70 % of the frames are an actual handstand (mean wrist y
  below mean ankle y); the rest are kick-ups and exits.

Useful flags: `--n`, `--seed` (the same seed always picks the same frames),
`--skills line press_up`, `--n-per-clip`, `--min-spacing-s`,
`--inverted-fraction`, `--out`.

The run prints what it did, and the numbers are worth reading before you start
labelling — in particular whether the skill filter was active:

```
wrote 300 of 300 requested frames from 76 clip(s) over 48 session(s) -> .../label_frames
  strata: clean_no_trainer=180/180 clean_trainer=75/75 trainer_contact=45/45
  inverted (hold): 271 of 300 (90.3 %), target 210
  skill filter: OFF, the catalogue carries no skills so every clip is used
```

300 frames is a couple of hours of careful work. If you want a quick pass first,
`--n 60` gives a sample with the same shape.

## 2. Install Label Studio

**Not a project dependency.** Do not `uv add label-studio`; it would pin a web
framework into the analysis pipeline. Either of these is fine:

```bash
# ephemeral, nothing installed: uv keeps it in its own cache
uv tool run --from label-studio label-studio start --port 8080 \
    --init --username me --password secret

# or a dedicated venv, which you keep between sessions
python3 -m venv ~/label-studio
~/label-studio/bin/pip install label-studio
~/label-studio/bin/label-studio start --port 8080 --init \
    --username me --password secret
```

`--init` creates the account if there is none yet (`--username` /
`--password` set it non-interactively; leave them out and Label Studio asks on
first start). Open <http://localhost:8080>.

## 3. Create the project

1. **Create Project**, name it e.g. `handstand-keypoints`.
2. Open **Settings → Labeling Interface** and paste the contents of
   [`tools/labeling/label_studio_config.xml`](../tools/labeling/label_studio_config.xml)
   (replacing the default one). It asks for 15 keypoints — `nose`, both
   shoulders, elbows, wrists, hips, knees, ankles and `foot_index` toes — using
   the exact joint names of `handstand.pose_mediapipe.JOINT_NAMES`, plus one
   `occluded_or_unsure` choice per image.
3. **Import** → *Upload selected files* → select
   `data/label_frames/*.jpg` (not `manifest.csv`).

Uploading is a copy: Label Studio stores each file under its own name with a
hash in front of it. That is fine — the converter finds the frame again from the
`<clip_id>_<frame_idx>` tail of the name (see below).

## 4. Label

For each frame, place one point per joint you can see. `left`/`right` is the
**athlete's** own left and right, not the left and right of the picture: in a
handstand filmed from the front, the athlete's left arm is on the right of the
screen. Ignore the trainer completely.

| what you see | what to do |
| --- | --- |
| the joint | place a point on it |
| the joint is hidden (behind the trainer, out of frame) | **place no point** |
| the whole frame is unusable | set `occluded_or_unsure` to `occluded` or `unsure` |

A missing point *is* the "not visible" answer: it produces no row in the CSV, and
a scorer counts it as a joint the model was never expected to find. Do not guess
where a hidden joint would be — a guessed point is worse than no point.

Label Studio (checked against 1.23.1) has no per-keypoint visibility attribute,
so the config does not pretend to have one; the `occluded_or_unsure` choice is
the per-image safety net. If a later version adds per-keypoint visibility, the
converter already reads the `occluded`/`visible` keys such a point would carry.

## 5. Export

**Export** → *Export to JSON*, with all annotations. Save it as
`data/label_studio_export.json` (any path works; the next step takes it as an
argument).

## 6. Convert to pixels

```bash
cd pipeline
uv run python -m handstand.labels import ../data/label_studio_export.json
```

Label Studio stores a keypoint as a percentage of the image, which is
resolution independent and therefore useless on its own. The converter reads
`data/label_frames/manifest.csv` for the display size of each frame and turns
percentages into display-frame pixels with the same rule the keypoint parquets
were written with (`x = percent / 100 * (width - 1)`):

```
data/labels/keypoints.csv
clip_id, frame_idx, joint, x, y, visible, labeler, labeled_at
6508f9b355bd,122,left_wrist,288.0,702.4,true,someone@example.com,2026-09-27T10:05:00Z
```

* `visible` is `false` for a point the tool flagged occluded and for every point
  of an image marked `occluded`/`unsure`. The coordinates are kept either way:
  "here, but I am not sure" is information a scorer can use.
* `labeler` and `labeled_at` come from the annotation that placed the point.
* **The import is idempotent.** Rows are keyed by
  `(clip_id, frame_idx, joint)`, so importing the same export twice changes
  nothing, and importing a corrected export rewrites exactly the keys it
  mentions and leaves the rest of the CSV alone. Re-import freely.
* Frames that are not in the manifest (an image you labelled by hand, a frame
  from an older sample) are skipped and counted — there is no recorded size to
  convert them against, and guessing one would silently shift every point.
* With several labelers, the last annotation of an export wins for a given joint
  and the import says how many joints were labelled twice.

Useful flags: `--manifest`, `--out`, `--data`.

## Troubleshooting

**`no frame can be sampled`** — the keypoints the sampler reads do not exist yet.
Run, in order:

```bash
uv run python -m handstand.pose_mediapipe --rotate auto --num-poses 3 \
    --running-mode image --min-detection 0.2 --min-presence 0.2
uv run python -m handstand.athlete --rotate auto
```

**The `clean_trainer` and `trainer_contact` strata come out empty** — the trainer
report is missing, so every clip counts as trainer-free. Generate it with
`uv run python -m handstand.trainer_report`.

**A frame is not the one you expected** — the sampler and the parquets agree by
construction (same `frame_idx`, same decoder), so check that you are looking at
`data/label_frames/` and not at a stale set from an earlier run; re-run the
sampler to rewrite both the images and the manifest together.

**Points look shifted by a pixel or two** — the display size in the manifest is
the size of the JPEG the labeler saw, and the conversion is exact in integer
pixels. If you resized an image by hand before importing it, re-import the
originals; the CSV's coordinates are in the frame's own pixels either way.
