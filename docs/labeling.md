# Keypoint labelling (Label Studio)

The pose-model bake-off (chainlink #16)
needs ground truth: frames a person has marked the athlete's joints on by hand,
so MediaPipe — and later Apple Vision and YOLO-pose — can be scored against a
human instead of against another model. This page is the whole loop: pick the
frames, label them, export, convert.

Label Studio runs in its own environment, never as a pipeline dependency; the
pipeline only reads the images and the JSON. RTMPose, which draws the
pre-labels, *is* a pipeline dependency (`rtmlib` + `onnxruntime`).

```
handstand.frame_sampler  ->  data/label_frames/*.jpg + manifest.csv
handstand.prelabel       ->  data/label_studio_prelabels.json + data/labels/review_queue.csv
Label Studio (separate)  ->  data/label_studio_export.json
handstand.labels import  ->  data/labels/keypoints.csv
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

## 2. Pre-label the frames

Clicking 15 joints on 300 frames is a couple of hours of identical work, so an
agent does a first pass and you only fix what is wrong:

```bash
cd pipeline
uv run python -m handstand.prelabel
```

```
pre-labelled 300 frame(s) with RTMPose in 142s
  model: rtmpose-prelabel (balanced); wrote .../data/label_studio_prelabels.json
  found somebody in 281/300 (93.7 %)
  rotated pass won 157/300 (52.3 %); athlete reference used in 281
  19 frame(s) have a reference athlete that rejected every body RTMPose found
  (usually only the trainer was detected), so they are written with no pre-label
  at all; 19 frame(s) in total are left for the labeler
  281 frame(s) have a measurable disagreement; 0.0 joint(s) per frame left for the labeler
  review queue: .../data/labels/review_queue.csv (worst first)
```

On the full 300-frame sample that is **under three minutes** of compute for
4 213 pre-placed points. 19 of the 300 frames get no pre-label at all, on
purpose — see the limitation below — and the other 281 come back with 15 of 15
joints or very nearly.

Two files come out:

| file | what it is |
| --- | --- |
| `data/label_studio_prelabels.json` | one Label Studio task per image, each with a `predictions` entry holding the model's points |
| `data/labels/review_queue.csv` | the same frames, worst first, with the reason each is doubtful |

Useful flags: `--mode lightweight|balanced|performance` (the RTMPose
detector/pose pair; `balanced` is the default), `--limit N` for a quick pass,
`--contact-sheet [N]` to draw the worst frames as one JPEG for a look before
importing anything, `--rotate`, `--image-base`, `--data`. The 0.3 score bar is a
constant, not a flag: it is the labelling rule above turned into a number, and a
run whose pre-labels were written at one bar and reviewed at another would be
comparing two different questions.

### Why RTMPose and not MediaPipe

The bake-off contestants are **MediaPipe** and **Apple Vision**. Drawing the
first-pass labels with a contestant would quietly bias the ground truth towards
it: the labeler corrects the *other* model's mistakes most, so the "human" labels
would end up measuring agreement with whoever drew them. RTMPose (a top-down
pose estimator on a YOLOX detector, trained on COCO-WholeBody) is a third,
independent model, so what you correct is a model's mistake rather than a
contestant's — and where RTMPose and a contestant disagree, the disagreement is
itself a reason to look at that frame first.

That last point is not theoretical on this sample: the median disagreement on
`clean_trainer` frames is several times the one on `clean_no_trainer`, and the
largest is over 4 body lengths — a foot placed a third of the image away from
where the other model put it. **Read the disagreement as "these two models do not
agree here", not as "this pre-label is wrong"**: on the frames that top the queue
it is often the *contestant* that has followed the trainer's leg, and the
pre-label itself is the good one. `6508f9b355bd_1`, the frame that was rank 1
before the athlete-matching fix below, is the clearest case — MediaPipe's athlete
has one ankle at the top of the frame and one at the bottom, and RTMPose's
pre-label is on the handstand next to it.

### What it does to each frame

* **Twice.** Once as displayed, once rotated 180°, and the pass with the higher
  mean keypoint score wins. The map-back is the same exact formula the MediaPipe
  runner uses, so both passes speak the frame's own pixels.
* **The athlete, not the trainer.** When RTMPose finds several people, the one
  whose **keypoints** agree with the MediaPipe athlete of the same frame
  (`keypoints/mediapipe_athlete/auto/`) is taken, so the two models are compared
  on the same body. Where no reference overlaps anybody, the most confident body
  is taken instead and the frame is marked as having had no reference. **See the
  limitation below** — this is the one step that can pick the wrong person.
* **The 15 joints the config asks for.** RTMPose's COCO-WholeBody output is
  translated to MediaPipe's names; `foot_index` is COCO-WholeBody's `big_toe`.
  Its 68 face and 42 hand points are ignored.
* **Nothing below 0.3 is written.** A guessed point is worse than no point — the
  rule below asks for a missing point rather than a guess, and a low-scoring
  pre-label fights that. Those joints are exactly what `low_score_joints` lists
  for you to place. On this sample only 2 points in 4 213 fell below 0.3.

#### Picking the athlete is the weak step — and what it still gets wrong

The athlete is chosen by comparing *keypoints* against the MediaPipe athlete,
not bounding boxes. That choice is deliberate and it is not academic: two people
standing next to each other overlap in **both** directions, and a trainer
crouching beside a handstand can cover more of the athlete's bounding box than
the athlete's own thin, inverted skeleton does. On `6508f9b355bd_1` the
trainer's squat scores **0.41** IoU against the MediaPipe athlete's box while
the athlete's own skeleton scores **0.17**, so ranking by box picked the trainer
and pre-labelled the one person this config says not to label. Ranking on
keypoints cannot make that mistake — the trainer's feet are side by side on the
floor and the athlete's are 320 px apart — and it takes that frame from 1.67 body
lengths of error to 0.15.

It is still the weak step, and the honest numbers on the 300-frame sample are:

| rule | pre-labels on the wrong body |
| --- | --- |
| box overlap (the obvious rule) | **20/300 (6.7 %)** |
| keypoint agreement | 17/300 (5.7 %) |
| keypoint agreement **+ dropping frames the reference rejects | **0/300 (0.0 %)** |

Measured as "the pre-labelled nose is more than 0.6 body lengths from the
MediaPipe athlete's nose". Every one of the 20 original failures had a trainer in
the shot, against a base rate of 101/300 — so the rate is 3× the background, and
it was concentrated in exactly the `clean_trainer` and `trainer_contact` strata.

Most of the remainder was not a selection failure at all: in 14 of the 17 cases
RTMPose's YOLOX detector reported **only one person**, and that person was the
trainer. There was no choosing to do — the athlete had never been detected. That
is why a frame where the reference athlete exists and *every* detected body is far
from it now gets **no pre-label at all** rather than the most confident body: an
empty frame the labeler fills in beats a confident pre-label of the trainer that
they have to notice and delete. The 19 frames that ended up empty are exactly
the ones that were wrong, so the trade is 19 frames labelled by hand out of 300
in exchange for never handing over somebody else's joints. The run reports the
count on its `rejected` line.

Two further consequences for you as the labeler:

* A frame with **no** pre-label is not necessarily a frame the model failed on —
  it may be a frame where the reference refused every detection. Label it from
  scratch.
* All the wrong-body frames land at the **top of the review queue**, because a
  skeleton on the wrong person disagrees with the MediaPipe athlete violently.
  But see above: a large `max_disagreement` is not by itself evidence that the
  *pre-label* is wrong.

### The rotated pass, and how often it wins

The rotated pass won **52 %** of the frames — less than you would guess from the
sampler, which says 271 of the 300 frames are an actual inverted hold. It wins
58 % of the holds and 3 % of the kick-ups. So RTMPose copes with a handstand
upright about as often as it does upside down, and the "inverted bodies are what
pose models are worst at" rule only half holds here. The run reports the count
either way, and the decision is made on the score rather than on the rule, so a
model change cannot quietly start feeding the labeler the wrong orientation.

### The review queue

`max_disagreement` is, per frame, the largest distance between any two models'
points for the same joint — RTMPose, the MediaPipe athlete, and the Vision
athlete when its selection exists — divided by that frame's body length, so a
tall and a short athlete are judged on the same terms. `worst_joint` names the
joint. Rows are sorted by that disagreement plus a penalty for every joint RTMPose
was unsure about, so a frame reaches the top either because two models put a
joint in different places or because the pre-labels are missing joints you have
to find yourself.

| column | meaning |
| --- | --- |
| `image`, `clip_id`, `frame_idx` | which frame |
| `stratum` | from the sampler manifest: `clean_no_trainer`, `clean_trainer`, `trainer_contact` |
| `max_disagreement` | worst joint gap, in body lengths |
| `worst_joint` | which joint that gap was on |
| `low_score_joints` | `;`-separated joints RTMPose scored below 0.3 — yours to place |
| `rotated_used` | `true` when the rotated pass won |

The Vision athlete has only been generated for a couple of clips so far, so most
frames are scored on RTMPose against MediaPipe alone. That is not a problem: a
joint only one model measured has no gap to report and is skipped rather than
counted as a disagreement with nobody.

On the full sample the disagreement is **0.10 body lengths at the median** and
0.05 at the tenth percentile, with a 90th percentile of 1.2 — so most frames
agree closely and roughly a tenth disagree so badly that they are the queue's
whole reason for existing. The feet are where it happens: `left_foot_index` and
`right_foot_index` are the worst joint on 151 of the 300 frames, because in a
handstand the two feet are together, partly out of frame and often behind the
trainer, and a toe is the smallest thing either model is asked to find.

## 3. Install Label Studio

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

## 4. Create the project

1. **Create Project**, name it e.g. `handstand-keypoints`.
2. Open **Settings → Labeling Interface** and paste the contents of
   [`tools/labeling/label_studio_config.xml`](../tools/labeling/label_studio_config.xml)
   (replacing the default one). It asks for 15 keypoints — `nose`, both
   shoulders, elbows, wrists, hips, knees, ankles and `foot_index` toes — using
   the exact joint names of `handstand.pose_mediapipe.JOINT_NAMES`, plus one
   `occluded_or_unsure` choice per image.
3. **Import** → *Import pre-annotations* (or drag the file in / use the API
   `POST /api/projects/<id/import`) and choose
   `data/label_studio_prelabels.json` from step 2.

Import the **JSON**, not the JPEGs. The JSON carries one task per image *with*
the pre-labels already in it, so Label Studio creates 300 tasks that arrive with
the model's points drawn and you only move what is wrong. Uploading the folder
instead still works — the converter finds each frame again from the
`<clip_id>_<frame_idx>` tail of the name either way — but then you click all
15 joints on all 300 frames, which is the work this step exists to avoid.

The file uses `file://` URIs, so Label Studio has to be allowed to read local
files. Start it with:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(realpath ../data/label_frames)
```

If you would rather serve the frames over HTTP, re-run the pre-labeller with
`--image-base http://localhost:8080/frames` and the URIs come out that way
instead.

Check the pre-labels arrived by opening one task: it should show a skeleton
already drawn. If a task is blank, `data/labels/review_queue.csv` tells you why
— `low_score_joints` lists what the model was unsure about, and a frame with
nobody in it is simply empty.

## 5. Label

Work through `data/labels/review_queue.csv` **top to bottom** — it is sorted so
the frames whose pre-labels are most likely wrong come first, which is where a
correction is worth the most. Match each row to its image by the `image`
column, and use the other columns as the hint they are:

* `max_disagreement` / `worst_joint` — two models put this joint in different
  places. Check it first.
* `low_score_joints` — RTMPose was unsure about these, so no point was drawn for
  them. Place them yourself, or leave them out if you cannot see them.
* `stratum` — `trainer_contact` frames are the hard ones by construction.
* `rotated_used` — the pre-labels came from the model looking at the frame
  upside down, which is the normal case for a hold.

Then do what the pre-labels did not do: for each joint you can see, check the
point and move it if it is wrong. `left`/`right` is the **athlete's** own left
and right, not the left and right of the picture: in a handstand filmed from
the front, the athlete's left arm is on the right of the screen. Ignore the
trainer completely.

| what you see | what to do |
| --- | --- |
| the pre-label is right | leave it |
| the pre-label is on the wrong spot | drag it to the joint |
| the joint is visible but has no pre-label | place a point on it |
| the joint is hidden (behind the trainer, out of frame) | **place no point** |
| the whole frame is unusable | set `occluded_or_unsure` to `occluded` or `unsure` |

A missing point *is* the "not visible" answer: it produces no row in the CSV, and
a scorer counts it as a joint the model was never expected to find. Do not guess
where a hidden joint would be — a guessed point is worse than no point.

Label Studio (checked against 1.23.1) has no per-keypoint visibility attribute,
so the config does not pretend to have one; the `occluded_or_unsure` choice is
the per-image safety net. If a later version adds per-keypoint visibility, the
converter already reads the `occluded`/`visible` keys such a point would carry.

## 6. Export

**Export** → *Export to JSON*, with all annotations. Save it as
`data/label_studio_export.json` (any path works; the next step takes it as an
argument).

This is unchanged by the pre-labels. A task you accepted without touching is an
ordinary annotation once you submit it, so the export is the same file it would
have been without step 2, and the import below does not know or care that the
first point on a joint came from a model.

## 7. Convert to pixels

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

A pre-label written by `handstand.prelabel` uses the **same** conversion in the
other direction, so a point that survives the export lands back on the pixel the
model predicted. A round trip of the whole file — pre-label to export to
`keypoints.csv` — is covered by `tests/test_prelabel.py`, so the two converters
cannot drift apart.

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

**Pre-labels do not show up in Label Studio** — the tasks were imported from
the folder of JPEGs rather than from `data/label_studio_prelabels.json`, or
Label Studio could not read the images. For the `file://` URIs it needs
`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and
`LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` pointing at `data/label_frames` (see
step 4); otherwise re-run with `--image-base http://...`. Importing the JSON
again creates a second set of tasks — delete the first.

**A pre-label is on the wrong joint** — RTMPose and MediaPipe disagree about
`left`/`right` in a handstand: it is easy for both to swap which arm is the
athlete's left when the athlete is upside down. The review queue's
`worst_joint` column is where this shows up first, and the `left`/`right` rule
in step 5 is the tiebreak.

**`max_disagreement` is 0.0000 for a whole column** — nothing was measured to
compare. Either `keypoints/mediapipe_athlete/auto/` is missing for those clips
(generate it as above) or the Vision athlete has no keypoints yet, so the frame
is scored on RTMPose's own confidence alone. A frame with one model measuring
every joint is not called disagreeing with nobody; it is simply unranked.
