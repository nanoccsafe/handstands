# Pose model bake-off

Which pose model the app should ship, measured against the user's hand labels
(chainlink #16). Three candidates are scored on the frames a human actually
labelled, every number comes out of `pipeline/handstand.bakeoff`, and the
decision — with the per-backend visibility gates the pipeline should use — is
written at the bottom of this page.

```
cd pipeline
uv run python -m handstand.bakeoff                 # full run, includes a fresh RTMPose pass
uv run python -m handstand.bakeoff --skip-rtmpose  # skip the ~2 min RTMPose inference
uv run python -m handstand.bakeoff --models mediapipe vision
uv run python -m handstand.bakeoff --flag-catalogue # one-off: flag multi-person clips in the catalogue
```

The CSVs land in `<data_dir>/reports/bakeoff/` (git-ignored); this document
holds only aggregate tables — no images, no keypoints, no personal data.

## The candidates

| model | where its keypoints come from | on-device? |
|---|---|---|
| **mediapipe** | `keypoints/mediapipe_athlete/best` (the pipeline's athlete selection on top of the multi-person run) | yes (Core ML, #45) |
| **vision** | `keypoints/vision_athlete/auto` — Apple Vision body pose, `tools/mac/run_vision.sh --all --rotate auto`, run fresh for this bake-off over **all 179 non-missing catalogue clips** (0 failures) | yes (built into iOS) |
| **rtmpose** | rtmlib *balanced*, run fresh on the 296 label JPEGs with `prelabel.make_model` + `predict_frame`, **without a reference** (comment 46: MediaPipe is never the referee), both orientations, best score | no — reference upper bound |

Vision's frame counts line up with MediaPipe's on all 76 labelled clips
(checked clip by clip), so a `frame_idx` is the same moment in both files.

## Person matching, two ways

Every model is reported twice:

- **pipeline** — the model's own person choice: the athlete parquet's person
  for mediapipe/vision; for RTMPose the *inverted* person with the best score,
  else the best score (in a handstand the athlete is the inverted body, a
  standing trainer never is).
- **oracle** — among all people the model detected in the frame, the one with
  the smallest mean keypoint error to the ground truth. This isolates keypoint
  accuracy from person choice.

**The decision is made on `pipeline`; `oracle` explains the gap.** On the
headline subset the two are identical for all three models — those frames are
clean and nearly all single-person — so person choice does not decide the
outcome. The gap shows up where person confusion lives:

- on the multi-person clips of the selection set (flag from the trainer
  report, #15), mediapipe's pipeline PCK@0.2 falls from **0.855** (single-person
  clips, 17 frames / 9 clips) to **0.641** (multi-person clips, 27 frames /
  12 clips) — and the oracle recovers **nothing** of that (also 0.641). The
  bodies are *stitched*, not mis-chosen: giving the model free person choice
  cannot fix a skeleton whose bones already run into the background person
  (the #64/#15 symptom);
- on the trainer-contact note frames, mediapipe's oracle lowers the miss rate
  (0.105 → 0.048) but its mean error *rises* (0.337 → 0.413): the oracle
  answers frames the pipeline refused, and those answers are sometimes far
  from the athlete.

## Ground truth rules

- **Human-reviewed frames only** for accuracy. `labels/auto_accepted.csv`
  lists the auto-accepted frames (RTMPose pre-labels); scoring accuracy on
  them would score RTMPose against itself (comment 47). They are reported
  separately as *agreement with RTMPose* and never as accuracy.
- **Missing clips excluded** — any clip whose catalogue row says
  `missing=true` (the duplicate 244275cd7059, comment 49) is dropped from
  every table.
- **One trainer mask, chosen once** — `trainer_contact` comes from the
  MediaPipe multi-person run (the `trainer_contact` column of the MediaPipe
  athlete parquet) and is applied to *every* model's keypoints (comments
  25/26). The app model is chosen on the clean strata
  (`clean_no_trainer` + `clean_trainer`, mask applied); trainer-contact frames
  are reported only as a note.
- **Normalisation** — `L` is the ground-truth torso length, shoulder midpoint
  to hip midpoint. A frame with no GT shoulder or no GT hip cannot be
  normalised and is excluded from the normalised metrics (counted in the
  run's summary). In practice every human-reviewed frame has at least one
  shoulder and one hip — 57 of 70 label a single side only, which is normal
  for a side view (the far shoulder sits behind the near one,
  `docs/swift.md`), so each midpoint falls on the labelled side — and **0
  frames** were excluded for missing torso GT.

### Sample sizes (from the final run)

| slice | frames | notes |
|---|---|---|
| labelled frames total | 296 | 76 clips, 3986 keypoints |
| human-reviewed, kept for accuracy | **70** | 28 clips; per stratum: `clean_no_trainer` 19, `clean_trainer` 36, `trainer_contact` 15 |
| auto-accepted, agreement only | 223 | per stratum: `clean_no_trainer` 161, `clean_trainer` 32, `trainer_contact` 30 |
| selection set (clean strata, mask applied) | 44 | 21 clips; drives the phase/shape/multi-person tables, the swap rate and the headline |
| mask applied, all strata | 48 | the 44 plus 4 unmasked `trainer_contact` frames; drives the per-stratum table and the gate sweep |
| trainer-contact note | 26 | reported as a note, never as a selection criterion |
| **headline subset** | **21** | hold frames of **line** holds in the clean strata, 10 clips |

The sample is small. Every 95 % interval below is a bootstrap **over clips**
(frames inside a clip move together), 1000 draws, seed 16 — and with only 10
clips in the headline the intervals are wide.

## Headline: hold + line + clean strata

Pipeline matching, each backend at its own recommended gate:

| model | gate | PCK@0.2 [95 % CI] | PCK@0.5 | mean err (L) | median err (L) | miss rate |
|---|---|---|---|---|---|---|
| mediapipe | 0.50 | **0.759** [0.629, 0.895] | 0.949 | 0.172 | 0.119 | 0.015 |
| vision | 0.75 | **0.109** [0.039, 0.211] | 0.117 | 0.312 | 0.086 | 0.869 |
| rtmpose | 0.00 | **0.898** [0.817, 0.977] | 0.956 | 0.157 | 0.079 | 0.000 |

137 ground-truth-visible schema joints over 21 frames. The oracle rows are
identical to the pipeline rows.

Gate sensitivity of the same subset (pipeline matching) — the ordering never
changes:

| gate | mediapipe | vision | rtmpose |
|---|---|---|---|
| 0 (no gate) | 0.759 | 0.606 [0.405, 0.836] | 0.898 |
| 0.50 (pipeline default) | 0.759 | 0.431 [0.229, 0.675] | 0.876 [0.754, 0.977] |
| recommended per backend | 0.759 | 0.109 [0.039, 0.211] | 0.898 |

### By joint group (headline, pipeline, recommended gate)

| group | mediapipe PCK@0.2 | vision PCK@0.2 | rtmpose PCK@0.2 |
|---|---|---|---|
| nose | 1.000 | 0.500 | 1.000 |
| shoulders | 0.762 | 0.000 (miss 1.000) | 1.000 |
| elbows | 0.800 | 0.250 | 0.900 |
| wrists | **0.619** | 0.095 | 0.810 |
| hips | 0.667 | 0.000 (miss 1.000) | 0.762 |
| knees | 0.955 | 0.045 | 0.955 |
| ankles | **0.667** | 0.125 | 0.917 |
| foot_index † | 0.640 | — | 0.880 |

† `foot_index` exists only for MediaPipe/RTMPose; Vision's schema stops at the
ankle, so it is reported separately and never mixes into the cross-model
numbers.

Vision's shoulders and hips score 0.000 not because the coordinates are
always wrong but because **every** shoulder/hip confidence in the headline
sits below its own recommended gate (miss = 1.000): the gate that buys
precision buys it with the torso.

### Breakdowns (selection set, pipeline, recommended gate)

Per stratum (mask applied; the `trainer_contact` row is the 4 frames of that
stratum the mask left through — the excluded ones are in the note table):

| stratum | mediapipe | vision | rtmpose | n (joints / frames) |
|---|---|---|---|---|
| clean_no_trainer | 0.850 [0.729, 0.948] | 0.179 | 0.886 | 140 / 17 |
| clean_trainer | 0.643 [0.482, 0.792] | 0.110 | 0.901 | 182 / 27 |
| trainer_contact (note) | 0.714 [0.250, 1.000] | 0.143 | 0.971 | 35 / 4 |

Per phase:

| phase | mediapipe | vision | rtmpose | n (joints / frames) |
|---|---|---|---|---|
| hold | 0.793 [0.684, 0.895] | 0.163 | 0.903 | 227 / 31 |
| kickup | 0.724 [0.429, 0.909] | 0.207 | 0.862 | 29 / 3 |
| pre | 0.547 [0.104, 0.833] | 0.019 | 0.981 | 53 / 8 |
| exit | 1.000 | 0.167 | 1.000 | 6 / 1 |
| unknown | 0.000 | 0.000 | 0.000 | 7 / 1 |

Per shape (hold shape of the frame's hold, `none` outside a hold):

| shape | mediapipe | vision | rtmpose | n (joints / frames) |
|---|---|---|---|---|
| line (= headline) | 0.759 | 0.109 | 0.898 | 137 / 21 |
| none (not in a hold) | 0.589 | 0.084 | 0.874 | 95 / 13 |
| straddle | 0.828 | 0.281 | 0.984 | 64 / 7 |
| tuck | 0.846 | 0.077 | 0.462 | 13 / 2 |
| mexican | 0.923 | 0.231 | 1.000 | 13 / 1 |

Multi-person split of the selection set (clip saw a second person in the
trainer report, #15):

| clip type | mediapipe | vision | rtmpose |
|---|---|---|---|
| single-person (17 frames / 9 clips) | 0.855 [0.735, 0.952] | 0.174 | 0.906 |
| multi-person (27 frames / 12 clips) | 0.641 [0.481, 0.800] | 0.114 | 0.886 |

Besides this table, the multi-person clips are flagged in
`data/catalogue.csv`'s `notes` column ("second person present", 89 clips) by
`bakeoff --flag-catalogue`, so later stages find them without reading the
trainer report.

Trainer-contact frames, reported **only as a note** (comments 25/26): 26
frames; mediapipe PCK@0.2 0.657 [0.483, 0.857] on the
`trainer_contact` stratum, vision 0.086, rtmpose 0.800.

### Left/right swap rate

Frames where the model's wrist or ankle sides are crossed relative to the
labels, evaluated only when the ground-truth sides are far enough apart to
mean anything (≥ 0.15 L — in a side view the sides often overlap) and both
model joints clear the gate:

| model | swap rate | evaluable frames |
|---|---|---|
| mediapipe | 0.08 (1 swapped) | 12 |
| vision | 0.00 | 4 |
| rtmpose | 0.00 | 12 |

Too few evaluable frames to conclude much — reported as-is.

### Agreement with RTMPose (auto-accepted frames — *not* accuracy)

223 frames, 2899 joints, at each backend's recommended gate:

| model | agreement PCK@0.2 | PCK@0.5 | miss rate |
|---|---|---|---|
| mediapipe | 0.843 | 0.884 | 0.109 |
| vision | 0.177 | 0.178 | 0.815 |

RTMPose itself is deliberately absent: on these frames it would be compared
with its own pre-labels.

## Visibility gate per backend

The pipeline drops joints below `MIN_VISIBILITY = 0.5`
(`postprocess.MIN_VISIBILITY`), a value tuned on MediaPipe's visibility
(comment 43). The bake-off tabulates error vs confidence on the
human-reviewed pipeline joints with the mask applied (48 frames;
`reports/bakeoff/gate_<model>.csv`) and recommends the **lowest gate at
which ≥ 90 % of the kept joints fall within 0.2 L**:

| backend | within 0.2 L @ gate 0 | @ 0.50 | recommended gate | target met? |
|---|---|---|---|---|
| mediapipe | 0.778 | 0.786 | **0.50** (keep the status quo) | **no** — precision is flat (0.78–0.80 across the whole sweep), no gate reaches 90 % |
| vision | 0.749 | 0.850 | **0.75** (0.926 within 0.2 L, keeps 18 % of joints) | yes |
| rtmpose | 0.902 | 0.924 | **0.00** | yes, already ungated |

Then `handstand.postprocess` runs over all 179 non-missing catalogue clips at
`min_visibility = 0.5` and at the recommended gate, counting clips with **no
body length** (unusable):

| backend | unusable at 0.50 | unusable at recommended gate |
|---|---|---|
| mediapipe | **3 / 179** | 3 / 179 (recommended = 0.50) |
| vision | **84 / 179** | **174 / 179** (at 0.75) |
| rtmpose | n/a — no full-clip keypoints exist for RTMPose; it is not an on-device backend | n/a |

Two things to read out of this table:

1. **mediapipe at 0.5 is cheap and stable**: 3 unusable clips, and raising
   the gate to 0.95 would buy under two points of precision (0.786 → 0.803)
   while discarding 14 % of the joints — so its recommendation is to keep 0.5
   and say the 90 % target is unreachable by gating (its errors are confident
   errors).
2. **vision's confidence scale cannot be gated into usefulness**: at 0.5 it
   already loses 84/179 clips (82 of them because the torso — both shoulders
   *and* both hips above the gate on ≥ 10 frames — is never measurable), and
   the gate that meets the precision target (0.75) loses 174/179. For vision
   the follow-up is **confidence recalibration**, not a different number in
   `PostProcessConfig.minVisibility`.

## Decision

The rule: *pick the on-device model with the higher headline PCK@0.2,
unless the clip-bootstrap intervals overlap heavily — then prefer Vision
(built into iOS, no model download) and say it is not decisive.*

- At the recommended gates the intervals are **disjoint**
  (mediapipe 0.759 [0.629, 0.895] vs vision 0.109 [0.039, 0.211]).
- At the common pipeline gate 0.5/0.5 the winner is unchanged
  (0.759 vs 0.431) but the intervals overlap — that comparison alone would
  not be decisive; the per-backend-gate comparison above is the one the rule
  specifies (miss rate is defined against each backend's gate).
- At gate 0 (raw detections, no confidence gate) the ordering still favours
  mediapipe (0.759 vs 0.606).
- The #17 rule (*both* on-device models below 0.6 PCK@0.2 on wrists or
  ankles in the headline) does **not** fire: mediapipe wrists 0.619, ankles
  0.667; vision wrists 0.095, ankles 0.125 — vision is far below, mediapipe
  is (barely) above. Re-check after more human review: mediapipe's wrist
  interval [0.391, 0.875] spans 0.6.

### The decision: ship MediaPipe as the on-device model

**mediapipe** wins on the headline subset with disjoint intervals, wins at
every gate setting, and is the only on-device backend whose gate keeps clips
usable (3/179 unusable vs vision's 84/179 already at 0.5).

**Recommended per-backend gate:**

| backend | gate | unusable clips at that gate |
|---|---|---|
| mediapipe | **0.50** (unchanged; 90 % precision target unreachable by gating) | 3 / 179 |
| vision | **0.75** by the rule — but it makes 174/179 clips unusable, so do **not** ship it as-is; recalibrate vision's confidence first (see above) | 174 / 179 |

**Headroom**: RTMPose, the reference upper bound, reaches 0.898 headline
PCK@0.2 against mediapipe's 0.759 — about **0.14** of PCK is recoverable by
a better on-device model or fine-tuning. RTMPose is not an on-device
candidate (CPU model, no full-clip run in this repository).

## Limitations

- **Small human-reviewed sample.** 70 accuracy frames over 28 clips; the
  headline decision rests on 21 frames over 10 clips. All intervals are wide
  and every headline number should be re-run as more frames are reviewed.
- **Side views only.** The dataset is side-view footage: far-side joints are
  usually unlabelled (57/70 frames label one side), torso midpoints fall on
  the labelled side, and left/right swap is only evaluable on 4–12 frames per
  model.
- **Trainer frames.** 26 frames are trainer-contact; they are excluded from
  the decision by one MediaPipe-chosen mask and reported only as a note
  (comments 25/26). Vision's inability to see a second person does not matter
  for the app (single-athlete assumption), only for these diagnostics.
- **Auto-accepted frames are agreement, never accuracy** (comment 47).
- **`foot_index` is MediaPipe/RTMPose only**; Vision's schema stops at the
  ankle, so cross-model numbers cover the 13 schema joints.
- **RTMPose has no full-clip keypoints** here, so its unusable-clip count is
  not applicable; its oracle pool is the orientation `predict_frame` kept.
- **Person pools are mostly single-person** on the clean set, so pipeline vs
  oracle separates person choice only weakly there; the multi-person and
  trainer tables carry that signal.
- **YOLO-pose (comment 20) was not run** in this pass: it is not among the
  task's three candidates and would add a heavyweight torch dependency to the
  shared pipeline. The multi-person questions it was meant to answer are
  covered here by the multi-person split, the trainer note and the
  oracle/pipeline pair.

## Next steps

1. **#45 — PoseService default**: the decision above is the input the lead
   asked for; MediaPipe wins, so the PoseService default follows it.
2. **Per-backend `PostProcessConfig.minVisibility`**: mediapipe keeps 0.5;
   vision needs confidence recalibration before any gate ships — at its
   recommended gate 174/179 clips have no body length.
3. **#17 (fine-tune)**: the rule does not fire as written (mediapipe wrists
   0.619 ≥ 0.6). Vision's wrists/ankles (0.095/0.125) would independently
   justify tuning *if* vision were chosen; with mediapipe chosen, re-check the
   rule after more human review — the wrist interval spans the threshold.
4. **More human review** tightens every interval here; the numbers in this
   document are from the run of 2026-10-02 and are regenerated by
   `uv run python -m handstand.bakeoff`.
5. Optional: run comment 20's YOLO-pose candidate in a follow-up if a
   multi-person detector with Core ML export is wanted on-device.

## Reproducing

```bash
# Vision keypoints (all non-missing catalogue clips, rotate auto) — Mac mini:
tools/mac/run_vision.sh --all --rotate auto
# then, on Linux:
cd pipeline
uv run python -m handstand.vision_import --rotate auto
uv run python -m handstand.athlete --source vision --rotate auto
uv run python -m handstand.bakeoff --flag-catalogue   # runs RTMPose fresh, writes the CSVs, prints the summary
```

Tests: `cd pipeline && uv run pytest -q` (the bake-off's own tests are
synthetic — hand-made frames, hand-made person pools, no real videos).
