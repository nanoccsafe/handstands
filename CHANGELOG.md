# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Orientation-aware athlete matching in the pre-labeler, and the re-run of the
  19 frames it left without a pre-label (chainlink #78). Keypoints identify a
  body only when both models read it the same way up: when the MediaPipe
  reference and an RTMPose candidate disagree on inversion (mean wrist y against
  mean ankle y), the distance between them measures the disagreement about
  *orientation*, not about which body is which, so `select_person` no longer
  refuses on it — it takes the **inverted** candidate whose box overlaps the
  reference's best (`handstand.athlete.box_iou`), above
  `MIN_ORIENTATION_FALLBACK_IOU` (0.1). The direction is the rule: a candidate
  that stands while the reference is inverted is still refused, because that is
  usually the trainer. The rotated pass is mapped back into display pixels
  *before* it is matched, so the rule is read in the frame the labeler sees —
  in the model's own coordinates it would flip with the pass and pick the
  trainer on exactly the frames it exists to fix (the keypoint comparison is
  unchanged by that: a 180° map-back preserves every distance and every box).
  Which rule decided a frame is the review queue's new `match_rule` column —
  `keypoint_gap` | `orientation_fallback` | `none`, and empty on the rows a
  partial run never touched — so the one step that can pick the wrong person
  now says why it did what it did
- `uv run python -m handstand.prelabel --only <images> --merge-review`: re-run
  part of the manifest and fold those rows back into the existing review queue
  instead of replacing the queue with them (an unknown image name is an error,
  because a subset run that silently drops a frame writes a file that looks
  complete), plus `--contact-sheet-out`/`--contact-sheet-empty` for a sheet of
  named frames — the empty ones are drawn captioned rather than skipped, since
  on such a sheet the absence is the subject — and `--contact-sheet N` now
  draws N tiles in the most square grid that holds them instead of always
  drawing 6. Used for #78: `data/label_studio_prelabels_fix78.json` (19 tasks,
  image URIs already in the `/data/local-files/?d=label_frames/<name>` form),
  the updated `data/labels/review_queue.csv` and
  `data/overlays/prelabels_fix78_contact_sheet.jpg`
- `--rotate best`, the recommended rotation mode: every frame is run in **both**
  orientations and the one the model is more sure of is kept, which breaks the
  `auto` trap of reading an inverted body as a standing person and then never
  rotating again because the misread says the body is upright (chainlink #79).
  The choice is temporal, because a body does not turn upside down between two
  frames: the per-frame margin `score_rotated - score_upright` is averaged over a
  centred 0.5 s window of clip time (VFR aware, NaN frames skipped) and the
  orientation only switches once that smoothed margin has held the other sign, by
  more than 0.05, for 0.3 s. Over the 180 clips that is 95 orientation changes in
  275 runs, of which **2** are three frames or shorter, against `auto`'s 3,639
  changes in 3,819 runs of which 3,112 are three frames or shorter and the
  per-frame rule's 16,418 — the flicker that made the per-frame version of this
  mode unusable is gone and the sustained runs still switch. Against `auto` on
  the frames both modes detected: 382 frames fixed and 567 broken (the lead
  measured the per-frame rule at 381/255; excluding `55ce46938d00`, which the
  model reads confidently and wrongly the other way round and which a clip-level
  decision therefore commits to for 687 of its 743 frames, it is 382/286). The
  stuck trap of the finding is fixed — `438c3693d6d7` frame 138 is written as
  the handstand it is (wrist y 749, ankle y 326) instead of a person standing on
  the mat (wrist y 427, ankle y 733), and `651b0b5783cc` goes from 81.1 % to
  100 % of its frames read as a handstand. Both scores are written out as
  `score_upright`/`score_rotated` next to the `rotated` flag they decided, so a
  frame's choice can be re-decided later without re-running the model
- `uv run python -m handstand.orient_measure`, the measurement behind the
  `--rotate best` numbers: it reads the keypoints of two rotation modes and the
  later stages' own outputs, and reports the orientation each mode used and how
  often it changed, the frames each read as a handstand, the frames one fixed
  and broke against the other with the run lengths of those changes, the clips
  that moved, and the usable clips, holds and trainer presence the chain found.
  It runs no model and writes nothing, so the tables in
  `docs/keypoint_schema.md` can be produced again rather than taken on trust
  (#79)
- Phase segmentation: `uv run python -m handstand.phases --all` labels every
  frame of a clip `pre`, `kickup`, `hold`, `exit`, `post` or `unknown` and
  numbers the holds, so scoring reads only the hold frames and hand steps
  (#71) and faults (#32) are events inside one. It reads the processed
  trajectories of #20 and the body frame of `handstand.bodyframe`, and is
  rule-based and readable: a hold is the ankle midpoint more than 0.6 body
  lengths above the wrist midpoint, within 35° of vertical, with the hands
  planted and not moving (no wrist faster than 0.3 body lengths per second over
  0.2 s), sustained for at least 0.3 s — a wrist that moves more than 0.1 body
  lengths over that window is a hand step and ends it, and a stretch of up to
  0.3 s of frames nobody could see does not. One visible wrist is enough when a
  side-on clip only reports one, which is what keeps 6508f9b355bd from losing its
  hold. Output goes to `data/phases/<source>/<clip_id>.parquet` (a row per frame
  with `phase`, `hold_id` and every signal the labels came from) and
  `data/phases/<source>/segments.csv` (one row per phase run, merged per clip so
  a run over a few clips keeps the rest). `--render` draws the phase and the hold
  number on every frame through a new `extra_captions` argument of
  `handstand.overlay.render_overlay`. Over the 180 MediaPipe clips: 178 usable,
  166 with at least one hold, a longest hold per clip of median 5.3 s (p25 2.9 s,
  p75 9.1 s, max 41.5 s) and 1467 s of hold time in total (#21)
- Keypoint post-processing: `uv run python -m handstand.postprocess --all` gates
  the athlete keypoints (trainer contact, low visibility), removes teleports
  above 8 body lengths per second, interpolates gaps of at most 0.2 s in time,
  smooths each joint coordinate with a time-aware One-Euro filter that restarts
  after every gap, and measures each clip's body length `L` as the sum of the
  90th-percentile torso, thigh and shin, so a split or a foreshortened frame
  cannot shrink the scale later stages normalise by. Output goes to
  `data/processed/<source>/` in the input's own long schema plus `x_raw`,
  `y_raw`, `valid` and `filled`, so `handstand.overlay` draws it unchanged and
  every later stage filters on two columns; `handstand.bodyframe` turns a
  position into the athlete's own coordinates (origin at the wrist midpoint,
  `u` right, `v` up, in body lengths). Over the 180 MediaPipe clips: 178 usable,
  73 % of samples keep a position (89 % in hold-like frames), and the median
  frame-to-frame displacement of the wrists and ankles drops from 0.0063 to
  0.0014 body lengths (#20)
- Agent pre-labels for the sampled frames: `uv run python -m handstand.prelabel`
  runs RTMPose (via `rtmlib`, deliberately a model outside the bake-off) over
  every frame in the labelling manifest, upright and rotated, and writes
  `data/label_studio_prelabels.json` (Label Studio tasks carrying a `predictions`
  entry) plus `data/labels/review_queue.csv`, which sorts the frames whose
  pre-labels most likely need correcting to the top with the reason for each
  (#77). The athlete is picked by **keypoint** agreement with the MediaPipe
  athlete, not bounding-box overlap: a crouching trainer's box can overlap the
  athlete's more than the athlete's own inverted skeleton does, which pre-labelled
  the trainer on 20 of 300 frames. Keypoint matching plus writing no pre-label at
  all where the reference rejects every detection brings that to 0/300, and those
  19 no-pre-label frames lead the review queue instead of sinking to the bottom —
  they are identifiable inside the specified columns by `low_score_joints`
  listing all 15 joints (#77)
- Athlete selection for the Apple Vision keypoints: `handstand.athlete --source
  vision` runs the same athlete and trainer-contact rules over `vision_multi/`
  and writes `vision_athlete/` in the same schema, so the bake-off compares the
  two models after the same selection; `handstand.overlay --source
  vision_athlete` draws it (#76)
- Apple Vision body-pose keypoints as a second pose model: a Swift CLI that runs
  on the Mac mini in the same schema as MediaPipe, a script that drives it from
  the workstation, and a CSV -> parquet import (`tools/mac/run_vision.sh`, then
  `uv run python -m handstand.vision_import`) (#15)
- Frame sampler + Label Studio setup for keypoint labelling: a stratified sample
  of frames to label, the labelling config, and an importer that turns a Label
  Studio export into display-frame keypoint pixels (#65)
- Catalogue every handstand video with ffprobe into a CSV to annotate by hand
  (`uv run python -m handstand.catalogue`), safe to re-run without losing
  annotations (#8)

### Fixed
- 11 of the 19 labelling frames that got no pre-label now get one (chainlink
  #78): 6 because #79 made the MediaPipe reference read the handstand the right
  way up and the keypoints agree again (`438c3693d6d7_138`, `64184de33f84_1`,
  `651b0b5783cc` frames 3/18/34/49), 2 through the orientation fallback (both
  `5050dcb30e08` frames, whose reference is still upright — the `auto` trap #79
  could not break for that clip), and 3 because `mediapipe_athlete/best` detects
  no athlete in that frame at all, so there is nothing to contradict the most
  confident body (`13479d9e86a6_13`, `5a9611088218_19`, `a79591b7be18_64` —
  the last of those was called a correct refusal in the #78 diagnosis, so its
  pre-label is worth checking before it is imported). The 8 frames that stay
  empty are all bodies a reference reading the *same* way up puts 0.4 to 1.5
  body lengths away, or a body upright against an inverted reference: no
  orientation excuse to fall back on

### Changed
- Port Scorer + reference loading to Swift, with a Python→Swift scorer parity
  fixture: `golden.py` now writes `expected.score` per hold and the folder's
  `parity_reference.json` (synthetic holds, not a real reference), and
  `HandstandCore.Scorer` mirrors `handstand.score` over
  `ClipFeatures.tableRounded()` (#41)
- Epic: Dev infrastructure (repo, Python env, Mac mini build host) (#1)
- Weighted z-score scorer (#29)
- Port FeatureExtractor + CentreOfMass + hold summary to Swift (#81)
- Port PhaseSegmenter to Swift (phases only; features are #81) (#40)
- Port postprocess to Swift (gating, outliers, gap fill, One-Euro, body length) (#39)
- Golden test fixtures for Python->Swift parity (synthetic, committed; real clips local-only) (#25)
- Recording with a live framing guide (AVFoundation + Vision) in the iOS app (#46)
- iOS app scaffold (SwiftUI, XcodeGen) consuming HandstandCore, ready to sideload (#44)
- HandstandCore Swift package skeleton + swift test on the Mac mini (#38)
- Center of mass (CoM) proxy, balance direction and hold stability features (#23)
- Investigate the 19 frames where the pre-label model found no athlete (#78)
- MediaPipe auto-rotation gets stuck reading inverted bodies upside down (#79)
- The multi-person pipeline defaults to `--rotate best` (chainlink #79):
  `pose_mediapipe` with `--num-poses > 1`, and `handstand.athlete`,
  `handstand.postprocess` and `handstand.trainer_report` (phases and features
  read the post-processed trajectory, which has no rotation mode of its own).
  `none`, `180`, `auto` and `best` all stay selectable, a single-person
  `pose_mediapipe` run still defaults to `none`, and the modes other than
  `best` are byte-identical to what they wrote before
- FeatureExtractor: stacking offsets, joint angles, line and leg-shape features per frame (#22)
- PhaseSegmenter: pre / kick-up / hold / exit / post per frame, with hold segments (#21)
- Keypoint post-processing: gating, outlier removal, gap fill, One-Euro smoothing, body length (#20)
- OrientationNormalizer: inversion detection, rotate/unrotate (#19)
- Agent pre-labels for the 300 frames (independent model) + review queue (#77)
- Agent pre-labels for the labelling sample (RTMPose, not a bake-off model) (#77)
- Apple Developer account decision (#6)
- Data storage and sync strategy for videos (#5)
- Check WhatsApp compression impact; recover originals if possible (#9)
- Write recording protocol for new clips (#11)
- Athlete selection for Vision multi-person keypoints (--source vision) (#76)
- Athlete selection over the Apple Vision keypoints (`--source vision`, unchanged output for `mediapipe`) (#76)
- Apple Vision body-pose runner (Swift CLI on the Mac mini) + import to keypoint schema (#15)
- Frame sampler + keypoint labeling setup (Label Studio) (#65)
- Refine bone-length contact rule: flag lengthened bones only (#70)
- Quantify trainer presence across the whole dataset (#69)
- Athlete selection + trainer-contact flag (per-frame multi-person detection) (#68)
- Multi-person keypoints: run MediaPipe with up to 3 poses per frame (#67)
- Overlay renderer: draw keypoints on a clip, single or side by side (#24)
- MediaPipe Pose Landmarker (Full) runner with optional 180 deg rotation (#14)
- Clip catalogue script (ffprobe metadata -> CSV with blank annotation columns) (#8)
- Python pipeline environment (#4)
- Create git monorepo and GitHub remote (#2)
- Set up Mac mini M1 as remote build host (#3)
