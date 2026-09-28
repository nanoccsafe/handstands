# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
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

### Changed
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
