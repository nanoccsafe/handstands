# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
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
