# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Catalogue every handstand video with ffprobe into a CSV to annotate by hand
  (`uv run python -m handstand.catalogue`), safe to re-run without losing
  annotations (#8)

### Fixed

### Changed
- MediaPipe Pose Landmarker (Full) runner with optional 180 deg rotation (#14)
- Clip catalogue script (ffprobe metadata -> CSV with blank annotation columns) (#8)
- Python pipeline environment (#4)
- Create git monorepo and GitHub remote (#2)
- Set up Mac mini M1 as remote build host (#3)
