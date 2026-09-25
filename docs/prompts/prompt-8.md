# Task: chainlink issue #8 — Clip catalogue script

Write a CLI that catalogues every handstand video with ffprobe and produces a CSV the user will annotate by hand.

## Where
- Module: `pipeline/handstand/catalogue.py`, runnable as `cd pipeline && uv run python -m handstand.catalogue`.
- Tests: `pipeline/tests/test_catalogue.py`.
- Use `handstand.paths.videos_dir()` for input and `handstand.paths.data_dir() / "catalogue.csv"` as the default output.
  CLI flags: `--videos DIR`, `--out FILE` to override both.

## Input facts (verified)
- Files: `*.mp4` in the videos dir (ignore `.jpeg` images). ~180 files named like
  `WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4`; some names have a suffix like ` (1)` before `.mp4`.
- Example ffprobe stream: `codec_name=h264 width=1024 height=576 r_frame_rate=30/1 avg_frame_rate=29280/979`
  with display-matrix side data `rotation=-90`. So videos are stored landscape with a rotation flag and have a
  VARIABLE frame rate.
- Get metadata with one subprocess call per file:
  `ffprobe -v error -print_format json -show_format -show_streams <file>` and parse the JSON
  (rotation is in `streams[i].side_data_list[j].rotation`, may be absent).

## Output: CSV columns, in this order
Metadata (script fills): `clip_id, filename, session_date, recorded_at, duration_s, fps_avg, fps_nominal, is_vfr,
width, height, rotation, display_width, display_height, bitrate_kbps, codec`
- `clip_id`: first 12 hex chars of the SHA-1 of the file bytes (stable across renames).
- `session_date`: `YYYY-MM-DD` parsed from the filename; `recorded_at`: ISO datetime from the filename
  (`7.09.23 PM` -> `19:09:23`). Empty if the name does not match.
- `fps_avg` from avg_frame_rate, `fps_nominal` from r_frame_rate (rounded to 3 decimals);
  `is_vfr` = true when they differ by more than 0.5%.
- `display_width/height` = width/height swapped when rotation is ±90 or ±270.
Annotation (script leaves EMPTY): `camera_angle, full_body, skill, outcome, hold_s, good_clip, notes`

Rows sorted by `recorded_at`, then filename.

## Re-run behaviour (important)
If the output CSV already exists, load it and keep every annotation column value for rows whose `clip_id`
still exists; refresh metadata columns; add new clips with empty annotations; keep rows for clips whose file
disappeared but set a `missing` column to true (add that column last). Never lose a human annotation.
Write atomically (temp file + rename).

## Tests (must not use the real videos)
- filename parsing: AM/PM, 12 AM/12 PM edge cases, ` (1)` suffix, non-matching name.
- rotation -> display dims.
- merge preserves annotations, marks missing clips, adds new ones (build DataFrames/CSVs in tmp_path;
  mock the ffprobe call).
- one integration test that generates a 1-second video with `ffmpeg -f lavfi -i testsrc=size=64x48:rate=10`
  into tmp_path and runs the full catalogue on it (skip if ffmpeg is not installed).

## Acceptance
- `cd pipeline && uv run pytest && uv run ruff check` pass.
- Running the CLI on the real videos dir finishes in under 1 minute and writes the CSV to the data dir
  (do run it once; do NOT commit the CSV). Report the row count and how many clips are VFR / rotated.

Follow AGENTS.md.
