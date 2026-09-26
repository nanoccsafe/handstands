# `handstand` — Python reference pipeline

Pose extraction, biomechanical features, phase segmentation, scoring and
classifier training for the handstand form analysis project. This is the
reference implementation; the on-device Swift port in `../swift/HandstandCore`
is checked against it via parity fixtures.

## Requirements

- Python 3.12 (`>=3.12,<3.13`)
- [uv](https://docs.astral.sh/uv/) — all dependency handling goes through it, never `pip`
- Shared workspace mounted at `/mnt/sharedOs/handstand-workspace` (see *Paths* below)

## Setup

```sh
cd pipeline
uv sync
```

`uv sync` installs the package in editable mode into `.venv/` using the locked
`uv.lock`, so local edits take effect without reinstalling.

To add a dependency, use uv (never `pip`) and commit the regenerated lock file:

```sh
uv add <package>          # runtime dependency
uv add --dev <package>    # development dependency (pytest, ruff, ...)
```

## Test

```sh
cd pipeline
uv run pytest
```

Tests use tiny synthetic inputs only — they must never read the real videos in
`/mnt/sharedOs/handstand-workspace/videos`. Add new tests under `tests/`.

## Lint

```sh
cd pipeline
uv run ruff check
uv run ruff format .
```

## Catalogue

The entry point of the pipeline: one CSV row per video, ffprobe metadata filled in
and the judgement calls left for a human.

```sh
cd pipeline
uv run python -m handstand.catalogue              # -> $HANDSTAND_DATA/catalogue.csv
uv run python -m handstand.catalogue --videos DIR --out FILE
```

The script fills `clip_id, filename, session_date, recorded_at, duration_s, fps_avg,
fps_nominal, is_vfr, width, height, rotation, display_width, display_height,
bitrate_kbps, codec` and leaves `camera_angle, full_body, skill, outcome, hold_s,
good_clip, notes` for you, sorted by recording time. `clip_id` is the first 12 hex
characters of the file's SHA-1, so a renamed clip keeps its row and its annotations.

Re-running is the normal case: metadata is refreshed from the files, annotations
already in the sheet are kept, clips that appeared since are added with blank
annotations, and rows whose file has disappeared are kept with `missing` set to
true. The sheet is written atomically (temp file plus rename), so a failed or
interrupted run never damages what you have typed.

## Trainer report

How much of the dataset the trainer is in: one row per catalogue clip, plus a
summary of the whole dataset.

```sh
cd pipeline
uv run python -m handstand.trainer_report                # report on existing keypoints
uv run python -m handstand.trainer_report --generate     # generate what is missing, then report
# -> $HANDSTAND_DATA/reports/trainer_report.csv, $HANDSTAND_DATA/reports/trainer_report.md
```

`--generate` runs the multi-person keypoints and the athlete selection for every
catalogue clip that is missing them (skipping the ones that have them, so an
interrupted run is resumed by running it again); without it the report is instant.
Per clip it reports frames, fps, the share of frames a second person was in, the
share flagged as trainer contact, the longest unbroken stretch of contact in
seconds, whether a second person was there for at least half a second in total,
the share of frames the selection dropped, and the catalogue's `notes`. The
markdown summary turns that into the dataset-level answer. Both files are derived
data and are never committed. See `docs/keypoint_schema.md` for every column.

## Paths

`handstand.paths` resolves the shared workspace at call time (never at import
time), so tests and the Mac/CI runners can redirect it:

| Function | Env var | Default |
|---|---|---|
| `handstand.paths.data_dir()` | `HANDSTAND_DATA` | `/mnt/sharedOs/handstand-workspace/data` |
| `handstand.paths.videos_dir()` | `HANDSTAND_VIDEOS` | `/mnt/sharedOs/handstand-workspace/videos` |

Neither function creates the directory. `videos_dir()` is read-only: raw media
never goes into git and is never written to.

```sh
export HANDSTAND_DATA=/scratch/handstand-data
export HANDSTAND_VIDEOS=/mnt/other/handstand-videos
```

## Layout

```
pipeline/
├── pyproject.toml     # project metadata, deps, pytest + ruff config
├── uv.lock            # pinned resolution, committed
├── handstand/         # the package (flat layout, ships py.typed)
├── tests/             # pytest suite
├── models/            # trained artifacts — git-ignored, never committed
└── README.md
```

Trained models and CoreML packages are build outputs: they are ignored by the
repo `.gitignore` and, if they ever need sharing, are published via releases.
