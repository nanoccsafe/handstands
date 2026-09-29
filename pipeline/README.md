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

## Rotation measure

What a rotation mode changed over the dataset, read off the keypoints and the
later stages' own outputs. Nothing is run and nothing is written: it is how the
numbers in `docs/keypoint_schema.md` are produced, so they can be produced again.

```sh
cd pipeline
uv run python -m handstand.orient_measure                 # best against auto, on the athlete pick
uv run python -m handstand.orient_measure --pick multi     # on every person of the frame
uv run python -m handstand.orient_measure --clips 438c3693d6d7 --no-chain
```

It reports the orientation each mode used and how often it changed (the flicker),
the frames each mode read as a handstand, the frames one mode fixed and broke
against the other with the run lengths of those changes, the clips that moved,
and — from `processed/`, `phases/` and `reports/trainer_report.csv` — the usable
clips and frames, the holds and the trainer presence the chain found.

## Keypoint post-processing

The trajectory every later stage measures: the athlete keypoints gated, de-spiked,
gap-filled and One-Euro smoothed, plus each clip's body length.

```sh
cd pipeline
uv run python -m handstand.postprocess --all              # -> $HANDSTAND_DATA/processed/mediapipe/
uv run python -m handstand.postprocess --source vision --all
uv run python -m handstand.postprocess --clips 6508f9b355bd --overwrite
```

Reads `keypoints/mediapipe_athlete/best/` (`--source vision` reads
`vision_athlete/`; `--rotate auto` reads the `auto` keypoints the dataset was
first measured with) and writes one parquet and one sidecar per clip under
`$HANDSTAND_DATA/processed/<source>/`. The parquet keeps the input's long schema
and rows, with `x`/`y` processed and NaN where there is no position, plus
`x_raw`/`y_raw` (what the model said), `valid` and `filled` — so
`handstand.overlay` draws it unchanged and a later stage filters on two columns.
The sidecar records the clip's body length, how many samples survived each step
and the jitter removed; the run prints the dataset-level summary. The thresholds
are module constants, and the sidecar records the ones a run used.
`handstand.bodyframe` turns a position into the athlete's own coordinates
(origin at the wrist midpoint, `u` right, `v` up, in body lengths). See
`docs/keypoint_schema.md` for every column and every rule.

## Phases

Which part of a clip each frame belongs to: `pre`, `kickup`, `hold`, `exit`,
`post` or `unknown`, with every hold numbered. Scoring reads only the `hold`
frames; hand steps (#71) and faults (#32) are events inside one.

```sh
cd pipeline
uv run python -m handstand.phases --all              # -> $HANDSTAND_DATA/phases/mediapipe/
uv run python -m handstand.phases --source vision --all
uv run python -m handstand.phases --clips 057c9e6c96af --overwrite
uv run python -m handstand.phases --clips 057c9e6c96af --render 057c9e6c96af
```

Reads `processed/<source>/` and writes one parquet per clip — a row per frame
with `phase`, `hold_id` and every signal the labels came from — plus
`segments.csv`, one row per phase run for the whole source. `--render` draws the
phase and the hold number on every frame of the clip, which is the only way to
see whether the boundaries landed where they should.

It is rule-based and readable, on purpose: a hold is `inverted` (the ankle
midpoint more than 0.6 body lengths above the wrist midpoint in the body frame)
and straight (within 35° of vertical) and `hands_down` (the hands are the
support and not moving), sustained for at least 0.3 s; a wrist that moves more
than 0.1 body lengths over 0.2 s is a hand step and ends it, while a stretch of
up to 0.3 s of unknown frames inside a hold does not. The thresholds are module
constants, documented at the top of `handstand.phases`. See
`docs/keypoint_schema.md` for every column and the whole state machine.

## Golden fixtures

The test data the Swift ports are checked against: synthetic clips, run through
the real `postprocess`, `phases` and `features` stages in memory, with input and
expected output written to one JSON file per case under
`swift/HandstandCore/Tests/HandstandCoreTests/Fixtures/golden/` (committed; the
format is documented in that folder's `README.md`).

```sh
cd pipeline
uv run python -m handstand.golden                    # regenerate every case
uv run python -m handstand.golden --only line_hold   # one of them
uv run python -m handstand.golden --real 6508f9b355bd   # local-only, never under swift/
```

The repo is public, so what is committed is always synthetic
(`meta.mode == "synthetic"`, generated from a seed) and `--real` refuses any
output path git would commit, plus `swift/`, `ios/`, `pipeline/` and `docs/`
outright, so real-clip fixtures only ever land in `<data_dir>/golden_real/`,
which is git-ignored. The data tree may sit *inside* the checkout, as it does in
the main one, and git ignoring it is exactly what allows that.
`tests/test_golden.py` reads the committed files back, re-runs the pipeline from
their `input` and checks the answers still match `expected` to the tolerances
stored in `meta`.

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
