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
