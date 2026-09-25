"""Download the MediaPipe Pose Landmarker (Full) model into ``pipeline/models/``.

The model file is a build artifact: ``pipeline/models/`` and ``*.task`` are
git-ignored, so every machine runs this once. Existing files are kept.

Usage::

    cd pipeline
    uv run python scripts/download_models.py [--force]
"""

from __future__ import annotations

import argparse
import pathlib
import urllib.error
import urllib.request

from handstand.pose_mediapipe import DEFAULT_MODEL_PATH, MODEL_URL

_CHUNK_BYTES = 1 << 20


def download_model(dest: pathlib.Path = DEFAULT_MODEL_PATH, *, force: bool = False) -> pathlib.Path:
    """Fetch ``MODEL_URL`` into ``dest`` unless it is already there.

    The download goes to a ``.part`` file first and is renamed only once it
    completed, so an interrupted run never leaves a truncated model behind.
    """
    if dest.is_file() and not force:
        print(f"already present: {dest} ({dest.stat().st_size} bytes)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    print(f"downloading {MODEL_URL}")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        print(f"\r  {written / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
                    else:
                        print(f"\r  {written / 1e6:.1f} MB", end="")
            print()
        if total and written != total:
            raise RuntimeError(f"incomplete download: {written} of {total} bytes")
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f"saved {dest} ({dest.stat().st_size} bytes)")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even when the model file already exists",
    )
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=DEFAULT_MODEL_PATH,
        help=f"where to put the model (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args(argv)
    try:
        download_model(args.dest, force=args.force)
    except (urllib.error.URLError, OSError, RuntimeError) as error:
        print(f"download failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
