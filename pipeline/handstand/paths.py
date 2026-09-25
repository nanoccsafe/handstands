"""Locations of the shared workspace: raw videos (read-only) and derived data.

Both are overridable through environment variables so the same code runs on the
Linux workstation, on the Mac mini and in tests without editing paths.

* ``HANDSTAND_DATA``   overrides the derived-data directory (writes).
* ``HANDSTAND_VIDEOS`` overrides the raw-videos directory (reads only).
"""

from __future__ import annotations

import os
import pathlib

__all__ = ["DATA_DIR", "VIDEOS_DIR", "data_dir", "videos_dir"]

#: Shared workspace layout on the Linux workstation.
DEFAULT_DATA_DIR = "/mnt/sharedOs/handstand-workspace/data"
DEFAULT_VIDEOS_DIR = "/mnt/sharedOs/handstand-workspace/videos"

#: Environment variables that override the defaults above.
DATA_ENV_VAR = "HANDSTAND_DATA"
VIDEOS_ENV_VAR = "HANDSTAND_VIDEOS"

#: Snapshot of the defaults, for callers that want the un-overridden values.
DATA_DIR = pathlib.Path(DEFAULT_DATA_DIR)
VIDEOS_DIR = pathlib.Path(DEFAULT_VIDEOS_DIR)


def data_dir() -> pathlib.Path:
    """Return the directory holding derived data (keypoints, features, models).

    Uses ``$HANDSTAND_DATA`` when set and non-empty, otherwise
    ``/mnt/sharedOs/handstand-workspace/data``. The directory is not created.
    """
    override = os.environ.get(DATA_ENV_VAR)
    if override:
        return pathlib.Path(override).expanduser()
    return DATA_DIR


def videos_dir() -> pathlib.Path:
    """Return the directory holding raw videos, which is never written to.

    Uses ``$HANDSTAND_VIDEOS`` when set and non-empty, otherwise
    ``/mnt/sharedOs/handstand-workspace/videos``. The directory is not created.
    """
    override = os.environ.get(VIDEOS_ENV_VAR)
    if override:
        return pathlib.Path(override).expanduser()
    return VIDEOS_DIR
