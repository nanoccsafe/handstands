"""Catalogue every handstand video: ffprobe metadata into a CSV the user annotates by hand.

Run it from the pipeline directory::

    uv run python -m handstand.catalogue [--videos DIR] [--out FILE]

The script owns the leading metadata columns and leaves the trailing annotation
columns empty for a human to fill in. It is safe to re-run: a second run refreshes
the metadata, keeps every annotation already written by hand, adds clips that
appeared since, and keeps rows whose file disappeared with ``missing`` set to
true. The CSV is written atomically, so an interrupted run cannot destroy
annotations that took an hour to type.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from handstand import paths

__all__ = [
    "ANNOTATION_COLUMNS",
    "COLUMNS",
    "METADATA_COLUMNS",
    "MISSING_COLUMN",
    "Catalogue",
    "FilenameStamp",
    "Probe",
    "Row",
    "build_catalogue",
    "build_row",
    "display_dimensions",
    "find_clips",
    "is_variable_frame_rate",
    "main",
    "merge_rows",
    "normalise_row",
    "parse_filename",
    "parse_frame_rate",
    "probe_media",
    "read_rows",
    "sort_rows",
    "summarise",
    "write_rows",
]

#: A catalogue row: every column in :data:`COLUMNS`, as strings ready for CSV.
Row = dict[str, str]

#: Columns the script fills from ffprobe and the filename. Order is part of the contract.
METADATA_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "filename",
    "session_date",
    "recorded_at",
    "duration_s",
    "fps_avg",
    "fps_nominal",
    "is_vfr",
    "width",
    "height",
    "rotation",
    "display_width",
    "display_height",
    "bitrate_kbps",
    "codec",
)

#: Columns a human fills in. The script only ever writes "" into them.
ANNOTATION_COLUMNS: tuple[str, ...] = (
    "camera_angle",
    "full_body",
    "skill",
    "outcome",
    "hold_s",
    "good_clip",
    "notes",
)

#: Trailing column flagging a row whose video file is no longer in the videos dir.
MISSING_COLUMN = "missing"

#: Full CSV header, in order. Always the same, so a re-run never shifts a column.
COLUMNS: tuple[str, ...] = METADATA_COLUMNS + ANNOTATION_COLUMNS + (MISSING_COLUMN,)

#: Hex characters kept from the SHA-1 of a clip's bytes.
CLIP_ID_LENGTH = 12

#: The average frame rate must differ from the nominal rate by more than this to be VFR.
VFR_TOLERANCE = 0.005

#: Decimal places used for the float columns.
FLOAT_DIGITS = 3

#: ffprobe binary, overridable for tests and unusual installs.
FFPROBE = "ffprobe"

#: Seconds to wait for a single ffprobe call before giving up on that file.
PROBE_TIMEOUT_S = 120

#: Chunk size for hashing clip bytes.
HASH_CHUNK_BYTES = 1 << 20

#: Mode of the written catalogue: readable by whoever shares the workspace, not private.
FILE_MODE = 0o644

TRUE = "true"
FALSE = "false"

_FILENAME_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2}) at "
    r"(?P<hour>\d{1,2})\.(?P<minute>\d{2})\.(?P<second>\d{2}) "
    r"(?P<meridiem>AM|PM)"
)

#: One ``ffprobe`` call for one file; injected in tests instead of shelling out.
Probe = Callable[[pathlib.Path], Mapping[str, Any]]


class FilenameStamp(NamedTuple):
    """What a filename says about when a clip was recorded.

    Both fields are ``""`` when the name does not carry a readable timestamp.
    """

    session_date: str
    recorded_at: str


class Catalogue(NamedTuple):
    """What a catalogue run produced: the rows to write and the files that failed."""

    rows: list[Row]
    failures: list[tuple[str, str]]


def _hour24(hour12: int, meridiem: str) -> int | None:
    """Convert a 12-hour clock reading to 0-23, or return None if it is not one."""
    if not 1 <= hour12 <= 12:
        return None
    if meridiem == "AM":
        return 0 if hour12 == 12 else hour12
    return hour12 if hour12 == 12 else hour12 + 12


def parse_filename(name: str) -> FilenameStamp:
    """Read the recording date and time out of a clip filename.

    Handles the WhatsApp export shape ``WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4``
    and the ``(1)`` duplicate suffix iOS appends (``... PM (1).mp4``): the timestamp
    may sit anywhere in the stem. ``7.09.23 PM`` becomes ``2026-07-22T19:09:23``,
    ``12.02.37 PM`` stays ``12:02:37`` and ``12.15.00 AM`` becomes ``00:15:00``.
    Names with no usable timestamp, or an impossible one, give empty strings.
    """
    match = _FILENAME_RE.search(pathlib.Path(name).stem)
    if match is None:
        return FilenameStamp("", "")
    hour = _hour24(int(match["hour"]), match["meridiem"])
    if hour is None:
        return FilenameStamp("", "")
    date = match["date"]
    try:
        stamp = dt.datetime(
            int(date[0:4]),
            int(date[5:7]),
            int(date[8:10]),
            hour,
            int(match["minute"]),
            int(match["second"]),
        )
    except ValueError:
        return FilenameStamp("", "")
    return FilenameStamp(stamp.date().isoformat(), stamp.isoformat())


def display_dimensions(width: int, height: int, rotation: int) -> tuple[int, int]:
    """Return the size a viewer sees: a quarter-turn rotation swaps width and height.

    Phones store portrait clips in a landscape buffer plus a display matrix, so a
    ``1024x576`` clip flagged ``rotation=-90`` is really ``576x1024`` on screen.
    """
    if rotation % 180 == 90:
        return height, width
    return width, height


def _as_float(value: object) -> float | None:
    """Coerce an ffprobe field to a finite float, or None when it is absent or junk."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: object) -> int:
    """Coerce a CSV cell to an int, 0 when it is blank or has been edited into junk.

    Annotation sheets get sorted and re-typed in a spreadsheet, so a cell the script
    wrote is not always a cell the script can still read back.
    """
    number = _as_float(value)
    return int(number) if number is not None else 0


def parse_frame_rate(value: object) -> float | None:
    """Parse an ffprobe frame rate such as ``"30/1"`` or ``"29280/979"`` into fps.

    WhatsApp clips are variable frame rate: the nominal rate is a clean ``30/1`` while
    the average carries the real, drifting rate. Returns None for ``"0/0"``, ``"N/A"``
    and anything else ffprobe uses for "unknown".
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if "/" not in text:
        return _as_float(text)
    numerator, _, denominator = text.partition("/")
    top, bottom = _as_float(numerator), _as_float(denominator)
    if top is None or bottom is None or bottom == 0:
        return None
    return top / bottom


def is_variable_frame_rate(fps_avg: float | None, fps_nominal: float | None) -> bool:
    """True when the average frame rate drifts from the nominal rate by over 0.5%."""
    if fps_avg is None or fps_nominal is None or fps_nominal <= 0:
        return False
    return abs(fps_avg - fps_nominal) / fps_nominal > VFR_TOLERANCE


def _format_float(value: float | None) -> str:
    """Format a float column rounded to :data:`FLOAT_DIGITS`; unknown stays empty."""
    if value is None:
        return ""
    return f"{value:.{FLOAT_DIGITS}f}"


def _format_bool(value: bool) -> str:
    """Format a flag column as lowercase ``true``/``false``."""
    return TRUE if value else FALSE


def file_clip_id(path: pathlib.Path, *, length: int = CLIP_ID_LENGTH) -> str:
    """Return a short id for a clip: the first ``length`` hex chars of its SHA-1.

    The id comes from the bytes, not the name, so re-exporting or renaming a clip
    keeps its row — and the annotations typed against it — in the catalogue.
    """
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def probe_media(path: pathlib.Path) -> Mapping[str, Any]:
    """Return ffprobe's JSON report for one file, in a single subprocess call.

    Raises ``subprocess.CalledProcessError`` when ffprobe is unhappy, which
    :func:`build_catalogue` turns into a warning rather than losing the whole run.
    """
    completed = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=PROBE_TIMEOUT_S,
    )
    return json.loads(completed.stdout)


def _video_stream(info: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the first video stream of an ffprobe report, or an empty mapping."""
    for stream in info.get("streams") or []:
        if isinstance(stream, Mapping) and stream.get("codec_type") == "video":
            return stream
    return {}


def _stream_rotation(stream: Mapping[str, Any]) -> int:
    """Return the display-matrix rotation of a stream, 0 when the clip carries none."""
    for side_data in stream.get("side_data_list") or []:
        if isinstance(side_data, Mapping) and side_data.get("rotation") is not None:
            return int(side_data["rotation"])
    return 0


def _identity_row(path: pathlib.Path) -> Row:
    """The part of a row the filesystem alone can fill in, including ``clip_id``."""
    stamp = parse_filename(path.name)
    row: Row = {
        "clip_id": file_clip_id(path),
        "filename": path.name,
        "session_date": stamp.session_date,
        "recorded_at": stamp.recorded_at,
    }
    for column in METADATA_COLUMNS:
        row.setdefault(column, "")
    return row


def build_row(path: pathlib.Path, probe: Probe | None = None) -> Row:
    """Probe one clip and return its metadata row, without the annotation columns.

    The row is a plain dict of strings keyed by :data:`METADATA_COLUMNS`, so a clip
    ffprobe cannot read degrades to :func:`_identity_row` instead of vanishing.
    """
    info = (probe or probe_media)(path)
    stream = _video_stream(info)
    container = info.get("format") or {}
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    rotation = _stream_rotation(stream)
    display_width, display_height = display_dimensions(width, height, rotation)
    fps_avg = parse_frame_rate(stream.get("avg_frame_rate"))
    fps_nominal = parse_frame_rate(stream.get("r_frame_rate"))
    bitrate = _as_float(container.get("bit_rate"))
    row = _identity_row(path)
    row.update(
        {
            "duration_s": _format_float(_as_float(container.get("duration"))),
            "fps_avg": _format_float(fps_avg),
            "fps_nominal": _format_float(fps_nominal),
            "is_vfr": _format_bool(is_variable_frame_rate(fps_avg, fps_nominal)),
            "width": str(width),
            "height": str(height),
            "rotation": str(rotation),
            "display_width": str(display_width),
            "display_height": str(display_height),
            "bitrate_kbps": "" if bitrate is None else str(round(bitrate / 1000)),
            "codec": str(stream.get("codec_name") or ""),
        }
    )
    return row


def find_clips(videos_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the ``*.mp4`` clips in ``videos_dir``, sorted by filename.

    Images sitting next to the videos (``.jpeg`` stills) are ignored. The directory
    is only read; raw media is never written to.
    """
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"videos dir does not exist: {videos_dir}")
    return sorted(path for path in videos_dir.glob("*.mp4") if path.is_file())


def normalise_row(
    row: Mapping[str, str],
    annotations: Mapping[str, str] | None = None,
    *,
    missing: bool = False,
) -> Row:
    """Return ``row`` with every column in :data:`COLUMNS` present and in order.

    ``annotations``, when given, is a previously written row whose human columns win
    over the script's — that is the whole mechanism for never losing an annotation.
    """
    out: Row = {column: row.get(column) or "" for column in COLUMNS}
    out[MISSING_COLUMN] = _format_bool(missing)
    if annotations is not None:
        for column in ANNOTATION_COLUMNS:
            out[column] = annotations.get(column) or ""
    return out


def sort_rows(rows: Iterable[Row]) -> list[Row]:
    """Order rows by recording time, then filename.

    ISO timestamps sort chronologically as text. Rows whose name carries no timestamp
    have an empty ``recorded_at`` and therefore sort first.
    """
    return sorted(rows, key=lambda row: (row.get("recorded_at", ""), row.get("filename", "")))


def merge_rows(fresh: Sequence[Row], existing: Sequence[Row]) -> list[Row]:
    """Merge freshly probed rows with a catalogue written earlier.

    Annotation values win for every ``clip_id`` present in both. Clips that appeared
    since the last run are added with empty annotations. Clips whose file has
    disappeared keep their row — annotations included — with ``missing`` set to true.
    A row whose ``clip_id`` is blank cannot match a clip either, so it is kept as a
    missing row rather than dropped on the floor with its annotations.
    """
    saved = {row["clip_id"]: row for row in existing if row.get("clip_id")}
    merged = [normalise_row(row, saved.get(row.get("clip_id") or "")) for row in fresh]
    present = {row.get("clip_id") or "" for row in fresh}
    merged += [
        normalise_row(row, row, missing=True)
        for clip_id, row in sorted(saved.items())
        if clip_id not in present
    ]
    merged += [normalise_row(row, row, missing=True) for row in existing if not row.get("clip_id")]
    return sort_rows(merged)


def read_rows(path: pathlib.Path) -> list[Row]:
    """Read a catalogue written earlier; a missing file is simply an empty one."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(rows: Sequence[Row], out: pathlib.Path) -> None:
    """Write the catalogue to ``out`` atomically: temp file beside it, then rename.

    A reader either sees the previous catalogue or the new one, never a half-written
    file, so annotations cannot be lost to a crash mid-write.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    temp_path = pathlib.Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        # mkstemp is deliberately private; the catalogue is a shared, hand-edited file.
        os.chmod(temp_path, FILE_MODE)
        temp_path.replace(out)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def build_catalogue(
    videos_dir: pathlib.Path,
    out: pathlib.Path,
    *,
    probe: Probe | None = None,
) -> Catalogue:
    """Catalogue every clip in ``videos_dir`` and write the merged CSV to ``out``.

    A clip ffprobe cannot read is reported in ``failures`` and still gets a row with
    its identity, so nothing silently disappears from the catalogue.
    """
    read_probe = probe or probe_media
    fresh: list[Row] = []
    failures: list[tuple[str, str]] = []
    for path in find_clips(videos_dir):
        try:
            fresh.append(build_row(path, read_probe))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            fresh.append(_identity_row(path))
            failures.append((path.name, f"{type(error).__name__}: {error}"))
    rows = merge_rows(fresh, read_rows(out))
    write_rows(rows, out)
    return Catalogue(rows=rows, failures=failures)


def summarise(catalogue: Catalogue) -> str:
    """Multi-line summary of a run, for the CLI to print."""
    rows = catalogue.rows
    present = [row for row in rows if row[MISSING_COLUMN] != TRUE]
    vfr = sum(row["is_vfr"] == TRUE for row in present)
    rotated = sum(_as_int(row["rotation"]) % 180 == 90 for row in present)
    dated = sum(bool(row["recorded_at"]) for row in present)
    return "\n".join(
        [
            f"wrote {len(rows)} rows",
            f"  clips present: {len(present)} ({len(rows) - len(present)} missing, "
            f"flagged in the {MISSING_COLUMN} column)",
            f"  variable frame rate: {vfr}",
            f"  rotated: {rotated}",
            f"  dated from filename: {dated} of {len(present)}",
            f"  ffprobe failures: {len(catalogue.failures)}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.catalogue``."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.catalogue",
        description="Catalogue handstand videos into a CSV to annotate by hand.",
    )
    parser.add_argument(
        "--videos",
        type=pathlib.Path,
        default=None,
        help="videos to catalogue (default: $HANDSTAND_VIDEOS, else the workspace videos dir)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="CSV to write; a re-run keeps the annotations already in it "
        "(default: $HANDSTAND_DATA/catalogue.csv)",
    )
    args = parser.parse_args(argv)

    videos = (args.videos or paths.videos_dir()).expanduser()
    out = (args.out or paths.data_dir() / "catalogue.csv").expanduser()
    if not videos.is_dir():
        print(f"catalogue: not a videos directory: {videos}", file=sys.stderr)
        return 2
    catalogue = build_catalogue(videos, out)
    for filename, reason in catalogue.failures:
        print(f"warning: ffprobe failed on {filename}: {reason}", file=sys.stderr)
    print(f"{videos} -> {out}")
    print(summarise(catalogue))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
