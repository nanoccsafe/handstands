"""Split every hold into per-shape segments, so the clean line part of a flow can be used.

A *hold* (``hold_summary.csv``) is one continuous inverted period, so a flow
(split to line, tuck to line) is ONE hold whose window mixes shapes. The hold
shape (:mod:`handstand.hold_shapes`) is a single label per hold; this module
finds the shape segments *inside* every hold and summarises each segment like a
hold, so a later stage (the reference, the scoring) can use the line part of a
mixed hold without the straddle or the tuck around it.

Phase 1 only: detection, a segment table, a report and a CLI. Nothing
downstream reads the table yet, and nothing existing changes: the hold ids,
``hold_summary.csv``, :func:`handstand.hold_shapes.predict_shape` and the
feature measurements are untouched.

The rule, per hold, over its **valid** frames in frame order:

1. A frame's shape is :func:`handstand.hold_shapes.predict_shape` applied to that
   frame's ``leg_separation``, ``knee_angle`` and ``hip_angle`` (the function is
   reused, its thresholds are not copied).
2. Runs are consecutive frames with the same shape. An
   :data:`handstand.hold_shapes.UNMEASURED` frame never starts a run: it joins
   the run in progress, and the ones before the first measured frame join the
   first run. A hold with no measured frame is one ``unmeasured`` segment.
3. A run lasts from its first frame's ``t_ms`` to its last frame's.
4. While there is more than one run and the shortest (the first, on a tie) lasts
   less than :data:`MIN_SEGMENT_MS`, it is merged into its longer neighbour (the
   previous one when they are equal, the only one at an end): its frames take
   that shape and adjacent runs of the same shape are coalesced.
5. The runs left are the segments, numbered ``0, 1, ...`` in time order; a hold
   with more than one is ``mixed``.

A segment row mirrors a hold-summary row: the same ``<feature>_median`` and
``<feature>_iqr`` columns with the same rounding, taken over the segment's
*measurable* frames (:meth:`handstand.features.ClipFeatures.measurable_hold_mask`
inside the segment's time range).

Everything it writes is under ``data/`` and git-ignored.

CLI::

    cd pipeline
    uv run python -m handstand.segments build
    uv run python -m handstand.segments show <clip_id>
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from handstand import features, hold_shapes
from handstand.paths import data_dir
from handstand.postprocess import DEFAULT_SOURCE, SOURCES

__all__ = [
    "MIN_SEGMENT_MS",
    "SEGMENTS_NAME",
    "SEGMENT_COLUMNS",
    "build_segments",
    "clip_segments",
    "frame_shapes",
    "line_segments",
    "load_clip",
    "main",
    "read_segments",
    "report",
    "segment_runs",
    "segments_path",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: A shape run shorter than this (ms) is merged into its neighbour.
MIN_SEGMENT_MS = 1000

#: The per-segment table of one source, next to ``hold_summary.csv``.
SEGMENTS_NAME = "hold_segments.csv"

#: The report subdirectory under ``<data_dir>/reports/``, one file per build.
REPORT_DIRNAME = "segments"

#: The table's columns, in write order: the segment's identity and extent, then
#: the hold summary's median and IQR of every feature.
SEGMENT_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "source",
    "hold_id",
    "segment_id",
    "shape_guess",
    "mixed",
    "segment_frames",
    "valid_frames",
    "start_ms",
    "end_ms",
    "duration_s",
    *(
        f"{feature.name}_{statistic}"
        for feature in features.FEATURES
        for statistic in ("median", "iqr")
    ),
)

#: The shortest line segment (seconds) the report counts as usable.
LINE_MIN_S = 2.0

#: Decimal places of ``duration_s``, as in the hold summary.
_DURATION_DIGITS = 3

# --------------------------------------------------------------------------- #
# Per-frame shapes and the runs they form
# --------------------------------------------------------------------------- #


def frame_shapes(leg: np.ndarray, knee: np.ndarray, hip: np.ndarray) -> list[str]:
    """Each frame's shape: :func:`handstand.hold_shapes.predict_shape` on its own angles.

    A NaN in any of the three is :data:`handstand.hold_shapes.UNMEASURED`.
    """
    return [
        hold_shapes.predict_shape(
            {
                "leg_separation_median": float(leg_value),
                "knee_angle_median": float(knee_value),
                "hip_angle_median": float(hip_value),
            }
        )[0]
        for leg_value, knee_value, hip_value in zip(leg, knee, hip, strict=True)
    ]


def _initial_runs(shapes: Sequence[str]) -> list[list]:
    """``[shape, first_index, last_index]`` per run; unmeasured frames join the run in progress."""
    runs: list[list] = []
    for index, shape in enumerate(shapes):
        if not runs:
            runs.append([shape, index, index])
        elif shape == hold_shapes.UNMEASURED:
            runs[-1][2] = index
        elif runs[-1][0] == hold_shapes.UNMEASURED:
            # Leading unmeasured frames: the first measured shape names the first run.
            runs[-1][0] = shape
            runs[-1][2] = index
        elif runs[-1][0] == shape:
            runs[-1][2] = index
        else:
            runs.append([shape, index, index])
    return runs


def _coalesce(runs: list[list]) -> list[list]:
    """Merge adjacent runs of the same shape."""
    merged: list[list] = []
    for run in runs:
        if merged and merged[-1][0] == run[0]:
            merged[-1][2] = run[2]
        else:
            merged.append(run)
    return merged


def segment_runs(
    t_ms: Sequence[int],
    shapes: Sequence[str],
    min_ms: int = MIN_SEGMENT_MS,
) -> list[tuple[str, int, int]]:
    """The shape segments of one hold: ``(shape, start_ms, end_ms)`` in time order.

    ``t_ms`` and ``shapes`` are the hold's valid frames in frame order. Runs of the
    same shape are found, then the shortest run under ``min_ms`` is merged into its
    longer neighbour until none is left (see the module docstring). Pure: no I/O.
    """
    if len(t_ms) != len(shapes):
        raise ValueError(f"t_ms has {len(t_ms)} entries, shapes has {len(shapes)}")
    if not shapes:
        return []
    times = [int(value) for value in t_ms]

    def duration(run: list) -> int:
        return times[run[2]] - times[run[1]]

    runs = _initial_runs(shapes)
    while len(runs) > 1:
        shortest = min(range(len(runs)), key=lambda index: (duration(runs[index]), index))
        if duration(runs[shortest]) >= min_ms:
            break
        if shortest == 0:
            neighbour = 1
        elif shortest == len(runs) - 1:
            neighbour = shortest - 1
        else:
            before, after = duration(runs[shortest - 1]), duration(runs[shortest + 1])
            neighbour = shortest + 1 if after > before else shortest - 1
        runs[shortest][0] = runs[neighbour][0]
        runs = _coalesce(runs)
    return [(str(run[0]), times[run[1]], times[run[2]]) for run in runs]


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def load_clip(
    clip_id: str,
    data: str | pathlib.Path,
    source: str = DEFAULT_SOURCE,
) -> features.ClipFeatures:
    """One clip's :class:`handstand.features.ClipFeatures` from its features parquet.

    The parquet is read with :func:`handstand.features.read_clip_features` (which
    checks its columns); this only puts its columns back into the clip object the
    hold summary is built from, so the segment rows are measured exactly like
    ``hold_rows``'s.
    """
    path = features.output_dir(data, source) / f"{clip_id}.parquet"
    table = features.read_clip_features(path)
    # The dtypes features.py itself gives them (see feature_table and
    # _unusable_features): floats for the measurements, a flag for com_complete,
    # words (NaN when unknown) for balance_zone.
    values: dict[str, np.ndarray] = {}
    for name in (*features.FEATURE_NAMES, *features.COM_COLUMNS):
        if name not in table.columns:
            continue
        if name == "com_complete":
            values[name] = table[name].to_numpy(dtype=bool)
        elif name == "balance_zone":
            values[name] = table[name].to_numpy(dtype=object)
        else:
            values[name] = table[name].to_numpy(dtype=np.float64)
    return features.ClipFeatures(
        clip_id=clip_id,
        source=source,
        frame_idx=table["frame_idx"].to_numpy(dtype=np.int64),
        t_ms=table["t_ms"].to_numpy(dtype=np.int64),
        phase=table["phase"].to_numpy(dtype=object),
        hold_id=table["hold_id"].to_numpy(dtype=np.int64),
        measured=features.BodyFeatures(values=values, valid=table["valid"].to_numpy(dtype=bool)),
    )


def clip_segments(
    clip: features.ClipFeatures,
    min_ms: int = MIN_SEGMENT_MS,
) -> list[dict[str, object]]:
    """Every segment of every hold of the clip, keyed by :data:`SEGMENT_COLUMNS`.

    Ascending ``hold_id``, then ``segment_id``. A hold with no valid frame at all
    is one ``unmeasured`` segment spanning the hold's own frames.
    """
    leg = clip.value("leg_separation")
    knee = clip.value("knee_angle")
    hip = clip.value("hip_angle")
    rows: list[dict[str, object]] = []
    for hold_id in clip.hold_ids():
        in_hold = clip.hold_mask(hold_id)
        valid = clip.measurable_hold_mask(hold_id)
        valid_index = np.flatnonzero(valid)
        shapes = frame_shapes(leg[valid_index], knee[valid_index], hip[valid_index])
        runs = segment_runs(clip.t_ms[valid_index].tolist(), shapes, min_ms)
        if not runs:
            hold_times = clip.t_ms[in_hold]
            runs = [(hold_shapes.UNMEASURED, int(hold_times[0]), int(hold_times[-1]))]
        for segment_id, (shape, start_ms, end_ms) in enumerate(runs):
            in_segment = in_hold & (clip.t_ms >= start_ms) & (clip.t_ms <= end_ms)
            measurable = valid & in_segment
            row: dict[str, object] = {
                "clip_id": clip.clip_id,
                "source": clip.source,
                "hold_id": hold_id,
                "segment_id": segment_id,
                "shape_guess": shape,
                "mixed": len(runs) > 1,
                "segment_frames": int(in_segment.sum()),
                "valid_frames": int(measurable.sum()),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_s": round((end_ms - start_ms) / 1000.0, _DURATION_DIGITS),
            }
            for feature in features.FEATURES:
                median, iqr = features._spread(clip.value(feature.name)[measurable])
                row[f"{feature.name}_median"] = features._rounded_scalar(median, feature.digits)
                row[f"{feature.name}_iqr"] = features._rounded_scalar(iqr, feature.digits)
            rows.append(row)
    return rows


def segments_path(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pathlib.Path:
    """``<data_dir>/features/<source>/hold_segments.csv``, the table this module writes."""
    return features.output_dir(data, source) / SEGMENTS_NAME


def read_segments(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pd.DataFrame:
    """The segment table, an empty :data:`SEGMENT_COLUMNS` frame when it is not written yet."""
    path = segments_path(data, source)
    if not path.is_file():
        return pd.DataFrame(columns=list(SEGMENT_COLUMNS))
    return pd.read_csv(path)


def _sorted(table: pd.DataFrame) -> pd.DataFrame:
    """Sorted by clip, hold and segment."""
    if table.empty:
        return table.reset_index(drop=True)
    return table.sort_values(["clip_id", "hold_id", "segment_id"], kind="stable").reset_index(
        drop=True
    )


def build_segments(
    data: str | pathlib.Path,
    source: str = DEFAULT_SOURCE,
    clips: Sequence[str] | None = None,
    min_ms: int = MIN_SEGMENT_MS,
    errors: list[str] | None = None,
) -> pd.DataFrame:
    """Segment every clip with a features parquet (or only ``clips``) and write the table.

    Returns the table as written, sorted by clip, hold and segment. With ``clips``
    the rows of the other clips already in the file are kept, as
    :func:`handstand.features.write_hold_summary` does for the hold summary. The
    file is written through a temporary file and a rename, so an interrupted run
    cannot leave a half-written table behind.

    One bad clip does not abort the build: its error is printed as
    ``<clip>: <error>``, appended to ``errors`` (when given) and the clip is
    skipped; rows it already had in the file are kept. A named clip without a
    features parquet is such an error.
    """
    root = pathlib.Path(data)
    if clips:
        names = [str(clip) for clip in clips]
    else:
        names = sorted(path.stem for path in features.output_dir(root, source).glob("*.parquet"))
    rows: list[dict[str, object]] = []
    built: set[str] = set()
    for name in names:
        try:
            rows.extend(clip_segments(load_clip(name, root, source), min_ms))
        except Exception as error:  # one bad clip must not lose the other clips' segments
            message = f"{name}: {error}"
            print(message, file=sys.stderr)
            if errors is not None:
                errors.append(message)
            continue
        built.add(name)
    table = pd.DataFrame(rows, columns=list(SEGMENT_COLUMNS))

    path = segments_path(root, source)
    if clips and path.is_file():
        existing = pd.read_csv(path)
        missing = [column for column in SEGMENT_COLUMNS if column not in existing.columns]
        if missing:
            raise ValueError(f"{path}: segment table is missing column(s) {missing}")
        keep = existing[~existing["clip_id"].astype(str).isin(built)]
        if not keep.empty:
            table = pd.concat([keep[list(SEGMENT_COLUMNS)], table], ignore_index=True)
    table = _sorted(table)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        table.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return table


def line_segments(table: pd.DataFrame, min_s: float = LINE_MIN_S) -> pd.DataFrame:
    """The segments whose shape is ``line`` and that last at least ``min_s`` seconds."""
    keep = (table["shape_guess"] == "line") & (table["duration_s"] >= min_s)
    return table[keep]


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def _seconds(value: float) -> str:
    return f"{value:.1f}s"


def _segment_text(group: pd.DataFrame) -> str:
    return ", ".join(
        f"{row.shape_guess} {_seconds(float(row.duration_s))}" for row in group.itertuples()
    )


def report(
    table: pd.DataFrame,
    labels: Mapping[tuple[str, int | str], str],
) -> str:
    """The build's text report: totals, the line seconds in mixed holds, the reviewed holds.

    ``labels`` is :func:`handstand.hold_shapes.load_hold_shapes`: the user's reviewed
    shape per hold. The text is written under ``data/`` only, never committed.
    """
    hold_groups = (
        []
        if table.empty
        else [
            (key, group.sort_values("segment_id"))
            for key, group in table.groupby(["clip_id", "hold_id"], sort=True)
        ]
    )
    mixed_count = sum(1 for _key, group in hold_groups if bool(group["mixed"].iloc[0]))
    lines = [f"holds: {len(hold_groups)}, mixed holds: {mixed_count}", "segments per shape:"]
    if table.empty:
        lines.append("  (none)")
    else:
        per_shape = table.groupby("shape_guess")["duration_s"].agg(["count", "sum"])
        for shape, row in per_shape.sort_index().iterrows():
            lines.append(f"  {shape}: {int(row['count'])} segments, {_seconds(float(row['sum']))}")

    inside = table[table["mixed"].astype(bool) & (table["shape_guess"] == "line")]
    long_inside = line_segments(inside)
    lines.append(
        f"line seconds inside mixed holds: {_seconds(float(inside['duration_s'].sum()))} "
        f"in {len(inside)} segments (>= {LINE_MIN_S:g} s: "
        f"{_seconds(float(long_inside['duration_s'].sum()))} in {len(long_inside)})"
    )

    reviewed: dict[str, list[int]] = {}
    reviewed_mixed: list[str] = []
    for (clip_id, hold_id), group in hold_groups:
        label = labels.get((str(clip_id), int(hold_id)))
        if label is None:
            continue
        counts = reviewed.setdefault(label, [0, 0])
        is_mixed = bool(group["mixed"].iloc[0])
        counts[1 if is_mixed else 0] += 1
        if is_mixed:
            reviewed_mixed.append(
                f"  {clip_id} h{int(hold_id)} label={label} segments={_segment_text(group)}"
            )
    lines.append("reviewed holds (label: single-segment / mixed):")
    if not reviewed:
        lines.append("  (none)")
    for label in sorted(reviewed):
        single, mixed = reviewed[label]
        lines.append(f"  {label}: {single} / {mixed}")
    if reviewed_mixed:
        lines.append("reviewed mixed holds:")
        lines.extend(reviewed_mixed)
    return "\n".join(lines)


def _report_path(root: pathlib.Path, when: dt.datetime | None = None) -> pathlib.Path:
    """``<data_dir>/reports/segments/<YYYY-MM-DD_HHMMSS>.txt``, a free name for one build."""
    stamp = (when or dt.datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    directory = root / features.REPORTS_DIRNAME / REPORT_DIRNAME
    path = directory / f"{stamp}.txt"
    counter = 1
    while path.exists():
        path = directory / f"{stamp}_{counter}.txt"
        counter += 1
    return path


def _write_report(root: pathlib.Path, text: str, when: dt.datetime | None = None) -> pathlib.Path:
    """Write the report under ``<data>/reports/segments/`` and return its path."""
    path = _report_path(root, when)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.segments`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.segments",
        description="Split every hold into per-shape segments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--data",
            type=pathlib.Path,
            default=None,
            help="data directory (default: $HANDSTAND_DATA)",
        )
        command.add_argument(
            "--source",
            choices=SOURCES,
            default=DEFAULT_SOURCE,
            help=f"which features to read (default: {DEFAULT_SOURCE})",
        )

    build = commands.add_parser(
        "build",
        help="segment every hold and write data/features/<source>/hold_segments.csv",
    )
    common(build)
    build.add_argument("--clips", nargs="+", default=None, metavar="ID", help="only these clips")
    build.add_argument(
        "--min-ms",
        type=int,
        default=MIN_SEGMENT_MS,
        help=f"merge runs shorter than this many ms (default: {MIN_SEGMENT_MS})",
    )

    show = commands.add_parser("show", help="print one clip's segments")
    common(show)
    show.add_argument("clip_id")
    return parser


def _show(table: pd.DataFrame, clip_id: str) -> str:
    rows = table[table["clip_id"].astype(str) == clip_id]
    if rows.empty:
        return f"no segments for {clip_id}"
    lines = [f"{clip_id}: {rows['hold_id'].nunique()} holds, {len(rows)} segments"]
    for row in rows.itertuples():
        lines.append(
            f"  h{int(row.hold_id)} s{int(row.segment_id)} {row.shape_guess} "
            f"{row.start_ms}-{row.end_ms} ms ({_seconds(float(row.duration_s))}, "
            f"{int(row.valid_frames)}/{int(row.segment_frames)} valid frames)"
            f"{' mixed' if bool(row.mixed) else ''}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the segment table and its report, or show one clip's segments.

    Exit codes: **2** when a file cannot be read or written; **1** when the table was
    written but some clip failed (named on stderr); **0** otherwise.
    """
    args = build_arg_parser().parse_args(argv)
    root = pathlib.Path(args.data) if args.data is not None else data_dir()
    try:
        if args.command == "show":
            print(_show(read_segments(root, args.source), args.clip_id))
            return 0
        errors: list[str] = []
        table = build_segments(root, args.source, args.clips, args.min_ms, errors)
        text = report(table, hold_shapes.load_hold_shapes(root))
        report_path = _write_report(root, text)
        print(text)
        print(f"segments {segments_path(root, args.source)}")
        print(f"report {report_path}")
        if errors:
            print(f"segments: {len(errors)} clip(s) failed", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError) as error:
        print(f"segments: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
