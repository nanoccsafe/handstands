"""Detect whether a hold was filmed from the side (#74).

The side-view features of :mod:`handstand.features` — hip angle, leg
separation, knee — are only honest from the side, so the reference build keeps
side-view holds only. The catalogue's ``camera_angle`` is a person's answer and
is empty for every clip nobody hand-labelled; this module measures the camera
instead, per hold, so a clip whose cell is empty can still be used.

The rule is the one the lead measured on the hand-labelled clips: from the side
the left and right shoulders (and hips) sit on top of each other in the image,
and from the front or at an angle they are apart. Per frame this module
measures the width between the two shoulders and between the two hips, in
pixels, and divides the mean of the ones it can see by the clip's body length
``L``::

    view_width = mean(shoulder width, hip width) / L

A pair contributes only when *both* its joints are valid, so a frame that sees
one shoulder and not the other has no shoulder width; a frame that sees neither
pair, or a clip with no usable ``L``, has no ``view_width`` at all.

Per hold, the median of the finite ``view_width`` values classifies the view::

    median <  SIDE_MAX (0.035)       -> "side"
    median >= NOT_SIDE_MIN (0.042)   -> "not_side"
    otherwise, or under MIN_VIEW_FRAMES (15) finite frames -> "unsure"

The gap between the two thresholds is deliberate: the lead's labelled side
holds measured 0.013–0.030 and the front holds 0.045–0.113, and a median in the
band between them is not evidence either way.

Input and output
----------------

Everything is read from the shared data directory, nothing is spelled out::

    <data_dir>/processed/<source>/<clip_id>.parquet   # x/y processed + valid
    <data_dir>/processed/<source>/<clip_id>.json      # body_length_px
    <data_dir>/features/<source>/<clip_id>.parquet    # hold_id and valid

Output is one table per source::

    <data_dir>/features/<source>/hold_views.csv        # one row per hold

CLI::

    cd pipeline
    uv run python -m handstand.view build [--data DIR] [--source mediapipe]

The build also writes a validation report to
``<data_dir>/reports/views/<YYYY-MM-DD_HHMMSS>.txt`` and prints it: the count
of catalogue label against detected view for the hand-labelled clips, a line
per disagreement, and how many holds were called side in all. That is what says
whether the detector can replace the empty cells.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from handstand import features, phases
from handstand.paths import data_dir
from handstand.postprocess import DEFAULT_SOURCE, SOURCES

__all__ = [
    "HOLD_VIEWS_NAME",
    "HOLD_VIEW_COLUMNS",
    "MIN_VIEW_FRAMES",
    "NOT_SIDE_MIN",
    "REPORT_DIRNAME",
    "SIDE_MAX",
    "VIEWS",
    "build_arg_parser",
    "build_hold_views",
    "classify",
    "clip_hold_views",
    "hold_views_path",
    "main",
    "read_hold_views",
    "validation_report",
    "view_width",
    "write_hold_views",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: A hold whose median ``view_width`` is under this is a side view.
SIDE_MAX = 0.035

#: A hold whose median ``view_width`` is at or over this is not a side view.
#: Between the two the measurement is not evidence either way.
NOT_SIDE_MIN = 0.042

#: A hold with fewer finite ``view_width`` frames than this is never classified:
#: a median of a handful of frames is not a camera angle.
MIN_VIEW_FRAMES = 15

#: Every classification :func:`classify` can return, in report order.
VIEWS: tuple[str, ...] = ("side", "not_side", "unsure")

#: The per-hold table of one source: every clip's holds in one file.
HOLD_VIEWS_NAME = "hold_views.csv"

#: The table's columns, in write order.
HOLD_VIEW_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "hold_id",
    "view_frames",
    "view_width",
    "view",
)

#: The report subdirectory under ``<data_dir>/reports/``, one file per build.
REPORT_DIRNAME = "views"

#: How many decimal places a hold's median ``view_width`` is written with.
_VIEW_DIGITS = 4

#: The catalogue labels :func:`validation_report` counts, in report order.
_LABELS: tuple[str, ...] = ("side", "front", "other")

#: The joint pairs whose image separation says which way the athlete faces the
#: camera, in the order they are averaged.
_WIDTH_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
)


# --------------------------------------------------------------------------- #
# The measurement, frame by frame
# --------------------------------------------------------------------------- #


def _pair_columns(joints: Sequence[str], left: str, right: str) -> tuple[int, int] | None:
    """The two columns of a joint pair, or ``None`` when this clip has no pair.

    A source that does not report a joint — a future model with no hips, a
    partially filled schema — simply has no column for it, and the pair is then
    not measurable rather than measured against a phantom joint.
    """
    names = list(joints)
    if left not in names or right not in names:
        return None
    return names.index(left), names.index(right)


def view_width(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    joints: Sequence[str],
    body_length: float,
) -> np.ndarray:
    """The per-frame ``view_width`` of a clip: ``(frames,)``, NaN where unmeasured.

    The mean of the shoulder width and the hip width that this frame can see,
    each the 2D pixel distance between the two joints of the pair and counted
    only when **both** joints are valid, divided by the clip's body length
    ``L``. A frame that sees neither pair, and every frame of a clip whose ``L``
    is not a positive finite number, is NaN: a width over no body length is not
    a ratio.
    """
    frames = int(x.shape[0])
    widths: list[np.ndarray] = []
    for left, right in _WIDTH_PAIRS:
        columns = _pair_columns(joints, left, right)
        if columns is None:
            continue
        first, second = columns
        both = np.asarray(valid, dtype=bool)[:, first] & np.asarray(valid, dtype=bool)[:, second]
        with np.errstate(invalid="ignore"):
            distance = np.hypot(
                x[:, first].astype(np.float64) - x[:, second].astype(np.float64),
                y[:, first].astype(np.float64) - y[:, second].astype(np.float64),
            )
        widths.append(np.where(both, distance, np.nan))
    if not widths:
        return np.full(frames, np.nan)
    stacked = np.stack(widths, axis=1)
    finite = np.isfinite(stacked)
    count = finite.sum(axis=1)
    total = np.where(finite, stacked, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_width = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    if not (math.isfinite(body_length) and body_length > 0.0):
        return np.full(frames, np.nan)
    return mean_width / float(body_length)


def classify(widths: np.ndarray) -> tuple[float, str]:
    """A hold's median ``view_width`` and its view, by the threshold rule.

    The median is of the finite values, NaN when there are none. A hold with
    fewer than :data:`MIN_VIEW_FRAMES` finite values is ``"unsure"`` whatever its
    median; otherwise the median decides, with the band between
    :data:`SIDE_MAX` and :data:`NOT_SIDE_MIN` left ``"unsure"`` on purpose.
    """
    values = np.asarray(widths, dtype=np.float64)
    finite = values[np.isfinite(values)]
    median = float(np.median(finite)) if finite.size else math.nan
    if finite.size < MIN_VIEW_FRAMES:
        return median, "unsure"
    if median < SIDE_MAX:
        return median, "side"
    if median >= NOT_SIDE_MIN:
        return median, "not_side"
    return median, "unsure"


def _rounded(value: float) -> float:
    """One median rounded for the table; NaN stays NaN."""
    return round(value, _VIEW_DIGITS) if math.isfinite(value) else value


# --------------------------------------------------------------------------- #
# One clip and the whole table
# --------------------------------------------------------------------------- #


def clip_hold_views(
    clip_id: str,
    data: str | pathlib.Path,
    source: str = DEFAULT_SOURCE,
) -> list[dict[str, Any]]:
    """One dict per hold of a clip, keyed by :data:`HOLD_VIEW_COLUMNS`.

    The widths are measured from the clip's processed parquet and its sidecar's
    body length, and the holds come from its features parquet: the two files
    must carry the same frames in the same order, and a mismatch is a
    :class:`ValueError` rather than a plausible-looking wrong answer. A hold's
    median is over the frames that are in the hold **and** valid, and its
    ``view_frames`` counts the finite widths among them, so a hold with no
    usable frame is reported and called ``"unsure"`` rather than dropped.
    """
    root = pathlib.Path(data)
    processed = features.input_dir(root, source) / f"{clip_id}.parquet"
    clip, valid = phases.read_processed(processed)
    body_length, _reason = phases.read_body_length(processed.with_suffix(".json"))
    table = features.read_clip_features(features.output_dir(root, source) / f"{clip_id}.parquet")
    if clip.frames != len(table):
        raise ValueError(
            f"{clip_id}: processed has {clip.frames} frames but features has {len(table)}; "
            "re-run the features stage"
        )
    widths = view_width(clip.x, clip.y, valid, clip.joints, body_length)
    hold_column = table["hold_id"].to_numpy()
    valid_column = table["valid"].to_numpy(dtype=bool)

    rows: list[dict[str, Any]] = []
    for hold_id in sorted({int(value) for value in hold_column if int(value) >= 0}):
        in_hold = (hold_column == hold_id) & valid_column
        hold_widths = widths[in_hold]
        median, view = classify(hold_widths)
        rows.append(
            {
                "clip_id": clip_id,
                "hold_id": hold_id,
                "view_frames": int(np.isfinite(hold_widths).sum()),
                "view_width": _rounded(median),
                "view": view,
            }
        )
    return rows


def build_hold_views(
    data: str | pathlib.Path,
    source: str = DEFAULT_SOURCE,
) -> pd.DataFrame:
    """Every clip that has a features parquet, as one :data:`HOLD_VIEW_COLUMNS` table.

    The rows are sorted by clip and hold, so a build is reproducible. A clip
    whose processed parquet is missing is skipped with a printed note rather
    than failing the run: a features file outliving its processed input is a
    clip somebody deleted, not a reason to lose the rest of the table.
    """
    root = pathlib.Path(data)
    rows: list[dict[str, Any]] = []
    for path in sorted(features.output_dir(root, source).glob("*.parquet")):
        try:
            rows.extend(clip_hold_views(path.stem, root, source))
        except FileNotFoundError:
            print(f"note: no processed keypoints for {path.stem}; skipped")
            continue
    table = pd.DataFrame(rows, columns=list(HOLD_VIEW_COLUMNS))
    if not table.empty:
        table = table.sort_values(["clip_id", "hold_id"], kind="stable").reset_index(drop=True)
    return table


def hold_views_path(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pathlib.Path:
    """``<data_dir>/features/<source>/hold_views.csv``, the table this module writes."""
    return features.output_dir(data, source) / HOLD_VIEWS_NAME


def write_hold_views(path: str | pathlib.Path, table: pd.DataFrame) -> pathlib.Path:
    """Write the table to ``path`` through a temporary file and a rename.

    An interrupted build cannot leave a half-written table behind: either the
    old file is still there or the new one is complete.
    """
    file = pathlib.Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f".{file.name}.tmp")
    try:
        table.to_csv(tmp, index=False)
        tmp.replace(file)
    finally:
        tmp.unlink(missing_ok=True)
    return file


def read_hold_views(
    data: str | pathlib.Path,
    source: str = DEFAULT_SOURCE,
) -> dict[tuple[str, int], str]:
    """{(clip_id, hold_id): view} from the table, ``{}`` when there is none.

    This is what the reference build reads: a hand label always wins, and the
    detected view is only consulted for the clips whose catalogue cell is
    empty.
    """
    path = hold_views_path(data, source)
    if not path.is_file():
        return {}
    table = pd.read_csv(path)
    return {
        (str(row.clip_id), int(row.hold_id)): str(row.view)
        for row in table.itertuples(index=False)
    }


# --------------------------------------------------------------------------- #
# The validation report
# --------------------------------------------------------------------------- #


def _width_text(value: float) -> str:
    """One median ``view_width`` as four decimals; NaN reads ``nan``."""
    return f"{float(value):.4f}"


def validation_report(
    table: pd.DataFrame,
    angles: Mapping[str, str],
) -> str:
    """Compare the detected views with the catalogue labels, and count.

    Only the holds of clips whose catalogue label is ``side``, ``front`` or
    ``other`` go into the count table and the mismatch lines — an empty or
    absent cell is exactly what this detector is for, so it is not evidence
    either way. A mismatch is a ``side`` label not detected side, or a
    ``front``/``other`` label detected side. The closing line is over **every**
    hold in the table.
    """
    rows = table.to_dict("records")
    labelled = [row for row in rows if angles.get(str(row["clip_id"]), "") in _LABELS]
    counts = {label: dict.fromkeys(VIEWS, 0) for label in _LABELS}
    for row in labelled:
        counts[angles[str(row["clip_id"])]][str(row["view"])] += 1

    lines = ["view validation: catalogue label x detected view", ""]
    lines.append("  label  " + "".join(f"{view:>10}" for view in VIEWS))
    for label in _LABELS:
        lines.append(f"  {label:<6} " + "".join(f"{counts[label][view]:>10}" for view in VIEWS))

    mismatches: list[str] = []
    for row in labelled:
        label = angles[str(row["clip_id"])]
        detected = str(row["view"])
        wrong_side = label == "side" and detected != "side"
        wrong_other = label in ("front", "other") and detected == "side"
        if wrong_side or wrong_other:
            mismatches.append(
                f"{row['clip_id']} h{int(row['hold_id'])} label={label} "
                f"detected={detected} width={_width_text(row['view_width'])}"
            )

    lines.append("")
    if mismatches:
        lines.append("mismatches:")
        lines.extend(f"  {line}" for line in mismatches)
    else:
        lines.append("mismatches: none")

    side_rows = [row for row in rows if str(row["view"]) == "side"]
    clips = {str(row["clip_id"]) for row in side_rows}
    lines.append("")
    lines.append(
        f"detected side: {len(side_rows)} holds in {len(clips)} clips (of {len(rows)} holds)"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _report_path(root: pathlib.Path, when: dt.datetime | None = None) -> pathlib.Path:
    """``<data_dir>/reports/views/<YYYY-MM-DD_HHMMSS>.txt`` for one build."""
    stamp = (when or dt.datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    return root / features.REPORTS_DIRNAME / REPORT_DIRNAME / f"{stamp}.txt"


def _write_report(root: pathlib.Path, text: str, when: dt.datetime | None = None) -> pathlib.Path:
    """Write the report under ``<data>/reports/views/`` and return its path."""
    path = _report_path(root, when)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.view`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.view",
        description="Detect whether each hold was filmed from the side.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build",
        help="measure every hold's view and write data/features/<source>/hold_views.csv",
    )
    build.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory (default: $HANDSTAND_DATA)",
    )
    build.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help=f"which features to read (default: {DEFAULT_SOURCE})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Measure every hold's view, write the table and the validation report.

    Exit codes: **2** when the table cannot be written; **0** otherwise. A
    missing catalogue is not an error — the report then counts no labelled
    clips and still writes the detected views.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = pathlib.Path(args.data) if args.data is not None else data_dir()

    # Imported here rather than at module level: `reference` imports this module
    # for the detected fallback, and the report only needs its catalogue reader.
    from handstand import reference

    try:
        table = build_hold_views(root, args.source)
        path = write_hold_views(hold_views_path(root, args.source), table)
        try:
            angles = reference.camera_angles(root)
        except FileNotFoundError:
            angles = {}
        report = validation_report(table, angles)
        report_path = _write_report(root, report)
        print(report)
        print(f"hold views {path}")
        print(f"report {report_path}")
        return 0
    except (OSError, ValueError) as error:
        print(f"view: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
