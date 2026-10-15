"""Build the scoring **reference** (#28) from the user's reviewed line holds.

A score is only a number next to a yardstick, and :mod:`handstand.score` refuses
to guess one: ``--reference`` is a required argument and no reference number is
ever committed to this public repository. This module is what writes that
yardstick, out of the holds the user reviewed in the reference review
(:mod:`handstand.reference_review`, #27): the per-feature mean and sample SD of
every :data:`handstand.score.SCORE_FEATURES` value over the good holds.

Input and output
----------------

Everything is read from the shared data directory, nothing is spelled out here::

    <data_dir>/labels/reference_review.csv          # #27: reference-worthy holds
    <data_dir>/catalogue.csv                        # camera_angle: which view
    <data_dir>/features/<source>/<clip_id>.parquet  # #22/#23: per-frame features
    <data_dir>/labels/hold_shapes.csv               # #86: which holds are lines

One hold's value of one feature is :func:`handstand.score.hold_values` — the same
athlete-signed medians (and the two spreads) a score is made of, so the reference
and the score cannot disagree about what a number means. A hold with fewer than
:data:`handstand.score.MIN_SCORE_FRAMES` valid frames is skipped and reported,
never averaged in: no honest median comes out of four frames.

Side views only
---------------

The side-view features (hip angle, leg separation, knee) are only honest from the
side, so by default the reference is built from holds whose clip's catalogue
``camera_angle`` is :data:`SIDE_VIEW` (``"side"``, stripped and lower-cased): a
``front`` or angled clip, or one whose cell is empty, is skipped with a note.
``camera_angle`` is a person's answer until view detection (#74) exists, and the
review (:mod:`handstand.reference_review`) says nothing about the camera.
``--any-view`` builds from every reviewed hold regardless of the camera, and a
missing catalogue is an error unless that flag is given.

The reference file is ``docs/scoring.md``'s ``handstand-reference`` schema v1::

    <data_dir>/references/reference-line.json       # git-ignored

with ``--install`` also copied to ``<repo>/ios/LocalResources/reference-line.json``
(git-ignored), the name the app's ``ReferenceLoader`` looks for
(``HoldType.line.referenceResourceName + ".json"``).

Each hold weighs the same
-------------------------

The per-feature statistics are taken over the **holds**, one value per hold, and
never over the pooled per-frame values: a hold's frames are not independent
samples of "what a good handstand looks like", and a long hold must not outvote
a short one. ``mean`` is the mean of the finite hold values, ``sd`` the sample
standard deviation (``ddof = 1``), ``n`` how many holds had a finite value. A
feature finite in fewer than :data:`MIN_FEATURE_HOLDS` holds is **left out** —
the format allows a partial reference, and a mean of three holds is not a
yardstick.

CLI::

    cd pipeline
    uv run python -m handstand.reference build [--data DIR] [--source SRC]
        [--out PATH] [--min-holds N] [--install] [--any-view]

The build also writes a sanity report to
``<data_dir>/reports/reference/<YYYY-MM-DD_HHMMSS>.txt`` and prints it: the
per-feature table, a leave-one-out of the reference holds, every other line hold
scored against the full reference, and the subset of those in side-view clips —
so a person can see that the reference holds score better than the rest before
anybody trusts the numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import shutil
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from handstand import catalogue, features, hold_shapes, reference_review, score
from handstand.paths import data_dir
from handstand.postprocess import DEFAULT_SOURCE, SOURCES

__all__ = [
    "DEFAULT_MIN_HOLDS",
    "DEFAULT_REFERENCE_NAME",
    "INSTALL_RELATIVE",
    "MIN_FEATURE_HOLDS",
    "REFERENCE_DIRNAME",
    "REPORT_DIRNAME",
    "REPORTS_DIRNAME",
    "SIDE_VIEW",
    "build_arg_parser",
    "build_reference",
    "built_from_text",
    "camera_angles",
    "collect_hold_values",
    "default_output",
    "evaluate",
    "filter_side_view",
    "install_path",
    "main",
    "repo_root",
    "write_reference",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Output root under ``<data_dir>/``: the built reference files, one per hold
#: type, exactly as ``features/`` and ``scores/`` have one.
REFERENCE_DIRNAME = "references"

#: The name the app's ``ReferenceLoader`` looks for — ``HoldType.line``'s
#: ``"reference-line"`` plus ``.json`` — and so the name this module writes.
DEFAULT_REFERENCE_NAME = "reference-line.json"

#: Where the sanity report goes, under ``<data_dir>/``.
REPORTS_DIRNAME = features.REPORTS_DIRNAME

#: The report subdirectory, one file per build.
REPORT_DIRNAME = "reference"

#: Where ``--install`` copies the reference, relative to the repository root.
INSTALL_RELATIVE: tuple[str, ...] = ("ios", "LocalResources", DEFAULT_REFERENCE_NAME)

#: A feature finite in fewer holds than this is left out of the reference: the
#: format allows a partial file, and a mean of four holds is not a yardstick.
MIN_FEATURE_HOLDS = 5

#: The fewest usable holds a build will accept before it refuses: below this the
#: whole reference is a handful of holds dressed up as a scale.
DEFAULT_MIN_HOLDS = 10

#: The catalogue ``camera_angle`` a hold must carry to build the reference: the
#: side-view features (hip angle, leg separation, knee) are only honest from the
#: side, and the review says nothing about the camera. Other views are skipped.
SIDE_VIEW = "side"

#: How many decimal places a reference mean and SD are rounded to.
_REFERENCE_DIGITS = 6


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: The repository root this checkout lives in (``pipeline/handstand/`` is two
#: levels below it). Tests pass their own ``repo_dir`` to :func:`main` so a
#: build never writes into a real checkout.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def repo_root() -> pathlib.Path:
    """``<repo>``, the directory ``--install`` copies into."""
    return _REPO_ROOT


def default_output(data: str | pathlib.Path) -> pathlib.Path:
    """``<data_dir>/references/reference-line.json``, the default build output."""
    return pathlib.Path(data) / REFERENCE_DIRNAME / DEFAULT_REFERENCE_NAME


def install_path(root: str | pathlib.Path) -> pathlib.Path:
    """``<repo>/ios/LocalResources/reference-line.json``, where ``--install`` copies."""
    return pathlib.Path(root).joinpath(*INSTALL_RELATIVE)


def _report_path(root: pathlib.Path, when: dt.datetime | None = None) -> pathlib.Path:
    """``<data_dir>/reports/reference/<YYYY-MM-DD_HHMMSS>.txt`` for one build."""
    stamp = (when or dt.datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    return root / REPORTS_DIRNAME / REPORT_DIRNAME / f"{stamp}.txt"


# --------------------------------------------------------------------------- #
# 1. One value per reviewed hold
# --------------------------------------------------------------------------- #


def _hold_number(value: object) -> int:
    """A reviewed hold id as an ``int``; the CSV writes strings and numbers alike."""
    return int(float(str(value).strip()))


def camera_angles(data: str | pathlib.Path) -> dict[str, str]:
    """Each clip's catalogue ``camera_angle``, stripped and lower-cased.

    The reference is built from **side-view** holds only: the side-view features
    are garbage from the front or at an angle, and the review records the camera
    nowhere. ``camera_angle`` is a person's answer until view detection (#74)
    exists, so an empty cell reads ``""`` here. A missing catalogue cannot be
    filtered at all, so this raises rather than letting a build quietly use every
    hold; ``--any-view`` skips the call on purpose.
    """
    path = reference_review.catalogue_path(data)
    if not path.is_file():
        raise FileNotFoundError(
            f"no catalogue to read camera_angle from: {path} "
            "(build with --any-view to ignore the view)"
        )
    return {
        clip_id: (row.get("camera_angle") or "").strip().lower()
        for row in catalogue.read_rows(path)
        if (clip_id := (row.get("clip_id") or "").strip())
    }


def filter_side_view(
    holds: Iterable[tuple[str, int | str]],
    angles: Mapping[str, str],
) -> tuple[set[tuple[str, int | str]], list[str]]:
    """Split reviewed holds into the side-view ones and a note per skip.

    A hold is kept only when its clip's ``camera_angle`` is :data:`SIDE_VIEW`
    after stripping and lower-casing, so ``"Side"`` and ``" side "`` count. Every
    other hold — ``front``, angled, or a clip with an empty or absent cell, which
    the note calls ``empty`` — is left out with the note the build prints, since
    only side-view holds are comparable when the side-view features are measured.
    """
    kept: set[tuple[str, int | str]] = set()
    notes: list[str] = []
    for clip_id, hold_id in sorted(holds, key=lambda item: (str(item[0]), _hold_number(item[1]))):
        clip = str(clip_id)
        angle = angles.get(clip, "")
        if angle == SIDE_VIEW:
            kept.add((clip_id, hold_id))
            continue
        notes.append(
            f"{clip} h{_hold_number(hold_id)}: not side view "
            f"(camera_angle={angle or 'empty'})"
        )
    return kept, notes


def collect_hold_values(
    holds: Iterable[tuple[str, int | str]],
    data: str | pathlib.Path,
    source: str = DEFAULT_SOURCE,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The reviewed holds' values, and a note for every hold that was skipped.

    ``holds`` is the ``(clip_id, hold_id)`` set
    :func:`handstand.reference_review.load_reference_holds` returns; the hold id
    is coerced with :func:`int`. Each hold's values are
    :func:`handstand.score.hold_values` of its clip's features parquet, keyed by
    :data:`handstand.score.SCORE_FEATURES` and carrying ``clip_id`` and
    ``hold_id`` beside them.

    A hold is skipped, with a note in the second return value, when its clip has
    no features parquet (``features file missing``), when the parquet has no
    frames for that hold (``hold id not in features``), or when it has fewer
    than :data:`handstand.score.MIN_SCORE_FRAMES` valid frames (``too few valid
    frames``). The rows are in clip and hold order, so a build is reproducible.
    """
    root = pathlib.Path(data)
    in_root = features.output_dir(root, source)

    by_clip: dict[str, list[int]] = {}
    for clip_id, hold_id in holds:
        by_clip.setdefault(str(clip_id), []).append(_hold_number(hold_id))

    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for clip_id in sorted(by_clip):
        hold_ids = sorted(set(by_clip[clip_id]))
        path = in_root / f"{clip_id}.parquet"
        try:
            table = features.read_clip_features(path)
        except FileNotFoundError:
            notes.extend(f"{clip_id} h{hold_id}: features file missing" for hold_id in hold_ids)
            continue
        hold_column = table["hold_id"].to_numpy()
        valid = table["valid"].to_numpy(dtype=bool)
        for hold_id in hold_ids:
            mask = hold_column == hold_id
            if not mask.any():
                notes.append(f"{clip_id} h{hold_id}: hold id not in features")
                continue
            if int(valid[mask].sum()) < score.MIN_SCORE_FRAMES:
                notes.append(f"{clip_id} h{hold_id}: too few valid frames")
                continue
            values = score.hold_values(table, hold_id)
            rows.append({"clip_id": clip_id, "hold_id": hold_id, **values})
    return rows, notes


def built_from_text(
    rows: Sequence[Mapping[str, Any]],
    data: str | pathlib.Path,
    *,
    side_only: bool = True,
) -> str:
    """The file's provenance line: how many holds, from how many clips, dated when.

    ``n`` and ``m`` are the holds and distinct clips actually used, the date is
    the modification date of ``data/labels/reference_review.csv`` — when the
    review was last touched, which is the honest date of the labels — and
    ``side_only`` says whether the side-view filter was on, so the text says
    which set the numbers came from. **No clip id, name or path appears here**:
    the file is built from the user's own data and the free text must not leak
    which recordings it came from.
    """
    path = reference_review.labels_path(data)
    try:
        date = dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        date = "unknown"
    clips = len({str(row["clip_id"]) for row in rows})
    view = "side-view " if side_only else ""
    return (
        f"{len(rows)} reviewed {view}line holds from {clips} clips "
        f"(reference review, labels updated {date})"
    )


# --------------------------------------------------------------------------- #
# 2. The statistics
# --------------------------------------------------------------------------- #


def _finite_value(row: Mapping[str, Any], name: str) -> float | None:
    """One hold's finite value of ``name``, or ``None`` when it has none."""
    value = row.get(name)
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def build_reference(
    rows: Sequence[Mapping[str, Any]],
    built_from: str,
    min_n: int = MIN_FEATURE_HOLDS,
) -> dict[str, Any]:
    """The reference file's contents: per-feature mean, sample SD and n over holds.

    Per name of :data:`handstand.score.SCORE_FEATURES`, over the **finite** values
    of that feature across ``rows`` — one value per hold, each hold weighing the
    same, frames never pooled:

    * ``mean`` — the arithmetic mean, rounded to six decimals;
    * ``sd`` — the **sample** standard deviation (``ddof = 1``), rounded to six
      decimals;
    * ``n`` — how many holds had a finite value for the feature.

    A feature finite in fewer than ``min_n`` holds is **left out** of
    ``features``: the schema allows a partial reference and missing features are
    simply not scored. ``n_holds`` is the number of holds used, and the markers
    are :mod:`handstand.score`'s, so the result is exactly what
    :func:`handstand.score.load_reference` accepts once written.
    """
    entries: dict[str, dict[str, float | int]] = {}
    for name in score.SCORE_FEATURES:
        values = [value for row in rows if (value := _finite_value(row, name)) is not None]
        if len(values) < min_n:
            continue
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        entries[name] = {
            "mean": round(mean, _REFERENCE_DIGITS),
            "sd": round(sd, _REFERENCE_DIGITS),
            "n": len(values),
        }
    return {
        "schema": score.REFERENCE_SCHEMA,
        "version": score.REFERENCE_VERSION,
        "signs": score.REFERENCE_SIGNS,
        "built_from": built_from,
        "n_holds": len(rows),
        "features": entries,
    }


def _as_reference(ref: Mapping[str, Any]) -> score.Reference:
    """A written reference dict as the :class:`handstand.score.Reference` to score with."""
    entries = ref["features"]
    return score.Reference(
        features={
            name: (float(entry["mean"]), float(entry["sd"]), int(entry["n"]))
            for name, entry in entries.items()
        },
        n_holds=int(ref["n_holds"]),
        built_from=str(ref["built_from"]),
    )


def write_reference(ref: Mapping[str, Any], path: str | pathlib.Path) -> pathlib.Path:
    """Write ``ref`` as JSON (indent 1) and hand it to :func:`handstand.score.load_reference`.

    The read-back is not a formality: it is the same validation the scoring stage
    and the app perform, so a build and a load cannot disagree about the format.
    A file that does not load raises the reader's error rather than being left on
    disk for somebody else to trip over.
    """
    file = pathlib.Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8")
    score.load_reference(file)
    return file


# --------------------------------------------------------------------------- #
# 3. The sanity report
# --------------------------------------------------------------------------- #


def _clip_table(
    clip_id: str,
    root: pathlib.Path,
    source: str,
    cache: dict[str, pd.DataFrame | None],
) -> pd.DataFrame | None:
    """One clip's features table, read once; ``None`` when the file is not there."""
    if clip_id not in cache:
        path = features.output_dir(root, source) / f"{clip_id}.parquet"
        try:
            cache[clip_id] = features.read_clip_features(path)
        except FileNotFoundError:
            cache[clip_id] = None
    return cache[clip_id]


def _score_text(values: Sequence[float]) -> str:
    """``median=X min=Y max=Z n=K`` for a list of scores, ``none`` when empty."""
    if not values:
        return "none"
    return (
        f"median={statistics.median(values):.1f} min={min(values):.1f} "
        f"max={max(values):.1f} n={len(values)}"
    )


def _median_text(values: Sequence[float]) -> str:
    """The median of a list of scores to one decimal, ``n/a`` when there are none."""
    return f"{statistics.median(values):.1f}" if values else "n/a"


def evaluate(
    rows_ref: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    data: str | pathlib.Path,
    source: str = DEFAULT_SOURCE,
    *,
    side_angles: Mapping[str, str] | None = None,
    skipped_not_side: int = 0,
) -> str:
    """The build's sanity report: what the reference says and how it reads holds.

    Sections, in order:

    1. per feature of :data:`handstand.score.SCORE_FEATURES`, the mean, SD and n
       the reference carries (or that it was left out and in how many holds it
       was finite);
    2. a **leave-one-out** of the reference holds: each hold scored against a
       reference rebuilt from the *other* holds (the same
       :func:`build_reference`), reported as median/min/max;
    3. every other **line** hold of ``data/labels/hold_shapes.csv`` (the reference
       holds themselves excluded) scored against the full reference, same
       statistics plus how many were skipped — and, when ``side_angles`` is given,
       the subset of those in side-view clips, the only comparable ones;
    4. one line comparing the two medians — the check that the reference holds do
       score better than the rest.

    ``side_angles`` is :func:`camera_angles` when the build filtered to side
    views: it adds the subset line in 3 and puts ``skipped_not_side`` in the
    header. ``None`` means every view was used and the report reads as before.

    Nothing here changes the reference; it only says whether the numbers deserve
    to be trusted.
    """
    root = pathlib.Path(data)
    cache: dict[str, pd.DataFrame | None] = {}
    reference_object = _as_reference(reference)

    used = len(rows_ref)
    if side_angles is None:
        header = f"reference report ({used} hold(s), source {source})"
    else:
        header = (
            f"reference report ({used} hold(s) used, {skipped_not_side} skipped as not "
            f"side view, source {source})"
        )
    lines = [header, "", "features:"]
    lines.append("  feature          mean            sd              n")
    for name in score.SCORE_FEATURES:
        entry = reference["features"].get(name)
        if entry is None:
            finite = sum(1 for row in rows_ref if _finite_value(row, name) is not None)
            lines.append(f"  {name:<15} left out (finite in {finite} hold(s))")
        else:
            lines.append(
                f"  {name:<15} {float(entry['mean']):<15.6f} {float(entry['sd']):<15.6f} "
                f"{int(entry['n'])}"
            )

    # Leave-one-out: does a hold look like a good hold when it did not build the
    # reference itself?
    loo_scores: list[float] = []
    for index, row in enumerate(rows_ref):
        others = list(rows_ref[:index]) + list(rows_ref[index + 1 :])
        loo = _as_reference(build_reference(others, str(reference["built_from"])))
        table = _clip_table(str(row["clip_id"]), root, source, cache)
        if table is None:
            continue
        result = score.score_hold(table, int(row["hold_id"]), loo)
        if result.score is not None:
            loo_scores.append(result.score)

    # Every other line hold (the reference holds excluded) against the full
    # reference: the reference should pull its own holds to the top.
    reference_keys = {(str(row["clip_id"]), int(row["hold_id"])) for row in rows_ref}
    shapes = hold_shapes.load_hold_shapes(root)
    other = sorted(
        (str(clip_id), int(hold_id))
        for (clip_id, hold_id), shape in shapes.items()
        if shape == "line" and (str(clip_id), int(hold_id)) not in reference_keys
    )
    other_scores: list[float] = []
    side_scores: list[float] = []
    skipped = 0
    for clip_id, hold_id in other:
        table = _clip_table(clip_id, root, source, cache)
        if table is None:
            skipped += 1
            continue
        result = score.score_hold(table, hold_id, reference_object)
        if result.score is None:
            skipped += 1
            continue
        other_scores.append(result.score)
        if side_angles is not None and side_angles.get(clip_id) == SIDE_VIEW:
            side_scores.append(result.score)

    lines += ["", f"leave-one-out of the reference holds: {_score_text(loo_scores)}"]
    lines.append(
        f"other line holds against the full reference: {_score_text(other_scores)}"
        + (f" (skipped {skipped})" if skipped else "")
    )
    if side_angles is not None:
        lines.append(f"other line holds in side-view clips: {_score_text(side_scores)}")
    lines.append(
        f"reference holds LOO median {_median_text(loo_scores)} vs other line holds median "
        f"{_median_text(other_scores)}"
    )
    return "\n".join(lines)


def _write_report(root: pathlib.Path, text: str, when: dt.datetime | None = None) -> pathlib.Path:
    """Write the report under ``<data>/reports/reference/`` and return its path."""
    path = _report_path(root, when)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.reference`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.reference",
        description="Build the scoring reference from the user's reviewed line holds.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build",
        help="build the reference from data/labels/reference_review.csv and the features",
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
    build.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=(
            "reference to write "
            f"(default: <data_dir>/{REFERENCE_DIRNAME}/{DEFAULT_REFERENCE_NAME})"
        ),
    )
    build.add_argument(
        "--min-holds",
        type=int,
        default=DEFAULT_MIN_HOLDS,
        metavar="N",
        help=(
            "refuse to build from fewer than this many usable holds "
            f"(default: {DEFAULT_MIN_HOLDS})"
        ),
    )
    build.add_argument(
        "--install",
        action="store_true",
        help=(
            "also copy the reference to <repo>/"
            + pathlib.Path(*INSTALL_RELATIVE).as_posix()
            + " for the app"
        ),
    )
    build.add_argument(
        "--any-view",
        action="store_true",
        help=(
            "build from every reviewed hold, not only those whose catalogue "
            "camera_angle is 'side' (the default); also skips reading the catalogue"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None, *, repo_dir: str | pathlib.Path | None = None) -> int:
    """Build the reference, write the report, optionally install it.

    Exit codes: **2** when the catalogue is missing without ``--any-view``, when
    there are fewer than ``--min-holds`` usable holds, or when the reference
    cannot be written; **0** otherwise. ``repo_dir`` is the directory
    ``--install`` copies into and defaults to this checkout; tests pass their own
    so a build never touches a real tree.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = pathlib.Path(args.data) if args.data is not None else data_dir()
    destination_root = pathlib.Path(repo_dir) if repo_dir is not None else repo_root()

    try:
        holds = reference_review.load_reference_holds(root)
        side_angles: dict[str, str] | None = None
        skipped_not_side = 0
        view_notes: list[str] = []
        if not args.any_view:
            side_angles = camera_angles(root)
            holds, view_notes = filter_side_view(holds, side_angles)
            skipped_not_side = len(view_notes)
        rows, notes = collect_hold_values(holds, root, args.source)
        for note in view_notes + notes:
            print(f"skip {note}")

        if len(rows) < args.min_holds:
            print(
                f"reference: only {len(rows)} usable hold(s), need at least "
                f"{args.min_holds}; not writing a reference",
                file=sys.stderr,
            )
            return 2

        built_from = built_from_text(rows, root, side_only=side_angles is not None)
        reference = build_reference(rows, built_from)
        out = pathlib.Path(args.out) if args.out is not None else default_output(root)
        write_reference(reference, out)
        report = evaluate(
            rows,
            reference,
            root,
            args.source,
            side_angles=side_angles,
            skipped_not_side=skipped_not_side,
        )
        report_path = _write_report(root, report)
        print(report)
        print(
            f"reference {out} n_holds={reference['n_holds']} "
            f"features={len(reference['features'])}: {', '.join(reference['features'])}"
        )
        print(f"report {report_path}")
        if args.install:
            installed = install_path(destination_root)
            installed.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out, installed)
            print(f"installed {installed}")
        return 0
    except (OSError, ValueError) as error:
        print(f"reference: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
