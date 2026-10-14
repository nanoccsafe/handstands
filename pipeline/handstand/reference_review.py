"""Auto-fill the catalogue and review LINE holds for the reference build.

The reference build (#27/#28) wants straight **line** holds only, and the
catalogue is the sheet the whole pipeline reads. Typing it by hand is slow and
error-prone, so this module fills what the pipeline already measured and asks the
user, per line hold, the only two questions a model cannot answer:

* is this hold good enough to be the *reference* ("reference-worthy"), and
* is it against a **wall** (the body touching a wall, which a reference must
  never do).

Everything else in the catalogue comes from ``hold_summary.csv`` and the trainer
report. The loop mirrors the hold-shape one (``docs/labeling.md``)::

    handstand.reference_review fill      -> fill EMPTY catalogue cells (backup first)
    handstand.reference_review prelabel  -> data/label_frames_reference/<clip>_h<hold>_<q>.jpg
                                           + data/label_studio_reference_review.json
                                           + tools/labeling/reference_review_config.xml
    Label Studio (separate)              -> an export JSON
    handstand.reference_review import    -> data/labels/reference_review.csv
                                           + catalogue good_clip and wall notes

``fill`` never touches a non-empty cell except to append the word ``trainer`` to
``notes`` (that cell may have been typed by hand, so the notes rule only ever
adds). ``import`` is the source of truth for ``good_clip``: it overwrites the
column from the reviewed holds, so a clip is good exactly when at least one of
its holds is a "yes" hold that is not against a wall.

Everything written from videos lives under ``data/`` and is git-ignored; only
the generated labeling config sits in the repository, under ``tools/labeling/``.

CLI::

    cd pipeline
    uv run python -m handstand.reference_review fill [--dry-run]
    uv run python -m handstand.reference_review prelabel [--min-duration SECONDS]
    uv run python -m handstand.reference_review import <export.json>
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import math
import os
import pathlib
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from handstand import catalogue as catalogue_module
from handstand import hold_shapes
from handstand.features import DEFAULT_SOURCE, HOLD_SUMMARY_NAME, output_dir
from handstand.frame_sampler import (
    CATALOGUE_NAME,
    FILE_MODE,
    REPORTS_DIRNAME,
    TRAINER_REPORT_NAME,
    read_trainer_present,
    write_jpeg,
)
from handstand.labels import LABELS_DIRNAME, read_export
from handstand.overlay import video_for_clip
from handstand.paths import data_dir
from handstand.pose_mediapipe import DisplayVideo
from handstand.prelabel import write_json

__all__ = [
    "ANNOTATION_FILL_COLUMNS",
    "BACKUPS_DIRNAME",
    "CONFIG_RELATIVE",
    "FRAMES_DIRNAME",
    "IMAGE_NAME",
    "LABEL_COLUMNS",
    "LABELS_CSV_NAME",
    "LOCAL_FILES_URL",
    "MIN_DURATION_S",
    "QUANTILES",
    "REFERENCE_CHOICE",
    "TASKS_FILENAME",
    "TRAINER_NOTE",
    "WALL_CHOICE",
    "WALL_NOTE",
    "FillReport",
    "ImportReport",
    "PrelabelReport",
    "build_arg_parser",
    "build_task",
    "catalogue_path",
    "config_xml",
    "default_config_path",
    "fill",
    "frame_name",
    "frames_dir",
    "hold_summary_path",
    "import_export",
    "labels_path",
    "load_reference_holds",
    "main",
    "prelabel",
    "read_label_rows",
    "summarise_fill",
    "summarise_import",
    "summarise_prelabel",
    "task_image_uri",
    "tasks_path",
    "trainer_report_path",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Name of the output directory inside the data directory: three JPEGs per hold.
FRAMES_DIRNAME = "label_frames_reference"
#: The Label Studio import, under ``<data_dir>/``.
TASKS_FILENAME = "label_studio_reference_review.json"
#: The imported labels, under ``<data_dir>/labels/``.
LABELS_CSV_NAME = "reference_review.csv"
#: Where the generated labeling config lives, relative to the repository root.
CONFIG_RELATIVE = ("tools", "labeling", "reference_review_config.xml")
#: Where ``fill`` and ``import`` copy the catalogue before writing it.
BACKUPS_DIRNAME = "backups"

#: The fractions of a hold the review shows, in order: quarter, middle, three-quarters.
QUANTILES: tuple[int, ...] = (25, 50, 75)

#: The shortest hold that is worth reviewing, in seconds. A hold under this can
#: never be the user's reference (the reference build wants a steady hold), so
#: ``prelabel`` queues no task and writes no frames for it — of the real line
#: holds, only a minority last this long, and the rest would waste the review.
#: ``prelabel --min-duration 0`` queues every line hold.
MIN_DURATION_S = 3.0

#: ``from_name``/``toName`` names of the two config choices; the import matches
#: a result by these names.
REFERENCE_CHOICE = "reference"
WALL_CHOICE = "wall"
#: ``name`` of the middle image, the ``toName`` of both choices.
IMAGE_NAME = "image50"

#: Label Studio local-files URL prefix: ``/data/local-files/?d=<path relative to
#: the data dir>``. A browser cannot open a ``file://`` URI, but Label Studio
#: serves these URLs from its local storage when it runs with
#: ``LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`` set to the data directory and the
#: project carries ``data/label_frames_reference`` as a local storage.
LOCAL_FILES_URL = "/data/local-files/?d="

#: The two notes this module knows how to append, never to repeat.
TRAINER_NOTE = "trainer"
WALL_NOTE = "wall"

#: The catalogue annotation columns ``fill`` owns, in the order the summary prints.
ANNOTATION_FILL_COLUMNS: tuple[str, ...] = (
    "hold_s",
    "outcome",
    "camera_angle",
    "skill",
    "notes",
)

#: Full header of ``reference_review.csv``, in order. Part of the contract: the
#: helpers read it back and ``load_reference_holds`` reads it for the reference build.
LABEL_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "hold_id",
    "reference",
    "wall",
    "labeled_at",
)

#: Decimal places of the hold duration in ``hold_s`` and of the two info numbers.
HOLD_S_DIGITS = 1
INFO_DIGITS = 2

#: What a boolean CSV cell says; lowercase, as everywhere else here.
TRUE = "true"
FALSE = "false"

#: Labels of the review choices, in config order.
REFERENCE_YES = "yes"
REFERENCE_NO = "no"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def _root(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """The data directory: ``data`` when given, else ``$HANDSTAND_DATA``."""
    return pathlib.Path(data) if data is not None else data_dir()


def frames_dir(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_frames_reference``, where the JPEGs go."""
    return _root(data) / FRAMES_DIRNAME


def tasks_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_studio_reference_review.json``, the Label Studio import."""
    return _root(data) / TASKS_FILENAME


def labels_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/labels/reference_review.csv``, the reviewed labels."""
    return _root(data) / LABELS_DIRNAME / LABELS_CSV_NAME


def catalogue_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/catalogue.csv``, the sheet ``fill`` and ``import`` write."""
    return _root(data) / CATALOGUE_NAME


def hold_summary_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/features/mediapipe/hold_summary.csv``."""
    return output_dir(_root(data), DEFAULT_SOURCE) / HOLD_SUMMARY_NAME


def trainer_report_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/reports/trainer_report.csv``."""
    return _root(data) / REPORTS_DIRNAME / TRAINER_REPORT_NAME


def default_config_path() -> pathlib.Path:
    """``tools/labeling/reference_review_config.xml`` in this checkout.

    Resolved from this file (``pipeline/handstand/`` is two levels below the
    repository root), so a run writes the config the docs point at. Tests pass
    ``config_path`` to keep a generated file out of the tree.
    """
    return pathlib.Path(__file__).resolve().parents[2].joinpath(*CONFIG_RELATIVE)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _as_float(value: object) -> float:
    """Coerce a cell to a finite float, ``nan`` when it is blank, absent or junk."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _as_bool(value: object) -> bool:
    """Coerce a CSV or JSON flag to a bool; a blank or absent cell is False."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == TRUE


def _hold_key(value: object) -> int | str:
    """A hold id as a dict key: an ``int`` when it reads as one, else its text."""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    return int(number)


def _hold_sort_key(value: object) -> tuple[int, str]:
    """Order hold ids numerically first, so hold 10 does not sort before hold 2."""
    key = _hold_key(value)
    if isinstance(key, int):
        return (0, f"{key:012d}")
    return (1, key)


def _group_by_clip(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Holds grouped by clip, in first-seen order."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["clip_id"]), []).append(row)
    return list(groups.items())


def _hold_duration(row: Mapping[str, Any]) -> float:
    """One hold's duration in seconds, falling back to its start/end times."""
    value = _as_float(row.get("hold_duration_s"))
    if not math.isnan(value):
        return round(value, 3)
    start = _as_float(row.get("hold_start_ms"))
    end = _as_float(row.get("hold_end_ms"))
    if math.isnan(start):
        start = 0.0
    if math.isnan(end):
        end = start
    return round(max(0.0, end - start) / 1000.0, 3)


def _target_ms(row: Mapping[str, Any], quantile: int) -> float:
    """The time ``quantile`` percent into a hold: ``start + (end - start) * q / 100``."""
    start = _as_float(row.get("hold_start_ms"))
    end = _as_float(row.get("hold_end_ms"))
    if math.isnan(start):
        start = 0.0
    if math.isnan(end):
        end = start
    return start + (end - start) * quantile / 100.0


def _has_word(text: object, word: str) -> bool:
    """Is ``word`` already somewhere in a notes cell, case-insensitively?"""
    return word.lower() in str(text or "").lower()


def _append_note(notes: str, word: str) -> str:
    """Append ``word`` to a notes cell, separated by ``"; "`` when it is not empty."""
    return f"{notes}; {word}" if notes.strip() else word


def frame_name(clip_id: str, hold_id: object, quantile: int) -> str:
    """File name of one hold's frame at one quantile: ``<clip_id>_h<hold_id>_<q>.jpg``."""
    return f"{clip_id}_h{_hold_key(hold_id)}_{quantile}.jpg"


def task_image_uri(
    image: str | pathlib.Path,
    data: str | pathlib.Path | None = None,
) -> str:
    """The ``data.image*`` URL for one hold frame, built like the hold-shape one.

    The default is Label Studio's local-files form,
    ``/data/local-files/?d=label_frames_reference/<clip_id>_h<hold_id>_<q>.jpg`` —
    the path of the JPEG relative to the data dir, which is what the project's
    local storage serves (a ``file://`` URI is refused by the browser).
    """
    path = pathlib.Path(image)
    root = _root(data)
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        # Outside the data dir (a direct call with some other path): the local
        # storage holds this module's frames, so name the file as it names them.
        relative = pathlib.Path(FRAMES_DIRNAME) / path.name
    return f"{LOCAL_FILES_URL}{relative.as_posix()}"


def _write_csv(
    path: pathlib.Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> pathlib.Path:
    """Write a CSV atomically: temp file beside it, then rename, mode 0644."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = pathlib.Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, FILE_MODE)
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


def _backup_catalogue(path: pathlib.Path) -> pathlib.Path:
    """Copy the catalogue to ``<data>/backups/catalogue-<timestamp>.csv``.

    Taken **before** any write, so an interrupted or wrong run can always be
    undone from the copy beside it.
    """
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    destination = path.parent / BACKUPS_DIRNAME / f"catalogue-{stamp}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    os.chmod(destination, FILE_MODE)
    return destination


# --------------------------------------------------------------------------- #
# 1. fill
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FillReport:
    """What one ``fill`` run produced."""

    path: pathlib.Path
    #: Cells filled per annotation column (the keys of :data:`ANNOTATION_FILL_COLUMNS`).
    counts: dict[str, int]
    #: The catalogue copy taken before writing; ``None`` on a dry run.
    backup: pathlib.Path | None
    dry_run: bool
    rows: int

    @property
    def filled(self) -> int:
        """Cells filled in total."""
        return sum(self.counts.values())


def _longest_hold_by_clip(root: pathlib.Path) -> dict[str, float]:
    """``clip_id ->`` the longest hold duration in seconds.

    A clip with no hold row is simply absent, which the caller reads as "no
    hold"; :func:`_hold_seconds` rounds the value to 0.1 when it writes the cell.
    """
    longest: dict[str, float] = {}
    for row in hold_shapes.read_hold_summary(hold_summary_path(root)):
        clip_id = str(row["clip_id"])
        longest[clip_id] = max(longest.get(clip_id, 0.0), _hold_duration(row))
    return longest


def _clip_skills(root: pathlib.Path) -> dict[str, str]:
    """``clip_id -> skill`` from the reviewed hold shapes, the catalogue's value.

    Reuses :func:`handstand.hold_shapes.build_summary` — the same helper
    ``write_catalogue_skill`` is fed from — so ``mixed`` and the exclusion of
    ``not_a_hold``-only clips are decided here exactly as there.
    """
    shapes = hold_shapes.load_hold_shapes(root)
    if not shapes:
        return {}
    rows = [
        {"clip_id": clip_id, "hold_id": hold_id, "shape": shape}
        for (clip_id, hold_id), shape in shapes.items()
    ]
    return hold_shapes.build_summary(rows).clip_skills


def _hold_seconds(value: float) -> str:
    """A ``hold_s`` cell: one decimal, or ``0`` for a clip with no hold row."""
    if value <= 0:
        return "0"
    return f"{value:.{HOLD_S_DIGITS}f}"


def fill(
    *,
    data: str | pathlib.Path | None = None,
    dry_run: bool = False,
) -> FillReport:
    """Fill the EMPTY catalogue cells from what the pipeline already measured.

    A non-empty cell is never changed — it may have been typed by hand — except
    ``notes``, where the word ``trainer`` is appended when the trainer report has
    a second person in the clip and the cell does not already carry the word. The
    file is read and written with :mod:`handstand.catalogue`'s atomic helpers, so
    an interrupted run cannot damage the hand-typed annotations; the catalogue is
    copied to ``<data>/backups/`` before the write.

    ``dry_run`` computes and reports the same counts without writing anything.
    """
    root = _root(data)
    path = catalogue_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"no catalogue to fill: {path}")
    rows = catalogue_module.read_rows(path)

    longest = _longest_hold_by_clip(root)
    skills = _clip_skills(root)
    trainer = read_trainer_present(trainer_report_path(root))
    missing_column = catalogue_module.MISSING_COLUMN

    counts = {column: 0 for column in ANNOTATION_FILL_COLUMNS}
    for row in rows:
        if _as_bool(row.get(missing_column)):
            continue
        clip_id = (row.get("clip_id") or "").strip()
        if not clip_id:
            continue
        if not (row.get("hold_s") or "").strip():
            row["hold_s"] = _hold_seconds(longest.get(clip_id, 0.0))
            counts["hold_s"] += 1
        if not (row.get("outcome") or "").strip():
            row["outcome"] = "hold" if clip_id in longest else "no_hold"
            counts["outcome"] += 1
        if not (row.get("camera_angle") or "").strip():
            # Every clip so far is a side view; the user confirmed it.
            row["camera_angle"] = "side"
            counts["camera_angle"] += 1
        if not (row.get("skill") or "").strip() and clip_id in skills:
            row["skill"] = skills[clip_id]
            counts["skill"] += 1
        if trainer.get(clip_id, False) and not _has_word(row.get("notes"), TRAINER_NOTE):
            row["notes"] = _append_note(row.get("notes") or "", TRAINER_NOTE)
            counts["notes"] += 1

    if dry_run:
        return FillReport(path=path, counts=counts, backup=None, dry_run=True, rows=len(rows))
    backup = _backup_catalogue(path)
    catalogue_module.write_rows(rows, path)
    return FillReport(path=path, counts=counts, backup=backup, dry_run=False, rows=len(rows))


def summarise_fill(report: FillReport) -> str:
    """Multi-line summary of a fill run, for the CLI to print."""
    mode = " (dry run; nothing written)" if report.dry_run else ""
    lines = [
        f"fill: {report.path}{mode}",
        f"  {report.rows} row(s), {report.filled} cell(s) filled",
    ]
    lines += [f"  {column}: {count}" for column, count in report.counts.items()]
    if report.backup is not None:
        lines.append(f"  backup: {report.backup}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. prelabel
# --------------------------------------------------------------------------- #


def config_xml() -> str:
    """The ``tools/labeling/reference_review_config.xml`` contents.

    A header line showing ``$info``, the three frames side by side, the required
    single ``reference`` choice (``yes`` hotkey 1, ``no`` hotkey 2) and the
    optional ``wall`` choice (``wall`` hotkey 3, multiple). No prediction is
    attached: the user decides from scratch.

    Label Studio 1.23 parses a ``value`` as ONE variable whose name runs from the
    ``$`` to the end of the attribute, so every ``$`` here must stay a lone
    ``$identifier``; :func:`build_task` joins the header into the single ``info``
    field instead of templating it.
    """
    return """<View>
  <!--
    Generated by pipeline/handstand/reference_review.py (prelabel); edit the
    module, not this file. The reference build (#27/#28) keeps only holds the
    user judged reference-worthy here and not against a wall. Hotkeys 1 and 2
    answer reference (yes / no), hotkey 3 checks wall. Answer no when unsure.
  -->
  <Text name="hold_info" value="$info" density="1"/>
  <View style="display:flex">
    <Image name="image25" value="$image25" zoom="true" zoomControl="true"/>
    <Image name="image50" value="$image50" zoom="true" zoomControl="true"/>
    <Image name="image75" value="$image75" zoom="true" zoomControl="true"/>
  </View>
  <Choices name="reference" toName="image50" choice="single" required="true">
    <Choice value="yes" hotkey="1"/>
    <Choice value="no" hotkey="2"/>
  </Choices>
  <Choices name="wall" toName="image50" choice="multiple">
    <Choice value="wall" hotkey="3"/>
  </Choices>
</View>
"""


def _number(value: float) -> str:
    """One info number to :data:`INFO_DIGITS`, ``n/a`` when it was never measured."""
    if math.isnan(value):
        return "n/a"
    return f"{value:.{INFO_DIGITS}f}"


def info_string(clip_id: str, hold_id: object, row: Mapping[str, Any]) -> str:
    """The single ``info`` field: clip, hold, duration, line deviation, sway.

    Joined into ONE task field because the config's ``<Text>`` reads a single
    ``$info`` (see :func:`config_xml`). Numbers are to two decimals; a feature the
    pipeline never measured is ``n/a``.
    """
    duration = _hold_duration(row)
    line_dev = _number(_as_float(row.get("line_deviation_median")))
    sway = _number(_as_float(row.get("com_sway_sd")))
    return (
        f"{clip_id} / hold {hold_id} / {duration:.{INFO_DIGITS}f} s / "
        f"line dev {line_dev} / sway sd {sway}"
    )


def build_task(
    clip_id: str,
    hold_id: int | str,
    row: Mapping[str, Any],
    frames_directory: pathlib.Path,
    data: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """One Label Studio task for one hold, carrying its three frames and header.

    No ``predictions``: the review is a judgement, not a pre-labelled answer.
    """
    images = {
        f"image{quantile}": task_image_uri(
            frames_directory / frame_name(clip_id, hold_id, quantile), data
        )
        for quantile in QUANTILES
    }
    return {
        "data": {
            **images,
            "clip_id": clip_id,
            "hold_id": hold_id,
            "duration_s": _hold_duration(row),
            "info": info_string(clip_id, hold_id, row),
        }
    }


@dataclasses.dataclass(frozen=True)
class PrelabelReport:
    """What one pre-label run produced."""

    #: Line holds queued: after skipping missing clips and holds under
    #: :attr:`min_duration_s`.
    holds: int
    #: The Label Studio tasks, longest hold first.
    tasks: list[dict[str, Any]]
    #: JPEGs written this run.
    frames_written: int
    #: JPEGs that already existed and were left alone.
    frames_skipped: int
    #: ``(clip_id, reason)`` for every clip whose video could not be read.
    clips_skipped: list[tuple[str, str]]
    #: The shortest hold the run queued, in seconds.
    min_duration_s: float
    #: Line holds the minimum left out: no task and no frames.
    left_out: int
    frames_dir: pathlib.Path
    json_path: pathlib.Path
    config_path: pathlib.Path

    @property
    def tasks_written(self) -> int:
        """Tasks written to the import."""
        return len(self.tasks)


def prelabel(
    *,
    data: str | pathlib.Path | None = None,
    videos: str | pathlib.Path | None = None,
    out_json: str | pathlib.Path | None = None,
    config_path: str | pathlib.Path | None = None,
    min_duration: float = MIN_DURATION_S,
    open_video: Callable[[Any], Any] | None = None,
) -> PrelabelReport:
    """Write three frames per line hold and the Label Studio import for the review.

    The holds are the **line** holds of ``hold_summary.csv`` whose clip is not
    flagged ``missing`` in the catalogue and whose duration is at least
    ``min_duration`` seconds (:data:`MIN_DURATION_S` by default), longest first.
    A shorter hold can never be the reference, so it gets no task and no frames;
    its count is :attr:`PrelabelReport.left_out`, and ``min_duration=0`` queues
    every line hold. For each of the 25 %, 50 % and 75 % points of
    ``[hold_start_ms, hold_end_ms]`` the nearest frame is decoded in display
    orientation — the sampler's own decoder — and written as
    ``<clip_id>_h<hold_id>_<q>.jpg``. A frame file that already exists is left
    alone; a clip whose video is missing or unreadable is skipped with a note
    rather than ending the run.

    The import goes to ``<data>/label_studio_reference_review.json`` and the
    config to ``tools/labeling/reference_review_config.xml`` in this checkout.
    """
    root = _root(data)
    holds = hold_shapes.line_holds(
        hold_shapes.read_hold_summary(hold_summary_path(root)),
        hold_shapes.load_hold_shapes(root),
    )
    missing = _missing_clips(root)
    holds = [hold for hold in holds if str(hold["clip_id"]) not in missing]
    eligible = [hold for hold in holds if _hold_duration(hold) >= min_duration]
    left_out = len(holds) - len(eligible)
    holds = sorted(eligible, key=_hold_duration, reverse=True)

    destination = frames_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    json_out = pathlib.Path(out_json) if out_json is not None else tasks_path(root)
    config_out = pathlib.Path(config_path) if config_path is not None else default_config_path()
    read = open_video or DisplayVideo

    frames_written = 0
    frames_skipped = 0
    clips_skipped: list[tuple[str, str]] = []
    succeeded: set[str] = set()

    for clip_id, clip_holds in _group_by_clip(holds):
        pending: dict[tuple[Any, int], dict[str, Any]] = {}
        for hold in clip_holds:
            hold_id = hold["hold_id"]
            for quantile in QUANTILES:
                name = frame_name(clip_id, hold_id, quantile)
                if (destination / name).is_file():
                    frames_skipped += 1
                    continue
                pending[(hold_id, quantile)] = {
                    "name": name,
                    "target": _target_ms(hold, quantile),
                    "dist": math.inf,
                    "frame": None,
                }
        if not pending:
            # Every frame already exists: the video does not even need to open.
            succeeded.add(clip_id)
            continue

        try:
            video_path = video_for_clip(clip_id, videos, root)
        except (FileNotFoundError, ValueError) as error:
            clips_skipped.append((clip_id, str(error)))
            continue

        try:
            frames_seen = 0
            with read(video_path) as video:
                for packet in video:
                    frames_seen += 1
                    for key in list(pending):
                        item = pending[key]
                        distance = abs(packet.t_ms - item["target"])
                        if distance < item["dist"]:
                            item["dist"] = distance
                            item["frame"] = packet.frame
                        # Past the target the distance only grows, so the nearest
                        # frame to this target is already known.
                        if packet.t_ms > item["target"]:
                            write_jpeg(item["frame"], destination / item["name"])
                            frames_written += 1
                            del pending[key]
                    if not pending:
                        break
        except (RuntimeError, OSError) as error:
            clips_skipped.append((clip_id, str(error)))
            continue

        if frames_seen == 0:
            clips_skipped.append((clip_id, "the video decoded no frames"))
            continue
        # The video ended before these targets: the nearest frame that exists is
        # still the nearest frame.
        for item in pending.values():
            write_jpeg(item["frame"], destination / item["name"])
            frames_written += 1
        succeeded.add(clip_id)

    tasks = [
        build_task(str(hold["clip_id"]), hold["hold_id"], hold, destination, root)
        for hold in holds
        if str(hold["clip_id"]) in succeeded
    ]

    write_json(tasks, json_out)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(config_xml(), encoding="utf-8")

    return PrelabelReport(
        holds=len(holds),
        tasks=tasks,
        frames_written=frames_written,
        frames_skipped=frames_skipped,
        clips_skipped=clips_skipped,
        min_duration_s=min_duration,
        left_out=left_out,
        frames_dir=destination,
        json_path=json_out,
        config_path=config_out,
    )


def summarise_prelabel(report: PrelabelReport) -> str:
    """Multi-line summary of a pre-label run, for the CLI to print."""
    lines = [
        f"pre-labelled {report.holds} line hold(s) -> {report.tasks_written} task(s)",
        f"  frames: {report.frames_written} written, {report.frames_skipped} already "
        f"present -> {report.frames_dir}",
        f"  tasks: {report.json_path}",
        f"  config: {report.config_path}",
        f"  {report.left_out} line hold(s) shorter than "
        f"{report.min_duration_s:.1f} s left out",
    ]
    if report.clips_skipped:
        lines.append(f"  {len(report.clips_skipped)} clip(s) skipped:")
        lines += [f"    {clip}: {reason}" for clip, reason in report.clips_skipped]
    return "\n".join(lines)


def _missing_clips(root: pathlib.Path) -> set[str]:
    """The clip ids flagged ``missing`` in the catalogue."""
    path = catalogue_path(root)
    if not path.is_file():
        return set()
    return {
        (row.get("clip_id") or "").strip()
        for row in catalogue_module.read_rows(path)
        if _as_bool(row.get(catalogue_module.MISSING_COLUMN))
    }


# --------------------------------------------------------------------------- #
# 3. import
# --------------------------------------------------------------------------- #


def _choices(result: Mapping[str, Any]) -> list[str]:
    """The values of a ``Choices`` result, whatever shape the export used."""
    value = result.get("value")
    if isinstance(value, Mapping):
        choices = value.get("choices")
        if isinstance(choices, list):
            return [str(choice) for choice in choices]
        if choices is not None:
            return [str(choices)]
        return []
    if isinstance(value, list):
        return [str(choice) for choice in value]
    return []


def _annotation_choice(annotation: Mapping[str, Any], name: str) -> str | None:
    """The value one annotation chose for ``name``, or ``None`` when it chose none."""
    results = annotation.get("result")
    chosen: str | None = None
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, Mapping):
            continue
        if str(result.get("from_name") or "") != name:
            continue
        choices = _choices(result)
        if choices:
            chosen = choices[0]
    return chosen


def _annotation_has(annotation: Mapping[str, Any], name: str, value: str) -> bool:
    """Whether one annotation checked ``value`` under ``name``."""
    results = annotation.get("result")
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, Mapping):
            continue
        if str(result.get("from_name") or "") != name:
            continue
        if value in _choices(result):
            return True
    return False


def _labeled_at(annotation: Mapping[str, Any]) -> str:
    """When one annotation was made, best effort and never empty."""
    for key in ("updated_at", "created_at", "completed_at"):
        value = annotation.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _task_review(task: Mapping[str, Any]) -> tuple[str, bool, str]:
    """``(reference, wall, labeled_at)`` of a task: the last non-cancelled decision.

    The last annotation that chose a reference wins, because that is the
    decision the user ended on; an annotation with no reference is not a decision
    and does not erase the one before it. A cancelled annotation is ignored.
    """
    reference = ""
    wall = False
    labeled_at = ""
    annotations = task.get("annotations")
    for annotation in annotations if isinstance(annotations, list) else []:
        if not isinstance(annotation, Mapping):
            continue
        if _as_bool(annotation.get("was_cancelled")):
            continue
        chosen = _annotation_choice(annotation, REFERENCE_CHOICE)
        if chosen is None:
            continue
        reference = chosen
        wall = _annotation_has(annotation, WALL_CHOICE, WALL_CHOICE)
        labeled_at = _labeled_at(annotation)
    return reference, wall, labeled_at


def read_label_rows(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Read a ``reference_review.csv`` written earlier; a missing file is empty."""
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            clip_id = (row.get("clip_id") or "").strip()
            if not clip_id:
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "hold_id": _hold_key(row.get("hold_id")),
                    "reference": (row.get("reference") or "").strip(),
                    "wall": (row.get("wall") or "").strip().lower(),
                    "labeled_at": row.get("labeled_at") or "",
                }
            )
    return rows


def load_reference_holds(data: str | pathlib.Path | None = None) -> set[tuple[str, int | str]]:
    """The reviewed holds that are reference-worthy: ``(clip_id, hold_id)``.

    A hold counts only when the review answered ``yes`` **and** did not check
    ``wall``; an unlabelled, a ``no`` and a wall hold are all left out. This is
    what the reference build (#27/#28) reads.
    """
    return {
        (row["clip_id"], row["hold_id"])
        for row in read_label_rows(labels_path(_root(data)))
        if row["reference"] == REFERENCE_YES and row["wall"] != TRUE
    }


@dataclasses.dataclass(frozen=True)
class ImportReport:
    """What one import of a Label Studio export produced."""

    #: Tasks in the export.
    tasks: int
    #: Tasks that carried a reference decision.
    annotated: int
    #: Tasks skipped: cancelled, no decision, or no clip/hold in their data.
    skipped: int
    #: Rows in ``reference_review.csv`` after the import.
    rows: int
    #: Catalogue rows whose ``good_clip`` was written.
    good_clips: int
    #: Catalogue rows whose ``notes`` gained the word ``wall``.
    wall_notes: int
    out_path: pathlib.Path
    catalogue_path: pathlib.Path


def import_export(
    export_path: str | pathlib.Path,
    *,
    data: str | pathlib.Path | None = None,
    out_csv: str | pathlib.Path | None = None,
    catalogue: str | pathlib.Path | None = None,
) -> ImportReport:
    """Import a Label Studio export of the review, then update the catalogue.

    Each annotated task becomes one ``reference_review.csv`` row
    (``clip_id, hold_id, reference, wall, labeled_at``); the last non-cancelled
    annotation wins and tasks without a decision are skipped. Rows already in the
    CSV whose key is not in this export are kept, so re-importing is idempotent
    and importing a corrected export changes exactly the holds it mentions.

    The catalogue is then updated, backup first: ``good_clip`` is **overwritten**
    for every clip with a reviewed hold — ``1`` when at least one hold is a
    ``yes`` that is not a wall, ``0`` otherwise — because the review is the
    source of truth for it; and a clip with any wall hold gets ``wall`` appended
    to ``notes`` once.
    """
    root = _root(data)
    destination = pathlib.Path(out_csv) if out_csv is not None else labels_path(root)
    tasks = read_export(export_path)

    merged: dict[tuple[str, int | str], dict[str, Any]] = {
        (row["clip_id"], row["hold_id"]): dict(row) for row in read_label_rows(destination)
    }
    annotated = skipped = 0
    for task in tasks:
        task_data = task.get("data")
        if not isinstance(task_data, Mapping):
            skipped += 1
            continue
        clip_id = str(task_data.get("clip_id") or "").strip()
        raw_hold = task_data.get("hold_id")
        if not clip_id or raw_hold is None or str(raw_hold).strip() == "":
            skipped += 1
            continue
        reference, wall, labeled_at = _task_review(task)
        if not reference:
            skipped += 1
            continue
        annotated += 1
        hold_id = _hold_key(raw_hold)
        merged[(clip_id, hold_id)] = {
            "clip_id": clip_id,
            "hold_id": hold_id,
            "reference": reference,
            "wall": TRUE if wall else FALSE,
            "labeled_at": labeled_at,
        }

    ordered = sorted(
        merged.values(), key=lambda row: (row["clip_id"], _hold_sort_key(row["hold_id"]))
    )
    _write_csv(destination, LABEL_COLUMNS, ordered)

    catalogue_file = pathlib.Path(catalogue) if catalogue is not None else catalogue_path(root)
    good_clips, wall_notes = _update_catalogue(ordered, catalogue_file)
    return ImportReport(
        tasks=len(tasks),
        annotated=annotated,
        skipped=skipped,
        rows=len(ordered),
        good_clips=good_clips,
        wall_notes=wall_notes,
        out_path=destination,
        catalogue_path=catalogue_file,
    )


def _update_catalogue(
    reviews: Sequence[Mapping[str, Any]], path: pathlib.Path
) -> tuple[int, int]:
    """Write ``good_clip`` and wall notes from the reviewed rows; backup first.

    Only clips that have at least one reviewed hold are touched. Returns
    ``(clips with good_clip written, clips whose notes gained "wall")``.
    """
    by_clip: dict[str, list[Mapping[str, Any]]] = {}
    for row in reviews:
        by_clip.setdefault(str(row["clip_id"]), []).append(row)
    if not by_clip or not path.is_file():
        return 0, 0

    rows = catalogue_module.read_rows(path)
    missing_column = catalogue_module.MISSING_COLUMN
    good_clips = wall_notes = 0
    changed = False
    for row in rows:
        if _as_bool(row.get(missing_column)):
            continue
        clip_id = (row.get("clip_id") or "").strip()
        clip_reviews = by_clip.get(clip_id)
        if not clip_reviews:
            continue
        good = any(
            review["reference"] == REFERENCE_YES and review["wall"] != TRUE
            for review in clip_reviews
        )
        row["good_clip"] = "1" if good else "0"
        good_clips += 1
        changed = True
        if any(review["wall"] == TRUE for review in clip_reviews) and not _has_word(
            row.get("notes"), WALL_NOTE
        ):
            row["notes"] = _append_note(row.get("notes") or "", WALL_NOTE)
            wall_notes += 1
    if not changed:
        return 0, 0
    _backup_catalogue(path)
    catalogue_module.write_rows(rows, path)
    return good_clips, wall_notes


def summarise_import(report: ImportReport) -> str:
    """Multi-line summary of an import, for the CLI to print."""
    lines = [
        f"imported {report.annotated} of {report.tasks} task(s) -> {report.out_path}",
        f"  {report.rows} reviewed hold(s); {report.skipped} task(s) skipped "
        "(cancelled, no decision, or no clip/hold in the task data)",
        f"  catalogue: good_clip written for {report.good_clips} clip(s), "
        f"wall note added to {report.wall_notes} -> {report.catalogue_path}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.reference_review`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.reference_review",
        description=(
            "Auto-fill the catalogue, pre-label the LINE holds for the reference "
            "review, and import the reviewed export."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fill_parser = commands.add_parser(
        "fill",
        help="fill the EMPTY catalogue cells from the measured data (backup first)",
    )
    fill_parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory (default: $HANDSTAND_DATA)",
    )
    fill_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the same summary and write nothing",
    )

    pre = commands.add_parser(
        "prelabel",
        help="write three frames per line hold, the LS import and the config",
    )
    pre.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory (default: $HANDSTAND_DATA)",
    )
    pre.add_argument(
        "--videos",
        type=pathlib.Path,
        default=None,
        help="raw videos to decode (default: $HANDSTAND_VIDEOS)",
    )
    pre.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=f"Label Studio import to write (default: <data_dir>/{TASKS_FILENAME})",
    )
    pre.add_argument(
        "--min-duration",
        type=float,
        default=MIN_DURATION_S,
        metavar="SECONDS",
        help=(
            "skip line holds shorter than this many seconds; they can never be "
            f"the reference (default: {MIN_DURATION_S:g}; 0 keeps every hold)"
        ),
    )
    pre.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=f"labeling config to write (default: <repo>/{pathlib.Path(*CONFIG_RELATIVE)})",
    )

    imp = commands.add_parser(
        "import",
        help="convert a Label Studio export into reference_review.csv and the catalogue",
    )
    imp.add_argument("export", type=pathlib.Path, help="the exported JSON")
    imp.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory (default: $HANDSTAND_DATA)",
    )
    imp.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=f"labels CSV to write (default: <data_dir>/labels/{LABELS_CSV_NAME})",
    )
    imp.add_argument(
        "--catalogue",
        type=pathlib.Path,
        default=None,
        help=f"catalogue to update (default: <data_dir>/{CATALOGUE_NAME})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.reference_review``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "fill":
            print(summarise_fill(fill(data=args.data, dry_run=args.dry_run)))
            return 0
        if args.command == "prelabel":
            print(
                summarise_prelabel(
                    prelabel(
                        data=args.data,
                        videos=args.videos,
                        out_json=args.out,
                        config_path=args.config,
                        min_duration=args.min_duration,
                    )
                )
            )
            return 0

        print(
            summarise_import(
                import_export(
                    args.export,
                    data=args.data,
                    out_csv=args.out,
                    catalogue=args.catalogue,
                )
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        # ValueError covers a broken export JSON; OSError covers a missing
        # hold summary, catalogue or video directory.
        print(f"reference_review: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
