"""Label the SHAPE of every segment of a mixed hold, from several frames.

The hold-shape review (:mod:`handstand.hold_shapes`) shows **one** frame per
hold: the middle one. That label describes that snapshot, not the whole hold,
and a *mixed* hold (:mod:`handstand.segments`) changes shape inside itself — a
flow is a straddle part and a line part. This module gives every segment of a
mixed hold its own Label Studio task, made from a strip of three frames at
25 %, 50 % and 75 % of the segment, so the labeler decides from the segment
rather than from one moment of it.

The loop mirrors the hold-shape one::

    handstand.segment_shapes prelabel -> data/label_frames_segments/<segment_key>.jpg
                                          + manifest.csv
                                          + data/label_studio_segment_shapes.json
                                          + tools/labeling/segment_shapes_config.xml
    Label Studio (separate)          -> an export JSON
    handstand.segment_shapes import  -> data/labels/segment_shapes.csv
    handstand.segment_shapes summary -> counts and the line windows
    handstand.segment_shapes push    -> the tasks into the Label Studio project

Everything it writes from videos lives under ``data/`` and is git-ignored; only
the generated labeling config sits in the repository, under ``tools/labeling/``.

CLI::

    cd pipeline
    uv run python -m handstand.segment_shapes prelabel
    uv run python -m handstand.segment_shapes import ../data/label_studio_segment_shapes_export.json
    uv run python -m handstand.segment_shapes summary
    uv run python -m handstand.segment_shapes push
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import pathlib
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from handstand import hold_shapes
from handstand import segments as segments_module
from handstand.features import DEFAULT_SOURCE, HOLD_SUMMARY_NAME, output_dir
from handstand.frame_sampler import write_jpeg
from handstand.ingest import (
    DEFAULT_LS_URL,
    LabelStudioClient,
    LabelStudioError,
    read_ls_credentials,
)
from handstand.labels import LABELS_DIRNAME, read_export
from handstand.overlay import video_for_clip
from handstand.paths import data_dir
from handstand.pose_mediapipe import DisplayVideo
from handstand.prelabel import image_uri, write_json

__all__ = [
    "CONFIG_RELATIVE",
    "FRAMES_DIRNAME",
    "LABEL_COLUMNS",
    "LABELS_CSV_NAME",
    "MANIFEST_COLUMNS",
    "MANIFEST_NAME",
    "MAX_STRIP_HEIGHT",
    "MODEL_VERSION",
    "PROJECT_TITLE",
    "SHORT_SEGMENT_MS",
    "TASKS_FILENAME",
    "ImportReport",
    "PrelabelReport",
    "PushReport",
    "build_arg_parser",
    "build_task",
    "config_xml",
    "default_config_path",
    "frames_dir",
    "import_export",
    "line_windows",
    "load_segment_shapes",
    "main",
    "manifest_path",
    "prelabel",
    "push",
    "read_label_rows",
    "segment_header",
    "segment_key",
    "segment_times",
    "strip_image",
    "summarise",
    "summarise_import",
    "summarise_prelabel",
    "summarise_push",
    "task_image_uri",
    "tasks_path",
    "labels_path",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Name of the output directory inside the data directory: one strip per segment.
FRAMES_DIRNAME = "label_frames_segments"
#: Name of the manifest inside :data:`FRAMES_DIRNAME`.
MANIFEST_NAME = "manifest.csv"
#: The Label Studio import, under ``<data_dir>/``.
TASKS_FILENAME = "label_studio_segment_shapes.json"
#: The imported labels, under ``<data_dir>/labels/``.
LABELS_CSV_NAME = "segment_shapes.csv"
#: Where the generated labeling config lives, relative to the repository root.
CONFIG_RELATIVE = ("tools", "labeling", "segment_shapes_config.xml")
#: The Label Studio project this module's tasks belong to.
PROJECT_TITLE = "handstand-segment-shapes"

#: The tallest a frame in the strip may be, in pixels.
MAX_STRIP_HEIGHT = 480
#: A segment shorter than this (ms) gets its middle frame only, because three
#: frames at 25/50/75 % of a few hundred milliseconds would be the same frame.
SHORT_SEGMENT_MS = 300
#: The fractions of the segment at which the three strip frames are decoded.
STRIP_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75)
#: Fractions used for a segment shorter than :data:`SHORT_SEGMENT_MS`.
SHORT_FRACTIONS: tuple[float, ...] = (0.5,)

#: Recorded in every prediction so a later run can tell which rules wrote it.
MODEL_VERSION = "segment-shapes-prelabel"

#: Full manifest header, in order.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "image",
    "clip_id",
    "hold_id",
    "segment_id",
    "segment_key",
    "start_ms",
    "end_ms",
    "duration_s",
    "hold_label",
    "predicted_shape",
)

#: Full header of ``segment_shapes.csv``, in order.
LABEL_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "hold_id",
    "segment_id",
    "start_ms",
    "end_ms",
    "shape",
    "predicted",
    "labeled_at",
)

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _as_int(value: object, default: int = 0) -> int:
    """Coerce a cell to an int, ``default`` when it is blank or junk."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return int(number) if math.isfinite(number) else default


def _as_bool(value: object) -> bool:
    """A CSV/pandas flag as a bool: ``True``, ``"true"`` and ``1`` are true."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _format_number(value: float) -> str:
    """A number for a header, without a trailing ``.0``."""
    return f"{value:g}"


def _seconds_text(ms: int) -> str:
    """Milliseconds as seconds for a header, e.g. ``2000`` -> ``2``."""
    return _format_number(ms / 1000.0)


def segment_key(clip_id: str, hold_id: object, segment_id: int | str) -> str:
    """The identity suffix of one segment: ``<clip_id>_h<hold_id>_s<segment_id>``."""
    return f"{clip_id}_h{hold_shapes._hold_key(hold_id)}_s{_as_int(segment_id)}"


def segment_times(start_ms: int, end_ms: int) -> list[int]:
    """The times (ms) at which a segment's strip frames are decoded.

    Three frames at 25 %, 50 % and 75 % of ``[start_ms, end_ms]``; a segment
    shorter than :data:`SHORT_SEGMENT_MS` gets only its middle frame.
    """
    span = max(0, int(end_ms) - int(start_ms))
    fractions = SHORT_FRACTIONS if span < SHORT_SEGMENT_MS else STRIP_FRACTIONS
    return [int(round(start_ms + span * fraction)) for fraction in fractions]


def segment_header(
    *,
    clip_id: str,
    hold_id: object,
    segment_id: int,
    segments: int,
    start_ms: int,
    end_ms: int,
    duration_s: float,
    hold_label: str,
) -> str:
    """The single ``info`` line shown above a segment's strip."""
    label = hold_label or "unlabelled"
    return (
        f"{clip_id} hold {hold_shapes._hold_key(hold_id)} "
        f"segment {segment_id + 1}/{segments}: "
        f"{_seconds_text(start_ms)}-{_seconds_text(end_ms)} s "
        f"({_format_number(duration_s)} s), hold labelled {label}"
    )


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def frames_dir(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_frames_segments``, where the strips and manifest go."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / FRAMES_DIRNAME


def manifest_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_frames_segments/manifest.csv``."""
    return frames_dir(data) / MANIFEST_NAME


def tasks_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_studio_segment_shapes.json``, the Label Studio import."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / TASKS_FILENAME


def labels_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/labels/segment_shapes.csv``, the reviewed labels."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / LABELS_DIRNAME / LABELS_CSV_NAME


def default_config_path() -> pathlib.Path:
    """``tools/labeling/segment_shapes_config.xml`` in this checkout."""
    return pathlib.Path(__file__).resolve().parents[2].joinpath(*CONFIG_RELATIVE)


# --------------------------------------------------------------------------- #
# The Label Studio config and tasks
# --------------------------------------------------------------------------- #


def config_xml() -> str:
    """The ``tools/labeling/segment_shapes_config.xml`` contents.

    The choices and hotkeys are :mod:`handstand.hold_shapes`' own constants, so
    the two review projects cannot drift apart: line 1 through mexican 9. The
    header reads the single ``$info`` task field :func:`build_task` writes.
    """
    choices = "\n".join(
        f'    <Choice value="{shape}" hotkey="{hold_shapes.HOTKEYS[shape]}"/>'
        for shape in hold_shapes.SHAPES
    )
    tag = (
        f'  <Choices name="{hold_shapes.CHOICE_NAME}" toName="{hold_shapes.IMAGE_NAME}" '
        'choice="single" required="true" showInline="true">'
    )
    return f"""<View>
  <!--
    Generated by pipeline/handstand/segment_shapes.py (prelabel); edit the
    module, not this file. The Choice values are handstand.hold_shapes.SHAPES
    and the hotkeys are its HOTKEYS, so this project labels line 1 through
    mexican 9 exactly like handstand-hold-shapes.

    The image is a strip of three frames of ONE segment of a mixed hold (25 %,
    50 % and 75 % of it), with the decoded time in seconds in each corner. The
    preselected choice is the segment's shape_guess, not a decision: check the
    whole strip and change it when it is wrong. A task with no preselection is
    a segment the features could not measure.
  -->
  <Text name="segment_info" value="$info" density="1"/>
  <Image name="{hold_shapes.IMAGE_NAME}" value="$image" zoom="true" zoomControl="true"/>
{tag}
{choices}
  </Choices>
</View>
"""


def task_image_uri(
    image: str | pathlib.Path,
    image_base: str | None = None,
    data: str | pathlib.Path | None = None,
) -> str:
    """The ``data.image`` URL for one segment's strip.

    The default is Label Studio's local-files form relative to the data dir,
    ``/data/local-files/?d=label_frames_segments/<segment_key>.jpg``.
    ``image_base`` serves the file from somewhere else instead, exactly as
    :func:`handstand.hold_shapes.task_image_uri` does.
    """
    path = pathlib.Path(image)
    if image_base:
        return image_uri(path, image_base)
    root = pathlib.Path(data) if data is not None else data_dir()
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = pathlib.Path(FRAMES_DIRNAME) / path.name
    return f"{hold_shapes.LOCAL_FILES_URL}{relative.as_posix()}"


def build_task(
    image: str | pathlib.Path,
    *,
    clip_id: str,
    hold_id: object,
    segment_id: int,
    segments: int,
    segment_key: str,
    start_ms: int,
    end_ms: int,
    duration_s: float,
    hold_label: str,
    shape: str,
    image_base: str | None = None,
    data: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """One Label Studio task for one segment of a mixed hold.

    ``shape`` preselects a choice only when it is one of
    :data:`handstand.hold_shapes.SHAPES`: an ``unmeasured`` segment (or any
    other guess) arrives with no ``predictions`` at all rather than a default
    that would label it by accident.
    """
    header = segment_header(
        clip_id=clip_id,
        hold_id=hold_id,
        segment_id=segment_id,
        segments=segments,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_s=duration_s,
        hold_label=hold_label,
    )
    task: dict[str, Any] = {
        "data": {
            "image": task_image_uri(image, image_base, data),
            "clip_id": clip_id,
            "hold_id": hold_shapes._hold_key(hold_id),
            "segment_id": int(segment_id),
            "segment_key": segment_key,
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "duration_s": float(duration_s),
            "hold_label": hold_label,
            "info": header,
        }
    }
    if shape in hold_shapes.SHAPES:
        task["predictions"] = [
            {
                "model_version": MODEL_VERSION,
                "score": 1.0,
                "result": [
                    {
                        "type": "choices",
                        "from_name": hold_shapes.CHOICE_NAME,
                        "to_name": hold_shapes.IMAGE_NAME,
                        "value": {"choices": [shape]},
                    }
                ],
            }
        ]
    return task


# --------------------------------------------------------------------------- #
# The strip image
# --------------------------------------------------------------------------- #


def _draw_time(frame: np.ndarray, t_ms: int) -> None:
    """Draw ``<seconds>s`` in the frame's top-left corner, white on black."""
    text = f"{t_ms / 1000.0:.1f}s"
    height = frame.shape[0]
    scale = max(0.4, min(1.2, height / 480.0))
    thickness = max(1, int(round(scale * 2)))
    position = (max(2, int(round(height * 0.04))), max(10, int(round(height * 0.15))))
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def strip_image(
    frames: Sequence[np.ndarray],
    times_ms: Sequence[int],
    max_height: int = MAX_STRIP_HEIGHT,
) -> np.ndarray:
    """The horizontal strip of ``frames``, all the same height, each time drawn.

    Every frame is scaled to the shortest input frame's height, capped at
    ``max_height``, so the strip joins cleanly. Pure: no I/O.
    """
    if not frames:
        raise ValueError("a strip needs at least one frame")
    if len(frames) != len(times_ms):
        raise ValueError(f"{len(frames)} frames but {len(times_ms)} times")
    target_height = min(max_height, *(frame.shape[0] for frame in frames))
    if target_height < 1:
        raise ValueError("a frame has no height")
    tiles: list[np.ndarray] = []
    for frame, t_ms in zip(frames, times_ms, strict=True):
        height, width = frame.shape[:2]
        tile_width = max(1, int(round(width * target_height / height)))
        tile = cv2.resize(frame, (tile_width, target_height), interpolation=cv2.INTER_AREA)
        tile = np.ascontiguousarray(tile)
        _draw_time(tile, int(t_ms))
        tiles.append(tile)
    return np.hstack(tiles)


# --------------------------------------------------------------------------- #
# Pre-label
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class PrelabelReport:
    """What one pre-label run produced."""

    #: The manifest rows: one per written segment.
    rows: list[dict[str, Any]]
    #: The Label Studio tasks: one per written segment.
    tasks: list[dict[str, Any]]
    #: Segments of mixed holds processed (written + skipped with their clips).
    segments: int
    #: Written segments per predicted shape, every hold shape plus unmeasured.
    counts: dict[str, int]
    #: ``(clip_id, reason)`` for every clip whose video could not be read.
    skipped: list[tuple[str, str]]
    frames_dir: pathlib.Path
    manifest: pathlib.Path
    json_path: pathlib.Path
    config_path: pathlib.Path
    runtime_seconds: float = 0.0
    #: Segments ``skip_existing`` left alone: already queued or reviewed.
    already_done: int = 0

    @property
    def written(self) -> int:
        """Segments with an image, a manifest row and a task."""
        return len(self.rows)

    @property
    def clips(self) -> int:
        """Distinct clips that contributed at least one image."""
        return len({str(row["clip_id"]) for row in self.rows})

    @property
    def skipped_segments(self) -> int:
        """Segments lost with their clips' videos."""
        return self.segments - self.written


def _segment_key_from_row(row: Mapping[str, Any]) -> str:
    return segment_key(
        str(row.get("clip_id") or "").strip(),
        row.get("hold_id"),
        row.get("segment_id"),  # type: ignore[arg-type]
    )


def _merge_manifest(
    path: pathlib.Path, rows: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """The manifest on disk minus this run's rows, then this run's rows."""
    existing: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = [dict(row) for row in csv.DictReader(handle)]
    fresh = {str(row.get("segment_key") or "") for row in rows}
    return [row for row in existing if str(row.get("segment_key") or "") not in fresh] + list(rows)


def _task_key(task: Any) -> str | None:
    """The identity of a task: its ``segment_key``, or ``None`` when absent."""
    data = task.get("data") if isinstance(task, Mapping) else None
    if not isinstance(data, Mapping):
        return None
    key = str(data.get("segment_key") or "").strip()
    return key or None


def _merge_tasks(path: pathlib.Path, tasks: Sequence[Any]) -> list[Any]:
    """The task file on disk minus this run's tasks, then this run's tasks."""
    existing: list[Any] = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, list):
            existing = loaded
    fresh = {key for key in (_task_key(task) for task in tasks) if key is not None}
    return [task for task in existing if _task_key(task) not in fresh] + list(tasks)


def _finish_segment(
    item: Mapping[str, Any],
    *,
    destination: pathlib.Path,
    root: pathlib.Path,
    labels: Mapping[tuple[str, int | str], str],
    image_base: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write one segment's strip, and return its manifest row and task."""
    clip_id = str(item["clip_id"])
    hold_id = item["hold_id"]
    segment_id = int(item["segment_id"])
    key = str(item["segment_key"])
    start_ms = int(item["start_ms"])
    end_ms = int(item["end_ms"])
    duration_s = float(item["duration_s"])
    shape = str(item["shape_guess"])
    hold_label = labels.get((clip_id, hold_shapes._hold_key(hold_id)), "")
    targets = [target for target in item["targets"] if target["frame"] is not None]
    if not targets:
        raise RuntimeError("no frame was decoded for this segment")
    frames = [target["frame"] for target in targets]
    times = [int(target["t_ms"]) for target in targets]
    strip = strip_image(frames, times)
    name = f"{key}.jpg"
    write_jpeg(strip, destination / name)
    row = {
        "image": name,
        "clip_id": clip_id,
        "hold_id": hold_id,
        "segment_id": segment_id,
        "segment_key": key,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_s": duration_s,
        "hold_label": hold_label,
        "predicted_shape": shape,
    }
    task = build_task(
        destination / name,
        clip_id=clip_id,
        hold_id=hold_id,
        segment_id=segment_id,
        segments=int(item["segments"]),
        segment_key=key,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_s=duration_s,
        hold_label=hold_label,
        shape=shape,
        image_base=image_base,
        data=root,
    )
    return row, task


def prelabel(
    *,
    data: str | pathlib.Path | None = None,
    videos: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
    clips: Iterable[str] | None = None,
    skip_existing: bool = False,
    image_base: str | None = None,
    out_json: str | pathlib.Path | None = None,
    config_path: str | pathlib.Path | None = None,
    open_video: Any = None,
) -> PrelabelReport:
    """Write a three-frame strip and a task for every mixed-hold segment.

    Only segments whose hold is ``mixed`` in
    ``<data>/features/<source>/hold_segments.csv`` get a task: a hold reviewed
    as one shape already has its hold-shape review. Each strip shows the
    display-oriented frames nearest 25 %, 50 % and 75 % of the segment (the
    middle frame only for a segment under 300 ms), scaled to the same height
    and capped at 480 px, with each frame's time in its corner.

    ``skip_existing`` skips segments that already have an image or a reviewed
    row in ``segment_shapes.csv`` and merges the manifest/task file with what is
    already on disk, so a rerun adds only new segment keys.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    table = segments_module.read_segments(root, source)
    if table.empty:
        rows: list[dict[str, Any]] = []
    else:
        table = table[table["mixed"].map(_as_bool)]  # type: ignore[attr-defined]
        rows = table.to_dict("records")
    if clips is not None:
        wanted = {str(clip).strip() for clip in clips}
        rows = [row for row in rows if str(row["clip_id"]) in wanted]
    rows.sort(
        key=lambda row: (
            str(row["clip_id"]),
            hold_shapes._hold_sort_key(row["hold_id"]),
            _as_int(row["segment_id"]),
        )
    )

    totals: Counter[tuple[str, int | str]] = Counter()
    all_mixed = segments_module.read_segments(root, source)
    if not all_mixed.empty:
        for row in all_mixed[all_mixed["mixed"].map(_as_bool)].to_dict("records"):
            totals[(str(row["clip_id"]), hold_shapes._hold_key(row["hold_id"]))] += 1

    destination = frames_dir(root)
    if not destination.is_absolute():
        destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_out = pathlib.Path(out_json) if out_json is not None else tasks_path(root)
    config_out = pathlib.Path(config_path) if config_path is not None else default_config_path()
    read = open_video or DisplayVideo
    labels = hold_shapes.load_hold_shapes(root)
    reviewed = load_segment_shapes(root) if skip_existing else {}

    manifest_rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    already_done = 0

    # Group the segments by clip so one decode serves every strip of that clip.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        hold_id = hold_shapes._hold_key(row["hold_id"])
        segment_id = _as_int(row["segment_id"])
        key = segment_key(str(row["clip_id"]), hold_id, segment_id)
        if skip_existing and (
            (str(row["clip_id"]), hold_id, segment_id) in reviewed
            or (destination / f"{key}.jpg").is_file()
        ):
            already_done += 1
            continue
        start_ms = _as_int(row["start_ms"])
        end_ms = _as_int(row["end_ms"])
        duration_s = float(row.get("duration_s") or (end_ms - start_ms) / 1000.0)
        item = {
            "clip_id": str(row["clip_id"]),
            "hold_id": hold_id,
            "segment_id": segment_id,
            "segment_key": key,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_s": duration_s,
            "shape_guess": str(row.get("shape_guess") or ""),
            "segments": totals[(str(row["clip_id"]), hold_id)],
            "targets": [
                {"ms": t_ms, "dist": math.inf, "t_ms": -1, "frame": None, "done": False}
                for t_ms in segment_times(start_ms, end_ms)
            ],
        }
        groups.setdefault(str(row["clip_id"]), []).append(item)

    started = time.perf_counter()
    for clip_id, pending in groups.items():
        try:
            video_path = video_for_clip(clip_id, videos, root)
        except (FileNotFoundError, ValueError) as error:
            skipped.append((clip_id, str(error)))
            continue

        try:
            with read(video_path) as video:
                for packet in video:
                    for item in pending:
                        for target in item["targets"]:
                            if target["done"]:
                                continue
                            distance = abs(packet.t_ms - target["ms"])
                            if target["frame"] is None or distance < target["dist"]:
                                target["dist"] = distance
                                target["t_ms"] = packet.t_ms
                                target["frame"] = packet.frame
                            # Past the target the distance only grows, so this
                            # target's nearest frame is already known.
                            if packet.t_ms > target["ms"]:
                                target["done"] = True
                    ready = [item for item in pending if all(t["done"] for t in item["targets"])]
                    for item in ready:
                        try:
                            row, task = _finish_segment(
                                item,
                                destination=destination,
                                root=root,
                                labels=labels,
                                image_base=image_base,
                            )
                        except RuntimeError:
                            continue
                        counts[str(item["shape_guess"])] += 1
                        manifest_rows.append(row)
                        tasks.append(task)
                        pending.remove(item)
                    if not pending:
                        break
        except (RuntimeError, OSError) as error:
            skipped.append((clip_id, str(error)))
            continue

        for item in list(pending):
            try:
                row, task = _finish_segment(
                    item,
                    destination=destination,
                    root=root,
                    labels=labels,
                    image_base=image_base,
                )
            except RuntimeError:
                skipped.append((clip_id, "the video decoded no frames"))
                continue
            counts[str(item["shape_guess"])] += 1
            manifest_rows.append(row)
            tasks.append(task)

    runtime = time.perf_counter() - started

    manifest_out = destination / MANIFEST_NAME
    manifest_final: Sequence[Mapping[str, Any]] = manifest_rows
    task_list: list[Any] = tasks
    if clips is not None or skip_existing:
        manifest_final = _merge_manifest(manifest_out, manifest_rows)
        task_list = _merge_tasks(json_out, tasks)
    hold_shapes._write_csv(manifest_out, MANIFEST_COLUMNS, manifest_final)
    write_json(task_list, json_out)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(config_xml(), encoding="utf-8")

    return PrelabelReport(
        rows=manifest_rows,
        tasks=tasks,
        segments=len(rows) + already_done,
        counts={
            shape: counts.get(shape, 0) for shape in (*hold_shapes.SHAPES, hold_shapes.UNMEASURED)
        },
        skipped=skipped,
        frames_dir=destination,
        manifest=manifest_out,
        json_path=json_out,
        config_path=config_out,
        runtime_seconds=runtime,
        already_done=already_done,
    )


def summarise_prelabel(report: PrelabelReport) -> str:
    """Multi-line summary of a pre-label run, for the CLI to print."""
    per_shape = " ".join(
        f"{shape}={report.counts.get(shape, 0)}"
        for shape in (*hold_shapes.SHAPES, hold_shapes.UNMEASURED)
    )
    lines = [
        f"pre-labelled {report.written} of {report.segments} mixed-hold segment(s) "
        f"in {report.runtime_seconds:.0f}s -> {report.manifest}",
        f"  written per predicted shape: {per_shape}",
        f"  images: {report.frames_dir} ({report.clips} clip(s))",
        f"  tasks: {report.json_path}",
        f"  config: {report.config_path}",
    ]
    if report.skipped:
        lines.append(
            f"  {report.skipped_segments} segment(s) skipped with {len(report.skipped)} clip(s):"
        )
        lines += [f"    {clip}: {reason}" for clip, reason in report.skipped]
    if report.already_done:
        lines.append(
            f"  {report.already_done} segment(s) already queued or reviewed, left untouched"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ImportReport:
    """What one import of a Label Studio export produced."""

    tasks: int
    annotated: int
    skipped: int
    added: int
    updated: int
    unchanged: int
    unknown_shapes: tuple[str, ...]
    rows: int
    out_path: pathlib.Path


def read_label_rows(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Read a ``segment_shapes.csv`` written earlier; a missing file is empty."""
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(LABEL_COLUMNS) <= set(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {', '.join(LABEL_COLUMNS)}")
        for raw in reader:
            clip_id = str(raw.get("clip_id") or "").strip()
            shape = str(raw.get("shape") or "").strip()
            if not clip_id or not shape:
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "hold_id": hold_shapes._hold_key(raw.get("hold_id")),
                    "segment_id": _as_int(raw.get("segment_id")),
                    "start_ms": _as_int(raw.get("start_ms")),
                    "end_ms": _as_int(raw.get("end_ms")),
                    "shape": shape,
                    "predicted": str(raw.get("predicted") or "").strip(),
                    "labeled_at": str(raw.get("labeled_at") or ""),
                }
            )
    return rows


def _manifest_predictions(path: pathlib.Path) -> dict[str, str]:
    """``segment_key -> predicted_shape`` from a pre-label manifest."""
    if not path.is_file():
        return {}
    found: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("segment_key") or "").strip()
            if key:
                found[key] = str(row.get("predicted_shape") or "").strip()
    return found


def import_export(
    export_path: str | pathlib.Path,
    *,
    data: str | pathlib.Path | None = None,
    out_csv: str | pathlib.Path | None = None,
    manifest: str | pathlib.Path | None = None,
) -> ImportReport:
    """Import a Label Studio export into ``<data_dir>/labels/segment_shapes.csv``.

    The last annotation of a task wins; tasks without a choice are skipped, and
    rows already in the CSV whose key is not in this export are kept.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    destination = pathlib.Path(out_csv) if out_csv is not None else labels_path(root)
    manifest_file = pathlib.Path(manifest) if manifest is not None else manifest_path(root)

    tasks = read_export(export_path)
    predicted = _manifest_predictions(manifest_file)
    merged: dict[tuple[str, int | str, int], dict[str, Any]] = {
        (row["clip_id"], hold_shapes._hold_key(row["hold_id"]), row["segment_id"]): dict(row)
        for row in read_label_rows(destination)
    }

    annotated = skipped = added = updated = unchanged = 0
    unknown: set[str] = set()
    for task in tasks:
        task_data = task.get("data")
        if not isinstance(task_data, Mapping):
            skipped += 1
            continue
        clip_id = str(task_data.get("clip_id") or "").strip()
        raw_hold = task_data.get("hold_id")
        raw_segment = task_data.get("segment_id")
        if (
            not clip_id
            or raw_hold is None
            or str(raw_hold).strip() == ""
            or raw_segment is None
            or str(raw_segment).strip() == ""
        ):
            skipped += 1
            continue
        hold_id = hold_shapes._hold_key(raw_hold)
        segment_id = _as_int(raw_segment)
        shape, labeled_at = hold_shapes._task_shape(task)
        if not shape:
            skipped += 1
            continue
        annotated += 1
        if shape not in hold_shapes.SHAPES:
            unknown.add(shape)
        key = (clip_id, hold_id, segment_id)
        task_key = str(task_data.get("segment_key") or segment_key(clip_id, hold_id, segment_id))
        prediction = hold_shapes._prediction_shape(task) or predicted.get(task_key, "")
        row = {
            "clip_id": clip_id,
            "hold_id": hold_id,
            "segment_id": segment_id,
            "start_ms": _as_int(task_data.get("start_ms")),
            "end_ms": _as_int(task_data.get("end_ms")),
            "shape": shape,
            "predicted": prediction,
            "labeled_at": labeled_at,
        }
        previous = merged.get(key)
        if previous is None:
            added += 1
        elif previous == row:
            unchanged += 1
        else:
            updated += 1
        merged[key] = row

    ordered = sorted(
        merged.values(),
        key=lambda row: (
            row["clip_id"],
            hold_shapes._hold_sort_key(row["hold_id"]),
            row["segment_id"],
        ),
    )
    hold_shapes._write_csv(destination, LABEL_COLUMNS, ordered)
    return ImportReport(
        tasks=len(tasks),
        annotated=annotated,
        skipped=skipped,
        added=added,
        updated=updated,
        unchanged=unchanged,
        unknown_shapes=tuple(sorted(unknown)),
        rows=len(ordered),
        out_path=destination,
    )


def summarise_import(report: ImportReport) -> str:
    """Multi-line summary of an import, for the CLI to print."""
    lines = [
        f"imported {report.annotated} of {report.tasks} task(s): {report.added} new, "
        f"{report.updated} changed, {report.unchanged} unchanged -> {report.out_path}",
        f"  {report.rows} label(s) in total; {report.skipped} task(s) skipped "
        "(no annotation, or no clip/hold/segment in the task data)",
    ]
    if report.unknown_shapes:
        lines.append("  not in SHAPES, imported as typed: " + ", ".join(report.unknown_shapes))
    return "\n".join(lines)


def load_segment_shapes(
    data: str | pathlib.Path | None = None,
) -> dict[tuple[str, int, int], str]:
    """Every reviewed segment shape: ``(clip_id, hold_id, segment_id) -> shape``."""
    return {
        (
            row["clip_id"],
            int(hold_shapes._hold_key(row["hold_id"])),
            int(row["segment_id"]),
        ): row["shape"]
        for row in read_label_rows(labels_path(data))
    }


# --------------------------------------------------------------------------- #
# Line windows and the summary
# --------------------------------------------------------------------------- #


def line_windows(
    data: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
    min_s: float = 2.0,
) -> list[dict[str, Any]]:
    """Every line window the review has confirmed, as ``hold`` or ``segment``.

    A window is:

    * a reviewed **line hold** that is not mixed in ``hold_segments.csv`` (a
      hold missing from that table counts as not mixed), or
    * a segment labelled ``line`` in ``segment_shapes.csv`` lasting at least
      ``min_s`` seconds.

    A mixed hold labelled ``line`` contributes only through its reviewed
    segments: its hold label came from one snapshot, so it is never a window by
    itself. Sorted by clip, hold and segment (a whole hold before its segments).
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    labels = hold_shapes.load_hold_shapes(root)
    summary_path = output_dir(root, source) / HOLD_SUMMARY_NAME
    holds = hold_shapes.read_hold_summary(summary_path)
    table = segments_module.read_segments(root, source)
    mixed: set[tuple[str, int | str]] = set()
    if not table.empty:
        for row in table.to_dict("records"):
            if _as_bool(row.get("mixed")):
                mixed.add((str(row["clip_id"]), hold_shapes._hold_key(row.get("hold_id"))))

    windows: list[dict[str, Any]] = []
    for hold in holds:
        clip_id = str(hold["clip_id"])
        hold_id = hold_shapes._hold_key(hold["hold_id"])
        if not hold_shapes.is_line_hold(clip_id, hold_id, labels):
            continue
        if (clip_id, hold_id) in mixed:
            continue
        start_ms = _as_int(hold.get("hold_start_ms"))
        end_ms = _as_int(hold.get("hold_end_ms"), start_ms)
        windows.append(
            {
                "clip_id": clip_id,
                "hold_id": hold_id,
                "segment_id": None,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_s": round(max(0, end_ms - start_ms) / 1000.0, 3),
                "origin": "hold",
            }
        )

    for row in read_label_rows(labels_path(root)):
        if row["shape"] != "line":
            continue
        start_ms = int(row["start_ms"])
        end_ms = int(row["end_ms"])
        duration_s = round(max(0, end_ms - start_ms) / 1000.0, 3)
        if duration_s < min_s:
            continue
        windows.append(
            {
                "clip_id": row["clip_id"],
                "hold_id": hold_shapes._hold_key(row["hold_id"]),
                "segment_id": int(row["segment_id"]),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_s": duration_s,
                "origin": "segment",
            }
        )

    windows.sort(
        key=lambda window: (
            window["clip_id"],
            hold_shapes._hold_sort_key(window["hold_id"]),
            -1 if window["segment_id"] is None else int(window["segment_id"]),
        )
    )
    return windows


def _read_tasks(path: pathlib.Path) -> list[Any]:
    """The task file as a list; a missing or unreadable file is empty."""
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return loaded if isinstance(loaded, list) else []


def summarise(
    data: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
) -> str:
    """Multi-line summary: queued/reviewed, labels per shape, line windows."""
    root = pathlib.Path(data) if data is not None else data_dir()
    queued = len(_read_tasks(tasks_path(root)))
    rows = read_label_rows(labels_path(root))

    per_shape: dict[str, list[float]] = {}
    for row in rows:
        counts = per_shape.setdefault(str(row["shape"]), [0, 0.0])
        counts[0] += 1
        counts[1] += max(0, int(row["end_ms"]) - int(row["start_ms"])) / 1000.0

    predicted = agreed = 0
    for row in rows:
        if row["predicted"] in hold_shapes.SHAPES:
            predicted += 1
            agreed += int(row["predicted"] == row["shape"])

    windows = line_windows(root, source)
    hold_windows = [window for window in windows if window["origin"] == "hold"]
    segment_windows = [window for window in windows if window["origin"] == "segment"]
    total_s = sum(float(window["duration_s"]) for window in windows)

    lines = [
        f"segment shapes: {queued} segment(s) queued, {len(rows)} reviewed",
        "labels per shape:",
    ]
    if not per_shape:
        lines.append("  (none)")
    for shape in sorted(per_shape):
        count, seconds = per_shape[shape]
        lines.append(f"  {shape}: {count} segment(s), {seconds:.1f}s")
    percent = 100.0 * agreed / predicted if predicted else 0.0
    lines.append(
        f"  prediction agreed with the review on {agreed} of {predicted} "
        f"measured prediction(s) ({percent:.1f} %)"
    )
    lines.append(
        f"line windows: {len(hold_windows)} hold(s) + {len(segment_windows)} "
        f"segment(s) = {total_s:.1f}s"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Push to Label Studio
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class PushReport:
    """What one push to Label Studio did."""

    project_id: int
    created: bool
    tasks: int
    imported: int
    skipped: int


def push(
    *,
    data: str | pathlib.Path | None = None,
    ls_url: str = DEFAULT_LS_URL,
    client: Any = None,
    credentials: tuple[str, str] | None = None,
) -> PushReport:
    """Import the tasks file into ``handstand-segment-shapes``.

    The project is created with the generated config when it does not exist;
    tasks whose ``segment_key`` is already in the project are skipped, so a
    rerun imports only new segments. The password never leaves the login call.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    tasks = [task for task in _read_tasks(tasks_path(root)) if isinstance(task, Mapping)]
    if client is None:
        user, password = credentials if credentials is not None else read_ls_credentials()
        client = LabelStudioClient(ls_url)
        client.login(user, password)

    created = False
    try:
        project_id = client.find_project_id(PROJECT_TITLE)
    except LabelStudioError:
        project_id = client.create_project(PROJECT_TITLE, config_xml())
        created = True

    # The tasks point at /data/local-files/?d=label_frames_segments/…, which
    # answers 404 unless the project has a local-files storage for that dir.
    storage = frames_dir(root)
    if not storage.is_absolute():
        storage = storage.resolve()
    client.ensure_local_storage(project_id, storage, "handstand-segment-shapes frames")

    existing = client.existing_keys(project_id, "segment_key")
    fresh = [task for task in tasks if _task_key(task) not in existing]
    if fresh:
        client.import_tasks(project_id, fresh)
    return PushReport(
        project_id=int(project_id),
        created=created,
        tasks=len(tasks),
        imported=len(fresh),
        skipped=len(tasks) - len(fresh),
    )


def summarise_push(report: PushReport) -> str:
    """Multi-line summary of a push, for the CLI to print."""
    created = "created" if report.created else "found"
    return (
        f"pushed {report.imported} of {report.tasks} task(s) to project "
        f"{report.project_id} ({created} {PROJECT_TITLE}): "
        f"{report.skipped} already present"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.segment_shapes`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.segment_shapes",
        description=(
            "Label the shape of every segment of a mixed hold: pre-label for "
            "Label Studio, import the reviewed export, summarise, push."
        ),
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
            default=DEFAULT_SOURCE,
            help=f"which segment table to read (default: {DEFAULT_SOURCE})",
        )

    pre = commands.add_parser(
        "prelabel",
        help="write a three-frame strip per mixed-hold segment, its manifest, tasks and config",
    )
    common(pre)
    pre.add_argument(
        "--videos",
        type=pathlib.Path,
        default=None,
        help="raw videos to decode (default: $HANDSTAND_VIDEOS)",
    )
    pre.add_argument(
        "--clips",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help="only these clips' segments (default: every mixed hold)",
    )
    pre.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "skip segments that already have an image or a row in "
            "segment_shapes.csv, and merge the manifest/tasks with the files "
            "already on disk"
        ),
    )
    pre.add_argument(
        "--image-base",
        default=None,
        metavar="URL",
        help="serve the strips from here instead of the Label Studio local-files URL",
    )
    pre.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=None,
        help=f"Label Studio import to write (default: <data_dir>/{TASKS_FILENAME})",
    )
    pre.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=f"labeling config to write (default: <repo>/{pathlib.Path(*CONFIG_RELATIVE)})",
    )

    imp = commands.add_parser("import", help="convert a Label Studio export to segment_shapes.csv")
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
        "--manifest",
        type=pathlib.Path,
        default=None,
        help=(
            "pre-label manifest to read missing predictions from "
            f"(default: <data_dir>/{FRAMES_DIRNAME}/{MANIFEST_NAME})"
        ),
    )

    summ = commands.add_parser(
        "summary", help="count reviewed segments, labels per shape and line windows"
    )
    common(summ)

    push_cmd = commands.add_parser(
        "push", help=f"import the tasks file into the {PROJECT_TITLE!r} project"
    )
    push_cmd.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory (default: $HANDSTAND_DATA)",
    )
    push_cmd.add_argument(
        "--ls-url",
        default=DEFAULT_LS_URL,
        help=f"Label Studio base URL (default: {DEFAULT_LS_URL})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.segment_shapes``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "prelabel":
            report = prelabel(
                data=args.data,
                videos=args.videos,
                source=args.source,
                clips=args.clips,
                skip_existing=args.skip_existing,
                image_base=args.image_base,
                out_json=args.out_json,
                config_path=args.config,
            )
            print(summarise_prelabel(report))
            return 0

        if args.command == "import":
            report = import_export(
                args.export, data=args.data, out_csv=args.out, manifest=args.manifest
            )
            print(summarise_import(report))
            return 0

        if args.command == "push":
            report = push(data=args.data, ls_url=args.ls_url)
            print(summarise_push(report))
            return 0

        print(summarise(args.data, args.source))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"segment_shapes: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
