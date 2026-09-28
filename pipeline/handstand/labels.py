"""Turn a Label Studio export into the keypoint CSV the pose bake-off scores against.

Label Studio stores a keypoint as a percentage of the image, so the export is
resolution independent but useless on its own: a point at ``x = 50`` on a
1024-pixel-wide JPEG is not the same pixel as ``x = 50`` on a 576-pixel-wide one.
This module reads ``<data_dir>/label_frames/manifest.csv`` — which carries the
display size of every frame a human was shown — and converts each point into
display-frame pixels with exactly the rule the keypoint parquets use
(``x = (percent / 100) * (width - 1)``, see
:func:`handstand.pose_mediapipe.normalized_to_pixels`). A hand label and a
MediaPipe landmark on the same frame then live in one coordinate system, which is
the entire point of the exercise::

    <data_dir>/labels/keypoints.csv

with the columns clip_id, frame_idx, joint, x, y, visible, labeler, labeled_at.

Importing is **idempotent**: rows are keyed by ``(clip_id, frame_idx, joint)``,
so re-importing the same export (or a corrected one) rewrites those rows in
place instead of appending a second copy. An existing CSV is kept and merged;
only the keys in the export change.

Conventions, all of them the labeler's rather than the tool's:

* A joint the labeler did **not** place produces **no row**. A missing point
  means "not visible"; :data:`VISIBLE` is for a point that was placed and the
  labeler stood behind it.
* A point the tool marks occluded is written with ``visible = false`` and its
  coordinates kept, because "I put it here but it is behind the trainer" is more
  useful to a scorer than silence.
* The per-image ``occluded_or_unsure`` choice (``occluded`` or ``unsure``)
  downgrades every keypoint of that image to ``visible = false``: the labeler is
  telling us the whole frame is not reliable.

Label Studio has spelled the label of a keypoint three ways over its life
(``value.label``, ``value.labels`` and, since 1.23, the plural
``value.keypointlabels`` its docs show), and all three are read here — see
:func:`joint_labels`. Per-keypoint *visibility*, on the other hand, does not
exist in 1.23: there is no attribute for it, which is why the config asks for a
missing point instead. Should a later version add one, the value keys
``occluded``/``visible`` are already honoured by :func:`keypoint_visible`, so
the import needs no change.

CLI::

    cd pipeline
    uv run python -m handstand.labels import ~/Downloads/label-studio-export.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from handstand.frame_sampler import FILE_MODE, MANIFEST_COLUMNS, manifest_path
from handstand.paths import data_dir
from handstand.pose_mediapipe import JOINT_NAMES, normalized_to_pixels

__all__ = [
    "COLUMNS",
    "LABELS_DIRNAME",
    "LABEL_JOINTS",
    "UNCLEAR_CHOICES",
    "UNCLEAR_FIELD",
    "ImportReport",
    "KeypointRow",
    "build_arg_parser",
    "image_index",
    "import_export",
    "load_manifest",
    "main",
    "normalise_joint",
    "percent_to_pixels",
    "read_export",
    "read_rows",
    "summarise",
    "write_rows",
]

#: Name of the output directory inside the data directory.
LABELS_DIRNAME = "labels"
#: The CSV the import writes, relative to :data:`LABELS_DIRNAME`.
CSV_NAME = "keypoints.csv"

#: Full CSV header, in order. Part of the contract.
COLUMNS: tuple[str, ...] = (
    "clip_id",
    "frame_idx",
    "joint",
    "x",
    "y",
    "visible",
    "labeler",
    "labeled_at",
)

#: The joints the Label Studio project asks for, in
#: :data:`handstand.pose_mediapipe.JOINT_NAMES` order. Any other MediaPipe joint
#: is still accepted (a labeler who adds one by hand does not make a mistake), but
#: this is the set the config puts on the screen.
LABEL_JOINTS: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot_index",
    "right_foot_index",
)

#: Name of the per-image "is this frame usable" ``Choices`` field, as the config
#: in ``tools/labeling/label_studio_config.xml`` spells it.
UNCLEAR_FIELD = "occluded_or_unsure"
#: Its choices; ``occluded`` and ``unsure`` downgrade every keypoint of the image
#: to ``visible = false``, and ``clear`` (the third) is the answer that does not.
UNCLEAR_CHOICES: tuple[str, ...] = ("occluded", "unsure")

#: What the CSV's ``visible`` column says; lowercase, as everywhere else here.
VISIBLE = "true"
INVISIBLE = "false"

#: Decimal places for the pixel columns: sub-pixel, but not 8 digits of a JPEG.
FLOAT_DIGITS = 2

#: Hand-written spellings mapped onto MediaPipe's own joint names. The sides are
#: written ``left_``/``right_`` in the schema; a config typed by a person who
#: thinks the other way round is still readable.
_ALIASES: dict[str, str] = {
    "foot_index_left": "left_foot_index",
    "foot_index_right": "right_foot_index",
    "left_foot": "left_foot_index",
    "right_foot": "right_foot_index",
    "left_toe": "left_foot_index",
    "right_toe": "right_foot_index",
    "l_wrist": "left_wrist",
    "r_wrist": "right_wrist",
    "wrist_l": "left_wrist",
    "wrist_r": "right_wrist",
    "head": "nose",
}

#: The keys a keypoint result's value may carry its label under, newest first.
_LABEL_KEYS = ("keypointlabels", "labels", "label")


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class KeypointRow:
    """One labelled joint in display-frame pixels."""

    clip_id: str
    frame_idx: int
    joint: str
    x: float
    y: float
    visible: bool
    labeler: str
    labeled_at: str

    @property
    def key(self) -> tuple[str, int, str]:
        """The row's identity: a joint of a frame of a clip."""
        return (self.clip_id, self.frame_idx, self.joint)

    def as_dict(self) -> dict[str, Any]:
        """The row as the CSV writes it, with the columns in :data:`COLUMNS` order."""
        return {
            "clip_id": self.clip_id,
            "frame_idx": self.frame_idx,
            "joint": self.joint,
            "x": f"{self.x:.{FLOAT_DIGITS}f}",
            "y": f"{self.y:.{FLOAT_DIGITS}f}",
            "visible": VISIBLE if self.visible else INVISIBLE,
            "labeler": self.labeler,
            "labeled_at": self.labeled_at,
        }

    def same_as(self, other: KeypointRow) -> bool:
        """Do two rows write the same line? Compared as written, not as computed.

        The CSV keeps :data:`FLOAT_DIGITS` decimals, so a row read back out of it
        is never quite the float it was written from. Comparing the written form
        is what lets a re-import of an unchanged export report "unchanged"
        instead of claiming it rewrote six rows it wrote out identically.
        """
        return self.as_dict() == other.as_dict()


@dataclasses.dataclass(frozen=True)
class ImportReport:
    """What one import changed."""

    rows_read: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    tasks: int = 0
    images: int = 0
    unknown_images: int = 0
    unknown_joints: tuple[str, ...] = ()
    conflicts: int = 0
    total: int = 0
    out_path: pathlib.Path | None = None

    def summarise(self) -> str:
        """Multi-line summary of the import, for the CLI to print."""
        lines = [
            f"read {self.rows_read} keypoint(s) from {self.tasks} task(s), "
            f"{self.images} image(s) matched by the manifest",
            f"  {self.added} added, {self.updated} updated, {self.unchanged} unchanged, "
            f"{self.total} row(s) in total",
        ]
        if self.unknown_images:
            lines.append(
                f"  {self.unknown_images} image(s) are not in the manifest and were skipped "
                "(a frame that was never sampled cannot be scored)"
            )
        if self.unknown_joints:
            lines.append(f"  ignored joint label(s): {', '.join(self.unknown_joints)}")
        if self.conflicts:
            lines.append(
                f"  {self.conflicts} keypoint(s) were labelled by more than one person; the last "
                "annotation in the export won"
            )
        if self.out_path is not None:
            lines.append(f"  wrote {self.out_path}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def load_manifest(path: str | pathlib.Path) -> list[dict[str, str]]:
    """Read the sampler manifest; a missing file is an empty one.

    Every manifest row is what a human was shown, so the display size in it is
    the one the percentages of the export have to be multiplied by. A manifest
    without those columns is a different sheet and is refused rather than read
    into keypoints at the wrong scale.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not set(MANIFEST_COLUMNS) <= fields:
            missing = [column for column in MANIFEST_COLUMNS if column not in fields]
            raise ValueError(f"{path}: manifest is missing column(s) {', '.join(missing)}")
        return [dict(row) for row in reader]


def _entries(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    """The usable manifest rows, as entries with their display size.

    A row without an image, a clip id or a frame index is not a frame anybody
    was shown, so it is left out of every index rather than half-indexed.
    """
    entries: list[dict[str, Any]] = []
    for row in rows:
        image = (row.get("image") or "").strip()
        frame_idx = _as_int(row.get("frame_idx"))
        clip_id = (row.get("clip_id") or "").strip()
        if not image or frame_idx < 0 or not clip_id:
            continue
        entries.append(
            {
                "image": image,
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "display_width": _as_int(row.get("display_width")),
                "display_height": _as_int(row.get("display_height")),
            }
        )
    return entries


def image_index(rows: Iterable[Mapping[str, str]]) -> dict[str, dict[str, Any]]:
    """Index the manifest by image file name, the way Label Studio names a task.

    Keys are the plain ``<clip_id>_<frame_idx>.jpg`` and the same name with any
    directory stripped, so a task whose image is a bare file name and one stored
    as a path both hit. The display size travels with the entry: it is the only
    thing that turns a percentage into a pixel.
    """
    index: dict[str, dict[str, Any]] = {}
    for entry in _entries(rows):
        index[entry["image"]] = entry
        index[pathlib.PurePath(entry["image"]).name] = entry
    return index


def frame_index(rows: Iterable[Mapping[str, str]]) -> dict[tuple[str, int], dict[str, Any]]:
    """The same manifest rows keyed by ``(clip_id, frame_idx)``.

    This is what finds a task whose image Label Studio renamed: an upload gets a
    hash in front of the file name, so ``f7b8a3c1-<clip>_<frame>.jpg`` no longer
    matches any key of :func:`image_index`, but its tail still ends with a clip
    id the manifest knows.
    """
    return {(entry["clip_id"], entry["frame_idx"]): entry for entry in _entries(rows)}


def _parse_image_name(value: str) -> tuple[str, int] | None:
    """``(<clip_id>, <frame_idx>)`` out of a frame's file name, however it is spelled."""
    name = pathlib.PurePath(value).name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    clip_part, _, frame_part = stem.rpartition("_")
    if not clip_part or not frame_part.isdigit():
        return None
    return clip_part, int(frame_part)


def _lookup_image(
    value: str,
    index: Mapping[str, dict[str, Any]],
    by_frame: Mapping[tuple[str, int], dict[str, Any]],
) -> dict[str, Any] | None:
    """The manifest entry for a task's image, however Label Studio names it.

    Three tries, in order: the value as it stands, its bare file name, and — when
    the upload prefix means neither matches — the clip id the file name *ends*
    with. Anything still unmatched is a frame nobody was shown, and is refused
    rather than converted against a size nobody recorded.
    """
    if not value:
        return None
    entry = index.get(value) or index.get(pathlib.PurePath(value).name)
    if entry is not None:
        return entry
    parsed = _parse_image_name(value)
    if parsed is None:
        return None
    clip_id, frame_idx = parsed
    known = by_frame.get((clip_id, frame_idx))
    if known is not None:
        return known
    for (candidate, candidate_frame), candidate_entry in by_frame.items():
        if candidate_frame == frame_idx and clip_id.endswith(candidate):
            return candidate_entry
    return None


def _as_int(value: object) -> int:
    """Coerce a CSV cell to an int, 0 when it is blank or junk."""
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# The export
# --------------------------------------------------------------------------- #


def read_export(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Read a Label Studio export into a list of tasks.

    Accepts the two shapes an export comes in: a bare list of tasks, and an
    object with a ``tasks`` key (what the API returns).
    """
    path = pathlib.Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError(f"{path}: expected a list of tasks or an object with a 'tasks' list")
    elif isinstance(payload, list):
        tasks = payload
    else:
        raise ValueError(f"{path}: expected a list of tasks, got {type(payload).__name__}")
    return [dict(task) for task in tasks if isinstance(task, Mapping)]


def normalise_joint(label: object) -> str | None:
    """Map a Label Studio label onto a MediaPipe joint name, or ``None``.

    Case, spaces, hyphens and dots are all the same separator, so a hand-edited
    config that says ``"Left Wrist"`` still lands on ``left_wrist``. A label that
    is not a joint at all (``"clear"``, a typo, a free-text note) is refused
    rather than written out as a joint nobody can score.
    """
    text = re.sub(r"[\s\-.]+", "_", str(label or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return None
    if text in JOINT_NAMES:
        return text
    return _ALIASES.get(text)


def percent_to_pixels(
    percent_x: float,
    percent_y: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Convert Label Studio's ``0..100`` percentages into display-frame pixels.

    The same rule the keypoint parquets were written with, so both land in one
    coordinate system: a percentage of the image is a fraction of ``size - 1``,
    because pixel *indices* run 0 to ``size - 1``.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"unknown display size: {width}x{height}")
    points = normalized_to_pixels(
        np.array([[percent_x / 100.0, percent_y / 100.0]], dtype=np.float64), width, height
    )
    return (float(points[0, 0]), float(points[0, 1]))


def _as_float(value: object) -> float:
    """Coerce an export value to a finite float, ``nan`` when it is not one."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _keypoint_visible(value: Mapping[str, Any], *, image_unclear: bool) -> bool:
    """Is a placed keypoint visible?

    The per-keypoint ``occluded`` flag (or an explicit ``visible``/``visibility``,
    from a version of Label Studio that writes one) decides on its own. The
    per-image ``occluded_or_unsure`` choice is the fallback: when the labeler
    says the whole frame is not reliable, nothing in it is.
    """
    if "visible" in value:
        return bool(value["visible"])
    if "visibility" in value:
        return bool(value["visibility"])
    if "occluded" in value:
        return not bool(value["occluded"])
    return not image_unclear


def joint_labels(value: Mapping[str, Any]) -> list[str]:
    """Every joint label a keypoint result carries, whichever key stored it.

    Label Studio has spelled this three ways: ``value.label`` (one string, the
    shape most converters assume), ``value.labels`` and — since 1.23, in the
    shape its docs show — the plural ``value.keypointlabels`` list. All three are
    read, and a value that carries two of them yields one row per label, because
    that is what such a value means.
    """
    labels: list[str] = []
    for key in _LABEL_KEYS:
        raw = value.get(key)
        if raw is None:
            continue
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        labels.extend(str(item) for item in items)
    return labels


def _is_keypoint_result(result: Mapping[str, Any]) -> bool:
    """Is this result a keypoint, whatever the export calls its type?

    Label Studio exports a ``<KeyPointLabels>`` point with type ``keypointlabels``
    (verified against a live 1.23 instance); ``keypoint``/``keypoints`` are accepted
    too. A result with no type at all is still read when it carries coordinates and
    a label, because an export trimmed by hand is exactly the kind of thing that
    arrives here.
    """
    kind = str(result.get("type") or "").lower()
    if kind in ("keypointlabels", "keypoints", "keypoint"):
        return True
    if kind:
        return False
    value = result.get("value")
    return isinstance(value, Mapping) and "x" in value and any(key in value for key in _LABEL_KEYS)


def _choice_values(result: Mapping[str, Any]) -> list[str]:
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


def _image_unclear(results: Sequence[Mapping[str, Any]]) -> bool:
    """Did the labeler mark this image ``occluded`` or ``unsure``?

    Any choice of ``occluded`` or ``unsure`` anywhere in the annotation counts,
    not only the field named :data:`UNCLEAR_FIELD`: those two words mean "do not
    trust this frame" whichever question they answered, and a project created
    before the field was named still has to import.
    """
    for result in results:
        for choice in _choice_values(result):
            if choice.strip().lower() in UNCLEAR_CHOICES:
                return True
    return False


def _results(annotation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The results of one annotation, ignoring anything that is not a mapping."""
    results = annotation.get("result")
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, Mapping)]


def _image_value(task: Mapping[str, Any]) -> str:
    """The image a task shows, as the string Label Studio stored."""
    data = task.get("data")
    if not isinstance(data, Mapping):
        return ""
    for key in ("image", "image_path", "path"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _annotation_meta(annotation: Mapping[str, Any]) -> tuple[str, str]:
    """``(labeler, labeled_at)`` of one annotation, best effort and never empty."""
    labeler = ""
    for key in ("completed_by", "created_by", "annotator"):
        value = annotation.get(key)
        if isinstance(value, str) and value:
            labeler = value
            break
        if isinstance(value, Mapping):
            for name_key in ("email", "username", "first_name"):
                name = value.get(name_key)
                if isinstance(name, str) and name:
                    labeler = name
                    break
            if labeler:
                break
    labeled_at = ""
    for key in ("updated_at", "created_at", "completed_at"):
        value = annotation.get(key)
        if isinstance(value, str) and value:
            labeled_at = value
            break
    return labeler, labeled_at


def _keypoint_results(
    annotation: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    labeler: str,
    labeled_at: str,
    image_unclear: bool,
) -> tuple[list[KeypointRow], list[str]]:
    """The rows of one annotation on one frame, and the joint labels refused."""
    rows: list[KeypointRow] = []
    unknown: list[str] = []
    for result in _results(annotation):
        if not _is_keypoint_result(result):
            continue
        value = result.get("value")
        if not isinstance(value, Mapping):
            continue
        percent_x = _as_float(value.get("x"))
        percent_y = _as_float(value.get("y"))
        if math.isnan(percent_x) or math.isnan(percent_y):
            continue
        try:
            x, y = percent_to_pixels(
                percent_x, percent_y, int(entry["display_width"]), int(entry["display_height"])
            )
        except ValueError:
            # No display size: the frame is not in the manifest, so there is
            # nothing to convert the percentage into. Refused, not guessed.
            unknown += joint_labels(value)
            continue
        for label in joint_labels(value):
            joint = normalise_joint(label)
            if joint is None:
                unknown.append(label)
                continue
            rows.append(
                KeypointRow(
                    clip_id=str(entry["clip_id"]),
                    frame_idx=int(entry["frame_idx"]),
                    joint=joint,
                    x=x,
                    y=y,
                    visible=_keypoint_visible(value, image_unclear=image_unclear),
                    labeler=labeler,
                    labeled_at=labeled_at,
                )
            )
    return rows, unknown


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


def import_export(
    export_path: str | pathlib.Path,
    manifest: str | pathlib.Path | None = None,
    out_csv: str | pathlib.Path | None = None,
    *,
    data: str | pathlib.Path | None = None,
) -> ImportReport:
    """Import a Label Studio export into ``<data_dir>/labels/keypoints.csv``.

    The export is read task by task; every keypoint result of every annotation
    of a task that the manifest knows becomes one row. Rows already in the CSV
    whose key is in the import are replaced, the rest are left untouched, so a
    re-import of the same file changes nothing and a re-import of a corrected one
    changes exactly the corrected keys.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    manifest_file = pathlib.Path(manifest) if manifest is not None else manifest_path(root)
    destination = pathlib.Path(out_csv) if out_csv is not None else root / LABELS_DIRNAME / CSV_NAME

    manifest_rows = load_manifest(manifest_file)
    index = image_index(manifest_rows)
    by_frame = frame_index(manifest_rows)
    tasks = read_export(export_path)

    existing = read_rows(destination)
    merged: dict[tuple[str, int, str], KeypointRow] = {row.key: row for row in existing}

    read = 0
    added = 0
    updated = 0
    unchanged = 0
    matched = 0
    unknown_images = 0
    unknown_joints: list[str] = []
    conflicts = 0
    seen_labelers: dict[tuple[str, int], str] = {}

    for task in tasks:
        image = _image_value(task)
        entry = _lookup_image(image, index, by_frame)
        if entry is None or int(entry["display_width"]) <= 0 or int(entry["display_height"]) <= 0:
            unknown_images += 1
            continue
        matched += 1
        annotations = task.get("annotations")
        for annotation in annotations if isinstance(annotations, list) else []:
            if not isinstance(annotation, Mapping):
                continue
            results = _results(annotation)
            unclear = _image_unclear(results)
            labeler, labeled_at = _annotation_meta(annotation)
            rows, unknown = _keypoint_results(
                annotation, entry, labeler=labeler, labeled_at=labeled_at, image_unclear=unclear
            )
            unknown_joints += unknown
            for row in rows:
                read += 1
                previous = merged.get(row.key)
                if previous is not None:
                    if previous.same_as(row):
                        unchanged += 1
                    else:
                        updated += 1
                    other = seen_labelers.get((row.clip_id, row.frame_idx))
                    if other is not None and other != row.labeler:
                        conflicts += 1
                else:
                    added += 1
                seen_labelers[(row.clip_id, row.frame_idx)] = row.labeler
                merged[row.key] = row

    if unknown_images:
        print(
            f"labels: {unknown_images} task(s) are not in {manifest_file} and were skipped",
            file=sys.stderr,
        )
    write_rows(list(merged.values()), destination)
    return ImportReport(
        rows_read=read,
        added=added,
        updated=updated,
        unchanged=unchanged,
        tasks=len(tasks),
        images=matched,
        unknown_images=unknown_images,
        unknown_joints=tuple(sorted(set(unknown_joints))),
        conflicts=conflicts,
        total=len(merged),
        out_path=destination,
    )


def read_rows(path: str | pathlib.Path) -> list[KeypointRow]:
    """Read a keypoint CSV written earlier; a missing file is simply an empty one.

    Rows that cannot be parsed are dropped rather than kept half-read: the file is
    rebuilt from the export on every import anyway, so one damaged line must not
    stop the rest of the sheet from being used.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    rows: list[KeypointRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(COLUMNS) <= set(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {', '.join(COLUMNS)}")
        for row in reader:
            frame_idx = _as_int(row.get("frame_idx"))
            joint = normalise_joint(row.get("joint"))
            if not row.get("clip_id") or joint is None or frame_idx < 0:
                continue
            x = _as_float(row.get("x"))
            y = _as_float(row.get("y"))
            if math.isnan(x) or math.isnan(y):
                continue
            rows.append(
                KeypointRow(
                    clip_id=str(row["clip_id"]),
                    frame_idx=frame_idx,
                    joint=joint,
                    x=x,
                    y=y,
                    visible=str(row.get("visible", "")).strip().lower() == VISIBLE,
                    labeler=str(row.get("labeler") or ""),
                    labeled_at=str(row.get("labeled_at") or ""),
                )
            )
    return rows


def write_rows(rows: Sequence[KeypointRow], path: str | pathlib.Path) -> pathlib.Path:
    """Write the keypoint CSV atomically: temp file beside it, then rename.

    Sorted by clip, frame and joint, so two imports of the same annotations give
    byte-identical files whatever order the export listed them in.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = pathlib.Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(COLUMNS), lineterminator="\n")
            writer.writeheader()
            for row in sorted(rows, key=lambda item: (item.clip_id, item.frame_idx, item.joint)):
                writer.writerow(row.as_dict())
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, FILE_MODE)
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


def summarise(report: ImportReport) -> str:
    """Multi-line summary of an import, for the CLI to print."""
    return report.summarise()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.labels`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.labels",
        description="Convert a Label Studio export into display-frame keypoint pixels.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser("import", help="import a Label Studio JSON export")
    importer.add_argument("export", type=pathlib.Path, help="the exported JSON file")
    importer.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=None,
        help="sampler manifest giving the display size of every frame "
        "(default: <data_dir>/label_frames/manifest.csv)",
    )
    importer.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=f"CSV to write (default: <data_dir>/{LABELS_DIRNAME}/{CSV_NAME})",
    )
    importer.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory holding label_frames/ and labels/ (default: $HANDSTAND_DATA)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.labels import <export.json>``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = import_export(
            args.export, manifest=args.manifest, out_csv=args.out, data=args.data
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"labels: {error}", file=sys.stderr)
        return 2
    print(summarise(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
