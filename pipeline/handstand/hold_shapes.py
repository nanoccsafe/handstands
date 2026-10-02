"""Label the SHAPE of every hold: pre-label, review in Label Studio, import, summarise.

The dataset mixes handstand types, and the shape matters per **hold**, not per
clip: one clip can hold a line and then a tuck. The reference (#28), the scoring
and tuning (#29/#30) and the line fault classifier (#32–#34) use LINE holds
only, so every other shape has to be identifiable and excluded; the pose
bake-off (#16) keeps every shape and reports per shape.

The loop mirrors the keypoint one (``docs/labeling.md``)::

    handstand.hold_shapes prelabel  -> data/label_frames_holds/<clip>_h<hold>.jpg
                                       + manifest.csv
                                       + data/label_studio_hold_shapes.json
                                       + tools/labeling/hold_shapes_config.xml
    Label Studio (separate)         -> an export JSON
    handstand.hold_shapes import    -> data/labels/hold_shapes.csv
    handstand.hold_shapes summary   -> counts, per-clip skill, prediction accuracy

``prelabel`` guesses each shape from the hold summary's medians
(:func:`predict_shape`) and preselects the guess in Label Studio. The guess is
**only a pre-label**: the labeler's review is the truth, and a hold that was
never reviewed is never a line hold downstream (:func:`is_line_hold`).

Everything it writes from videos lives under ``data/`` and is git-ignored; only
the generated labeling config sits in the repository, under ``tools/labeling/``.

CLI::

    cd pipeline
    uv run python -m handstand.hold_shapes prelabel
    uv run python -m handstand.hold_shapes import ../data/label_studio_hold_shapes_export.json
    uv run python -m handstand.hold_shapes summary --write-catalogue
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import pathlib
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from handstand import catalogue as catalogue_module
from handstand.features import DEFAULT_SOURCE, HOLD_SUMMARY_NAME, output_dir
from handstand.frame_sampler import CATALOGUE_NAME, FILE_MODE, write_jpeg
from handstand.labels import LABELS_DIRNAME, read_export
from handstand.overlay import video_for_clip
from handstand.paths import data_dir
from handstand.pose_mediapipe import DisplayVideo
from handstand.prelabel import image_uri, write_json

__all__ = [
    "CHOICE_NAME",
    "CONFIG_RELATIVE",
    "FRAMES_DIRNAME",
    "HOTKEYS",
    "IMAGE_NAME",
    "LABEL_COLUMNS",
    "LABELS_CSV_NAME",
    "LOCAL_FILES_URL",
    "MANIFEST_COLUMNS",
    "MANIFEST_NAME",
    "MODEL_VERSION",
    "PIKE_HIP_MAX_DEG",
    "SHAPES",
    "STRADDLE_LEG_DEG",
    "TASKS_FILENAME",
    "TUCK_KNEE_DEG",
    "UNMEASURED",
    "ImportReport",
    "PrelabelReport",
    "ShapeSummary",
    "build_arg_parser",
    "build_summary",
    "build_task",
    "config_xml",
    "default_config_path",
    "frames_dir",
    "hold_image_name",
    "import_export",
    "is_line_hold",
    "labels_path",
    "line_holds",
    "load_hold_shapes",
    "main",
    "manifest_path",
    "prelabel",
    "predict_shape",
    "read_hold_summary",
    "read_label_rows",
    "summarise_import",
    "summarise_prelabel",
    "summarise_shapes",
    "task_image_uri",
    "tasks_path",
    "write_catalogue_skill",
]

# --------------------------------------------------------------------------- #
# Shapes and the pre-label thresholds
# --------------------------------------------------------------------------- #

#: The shapes a hold can be labelled with, in the order the config lists them.
#: This tuple is the single source of truth: the labeling config's choices, the
#: import's accepted values and the summary's columns are all built from it.
SHAPES: tuple[str, ...] = ("line", "straddle", "split_stag", "tuck", "pike", "other")

#: What a hold whose shape medians were never measured is called — **pre-label
#: only**, never a final label: it never appears in ``hold_shapes.csv``, and a
#: hold predicted as this gets no preselection in Label Studio.
UNMEASURED = "unmeasured"

#: The knee angle under which a pre-labelled hold is a tuck, in degrees.
#: Pre-labelling thresholds ONLY: they preselect a choice the labeler confirms
#: or changes, and the reviewed ``data/labels/hold_shapes.csv`` is the truth.
TUCK_KNEE_DEG = 120.0

#: The leg separation over which a pre-labelled hold has the legs apart, in
#: degrees. Pre-labelling ONLY — see :data:`TUCK_KNEE_DEG`. The labeler may
#: change the preselected ``straddle`` to ``split_stag``, which no number here
#: can decide: it is which way the legs point, not how far apart they are.
STRADDLE_LEG_DEG = 45.0

#: The hip angle under which a pre-labelled hold is a pike, in degrees.
#: Pre-labelling ONLY — see :data:`TUCK_KNEE_DEG`.
PIKE_HIP_MAX_DEG = 140.0

#: Keyboard shortcuts for the six choices, in :data:`SHAPES` order.
HOTKEYS: dict[str, str] = {shape: str(index + 1) for index, shape in enumerate(SHAPES)}

#: ``from_name``/``toName`` of the config's ``<Choices>`` tag; the import
#: matches a result by this name as well as by its ``choices`` type.
CHOICE_NAME = "shape"
#: ``name`` of the config's ``<Image>`` tag, the ``toName`` of the choices.
IMAGE_NAME = "image"

#: Recorded in every prediction so a later run can tell which rules wrote it.
MODEL_VERSION = "hold-shapes-prelabel"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Name of the output directory inside the data directory: one JPEG per hold.
FRAMES_DIRNAME = "label_frames_holds"
#: Name of the manifest inside :data:`FRAMES_DIRNAME`.
MANIFEST_NAME = "manifest.csv"
#: The Label Studio import, under ``<data_dir>/``.
TASKS_FILENAME = "label_studio_hold_shapes.json"
#: The imported labels, under ``<data_dir>/labels/``.
LABELS_CSV_NAME = "hold_shapes.csv"
#: Where the generated labeling config lives, relative to the repository root.
CONFIG_RELATIVE = ("tools", "labeling", "hold_shapes_config.xml")

#: Label Studio local-files URL prefix: ``/data/local-files/?d=<path relative to
#: the data dir>`` — the default form of every task's ``image``. A browser
#: cannot open a ``file://`` URI, but Label Studio serves these URLs from its
#: local storage when it runs with
#: ``LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`` set to the data directory and the
#: project carries ``data/label_frames_holds`` as a local storage.
LOCAL_FILES_URL = "/data/local-files/?d="

#: Full manifest header, in order. Part of the contract: the CLI prints it and
#: :func:`import_export` reads the predictions back from it.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "image",
    "clip_id",
    "hold_id",
    "frame_idx",
    "t_ms",
    "hold_duration_s",
    "predicted_shape",
    "reason",
)

#: Full header of ``hold_shapes.csv``, in order. Part of the contract: the
#: helpers read it back and ``summary`` counts from it.
LABEL_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "hold_id",
    "shape",
    "predicted_shape",
    "agreed",
    "labeled_at",
)

#: What the CSV's ``agreed`` column says; lowercase, as everywhere else here.
TRUE = "true"
FALSE = "false"


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


def _hold_key(value: object) -> int | str:
    """A hold id as a dict key: an ``int`` when it reads as one, else its text.

    The hold summary writes integers, but a CSV re-read by hand can hold
    anything, and two spellings of the same hold must not become two keys.
    """
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


def hold_image_name(clip_id: str, hold_id: object) -> str:
    """File name of one hold's frame: ``<clip_id>_h<hold_id>.jpg``.

    The ``h`` keeps it apart from the sampler's ``<clip_id>_<frame_idx>.jpg``:
    the two directories are different Label Studio projects, and a converter
    that parses names must not confuse a hold number with a frame number.
    """
    return f"{clip_id}_h{_hold_key(hold_id)}.jpg"


def frames_dir(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_frames_holds``, where the JPEGs and the manifest go."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / FRAMES_DIRNAME


def manifest_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_frames_holds/manifest.csv``."""
    return frames_dir(data) / MANIFEST_NAME


def tasks_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_studio_hold_shapes.json``, the Label Studio import."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / TASKS_FILENAME


def labels_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/labels/hold_shapes.csv``, the reviewed labels."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / LABELS_DIRNAME / LABELS_CSV_NAME


def default_config_path() -> pathlib.Path:
    """``tools/labeling/hold_shapes_config.xml`` in this checkout.

    Resolved from this file (``pipeline/handstand/`` is two levels below the
    repository root), so a run writes the config the docs point at. Tests pass
    ``--config`` to keep a generated file out of the tree.
    """
    return pathlib.Path(__file__).resolve().parents[2].joinpath(*CONFIG_RELATIVE)


def _write_csv(
    path: pathlib.Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> pathlib.Path:
    """Write a CSV atomically: temp file beside it, then rename, mode 0644.

    A reader either sees the previous file or the new one — never a half-written
    work list (the manifest the labeler works from) or a truncated label sheet.
    """
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


# --------------------------------------------------------------------------- #
# Predicting a shape
# --------------------------------------------------------------------------- #


def predict_shape(row: Mapping[str, object]) -> tuple[str, str]:
    """Guess one hold's shape from its hold-summary medians: ``(shape, reason)``.

    Reads ``leg_separation_median``, ``knee_angle_median`` and ``hip_angle_median``
    (blank or unparseable cells count as not measured) and returns a shape from
    :data:`SHAPES` or :data:`UNMEASURED` plus a short reason such as
    ``"legs 62° apart"`` for the Label Studio header.

    The thresholds are **pre-labelling only** — see :data:`TUCK_KNEE_DEG` — so
    the guess arrives preselected and the labeler's review stays the truth. A
    hold missing any of the three medians is ``unmeasured``: it still gets an
    image and a task, just no preselection.

    The order is the one a viewer sees: bent knees are a tuck whatever the
    legs do, legs apart are a straddle (the labeler changes it to ``split_stag``
    when one leg points forward), and only a hold that is straight everywhere
    comes back a line.
    """
    leg = _as_float(row.get("leg_separation_median"))
    knee = _as_float(row.get("knee_angle_median"))
    hip = _as_float(row.get("hip_angle_median"))
    missing = [
        name
        for name, value in (
            ("leg_separation", leg),
            ("knee_angle", knee),
            ("hip_angle", hip),
        )
        if math.isnan(value)
    ]
    if missing:
        return UNMEASURED, "not measured: " + ", ".join(missing)
    if knee < TUCK_KNEE_DEG:
        return "tuck", f"knees {knee:.0f}°"
    if leg > STRADDLE_LEG_DEG:
        return "straddle", f"legs {leg:.0f}° apart"
    if hip < PIKE_HIP_MAX_DEG:
        return "pike", f"hips {hip:.0f}°"
    return "line", f"legs {leg:.0f}°, knees {knee:.0f}°, hips {hip:.0f}°"


# --------------------------------------------------------------------------- #
# The Label Studio config and tasks
# --------------------------------------------------------------------------- #


def config_xml() -> str:
    """The ``tools/labeling/hold_shapes_config.xml`` contents.

    One image, one required single choice built from :data:`SHAPES`, and a
    header line that shows which hold this is, how long it was and why the
    pre-label says what it says — everything needed to decide without leaving
    the task. The choices carry hotkeys 1–6 in :data:`SHAPES` order.

    The header reads the single ``$info`` task field (:func:`build_task` joins
    the four parts). Label Studio 1.23 parses a ``value`` as ONE variable whose
    name runs from the ``$`` to the end of the attribute, so a value of
    ``"$clip_id / hold $hold_id / ..."`` asks for a task key literally named
    ``clip_id / hold $hold_id / ...`` and every import fails with
    ``key is expected in task data``. Every ``$`` here must stay a lone
    ``$identifier`` — ``tests/test_hold_shapes.py`` asserts it.
    """
    choices = "\n".join(
        f'    <Choice value="{shape}" hotkey="{HOTKEYS[shape]}"/>' for shape in SHAPES
    )
    tag = f'  <Choices name="{CHOICE_NAME}" toName="{IMAGE_NAME}" choice="single" '
    return f"""<View>
  <!--
    Generated by pipeline/handstand/hold_shapes.py (prelabel); edit the module,
    not this file. The Choice values are the module's SHAPES constants and the
    import matches them by name: a renamed choice here is an unlabelled hold
    there. Hotkeys 1-6 select line, straddle, split_stag, tuck, pike, other.

    The preselected choice is the pre-label from the hold's features, not a
    decision: check the image and change it when it is wrong. A task with no
    preselection is a hold the features could not measure.
  -->
  <Text name="hold_info" value="$info" density="1"/>
  <Image name="{IMAGE_NAME}" value="$image" zoom="true" zoomControl="true"/>
{tag}required="true" showInline="true">
{choices}
  </Choices>
</View>
"""


def task_image_uri(
    image: str | pathlib.Path,
    image_base: str | None = None,
    data: str | pathlib.Path | None = None,
) -> str:
    """The ``data.image`` URL for one hold's frame.

    The default is Label Studio's local-files form,
    ``/data/local-files/?d=<path relative to the data dir>`` — for this module
    ``/data/local-files/?d=label_frames_holds/<clip_id>_h<hold_id>.jpg``. A
    browser refuses to open a ``file://`` URI, but Label Studio serves these
    URLs from its local storage when it is started with
    ``LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`` set to the data directory and
    the project carries ``data/label_frames_holds`` as a local storage (the
    setup ``docs/labeling.md`` documents for project 3, ``handstand-hold-shapes``).

    ``image_base`` overrides it, prefixing the file name exactly as
    :func:`handstand.prelabel.image_uri` does: an ``http(s)://`` prefix or a
    directory serving the JPEGs by name.
    """
    path = pathlib.Path(image)
    if image_base:
        return image_uri(path, image_base)
    root = pathlib.Path(data) if data is not None else data_dir()
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        # Outside the data dir (a direct call with some other path): the local
        # storage holds this module's frames, so name the file as it names them.
        relative = pathlib.Path(FRAMES_DIRNAME) / path.name
    return f"{LOCAL_FILES_URL}{relative.as_posix()}"


def build_task(
    image: str | pathlib.Path,
    *,
    clip_id: str,
    hold_id: int | str,
    duration_s: float,
    reason: str,
    shape: str,
    image_base: str | None = None,
    data: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """One Label Studio task for one hold, carrying the pre-selected shape.

    ``image`` is a local path; it becomes the ``/data/local-files/?d=`` URL
    :func:`task_image_uri` builds from its place under the data dir, so the
    project's local storage serves it (``image_base`` serves it from somewhere
    else instead).

    ``info`` is the header line joined into ONE task field — ``clip / hold /
    duration / reason`` — because that is the single ``$info`` the config's
    ``<Text>`` reads: Label Studio 1.23 takes a multi-``$`` value as one
    variable name (see :func:`config_xml`). A :data:`UNMEASURED` hold gets
    **no** ``predictions`` entry — there is nothing to preselect, and
    inventing a default would label it by accident.
    """
    task: dict[str, Any] = {
        "data": {
            "image": task_image_uri(image, image_base, data),
            "clip_id": clip_id,
            "hold_id": hold_id,
            "duration_s": duration_s,
            "reason": reason,
            "info": f"{clip_id} / hold {hold_id} / {duration_s:.1f} s / {reason}",
        }
    }
    if shape != UNMEASURED:
        task["predictions"] = [
            {
                "model_version": MODEL_VERSION,
                "score": 1.0,
                "result": [
                    {
                        "type": "choices",
                        "from_name": CHOICE_NAME,
                        "to_name": IMAGE_NAME,
                        "value": {"choices": [shape]},
                    }
                ],
            }
        ]
    return task


# --------------------------------------------------------------------------- #
# Reading the hold summary
# --------------------------------------------------------------------------- #


def read_hold_summary(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Read ``hold_summary.csv`` into rows with a typed id and duration.

    A missing file raises with the command that produces it: the pre-label has
    nothing to predict from and guessing would be worse than stopping. Only the
    columns this module needs are normalised; every other cell is kept as read,
    so :func:`predict_shape` can be handed the row whole.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; generate it with: "
            "cd pipeline && uv run python -m handstand.features --all"
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            clip_id = str(raw.get("clip_id") or "").strip()
            if not clip_id:
                continue
            row: dict[str, Any] = dict(raw)
            row["clip_id"] = clip_id
            row["hold_id"] = _hold_key(raw.get("hold_id"))
            if math.isnan(_as_float(row.get("hold_duration_s"))):
                start = _as_float(row.get("hold_start_ms"))
                end = _as_float(row.get("hold_end_ms"))
                if math.isnan(start):
                    start = 0.0
                if math.isnan(end):
                    end = start
                row["hold_duration_s"] = round(max(0.0, end - start) / 1000.0, 3)
            rows.append(row)
    return rows


def _duration(row: Mapping[str, Any]) -> float:
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


def _group_by_clip(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Hold rows grouped by clip, in first-seen order."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["clip_id"]), []).append(row)
    return list(groups.items())


# --------------------------------------------------------------------------- #
# Pre-label
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class PrelabelReport:
    """What one pre-label run produced."""

    #: The manifest rows: one per written hold.
    rows: list[dict[str, Any]]
    #: The Label Studio tasks: one per written hold.
    tasks: list[dict[str, Any]]
    #: Holds processed (written + skipped with their clips).
    holds: int
    #: Written holds per predicted shape, every shape of :data:`SHAPES` present.
    counts: dict[str, int]
    #: ``(clip_id, reason)`` for every clip whose video could not be read.
    skipped: list[tuple[str, str]]
    frames_dir: pathlib.Path
    manifest: pathlib.Path
    json_path: pathlib.Path
    config_path: pathlib.Path
    runtime_seconds: float = 0.0

    @property
    def written(self) -> int:
        """Holds with an image, a manifest row and a task."""
        return len(self.rows)

    @property
    def clips(self) -> int:
        """Distinct clips that contributed at least one image."""
        return len({str(row["clip_id"]) for row in self.rows})

    @property
    def skipped_holds(self) -> int:
        """Holds lost with their clips' videos."""
        return self.holds - self.written


def prelabel(
    *,
    data: str | pathlib.Path | None = None,
    videos: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
    limit: int | None = None,
    image_base: str | None = None,
    out_json: str | pathlib.Path | None = None,
    config_path: str | pathlib.Path | None = None,
    open_video: Callable[[Any], Any] | None = None,
) -> PrelabelReport:
    """Write one middle-of-hold frame per hold, its manifest, tasks and config.

    For every row of ``<data>/features/<source>/hold_summary.csv`` the frame at
    the hold's middle time ``((hold_start_ms + hold_end_ms) / 2)`` is decoded in
    display orientation — the same decoder the sampler uses — and kept when it
    is the nearest frame to that time. Each hold becomes a JPEG named
    ``<clip_id>_h<hold_id>.jpg``, a manifest row and a Label Studio task whose
    prediction preselects :func:`predict_shape`'s answer.

    A clip whose video is missing or unreadable is skipped with a note in
    :attr:`PrelabelReport.skipped` rather than ending the run: one lost file
    must not cost the other 160 clips their labels. ``open_video`` defaults to
    :class:`handstand.pose_mediapipe.DisplayVideo`, resolved at call time so a
    test can replace the decoder.

    ``config_path`` defaults to ``tools/labeling/hold_shapes_config.xml`` in
    this checkout (see :func:`default_config_path`).
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    summary_path = output_dir(root, source) / HOLD_SUMMARY_NAME
    holds = read_hold_summary(summary_path)
    if limit is not None:
        holds = holds[: max(0, limit)]

    destination = frames_dir(root)
    if not destination.is_absolute():
        destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_out = pathlib.Path(out_json) if out_json is not None else tasks_path(root)
    config_out = pathlib.Path(config_path) if config_path is not None else default_config_path()
    read = open_video or DisplayVideo

    rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()

    def finish(clip_id: str, item: Mapping[str, Any]) -> None:
        """Write one hold's image, manifest row and task."""
        hold = item["hold"]
        hold_id = hold["hold_id"]
        shape, reason = predict_shape(hold)
        name = hold_image_name(clip_id, hold_id)
        write_jpeg(item["frame"], destination / name)
        duration = _duration(hold)
        counts[shape] += 1
        rows.append(
            {
                "image": name,
                "clip_id": clip_id,
                "hold_id": hold_id,
                "frame_idx": int(item["frame_idx"]),
                "t_ms": int(item["t_ms"]),
                "hold_duration_s": duration,
                "predicted_shape": shape,
                "reason": reason,
            }
        )
        tasks.append(
            build_task(
                destination / name,
                clip_id=clip_id,
                hold_id=hold_id,
                duration_s=duration,
                reason=reason,
                shape=shape,
                image_base=image_base,
                data=root,
            )
        )

    started = time.perf_counter()
    for clip_id, clip_holds in _group_by_clip(holds):
        try:
            video_path = video_for_clip(clip_id, videos, root)
        except (FileNotFoundError, ValueError) as error:
            skipped.append((clip_id, str(error)))
            continue

        pending: dict[int | str, dict[str, Any]] = {}
        for hold in clip_holds:
            start = _as_float(hold.get("hold_start_ms"))
            end = _as_float(hold.get("hold_end_ms"))
            if math.isnan(start):
                start = 0.0
            if math.isnan(end):
                end = start
            pending[hold["hold_id"]] = {
                "hold": hold,
                "mid": (start + end) / 2.0,
                "dist": math.inf,
                "frame": None,
                "frame_idx": -1,
                "t_ms": -1,
            }

        try:
            with read(video_path) as video:
                for packet in video:
                    for key in list(pending):
                        item = pending[key]
                        distance = abs(packet.t_ms - item["mid"])
                        if distance < item["dist"]:
                            item["dist"] = distance
                            item["frame"] = packet.frame
                            item["frame_idx"] = packet.frame_idx
                            item["t_ms"] = packet.t_ms
                        # Past the middle the distance only grows, so the
                        # nearest frame to this hold is already known.
                        if packet.t_ms > item["mid"]:
                            finish(clip_id, item)
                            del pending[key]
                    if not pending:
                        break
        except (RuntimeError, OSError) as error:
            skipped.append((clip_id, str(error)))
            continue

        if pending:
            empty = [item for item in pending.values() if item["frame"] is None]
            if empty:
                skipped.append((clip_id, "the video decoded no frames"))
            for item in pending.values():
                if item["frame"] is not None:
                    # The video ended before this hold's middle: the nearest
                    # frame that exists is still the nearest frame.
                    finish(clip_id, item)

    runtime = time.perf_counter() - started

    manifest_out = destination / MANIFEST_NAME
    _write_csv(manifest_out, MANIFEST_COLUMNS, rows)
    write_json(tasks, json_out)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(config_xml(), encoding="utf-8")

    return PrelabelReport(
        rows=rows,
        tasks=tasks,
        holds=len(holds),
        counts={shape: counts.get(shape, 0) for shape in (*SHAPES, UNMEASURED)},
        skipped=skipped,
        frames_dir=destination,
        manifest=manifest_out,
        json_path=json_out,
        config_path=config_out,
        runtime_seconds=runtime,
    )


def summarise_prelabel(report: PrelabelReport) -> str:
    """Multi-line summary of a pre-label run, for the CLI to print."""
    per_shape = " ".join(
        f"{shape}={report.counts.get(shape, 0)}" for shape in (*SHAPES, UNMEASURED)
    )
    lines = [
        f"pre-labelled {report.written} of {report.holds} hold(s) in "
        f"{report.runtime_seconds:.0f}s -> {report.manifest}",
        f"  written per predicted shape: {per_shape}",
        f"  images: {report.frames_dir} ({report.clips} clip(s))",
        f"  tasks: {report.json_path}",
        f"  config: {report.config_path}",
    ]
    if report.skipped:
        holds = report.skipped_holds
        lines.append(f"  {holds} hold(s) skipped with {len(report.skipped)} clip(s):")
        lines += [f"    {clip}: {reason}" for clip, reason in report.skipped]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Import
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


def _annotation_shape(annotation: Mapping[str, Any]) -> str | None:
    """The shape one annotation chose, or ``None`` when it chose nothing."""
    results = annotation.get("result")
    if not isinstance(results, list):
        return None
    shape: str | None = None
    for result in results:
        if not isinstance(result, Mapping):
            continue
        kind = str(result.get("type") or "").lower()
        from_name = str(result.get("from_name") or "")
        if kind != "choices" and from_name != CHOICE_NAME:
            continue
        choices = _choices(result)
        if choices:
            shape = choices[0]
    return shape


def _labeled_at(annotation: Mapping[str, Any]) -> str:
    """When one annotation was made, best effort and never empty."""
    for key in ("updated_at", "created_at", "completed_at"):
        value = annotation.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _task_shape(task: Mapping[str, Any]) -> tuple[str, str]:
    """``(shape, labeled_at)`` of a task: the **last** annotation that chose one.

    The last annotation wins because that is the decision the labeler ended on;
    an annotation with no choice (submitted to keep a note) is not a decision
    and does not erase the one before it.
    """
    annotations = task.get("annotations")
    shape = ""
    labeled_at = ""
    for annotation in annotations if isinstance(annotations, list) else []:
        if not isinstance(annotation, Mapping):
            continue
        chosen = _annotation_shape(annotation)
        if chosen:
            shape = chosen
            labeled_at = _labeled_at(annotation)
    return shape, labeled_at


def _prediction_shape(task: Mapping[str, Any]) -> str:
    """The pre-selected shape the import gave this task, or ``""``."""
    predictions = task.get("predictions")
    for prediction in predictions if isinstance(predictions, list) else []:
        if not isinstance(prediction, Mapping):
            continue
        results = prediction.get("result")
        for result in results if isinstance(results, list) else []:
            if not isinstance(result, Mapping):
                continue
            kind = str(result.get("type") or "").lower()
            if kind != "choices" and str(result.get("from_name") or "") != CHOICE_NAME:
                continue
            choices = _choices(result)
            if choices:
                return choices[0]
    return ""


def read_label_rows(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Read a ``hold_shapes.csv`` written earlier; a missing file is an empty one.

    A file without the full header raises: it is this module's own contract, and
    a silently empty read would make ``summary`` report "0 labels" over a
    mis-typed sheet.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(LABEL_COLUMNS) <= set(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {', '.join(LABEL_COLUMNS)}")
        for row in reader:
            clip_id = str(row.get("clip_id") or "").strip()
            shape = str(row.get("shape") or "").strip()
            if not clip_id or not shape:
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "hold_id": _hold_key(row.get("hold_id")),
                    "shape": shape,
                    "predicted_shape": str(row.get("predicted_shape") or "").strip(),
                    "agreed": str(row.get("agreed") or "").strip().lower(),
                    "labeled_at": str(row.get("labeled_at") or ""),
                }
            )
    return rows


def _manifest_predictions(path: pathlib.Path) -> dict[tuple[str, int | str], str]:
    """``(clip_id, hold_id) -> predicted_shape`` from a pre-label manifest."""
    if not path.is_file():
        return {}
    found: dict[tuple[str, int | str], str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = str(row.get("clip_id") or "").strip()
            if not clip_id:
                continue
            found[(clip_id, _hold_key(row.get("hold_id")))] = str(
                row.get("predicted_shape") or ""
            ).strip()
    return found


@dataclasses.dataclass(frozen=True)
class ImportReport:
    """What one import of a Label Studio export produced."""

    #: Tasks in the export.
    tasks: int
    #: Tasks that carried an annotation with a shape.
    annotated: int
    #: Tasks skipped: no annotations, or no clip/hold in their data.
    skipped: int
    added: int
    updated: int
    unchanged: int
    #: Shapes the export used that are not in :data:`SHAPES`.
    unknown_shapes: tuple[str, ...]
    rows: int
    out_path: pathlib.Path


def import_export(
    export_path: str | pathlib.Path,
    *,
    data: str | pathlib.Path | None = None,
    out_csv: str | pathlib.Path | None = None,
    manifest: str | pathlib.Path | None = None,
) -> ImportReport:
    """Import a Label Studio export into ``<data_dir>/labels/hold_shapes.csv``.

    Each annotated task becomes one row: its shape, the prediction it started
    from (from the export's ``predictions``, else from the pre-label manifest),
    whether the two agree and when the annotation was made. The **last**
    annotation of a task wins, tasks without one are skipped, and rows already
    in the CSV whose key is not in this export are kept, so re-importing is
    idempotent and importing a corrected export changes exactly the holds it
    mentions.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    destination = pathlib.Path(out_csv) if out_csv is not None else labels_path(root)
    manifest_file = pathlib.Path(manifest) if manifest is not None else manifest_path(root)

    tasks = read_export(export_path)
    predicted = _manifest_predictions(manifest_file)
    merged: dict[tuple[str, int | str], dict[str, Any]] = {
        (row["clip_id"], _hold_key(row["hold_id"])): dict(row)
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
        if not clip_id or raw_hold is None or str(raw_hold).strip() == "":
            skipped += 1
            continue
        hold_id = _hold_key(raw_hold)
        shape, labeled_at = _task_shape(task)
        if not shape:
            skipped += 1
            continue
        annotated += 1
        if shape not in SHAPES:
            unknown.add(shape)
        key = (clip_id, hold_id)
        prediction = _prediction_shape(task) or predicted.get(key, "")
        row = {
            "clip_id": clip_id,
            "hold_id": hold_id,
            "shape": shape,
            "predicted_shape": prediction,
            "agreed": TRUE if shape == prediction else FALSE,
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
        merged.values(), key=lambda row: (row["clip_id"], _hold_sort_key(row["hold_id"]))
    )
    _write_csv(destination, LABEL_COLUMNS, ordered)
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
        "(no annotation, or no clip/hold in the task data)",
    ]
    if report.unknown_shapes:
        lines.append("  not in SHAPES, imported as typed: " + ", ".join(report.unknown_shapes))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Summary and the library helpers
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ShapeSummary:
    """What ``summary`` counted: shapes, per-clip skill, prediction accuracy."""

    #: Labelled holds in total.
    total: int
    #: Holds per shape; every shape of :data:`SHAPES` is present, even at 0.
    counts: dict[str, int]
    #: ``clip_id ->`` the single shape all its holds share, or ``"mixed"``.
    clip_skills: dict[str, str]
    #: Rows whose prediction was one of :data:`SHAPES` (a real preselection).
    predicted: int
    #: Of those, the ones the review agreed with.
    agreed: int
    #: Rows predicted :data:`UNMEASURED` — a guess that was never made.
    unmeasured: int
    #: Rows with no recorded prediction at all.
    unrecorded: int

    @property
    def agreement_percent(self) -> float:
        """Share of the measured predictions the review confirmed."""
        return 100.0 * self.agreed / self.predicted if self.predicted else 0.0


def build_summary(rows: Sequence[Mapping[str, object]]) -> ShapeSummary:
    """Count labelled holds per shape, per clip and against their predictions.

    A clip's skill is the single shape all its labelled holds share and
    ``"mixed"`` otherwise — which is exactly why the shape is labelled per hold:
    a clip is only as uniform as its holds are.
    """
    counts: Counter[str] = Counter(str(row.get("shape") or "") for row in rows)
    ordered = {shape: counts.get(shape, 0) for shape in SHAPES}
    ordered.update({shape: n for shape, n in sorted(counts.items()) if shape not in SHAPES})

    per_clip: dict[str, set[str]] = {}
    for row in rows:
        clip_id = str(row.get("clip_id") or "").strip()
        shape = str(row.get("shape") or "").strip()
        if clip_id and shape:
            per_clip.setdefault(clip_id, set()).add(shape)
    clip_skills = {
        clip_id: next(iter(shapes)) if len(shapes) == 1 else "mixed"
        for clip_id, shapes in per_clip.items()
    }

    predicted = agreed = unmeasured = unrecorded = 0
    for row in rows:
        prediction = str(row.get("predicted_shape") or "").strip()
        shape = str(row.get("shape") or "").strip()
        if prediction in SHAPES:
            predicted += 1
            agreed += int(shape == prediction)
        elif prediction == UNMEASURED:
            unmeasured += 1
        else:
            unrecorded += 1

    return ShapeSummary(
        total=len(rows),
        counts=ordered,
        clip_skills=clip_skills,
        predicted=predicted,
        agreed=agreed,
        unmeasured=unmeasured,
        unrecorded=unrecorded,
    )


def summarise_shapes(summary: ShapeSummary) -> str:
    """Multi-line summary of the labels, for the CLI to print."""
    holds = " ".join(f"{shape}={count}" for shape, count in summary.counts.items())
    skills: Counter[str] = Counter(summary.clip_skills.values())
    clip_lines = []
    for skill in (*SHAPES, "mixed"):
        if skills.get(skill):
            clip_lines.append(f"{skill}={skills.pop(skill)}")
    clip_lines += [f"{skill}={count}" for skill, count in sorted(skills.items())]
    lines = [
        f"hold shapes: {summary.total} labelled hold(s) over {len(summary.clip_skills)} clip(s)",
        f"  holds: {holds}",
        f"  clips: {' '.join(clip_lines) if clip_lines else 'none'}",
        f"  prediction agreed with the review on {summary.agreed} of "
        f"{summary.predicted} measured prediction(s) "
        f"({summary.agreement_percent:.1f} %)",
        f"  {summary.unmeasured} hold(s) were pre-labelled unmeasured "
        f"(no prediction to agree with); {summary.unrecorded} had none recorded",
    ]
    mixed = sorted(clip for clip, skill in summary.clip_skills.items() if skill == "mixed")
    if mixed:
        lines.append(f"  clips whose holds disagree ({len(mixed)}): {', '.join(mixed)}")
    return "\n".join(lines)


def load_hold_shapes(data: str | pathlib.Path | None = None) -> dict[tuple[str, int | str], str]:
    """Every reviewed hold shape: ``(clip_id, hold_id) -> shape``.

    Empty when the labels do not exist yet — an unreviewed dataset has no line
    holds, which is what :func:`is_line_hold` turns into ``False``.
    """
    return {
        (row["clip_id"], _hold_key(row["hold_id"])): row["shape"]
        for row in read_label_rows(labels_path(data))
    }


def is_line_hold(
    clip_id: str,
    hold_id: object,
    shapes: Mapping[tuple[str, int | str], str],
) -> bool:
    """Is this hold a **confirmed** line hold?

    ``False`` for an unlabelled hold, so an unreviewed hold can never sneak
    into line-only data (#28's reference, #29/#30's scoring and tuning, and the
    #32–#34 fault classifier all read only this answer).
    """
    return shapes.get((str(clip_id).strip(), _hold_key(hold_id))) == "line"


def line_holds(
    hold_summary_rows: Iterable[Mapping[str, object]],
    shapes: Mapping[tuple[str, int | str], str],
) -> list[Mapping[str, object]]:
    """The hold-summary rows whose hold was reviewed as a line hold, in order.

    Unlabelled holds are dropped along with every other shape: this is the one
    filter downstream line-only stages need.
    """
    return [
        row
        for row in hold_summary_rows
        if is_line_hold(str(row.get("clip_id") or ""), row.get("hold_id"), shapes)
    ]


def write_catalogue_skill(
    clip_skills: Mapping[str, str],
    catalogue_path: str | pathlib.Path,
) -> tuple[pathlib.Path, int]:
    """Fill the catalogue's ``skill`` column from per-clip shapes, atomically.

    Only clips that have reviewed holds are touched, and only the ``skill``
    column: every other column keeps exactly the value it had, and the file is
    written by :func:`handstand.catalogue.write_rows` (temp file plus rename),
    so an interrupted run cannot damage the annotations typed by hand.
    ``"mixed"`` is written for a clip whose holds do not share one shape —
    deliberately not a guess at which shape the clip mostly is.
    """
    path = pathlib.Path(catalogue_path)
    if not path.is_file():
        raise FileNotFoundError(f"no catalogue to fill: {path}")
    rows = catalogue_module.read_rows(path)
    if not rows or not clip_skills:
        return path, 0
    changed = 0
    for row in rows:
        skill = clip_skills.get(row.get("clip_id") or "")
        if not skill:
            continue
        if row.get("skill") != skill:
            changed += 1
        row["skill"] = skill
    catalogue_module.write_rows(rows, path)
    return path, changed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.hold_shapes`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.hold_shapes",
        description=(
            "Label the shape of every hold: pre-label for Label Studio, import "
            "the reviewed export, summarise."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pre = commands.add_parser(
        "prelabel",
        help="write one middle frame per hold, its manifest, the LS import and the config",
    )
    pre.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"which hold summary to read (default: {DEFAULT_SOURCE})",
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
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="pre-label at most N holds (default: every hold in the summary)",
    )
    pre.add_argument(
        "--image-base",
        default=None,
        metavar="URL",
        help=(
            "serve the frames from here instead of the Label Studio local-files "
            "URL, e.g. an http(s):// prefix or a directory (default: "
            "/data/local-files/?d=label_frames_holds/<name>, which needs the "
            "local storage on the project and LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT "
            "set to the data dir)"
        ),
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

    imp = commands.add_parser("import", help="convert a Label Studio export to hold_shapes.csv")
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
        "summary",
        help="count shapes, per-clip skill and prediction accuracy",
    )
    summ.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory (default: $HANDSTAND_DATA)",
    )
    summ.add_argument(
        "--write-catalogue",
        action="store_true",
        help="fill the catalogue's skill column from the reviewed shapes",
    )
    summ.add_argument(
        "--catalogue",
        type=pathlib.Path,
        default=None,
        help=f"catalogue to fill (default: <data_dir>/{CATALOGUE_NAME})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.hold_shapes``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "prelabel":
            if args.limit is not None and args.limit < 0:
                parser.error("--limit must be >= 0")
            report = prelabel(
                data=args.data,
                videos=args.videos,
                source=args.source,
                limit=args.limit,
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

        root = pathlib.Path(args.data) if args.data is not None else data_dir()
        rows = read_label_rows(labels_path(root))
        summary = build_summary(rows)
        print(summarise_shapes(summary))
        if args.write_catalogue:
            catalogue_path = args.catalogue if args.catalogue is not None else root / CATALOGUE_NAME
            path, changed = write_catalogue_skill(summary.clip_skills, catalogue_path)
            print(
                f"  catalogue: wrote the skill of {changed} clip(s) "
                f"(of {len(summary.clip_skills)} with labels) -> {path}"
            )
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        # ValueError covers a broken export JSON and a CSV with the wrong
        # header; OSError covers a missing hold summary or catalogue.
        print(f"hold_shapes: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
