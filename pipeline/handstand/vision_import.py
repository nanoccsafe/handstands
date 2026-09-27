"""Import the Apple Vision body-pose CSV into the pipeline's parquet schema.

The Vision runner (`swift/VisionPose`, driven by `tools/mac/run_vision.sh`)
writes one CSV per clip and rotation mode::

    <data_dir>/keypoints/vision_raw/<rotate>/<clip_id>.csv
    <data_dir>/keypoints/vision_raw/<rotate>/<clip_id>.json   # its run manifest

and this module turns each into the two parquet files every other stage
already knows how to read::

    <data_dir>/keypoints/vision_multi/<rotate>/<clip_id>.parquet   # every person
    <data_dir>/keypoints/vision/<rotate>/<clip_id>.parquet          # one person

plus a JSON sidecar next to each. So a Vision run and a MediaPipe run of the
same clip and the same rotation mode differ only in which directory they are
in, and the bake-off (chainlink #16) is a comparison of two files with the
same columns.

The schemas are the MediaPipe ones, unchanged:

* multi: :data:`pose_mediapipe.PARQUET_COLUMNS_MULTI` — one block per person,
  ``person_idx`` 0..P-1, and the single all-NaN block with
  ``person_idx = NO_PERSON_IDX`` for a frame nobody was detected in.
* single: :data:`pose_mediapipe.PARQUET_COLUMNS`, no ``person_idx``, holding the
  person whose **wrists are lowest** in each frame — the same rule
  :func:`pose_mediapipe.lowest_wrist_pose` applies to the MediaPipe run, so
  "which body is this frame about" is decided the same way for both models.
  Picking the athlete out of several people is still
  :mod:`handstand.athlete`'s job, not this module's.

Three columns do not survive the crossing, and are the whole of the difference
between a Vision parquet and a MediaPipe one:

==============  ==================================================================
``visibility``   Vision's per-joint confidence.
``presence``    NaN — Vision reports no separate presence score.
``z``           NaN — Vision's body-pose model reports no depth.
==============  ==================================================================

Coordinates need no conversion: the runner already wrote display-frame pixels
(origin top left, ``y`` down), the same space MediaPipe's parquets hold, and
the CSV is a lossless enough carrier (4 decimals of a pixel, 6 of a confidence).

CLI::

    cd pipeline
    uv run python -m handstand.vision_import --rotate auto
    uv run python -m handstand.vision_import --rotate none --rotate auto \\
        --clips 6508f9b355bd 64184de33f84
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from handstand.paths import data_dir
from handstand.pose_mediapipe import (
    NO_PERSON_IDX,
    PARQUET_COLUMNS,
    PARQUET_COLUMNS_MULTI,
    PERSON_COLUMN,
)

__all__ = [
    "CSV_COLUMNS",
    "JOINT_NAMES",
    "MEDIAPIPE_SHARED_JOINTS",
    "MODEL_NAME",
    "MULTI_OUTPUT_DIRNAME",
    "NO_PERSON_IDX",
    "PARQUET_COLUMNS",
    "PARQUET_COLUMNS_MULTI",
    "PERSON_COLUMN",
    "RAW_OUTPUT_DIRNAME",
    "SINGLE_OUTPUT_DIRNAME",
    "VISION_ONLY_JOINTS",
    "WRIST_JOINTS",
    "ClipReport",
    "build_arg_parser",
    "collect_clips",
    "load_manifest",
    "lowest_wrist_person_index",
    "main",
    "mean_wrist_y",
    "parse_bool",
    "read_csv",
    "run_clip",
    "to_multi_table",
    "to_single_table",
]

#: What the runner reports the model as, in the sidecar.
MODEL_NAME = "apple-vision-body-pose"

#: Output roots under ``<data_dir>/keypoints/``.
RAW_OUTPUT_DIRNAME = "vision_raw"
MULTI_OUTPUT_DIRNAME = "vision_multi"
SINGLE_OUTPUT_DIRNAME = "vision"

#: The CSV the runner writes, in the order it writes it.
CSV_COLUMNS: tuple[str, ...] = (
    "frame_idx",
    "t_ms",
    "person_idx",
    "joint",
    "x",
    "y",
    "confidence",
    "rotated",
    "detected",
)

#: Vision's 19 body-pose joints, in the runner's joint order (which is the CSV
#: row order, and therefore the row order inside a person's block). The
#: authoritative list on the Swift side is ``VisionJoint.columnNames``; the 17
#: shared names are spelled exactly as ``pose_mediapipe.JOINT_NAMES`` spells
#: them so the two can be compared joint by joint.
JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "neck",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "root",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

#: The two joints MediaPipe's 33 landmarks have no name for: Vision's shoulder
#: and hip midpoints. They are Vision's alone and every other stage, which
#: indexes joints by MediaPipe's names, simply never sees them.
VISION_ONLY_JOINTS: tuple[str, ...] = ("neck", "root")

#: The ``JOINT_NAMES`` that MediaPipe also has — 19 minus the two above.
MEDIAPIPE_SHARED_JOINTS: tuple[str, ...] = tuple(
    joint for joint in JOINT_NAMES if joint not in VISION_ONLY_JOINTS
)

#: The joints the lowest-wrist rule reads.
WRIST_JOINTS: tuple[str, ...] = ("left_wrist", "right_wrist")


# --------------------------------------------------------------------------- #
# Reading the CSV
# --------------------------------------------------------------------------- #


def parse_bool(value: object, column: str) -> bool:
    """Read one of the CSV's boolean fields, which the runner writes lower-case.

    Raises :class:`ValueError` for anything else rather than guessing: a
    silently ``False`` ``rotated`` would quietly turn an ``auto`` run into a
    ``none`` one.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"{column}: expected true or false, got {value!r}")


def read_csv(path: str | pathlib.Path) -> pd.DataFrame:
    """Read one runner CSV into a typed table.

    Empty ``x``/``y``/``confidence`` fields — the placeholder block of a frame
    nobody was detected in — come out as NaN, and the dtypes are the ones the
    parquet files are written with, so the conversion is a column rename rather
    than a cast.
    """
    path = pathlib.Path(path)
    frame = pd.read_csv(
        path,
        dtype={
            "frame_idx": "int64",
            "t_ms": "int64",
            "person_idx": "int64",
            "joint": "str",
            "x": "float64",
            "y": "float64",
            "confidence": "float64",
            "rotated": "str",
            "detected": "str",
        },
        na_values=[""],
        keep_default_na=True,
    )
    missing = [column for column in CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}")
    if frame.empty:
        raise ValueError(f"{path.name}: no rows")
    return pd.DataFrame(
        {
            "frame_idx": frame["frame_idx"].astype("int64"),
            "t_ms": frame["t_ms"].astype("int64"),
            "person_idx": frame["person_idx"].astype("int64"),
            "joint": frame["joint"].astype("str"),
            "x": frame["x"].astype("float64"),
            "y": frame["y"].astype("float64"),
            "confidence": frame["confidence"].astype("float64"),
            "rotated": [parse_bool(value, "rotated") for value in frame["rotated"]],
            "detected": [parse_bool(value, "detected") for value in frame["detected"]],
        }
    )


# --------------------------------------------------------------------------- #
# The lowest-wrist person
# --------------------------------------------------------------------------- #


def mean_wrist_y(wrist_ys: Sequence[float]) -> float:
    """Mean y of both wrists, or NaN when either wrist is missing.

    Not a mean of whatever is there: a body with one wrist found cannot be
    ranked against one with both, and silently averaging the single value would
    make it rankable. Mirrors :func:`pose_mediapipe.mean_wrist_y`.
    """
    values = np.asarray(wrist_ys, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(values.mean())


def lowest_wrist_person_index(wrist_means: Sequence[float]) -> int:
    """Which of several people has their wrists lowest in the image.

    The winner is the largest mean wrist ``y`` — the body most likely to be on
    its hands. People whose wrists are not both visible score NaN and are
    skipped; when nobody can be ranked, or there is only one person, the first
    person is returned, so a single-person frame is judged exactly as the
    MediaPipe runner judges it.

    Raises :class:`ValueError` for an empty sequence: there is no person to
    pick, and a caller that got here has a bug.
    """
    scores = np.asarray(wrist_means, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1:
        raise ValueError(f"expected at least one person, got {scores.shape}")
    if scores.size == 1:
        return 0
    rankable = np.flatnonzero(np.isfinite(scores))
    if rankable.size == 0:
        return 0
    return int(rankable[np.argmax(scores[rankable])])


def _wrist_means_by_person(frame: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Mean wrist y of every (frame, person) that has a ``left_wrist``/``right_wrist``.

    Both wrists are required, so a person missing one gets NaN and is skipped
    by :func:`lowest_wrist_person_index` rather than ranked on half a body.
    """
    wrists = frame[frame["joint"].isin(WRIST_JOINTS)]
    grouped: dict[tuple[int, int], float] = {}
    if wrists.empty:
        return grouped
    for (frame_idx, person_idx), block in wrists.groupby(["frame_idx", "person_idx"], sort=False):
        values = {str(joint): block.loc[block["joint"] == joint, "y"] for joint in WRIST_JOINTS}
        ordered = []
        for joint in WRIST_JOINTS:
            column = values[joint]
            ordered.append(float(column.iloc[0]) if len(column) == 1 else float("nan"))
        grouped[(int(frame_idx), int(person_idx))] = mean_wrist_y(ordered)
    return grouped


# --------------------------------------------------------------------------- #
# CSV -> parquet tables
# --------------------------------------------------------------------------- #


def to_multi_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Every person, in the MediaPipe multi-person schema.

    One block of :data:`JOINT_NAMES` rows per person, ordered by ``frame_idx``,
    then ``person_idx``, then joint. A frame with nobody in it keeps the single
    all-NaN block with ``person_idx = NO_PERSON_IDX`` and ``detected = false``
    the runner wrote, so no frame is ever missing from the file.

    The three columns Vision has nothing to say about (``z``, ``presence``, and
    ``visibility``, which becomes the confidence) are filled in as described in
    the module docstring.
    """
    ordered = frame.sort_values(["frame_idx", "person_idx"], kind="stable")
    rows = len(ordered)
    return pd.DataFrame(
        {
            "frame_idx": ordered["frame_idx"].to_numpy(dtype="int64"),
            "t_ms": ordered["t_ms"].to_numpy(dtype="int64"),
            "joint": ordered["joint"].to_numpy(dtype="str"),
            "x": ordered["x"].to_numpy(dtype="float64"),
            "y": ordered["y"].to_numpy(dtype="float64"),
            "z": np.full(rows, np.nan, dtype="float64"),
            "visibility": ordered["confidence"].to_numpy(dtype="float64"),
            "presence": np.full(rows, np.nan, dtype="float64"),
            "rotated": ordered["rotated"].to_numpy(dtype=bool),
            "detected": ordered["detected"].to_numpy(dtype=bool),
            PERSON_COLUMN: ordered["person_idx"].to_numpy(dtype="int64"),
        },
        columns=list(PARQUET_COLUMNS_MULTI),
    )


def to_single_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One person per frame — the lowest-wrist one — in the single-person schema.

    The MediaPipe schema with no ``person_idx``: 19 rows per frame, ordered by
    ``frame_idx`` then joint. A frame whose people cannot be ranked (or that has
    none) keeps the runner's placeholder block: NaN coordinates and
    ``detected = false``.
    """
    wrist_means = _wrist_means_by_person(frame)
    blocks: list[pd.DataFrame] = []
    for frame_idx, block in frame.groupby("frame_idx", sort=True):
        detected = block[block["detected"]]
        if detected.empty:
            blocks.append(block[block[PERSON_COLUMN] == NO_PERSON_IDX])
            continue
        people = detected[PERSON_COLUMN].drop_duplicates().to_numpy(dtype="int64")
        scores = [wrist_means.get((int(frame_idx), int(person)), float("nan")) for person in people]
        winner = people[lowest_wrist_person_index(scores)]
        blocks.append(detected[detected[PERSON_COLUMN] == winner])
    if not blocks:
        raise ValueError("no frames in the CSV")

    selected = pd.concat(blocks, ignore_index=True)
    rows = len(selected)
    return pd.DataFrame(
        {
            "frame_idx": selected["frame_idx"].to_numpy(dtype="int64"),
            "t_ms": selected["t_ms"].to_numpy(dtype="int64"),
            "joint": selected["joint"].to_numpy(dtype="str"),
            "x": selected["x"].to_numpy(dtype="float64"),
            "y": selected["y"].to_numpy(dtype="float64"),
            "z": np.full(rows, np.nan, dtype="float64"),
            "visibility": selected["confidence"].to_numpy(dtype="float64"),
            "presence": np.full(rows, np.nan, dtype="float64"),
            "rotated": selected["rotated"].to_numpy(dtype=bool),
            "detected": selected["detected"].to_numpy(dtype=bool),
        },
        columns=list(PARQUET_COLUMNS),
    )


# --------------------------------------------------------------------------- #
# Per-clip import
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipReport:
    """What happened to one clip."""

    clip_id: str
    rotate: str
    csv_path: pathlib.Path
    frame_count: int
    detected_frames: int
    rotated_frames: int
    #: Frames per number of detected people, ``{people: frames}``.
    frames_by_people: dict[int, int]
    multi_parquet_path: pathlib.Path
    multi_json_path: pathlib.Path
    single_parquet_path: pathlib.Path
    single_json_path: pathlib.Path
    #: Frames the single-person table attributes to a person.
    selected_frames: int = 0
    skipped: bool = False

    @property
    def detected_percent(self) -> float:
        """Percentage of frames with at least one person."""
        if not self.frame_count:
            return 0.0
        return 100.0 * self.detected_frames / self.frame_count

    @property
    def rotated_percent(self) -> float:
        """Percentage of frames fed to the model rotated 180°."""
        if not self.frame_count:
            return 0.0
        return 100.0 * self.rotated_frames / self.frame_count

    @property
    def people_summary(self) -> str:
        """``"0:44 1:200"`` — frames per number of detected people."""
        return " ".join(
            f"{people}:{self.frames_by_people[people]}" for people in sorted(self.frames_by_people)
        )


def load_manifest(csv_path: str | pathlib.Path) -> dict[str, object]:
    """Read the runner's ``<clip_id>.json`` next to a CSV, or ``{}`` if absent.

    The CSV is nothing but keypoints; the macOS version, the Vision revision and
    the runtime only exist on the Mac, so they come back in this file and are
    folded into the parquet's sidecar.
    """
    manifest_path = pathlib.Path(csv_path).with_suffix(".json")
    if not manifest_path.is_file():
        return {}
    try:
        loaded = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{manifest_path.name}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{manifest_path.name}: expected a JSON object")
    return loaded


def _write_table(table: pd.DataFrame, parquet_path: pathlib.Path) -> None:
    """Write a parquet atomically — its existence is what marks a clip as done."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = parquet_path.with_name(f".{parquet_path.name}.tmp")
    try:
        table.to_parquet(temporary, index=False)
        temporary.replace(parquet_path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_sidecar(sidecar: Mapping[str, object], json_path: pathlib.Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_name(f".{json_path.name}.tmp")
    temporary.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    temporary.replace(json_path)


def _base_sidecar(
    clip_id: str,
    rotate: str,
    csv_path: pathlib.Path,
    manifest: Mapping[str, object],
    frame_count: int,
    detected_frames: int,
    rotated_frames: int,
    frames_by_people: Mapping[int, int],
) -> dict[str, object]:
    """The keys both sidecars share, from the CSV and the run manifest."""
    sidecar: dict[str, object] = {
        "clip_id": clip_id,
        "source_file": str(manifest.get("source_file", "")),
        "source_csv": csv_path.name,
        "model": MODEL_NAME,
        "rotate": rotate,
        "frame_count": frame_count,
        "detected_frame_count": detected_frames,
        "rotated_frame_count": rotated_frames,
        "display_width": int(manifest["display_width"]) if "display_width" in manifest else None,
        "display_height": int(manifest["display_height"]) if "display_height" in manifest else None,
        "macos_version": str(manifest.get("macos_version", "unknown")),
        "runtime_seconds": manifest.get("runtime_seconds"),
        "frames_by_people": {
            str(people): count for people, count in sorted(frames_by_people.items())
        },
    }
    if "vision_revision" in manifest:
        sidecar["vision_revision"] = manifest["vision_revision"]
    return sidecar


def run_clip(
    csv_path: str | pathlib.Path,
    *,
    clip_id: str,
    rotate: str,
    out_root: pathlib.Path,
    overwrite: bool = False,
) -> ClipReport:
    """Import one runner CSV into the multi- and single-person parquets.

    ``out_root`` is the ``<data_dir>/keypoints`` directory; the two files land
    in ``vision_multi/<rotate>/`` and ``vision/<rotate>/``, each with a JSON
    sidecar. A clip whose parquets already exist is skipped unless ``overwrite``
    is set.
    """
    csv_path = pathlib.Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"no runner CSV: {csv_path}")
    out_root = pathlib.Path(out_root)
    multi_dir = out_root / MULTI_OUTPUT_DIRNAME / rotate
    single_dir = out_root / SINGLE_OUTPUT_DIRNAME / rotate
    multi_parquet = multi_dir / f"{clip_id}.parquet"
    single_parquet = single_dir / f"{clip_id}.parquet"
    multi_json = multi_dir / f"{clip_id}.json"
    single_json = single_dir / f"{clip_id}.json"
    if multi_parquet.exists() and single_parquet.exists() and not overwrite:
        return ClipReport(
            clip_id=clip_id,
            rotate=rotate,
            csv_path=csv_path,
            frame_count=0,
            detected_frames=0,
            rotated_frames=0,
            frames_by_people={},
            multi_parquet_path=multi_parquet,
            multi_json_path=multi_json,
            single_parquet_path=single_parquet,
            single_json_path=single_json,
            skipped=True,
        )

    frame = read_csv(csv_path)
    manifest = load_manifest(csv_path)

    per_frame = frame.groupby("frame_idx", sort=True)
    frame_count = int(per_frame.ngroups)
    detected = per_frame["detected"].any()
    rotated = per_frame["rotated"].first()
    detected_frames = int(detected.sum())
    rotated_frames = int(rotated.sum())
    people_per_frame = [
        block.loc[block["detected"], PERSON_COLUMN].nunique() for _, block in per_frame
    ]
    frames_by_people: dict[int, int] = {}
    for people in people_per_frame:
        frames_by_people[int(people)] = frames_by_people.get(int(people), 0) + 1

    multi_table = to_multi_table(frame)
    single_table = to_single_table(frame)
    selected_frames = int(single_table.loc[single_table["detected"], "frame_idx"].nunique())

    # The sidecar's frame count has to agree with the parquet's own: a
    # disagreement means the CSV and its manifest are from different runs.
    manifest_frames = manifest.get("frame_count")
    if manifest_frames is not None and int(manifest_frames) != frame_count:
        raise ValueError(
            f"{csv_path.name}: the run manifest says {manifest_frames} frames "
            f"but the CSV has {frame_count}; they are from different runs"
        )

    sidecar = _base_sidecar(
        clip_id=clip_id,
        rotate=rotate,
        csv_path=csv_path,
        manifest=manifest,
        frame_count=frame_count,
        detected_frames=detected_frames,
        rotated_frames=rotated_frames,
        frames_by_people=frames_by_people,
    )
    single_sidecar = dict(sidecar)
    single_sidecar["selection"] = "lowest_wrist"
    single_sidecar["selected_frame_count"] = selected_frames

    _write_table(multi_table, multi_parquet)
    _write_sidecar(sidecar, multi_json)
    _write_table(single_table, single_parquet)
    _write_sidecar(single_sidecar, single_json)

    return ClipReport(
        clip_id=clip_id,
        rotate=rotate,
        csv_path=csv_path,
        frame_count=frame_count,
        detected_frames=detected_frames,
        rotated_frames=rotated_frames,
        frames_by_people=frames_by_people,
        multi_parquet_path=multi_parquet,
        multi_json_path=multi_json,
        single_parquet_path=single_parquet,
        single_json_path=single_json,
        selected_frames=selected_frames,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def collect_clips(
    clips: Sequence[str] | None,
    raw_dir: pathlib.Path,
    rotate: str,
) -> list[pathlib.Path]:
    """Resolve ``--clips`` (or every CSV in the raw directory for ``rotate``).

    The argument is a clip id and may be written with or without the ``.csv``
    suffix, and may be a path.
    """
    if clips:
        paths = []
        for clip in clips:
            path = pathlib.Path(clip)
            if path.suffix != ".csv":
                path = path.with_suffix(".csv")
            if not path.is_absolute() and not path.is_file():
                path = raw_dir / path.name
            paths.append(path)
    else:
        paths = sorted(raw_dir.glob("*.csv"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "no runner CSV for "
            f"{rotate}: " + ", ".join(str(path) for path in missing)
            + "\nrun tools/mac/run_vision.sh --rotate " + rotate + " on the Mac mini first"
        )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m handstand.vision_import",
        description="Import the Apple Vision body-pose CSVs into the keypoint schema.",
    )
    parser.add_argument(
        "--rotate",
        action="append",
        choices=[mode for mode in ("none", "180", "auto")],
        default=None,
        help=(
            "rotation mode to import; repeat for several (default: every mode "
            "present under keypoints/vision_raw/)"
        ),
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help="clips to import (default: every CSV of the chosen rotation mode)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-import clips whose parquet already exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = data_dir() / "keypoints"
    raw_root = root / RAW_OUTPUT_DIRNAME
    if not raw_root.is_dir():
        print(f"no {raw_root}; run tools/mac/run_vision.sh on the Mac mini first")
        return 0
    rotates = args.rotate or sorted(path.name for path in raw_root.iterdir() if path.is_dir())

    failures = 0
    imported = 0
    for rotate in rotates:
        raw_dir = raw_root / rotate
        if not raw_dir.is_dir():
            print(f"skip  {rotate}: no {raw_dir}")
            continue
        clips = collect_clips(args.clips, raw_dir, rotate)
        if not clips:
            print(f"skip  {rotate}: no CSVs in {raw_dir}")
            continue
        for csv_path in clips:
            clip_id = csv_path.stem
            try:
                report = run_clip(
                    csv_path,
                    clip_id=clip_id,
                    rotate=rotate,
                    out_root=root,
                    overwrite=args.overwrite,
                )
            except Exception as error:  # one bad clip must not kill the batch
                failures += 1
                print(f"fail  {rotate}/{clip_id}: {error}")
                continue
            imported += 1
            if report.skipped:
                print(f"skip  {rotate}/{clip_id} ({report.multi_parquet_path} exists)")
                continue
            print(
                f"clip  {rotate}/{clip_id} frames={report.frame_count} "
                f"detected={report.detected_percent:.1f}% "
                f"rotated={report.rotated_percent:.1f}% "
                f"people[{report.people_summary}] "
                f"selected={report.selected_frames}/{report.frame_count} "
                f"-> {report.multi_parquet_path}, {report.single_parquet_path}"
            )
    if not imported and not failures:
        print("nothing to import")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
