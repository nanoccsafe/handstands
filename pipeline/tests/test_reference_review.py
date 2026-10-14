"""Tests for :mod:`handstand.reference_review`.

Nothing here touches the real videos or the real data dir: the data directory is
built in ``tmp_path`` from a tiny ``hold_summary.csv``, catalogue, reviewed
``hold_shapes.csv`` and trainer report, and the one clip is generated with
``cv2.VideoWriter`` (flat frames whose grey level encodes the frame index).
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from handstand import catalogue as catalogue_module
from handstand import reference_review as reference_review
from handstand.reference_review import (
    FRAMES_DIRNAME,
    build_task,
    fill,
    import_export,
    load_reference_holds,
    prelabel,
    read_label_rows,
)

# --------------------------------------------------------------------------- #
# Fixture constants
# --------------------------------------------------------------------------- #

WIDTH = 64
HEIGHT = 96
FPS = 30
FRAMES = 20
FRAME_STEP = 12
VIDEO_NAME = "generated clip.avi"

CLIP = "aaaaaaaaaaaa"
OTHER_CLIP = "bbbbbbbbbbbb"
MISSING_CLIP = "cccccccccccc"
GONE_CLIP = "dddddddddddd"

MISSING_KEY = catalogue_module.MISSING_COLUMN

#: Hold-summary columns the module reads (plus the two info features).
HOLD_COLUMNS = (
    "clip_id",
    "source",
    "hold_id",
    "hold_frames",
    "valid_frames",
    "hold_start_ms",
    "hold_end_ms",
    "hold_duration_s",
    "line_deviation_median",
    "com_sway_sd",
    "leg_separation_median",
    "knee_angle_median",
    "hip_angle_median",
)


def hold(
    clip_id: str,
    hold_id: int,
    start_ms: int,
    end_ms: int,
    *,
    line_dev: float = 0.05,
    sway: float = 0.012,
    leg: float = 5.0,
    knee: float = 176.0,
    hip: float = 172.0,
) -> dict[str, object]:
    """One hold-summary row with the medians that make it a line by the pre-label."""
    return {
        "clip_id": clip_id,
        "source": "mediapipe",
        "hold_id": hold_id,
        "hold_frames": 12,
        "valid_frames": 12,
        "hold_start_ms": start_ms,
        "hold_end_ms": end_ms,
        "hold_duration_s": round((end_ms - start_ms) / 1000.0, 3),
        "line_deviation_median": line_dev,
        "com_sway_sd": sway,
        "leg_separation_median": leg,
        "knee_angle_median": knee,
        "hip_angle_median": hip,
    }


#: CLIP has two line holds (0.6 s and 0.3 s); OTHER's only hold is a tuck.
LINE_LONG = hold(CLIP, 0, 0, 600)
LINE_SHORT = hold(CLIP, 1, 200, 500, line_dev=0.08, sway=0.02)
TUCK = hold(OTHER_CLIP, 0, 0, 500, leg=8.0, knee=100.0, hip=170.0)
MISSING_LINE = hold(MISSING_CLIP, 0, 0, 400)
GONE_LINE = hold(GONE_CLIP, 0, 0, 400, line_dev=0.4, sway=0.3)

#: Reviewed shapes: the two CLIP holds are line, OTHER's is a tuck.
SHAPE_ROWS = [
    {"clip_id": CLIP, "hold_id": 0, "shape": "line", "predicted_shape": "line"},
    {"clip_id": CLIP, "hold_id": 1, "shape": "line", "predicted_shape": "line"},
    {"clip_id": OTHER_CLIP, "hold_id": 0, "shape": "tuck", "predicted_shape": "tuck"},
    {"clip_id": MISSING_CLIP, "hold_id": 0, "shape": "line", "predicted_shape": "line"},
    {"clip_id": GONE_CLIP, "hold_id": 0, "shape": "line", "predicted_shape": "line"},
]


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #


def write_video(path: pathlib.Path) -> pathlib.Path:
    """A tiny MJPEG clip: frame ``i`` is a flat grey of ``i * FRAME_STEP``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), f"cv2.VideoWriter could not open {path}"
    for index in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), index * FRAME_STEP, np.uint8))
    writer.release()
    return path


def write_csv(
    path: pathlib.Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_catalogue(path: pathlib.Path, rows: list[dict[str, str]]) -> pathlib.Path:
    """A catalogue with every column, so ``fill``/``import`` have a real sheet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(catalogue_module.COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            full = {column: "" for column in catalogue_module.COLUMNS}
            full[MISSING_KEY] = "false"
            full.update(row)
            writer.writerow(full)
    return path


def write_hold_shapes(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    columns = ("clip_id", "hold_id", "shape", "predicted_shape", "agreed", "labeled_at")
    full = [{**{"agreed": "true", "labeled_at": ""}, **row} for row in rows]
    return write_csv(path, columns, full)


def write_trainer_report(path: pathlib.Path, present: dict[str, bool]) -> pathlib.Path:
    rows = [
        {"clip_id": clip, "trainer_present": "true" if value else "false"}
        for clip, value in present.items()
    ]
    return write_csv(path, ("clip_id", "trainer_present"), rows)


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A data dir and a videos dir with one generated clip, four clips in all."""
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    write_video(videos / VIDEO_NAME)
    write_csv(
        data / "features" / "mediapipe" / "hold_summary.csv",
        HOLD_COLUMNS,
        [LINE_LONG, LINE_SHORT, TUCK, MISSING_LINE, GONE_LINE],
    )
    write_hold_shapes(data / "labels" / "hold_shapes.csv", SHAPE_ROWS)
    return data, videos


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_catalogue_by_clip(path: pathlib.Path) -> dict[str, dict[str, str]]:
    return {row["clip_id"]: row for row in catalogue_module.read_rows(path)}


# --------------------------------------------------------------------------- #
# fill
# --------------------------------------------------------------------------- #


def test_fill_fills_only_empty_cells_and_leaves_non_empty_ones(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, _ = workspace
    catalogue = write_catalogue(
        data / "catalogue.csv",
        [
            # Non-empty outcome/camera_angle must survive; empty skill/notes filled.
            {
                "clip_id": CLIP,
                "filename": VIDEO_NAME,
                "outcome": "hold",
                "camera_angle": "side",
            },
            # Non-empty hold_s and skill must survive.
            {
                "clip_id": OTHER_CLIP,
                "filename": VIDEO_NAME,
                "hold_s": "9.9",
                "skill": "tuck",
            },
            # A missing clip is skipped entirely.
            {"clip_id": MISSING_CLIP, "filename": "gone.mp4", MISSING_KEY: "true"},
            # A trainer in the clip: the note is appended.
            {"clip_id": GONE_CLIP, "filename": "gone2.mp4"},
        ],
    )
    write_trainer_report(data / "reports" / "trainer_report.csv", {CLIP: True, GONE_CLIP: False})

    report = fill(data=data)

    assert report.dry_run is False
    assert report.backup is not None and report.backup.is_file()
    assert report.backup.parent == data / "backups"
    assert report.counts == {
        "hold_s": 2,  # CLIP (0.6) and GONE_CLIP (0.4); OTHER_CLIP already had 9.9
        "outcome": 2,  # OTHER_CLIP and GONE_CLIP; CLIP already had "hold"
        "camera_angle": 2,  # OTHER_CLIP and GONE_CLIP; CLIP already had "side"
        "skill": 2,  # CLIP (line) and GONE_CLIP (line); OTHER_CLIP already "tuck"
        "notes": 1,  # CLIP got "trainer"
    }

    rows = read_catalogue_by_clip(catalogue)
    assert rows[CLIP]["hold_s"] == "0.6"
    assert rows[CLIP]["outcome"] == "hold"  # untouched
    assert rows[CLIP]["camera_angle"] == "side"  # untouched
    assert rows[CLIP]["skill"] == "line"
    assert rows[CLIP]["notes"] == "trainer"

    assert rows[OTHER_CLIP]["hold_s"] == "9.9"  # untouched
    assert rows[OTHER_CLIP]["skill"] == "tuck"  # untouched
    assert rows[OTHER_CLIP]["outcome"] == "hold"
    assert rows[OTHER_CLIP]["camera_angle"] == "side"

    assert rows[GONE_CLIP]["hold_s"] == "0.4"
    assert rows[GONE_CLIP]["skill"] == "line"

    # The missing row was skipped: every annotation column is still empty.
    missing = rows[MISSING_CLIP]
    assert all(not missing[column] for column in catalogue_module.ANNOTATION_COLUMNS)


def test_fill_appends_trainer_once(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, _ = workspace
    write_catalogue(
        data / "catalogue.csv",
        [
            {"clip_id": CLIP, "filename": VIDEO_NAME, "notes": "second person present"},
        ],
    )
    write_trainer_report(data / "reports" / "trainer_report.csv", {CLIP: True})

    first = fill(data=data)
    assert first.counts["notes"] == 1
    assert read_catalogue_by_clip(data / "catalogue.csv")[CLIP]["notes"] == (
        "second person present; trainer"
    )

    # Running again changes nothing: the word is already there.
    second = fill(data=data)
    assert second.counts["notes"] == 0
    assert read_catalogue_by_clip(data / "catalogue.csv")[CLIP]["notes"].count("trainer") == 1


def test_fill_dry_run_writes_nothing(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, _ = workspace
    catalogue = write_catalogue(
        data / "catalogue.csv", [{"clip_id": CLIP, "filename": VIDEO_NAME}]
    )
    write_trainer_report(data / "reports" / "trainer_report.csv", {CLIP: True})
    before = catalogue.read_bytes()

    report = fill(data=data, dry_run=True)

    assert report.dry_run is True
    assert report.backup is None
    assert report.counts["hold_s"] == 1
    assert catalogue.read_bytes() == before
    assert not (data / "backups").exists()


# --------------------------------------------------------------------------- #
# prelabel
# --------------------------------------------------------------------------- #


def test_prelabel_prelabels_only_line_holds_longest_first(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    write_catalogue(
        data / "catalogue.csv",
        [
            {"clip_id": CLIP, "filename": VIDEO_NAME},
            {"clip_id": OTHER_CLIP, "filename": "other.avi"},
            {"clip_id": MISSING_CLIP, "filename": "gone.mp4", MISSING_KEY: "true"},
            {"clip_id": GONE_CLIP, "filename": "gone2.mp4"},
        ],
    )

    report = prelabel(
        data=data, videos=videos, config_path=tmp_path / "reference_review_config.xml"
    )

    # CLIP's two line holds (0.6 s then 0.3 s); the tuck, the missing and the
    # video-less clips are not.
    assert report.holds == 3  # CLIP, MISSING and GONE line holds survive the shape filter
    assert report.tasks_written == 2
    # Only the video-less clip is skipped; the missing clip never entered the list.
    assert [clip for clip, _ in report.clips_skipped] == [GONE_CLIP]

    tasks = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert [task["data"]["hold_id"] for task in tasks] == [0, 1]  # longest first
    assert [task["data"]["duration_s"] for task in tasks] == [0.6, 0.3]
    assert set(tasks[0]["data"]) == {
        "image25",
        "image50",
        "image75",
        "clip_id",
        "hold_id",
        "duration_s",
        "info",
    }
    assert tasks[0]["data"]["info"] == (
        f"{CLIP} / hold 0 / 0.60 s / line dev 0.05 / sway sd 0.01"
    )
    assert tasks[1]["data"]["info"] == (
        f"{CLIP} / hold 1 / 0.30 s / line dev 0.08 / sway sd 0.02"
    )
    for task in tasks:
        for quantile in (25, 50, 75):
            url = task["data"][f"image{quantile}"]
            assert url.startswith("/data/local-files/?d=label_frames_reference/")
            assert url.endswith(f"{CLIP}_h{task['data']['hold_id']}_{quantile}.jpg")
    assert "predictions" not in tasks[0]

    # Three frames per line hold, and only for the CLIP holds.
    frames = sorted((data / FRAMES_DIRNAME).glob("*.jpg"))
    assert [path.name for path in frames] == [
        f"{CLIP}_h0_25.jpg",
        f"{CLIP}_h0_50.jpg",
        f"{CLIP}_h0_75.jpg",
        f"{CLIP}_h1_25.jpg",
        f"{CLIP}_h1_50.jpg",
        f"{CLIP}_h1_75.jpg",
    ]


def test_prelabel_skips_existing_frames_and_rewrites_tasks(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    write_catalogue(
        data / "catalogue.csv",
        [
            {"clip_id": CLIP, "filename": VIDEO_NAME},
            {"clip_id": GONE_CLIP, "filename": "gone2.mp4"},
        ],
    )
    config = tmp_path / "config.xml"
    first = prelabel(data=data, videos=videos, config_path=config)
    assert first.frames_written == 6
    assert first.frames_skipped == 0

    second = prelabel(data=data, videos=videos, config_path=config)
    assert second.frames_written == 0
    assert second.frames_skipped == 6
    assert second.tasks_written == first.tasks_written


def test_prelabel_config_is_single_variable_per_value(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """Label Studio 1.23 reads a ``value`` as ONE variable, name and all."""
    data, videos = workspace
    write_catalogue(data / "catalogue.csv", [{"clip_id": CLIP, "filename": VIDEO_NAME}])
    config = tmp_path / "config.xml"
    report = prelabel(data=data, videos=videos, config_path=config)

    root = ET.parse(config).getroot()
    assert root.tag == "View"
    info = root.find("Text")
    assert info is not None and info.get("value") == "$info"
    images = list(root.iter("Image"))
    assert len(images) == 3
    assert [image.get("value") for image in images] == ["$image25", "$image50", "$image75"]
    reference = root.find("Choices[@name='reference']")
    assert reference is not None
    assert reference.get("toName") == "image50"
    assert reference.get("choice") == "single"
    assert reference.get("required") == "true"
    assert [(c.get("value"), c.get("hotkey")) for c in reference] == [
        ("yes", "1"),
        ("no", "2"),
    ]
    wall = root.find("Choices[@name='wall']")
    assert wall is not None
    assert wall.get("toName") == "image50"
    assert wall.get("choice") == "multiple"
    assert [(c.get("value"), c.get("hotkey")) for c in wall] == [("wall", "3")]

    tasks = json.loads(report.json_path.read_text(encoding="utf-8"))
    values = [value for element in root.iter() if (value := element.get("value")) and "$" in value]
    assert values
    for value in values:
        assert re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", value), value
        for task in tasks:
            assert value[1:] in task["data"]

    # The committed config is generated from the module: it must stay in sync.
    assert reference_review.default_config_path().name == "reference_review_config.xml"
    assert reference_review.default_config_path().read_text(encoding="utf-8") == (
        reference_review.config_xml()
    )


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #


def annotation(
    *,
    reference: str | None = None,
    wall: bool = False,
    labeled_at: str = "2026-10-14T10:00:00Z",
    cancelled: bool = False,
) -> dict[str, object]:
    """One review annotation: a required reference and an optional wall check."""
    result: list[dict[str, object]] = []
    if reference is not None:
        result.append(
            {
                "type": "choices",
                "from_name": "reference",
                "to_name": "image50",
                "value": {"choices": [reference]},
            }
        )
    if wall:
        result.append(
            {
                "type": "choices",
                "from_name": "wall",
                "to_name": "image50",
                "value": {"choices": ["wall"]},
            }
        )
    return {"result": result, "updated_at": labeled_at, "was_cancelled": cancelled}


def write_export(path: pathlib.Path, tasks: list[dict[str, object]]) -> pathlib.Path:
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


def test_import_writes_the_review_and_updates_the_catalogue(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, _ = workspace
    catalogue = write_catalogue(
        data / "catalogue.csv",
        [
            {"clip_id": CLIP, "filename": VIDEO_NAME, "notes": "second person present"},
            {"clip_id": OTHER_CLIP, "filename": "other.avi"},
            {"clip_id": GONE_CLIP, "filename": "gone2.mp4"},
        ],
    )
    export = write_export(
        tmp_path / "export.json",
        [
            # Two annotations: the last one wins, and it says yes.
            {
                "data": {"clip_id": CLIP, "hold_id": 0},
                "annotations": [
                    annotation(reference="no"),
                    annotation(reference="yes", labeled_at="2026-10-14T11:00:00Z"),
                ],
            },
            {"data": {"clip_id": CLIP, "hold_id": 1}, "annotations": [annotation(reference="no")]},
            # A yes against a wall is not reference-worthy and notes the wall.
            {
                "data": {"clip_id": OTHER_CLIP, "hold_id": 0},
                "annotations": [annotation(reference="yes", wall=True)],
            },
            # The cancelled yes is ignored; the live "no" is the decision.
            {
                "data": {"clip_id": GONE_CLIP, "hold_id": 0},
                "annotations": [
                    annotation(reference="yes", cancelled=True),
                    annotation(reference="no"),
                ],
            },
            # No clip/hold, and no decision: both skipped.
            {"data": {}, "annotations": [annotation(reference="yes")]},
            {"data": {"clip_id": CLIP, "hold_id": 2}, "annotations": []},
        ],
    )

    report = import_export(export, data=data)

    assert (report.tasks, report.annotated, report.skipped) == (6, 4, 2)
    rows = {(row["clip_id"], row["hold_id"]): row for row in read_label_rows(report.out_path)}
    assert set(rows) == {(CLIP, 0), (CLIP, 1), (OTHER_CLIP, 0), (GONE_CLIP, 0)}
    assert rows[(CLIP, 0)]["reference"] == "yes"
    assert rows[(CLIP, 0)]["labeled_at"] == "2026-10-14T11:00:00Z"
    assert rows[(CLIP, 1)]["reference"] == "no"
    assert rows[(OTHER_CLIP, 0)]["reference"] == "yes"
    assert rows[(OTHER_CLIP, 0)]["wall"] == "true"
    assert rows[(GONE_CLIP, 0)]["reference"] == "no"

    # good_clip is overwritten from the review: a yes-not-wall hold is a good clip.
    rows_by_clip = read_catalogue_by_clip(catalogue)
    assert rows_by_clip[CLIP]["good_clip"] == "1"
    assert rows_by_clip[OTHER_CLIP]["good_clip"] == "0"
    assert rows_by_clip[GONE_CLIP]["good_clip"] == "0"
    # The wall is noted once, after whatever was already typed.
    assert rows_by_clip[CLIP]["notes"] == "second person present"
    assert rows_by_clip[OTHER_CLIP]["notes"] == "wall"

    assert load_reference_holds(data=data) == {(CLIP, 0)}


def test_import_is_idempotent(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, _ = workspace
    write_catalogue(
        data / "catalogue.csv",
        [{"clip_id": OTHER_CLIP, "filename": VIDEO_NAME}],
    )
    export = write_export(
        tmp_path / "export.json",
        [
            {
                "data": {"clip_id": OTHER_CLIP, "hold_id": 0},
                "annotations": [annotation(reference="yes", wall=True)],
            }
        ],
    )

    import_export(export, data=data)
    first_csv = (data / "labels" / "reference_review.csv").read_bytes()
    first_notes = read_catalogue_by_clip(data / "catalogue.csv")[OTHER_CLIP]["notes"]

    report = import_export(export, data=data)

    assert (data / "labels" / "reference_review.csv").read_bytes() == first_csv
    assert report.rows == 1
    assert read_catalogue_by_clip(data / "catalogue.csv")[OTHER_CLIP]["notes"] == first_notes
    assert first_notes.count("wall") == 1
    assert read_catalogue_by_clip(data / "catalogue.csv")[OTHER_CLIP]["good_clip"] == "0"


# --------------------------------------------------------------------------- #
# small helpers / CLI
# --------------------------------------------------------------------------- #


def test_frame_name_and_task_image_uri(tmp_path: pathlib.Path) -> None:
    assert reference_review.frame_name(CLIP, 3, 50) == f"{CLIP}_h3_50.jpg"
    path = tmp_path / FRAMES_DIRNAME / f"{CLIP}_h3_50.jpg"
    assert reference_review.task_image_uri(path, tmp_path) == (
        f"/data/local-files/?d={FRAMES_DIRNAME}/{CLIP}_h3_50.jpg"
    )


def test_build_task_marks_missing_numbers_as_na() -> None:
    row = hold(CLIP, 0, 0, 500, line_dev=0.123, sway=0.456)
    row["line_deviation_median"] = ""
    row["com_sway_sd"] = ""
    task = build_task(CLIP, 0, row, pathlib.Path("/tmp/frames"), pathlib.Path("/tmp"))
    assert task["data"]["info"] == f"{CLIP} / hold 0 / 0.50 s / line dev n/a / sway sd n/a"


def test_main_fill_dry_run_prints_counts(
    workspace: tuple[pathlib.Path, pathlib.Path], capsys: pytest.CaptureFixture[str]
) -> None:
    data, _ = workspace
    write_catalogue(data / "catalogue.csv", [{"clip_id": CLIP, "filename": VIDEO_NAME}])
    write_trainer_report(data / "reports" / "trainer_report.csv", {CLIP: False})

    code = reference_review.main(["fill", "--data", str(data), "--dry-run"])

    assert code == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "hold_s: 1" in out
    assert not (data / "backups").exists()
