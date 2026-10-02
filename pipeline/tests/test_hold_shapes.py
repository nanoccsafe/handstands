"""Tests for :mod:`handstand.hold_shapes`.

Nothing here touches the real videos: the data directory is built in ``tmp_path``
out of a tiny ``hold_summary.csv`` and a catalogue, and the one clip is generated
with ``cv2.VideoWriter`` — flat frames whose grey level encodes the frame index,
so a test can prove the frame written for a hold is the frame at the hold's
middle time and not merely *a* frame.
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
from handstand import hold_shapes as hold_shapes
from handstand.hold_shapes import (
    LABEL_COLUMNS,
    MANIFEST_COLUMNS,
    SHAPES,
    UNMEASURED,
    import_export,
    is_line_hold,
    line_holds,
    load_hold_shapes,
    predict_shape,
    prelabel,
    read_hold_summary,
    read_label_rows,
)

#: The generated clip: 20 frames at 30 fps, i.e. 633 ms, all display-oriented.
WIDTH = 64
HEIGHT = 96
FPS = 30
FRAMES = 20
#: Frame ``i`` is a flat grey of ``i * FRAME_STEP`` (MJPEG adds less than a
#: level), so a frame picked one hold too early or late is off by 12 — far more
#: than the tolerance any assertion uses.
FRAME_STEP = 12
VIDEO_NAME = "generated clip.avi"

CLIP = "aaaaaaaaaaaa"
OTHER_CLIP = "bbbbbbbbbbbb"
MISSING_CLIP = "cccccccccccc"

#: Hold 0 spans 0-400 ms (middle 200 ms = frame 6 exactly), hold 1 spans
#: 300-500 ms (middle 400 ms = frame 12 exactly), so neither pick is a tie.
#: The catalogue's trailing flag column, so the rows read as present.
MISSING_KEY = catalogue_module.MISSING_COLUMN

HOLD_COLUMNS = (
    "clip_id",
    "source",
    "hold_id",
    "hold_frames",
    "valid_frames",
    "hold_start_ms",
    "hold_end_ms",
    "hold_duration_s",
    "leg_separation_median",
    "knee_angle_median",
    "hip_angle_median",
)

#: The two hold-summary rows the whole fixture is built on: a line and a tuck.
LINE_HOLD = {
    "clip_id": CLIP,
    "source": "mediapipe",
    "hold_id": 0,
    "hold_frames": 12,
    "valid_frames": 12,
    "hold_start_ms": 0,
    "hold_end_ms": 400,
    "hold_duration_s": 0.4,
    "leg_separation_median": 5.0,
    "knee_angle_median": 176.0,
    "hip_angle_median": 172.0,
}
TUCK_HOLD = {
    **LINE_HOLD,
    "hold_id": 1,
    "hold_start_ms": 300,
    "hold_end_ms": 500,
    "hold_duration_s": 0.2,
    "leg_separation_median": 8.0,
    "knee_angle_median": 100.0,
    "hip_angle_median": 170.0,
}


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


def write_hold_summary(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """A one-clip hold summary in the schema :func:`handstand.features` writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HOLD_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_catalogue(path: pathlib.Path, rows: list[dict[str, str]]) -> pathlib.Path:
    """A catalogue with every column, so ``--write-catalogue`` has a real sheet."""
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


def write_labels(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """A reviewed ``hold_shapes.csv``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LABEL_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A data dir and a videos dir: one clip, two holds, one generated video."""
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    write_video(videos / VIDEO_NAME)
    write_hold_summary(data / "features" / "mediapipe" / "hold_summary.csv", [LINE_HOLD, TUCK_HOLD])
    write_catalogue(data / "catalogue.csv", [{"clip_id": CLIP, "filename": VIDEO_NAME}])
    return data, videos


def run_prelabel(
    data: pathlib.Path, videos: pathlib.Path, tmp_path: pathlib.Path, **kwargs: object
) -> hold_shapes.PrelabelReport:
    """Pre-label the fixture, with the generated config kept out of the tree."""
    return prelabel(
        data=data, videos=videos, config_path=tmp_path / "hold_shapes_config.xml", **kwargs
    )


def read_manifest(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def brightness(path: pathlib.Path) -> float:
    """Mean grey level of a written JPEG."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert image is not None, f"could not read back {path}"
    return float(image.mean())


# --------------------------------------------------------------------------- #
# predict_shape
# --------------------------------------------------------------------------- #


def test_predict_shape_reads_every_branch_off_the_medians() -> None:
    assert predict_shape(
        {"leg_separation_median": 5.0, "knee_angle_median": 176.0, "hip_angle_median": 172.0}
    ) == ("line", "legs 5°, knees 176°, hips 172°")
    assert predict_shape(
        {"leg_separation_median": 8.0, "knee_angle_median": 100.0, "hip_angle_median": 170.0}
    ) == ("tuck", "knees 100°")
    straddle, reason = predict_shape(
        {"leg_separation_median": 62.0, "knee_angle_median": 176.0, "hip_angle_median": 171.0}
    )
    assert straddle == "straddle"
    assert reason == "legs 62° apart"
    assert predict_shape(
        {"leg_separation_median": 4.0, "knee_angle_median": 170.0, "hip_angle_median": 130.0}
    ) == ("pike", "hips 130°")


def test_predict_shape_is_unmeasured_when_a_median_is_missing() -> None:
    # A blank CSV cell and a NaN are the same absence: nobody measured it.
    shape, reason = predict_shape(
        {"leg_separation_median": "", "knee_angle_median": 176.0, "hip_angle_median": 172.0}
    )
    assert (shape, reason) == (UNMEASURED, "not measured: leg_separation")
    assert (
        predict_shape(
            {
                "leg_separation_median": 4.0,
                "knee_angle_median": float("nan"),
                "hip_angle_median": 140.0,
            }
        )[0]
        == UNMEASURED
    )
    assert predict_shape({})[0] == UNMEASURED
    assert "hip_angle" in predict_shape({"hip_angle_median": ""})[1]


def test_predict_shape_threshold_edges_stay_on_the_line_side() -> None:
    # Exactly on a threshold is not over it: the pre-label must not drift.
    edges = {"leg_separation_median": 45.0, "knee_angle_median": 120.0, "hip_angle_median": 140.0}
    assert predict_shape(edges) == ("line", "legs 45°, knees 120°, hips 140°")
    assert predict_shape({**edges, "knee_angle_median": 119.9})[0] == "tuck"
    assert predict_shape({**edges, "leg_separation_median": 45.1})[0] == "straddle"
    assert predict_shape({**edges, "hip_angle_median": 139.9})[0] == "pike"


def test_predict_shape_orders_the_branches_the_way_a_viewer_sees_them() -> None:
    # Bent knees win over legs apart, which wins over a folded hip: a tuck with
    # the legs wide is still a tuck, and a straddle with bent knees is still
    # read as the tuck it most resembles at a glance.
    both = {"leg_separation_median": 60.0, "knee_angle_median": 100.0, "hip_angle_median": 130.0}
    assert predict_shape(both)[0] == "tuck"
    assert predict_shape({**both, "knee_angle_median": 150.0})[0] == "straddle"
    folded = {**both, "knee_angle_median": 150.0, "leg_separation_median": 10.0}
    assert predict_shape(folded)[0] == "pike"


# --------------------------------------------------------------------------- #
# prelabel
# --------------------------------------------------------------------------- #


def test_prelabel_writes_one_middle_frame_per_hold(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    report = run_prelabel(data, videos, tmp_path)

    assert report.written == 2
    assert report.holds == 2
    assert report.skipped == []
    assert report.counts == {
        **{shape: 0 for shape in SHAPES},
        "line": 1,
        "tuck": 1,
        UNMEASURED: 0,
    }

    rows = read_manifest(report.manifest)
    assert [row["image"] for row in rows] == [f"{CLIP}_h0.jpg", f"{CLIP}_h1.jpg"]
    # The middle of hold 0 (200 ms) is frame 6; of hold 1 (400 ms) is frame 12.
    assert [(row["frame_idx"], row["t_ms"]) for row in rows] == [("6", "200"), ("12", "400")]
    assert [row["hold_duration_s"] for row in rows] == ["0.4", "0.2"]
    assert [row["predicted_shape"] for row in rows] == ["line", "tuck"]
    assert rows[1]["reason"] == "knees 100°"
    assert set(rows[0]) == set(MANIFEST_COLUMNS)

    # The images really are those frames: the grey level of frame i is i * 12.
    for row in rows:
        expected = int(row["frame_idx"]) * FRAME_STEP
        assert brightness(report.frames_dir / row["image"]) == pytest.approx(expected, abs=6)


def test_prelabel_writes_a_task_with_the_shape_preselected(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    report = run_prelabel(data, videos, tmp_path)

    tasks = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert len(tasks) == 2
    assert set(tasks[0]["data"]) == {
        "image",
        "clip_id",
        "hold_id",
        "duration_s",
        "reason",
        "info",
    }
    assert tasks[0]["data"]["clip_id"] == CLIP
    assert tasks[0]["data"]["hold_id"] == 0
    assert tasks[0]["data"]["duration_s"] == 0.4
    # The header the config shows is joined into ONE field: Label Studio 1.23
    # reads a value with several $... as a single variable name.
    assert tasks[0]["data"]["info"] == f"{CLIP} / hold 0 / 0.4 s / legs 5°, knees 176°, hips 172°"
    assert tasks[1]["data"]["info"] == f"{CLIP} / hold 1 / 0.2 s / knees 100°"
    assert tasks[0]["data"]["image"].startswith("/data/local-files/?d=label_frames_holds/")
    assert tasks[0]["data"]["image"].endswith(f"{CLIP}_h0.jpg")
    first = tasks[0]["predictions"][0]
    assert first["model_version"] == hold_shapes.MODEL_VERSION
    assert first["result"][0]["type"] == "choices"
    assert first["result"][0]["from_name"] == "shape"
    assert first["result"][0]["value"]["choices"] == ["line"]
    assert tasks[1]["predictions"][0]["result"][0]["value"]["choices"] == ["tuck"]


def test_prelabel_leaves_an_unmeasured_hold_without_a_preselection(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    write_hold_summary(
        data / "features" / "mediapipe" / "hold_summary.csv",
        [{**LINE_HOLD, "leg_separation_median": ""}],
    )
    report = run_prelabel(data, videos, tmp_path)

    assert report.counts[UNMEASURED] == 1
    tasks = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert "predictions" not in tasks[0]
    assert tasks[0]["data"]["reason"] == "not measured: leg_separation"
    rows = read_manifest(report.manifest)
    assert rows[0]["predicted_shape"] == UNMEASURED


def test_prelabel_writes_the_labeling_config(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    report = run_prelabel(data, videos, tmp_path)

    assert report.config_path == tmp_path / "hold_shapes_config.xml"
    root = ET.parse(report.config_path).getroot()
    assert root.tag == "View"
    info = root.find("Text")
    assert info is not None
    # One variable, one task field: Label Studio 1.23 parses the whole value as
    # one variable name, so "$clip_id / hold $hold_id / ..." cannot be a template.
    assert info.get("value") == "$info"
    assert info.get("name") == "hold_info"
    image = root.find("Image")
    assert image is not None
    assert image.get("value") == "$image"
    choices = root.find("Choices")
    assert choices is not None
    assert choices.get("name") == "shape"
    assert choices.get("toName") == "image"
    assert choices.get("choice") == "single"
    assert choices.get("required") == "true"
    assert [choice.get("value") for choice in choices] == list(SHAPES)
    assert [choice.get("hotkey") for choice in choices] == ["1", "2", "3", "4", "5", "6"]
    assert hold_shapes.default_config_path().name == "hold_shapes_config.xml"


def test_every_config_variable_is_one_identifier_present_in_every_task(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """Label Studio 1.23 reads a ``value`` as ONE variable, name and all.

    ``value="$clip_id / hold $hold_id / ..."`` asks for a task key literally
    named ``clip_id / hold $hold_id / ...``, so every import fails with
    ``key is expected in task data``. Each ``$`` in the generated config must
    therefore be a lone ``$identifier``, and that identifier must be a key of
    the ``data`` of every task the pre-label wrote.
    """
    data, videos = workspace
    report = run_prelabel(data, videos, tmp_path)

    root = ET.parse(report.config_path).getroot()
    tasks = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert tasks
    values = [value for element in root.iter() if (value := element.get("value")) and "$" in value]
    assert values, "the config must reference task data"
    for value in values:
        assert re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", value), (
            "not a single identifier: Label Studio reads the whole value as one "
            f"variable name, so it must be $identifier, got {value!r}"
        )
        for task in tasks:
            assert value[1:] in task["data"], f"{value} is not a key of {task['data']}"


def test_default_image_uri_is_a_local_files_url(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """The task image defaults to the local-files URL, ``--image-base`` overrides."""
    data, videos = workspace
    report = run_prelabel(data, videos, tmp_path)

    tasks = json.loads(report.json_path.read_text(encoding="utf-8"))
    for task in tasks:
        assert task["data"]["image"].startswith("/data/local-files/?d=label_frames_holds/")
    # The path is relative to the data dir (LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT).
    assert tasks[0]["data"]["image"] == (
        f"/data/local-files/?d={hold_shapes.FRAMES_DIRNAME}/{CLIP}_h0.jpg"
    )

    # --image-base is the override: the file name is prefixed, as before.
    local = data / hold_shapes.FRAMES_DIRNAME / f"{CLIP}_h0.jpg"
    assert hold_shapes.task_image_uri(local, "http://localhost:8080/frames/", data) == (
        f"http://localhost:8080/frames/{CLIP}_h0.jpg"
    )
    assert hold_shapes.task_image_uri(local, None, data).startswith(hold_shapes.LOCAL_FILES_URL)


def test_prelabel_skips_a_clip_whose_video_is_missing(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    videos.mkdir()
    write_hold_summary(
        data / "features" / "mediapipe" / "hold_summary.csv",
        [{**LINE_HOLD, "clip_id": MISSING_CLIP}],
    )
    write_catalogue(
        data / "catalogue.csv",
        [{"clip_id": MISSING_CLIP, "filename": "gone.mp4"}],
    )

    code = hold_shapes.main(
        [
            "prelabel",
            "--data",
            str(data),
            "--videos",
            str(videos),
            "--config",
            str(tmp_path / "config.xml"),
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "skipped" in out
    assert MISSING_CLIP in out
    assert "written per predicted shape: line=0" in out
    # Nothing was written for the clip, and the run still produced every output.
    assert list((data / "label_frames_holds").glob("*.jpg")) == []
    rows = read_manifest(data / "label_frames_holds" / "manifest.csv")
    assert rows == []
    tasks = json.loads((data / "label_studio_hold_shapes.json").read_text(encoding="utf-8"))
    assert tasks == []
    assert (tmp_path / "config.xml").is_file()


def test_prelabel_reports_a_missing_hold_summary(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="handstand.features"):
        prelabel(data=tmp_path, videos=tmp_path, config_path=tmp_path / "config.xml")


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #


def annotation(shape: str, labeled_at: str) -> dict[str, object]:
    """One Label Studio annotation that chose ``shape``."""
    return {
        "result": [
            {
                "type": "choices",
                "from_name": "shape",
                "to_name": "image",
                "value": {"choices": [shape]},
            }
        ],
        "updated_at": labeled_at,
    }


def test_import_takes_the_last_annotation_and_skips_unannotated_tasks(
    tmp_path: pathlib.Path,
) -> None:
    data = tmp_path / "data"
    manifest = data / "label_frames_holds" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "image": f"{OTHER_CLIP}_h1.jpg",
                "clip_id": OTHER_CLIP,
                "hold_id": 1,
                "frame_idx": 5,
                "t_ms": 167,
                "hold_duration_s": 0.2,
                "predicted_shape": "tuck",
                "reason": "knees 100°",
            }
        )
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            [
                {
                    "data": {"clip_id": CLIP, "hold_id": 0},
                    "predictions": [
                        {
                            "result": [
                                {
                                    "type": "choices",
                                    "from_name": "shape",
                                    "to_name": "image",
                                    "value": {"choices": ["line"]},
                                }
                            ]
                        }
                    ],
                    "annotations": [
                        annotation("line", "2026-10-01T10:00:00Z"),
                        annotation("tuck", "2026-10-01T11:00:00Z"),
                    ],
                },
                # No predictions on the task: the prediction comes from the manifest.
                {
                    "data": {"clip_id": OTHER_CLIP, "hold_id": 1},
                    "annotations": [annotation("line", "2026-10-01T12:00:00Z")],
                },
                # Never annotated: skipped rather than imported as blank.
                {"data": {"clip_id": MISSING_CLIP, "hold_id": 0}, "annotations": []},
            ]
        ),
        encoding="utf-8",
    )

    report = import_export(export, data=data)

    assert (report.tasks, report.annotated, report.skipped) == (3, 2, 1)
    rows = read_label_rows(report.out_path)
    assert len(rows) == 2
    by_key = {(row["clip_id"], row["hold_id"]): row for row in rows}
    # The last annotation wins, and it disagrees with the pre-label it replaced.
    first = by_key[(CLIP, 0)]
    assert first["shape"] == "tuck"
    assert first["predicted_shape"] == "line"
    assert first["agreed"] == "false"
    assert first["labeled_at"] == "2026-10-01T11:00:00Z"
    second = by_key[(OTHER_CLIP, 1)]
    assert second["shape"] == "line"
    assert second["predicted_shape"] == "tuck"
    assert second["agreed"] == "false"

    # Re-importing the same export changes nothing.
    again = import_export(export, data=data)
    assert (again.added, again.updated, again.unchanged, again.rows) == (0, 0, 2, 2)


def test_import_marks_a_confirmed_prediction_as_agreed(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            [
                {
                    "data": {"clip_id": CLIP, "hold_id": 0},
                    "predictions": [
                        {
                            "result": [
                                {
                                    "type": "choices",
                                    "from_name": "shape",
                                    "to_name": "image",
                                    "value": {"choices": ["line"]},
                                }
                            ]
                        }
                    ],
                    "annotations": [annotation("line", "2026-10-01T10:00:00Z")],
                }
            ]
        ),
        encoding="utf-8",
    )
    report = import_export(export, data=data)
    row = read_label_rows(report.out_path)[0]
    assert row["agreed"] == "true"


# --------------------------------------------------------------------------- #
# summary, the catalogue and the library helpers
# --------------------------------------------------------------------------- #


#: Six reviewed holds over three clips: one uniform clip and two mixed ones.
LABEL_ROWS: list[dict[str, object]] = [
    {"clip_id": CLIP, "hold_id": 0, "shape": "line", "predicted_shape": "line"},
    {"clip_id": CLIP, "hold_id": 1, "shape": "line", "predicted_shape": "straddle"},
    {"clip_id": OTHER_CLIP, "hold_id": 0, "shape": "tuck", "predicted_shape": "tuck"},
    {"clip_id": OTHER_CLIP, "hold_id": 1, "shape": "pike", "predicted_shape": UNMEASURED},
    {"clip_id": MISSING_CLIP, "hold_id": 0, "shape": "split_stag", "predicted_shape": ""},
    {"clip_id": MISSING_CLIP, "hold_id": 1, "shape": "other", "predicted_shape": "other"},
]


def test_summary_counts_shapes_skills_and_prediction_accuracy(tmp_path: pathlib.Path) -> None:
    path = write_labels(tmp_path / "hold_shapes.csv", LABEL_ROWS)
    summary = hold_shapes.build_summary(read_label_rows(path))

    assert summary.total == 6
    assert summary.counts == {
        "line": 2,
        "straddle": 0,
        "split_stag": 1,
        "tuck": 1,
        "pike": 1,
        "other": 1,
    }
    assert summary.clip_skills == {CLIP: "line", OTHER_CLIP: "mixed", MISSING_CLIP: "mixed"}
    assert (summary.predicted, summary.agreed) == (4, 3)
    assert (summary.unmeasured, summary.unrecorded) == (1, 1)
    assert summary.agreement_percent == pytest.approx(75.0)

    text = hold_shapes.summarise_shapes(summary)
    assert "line=2" in text
    assert "mixed=2" in text
    assert "75.0 %" in text
    # The two mixed clips are named (they are the exception); the uniform one
    # is only counted, not listed.
    assert f"{OTHER_CLIP}, {MISSING_CLIP}" in text
    assert CLIP not in text


def test_main_summary_writes_only_the_catalogue_skill_column(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    write_labels(data / "labels" / "hold_shapes.csv", LABEL_ROWS)
    catalogue = data / "catalogue.csv"
    write_catalogue(
        catalogue,
        [
            {"clip_id": CLIP, "filename": "one.mp4", "skill": ""},
            {
                "clip_id": OTHER_CLIP,
                "filename": "two.mp4",
                "skill": "line",
                "outcome": "hold",
                "hold_s": "5.7",
                "notes": "trainer",
                "good_clip": "1",
            },
            {"clip_id": MISSING_CLIP, "filename": "three.mp4", "skill": "", "camera_angle": "side"},
        ],
    )
    before = {row["clip_id"]: dict(row) for row in catalogue_module.read_rows(catalogue)}

    assert hold_shapes.main(["summary", "--data", str(data), "--write-catalogue"]) == 0
    out = capsys.readouterr().out
    assert "hold shapes: 6 labelled hold(s) over 3 clip(s)" in out
    assert "wrote the skill of 3 clip(s)" in out

    after = {row["clip_id"]: dict(row) for row in catalogue_module.read_rows(catalogue)}
    assert after[CLIP]["skill"] == "line"
    assert after[OTHER_CLIP]["skill"] == "mixed"  # replaced the hand-typed "line"
    assert after[MISSING_CLIP]["skill"] == "mixed"
    for clip_id, row in after.items():
        for column, value in row.items():
            if column == "skill":
                continue
            assert value == before[clip_id][column], f"{clip_id}.{column} changed"


def test_line_helpers_refuse_an_unlabelled_hold(tmp_path: pathlib.Path) -> None:
    # Nothing reviewed yet: no hold may count as a line hold.
    assert load_hold_shapes(tmp_path) == {}
    assert is_line_hold(CLIP, 0, {}) is False

    write_labels(
        tmp_path / "labels" / "hold_shapes.csv",
        [
            {"clip_id": CLIP, "hold_id": 0, "shape": "line", "predicted_shape": "line"},
            {"clip_id": CLIP, "hold_id": 1, "shape": "tuck", "predicted_shape": "tuck"},
            {
                "clip_id": OTHER_CLIP,
                "hold_id": 0,
                "shape": "straddle",
                "predicted_shape": "straddle",
            },
        ],
    )
    shapes = load_hold_shapes(tmp_path)
    assert shapes == {(CLIP, 0): "line", (CLIP, 1): "tuck", (OTHER_CLIP, 0): "straddle"}
    assert is_line_hold(CLIP, 0, shapes) is True
    # A hold that exists but is not labelled, a hold of another clip, a clip
    # nobody has seen, and a label that is not a line: all False.
    assert is_line_hold(CLIP, 7, shapes) is False
    assert is_line_hold(MISSING_CLIP, 0, shapes) is False
    assert is_line_hold(OTHER_CLIP, 0, shapes) is False
    # The hold id arrives as text from a CSV read and as an int from pandas.
    assert is_line_hold(CLIP, "0", shapes) is True

    summary_path = tmp_path / "features" / "mediapipe" / "hold_summary.csv"
    write_hold_summary(
        summary_path,
        [LINE_HOLD, TUCK_HOLD, {**TUCK_HOLD, "clip_id": OTHER_CLIP, "hold_id": 0}],
    )
    rows = read_hold_summary(summary_path)
    selected = line_holds(rows, shapes)
    assert [(row["clip_id"], row["hold_id"]) for row in selected] == [(CLIP, 0)]
    # The whole summary read back is the input, and only 1 of 3 rows survives.
    assert len(rows) == 3
