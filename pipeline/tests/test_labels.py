"""Tests for :mod:`handstand.labels`.

Nothing here talks to Label Studio: the export is a small hand-written JSON file
in the shapes Label Studio actually writes, and the manifest is a couple of rows
of the sampler's own CSV. The point is the conversion — percentages to display
pixels, joint names to MediaPipe's, and an import that can be run twice.
"""

from __future__ import annotations

import csv
import json
import pathlib

import pytest

from handstand import labels
from handstand.frame_sampler import MANIFEST_COLUMNS
from handstand.labels import COLUMNS, KeypointRow, import_export, percent_to_pixels

#: A display size one pixel wider than 600/800, so ``(size - 1)`` is exact:
#: 50 % of 601 wide is 300.0 and 25 % of 801 high is 200.0.
WIDTH = 601
HEIGHT = 801
CLIP = "6508f9b355bd"
OTHER_CLIP = "64184de33f84"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def write_manifest(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """A sampler manifest, with every column the converter needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults: dict[str, object] = {
        "t_ms": 1000,
        "stratum": "clean_no_trainer",
        "skill": "line",
        "trainer_present": "false",
        "trainer_contact": "false",
        "display_width": WIDTH,
        "display_height": HEIGHT,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            merged = {**defaults, **row}
            writer.writerow(merged)
    return path


@pytest.fixture
def manifest(tmp_path: pathlib.Path) -> pathlib.Path:
    return write_manifest(
        tmp_path / "data" / "label_frames" / "manifest.csv",
        [
            {"image": f"{CLIP}_10.jpg", "clip_id": CLIP, "frame_idx": 10},
            {"image": f"{CLIP}_11.jpg", "clip_id": CLIP, "frame_idx": 11},
            {"image": f"{OTHER_CLIP}_3.jpg", "clip_id": OTHER_CLIP, "frame_idx": 3},
        ],
    )


def keypoint(label: str, x: float, y: float, **extra: object) -> dict[str, object]:
    """One ``keypoints`` result, in the shape Label Studio writes."""
    result = {
        "type": "keypoints",
        "from_name": "keypoints",
        "to_name": "image",
        "value": {"x": x, "y": y, "width": 1.0, "rotation": 0, "label": label},
    }
    result["value"].update(extra)  # type: ignore[union-attr]
    return result


def choice(value: str, from_name: str = "occluded_or_unsure") -> dict[str, object]:
    """One ``Choices`` result, in the shape Label Studio writes."""
    return {
        "type": "choices",
        "from_name": from_name,
        "to_name": "image",
        "value": {"choices": [value]},
    }


def task(image: str, results: list[dict[str, object]], **meta: object) -> dict[str, object]:
    """One exported task with a single annotation."""
    annotation = {
        "result": results,
        "completed_by": "labeler@example.com",
        "created_at": "2026-09-27T10:00:00Z",
        "updated_at": "2026-09-27T10:05:00Z",
    }
    annotation.update(meta)  # type: ignore[union-attr]
    return {"id": 1, "data": {"image": image}, "annotations": [annotation]}


def write_export(path: pathlib.Path, tasks: object) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return path


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rows_of(path: pathlib.Path) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(row["clip_id"], row["frame_idx"], row["joint"]): row for row in read_csv(path)}


# --------------------------------------------------------------------------- #
# Percentages to pixels
# --------------------------------------------------------------------------- #


def test_a_percentage_becomes_a_display_pixel() -> None:
    assert percent_to_pixels(50.0, 25.0, WIDTH, HEIGHT) == pytest.approx((300.0, 200.0))


def test_the_whole_image_spans_every_pixel() -> None:
    assert percent_to_pixels(0.0, 0.0, WIDTH, HEIGHT) == pytest.approx((0.0, 0.0))
    assert percent_to_pixels(100.0, 100.0, WIDTH, HEIGHT) == pytest.approx((WIDTH - 1, HEIGHT - 1))


def test_pixels_use_the_same_rule_as_the_keypoint_parquets() -> None:
    # The parquets were written with x = x_norm * (width - 1); a percentage is
    # x_norm * 100, so the two have to agree exactly.
    from handstand.pose_mediapipe import normalized_to_pixels

    x, y = percent_to_pixels(37.5, 62.5, 1080, 1920)
    assert (x, y) == pytest.approx(tuple(normalized_to_pixels([[0.375, 0.625]], 1080, 1920)[0]))


def test_an_unknown_display_size_is_refused() -> None:
    with pytest.raises(ValueError):
        percent_to_pixels(50.0, 50.0, 0, 0)


# --------------------------------------------------------------------------- #
# Joint names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("left_wrist", "left_wrist"),
        ("Left Wrist", "left_wrist"),
        ("left-ankle", "left_ankle"),
        ("Right Foot Index", "right_foot_index"),
        ("foot_index_left", "left_foot_index"),
        ("left_foot", "left_foot_index"),
        ("head", "nose"),
        ("NOSE", "nose"),
    ],
)
def test_joint_names_are_normalised(label: str, expected: str) -> None:
    assert labels.normalise_joint(label) == expected


@pytest.mark.parametrize("label", ["", "   ", "clear", "left_wrist_typo", None])
def test_a_label_that_is_not_a_joint_is_refused(label: object) -> None:
    assert labels.normalise_joint(label) is None


def test_every_joint_the_config_asks_for_is_a_mediapipe_joint() -> None:
    import pathlib as _pathlib

    from handstand.pose_mediapipe import JOINT_NAMES

    config = (
        _pathlib.Path(__file__).resolve().parents[2]
        / "tools"
        / "labeling"
        / "label_studio_config.xml"
    )
    text = config.read_text(encoding="utf-8")
    for joint in labels.LABEL_JOINTS:
        assert joint in JOINT_NAMES
        assert f'value="{joint}"' in text, f"{joint} is missing from {config.name}"


def test_keypoint_joints_are_label_tags_inside_keypointlabels() -> None:
    """Label Studio rejects <Choice> inside <KeyPointLabels> ("Not expecting tag: choice")."""
    import pathlib as _pathlib
    import xml.etree.ElementTree as ET

    config = (
        _pathlib.Path(__file__).resolve().parents[2]
        / "tools"
        / "labeling"
        / "label_studio_config.xml"
    )
    root = ET.fromstring(config.read_text(encoding="utf-8"))
    kp = root.find("KeyPointLabels")
    assert kp is not None
    children = list(kp)
    assert children and all(child.tag == "Label" for child in children)
    assert {child.get("value") for child in children} >= set(labels.LABEL_JOINTS)


def test_the_config_has_the_per_image_visibility_field_the_converter_reads() -> None:
    import pathlib as _pathlib

    config = (
        _pathlib.Path(__file__).resolve().parents[2]
        / "tools"
        / "labeling"
        / "label_studio_config.xml"
    )
    text = config.read_text(encoding="utf-8")
    assert f'name="{labels.UNCLEAR_FIELD}"' in text
    for choice in (*labels.UNCLEAR_CHOICES, "clear"):
        assert f'value="{choice}"' in text


# --------------------------------------------------------------------------- #
# Importing
# --------------------------------------------------------------------------- #


def test_a_keypoint_lands_in_display_pixels(tmp_path: pathlib.Path, manifest: pathlib.Path) -> None:
    export = write_export(
        tmp_path / "export.json",
        [task(f"{CLIP}_10.jpg", [keypoint("left_wrist", 50.0, 25.0)])],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    rows = rows_of(report.out_path)
    assert len(rows) == 1
    row = rows[(CLIP, "10", "left_wrist")]
    assert float(row["x"]) == pytest.approx(300.0)
    assert float(row["y"]) == pytest.approx(200.0)
    assert row["visible"] == "true"
    assert row["labeler"] == "labeler@example.com"
    assert row["labeled_at"] == "2026-09-27T10:05:00Z"
    assert report.added == 1


def test_the_csv_has_the_documented_header(tmp_path: pathlib.Path, manifest: pathlib.Path) -> None:
    export = write_export(tmp_path / "export.json", [task(f"{CLIP}_10.jpg", [])])
    report = import_export(export, manifest, data=tmp_path / "data")
    assert report.out_path == tmp_path / "data" / "labels" / "keypoints.csv"
    assert report.out_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(COLUMNS)


def test_a_joint_the_labeler_left_out_gets_no_row(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    export = write_export(
        tmp_path / "export.json",
        [
            task(
                f"{CLIP}_10.jpg",
                [keypoint("left_wrist", 50.0, 25.0), keypoint("right_ankle", 10.0, 90.0)],
            )
        ],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    joints = {row["joint"] for row in read_csv(report.out_path)}
    assert joints == {"left_wrist", "right_ankle"}
    # A missing point is the labeler's way of saying "not visible", and a missing
    # point is simply absent: there is no row to say so with.
    assert "left_ankle" not in joints


def test_the_new_1_23_export_shape_is_read(tmp_path: pathlib.Path, manifest: pathlib.Path) -> None:
    # Label Studio 1.23 stores the label as a plural list; 1.16 and earlier as a
    # single string. Both are in the wild and both must import.
    result = keypoint("ignored", 50.0, 25.0)
    result["value"] = {"x": 50.0, "y": 25.0, "keypointlabels": ["left_wrist"]}  # type: ignore[index]
    export = write_export(tmp_path / "export.json", [task(f"{CLIP}_10.jpg", [result])])
    report = import_export(export, manifest, data=tmp_path / "data")
    assert {row["joint"] for row in read_csv(report.out_path)} == {"left_wrist"}


def test_an_occluded_keypoint_keeps_its_coordinates_but_is_not_visible(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    export = write_export(
        tmp_path / "export.json",
        [
            task(
                f"{CLIP}_10.jpg",
                [
                    keypoint("left_wrist", 50.0, 25.0),
                    keypoint("left_hip", 50.0, 40.0, occluded=True),
                    keypoint("right_hip", 50.0, 40.0, visible=False),
                ],
            )
        ],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    rows = rows_of(report.out_path)
    assert rows[(CLIP, "10", "left_wrist")]["visible"] == "true"
    assert rows[(CLIP, "10", "left_hip")]["visible"] == "false"
    assert float(rows[(CLIP, "10", "left_hip")]["y"]) == pytest.approx(320.0)
    assert rows[(CLIP, "10", "right_hip")]["visible"] == "false"


def test_an_unusable_image_downgrades_all_its_keypoints(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    export = write_export(
        tmp_path / "export.json",
        [
            task(
                f"{CLIP}_10.jpg",
                [keypoint("left_wrist", 50.0, 25.0), choice("unsure")],
            )
        ],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    row = rows_of(report.out_path)[(CLIP, "10", "left_wrist")]
    assert row["visible"] == "false"
    assert float(row["x"]) == pytest.approx(300.0)


def test_a_clear_image_leaves_its_keypoints_visible(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    export = write_export(
        tmp_path / "export.json",
        [task(f"{CLIP}_10.jpg", [keypoint("left_wrist", 50.0, 25.0), choice("clear")])],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    assert rows_of(report.out_path)[(CLIP, "10", "left_wrist")]["visible"] == "true"


def test_an_image_label_studio_renamed_on_upload_is_still_found(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    # Uploading through the UI stores the file as "<hash>-<original name>".
    export = write_export(
        tmp_path / "export.json",
        [task(f"/data/upload/2/f7b8a3c1-{CLIP}_10.jpg", [keypoint("left_wrist", 50.0, 25.0)])],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    assert report.unknown_images == 0
    assert (CLIP, "10", "left_wrist") in rows_of(report.out_path)


def test_an_image_outside_the_manifest_is_skipped_and_counted(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    export = write_export(
        tmp_path / "export.json",
        [
            task(f"{CLIP}_10.jpg", [keypoint("left_wrist", 50.0, 25.0)]),
            task("999999999999_4.jpg", [keypoint("left_wrist", 50.0, 25.0)]),
        ],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    assert report.unknown_images == 1
    assert report.total == 1
    assert "not in" in report.summarise()


def test_an_unknown_joint_label_is_refused_not_written(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    export = write_export(
        tmp_path / "export.json",
        [
            task(
                f"{CLIP}_10.jpg",
                [keypoint("left_wrist", 50.0, 25.0), keypoint("eyebrow", 1.0, 2.0)],
            )
        ],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    assert report.unknown_joints == ("eyebrow",)
    assert {row["joint"] for row in read_csv(report.out_path)} == {"left_wrist"}


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def test_importing_the_same_export_twice_changes_nothing(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    # Percentages that do not land on a whole pixel, so a re-import compares a
    # float with the two decimals it was written as.
    export = write_export(
        tmp_path / "export.json",
        [task(f"{CLIP}_10.jpg", [keypoint("left_wrist", 33.3, 66.6), keypoint("nose", 10.0, 5.0)])],
    )
    first = import_export(export, manifest, data=tmp_path / "data")
    text = first.out_path.read_text(encoding="utf-8")
    assert "199.80" in text
    second = import_export(export, manifest, data=tmp_path / "data")
    assert second.out_path.read_text(encoding="utf-8") == text
    assert (second.added, second.updated, second.unchanged) == (0, 0, 2)
    assert len(read_csv(second.out_path)) == 2


def test_a_corrected_export_rewrites_only_that_key(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    first_export = write_export(
        tmp_path / "first.json",
        [task(f"{CLIP}_10.jpg", [keypoint("left_wrist", 50.0, 25.0), keypoint("nose", 10.0, 5.0)])],
    )
    first = import_export(first_export, manifest, data=tmp_path / "data")
    assert first.added == 2

    corrected = write_export(
        tmp_path / "corrected.json",
        [task(f"{CLIP}_10.jpg", [keypoint("left_wrist", 20.0, 30.0)])],
    )
    second = import_export(corrected, manifest, data=tmp_path / "data")
    assert (second.added, second.updated, second.unchanged) == (0, 1, 0)
    rows = rows_of(second.out_path)
    assert len(rows) == 2  # the nose row of the first import is still there
    assert float(rows[(CLIP, "10", "left_wrist")]["x"]) == pytest.approx(120.0)
    assert float(rows[(CLIP, "10", "left_wrist")]["y"]) == pytest.approx(240.0)


def test_an_export_with_a_tasks_object_is_read(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    export = write_export(
        tmp_path / "export.json",
        {"tasks": [task(f"{CLIP}_10.jpg", [keypoint("nose", 50.0, 25.0)])]},
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    assert report.total == 1


def test_two_labelers_on_one_image_leave_the_last_annotation(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    export = write_export(
        tmp_path / "export.json",
        [
            {
                "id": 1,
                "data": {"image": f"{CLIP}_10.jpg"},
                "annotations": [
                    {
                        "result": [keypoint("left_wrist", 50.0, 25.0)],
                        "completed_by": "first@example.com",
                        "updated_at": "2026-09-27T10:00:00Z",
                    },
                    {
                        "result": [keypoint("left_wrist", 10.0, 10.0)],
                        "completed_by": "second@example.com",
                        "updated_at": "2026-09-27T11:00:00Z",
                    },
                ],
            }
        ],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    row = rows_of(report.out_path)[(CLIP, "10", "left_wrist")]
    assert row["labeler"] == "second@example.com"
    assert float(row["x"]) == pytest.approx(60.0)
    assert report.conflicts == 1
    assert "more than one person" in report.summarise()


def test_a_task_with_no_annotation_writes_nothing(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    export = write_export(
        tmp_path / "export.json",
        [{"id": 1, "data": {"image": f"{CLIP}_10.jpg"}, "annotations": []}],
    )
    report = import_export(export, manifest, data=tmp_path / "data")
    assert report.total == 0
    assert report.out_path.read_text(encoding="utf-8").strip() == ",".join(COLUMNS)


def test_a_frame_in_the_manifest_of_another_clip_keeps_its_own_size(
    tmp_path: pathlib.Path, manifest: pathlib.Path
) -> None:
    small = write_manifest(
        tmp_path / "small.csv",
        [
            {
                "image": f"{OTHER_CLIP}_3.jpg",
                "clip_id": OTHER_CLIP,
                "frame_idx": 3,
                "display_width": 101,
                "display_height": 201,
            }
        ],
    )
    export = write_export(
        tmp_path / "export.json",
        [task(f"{OTHER_CLIP}_3.jpg", [keypoint("nose", 50.0, 50.0)])],
    )
    report = import_export(export, small, data=tmp_path / "data")
    row = rows_of(report.out_path)[(OTHER_CLIP, "3", "nose")]
    assert float(row["x"]) == pytest.approx(50.0)
    assert float(row["y"]) == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Reading a CSV written earlier
# --------------------------------------------------------------------------- #


def test_reading_a_missing_csv_is_an_empty_one(tmp_path: pathlib.Path) -> None:
    assert labels.read_rows(tmp_path / "nope.csv") == []


def test_a_damaged_row_is_dropped_rather_than_half_read(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "keypoints.csv"
    good = KeypointRow(CLIP, 10, "left_wrist", 1.0, 2.0, True, "a", "2026-09-27T10:00:00Z")
    bad = KeypointRow(CLIP, 11, "left_wrist", float("nan"), 2.0, True, "a", "")
    labels.write_rows([good, bad], path)
    assert labels.read_rows(path) == [good]


def test_a_csv_with_the_wrong_columns_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "wrong.csv"
    path.write_text("clip_id,frame_idx\nx,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        labels.read_rows(path)


def test_writing_twice_gives_the_same_bytes(tmp_path: pathlib.Path) -> None:
    rows = [
        KeypointRow(OTHER_CLIP, 3, "nose", 1.0, 2.0, True, "a", "t"),
        KeypointRow(CLIP, 11, "left_wrist", 3.0, 4.0, False, "a", "t"),
        KeypointRow(CLIP, 10, "left_wrist", 5.0, 6.0, True, "a", "t"),
    ]
    first = labels.write_rows(rows, tmp_path / "a.csv").read_bytes()
    second = labels.write_rows(list(reversed(rows)), tmp_path / "b.csv").read_bytes()
    assert first == second


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def test_main_imports_and_prints(
    tmp_path: pathlib.Path, manifest: pathlib.Path, capsys, monkeypatch
) -> None:
    export = write_export(
        tmp_path / "export.json", [task(f"{CLIP}_10.jpg", [keypoint("left_wrist", 50.0, 25.0)])]
    )
    monkeypatch.setenv("HANDSTAND_DATA", str(tmp_path / "data"))
    assert labels.main(["import", str(export)]) == 0
    out = capsys.readouterr().out
    assert "1 added" in out
    assert (tmp_path / "data" / "labels" / "keypoints.csv").is_file()


def test_main_reports_a_missing_export(tmp_path: pathlib.Path, capsys) -> None:
    assert labels.main(["import", str(tmp_path / "nope.json")]) == 2
    assert "labels:" in capsys.readouterr().err


def test_main_rejects_an_export_that_is_not_json(tmp_path: pathlib.Path, capsys) -> None:
    path = tmp_path / "export.json"
    path.write_text("{not json", encoding="utf-8")
    assert labels.main(["import", str(path)]) == 2
    assert "labels:" in capsys.readouterr().err
