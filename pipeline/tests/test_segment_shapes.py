"""Tests for :mod:`handstand.segment_shapes`.

Nothing here touches the real videos or data: the data directory is built in
``tmp_path`` from small CSV fixtures, and the clip is a generated MJPEG video
whose frames have distinct grey levels so a strip can be proved to hold the
right number of frames. Label Studio is replaced by a fake client, so no test
opens a network connection.
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

from handstand import hold_shapes
from handstand import segment_shapes as segment_shapes
from handstand.ingest import LabelStudioError

CLIP = "aaaaaaaaaaaa"
OTHER_CLIP = "bbbbbbbbbbbb"
VIDEO_NAME = "generated clip.avi"
WIDTH = 64
HEIGHT = 96
FPS = 30
FRAMES = 80
FRAME_STEP = 3

SEGMENT_COLUMNS = (
    "clip_id",
    "source",
    "hold_id",
    "segment_id",
    "shape_guess",
    "mixed",
    "segment_frames",
    "valid_frames",
    "start_ms",
    "end_ms",
    "duration_s",
)

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


def write_video(path: pathlib.Path) -> pathlib.Path:
    """A tiny MJPEG clip: frame ``i`` is a flat grey of ``i * FRAME_STEP``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), f"cv2.VideoWriter could not open {path}"
    for index in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), index * FRAME_STEP, np.uint8))
    writer.release()
    return path


def write_segments(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """A minimal ``hold_segments.csv``; every other column is blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SEGMENT_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            full = {column: "" for column in SEGMENT_COLUMNS}
            full.update(row)
            writer.writerow(full)
    return path


def write_hold_summary(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HOLD_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            full = {column: "" for column in HOLD_COLUMNS}
            full.update(row)
            writer.writerow(full)
    return path


def write_hold_labels(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(hold_shapes.LABEL_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            full = {column: "" for column in hold_shapes.LABEL_COLUMNS}
            full.update(row)
            writer.writerow(full)
    return path


def write_segment_labels(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(segment_shapes.LABEL_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            full = {column: "" for column in segment_shapes.LABEL_COLUMNS}
            full.update(row)
            writer.writerow(full)
    return path


def write_catalogue(path: pathlib.Path, clip_id: str = CLIP) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "filename"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({"clip_id": clip_id, "filename": VIDEO_NAME})
    return path


def write_tasks(path: pathlib.Path, tasks: list[dict[str, object]]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


def annotation(shape: str, labeled_at: str) -> dict[str, object]:
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


def read_manifest(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    write_video(videos / VIDEO_NAME)
    write_catalogue(data / "catalogue.csv")
    write_segments(
        data / "features" / "mediapipe" / "hold_segments.csv",
        [
            {
                "clip_id": CLIP,
                "source": "mediapipe",
                "hold_id": 0,
                "segment_id": 0,
                "shape_guess": "line",
                "mixed": True,
                "start_ms": 0,
                "end_ms": 2000,
                "duration_s": 2.0,
            },
            {
                "clip_id": CLIP,
                "source": "mediapipe",
                "hold_id": 0,
                "segment_id": 1,
                "shape_guess": "unmeasured",
                "mixed": True,
                "start_ms": 2000,
                "end_ms": 2200,
                "duration_s": 0.2,
            },
            {
                "clip_id": CLIP,
                "source": "mediapipe",
                "hold_id": 1,
                "segment_id": 0,
                "shape_guess": "tuck",
                "mixed": False,
                "start_ms": 2200,
                "end_ms": 2600,
                "duration_s": 0.4,
            },
        ],
    )
    write_hold_labels(
        data / "labels" / "hold_shapes.csv",
        [{"clip_id": CLIP, "hold_id": 0, "shape": "straddle", "predicted_shape": "straddle"}],
    )
    return data, videos


def run_prelabel(
    data: pathlib.Path, videos: pathlib.Path, tmp_path: pathlib.Path, **kwargs: object
) -> segment_shapes.PrelabelReport:
    return segment_shapes.prelabel(
        data=data,
        videos=videos,
        config_path=tmp_path / "segment_shapes_config.xml",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


def test_only_mixed_holds_segments_get_tasks(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    report = run_prelabel(data, videos, tmp_path)

    assert report.written == 2
    assert report.segments == 2
    assert report.skipped == []
    keys = [row["segment_key"] for row in read_manifest(report.manifest)]
    assert keys == [f"{CLIP}_h0_s0", f"{CLIP}_h0_s1"]
    tasks = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert [task["data"]["segment_key"] for task in tasks] == keys


def test_task_carries_the_segment_data_and_header(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    report = run_prelabel(data, videos, tmp_path)
    tasks = json.loads(report.json_path.read_text(encoding="utf-8"))
    long_task, short_task = tasks

    assert set(long_task["data"]) == {
        "image",
        "clip_id",
        "hold_id",
        "segment_id",
        "segment_key",
        "start_ms",
        "end_ms",
        "duration_s",
        "hold_label",
        "info",
    }
    assert long_task["data"]["clip_id"] == CLIP
    assert long_task["data"]["hold_id"] == 0
    assert long_task["data"]["segment_id"] == 0
    assert long_task["data"]["start_ms"] == 0
    assert long_task["data"]["end_ms"] == 2000
    assert long_task["data"]["duration_s"] == 2.0
    assert long_task["data"]["hold_label"] == "straddle"
    assert long_task["data"]["info"] == (
        f"{CLIP} hold 0 segment 1/2: 0-2 s (2 s), hold labelled straddle"
    )
    assert long_task["data"]["image"] == (
        f"/data/local-files/?d={segment_shapes.FRAMES_DIRNAME}/{CLIP}_h0_s0.jpg"
    )
    # The shape guess is the preselected choice only when it is a real shape.
    result = long_task["predictions"][0]["result"][0]
    assert result["from_name"] == hold_shapes.CHOICE_NAME
    assert result["value"]["choices"] == ["line"]
    assert "predictions" not in short_task
    assert short_task["data"]["info"] == (
        f"{CLIP} hold 0 segment 2/2: 2-2.2 s (0.2 s), hold labelled straddle"
    )


def test_the_config_reuses_the_hold_shape_choices_and_hotkeys(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    report = run_prelabel(data, videos, tmp_path)

    root = ET.parse(report.config_path).getroot()
    info = root.find("Text")
    assert info is not None and info.get("value") == "$info"
    choices = root.find("Choices")
    assert choices is not None
    assert [choice.get("value") for choice in choices] == list(hold_shapes.SHAPES)
    assert [choice.get("hotkey") for choice in choices] == [
        hold_shapes.HOTKEYS[shape] for shape in hold_shapes.SHAPES
    ]
    # Every $variable in the config is a task field, as Label Studio 1.23 needs.
    tasks = json.loads(report.json_path.read_text(encoding="utf-8"))
    values = [value for element in root.iter() if (value := element.get("value")) and "$" in value]
    assert values
    for value in values:
        assert re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", value)
        for task in tasks:
            assert value[1:] in task["data"]


# --------------------------------------------------------------------------- #
# The strip image
# --------------------------------------------------------------------------- #


def test_a_long_segment_gets_three_frames_side_by_side(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    report = run_prelabel(data, videos, tmp_path)

    long_image = cv2.imread(str(report.frames_dir / f"{CLIP}_h0_s0.jpg"), cv2.IMREAD_GRAYSCALE)
    short_image = cv2.imread(str(report.frames_dir / f"{CLIP}_h0_s1.jpg"), cv2.IMREAD_GRAYSCALE)
    assert long_image is not None and short_image is not None

    # Three frames at the same height, the short segment's one middle frame.
    assert long_image.shape[0] == short_image.shape[0] <= segment_shapes.MAX_STRIP_HEIGHT
    assert long_image.shape[1] > 2.5 * short_image.shape[1]
    # The three thirds are different source frames (25 %, 50 %, 75 %), each a
    # flat grey: their middle rows cannot all be the same brightness.
    third = long_image.shape[1] // 3
    levels = [
        float(long_image[long_image.shape[0] // 2, index * third + third // 2])
        for index in range(3)
    ]
    assert max(levels) - min(levels) > 20


def test_strip_image_scales_to_one_height_and_caps_it() -> None:
    tall = np.full((800, 40, 3), 120, np.uint8)
    short = np.full((600, 20, 3), 200, np.uint8)

    strip = segment_shapes.strip_image([tall, short], [250, 750])

    assert strip.shape[0] == segment_shapes.MAX_STRIP_HEIGHT
    assert strip.shape[1] > 0


def test_segment_times_uses_the_middle_frame_when_short() -> None:
    assert segment_shapes.segment_times(0, 2000) == [500, 1000, 1500]
    assert segment_shapes.segment_times(1000, 1200) == [1100]


# --------------------------------------------------------------------------- #
# skip_existing
# --------------------------------------------------------------------------- #


def test_skip_existing_adds_only_new_segment_keys(
    workspace: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    data, videos = workspace
    first = run_prelabel(data, videos, tmp_path, skip_existing=True)
    assert first.written == 2

    # A new segment appears in the table; the two queued ones stay untouched.
    write_segments(
        data / "features" / "mediapipe" / "hold_segments.csv",
        [
            {
                "clip_id": CLIP,
                "hold_id": 0,
                "segment_id": 0,
                "shape_guess": "line",
                "mixed": True,
                "start_ms": 0,
                "end_ms": 2000,
                "duration_s": 2.0,
            },
            {
                "clip_id": CLIP,
                "hold_id": 0,
                "segment_id": 1,
                "shape_guess": "unmeasured",
                "mixed": True,
                "start_ms": 2000,
                "end_ms": 2200,
                "duration_s": 0.2,
            },
            {
                "clip_id": CLIP,
                "hold_id": 0,
                "segment_id": 2,
                "shape_guess": "tuck",
                "mixed": True,
                "start_ms": 2200,
                "end_ms": 2500,
                "duration_s": 0.3,
            },
        ],
    )
    again = run_prelabel(data, videos, tmp_path, skip_existing=True)

    assert (again.written, again.already_done) == (1, 2)
    assert [row["segment_key"] for row in read_manifest(again.manifest)] == [
        f"{CLIP}_h0_s0",
        f"{CLIP}_h0_s1",
        f"{CLIP}_h0_s2",
    ]
    tasks = json.loads(again.json_path.read_text(encoding="utf-8"))
    assert [task["data"]["segment_key"] for task in tasks] == [
        f"{CLIP}_h0_s0",
        f"{CLIP}_h0_s1",
        f"{CLIP}_h0_s2",
    ]


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


def test_import_takes_the_last_annotation_and_writes_the_columns(
    tmp_path: pathlib.Path,
) -> None:
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            [
                {
                    "data": {
                        "clip_id": CLIP,
                        "hold_id": 0,
                        "segment_id": 0,
                        "segment_key": f"{CLIP}_h0_s0",
                        "start_ms": 0,
                        "end_ms": 2000,
                    },
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
                        annotation("line", "2026-11-02T10:00:00Z"),
                        annotation("tuck", "2026-11-02T11:00:00Z"),
                    ],
                },
                {
                    "data": {
                        "clip_id": CLIP,
                        "hold_id": 0,
                        "segment_id": 1,
                        "segment_key": f"{CLIP}_h0_s1",
                        "start_ms": 2000,
                        "end_ms": 2200,
                    },
                    "annotations": [annotation("line", "2026-11-02T12:00:00Z")],
                },
                {"data": {"clip_id": CLIP, "hold_id": 0}, "annotations": []},
            ]
        ),
        encoding="utf-8",
    )

    report = segment_shapes.import_export(export, data=tmp_path / "data")

    assert (report.tasks, report.annotated, report.skipped) == (3, 2, 1)
    with report.out_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(segment_shapes.LABEL_COLUMNS)
        raw = list(reader)
    assert [(row["hold_id"], row["segment_id"], row["shape"]) for row in raw] == [
        ("0", "0", "tuck"),
        ("0", "1", "line"),
    ]
    rows = segment_shapes.read_label_rows(report.out_path)
    assert rows[0]["predicted"] == "line"
    assert rows[0]["labeled_at"] == "2026-11-02T11:00:00Z"
    assert segment_shapes.load_segment_shapes(tmp_path / "data") == {
        (CLIP, 0, 0): "tuck",
        (CLIP, 0, 1): "line",
    }

    again = segment_shapes.import_export(export, data=tmp_path / "data")
    assert (again.added, again.updated, again.unchanged, again.rows) == (0, 0, 2, 2)


# --------------------------------------------------------------------------- #
# line_windows
# --------------------------------------------------------------------------- #


def line_window_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    data = tmp_path / "data"
    write_hold_summary(
        data / "features" / "mediapipe" / "hold_summary.csv",
        [
            {"clip_id": CLIP, "hold_id": 0, "hold_start_ms": 0, "hold_end_ms": 3000},
            {"clip_id": CLIP, "hold_id": 1, "hold_start_ms": 3000, "hold_end_ms": 7000},
            {"clip_id": CLIP, "hold_id": 2, "hold_start_ms": 7000, "hold_end_ms": 10000},
            {"clip_id": CLIP, "hold_id": 3, "hold_start_ms": 10000, "hold_end_ms": 12000},
        ],
    )
    write_hold_labels(
        data / "labels" / "hold_shapes.csv",
        [
            {"clip_id": CLIP, "hold_id": 0, "shape": "line"},
            {"clip_id": CLIP, "hold_id": 1, "shape": "line"},
            {"clip_id": CLIP, "hold_id": 2, "shape": "line"},
            {"clip_id": CLIP, "hold_id": 3, "shape": "tuck"},
        ],
    )
    write_segments(
        data / "features" / "mediapipe" / "hold_segments.csv",
        [
            {
                "clip_id": CLIP,
                "hold_id": 1,
                "segment_id": 0,
                "shape_guess": "line",
                "mixed": True,
                "start_ms": 3000,
                "end_ms": 5500,
                "duration_s": 2.5,
            },
            {
                "clip_id": CLIP,
                "hold_id": 1,
                "segment_id": 1,
                "shape_guess": "line",
                "mixed": True,
                "start_ms": 5500,
                "end_ms": 7000,
                "duration_s": 1.5,
            },
            {
                "clip_id": CLIP,
                "hold_id": 1,
                "segment_id": 2,
                "shape_guess": "straddle",
                "mixed": True,
                "start_ms": 6500,
                "end_ms": 7000,
                "duration_s": 0.5,
            },
        ],
    )
    # Hold 2 is missing from the segment table: it counts as not mixed.
    return data


def test_line_windows_uses_only_unmixed_holds_and_long_line_segments(
    tmp_path: pathlib.Path,
) -> None:
    data = line_window_fixture(tmp_path)
    write_segment_labels(
        data / "labels" / "segment_shapes.csv",
        [
            {
                "clip_id": CLIP,
                "hold_id": 1,
                "segment_id": 0,
                "start_ms": 3000,
                "end_ms": 5500,
                "shape": "line",
                "predicted": "line",
            },
            {
                "clip_id": CLIP,
                "hold_id": 1,
                "segment_id": 1,
                "start_ms": 5500,
                "end_ms": 7000,
                "shape": "line",
                "predicted": "line",
            },
            {
                "clip_id": CLIP,
                "hold_id": 1,
                "segment_id": 2,
                "start_ms": 6500,
                "end_ms": 7000,
                "shape": "straddle",
                "predicted": "straddle",
            },
        ],
    )

    windows = segment_shapes.line_windows(data)

    assert [
        (window["hold_id"], window["segment_id"], window["origin"], window["duration_s"])
        for window in windows
    ] == [
        (0, None, "hold", 3.0),
        (1, 0, "segment", 2.5),
        (2, None, "hold", 3.0),
    ]


def test_a_mixed_hold_labelled_line_without_segment_labels_is_absent(
    tmp_path: pathlib.Path,
) -> None:
    data = line_window_fixture(tmp_path)

    windows = segment_shapes.line_windows(data)

    assert [(window["hold_id"], window["origin"]) for window in windows] == [
        (0, "hold"),
        (2, "hold"),
    ]


def test_summary_counts_queued_reviewed_and_line_windows(tmp_path: pathlib.Path) -> None:
    data = line_window_fixture(tmp_path)
    write_tasks(
        segment_shapes.tasks_path(data),
        [
            {"data": {"segment_key": f"{CLIP}_h1_s0"}},
            {"data": {"segment_key": f"{CLIP}_h1_s1"}},
        ],
    )
    write_segment_labels(
        data / "labels" / "segment_shapes.csv",
        [
            {
                "clip_id": CLIP,
                "hold_id": 1,
                "segment_id": 0,
                "start_ms": 3000,
                "end_ms": 5500,
                "shape": "line",
                "predicted": "line",
            },
        ],
    )

    text = segment_shapes.summarise(data)

    assert "2 segment(s) queued, 1 reviewed" in text
    assert "line: 1 segment(s), 2.5s" in text
    assert "on 1 of 1 measured prediction(s) (100.0 %)" in text
    assert "line windows: 2 hold(s) + 1 segment(s) = 8.5s" in text


# --------------------------------------------------------------------------- #
# Push
# --------------------------------------------------------------------------- #


class FakeClient:
    """The slice of :class:`handstand.ingest.LabelStudioClient` push needs."""

    def __init__(
        self,
        project_id: int | None,
        existing: set[str],
        storages: list[dict[str, object]] | None = None,
    ) -> None:
        self.project_id = project_id
        self.existing = existing
        self.created: tuple[str, str] | None = None
        self.imported: list[dict[str, object]] = []
        self.key_field: str | None = None
        self.storages = list(storages or [])
        self.storage_calls: list[tuple[int, str, str]] = []
        self.storage_created: list[dict[str, object]] = []
        self.sync_calls: list[int] = []

    def find_project_id(self, title: str) -> int:
        if self.project_id is None:
            raise LabelStudioError(f"no Label Studio project titled {title!r}")
        return self.project_id

    def create_project(self, title: str, label_config: str) -> int:
        self.created = (title, label_config)
        self.project_id = 7
        return 7

    def ensure_local_storage(self, project_id: int, path: str, title: str) -> int:
        self.storage_calls.append((project_id, str(path), title))
        for storage in self.storages:
            if storage["path"] == str(path):
                return int(storage["id"])  # type: ignore[arg-type]
        storage = {
            "id": 11 + len(self.storages),
            "project": project_id,
            "path": str(path),
            "title": title,
        }
        self.storages.append(storage)
        self.storage_created.append(storage)
        return int(storage["id"])  # type: ignore[arg-type]

    def sync_storage(self, storage_id: int) -> None:
        """Never called by push: a sync would import every image as a task."""
        self.sync_calls.append(storage_id)

    def existing_keys(self, project_id: int, key_field: str) -> set[str]:
        assert project_id == self.project_id
        self.key_field = key_field
        return self.existing

    def import_tasks(self, project_id: int, tasks: list[dict[str, object]]) -> int:
        assert project_id == self.project_id
        self.imported.extend(tasks)
        return len(tasks)


def tasks_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    data = tmp_path / "data"
    write_tasks(
        segment_shapes.tasks_path(data),
        [
            {"data": {"segment_key": f"{CLIP}_h0_s0"}},
            {"data": {"segment_key": f"{CLIP}_h0_s1"}},
        ],
    )
    return data


def test_push_creates_the_project_and_skips_existing_keys(tmp_path: pathlib.Path) -> None:
    data = tasks_fixture(tmp_path)
    client = FakeClient(project_id=None, existing={f"{CLIP}_h0_s0"})

    report = segment_shapes.push(data=data, client=client)

    assert report.created is True
    assert report.project_id == 7
    assert (report.tasks, report.imported, report.skipped) == (2, 1, 1)
    assert client.created is not None
    title, label_config = client.created
    assert title == segment_shapes.PROJECT_TITLE
    assert 'value="line" hotkey="1"' in label_config
    assert client.key_field == "segment_key"
    assert [task["data"]["segment_key"] for task in client.imported] == [f"{CLIP}_h0_s1"]


def test_push_finds_an_existing_project_and_imports_nothing_new(tmp_path: pathlib.Path) -> None:
    data = tasks_fixture(tmp_path)
    client = FakeClient(
        project_id=3,
        existing={f"{CLIP}_h0_s0", f"{CLIP}_h0_s1"},
    )

    report = segment_shapes.push(data=data, client=client)

    assert report.created is False
    assert (report.project_id, report.imported, report.skipped) == (3, 0, 2)
    assert client.imported == []
    assert client.created is None


def test_push_creates_the_local_files_storage_once_and_never_syncs(
    tmp_path: pathlib.Path,
) -> None:
    data = tasks_fixture(tmp_path)
    client = FakeClient(project_id=None, existing=set())
    expected = segment_shapes.frames_dir(data).resolve()

    first = segment_shapes.push(data=data, client=client)

    assert first.created is True
    assert client.storage_calls == [(7, str(expected), "handstand-segment-shapes frames")]
    assert [(storage["project"], storage["path"]) for storage in client.storage_created] == [
        (7, str(expected))
    ]
    assert client.sync_calls == [], "a sync would import every image as a task"

    segment_shapes.push(data=data, client=client)

    assert client.storage_calls == [
        (7, str(expected), "handstand-segment-shapes frames"),
        (7, str(expected), "handstand-segment-shapes frames"),
    ]
    assert len(client.storage_created) == 1, "the rerun found the storage it created"
    assert client.sync_calls == []


def test_push_keeps_an_existing_storage_for_the_same_path(tmp_path: pathlib.Path) -> None:
    data = tasks_fixture(tmp_path)
    expected = segment_shapes.frames_dir(data).resolve()
    client = FakeClient(
        project_id=3,
        existing=set(),
        storages=[{"id": 5, "project": 3, "path": str(expected)}],
    )

    report = segment_shapes.push(data=data, client=client)

    assert report.project_id == 3
    assert client.storage_created == []
    assert client.storage_calls == [(3, str(expected), "handstand-segment-shapes frames")]
    assert client.sync_calls == []
