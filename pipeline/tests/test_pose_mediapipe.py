"""Tests for :mod:`handstand.pose_mediapipe`.

Nothing here touches the real handstand videos or the real ``.task`` model:
videos are 20-frame ``ffmpeg testsrc`` clips generated into ``tmp_path``, and
inference goes through an injected stub landmarker.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import pathlib
import re
import shutil
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from handstand import pose_mediapipe as pm
from handstand.rotation import inverse_rotate_points, rotate_points

FRAME_COUNT = 20
WIDTH = 128
HEIGHT = 96
FPS = 10


# --------------------------------------------------------------------------- #
# Fixtures and stubs
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A 20-frame 128x96 clip made from ffmpeg's testsrc (no rotation metadata)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path_factory.mktemp("videos") / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={WIDTH}x{HEIGHT}:rate={FPS}",
            "-frames:v",
            str(FRAME_COUNT),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@dataclasses.dataclass
class StubLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


def make_pose(x: float = 0.5, y: float = 0.5, **overrides: float) -> list[StubLandmark]:
    """33 stub landmarks; ``overrides`` sets y per joint name (e.g. left_wrist=0.9)."""
    return [
        StubLandmark(x=x, y=overrides.get(name, y), z=-0.1, visibility=0.9, presence=0.8)
        for name in pm.JOINT_NAMES
    ]


def upright_pose(x: float = 0.5) -> list[StubLandmark]:
    """Standing pose: wrists above (smaller y than) ankles in display coords."""
    return make_pose(x=x, y=0.5, left_wrist=0.2, right_wrist=0.2, left_ankle=0.8, right_ankle=0.8)


def inverted_pose(x: float = 0.5) -> list[StubLandmark]:
    """Handstand pose: wrists below (greater y than) ankles in display coords."""
    return make_pose(x=x, y=0.5, left_wrist=0.9, right_wrist=0.9, left_ankle=0.1, right_ankle=0.1)


def landmark_array(pose: list[StubLandmark]) -> np.ndarray:
    """``(33, 5)`` float array — the layout the runner builds from a detection."""
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in pose],
        dtype=np.float64,
    )


class StubLandmarker:
    """Fake ``PoseLandmarker``: pops the next scripted result per call."""

    def __init__(
        self,
        script: list[list[StubLandmark] | None],
        counter: itertools.count[int],
        fed_images: list[np.ndarray],
    ) -> None:
        self._script = script
        self._counter = counter
        self.fed_images = fed_images
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image: object, timestamp_ms: int) -> SimpleNamespace:
        self.timestamps.append(timestamp_ms)
        self.fed_images.append(image.numpy_view().copy())
        index = next(self._counter)
        pose = self._script[index] if index < len(self._script) else None
        if pose is None:
            return SimpleNamespace(pose_landmarks=[])
        return SimpleNamespace(pose_landmarks=[pose])

    def close(self) -> None:
        self.closed = True


def stub_factory(
    script: list[list[StubLandmark] | None],
) -> tuple[object, list[StubLandmarker], list[np.ndarray]]:
    """Return ``(factory, created_landmarkers, fed_images)`` for a scripted run."""
    counter = itertools.count()
    created: list[StubLandmarker] = []
    fed_images: list[np.ndarray] = []

    def factory() -> StubLandmarker:
        landmarker = StubLandmarker(script, counter, fed_images)
        created.append(landmarker)
        return landmarker

    return factory, created, fed_images


def default_script() -> list[list[StubLandmark] | None]:
    """Frames 0-9 and 13-19 detect; frames 10-12 have no pose at all."""
    script: list[list[StubLandmark] | None] = []
    for index in range(FRAME_COUNT):
        if index in (10, 11, 12):
            script.append(None)
        else:
            script.append(make_pose(x=(index + 0.5) / FRAME_COUNT, y=(index + 0.5) / FRAME_COUNT))
    return script


def raw_display_frame(video_path: pathlib.Path, index: int) -> np.ndarray:
    """Decode frame ``index`` with plain OpenCV auto-orientation (ground truth)."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame = None
        for _ in range(index + 1):
            ok, frame = capture.read()
            assert ok, f"could not read frame {index}"
        assert frame is not None
        return frame
    finally:
        capture.release()


def run(
    video_path: pathlib.Path,
    tmp_path: pathlib.Path,
    factory: object,
    rotate_mode: str = "none",
    clip_id: str = "clip0000000001",
    **kwargs: object,
) -> tuple[pm.ClipReport, pathlib.Path]:
    out_root = tmp_path / "keypoints" / "mediapipe"
    report = pm.run_clip(
        video_path,
        clip_id=clip_id,
        rotate_mode=rotate_mode,
        out_root=out_root,
        landmarker_factory=factory,
        **kwargs,  # type: ignore[arg-type]
    )
    return report, out_root / rotate_mode


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_joint_names_are_33_snake_case_landmarks() -> None:
    assert len(pm.JOINT_NAMES) == 33
    assert len(set(pm.JOINT_NAMES)) == 33
    assert pm.JOINT_NAMES[0] == "nose"
    assert pm.JOINT_NAMES[-1] == "right_foot_index"
    assert pm.JOINT_NAMES[pm.JOINT_INDEX["left_wrist"]] == "left_wrist"
    for name in pm.JOINT_NAMES:
        assert re.fullmatch(r"[a-z]+(_[a-z]+)*", name), name


def test_parquet_columns_match_the_documented_schema() -> None:
    assert pm.PARQUET_COLUMNS == (
        "frame_idx",
        "t_ms",
        "joint",
        "x",
        "y",
        "z",
        "visibility",
        "presence",
        "rotated",
        "detected",
    )


def test_normalized_to_pixels_uses_edge_pixel_indices() -> None:
    out = pm.normalized_to_pixels(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]), 100, 50)
    assert np.allclose(out, [[0.0, 0.0], [99.0, 49.0], [49.5, 24.5]])
    with pytest.raises(ValueError, match="shape"):
        pm.normalized_to_pixels(np.zeros((2, 3)), 100, 50)


def test_normalized_to_pixels_pairs_with_the_180_map_back() -> None:
    """A point and its mirrored normalised coords give the same display pixel."""
    display = pm.normalized_to_pixels(np.array([[0.25, 0.75]]), 100, 50)
    mirrored = pm.normalized_to_pixels(np.array([[0.75, 0.25]]), 100, 50)
    assert np.allclose(inverse_rotate_points(mirrored, 180, 100, 50), display)
    assert np.allclose(rotate_points(display, 180, 100, 50), mirrored)


@pytest.mark.parametrize(
    ("pose", "expected"),
    [
        (upright_pose(), False),
        (inverted_pose(), True),
    ],
)
def test_is_inverted_rule_on_synthetic_landmarks(pose: list[StubLandmark], expected: bool) -> None:
    assert pm.is_inverted(landmark_array(pose)) is expected


def test_is_inverted_uses_wrists_versus_ankles_only() -> None:
    # Nose position is irrelevant; only the wrist/ankle y's decide.
    pose = inverted_pose()
    pose[pm.JOINT_INDEX["nose"]] = dataclasses.replace(pose[pm.JOINT_INDEX["nose"]], y=0.0)
    assert pm.is_inverted(landmark_array(pose)) is True

    # Tied means are not "inverted": the rule is strictly greater.
    tied = make_pose(y=0.5, left_wrist=0.5, right_wrist=0.5, left_ankle=0.5, right_ankle=0.5)
    assert pm.is_inverted(landmark_array(tied)) is False

    # Wrists a pixel above the ankles is upright.
    barely = make_pose(y=0.5, left_wrist=0.49, right_wrist=0.49, left_ankle=0.5, right_ankle=0.5)
    assert pm.is_inverted(landmark_array(barely)) is False


def test_is_inverted_ignores_unusable_coordinates() -> None:
    pose = inverted_pose()
    pose[pm.JOINT_INDEX["left_wrist"]] = dataclasses.replace(
        pose[pm.JOINT_INDEX["left_wrist"]], y=float("nan")
    )
    assert pm.is_inverted(landmark_array(pose)) is False
    with pytest.raises(ValueError, match="landmarks"):
        pm.is_inverted(np.zeros((12, 2)))


def test_display_orientation_when_opencv_already_rotated() -> None:
    orientation = pm.display_orientation(576, 1024, 90, True)
    assert orientation == pm.DisplayOrientation(0, 576, 1024)
    assert orientation.display_shape == (1024, 576)


@pytest.mark.parametrize(
    ("meta", "expected"),
    [(0, (0, 1024, 576)), (90, (90, 576, 1024)), (-90, (270, 576, 1024)), (180, (180, 1024, 576))],
)
def test_display_orientation_from_rotation_metadata(meta: float, expected: tuple) -> None:
    orientation = pm.display_orientation(1024, 576, meta, False)
    assert (
        orientation.rotate_cw_deg,
        orientation.display_width,
        orientation.display_height,
    ) == expected


def test_display_orientation_rejects_impossible_metadata() -> None:
    with pytest.raises(ValueError, match="rotation metadata"):
        pm.display_orientation(1024, 576, 45, False)


def test_apply_display_rotation_matches_numpy() -> None:
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert pm.apply_display_rotation(frame, 0) is frame
    assert np.array_equal(pm.apply_display_rotation(frame, 90), np.rot90(frame, -1))
    assert np.array_equal(pm.apply_display_rotation(frame, 180), np.rot90(frame, 2))
    assert np.array_equal(pm.apply_display_rotation(frame, 270), np.rot90(frame, 1))
    with pytest.raises(ValueError, match="display rotation"):
        pm.apply_display_rotation(frame, 45)


def test_clip_id_is_first_12_hex_of_sha1(tmp_path: pathlib.Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"handstand")
    expected = hashlib.sha1(b"handstand").hexdigest()[:12]
    assert pm.resolve_clip_id(video) == expected
    assert len(expected) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", pm.resolve_clip_id(video))


def test_catalogue_supplies_the_clip_id(tmp_path: pathlib.Path) -> None:
    catalogue = tmp_path / "catalogue.csv"
    catalogue.write_text("clip_id,filename\nabc123def456,a.mp4\nfedcba654321,other/b.mp4\n")
    mapping = pm.load_catalogue(catalogue)
    assert mapping is not None
    assert pm.resolve_clip_id(tmp_path / "a.mp4", mapping) == "abc123def456"
    assert pm.resolve_clip_id(tmp_path / "b.mp4", mapping) == "fedcba654321"
    assert pm.resolve_clip_id(tmp_path / "sub" / "b.mp4", mapping) == "fedcba654321"
    # Unknown file falls back to the hash.
    unknown = tmp_path / "c.mp4"
    unknown.write_bytes(b"x")
    assert pm.resolve_clip_id(unknown, mapping) == hashlib.sha1(b"x").hexdigest()[:12]


def test_missing_or_broken_catalogue(tmp_path: pathlib.Path) -> None:
    assert pm.load_catalogue(tmp_path / "nope.csv") is None
    broken = tmp_path / "catalogue.csv"
    broken.write_text("name,id\na,1\n")
    with pytest.raises(ValueError, match="clip_id"):
        pm.load_catalogue(broken)


# --------------------------------------------------------------------------- #
# Runner: schema
# --------------------------------------------------------------------------- #


def test_none_mode_writes_the_documented_parquet_schema(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    factory, created, _ = stub_factory(default_script())
    report, out_dir = run(synthetic_video, tmp_path, factory, model_name="stub.task")

    parquet_path = out_dir / "clip0000000001.parquet"
    assert report.parquet_path == parquet_path
    assert parquet_path.is_file()
    assert report.frame_count == FRAME_COUNT
    assert report.detected_frames == FRAME_COUNT - 3
    assert report.rotated_frames == 0
    assert not report.skipped

    table = pd.read_parquet(parquet_path)
    assert list(table.columns) == list(pm.PARQUET_COLUMNS)

    # dtypes the Apple runner will match
    assert table["frame_idx"].dtype == "int64"
    assert table["t_ms"].dtype == "int64"
    assert pd.api.types.is_string_dtype(table["joint"])
    for column in ("x", "y", "z", "visibility", "presence"):
        assert table[column].dtype == "float64", column
    assert table["rotated"].dtype == bool
    assert table["detected"].dtype == bool

    # One row per (frame, joint): every frame carries all 33 joints exactly once.
    assert len(table) == FRAME_COUNT * len(pm.JOINT_NAMES)
    sizes = table.groupby("frame_idx").size()
    assert sizes.tolist() == [len(pm.JOINT_NAMES)] * FRAME_COUNT
    for _, group in table.groupby("frame_idx"):
        assert set(group["joint"]) == set(pm.JOINT_NAMES)

    # Timestamps come from POS_MSEC: ints, strictly increasing, starting at 0.
    stamps = table.groupby("frame_idx")["t_ms"].first().tolist()
    assert stamps[0] == 0
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))

    # Frames without a detection still exist, with NaN coordinates.
    missed = table[table["frame_idx"].isin([10, 11, 12])]
    assert not missed.empty
    assert not missed["detected"].any()
    assert missed[["x", "y", "z", "visibility", "presence"]].isna().all().all()
    found = table[table["detected"]]
    assert len(found) == (FRAME_COUNT - 3) * len(pm.JOINT_NAMES)
    assert not found[["x", "y", "z", "visibility", "presence"]].isna().any().any()

    # Coordinates are display-frame pixels: x_norm * (width - 1).
    for frame_idx in (0, 5, 19):
        normalised = (frame_idx + 0.5) / FRAME_COUNT
        rows = table[table["frame_idx"] == frame_idx]
        assert np.allclose(rows["x"], normalised * (WIDTH - 1))
        assert np.allclose(rows["y"], normalised * (HEIGHT - 1))
        assert rows["x"].between(0, WIDTH - 1).all()
        assert rows["y"].between(0, HEIGHT - 1).all()
    assert not table["rotated"].any()

    # Exactly one landmarker instance for --rotate none, and it was closed.
    assert len(created) == 1
    assert created[0].closed
    assert len(created[0].timestamps) == FRAME_COUNT

    # JSON sidecar next to the parquet.
    sidecar = json.loads(report.json_path.read_text())
    assert report.json_path == out_dir / "clip0000000001.json"
    assert sidecar["clip_id"] == "clip0000000001"
    assert sidecar["source_file"] == synthetic_video.name
    assert sidecar["model"] == "stub.task"
    assert sidecar["rotate"] == "none"
    assert sidecar["display_width"] == WIDTH
    assert sidecar["display_height"] == HEIGHT
    assert sidecar["frame_count"] == FRAME_COUNT
    assert sidecar["detected_frame_count"] == FRAME_COUNT - 3
    assert sidecar["rotated_frame_count"] == 0
    assert sidecar["mediapipe_version"] == pm.mediapipe_version()
    assert sidecar["runtime_seconds"] > 0


# --------------------------------------------------------------------------- #
# Runner: rotation modes
# --------------------------------------------------------------------------- #


def test_180_mode_rotates_every_frame_and_maps_back(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    script = [make_pose(x=0.25, y=0.75) for _ in range(FRAME_COUNT)]
    factory, created, fed_images = stub_factory(script)
    report, out_dir = run(synthetic_video, tmp_path, factory, rotate_mode="180")

    assert report.rotated_frames == FRAME_COUNT
    assert len(created) == 1  # a single tracker, consistently fed rotated frames

    table = pd.read_parquet(out_dir / "clip0000000001.parquet")
    assert table["rotated"].all()
    # Inverse of the 180° map-back: display = (W-1) - x_rotated.
    assert np.allclose(table["x"], (WIDTH - 1) * (1 - 0.25))
    assert np.allclose(table["y"], (HEIGHT - 1) * (1 - 0.75))

    # The model really saw 180°-rotated pixels (BGR frame rotated, then RGB).
    for frame_idx, fed in enumerate(fed_images):
        expected = cv2.cvtColor(
            cv2.rotate(raw_display_frame(synthetic_video, frame_idx), cv2.ROTATE_180),
            cv2.COLOR_BGR2RGB,
        )
        assert np.array_equal(fed, expected), f"frame {frame_idx} was not rotated"


def test_auto_mode_decides_from_the_previous_frame(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    script: list[list[StubLandmark] | None] = [
        upright_pose(),  # frame 0: upright -> frame 1 stays upright
        inverted_pose(),  # frame 1: inverted -> frame 2 gets rotated
        make_pose(x=0.25, y=0.75),  # frame 2: fed rotated, uniform y -> frame 3 upright
    ]
    script += [upright_pose() for _ in range(FRAME_COUNT - 3)]
    factory, created, fed_images = stub_factory(script)
    report, out_dir = run(synthetic_video, tmp_path, factory, rotate_mode="auto")

    # Two trackers: one that only sees upright frames, one that only sees rotated.
    assert len(created) == 2
    assert report.rotated_frames == 1
    assert report.frame_count == FRAME_COUNT
    assert all(marker.closed for marker in created)

    table = pd.read_parquet(out_dir / "clip0000000001.parquet")
    rotated_by_frame = table.groupby("frame_idx")["rotated"].first().tolist()
    expected = [False, False, True] + [False] * (FRAME_COUNT - 3)
    assert rotated_by_frame == expected
    # The first frame is never rotated.
    assert rotated_by_frame[0] is False

    # Frame 2 was fed rotated pixels and its keypoints are back in display pixels.
    expected_rotated = cv2.cvtColor(
        cv2.rotate(raw_display_frame(synthetic_video, 2), cv2.ROTATE_180), cv2.COLOR_BGR2RGB
    )
    assert np.array_equal(fed_images[2], expected_rotated)
    frame2 = table[table["frame_idx"] == 2]
    assert np.allclose(frame2["x"], (WIDTH - 1) * (1 - 0.25))
    assert np.allclose(frame2["y"], (HEIGHT - 1) * (1 - 0.75))
    frame0 = table[table["frame_idx"] == 0].set_index("joint")
    assert np.allclose(frame0["x"], (WIDTH - 1) * 0.5)
    assert frame0.loc["nose", "y"] == pytest.approx((HEIGHT - 1) * 0.5)
    assert frame0.loc["left_wrist", "y"] == pytest.approx((HEIGHT - 1) * 0.2)
    assert frame0.loc["left_ankle", "y"] == pytest.approx((HEIGHT - 1) * 0.8)

    # One detection per frame across both instances, strictly increasing ms.
    stamps = sorted(stamp for marker in created for stamp in marker.timestamps)
    assert len(stamps) == FRAME_COUNT
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))


def test_auto_mode_keeps_the_last_decision_when_detection_is_lost(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    # Frame 0 is inverted, frames 1-2 are lost, so frame 3 must still be rotated.
    # Frames 3+ are orientation-neutral (all joints share a y) so they never
    # request a rotation of their own, whatever frame they end up in.
    script: list[list[StubLandmark] | None] = [inverted_pose(), None, None]
    script += [make_pose() for _ in range(FRAME_COUNT - 3)]
    factory, created, _ = stub_factory(script)
    report, out_dir = run(synthetic_video, tmp_path, factory, rotate_mode="auto")

    assert len(created) == 2
    assert report.detected_frames == FRAME_COUNT - 2
    table = pd.read_parquet(out_dir / "clip0000000001.parquet")
    rotated_by_frame = table.groupby("frame_idx")["rotated"].first().tolist()
    assert rotated_by_frame == [False, True, True, True] + [False] * (FRAME_COUNT - 4)


# --------------------------------------------------------------------------- #
# Runner: bookkeeping
# --------------------------------------------------------------------------- #


def test_existing_output_is_skipped_unless_overwrite(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    factory, created, _ = stub_factory(default_script())
    report, _ = run(synthetic_video, tmp_path, factory)
    assert not report.skipped
    first_bytes = report.parquet_path.read_bytes()

    def explode() -> object:
        raise AssertionError("landmarker must not be created for a skipped clip")

    skipped, _ = run(synthetic_video, tmp_path, explode)
    assert skipped.skipped
    assert skipped.parquet_path == report.parquet_path
    assert report.parquet_path.read_bytes() == first_bytes

    rerun, _ = run(synthetic_video, tmp_path, factory, overwrite=True)
    assert not rerun.skipped
    assert len(created) == 2  # the second run really re-executed


def test_unknown_rotate_mode_is_rejected(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    factory, _, _ = stub_factory(default_script())
    with pytest.raises(ValueError, match="rotate mode"):
        run(synthetic_video, tmp_path, factory, rotate_mode="90")


def test_cli_parser_defaults_and_choices() -> None:
    parser = pm.build_arg_parser()
    args = parser.parse_args([])
    assert args.rotate == "none"
    assert args.clips is None
    assert args.limit is None
    assert args.overwrite is False
    assert args.model == pm.DEFAULT_MODEL_PATH

    args = parser.parse_args(
        ["--rotate", "auto", "--clips", "a.mp4", "b.mp4", "--limit", "3", "--overwrite"]
    )
    assert args.rotate == "auto"
    assert args.clips == ["a.mp4", "b.mp4"]
    assert args.limit == 3
    assert args.overwrite is True

    with pytest.raises(SystemExit):
        parser.parse_args(["--rotate", "sideways"])


def test_collect_clips_defaults_to_every_mp4(tmp_path: pathlib.Path) -> None:
    for name in ("b.mp4", "a.mp4", "ignored.mov", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [path.name for path in pm.collect_clips(None, tmp_path)] == ["a.mp4", "b.mp4"]
    assert [path.name for path in pm.collect_clips([str(tmp_path / "ignored.mov")], tmp_path)] == [
        "ignored.mov"
    ]
    with pytest.raises(FileNotFoundError, match="missing video"):
        pm.collect_clips([str(tmp_path / "nope.mp4")], tmp_path)
