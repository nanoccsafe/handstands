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
    return handstand_pose(ankles_y=0.8, wrists_y=0.2, x=x)


def inverted_pose(x: float = 0.5) -> list[StubLandmark]:
    """Handstand pose: wrists below (greater y than) ankles in display coords."""
    return handstand_pose(ankles_y=0.1, wrists_y=0.9, x=x)


def handstand_pose(ankles_y: float, wrists_y: float, x: float = 0.5) -> list[StubLandmark]:
    """Pose with both wrists at ``wrists_y`` and both ankles at ``ankles_y``.

    Everything else sits at ``y = 0.5``, so only the wrist/ankle comparison
    decides :func:`pm.is_inverted`. ``wrists_y > ankles_y`` is a handstand.
    """
    return make_pose(
        x=x,
        y=0.5,
        left_wrist=wrists_y,
        right_wrist=wrists_y,
        left_ankle=ankles_y,
        right_ankle=ankles_y,
    )


def landmark_array(pose: list[StubLandmark]) -> np.ndarray:
    """``(33, 5)`` float array — the layout the runner builds from a detection."""
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in pose],
        dtype=np.float64,
    )


class StubLandmarker:
    """Fake ``PoseLandmarker``: pops the next scripted result per call.

    A scripted entry is either ``None`` (nobody detected) or a *list of people*,
    each one a list of 33 :class:`StubLandmark`. People come back in script
    order, which is what a real ``num_poses > 1`` run does too.
    """

    def __init__(
        self,
        script: list[list[list[StubLandmark]] | None],
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
        people = self._script[index] if index < len(self._script) else None
        if people is None:
            return SimpleNamespace(pose_landmarks=[])
        return SimpleNamespace(pose_landmarks=list(people))

    def close(self) -> None:
        self.closed = True


def stub_factory(
    script: list[list[list[StubLandmark]] | None],
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


def default_script() -> list[list[list[StubLandmark]] | None]:
    """One person per frame; frames 0-9 and 13-19 detect, 10-12 nobody at all."""
    script: list[list[list[StubLandmark]] | None] = []
    for index in range(FRAME_COUNT):
        if index in (10, 11, 12):
            script.append(None)
        else:
            script.append([make_pose(x=(index + 0.5) / FRAME_COUNT, y=(index + 0.5) / FRAME_COUNT)])
    return script


def people_script(
    *people_per_frame: list[list[StubLandmark]] | None,
) -> list[list[list[StubLandmark]] | None]:
    """Script exactly the given frames; short scripts answer ``None`` afterwards."""
    return list(people_per_frame)


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
    """Run a clip into ``<tmp_path>/keypoints/<root>`` and return (report, out dir).

    ``root`` is the sub-directory :func:`pm.output_dirname` would pick for
    ``num_poses``, so a multi-person run cannot land in the single-person root.
    """
    num_poses = int(kwargs.get("num_poses", pm.DEFAULT_NUM_POSES))  # type: ignore[arg-type]
    out_root = tmp_path / "keypoints" / pm.output_dirname(num_poses)
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
# A video that does not need ffmpeg: fixed frames, fixed timestamps
# --------------------------------------------------------------------------- #

STUB_VIDEO_WIDTH = 64
STUB_VIDEO_HEIGHT = 48
STUB_VIDEO_FRAMES = 4


class StubVideo:
    """Stand-in for :class:`pm.DisplayVideo` with fixed frames and timestamps.

    Decoding an ffmpeg clip makes the output depend on the ffmpeg build, which
    is no good for a byte-for-byte regression test: these frames are plain
    black images with a constant ``t_ms`` ladder.
    """

    def __init__(self, video_path: str | pathlib.Path) -> None:
        self.video_path = pathlib.Path(video_path)

    @property
    def display_width(self) -> int:
        return STUB_VIDEO_WIDTH

    @property
    def display_height(self) -> int:
        return STUB_VIDEO_HEIGHT

    def __iter__(self):
        for index in range(STUB_VIDEO_FRAMES):
            frame = np.zeros((STUB_VIDEO_HEIGHT, STUB_VIDEO_WIDTH, 3), dtype=np.uint8)
            frame[:, :, 0] = index
            yield pm.FramePacket(frame_idx=index, t_ms=100 * index, frame=frame)

    def close(self) -> None:
        pass

    def __enter__(self) -> StubVideo:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


@pytest.fixture
def stub_video(monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Make the runner read :class:`StubVideo` instead of a real video file."""
    monkeypatch.setattr(pm, "DisplayVideo", StubVideo)
    return pathlib.Path("synthetic.mp4")


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


def test_multi_schema_is_the_single_schema_plus_person_idx() -> None:
    assert pm.PARQUET_COLUMNS_MULTI == (*pm.PARQUET_COLUMNS, "person_idx")
    assert pm.PERSON_COLUMN == "person_idx"
    assert pm.NO_PERSON_IDX == -1
    assert pm.DEFAULT_NUM_POSES == 1
    assert pm.MAX_NUM_POSES == 5


def test_multi_person_runs_get_their_own_output_root() -> None:
    assert pm.output_dirname() == "mediapipe"
    assert pm.output_dirname(1) == "mediapipe"
    assert pm.output_dirname(2) == "mediapipe_multi"
    assert pm.output_dirname(5) == "mediapipe_multi"
    assert pm.SINGLE_OUTPUT_DIRNAME == "mediapipe"
    assert pm.MULTI_OUTPUT_DIRNAME == "mediapipe_multi"
    for invalid in (0, 6, -1):
        with pytest.raises(ValueError, match="num_poses"):
            pm.output_dirname(invalid)


def test_mean_wrist_y_is_the_average_of_both_wrists() -> None:
    pose = landmark_array(handstand_pose(ankles_y=0.1, wrists_y=0.9))
    assert pm.mean_wrist_y(pose) == pytest.approx(0.9)
    one_wrist = make_pose(y=0.5, left_wrist=0.4, right_wrist=0.6)
    assert pm.mean_wrist_y(landmark_array(one_wrist)) == pytest.approx(0.5)
    # A missing wrist cannot be ranked.
    missing = make_pose(y=0.5, left_wrist=0.4, right_wrist=float("nan"))
    assert np.isnan(pm.mean_wrist_y(landmark_array(missing)))
    with pytest.raises(ValueError, match="landmarks"):
        pm.mean_wrist_y(np.zeros((12, 2)))


def test_lowest_wrist_pose_picks_the_person_lowest_in_the_image() -> None:
    # Listed first, but with wrists higher up in the image (y=0.4) than the
    # second person's (y=0.8), so the second person is the one on their hands.
    standing = handstand_pose(ankles_y=0.9, wrists_y=0.4, x=0.2)
    on_hands = handstand_pose(ankles_y=0.1, wrists_y=0.8, x=0.8)
    picked = pm.lowest_wrist_pose(np.stack([landmark_array(standing), landmark_array(on_hands)]))
    assert picked is not None
    assert picked[:, 0].mean() == pytest.approx(0.8)

    # Order does not matter: the lowest wrists win whoever they belong to.
    picked = pm.lowest_wrist_pose(np.stack([landmark_array(on_hands), landmark_array(standing)]))
    assert picked is not None
    assert picked[:, 0].mean() == pytest.approx(0.8)

    # A lone person is always its own winner, so nothing changes for them.
    picked = pm.lowest_wrist_pose(np.stack([landmark_array(standing)]))
    assert picked is not None
    assert picked[:, 0].mean() == pytest.approx(0.2)


def test_lowest_wrist_pose_skips_people_without_visible_wrists() -> None:
    unknown = make_pose(x=0.2, y=0.5, left_wrist=0.9, right_wrist=0.9)
    broken = make_pose(x=0.8, y=0.5, left_wrist=float("nan"), right_wrist=float("nan"))
    picked = pm.lowest_wrist_pose(np.stack([landmark_array(broken), landmark_array(unknown)]))
    assert picked is not None
    assert picked[:, 0].mean() == pytest.approx(0.2)

    # Nobody rankable -> the first person, exactly as a single-person frame is judged.
    picked = pm.lowest_wrist_pose(np.stack([landmark_array(broken), landmark_array(broken)]))
    assert picked is not None
    assert picked[:, 0].mean() == pytest.approx(0.8)

    with pytest.raises(ValueError, match="poses"):
        pm.lowest_wrist_pose(np.zeros((0, 33, 2)))
    with pytest.raises(ValueError, match="poses"):
        pm.lowest_wrist_pose(np.zeros((2, 12, 2)))


def test_detect_pose_keeps_every_person_in_the_order_media_pipe_returned() -> None:
    """``(P, 33, 5)`` per frame: nobody is dropped or re-sorted."""
    people = [make_pose(x=0.2), make_pose(x=0.7), make_pose(x=0.4)]
    landmarker = SimpleNamespace(
        detect_for_video=lambda image, t_ms: SimpleNamespace(pose_landmarks=people)
    )
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    poses = pm._detect_pose(landmarker, frame, 0, pathlib.Path("clip.mp4"))
    assert poses.shape == (3, len(pm.JOINT_NAMES), 5)
    assert poses.dtype == np.float64
    assert np.allclose(poses[:, 0, 0], [0.2, 0.7, 0.4])  # x of the nose, per person
    assert np.allclose(poses[:, :, 3], 0.9)  # visibility survives

    empty = pm._detect_pose(
        SimpleNamespace(detect_for_video=lambda image, t_ms: SimpleNamespace(pose_landmarks=[])),
        frame,
        1,
        pathlib.Path("clip.mp4"),
    )
    assert empty.shape == (0, len(pm.JOINT_NAMES), 5)
    assert not np.isfinite(empty).any()

    with pytest.raises(RuntimeError, match="landmarks"):
        pm._detect_pose(
            SimpleNamespace(
                detect_for_video=lambda image, t_ms: SimpleNamespace(
                    pose_landmarks=[[StubLandmark(0, 0, 0, 0, 0)]]
                )
            ),
            frame,
            2,
            pathlib.Path("clip.mp4"),
        )


def test_people_histogram_counts_frames_per_number_of_people() -> None:
    assert pm.people_histogram([]) == {}
    assert pm.people_histogram([0, 1, 1, 3]) == {0: 1, 1: 2, 3: 1}


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
    script = [[make_pose(x=0.25, y=0.75)] for _ in range(FRAME_COUNT)]
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
    script: list[list[list[StubLandmark]] | None] = [
        [upright_pose()],  # frame 0: upright -> frame 1 stays upright
        [inverted_pose()],  # frame 1: inverted -> frame 2 gets rotated
        # frame 2: fed rotated, uniform y -> frame 3 upright
        [make_pose(x=0.25, y=0.75)],
    ]
    script += [[upright_pose()] for _ in range(FRAME_COUNT - 3)]
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
    script: list[list[list[StubLandmark]] | None] = [[inverted_pose()], None, None]
    script += [[make_pose()] for _ in range(FRAME_COUNT - 3)]
    factory, created, _ = stub_factory(script)
    report, out_dir = run(synthetic_video, tmp_path, factory, rotate_mode="auto")

    assert len(created) == 2
    assert report.detected_frames == FRAME_COUNT - 2
    table = pd.read_parquet(out_dir / "clip0000000001.parquet")
    rotated_by_frame = table.groupby("frame_idx")["rotated"].first().tolist()
    assert rotated_by_frame == [False, True, True, True] + [False] * (FRAME_COUNT - 4)


# --------------------------------------------------------------------------- #
# Runner: multi-person
# --------------------------------------------------------------------------- #


def test_multi_person_frame_gets_one_33_row_block_per_person(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Two people in one frame: 33 rows each, labelled 0 and 1."""
    athlete = make_pose(x=0.25, y=0.30)
    trainer = make_pose(x=0.75, y=0.70)
    script = [[athlete, trainer]] + [[upright_pose()] for _ in range(FRAME_COUNT - 1)]
    factory, created, _ = stub_factory(script)
    report, out_dir = run(synthetic_video, tmp_path, factory, num_poses=3)

    assert not report.skipped
    assert report.num_poses == 3
    assert report.frame_count == FRAME_COUNT
    assert report.detected_frames == FRAME_COUNT

    table = pd.read_parquet(report.parquet_path)
    assert list(table.columns) == list(pm.PARQUET_COLUMNS_MULTI)
    assert table["person_idx"].dtype == "int64"
    assert table["frame_idx"].dtype == "int64"
    assert table["rotated"].dtype == bool
    assert table["detected"].dtype == bool

    frame0 = table[table["frame_idx"] == 0]
    assert len(frame0) == 2 * len(pm.JOINT_NAMES)
    blocks = frame0.groupby("person_idx")
    assert sorted(blocks.groups) == [0, 1]
    for _, block in blocks:
        assert len(block) == len(pm.JOINT_NAMES)
        assert set(block["joint"]) == set(pm.JOINT_NAMES)
        assert block["detected"].all()
        assert not block[["x", "y", "z", "visibility", "presence"]].isna().any().any()

    # People keep MediaPipe's order: person 0 is the athlete we scripted first.
    assert np.allclose(blocks.get_group(0)["x"], (WIDTH - 1) * 0.25)
    assert np.allclose(blocks.get_group(0)["y"], (HEIGHT - 1) * 0.30)
    assert np.allclose(blocks.get_group(1)["x"], (WIDTH - 1) * 0.75)
    assert np.allclose(blocks.get_group(1)["y"], (HEIGHT - 1) * 0.70)

    # Every other frame has one person, so 33 rows and person_idx 0 only.
    others = table[table["frame_idx"] > 0]
    assert set(others["person_idx"]) == {0}
    assert others.groupby("frame_idx").size().tolist() == [len(pm.JOINT_NAMES)] * (FRAME_COUNT - 1)
    assert len(table) == len(pm.JOINT_NAMES) * (2 + FRAME_COUNT - 1)


def test_multi_person_frame_without_anyone_keeps_one_nan_block(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A frame with nobody in it is still there: 33 NaN rows, person_idx -1."""
    script = [
        [upright_pose(), upright_pose()],  # frame 0: two people
        None,  # frame 1: nobody at all
        [make_pose(x=0.2), make_pose(x=0.4), make_pose(x=0.6)],  # frame 2: three
    ]
    script += [[upright_pose()] for _ in range(FRAME_COUNT - 3)]
    factory, created, _ = stub_factory(script)
    report, out_dir = run(synthetic_video, tmp_path, factory, num_poses=3)

    table = pd.read_parquet(report.parquet_path)
    empty = table[table["frame_idx"] == 1]
    assert len(empty) == len(pm.JOINT_NAMES)
    assert set(empty["joint"]) == set(pm.JOINT_NAMES)
    assert (empty["person_idx"] == pm.NO_PERSON_IDX).all()
    assert not empty["detected"].any()
    assert empty[["x", "y", "z", "visibility", "presence"]].isna().all().all()
    # t_ms/rotated still describe the frame itself.
    assert empty["t_ms"].nunique() == 1

    # Three people in one frame: person_idx 0, 1 and 2, in script order.
    three = table[table["frame_idx"] == 2]
    assert sorted(three["person_idx"].unique()) == [0, 1, 2]
    assert three.groupby("person_idx").size().tolist() == [len(pm.JOINT_NAMES)] * 3
    assert np.allclose(
        [block["x"].mean() for _, block in three.groupby("person_idx")],
        [(WIDTH - 1) * 0.2, (WIDTH - 1) * 0.4, (WIDTH - 1) * 0.6],
    )

    assert report.frames_by_people == {2: 1, 0: 1, 3: 1, 1: FRAME_COUNT - 3}
    assert report.people_summary == "0:1 1:17 2:1 3:1"
    assert report.detected_frames == FRAME_COUNT - 1


def test_multi_person_run_writes_to_the_multi_root_with_num_poses_in_the_sidecar(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    factory, _, _ = stub_factory(default_script())
    report, out_dir = run(
        synthetic_video, tmp_path, factory, rotate_mode="auto", num_poses=2, model_name="stub.task"
    )
    assert out_dir == tmp_path / "keypoints" / "mediapipe_multi" / "auto"
    assert report.parquet_path == out_dir / "clip0000000001.parquet"
    assert report.json_path == out_dir / "clip0000000001.json"
    assert report.parquet_path.is_file()

    sidecar = json.loads(report.json_path.read_text())
    assert sidecar["num_poses"] == 2
    assert sidecar["frames_by_people"] == {"0": 3, "1": FRAME_COUNT - 3}
    assert sidecar["clip_id"] == "clip0000000001"
    assert sidecar["model"] == "stub.task"
    assert sidecar["rotate"] == "auto"
    assert sidecar["display_width"] == WIDTH
    assert sidecar["frame_count"] == FRAME_COUNT
    assert sidecar["detected_frame_count"] == FRAME_COUNT - 3

    # Nothing landed in the single-person root.
    assert not (tmp_path / "keypoints" / "mediapipe").exists()


def test_multi_person_run_leaves_the_single_person_output_untouched(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    single_factory, _, _ = stub_factory(default_script())
    single, single_dir = run(synthetic_video, tmp_path, single_factory)
    before = single.parquet_path.read_bytes()
    sidecar_before = single.json_path.read_text()

    multi_factory, _, _ = stub_factory(default_script())
    multi, multi_dir = run(synthetic_video, tmp_path, multi_factory, num_poses=2)
    assert single_dir != multi_dir
    assert single.parquet_path.read_bytes() == before
    assert single.json_path.read_text() == sidecar_before
    assert multi.parquet_path.is_file()

    # And the two schemas agree wherever they overlap: one person per frame,
    # and the same -1 marker on the frames nobody was found in.
    single_table = pd.read_parquet(single.parquet_path)
    multi_table = pd.read_parquet(multi.parquet_path)
    assert list(single_table.columns) == list(pm.PARQUET_COLUMNS)
    assert list(multi_table.columns) == list(pm.PARQUET_COLUMNS_MULTI)
    assert set(multi_table["person_idx"].unique()) == {pm.NO_PERSON_IDX, 0}
    assert (multi_table.loc[multi_table["detected"], "person_idx"] == 0).all()
    assert (multi_table.loc[~multi_table["detected"], "person_idx"] == pm.NO_PERSON_IDX).all()
    assert multi_table.drop(columns=["person_idx"]).equals(single_table)


#: SHA-256 of the parquet the runner wrote for :func:`single_person_script`
#: *before* ``--num-poses`` existed (recorded at commit f8cbff1, so a change
#: here means the single-person output changed and that has to be deliberate).
#: Regenerate from a stub-video run of that script if the pandas/pyarrow
#: versions pinned in ``uv.lock`` ever change.
SINGLE_PARQUET_SHA256 = "94652ac49f143e0ded5c14739ade95e7c139006d1a2f4012be194a6e5a01a5e4"
#: Same, for the sidecar with ``runtime_seconds`` removed.
SINGLE_SIDECAR_SHA256 = "768ef63c668819e61f0aa55e272095b9876d9d05466c961d1c7cc5ba342b45b1"


def single_person_script() -> list[list[list[StubLandmark]] | None]:
    """One person per frame, frame 2 empty — the pre-``--num-poses`` behaviour."""
    return [
        [make_pose(0.25 + 0.1 * index, 0.75 - 0.05 * index)] if index != 2 else None
        for index in range(STUB_VIDEO_FRAMES)
    ]


def test_single_person_output_is_byte_identical(
    stub_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """``--num-poses 1`` must not move a single byte of the existing outputs."""
    factory, _, _ = stub_factory(single_person_script())
    report, out_dir = run(stub_video, tmp_path, factory, model_name="stub.task")

    assert out_dir == tmp_path / "keypoints" / "mediapipe" / "none"
    assert report.parquet_path == out_dir / "clip0000000001.parquet"
    assert report.num_poses == 1
    assert report.frames_by_people == {}
    assert report.people_summary == ""
    assert hashlib.sha256(report.parquet_path.read_bytes()).hexdigest() == SINGLE_PARQUET_SHA256

    sidecar = json.loads(report.json_path.read_text())
    assert "num_poses" not in sidecar
    assert "frames_by_people" not in sidecar
    sidecar.pop("runtime_seconds")
    sidecar_text = json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
    assert hashlib.sha256(sidecar_text.encode()).hexdigest() == SINGLE_SIDECAR_SHA256


def test_auto_mode_with_several_people_follows_the_lowest_wrist_person(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Rotation follows whoever has their hands lowest, not the first person.

    Frame 0 has two people. Person 0 is on their hands but high up in the
    frame (wrists at y=0.4), so person 1 — standing with arms hanging at
    y=0.9, i.e. lower in the image — wins the "lowest wrists" rule and is
    upright. Frame 1 must therefore NOT be rotated, even though the first
    person MediaPipe returned looks like a handstand.
    """
    on_hands = handstand_pose(ankles_y=0.05, wrists_y=0.4, x=0.3)  # person 0, listed first
    arms_down = handstand_pose(ankles_y=0.98, wrists_y=0.9, x=0.7)  # person 1, wrists lower
    assert pm.is_inverted(landmark_array(on_hands)) is True
    assert pm.is_inverted(landmark_array(arms_down)) is False
    assert pm.mean_wrist_y(landmark_array(arms_down)) > pm.mean_wrist_y(landmark_array(on_hands))

    script = people_script(
        [on_hands, arms_down],
        # Frame 1 is orientation-neutral, so it never asks for a rotation itself.
        [make_pose(), make_pose()],
    )
    script += [[upright_pose(), upright_pose()] for _ in range(FRAME_COUNT - 2)]
    factory, created, _ = stub_factory(script)
    report, out_dir = run(synthetic_video, tmp_path, factory, rotate_mode="auto", num_poses=3)

    table = pd.read_parquet(report.parquet_path)
    rotated_by_frame = table.groupby("frame_idx")["rotated"].first().tolist()
    assert rotated_by_frame == [False] * FRAME_COUNT
    assert report.rotated_frames == 0
    # Both people are still written out for frame 0.
    assert len(table[table["frame_idx"] == 0]) == 2 * len(pm.JOINT_NAMES)


def test_auto_mode_rotates_when_the_second_person_is_the_one_on_their_hands(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Same rule, other way round: the handstand person listed second wins."""
    arms_down = handstand_pose(ankles_y=0.95, wrists_y=0.4, x=0.3)  # person 0
    on_hands = handstand_pose(ankles_y=0.1, wrists_y=0.9, x=0.7)  # person 1, wrists lower
    script = people_script(
        [arms_down, on_hands],
        [make_pose(), make_pose()],  # neutral: whatever happens next, no own opinion
    )
    script += [[make_pose(), make_pose()] for _ in range(FRAME_COUNT - 2)]
    factory, created, _ = stub_factory(script)
    report, out_dir = run(synthetic_video, tmp_path, factory, rotate_mode="auto", num_poses=2)

    assert len(created) == 2  # upright tracker + rotated tracker, as in single-person auto
    table = pd.read_parquet(report.parquet_path)
    rotated_by_frame = table.groupby("frame_idx")["rotated"].first().tolist()
    assert rotated_by_frame == [False, True] + [False] * (FRAME_COUNT - 2)
    assert report.rotated_frames == 1
    # Frame 1 was fed rotated pixels; its keypoints are back in display pixels.
    frame1 = table[table["frame_idx"] == 1]
    assert np.allclose(frame1["x"], 0.5 * (WIDTH - 1))
    assert np.allclose(frame1["y"], 0.5 * (HEIGHT - 1))


def test_auto_mode_keeps_every_person_of_a_frame_that_nobody_was_found_in(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A lost frame keeps the previous decision and still writes its NaN block."""
    on_hands = handstand_pose(ankles_y=0.05, wrists_y=0.4, x=0.3)  # person 0
    arms_down = handstand_pose(ankles_y=0.98, wrists_y=0.9, x=0.7)  # person 1 wins, upright
    script = people_script(
        [on_hands, arms_down],
        None,  # nobody detected: the previous decision (upright) carries over
        None,
    )
    script += [[upright_pose(), upright_pose()] for _ in range(FRAME_COUNT - 3)]
    factory, _, _ = stub_factory(script)
    report, out_dir = run(synthetic_video, tmp_path, factory, rotate_mode="auto", num_poses=2)

    table = pd.read_parquet(report.parquet_path)
    assert not table[table["frame_idx"].isin([1, 2])]["detected"].any()
    assert (table[table["frame_idx"] == 1]["person_idx"] == pm.NO_PERSON_IDX).all()
    assert report.detected_frames == FRAME_COUNT - 2
    assert report.rotated_frames == 0


@pytest.mark.parametrize("num_poses", [0, 6, -1])
def test_run_clip_rejects_an_unsupported_number_of_poses(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path, num_poses: int
) -> None:
    factory, _, _ = stub_factory(default_script())
    with pytest.raises(ValueError, match="num_poses"):
        run(synthetic_video, tmp_path, factory, num_poses=num_poses)


def test_make_landmarker_factory_passes_num_poses_to_the_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag has to reach ``PoseLandmarkerOptions`` or the model stays at 1."""
    model = tmp_path / "pose.task"
    model.write_bytes(b"stub model")
    seen: dict[str, object] = {}

    class FakeOptions:
        def __init__(self, **kwargs: object) -> None:
            seen["options"] = kwargs

    class FakeLandmarker:
        @staticmethod
        def create_from_options(options: object) -> str:
            seen["created_with"] = options
            return "landmarker"

    monkeypatch.setattr(pm.mp_vision, "PoseLandmarkerOptions", FakeOptions)
    monkeypatch.setattr(pm.mp_vision, "PoseLandmarker", FakeLandmarker)

    factory = pm.make_landmarker_factory(model, 3)
    assert factory() == "landmarker"
    options = seen["options"]
    assert isinstance(options, dict)
    assert options["num_poses"] == 3
    assert options["running_mode"] == pm.mp_vision.RunningMode.VIDEO

    pm.make_landmarker_factory(model)()  # default is still one person
    assert isinstance(seen["options"], dict)
    assert seen["options"]["num_poses"] == 1

    for invalid in (0, 6):
        with pytest.raises(ValueError, match="num_poses"):
            pm.make_landmarker_factory(model, invalid)


def test_make_landmarker_factory_reports_a_missing_model_when_used(tmp_path: pathlib.Path) -> None:
    """Building the factory is cheap; only using it needs the model on disk."""
    missing = tmp_path / "absent.task"
    factory = pm.make_landmarker_factory(missing, 2)  # building it is fine
    with pytest.raises(FileNotFoundError, match="pose model not found"):
        factory()


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
    assert args.num_poses == pm.DEFAULT_NUM_POSES == 1
    assert args.clips is None
    assert args.limit is None
    assert args.overwrite is False
    assert args.model == pm.DEFAULT_MODEL_PATH

    args = parser.parse_args(
        [
            "--rotate",
            "auto",
            "--clips",
            "a.mp4",
            "b.mp4",
            "--limit",
            "3",
            "--overwrite",
            "--num-poses",
            "3",
        ]
    )
    assert args.rotate == "auto"
    assert args.clips == ["a.mp4", "b.mp4"]
    assert args.limit == 3
    assert args.overwrite is True
    assert args.num_poses == 3

    with pytest.raises(SystemExit):
        parser.parse_args(["--rotate", "sideways"])
    for out_of_range in ("0", "6", "-2"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--num-poses", out_of_range])


def test_cli_routes_each_num_poses_to_its_own_output_root(
    synthetic_video: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--num-poses 1`` keeps the old root, ``--num-poses > 1`` gets its own."""
    monkeypatch.setattr(pm, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(pm, "videos_dir", lambda: tmp_path)
    requested: list[int] = []

    def fake_factory(model_path: object, num_poses: int = 1) -> object:
        requested.append(num_poses)
        return stub_factory(default_script())[0]

    monkeypatch.setattr(pm, "make_landmarker_factory", fake_factory)
    clip_id = pm.sha1_clip_id(synthetic_video)
    common = ["--clips", str(synthetic_video), "--model", "stub.task"]

    assert pm.main(common) == 0
    assert requested == [1]
    single = tmp_path / "keypoints" / "mediapipe" / "none" / f"{clip_id}.parquet"
    assert single.is_file()
    assert list(pd.read_parquet(single).columns) == list(pm.PARQUET_COLUMNS)

    assert pm.main(["--num-poses", "2", *common]) == 0
    assert requested == [1, 2]
    multi = tmp_path / "keypoints" / "mediapipe_multi" / "none" / f"{clip_id}.parquet"
    assert multi.is_file()
    table = pd.read_parquet(multi)
    assert list(table.columns) == list(pm.PARQUET_COLUMNS_MULTI)
    assert json.loads(multi.with_suffix(".json").read_text())["num_poses"] == 2

    printed = capsys.readouterr().out
    assert "num_poses=2" in printed
    assert "people[0:3 1:17]" in printed


def test_collect_clips_defaults_to_every_mp4(tmp_path: pathlib.Path) -> None:
    for name in ("b.mp4", "a.mp4", "ignored.mov", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [path.name for path in pm.collect_clips(None, tmp_path)] == ["a.mp4", "b.mp4"]
    assert [path.name for path in pm.collect_clips([str(tmp_path / "ignored.mov")], tmp_path)] == [
        "ignored.mov"
    ]
    with pytest.raises(FileNotFoundError, match="missing video"):
        pm.collect_clips([str(tmp_path / "nope.mp4")], tmp_path)
