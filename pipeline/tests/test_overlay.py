"""Tests for :mod:`handstand.overlay`.

Nothing here touches the real handstand videos or the real ``.task`` model: the
clip is a 20-frame ``ffmpeg testsrc`` video generated into ``tmp_path`` and the
keypoints are hand-built parquets in the documented schema.
"""

from __future__ import annotations

import math
import pathlib
import shutil
import subprocess
from collections.abc import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import pytest

from handstand import overlay, paths
from handstand.pose_mediapipe import JOINT_NAMES, resolve_clip_id

FRAME_COUNT = 20
WIDTH = 128
HEIGHT = 96
FPS = 10


# --------------------------------------------------------------------------- #
# Fixtures and helpers
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


@pytest.fixture(scope="module")
def flat_video(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A 20-frame 128x96 clip of flat grey, so codec noise stays tiny."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path_factory.mktemp("flat") / "flat.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s={WIDTH}x{HEIGHT}:r={FPS}",
            "-frames:v",
            str(FRAME_COUNT),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def blank_frame(height: int = 240, width: int = 320) -> np.ndarray:
    """A black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def pose(
    points: dict[str, tuple[float, float]],
    visibility: dict[str, float] | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """``(joints_xy, visibility)`` for a hand-built pose, NaN-scored elsewhere."""
    scores = {name: 0.95 for name in points}
    if visibility:
        scores.update(visibility)
    return dict(points), scores


def changed_pixels(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Boolean mask of the pixels that differ between two frames."""
    return np.any(before != after, axis=-1)


def joint_frame(
    joint: str,
    x: int,
    y: int,
    visibility: float,
    min_visibility: float = overlay.MIN_VISIBILITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one joint alone and return ``(input, output)`` frames."""
    frame = blank_frame()
    drawn = overlay.draw_pose(
        frame,
        {joint: (float(x), float(y))},
        {joint: visibility},
        min_visibility=min_visibility,
    )
    return frame, drawn


def keypoint_table(
    frame_indices: Iterable[int],
    poses: dict[int, dict[str, tuple[float, float]]] | None = None,
    visibility: float = 0.9,
) -> pd.DataFrame:
    """A long-format keypoint table for ``frame_indices``, schema-compatible.

    Frames missing from ``poses`` get their 33 rows of NaN and ``detected``
    false, exactly as the runner writes a frame the model missed.
    """
    poses = poses or {}
    rows = []
    for frame_idx in frame_indices:
        pose = poses.get(frame_idx)
        t_ms = int(frame_idx * 1000 / FPS)
        for name in JOINT_NAMES:
            if pose is None or name not in pose:
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "t_ms": t_ms,
                        "joint": name,
                        "x": math.nan,
                        "y": math.nan,
                        "z": math.nan,
                        "visibility": math.nan,
                        "presence": math.nan,
                        "rotated": False,
                        "detected": False,
                    }
                )
            else:
                x, y = pose[name]
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "t_ms": t_ms,
                        "joint": name,
                        "x": x,
                        "y": y,
                        "z": 0.0,
                        "visibility": visibility,
                        "presence": 0.8,
                        "rotated": False,
                        "detected": True,
                    }
                )
    return pd.DataFrame(rows)


def line_pose(offset: float = 0.0) -> dict[str, tuple[float, float]]:
    """A pose whose bones are a diagonal chain across the frame."""
    return {
        "left_ankle": (WIDTH * 0.3 + offset, HEIGHT * 0.8),
        "left_knee": (WIDTH * 0.4 + offset, HEIGHT * 0.6),
        "left_hip": (WIDTH * 0.5 + offset, HEIGHT * 0.4),
        "left_shoulder": (WIDTH * 0.6 + offset, HEIGHT * 0.25),
        "left_elbow": (WIDTH * 0.7 + offset, HEIGHT * 0.15),
        "left_wrist": (WIDTH * 0.8 + offset, HEIGHT * 0.08),
    }


def every_frame(offset: float = 0.0, first: int = 0) -> dict[int, dict[str, tuple[float, float]]]:
    """A :func:`line_pose` on every frame from ``first`` on."""
    return {index: line_pose(offset) for index in range(first, FRAME_COUNT)}


def first_frames(count: int, offset: float = 0.0) -> dict[int, dict[str, tuple[float, float]]]:
    """A :func:`line_pose` on the first ``count`` frames only."""
    return {index: line_pose(offset) for index in range(count)}


CLIP_ID = "abc123def456"


def write_panels(
    root: pathlib.Path,
    panels: Sequence[tuple[str, pd.DataFrame]],
    mode_root: bool = True,
    clip_id: str = CLIP_ID,
) -> list[pathlib.Path]:
    """Write ``(label, table)`` pairs as ``<root>/<label>/<clip_id>.parquet``."""
    paths = []
    for label, table in panels:
        directory = root / label if mode_root else root
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{clip_id}.parquet"
        table.to_parquet(path, index=False)
        paths.append(path)
    return paths


def decoded_frames(video_path: pathlib.Path) -> list[np.ndarray]:
    """Every frame of a clip, decoded plainly (ground truth for the output)."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                return frames
            frames.append(frame)
    finally:
        capture.release()


# --------------------------------------------------------------------------- #
# Bones and colours
# --------------------------------------------------------------------------- #


def test_bones_are_mediapipe_connections_named_by_joint() -> None:
    from mediapipe.tasks.python import vision as mp_vision

    expected = {
        (JOINT_NAMES[connection.start], JOINT_NAMES[connection.end])
        for connection in mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
    }
    assert set(overlay.BONES) == expected
    assert all(name in JOINT_NAMES for bone in overlay.BONES for name in bone)
    # A wrist hangs off an elbow, not the other way round.
    assert ("left_elbow", "left_wrist") in overlay.BONES


@pytest.mark.parametrize(
    ("visibility", "colour"),
    [
        (1.0, overlay.COLOR_GOOD),
        (overlay.VISIBILITY_GOOD, overlay.COLOR_GOOD),
        (0.79, overlay.COLOR_OK),
        (overlay.VISIBILITY_OK, overlay.COLOR_OK),
        (0.49, overlay.COLOR_LOW),
        (0.0, overlay.COLOR_LOW),
        (math.nan, overlay.COLOR_LOW),
    ],
)
def test_visibility_colour_thresholds(visibility: float, colour: tuple) -> None:
    assert overlay.visibility_colour(visibility) == colour


# --------------------------------------------------------------------------- #
# draw_pose
# --------------------------------------------------------------------------- #


def test_draw_pose_marks_the_joints_and_leaves_the_rest_alone() -> None:
    # A horizontal limb, so the bone lands on an exact row of pixels.
    joints, scores = pose({"left_knee": (100.0, 150.0), "left_hip": (200.0, 150.0)})
    frame = blank_frame()
    before = frame.copy()

    drawn = overlay.draw_pose(frame, joints, scores)

    assert drawn.shape == frame.shape
    assert np.array_equal(frame, before), "the input frame must not be modified"
    assert not np.shares_memory(drawn, frame)

    changed = changed_pixels(frame, drawn)
    assert changed.any(), "nothing was drawn"
    for name, (x, y) in joints.items():
        assert changed[int(round(y)), int(round(x))], f"{name} was not drawn"
    # The corners of a 240x320 frame are nowhere near the skeleton.
    assert not changed[0, 0]
    assert not changed[239, 319]

    # The bones are MediaPipe's connections: every pixel between knee and hip is
    # ink, and nothing is drawn outside that band.
    assert changed[150, 150], "the left_knee->left_hip bone is missing"
    assert not changed[140, 150]
    assert not changed[160, 150]
    assert not changed[150, 90]


def test_draw_pose_colours_by_visibility() -> None:
    x = y = 120
    for visibility, colour in (
        (0.95, overlay.COLOR_GOOD),
        (0.6, overlay.COLOR_OK),
    ):
        frame, drawn = joint_frame("left_wrist", x, y, visibility)
        assert tuple(int(channel) for channel in drawn[y, x]) == colour

    # Red needs min_visibility lowered, otherwise the joint is skipped entirely.
    frame, drawn = joint_frame("left_wrist", x, y, 0.2, min_visibility=0.0)
    assert tuple(int(channel) for channel in drawn[y, x]) == overlay.COLOR_LOW


def test_draw_pose_skips_nan_and_low_visibility_joints() -> None:
    frame = blank_frame()
    # "good" is far away from the two skipped joints, so their pixels cannot be
    # reached by a bone or a circle belonging to it.
    joints = {
        "left_wrist": (100.0, 100.0),
        "left_elbow": (math.nan, math.nan),  # never reported by the model
        "right_wrist": (200.0, 200.0),
    }
    scores = {"left_wrist": 0.9, "left_elbow": 0.9, "right_wrist": 0.1}  # below the floor

    drawn = overlay.draw_pose(frame, joints, scores)

    changed = changed_pixels(frame, drawn)
    assert changed[100, 100], "the confident joint is drawn"
    assert not changed[200, 200], "the low-visibility joint is skipped"
    # Nothing beyond that one disc: no half limbs reaching towards the skipped ones.
    assert changed[95:106, 95:106].sum() == changed.sum()
    assert not changed[150, 150]  # halfway between the wrist and the elbow
    assert np.array_equal(frame, blank_frame())


def test_draw_pose_skips_everything_when_nothing_is_usable() -> None:
    frame = blank_frame()
    joints = {"left_wrist": (50.0, 50.0), "right_wrist": (150.0, 150.0)}
    scores = {"left_wrist": math.nan, "right_wrist": 0.2}
    assert np.array_equal(overlay.draw_pose(frame, joints, scores), frame)


def test_draw_pose_only_draws_a_bone_when_both_ends_are_drawn() -> None:
    frame = blank_frame()
    joints = {"left_wrist": (60.0, 60.0), "left_elbow": (200.0, 200.0)}
    # The elbow is present but not confident: the bone must not appear.
    without = overlay.draw_pose(frame, joints, {"left_wrist": 0.9, "left_elbow": 0.4})
    changed = changed_pixels(frame, without)
    assert changed[60, 60]
    assert not changed[200, 200]
    assert not changed[130, 130]  # halfway along the missing bone

    # Drawing the elbow too puts the bone back.
    with_bone = overlay.draw_pose(frame, joints, {"left_wrist": 0.9, "left_elbow": 0.9})
    assert changed_pixels(frame, with_bone).sum() > changed.sum()


def test_draw_pose_scales_with_frame_height() -> None:
    scores = {"left_wrist": 0.9}
    counts = []
    for height in (96, 240, 720, 1440):
        frame = np.zeros((height, height, 3), dtype=np.uint8)
        drawn = overlay.draw_pose(frame, {"left_wrist": (height * 0.5, height * 0.5)}, scores)
        counts.append(int(changed_pixels(frame, drawn).sum()))
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]

    thickness, radius = overlay.draw_metrics(96)
    assert (thickness, radius) == (1, 2)
    thick, big = overlay.draw_metrics(1440)
    assert thick > thickness and big > radius


def test_draw_pose_skips_a_joint_with_no_visibility_score() -> None:
    """A missing score is NaN, and NaN is never confident enough to draw."""
    frame = blank_frame()
    drawn = overlay.draw_pose(frame, {"left_wrist": (40.0, 40.0)}, {}, min_visibility=0.0)
    assert not changed_pixels(frame, drawn).any()


# --------------------------------------------------------------------------- #
# draw_caption
# --------------------------------------------------------------------------- #


def test_draw_caption_writes_text_without_touching_the_input() -> None:
    frame = blank_frame()
    before = frame.copy()
    drawn = overlay.draw_caption(frame, ["auto", "frame_idx 7", "t_ms 733"])

    assert np.array_equal(frame, before)
    changed = changed_pixels(frame, drawn)
    assert changed.any()
    # The block sits in the top-left and is a fraction of the frame.
    assert changed[:HEIGHT, :WIDTH].any()
    assert not changed[HEIGHT // 2 :, WIDTH // 2 :].any()


def test_draw_caption_of_nothing_is_an_unchanged_copy() -> None:
    frame = blank_frame()
    drawn = overlay.draw_caption(frame, [])
    assert np.array_equal(drawn, frame)
    assert not np.shares_memory(drawn, frame)


def test_draw_caption_of_a_caption_twice_is_the_same() -> None:
    frame = blank_frame()
    lines = ["none", "frame_idx 0", "t_ms 0"]
    assert np.array_equal(overlay.draw_caption(frame, lines), overlay.draw_caption(frame, lines))


def test_caption_metrics_scale_with_frame_height() -> None:
    small_scale, small_thickness = overlay.caption_metrics(96)
    big_scale, big_thickness = overlay.caption_metrics(1440)
    assert big_scale > small_scale
    assert small_scale >= 0.4  # a floor keeps tiny panels legible
    assert big_thickness > small_thickness


# --------------------------------------------------------------------------- #
# load_panel
# --------------------------------------------------------------------------- #


def test_load_panel_reads_the_documented_schema(tmp_path: pathlib.Path) -> None:
    table = keypoint_table(range(3), {1: line_pose(0.0)})
    path = tmp_path / "clip.parquet"
    table.to_parquet(path, index=False)

    panel = overlay.load_panel("auto", path)

    assert panel.label == "auto"
    assert panel.path == path
    assert panel.frame_count == 3
    assert panel.max_frame_idx == 2
    assert panel.detected_frames == 1
    assert panel.frames[1].t_ms == 100
    assert panel.frames[1].detected
    assert math.isnan(panel.frames[0].visibility["left_wrist"])
    assert all(math.isnan(value) for value in panel.frames[0].joints_xy["left_wrist"])
    assert panel.frames[1].joints_xy["left_hip"] == pytest.approx(line_pose()["left_hip"])
    assert not panel.frames[2].detected
    assert panel.frame(99) is None

    # 20 frames of 100 ms each: the average frame rate of the clip.
    assert panel.fps() == pytest.approx(3 / 0.2)


def test_load_panel_needs_the_documented_columns(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "clip.parquet"
    keypoint_table(range(2)).drop(columns=["visibility"]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="visibility"):
        overlay.load_panel("none", path)


# --------------------------------------------------------------------------- #
# render_overlay
# --------------------------------------------------------------------------- #


def test_render_overlay_two_panels_side_by_side(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    left = keypoint_table(range(FRAME_COUNT), every_frame(0.0))
    right = keypoint_table(range(FRAME_COUNT), every_frame(4.0))
    left_path, right_path = write_panels(tmp_path / "keypoints", [("none", left), ("auto", right)])
    out_path = tmp_path / "overlays" / f"{CLIP_ID}_none-vs-auto.mp4"

    report = overlay.render_overlay(
        synthetic_video, [("none", left_path), ("auto", right_path)], out_path
    )

    assert out_path.is_file()
    assert report.out_path == out_path
    assert report.labels == ("none", "auto")
    assert report.frame_count == FRAME_COUNT
    assert report.width == 2 * WIDTH
    assert report.height == HEIGHT
    assert report.detected_frames == (FRAME_COUNT, FRAME_COUNT)
    assert report.no_pose_frames == (0, 0)
    assert report.fps == pytest.approx(FRAME_COUNT / ((FRAME_COUNT - 1) * 0.1))
    assert report.runtime_seconds > 0

    frames = decoded_frames(out_path)
    assert len(frames) == FRAME_COUNT
    assert frames[0].shape[:2] == (HEIGHT, 2 * WIDTH)

    # The two panels are the same clip with two different poses, not a copy.
    left_tile = frames[0][:, :WIDTH]
    right_tile = frames[0][:, WIDTH:]
    assert not np.array_equal(left_tile, right_tile)
    # ... and each panel is its own pose, not a squeezed one.
    assert np.array_equal(left_tile, frames[0][:, :WIDTH])


def test_render_overlay_single_panel(synthetic_video: pathlib.Path, tmp_path: pathlib.Path) -> None:
    table = keypoint_table(range(FRAME_COUNT), every_frame())
    (path,) = write_panels(tmp_path / "keypoints", [("auto", table)])
    out_path = tmp_path / "overlays" / f"{CLIP_ID}_auto.mp4"

    report = overlay.render_overlay(synthetic_video, [("auto", path)], out_path)

    assert out_path.is_file()
    assert (report.width, report.height) == (WIDTH, HEIGHT)
    frames = decoded_frames(out_path)
    assert len(frames) == FRAME_COUNT
    assert frames[0].shape[:2] == (HEIGHT, WIDTH)


def test_render_overlay_leaves_a_frame_without_a_pose_alone(
    flat_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    # Frames 0-2 have no pose, the rest do.
    table = keypoint_table(range(FRAME_COUNT), every_frame(first=3))
    (path,) = write_panels(tmp_path / "keypoints", [("auto", table)])
    out_path = tmp_path / "overlays" / "no-pose.mp4"

    report = overlay.render_overlay(flat_video, [("auto", path)], out_path)

    assert report.detected_frames == (FRAME_COUNT - 3,)
    assert report.no_pose_frames == (3,)

    frames = decoded_frames(out_path)
    raw = decoded_frames(flat_video)
    # Bottom-left corner: where the left_ankle joint is drawn on a detected frame,
    # and well clear of the caption plate in the top-left. The clip is flat grey,
    # so anything but grey there is skeleton ink.
    ankle = (slice(int(HEIGHT * 0.72), int(HEIGHT * 0.92)), slice(0, WIDTH // 3))

    def difference(index: int) -> np.ndarray:
        return np.abs(frames[index][ankle].astype(np.int32) - raw[index][ankle].astype(np.int32))

    # Frame 0 has no pose: the frame is the decoded one (give or take the codec).
    assert difference(0).max() <= 8
    # Frame 5 does have one, and the skeleton is unmistakable.
    assert difference(5).max() > 60
    # ... and the caption is on both.
    assert not np.array_equal(frames[0][:40, :80], raw[0][:40, :80])


def test_render_overlay_max_frames_truncates_but_keeps_the_frame_rate(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    table = keypoint_table(range(FRAME_COUNT), every_frame())
    (path,) = write_panels(tmp_path / "keypoints", [("auto", table)])
    out_path = tmp_path / "overlays" / "short.mp4"

    report = overlay.render_overlay(synthetic_video, [("auto", path)], out_path, max_frames=5)

    assert report.frame_count == 5
    assert len(decoded_frames(out_path)) == 5
    # The clip's own rate, so the preview plays at the recorded speed.
    assert report.fps == pytest.approx(FRAME_COUNT / ((FRAME_COUNT - 1) * 0.1))


def test_render_overlay_output_is_renamed_only_once_complete(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    table = keypoint_table(range(FRAME_COUNT), every_frame())
    (path,) = write_panels(tmp_path / "keypoints", [("auto", table)])
    out_path = tmp_path / "overlays" / "done.mp4"

    overlay.render_overlay(synthetic_video, [("auto", path)], out_path)

    assert [p.name for p in out_path.parent.iterdir()] == ["done.mp4"]


def test_render_overlay_rejects_a_video_longer_than_the_keypoints(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    short = keypoint_table(range(5), {index: line_pose() for index in range(5)})
    (path,) = write_panels(tmp_path / "keypoints", [("auto", short)])
    out_path = tmp_path / "overlays" / "mismatch.mp4"

    with pytest.raises(RuntimeError, match="no keypoints for frame 5"):
        overlay.render_overlay(synthetic_video, [("auto", path)], out_path)

    assert not out_path.exists()
    assert list(out_path.parent.iterdir()) == []


def test_render_overlay_rejects_keypoints_longer_than_the_video(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    long = keypoint_table(range(FRAME_COUNT + 5), first_frames(FRAME_COUNT + 5))
    (path,) = write_panels(tmp_path / "keypoints", [("auto", long)])
    out_path = tmp_path / "overlays" / "mismatch.mp4"

    with pytest.raises(RuntimeError, match="off by 5 frames"):
        overlay.render_overlay(synthetic_video, [("auto", path)], out_path)

    assert not out_path.exists()


def test_render_overlay_allows_one_frame_of_disagreement(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    # A parquet that stops one frame early still fails: the frames have to exist.
    almost = keypoint_table(range(FRAME_COUNT - 1), first_frames(FRAME_COUNT - 1))
    (path,) = write_panels(tmp_path / "keypoints", [("auto", almost)])
    out_path = tmp_path / "overlays" / "off-by-one.mp4"
    with pytest.raises(RuntimeError, match="no keypoints for frame 19"):
        overlay.render_overlay(synthetic_video, [("auto", path)], out_path)

    # A parquet one frame *ahead* of the video is within the tolerance.
    longer = keypoint_table(range(FRAME_COUNT + 1), first_frames(FRAME_COUNT + 1))
    (path2,) = write_panels(tmp_path / "keypoints2", [("auto", longer)])
    report = overlay.render_overlay(synthetic_video, [("auto", path2)], out_path)
    assert report.frame_count == FRAME_COUNT


def test_render_overlay_needs_a_panel(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    with pytest.raises(ValueError, match="at least one panel"):
        overlay.render_overlay(synthetic_video, [], tmp_path / "x.mp4")


def test_render_overlay_rejects_a_negative_frame_budget(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    table = keypoint_table(range(FRAME_COUNT))
    (path,) = write_panels(tmp_path / "keypoints", [("auto", table)])
    with pytest.raises(ValueError, match="max_frames"):
        overlay.render_overlay(synthetic_video, [("auto", path)], tmp_path / "x.mp4", max_frames=-1)


# --------------------------------------------------------------------------- #
# Names and locations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["auto"], "abc123def456_auto.mp4"),
        (["none", "auto"], "abc123def456_none-vs-auto.mp4"),
        (["none", "180", "auto"], "abc123def456_none-vs-180-vs-auto.mp4"),
    ],
)
def test_overlay_filename(labels: list[str], expected: str) -> None:
    assert overlay.overlay_filename(CLIP_ID, labels) == expected


def test_overlay_filename_needs_a_label() -> None:
    with pytest.raises(ValueError, match="at least one label"):
        overlay.overlay_filename(CLIP_ID, [])


def test_output_locations_live_under_the_data_dir() -> None:
    assert overlay.keypoints_root("/data") == pathlib.Path("/data/keypoints/mediapipe")
    assert overlay.panel_parquet_path("abc", "auto", "/data") == pathlib.Path(
        "/data/keypoints/mediapipe/auto/abc.parquet"
    )
    assert overlay.overlays_dir("/data") == pathlib.Path("/data/overlays")
    # Without an argument both fall back to the shared workspace, read at call time.
    assert overlay.overlays_dir() == paths.data_dir() / "overlays"


def test_missing_parquet_error_names_the_command(tmp_path: pathlib.Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    path = tmp_path / "keypoints" / "mediapipe" / "auto" / f"{CLIP_ID}.parquet"

    with pytest.raises(FileNotFoundError) as excinfo:
        overlay.require_parquet(CLIP_ID, "auto", tmp_path, video)

    message = str(excinfo.value)
    assert "python -m handstand.pose_mediapipe --rotate auto" in message
    assert str(video) in message
    assert str(path) in message


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A ``(videos, data)`` pair in the shape the CLI expects."""
    videos = tmp_path / "videos"
    data = tmp_path / "data"
    videos.mkdir()
    (data / "keypoints").mkdir(parents=True)
    (videos / "clip.mp4").write_bytes(b"not really a video")
    (data / "catalogue.csv").write_text(f"clip_id,filename\n{CLIP_ID},clip.mp4\n")
    return videos, data


def test_cli_missing_parquet_reports_the_command(
    workspace: tuple[pathlib.Path, pathlib.Path], capsys: pytest.CaptureFixture[str]
) -> None:
    videos, data = workspace
    exit_code = overlay.main(
        ["--clip", CLIP_ID, "--modes", "auto", "--videos", str(videos), "--data", str(data)]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "python -m handstand.pose_mediapipe --rotate auto" in captured.err
    assert f"keypoints/mediapipe/auto/{CLIP_ID}.parquet" in captured.err
    assert not (data / "overlays").exists()


def test_cli_renders_a_clip(
    synthetic_video: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    videos = tmp_path / "videos"
    data = tmp_path / "data"
    videos.mkdir()
    (data / "keypoints").mkdir(parents=True)
    shutil.copy(synthetic_video, videos / "clip.mp4")
    (data / "catalogue.csv").write_text(f"clip_id,filename\n{CLIP_ID},clip.mp4\n")

    poses = every_frame(0.0)
    shifted = every_frame(3.0)
    write_panels(
        data / "keypoints" / "mediapipe",
        [
            ("none", keypoint_table(range(FRAME_COUNT), poses)),
            ("auto", keypoint_table(range(FRAME_COUNT), shifted)),
        ],
    )

    exit_code = overlay.main(
        ["--clip", CLIP_ID, "--modes", "none", "auto", "--videos", str(videos), "--data", str(data)]
    )
    captured = capsys.readouterr()

    out_path = data / "overlays" / f"{CLIP_ID}_none-vs-auto.mp4"
    assert exit_code == 0, captured.err
    assert out_path.is_file()
    assert f"frames={FRAME_COUNT}" in captured.out
    assert "panels=none|auto" in captured.out

    # One panel per mode, both written in the same order as --modes.
    frames = decoded_frames(out_path)
    assert len(frames) == FRAME_COUNT
    assert frames[0].shape[:2] == (HEIGHT, 2 * WIDTH)


def test_cli_renders_a_single_mode(synthetic_video: pathlib.Path, tmp_path: pathlib.Path) -> None:
    videos = tmp_path / "videos"
    data = tmp_path / "data"
    videos.mkdir()
    (data / "keypoints").mkdir(parents=True)
    shutil.copy(synthetic_video, videos / "clip.mp4")
    # No catalogue: the clip id is the hash of the file, and the parquet follows it.
    clip_id = resolve_clip_id(videos / "clip.mp4")
    write_panels(
        data / "keypoints" / "mediapipe",
        [("auto", keypoint_table(range(FRAME_COUNT), every_frame()))],
        clip_id=clip_id,
    )

    exit_code = overlay.main(
        [
            "--clip",
            clip_id,
            "--modes",
            "auto",
            "--max-frames",
            "4",
            "--videos",
            str(videos),
            "--data",
            str(data),
        ]
    )

    out_path = data / "overlays" / f"{clip_id}_auto.mp4"
    assert exit_code == 0
    assert len(decoded_frames(out_path)) == 4


def test_cli_unknown_clip(
    workspace: tuple[pathlib.Path, pathlib.Path], capsys: pytest.CaptureFixture[str]
) -> None:
    videos, data = workspace
    write_panels(
        data / "keypoints" / "mediapipe",
        [("auto", keypoint_table(range(FRAME_COUNT)))],
    )

    # An unknown clip id resolves to no video at all, reported as an error.
    assert (
        overlay.main(
            [
                "--clip",
                "ffffffffffff",
                "--modes",
                "auto",
                "--videos",
                str(videos),
                "--data",
                str(data),
            ]
        )
        == 2
    )
    assert "no video for clip ffffffffffff" in capsys.readouterr().err


def test_cli_parser_rejects_an_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        overlay.build_arg_parser().parse_args(["--clip", CLIP_ID, "--modes", "sideways"])

    with pytest.raises(SystemExit):
        overlay.build_arg_parser().parse_args([])  # --clip is required

    args = overlay.build_arg_parser().parse_args(["--clip", CLIP_ID])
    assert args.modes == ["auto"]
    assert args.max_frames is None
    assert args.videos is None
    assert args.data is None


def test_cli_rejects_a_negative_frame_budget() -> None:
    with pytest.raises(SystemExit):
        overlay.main(["--clip", CLIP_ID, "--max-frames", "-1"])


# --------------------------------------------------------------------------- #
# video_for_clip
# --------------------------------------------------------------------------- #


def test_video_for_clip_uses_the_catalogue(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    shutil.copy(synthetic_video, videos / "clip.mp4")
    (tmp_path / "catalogue.csv").write_text(f"clip_id,filename\n{CLIP_ID},clip.mp4\n")
    assert overlay.video_for_clip(CLIP_ID, videos, tmp_path) == videos / "clip.mp4"


def test_video_for_clip_falls_back_to_the_file_hash(
    synthetic_video: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    target = videos / "renamed.mp4"
    shutil.copy(synthetic_video, target)

    # No catalogue: the id is the SHA-1 of the bytes, so a renamed clip still resolves.
    assert overlay.video_for_clip(resolve_clip_id(target), videos, tmp_path) == target


def test_video_for_clip_without_a_match(tmp_path: pathlib.Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    with pytest.raises(FileNotFoundError, match="no video for clip"):
        overlay.video_for_clip("ffffffffffff", videos, tmp_path)
    with pytest.raises(FileNotFoundError, match="videos dir"):
        overlay.video_for_clip(CLIP_ID, tmp_path / "absent", tmp_path)


def test_video_for_clip_names_a_catalogue_file_that_is_gone(tmp_path: pathlib.Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (tmp_path / "catalogue.csv").write_text(f"clip_id,filename\n{CLIP_ID},gone.mp4\n")
    with pytest.raises(FileNotFoundError, match="gone.mp4"):
        overlay.video_for_clip(CLIP_ID, videos, tmp_path)
