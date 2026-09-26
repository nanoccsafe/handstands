"""Batch MediaPipe Pose Landmarker (Full) runner for handstand videos.

Decodes videos in **display orientation** (upright, as a person views them),
runs ``pose_landmarker_full.task`` over every frame in VIDEO mode and writes
one long-format parquet per clip plus a JSON sidecar::

    <data_dir>/keypoints/mediapipe/<rotate>/<clip_id>.parquet
    <data_dir>/keypoints/mediapipe/<rotate>/<clip_id>.json

Rotation modes (``--rotate``):

``none``
    Feed frames exactly as displayed.
``180``
    Rotate every frame 180° before inference — inverted bodies are what pose
    models are worst at — then map keypoints back to the un-rotated frame.
``auto``
    Rotate a frame when the *previous* frame's result looked inverted (mean y
    of both wrists below-in-image / greater than mean y of both ankles).
    MediaPipe VIDEO mode tracks between frames, so auto mode owns **two**
    landmarker instances — one that only ever sees upright frames and one that
    only ever sees rotated frames — and each tracker therefore observes a
    consistent orientation.

People per frame (``--num-poses``):

``1`` (default)
    What the model is best at, and what the first consumer of these files
    expects: one body per frame, 33 rows per frame, no ``person_idx`` column.
    Output is byte-identical to a run without the flag.
``2..5``
    Clips often contain the trainer next to the athlete, and a single-pose
    detector happily snaps the athlete's joints onto whoever is closer to the
    camera. ``--num-poses N`` keeps **every** person the model reports, in the
    order MediaPipe returned them, and writes one 33-row block per person::

        <data_dir>/keypoints/mediapipe_multi/<rotate>/<clip_id>.parquet
        <data_dir>/keypoints/mediapipe_multi/<rotate>/<clip_id>.json

    The blocks are labelled by a new ``person_idx`` column (0..P-1 within the
    frame); a frame with nobody in it keeps a single NaN block with
    ``person_idx = -1`` and ``detected = false``, so every frame is still
    represented. Picking the athlete out of the people is a *later* step —
    this module only keeps them all. In ``auto`` mode the rotation decision is
    taken from the person whose wrists are lowest in the image (largest mean
    wrist y), i.e. whoever is most likely the one on their hands.

Coordinates written out are always **display-frame pixel indices** (origin top
left, ``x`` right, ``y`` down, ``0 <= x <= width - 1``); rotated-frame
coordinates never leave this module. MediaPipe's normalised output is turned
into pixels with ``x = x_norm * (width - 1)``, which is exactly the pair of the
180° map-back ``x = width - 1 - x_rotated`` (see ``docs/keypoint_schema.md``).

CLI::

    cd pipeline
    uv run python scripts/download_models.py
    uv run python -m handstand.pose_mediapipe --rotate auto --limit 3
    uv run python -m handstand.pose_mediapipe --rotate auto --num-poses 3
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import pathlib
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options

from handstand.paths import data_dir, videos_dir
from handstand.rotation import inverse_rotate_points

__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_NUM_POSES",
    "JOINT_NAMES",
    "MAX_NUM_POSES",
    "MODEL_FILENAME",
    "MODEL_URL",
    "MULTI_OUTPUT_DIRNAME",
    "NO_PERSON_IDX",
    "PARQUET_COLUMNS",
    "PARQUET_COLUMNS_MULTI",
    "PERSON_COLUMN",
    "ROTATE_MODES",
    "SINGLE_OUTPUT_DIRNAME",
    "ClipReport",
    "DisplayOrientation",
    "DisplayVideo",
    "FramePacket",
    "LandmarkerLike",
    "apply_display_rotation",
    "build_arg_parser",
    "collect_clips",
    "display_orientation",
    "is_inverted",
    "load_catalogue",
    "lowest_wrist_pose",
    "main",
    "make_landmarker_factory",
    "mean_wrist_y",
    "mediapipe_version",
    "normalized_to_pixels",
    "output_dirname",
    "people_histogram",
    "resolve_clip_id",
    "run_clip",
]

#: Model downloaded by ``scripts/download_models.py`` (git-ignored).
MODEL_FILENAME = "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
#: ``pipeline/models/<MODEL_FILENAME>``, relative to this file, not to the cwd.
DEFAULT_MODEL_PATH = pathlib.Path(__file__).resolve().parents[1] / "models" / MODEL_FILENAME

#: ``--rotate`` choices; also the name of the output sub-directory.
ROTATE_MODES: tuple[str, ...] = ("none", "180", "auto")

#: People per frame: the default is one body, multi-person runs are opt-in.
DEFAULT_NUM_POSES = 1
#: Upper bound of ``--num-poses``; MediaPipe's own PoseLandmarkerOptions max.
MAX_NUM_POSES = 5

#: Output roots under ``<data_dir>/keypoints/``. Single-person runs keep the
#: historical root, so a multi-person run can never overwrite them.
SINGLE_OUTPUT_DIRNAME = "mediapipe"
MULTI_OUTPUT_DIRNAME = "mediapipe_multi"

#: The 33 MediaPipe pose landmarks, snake_case, in model order.
JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
JOINT_INDEX: dict[str, int] = {name: index for index, name in enumerate(JOINT_NAMES)}

#: Joints used by the auto-mode inversion rule.
WRIST_JOINTS = ("left_wrist", "right_wrist")
ANKLE_JOINTS = ("left_ankle", "right_ankle")

#: Column order of the output parquet (see ``docs/keypoint_schema.md``).
PARQUET_COLUMNS: tuple[str, ...] = (
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

#: ``--num-poses > 1`` adds this column: which person of the frame a row is
#: about (0..P-1, ``NO_PERSON_IDX`` on the empty block of an undetected frame).
PERSON_COLUMN = "person_idx"
#: :data:`PARQUET_COLUMNS` plus the per-person label.
PARQUET_COLUMNS_MULTI: tuple[str, ...] = (*PARQUET_COLUMNS, PERSON_COLUMN)
#: ``person_idx`` of the single NaN block written for a frame with nobody in it.
NO_PERSON_IDX = -1

#: ``(x, y, z, visibility, presence)`` per landmark.
_LANDMARK_FIELDS = 5


class LandmarkerLike(Protocol):
    """Structural type of ``mediapipe...PoseLandmarker`` in VIDEO mode.

    Tests inject a fake with this shape, so the runner never needs the real
    ``.task`` model.
    """

    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any:
        """Return an object exposing ``pose_landmarks`` (list of poses)."""
        ...


# --------------------------------------------------------------------------- #
# Display orientation (decode helpers)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class DisplayOrientation:
    """How to turn decoded frames into display (upright) orientation.

    ``rotate_cw_deg`` is applied to every decoded frame: OpenCV reports
    ``CAP_PROP_ORIENTATION_META`` as the *clockwise* angle that makes the
    stream upright, so ``90`` means ``cv2.ROTATE_90_CLOCKWISE``.
    """

    rotate_cw_deg: int
    display_width: int
    display_height: int

    @property
    def display_shape(self) -> tuple[int, int]:
        """``(height, width)`` — the order used by ``ndarray.shape``."""
        return (self.display_height, self.display_width)


def display_orientation(
    frame_width: int,
    frame_height: int,
    rotation_meta_deg: float,
    auto_applied: bool,
) -> DisplayOrientation:
    """Work out the rotation needed for frames decoded from a capture.

    Parameters
    ----------
    frame_width, frame_height:
        ``CAP_PROP_FRAME_WIDTH/HEIGHT`` as reported by the capture.
    rotation_meta_deg:
        ``CAP_PROP_ORIENTATION_META`` (clockwise degrees to upright the
        stream), or ``0`` when the property is unavailable.
    auto_applied:
        ``CAP_PROP_ORIENTATION_AUTO != 0``, i.e. OpenCV already rotated the
        frames *and* reports display dimensions.
    """
    if auto_applied:
        # OpenCV rotated the frames for us; the reported size is the display size.
        return DisplayOrientation(0, int(frame_width), int(frame_height))

    clockwise = int(round(float(rotation_meta_deg))) % 360
    if clockwise not in (0, 90, 180, 270):
        raise ValueError(f"unsupported rotation metadata: {rotation_meta_deg} degrees")
    if clockwise in (90, 270):
        return DisplayOrientation(clockwise, int(frame_height), int(frame_width))
    return DisplayOrientation(clockwise, int(frame_width), int(frame_height))


def apply_display_rotation(frame: np.ndarray, rotate_cw_deg: int) -> np.ndarray:
    """Rotate a decoded frame clockwise by 0/90/180/270 degrees."""
    if rotate_cw_deg == 0:
        return frame
    if rotate_cw_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate_cw_deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotate_cw_deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported display rotation: {rotate_cw_deg} degrees")


@dataclasses.dataclass(frozen=True)
class FramePacket:
    """One decoded frame in display orientation."""

    frame_idx: int
    t_ms: int
    frame: np.ndarray


class DisplayVideo:
    """OpenCV capture that yields frames in display orientation.

    Frames are decoded once and checked against the display size on every
    read: if ``CAP_PROP_ORIENTATION_AUTO`` claims to be on but the frame comes
    back in storage orientation (or the rotation metadata is unreadable) the
    mismatch raises instead of silently producing sideways keypoints.

    Timestamps come from ``CAP_PROP_POS_MSEC`` (variable frame rate — never
    ``frame_idx / fps``) and are forced to be strictly increasing integers,
    which is what MediaPipe VIDEO mode requires.
    """

    def __init__(self, video_path: str | pathlib.Path) -> None:
        self.video_path = pathlib.Path(video_path)
        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"could not open video: {self.video_path}")

        auto = 0.0
        meta = 0.0
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            auto = self._capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            meta = self._capture.get(cv2.CAP_PROP_ORIENTATION_META)
        self.orientation = display_orientation(
            int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            meta,
            bool(auto),
        )
        self._previous_t_ms = -1

    @property
    def display_width(self) -> int:
        return self.orientation.display_width

    @property
    def display_height(self) -> int:
        return self.orientation.display_height

    def __iter__(self) -> Iterator[FramePacket]:
        frame_idx = 0
        expected_shape = self.orientation.display_shape
        while True:
            ok, frame = self._capture.read()
            if not ok:
                return
            t_ms = int(round(self._capture.get(cv2.CAP_PROP_POS_MSEC)))
            if t_ms <= self._previous_t_ms:  # MediaPipe wants strictly increasing ms.
                t_ms = self._previous_t_ms + 1
            self._previous_t_ms = t_ms

            frame = apply_display_rotation(frame, self.orientation.rotate_cw_deg)
            if frame.shape[0] != expected_shape[0] or frame.shape[1] != expected_shape[1]:
                raise RuntimeError(
                    f"{self.video_path.name}: frame {frame_idx} decoded as "
                    f"{(frame.shape[1], frame.shape[0])} but display orientation is "
                    f"{(self.orientation.display_width, self.orientation.display_height)}; "
                    "rotation metadata and decoded frames disagree"
                )
            yield FramePacket(frame_idx=frame_idx, t_ms=t_ms, frame=frame)
            frame_idx += 1

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> DisplayVideo:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Keypoint maths
# --------------------------------------------------------------------------- #


def normalized_to_pixels(norm_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert MediaPipe normalised ``(x, y)`` to pixel indices of a ``width x height`` frame.

    MediaPipe reports ``0..1`` where ``0`` is the left/top edge and ``1`` the
    right/bottom edge. Multiplying by ``size - 1`` lands those edges on pixel
    indices ``0`` and ``size - 1``, which makes the 180° map-back
    ``x = width - 1 - x_rotated`` an exact round trip.
    """
    pts = np.asarray(norm_xy, dtype=np.float64)
    if pts.ndim == 0 or pts.shape[-1] != 2:
        raise ValueError(f"points must have shape (..., 2), got {pts.shape}")
    out = np.empty_like(pts)
    out[..., 0] = pts[..., 0] * (width - 1)
    out[..., 1] = pts[..., 1] * (height - 1)
    return out


def is_inverted(landmarks_px: np.ndarray) -> bool:
    """Does a pose in display pixels look upside down (a handstand)?

    ``landmarks_px`` is shaped ``(33, >= 2)`` in display-frame pixels. The rule
    used by ``--rotate auto``: mean y of both wrists greater (lower in the
    image) than mean y of both ankles. Non-finite coordinates cannot judge an
    orientation and count as "not inverted".
    """
    pts = np.asarray(landmarks_px, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] != len(JOINT_NAMES) or pts.shape[1] < 2:
        raise ValueError(f"expected ({len(JOINT_NAMES)}, >=2) landmarks, got {pts.shape}")
    wrists = pts[[JOINT_INDEX[name] for name in WRIST_JOINTS], 1]
    ankles = pts[[JOINT_INDEX[name] for name in ANKLE_JOINTS], 1]
    if not (np.all(np.isfinite(wrists)) and np.all(np.isfinite(ankles))):
        return False
    return bool(wrists.mean() > ankles.mean())


def mean_wrist_y(landmarks_px: np.ndarray) -> float:
    """Mean y of both wrists, or NaN when either wrist is missing.

    Bigger means lower in the image, which is what the multi-person
    ``--rotate auto`` rule sorts people by.
    """
    pts = np.asarray(landmarks_px, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] != len(JOINT_NAMES) or pts.shape[1] < 2:
        raise ValueError(f"expected ({len(JOINT_NAMES)}, >=2) landmarks, got {pts.shape}")
    wrists = pts[[JOINT_INDEX[name] for name in WRIST_JOINTS], 1]
    if not np.all(np.isfinite(wrists)):
        return float("nan")
    return float(wrists.mean())


def lowest_wrist_pose(poses_px: np.ndarray) -> np.ndarray | None:
    """Of several people in a frame, the one whose wrists sit lowest.

    ``poses_px`` is shaped ``(P, 33, >= 2)`` in display-frame pixels, in the
    order MediaPipe returned them. The winner is the person with the largest
    mean wrist y (:func:`mean_wrist_y`) — the one most likely to be on their
    hands — and it is the pose ``--rotate auto`` judges the next frame by.

    People whose wrists are not both visible cannot be ranked and are skipped;
    when nobody can be (or when there is only one person) the first person is
    returned, so a single-person frame is judged exactly as it always was. An
    empty ``(0, 33, 2)`` array has no person to pick and raises.
    """
    pts = np.asarray(poses_px, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[0] < 1 or pts.shape[1] != len(JOINT_NAMES) or pts.shape[2] < 2:
        raise ValueError(f"expected (P, {len(JOINT_NAMES)}, >=2) poses, got {pts.shape}")
    if pts.shape[0] == 1:
        return pts[0]
    scores = [mean_wrist_y(pose) for pose in pts]
    rankable = [index for index, score in enumerate(scores) if np.isfinite(score)]
    if not rankable:
        return pts[0]
    return pts[max(rankable, key=lambda index: scores[index])]


def _detect_pose(
    landmarker: LandmarkerLike,
    frame_bgr: np.ndarray,
    t_ms: int,
    source: pathlib.Path,
) -> np.ndarray:
    """Run one frame; return **every** pose as ``(P, 33, 5)`` normalised.

    ``P`` is 0 when nobody was detected, so callers can tell "no people" from
    "one person with no landmarks". People keep the order MediaPipe returned
    them in (most confident first); choosing the athlete among them is a later
    step, so nothing is filtered or re-sorted here.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect_for_video(image, t_ms)
    poses = getattr(result, "pose_landmarks", None) or []
    if not poses:
        return np.empty((0, len(JOINT_NAMES), _LANDMARK_FIELDS), dtype=np.float64)
    for pose in poses:
        if len(pose) != len(JOINT_NAMES):
            raise RuntimeError(
                f"{source.name}: expected {len(JOINT_NAMES)} landmarks, got {len(pose)}"
            )
    # Missing visibility/presence come through as None -> NaN.
    return np.array(
        [[[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in pose] for pose in poses],
        dtype=np.float64,
    )


def _close(landmarker: LandmarkerLike | None) -> None:
    if landmarker is None:
        return
    close = getattr(landmarker, "close", None)
    if callable(close):
        close()


# --------------------------------------------------------------------------- #
# Long-format tables
# --------------------------------------------------------------------------- #


def _check_poses(poses_px: np.ndarray) -> np.ndarray:
    """Validate one frame's ``(P, 33, 5)`` display-pixel poses and return them."""
    pts = np.asarray(poses_px, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[1:] != (len(JOINT_NAMES), _LANDMARK_FIELDS):
        raise ValueError(
            f"expected (P, {len(JOINT_NAMES)}, {_LANDMARK_FIELDS}) poses, got {pts.shape}"
        )
    return pts


def _frame_blocks(poses_px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten one frame's people into ``(rows, 5)`` landmarks + labels.

    Each detected person contributes one block of 33 rows labelled with their
    ``person_idx`` (0..P-1, MediaPipe's order) and ``detected = True``. A frame
    with nobody in it still contributes one block of 33 all-NaN rows, labelled
    ``person_idx = NO_PERSON_IDX`` and ``detected = False``, so every frame is
    represented exactly once in the output.
    """
    pts = _check_poses(poses_px)
    if pts.shape[0] == 0:
        rows = len(JOINT_NAMES)
        return (
            np.full((rows, _LANDMARK_FIELDS), np.nan),
            np.full(rows, NO_PERSON_IDX, dtype=np.int64),
            np.zeros(rows, dtype=bool),
        )
    return (
        pts.reshape(-1, _LANDMARK_FIELDS),
        np.repeat(np.arange(pts.shape[0], dtype=np.int64), len(JOINT_NAMES)),
        np.ones(pts.shape[0] * len(JOINT_NAMES), dtype=bool),
    )


def _single_table(
    frame_idx: Sequence[int],
    frame_t_ms: Sequence[int],
    frame_rotated: Sequence[bool],
    poses: Sequence[np.ndarray],
) -> pd.DataFrame:
    """One row per (frame, joint): the single-person schema, no ``person_idx``.

    ``--num-poses 1`` means the model reported at most one person per frame, so
    person 0 is simply that person's pose and the layout is unchanged.
    """
    frame_count = len(poses)
    landmark_data = np.full((frame_count, len(JOINT_NAMES), _LANDMARK_FIELDS), np.nan)
    detected = np.zeros(frame_count, dtype=bool)
    for index, pose in enumerate(poses):
        pts = _check_poses(pose)
        if pts.shape[0]:
            landmark_data[index] = pts[0]
            detected[index] = True

    joints = np.asarray(JOINT_NAMES)
    return pd.DataFrame(
        {
            "frame_idx": np.repeat(np.asarray(frame_idx, dtype=np.int64), len(JOINT_NAMES)),
            "t_ms": np.repeat(np.asarray(frame_t_ms, dtype=np.int64), len(JOINT_NAMES)),
            "joint": np.tile(joints, frame_count),
            "x": landmark_data[..., 0].reshape(-1),
            "y": landmark_data[..., 1].reshape(-1),
            "z": landmark_data[..., 2].reshape(-1),
            "visibility": landmark_data[..., 3].reshape(-1),
            "presence": landmark_data[..., 4].reshape(-1),
            "rotated": np.repeat(np.asarray(frame_rotated, dtype=bool), len(JOINT_NAMES)),
            "detected": np.repeat(detected, len(JOINT_NAMES)),
        }
    )


def _multi_table(
    frame_idx: Sequence[int],
    frame_t_ms: Sequence[int],
    frame_rotated: Sequence[bool],
    poses: Sequence[np.ndarray],
) -> pd.DataFrame:
    """One 33-row block per person: :data:`PARQUET_COLUMNS_MULTI`.

    Rows are ordered by ``frame_idx``, then ``person_idx`` (0..P-1, the order
    MediaPipe returned the people in), then landmark index.
    """
    joints = np.asarray(JOINT_NAMES)
    landmark_blocks: list[np.ndarray] = []
    person_blocks: list[np.ndarray] = []
    detected_blocks: list[np.ndarray] = []
    joint_blocks: list[np.ndarray] = []
    rows_per_frame: list[int] = []
    for pose in poses:
        landmarks, person_idx, detected = _frame_blocks(pose)
        landmark_blocks.append(landmarks)
        person_blocks.append(person_idx)
        detected_blocks.append(detected)
        blocks = person_idx.size // len(JOINT_NAMES)
        joint_blocks.append(np.tile(joints, blocks))
        rows_per_frame.append(person_idx.size)

    stacked = np.concatenate(landmark_blocks, axis=0)
    repeat = np.asarray(rows_per_frame, dtype=np.int64)
    return pd.DataFrame(
        {
            "frame_idx": np.repeat(np.asarray(frame_idx, dtype=np.int64), repeat),
            "t_ms": np.repeat(np.asarray(frame_t_ms, dtype=np.int64), repeat),
            "joint": np.concatenate(joint_blocks),
            "x": stacked[:, 0],
            "y": stacked[:, 1],
            "z": stacked[:, 2],
            "visibility": stacked[:, 3],
            "presence": stacked[:, 4],
            "rotated": np.repeat(np.asarray(frame_rotated, dtype=bool), repeat),
            "detected": np.concatenate(detected_blocks),
            PERSON_COLUMN: np.concatenate(person_blocks),
        }
    )


# --------------------------------------------------------------------------- #
# Per-clip run
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipReport:
    """What happened to one clip."""

    clip_id: str
    video_path: pathlib.Path
    rotate: str
    frame_count: int
    detected_frames: int
    rotated_frames: int
    runtime_seconds: float
    parquet_path: pathlib.Path
    json_path: pathlib.Path
    skipped: bool = False
    num_poses: int = DEFAULT_NUM_POSES
    #: How many frames had how many people: ``{people: frames}``. Empty for
    #: skipped clips and single-person runs, which do not count people.
    frames_by_people: dict[int, int] = dataclasses.field(default_factory=dict)

    @property
    def detected_percent(self) -> float:
        """Percentage of frames with at least one pose."""
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
        """``"0:2 1:150 2:80"`` — frames per number of detected people."""
        return " ".join(
            f"{people}:{self.frames_by_people[people]}" for people in sorted(self.frames_by_people)
        )


def people_histogram(people_per_frame: Sequence[int]) -> dict[int, int]:
    """Count frames per number of detected people, including the empty ones."""
    histogram: dict[int, int] = {}
    for people in people_per_frame:
        key = int(people)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def run_clip(
    video_path: str | pathlib.Path,
    *,
    clip_id: str,
    rotate_mode: str,
    out_root: pathlib.Path,
    landmarker_factory: Callable[[], LandmarkerLike],
    overwrite: bool = False,
    model_name: str = MODEL_FILENAME,
    num_poses: int = DEFAULT_NUM_POSES,
) -> ClipReport:
    """Run the landmarker over one video and write its parquet + JSON.

    ``out_root`` is the ``<data_dir>/keypoints/mediapipe`` directory (or
    ``mediapipe_multi`` for a multi-person run — :func:`output_dirname`);
    outputs land in ``<out_root>/<rotate_mode>/<clip_id>.{parquet,json}``.
    Clips whose parquet already exists are skipped unless ``overwrite`` is true.

    ``landmarker_factory`` is called once per needed tracker: once for
    ``none``/``180``, twice for ``auto`` (upright tracker + rotated tracker).
    Tests pass a factory that returns a stub detector instead of loading the
    model.

    ``num_poses`` is how many people per frame the landmarker was asked for.
    With ``1`` the output is exactly the single-person schema and the file
    lands under the single-person root; above ``1`` every detected person gets
    their own 33-row block labelled by ``person_idx``.
    """
    video_path = pathlib.Path(video_path)
    if rotate_mode not in ROTATE_MODES:
        raise ValueError(f"unknown rotate mode {rotate_mode!r}; expected one of {ROTATE_MODES}")
    if not 1 <= num_poses <= MAX_NUM_POSES:
        raise ValueError(f"num_poses must be between 1 and {MAX_NUM_POSES}, got {num_poses}")
    multi = num_poses > 1

    out_dir = pathlib.Path(out_root) / rotate_mode
    parquet_path = out_dir / f"{clip_id}.parquet"
    json_path = out_dir / f"{clip_id}.json"
    if parquet_path.exists() and not overwrite:
        return ClipReport(
            clip_id=clip_id,
            video_path=video_path,
            rotate=rotate_mode,
            frame_count=0,
            detected_frames=0,
            rotated_frames=0,
            runtime_seconds=0.0,
            parquet_path=parquet_path,
            json_path=json_path,
            skipped=True,
            num_poses=num_poses,
        )

    started = time.perf_counter()
    if rotate_mode == "auto":
        upright_landmarker: LandmarkerLike | None = landmarker_factory()
        rotated_landmarker: LandmarkerLike | None = landmarker_factory()
    elif rotate_mode == "180":
        upright_landmarker = None
        rotated_landmarker = landmarker_factory()
    else:
        upright_landmarker = landmarker_factory()
        rotated_landmarker = None

    frame_idx: list[int] = []
    frame_t_ms: list[int] = []
    frame_rotated: list[bool] = []
    #: One ``(P, 33, 5)`` array of display-pixel poses per frame, ``P == 0``
    #: when nobody was detected.
    poses: list[np.ndarray] = []
    display_width = display_height = 0

    try:
        with DisplayVideo(video_path) as video:
            display_width = video.display_width
            display_height = video.display_height
            rotate_next = False  # first frame is never rotated
            for packet in video:
                rotated = rotate_mode == "180" or (rotate_mode == "auto" and rotate_next)
                landmarker = rotated_landmarker if rotated else upright_landmarker
                assert landmarker is not None  # mode/instance pairing above

                inference_frame = (
                    apply_display_rotation(packet.frame, 180) if rotated else packet.frame
                )
                pose_norm = _detect_pose(landmarker, inference_frame, packet.t_ms, video_path)

                # (P, 33, 5): scale x/y to pixel indices of the frame the model
                # saw, then map back into display pixels; z/visibility/
                # presence pass through untouched.
                pose_px = pose_norm.copy()
                pose_px[..., :2] = normalized_to_pixels(
                    pose_norm[..., :2],
                    inference_frame.shape[1],
                    inference_frame.shape[0],
                )
                if rotated:
                    pose_px[..., :2] = inverse_rotate_points(
                        pose_px[..., :2], 180, display_width, display_height
                    )

                frame_idx.append(packet.frame_idx)
                frame_t_ms.append(packet.t_ms)
                frame_rotated.append(rotated)
                poses.append(pose_px)

                if rotate_mode == "auto" and pose_px.shape[0]:
                    # Decide from the PREVIOUS frame, in display coordinates,
                    # from the person whose wrists are lowest in the image —
                    # the one most likely to be the athlete on their hands. A
                    # frame we failed to detect keeps the previous decision.
                    rotate_next = is_inverted(lowest_wrist_pose(pose_px))
    finally:
        _close(upright_landmarker)
        _close(rotated_landmarker)

    if not poses:
        raise RuntimeError(f"{video_path.name}: no frames decoded")

    frame_count = len(poses)
    people_per_frame = [pose.shape[0] for pose in poses]
    detected_frames = sum(people > 0 for people in people_per_frame)
    rotated_frames = sum(frame_rotated)
    frames_by_people = people_histogram(people_per_frame) if multi else {}

    table = (
        _multi_table(frame_idx, frame_t_ms, frame_rotated, poses)
        if multi
        else _single_table(frame_idx, frame_t_ms, frame_rotated, poses)
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    runtime_seconds = time.perf_counter() - started
    sidecar = {
        "clip_id": clip_id,
        "source_file": video_path.name,
        "model": model_name,
        "rotate": rotate_mode,
        "display_width": display_width,
        "display_height": display_height,
        "frame_count": frame_count,
        "detected_frame_count": detected_frames,
        "rotated_frame_count": rotated_frames,
        "mediapipe_version": mediapipe_version(),
        "runtime_seconds": round(runtime_seconds, 3),
    }
    if multi:
        # Only multi-person sidecars carry these: a single-person run must stay
        # byte-identical to what the runner wrote before --num-poses existed.
        sidecar["num_poses"] = num_poses
        sidecar["frames_by_people"] = {
            str(people): count for people, count in frames_by_people.items()
        }
    # The parquet is written last and atomically: its existence is what marks a clip as done
    # (see the skip check above), so an interrupted run must never leave a partial one behind.
    json_tmp = json_path.with_name(f".{json_path.name}.tmp")
    json_tmp.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    json_tmp.replace(json_path)
    parquet_tmp = parquet_path.with_name(f".{parquet_path.name}.tmp")
    try:
        table.to_parquet(parquet_tmp, index=False)
        parquet_tmp.replace(parquet_path)
    finally:
        parquet_tmp.unlink(missing_ok=True)

    return ClipReport(
        clip_id=clip_id,
        video_path=video_path,
        rotate=rotate_mode,
        frame_count=frame_count,
        detected_frames=detected_frames,
        rotated_frames=rotated_frames,
        runtime_seconds=runtime_seconds,
        parquet_path=parquet_path,
        json_path=json_path,
        num_poses=num_poses,
        frames_by_people=frames_by_people,
    )


# --------------------------------------------------------------------------- #
# Clip ids
# --------------------------------------------------------------------------- #


def sha1_clip_id(video_path: str | pathlib.Path) -> str:
    """First 12 hex chars of the SHA-1 of the file bytes (catalogue definition)."""
    digest = hashlib.sha1()
    with open(video_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def load_catalogue(catalogue_path: str | pathlib.Path) -> dict[str, str] | None:
    """Load ``<data_dir>/catalogue.csv`` (``clip_id,filename``) or ``None``.

    Keys include both the stored filename and its basename so lookups work
    whichever form the ``filename`` column uses.
    """
    path = pathlib.Path(catalogue_path)
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"clip_id", "filename"} <= fields:
            raise ValueError(f"{path}: expected columns clip_id and filename, got {sorted(fields)}")
        catalogue: dict[str, str] = {}
        for row in reader:
            filename = row["filename"]
            clip_id = row["clip_id"]
            catalogue[filename] = clip_id
            catalogue.setdefault(pathlib.Path(filename).name, clip_id)
        return catalogue


def resolve_clip_id(
    video_path: str | pathlib.Path,
    catalogue: Mapping[str, str] | None = None,
) -> str:
    """Return the clip id from the catalogue, else the SHA-1 based fallback."""
    path = pathlib.Path(video_path)
    if catalogue:
        for key in (path.name, str(path)):
            if key in catalogue:
                return catalogue[key]
    return sha1_clip_id(path)


def mediapipe_version() -> str:
    """Version of the mediapipe distribution this module imported."""
    version = getattr(mp, "__version__", None)
    if version:
        return str(version)
    try:
        return importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def output_dirname(num_poses: int = DEFAULT_NUM_POSES) -> str:
    """Which directory under ``<data_dir>/keypoints/`` a run writes to.

    Multi-person runs get their own root so the single-person parquets that
    every other stage reads are never touched.
    """
    if not 1 <= num_poses <= MAX_NUM_POSES:
        raise ValueError(f"num_poses must be between 1 and {MAX_NUM_POSES}, got {num_poses}")
    return MULTI_OUTPUT_DIRNAME if num_poses > 1 else SINGLE_OUTPUT_DIRNAME


def make_landmarker_factory(
    model_path: str | pathlib.Path,
    num_poses: int = DEFAULT_NUM_POSES,
) -> Callable[[], LandmarkerLike]:
    """Build a factory creating a VIDEO-mode ``PoseLandmarker`` per call.

    ``num_poses`` is how many people per frame the model is allowed to report
    (1..5); a factory built for a multi-person run cannot write into the
    single-person root, see :func:`output_dirname`.

    The model file is only checked when the factory is actually used, so a run
    where every clip is skipped does not need the model present.
    """
    model_path = pathlib.Path(model_path)
    if not 1 <= num_poses <= MAX_NUM_POSES:
        raise ValueError(f"num_poses must be between 1 and {MAX_NUM_POSES}, got {num_poses}")

    def factory() -> LandmarkerLike:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"pose model not found: {model_path} "
                "(run `uv run python scripts/download_models.py` first)"
            )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_base_options.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    return factory


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def collect_clips(
    clips: Sequence[str] | None,
    videos_directory: pathlib.Path,
) -> list[pathlib.Path]:
    """Resolve ``--clips`` (or every ``*.mp4`` in the videos directory)."""
    if clips:
        paths = [pathlib.Path(clip).expanduser() for clip in clips]
    else:
        paths = sorted(videos_directory.glob("*.mp4"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing video file(s): " + ", ".join(str(p) for p in missing))
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m handstand.pose_mediapipe",
        description="Extract MediaPipe pose keypoints for handstand videos.",
    )
    parser.add_argument(
        "--rotate",
        choices=ROTATE_MODES,
        default="none",
        help="frame rotation fed to the model (default: none)",
    )
    parser.add_argument(
        "--num-poses",
        type=int,
        choices=range(1, MAX_NUM_POSES + 1),
        default=DEFAULT_NUM_POSES,
        metavar="N",
        help=(
            "people to keep per frame, 1..5; above 1 every detected person gets "
            f"a {PERSON_COLUMN} block and the run writes to "
            f"keypoints/{MULTI_OUTPUT_DIRNAME}/ (default: {DEFAULT_NUM_POSES})"
        ),
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        metavar="FILE",
        default=None,
        help="videos to process (default: every *.mp4 in the videos directory)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="process at most N clips",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run clips whose parquet already exists",
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=DEFAULT_MODEL_PATH,
        help=f"pose landmarker .task file (default: {DEFAULT_MODEL_PATH})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    clips = collect_clips(args.clips, videos_dir())
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        print("no clips to process")
        return 0

    catalogue = load_catalogue(data_dir() / "catalogue.csv")
    landmarker_factory = make_landmarker_factory(args.model, args.num_poses)
    out_root = data_dir() / "keypoints" / output_dirname(args.num_poses)
    model_name = pathlib.Path(args.model).name
    print(
        f"rotate={args.rotate} num_poses={args.num_poses} model={model_name} "
        f"clips={len(clips)} out={out_root}"
    )

    failures = 0
    for path in clips:
        clip_id = resolve_clip_id(path, catalogue)
        try:
            report = run_clip(
                path,
                clip_id=clip_id,
                rotate_mode=args.rotate,
                out_root=out_root,
                landmarker_factory=landmarker_factory,
                overwrite=args.overwrite,
                model_name=model_name,
                num_poses=args.num_poses,
            )
        except Exception as error:  # one bad clip must not kill the whole batch
            failures += 1
            print(f"fail  {path.name}: {error}")
            continue
        if report.skipped:
            print(f"skip  {path.name} ({report.parquet_path} exists)")
            continue
        people = f" people[{report.people_summary}]" if report.frames_by_people else ""
        print(
            f"clip  {path.name} id={clip_id} frames={report.frame_count} "
            f"detected={report.detected_percent:.1f}% rotated={report.rotated_percent:.1f}%"
            f"{people} runtime={report.runtime_seconds:.1f}s -> {report.parquet_path}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
