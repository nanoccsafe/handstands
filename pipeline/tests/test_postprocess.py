"""Tests for :mod:`handstand.postprocess` and :mod:`handstand.bodyframe`.

Nothing here touches the real handstand videos. A clip is a synthetic skeleton —
a side-on body with a torso, a thigh, a shin, hands below the hips and, by
default, a fixed set of timestamps — folded into the same long parquet schema the
pipeline writes, so the tests exercise the real loader, the real thresholds and
the real atomic writes on input that is small enough to reason about exactly.
The pixel values are chosen so the body length is a known number, which is what
lets a test say "within 5 % of the true length" instead of "it did not crash".
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from handstand import athlete, bodyframe, overlay
from handstand import pose_mediapipe as pm
from handstand import postprocess as pp

# --------------------------------------------------------------------------- #
# A synthetic body
# --------------------------------------------------------------------------- #

#: The joints the skeleton builder places from the body itself; the rest are
#: parked on their parent joint, which is all a test needs of them.
_FACE = (
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
)
_FINGERS = ("left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb")
_FEET = ("left_heel", "right_heel", "left_foot_index", "right_foot_index")


@dataclasses.dataclass(frozen=True)
class Skeleton:
    """A side-on body as ``(frames, joints)`` arrays plus the body length it implies.

    The body is **inverted**, as a handstand is: the origin is the hip midpoint,
    ``y`` grows downwards, the shoulders are ``torso`` below it (towards the
    hands) and the knees and ankles ``thigh`` and ``thigh + shin`` above it
    (towards the feet). So the spans :data:`handstand.postprocess.LENGTH_PARTS`
    measures are exactly ``torso``, ``thigh`` and ``shin``, and the mean wrist
    ``y`` is below the mean ankle ``y`` — which is what "hold-like" means in this
    codebase.
    """

    x: np.ndarray
    y: np.ndarray
    joints: tuple[str, ...]
    body_length: float
    t_ms: np.ndarray

    @property
    def frames(self) -> int:
        """How many frames the skeleton has."""
        return int(self.x.shape[0])

    def column(self, joint: str) -> int:
        """The column of ``joint``."""
        return self.joints.index(joint)


def skeleton(
    frames: int = 40,
    *,
    t_ms: Sequence[int] | None = None,
    joints: Sequence[str] = pm.JOINT_NAMES,
    torso: float = 100.0,
    thigh: float = 120.0,
    shin: float = 120.0,
    hand_drop: float = 90.0,
    head: float = 40.0,
    #: Per-frame scale of each leg's thigh and shin: 1.0 is seen side-on, 0.0 is
    #: pointing straight at the camera. A split is one side at 1.0 and the other
    #: at 0.0 for half the clip.
    leg_scale: tuple[np.ndarray, np.ndarray] | None = None,
    drift: float = 0.0,
) -> Skeleton:
    """A side-on inverted body, ``frames`` long, with the hands on the floor.

    ``leg_scale`` is ``(left, right)``, one array of per-frame scales each; both
    default to 1.0. ``drift`` moves the whole body ``drift`` pixels to the right
    per frame, which is how a test produces a known speed.
    """
    times = (
        np.arange(frames, dtype=np.int64) * 33 if t_ms is None else np.asarray(t_ms, dtype=np.int64)
    )
    if times.size != frames:
        raise ValueError("t_ms must hold one timestamp per frame")
    names = tuple(joints)
    x = np.zeros((frames, len(names)))
    y = np.zeros((frames, len(names)))
    steps = np.arange(frames, dtype=np.float64)
    scales = (np.ones(frames), np.ones(frames)) if leg_scale is None else leg_scale

    def put(joint: str, px: np.ndarray, py: np.ndarray) -> None:
        if joint in names:
            x[:, names.index(joint)] = px
            y[:, names.index(joint)] = py

    # An inverted body, as a handstand is: the feet are above the hip, the hands
    # below it, and `y` grows downwards from the ankles through the hips out to
    # the hands. The spans LENGTH_PARTS measures are then exactly the three
    # lengths passed in.
    zero, hand = np.zeros(frames), np.full(frames, torso + hand_drop)
    for side, scale in zip(("left", "right"), scales, strict=True):
        sign = -1.0 if side == "left" else 1.0
        side_x = sign * 8.0 + steps * drift
        foot_y = -(thigh + shin) * scale
        put(f"{side}_hip", side_x, zero)
        put(f"{side}_shoulder", side_x, np.full(frames, torso))
        put(f"{side}_knee", side_x, -thigh * scale)
        put(f"{side}_ankle", side_x, foot_y)
        put(f"{side}_wrist", side_x, hand)
        put(f"{side}_heel", side_x, foot_y + 4.0)
        put(f"{side}_foot_index", side_x, foot_y + 8.0)
    put("nose", np.zeros(frames), np.full(frames, torso + head))
    for joint in _FINGERS:
        side = "left" if joint.startswith("left") else "right"
        sign = -1.0 if side == "left" else 1.0
        put(joint, sign * 8.0 + steps * drift, hand + 5.0)
    for index, joint in enumerate(j for j in names if j in _FACE and j != "nose"):
        put(joint, np.full(frames, 3.0 * (index - 5)), np.full(frames, torso + head))
    return Skeleton(
        x=x,
        y=y,
        joints=names,
        body_length=torso + thigh + shin,
        t_ms=times,
    )


@dataclasses.dataclass
class Clip:
    """A synthetic athlete parquet: a skeleton plus the columns around it."""

    skeleton: Skeleton
    visibility: np.ndarray
    detected: np.ndarray
    contact: np.ndarray
    z: np.ndarray | None = None

    @property
    def columns(self) -> list[str]:
        return list(athlete.ATHLETE_COLUMNS)

    def table(self) -> pd.DataFrame:
        """The long schema, one row per ``(frame, joint)``, 33-ish rows a frame."""
        body = self.skeleton
        frames, joints = body.frames, len(body.joints)
        return pd.DataFrame(
            {
                "frame_idx": np.repeat(np.arange(frames, dtype=np.int64), joints),
                "t_ms": np.repeat(body.t_ms, joints),
                "joint": np.tile(np.asarray(body.joints), frames),
                "x": body.x.reshape(-1),
                "y": body.y.reshape(-1),
                "z": (self.z.reshape(-1) if self.z is not None else np.zeros(body.x.size)),
                "visibility": self.visibility.reshape(-1),
                "presence": self.visibility.reshape(-1),
                "rotated": np.zeros(body.x.size, dtype=bool),
                "detected": np.repeat(self.detected, joints),
                athlete.SCORE_COLUMN: np.repeat(np.where(self.detected, 0.9, np.nan), joints),
                athlete.N_PEOPLE_COLUMN: np.repeat(
                    np.where(self.contact, 2, 1).astype(np.int64), joints
                ),
                athlete.CONTACT_COLUMN: np.repeat(self.contact, joints),
                athlete.CONTACT_REASON_COLUMN: np.repeat(
                    np.where(self.contact, athlete.REASON_BOX_IOU, athlete.REASON_NONE), joints
                ),
            }
        )


def clip(
    body: Skeleton | None = None,
    *,
    visibility: float = 0.95,
    detected: bool = True,
    contact: bool = False,
) -> Clip:
    """A clip of ``body`` (a fresh :func:`skeleton` by default) with flat columns."""
    body = body if body is not None else skeleton()
    shape = body.x.shape
    return Clip(
        skeleton=body,
        visibility=np.full(shape, float(visibility)),
        detected=np.full(body.frames, bool(detected)),
        contact=np.full(body.frames, bool(contact)),
    )


def write_clip(
    directory: pathlib.Path,
    clip_: Clip,
    clip_id: str = "0123456789ab",
    rotate: str = "auto",
) -> pathlib.Path:
    """Write ``<directory>/<rotate>/<clip_id>.parquet`` and return the path."""
    root = directory / rotate
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{clip_id}.parquet"
    clip_.table().to_parquet(path, index=False)
    return path


def read_back(directory: pathlib.Path, clip_id: str = "0123456789ab") -> pp.ClipKeypoints:
    """Run the real loader over a written clip."""
    return pp.read_clip(directory / "auto" / f"{clip_id}.parquet")


def process(directory: pathlib.Path, clip_: Clip, **kwargs) -> pp.ProcessedClip:
    """Write a clip, read it back and post-process it: the whole path, in memory."""
    write_clip(directory, clip_)
    return pp.process_clip(read_back(directory), **kwargs)


def processed_frame(processed: pp.ProcessedClip, frame: int, joint: str) -> tuple[float, float]:
    """The processed ``(x, y)`` of one joint in one frame."""
    column = processed.clip.column(joint)
    return (
        float(processed.x[frame, column]),
        float(processed.y[frame, column]),
    )


def vfr_times(*gaps_ms: int) -> list[int]:
    """Timestamps built from the gaps between frames, e.g. ``vfr_times(33, 66)``."""
    times, clock = [], 0
    for gap in gaps_ms:
        times.append(clock)
        clock += gap
    times.append(clock)
    return times


# --------------------------------------------------------------------------- #
# The body frame
# --------------------------------------------------------------------------- #


def test_to_body_frame_puts_the_wrist_midpoint_at_the_origin() -> None:
    assert bodyframe.to_body_frame(120.0, 300.0, 120.0, 300.0, 200.0) == (0.0, 0.0)


def test_to_body_frame_v_is_positive_above_the_hands() -> None:
    """A handstand is upside down, so "up" has to be flipped to mean "away from the floor"."""
    above = bodyframe.to_body_frame(120.0, 200.0, 120.0, 300.0, 200.0)
    below = bodyframe.to_body_frame(120.0, 400.0, 120.0, 300.0, 200.0)
    assert above[1] > 0.0
    assert below[1] < 0.0
    assert above[1] == pytest.approx(0.5)
    assert below[1] == pytest.approx(-0.5)


def test_to_body_frame_u_is_positive_to_the_right() -> None:
    right = bodyframe.to_body_frame(220.0, 300.0, 120.0, 300.0, 200.0)
    left = bodyframe.to_body_frame(20.0, 300.0, 120.0, 300.0, 200.0)
    assert right[0] == pytest.approx(0.5)
    assert left[0] == pytest.approx(-0.5)


def test_to_body_frame_is_one_body_length_up() -> None:
    assert bodyframe.to_body_frame(120.0, 100.0, 120.0, 300.0, 200.0) == pytest.approx((0.0, 1.0))


def test_to_body_frame_works_on_arrays() -> None:
    points = np.array([[[120.0, 300.0], [320.0, 200.0]], [[120.0, 300.0], [120.0, 100.0]]])
    got = bodyframe.body_frame_points(points, (120.0, 300.0), 200.0)
    expected = np.array([[[0.0, 0.0], [1.0, 0.5]], [[0.0, 0.0], [0.0, 1.0]]])
    assert got.shape == expected.shape
    assert got == pytest.approx(expected)
    for index in np.ndindex(points.shape[:-1]):
        scalar = bodyframe.to_body_frame(points[index][0], points[index][1], 120.0, 300.0, 200.0)
        assert tuple(got[index]) == pytest.approx(scalar)


def test_to_body_frame_rejects_a_useless_scale() -> None:
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="body_length"):
            bodyframe.to_body_frame(1.0, 1.0, 0.0, 0.0, bad)


def test_body_frame_points_rejects_a_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        bodyframe.body_frame_points(np.zeros((4, 3)), (0.0, 0.0), 100.0)


def test_wrist_midpoint_is_the_mean_of_the_two_wrists() -> None:
    assert bodyframe.wrist_midpoint((10.0, 20.0), (30.0, 40.0)) == (20.0, 30.0)


def test_midpoint_of_a_pair_of_points() -> None:
    assert bodyframe.midpoint((0.0, 0.0), (10.0, 20.0)) == (5.0, 10.0)
    with pytest.raises(ValueError, match="a point is"):
        bodyframe.midpoint((0.0, 0.0), (1.0, 2.0, 3.0))


def test_body_frame_of_a_whole_skeleton_orders_the_body_from_the_floor_up() -> None:
    """The feet are furthest from the hands, the nose is in between, the hands are 0."""
    body = skeleton(torso=100.0, thigh=120.0, shin=120.0, hand_drop=90.0, head=40.0)
    hands = bodyframe.wrist_midpoint(
        (body.x[0, body.column("left_wrist")], body.y[0, body.column("left_wrist")]),
        (body.x[0, body.column("right_wrist")], body.y[0, body.column("right_wrist")]),
    )
    heights = {
        joint: bodyframe.to_body_frame(
            body.x[0, body.column(joint)], body.y[0, body.column(joint)], *hands, body.body_length
        )
        for joint in ("left_wrist", "right_wrist", "nose", "left_ankle", "left_hip")
    }
    assert hands == (0.0, body.y[0, body.column("left_wrist")])
    assert heights["left_wrist"][1] == 0.0 and heights["right_wrist"][1] == 0.0
    assert 0.0 < heights["nose"][1] < heights["left_ankle"][1]
    assert heights["left_hip"][1] == pytest.approx(190.0 / body.body_length)
    assert heights["left_ankle"][1] == pytest.approx(430.0 / body.body_length)
    for u, _ in heights.values():
        assert abs(u) < 0.05  # everything is between the two wrists in this body


# --------------------------------------------------------------------------- #
# Step 1: gating
# --------------------------------------------------------------------------- #


def test_gating_keeps_a_confident_sample() -> None:
    x = np.array([[1.0, 2.0]])
    valid = pp.gated_valid(x, x.copy(), np.array([[0.9, 0.5]]), np.array([True]))
    assert valid.tolist() == [[True, True]]


def test_gating_drops_a_frame_the_model_found_nobody_in() -> None:
    x = np.array([[1.0], [2.0]])
    valid = pp.gated_valid(x, x.copy(), np.array([[0.9], [0.9]]), np.array([True, False]))
    assert valid[:, 0].tolist() == [True, False]


def test_gating_drops_trainer_contact() -> None:
    """A contact frame is two bodies' keypoints stitched together: all of it goes."""
    x = np.ones((2, 3))
    valid = pp.gated_valid(x, x.copy(), np.full((2, 3), 0.9), np.array([True, False]))
    assert valid[0].all()
    assert not valid[1].any()


def test_gating_drops_a_joint_below_the_visibility_threshold() -> None:
    x = np.ones((1, 3))
    visibility = np.array([[0.49, 0.5, math.nan]])
    valid = pp.gated_valid(x, x.copy(), visibility, np.array([True]))
    assert valid.tolist() == [[False, True, False]]


def test_gating_drops_a_joint_with_no_coordinates() -> None:
    x = np.array([[1.0, math.nan]])
    y = np.array([[1.0, 2.0]])
    valid = pp.gated_valid(x, y, np.full((1, 2), 0.9), np.array([True]))
    assert valid.tolist() == [[True, False]]


def test_gating_checks_the_shapes_it_is_given() -> None:
    x = np.zeros((2, 3))
    with pytest.raises(ValueError, match="frame_valid"):
        pp.gated_valid(x, x.copy(), np.zeros((2, 3)), np.array([True]))
    with pytest.raises(ValueError, match="same"):
        pp.gated_valid(x, x.copy(), np.zeros((2, 2)), np.array([True, True]))


def test_a_contact_frame_comes_out_all_nan(tmp_path: pathlib.Path) -> None:
    """End to end: a contact frame is written with no position in it at all.

    The run of contact frames is 20 frames long, so it is longer than
    :data:`postprocess.MAX_GAP_S` and nothing bridges it.
    """
    clip_ = clip(contact=False)
    clip_.contact[10:30] = True
    clip_.detected[20] = False
    processed = process(tmp_path, clip_)
    for frame in (10, 11, 29, 20):
        assert np.isnan(processed.x[frame]).all()
        assert np.isnan(processed.y[frame]).all()
        assert not processed.valid[frame].any()
        assert not processed.filled[frame].any()
    assert processed.valid[0].all()
    assert processed.valid[35].all()
    # The model's own coordinates survive untouched in x_raw.
    assert processed.clip.x[10, 0] == clip_.skeleton.x[10, 0]


def test_a_short_contact_blip_is_bridged_and_marked_as_filled(
    tmp_path: pathlib.Path,
) -> None:
    """Four contact frames are under the gap limit, so they come back as a bridge.

    They are marked ``filled`` precisely so that a later stage can refuse them: the
    interpolation is continuity, not observation.
    """
    clip_ = clip(contact=False)
    clip_.contact[10:14] = True
    processed = process(tmp_path, clip_)
    assert processed.filled[10:14].all()
    assert processed.valid[10:14].all()
    assert not processed.gated[10:14].any()
    assert processed.filled[9].sum() == 0 and processed.filled[14].sum() == 0


def test_a_low_visibility_joint_is_gated_but_its_raw_coordinates_survive(tmp_path) -> None:
    clip_ = clip()
    clip_.visibility[:, clip_.skeleton.column("right_ankle")] = 0.2
    processed = process(tmp_path, clip_)
    column = processed.clip.column("right_ankle")
    assert not processed.valid[:, column].any()
    assert np.isfinite(processed.clip.x[:, column]).all()


# --------------------------------------------------------------------------- #
# Step 2: body length
# --------------------------------------------------------------------------- #


def test_body_length_is_the_sum_of_its_three_parts(tmp_path: pathlib.Path) -> None:
    body = skeleton(torso=100.0, thigh=120.0, shin=90.0)
    processed = process(tmp_path, clip(body))
    measured = processed.body_length
    assert measured.usable
    assert measured.torso == pytest.approx(100.0)
    assert measured.thigh == pytest.approx(120.0)
    assert measured.shin == pytest.approx(90.0)
    assert measured.total == pytest.approx(310.0)
    assert measured.parts() == {
        "torso": pytest.approx(100.0),
        "thigh": pytest.approx(120.0),
        "shin": pytest.approx(90.0),
    }


def test_body_length_survives_a_split_that_shrinks_half_the_frames(
    tmp_path: pathlib.Path,
) -> None:
    """A split can only foreshorten a leg, and the other leg of the frame is intact."""
    frames = 40
    right = np.ones(frames)
    left = np.ones(frames)
    left[frames // 2 :] = 0.0  # the left leg points at the camera in the second half
    body = skeleton(frames=frames, leg_scale=(left, right))
    processed = process(tmp_path, clip(body))
    assert processed.body_length.total == pytest.approx(body.body_length, rel=0.05)


def test_body_length_survives_a_foreshortened_torso(tmp_path: pathlib.Path) -> None:
    """The torso is measured on its own percentile too, so a tucked frame is outvoted."""
    frames = 40
    body = skeleton(frames=frames)
    tucked = frames // 2
    for joint in ("left_shoulder", "right_shoulder"):
        body.y[tucked:, body.column(joint)] = body.y[0, body.column(joint)] * 0.3
    processed = process(tmp_path, clip(body))
    assert processed.body_length.total == pytest.approx(body.body_length, rel=0.05)


def test_body_length_needs_enough_frames_to_measure_a_part(tmp_path: pathlib.Path) -> None:
    body = skeleton(frames=40)
    clip_ = clip(body)
    clip_.contact[:35] = True  # only the last five frames survive the gate
    processed = process(tmp_path, clip_)
    assert not processed.usable
    assert "need" in processed.body_length.reason
    # The counts are still reported, so a clip lost to the trainer is told apart
    # from one the model never saw.
    assert processed.body_length.frames() == {"torso": 5, "thigh": 5, "shin": 5}
    assert math.isnan(processed.body_length.total)


def test_an_unusable_clip_is_written_with_every_joint_invalid(tmp_path: pathlib.Path) -> None:
    """No body length, no scale: the parquet says so rather than pretending."""
    body = skeleton(frames=40)
    clip_ = clip(body)
    clip_.detected[:] = False
    processed = process(tmp_path, clip_)
    assert not processed.usable
    assert not processed.valid.any()
    assert not processed.filled.any()
    assert np.isnan(processed.x).all()
    assert np.isfinite(processed.clip.x).all()  # the raw coordinates are still there


def test_body_length_needs_both_shoulders_for_the_torso(tmp_path: pathlib.Path) -> None:
    body = skeleton(frames=40)
    clip_ = clip(body)
    clip_.visibility[:, body.column("left_shoulder")] = 0.1
    clip_.visibility[:, body.column("right_shoulder")] = 0.1
    processed = process(tmp_path, clip_)
    assert not processed.usable
    assert "torso" in processed.body_length.reason


def test_body_length_of_a_clip_whose_leg_is_missing_uses_the_other_one() -> None:
    """One leg out of every frame still gives a length: the visible leg sets it."""
    body = skeleton(frames=20)
    valid = np.ones(body.x.shape, dtype=bool)
    valid[:, body.column("left_knee")] = False
    valid[:, body.column("left_ankle")] = False
    measured = pp.estimate_body_length(body.joints, body.x, body.y, valid)
    assert measured.usable
    assert measured.total == pytest.approx(body.body_length)
    assert measured.thigh_frames == 20


def test_a_source_without_foot_index_still_gets_a_body_length(tmp_path: pathlib.Path) -> None:
    """Apple Vision reports no ``foot_index``; the module must not need one."""
    joints = tuple(name for name in pm.JOINT_NAMES if name not in pp.FOOT_INDEX_JOINTS)
    body = skeleton(joints=joints)
    processed = process(tmp_path, clip(body))
    assert processed.usable
    assert processed.body_length.total == pytest.approx(body.body_length)


def test_estimate_body_length_checks_its_arrays() -> None:
    x = np.zeros((3, 2))
    with pytest.raises(ValueError, match="same"):
        pp.estimate_body_length(("a", "b"), x, x, np.zeros((3, 3), dtype=bool))


# --------------------------------------------------------------------------- #
# Step 3: outliers
# --------------------------------------------------------------------------- #


def test_a_joint_that_jumps_is_invalid_in_that_frame_only() -> None:
    t = np.arange(10) / 30.0
    x = np.zeros((10, 1))
    y = np.zeros((10, 1))
    valid = np.ones((10, 1), dtype=bool)
    x[4] = 500.0  # 500 px in a body length of 100 px, in 1/30 s: 150 L/s
    kept = pp.remove_speed_outliers(t, x, y, valid, 100.0)
    assert not kept[4, 0]
    assert kept.sum() == 9
    # The spike does not drag the next sample down with it: the reference stays
    # the last good sample, so a whole run of spikes collapses to its first one.
    assert kept[5:, 0].all()


def test_a_joint_that_moves_at_handstand_speed_is_kept() -> None:
    frames = 60
    t = np.arange(frames) / 30.0
    x = (0.5 * t)[:, None]  # 0.5 L/s in a 100 px body length
    kept = pp.remove_speed_outliers(t, x, x.copy(), np.ones_like(x, dtype=bool), 100.0)
    assert kept.all()


def test_the_speed_limit_is_measured_in_body_lengths() -> None:
    """The same pixel jump is 20 L/s for a 25 px body and 0.5 L/s for a 1000 px one."""
    frames = 30
    t = np.arange(frames) / 30.0
    x = np.zeros((frames, 1))
    x[10] = 50.0
    for length, survives in ((25.0, False), (1000.0, True)):
        kept = pp.remove_speed_outliers(t, x, x.copy(), np.ones_like(x, dtype=bool), length)
        assert bool(kept[10, 0]) is survives


def test_the_speed_limit_uses_the_real_frame_times() -> None:
    """Variable frame rate: the same 40 px step is a jump at 30 fps, a drift at 5 fps."""
    x = np.zeros((4, 1))
    x[1] = 40.0
    valid = np.ones_like(x, dtype=bool)
    fast = pp.remove_speed_outliers(np.array([0.0, 0.033, 0.066, 0.099]), x, x.copy(), valid, 100.0)
    slow = pp.remove_speed_outliers(np.array([0.0, 0.2, 0.4, 0.6]), x, x.copy(), valid, 100.0)
    assert not fast[1, 0]
    assert slow[1, 0]


def test_a_duplicate_timestamp_is_not_evidence_of_a_jump() -> None:
    x = np.zeros((3, 1))
    x[1] = 90.0
    kept = pp.remove_speed_outliers(
        np.array([0.0, 0.0, 0.1]), x, x.copy(), np.ones_like(x, dtype=bool), 100.0
    )
    assert kept[1, 0]


def test_remove_speed_outliers_checks_its_arguments() -> None:
    x = np.zeros((3, 1))
    with pytest.raises(ValueError, match="body_length"):
        pp.remove_speed_outliers(np.zeros(3), x, x.copy(), np.ones_like(x, dtype=bool), 0.0)
    with pytest.raises(ValueError, match="t_seconds"):
        pp.remove_speed_outliers(np.zeros(2), x, x.copy(), np.ones_like(x, dtype=bool), 100.0)


def test_a_jump_in_the_data_is_dropped_end_to_end(tmp_path: pathlib.Path) -> None:
    body = skeleton(frames=40)
    body.x[20, body.column("left_wrist")] += 400.0
    processed = process(tmp_path, clip(body))
    column = processed.clip.column("left_wrist")
    # The jump is not a measurement, so it is not in ``outliers``; the frame is
    # then bridged, because its neighbours are one frame away on either side.
    assert processed.gated[20, column] and not processed.outliers[20, column]
    assert processed.filled[20, column] and processed.valid[20, column]
    assert processed.valid[19, column] and processed.valid[21, column]
    # The model really did report the teleported position; it is in the raw columns.
    assert processed.clip.x[20, column] == body.x[20, column]


# --------------------------------------------------------------------------- #
# Step 4: gap fill
# --------------------------------------------------------------------------- #


def fill_case(mask: Sequence[bool], values: Sequence[float], t_ms: Sequence[int]):
    """Run :func:`fill_gaps` over one joint with the given validity and positions."""
    valid = np.array(mask, dtype=bool)[:, None]
    x = np.array(values, dtype=np.float64)[:, None]
    return pp.fill_gaps(np.asarray(t_ms, dtype=np.float64) / 1000.0, x, x.copy(), valid)


def test_a_short_gap_is_interpolated_and_marked_filled() -> None:
    t_ms = vfr_times(33, 33, 33)  # 0, 33, 66, 99 ms
    x, _, valid, filled = fill_case([True, False, True, True], [0.0, 0.0, 66.0, 99.0], t_ms)
    assert valid[:, 0].tolist() == [True, True, True, True]
    assert filled[:, 0].tolist() == [False, True, False, False]
    assert x[1, 0] == pytest.approx(33.0)  # half the way in time, not in frames
    assert x[2, 0] == pytest.approx(66.0)


def test_a_long_gap_is_left_alone() -> None:
    t_ms = [0, 50, 100, 200, 300]  # 0.3 s between the two samples that bracket it
    x, _, valid, filled = fill_case(
        [True, False, False, False, True], [0.0, 1.0, 2.0, 3.0, 300.0], t_ms
    )
    assert valid[:, 0].tolist() == [True, False, False, False, True]
    assert not filled[:, 0].any()
    assert np.isnan(x[1:4, 0]).all()


def test_the_gap_boundary_is_the_constant() -> None:
    """Exactly :data:`postprocess.MAX_GAP_S` between the two samples is filled."""
    _, _, valid, filled = fill_case([True, False, True], [0.0, 0.0, 200.0], [0, 100, 200])
    assert valid[:, 0].all() and filled[1, 0]
    x, _, valid, filled = fill_case([True, False, True], [0.0, 0.0, 201.0], [0, 100, 201])
    assert valid[:, 0].tolist() == [True, False, True]
    assert not filled[:, 0].any()
    assert np.isnan(x[1, 0])


def test_a_track_does_not_start_or_end_with_an_invention() -> None:
    """Nothing to interpolate between means nothing is written."""
    x, _, valid, filled = fill_case(
        [False, False, True, True], [5.0, 6.0, 7.0, 8.0], [0, 33, 66, 99]
    )
    assert valid[:, 0].tolist() == [False, False, True, True]
    assert not filled[:, 0].any()
    assert np.isnan(x[0, 0])


def test_fill_gaps_interpolates_in_time_not_in_frames() -> None:
    """Variable frame rate: 100 ms in, 100 ms out, is half the gap, not a third of it."""
    t_ms = [0, 30, 60, 100, 130, 200]
    x, _, _, _ = fill_case(
        [True, False, False, False, False, True], [0.0, 0.0, 0.0, 0.0, 0.0, 200.0], t_ms
    )
    assert x[3, 0] == pytest.approx(100.0)  # the 100 ms sample, not 4/5 of the way
    assert x[1, 0] == pytest.approx(30.0)


def test_fill_gaps_checks_its_arguments() -> None:
    x = np.zeros((2, 1))
    with pytest.raises(ValueError, match="t_seconds"):
        pp.fill_gaps(np.zeros(3), x, x.copy(), np.ones_like(x, dtype=bool))


def test_a_gap_in_the_data_is_bridged_end_to_end(tmp_path: pathlib.Path) -> None:
    clip_ = clip()
    body = clip_.skeleton
    body.x[10:14, body.column("left_wrist")] = math.nan
    body.y[10:14, body.column("left_wrist")] = math.nan
    processed = process(tmp_path, clip_)
    column = processed.clip.column("left_wrist")
    assert processed.filled[10:14, column].all()
    assert processed.valid[10:14, column].all()
    assert processed.x[12, column] == pytest.approx(
        (processed.clip.x[9, column] + processed.clip.x[14, column]) / 2.0, rel=0.2
    )
    assert not processed.filled[9, column] and not processed.filled[14, column]


def test_a_long_gap_in_the_data_stays_nan_end_to_end(tmp_path: pathlib.Path) -> None:
    clip_ = clip()
    body = clip_.skeleton
    body.x[10:25, body.column("left_wrist")] = math.nan
    body.y[10:25, body.column("left_wrist")] = math.nan
    processed = process(tmp_path, clip_)
    column = processed.clip.column("left_wrist")
    assert not processed.valid[10:25, column].any()
    assert not processed.filled[10:25, column].any()
    assert np.isnan(processed.x[10:25, column]).all()


# --------------------------------------------------------------------------- #
# Step 5: One-Euro smoothing
# --------------------------------------------------------------------------- #


def noisy_ramp(
    seconds: float = 8.0, fps: float = 30.0, speed: float = 1.0, sigma: float = 0.02, seed: int = 7
):
    """A joint creeping along at ``speed`` body lengths a second, with noise on top.

    Returned in body lengths, so the caller can scale it into whatever pixel body
    length its clip has.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fps)) / fps
    return t, speed * t + rng.normal(0.0, sigma, t.size)


def test_the_first_sample_passes_through_and_a_reset_starts_again() -> None:
    one_euro = pp.OneEuroFilter()
    assert one_euro(4.0, 0.0) == 4.0
    smoothed = one_euro(9.0, 0.1)
    assert 4.0 < smoothed < 9.0
    one_euro.reset()
    assert one_euro(-1.0, 0.2) == -1.0


def test_the_filter_needs_positive_cutoffs() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        pp.OneEuroFilter(min_cutoff=0.0)


def test_one_euro_cuts_the_jitter_of_a_slow_trajectory() -> None:
    """A still-ish joint has to get smoother, and it must not be dragged off course."""
    t, truth = noisy_ramp()
    noisy = truth + np.random.default_rng(3).normal(0.0, 0.02, truth.size)
    one_euro = pp.OneEuroFilter()
    smooth = np.array([one_euro(value, when) for value, when in zip(noisy, t, strict=True)])
    raw_steps = np.diff(noisy)
    smooth_steps = np.diff(smooth)
    assert smooth_steps.std() < 0.5 * raw_steps.std()
    # A second of settling first: the filter's lag is a time constant, not a step.
    settled = t > 1.0
    assert np.abs(smooth - truth)[settled].mean() < 0.1


def test_smooth_track_keeps_a_still_joint_where_it_is() -> None:
    """Smoothing must not shrink a body: a constant track stays a constant track."""
    t = np.arange(60) / 30.0
    x = np.full((60, 1), 250.0)
    smooth_x, smooth_y = pp.smooth_track(t, x, x.copy(), np.ones_like(x, dtype=bool), 100.0)
    assert smooth_x == pytest.approx(np.full((60, 1), 250.0))
    assert smooth_y == pytest.approx(np.full((60, 1), 250.0))


def test_smooth_track_restarts_after_a_gap() -> None:
    """Across a gap the filter must not remember: the first sample back is itself."""
    frames = 20
    t = np.arange(frames) / 30.0
    x = np.zeros((frames, 1))
    valid = np.ones((frames, 1), dtype=bool)
    x[:8] = 0.0
    x[8:] = 5.0  # five body lengths away, which the filter would take many frames to reach
    valid[8:10] = False
    smooth, _ = pp.smooth_track(t, x, x.copy(), valid, 100.0)
    assert smooth[10, 0] == pytest.approx(5.0)
    assert np.isnan(smooth[8:10, 0]).all()
    # Without the restart the first sample back would be dragged towards the old run.
    assert smooth[10, 0] != pytest.approx(0.0)


def test_smooth_track_filters_in_body_lengths_not_in_pixels() -> None:
    """The same motion at two pixel scales: the same lag and the same jitter, in L.

    A cutoff of 1 Hz has to mean the same thing for a 100 px body and a 400 px one,
    which is the whole reason the filter runs on positions divided by ``L``.
    """
    frames = 240
    t = np.arange(frames) / 30.0
    truth = t  # one body length a second, measured in body lengths
    raw = truth + np.random.default_rng(5).normal(0.0, 0.01, frames)
    measured = {}
    for length in (100.0, 400.0):
        smooth, _ = pp.smooth_track(
            t,
            (raw * length)[:, None],
            (raw * length)[:, None],
            np.ones((frames, 1), dtype=bool),
            length,
        )
        settled = t > 1.0
        back = smooth[settled, 0] / length
        measured[length] = (
            float(np.abs(back - truth[settled]).mean()),  # the lag, in body lengths
            float(np.diff(back).std() / np.diff(raw[settled]).std()),  # jitter left
        )
        assert measured[length][0] < 0.1
    assert measured[100.0] == pytest.approx(measured[400.0], rel=1e-9)


def test_smooth_track_checks_its_arguments() -> None:
    x = np.zeros((3, 1))
    with pytest.raises(ValueError, match="body_length"):
        pp.smooth_track(np.zeros(3), x, x.copy(), np.ones_like(x, dtype=bool), math.nan)
    with pytest.raises(ValueError, match="same"):
        pp.smooth_track(np.zeros(3), x, x.copy(), np.ones((3, 2), dtype=bool), 100.0)


def test_the_processed_track_is_smoother_than_the_raw_one_end_to_end(
    tmp_path: pathlib.Path,
) -> None:
    body = skeleton(frames=180)
    rng = np.random.default_rng(11)
    for joint in pp.HOLD_LIKE_JOINTS:
        column = body.column(joint)
        body.x[:, column] += rng.normal(0.0, 1.0, body.frames)
        body.y[:, column] += rng.normal(0.0, 1.0, body.frames)
    processed = process(tmp_path, clip(body))
    columns = [processed.clip.column(joint) for joint in pp.HOLD_LIKE_JOINTS]
    raw = np.diff(processed.clip.x[:, columns], axis=0)
    smooth = np.diff(processed.x[:, columns], axis=0)
    assert smooth.std() < 0.5 * raw.std()
    # ... and it has not wandered: the track stays within a couple of pixels of it.
    assert np.abs(processed.x[:, columns] - processed.clip.x[:, columns]).max() < 5.0


# --------------------------------------------------------------------------- #
# The output parquet
# --------------------------------------------------------------------------- #


def test_the_output_keeps_the_input_schema_and_adds_four_columns(
    tmp_path: pathlib.Path,
) -> None:
    clip_ = clip()
    write_clip(tmp_path, clip_)
    processed = pp.process_clip(read_back(tmp_path))
    out = pp.processed_table(processed)
    assert list(out.columns) == [*clip_.table().columns, *pp.EXTRA_COLUMNS]
    assert len(out) == len(clip_.table())
    assert out["frame_idx"].tolist() == clip_.table()["frame_idx"].tolist()
    assert out["joint"].tolist() == clip_.table()["joint"].tolist()
    # Every input column that is not a coordinate is passed through untouched.
    for column in (
        "t_ms",
        "z",
        "visibility",
        "rotated",
        "detected",
        athlete.SCORE_COLUMN,
        athlete.N_PEOPLE_COLUMN,
        athlete.CONTACT_COLUMN,
        athlete.CONTACT_REASON_COLUMN,
    ):
        assert out[column].tolist() == clip_.table()[column].tolist()


def test_x_raw_is_what_the_model_said(tmp_path: pathlib.Path) -> None:
    clip_ = clip()
    clip_.contact[5:15] = True  # long enough that nothing bridges it
    clip_.visibility[12, 3] = 0.1
    write_clip(tmp_path, clip_)
    processed = pp.process_clip(read_back(tmp_path))
    out = pp.processed_table(processed)
    assert out["x_raw"].tolist() == clip_.table()["x"].tolist()
    assert out["y_raw"].tolist() == clip_.table()["y"].tolist()
    # Gated-out samples still carry the model's own number in the raw columns.
    assert not out.loc[out["frame_idx"] == 5, "valid"].any()
    assert not out.loc[out["frame_idx"] == 5, "filled"].any()
    assert not out.loc[out["frame_idx"] == 5, "x"].notna().any()
    assert out.loc[out["frame_idx"] == 5, "x_raw"].notna().all()


def test_valid_is_exactly_where_there_is_a_position(tmp_path: pathlib.Path) -> None:
    clip_ = clip()
    clip_.contact[5:15] = True
    write_clip(tmp_path, clip_)
    out = pp.processed_table(pp.process_clip(read_back(tmp_path)))
    assert (out["valid"] == out["x"].notna()).all()
    assert (out["valid"] == out["y"].notna()).all()
    assert (out["filled"] <= out["valid"]).all()


def test_a_vision_style_clip_without_the_athlete_columns_still_loads(
    tmp_path: pathlib.Path,
) -> None:
    """The single-person schema has no ``trainer_contact``: the loader says "no contact"."""
    clip_ = clip()
    table = clip_.table()[list(pm.PARQUET_COLUMNS)]
    root = tmp_path / "auto"
    root.mkdir(parents=True)
    table.to_parquet(root / "0123456789ab.parquet", index=False)
    loaded = pp.read_clip(root / "0123456789ab.parquet")
    assert not loaded.trainer_contact.any()
    processed = pp.process_clip(loaded)
    assert processed.usable
    assert processed.valid.all()


def test_the_output_draws_in_the_overlay_unchanged(tmp_path: pathlib.Path) -> None:
    """The point of keeping the schema: ``handstand.overlay`` reads the processed file."""
    clip_ = clip()
    clip_.contact[10:20] = True
    write_clip(tmp_path, clip_)
    out = tmp_path / "processed.parquet"
    pp.processed_table(pp.process_clip(read_back(tmp_path))).to_parquet(out, index=False)
    panel = overlay.load_panel("processed", out)
    raw = overlay.load_panel("raw", tmp_path / "auto" / "0123456789ab.parquet")
    assert panel.frame_count == raw.frame_count
    assert panel.max_frame_idx == raw.max_frame_idx
    assert overlay.draw_pose(
        np.zeros((64, 64, 3), dtype=np.uint8), panel.frame(0).joints_xy, panel.frame(0).visibility
    ).shape == (64, 64, 3)
    # A contact frame is drawn on the raw panel and left empty on the processed one.
    assert raw.frame(10).trainer_contact
    assert panel.frame(10).trainer_contact  # the flag is passed through
    assert all(math.isnan(xy[0]) for xy in panel.frame(10).joints_xy.values())


def test_read_clip_folds_the_long_schema_into_arrays(tmp_path: pathlib.Path) -> None:
    clip_ = clip()
    write_clip(tmp_path, clip_)
    loaded = read_back(tmp_path)
    assert loaded.frames == clip_.skeleton.frames
    assert loaded.joint_count == len(clip_.skeleton.joints)
    assert loaded.x == pytest.approx(clip_.skeleton.x)
    assert loaded.y == pytest.approx(clip_.skeleton.y)
    assert loaded.t_ms.tolist() == clip_.skeleton.t_ms.tolist()
    assert loaded.t_seconds.tolist() == [t / 1000.0 for t in clip_.skeleton.t_ms]
    assert loaded.detected.all() and not loaded.trainer_contact.any()


def test_read_clip_sorts_a_shuffled_file(tmp_path: pathlib.Path) -> None:
    """Rows are matched by ``(frame_idx, joint)``, never by position."""
    clip_ = clip()
    path = write_clip(tmp_path, clip_)
    shuffled = pd.read_parquet(path).sample(frac=1.0, random_state=1)
    shuffled.to_parquet(path, index=False)
    loaded = read_back(tmp_path)
    assert loaded.x == pytest.approx(clip_.skeleton.x)
    assert loaded.joints == clip_.skeleton.joints


def test_read_clip_rejects_a_file_it_cannot_read(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "auto"
    root.mkdir(parents=True)
    good = clip().table()

    (root / "nocol.parquet").unlink(missing_ok=True)
    without = good.drop(columns=["visibility"])
    without.to_parquet(root / "nocol.parquet", index=False)
    with pytest.raises(ValueError, match="missing column"):
        pp.read_clip(root / "nocol.parquet")

    short = good[~((good["frame_idx"] == 0) & (good["joint"] == "nose"))]
    short.to_parquet(root / "short.parquet", index=False)
    with pytest.raises(ValueError, match="once each"):
        pp.read_clip(root / "short.parquet")

    swapped = good.copy()
    mask = (swapped["frame_idx"] == 0) & (swapped["joint"] == "nose")
    swapped.loc[mask, "joint"] = "left_ear"  # the same row count, a different joint
    swapped.to_parquet(root / "swapped.parquet", index=False)
    with pytest.raises(ValueError, match="once"):
        pp.read_clip(root / "swapped.parquet")

    odd = good.copy()
    odd["joint"] = odd["joint"].replace({"nose": "nose_tip"})
    odd.to_parquet(root / "odd.parquet", index=False)
    with pytest.raises(ValueError, match="not part of the keypoint schema"):
        pp.read_clip(root / "odd.parquet")

    pd.DataFrame(columns=good.columns).to_parquet(root / "empty.parquet", index=False)
    with pytest.raises(ValueError, match="no rows"):
        pp.read_clip(root / "empty.parquet")


def test_read_clip_rejects_a_frame_whose_rows_disagree_about_their_time(
    tmp_path: pathlib.Path,
) -> None:
    clip_ = clip()
    path = write_clip(tmp_path, clip_)
    table = pd.read_parquet(path)
    rows = table.index[table["frame_idx"] == 4]
    table.loc[rows[:3], "t_ms"] = 999  # only some of the frame's rows
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="disagree about t_ms"):
        pp.read_clip(path)


# --------------------------------------------------------------------------- #
# Holding, jitter and the sidecar
# --------------------------------------------------------------------------- #


def test_hold_like_frames_reads_the_inverted_body(tmp_path: pathlib.Path) -> None:
    """The synthetic body has its hands below its hips, so every frame is a hold."""
    body = skeleton()
    valid = np.ones(body.x.shape, dtype=bool)
    assert pp.hold_like_frames(body.x, body.y, valid, body.joints).all()


def test_an_upright_body_is_not_a_hold() -> None:
    """Same body, hands up: the wrists are above the ankles, which is not a handstand."""
    body = skeleton()
    flipped = body.y * -1.0
    valid = np.ones(body.x.shape, dtype=bool)
    assert not pp.hold_like_frames(body.x, flipped, valid, body.joints).any()


def test_a_frame_with_one_visible_wrist_is_still_classified() -> None:
    body = skeleton(frames=3)
    valid = np.ones(body.x.shape, dtype=bool)
    valid[:, body.column("left_wrist")] = False
    assert pp.hold_like_frames(body.x, body.y, valid, body.joints).all()


def test_a_frame_with_no_ankle_is_not_classified() -> None:
    body = skeleton(frames=3)
    valid = np.ones(body.x.shape, dtype=bool)
    valid[:, [body.column("left_ankle"), body.column("right_ankle")]] = False
    assert not pp.hold_like_frames(body.x, body.y, valid, body.joints).any()


def test_hold_like_frames_needs_both_sides_of_the_body() -> None:
    x, y = np.zeros((2, 4)), np.zeros((2, 4))
    assert not pp.hold_like_frames(x, y, np.ones((2, 4), dtype=bool), ("left_wrist",)).any()


def test_median_jitter_l_measures_the_step_it_is_given() -> None:
    t = np.arange(30) / 30.0
    x = np.stack([np.arange(30.0) * 30.0] * 2, axis=1)  # 30 px a frame
    y = np.zeros_like(x)
    jitter = pp.median_jitter_l(t, x, y, np.ones_like(x, dtype=bool), 100.0)
    assert jitter == pytest.approx(0.3)
    doubled = pp.median_jitter_l(t, x, y, np.ones_like(x, dtype=bool), 50.0)
    assert doubled == pytest.approx(0.6)


def test_median_jitter_l_needs_a_pair_to_measure() -> None:
    x = np.zeros((3, 1))
    assert math.isnan(
        pp.median_jitter_l(np.arange(3) / 30.0, x, x.copy(), np.zeros_like(x, dtype=bool), 100.0)
    )
    assert math.isnan(
        pp.median_jitter_l(np.arange(3) / 30.0, x, x.copy(), np.ones_like(x, dtype=bool), 0.0)
    )
    with pytest.raises(ValueError, match="same"):
        pp.median_jitter_l(np.arange(3) / 30.0, x, x.copy(), np.zeros((3, 2), dtype=bool), 100.0)


def test_the_sidecar_says_what_the_run_did(tmp_path: pathlib.Path) -> None:
    clip_ = clip()
    clip_.contact[5:15] = True  # long enough that the gap fill leaves it alone
    write_clip(tmp_path, clip_)
    processed = pp.process_clip(read_back(tmp_path))
    stats = pp.measure("0123456789ab", "mediapipe", "auto", processed, 0.5)
    sidecar = stats.sidecar("keypoints/mediapipe_athlete/auto/0123456789ab.parquet")

    assert sidecar["clip_id"] == "0123456789ab"
    assert sidecar["rotate"] == "auto"
    assert sidecar["source"] == "mediapipe"
    assert sidecar["usable"] is True
    assert sidecar["unusable_reason"] is None
    assert sidecar["body_length_px"] == pytest.approx(340.0, rel=0.01)
    assert set(sidecar["body_length_parts_px"]) == {"torso", "thigh", "shin"}
    assert sidecar["parameters"]["one_euro"] == {
        "min_cutoff": pp.MIN_CUTOFF,
        "beta": pp.BETA,
        "d_cutoff": pp.D_CUTOFF,
    }
    assert sidecar["parameters"]["max_gap_s"] == pp.MAX_GAP_S
    # Counts add up: every sample is measured, interpolated, or lost — and the
    # loss says which rule lost it.
    assert (
        sidecar["measured_sample_count"]
        + sidecar["filled_sample_count"]
        + sidecar["unfilled_sample_count"]
        == sidecar["total_sample_count"]
    )
    assert sidecar["unfilled_sample_count"] == (
        sidecar["gated_out_sample_count"] + sidecar["outlier_sample_count"]
    )
    assert sidecar["valid_sample_count"] == (
        sidecar["measured_sample_count"] + sidecar["filled_sample_count"]
    )
    assert sidecar["gated_out_sample_count"] == 10 * clip_.skeleton.x.shape[1]
    assert sidecar["hold_like_frame_count"] == clip_.skeleton.frames - 10
    assert 0.0 < sidecar["pct_valid_tracked_samples_hold_like"] <= 100.0
    assert json.loads(json.dumps(sidecar)) == sidecar  # no NaN in the JSON


def test_the_sidecar_of_an_unusable_clip_says_why(tmp_path: pathlib.Path) -> None:
    clip_ = clip()
    clip_.detected[:] = False
    write_clip(tmp_path, clip_)
    stats = pp.measure(
        "0123456789ab", "mediapipe", "auto", pp.process_clip(read_back(tmp_path)), 0.1
    )
    sidecar = stats.sidecar("keypoints/mediapipe_athlete/auto/0123456789ab.parquet")
    assert sidecar["usable"] is False
    assert sidecar["unusable_reason"]
    assert sidecar["body_length_px"] is None
    assert sidecar["jitter_l_raw"] is None
    assert "unusable" in stats.line()


def test_the_counts_add_up_over_a_whole_clip(tmp_path: pathlib.Path) -> None:
    """One invariant over the real pipeline: nothing is lost and nothing is invented."""
    clip_ = clip()
    clip_.contact[5:9] = True
    clip_.visibility[20:24, 2] = 0.1
    body = clip_.skeleton
    body.x[30:33, body.column("right_ankle")] = math.nan
    body.y[30:33, body.column("right_ankle")] = math.nan
    write_clip(tmp_path, clip_)
    processed = pp.process_clip(read_back(tmp_path))
    stats = pp.measure("0123456789ab", "mediapipe", "auto", processed, 0.1)
    assert stats.total_samples == clip_.skeleton.frames * clip_.skeleton.x.shape[1]
    assert (
        stats.measured_samples + stats.filled_samples + stats.unfilled_samples
        == stats.total_samples
    )
    assert stats.unfilled_samples == stats.gated_out_samples + stats.outlier_samples
    assert stats.valid_samples == int(processed.valid.sum())
    assert stats.filled_samples == int(processed.filled.sum())
    assert stats.measured_samples == stats.valid_samples - stats.filled_samples
    assert 0.0 < stats.tracked_valid_percent <= 100.0
    # A contact frame has no visible joints, so it cannot be classified as a hold
    # or as anything else; every other frame can.
    assert stats.hold_like_frames == clip_.skeleton.frames - int(clip_.contact.sum())


def test_the_summary_answers_the_dataset_questions(tmp_path: pathlib.Path) -> None:
    clip_ = clip()
    write_clip(tmp_path, clip_)
    good = pp.measure("aaa", "mediapipe", "auto", pp.process_clip(read_back(tmp_path)), 0.1)
    broken = clip()
    broken.detected[:] = False
    write_clip(tmp_path, broken, clip_id="bbb")
    bad = pp.measure("bbb", "mediapipe", "auto", pp.process_clip(read_back(tmp_path, "bbb")), 0.1)
    text = pp.summary([good, bad])
    assert "clips=2 usable=1 unusable=1" in text
    assert "valid samples" in text
    assert "valid in holds" in text
    assert "jitter (L)" in text
    assert "body length (px)" in text
    assert "bbb (" in text  # the unusable clip is named with its reason
    assert pp.summary([]).startswith("summary clips=0")


def test_percentages_of_nothing_are_zero_not_a_crash() -> None:
    empty = pp.ClipStats(
        clip_id="x",
        source="mediapipe",
        rotate="auto",
        frame_count=0,
        joint_count=0,
        usable=False,
        unusable_reason="nothing",
        body_length=math.nan,
        body_length_parts={},
        body_length_frames={},
        valid_samples=0,
        total_samples=0,
        tracked_valid_samples=0,
        tracked_total_samples=0,
        hold_like_frames=0,
        hold_like_valid_samples=0,
        hold_like_total_samples=0,
        gated_out_samples=0,
        outlier_samples=0,
        unfilled_samples=0,
        filled_samples=0,
        jitter_raw_l=math.nan,
        jitter_processed_l=math.nan,
        runtime_seconds=0.0,
    )
    assert empty.valid_percent == 0.0
    assert empty.hold_like_valid_percent == 0.0
    assert math.isnan(empty.jitter_reduction)
    assert "clips=1 usable=0" in pp.summary([empty])


# --------------------------------------------------------------------------- #
# One clip on disk, and the CLI
# --------------------------------------------------------------------------- #


def test_run_clip_writes_both_files_and_no_temporary_ones(tmp_path: pathlib.Path) -> None:
    in_root, out_root = tmp_path / "keypoints" / pp.input_dirname(), tmp_path / "out"
    write_clip(in_root, clip())
    report = pp.run_clip("0123456789ab", rotate_mode="auto", in_root=in_root, out_root=out_root)
    assert not report.skipped
    assert report.stats is not None
    assert report.parquet_path == out_root / "0123456789ab.parquet"
    assert report.json_path.is_file()
    sidecar = json.loads(report.json_path.read_text())
    assert sidecar["clip_id"] == "0123456789ab"
    assert sidecar["source_parquet"] == "keypoints/mediapipe_athlete/auto/0123456789ab.parquet"
    assert sorted(path.name for path in out_root.iterdir()) == [
        "0123456789ab.json",
        "0123456789ab.parquet",
    ]
    panel = overlay.load_panel("processed", report.parquet_path)
    assert panel.frame_count == report.stats.frame_count


def test_run_clip_skips_a_clip_that_is_already_there(tmp_path: pathlib.Path) -> None:
    in_root, out_root = tmp_path / "keypoints" / pp.input_dirname(), tmp_path / "out"
    write_clip(in_root, clip())
    first = pp.run_clip("0123456789ab", rotate_mode="auto", in_root=in_root, out_root=out_root)
    second = pp.run_clip("0123456789ab", rotate_mode="auto", in_root=in_root, out_root=out_root)
    assert not first.skipped and second.skipped
    assert second.stats is None
    assert "skip" in second.line()
    third = pp.run_clip(
        "0123456789ab", rotate_mode="auto", in_root=in_root, out_root=out_root, overwrite=True
    )
    assert not third.skipped


def test_run_clip_names_the_command_that_makes_the_input(tmp_path: pathlib.Path) -> None:
    in_root, out_root = tmp_path / "in", tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="handstand.athlete --rotate auto"):
        pp.run_clip("0123456789ab", rotate_mode="auto", in_root=in_root, out_root=out_root)
    with pytest.raises(FileNotFoundError, match="--source vision"):
        pp.run_clip(
            "0123456789ab", rotate_mode="auto", in_root=in_root, out_root=out_root, source="vision"
        )


def test_run_clip_rejects_an_unknown_mode_or_source(tmp_path: pathlib.Path) -> None:
    in_root, out_root = tmp_path / "in", tmp_path / "out"
    with pytest.raises(ValueError, match="rotate mode"):
        pp.run_clip("x", rotate_mode="180x", in_root=in_root, out_root=out_root)
    with pytest.raises(ValueError, match="unknown source"):
        pp.run_clip("x", rotate_mode="auto", in_root=in_root, out_root=out_root, source="rtmpose")


def test_available_clips_lists_or_takes(tmp_path: pathlib.Path) -> None:
    in_root = tmp_path / "in"
    write_clip(in_root, clip(), clip_id="bbb")
    write_clip(in_root, clip(), clip_id="aaa")
    assert pp.available_clips("auto", in_root) == ["aaa", "bbb"]
    assert pp.available_clips("auto", in_root, ["zzz"]) == ["zzz"]
    with pytest.raises(FileNotFoundError, match="handstand.athlete"):
        pp.available_clips("auto", tmp_path / "nowhere")
    with pytest.raises(ValueError, match="rotate mode"):
        pp.available_clips("sideways", in_root)


def test_the_paths_are_where_the_docs_say(tmp_path: pathlib.Path) -> None:
    assert pp.output_dir(tmp_path) == tmp_path / "processed" / "mediapipe"
    assert pp.output_dir(tmp_path, "vision") == tmp_path / "processed" / "vision"
    assert pp.input_dirname() == "mediapipe_athlete"
    assert pp.input_dirname("vision") == "vision_athlete"
    assert (
        pp.clip_table("abc", "auto", "mediapipe") == "keypoints/mediapipe_athlete/auto/abc.parquet"
    )
    assert pp.PROCESSED_DIRNAME == "processed"


def test_the_cli_wants_to_know_which_clips(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        pp.build_arg_parser().parse_args([])
    with pytest.raises(SystemExit):
        pp.build_arg_parser().parse_args(["--clips", "a", "--all"])
    assert pp.build_arg_parser().parse_args(["--all"]).all_clips is True
    assert pp.build_arg_parser().parse_args(["--clips", "a"]).clips == ["a"]


def test_the_cli_runs_a_batch_and_prints_the_summary(tmp_path, capsys) -> None:
    # No --rotate: the CLI reads the mode `handstand.athlete` defaults to.
    in_root = tmp_path / "keypoints" / pp.input_dirname()
    assert pp.DEFAULT_ROTATE == "best"
    write_clip(in_root, clip(), clip_id="aaa", rotate=pp.DEFAULT_ROTATE)
    write_clip(in_root, clip(), clip_id="bbb", rotate=pp.DEFAULT_ROTATE)
    assert pp.main(["--data", str(tmp_path), "--all"]) == 0
    out = capsys.readouterr().out
    assert "clips=2" in out
    assert "summary clips=2 usable=2 unusable=0" in out
    assert (tmp_path / "processed" / "mediapipe" / "aaa.parquet").is_file()
    assert (tmp_path / "processed" / "mediapipe" / "aaa.json").is_file()


def test_the_cli_limits_and_skips(tmp_path, capsys) -> None:
    in_root = tmp_path / "keypoints" / pp.input_dirname()
    write_clip(in_root, clip(), clip_id="aaa", rotate=pp.DEFAULT_ROTATE)
    write_clip(in_root, clip(), clip_id="bbb", rotate=pp.DEFAULT_ROTATE)
    assert pp.main(["--data", str(tmp_path), "--all", "--limit", "1"]) == 0
    assert (tmp_path / "processed" / "mediapipe" / "aaa.parquet").is_file()
    assert not (tmp_path / "processed" / "mediapipe" / "bbb.parquet").exists()
    assert pp.main(["--data", str(tmp_path), "--all"]) == 0
    assert "skip" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pp.main(["--data", str(tmp_path), "--all", "--limit", "-1"])


def test_the_cli_reports_a_missing_input_directory(tmp_path, capsys) -> None:
    assert pp.main(["--data", str(tmp_path), "--all"]) == 2
    assert "handstand.athlete" in capsys.readouterr().out
    assert pp.main(["--data", str(tmp_path), "--clips", "nope"]) == 1
    assert "fail  nope" in capsys.readouterr().out


def test_the_cli_carries_on_after_a_bad_clip(tmp_path, capsys) -> None:
    in_root = tmp_path / "keypoints" / pp.input_dirname()
    write_clip(in_root, clip(), clip_id="aaa", rotate=pp.DEFAULT_ROTATE)
    (in_root / pp.DEFAULT_ROTATE / "broken.parquet").write_bytes(b"not a parquet")
    assert pp.main(["--data", str(tmp_path), "--all"]) == 1
    out = capsys.readouterr().out
    assert "fail  broken" in out
    assert "summary clips=1 usable=1" in out


def test_the_constants_are_the_documented_ones() -> None:
    assert pp.MIN_VISIBILITY == 0.5
    assert pp.BODY_LENGTH_PERCENTILE == 90.0
    assert pp.MAX_SPEED_L_PER_S == 8.0
    assert pp.MAX_GAP_S == 0.2
    assert (pp.MIN_CUTOFF, pp.BETA, pp.D_CUTOFF) == (1.0, 0.3, 1.0)
    assert pp.SOURCES == ("mediapipe", "vision")
    assert set(pp.CORE_JOINTS) == set(athlete.SHARED_JOINTS) - {
        "left_eye_inner",
        "right_eye_inner",
        "left_eye_outer",
        "right_eye_outer",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
    }
    assert pp.TRACKED_JOINTS == (*pp.CORE_JOINTS, *pp.FOOT_INDEX_JOINTS)
    assert set(pp.CORE_JOINTS) <= set(pm.JOINT_NAMES)
    assert set(pp.FOOT_INDEX_JOINTS) <= set(pm.JOINT_NAMES)
    assert pp.LENGTH_PARTS == ("torso", "thigh", "shin")
    assert [name for name, _, _ in pp.LEG_SEGMENTS] == ["thigh", "shin"]
