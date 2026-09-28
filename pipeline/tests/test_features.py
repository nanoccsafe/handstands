"""Tests for :mod:`handstand.features`.

Nothing here touches the real handstand videos. Most of the file drives the
geometry from poses written by hand **in the body frame** — the wrist midpoint is
the origin, ``v`` is up, both in body lengths — because that is what makes a
feature readable: a line handstand is :data:`LINE`, a pike moves the knees and the
ankles down towards the hands, and a bent arm moves one elbow. A second section
folds the same poses into the processed long parquet that #20 writes and the
phases parquet that #21 writes, so the loader, the CSV, the atomic writes and the
CLI run against exactly what a real run reads.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from handstand import features as ft
from handstand import pose_mediapipe as pm

CLIP_ID = "abc123def456"
#: The clip's body length in pixels. Any positive number works; 300 keeps the
#: coordinates in a comfortable range and makes 0.1 L exactly 30 px.
LENGTH = 300.0
#: The frame interval of the synthetic clips, in milliseconds. Real clips are
#: variable frame rate, so the tests that care about time say so explicitly.
FRAME_MS = 33


# --------------------------------------------------------------------------- #
# A synthetic body, written in the body frame
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Pose:
    """One frame of a synthetic athlete, in the body frame.

    Every number is a body length with the wrist midpoint at ``(0, 0)`` and ``v``
    up, so a held handstand is a column of stations up the image
    (:data:`LINE`) and a failing assertion says which station moved. The two sides
    of a part are the station plus and minus that part's ``spread``, which is what
    a straddle widens and a side view leaves at zero.
    """

    #: The height of each station up the body. An inverted body is upside down, so
    #: the ankles are the highest and the knees sit between the hips and them.
    v_shoulder: float = 0.39
    v_hip: float = 0.75
    v_knee: float = 0.95
    v_ankle: float = 1.15
    #: The horizontal offset of each station from the vertical through the hands:
    #: the stacking errors, which are 0 for a line and non-zero for a stack that
    #: has drifted over one hand.
    u_shoulder: float = 0.0
    u_hip: float = 0.0
    u_knee: float = 0.0
    u_ankle: float = 0.0
    #: The nose: a handstand's head hangs *past* the shoulders towards the floor,
    #: so the nose is below the shoulder line and a little in front of the chest.
    u_nose: float = 0.04
    v_nose: float = 0.30
    #: The elbow of a straight arm is halfway between the shoulder and the hands,
    #: which are the origin: the arm is a straight line through the elbow.
    u_elbow: float = 0.0
    v_elbow: float = 0.195
    #: How far apart the two sides of a part are, in u, per part. Empty is a side
    #: view, where the near and far side overlap; ``{"ankle": 0.5}`` is a straddle.
    spread: Mapping[str, float] = dataclasses.field(default_factory=dict)
    #: How wide the hands and the shoulders are. Both are 0 in a side view, where
    #: both pairs project onto one another, and both are set in the hand_width
    #: tests, which is the only place they are read.
    hand_span: float = 0.0
    shoulder_span: float = 0.0
    #: Joint names to leave unseen, for the tests about one side or one joint
    #: missing.
    hidden: tuple[str, ...] = ()
    #: False for a frame the model saw nobody in, which is every joint at once.
    visible: bool = True

    def stations(self) -> dict[str, tuple[float, float]]:
        """Every joint's ``(u, v)`` by name, both sides and the nose."""
        centres = {
            "shoulder": (self.u_shoulder, self.v_shoulder),
            "elbow": (self.u_elbow, self.v_elbow),
            "hip": (self.u_hip, self.v_hip),
            "knee": (self.u_knee, self.v_knee),
            "ankle": (self.u_ankle, self.v_ankle),
        }
        points: dict[str, tuple[float, float]] = {ft.NOSE: (self.u_nose, self.v_nose)}
        for part, (u, v) in centres.items():
            half = self.spread.get(part, 0.0) / 2.0
            points[f"left_{part}"] = (u - half, v)
            points[f"right_{part}"] = (u + half, v)
        # The wrists are the origin by definition, so their spread is the only
        # thing about them, and only hand_width reads it.
        points["left_wrist"] = (-self.hand_span / 2.0, 0.0)
        points["right_wrist"] = (self.hand_span / 2.0, 0.0)
        half_shoulders = self.shoulder_span / 2.0
        points["left_shoulder"] = (self.u_shoulder - half_shoulders, self.v_shoulder)
        points["right_shoulder"] = (self.u_shoulder + half_shoulders, self.v_shoulder)
        return points

    def mirrored(self) -> Pose:
        """This pose seen from the other side: every ``u`` negated.

        The athlete still faces the same way *anatomically*, so a shape feature
        must read the same number on both, and a directional one must flip.
        """
        return dataclasses.replace(
            self,
            u_shoulder=-self.u_shoulder,
            u_hip=-self.u_hip,
            u_knee=-self.u_knee,
            u_ankle=-self.u_ankle,
            u_nose=-self.u_nose,
            u_elbow=-self.u_elbow,
            spread={part: -width for part, width in self.spread.items()},
            hand_span=-self.hand_span,
        )


#: A held line handstand: every station on the vertical through the hands, every
#: limb straight, the nose a little in front of the chest.
LINE = Pose()

#: A pike: the hips above the shoulders and the legs folded back down towards the
#: hands, which is the shape #21 lets through as a hold and this module calls out.
PIKE = Pose(v_hip=0.95, v_knee=0.55, v_ankle=0.40, v_nose=0.26, v_elbow=0.195)

#: A banana: the hips off the shoulder→ankle line, the knees a third of the way
#: back to it, everything else where a line is.
BANANA = Pose(u_hip=0.08, u_knee=0.03)

#: A bent arm: the elbow forward of the shoulder→wrist line, which in a side view
#: is the only way a straight arm can leave the image plane.
BENT_ARMS = Pose(u_elbow=0.15, v_elbow=0.30)

#: A straddle: the ankles a leg's width apart, the hips where they were.
STRADDLE = Pose(spread={"ankle": 0.5})

#: A front-view handstand, the only geometry in which the two widths
#: :data:`handstand.features.FEATURES` calls a ratio can be read.
FRONTAL = Pose(hand_span=0.24, shoulder_span=0.24)


def track_of(poses: Sequence[Pose]) -> ft.BodyTrack:
    """A pose list as the :class:`handstand.features.BodyTrack` the geometry reads."""
    frames = len(poses)
    joints = tuple(ft.WANTED_JOINTS)
    uv = np.full((frames, len(joints), 2), np.nan)
    for index, pose in enumerate(poses):
        for name, point in pose.stations().items():
            uv[index, joints.index(name)] = point
        if not pose.visible:
            uv[index] = np.nan
        for name in pose.hidden:
            uv[index, joints.index(name)] = np.nan
    return ft.BodyTrack(uv=uv, joints=joints)


def measure(poses: Sequence[Pose]) -> ft.BodyFeatures:
    """The features of a pose list, with no file in the way."""
    return ft.body_features(track_of(poses))


def value(poses: Sequence[Pose], name: str) -> float:
    """One feature of a one-frame pose list."""
    return float(measure(poses).value(name)[0])


# --------------------------------------------------------------------------- #
# The features of a line handstand
# --------------------------------------------------------------------------- #


def test_a_line_handstand_reads_as_one() -> None:
    features = measure([LINE])

    assert bool(features.valid[0])
    for name in ("off_shoulder", "off_hip", "off_knee", "off_ankle"):
        assert features.value(name)[0] == pytest.approx(0.0, abs=1e-9)
    assert features.value("line_deviation")[0] == pytest.approx(0.0, abs=1e-9)
    assert features.value("body_angle")[0] == pytest.approx(0.0, abs=1e-6)
    for name in ("shoulder_angle", "hip_angle", "knee_angle", "elbow_angle"):
        assert features.value(name)[0] == pytest.approx(180.0, abs=0.5)
    assert features.value("banana")[0] == pytest.approx(0.0, abs=1e-9)
    # The head hangs 24 degrees off the spine, which is the relaxed handstand band
    # the docstring quotes, and the legs overlap so the separation is zero.
    assert features.value("head")[0] == pytest.approx(24.0, abs=0.5)
    assert 15.0 < features.value("head")[0] < 30.0
    assert features.value("leg_separation")[0] == pytest.approx(0.0, abs=1e-6)
    # A side view has no shoulder width to divide the hand width by, and the
    # feature says so rather than inventing a ratio.
    assert math.isnan(features.value("hand_width")[0])


def test_every_feature_is_measured_on_every_frame_of_a_clip() -> None:
    features = measure([LINE, PIKE, BANANA])

    assert features.frames == 3
    assert sorted(features.values) == sorted(ft.FEATURE_NAMES)
    for name in ft.FEATURE_NAMES:
        assert features.value(name).shape == (3,)
    assert features.value("hip_angle")[1] < features.value("hip_angle")[0]


def test_a_stack_that_has_drifted_over_one_hand_is_read_as_one() -> None:
    pose = dataclasses.replace(LINE, u_shoulder=-0.06, u_hip=-0.05, u_knee=-0.03, u_ankle=-0.01)

    assert value([pose], "off_shoulder") == pytest.approx(-0.06)
    assert value([pose], "off_hip") == pytest.approx(-0.05)
    assert value([pose], "off_knee") == pytest.approx(-0.03)
    assert value([pose], "off_ankle") == pytest.approx(-0.01)
    # line_deviation is the worst of the four, and the body angle is the same
    # error read as a lean of the whole line.
    assert value([pose], "line_deviation") == pytest.approx(0.06)
    assert value([pose], "body_angle") == pytest.approx(math.degrees(math.atan2(-0.01, 1.15)))


def test_line_deviation_uses_the_stations_the_frame_can_see() -> None:
    # A frame with no ankles is judged on the three stations it has, rather than
    # thrown away: a hold with a trainer's knee in front of the feet is still a
    # hold.
    pose = dataclasses.replace(LINE, u_hip=0.07, hidden=("left_ankle", "right_ankle"))

    assert value([pose], "line_deviation") == pytest.approx(0.07)
    assert math.isnan(value([pose], "body_angle"))
    assert math.isnan(value([pose], "banana"))
    # It is still not a valid frame: the body angle and the banana are both
    # measured against the feet.
    assert not bool(measure([pose]).valid[0])
    # With none of the four stations there is no line at all, and the row says so
    # rather than reporting the best of nothing.
    nothing = dataclasses.replace(
        LINE,
        u_hip=0.07,
        hidden=(
            "left_ankle",
            "right_ankle",
            "left_knee",
            "right_knee",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        ),
    )
    assert math.isnan(value([nothing], "line_deviation"))
    assert not bool(measure([nothing]).valid[0])


# --------------------------------------------------------------------------- #
# The shapes that are not a line
# --------------------------------------------------------------------------- #


def test_a_pike_is_a_pike() -> None:
    features = measure([PIKE])

    assert features.value("hip_angle")[0] < ft.PIKE_HIP_DEG
    assert features.value("hip_angle")[0] < ft.OPEN_SHOULDER_DEG
    # The legs are straight even in a pike, and the arms are straight whatever the
    # hips do, which is what tells a pike from a knee-bent, arm-bent fault.
    assert features.value("knee_angle")[0] == pytest.approx(180.0, abs=0.5)
    assert features.value("elbow_angle")[0] == pytest.approx(180.0, abs=0.5)
    # A pike is a body folded about the hip, not a body off the vertical.
    assert features.value("line_deviation")[0] == pytest.approx(0.0, abs=1e-9)
    assert features.value("banana")[0] == pytest.approx(0.0, abs=1e-9)


def test_a_banana_moves_the_hip_and_is_signed_towards_the_face() -> None:
    features = measure([BANANA])

    assert features.value("off_hip")[0] == pytest.approx(0.08)
    assert features.value("banana")[0] == pytest.approx(0.08, abs=0.01)
    # The nose of this pose is at +u, so the hips displaced to +u are towards the
    # way the athlete faces: positive.
    assert features.value("banana")[0] > 0.0
    # A curve is also a bent hip — the same fault seen from two sides — so the two
    # features overlap and a score has to say which of them it is counting. What
    # the curve is not is a pike: the legs are still long and the body is not
    # folded in half.
    assert features.value("hip_angle")[0] < 180.0
    assert features.value("hip_angle")[0] > 130.0
    assert features.value("knee_angle")[0] > 170.0
    # A bigger curve is what the tolerance is for, and the feature is the number
    # the tolerance is applied to.
    worse = measure([dataclasses.replace(BANANA, u_hip=0.14, u_knee=0.05)])
    assert abs(worse.value("banana")[0]) > ft.BANANA_TOLERANCE_L
    assert not ft.fails_tolerance(
        float(features.value("banana")[0]), ft.BANANA_TOLERANCE_L, ft.EITHER
    )
    # A body that curves the other way is the same fault with the other sign.
    other = measure([BANANA.mirrored()])
    assert other.value("off_hip")[0] == pytest.approx(-0.08)
    assert other.value("banana")[0] == pytest.approx(0.08, abs=0.01)


def test_a_straight_body_leaning_off_the_vertical_is_not_a_banana() -> None:
    # A body that has come out of line without bending at all: every joint, the
    # hands included, is on one straight line leaning off the vertical. Every joint
    # is still straight, the curve is zero, and what sees it is the stacking
    # offsets and the body angle — which is why they are four features and not one.
    def on_the_line(v: float) -> float:
        return 0.18 * v / 1.15

    lean = dataclasses.replace(
        LINE,
        u_shoulder=on_the_line(0.39),
        u_hip=on_the_line(0.75),
        u_knee=on_the_line(0.95),
        u_ankle=0.18,
        u_nose=on_the_line(0.95) + 0.05,
        u_elbow=on_the_line(0.195),
    )
    features = measure([lean])

    assert features.value("banana")[0] == pytest.approx(0.0, abs=1e-9)
    assert features.value("line_deviation")[0] == pytest.approx(0.18)
    assert features.value("body_angle")[0] == pytest.approx(
        math.degrees(math.atan2(0.18, 1.15)), abs=1e-6
    )
    # The stack has come out with it: the shoulders are off the hands too, which is
    # what a rigid lean looks like and what a pike does not do.
    assert features.value("off_shoulder")[0] == pytest.approx(0.061, abs=1e-3)
    for name in ("shoulder_angle", "hip_angle", "knee_angle", "elbow_angle"):
        assert features.value(name)[0] == pytest.approx(180.0, abs=0.5)


def test_a_shape_feature_does_not_depend_on_which_way_the_camera_was() -> None:
    line = measure([LINE])
    banana = measure([BANANA])

    # The offsets are a direction in the image, so a mirrored recording flips them.
    assert measure([BANANA.mirrored()]).value("off_hip")[0] == pytest.approx(-0.08)
    # The shape features are anatomical: same athlete, same number, in either case
    # because the head's sign is not there to flip.
    for name in ("banana", "head"):
        assert banana.value(name)[0] == pytest.approx(
            measure([BANANA.mirrored()]).value(name)[0], abs=1e-6
        )
    assert line.value("banana")[0] == pytest.approx(0.0, abs=1e-9)


def test_a_dropped_head_reads_as_a_head_off_the_spine() -> None:
    neutral = measure([LINE]).value("head")[0]
    # A head dropped to look at the floor between the hands hangs straight down, so
    # the nose is level with the shoulders and well in front of them.
    dropped = measure([dataclasses.replace(LINE, u_nose=0.12, v_nose=0.38)]).value("head")[0]

    assert neutral < ft.HEAD_FLEXION_DEG
    assert dropped == pytest.approx(85.0, abs=1.0)
    assert dropped > ft.HEAD_FLEXION_DEG
    # It is a magnitude, not a signed angle: a head thrown back is the same fault
    # and the same number, because which side of the spine the nose is on is the
    # only facing cue a side view has.
    thrown = measure([dataclasses.replace(LINE, u_nose=-0.12, v_nose=0.38)]).value("head")[0]
    assert thrown == pytest.approx(dropped)
    # No nose, no head and no signed shape: the magnitude is not given up either.
    blind = measure([dataclasses.replace(LINE, hidden=("nose",))])
    assert math.isnan(blind.value("head")[0])
    assert math.isnan(blind.value("banana")[0])
    # The nose is the only joint whose absence does not make the frame invalid.
    assert bool(blind.valid[0])


def test_bent_arms_are_a_severe_fault() -> None:
    features = measure([BENT_ARMS])

    assert features.value("elbow_angle")[0] < ft.BENT_ELBOW_DEG
    assert features.value("elbow_angle")[0] < 120.0
    # The rest of the body is still straight, so the arm angle is the one feature
    # that says anything about this pose.
    assert features.value("hip_angle")[0] == pytest.approx(180.0, abs=0.5)
    assert features.value("line_deviation")[0] == pytest.approx(0.0, abs=1e-9)


def test_a_straddle_is_a_leg_separation_and_not_a_stack_error() -> None:
    features = measure([STRADDLE])

    assert features.value("leg_separation")[0] > ft.SPLIT_DEG
    assert features.value("leg_separation")[0] == pytest.approx(64.0, abs=1.0)
    # The ankles are apart but their midpoint is where a line puts it, so this is
    # not a stacking fault and a scorer must not read it as one.
    assert features.value("off_ankle")[0] == pytest.approx(0.0, abs=1e-9)
    assert features.value("leg_separation")[0] > measure([LINE]).value("leg_separation")[0]


def test_a_split_needs_both_legs() -> None:
    # A split in a side view has one leg in front of the other, so the far leg is
    # the only thing that tells it from a line. One side is still enough for every
    # feature that is a midpoint, so the frame is a good frame and the separation
    # alone is NaN: a zero would be a claim nobody made.
    one_leg = dataclasses.replace(LINE, hidden=("left_ankle",))
    features = measure([one_leg])

    assert math.isnan(features.value("leg_separation")[0])
    assert bool(features.valid[0])
    assert features.value("off_ankle")[0] == pytest.approx(0.0)


def test_hand_width_is_a_ratio_of_two_widths() -> None:
    assert value([FRONTAL], "hand_width") == pytest.approx(1.0)
    assert value([dataclasses.replace(FRONTAL, hand_span=0.12)], "hand_width") == pytest.approx(0.5)
    # It is the one feature that needs both sides, so hiding one is enough to lose
    # it while the angles it shares its joints with survive.
    one_side = dataclasses.replace(FRONTAL, hidden=("right_shoulder",))
    assert math.isnan(measure([one_side]).value("hand_width")[0])
    assert bool(measure([one_side]).valid[0])


# --------------------------------------------------------------------------- #
# Sides, and frames the model cannot see into
# --------------------------------------------------------------------------- #


def test_one_visible_side_is_used_when_the_other_is_not() -> None:
    left_only = dataclasses.replace(
        LINE, hidden=("right_hip", "right_knee", "right_ankle", "right_elbow")
    )
    right_only = dataclasses.replace(
        LINE, hidden=("left_hip", "left_knee", "left_ankle", "left_elbow")
    )
    both = measure([LINE])

    for name in ("off_hip", "off_knee", "off_ankle", "line_deviation", "body_angle"):
        assert measure([left_only]).value(name)[0] == pytest.approx(both.value(name)[0])
        assert measure([right_only]).value(name)[0] == pytest.approx(both.value(name)[0])
    for name in ("shoulder_angle", "hip_angle", "knee_angle", "elbow_angle"):
        assert measure([left_only]).value(name)[0] == pytest.approx(180.0, abs=0.5)
        assert measure([right_only]).value(name)[0] == pytest.approx(180.0, abs=0.5)
    assert bool(measure([left_only]).valid[0])


def test_the_two_sides_are_averaged_where_both_are_there() -> None:
    # One hip drifted: with both sides visible the midpoint is between them, with
    # one side visible it is the side that is there.
    both = dataclasses.replace(
        LINE, u_hip=0.10, spread={"hip": 0.20, "knee": 0.20, "ankle": 0.20}
    )
    left = dataclasses.replace(both, hidden=("right_hip", "right_knee", "right_ankle"))

    assert measure([both]).value("off_hip")[0] == pytest.approx(0.10)
    assert measure([left]).value("off_hip")[0] == pytest.approx(0.0)


def test_a_frame_the_model_cannot_see_into_is_all_nan() -> None:
    features = measure([LINE, dataclasses.replace(LINE, visible=False), LINE])

    assert features.valid.tolist() == [True, False, True]
    for name in ft.FEATURE_NAMES:
        assert math.isnan(float(features.value(name)[1]))
    assert not math.isnan(float(features.value("line_deviation")[0]))


def test_a_frame_without_a_wrist_has_no_body_frame() -> None:
    # Nothing to measure from: the origin is the wrist midpoint, so a frame with no
    # wrist is a frame whose body frame was never placed, and every feature comes
    # out NaN rather than a number measured against an origin of zeros.
    pose = dataclasses.replace(LINE, hidden=("left_wrist", "right_wrist"))
    features = ft.body_features(track_from_pixels([pose]))

    assert not bool(features.valid[0])
    for name in ft.FEATURE_NAMES:
        assert math.isnan(float(features.value(name)[0]))


def test_side_view_is_assumed_until_the_camera_angle_is_measured() -> None:
    assert ft.side_view_assumed(4).tolist() == [True, True, True, True]
    assert "side_view" in ft.feature_table.__doc__


# --------------------------------------------------------------------------- #
# The body frame
# --------------------------------------------------------------------------- #


def to_pixels(
    pose: Pose, *, wrist_x: float = 200.0, wrist_y: float = 500.0, length: float = LENGTH
) -> dict[str, tuple[float, float]]:
    """A pose's body-frame joints as display-frame pixels, for the parquet tests.

    ``u`` is to the right and ``v`` is up, so a point in the body frame is
    ``(wrist_x + u L, wrist_y - v L)`` on the way back to the image.
    """
    return {
        name: (wrist_x + u * length, wrist_y - v * length)
        for name, (u, v) in pose.stations().items()
    }


def track_from_pixels(
    poses: Sequence[Pose],
    *,
    wrist_x: float = 200.0,
    wrist_y: float = 500.0,
    length: float = LENGTH,
) -> ft.BodyTrack:
    """The :class:`BodyTrack` a processed parquet of these poses reads back as."""
    return ft.body_frame_track(
        *_arrays(poses, wrist_x=wrist_x, wrist_y=wrist_y, length=length),
        pm.JOINT_NAMES,
        length,
    )


def _arrays(
    poses: Sequence[Pose],
    *,
    wrist_x: float = 200.0,
    wrist_y: float = 500.0,
    length: float = LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pose list as the ``(x, y, valid)`` arrays of a processed clip."""
    frames = len(poses)
    x = np.full((frames, len(pm.JOINT_NAMES)), np.nan)
    y = np.full((frames, len(pm.JOINT_NAMES)), np.nan)
    valid = np.zeros((frames, len(pm.JOINT_NAMES)), dtype=bool)
    for index, pose in enumerate(poses):
        for name, (point_x, point_y) in to_pixels(
            pose, wrist_x=wrist_x, wrist_y=wrist_y, length=length
        ).items():
            column = pm.JOINT_NAMES.index(name)
            x[index, column] = point_x
            y[index, column] = point_y
            valid[index, column] = pose.visible and name not in pose.hidden
    return x, y, valid


def test_the_body_frame_is_the_wrist_midpoint_in_body_lengths() -> None:
    here = track_from_pixels([LINE])
    there = track_from_pixels([LINE], wrist_x=900.0, wrist_y=120.0)
    twice = track_from_pixels([LINE], length=2.0 * LENGTH)

    # The origin is where the hands are, so moving the hands moves nothing else.
    assert np.allclose(here.uv, there.uv, atol=1e-6, equal_nan=True)
    # The unit is the body length, so a clip of a twice-as-tall athlete measures
    # the same: every offset and angle is the same number.
    assert np.allclose(here.uv, twice.uv, atol=1e-6, equal_nan=True)
    # And the wrists are the origin, in units of L.
    assert here.point("left_wrist")[0] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert here.point("right_wrist")[0] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert here.point("left_ankle")[0] == pytest.approx((0.0, 1.15), abs=1e-9)


def test_the_track_leaves_an_invisible_joint_at_nan_and_a_source_without_one_too() -> None:
    x, y, valid = _arrays([dataclasses.replace(LINE, hidden=("left_hip",))])
    track = ft.body_frame_track(x, y, valid, pm.JOINT_NAMES, LENGTH)

    assert math.isnan(float(track.point("left_hip")[0][0]))
    assert not math.isnan(float(track.point("right_hip")[0][0]))
    # A source that reports a different set of joints answers NaN for the ones it
    # does not have, rather than raising or inventing a position: this is the whole
    # schema a third model would need, and the one module reads whatever columns it
    # has. The wrist is in it, because without one there is no body frame at all.
    names = ("nose", "left_wrist", "left_hip")
    columns = [pm.JOINT_NAMES.index(name) for name in names]
    without = ft.body_frame_track(
        x[:, columns], y[:, columns], valid[:, columns], names, LENGTH
    )
    assert without.point("left_wrist")[0] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert without.point("nose")[0] == pytest.approx((0.04, 0.30))
    assert math.isnan(float(without.point("left_hip")[0][0]))
    assert math.isnan(float(without.point("right_hip")[0][0]))


def test_the_track_refuses_arrays_it_cannot_read() -> None:
    x, y, valid = _arrays([LINE])
    with pytest.raises(ValueError, match="same \\(frames, joints\\) shape"):
        ft.body_frame_track(x, y[:-1], valid, pm.JOINT_NAMES, LENGTH)
    with pytest.raises(ValueError, match="joint columns"):
        ft.body_frame_track(x, y, valid, pm.JOINT_NAMES[:-1], LENGTH)
    with pytest.raises(ValueError, match="positive, finite"):
        ft.body_frame_track(x, y, valid, pm.JOINT_NAMES, 0.0)
    with pytest.raises(ValueError, match="positive, finite"):
        ft.body_frame_track(x, y, valid, pm.JOINT_NAMES, math.nan)


# --------------------------------------------------------------------------- #
# The tables
# --------------------------------------------------------------------------- #


def phases_for(poses: Sequence[Pose], labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """The ``(phase, hold_id)`` columns for a pose list, from a label per frame.

    Holds are numbered from 0 in time order, as :mod:`handstand.phases` numbers
    them, so a stretch of hold frames is a different hold from the one before it.
    """
    phase = np.asarray([str(label) for label in labels], dtype=object)
    numbers: list[int] = []
    current = -1
    for label in labels:
        if label != ft.HOLD_PHASE:
            numbers.append(ft.NO_HOLD)
            continue
        if not numbers or numbers[-1] == ft.NO_HOLD:
            current += 1
        numbers.append(current)
    return phase, np.asarray(numbers, dtype=np.int64)


def extract(
    poses: Sequence[Pose], labels: Sequence[str] | None = None, **kwargs
) -> ft.ClipFeatures:
    """Measure a pose list straight, with no file in the way."""
    frames = len(poses)
    x, y, valid = _arrays(poses)
    times = np.arange(frames, dtype=np.int64) * FRAME_MS
    phase, hold = (
        phases_for(poses, labels)
        if labels is not None
        else (np.full(frames, "unknown", dtype=object), np.full(frames, ft.NO_HOLD, dtype=np.int64))
    )
    return ft.extract_clip(
        CLIP_ID,
        "mediapipe",
        times,
        x,
        y,
        valid,
        pm.JOINT_NAMES,
        kwargs.pop("body_length", LENGTH),
        phase=phase,
        hold_id=hold,
        **kwargs,
    )


def test_the_per_frame_table_carries_the_labels_and_every_feature() -> None:
    features = extract([LINE, PIKE, BANANA], ["hold", "hold", "unknown"])
    table = ft.feature_table(features)

    assert list(table.columns) == [
        "frame_idx",
        "t_ms",
        "phase",
        "hold_id",
        "valid",
        "side_view",
        *ft.FEATURE_NAMES,
    ]
    assert len(table) == features.frames
    assert table["frame_idx"].tolist() == [0, 1, 2]
    assert table["t_ms"].tolist() == [0, FRAME_MS, 2 * FRAME_MS]
    assert table["phase"].tolist() == ["hold", "hold", "unknown"]
    assert table["hold_id"].tolist() == [0, 0, ft.NO_HOLD]
    assert table["valid"].tolist() == [True, True, True]
    assert table["side_view"].tolist() == [True, True, True]
    # The table is the answer, not a summary of it: the pike's hip is folded, the
    # banana's hip is bent by the curve rather than folded.
    assert table["hip_angle"].tolist() == [
        pytest.approx(180.0, abs=0.5),
        pytest.approx(0.0, abs=0.5),
        pytest.approx(153.4, abs=0.5),
    ]
    assert table["line_deviation"].tolist() == [
        pytest.approx(0.0),
        pytest.approx(0.0),
        pytest.approx(0.08),
    ]


def test_the_hold_table_summarises_each_hold() -> None:
    labels = ["hold"] * 4 + ["unknown"] + ["hold"] * 3
    table = ft.hold_summary_table([extract([LINE] * 8, labels)])

    assert list(table.columns) == list(ft.HOLD_SUMMARY_COLUMNS)
    assert len(table) == 2
    assert table["hold_id"].tolist() == [0, 1]
    assert table["hold_frames"].tolist() == [4, 3]
    assert table["valid_frames"].tolist() == [4, 3]
    assert table["hold_start_ms"].tolist() == [0, 5 * FRAME_MS]
    # (end - start) of the timestamps of the hold's own frames.
    assert table["hold_duration_s"].tolist() == [
        pytest.approx(3 * FRAME_MS / 1000, abs=1e-3),
        pytest.approx(2 * FRAME_MS / 1000, abs=1e-3),
    ]
    assert table["line_deviation_median"].tolist() == [0.0, 0.0]
    assert table["hip_angle_median"][0] == pytest.approx(180.0, abs=0.5)
    assert table["hip_angle_iqr"][0] == pytest.approx(0.0, abs=0.5)


def test_a_hold_is_summarised_over_the_frames_that_could_be_measured() -> None:
    # Four hold frames, one of them a frame the model saw nobody in, and one that
    # is a hold frame but not measurable: the medians are over the other three and
    # the row says so.
    poses = [
        LINE,
        dataclasses.replace(LINE, u_hip=0.10),
        LINE,
        dataclasses.replace(LINE, visible=False),
    ]
    features = extract(poses, ["hold"] * 4)
    row = ft.hold_rows(features)[0]

    assert row["hold_frames"] == 4
    assert row["valid_frames"] == 3
    # Three measurable hold frames: 0, 0.10 and 0, so the median is the line and
    # the IQR is the spread of the frames that moved.
    assert row["off_hip_median"] == pytest.approx(0.0)
    assert row["off_hip_iqr"] == pytest.approx(0.05)


def test_the_longest_hold_is_the_one_a_clip_is_ranked_by() -> None:
    labels = ["hold"] * 5 + ["exit"] * 2 + ["hold"] * 9
    features = extract([LINE] * 16, labels)

    assert features.hold_ids() == (0, 1)
    assert features.longest_hold_id() == 1
    assert features.hold_duration_s(0) == pytest.approx(4 * FRAME_MS / 1000)
    assert features.hold_duration_s(1) == pytest.approx(8 * FRAME_MS / 1000)
    assert features.hold_mask(1).tolist() == [False] * 7 + [True] * 9
    assert features.measurable_hold_mask(1).tolist() == [False] * 7 + [True] * 9


def test_a_clip_with_no_hold_has_no_longest_one() -> None:
    features = extract([LINE] * 4, ["pre"] * 4)

    assert features.hold_ids() == ()
    assert features.longest_hold_id() == ft.NO_HOLD
    assert ft.hold_rows(features) == []
    stats = ft.measure(features)
    assert not stats.has_hold
    assert stats.longest_hold_s == 0.0
    assert "n/a" in stats.line()


def test_a_clip_with_no_scale_has_no_features() -> None:
    features = extract([LINE] * 3, ["hold"] * 3, body_length=math.nan)

    assert not features.usable
    assert "no body length" in features.unusable_reason
    assert not bool(features.valid.any())
    for name in ft.FEATURE_NAMES:
        assert np.isnan(features.value(name)).all()
    assert ft.hold_rows(features)[0]["line_deviation_median"] is math.nan


def test_a_frame_the_trainer_is_in_the_way_is_not_measured() -> None:
    features = extract([LINE] * 3, ["hold"] * 3, trainer_contact=np.array([False, True, False]))

    assert features.valid.tolist() == [True, False, True]
    # The numbers of the contact frame are still in the table, as everywhere else
    # in the pipeline: what is not true is that they are the athlete's.
    assert features.value("line_deviation")[1] == pytest.approx(0.0)


def test_extract_refuses_labels_that_are_not_one_per_frame() -> None:
    with pytest.raises(ValueError, match="phase has"):
        extract([LINE] * 3, ["hold"] * 2)
    with pytest.raises(ValueError, match="hold_id has"):
        ft.extract_clip(
            CLIP_ID,
            "mediapipe",
            np.zeros(3, dtype=np.int64),
            *_arrays([LINE] * 3),
            pm.JOINT_NAMES,
            LENGTH,
            hold_id=np.zeros(2, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="trainer_contact"):
        extract([LINE] * 3, ["hold"] * 3, trainer_contact=np.array([False, True]))


# --------------------------------------------------------------------------- #
# What a run reports
# --------------------------------------------------------------------------- #


def straight_report(clip_id: str, line_deviation: float, *, banana: float = 0.0) -> ft.ClipStats:
    """A clip whose longest hold's medians are the numbers a test wants to rank."""
    medians = {name: 0.0 for name in ft.FEATURE_NAMES}
    medians["line_deviation"] = line_deviation
    medians["banana"] = banana
    values = {name: np.asarray([medians[name]]) for name in ft.FEATURE_NAMES}
    return ft.ClipStats(
        clip_id=clip_id,
        source="mediapipe",
        frames=100,
        valid_frames=100,
        hold_count=1,
        hold_frames=50,
        measured_hold_frames=50,
        longest_hold_id=0,
        longest_hold_s=3.0,
        hold_values=values,
        longest_medians=medians,
    )


def test_measure_reads_the_numbers_the_summary_ranks_by() -> None:
    # Two holds, the second one the longer: a clip is ranked by its longest hold and
    # not by its first, which is the case a state machine's output makes easy to
    # get wrong.
    labels = ["pre"] * 4 + ["hold"] * 6 + ["exit"] * 4 + ["hold"] * 12
    stats = ft.measure(extract([dataclasses.replace(LINE, u_hip=0.05)] * 26, labels), 0.25)

    assert stats.frames == 26
    assert stats.valid_frames == 26
    assert stats.hold_count == 2
    assert stats.hold_frames == 18
    assert stats.measured_hold_frames == 18
    assert stats.longest_hold_id == 1
    assert stats.longest_hold_s == pytest.approx(11 * FRAME_MS / 1000)
    assert stats.longest_medians["line_deviation"] == pytest.approx(0.05)
    # Every hold frame of every feature is pooled, both holds, not just the longest.
    assert stats.hold_values["line_deviation"].size == 18
    assert "holds=2" in stats.line()
    assert "line=0.050" in stats.line()


def test_the_summary_answers_the_dataset_questions() -> None:
    def clip(clip_id: str, poses: Sequence[Pose], labels: Sequence[str], **kwargs) -> ft.ClipStats:
        return ft.measure(dataclasses.replace(extract(poses, labels, **kwargs), clip_id=clip_id))

    reports = [
        straight_report("aaa000000000", 0.01),
        straight_report("bbb111111111", 0.02, banana=0.05),
        straight_report("ccc222222222", 0.30, banana=-0.20),
        clip("ddd333333333", [LINE] * 4, ["pre"] * 4),
        # A clip with no scale is labelled unknown by #21 as well, so it has no
        # hold to report and no median to rank.
        clip("eee444444444", [LINE] * 4, ["unknown"] * 4, body_length=math.nan),
    ]
    text = ft.summary(reports)

    assert "clips=5" in text
    assert "usable=4" in text
    assert "with_hold=3" in text
    # The per-feature medians are over every hold frame of the dataset.
    assert "line_deviation" in text
    assert "iqr=" in text
    # The straightest line and the biggest curve, with their holds.
    assert "aaa000000000  0.010  hold 0  3.0s" in text
    assert "ccc222222222  0.300" in text
    assert "best line" in text
    assert "most banana" in text
    # The targets are counted, so a stage that scores against them can see whether
    # they discriminate: one of these three clips misses the line target, none is a
    # pike, and the one with the biggest curve misses the curve target.
    assert "line_deviation  over 0.1 L" in text
    assert "hip_angle       under 165 deg" in text
    assert "banana          over 0.1 L" in text
    assert "no hold" in text
    assert "ddd333333333" in text
    assert "unusable clips" in text
    assert "eee444444444" in text
    assert ft.summary([]) == "summary clips=0"


def test_the_rankings_skip_the_clips_that_have_nothing_to_rank() -> None:
    reports = [
        straight_report("aaa000000000", 0.05),
        straight_report("bbb111111111", 0.02),
        straight_report("ccc222222222", 0.01),
        straight_report("ddd333333333", 0.03),
        straight_report("eee444444444", 0.04),
        straight_report("fff555555555", 0.06),
        straight_report("ggg666666666", 0.07),
        ft.measure(extract([LINE] * 4, ["pre"] * 4)),
    ]
    ranked = ft.ranked(reports, "line_deviation", largest=False)

    assert [report.clip_id for report in ranked] == [
        "ccc222222222",
        "bbb111111111",
        "ddd333333333",
        "eee444444444",
        "aaa000000000",
    ]
    # The list is capped at what a person reads; the whole table is a filter away.
    assert len(ft.ranked(reports, "line_deviation", largest=True)) == ft.LISTED_CLIPS
    # The biggest curve is ranked by its magnitude, whichever way it curves.
    curved = [straight_report("aaa", 0.01, banana=-0.2), straight_report("bbb", 0.01, banana=0.1)]
    assert [report.clip_id for report in ft.ranked(curved, "banana", absolute=True)] == [
        "aaa",
        "bbb",
    ]
    assert [report.clip_id for report in ft.ranked(curved, "banana")] == ["bbb", "aaa"]


def test_a_plot_panel_draws_the_line_at_which_a_feature_becomes_a_fault() -> None:
    # The tolerance rather than the ideal: a joint angle's ideal is 180 degrees and
    # its data is nowhere near it, so a panel drawn to the ideal cannot show the
    # trace. The label says which side of the line is the fault.
    def of(name: str) -> ft.Feature:
        return next(feature for feature in ft.FEATURES if feature.name == name)

    lines, test = ft.fault_line(of("hip_angle"))
    assert lines == (ft.PIKE_HIP_DEG,)
    assert test == f"under {ft.PIKE_HIP_DEG:g} deg"
    # A feature whose fault is a magnitude either side of the target draws both.
    assert ft.fault_line(of("body_angle"))[0] == (-ft.BODY_ANGLE_TOL_DEG, ft.BODY_ANGLE_TOL_DEG)
    # The four offsets have no tolerance and draw the ideal they are measured against.
    lines, test = ft.fault_line(of("off_hip"))
    assert lines == (0.0,)
    assert "target 0 L" in test
    for feature in ft.FEATURES:
        drawn, label = ft.fault_line(feature)
        assert drawn and label
        assert feature.unit in label or "L" in label


def test_the_tolerance_tests_ask_the_right_side_of_each_target() -> None:
    assert ft.fails_tolerance(0.2, 0.1, ft.EITHER)
    assert ft.fails_tolerance(-0.2, 0.1, ft.EITHER)
    assert not ft.fails_tolerance(0.05, 0.1, ft.EITHER)
    assert ft.fails_tolerance(150.0, 165.0, ft.BELOW)
    assert not ft.fails_tolerance(180.0, 165.0, ft.BELOW)
    assert ft.fails_tolerance(40.0, 30.0, ft.ABOVE)
    assert not ft.fails_tolerance(40.0, 30.0, ft.BELOW)
    # An unmeasured hold is not a hold that missed a target.
    assert not ft.fails_tolerance(math.nan, 0.1, ft.EITHER)
    assert {side for _, _, _, side in ft.TOLERANCES} == {ft.ABOVE, ft.BELOW, ft.EITHER}


def test_the_features_and_their_targets_are_documented() -> None:
    for feature in ft.FEATURES:
        assert feature.name in ft.FEATURE_NAMES
        assert feature.unit in {"L", "deg", "ratio"}
        assert math.isfinite(feature.target)
        assert feature.digits == (2 if feature.unit == "deg" else 5)
    assert ft.__doc__ is not None
    for feature in ft.FEATURES:
        assert feature.name in ft.__doc__, f"{feature.name} is not in the module docstring"
    for name, _, _, _ in ft.TOLERANCES:
        assert name in ft.FEATURE_NAMES


# --------------------------------------------------------------------------- #
# Reading and writing files
# --------------------------------------------------------------------------- #


def write_processed(
    root: pathlib.Path,
    poses: Sequence[Pose],
    *,
    clip_id: str = CLIP_ID,
    body_length: float = LENGTH,
    usable: bool = True,
    contact: Sequence[bool] | None = None,
) -> pathlib.Path:
    """Write a synthetic processed parquet + sidecar, and return the parquet path.

    The parquet is the schema :mod:`handstand.postprocess` writes — the shared long
    one plus ``valid`` and ``filled`` — and the sidecar carries the body length, so
    the tests run against exactly what a real run reads.
    """
    x, y, valid = _arrays(poses)
    frames = x.shape[0]
    contact = [False] * frames if contact is None else list(contact)
    rows = []
    for index in range(frames):
        for joint in range(x.shape[1]):
            known = bool(valid[index, joint])
            rows.append(
                {
                    "frame_idx": index,
                    "t_ms": index * FRAME_MS,
                    "joint": pm.JOINT_NAMES[joint],
                    "x": float(x[index, joint]) if known else math.nan,
                    "y": float(y[index, joint]) if known else math.nan,
                    "z": math.nan,
                    "visibility": 0.9 if known else math.nan,
                    "presence": 0.9 if known else math.nan,
                    "rotated": False,
                    "detected": known,
                    "athlete_score": 0.9,
                    "n_people": 2,
                    "trainer_contact": bool(contact[index]),
                    "contact_reason": pm.JOINT_NAMES[joint] if contact[index] else "",
                    "x_raw": float(x[index, joint]) if known else math.nan,
                    "y_raw": float(y[index, joint]) if known else math.nan,
                    "valid": known,
                    "filled": False,
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{clip_id}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    sidecar = {
        "clip_id": clip_id,
        "source": "mediapipe",
        "usable": usable,
        "unusable_reason": "" if usable else "torso was measurable on 3 frame(s), need 10",
        "body_length_px": body_length if usable else None,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return path


def write_phases(
    root: pathlib.Path,
    labels: Sequence[str],
    *,
    clip_id: str = CLIP_ID,
    t_ms: Sequence[int] | None = None,
) -> pathlib.Path:
    """Write a synthetic phases parquet, the way :mod:`handstand.phases` writes it."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{clip_id}.parquet"
    times = [index * FRAME_MS for index in range(len(labels))]
    phase, hold = phases_for([], labels)
    pd.DataFrame(
        {
            "frame_idx": list(range(len(labels))),
            "t_ms": list(t_ms) if t_ms is not None else times,
            "phase": phase,
            "hold_id": hold,
        }
    ).to_parquet(path, index=False)
    return path


def test_run_clip_writes_the_parquet_and_the_hold_table(tmp_path: pathlib.Path) -> None:
    in_root = ft.input_dir(tmp_path, "mediapipe")
    labels_root = ft.phases_dir(tmp_path, "mediapipe")
    out_root = ft.output_dir(tmp_path, "mediapipe")
    write_processed(in_root, [LINE] * 12)
    write_phases(labels_root, ["hold"] * 10 + ["exit", "exit"])

    report = ft.run_clip(
        CLIP_ID, in_root=in_root, labels_root=labels_root, out_root=out_root
    )

    assert not report.skipped
    assert report.parquet_path == out_root / f"{CLIP_ID}.parquet"
    assert report.stats.hold_count == 1
    assert report.stats.hold_frames == 10
    assert "holds=1" in report.line()
    assert [p.name for p in out_root.iterdir()] == [f"{CLIP_ID}.parquet"]

    # The parquet reads back with everything the columns promise, and the features
    # are the ones the geometry gives.
    table = ft.read_clip_features(report.parquet_path)
    assert len(table) == 12
    assert table["phase"].tolist() == ["hold"] * 10 + ["exit"] * 2
    assert table["line_deviation"].tolist() == [0.0] * 12
    assert table["hip_angle"].tolist() == [pytest.approx(180.0, abs=0.5)] * 12

    # ... and re-running it is a no-op unless asked.
    again = ft.run_clip(CLIP_ID, in_root=in_root, labels_root=labels_root, out_root=out_root)
    assert again.skipped
    assert again.line().startswith("skip ")
    again = ft.run_clip(
        CLIP_ID, in_root=in_root, labels_root=labels_root, out_root=out_root, overwrite=True
    )
    assert not again.skipped


def test_run_clip_writes_a_clip_with_no_scale_as_all_nan(tmp_path: pathlib.Path) -> None:
    in_root = ft.input_dir(tmp_path)
    labels_root = ft.phases_dir(tmp_path)
    out_root = ft.output_dir(tmp_path)
    write_processed(in_root, [LINE] * 8, usable=False)
    write_phases(labels_root, ["hold"] * 8)

    report = ft.run_clip(
        CLIP_ID, in_root=in_root, labels_root=labels_root, out_root=out_root
    )

    assert not report.stats.usable
    assert "no body length" in report.stats.unusable_reason
    table = ft.read_clip_features(report.parquet_path)
    assert not table["valid"].any()
    assert table["line_deviation"].isna().all()
    assert "unusable" in report.line()


def test_run_clip_refuses_a_clip_that_was_never_prepared(tmp_path: pathlib.Path) -> None:
    in_root = ft.input_dir(tmp_path)
    labels_root = ft.phases_dir(tmp_path)
    out_root = ft.output_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="handstand.postprocess --all"):
        ft.run_clip(CLIP_ID, in_root=in_root, labels_root=labels_root, out_root=out_root)
    write_processed(in_root, [LINE] * 4)
    with pytest.raises(FileNotFoundError, match="handstand.phases --all"):
        ft.run_clip(CLIP_ID, in_root=in_root, labels_root=labels_root, out_root=out_root)


def test_labels_that_are_not_one_per_frame_are_refused(tmp_path: pathlib.Path) -> None:
    in_root = ft.input_dir(tmp_path)
    labels_root = ft.phases_dir(tmp_path)
    write_processed(in_root, [LINE] * 8)
    write_phases(labels_root, ["hold"] * 8)
    write_processed(in_root, [LINE] * 8, clip_id="short")
    write_phases(labels_root, ["hold"] * 4, clip_id="short")

    with pytest.raises(ValueError, match="4 frame\\(s\\) of phases"):
        ft.run_clip(
            "short", in_root=in_root, labels_root=labels_root, out_root=ft.output_dir(tmp_path)
        )


def test_reading_a_phases_parquet_as_features_says_to_run_this_stage(
    tmp_path: pathlib.Path,
) -> None:
    path = write_phases(tmp_path, ["hold"] * 4)
    with pytest.raises(ValueError, match="handstand.features --all"):
        ft.read_clip_features(path)


def test_write_hold_summary_replaces_only_the_clips_this_run_covered(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / ft.HOLD_SUMMARY_NAME
    other = dataclasses.replace(extract([LINE] * 4, ["hold"] * 4), clip_id="other000000")
    ft.write_hold_summary(
        path, ft.hold_summary_table([extract([LINE] * 4, ["hold"] * 4), other])
    )
    # A run over a third clip adds it and leaves the other two alone.
    third = dataclasses.replace(extract([LINE] * 6, ["hold"] * 6), clip_id="aaa000000000")
    ft.write_hold_summary(path, ft.hold_summary_table([third]))

    written = pd.read_csv(path)
    assert sorted(written["clip_id"].unique()) == sorted(
        [CLIP_ID, "aaa000000000", "other000000"]
    )
    assert len(written) == 3
    # Re-running the first clip replaces its row rather than adding a second one.
    ft.write_hold_summary(
        path, ft.hold_summary_table([extract([LINE] * 4, ["hold"] * 3 + ["exit"])])
    )
    written = pd.read_csv(path)
    assert len(written) == 3
    assert written[written["clip_id"] == CLIP_ID]["hold_frames"].tolist() == [3]
    assert list(written.columns) == list(ft.HOLD_SUMMARY_COLUMNS)
    assert written["clip_id"].tolist() == sorted(written["clip_id"].tolist())
    assert [p.name for p in tmp_path.iterdir()] == [ft.HOLD_SUMMARY_NAME]


def test_write_hold_summary_refuses_a_table_it_cannot_merge(tmp_path: pathlib.Path) -> None:
    path = tmp_path / ft.HOLD_SUMMARY_NAME
    path.write_text("clip_id,hold_id\nx,0\n")
    with pytest.raises(ValueError, match="missing column"):
        ft.write_hold_summary(path, ft.hold_summary_table([extract([LINE] * 4, ["hold"] * 4)]))


# --------------------------------------------------------------------------- #
# The plot
# --------------------------------------------------------------------------- #


def test_the_plot_names_the_file_and_draws_every_feature(tmp_path: pathlib.Path) -> None:
    assert ft.plot_path(tmp_path, CLIP_ID) == tmp_path / "reports" / f"features_{CLIP_ID}.png"
    assert ft.plot_path(tmp_path, CLIP_ID, "vision").name == f"{CLIP_ID}_vision_features.png"
    assert ft.reports_dir(tmp_path) == tmp_path / "reports"

    labels = ["pre"] * 4 + ["hold"] * 10 + ["exit"] * 3 + ["hold"] * 5
    table = ft.feature_table(extract([LINE] * 22, labels))
    path = ft.plot_clip(table, ft.plot_path(tmp_path, CLIP_ID))

    assert path.is_file()
    assert path.stat().st_size > 0
    # Nothing is left behind but the plot: the image is written to a temporary
    # name and renamed.
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_the_plot_of_a_clip_with_nothing_to_show_still_draws(tmp_path: pathlib.Path) -> None:
    table = ft.feature_table(extract([dataclasses.replace(LINE, visible=False)] * 6, ["pre"] * 6))
    path = ft.plot_clip(table, tmp_path / "empty.png")

    assert path.is_file()


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def test_the_cli_measures_clips_and_writes_the_hold_table(tmp_path: pathlib.Path) -> None:
    in_root = ft.input_dir(tmp_path)
    labels_root = ft.phases_dir(tmp_path)
    for clip_id, count in (("aaa000000000", 12), ("bbb111111111", 8)):
        write_processed(in_root, [LINE] * count, clip_id=clip_id)
        write_phases(labels_root, ["hold"] * (count - 2) + ["exit", "exit"], clip_id=clip_id)

    assert ft.main(["--all", "--data", str(tmp_path)]) == 0

    summary = pd.read_csv(ft.output_dir(tmp_path) / ft.HOLD_SUMMARY_NAME)
    assert sorted(summary["clip_id"].unique()) == ["aaa000000000", "bbb111111111"]
    assert (ft.output_dir(tmp_path) / "aaa000000000.parquet").is_file()
    assert (ft.output_dir(tmp_path) / "bbb111111111.parquet").is_file()
    assert ft.summary([]) == "summary clips=0"


def test_the_cli_says_so_when_the_input_is_not_there(tmp_path: pathlib.Path, capsys) -> None:
    assert ft.main(["--all", "--data", str(tmp_path)]) == 2
    assert "handstand.postprocess" in capsys.readouterr().out

    write_processed(ft.input_dir(tmp_path), [LINE] * 4)
    assert ft.main(["--all", "--data", str(tmp_path)]) == 2
    assert "handstand.phases --all" in capsys.readouterr().out


def test_the_cli_survives_a_clip_it_cannot_read(tmp_path: pathlib.Path, capsys) -> None:
    in_root = ft.input_dir(tmp_path)
    labels_root = ft.phases_dir(tmp_path)
    write_processed(in_root, [LINE] * 6, clip_id="aaa000000000")
    write_phases(labels_root, ["hold"] * 6, clip_id="aaa000000000")

    assert ft.main(["--clips", "aaa000000000", "gone", "--data", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "fail  gone" in out
    assert "clips=2" in out


def test_the_cli_needs_clips_or_all(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        ft.main(["--data", str(tmp_path)])
    with pytest.raises(SystemExit):
        ft.main(["--all", "--clips", CLIP_ID, "--data", str(tmp_path)])
    with pytest.raises(SystemExit):
        ft.main(["--clips", CLIP_ID, "--limit", "-1", "--data", str(tmp_path)])
    with pytest.raises(SystemExit):
        ft.main(["--clips", CLIP_ID, "--source", "nope", "--data", str(tmp_path)])


def test_the_cli_plots_a_clip_it_has_just_measured(tmp_path: pathlib.Path) -> None:
    in_root = ft.input_dir(tmp_path)
    labels_root = ft.phases_dir(tmp_path)
    write_processed(in_root, [LINE] * 10, clip_id=CLIP_ID)
    write_phases(labels_root, ["hold"] * 8 + ["exit", "exit"], clip_id=CLIP_ID)

    assert ft.main(["--all", "--plot", CLIP_ID, "--data", str(tmp_path)]) == 0
    assert ft.plot_path(tmp_path, CLIP_ID).is_file()


def test_the_cli_reports_a_plot_it_cannot_draw(tmp_path: pathlib.Path, capsys) -> None:
    write_processed(ft.input_dir(tmp_path), [LINE] * 6, clip_id=CLIP_ID)
    write_phases(ft.phases_dir(tmp_path), ["hold"] * 6, clip_id=CLIP_ID)

    assert ft.main(["--all", "--plot", "gone", "--data", str(tmp_path)]) == 1
    assert "plot gone" in capsys.readouterr().err


def test_the_cli_documents_the_documented_flags() -> None:
    parser = ft.build_arg_parser()
    args = parser.parse_args(["--all", "--source", "vision", "--limit", "3", "--overwrite"])

    assert args.all_clips
    assert args.source == "vision"
    assert args.limit == 3
    assert args.overwrite
    args = parser.parse_args(["--clips", "a", "b", "--plot", "a"])
    assert args.clips == ["a", "b"]
    assert args.plot == ["a"]
    # The message about the stage before names the stage that writes its input, and
    # it is the one handstand.phases writes rather than a second wording of it.
    assert ft.missing_input_message is ft.phases.missing_input_message
    assert "handstand.postprocess --all" in ft.missing_input_message("x", "vision")
    assert "handstand.phases --all" in ft.missing_phases_message("x", "vision")
    assert "handstand.phases --all --source vision" in ft.missing_phases_message("x", "vision")
    assert ft.FEATURES_DIRNAME == "features"
    assert ft.HOLD_SUMMARY_NAME == "hold_summary.csv"
    assert ft.HOLD_PHASE == "hold"
    assert ft.NO_HOLD == -1
