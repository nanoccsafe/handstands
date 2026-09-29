"""Tests for :mod:`handstand.com`, the segment model and the balance geometry.

Nothing here touches the real handstand videos: every pose is written by hand in
the body frame — the wrist midpoint at the origin, ``v`` up, both in body
lengths — so each assertion is arithmetic on Winter's constants that a reader
can check with a pencil. The pose is a symmetric vertical handstand
(:data:`LINE`), and the tests move one part of it and say exactly how far the
centre of mass should have followed, which is the only way to tell a segment
model that works from one that happens to give a plausible number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from handstand import com

#: The joints a side view reports. There is no ``foot_index`` here, so this is
#: both the schema Apple Vision has and the one the features tests drive: the
#: foot segment ends at the ankle (:data:`FOOT_JOINTS` adds MediaPipe's toes).
JOINTS: tuple[str, ...] = (
    "nose",
    *(
        f"{side}_{part}"
        for side in com.SIDES
        for part in ("ankle", "elbow", "hip", "knee", "shoulder", "wrist")
    ),
)

#: The same schema with the toes, for the tests about them.
FOOT_JOINTS: tuple[str, ...] = (*JOINTS, "left_foot_index", "right_foot_index")

#: A symmetric vertical handstand in the body frame: every station on the
#: vertical through the hands, the arms straight, the nose in front of the
#: chest — the same pose the features tests call LINE.
LINE: dict[str, tuple[float, float]] = {
    "nose": (0.04, 0.30),
    "left_wrist": (0.0, 0.0),
    "right_wrist": (0.0, 0.0),
    "left_elbow": (0.0, 0.195),
    "right_elbow": (0.0, 0.195),
    "left_shoulder": (0.0, 0.39),
    "right_shoulder": (0.0, 0.39),
    "left_hip": (0.0, 0.75),
    "right_hip": (0.0, 0.75),
    "left_knee": (0.0, 0.95),
    "right_knee": (0.0, 0.95),
    "left_ankle": (0.0, 1.15),
    "right_ankle": (0.0, 1.15),
}

#: The joints every test moves to shift the legs, both sides below the hip.
LEG_JOINTS: tuple[str, ...] = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def shifted(
    points: Mapping[str, tuple[float, float]] = LINE, **moves: tuple[float, float]
) -> dict[str, tuple[float, float]]:
    """A copy of ``points`` (or :data:`LINE`) with the named joints moved."""
    updated = dict(points)
    updated.update(moves)
    return updated


def uv_of(
    points: Mapping[str, tuple[float, float]] = LINE,
    joints: Sequence[str] = JOINTS,
    *,
    hidden: Sequence[str] = (),
    frames: int = 1,
) -> np.ndarray:
    """One pose as a ``(frames, joints, 2)`` body-frame track.

    A joint that is not in ``points`` — or that ``hidden`` names — is NaN there,
    which is exactly how :func:`handstand.features.body_frame_track` writes a
    joint the processed data had nothing for.
    """
    uv = np.full((frames, len(joints), 2), np.nan)
    for index, name in enumerate(joints):
        if name in points and name not in hidden:
            uv[:, index, :] = points[name]
    return uv


def com_of(
    points: Mapping[str, tuple[float, float]] = LINE, joints: Sequence[str] = JOINTS
) -> np.ndarray:
    """The ``(frames, 2)`` centre of mass of one pose."""
    return com.centre_of_mass(uv_of(points, joints), joints).com


def at_u(
    points: Mapping[str, tuple[float, float]], names: Sequence[str], u: float
) -> dict[str, tuple[float, float]]:
    """``points`` with every named joint moved to ``u``, keeping its ``v``."""
    return shifted(points, **{name: (u, points[name][1]) for name in names})


# --------------------------------------------------------------------------- #
# The segment model
# --------------------------------------------------------------------------- #


def test_the_segments_are_winters_masses_and_add_to_a_whole_body() -> None:
    # The model only balances if the constants are the ones they say they are:
    # seven segments a side plus two midline ones, every mass a fraction of the
    # same body, and every position Winter's fraction along its segment.
    assert com.SEGMENT_SOURCE.startswith("Winter, D. A. (2009)")
    total = com.MASS_HEAD_NECK + com.MASS_TRUNK + 2 * sum(
        segment.mass for segment in com.SEGMENTS if segment.sides
    )
    assert total == pytest.approx(1.0, abs=1e-12)
    assert com.MASS_HEAD_NECK == pytest.approx(0.081)
    assert com.MASS_TRUNK == pytest.approx(0.497)

    # The segment table and the named constants are the same numbers, so the
    # two cannot drift apart.
    masses = {segment.name: segment.mass for segment in com.SEGMENTS}
    assert masses == {
        "trunk": com.MASS_TRUNK,
        "upper_arm": com.MASS_UPPER_ARM,
        "forearm_hand": com.MASS_FOREARM_HAND,
        "thigh": com.MASS_THIGH,
        "shank": com.MASS_SHANK,
        "foot": com.MASS_FOOT,
    }
    fractions = {segment.name: segment.fraction for segment in com.SEGMENTS}
    assert fractions["upper_arm"] == pytest.approx(0.436)
    assert fractions["forearm_hand"] == pytest.approx(0.682)
    assert fractions["thigh"] == pytest.approx(0.433)
    assert fractions["shank"] == pytest.approx(0.433)
    assert fractions["trunk"] == pytest.approx(0.5)
    assert fractions["foot"] == pytest.approx(0.5)
    assert com.COM_HEAD_NECK == 1.0


def test_a_symmetric_vertical_handstand_has_its_com_over_the_hands() -> None:
    result = com.centre_of_mass(uv_of(), JOINTS)

    assert result.frames == 1
    # Symmetric about the vertical through the wrists: the only thing off the
    # line is the head, which the nose puts a whole 0.04 L to one side.
    assert result.com[0, 0] == pytest.approx(com.MASS_HEAD_NECK * 0.04, abs=1e-12)
    assert abs(result.com[0, 0]) < 0.01
    # And it sits between the hands and the hips, below the trunk's own centre.
    assert 0.0 < result.com[0, 1] < LINE["left_hip"][1]
    assert bool(result.complete[0])


def test_the_com_follows_the_mass_that_moved() -> None:
    base = com_of()

    # Whole body one body length sideways: every segment moved the same, so the
    # weighted mean of their centres moved the same. This is the check that the
    # masses add to one — a model whose masses summed to 0.9 would move by
    # 1/0.9 of x here.
    sideways = com_of(shifted(**{name: (u + 0.3, v) for name, (u, v) in LINE.items()}))
    assert sideways[0, 0] - base[0, 0] == pytest.approx(0.3, abs=1e-12)

    # The legs below the hip, both sides, forwards by x: thigh, shank and foot
    # take their centres with them, so the leg share of the body (0.322) moves
    # by x — and the trunk's lower end moves with the hips, so its midpoint
    # follows half the way, carrying 0.497 / 2 of the body with it.
    x = 0.1
    leg_share = 2 * (com.MASS_THIGH + com.MASS_SHANK + com.MASS_FOOT)
    assert leg_share == pytest.approx(0.322)
    forward = com_of(at_u(LINE, LEG_JOINTS, x))
    expected = leg_share * x + com.MASS_TRUNK * 0.5 * x
    assert forward[0, 0] - base[0, 0] == pytest.approx(expected, abs=1e-12)
    # The direction is the one the legs went.
    assert forward[0, 0] > base[0, 0]
    assert com_of(at_u(LINE, LEG_JOINTS, -x))[0, 0] < base[0, 0]

    # With the hip pinned, only the knee and the ankle move: the thigh's centre
    # is 0.433 of the way down it, so it follows only that fraction, while the
    # shank and the foot follow all the way.
    below = at_u(LINE, ("left_knee", "right_knee", "left_ankle", "right_ankle"), x)
    expected_below = 2 * (
        com.COM_THIGH * com.MASS_THIGH + com.MASS_SHANK + com.MASS_FOOT
    ) * x
    assert com_of(below)[0, 0] - base[0, 0] == pytest.approx(expected_below, abs=1e-12)
    assert expected_below < leg_share * x


def test_the_head_is_the_nose_and_falls_back_to_the_spine() -> None:
    base = com_of()

    # The head's mass is at the nose, so moving the nose moves the CoM by the
    # head's share of the move — which is where 0.081 shows up in a measurement.
    further = com_of(shifted(nose=(0.14, 0.30)))
    assert further[0, 0] - base[0, 0] == pytest.approx(com.MASS_HEAD_NECK * 0.10, abs=1e-12)

    # Without a nose the head goes 0.5 of the way from the shoulder midpoint
    # towards the body's axis extended past the shoulder — here straight down
    # the image, to v = 0.39 + 0.5 (0.39 - 0.75) = 0.21 and u = 0.
    head_v = LINE["left_shoulder"][1] + com.HEAD_FALLBACK * (
        LINE["left_shoulder"][1] - LINE["left_hip"][1]
    )
    assert head_v == pytest.approx(0.21)
    without_nose = uv_of()
    without_nose[0, JOINTS.index("nose"), :] = np.nan
    fallback = com.centre_of_mass(without_nose, JOINTS)
    at_fallback = com_of(shifted(nose=(0.0, head_v)))
    assert fallback.com[0] == pytest.approx(at_fallback[0], abs=1e-12)
    # An estimated head is not a missing head: the frame stays complete.
    assert bool(fallback.complete[0])

    # With neither a nose nor a hip to extend the spine from, the head is a
    # segment that is not there: the rest of the body is still weighed, and the
    # frame says so rather than pretending otherwise.
    no_spine = uv_of(hidden=("left_hip", "right_hip"))
    no_spine[0, JOINTS.index("nose"), :] = np.nan
    missing = com.centre_of_mass(no_spine, JOINTS)
    assert not bool(missing.complete[0])
    assert np.isfinite(missing.com).all()


def test_a_foot_index_the_schema_does_not_have_ends_the_foot_at_the_ankle() -> None:
    # Apple Vision reports no toes: the foot is still a foot, it is just
    # zero-length, and its centre is the ankle's.
    without_toes = com_of(joints=JOINTS)
    toes_at_ankle = {
        **LINE,
        "left_foot_index": LINE["left_ankle"],
        "right_foot_index": LINE["right_ankle"],
    }
    assert com_of(toes_at_ankle, FOOT_JOINTS)[0] == pytest.approx(without_toes[0], abs=1e-12)

    # A schema that *has* the toes but lost them on this frame drops the 2.9 %
    # of the body they carry rather than inventing where they were.
    lost = com.centre_of_mass(uv_of(joints=FOOT_JOINTS), FOOT_JOINTS)
    assert not bool(lost.complete[0])
    assert np.isfinite(lost.com).all()

    # And real toes move the CoM by their mass share of the move, taken where
    # along the foot that move lands: both feet, 0.0145 each, the toes 0.1 L
    # forwards and the foot's centre 0.5 of the way to them.
    toes = shifted(toes_at_ankle, left_foot_index=(0.1, 1.2), right_foot_index=(0.1, 1.2))
    moved = com_of(toes, FOOT_JOINTS)
    expected = 2 * com.MASS_FOOT * com.COM_FOOT * (0.1 - LINE["left_ankle"][0])
    assert moved[0, 0] - com_of(toes_at_ankle, FOOT_JOINTS)[0, 0] == pytest.approx(
        expected, abs=1e-12
    )
    assert bool(com.centre_of_mass(uv_of(toes, FOOT_JOINTS), FOOT_JOINTS).complete[0])


def test_a_segment_missing_on_one_side_is_taken_from_the_other() -> None:
    # A side view hides the far side. In a side view the two sides overlap, so
    # the far thigh is at the near thigh's coordinates — not its mirror image —
    # and losing one side of every segment changes nothing.
    both = com_of()
    one_side = com.centre_of_mass(
        uv_of(hidden=("left_hip", "left_knee", "left_ankle", "left_shoulder", "left_elbow")),
        JOINTS,
    )

    assert one_side.com[0] == pytest.approx(both[0], abs=1e-12)
    assert bool(one_side.complete[0])


def test_a_segment_missing_on_both_sides_is_renormalised_and_flagged() -> None:
    # Both shanks and both feet out of the picture. The CoM of what is there,
    # over the mass that is there — and the frame flagged as incomplete, because
    # a quarter of the body went missing.
    hidden = ("left_knee", "right_knee", "left_ankle", "right_ankle")
    gone = com.centre_of_mass(uv_of(hidden=hidden), JOINTS)
    kept = 1.0 - 2 * (com.MASS_THIGH + com.MASS_SHANK + com.MASS_FOOT)

    assert not bool(gone.complete[0])
    assert kept == pytest.approx(0.678)
    kept_by_hand = com.MASS_HEAD_NECK + com.MASS_TRUNK + 2 * (
        com.MASS_UPPER_ARM + com.MASS_FOREARM_HAND
    )
    assert kept == pytest.approx(kept_by_hand, abs=1e-12)

    # The surviving segments, written out: the nose, the trunk's midpoint, and
    # per side the arm halfway to the elbow's own fraction and the forearm's.
    shoulder = np.array(LINE["left_shoulder"])
    elbow = np.array(LINE["left_elbow"])
    wrist = np.array(LINE["left_wrist"])
    hip = np.array(LINE["left_hip"])
    moment = (
        com.MASS_HEAD_NECK * np.array(LINE["nose"])
        + com.MASS_TRUNK * (shoulder + com.COM_TRUNK * (hip - shoulder))
        + 2 * com.MASS_UPPER_ARM * (shoulder + com.COM_UPPER_ARM * (elbow - shoulder))
        + 2 * com.MASS_FOREARM_HAND * (elbow + com.COM_FOREARM_HAND * (wrist - elbow))
    )
    assert gone.com[0] == pytest.approx(moment / kept, abs=1e-12)
    # And a frame that loses even the trunk's ends still has something to weigh.
    trunkless = com.centre_of_mass(
        uv_of(hidden=("left_shoulder", "right_shoulder", "left_elbow", "right_elbow")), JOINTS
    )
    assert not bool(trunkless.complete[0])
    assert np.isfinite(trunkless.com).all()


def test_a_frame_with_nothing_visible_has_no_centre_of_mass() -> None:
    nothing = com.centre_of_mass(np.full((2, len(JOINTS), 2), np.nan), JOINTS)

    assert np.isnan(nothing.com).all()
    assert not nothing.complete.any()


def test_the_model_refuses_a_track_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="joint names"):
        com.centre_of_mass(np.zeros((3, 4, 2)), JOINTS)
    with pytest.raises(ValueError, match="joint names"):
        com.centre_of_mass(np.zeros((3, len(JOINTS), 3)), JOINTS)


# --------------------------------------------------------------------------- #
# Which way the athlete faces
# --------------------------------------------------------------------------- #


def torso_line() -> tuple[np.ndarray, np.ndarray]:
    """The shoulder midpoint and hip midpoint of :data:`LINE`, as one-frame tracks."""
    shoulder = np.array([[*LINE["left_shoulder"]]])
    hip = np.array([[*LINE["left_hip"]]])
    return shoulder, hip


def test_the_facing_sign_is_the_side_of_the_torso_line_the_nose_is_on() -> None:
    shoulder, hip = torso_line()

    # The nose is in front of the chest, so the nose's side of the torso line is
    # the way the fingers point: +u here, +1.
    assert com.facing_sign(shoulder, hip, np.array([[0.04, 0.30]]))[0] == 1.0
    assert com.facing_sign(shoulder, hip, np.array([[-0.04, 0.30]]))[0] == -1.0
    # No nose is no direction, and neither is a nose sitting on the line.
    assert math.isnan(com.facing_sign(shoulder, hip, np.array([[np.nan, np.nan]]))[0])
    assert math.isnan(com.facing_sign(shoulder, hip, np.array([[0.0, 0.30]]))[0])

    # The side of the line does not depend on which way up the torso is drawn:
    # a hip *below* the shoulder still has a nose at +u on its +u side.
    assert com.facing_sign(
        shoulder, shoulder + np.array([[0.0, -0.36]]), np.array([[0.04, 0.30]])
    )[0] == 1.0

    # And a shoulder and hip in the same place is not a line at all.
    same = np.array([[0.0, 0.39]])
    assert math.isnan(com.facing_sign(same, same, np.array([[0.04, 0.30]]))[0])


def test_the_facing_sign_is_decided_per_hold_and_not_per_frame() -> None:
    votes = np.array([1.0, -1.0, 1.0, 1.0, np.nan, -1.0, -1.0, np.nan, 1.0, -1.0, np.nan])
    holds = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, -1])

    decided = com.majority_per_hold(votes, holds)

    # Hold 0 voted three to one for +1: every frame of it faces +u, including
    # the one that had no nose to vote with.
    assert decided[:5].tolist() == [1.0] * 5
    # Hold 1's only finite votes are -1, so its noseless frame takes them too.
    assert decided[5:8].tolist() == [-1.0] * 3
    # Hold 2 is an exact tie: no majority, so the frames keep their own votes
    # rather than being told a direction nothing agreed on.
    assert decided[8:10].tolist() == [1.0, -1.0]
    # A frame outside a hold keeps its own vote, nose or no nose.
    assert math.isnan(decided[10])

    # The same two rules without the tie in the way: a hold that agrees is
    # unchanged, and a hold with no finite vote at all stays NaN.
    assert com.majority_per_hold(np.array([1.0, 1.0]), np.array([3, 3])).tolist() == [1.0, 1.0]
    assert np.isnan(com.majority_per_hold(np.array([np.nan, np.nan]), np.array([3, 3]))).all()


def test_com_forward_is_the_com_in_the_direction_the_fingers_point() -> None:
    forward = com.com_forward(np.array([0.05, -0.05, 0.05]), np.array([1.0, 1.0, -1.0]))

    # A CoM to the right is overbalanced for an athlete facing right and
    # underbalanced for one facing left: the same position, the other way up.
    assert forward.tolist() == [0.05, -0.05, -0.05]
    # An unmeasured sign is an unmeasured direction, not a sign of +1.
    assert math.isnan(com.com_forward(np.array([0.05]), np.array([np.nan]))[0])


# --------------------------------------------------------------------------- #
# The zones
# --------------------------------------------------------------------------- #


def test_the_balance_zones_are_cut_at_the_edges_of_the_base_of_support() -> None:
    zone = com.balance_zone(
        np.array(
            [
                -com.BASE_BACK - 1e-9,
                -com.BASE_BACK,
                0.0,
                com.BASE_FRONT,
                com.BASE_FRONT + 1e-9,
                np.nan,
            ]
        )
    )

    # Just past the heel of the hand is under, just inside the fingertips is ok,
    # just past them is over, and the edges themselves are still ok: the test is
    # strictly outside the base.
    assert zone[:5].tolist() == ["under", "ok", "ok", "ok", "over"]
    # A frame with no CoM has no zone, and the NaN does not become a word.
    assert zone[5] != zone[5]

    # The edges are the order of a hand's own length, they are approximate, and
    # they are asymmetric: there is more room towards the fingertips.
    assert com.BASE_BACK == pytest.approx(0.03)
    assert com.BASE_FRONT == pytest.approx(0.06)
    assert com.BASE_BACK < com.BASE_FRONT
    assert com.ZONE_NAMES == ("under", "ok", "over")
