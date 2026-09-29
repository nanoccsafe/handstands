"""The centre of mass, and where it sits over the hands.

A handstand is held by keeping the body's centre of mass (CoM) over the hands.
Handstand research measures balance as the CoM's position relative to the
support — over- or underbalanced — and reads corrections at the hips (the "hip
strategy") as the sign of a weaker balance than corrections at the shoulders,
because a body that has already tipped can only be brought back by moving the
CoM itself, which from the hips is a bigger lever than from the shoulders. This
module is #23's answer: where the CoM is, which way the athlete faces, whether
the CoM is inside the base of support, and one sign per hold so a sideways
recording and its mirror read the same.

Everything here is model-agnostic and side-view
(``docs/recording_protocol.md``): the input is a clip's joints already in the
body frame of :mod:`handstand.bodyframe` — origin at the wrist midpoint, ``u``
to the right, ``v`` up, units of body length ``L`` — and the output is in the
same units, so "the CoM is at ``0.02 L``" is a distance a judge could see.

The segment model
-----------------

The body is seven segments per side plus two midline ones, each with a mass
fraction of the whole body and a centre of mass at a fraction of the segment's
length measured **from its proximal end** (the end nearer the torso). Both
numbers are Winter's anthropometric constants — :data:`SEGMENT_SOURCE` names
the source, and every constant is at the top of this module so the model can be
checked against the table it came from:

===========================  ======  ==================================
segment                      mass    CoM position
===========================  ======  ==================================
head + neck (midline)        0.081   at the nose (fallback below)
trunk (midline)              0.497   0.5 along shoulder_mid → hip_mid
upper arm (each side)        0.028   0.436 along shoulder → elbow
forearm + hand (each side)   0.022   0.682 along elbow → wrist
thigh (each side)            0.100   0.433 along hip → knee
shank (each side)            0.0465  0.433 along knee → ankle
foot (each side)             0.0145  0.5 along ankle → foot_index
===========================  ======  ==================================

The two feet land on the ankle when the schema has no ``foot_index`` — Apple
Vision reports no toes — because a foot segment of zero length is a foot whose
CoM is at the ankle rather than a segment to drop. The head's fallback is the
one place the model estimates rather than reads: **when the nose is missing the
head's CoM is taken 0.5 of the way from the shoulder midpoint towards the
nose's place on the body's axis extended past the shoulders**
(``shoulder_mid + 0.5 (shoulder_mid − hip_mid)``), which in a handstand — where
the spine's direction points at the head — is where the head hangs. The nose is
the head's landmark; without one the spine still says which way the head is.

What is missing, and what that costs
------------------------------------

A side view sees the near side and hides the far one, so:

* a segment missing on one side is measured on **the other side's coordinates**
  (mirror-free: in a side view the two sides overlap, so the far leg's thigh is
  at the same place the near leg's thigh is, not its mirror image);
* a segment missing on **both** sides is dropped and the remaining masses are
  **renormalised** to sum to one — the CoM of the parts that are there — and the
  frame is flagged ``com_complete = false`` so a later stage can filter on it;
* a frame with nothing left (no visible joints at all, or a body frame that
  could not be placed) has no CoM: ``com``, ``com_u`` and ``com_v`` are NaN.

``com_complete`` is *not* set false by a fallback — the foot at the ankle and
the head off the spine are the model's documented approximations of segments
that are there, not segments that are not.

Which way is forward
--------------------

``facing_sign`` is ``+1`` when the athlete faces ``+u`` and ``-1`` when they
face ``-u``, read from **which side of the shoulder_mid → hip_mid line the nose
lies on**: in a handstand the fingers point the way the chest faces, so the
nose's side of the torso line is the direction of the fingers. The sign is then
decided **per hold by majority** (:func:`majority_per_hold`), because the nose
clears that line by only a couple of hundredths of a body length and a per-frame
vote flickers; frames outside a hold keep their own vote, and a hold with no
nose at all keeps its frames' NaN. ``com_forward = com_u × facing_sign`` is the
answer in anatomical rather than image coordinates: positive is towards the
fingers (the overbalance side), negative towards the heel of the hand (the
underbalance side).

The zones
---------

:data:`BASE_BACK` and :data:`BASE_FRONT` cut ``com_forward`` into
``balance_zone`` — ``under``, ``ok`` or ``over`` — and the two are **approximate
constants**, not measured ones: the base of support of a hand runs from the
heel of the hand to the fingertips, roughly 0.03 L behind the wrist midpoint
and 0.06 L in front of it on a splayed hand, and both numbers are the order of
a hand's own length in body lengths. They are asymmetric because the hand is:
there is more room towards the fingers than towards the heel, so the CoM is
allowed further over before it is lost.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

__all__ = [
    "BASE_BACK",
    "BASE_FRONT",
    "COM_FOREARM_HAND",
    "COM_FOOT",
    "COM_HEAD_NECK",
    "COM_SHANK",
    "COM_THIGH",
    "COM_TRUNK",
    "COM_UPPER_ARM",
    "MASS_FOREARM_HAND",
    "MASS_FOOT",
    "MASS_HEAD_NECK",
    "MASS_SHANK",
    "MASS_THIGH",
    "MASS_TRUNK",
    "MASS_UPPER_ARM",
    "NOSE",
    "SEGMENTS",
    "SEGMENT_SOURCE",
    "SIDES",
    "ZONE_NAMES",
    "CentreOfMass",
    "Segment",
    "balance_zone",
    "centre_of_mass",
    "com_forward",
    "facing_sign",
    "majority_per_hold",
]

# --------------------------------------------------------------------------- #
# The segment model: Winter (2009)
# --------------------------------------------------------------------------- #

#: Where every mass and every fraction below comes from, named because a
#: segment model is only as good as the table it was copied from: Winter's
#: anthropometric data, the standard reference for whole-body segment parameters
#: (segment mass as a fraction of body mass, segment centre of mass as a
#: fraction of the segment's length from its proximal end).
SEGMENT_SOURCE = (
    "Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement, "
    "3rd edition, Wiley. Anthropometric tables: segment mass as a fraction of "
    "body mass, segment centre of mass as a fraction of segment length from the "
    "proximal end."
)

#: Mass fractions of the whole body, adding to 1.0 over the twelve segments
#: below (two of each paired segment, one trunk, one head + neck).
MASS_HEAD_NECK = 0.081
MASS_TRUNK = 0.497
MASS_UPPER_ARM = 0.028
MASS_FOREARM_HAND = 0.022
MASS_THIGH = 0.100
MASS_SHANK = 0.0465
MASS_FOOT = 0.0145

#: Where each segment's centre of mass sits along it, from the proximal end.
#: The head's is 1.0 because its segment is shoulder_mid → nose: the nose *is*
#: the head's centre of mass in this model, as the task's model specifies.
COM_HEAD_NECK = 1.0
COM_TRUNK = 0.5
COM_UPPER_ARM = 0.436
COM_FOREARM_HAND = 0.682
COM_THIGH = 0.433
COM_SHANK = 0.433
COM_FOOT = 0.5

#: How far towards the nose the head's centre of mass goes when there is no
#: nose to put it at: 0.5 of the way from the shoulder midpoint to the body's
#: axis extended past the shoulder, i.e. ``shoulder_mid + 0.5 (shoulder_mid −
#: hip_mid)``. See the module docstring.
HEAD_FALLBACK = 0.5

#: The two sides, as the schema names them.
SIDES: tuple[str, ...] = ("left", "right")

#: The nose: the only joint that says which way the athlete is facing.
NOSE = "nose"

#: The joint the foot segment ends at. A schema without it (Apple Vision has no
#: toes) ends the foot at the ankle instead — see :class:`Segment`.
FOOT_DISTAL = "foot_index"


@dataclasses.dataclass(frozen=True)
class Segment:
    """One segment of the model: its mass, its two joints, and its CoM along them.

    ``proximal`` and ``distal`` are *part* names, which the segment looks up per
    side (``left_hip``, ``right_hip``) unless ``sides`` is False, in which case
    both are midpoints of the two sides (``shoulder_mid``, ``hip_mid``).
    ``fallback`` is the distal part to use when the schema reports no such joint
    at all, and ``None`` when there is nothing to fall back to.
    """

    name: str
    mass: float
    proximal: str
    distal: str
    fraction: float
    sides: bool = True
    fallback: str | None = None


#: Every segment except the head, whose rule is its own (:func:`_head_points`).
#: Order is the order the body is read in, hands to feet; it does not affect the
#: sum, and the paired ones contribute twice.
SEGMENTS: tuple[Segment, ...] = (
    Segment("trunk", MASS_TRUNK, "shoulder", "hip", COM_TRUNK, sides=False),
    Segment("upper_arm", MASS_UPPER_ARM, "shoulder", "elbow", COM_UPPER_ARM),
    Segment("forearm_hand", MASS_FOREARM_HAND, "elbow", "wrist", COM_FOREARM_HAND),
    Segment("thigh", MASS_THIGH, "hip", "knee", COM_THIGH),
    Segment("shank", MASS_SHANK, "knee", "ankle", COM_SHANK),
    Segment("foot", MASS_FOOT, "ankle", FOOT_DISTAL, COM_FOOT, fallback="ankle"),
)

#: How far behind the wrist midpoint the heel of the hand reaches, in body
#: lengths — the back edge of the base of support. **Approximate**: it is a
#: hand's own length read off a side view, not a measured constant, and a
#: splayed or a rolled hand moves it.
BASE_BACK = 0.03

#: How far in front of the wrist midpoint the fingertips reach, in body
#: lengths — the front edge of the base of support, longer than the back edge
#: because a hand is. Also **approximate**, for the same reason
#: :data:`BASE_BACK` is.
BASE_FRONT = 0.06

#: The three zones :func:`balance_zone` can name, in the order they run along
#: ``com_forward``.
ZONE_NAMES: tuple[str, ...] = ("under", "ok", "over")


# --------------------------------------------------------------------------- #
# The geometry of one clip
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class CentreOfMass:
    """One clip's centre of mass: where it is, and whether every segment was there.

    ``com`` is ``(frames, 2)`` in body-frame units, NaN on a frame with nothing
    to measure; ``complete`` is ``(frames,)`` and is False where a segment was
    missing on both sides and the masses were renormalised without it.
    """

    com: np.ndarray
    complete: np.ndarray

    @property
    def frames(self) -> int:
        """How many frames the answer covers."""
        return int(self.com.shape[0])


def _points(uv: np.ndarray, joints: Sequence[str], name: str) -> np.ndarray:
    """The ``(frames, 2)`` track of one joint, NaN when the schema has no such joint.

    A source that does not report a joint (Apple Vision has no ``foot_index``)
    answers NaN rather than raising or inventing a position, which is what lets
    the same model run over every source.
    """
    if name not in joints:
        return np.full((uv.shape[0], 2), np.nan)
    return uv[:, list(joints).index(name), :]


def _side_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Both sides of a point as one: the mean where both are, the one that is where
    only one is, NaN where neither is — the same rule every midpoint uses."""
    both = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    either = np.isfinite(left).all(axis=1) | np.isfinite(right).all(axis=1)
    mean = (left + right) / 2.0
    chosen = np.where(np.isfinite(left).all(axis=1)[:, None], left, right)
    return np.where(both[:, None], mean, np.where(either[:, None], chosen, np.nan))


def _midpoint(uv: np.ndarray, joints: Sequence[str], part: str) -> np.ndarray:
    """The ``(frames, 2)`` midpoint of both sides of one body part, one side being enough."""
    return _side_mean(
        _points(uv, joints, f"left_{part}"), _points(uv, joints, f"right_{part}")
    )


def _along(proximal: np.ndarray, distal: np.ndarray, fraction: float) -> np.ndarray:
    """The point ``fraction`` of the way from ``proximal`` to ``distal``, NaN where either end is.

    NaN propagates rather than being skipped: a segment whose end is not visible
    is a segment that is not measured, and the caller decides what to do about
    that (take the other side, or drop the mass).
    """
    return proximal + fraction * (distal - proximal)


def _head_points(uv: np.ndarray, joints: Sequence[str]) -> np.ndarray:
    """The head + neck segment's CoM: the nose, or the documented fallback without one.

    With a nose the head's CoM *is* the nose (``COM_HEAD_NECK = 1.0`` along
    shoulder_mid → nose). Without one it is 0.5 of the way from the shoulder
    midpoint towards the nose's place on the body's axis extended past the
    shoulder — in a handstand the spine points at the head, so that is where the
    head hangs. Both ends of the fallback need the shoulder midpoint and the hip
    midpoint; with neither the head is missing like any other segment.
    """
    nose = _points(uv, joints, NOSE)
    shoulder = _midpoint(uv, joints, "shoulder")
    hip = _midpoint(uv, joints, "hip")
    seen = np.isfinite(nose).all(axis=1)[:, None]
    extended = shoulder + HEAD_FALLBACK * (shoulder - hip)
    return np.where(seen, nose, extended)


def _segment_points(uv: np.ndarray, joints: Sequence[str], segment: Segment) -> np.ndarray:
    """Both sides of one segment's CoM, ``(frames, 2, 2)``, each side filled from the other.

    A side whose endpoints are not both visible takes the other side's
    **coordinates** — not its mirror image: in a side view of a handstand the two
    sides overlap in the image, so the far thigh is where the near thigh is. A
    segment missing on both sides comes back NaN on both, and the caller drops
    its mass.
    """
    distal = segment.distal
    if segment.fallback and not any(f"{side}_{distal}" in joints for side in SIDES):
        distal = segment.fallback
    side_points = []
    for side in SIDES:
        side_points.append(
            _along(
                _points(uv, joints, f"{side}_{segment.proximal}"),
                _points(uv, joints, f"{side}_{distal}"),
                segment.fraction,
            )
        )
    left, right = side_points
    left_seen = np.isfinite(left).all(axis=1)[:, None]
    right_seen = np.isfinite(right).all(axis=1)[:, None]
    filled_left = np.where(left_seen, left, right)
    filled_right = np.where(right_seen, right, left)
    return np.stack((filled_left, filled_right), axis=1)


def _midline_points(uv: np.ndarray, joints: Sequence[str], segment: Segment) -> np.ndarray:
    """One midline segment's CoM, ``(frames, 1, 2)``: a midpoint is one side being enough."""
    point = _along(
        _midpoint(uv, joints, segment.proximal),
        _midpoint(uv, joints, segment.distal),
        segment.fraction,
    )
    return point[:, None, :]


def centre_of_mass(uv: np.ndarray, joints: Sequence[str]) -> CentreOfMass:
    """The whole clip's centre of mass from its body-frame joints.

    ``uv`` is ``(frames, joints, 2)`` in body-frame units with NaN wherever a
    joint was not visible (which is how :func:`handstand.features.body_frame_track`
    writes a frame the processed data has nothing for), and ``joints`` is the
    schema's names. The answer is the mass-weighted mean of the twelve segment
    CoMs — the two midline ones and both sides of the five paired ones — with
    the missing ones handled as the module documents: the other side's
    coordinates, or renormalised away with ``complete`` False.

    A frame where every segment is missing — nothing visible at all — has no
    CoM: NaN rather than the mean of nothing.
    """
    if uv.ndim != 3 or uv.shape[-1] != 2 or uv.shape[1] != len(joints):
        raise ValueError(
            f"uv must be (frames, {len(joints)}, 2) for those joint names, got {uv.shape}"
        )
    points: list[np.ndarray] = [_head_points(uv, joints)[:, None, :]]
    masses: list[float] = [MASS_HEAD_NECK]
    for segment in SEGMENTS:
        if segment.sides:
            # Both sides are separate entries of the same mass: a segment that
            # was missing on one side is already filled from the other inside
            # _segment_points, so a side view contributes its mass twice at the
            # coordinates of the side that is there. The swap puts the side on
            # the outside so the loop is over sides rather than frames, and each
            # entry is given the segment axis the concatenation below expects.
            both_sides = np.swapaxes(_segment_points(uv, joints, segment), 0, 1)
            for side_point in both_sides:
                points.append(side_point[:, None, :])
                masses.append(segment.mass)
        else:
            points.append(_midline_points(uv, joints, segment))
            masses.append(segment.mass)

    stacked = np.concatenate(points, axis=1)  # (frames, n_segments, 2)
    weights = np.asarray(masses, dtype=np.float64)  # (n_segments,)
    present = np.isfinite(stacked).all(axis=2)  # (frames, n_segments)
    per_frame = np.where(present, weights[None, :], 0.0)
    total = per_frame.sum(axis=1)  # the masses that survived, 1.0 when nothing is missing
    # The missing segments are zeroed before the moment is summed: NaN * 0 is
    # still NaN, and one absent segment must not take the rest of the frame
    # with it. The division below renormalises what is left onto the sum of the
    # masses that are there.
    moment = (np.where(present[:, :, None], stacked, 0.0) * per_frame[:, :, None]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        com = moment / np.where(total > 0.0, total, np.nan)[:, None]
    complete = present.all(axis=1)
    return CentreOfMass(com=com, complete=complete)


# --------------------------------------------------------------------------- #
# Which way is forward, and where the CoM is relative to the hands
# --------------------------------------------------------------------------- #


def facing_sign(
    shoulder_mid: np.ndarray, hip_mid: np.ndarray, nose: np.ndarray
) -> np.ndarray:
    """``+1`` when the athlete faces ``+u``, ``-1`` when they face ``-u``, NaN without a nose.

    The torso line is the shoulder midpoint to the hip midpoint, and the nose is
    on one side of it or the other; the side that is towards ``+u`` is ``+1``.
    In a handstand the fingers point the way the chest faces, so this is the
    direction of the fingers — which is what makes ``com_forward`` mean
    "towards the fingers" rather than "towards the right of the image".

    The sign is the nose's side of that line *per frame*; the hold-level vote is
    :func:`majority_per_hold`'s job, because the nose clears the line by only a
    couple of hundredths of a body length and one bad frame must not flip the
    whole hold. NaN when the nose is missing or on the line, and NaN when the
    torso line has no direction (the shoulder and hip midpoints coincide).
    """
    axis = hip_mid - shoulder_mid
    # The normal that points to +u: for a handstand's upright torso line that is
    # (axis_v, -axis_u) already, and flipping it when it points the other way
    # keeps "+1 is the side the nose is on towards +u" true however the torso is
    # tilted.
    across = np.stack((axis[:, 1], -axis[:, 0]), axis=1)
    flip = across[:, 0] < 0.0
    across = np.where(flip[:, None], -across, across)
    front = ((nose - shoulder_mid) * across).sum(axis=1)
    span = np.hypot(across[:, 0], across[:, 1])
    return np.where(
        (front > 0.0) & (span > 0.0),
        1.0,
        np.where((front < 0.0) & (span > 0.0), -1.0, np.nan),
    )


def majority_per_hold(sign: np.ndarray, hold_id: np.ndarray) -> np.ndarray:
    """One facing sign per hold, by majority: the sign most of the hold's frames had.

    The nose's clearance of the torso line is a small number measured on a noisy
    coordinate, so a per-frame vote flickers between holds of the same clip and
    ``com_forward`` would change sign mid-hold for no anatomical reason. The
    frames of a hold all take the sign most of them voted for (NaN votes do not
    count), frames outside a hold keep their own vote, and a hold with no
    majority — no finite votes, or an exact tie — keeps its frames as they were
    rather than being told a direction nothing agreed on.
    """
    decided = np.asarray(sign, dtype=np.float64).copy()
    ids = np.asarray(hold_id)
    for number in np.unique(ids[ids >= 0]):
        frame = ids == number
        votes = decided[frame]
        finite = votes[np.isfinite(votes)]
        if finite.size == 0:
            continue
        forward = int(np.count_nonzero(finite > 0.0))
        backward = int(np.count_nonzero(finite < 0.0))
        if forward > backward:
            decided[frame] = 1.0
        elif backward > forward:
            decided[frame] = -1.0
    return decided


def com_forward(com_u: np.ndarray, facing: np.ndarray) -> np.ndarray:
    """``com_u`` in the athlete's own direction: positive towards the fingers.

    The same multiplication in two places — once when the frame is measured and
    once when the hold has voted on its facing — so it is written once: a CoM to
    the right of the hands is *overbalanced* for an athlete facing right and
    *underbalanced* for one facing left, and this is the number that says so.
    """
    return np.asarray(com_u, dtype=np.float64) * np.asarray(facing, dtype=np.float64)


def balance_zone(forward: np.ndarray) -> np.ndarray:
    """Which side of the base of support each frame's CoM is on: ``under``, ``ok``, ``over``.

    ``under`` behind the heel of the hand (``com_forward < -BASE_BACK``),
    ``over`` in front of the fingertips (``com_forward > BASE_FRONT``), ``ok``
    in between, and NaN where ``com_forward`` is NaN — a frame with no CoM has
    no zone. The two edges are approximate, as their docstrings say: they are
    the order of a hand's own length, and they are asymmetric because the base
    of support is.
    """
    values = np.asarray(forward, dtype=np.float64)
    zone = np.full(values.shape, np.nan, dtype=object)
    known = np.isfinite(values)
    zone[known & (values < -BASE_BACK)] = "under"
    zone[known & (values > BASE_FRONT)] = "over"
    zone[known & (values >= -BASE_BACK) & (values <= BASE_FRONT)] = "ok"
    return zone
