"""Per-frame biomechanical features: stacking, joint angles, line and leg shape.

A handstand form is a handful of numbers. Is the stack over the hands, is the body
one straight line, are the arms straight, is the back arched, are the legs
together. A judge says those things in words, a fault classifier (#34) says them
in labels, a score (#29) says them in a number out of ten, and a line-vs-pike
reference (#28) says which of them a good handstand does better. All of them need
the same measurements, taken the same way, so they are measured once here, per
frame, and every later stage reads the same parquet.

Input and output
----------------

Input is the post-processed trajectory of :mod:`handstand.postprocess` (#20) and
the phase labels of :mod:`handstand.phases` (#21)::

    <data_dir>/processed/<source>/<clip_id>.parquet   # x/y processed, valid, filled
    <data_dir>/processed/<source>/<clip_id>.json      # body_length_px
    <data_dir>/phases/<source>/<clip_id>.parquet       # phase, hold_id

Output is one parquet per clip and one hold table per source::

    <data_dir>/features/<source>/<clip_id>.parquet   # one row per frame
    <data_dir>/features/<source>/hold_summary.csv     # one row per clip and hold
    <data_dir>/reports/features_<clip_id>.png         # what the numbers do over time

CLI::

    cd pipeline
    uv run python -m handstand.features --all
    uv run python -m handstand.features --clips 057c9e6c96af --overwrite
    uv run python -m handstand.features --all --plot 057c9e6c96af

The per-frame table is one row per frame, in frame order: ``frame_idx``,
``t_ms``, the ``phase`` and ``hold_id`` the frame carries over from #21, then
``valid``, ``side_view`` and every feature in :data:`FEATURES`. A feature the
frame could not support is NaN — never a number made of nothing — and ``valid``
is the column to filter on. :data:`HOLD_SUMMARY_NAME` is the same answer per
hold: the median and the inter-quartile range of every feature over that hold's
measurable frames, and how long the hold lasted. A hold is what gets scored, and
a median over a hold says what the hold looked like while an IQR says how much
it wobbled, which is the difference between a fault and a bad frame.

Nothing here knows which pose model produced the keypoints: it reads a source's
processed parquet and nothing else, so ``--source mediapipe`` and
``--source vision`` are the same code.

What a run reports
------------------

:func:`summary` prints the dataset-level answer, and the four questions it answers
are the ones the stages that read this one ask:

* how many clips are usable and how many have a hold, over how many hold frames;
* the **median and inter-quartile range of every feature over every hold frame in
  the dataset** — what a handstand looks like in this dataset, which is the number
  a fault classifier's class balance and a scorer's scale start from;
* **how many clips miss each feature's target** over their longest hold, which says
  whether a target discriminates at all before anything is scored against it;
* the extremes: the clips with the smallest median ``line_deviation`` over their
  longest hold (the straightest lines in the dataset, the ones a line-vs-pike
  reference like #28 wants as positives) and the largest ``banana`` (the curves).
  The medians are over the *longest* hold because a short one is a hop on the hands
  and says more about balance than about form.

Side view
---------

The recording protocol (``docs/recording_protocol.md``) asks for a **side view**
for straight handstands, and that is what every angle here assumes: the joint
angles are 2D projections, so a straight arm pointed at the camera reads as bent,
and the stacking offsets are the horizontal error a judge would see from the
side. The features are computed for every frame regardless, and the ``side_view``
column is where a later stage filters that assumption out — camera-angle
detection is #74, and until it lands this column is :func:`side_view_assumed`, a
column that is true everywhere and says so.

Units, and the body frame
-------------------------

Everything is read in the body frame of :mod:`handstand.bodyframe`: the origin is
the wrist midpoint, ``u`` runs to the right, ``v`` runs **up** (display-frame ``y``
with the sign flipped), and both are in the clip's body length ``L``. So a
feature in ``L`` is "a tenth of the way from the hands to the feet", the same
number on a 350 px athlete and a 600 px one, and an offset of ``0`` is a joint
exactly over the hands.

Three rules, and every feature is one of them:

* **Sides.** A left/right pair is averaged when both sides are visible, taken
  from the one that is when only one is, and NaN when neither is. A side-on
  recording sees the near side and hides the far one, and throwing away every
  hold because the far ankle is behind a leg would defeat the point of this
  module. A feature that *is* a distance between the two sides — ``leg_separation``
  and ``hand_width`` — needs both, and is NaN with one.
* **Signs.** A feature whose sign only says "which way" is signed in ``u``, which
  is a direction, not an anatomy: ``off_hip`` is positive when the hips are to the
  right of the vertical through the wrists, and ``body_angle`` is positive when
  the feet lean that way too. The one *shape* feature with a sign — ``banana`` —
  is signed against the direction the athlete **faces**, read off the nose, so a
  mirrored recording (or a clip shot from the other side of the athlete) gives the
  same sign; it is NaN without a nose, because a curve with no known direction is
  not a curve with a sign. ``head`` is the exception that proves the rule: the nose
  is the only facing cue a side view has, so signing the head against it would be
  signing it against itself, and it is left unsigned with the fault as a magnitude.
* **No half-measurements.** A joint the source does not report is NaN, not zero,
  and a frame where a limb cannot be seen does not get a plausible number out of
  the limb that can.

The features
------------

In :data:`FEATURES` order, with the value a good *line* handstand should show.
Lengths are in body lengths, angles in degrees, and ``L`` in both means body
lengths; the tolerances are the module constants below, named so the stages that
judge form can use the same numbers this docstring quotes.

``off_shoulder``, ``off_hip``, ``off_knee``, ``off_ankle`` — **0 L, target**
    The horizontal offset of each station's **midpoint** (both sides averaged
    where both are visible) from the vertical line through the wrist midpoint,
    signed. These are the stacking errors a judge calls "over the hands" and they
    are the reason the whole body frame has its origin where it has it: an
    offset of zero is a joint directly above the hands.
``line_deviation`` — **0 L, target; under :data:`LINE_TOLERANCE_L` is straight**
    The largest of the four offsets above, in absolute value: the single number
    for "how far is anything from the line", and the one a hold is ranked by. The
    largest of the four the frame *could* see, so a frame missing an ankle is
    judged on the rest; NaN only when the frame sees none of them.
``body_angle`` — **0°, target; under :data:`BODY_ANGLE_TOL_DEG` is straight**
    The signed angle in degrees of the wrist→ankle vector away from vertical. The
    same measurement #21 holds a hold to 35° of, read here as a form number
    instead of a threshold.
``shoulder_angle`` — **~180°; "open shoulders"**
    The interior angle at the shoulder between the hip and the wrist, averaged
    over the sides. Straight is 180°, and a handstand with the shoulders pushed
    forward and the chest arched open reads under :data:`OPEN_SHOULDER_DEG`.
``hip_angle`` — **~180°; a pike is under :data:`PIKE_HIP_DEG`**
    The interior angle at the hip between the shoulder and the knee, averaged
    over the sides. This is the difference between a line and a pike, and it is
    the one joint angle a kick-up that has not arrived fails first.
``knee_angle`` — **~180°; a locked knee is over :data:`STRAIGHT_KNEE_DEG`**
    The interior angle at the knee between the hip and the ankle, averaged over
    the sides. Below the tolerance the legs are bent, which in a held handstand
    means a pike with a soft knee rather than a straight line.
``elbow_angle`` — **~180°; a bent arm is a severe fault under
:data:`BENT_ELBOW_DEG`**
    The interior angle at the elbow between the shoulder and the wrist, averaged
    over the sides. Straight arms are the one thing a handstand cannot fake, and
    this is why the feature is here even though the model is worst at elbows
    (roughly two thirds of hold frames have the far elbow visible at all).
``banana`` — **0 L, target; over :data:`BANANA_TOLERANCE_L` is an arched back**
    The signed distance of the hip midpoint from the straight line joining the
    shoulder midpoint to the ankle midpoint. Zero is a body whose three midpoints
    are collinear; large is the banana shape, the curve a judge calls "arched"
    and the fault classifier calls a shape fault. Signed towards the direction
    the athlete faces — read off the nose, so positive is the belly side of the
    curve and negative the back side whichever way round the camera was — which
    makes the sign as reliable as the nose's own clearance of the line, a couple
    of hundredths of a body length on a real clip. The magnitude is the
    measurement; the sign says which way the body curved.
``head`` — **0° to ~30°; a dropped head is over :data:`HEAD_FLEXION_DEG`**
    The angle in degrees of the shoulder→nose vector against the torso, taken as
    the vector from the hip midpoint to the shoulder midpoint. In an inverted body
    that points down the image, towards the head, which is where a head in line
    with the body points too — a handstand's head hangs *past* the shoulders
    towards the floor, so its nose is below the shoulder line and not above it. So
    0 is a head in line with the spine, 15–30 is a relaxed handstand head with the
    nose a little in front of the chest, and 90 is a head hanging straight down,
    looking at the floor between the hands: a fault in a hold and ordinary while
    kicking up. Unsigned on purpose: which side of the spine the nose is on is the
    only facing cue this schema has in a side view, so a head thrown back and a
    head dropped forward differ by the sign of the same number, and both are the
    same fault.
``leg_separation`` — **~0° in a side view; over :data:`SPLIT_DEG` is a split**
    The angle in degrees between the left and the right hip→ankle vector. In the
    side view of a line handstand the two legs overlap and this is small; a split
    or a stag in a front or three-quarter view is a large one, which is what
    separates a line from a straddle (#73) and a stag (#57). It needs both sides
    visible, so it is NaN on a clip where only one leg is ever seen.
``hand_width`` — **~1, i.e. hands about shoulder-width apart**
    The wrist distance over the shoulder distance, in the image plane. In a front
    view it is a real number a judge would measure; in the side view both
    distances are foreshortened and the ratio is kept only as a shape flag — it is
    in the table because #73 and #57 need to tell a straddle from a split, and
    because a fault classifier that is handed a ratio it cannot trust in most of
    the dataset should learn to ignore it.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import sys
import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from handstand import bodyframe, phases
from handstand.paths import data_dir
from handstand.postprocess import DEFAULT_SOURCE, SOURCES

__all__ = [
    "ABOVE",
    "BANANA_TOLERANCE_L",
    "BELOW",
    "BENT_ELBOW_DEG",
    "BODY_ANGLE_TOL_DEG",
    "BODY_PARTS",
    "EITHER",
    "FEATURES",
    "FEATURE_NAMES",
    "FEATURES_DIRNAME",
    "HOLD_PHASE",
    "HOLD_SUMMARY_COLUMNS",
    "HOLD_SUMMARY_NAME",
    "HEAD_FLEXION_DEG",
    "LINE_TOLERANCE_L",
    "LISTED_CLIPS",
    "MIN_SEGMENT_L",
    "NOSE",
    "NO_HOLD",
    "OPEN_SHOULDER_DEG",
    "PIKE_HIP_DEG",
    "REPORTS_DIRNAME",
    "SPLIT_DEG",
    "STRAIGHT_KNEE_DEG",
    "TOLERANCES",
    "WANTED_JOINTS",
    "BodyFeatures",
    "BodyTrack",
    "ClipFeatures",
    "ClipReport",
    "ClipStats",
    "Feature",
    "available_clips",
    "body_features",
    "body_frame_track",
    "build_arg_parser",
    "extract_clip",
    "fault_line",
    "fails_tolerance",
    "feature_table",
    "hold_rows",
    "hold_summary_table",
    "input_dir",
    "main",
    "measure",
    "missing_input_message",
    "missing_phases_message",
    "output_dir",
    "phases_dir",
    "plot_clip",
    "plot_path",
    "ranked",
    "read_clip_features",
    "reports_dir",
    "run_clip",
    "side_view_assumed",
    "summary",
    "write_hold_summary",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Output root under ``<data_dir>/``, one directory per source exactly as
#: ``processed/`` and ``phases/`` have one, so one source's features never
#: overwrite another's.
FEATURES_DIRNAME = "features"

#: The per-hold table of one source: every clip's holds in one file.
HOLD_SUMMARY_NAME = "hold_summary.csv"

#: Where :func:`plot_clip` writes, relative to ``<data_dir>/``. The same directory
#: every other stage's report goes to, so everything a person reads is in one
#: place.
REPORTS_DIRNAME = "reports"

#: The two sides, in the order the schema names them.
SIDES: tuple[str, ...] = ("left", "right")

#: The nose, the only joint that says which way the athlete is facing, and so the
#: only one that gives ``banana`` and ``head`` a sign.
NOSE = "nose"

#: The six body parts every feature is read off, in the order the body is read
#: from the hands up: the wrists place the origin, the shoulders and hips give the
#: torso, the knees the middle of the legs and the ankles their ends. A side view
#: sees one of each pair, which is what :func:`_side_mean` is for.
BODY_PARTS: tuple[str, ...] = ("wrist", "elbow", "shoulder", "hip", "knee", "ankle")

#: Every joint the features are read off, as the schema names it. A source that
#: does not report one of them leaves it NaN and the features that need it NaN
#: with it, which is what makes this module model-agnostic.
WANTED_JOINTS: tuple[str, ...] = (
    *(f"{side}_{part}" for part in BODY_PARTS for side in SIDES),
    NOSE,
)

#: The phase whose frames are the handstand, i.e. the frames a score is made of.
HOLD_PHASE = "hold"

#: ``hold_id`` outside a hold, and the clip's frames when it has no hold at all.
#: Re-used from :mod:`handstand.phases` so the two modules cannot disagree about
#: what "no hold" looks like in a column.
NO_HOLD = phases.NO_HOLD

#: A span shorter than this many body lengths has no direction and no magnitude to
#: measure against — a shoulder midpoint sitting exactly on a hip is a degenerate
#: frame, not a 0° angle, and a pair of shoulders projected onto one another is
#: not a shoulder width, so both are reported as no measurement.
MIN_SEGMENT_L = 1e-9

#: How far the worst stacking error may be, in body lengths, before the line is
#: bent. A tenth of the body is about the width of the athlete's own shoulder, so
#: under it the stack is over the hands as far as a judge or a model can tell.
LINE_TOLERANCE_L = 0.1

#: How far the wrist→ankle line may lean from vertical, in degrees, before the
#: body is not straight any more. Well inside the 35° #21 gives up a hold at,
#: because this one is a form number and not a boundary.
BODY_ANGLE_TOL_DEG = 10.0

#: The hip angle under which a hold is a pike, in degrees. #21 calls anything
#: within 35° of vertical a hold; this is the shape inside that window.
PIKE_HIP_DEG = 165.0

#: The shoulder angle under which the shoulders are closed and the chest arched
#: open, in degrees. "Open shoulders" is the one piece of style advice every
#: handstand judge gives, and it is an angle.
OPEN_SHOULDER_DEG = 160.0

#: The knee angle under which the legs are bent rather than locked, in degrees.
STRAIGHT_KNEE_DEG = 165.0

#: The elbow angle under which the arms are bent, in degrees. A bent arm in a
#: handstand is a severe fault rather than a style, so the tolerance is the
#: loosest of the four joint angles: the model is least sure about elbows, and a
#: fault that fires on a 20° error is a fault nobody trusts.
BENT_ELBOW_DEG = 160.0

#: How far the hip midpoint may sit from the shoulder→ankle line, in body
#: lengths, before the shape is an arched back rather than a line. A tenth of the
#: body again, which is the same yardstick as the stacking tolerance: at that
#: size the curve is visible without measuring it.
BANANA_TOLERANCE_L = 0.1

#: How far the head may leave the torso's line, in degrees, before it is dropped
#: rather than merely looking. A neutral head in a handstand reads 0–30° because
#: the athlete is looking at the floor between their hands.
HEAD_FLEXION_DEG = 60.0

#: The leg separation over which the legs are apart in the image rather than
#: together, in degrees. Small in a side view of a line; large in a front view of
#: a split or a stag, which is what #73 and #57 read it for.
SPLIT_DEG = 30.0

#: How many clips the run summary names per ranking. A list a person reads is
#: short; the whole thing is two filters away in :data:`HOLD_SUMMARY_NAME`.
LISTED_CLIPS = 5

#: Rounding of a feature in the parquet, per feature: five decimals on lengths and
#: ratios, two on degrees. Enough to see a change between two frames, not so much
#: that a float's last bits turn into a difference.
_LENGTH_DIGITS = 5
_ANGLE_DIGITS = 2


@dataclasses.dataclass(frozen=True)
class Feature:
    """One feature: what the column is called, the unit it is in, and its target.

    The dataclass is the machine-readable half of the docstring's feature list, and
    the three numbers are what the plot draws (unit on the axis, target as a line
    to compare against) and what the hold summary's columns are named after.
    """

    name: str
    unit: str
    #: The value a good line handstand shows, as the docstring's table quotes it.
    target: float
    digits: int = _LENGTH_DIGITS


#: Every feature, in the order the parquet and the hold summary carry them. The
#: order is the order a judge watches an athlete in: where the stack is, how
#: straight the line is, the joints from the shoulders down, then the shape.
FEATURES: tuple[Feature, ...] = (
    Feature("off_shoulder", "L", 0.0),
    Feature("off_hip", "L", 0.0),
    Feature("off_knee", "L", 0.0),
    Feature("off_ankle", "L", 0.0),
    Feature("line_deviation", "L", 0.0),
    Feature("body_angle", "deg", 0.0, _ANGLE_DIGITS),
    Feature("shoulder_angle", "deg", 180.0, _ANGLE_DIGITS),
    Feature("hip_angle", "deg", 180.0, _ANGLE_DIGITS),
    Feature("knee_angle", "deg", 180.0, _ANGLE_DIGITS),
    Feature("elbow_angle", "deg", 180.0, _ANGLE_DIGITS),
    Feature("banana", "L", 0.0),
    Feature("head", "deg", 0.0, _ANGLE_DIGITS),
    Feature("leg_separation", "deg", 0.0, _ANGLE_DIGITS),
    # No target in a side view: see the docstring. 1.0 is the front-view answer
    # (hands about shoulder-width apart), drawn on the plot as the reference a
    # frontal recording would be read against.
    Feature("hand_width", "ratio", 1.0),
)

#: The feature names alone, in :data:`FEATURES` order, for the loops that do not
#: care about the rest.
FEATURE_NAMES: tuple[str, ...] = tuple(feature.name for feature in FEATURES)

_BY_NAME: Mapping[str, Feature] = {feature.name: feature for feature in FEATURES}

#: The per-hold table's columns, in write order. Every feature contributes a
#: median and an inter-quartile range, so the table answers "what shape was this
#: hold" and "how steady was it" in the same row.
HOLD_SUMMARY_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "source",
    "hold_id",
    "hold_frames",
    "valid_frames",
    "hold_start_ms",
    "hold_end_ms",
    "hold_duration_s",
    *(f"{feature.name}_{stat}" for feature in FEATURES for stat in ("median", "iqr")),
)


# --------------------------------------------------------------------------- #
# The body frame, one clip at a time
# --------------------------------------------------------------------------- #


def _joint_columns(joints: Sequence[str], wanted: Sequence[str]) -> list[int]:
    """The columns of ``wanted`` this clip has, in that order.

    A source that does not report one of them — Apple Vision has no
    ``foot_index``, and a future model may have no elbows — simply has no column
    for it, and the measurement over these names skips it rather than padding it
    with zeros, which would put a phantom joint at the origin.
    """
    return [list(joints).index(name) for name in wanted if name in joints]


def _pixel_midpoint(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    columns: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """A joint group's per-frame midpoint over its visible members, in pixels.

    One wrist is enough to place the origin and one ankle is enough to say which
    way up the body is, so a side-on clip with one wrist in shot is still
    measured; a frame where no member of the group is visible is left NaN rather
    than averaged over zeros, which for the wrist group would be a body frame
    placed at ``(0, 0)`` of the image and every feature measured against it.
    """
    frames = x.shape[0]
    if not columns:
        return np.full(frames, np.nan), np.full(frames, np.nan)
    known = np.asarray(valid, dtype=bool)[:, list(columns)]
    count = known.sum(axis=1)
    safe = np.maximum(count, 1)
    mid_x = np.where(known, x[:, list(columns)], 0.0).sum(axis=1) / safe
    mid_y = np.where(known, y[:, list(columns)], 0.0).sum(axis=1) / safe
    visible = count > 0
    return np.where(visible, mid_x, np.nan), np.where(visible, mid_y, np.nan)


@dataclasses.dataclass(frozen=True)
class BodyTrack:
    """A clip's joints in the body frame, with a name for every column.

    ``uv`` is ``(frames, joints, 2)`` in the units of :mod:`handstand.bodyframe`:
    the origin is the wrist midpoint of *that* frame, ``u`` is to the right and
    ``v`` is up, both in body lengths. A joint that was not visible on a frame is
    NaN there, so every measurement below can be written as "is this finite" and a
    feature the frame cannot support comes out NaN instead of a number.
    """

    uv: np.ndarray
    joints: tuple[str, ...]

    @property
    def frames(self) -> int:
        """How many frames the clip has."""
        return int(self.uv.shape[0])

    def point(self, name: str) -> np.ndarray:
        """The ``(frames, 2)`` body-frame track of ``name``.

        All NaN when this clip's schema has no such joint, so a caller can ask for
        any name in :data:`WANTED_JOINTS` and get NaN rather than a KeyError.
        """
        if name not in self.joints:
            return np.full((self.frames, 2), np.nan)
        return self.uv[:, self.joints.index(name), :]


def body_frame_track(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    joints: Sequence[str],
    body_length: float,
) -> BodyTrack:
    """A processed clip as a :class:`BodyTrack`: the joints in the body frame.

    This is :func:`handstand.bodyframe.to_body_frame` over a whole clip, one frame
    at a time. The origin is the wrist midpoint of *each* frame, which is a
    per-frame quantity and the one thing :func:`to_body_frame` takes a scalar for,
    so the conversion is a loop over a few hundred frames of four numbers each —
    the same trade :mod:`handstand.phases` makes for the same reason. A frame
    whose origin is NaN, because no wrist was visible on it, comes back as NaN
    throughout: a body frame placed at the image's own origin would measure every
    offset against a place the athlete is not.
    """
    if x.ndim != 2 or y.shape != x.shape or valid.shape != x.shape:
        raise ValueError("x, y and valid must have the same (frames, joints) shape")
    if x.shape[1] != len(joints):
        raise ValueError(f"x has {x.shape[1]} joint columns but {len(joints)} joint names")
    wrist_columns = _joint_columns(joints, phases.WRIST_JOINTS)
    origin_x, origin_y = _pixel_midpoint(x, y, valid, wrist_columns)
    uv = np.array(
        [
            bodyframe.to_body_frame(
                x[index], y[index], origin_x[index], origin_y[index], body_length
            )
            for index in range(x.shape[0])
        ],
        dtype=np.float64,
    ).reshape(x.shape[0], len(joints), 2)
    seen = np.asarray(valid, dtype=bool)
    return BodyTrack(uv=np.where(seen[:, :, None], uv, np.nan), joints=tuple(joints))


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def _side_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The two sides of a measurement as one: the mean of both, one where there is
    one, and NaN where there is none.

    A side-on recording sees the near side and hides the far one, so averaging the
    two and dividing by two is not an option and dropping the frame is worse; the
    one side that *is* there measured the athlete just as well.
    """
    out = np.where(np.isfinite(left), left, right)
    both = np.isfinite(left) & np.isfinite(right)
    return np.where(both, (left + right) / 2.0, out)


def _angle_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The unsigned angle in degrees between two ``(frames, 2)`` vectors, in ``[0, 180]``.

    ``atan2`` of the cross against the dot rather than ``arccos`` of the
    normalised dot, so a nearly straight angle stays accurate and a NaN in either
    vector comes out NaN rather than being clipped into the range.
    """
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    dot = first[:, 0] * second[:, 0] + first[:, 1] * second[:, 1]
    with np.errstate(invalid="ignore"):
        angle = np.degrees(np.arctan2(np.abs(cross), dot))
    usable = (
        np.isfinite(cross)
        & np.isfinite(dot)
        & (np.hypot(first[:, 0], first[:, 1]) > MIN_SEGMENT_L)
        & (np.hypot(second[:, 0], second[:, 1]) > MIN_SEGMENT_L)
    )
    return np.where(usable, angle, np.nan)


def _angle_at(first: np.ndarray, middle: np.ndarray, last: np.ndarray) -> np.ndarray:
    """The interior angle in degrees at ``middle`` of the chain first→middle→last.

    A joint angle is the angle between the two limbs *at* the joint, which is why
    a straight arm and a straight leg both read 180° rather than 0°: the athlete
    is upside down, so the two segments point in opposite directions from it.
    """
    return _angle_between(first - middle, last - middle)


def _row_max(values: np.ndarray) -> np.ndarray:
    """The largest finite value of each row, NaN where a row is all NaN.

    ``nanmax`` warns on an all-NaN row, and a frame whose four stacking stations
    are all invisible is exactly that, so the rows without a finite value are
    taken out before the maximum is taken.
    """
    if values.size == 0 or values.shape[1] == 0:
        return np.full(values.shape[0], np.nan)
    out = np.full(values.shape[0], np.nan)
    rows = np.isfinite(values).any(axis=1)
    masked = np.where(np.isfinite(values), values, -np.inf)
    out[rows] = masked[rows].max(axis=1)
    return out


def _facing_sign(shoulder_mid: np.ndarray, ankle_mid: np.ndarray, nose: np.ndarray) -> np.ndarray:
    """Which way the athlete faces: ``+1``, ``-1``, or NaN when the nose is on the line.

    The body's own axis — shoulder midpoint to ankle midpoint, which in a handstand
    runs up the image — has a normal to it, and the nose is on one side of that
    line or the other. The sign is what makes ``banana`` anatomical rather than
    directional: without it a clip shot from the other side of the athlete, or one
    flipped horizontally, would report every curve as the mirror of what it is. NaN
    when the nose is on the line, because a curve with no known direction has no
    known sign either — and the nose's clearance of that line is a couple of
    hundredths of a body length on a real clip, which is the limit on how much the
    sign is worth.
    """
    axis = ankle_mid - shoulder_mid
    span = np.hypot(axis[:, 0], axis[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        across = np.stack((-axis[:, 1], axis[:, 0]), axis=1) / span[:, None]
    front = ((nose - shoulder_mid) * across).sum(axis=1)
    return np.where(front > 0.0, 1.0, np.where(front < 0.0, -1.0, np.nan))


def _banana(
    shoulder_mid: np.ndarray,
    hip_mid: np.ndarray,
    ankle_mid: np.ndarray,
    nose: np.ndarray,
) -> np.ndarray:
    """The signed distance of the hip midpoint from the shoulder→ankle line, in L.

    A line handstand's three midpoints are collinear, so this is 0 for one and
    grows with how far the body curves — the banana shape, the arched back a judge
    calls a shape fault. The sign is towards the direction the athlete faces, so
    positive is the belly side of the curve and negative the back side.
    """
    axis = ankle_mid - shoulder_mid
    span = np.hypot(axis[:, 0], axis[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        across = np.stack((-axis[:, 1], axis[:, 0]), axis=1) / span[:, None]
    offset = ((hip_mid - shoulder_mid) * across).sum(axis=1)
    return offset * _facing_sign(shoulder_mid, ankle_mid, nose)


def _head(hip_mid: np.ndarray, shoulder_mid: np.ndarray, nose: np.ndarray) -> np.ndarray:
    """The angle of the shoulder→nose vector against the spine, in degrees, unsigned.

    The reference is the torso direction as a vector from the hip midpoint to the
    shoulder midpoint. In an inverted body that points *down* the image, towards
    the head, which is where a head in line with the body points too — a
    handstand's head hangs past the shoulders towards the floor, so its nose is
    below the shoulder line, not above it. So 0 is a head in line with the spine,
    15–30 is a relaxed handstand head with the nose a little in front of the
    chest, and 90 is a head hanging straight down looking at the floor.

    There is no sign, and that is a measurement rather than a gap: a nose displaced
    to either side of the spine is the *only* facing cue this schema has in a side
    view, so a head thrown back and a head dropped forward differ by the sign of
    the same small number, and signing this against the nose would be signing it
    against itself. Both are faults, and both are the same number, so the
    magnitude is the feature.
    """
    return _angle_between(shoulder_mid - hip_mid, nose - shoulder_mid)


# --------------------------------------------------------------------------- #
# One clip's features
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class BodyFeatures:
    """Every feature of a clip's frames, measured from its joints in the body frame.

    ``values`` holds one ``(frames,)`` array per feature, NaN wherever the frame
    could not support that feature, and ``valid`` says whether the frame could
    support them all: a visible joint on both sides of the body frame's own
    measurement (one side is enough), a placeable origin, and no trainer in the
    way. The nose is deliberately not part of it — it only gives ``banana`` its
    sign and ``head`` its number, so a frame without one is a frame with two NaNs
    rather than a frame to throw away.
    """

    values: dict[str, np.ndarray]
    valid: np.ndarray

    @property
    def frames(self) -> int:
        """How many frames the measurements cover."""
        return int(self.valid.size)

    def value(self, name: str) -> np.ndarray:
        """One feature's ``(frames,)`` values, NaN for a name that is not a feature."""
        return self.values.get(name, np.full(self.frames, np.nan))


def _group_midpoint(points: Mapping[str, np.ndarray], part: str, frames: int) -> np.ndarray:
    """The ``(frames, 2)`` midpoint of both sides of one body part.

    The mean where both sides are visible, the one that is where only one is, NaN
    where neither is — the same rule as :func:`_side_mean`, on points rather than
    on a measurement, and for the same reason: the stacking offsets are of a
    *midpoint*, and a side-on clip only ever has one of each pair.
    """
    left = points.get(f"left_{part}", np.full((frames, 2), np.nan))
    right = points.get(f"right_{part}", np.full((frames, 2), np.nan))
    return _side_mean(left, right)


def _side_angle(
    points: Mapping[str, np.ndarray], parts: tuple[str, str, str], side: str
) -> np.ndarray:
    """One side's joint angle: the angle at the middle of the three named parts.

    The chain is named from the first joint through the middle one to the last, so
    ``("hip", "shoulder", "wrist")`` is the angle at the shoulder.
    """
    first, middle, last = (f"{side}_{part}" for part in parts)
    return _angle_at(points[first], points[middle], points[last])


def _both_sides(points: Mapping[str, np.ndarray], parts: tuple[str, str, str]) -> np.ndarray:
    """A joint angle averaged over the sides, or taken from whichever side is there."""
    return _side_mean(_side_angle(points, parts, "left"), _side_angle(points, parts, "right"))


def body_features(track: BodyTrack) -> BodyFeatures:
    """Every feature of :data:`FEATURES` for a whole clip, from its body-frame joints.

    The input is a :class:`BodyTrack` — a clip's joints as ``(frames, 2)`` points
    with NaN where a joint was not visible — and the output is one ``(frames,)``
    array per feature. This is the whole geometry of the module and it knows
    nothing about files, phases or parquet, which is why the tests can drive it
    from a pose written by hand.
    """
    frames = track.frames
    points = {name: track.point(name) for name in WANTED_JOINTS}
    parts = {part: _group_midpoint(points, part, frames) for part in BODY_PARTS}
    shoulder_mid, hip_mid, knee_mid, ankle_mid = (
        parts["shoulder"],
        parts["hip"],
        parts["knee"],
        parts["ankle"],
    )
    nose = points[NOSE]

    # The stacking offsets are the u of the midpoints: the body frame's origin is
    # the wrist midpoint, so u = 0 *is* the vertical line through the hands, and a
    # feature of 0 needs no reference line to be measured against.
    offsets = [mid[:, 0] for mid in (shoulder_mid, hip_mid, knee_mid, ankle_mid)]

    values: dict[str, np.ndarray] = {
        "off_shoulder": offsets[0],
        "off_hip": offsets[1],
        "off_knee": offsets[2],
        "off_ankle": offsets[3],
        # The worst of the four as a magnitude: a stack that has drifted to the
        # left is as wrong as one that has drifted to the right, and this is the
        # one number both are judged by.
        "line_deviation": _row_max(np.abs(np.stack(offsets, axis=1))),
        # atan2 of the horizontal against the vertical component: the lean of the
        # wrist->ankle vector away from straight up, signed by which way it leans,
        # which is the same measurement #21's hold condition thresholds at 35 deg.
        "body_angle": np.degrees(np.arctan2(offsets[3], ankle_mid[:, 1])),
        "shoulder_angle": _both_sides(points, ("hip", "shoulder", "wrist")),
        "hip_angle": _both_sides(points, ("shoulder", "hip", "knee")),
        "knee_angle": _both_sides(points, ("hip", "knee", "ankle")),
        "elbow_angle": _both_sides(points, ("shoulder", "elbow", "wrist")),
        "banana": _banana(shoulder_mid, hip_mid, ankle_mid, nose),
        "head": _head(hip_mid, shoulder_mid, nose),
    }
    # The last two are the only features that *are* a measurement between the two
    # sides, so they need both of them and are NaN with one — a leg separation of
    # 0 because the far leg was never seen is a leg separation nobody measured.
    values["leg_separation"] = _angle_between(
        points["left_ankle"] - points["left_hip"], points["right_ankle"] - points["right_hip"]
    )
    values["hand_width"] = _ratio(
        _distance(points["left_wrist"], points["right_wrist"]),
        _distance(points["left_shoulder"], points["right_shoulder"]),
    )

    # Every part the features are read off has a midpoint, and the wrist group is
    # where the origin comes from, so a frame whose body frame could not be placed
    # at all — no wrist visible, or a source that reports no wrists — is not
    # measurable however good the rest of the pose looks.
    valid = np.ones(frames, dtype=bool)
    for mid in parts.values():
        valid &= np.isfinite(mid[:, 0]) & np.isfinite(mid[:, 1])
    return BodyFeatures(values=values, valid=valid)


def _distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The distance between two ``(frames, 2)`` points, NaN where either is missing."""
    delta = first - second
    with np.errstate(invalid="ignore"):
        return np.hypot(delta[:, 0], delta[:, 1])


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / denominator``, NaN where the denominator is missing or zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            np.isfinite(denominator) & (np.abs(denominator) > MIN_SEGMENT_L),
            numerator / denominator,
            np.nan,
        )


def side_view_assumed(frames: int) -> np.ndarray:
    """``frames`` true values: every frame is assumed to be a side view.

    The recording protocol asks for one, and the angles in this module are 2D
    projections read as if it were true. #74 detects the camera angle per clip and
    is the stage that turns this into a measurement; until it lands, this column
    is where a later stage filters out the one thing that makes a feature
    meaningless — a clip shot head-on, where a straight arm points at the camera and
    reads as bent.
    """
    return np.ones(frames, dtype=bool)


def _unusable_features(frames: int) -> BodyFeatures:
    """The features of a clip with no scale: nothing measured, every frame invalid."""
    return BodyFeatures(
        values={name: np.full(frames, np.nan) for name in FEATURE_NAMES},
        valid=np.zeros(frames, dtype=bool),
    )


@dataclasses.dataclass(frozen=True)
class ClipFeatures:
    """One clip's features, with the frame identity and the phase labels they sit in.

    ``phase`` and ``hold_id`` come straight from #21 and are carried here so a
    later stage can filter on "inside a hold" without opening a second file, and
    so the hold summary can be built without one.
    """

    clip_id: str
    source: str
    frame_idx: np.ndarray
    t_ms: np.ndarray
    phase: np.ndarray
    hold_id: np.ndarray
    measured: BodyFeatures
    unusable_reason: str = ""

    @property
    def frames(self) -> int:
        """How many frames the clip has."""
        return int(self.frame_idx.size)

    @property
    def valid(self) -> np.ndarray:
        """Which frames every feature could be measured on."""
        return self.measured.valid

    @property
    def usable(self) -> bool:
        """Does this clip have a scale, and so any features at all?"""
        return not self.unusable_reason

    def value(self, name: str) -> np.ndarray:
        """One feature's ``(frames,)`` values."""
        return self.measured.value(name)

    def hold_ids(self) -> tuple[int, ...]:
        """The clip's hold numbers, in time order."""
        return tuple(sorted({int(value) for value in self.hold_id if int(value) >= 0}))

    def hold_mask(self, hold_id: int) -> np.ndarray:
        """The frames of one hold, inclusive of the unknown frames bridged into it."""
        return self.hold_id == hold_id

    def measurable_hold_mask(self, hold_id: int) -> np.ndarray:
        """The frames of one hold the features could actually be measured on.

        This is the filter every number about a hold is taken over: inside the
        hold, and measurable. A hold with a trainer in front of the camera for a
        third of it is then summarised over the two thirds a human could see, and
        the row says how many frames that was.
        """
        return self.hold_mask(hold_id) & self.valid

    def longest_hold_id(self) -> int:
        """The clip's longest hold, or :data:`NO_HOLD` when it has none.

        The longest hold of a clip is the one that is representative of it: a short
        one is a hop on the hands, and the median of a hop says more about the
        athlete's balance than about their form. Ties go to the earlier hold.
        """
        best, best_s = NO_HOLD, -1.0
        for hold_id in self.hold_ids():
            frames = np.flatnonzero(self.hold_mask(hold_id))
            span = float(self.t_ms[frames[-1]] - self.t_ms[frames[0]])
            if span > best_s:
                best, best_s = hold_id, span
        return best

    def hold_duration_s(self, hold_id: int) -> float:
        """How long one hold lasted, first frame to last, in seconds."""
        frames = np.flatnonzero(self.hold_mask(hold_id))
        if frames.size == 0:
            return 0.0
        return float(self.t_ms[frames[-1]] - self.t_ms[frames[0]]) / 1000.0


def extract_clip(
    clip_id: str,
    source: str,
    t_ms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    joints: Sequence[str],
    body_length: float,
    *,
    phase: Sequence[str] | None = None,
    hold_id: Sequence[int] | None = None,
    trainer_contact: np.ndarray | None = None,
    reason: str = "",
) -> ClipFeatures:
    """Measure one clip's features, from its processed arrays to its per-frame answer.

    ``phase`` and ``hold_id`` are the labels of :mod:`handstand.phases`; without
    them every frame is ``unknown`` outside a hold, which is the honest answer for
    a clip nobody has labelled. ``trainer_contact`` is that module's flag for the
    frames where the keypoints are two bodies stitched together: the features are
    measured on them as everything else is — the numbers are in the table so a
    person can see what the stitched frame said — and ``valid`` is False, so no
    measurement takes them for the athlete's.

    A clip with no body length comes back with every feature NaN, every frame
    invalid and :attr:`ClipFeatures.unusable_reason` set: without ``L`` there is
    no unit to measure an offset or an angle in, and a feature in pixels is not a
    feature.
    """
    times = np.asarray(t_ms, dtype=np.int64)
    frames = times.size
    frame_idx = np.arange(frames, dtype=np.int64)
    names = (
        np.full(frames, phases.PHASES[phases.UNKNOWN], dtype=object)
        if phase is None
        else np.asarray(phase, dtype=object)
    )
    holds = (
        np.full(frames, NO_HOLD, dtype=np.int64)
        if hold_id is None
        else np.asarray(hold_id, dtype=np.int64)
    )
    if names.shape != (frames,):
        raise ValueError(f"phase has {names.shape} entries, t_ms has {frames}")
    if holds.shape != (frames,):
        raise ValueError(f"hold_id has {holds.shape} entries, t_ms has {frames}")
    if not reason and (not math.isfinite(body_length) or body_length <= 0.0):
        reason = f"no body length ({body_length!r} px)"

    measured: BodyFeatures
    if reason:
        measured = _unusable_features(frames)
    else:
        measured = body_features(body_frame_track(x, y, valid, joints, body_length))
        if trainer_contact is not None:
            contact = np.asarray(trainer_contact, dtype=bool)
            if contact.shape != (frames,):
                raise ValueError(
                    f"trainer_contact must have one entry per frame ({frames}), got {contact.shape}"
                )
            measured = dataclasses.replace(measured, valid=measured.valid & ~contact)
    return ClipFeatures(
        clip_id=clip_id,
        source=source,
        frame_idx=frame_idx,
        t_ms=times,
        phase=names,
        hold_id=holds,
        measured=measured,
        unusable_reason=reason,
    )


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def _rounded(values: np.ndarray, digits: int) -> np.ndarray:
    """``values`` rounded to ``digits`` — NaN stays NaN, so an unmeasured feature is
    still visibly unmeasured in the table."""
    return np.round(np.asarray(values, dtype=np.float64), digits)


def _rounded_scalar(value: float, digits: int) -> float:
    """One number rounded for the CSV, with NaN left as NaN rather than written out."""
    return round(value, digits) if math.isfinite(value) else math.nan


def feature_table(features: ClipFeatures) -> pd.DataFrame:
    """One clip's per-frame features as a table, in frame order.

    The frame's identity first, then the labels it carries from #21, then the two
    columns that say whether to believe the row at all (``valid`` and
    ``side_view``), then every feature in :data:`FEATURES` order. A feature the
    frame could not support is NaN, which survives the rounding on purpose.
    """
    columns: dict[str, object] = {
        "frame_idx": features.frame_idx,
        "t_ms": features.t_ms,
        "phase": features.phase,
        "hold_id": features.hold_id,
        "valid": features.valid,
        "side_view": side_view_assumed(features.frames),
    }
    for feature in FEATURES:
        columns[feature.name] = _rounded(features.value(feature.name), feature.digits)
    return pd.DataFrame(columns)


def _spread(values: np.ndarray) -> tuple[float, float]:
    """The median and the inter-quartile range of the finite values of ``values``.

    The median because a hold is judged on its shape rather than on one frame of
    it, and the IQR because a handstand that is perfect for 90 % of a hold and
    bent for the rest is a different hold from one that is the same all the way
    through — which is the difference a fault classifier and a score both need.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.median(finite)), float(np.percentile(finite, 75) - np.percentile(finite, 25))


def hold_rows(features: ClipFeatures) -> list[dict[str, object]]:
    """One row per hold of the clip: every feature's median and IQR, and its length.

    Measured over the hold's *measurable* frames
    (:meth:`ClipFeatures.measurable_hold_mask`), so a hold nobody could see for
    part of its length is summarised over the rest, and the row carries both the
    number of frames in the hold and the number of frames the median is over.
    """
    rows: list[dict[str, object]] = []
    for hold_id in features.hold_ids():
        mask = features.measurable_hold_mask(hold_id)
        frames = np.flatnonzero(features.hold_mask(hold_id))
        row: dict[str, object] = {
            "clip_id": features.clip_id,
            "source": features.source,
            "hold_id": hold_id,
            "hold_frames": int(frames.size),
            "valid_frames": int(mask.sum()),
            "hold_start_ms": int(features.t_ms[frames[0]]),
            "hold_end_ms": int(features.t_ms[frames[-1]]),
            "hold_duration_s": round(features.hold_duration_s(hold_id), 3),
        }
        for feature in FEATURES:
            median, iqr = _spread(features.value(feature.name)[mask])
            row[f"{feature.name}_median"] = _rounded_scalar(median, feature.digits)
            row[f"{feature.name}_iqr"] = _rounded_scalar(iqr, feature.digits)
        rows.append(row)
    return rows


def hold_summary_table(features: Sequence[ClipFeatures]) -> pd.DataFrame:
    """The holds of one or more clips as the hold table's rows, in clip and hold order."""
    rows = [row for clip in features for row in hold_rows(clip)]
    return pd.DataFrame(rows, columns=list(HOLD_SUMMARY_COLUMNS))


def write_hold_summary(path: str | pathlib.Path, table: pd.DataFrame) -> pathlib.Path:
    """Write the hold table, replacing the rows of the clips in ``table``.

    The file is per source, not per run: a run over ten clips must not delete the
    other hundred and seventy, so the rows of the clips this run covered are
    dropped and these written in their place, sorted by clip and hold. A run over
    every clip therefore writes the whole dataset and a re-run of one clip refreshes
    just that clip. It is written to a temporary file and renamed, so an
    interrupted run cannot leave a half-written table behind.
    """
    path = pathlib.Path(path)
    keep = pd.DataFrame(columns=list(HOLD_SUMMARY_COLUMNS))
    if path.is_file():
        existing = pd.read_csv(path)
        missing = [column for column in HOLD_SUMMARY_COLUMNS if column not in existing.columns]
        if missing:
            raise ValueError(f"{path}: hold summary is missing column(s) {missing}")
        clips = set(table["clip_id"].astype(str))
        keep = existing[~existing["clip_id"].astype(str).isin(clips)]
    merged = pd.concat([keep, table], ignore_index=True)
    merged = merged.sort_values(["clip_id", "hold_id"], kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        merged.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_clip_features(parquet_path: str | pathlib.Path) -> pd.DataFrame:
    """Read one features parquet back, for the plot and for the later stages.

    :class:`ValueError` when the frame columns are missing: that is a file
    :mod:`handstand.phases` wrote, and this module's own table is what #29 and #34
    read.
    """
    path = pathlib.Path(parquet_path)
    table = pd.read_parquet(path)
    required = ("frame_idx", "t_ms", "phase", "hold_id", "valid", *FEATURE_NAMES)
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(
            f"{path}: features are missing column(s) {missing}; "
            "generate them with: cd pipeline && uv run python -m handstand.features --all"
        )
    return table


# --------------------------------------------------------------------------- #
# What a run reports
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipStats:
    """What one clip's run produced, in the numbers the summary is built from.

    The per-feature arrays are kept rather than a few headline scalars, because the
    dataset questions are per feature ("what is the median ``banana`` over every
    hold in the dataset") and re-reading every parquet to answer them is what this
    stage exists to avoid. The medians of the clip's *longest* hold are the ranking
    key, one per clip, which is the one hold of a clip that represents it.
    """

    clip_id: str
    source: str
    frames: int
    valid_frames: int
    hold_count: int
    hold_frames: int
    measured_hold_frames: int
    longest_hold_id: int
    longest_hold_s: float
    unusable_reason: str = ""
    runtime_seconds: float = 0.0
    #: One array of the finite values of that feature over the clip's measurable
    #: hold frames, pooled by the summary over every clip.
    hold_values: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    #: One median per feature over the clip's longest hold, NaN where it has none.
    longest_medians: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Does this clip have a scale, and so any features at all?"""
        return not self.unusable_reason

    @property
    def has_hold(self) -> bool:
        """Did the clip contain at least one hold?"""
        return self.hold_count > 0

    def line(self) -> str:
        """The one progress line the CLI prints for this clip."""
        scale = "unusable" if not self.usable else f"holds={self.hold_count}"
        line_median = self.longest_medians.get("line_deviation", math.nan)
        hip_median = self.longest_medians.get("hip_angle", math.nan)
        return (
            f"clip  {self.clip_id} frames={self.frames} valid={self.valid_frames} "
            f"{scale} hold_frames={self.hold_frames} longest={self.longest_hold_s:.2f}s "
            f"line={_num(line_median)} hip={_num(hip_median)} "
            f"runtime={self.runtime_seconds:.2f}s"
        )


def _num(value: float, digits: int = 3) -> str:
    """A number for a progress line, or ``n/a`` where there is none."""
    return f"{value:.{digits}f}" if math.isfinite(value) else "n/a"


def measure(features: ClipFeatures, runtime_seconds: float = 0.0) -> ClipStats:
    """Everything one clip's run produced, measured off its features.

    Two sets of numbers per clip: every measurable hold frame's value of every
    feature, which the dataset medians are pooled over, and the median of every
    feature over the clip's longest hold, which is what the clip is ranked by.
    """
    holds = features.hold_ids()
    hold_values: dict[str, np.ndarray] = {}
    for feature in FEATURES:
        collected: list[np.ndarray] = []
        for hold_id in holds:
            mask = features.measurable_hold_mask(hold_id)
            values = features.value(feature.name)[mask]
            collected.append(values[np.isfinite(values)])
        hold_values[feature.name] = (
            np.concatenate(collected) if collected else np.zeros(0, dtype=np.float64)
        )
    longest = features.longest_hold_id()
    longest_medians: dict[str, float] = {}
    if longest != NO_HOLD:
        mask = features.measurable_hold_mask(longest)
        for feature in FEATURES:
            longest_medians[feature.name] = _spread(features.value(feature.name)[mask])[0]
    hold_masks = [features.measurable_hold_mask(hold_id) for hold_id in holds]
    return ClipStats(
        clip_id=features.clip_id,
        source=features.source,
        frames=features.frames,
        valid_frames=int(features.valid.sum()),
        hold_count=len(holds),
        hold_frames=sum(int(features.hold_mask(hold_id).sum()) for hold_id in holds),
        measured_hold_frames=sum(int(mask.sum()) for mask in hold_masks),
        longest_hold_id=longest,
        longest_hold_s=(features.hold_duration_s(longest) if longest != NO_HOLD else 0.0),
        unusable_reason=features.unusable_reason,
        runtime_seconds=runtime_seconds,
        hold_values=hold_values,
        longest_medians=longest_medians,
    )


#: Which side of a target is the wrong side, for the features that have one.
ABOVE = "above"
BELOW = "below"
EITHER = "either"

#: Every feature's target, as the summary counts clips against: the feature, the
#: test its longest-hold median fails, and the numbers the test is. Written once,
#: from the same constants the docstring quotes, so the targets a score is
#: tuned against and the targets this module reports cannot drift apart.
TOLERANCES: tuple[tuple[str, str, float, str], ...] = (
    ("line_deviation", f"over {LINE_TOLERANCE_L:g} L", LINE_TOLERANCE_L, EITHER),
    ("body_angle", f"over {BODY_ANGLE_TOL_DEG:g} deg", BODY_ANGLE_TOL_DEG, EITHER),
    ("shoulder_angle", f"under {OPEN_SHOULDER_DEG:g} deg", OPEN_SHOULDER_DEG, BELOW),
    ("hip_angle", f"under {PIKE_HIP_DEG:g} deg", PIKE_HIP_DEG, BELOW),
    ("knee_angle", f"under {STRAIGHT_KNEE_DEG:g} deg", STRAIGHT_KNEE_DEG, BELOW),
    ("elbow_angle", f"under {BENT_ELBOW_DEG:g} deg", BENT_ELBOW_DEG, BELOW),
    ("banana", f"over {BANANA_TOLERANCE_L:g} L", BANANA_TOLERANCE_L, EITHER),
    ("head", f"over {HEAD_FLEXION_DEG:g} deg", HEAD_FLEXION_DEG, EITHER),
    ("leg_separation", f"over {SPLIT_DEG:g} deg", SPLIT_DEG, ABOVE),
)


def fails_tolerance(value: float, threshold: float, side: str) -> bool:
    """Is one hold's median outside the target for that feature?

    ``side`` says which way is wrong. ``either`` is for the two shape features,
    where bending one way is as much a fault as bending the other, and ``below``
    is for the joint angles, where straight is 180° and any value under the
    threshold is bent however the athlete is facing. An unmeasured median fails
    nothing: a hold the model could not see is not a hold that missed a target.
    """
    if not math.isfinite(value):
        return False
    if side == EITHER:
        return abs(value) > threshold
    if side == BELOW:
        return value < threshold
    return value > threshold


def summary(reports: Sequence[ClipStats]) -> str:
    """The dataset-level answer, printed at the end of a run.

    Four questions, in this order: how many clips are usable and how many have a
    hold; what every feature looks like over all the hold frames of the dataset,
    as a median and an inter-quartile range; how many clips miss each target over
    their longest hold, which is what says whether the targets are worth scoring
    against; and which clips are the extremes — the straightest lines and the
    biggest curves. The two rankings are capped at :data:`LISTED_CLIPS` names
    because a summary that printed a hundred and eighty clip ids would be read
    past; the whole thing is a filter away in :data:`HOLD_SUMMARY_NAME`.
    """
    lines = [f"summary clips={len(reports)}"]
    if not reports:
        return "\n".join(lines)
    usable = [report for report in reports if report.usable]
    with_holds = [report for report in reports if report.has_hold]
    lines[0] += f" usable={len(usable)} with_hold={len(with_holds)}"
    total_hold_frames = sum(report.hold_frames for report in reports)
    lines.append(
        f"  hold frames       {sum(report.measured_hold_frames for report in reports)} "
        f"measurable of {total_hold_frames} in {len(with_holds)} clip(s), "
        f"{sum(report.hold_count for report in reports)} hold(s) in total"
    )
    lines.append("  every hold frame  median and IQR of each feature over all of them")
    for feature in FEATURES:
        pooled = np.concatenate([report.hold_values[feature.name] for report in reports])
        median, iqr = _spread(pooled)
        lines.append(
            f"    {feature.name:<15} median={_num(median, feature.digits + 1):>9} "
            f"iqr={_num(iqr, feature.digits + 1):>8} {feature.unit}"
        )
    lines.append(
        f"  misses its target over the longest hold, of {len(with_holds)} clip(s) with a hold"
    )
    for name, test, threshold, side in TOLERANCES:
        failing = sum(
            1
            for report in with_holds
            if fails_tolerance(report.longest_medians.get(name, math.nan), threshold, side)
        )
        lines.append(f"    {name:<15} {test:<18} {failing:>4}")
    lines.extend(
        _ranking_lines(
            "best line",
            "line_deviation",
            ranked(reports, "line_deviation", largest=False),
            "smallest median line_deviation over the longest hold",
        )
    )
    lines.extend(
        _ranking_lines(
            "most banana",
            "banana",
            ranked(reports, "banana", largest=True, absolute=True),
            "largest median |banana| over the longest hold",
        )
    )
    lines.extend(
        _ranking_lines(
            "banana signed",
            "banana",
            ranked(reports, "banana", largest=True),
            "largest median banana, positive being the side the athlete faces",
        )
    )
    without = [report for report in reports if not report.has_hold]
    if without:
        listed = without[:LISTED_CLIPS]
        names = ", ".join(report.clip_id for report in listed)
        rest = len(without) - len(listed)
        more = "" if rest == 0 else f" (+{rest} more)"
        lines.append(f"  no hold            {names}{more}")
    unusable = [report for report in reports if not report.usable]
    if unusable:
        lines.append(
            "  unusable clips     "
            + ", ".join(f"{report.clip_id} ({report.unusable_reason})" for report in unusable)
        )
    return "\n".join(lines)


def ranked(
    reports: Sequence[ClipStats], feature: str, *, largest: bool = True, absolute: bool = False
) -> list[ClipStats]:
    """The clips with a hold, ranked by their longest hold's median of ``feature``.

    Clips with no hold, and clips whose median is NaN because the longest hold's
    frames could not be measured, are left out rather than ranked as zero. With
    ``absolute`` the magnitude is ranked, which is the right test for the two
    shape features: bending one way is as much a fault as bending the other.
    """
    usable = [
        report
        for report in reports
        if report.has_hold and math.isfinite(report.longest_medians.get(feature, math.nan))
    ]

    def key(report: ClipStats) -> float:
        value = report.longest_medians[feature]
        return abs(value) if absolute else value

    return sorted(usable, key=key, reverse=largest)[:LISTED_CLIPS]


def _ranking_lines(
    title: str, feature: str, reports: Sequence[ClipStats], question: str
) -> list[str]:
    """The heading and the rows of one ranking in the summary."""
    if not reports:
        return [f"  {title:<15} none: no clip has a measurable hold"]
    lines = [f"  {title:<15} {len(reports)} clip(s), {question}"]
    for report in reports:
        lines.append(
            f"    {report.clip_id}  {report.longest_medians[feature]:.3f}  "
            f"hold {report.longest_hold_id}  {report.longest_hold_s:.1f}s"
        )
    return lines


# --------------------------------------------------------------------------- #
# Locations, writing, one clip
# --------------------------------------------------------------------------- #


def input_dir(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pathlib.Path:
    """``<data_dir>/processed/<source>``, the trajectories this stage reads."""
    return phases.input_dir(data, source)


def phases_dir(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pathlib.Path:
    """``<data_dir>/phases/<source>``, the hold labels this stage reads."""
    return phases.output_dir(data, source)


def output_dir(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pathlib.Path:
    """``<data_dir>/features/<source>``, where that source's features go."""
    return pathlib.Path(data) / FEATURES_DIRNAME / source


def missing_phases_message(path: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> str:
    """The error shown when the phase labels have not been generated yet."""
    command = "uv run python -m handstand.phases --all"
    if source != DEFAULT_SOURCE:
        command += f" --source {source}"
    return (
        f"no phases for source {source!r}: {path}\n"
        "generate them first with:\n"
        f"  cd pipeline && {command}"
    )


#: Re-exported from :mod:`handstand.phases` so the message about the missing input
#: of the stage before it is the one that module writes.
missing_input_message = phases.missing_input_message


def available_clips(
    in_root: str | pathlib.Path,
    clips: Sequence[str] | None = None,
    source: str = DEFAULT_SOURCE,
) -> list[str]:
    """The clip ids to measure: ``clips`` as given, else every processed clip found.

    Raises :class:`FileNotFoundError` naming the command that writes the input
    when there is no processed directory for this source at all.
    """
    root = pathlib.Path(in_root)
    if clips:
        return [str(clip) for clip in clips]
    if not root.is_dir():
        raise FileNotFoundError(missing_input_message(root, source))
    return sorted(path.stem for path in root.glob("*.parquet"))


@dataclasses.dataclass(frozen=True)
class ClipReport:
    """What one clip's run produced."""

    clip_id: str
    features: ClipFeatures
    stats: ClipStats
    parquet_path: pathlib.Path
    skipped: bool = False

    def line(self) -> str:
        """The one progress line the CLI prints for this clip."""
        if self.skipped:
            return f"skip  {self.clip_id} ({self.parquet_path.name} exists)"
        return self.stats.line()


def read_phase_labels(
    phases_path: str | pathlib.Path, frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """One clip's ``phase`` and ``hold_id`` columns, checked against its frame count.

    The features are per frame, so the labels have to be per frame too: a phases
    parquet with a different number of rows is not this clip's phases and is
    refused rather than padded or truncated.
    """
    path = pathlib.Path(phases_path)
    table = phases.read_clip_phases(path)
    if len(table) != frames:
        raise ValueError(
            f"{path}: {len(table)} frame(s) of phases for a clip of {frames} frame(s); "
            "re-label it with: cd pipeline && uv run python -m handstand.phases --overwrite"
        )
    return (
        np.asarray(table["phase"].astype(str).to_numpy(), dtype=object),
        table["hold_id"].to_numpy(dtype=np.int64),
    )


def run_clip(
    clip_id: str,
    *,
    in_root: str | pathlib.Path,
    labels_root: str | pathlib.Path,
    out_root: str | pathlib.Path,
    source: str = DEFAULT_SOURCE,
    overwrite: bool = False,
) -> ClipReport:
    """Measure one clip and write its parquet.

    Reads ``<in_root>/<clip_id>.parquet`` and the sidecar beside it, the phase
    labels from ``<labels_root>/<clip_id>.parquet``, and writes
    ``<out_root>/<clip_id>.parquet``. A clip whose parquet is already there is
    skipped unless ``overwrite``, which is what makes a batch over the dataset
    resumable. The parquet is written to a temporary file and renamed last: its
    existence is what marks the clip as done.
    """
    in_path = pathlib.Path(in_root) / f"{clip_id}.parquet"
    if not in_path.is_file():
        raise FileNotFoundError(missing_input_message(in_path, source))
    labels_path = pathlib.Path(labels_root) / f"{clip_id}.parquet"
    if not labels_path.is_file():
        raise FileNotFoundError(missing_phases_message(labels_path, source))
    out_dir = pathlib.Path(out_root)
    parquet_path = out_dir / f"{clip_id}.parquet"

    started = time.perf_counter()
    clip, valid = phases.read_processed(in_path)
    body_length, reason = phases.read_body_length(in_path.with_suffix(".json"))
    phase, hold_id = read_phase_labels(labels_path, clip.frames)
    features = extract_clip(
        clip_id,
        source,
        clip.t_ms,
        clip.x,
        clip.y,
        valid,
        clip.joints,
        body_length,
        phase=phase,
        hold_id=hold_id,
        trainer_contact=clip.trainer_contact,
        reason=reason,
    )
    runtime_seconds = time.perf_counter() - started
    stats = measure(features, runtime_seconds)

    if parquet_path.exists() and not overwrite:
        return ClipReport(
            clip_id=clip_id, features=features, stats=stats, parquet_path=parquet_path, skipped=True
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = parquet_path.with_name(f".{parquet_path.name}.tmp")
    try:
        feature_table(features).to_parquet(tmp, index=False)
        tmp.replace(parquet_path)
    finally:
        tmp.unlink(missing_ok=True)
    return ClipReport(
        clip_id=clip_id, features=features, stats=stats, parquet_path=parquet_path, skipped=False
    )


# --------------------------------------------------------------------------- #
# The plot
# --------------------------------------------------------------------------- #


def reports_dir(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/reports``, where :func:`plot_clip` writes. Created on write."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / REPORTS_DIRNAME


def plot_path(
    data: str | pathlib.Path, clip_id: str, source: str = DEFAULT_SOURCE
) -> pathlib.Path:
    """``<data_dir>/reports/features_<clip_id>.png``, the plot of one clip.

    The name carries the source for a non-default one, the same rule the overlays
    use, so a MediaPipe plot and a Vision plot of the same clip never overwrite
    each other.
    """
    name = f"features_{clip_id}.png"
    if source != DEFAULT_SOURCE:
        name = f"{clip_id}_{source}_features.png"
    return reports_dir(data) / name


def fault_line(feature: Feature) -> tuple[tuple[float, ...], str]:
    """The values a plot panel draws as its fault line, and the label that says which
    side of them is wrong.

    The line a panel wants is the value at which the feature stops being a line
    handstand — the tolerance — not the ideal, which for a joint angle is 180° and
    is usually nowhere near the data: stretching a panel to show 180 on a pike that
    sits at 70 flattens the part of the trace there is something to see. The
    tolerance is written into the axis label either way, so a panel whose fault line
    is off its own range still says where the line is. The four stacking offsets and
    ``hand_width`` have no tolerance and draw their target instead.
    """
    for name, test, threshold, side in TOLERANCES:
        if name == feature.name:
            values = (-threshold, threshold) if side == EITHER else (threshold,)
            return values, test
    return (feature.target,), f"target {feature.target:g} {feature.unit}"


def plot_clip(
    table: pd.DataFrame, out_path: str | pathlib.Path, *, title: str = ""
) -> pathlib.Path:
    """Draw every feature over time for one clip, with the hold frames shaded.

    One panel per feature in :data:`FEATURES` order, time in seconds along the
    bottom two, the feature's fault line as a dotted line and the test that says
    which side of it is wrong in the axis label, and the hold frames shaded — a
    feature outside a hold is not a fault, and a panel without the shading invites
    exactly that reading. The longest hold is shaded darker: it is the hold the clip
    is ranked by.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(table["t_ms"].to_numpy(), dtype=np.float64) / 1000.0
    phases = np.asarray(table["phase"].astype(str).to_numpy())
    holds = np.asarray(table["hold_id"].to_numpy(), dtype=np.int64)
    hold_mask = (phases == HOLD_PHASE) & (holds >= 0)
    # The longest hold is shaded darker: it is the hold the clip is ranked by, and
    # the stretch of a plot a person compares against another clip's.
    runs = _hold_runs(hold_mask)
    longest_run = max(runs, key=lambda run: run[1] - run[0], default=None)

    columns = 2
    rows = math.ceil(len(FEATURES) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(13.0, 2.4 * rows), sharex=True, squeeze=False
    )
    flat = [axis for row in axes for axis in row]
    for feature, axis in zip(FEATURES, flat, strict=True):
        for first, last in runs:
            axis.axvspan(
                times[first],
                times[last],
                color="tab:orange",
                alpha=0.25 if (first, last) == longest_run else 0.10,
                linewidth=0,
            )
        values = np.asarray(table[feature.name].to_numpy(), dtype=np.float64)
        axis.plot(times, values, color="tab:blue", linewidth=0.8)
        lines, test = fault_line(feature)
        for line in lines:
            axis.axhline(line, color="0.35", linestyle=":", linewidth=0.8)
        axis.set_ylabel(f"{feature.name} ({feature.unit})\n{test}", fontsize=8)
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=7)
    for axis in flat[len(FEATURES) :]:
        axis.set_axis_off()
    axes[-1][0].set_xlabel("t (s)", fontsize=9)
    axes[-1][1].set_xlabel("t (s)", fontsize=9)
    heading = title or f"{len(table)} frames, {int(hold_mask.sum())} hold frame(s)"
    fig.suptitle(f"handstand features per frame — {heading}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        # The format has to be named: the temporary file is called ``.tmp``, and
        # matplotlib reads the format off the name.
        fig.savefig(tmp, dpi=110, format=path.suffix.lstrip(".").lower() or "png")
        plt.close(fig)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _hold_runs(hold_mask: np.ndarray) -> list[tuple[int, int]]:
    """The stretches of hold frames as inclusive ``(first, last)`` index pairs.

    The plot shades them, and a clip with three holds in it has three bands — the
    longest of them being the one the clip is ranked by.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(hold_mask.size + 1):
        holding = index < hold_mask.size and bool(hold_mask[index])
        if holding and start is None:
            start = index
        elif not holding and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The argument parser of ``python -m handstand.features``."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.features",
        description=(
            "Measure the stacking offsets, joint angles, line and leg shape of every frame."
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--clips",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help="clip ids to measure",
    )
    selection.add_argument(
        "--all",
        dest="all_clips",
        action="store_true",
        help="measure every clip with processed keypoints and phases",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help=f"which processed keypoints to read (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory holding processed/, phases/ and features/ (default: $HANDSTAND_DATA)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="measure at most N clips",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-measure clips whose parquet already exists",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help=(
            "after the batch, draw these clips' features over time into "
            "<data_dir>/reports (the clip must already have features)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batch, write the hold table, plot, and print the summary."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    root = pathlib.Path(args.data) if args.data is not None else data_dir()
    in_root = input_dir(root, args.source)
    labels_root = phases_dir(root, args.source)
    out_root = output_dir(root, args.source)
    try:
        clip_ids = available_clips(in_root, None if args.all_clips else args.clips, args.source)
    except FileNotFoundError as error:
        print(f"features: {error}")
        return 2
    if not labels_root.is_dir():
        print(f"features: {missing_phases_message(labels_root, args.source)}")
        return 2
    if args.limit is not None:
        clip_ids = clip_ids[: args.limit]
    if not clip_ids:
        print("no clips to measure")
        return 0

    print(f"source={args.source} clips={len(clip_ids)} in={in_root} out={out_root}")
    reports: list[ClipStats] = []
    measured: list[ClipFeatures] = []
    failures = 0
    for clip_id in clip_ids:
        try:
            report = run_clip(
                clip_id,
                in_root=in_root,
                labels_root=labels_root,
                out_root=out_root,
                source=args.source,
                overwrite=args.overwrite,
            )
        except Exception as error:  # one bad clip must not kill the whole batch
            failures += 1
            print(f"fail  {clip_id}: {error}")
            continue
        print(report.line(), flush=True)
        if not report.skipped:
            reports.append(report.stats)
            measured.append(report.features)

    if measured:
        path = write_hold_summary(
            out_root / HOLD_SUMMARY_NAME, hold_summary_table(measured)
        )
        print(f"hold summary {path} ({len(clip_ids) - len(measured)} clip(s) already measured)")
    print(summary(reports))

    for clip_id in args.plot or ():
        parquet = output_dir(root, args.source) / f"{clip_id}.parquet"
        try:
            table = read_clip_features(parquet)
            path = plot_clip(
                table, plot_path(root, clip_id, args.source), title=f"{clip_id} ({args.source})"
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"plot {clip_id}: {error}", file=sys.stderr)
            failures += 1
            continue
        print(f"plot  {clip_id} {len(table)} frames -> {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
