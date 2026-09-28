"""Phase segmentation: pre, kick-up, hold, exit, post — labelled on every frame.

A clip is not one thing. The athlete walks up, plants their hands, kicks up,
holds, comes down and walks off, and only a few seconds of that is a handstand.
Every stage after this one needs to know which part of the clip it is looking
at: scoring reads only the ``hold`` frames, hand steps (#71) and faults (#32) are
events *inside* a hold, and a feature (#22) measured over a whole clip would
average a fall in with a handstand. So this module labels **every frame** with
the phase it belongs to and numbers each hold.

It is rule-based and readable, on purpose. A phase is a thing an athlete does
that anyone can point at on the video — a person about to kick up, a person
holding, a person coming down — and a threshold per phase is a sentence somebody
can argue with. There is no model here, and no training set for one: the labels
come from geometry (:mod:`handstand.bodyframe`) and time.

Input and output
----------------

Input is the post-processed trajectory of :mod:`handstand.postprocess` (#20)::

    <data_dir>/processed/<source>/<clip_id>.parquet   # x/y smoothed, valid, filled
    <data_dir>/processed/<source>/<clip_id>.json      # body_length_px

Output is one parquet per clip and one segment table for the whole source::

    <data_dir>/phases/<source>/<clip_id>.parquet       # one row per frame
    <data_dir>/phases/<source>/segments.csv            # one row per phase run

The per-frame parquet is one row per frame, in frame order: ``frame_idx``,
``t_ms``, ``phase``, ``hold_id`` and every signal below, so a later stage can
re-derive its own decision from the measurements rather than trusting a label.
``hold_id`` numbers the holds of a clip from 0 and is ``-1`` outside one.
``segments.csv`` is the same answer as intervals — ``clip_id``, ``phase``,
``hold_id``, ``start_frame``, ``end_frame``, ``start_ms``, ``end_ms``,
``duration_s`` — which is what a person reads to check where the boundaries
landed. Both are derived data and are never committed.

Nothing here knows which pose model produced the keypoints: it reads a source's
processed parquet and nothing else, so ``--source mediapipe`` and
``--source vision`` are the same code.

CLI::

    cd pipeline
    uv run python -m handstand.phases --all
    uv run python -m handstand.phases --clips 057c9e6c96af 6508f9b355bd
    uv run python -m handstand.phases --clips 057c9e6c96af --render 057c9e6c96af

The signals
-----------

All of them are read in the body frame (:mod:`handstand.bodyframe`: origin at
the wrist midpoint, ``u`` right, ``v`` **up**, both in body lengths ``L``), so
one threshold means the same thing on a 350 px athlete and a 600 px one. These
clips are variable frame rate, so every duration is measured against ``t_ms`` and
never against a frame count.

``inverted``
    The ankle midpoint is more than :data:`INVERTED_MIN` above the wrist
    midpoint — ``v_ankle_mid > 0.6``. It is the "upside down" test, with a
    margin: a body standing on its head is not perfectly straight, and a person
    lying on the floor is not inverted even though their ankles can be level
    with their hands.
``body_angle``
    The angle in degrees between the wrist→ankle vector and straight up, signed
    (negative when the feet are to the left of the hands). A handstand is
    straight, a pike or a banana is not, and :data:`HOLD_MAX_ANGLE` is where the
    handstand ends and the shape starts.
``hands_low``
    The hip midpoint is at least :data:`HANDS_LOW_MIN_V` above the wrist
    midpoint, i.e. the athlete's mass is above their hands. That is what "the
    hands are the support" means: standing or walking, the hips are at or below
    the hands and the feet are on the floor.
``hands_down``
    ``hands_low``, and the hands are not moving: every wrist visible in a frame
    and in the frame :data:`HAND_STILL_WINDOW_S` earlier has moved slower than
    :data:`HAND_STILL_L_PER_S` over that window. Planted hands are what separates
    standing up to the wall from going up it, and the measurement is over a
    window rather than frame to frame, so one jittery frame is not a movement.
    One wrist is enough when the source only reports one — a rule that needs both
    throws away every hold of a side-on clip, where the near hand is the one the
    model sees.
``hand_step``
    The same measurement read the other way: a wrist that moved further than
    :data:`HAND_STEP_L` over the window. It ends the current hold, and at
    0.1 L per 0.2 s it is necessarily a failure of ``hands_down`` too, which is
    0.3 L/s. It is kept as its own term and its own column because it is a
    different question — a threshold tuned to stop a hold drifting is not a
    threshold tuned to catch a hand step, and #71 tunes that one against labels.
``legs_rising`` / ``legs_falling``
    The sign of the ankle midpoint's vertical velocity — how fast the feet are
    going away from the floor or coming back to it, measured over
    :data:`LEG_VELOCITY_WINDOW_S` and dead-banded at
    :data:`LEG_VELOCITY_MIN_L_PER_S` so keypoint noise is not read as motion.

The phases
----------

A state machine over time, one phase per frame, with ``unknown`` for the frames
it cannot see into. The rules, in the order they are asked:

``pre``
    Standing, walking up: anything before the hands are down.
``kickup``
    Hands down, the legs going up, not yet holding — the attempt.
``hold``
    ``inverted`` and ``|body_angle| <= HOLD_MAX_ANGLE`` and ``hands_down`` and no
    hand step, sustained for at least :data:`MIN_HOLD_S`. Every such run is one
    hold and gets the next ``hold_id``.
``exit``
    After a hold: the legs falling, no longer inverted, or the hands on the floor
    without holding — until the feet land or the hands leave the floor.
``post``
    After the last exit. The body is no longer above its hands, so the clip is not
    about a handstand any more.
``unknown``
    A frame the model could not see into: trainer contact, or a wrist, an ankle
    or a hip with no position. An unknown stretch of at most
    :data:`HOLD_BREAK_MAX_S` *inside* a hold does not break it — the athlete did
    not stop holding because a trainer walked in front of the camera — and the
    frames of the gap are labelled with the hold they are inside. A longer one
    does break it: a hold cannot be claimed across a stretch of frames nobody
    was watching. The same window forgives a *known* frame the hold condition
    failed on, because a held handstand sways and its ankle midpoint crosses
    :data:`INVERTED_MIN` now and then for a single frame; what it does not
    forgive is a hand step, which is the one thing that really does end a hold.

Transitions, in one list: ``pre`` becomes ``kickup`` when the hands go down; a
run that qualifies is a ``hold``; a hold that stops holding becomes ``exit``;
``kickup`` and ``exit`` become ``post`` when the body is no longer above the
hands; and ``kickup``, ``exit`` and ``post`` become ``kickup`` again when the
hands are down and the legs are rising, which is a second attempt in a clip that
holds several times. A hold's first frame is where its ``hold_id`` starts, so a
hand step in the middle of a hold gives two holds rather than one with a hole in
it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from handstand import bodyframe, overlay
from handstand.paths import data_dir
from handstand.postprocess import (
    DEFAULT_SOURCE,
    PROCESSED_DIRNAME,
    SOURCES,
    ClipKeypoints,
    read_clip,
)

__all__ = [
    "ANKLE_JOINTS",
    "BODY_LENGTH_KEY",
    "DEFAULT_SOURCE",
    "HAND_STEP_L",
    "HAND_STILL_L_PER_S",
    "HAND_STILL_WINDOW_S",
    "HANDS_LOW_MIN_V",
    "HIP_JOINTS",
    "HOLD_BREAK_MAX_S",
    "HOLD_MAX_ANGLE",
    "INVERTED_MIN",
    "LEG_VELOCITY_MIN_L_PER_S",
    "LEG_VELOCITY_WINDOW_S",
    "MIN_HOLD_S",
    "NO_HOLD",
    "NO_HOLD_CLIPS_LISTED",
    "PHASES",
    "PHASES_DIRNAME",
    "REASON_ANKLES",
    "REASON_HIPS",
    "REASON_TRAINER",
    "REASON_WRISTS",
    "SEGMENT_COLUMNS",
    "SEGMENTS_NAME",
    "SOURCES",
    "WRIST_JOINTS",
    "ClipPhases",
    "ClipReport",
    "ClipStats",
    "FrameSignals",
    "HoldRun",
    "Segment",
    "assign_phases",
    "available_clips",
    "build_arg_parser",
    "classify_clip",
    "frame_signals",
    "hold_condition",
    "hold_runs",
    "input_dir",
    "main",
    "measure",
    "missing_input_message",
    "output_dir",
    "phase_captions",
    "phase_names",
    "phase_table",
    "read_body_length",
    "read_clip_phases",
    "read_processed",
    "render_clip",
    "run_clip",
    "segment_phases",
    "segments_table",
    "summary",
    "velocity_l_per_s",
    "window_motion",
    "write_segments",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The six labels, in the order a clip passes through them. The integers below
#: are what the state machine uses; :func:`phase_names` turns them back into
#: these names for the parquet, the CSV and the overlay caption.
PHASES: tuple[str, ...] = ("pre", "kickup", "hold", "exit", "post", "unknown")
PRE, KICKUP, HOLD, EXIT, POST, UNKNOWN = range(len(PHASES))

#: ``hold_id`` outside a hold. A negative number cannot be a hold's number, and a
#: number keeps the column numeric in both the parquet and the CSV.
NO_HOLD = -1

#: How far above the wrist midpoint the ankle midpoint has to be before the
#: athlete counts as inverted, in body lengths. Handstands are not perfectly
#: straight and a body on the floor is not inverted, so this is a margin and not
#: a sign change: a 0.6 L gap between hands and feet is a body standing on its
#: hands and nothing else in a clip gets there.
INVERTED_MIN = 0.6

#: How far the wrist→ankle vector may lean from straight up inside a hold, in
#: degrees. A held handstand is straight; 35° is well past the lean of a real one
#: and well before the pike of a kick-up that has not arrived.
HOLD_MAX_ANGLE = 35.0

#: How long a run of hold frames has to last to count as a hold, in seconds. A
#: third of a second is long enough that a hop on the hands is not a hold and
#: short enough that the small balance corrections at the end of a real one are
#: not thrown away.
MIN_HOLD_S = 0.3

#: How fast a wrist may move and still count as planted, in body lengths per
#: second. The balance corrections of a held handstand are a few hundredths of a
#: body length a second; a hand step is several times this.
HAND_STILL_L_PER_S = 0.3

#: The window both hand measurements are taken over, in seconds. A frame-to-frame
#: difference is one noisy sample of a smooth track; a fifth of a second is long
#: enough to be a movement and short enough to catch the start of a step.
HAND_STILL_WINDOW_S = 0.2

#: How far a wrist may move over :data:`HAND_STILL_WINDOW_S` and still be the
#: same hold, in body lengths. This is a hand step — the hands are put down
#: somewhere else — and a tenth of the body is more than a balance correction
#: covers in a fifth of a second. #71 tunes it against labelled hand steps.
HAND_STEP_L = 0.1

#: How far above the wrist midpoint the hip midpoint has to be for the hands to
#: count as the support, in body lengths. Standing or walking, the hips are at
#: or below the hands; once the hands are planted with the body above them, the
#: athlete is inverted in the only sense a phase cares about.
HANDS_LOW_MIN_V = 0.1

#: The deadband on the ankle midpoint's vertical velocity, in body lengths per
#: second. Below it the legs are neither rising nor falling: the smoothed track of
#: a body that is holding has a velocity of a few hundredths, and calling that
#: "rising" would turn every hold into a kick-up.
LEG_VELOCITY_MIN_L_PER_S = 0.1

#: The window the vertical velocity is measured over, in seconds. Centred, so it
#: is half this far either side of the frame: a forward difference of neighbours
#: would put the velocity at the midpoint of a step, where the body is not moving.
LEG_VELOCITY_WINDOW_S = 0.1

#: The longest stretch of frames that may fail to be a hold without ending one, in
#: seconds. It covers both kinds of stretch a hold survives: frames nobody could
#: see into, where a trainer walked in front of the camera, and frames the
#: geometry blipped on — an ankle midpoint a thousandth of a body length under
#: :data:`INVERTED_MIN` for one frame is not the athlete letting go. A hand step
#: is never bridged: the hands were put down somewhere else, and that is the end
#: of this hold whatever it took.
HOLD_BREAK_MAX_S = 0.3

#: Output root under ``<data_dir>/``, one directory per source exactly as
#: ``processed/`` has one, so one source's phases never overwrite another's.
PHASES_DIRNAME = "phases"

#: The segment table of one source: every clip's segments in one file.
SEGMENTS_NAME = "segments.csv"

#: Its columns, in write order. ``hold_id`` is :data:`NO_HOLD` outside a hold and
#: ``duration_s`` is ``(end_ms - start_ms) / 1000``, so the four time columns of a
#: row always agree with each other.
SEGMENT_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "phase",
    "hold_id",
    "start_frame",
    "end_frame",
    "start_ms",
    "end_ms",
    "duration_s",
)

#: The column of a processed parquet that says a sample has a position, and the
#: key of the sidecar that holds the clip's scale: the two things this module
#: reads that the raw keypoints do not have.
VALID_COLUMN = "valid"
BODY_LENGTH_KEY = "body_length_px"

#: Why a frame is ``unknown``, one name per reason, asked in this order: a trainer
#: touching the athlete first, then the three joints a phase is read off. The
#: first that applies is the frame's reason.
REASON_TRAINER = "trainer_contact"
REASON_WRISTS = "no_visible_wrist"
REASON_ANKLES = "no_visible_ankle"
REASON_HIPS = "no_visible_hip"

#: How many of the clips without a hold the run summary names. The rest of the
#: list is in :data:`SEGMENTS_NAME`; ten is what a person reads, and a summary
#: that printed eighty clip ids would be read past.
NO_HOLD_CLIPS_LISTED = 10

#: The joints a phase is read off, in the order the body frame needs them.
WRIST_JOINTS: tuple[str, ...] = ("left_wrist", "right_wrist")
ANKLE_JOINTS: tuple[str, ...] = ("left_ankle", "right_ankle")
HIP_JOINTS: tuple[str, ...] = ("left_hip", "right_hip")

#: Slack, in seconds, when a measurement window is compared against a timestamp.
#: A nanosecond is a millionth of a frame at 30 fps, so it decides nothing about
#: the data; it only stops a window that lands exactly on a frame from being
#: decided by the last bit of a float.
_TIME_EPS = 1e-9


def phase_names(codes: np.ndarray) -> np.ndarray:
    """Phase codes as the names the parquet and the CSV carry."""
    return np.asarray(PHASES, dtype=object)[np.asarray(codes, dtype=np.int64)]


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #


def _joint_columns(joints: Sequence[str], wanted: Sequence[str]) -> list[int]:
    """The columns of ``wanted`` this clip has, in that order.

    A source that does not report one of them (Apple Vision has no
    ``foot_index``) simply has no column for it, and every measurement over these
    names skips it rather than padding it with zeros.
    """
    return [list(joints).index(name) for name in wanted if name in joints]


def _midpoint(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    columns: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A joint group's per-frame midpoint over its visible members, and the count.

    One wrist is enough to place the origin and one ankle is enough to say which
    way up the body is, so a side-on clip with one wrist in shot is still
    classifiable; a frame where no member of the group is visible is left NaN
    rather than averaged over zeros.
    """
    if not columns:
        zeros = np.zeros(x.shape[0], dtype=np.float64)
        return zeros.copy(), zeros.copy(), zeros.astype(np.int64)
    known = valid[:, columns]
    count = known.sum(axis=1).astype(np.int64)
    safe = np.maximum(count, 1)
    mid_x = np.where(known, x[:, columns], 0.0).sum(axis=1) / safe
    mid_y = np.where(known, y[:, columns], 0.0).sum(axis=1) / safe
    visible = count > 0
    return (
        np.where(visible, mid_x, np.nan),
        np.where(visible, mid_y, np.nan),
        count,
    )


def _largest(values: np.ndarray) -> np.ndarray:
    """The largest finite value of each row; NaN where a row is all NaN.

    ``np.nanmax`` warns on an all-NaN slice, and a wrist that cannot be seen at
    both ends of a window is exactly that: the rows without a finite value are
    taken out before the maximum is taken.
    """
    if values.size == 0 or values.shape[1] == 0:
        return np.full(values.shape[0], np.nan)
    out = np.full(values.shape[0], np.nan)
    rows = np.isfinite(values).any(axis=1)
    masked = np.where(np.isfinite(values), values, -np.inf)
    out[rows] = masked[rows].max(axis=1)
    return out


def _body_frame_uv(
    x: np.ndarray,
    y: np.ndarray,
    origin_x: np.ndarray,
    origin_y: np.ndarray,
    body_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``handstand.bodyframe.to_body_frame`` over a whole clip, one frame at a time.

    The origin is a per-frame quantity — the wrist midpoint of *that* frame — so
    this is the call :func:`handstand.bodyframe.to_body_frame` is written for. A
    clip is a few hundred frames and each call is four numbers, so the loop is
    not the cost of anything here.
    """
    points = np.array(
        [
            bodyframe.to_body_frame(x[i], y[i], origin_x[i], origin_y[i], body_length)
            for i in range(x.size)
        ],
        dtype=np.float64,
    )
    return points[:, 0], points[:, 1]


def window_motion(
    t_seconds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    known: np.ndarray,
    body_length: float,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """How far, and how fast, a point moved over the ``window_s`` before each frame.

    The window is the frames from the last one at or before ``t - window_s`` up to
    the frame itself, which is what "over the last fifth of a second" means on a
    variable frame rate clip. The number is NaN whenever the question cannot be
    asked of the clip: when the window reaches back before its first frame, when
    no time has passed inside it, or when either end is a frame that cannot see
    the point. A half-measurement is never dressed up as a measurement.
    """
    if window_s <= 0.0:
        raise ValueError(f"window_s must be > 0, got {window_s!r}")
    frames = t_seconds.size
    step_l = np.full(frames, np.nan)
    speed_l_per_s = np.full(frames, np.nan)
    for index in range(frames):
        earlier = int(np.searchsorted(t_seconds, t_seconds[index] - window_s, side="right")) - 1
        if earlier < 0 or not known[index] or not known[earlier]:
            continue
        elapsed = float(t_seconds[index] - t_seconds[earlier])
        if elapsed <= 0.0:
            continue
        step_l[index] = (
            math.hypot(float(x[index] - x[earlier]), float(y[index] - y[earlier])) / body_length
        )
        speed_l_per_s[index] = step_l[index] / elapsed
    return step_l, speed_l_per_s


def velocity_l_per_s(
    t_seconds: np.ndarray,
    values: np.ndarray,
    known: np.ndarray,
    span_s: float,
) -> np.ndarray:
    """The rate of change of ``values``, over ``span_s`` centred on each frame.

    Centred, because a forward difference of a smoothed track reads the motion as
    happening *after* the frame it is attached to, and a phase boundary is where
    the motion is. The window is clipped to the frames the clip has — the first
    and last frames are measured against the one side they have — and is NaN
    whenever the frame itself or the one it is compared with is a frame the
    module cannot see into, so the edges of a gap are not velocities.
    """
    if span_s <= 0.0:
        raise ValueError(f"span_s must be > 0, got {span_s!r}")
    frames = t_seconds.size
    half = span_s / 2.0
    out = np.full(frames, np.nan)
    for index in range(frames):
        # The nearest frame on each side of the window, clipped to the clip: the
        # first and last frames are measured against the one side they have. The
        # slack keeps a window that lands exactly on a frame from being decided
        # by the last bit of a timestamp.
        before = (
            int(np.searchsorted(t_seconds, t_seconds[index] - half + _TIME_EPS, side="right")) - 1
        )
        after = int(np.searchsorted(t_seconds, t_seconds[index] + half - _TIME_EPS, side="left"))
        before = max(min(before, index), 0)
        after = min(max(after, index), frames - 1)
        elapsed = float(t_seconds[after] - t_seconds[before])
        if elapsed <= 0.0 or not known[index] or not known[before] or not known[after]:
            continue
        out[index] = float(values[after] - values[before]) / elapsed
    return out


@dataclasses.dataclass(frozen=True)
class FrameSignals:
    """Every per-frame measurement the state machine reads, one array per signal.

    All of them are ``(frames,)``, in frame order, and all in the body frame's
    units: positions and lengths in body lengths, angles in degrees, speeds in
    body lengths per second. Every boolean is ``False`` on a frame the module
    cannot see into — :attr:`known` is the column to filter on and
    :attr:`unknown_reason` says which joint, or which person, was in the way.
    """

    known: np.ndarray
    unknown_reason: np.ndarray
    v_ankle_mid: np.ndarray
    u_ankle_mid: np.ndarray
    v_hip_mid: np.ndarray
    body_angle_deg: np.ndarray
    inverted: np.ndarray
    hands_low: np.ndarray
    wrist_speed_l_per_s: np.ndarray
    wrist_step_l: np.ndarray
    hands_down: np.ndarray
    hand_step: np.ndarray
    legs_velocity_l_per_s: np.ndarray
    legs_rising: np.ndarray
    legs_falling: np.ndarray

    @property
    def frames(self) -> int:
        """How many frames the clip has."""
        return int(self.known.size)

    @property
    def upside_down(self) -> np.ndarray:
        """The geometry of a hold on its own: inverted and straight.

        This is what a later stage asks when it only wants to know whether the
        athlete was upside down at all; :func:`hold_condition` adds the hands.
        """
        return self.known & self.inverted & (np.abs(self.body_angle_deg) <= HOLD_MAX_ANGLE)


def _unusable_signals(frames: int, reason: str) -> FrameSignals:
    """The signals of a clip with no scale: nothing measured, everything unknown."""
    unknown = np.full(frames, reason, dtype=object)
    return FrameSignals(
        known=np.zeros(frames, dtype=bool),
        unknown_reason=unknown,
        v_ankle_mid=np.full(frames, np.nan),
        u_ankle_mid=np.full(frames, np.nan),
        v_hip_mid=np.full(frames, np.nan),
        body_angle_deg=np.full(frames, np.nan),
        inverted=np.zeros(frames, dtype=bool),
        hands_low=np.zeros(frames, dtype=bool),
        wrist_speed_l_per_s=np.full(frames, np.nan),
        wrist_step_l=np.full(frames, np.nan),
        hands_down=np.zeros(frames, dtype=bool),
        hand_step=np.zeros(frames, dtype=bool),
        legs_velocity_l_per_s=np.full(frames, np.nan),
        legs_rising=np.zeros(frames, dtype=bool),
        legs_falling=np.zeros(frames, dtype=bool),
    )


def frame_signals(
    t_ms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    trainer_contact: np.ndarray,
    joints: Sequence[str],
    body_length: float,
) -> FrameSignals:
    """Measure every signal of :class:`FrameSignals` for one clip.

    Parameters
    ----------
    t_ms:
        The clip's frame timestamps in milliseconds — the only clock these clips
        have, and variable from frame to frame.
    x, y, valid:
        ``(frames, joints)`` arrays of the *processed* track in display pixels,
        and the column saying a sample has a position.
    trainer_contact:
        Per frame, whether the athlete selection flagged the trainer overlapping
        or touching the athlete. Such a frame is two bodies' keypoints stitched
        together, and is ``unknown`` whatever the numbers say.
    joints:
        The clip's joint names, in column order, so the wrists, ankles and hips
        can be found.
    body_length:
        The clip's scale ``L`` in pixels, from the sidecar. Positive and finite:
        a clip without one has no unit to measure in and is refused here rather
        than divided by later.
    """
    if not math.isfinite(body_length) or body_length <= 0.0:
        raise ValueError(f"body_length must be a positive, finite number, got {body_length!r}")
    times = np.asarray(t_ms, dtype=np.float64) / 1000.0
    if x.ndim != 2 or y.shape != x.shape or valid.shape != x.shape:
        raise ValueError("x, y and valid must have the same (frames, joints) shape")
    if x.shape[0] != times.size:
        raise ValueError(f"x has {x.shape[0]} frames, t_ms has {times.size}")
    if trainer_contact.shape != (times.size,):
        raise ValueError(
            f"trainer_contact must have one entry per frame ({times.size}), "
            f"got {trainer_contact.shape}"
        )
    if x.shape[1] != len(joints):
        raise ValueError(f"x has {x.shape[1]} joint columns but {len(joints)} joint names")

    wrist_x, wrist_y, wrist_count = _midpoint(x, y, valid, _joint_columns(joints, WRIST_JOINTS))
    ankle_x, ankle_y, ankle_count = _midpoint(x, y, valid, _joint_columns(joints, ANKLE_JOINTS))
    hip_x, hip_y, hip_count = _midpoint(x, y, valid, _joint_columns(joints, HIP_JOINTS))

    known = (wrist_count > 0) & (ankle_count > 0) & (hip_count > 0) & ~trainer_contact
    reason = np.full(times.size, "", dtype=object)
    reason[trainer_contact] = REASON_TRAINER
    reason[~trainer_contact & (wrist_count == 0)] = REASON_WRISTS
    reason[~trainer_contact & (wrist_count > 0) & (ankle_count == 0)] = REASON_ANKLES
    reason[~trainer_contact & (wrist_count > 0) & (ankle_count > 0) & (hip_count == 0)] = (
        REASON_HIPS
    )

    u_ankle, v_ankle = _body_frame_uv(ankle_x, ankle_y, wrist_x, wrist_y, body_length)
    _, v_hip = _body_frame_uv(hip_x, hip_y, wrist_x, wrist_y, body_length)

    # atan2 of the horizontal against the vertical component: the angle of the
    # wrist->ankle vector from straight up, signed by which way it leans.
    with np.errstate(invalid="ignore"):
        angle = np.degrees(np.arctan2(u_ankle, v_ankle))
    inverted = known & (v_ankle > INVERTED_MIN)
    hands_low = known & (v_hip > HANDS_LOW_MIN_V)

    steps, speeds = [], []
    for column in _joint_columns(joints, WRIST_JOINTS):
        step_l, speed = window_motion(
            times, x[:, column], y[:, column], valid[:, column], body_length, HAND_STILL_WINDOW_S
        )
        steps.append(step_l)
        speeds.append(speed)
    wrist_step_l = _largest(np.stack(steps, axis=1))
    wrist_speed = _largest(np.stack(speeds, axis=1))

    legs_velocity = velocity_l_per_s(times, v_ankle, known, LEG_VELOCITY_WINDOW_S)
    # A wrist whose speed could not be measured is not evidence that the hands
    # moved. That happens on the first fifth of a second after a stretch of
    # frames nobody could see, where there is no earlier frame to compare with;
    # reading it as "the hands are moving" would end a hold every time a trainer
    # walked in front of the camera, which is the one thing a gap must not do.
    still = np.where(np.isfinite(wrist_speed), wrist_speed < HAND_STILL_L_PER_S, True)
    return FrameSignals(
        known=known,
        unknown_reason=reason,
        v_ankle_mid=v_ankle,
        u_ankle_mid=u_ankle,
        v_hip_mid=v_hip,
        body_angle_deg=angle,
        inverted=inverted,
        hands_low=hands_low,
        wrist_speed_l_per_s=wrist_speed,
        wrist_step_l=wrist_step_l,
        hands_down=hands_low & still,
        hand_step=known & (wrist_step_l > HAND_STEP_L),
        legs_velocity_l_per_s=legs_velocity,
        legs_rising=known & (legs_velocity > LEG_VELOCITY_MIN_L_PER_S),
        legs_falling=known & (legs_velocity < -LEG_VELOCITY_MIN_L_PER_S),
    )


# --------------------------------------------------------------------------- #
# Holds
# --------------------------------------------------------------------------- #


def hold_condition(signals: FrameSignals) -> np.ndarray:
    """The tests a hold frame has to pass, all of them at once.

    Inverted and straight (:attr:`FrameSignals.upside_down`), the hands planted
    (:attr:`FrameSignals.hands_down`) and no hand step
    (:attr:`FrameSignals.hand_step`) — the last of which is necessarily a failure
    of the third, since :data:`HAND_STEP_L` over :data:`HAND_STILL_WINDOW_S` is
    faster than :data:`HAND_STILL_L_PER_S`. It stays its own term because the two
    answer different questions and #71 re-tunes the step.
    """
    return signals.upside_down & signals.hands_down & ~signals.hand_step


@dataclasses.dataclass(frozen=True)
class HoldRun:
    """One hold: the frames it covers, inclusive, and the number it is."""

    start: int
    end: int
    hold_id: int = NO_HOLD

    @property
    def frames(self) -> int:
        """How many frames the run covers, including any bridged unknown ones."""
        return self.end - self.start + 1


def hold_runs(
    t_seconds: np.ndarray,
    holding: np.ndarray,
    known: np.ndarray,
    event: np.ndarray | None = None,
    *,
    min_hold_s: float = MIN_HOLD_S,
    max_break_s: float = HOLD_BREAK_MAX_S,
) -> tuple[HoldRun, ...]:
    """The stretches of frames a hold occupies, the long ones only, numbered from 0.

    A run starts at the first frame that passes :func:`hold_condition` and ends at
    the last one before a break that ends it. Three kinds of frame in between do
    not end a run, and are the reason the answer is not simply "every frame the
    condition passed":

    * an ``event`` frame — a hand step — ends the run at once, wherever it is;
    * an unknown frame, which is a trainer in the way rather than a change in
      what the athlete is doing;
    * a known frame the condition failed on, for as long as the whole break
      lasts less than ``max_break_s``.

    The last one is the geometry wobbling: a held handstand sways, and its ankle
    midpoint crosses :data:`INVERTED_MIN` now and then for a frame. A stretch of
    failures longer than ``max_break_s`` is the athlete coming down, and it ends
    the hold. The frames a run bridges are inside it, so they carry the hold's
    phase and number: the alternative — a hold with a hole in it — is a thing
    every later stage would have to special-case.

    Runs at least ``min_hold_s`` long come back numbered in time order; the
    shorter ones are not holds and are left to the state machine.
    """
    if min_hold_s < 0.0:
        raise ValueError(f"min_hold_s must be >= 0, got {min_hold_s!r}")
    if max_break_s < 0.0:
        raise ValueError(f"max_break_s must be >= 0, got {max_break_s!r}")
    if holding.shape != known.shape or holding.shape != t_seconds.shape:
        raise ValueError("holding, known and t_seconds must have the same shape")
    if event is not None and event.shape != holding.shape:
        raise ValueError("event must have the same shape as holding")

    runs: list[tuple[int, int]] = []
    start: int | None = None
    break_start: int | None = None
    for index in range(t_seconds.size):
        if holding[index]:
            if start is None:
                start = index
            break_start = None
            continue
        if start is None:
            continue
        if event is not None and event[index]:
            runs.append((start, index - 1))
            start = break_start = None
            continue
        if break_start is None:
            break_start = index
        elif float(t_seconds[index] - t_seconds[break_start]) > max_break_s:
            runs.append((start, break_start - 1))
            start = break_start = None
    if start is not None:
        runs.append((start, (break_start - 1) if break_start is not None else t_seconds.size - 1))

    return tuple(
        HoldRun(start, end, hold_id)
        for hold_id, (start, end) in enumerate(
            run for run in runs if float(t_seconds[run[1]] - t_seconds[run[0]]) >= min_hold_s
        )
    )


# --------------------------------------------------------------------------- #
# The state machine
# --------------------------------------------------------------------------- #


def assign_phases(
    signals: FrameSignals,
    runs: Sequence[HoldRun],
) -> tuple[np.ndarray, np.ndarray]:
    """Label every frame with its phase and, inside a hold, its hold number.

    A hold's own frames are labelled first, so the walk below never has to know
    whether a run qualified. The rest is four rules, in this order, and they are
    the whole of it:

    1. a frame the module cannot see into is ``unknown`` and does not move the
       state on — a gap is not an event;
    2. a hold frame puts the state in ``hold``, and the first known frame that is
       not holding takes it to ``exit``: a hold that stops holding is an exit,
       whatever the athlete does next;
    3. hands down with the legs rising is ``kickup``, whenever it happens: a new
       attempt, whether it is the first, the third, or the one after a fall;
    4. the body is no longer above the hands ends the attempt: ``pre`` stays
       ``pre`` and every other state becomes ``post``, because once the hands are
       not the support any more the clip is not about a handstand any more.

    Anything else stands, so ``kickup`` and ``exit`` last until one of those four
    says otherwise. ``kickup`` and ``exit`` are the same thing from two sides —
    the hands are on the floor and the athlete is not holding — and what
    separates them is the legs (rising means a kick-up) and what came before (a
    hold means an exit). Note that rule 4 asks about :attr:`FrameSignals.hands_low`
    and not about ``hands_down``: a hand step is the hands *moving* while they
    stay on the floor, which ends the hold (rule 2) and is an exit, not the end of
    the attempt.
    """
    frames = signals.frames
    phase = np.full(frames, UNKNOWN, dtype=np.int64)
    hold_id = np.full(frames, NO_HOLD, dtype=np.int64)
    holding = np.zeros(frames, dtype=bool)
    for run in runs:
        holding[run.start : run.end + 1] = True
        hold_id[run.start : run.end + 1] = run.hold_id
        phase[run.start : run.end + 1] = HOLD

    state = PRE
    for index in range(frames):
        if not signals.known[index]:
            continue
        if holding[index]:
            state = HOLD
            continue
        if state == HOLD:
            state = EXIT
        elif signals.hands_down[index] and signals.legs_rising[index]:
            state = KICKUP
        elif not signals.hands_low[index]:
            state = PRE if state == PRE else POST
        phase[index] = state
    return phase, hold_id


# --------------------------------------------------------------------------- #
# One clip
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipPhases:
    """One clip's per-frame answer, with the numbers it was made from."""

    clip_id: str
    source: str
    frame_idx: np.ndarray
    t_ms: np.ndarray
    phase: np.ndarray
    hold_id: np.ndarray
    signals: FrameSignals
    runs: tuple[HoldRun, ...] = ()
    usable: bool = True
    unusable_reason: str = ""

    @property
    def frames(self) -> int:
        """How many frames the clip has."""
        return int(self.frame_idx.size)

    @property
    def hold_count(self) -> int:
        """How many holds the clip has."""
        return len(self.runs)

    @property
    def phase_codes(self) -> np.ndarray:
        """The phase of each frame as its name."""
        return phase_names(self.phase)

    def phase_counts(self) -> dict[str, int]:
        """How many frames each phase got, with every phase present."""
        codes = self.phase_codes
        return {name: int((codes == name).sum()) for name in PHASES}

    def hold_durations_s(self) -> tuple[float, ...]:
        """The duration of every hold in seconds, in time order."""
        times = self.t_ms.astype(np.float64) / 1000.0
        return tuple(float(times[run.end] - times[run.start]) for run in self.runs)

    def longest_hold_s(self) -> float:
        """The longest hold of the clip, or 0.0 when it has none."""
        return max(self.hold_durations_s(), default=0.0)


def classify_clip(
    clip_id: str,
    source: str,
    t_ms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    trainer_contact: np.ndarray,
    body_length: float,
    joints: Sequence[str],
    *,
    reason: str = "",
) -> ClipPhases:
    """Label one clip, from its processed arrays to its phases.

    A clip with no body length comes back with every frame ``unknown`` and
    :attr:`ClipPhases.unusable` set: without ``L`` there is no unit to measure
    inversion, angle or speed in, and a phase read in pixels is not a phase.
    """
    times = np.asarray(t_ms, dtype=np.int64)
    if not reason and (not math.isfinite(body_length) or body_length <= 0.0):
        reason = f"no body length ({body_length!r} px)"
    frame_idx = np.arange(times.size, dtype=np.int64)
    if reason:
        return ClipPhases(
            clip_id=clip_id,
            source=source,
            frame_idx=frame_idx,
            t_ms=times,
            phase=np.full(times.size, UNKNOWN, dtype=np.int64),
            hold_id=np.full(times.size, NO_HOLD, dtype=np.int64),
            signals=_unusable_signals(times.size, reason),
            usable=False,
            unusable_reason=reason,
        )
    signals = frame_signals(times, x, y, valid, trainer_contact, joints, body_length)
    runs = hold_runs(
        times.astype(np.float64) / 1000.0,
        hold_condition(signals),
        signals.known,
        signals.hand_step,
    )
    phase, hold_id = assign_phases(signals, runs)
    return ClipPhases(
        clip_id=clip_id,
        source=source,
        frame_idx=frame_idx,
        t_ms=times,
        phase=phase,
        hold_id=hold_id,
        signals=signals,
        runs=runs,
    )


def read_processed(parquet_path: str | pathlib.Path) -> tuple[ClipKeypoints, np.ndarray]:
    """Read a processed parquet into its clip and its ``(frames, joints)`` validity.

    Raises :class:`ValueError` when the file has no ``valid`` column: that is the
    column :mod:`handstand.postprocess` adds, so a file without it is a raw
    keypoint parquet and this module would be measuring a track nobody cleaned.
    """
    path = pathlib.Path(parquet_path)
    clip = read_clip(path)
    if VALID_COLUMN not in clip.table.columns:
        raise ValueError(
            f"{path}: no {VALID_COLUMN!r} column, so this is not a processed parquet; "
            "generate it with: cd pipeline && uv run python -m handstand.postprocess --all"
        )
    valid = clip.table[VALID_COLUMN].to_numpy(dtype=bool).reshape(clip.frames, clip.joint_count)
    return clip, valid


def read_body_length(sidecar_path: str | pathlib.Path) -> tuple[float, str]:
    """The clip's body length ``L`` in pixels from its sidecar, or why it has none.

    Returns ``(nan, reason)`` for a missing or unreadable sidecar and for a clip
    the post-processing called unusable, so the caller can label the clip
    ``unknown`` and say why instead of dividing by nothing.
    """
    path = pathlib.Path(sidecar_path)
    try:
        sidecar = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return math.nan, f"sidecar {path.name} is unreadable ({error})"
    if not sidecar.get("usable", True):
        return math.nan, f"no body length: {sidecar.get('unusable_reason')}"
    length = sidecar.get(BODY_LENGTH_KEY)
    if length is None:
        return math.nan, f"sidecar {path.name} has no {BODY_LENGTH_KEY}"
    value = float(length)
    if not math.isfinite(value) or value <= 0.0:
        return math.nan, f"sidecar {path.name} has a {BODY_LENGTH_KEY} of {length!r}"
    return value, ""


def _rounded(values: np.ndarray, digits: int = 5) -> np.ndarray:
    """``values`` rounded to five decimals — NaN stays NaN, so an unmeasured
    signal is still visibly unmeasured in the table."""
    return np.round(np.asarray(values, dtype=np.float64), digits)


def phase_table(result: ClipPhases) -> pd.DataFrame:
    """One clip's per-frame answer as a table, in frame order.

    The columns are the frame's identity (``frame_idx``, ``t_ms``), the answer
    (``phase``, ``hold_id``) and then every signal, so a later stage can re-derive
    its own decision from the measurements rather than trusting a label.
    """
    signals = result.signals
    return pd.DataFrame(
        {
            "frame_idx": result.frame_idx,
            "t_ms": result.t_ms,
            "phase": result.phase_codes,
            "hold_id": result.hold_id,
            "known": signals.known,
            "unknown_reason": signals.unknown_reason,
            "v_ankle_mid": _rounded(signals.v_ankle_mid),
            "u_ankle_mid": _rounded(signals.u_ankle_mid),
            "v_hip_mid": _rounded(signals.v_hip_mid),
            "body_angle_deg": _rounded(signals.body_angle_deg, 2),
            "inverted": signals.inverted,
            "hands_low": signals.hands_low,
            "wrist_speed_l_per_s": _rounded(signals.wrist_speed_l_per_s),
            "wrist_step_l": _rounded(signals.wrist_step_l),
            "hands_down": signals.hands_down,
            "hand_step": signals.hand_step,
            "legs_velocity_l_per_s": _rounded(signals.legs_velocity_l_per_s),
            "legs_rising": signals.legs_rising,
            "legs_falling": signals.legs_falling,
        }
    )


def read_clip_phases(parquet_path: str | pathlib.Path) -> pd.DataFrame:
    """Read one phases parquet back, for the overlay caption and for later stages."""
    path = pathlib.Path(parquet_path)
    table = pd.read_parquet(path)
    required = ("frame_idx", "t_ms", "phase", "hold_id")
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(
            f"{path}: phases are missing column(s) {missing}; "
            "generate them with: cd pipeline && uv run python -m handstand.phases --all"
        )
    return table


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Segment:
    """One stretch of frames sharing a phase, as :data:`SEGMENT_COLUMNS` records it."""

    clip_id: str
    phase: str
    hold_id: int
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int

    @property
    def duration_s(self) -> float:
        """The segment's length in seconds, first frame to last."""
        return (self.end_ms - self.start_ms) / 1000.0

    def row(self) -> dict[str, object]:
        """This segment as one CSV row."""
        return {
            "clip_id": self.clip_id,
            "phase": self.phase,
            "hold_id": self.hold_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_s": round(self.duration_s, 3),
        }


def segment_phases(result: ClipPhases) -> list[Segment]:
    """Every maximal stretch of frames sharing a phase, as :class:`Segment` rows.

    Holds are their own segments because the ``hold_id`` is part of what makes a
    segment one: two holds separated by a hand step are two segments, not one
    hold with a hole in it. Stretches of ``unknown`` are segments too — where
    the module could not look is an answer a later stage needs.
    """
    segments: list[Segment] = []
    start = 0
    for index in range(1, result.frames + 1):
        if index < result.frames and _same_segment(result, start, index):
            continue
        segments.append(
            Segment(
                clip_id=result.clip_id,
                phase=str(result.phase_codes[start]),
                hold_id=int(result.hold_id[start]),
                start_frame=int(result.frame_idx[start]),
                end_frame=int(result.frame_idx[index - 1]),
                start_ms=int(result.t_ms[start]),
                end_ms=int(result.t_ms[index - 1]),
            )
        )
        start = index
    return segments


def _same_segment(result: ClipPhases, first: int, second: int) -> bool:
    """Do frames ``first`` and ``second`` belong to the same segment?"""
    return bool(result.phase[first] == result.phase[second]) and bool(
        result.hold_id[first] == result.hold_id[second]
    )


def segments_table(results: Sequence[ClipPhases]) -> pd.DataFrame:
    """The segments of one or more clips as the CSV's table."""
    rows = [segment.row() for result in results for segment in segment_phases(result)]
    return pd.DataFrame(rows, columns=list(SEGMENT_COLUMNS))


def write_segments(path: str | pathlib.Path, table: pd.DataFrame) -> pathlib.Path:
    """Write the segment table, replacing the rows of the clips in ``table``.

    The file is per source, not per run: a run over ten clips must not delete the
    other hundred and seventy, so the rows of the clips this run covered are
    dropped and these written in their place, sorted by clip and start frame. A
    run over every clip therefore writes the whole dataset, and a re-run of one
    clip refreshes just that clip. It is written to a temporary file and renamed,
    so an interrupted run cannot leave a half-written table behind.
    """
    path = pathlib.Path(path)
    keep = pd.DataFrame(columns=list(SEGMENT_COLUMNS))
    if path.is_file():
        existing = pd.read_csv(path)
        missing = [column for column in SEGMENT_COLUMNS if column not in existing.columns]
        if missing:
            raise ValueError(f"{path}: segments are missing column(s) {missing}")
        clips = set(table["clip_id"].astype(str))
        keep = existing[~existing["clip_id"].astype(str).isin(clips)]
    merged = pd.concat([keep, table], ignore_index=True)
    merged = merged.sort_values(["clip_id", "start_frame"], kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        merged.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


# --------------------------------------------------------------------------- #
# What a run reports
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipStats:
    """What one clip's run produced, in the numbers the summary is built from."""

    clip_id: str
    source: str
    frames: int
    known_frames: int
    hold_count: int
    longest_hold_s: float
    total_hold_s: float
    phase_frames: dict[str, int]
    unusable_reason: str = ""
    runtime_seconds: float = 0.0

    @property
    def usable(self) -> bool:
        """Does this clip have a scale, and so any phases at all?"""
        return not self.unusable_reason

    @property
    def has_hold(self) -> bool:
        """Did the clip contain at least one hold?"""
        return self.hold_count > 0

    def line(self) -> str:
        """The one progress line the CLI prints for this clip.

        The phases are counted in frames, not seconds: a clip is variable frame
        rate, so a frame count is the honest per-phase summary here, and the
        durations in seconds are in the segment table and the run summary.
        """
        scale = "unusable" if not self.usable else f"holds={self.hold_count}"
        return (
            f"clip  {self.clip_id} frames={self.frames} known={self.known_frames} "
            f"{scale} longest={self.longest_hold_s:.2f}s total={self.total_hold_s:.2f}s "
            f"pre={_frames(self.phase_frames, 'pre')} "
            f"kickup={_frames(self.phase_frames, 'kickup')} "
            f"hold={_frames(self.phase_frames, 'hold')} "
            f"exit={_frames(self.phase_frames, 'exit')} "
            f"post={_frames(self.phase_frames, 'post')} "
            f"unknown={_frames(self.phase_frames, 'unknown')} "
            f"runtime={self.runtime_seconds:.2f}s"
        )


def _frames(phase_frames: dict[str, int], phase: str) -> int:
    """How many frames one phase got."""
    return int(phase_frames.get(phase, 0))


def measure(result: ClipPhases, runtime_seconds: float = 0.0) -> ClipStats:
    """Everything one clip's run produced, measured off its answer."""
    durations = result.hold_durations_s()
    return ClipStats(
        clip_id=result.clip_id,
        source=result.source,
        frames=result.frames,
        known_frames=int(result.signals.known.sum()),
        hold_count=result.hold_count,
        longest_hold_s=max(durations, default=0.0),
        total_hold_s=float(sum(durations)),
        phase_frames=result.phase_counts(),
        unusable_reason=result.unusable_reason,
        runtime_seconds=runtime_seconds,
    )


def summary(reports: Sequence[ClipStats]) -> str:
    """The dataset-level answer, printed at the end of a run.

    Four questions, in this order: how many clips have a hold at all, how long
    the holds are, how much hold time the dataset holds, and which clips have
    none. The last one is capped at :data:`NO_HOLD_CLIPS_LISTED` names because a
    summary that printed eighty clip ids would be read past; the full list is one
    filter away in :data:`SEGMENTS_NAME`.
    """
    lines = [f"summary clips={len(reports)}"]
    if not reports:
        return "\n".join(lines)
    with_holds = [report for report in reports if report.has_hold]
    without = [report for report in reports if not report.has_hold]
    usable = [report for report in reports if report.usable]
    lines[0] += f" usable={len(usable)} with_hold={len(with_holds)} no_hold={len(without)}"
    longest = np.array(
        [report.longest_hold_s for report in reports if report.longest_hold_s > 0.0], dtype=float
    )
    if longest.size:
        lines.append(
            f"  longest hold (s)  median={np.median(longest):.2f} "
            f"p25={np.percentile(longest, 25):.2f} p75={np.percentile(longest, 75):.2f} "
            f"max={longest.max():.2f} over {longest.size} clip(s) with a hold"
        )
    else:
        lines.append("  longest hold (s)  none: no clip has a hold")
    lines.append(
        f"  hold time         {sum(report.total_hold_s for report in reports):.1f} s "
        f"in {len(with_holds)} clip(s), "
        f"{sum(report.hold_count for report in reports)} hold(s) in total"
    )
    phase_totals = {
        phase: sum(report.phase_frames.get(phase, 0) for report in reports) for phase in PHASES
    }
    frames = sum(phase_totals.values())
    lines.append(
        "  phase frames      "
        + "  ".join(
            f"{phase}={phase_totals[phase]} ({_pct(phase_totals[phase], frames)})"
            for phase in PHASES
        )
    )
    if without:
        listed = without[:NO_HOLD_CLIPS_LISTED]
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


def _pct(part: int, total: int) -> str:
    return f"{100.0 * part / total:.1f} %" if total else "n/a"


# --------------------------------------------------------------------------- #
# Locations, writing, one clip
# --------------------------------------------------------------------------- #


def input_dir(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pathlib.Path:
    """``<data_dir>/processed/<source>``, the trajectories this stage reads."""
    return pathlib.Path(data) / PROCESSED_DIRNAME / source


def output_dir(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pathlib.Path:
    """``<data_dir>/phases/<source>``, where that source's phases go."""
    return pathlib.Path(data) / PHASES_DIRNAME / source


def missing_input_message(path: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> str:
    """The error shown when the processed keypoints have not been generated yet."""
    command = "uv run python -m handstand.postprocess --all"
    if source != DEFAULT_SOURCE:
        command += f" --source {source}"
    return (
        f"no processed keypoints for source {source!r}: {path}\n"
        "generate them first with:\n"
        f"  cd pipeline && {command}"
    )


def available_clips(
    in_root: str | pathlib.Path,
    clips: Sequence[str] | None = None,
    source: str = DEFAULT_SOURCE,
) -> list[str]:
    """The clip ids to label: ``clips`` as given, else every processed clip found.

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
    result: ClipPhases
    stats: ClipStats
    parquet_path: pathlib.Path
    skipped: bool = False

    def line(self) -> str:
        """The one progress line the CLI prints for this clip."""
        if self.skipped:
            return f"skip  {self.clip_id} ({self.parquet_path.name} exists)"
        return self.stats.line()


def run_clip(
    clip_id: str,
    *,
    in_root: str | pathlib.Path,
    out_root: str | pathlib.Path,
    source: str = DEFAULT_SOURCE,
    overwrite: bool = False,
) -> ClipReport:
    """Label one clip and write its parquet.

    Reads ``<in_root>/<clip_id>.parquet`` and the sidecar beside it, and writes
    ``<out_root>/<clip_id>.parquet``. A clip whose parquet is already there is
    skipped unless ``overwrite``, which is what makes a batch over the dataset
    resumable. The parquet is written to a temporary file and renamed last: its
    existence is what marks the clip as done.
    """
    in_path = pathlib.Path(in_root) / f"{clip_id}.parquet"
    if not in_path.is_file():
        raise FileNotFoundError(missing_input_message(in_path, source))
    out_dir = pathlib.Path(out_root)
    parquet_path = out_dir / f"{clip_id}.parquet"

    started = time.perf_counter()
    clip, valid = read_processed(in_path)
    body_length, reason = read_body_length(in_path.with_suffix(".json"))
    result = classify_clip(
        clip_id,
        source,
        clip.t_ms,
        clip.x,
        clip.y,
        valid,
        clip.trainer_contact,
        body_length,
        clip.joints,
        reason=reason,
    )
    runtime_seconds = time.perf_counter() - started
    stats = measure(result, runtime_seconds)

    if parquet_path.exists() and not overwrite:
        return ClipReport(
            clip_id=clip_id,
            result=result,
            stats=stats,
            parquet_path=parquet_path,
            skipped=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = parquet_path.with_name(f".{parquet_path.name}.tmp")
    try:
        phase_table(result).to_parquet(tmp, index=False)
        tmp.replace(parquet_path)
    finally:
        tmp.unlink(missing_ok=True)
    return ClipReport(
        clip_id=clip_id,
        result=result,
        stats=stats,
        parquet_path=parquet_path,
        skipped=False,
    )


# --------------------------------------------------------------------------- #
# The overlay
# --------------------------------------------------------------------------- #


def phase_captions(table: pd.DataFrame) -> dict[int, list[str]]:
    """``{frame_idx: [caption lines]}`` for :func:`handstand.overlay.render_overlay`.

    Two lines at most: the phase, and the hold number when the frame is inside a
    hold. That is the whole point of the overlay — watching where the boundaries
    land, which is the only way to tell a detector that is right from one that
    is confidently wrong.
    """
    captions: dict[int, list[str]] = {}
    for row in table.itertuples(index=False):
        lines = [f"phase {row.phase}"]
        if int(row.hold_id) != NO_HOLD:
            lines.append(f"hold {int(row.hold_id)}")
        captions[int(row.frame_idx)] = lines
    return captions


def render_clip(
    clip_id: str,
    *,
    data: str | pathlib.Path | None = None,
    videos: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
    max_frames: int | None = None,
) -> overlay.OverlayReport:
    """Render ``clip_id`` with its phase and hold number on every frame.

    The panel drawn is the *processed* trajectory this module read, so the
    skeleton and the caption are of the same frame of the same track, and the
    caption comes from the phases parquet. The output lands beside the other
    overlays as ``<clip_id>_phases.mp4`` (``<clip_id>_phases_<source>.mp4`` for a
    non-default source), and the videos directory is only ever read.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    phases_path = output_dir(root, source) / f"{clip_id}.parquet"
    if not phases_path.is_file():
        raise FileNotFoundError(
            f"no phases for clip {clip_id}: {phases_path}\n"
            "generate them first with:\n"
            f"  cd pipeline && uv run python -m handstand.phases --clips {clip_id}"
        )
    table = read_clip_phases(phases_path)
    video_path = overlay.video_for_clip(clip_id, videos, root)
    processed = input_dir(root, source) / f"{clip_id}.parquet"
    if not processed.is_file():
        raise FileNotFoundError(missing_input_message(processed, source))
    out_path = overlay.overlays_dir(root) / overlay.overlay_filename(
        clip_id, [PHASES_DIRNAME], source=source
    )
    return overlay.render_overlay(
        video_path,
        [(PHASES_DIRNAME, processed)],
        out_path,
        max_frames=max_frames,
        extra_captions=phase_captions(table),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The argument parser of ``python -m handstand.phases``."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.phases",
        description=(
            "Label every frame of a clip pre, kick-up, hold, exit or post, and number the holds."
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--clips",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help="clip ids to label",
    )
    selection.add_argument(
        "--all",
        dest="all_clips",
        action="store_true",
        help="label every clip with processed keypoints",
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
        help="data directory holding processed/ and phases/ (default: $HANDSTAND_DATA)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="label at most N clips",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-label clips whose parquet already exists",
    )
    parser.add_argument(
        "--render",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help=(
            "after the batch, render these clips with the phase and hold number on "
            "every frame (the clip must already have phases)"
        ),
    )
    parser.add_argument(
        "--videos",
        type=pathlib.Path,
        default=None,
        help="videos directory (default: $HANDSTAND_VIDEOS, else the workspace videos dir)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="render at most N frames of each --render clip",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batch, write the segment table, render, and print the summary."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.max_frames is not None and args.max_frames < 0:
        parser.error("--max-frames must be >= 0")

    root = pathlib.Path(args.data) if args.data is not None else data_dir()
    in_root = input_dir(root, args.source)
    out_root = output_dir(root, args.source)
    try:
        clip_ids = available_clips(in_root, None if args.all_clips else args.clips, args.source)
    except FileNotFoundError as error:
        print(f"phases: {error}")
        return 2
    if args.limit is not None:
        clip_ids = clip_ids[: args.limit]
    if not clip_ids:
        print("no clips to label")
        return 0

    print(f"source={args.source} clips={len(clip_ids)} in={in_root} out={out_root}")
    reports: list[ClipStats] = []
    results: list[ClipPhases] = []
    failures = 0
    for clip_id in clip_ids:
        try:
            report = run_clip(
                clip_id,
                in_root=in_root,
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
            results.append(report.result)

    if results:
        path = write_segments(out_root / SEGMENTS_NAME, segments_table(results))
        print(f"segments {path} ({len(clip_ids) - len(results)} clip(s) already labelled)")
    print(summary(reports))

    for clip_id in args.render or ():
        try:
            report = render_clip(
                clip_id,
                data=root,
                videos=args.videos,
                source=args.source,
                max_frames=args.max_frames,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"overlay {clip_id}: {error}", file=sys.stderr)
            failures += 1
            continue
        print(
            f"render {clip_id} frames={report.frame_count} "
            f"size={report.width}x{report.height} -> {report.out_path}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
