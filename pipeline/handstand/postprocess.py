"""Keypoint post-processing: gating, outliers, gap fill, One-Euro smoothing, body length.

A pose model reports a *guess per frame*, and the guesses do not agree with each
other: a joint flickers in and out of visibility, a hand jumps to the trainer's
wrist for a few frames, a keypoint jitters by a couple of pixels while the
athlete stands still. Every stage after this one — phases (#21), features (#22),
centre of mass (#23) — measures *how the body moved*, so they need a trajectory
that is clean, smooth and in the athlete's own units, not a per-frame guess.
That is what this module produces::

    <data_dir>/processed/<source>/<clip_id>.parquet   # the input schema, x/y processed
    <data_dir>/processed/<source>/<clip_id>.json      # what was done to the clip

CLI::

    cd pipeline
    uv run python -m handstand.postprocess --source mediapipe --rotate auto --all
    uv run python -m handstand.postprocess --clips 6508f9b355bd --overwrite

The input is the athlete selection's parquet (``keypoints/mediapipe_athlete/``
by default, ``--source vision`` for ``vision_athlete/``); the two are the same
schema, so the rules below do not know which model produced the keypoints and
adding a third model later is a flag, not a rewrite.

The output
----------

The **same long schema as the input**, one row per ``(frame, joint)`` and the
same rows in the same order, so :mod:`handstand.overlay` draws it unchanged: the
columns the model wrote (``z``, ``visibility``, ``detected``, ``trainer_contact``,
…) are passed through untouched and ``x``/``y`` carry the processed position,
NaN wherever the position is not known. Four columns are added:

``x_raw``, ``y_raw``
    What the model said, before any of this. Kept so a later stage (or a human
    looking at an overlay) can see the difference the processing made.
``valid``
    Does this sample have a position at all. False exactly where ``x``/``y`` are
    NaN, so one filter drops every unusable sample.
``filled``
    Was this position interpolated across a short gap rather than measured. A
    filled sample is valid, but it is a bridge, not an observation.

What is done to a clip
----------------------

Five steps, in this order, all thresholds being the module constants at the top:

1. **Gating** (:func:`gated_valid`) — a frame the model found nobody in
   (``detected = false``) or the athlete selection flagged as trainer contact
   has *every* joint invalid, and so has any joint below
   :data:`MIN_VISIBILITY` (or with no visibility score at all, which
   :mod:`handstand.overlay` also refuses to draw). A handstand is upside down and
   the trainer is regularly within a foot of it, so a contact frame's keypoints
   are two bodies' keypoints stitched together; carrying them into a trajectory
   is how a CoM estimate ends up on the wrong person.
2. **Body length** (:func:`estimate_body_length`) — ``L`` in pixels: torso
   (shoulder midpoint to hip midpoint) + thigh + shin, each the **90th
   percentile** of that span over the clip's valid frames. The percentile is
   what makes it a scale rather than a pose: a split, a stag, a leg swung out of
   the picture plane or a body foreshortened by the camera can only make a
   projected span *shorter*, so the frames where the limb lies in the picture
   plane set the yardstick and the others are measured against it (the same idea
   as the bone-length rule of #70). A clip with too few valid frames to measure
   has no ``L`` and therefore no scale: it is marked unusable in the sidecar and
   written with every joint invalid, so the dataset can be counted without it.
3. **Outliers** (:func:`remove_speed_outliers`) — a joint that moves more than
   :data:`MAX_SPEED_L_PER_S` body lengths in a second between two consecutive
   valid samples has jumped, not travelled, and is invalid for that frame. The
   limit is in body lengths rather than pixels so that a fast jump on a small
   athlete in the frame is judged as harshly as on a tall one.
4. **Gap fill** (:func:`fill_gaps`) — a run of invalid samples is interpolated
   linearly *in time* across it when its two ends are at most
   :data:`MAX_GAP_S` apart, and left NaN when they are further apart: over a
   fifth of a second there is no honest straight line to draw, and a long
   bridge would let a phase detector read the interpolation as movement. Filled
   samples are marked as such.
5. **Smoothing** (:func:`smooth_track`) — a One-Euro filter per joint
   coordinate, with the real ``dt`` from ``t_ms`` (these clips are variable frame
   rate, so a fixed ``dt`` would smooth a 15 fps frame as hard as a 30 fps one).
   The filter runs on positions in body lengths, is fed the interpolated gaps
   and is **restarted after every invalid run**, because its whole point is
   continuity, and a filter dragged across a trainer's occlusion would smooth
   the athlete into them. One-Euro rather than a plain low-pass because its
   cutoff follows the speed: a still athlete is smoothed hard, a moving one
   hardly at all, so phase boundaries survive.

The body frame
--------------

:mod:`handstand.bodyframe` holds the coordinate system the later stages work in
— origin at the wrist midpoint, ``u`` to the right, ``v`` up, both in body
lengths — plus :func:`to_body_frame`, which is all a later stage needs to place
a point in it.

Reading the result
------------------

Everything a later stage needs is either a column or a constant: a sample is
usable when ``valid`` is true, the clip's scale is ``body_length_px`` in the
sidecar, and ``bodyframe.to_body_frame`` turns a position into the athlete's
frame. ``valid`` and ``filled`` are the two columns worth filtering on, and
neither of them needs the sidecar.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from handstand import athlete
from handstand.paths import data_dir
from handstand.pose_mediapipe import JOINT_INDEX, PARQUET_COLUMNS, ROTATE_MODES

__all__ = [
    "BETA",
    "BODY_LENGTH_PERCENTILE",
    "CORE_JOINTS",
    "D_CUTOFF",
    "DEFAULT_ROTATE",
    "DEFAULT_SOURCE",
    "EXTRA_COLUMNS",
    "FOOT_INDEX_JOINTS",
    "HOLD_LIKE_JOINTS",
    "LEG_SEGMENTS",
    "LEG_SIDES",
    "LENGTH_PARTS",
    "MAX_GAP_S",
    "MAX_SPEED_L_PER_S",
    "MIN_BODY_FRAMES",
    "MIN_CUTOFF",
    "MIN_SAMPLE_DT",
    "MIN_VISIBILITY",
    "PROCESSED_DIRNAME",
    "SOURCES",
    "TORSO_ENDS",
    "TRACKED_JOINTS",
    "BodyLength",
    "ClipKeypoints",
    "ClipReport",
    "ClipStats",
    "OneEuroFilter",
    "ProcessedClip",
    "available_clips",
    "build_arg_parser",
    "clip_table",
    "estimate_body_length",
    "fill_gaps",
    "gated_valid",
    "hold_like_frames",
    "input_dirname",
    "joint_columns",
    "main",
    "measure",
    "median_jitter_l",
    "missing_input_message",
    "output_dir",
    "process_clip",
    "processed_table",
    "read_clip",
    "remove_speed_outliers",
    "run_clip",
    "smooth_track",
    "summary",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: ``--source`` values, in the order the flag lists them. The input root of each
#: is the matching :mod:`handstand.athlete` output, and the output root is
#: ``processed/<source>/``, so the two models' processed clips never overwrite
#: each other.
SOURCES: tuple[str, ...] = athlete.SOURCES
DEFAULT_SOURCE = athlete.DEFAULT_SOURCE
DEFAULT_ROTATE = athlete.DEFAULT_ROTATE

#: Output root under ``<data_dir>/``: the processed clips, one directory per
#: source. Unlike ``keypoints/`` it holds no rotation-mode sub-directory, so one
#: clip has one processed trajectory — the recommended ``auto`` mode, which is
#: what ``--rotate`` defaults to. Re-running a clip in a different mode needs
#: ``--overwrite``, and the mode is recorded in the sidecar either way.
PROCESSED_DIRNAME = "processed"

#: Columns the output adds to the input schema, in write order.
EXTRA_COLUMNS: tuple[str, ...] = ("x_raw", "y_raw", "valid", "filled")

#: A joint below this visibility is not a position, it is a rumour — the same
#: threshold :mod:`handstand.overlay` draws at, and the same one the athlete
#: selection measures its bones with, so the two agree on which frames carry
#: evidence. A visibility of NaN counts as below: the model did not report one.
MIN_VISIBILITY = 0.5

#: The percentile of a span's per-frame length that counts as that span's full
#: length. 90 % of the frames sit below it, so the few where a limb is
#: contaminated cannot drag the yardstick, while a limb that is foreshortened in
#: most of the clip is still measured against the frames where it is not.
BODY_LENGTH_PERCENTILE = 90.0

#: A part of the body length needs this many valid frames before it is measured
#: at all. Ten frames is about a third of a second at 30 fps: enough for a
#: percentile to mean something, few enough that a clip is not written off
#: because it starts late.
MIN_BODY_FRAMES = 10

#: A joint that covers this many body lengths in a second between two
#: consecutive valid samples has jumped rather than travelled. Nothing in a
#: handstand moves that fast: the fastest thing in a clip is a leg kicking into
#: a handstand, a few body lengths per second, so eight leaves room for the real
#: thing and cuts the model's teleports.
MAX_SPEED_L_PER_S = 8.0

#: A gap between two valid samples is filled when it is at most this long, and
#: left NaN when it is longer. A fifth of a second is a blink, not a bridge: past
#: it, the straight line between the two ends says nothing about where the joint
#: was in between, and a phase detector reading it as movement would invent a
#: phase the athlete never performed.
MAX_GAP_S = 0.2

#: One-Euro filter cutoffs and speed weight, in the units of the filter: cutoffs
#: in Hz on positions measured in body lengths, ``BETA`` multiplying the filtered
#: speed (body lengths per second) to raise the cutoff while the joint moves.
MIN_CUTOFF = 1.0
BETA = 0.3
D_CUTOFF = 1.0
#: The smallest ``dt`` the filter is allowed to see, in seconds. The schema says
#: ``t_ms`` strictly increases, so this only guards a duplicate timestamp: there
#: is no speed to measure across one, so the filter follows the sample instead
#: of smoothing it.
MIN_SAMPLE_DT = 1e-3

#: The joints both pose models report, and the only ones the body length, the
#: body frame and the report's numbers are built from: the nose plus both
#: shoulders, elbows, wrists, hips, knees and ankles. A clip of a source that
#: does not report one of them (Apple Vision has no ``foot_index``) leaves that
#: joint NaN rather than guessing it, which is why the foot indexes are listed
#: separately in :data:`FOOT_INDEX_JOINTS`.
CORE_JOINTS: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
#: MediaPipe's two big-toe landmarks. They are kept when the source has them and
#: are NaN when it does not; every measurement that matters is in
#: :data:`CORE_JOINTS`, so nothing downstream depends on a model reporting them.
FOOT_INDEX_JOINTS: tuple[str, ...] = ("left_foot_index", "right_foot_index")
#: :data:`CORE_JOINTS` plus the foot indexes: every joint this module's own
#: numbers are measured over, i.e. the joints of the shared schema.
TRACKED_JOINTS: tuple[str, ...] = (*CORE_JOINTS, *FOOT_INDEX_JOINTS)

#: Wrists and ankles: where a handstand's hands and a standing person's feet meet
#: the floor. The mean ``y`` of the wrists is below the mean ``y`` of the ankles
#: in every frame where the athlete is not inverted, so these four joints are what
#: a hold is read off — and they are the four this module's own jitter number is
#: measured on, since a wrist on the floor is where a detector is least sure.
HOLD_LIKE_JOINTS: tuple[str, ...] = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")

#: The ends of the torso span, as two joint pairs: the span is measured from the
#: midpoint of the first pair to the midpoint of the second, so both shoulders and
#: both hips have to be visible in a frame for that frame to count.
TORSO_ENDS: tuple[tuple[str, str], tuple[str, str]] = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
)
#: The segments of a leg, as ``(name, first joint, second joint)``, each measured
#: on **both** sides: the **longer** leg of a frame is the one that sets the
#: length (:func:`estimate_body_length`), because a split or a stag foreshortens
#: one leg and never lengthens it.
LEG_SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("thigh", "hip", "knee"),
    ("shin", "knee", "ankle"),
)
#: The sides each leg segment is measured on.
LEG_SIDES: tuple[str, ...] = ("left", "right")
#: Every part of the body length, in the order the sidecar lists them.
LENGTH_PARTS: tuple[str, ...] = ("torso", *(name for name, _, _ in LEG_SEGMENTS))


# --------------------------------------------------------------------------- #
# One clip as arrays
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipKeypoints:
    """One keypoint parquet as ``(frames, joints)`` arrays, plus the long table.

    The long schema is one row per ``(frame, joint)`` with the frame's columns
    repeated on each of its rows, which is a ``(F, J)`` array wearing a disguise.
    :func:`read_clip` does the unfolding, checks that the disguise is consistent
    (every frame holds each joint exactly once, and a frame's own columns agree
    across its rows) and keeps the sorted table, so the output can be written as
    the same rows in the same order whatever order the file happened to use.
    """

    table: pd.DataFrame
    frame_idx: np.ndarray
    t_ms: np.ndarray
    joints: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    visibility: np.ndarray
    detected: np.ndarray
    trainer_contact: np.ndarray
    rotated: np.ndarray
    has_contact_column: bool = False

    @property
    def frames(self) -> int:
        """How many frames the clip has."""
        return int(self.frame_idx.size)

    @property
    def joint_count(self) -> int:
        """How many joints the clip's schema has."""
        return len(self.joints)

    @property
    def t_seconds(self) -> np.ndarray:
        """Frame timestamps in seconds — the only clock these clips have.

        They are variable frame rate, so nothing in here may assume a constant
        step between frames.
        """
        return self.t_ms.astype(np.float64) / 1000.0

    def column(self, joint: str) -> int:
        """The column of ``joint``, or :class:`KeyError` when this clip has none."""
        try:
            return self.joints.index(joint)
        except ValueError:
            raise KeyError(f"{joint} is not one of this clip's joints") from None


def joint_columns(joints: Sequence[str], wanted: str) -> int:
    """The column of ``wanted`` in ``joints``, or ``-1`` when it is not there.

    Every measurement below is written against *names* rather than indices and
    skips a joint the source does not report, which is what lets a MediaPipe file
    (33 joints) and a Vision file (the same 33 rows, 14 of them NaN) go through
    the same code.
    """
    try:
        return list(joints).index(wanted)
    except ValueError:
        return -1


def read_clip(parquet_path: str | pathlib.Path) -> ClipKeypoints:
    """Read one keypoint parquet into :class:`ClipKeypoints`, in frame order.

    The joints are put back into the schema's own order — the model's landmark
    order, which is what ``docs/keypoint_schema.md`` says the rows of a frame
    are in — rather than the order this file happened to write them, so a shuffled
    file and a sorted one read identically and the output is canonical. The rows
    keep the input's columns, so the output of a `mediapipe_athlete` file and of a
    `vision_athlete` file of the same clip line up row for row.

    Raises :class:`ValueError` when the file is not the long schema, when a joint
    name is not one the schema knows, or when a frame does not hold each joint
    exactly once — all of which mean the file is not what this module reads, and
    quietly re-indexing it would produce a plausible-looking wrong answer.
    """
    path = pathlib.Path(parquet_path)
    table = pd.read_parquet(path)
    missing = [column for column in PARQUET_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"{path}: keypoints are missing column(s) {missing}; see docs/keypoint_schema.md"
        )
    if table.empty:
        raise ValueError(f"{path}: no rows in the keypoint parquet")

    names = list(dict.fromkeys(str(name) for name in table["joint"]))
    unknown = [name for name in names if name not in JOINT_INDEX]
    if unknown:
        raise ValueError(
            f"{path}: joint(s) {unknown} are not part of the keypoint schema; "
            "see docs/keypoint_schema.md"
        )
    joints = tuple(sorted(names, key=JOINT_INDEX.__getitem__))
    frame_values = table["frame_idx"].to_numpy(dtype=np.int64)
    frames = np.unique(frame_values)
    columns = np.array([JOINT_INDEX[name] for name in table["joint"]], dtype=np.int64)
    rows = np.searchsorted(frames, frame_values)
    # Model order, not the file's joint order, is what the completeness check is
    # made against: every frame must hold each of the schema's joints it has once.
    order = np.lexsort((columns, rows))
    per_frame = len(joints)
    expected_rows = np.repeat(np.arange(frames.size), per_frame)
    if rows[order].size != frames.size * per_frame or not np.array_equal(
        rows[order], expected_rows
    ):
        raise ValueError(
            f"{path}: frames do not hold {per_frame} joints once each; "
            "re-run the keypoint extraction"
        )
    expected_columns = np.array([JOINT_INDEX[name] for name in joints], dtype=np.int64)
    if not np.array_equal(
        columns[order].reshape(frames.size, per_frame),
        np.tile(expected_columns, (frames.size, 1)),
    ):
        raise ValueError(
            f"{path}: a frame does not hold each of the {per_frame} joints once; "
            "re-run the keypoint extraction"
        )

    ordered = table.iloc[order].reset_index(drop=True)
    x = _frame_joint(ordered, "x", frames.size, per_frame)
    y = _frame_joint(ordered, "y", frames.size, per_frame)
    t_ms = _first_of_frame(ordered, "t_ms", frames.size, per_frame)
    if not np.all(_frame_joint(ordered, "t_ms", frames.size, per_frame) == t_ms[:, None]):
        raise ValueError(f"{path}: the rows of a frame disagree about t_ms")
    rotated = _first_of_frame(ordered, "rotated", frames.size, per_frame)
    detected = _any_of_frame(ordered, "detected", frames.size, per_frame)
    has_contact = athlete.CONTACT_COLUMN in ordered.columns
    contact = (
        _any_of_frame(ordered, athlete.CONTACT_COLUMN, frames.size, per_frame)
        if has_contact
        else np.zeros(frames.size, dtype=bool)
    )
    return ClipKeypoints(
        table=ordered,
        frame_idx=frames,
        t_ms=t_ms.astype(np.int64),
        joints=joints,
        x=x,
        y=y,
        visibility=_frame_joint(ordered, "visibility", frames.size, per_frame),
        detected=detected,
        trainer_contact=contact,
        rotated=rotated,
        has_contact_column=has_contact,
    )


def _frame_joint(table: pd.DataFrame, column: str, frames: int, joints: int) -> np.ndarray:
    """One float column of the long schema as a ``(frames, joints)`` array."""
    return table[column].to_numpy(dtype=np.float64).reshape(frames, joints)


def _first_of_frame(table: pd.DataFrame, column: str, frames: int, joints: int) -> np.ndarray:
    """The first row of each frame's column, as a ``(frames,)`` array."""
    values = table[column].to_numpy()
    return np.asarray(values[::joints])


def _any_of_frame(table: pd.DataFrame, column: str, frames: int, joints: int) -> np.ndarray:
    """True where any row of the frame has the column set — the per-frame ``any``.

    The frame-level columns are repeated on every row of a frame, so this only
    differs from the first row when the file is inconsistent; ``any`` is the
    conservative reading, the same one :func:`handstand.overlay.load_panel` uses.
    """
    values = table[column].to_numpy(dtype=bool)
    return values.reshape(frames, joints).any(axis=1)


# --------------------------------------------------------------------------- #
# Step 1: gating
# --------------------------------------------------------------------------- #


def gated_valid(
    x: np.ndarray,
    y: np.ndarray,
    visibility: np.ndarray,
    frame_valid: np.ndarray,
    min_visibility: float = MIN_VISIBILITY,
) -> np.ndarray:
    """Which ``(frame, joint)`` samples the rest of the pipeline may use.

    A sample survives when all four of these hold: its frame carries a pose that
    is not in trainer contact (``frame_valid``, i.e. ``detected and not
    trainer_contact``), its coordinates are finite, and its visibility is at or
    above ``min_visibility``. A visibility that is NaN fails the comparison, which
    is the same answer :mod:`handstand.overlay` gives: a joint the model refused
    to score is not a position.

    Parameters
    ----------
    x, y, visibility:
        ``(frames, joints)`` arrays of the clip.
    frame_valid:
        ``(frames,)`` booleans, true where the frame is usable at all.
    min_visibility:
        The score at or above which a joint is confident.
    """
    frames, joints = x.shape
    if y.shape != x.shape or visibility.shape != x.shape:
        raise ValueError("x, y and visibility must have the same (frames, joints) shape")
    if frame_valid.shape != (frames,):
        raise ValueError(f"frame_valid must have {frames} entries, got {frame_valid.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    confident = visibility >= min_visibility
    return np.asarray(frame_valid, dtype=bool)[:, None] & finite & confident


# --------------------------------------------------------------------------- #
# Step 2: body length
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class BodyLength:
    """One clip's scale, in display-frame pixels, and whether it exists.

    ``L`` is the sum of the three :data:`LENGTH_PARTS` percentiles, so every
    part of the body is measured against the frames where *that part* lies in the
    picture plane. :attr:`reason` says why a clip has no ``L``, and is empty when
    it has one — an unusable clip is a fact about the clip, recorded rather than
    thrown away.
    """

    torso: float
    thigh: float
    shin: float
    torso_frames: int
    thigh_frames: int
    shin_frames: int
    reason: str = ""

    @property
    def usable(self) -> bool:
        """Is there a body length to normalise by?"""
        return not self.reason

    @property
    def total(self) -> float:
        """``L`` in pixels, or NaN when the clip is unusable."""
        return self.torso + self.thigh + self.shin if self.usable else math.nan

    def parts(self) -> dict[str, float]:
        """The three parts, keyed as in :data:`LENGTH_PARTS`, for the sidecar."""
        return {"torso": self.torso, "thigh": self.thigh, "shin": self.shin}

    def frames(self) -> dict[str, int]:
        """How many valid frames each part was measured on, for the sidecar."""
        return {
            "torso": self.torso_frames,
            "thigh": self.thigh_frames,
            "shin": self.shin_frames,
        }


def estimate_body_length(
    joints: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    *,
    percentile: float = BODY_LENGTH_PERCENTILE,
    min_frames: int = MIN_BODY_FRAMES,
) -> BodyLength:
    """Measure the clip's body length ``L`` in pixels, or say why it has none.

    Each part of :data:`LENGTH_PARTS` is the ``percentile``-th percentile of its
    per-frame length over the frames where it could be measured. The torso is
    measured midpoint to midpoint; the thigh and the shin are measured on each
    side and the **longer** leg of a frame is the one kept, so a split, a stag or
    a straddle — which foreshortens one leg and never lengthens it — cannot
    shrink the yardstick the other frames are measured against.

    A part measured on fewer than ``min_frames`` frames has no percentile worth
    the name, and neither has a clip whose three parts do not add up to
    :data:`handstand.athlete.MIN_BODY_LENGTH_PIXELS`; either way the returned
    :class:`BodyLength` is unusable and says so in :attr:`BodyLength.reason`.

    Parameters
    ----------
    joints:
        The clip's joint names, in column order, so the parts can be found.
    x, y, valid:
        ``(frames, joints)`` arrays — the *gated* coordinates and the gate's
        answer. Outlier removal runs after this, because a jump does not make a
        frame foreshortened, only a limb pointing at the camera does.
    percentile:
        Which percentile of the per-frame lengths is the full length.
    min_frames:
        How many frames a part needs before it is measured.
    """
    if x.shape != y.shape or valid.shape != x.shape:
        raise ValueError("x, y and valid must have the same (frames, joints) shape")

    spans = {"torso": _torso_span(joints, x, y, valid)}
    spans.update(
        {
            name: _leg_span(joints, x, y, valid, first, second)
            for name, first, second in LEG_SEGMENTS
        }
    )
    counts = {name: int(np.isfinite(span).sum()) for name, span in spans.items()}
    parts = {name: _percentile(span, percentile) for name, span in spans.items()}

    # An unusable clip still records how many frames each part *was* measurable on:
    # a clip that is unusable because the trainer is in every frame of it is worth
    # telling apart from one that is unusable because the model saw nothing.
    def unusable(reason: str) -> BodyLength:
        return _unusable(reason, counts)

    for name, (value, count) in parts.items():
        if count < min_frames:
            return unusable(f"{name} was measurable on {count} frame(s), need {min_frames}")
        if not math.isfinite(value) or value <= 0.0:
            return unusable(f"{name} has no measurable length")
    total = sum(value for value, _ in parts.values())
    if total < athlete.MIN_BODY_LENGTH_PIXELS:
        return unusable(f"body length {total:.3f} px is below {athlete.MIN_BODY_LENGTH_PIXELS} px")
    return BodyLength(
        torso=parts["torso"][0],
        thigh=parts["thigh"][0],
        shin=parts["shin"][0],
        torso_frames=counts["torso"],
        thigh_frames=counts["thigh"],
        shin_frames=counts["shin"],
    )


def _unusable(reason: str, counts: Mapping[str, int]) -> BodyLength:
    """A :class:`BodyLength` that has none, carrying ``reason`` and the counts."""
    return BodyLength(
        torso=math.nan,
        thigh=math.nan,
        shin=math.nan,
        torso_frames=counts["torso"],
        thigh_frames=counts["thigh"],
        shin_frames=counts["shin"],
        reason=reason,
    )


def _percentile(values: np.ndarray, percentile: float) -> tuple[float, int]:
    """``(value, how many samples)`` of the percentile of the finite ``values``."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, 0
    return float(np.percentile(finite, percentile)), int(finite.size)


def _torso_span(
    joints: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """The per-frame shoulder-midpoint to hip-midpoint span, NaN where unmeasured.

    The ends are :data:`TORSO_ENDS`, two *pairs* of joints, so both shoulders and
    both hips have to be valid in a frame for that frame to count: a frame with
    one shoulder missing has no shoulder midpoint, and using the one shoulder on
    its own would make the measurement depend on which side happened to be
    visible.
    """
    shoulders, hips = (
        _midpoint(joints, x, y, valid, TORSO_ENDS[0]),
        _midpoint(joints, x, y, valid, TORSO_ENDS[1]),
    )
    length = np.hypot(shoulders[0] - hips[0], shoulders[1] - hips[1])
    return np.where(shoulders[2] & hips[2], length, np.nan)


def _leg_span(
    joints: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    first: str,
    second: str,
) -> np.ndarray:
    """The longer leg of a frame, as the per-frame ``first`` to ``second`` span.

    Per side, then the max over the sides: whichever leg is seen side-on that
    frame sets the length, and a frame where only one leg is visible contributes
    that leg rather than nothing.
    """
    sides = [
        _pair_span(joints, x, y, valid, f"{side}_{first}", f"{side}_{second}") for side in LEG_SIDES
    ]
    stacked = np.stack(sides)
    known = np.isfinite(stacked).any(axis=0)
    longest = np.max(np.where(np.isfinite(stacked), stacked, -np.inf), axis=0)
    return np.where(known, longest, np.nan)


def _pair_span(
    joints: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    first: str,
    second: str,
) -> np.ndarray:
    """Per-frame distance between two named joints, NaN where either is invalid."""
    a, b = joint_columns(joints, first), joint_columns(joints, second)
    if a < 0 or b < 0:
        return np.full(x.shape[0], np.nan)
    length = np.hypot(x[:, a] - x[:, b], y[:, a] - y[:, b])
    return np.where(valid[:, a] & valid[:, b], length, np.nan)


def _midpoint(
    joints: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    pair: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The midpoint of two named joints, and whether both of them are valid."""
    a, b = joint_columns(joints, pair[0]), joint_columns(joints, pair[1])
    if a < 0 or b < 0:
        nan = np.full(x.shape[0], np.nan)
        return nan, nan, np.zeros(x.shape[0], dtype=bool)
    both = valid[:, a] & valid[:, b]
    return (x[:, a] + x[:, b]) / 2.0, (y[:, a] + y[:, b]) / 2.0, both


# --------------------------------------------------------------------------- #
# Step 3: outliers
# --------------------------------------------------------------------------- #


def remove_speed_outliers(
    t_seconds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    body_length: float,
    max_speed: float = MAX_SPEED_L_PER_S,
) -> np.ndarray:
    """The validity mask with the teleports taken out.

    A joint that covers more than ``max_speed`` body lengths in a second between
    two consecutive valid samples is invalid in the *later* of the two: the model
    lost it, it did not travel that fast. The earlier sample is kept and stays
    the reference, so a single spike is dropped and the joint is not dragged
    invalid along with it, and a whole run of spikes collapses to its first
    sample. Samples that were already invalid are never used as a reference, and
    a pair with no positive ``dt`` between them is not evidence of a jump (the
    schema says ``t_ms`` increases, so this only guards a duplicate).

    Parameters
    ----------
    t_seconds:
        ``(frames,)`` timestamps in seconds — the real clock, because these
        clips are variable frame rate.
    x, y, valid:
        ``(frames, joints)`` arrays and the mask so far.
    body_length:
        ``L`` in pixels: the unit ``max_speed`` is measured in.
    max_speed:
        The limit, in body lengths per second.
    """
    if t_seconds.shape != (x.shape[0],):
        raise ValueError(f"t_seconds must have {x.shape[0]} entries, got {t_seconds.shape}")
    if y.shape != x.shape or valid.shape != x.shape:
        raise ValueError("x, y and valid must have the same (frames, joints) shape")
    length = float(body_length)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"body_length must be a positive, finite number, got {body_length!r}")

    kept = np.array(valid, dtype=bool, copy=True)
    for column in range(x.shape[1]):
        last = -1
        for index in np.flatnonzero(valid[:, column]):
            if last < 0:
                last = int(index)
                continue
            dt = float(t_seconds[index] - t_seconds[last])
            if dt > 0.0:
                jump = math.hypot(
                    float(x[index, column]) - float(x[last, column]),
                    float(y[index, column]) - float(y[last, column]),
                ) / (length * dt)
                if jump > max_speed:
                    kept[index, column] = False
                    continue
            last = int(index)
    return kept


# --------------------------------------------------------------------------- #
# Step 4: gap fill
# --------------------------------------------------------------------------- #


def fill_gaps(
    t_seconds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    max_gap_s: float = MAX_GAP_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bridge the short gaps in each joint: ``(x, y, valid, filled)``.

    A run of invalid samples is interpolated linearly **in time** between the
    valid samples on either side of it when those two are at most ``max_gap_s``
    apart, and left NaN when they are further apart or when there is nothing to
    interpolate between — the head and tail of a joint's track have no far side
    to lean on, and a track that starts 30 frames late does not get a made-up
    beginning. The returned ``x``/``y`` carry the bridges and NaN everywhere the
    sample is not valid, so "has a position" is one test rather than two;
    ``valid`` gains the bridges and ``filled`` marks exactly them.

    Interpolating in ``t_ms`` rather than in frame index is the whole point: a
    0.1 s gap is three frames in a 30 fps clip and one frame in a 10 fps one, and
    the samples on either side of it are in the middle of their own dt either way.
    """
    if t_seconds.shape != (x.shape[0],):
        raise ValueError(f"t_seconds must have {x.shape[0]} entries, got {t_seconds.shape}")
    if y.shape != x.shape or valid.shape != x.shape:
        raise ValueError("x, y and valid must have the same (frames, joints) shape")

    out_x, out_y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    out_valid = np.array(valid, dtype=bool, copy=True)
    filled = np.zeros(x.shape, dtype=bool)
    times = np.asarray(t_seconds, dtype=np.float64)
    for column in range(x.shape[1]):
        known = np.flatnonzero(valid[:, column])
        for first, second in zip(known[:-1], known[1:], strict=True):
            if second <= first + 1:
                continue
            span = float(times[second] - times[first])
            if span <= 0.0 or span > max_gap_s:
                continue
            inside = np.arange(first + 1, second)
            weight = (times[inside] - times[first]) / span
            out_x[inside, column] = x[first, column] + weight * (
                x[second, column] - x[first, column]
            )
            out_y[inside, column] = y[first, column] + weight * (
                y[second, column] - y[first, column]
            )
            out_valid[inside, column] = True
            filled[inside, column] = True
    return out_x, out_y, out_valid, filled


# --------------------------------------------------------------------------- #
# Step 5: One-Euro smoothing
# --------------------------------------------------------------------------- #


class OneEuroFilter:
    """A One-Euro filter for one scalar, driven by the real ``dt`` of each sample.

    Casiez, Roussel & Vogel (CHI 2012). The filter is a one-pole low-pass whose
    cutoff rises with the speed of the signal, so a still signal is smoothed hard
    and a moving one hardly at all: the jitter of a handstand at the top goes,
    while the boundary between the kick-up and the hold stays where it was — the
    one thing a fixed low-pass cannot do. ``beta`` is that coupling, in cutoff
    units per (signal unit / second); ``d_cutoff`` smooths the speed estimate
    itself, so a single noisy sample cannot swing the cutoff.

    The filter holds no state across a gap: :meth:`reset` starts it again from
    the next sample, which is what the post-processing wants after an occlusion —
    its own memory is what would drag the athlete towards the trainer.
    """

    def __init__(
        self,
        min_cutoff: float = MIN_CUTOFF,
        beta: float = BETA,
        d_cutoff: float = D_CUTOFF,
    ) -> None:
        if min_cutoff <= 0.0 or d_cutoff <= 0.0:
            raise ValueError("min_cutoff and d_cutoff must be positive")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        """Forget everything, so the next :meth:`__call__` starts the filter."""
        self._started = False
        self._value = 0.0
        self._speed = 0.0
        self._time = 0.0

    def _alpha(self, cutoff: float, dt: float) -> float:
        """The one-pole smoothing factor for a cutoff and a timestep."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: float, t_seconds: float) -> float:
        """Filter one sample measured at ``t_seconds`` and return the new value.

        The first sample after a :meth:`reset` is passed through untouched, the
        way every One-Euro implementation seeds itself: there is nothing to
        compare it against yet, and starting a filter from an average would put a
        spike in front of the first real sample.
        """
        if not self._started:
            self._started = True
            self._value = float(value)
            self._speed = 0.0
            self._time = float(t_seconds)
            return self._value
        dt = max(float(t_seconds) - self._time, MIN_SAMPLE_DT)
        speed = (float(value) - self._value) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        self._speed = alpha_d * speed + (1.0 - alpha_d) * self._speed
        alpha = self._alpha(self.min_cutoff + self.beta * abs(self._speed), dt)
        self._value = alpha * float(value) + (1.0 - alpha) * self._value
        self._time = float(t_seconds)
        return self._value


def smooth_track(
    t_seconds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    body_length: float,
    *,
    min_cutoff: float = MIN_CUTOFF,
    beta: float = BETA,
    d_cutoff: float = D_CUTOFF,
) -> tuple[np.ndarray, np.ndarray]:
    """One-Euro smooth every joint coordinate of a clip, in pixels.

    The filter runs on positions in **body lengths** and the results are scaled
    back, because its cutoffs are frequencies of a signal whose unit is the
    body's own: a cutoff of 1 Hz means "jitter slower than a second at a
    hundredth of a body length a second", and that is the same statement for a
    350 px athlete and a 600 px one. It is also what makes a cut-off constant
    mean the same thing on every clip in the dataset.

    The filter is restarted at every invalid sample, so a joint's track is
    smoothed within each run of valid samples and never across a gap it does not
    know the shape of. Invalid samples come back as NaN.
    """
    if t_seconds.shape != (x.shape[0],):
        raise ValueError(f"t_seconds must have {x.shape[0]} entries, got {t_seconds.shape}")
    if y.shape != x.shape or valid.shape != x.shape:
        raise ValueError("x, y and valid must have the same (frames, joints) shape")
    length = float(body_length)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"body_length must be a positive, finite number, got {body_length!r}")

    out_x = np.full(x.shape, np.nan)
    out_y = np.full(x.shape, np.nan)
    times = np.asarray(t_seconds, dtype=np.float64)
    for column in range(x.shape[1]):
        filters = (
            OneEuroFilter(min_cutoff, beta, d_cutoff),
            OneEuroFilter(min_cutoff, beta, d_cutoff),
        )
        for index in range(x.shape[0]):
            if not valid[index, column]:
                # Restart at every gap: the filter's memory is the athlete's own
                # motion, and across an invalid run that memory is a guess about
                # what happened while nobody was watching.
                filters[0].reset()
                filters[1].reset()
                continue
            out_x[index, column] = length * filters[0](x[index, column] / length, times[index])
            out_y[index, column] = length * filters[1](y[index, column] / length, times[index])
    return out_x, out_y


# --------------------------------------------------------------------------- #
# The processed clip
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ProcessedClip:
    """One clip's processed trajectory and the counts that explain it."""

    clip: ClipKeypoints
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray
    filled: np.ndarray
    gated: np.ndarray
    outliers: np.ndarray
    body_length: BodyLength
    hold_like: np.ndarray

    @property
    def usable(self) -> bool:
        """Does this clip have a body length, i.e. any usable scale?"""
        return self.body_length.usable

    @property
    def unfilled(self) -> np.ndarray:
        """Samples with no position left: a gap too long to bridge, or a track's end."""
        return ~self.valid

    def filled_samples(self) -> int:
        """How many samples were interpolated across a short gap."""
        return int(self.filled.sum())


def process_clip(
    clip: ClipKeypoints,
    *,
    min_visibility: float = MIN_VISIBILITY,
    percentile: float = BODY_LENGTH_PERCENTILE,
    min_frames: int = MIN_BODY_FRAMES,
    max_speed: float = MAX_SPEED_L_PER_S,
    max_gap_s: float = MAX_GAP_S,
    min_cutoff: float = MIN_CUTOFF,
    beta: float = BETA,
    d_cutoff: float = D_CUTOFF,
) -> ProcessedClip:
    """Run the five steps over one clip, in order.

    A clip with no body length — too few valid frames to measure one — comes back
    with every joint invalid, and :attr:`ProcessedClip.usable` false: without ``L``
    there is no scale to normalise by, no speed limit to judge a jump against and
    no meaningful unit for the filter, so the honest output is a clip that says
    so. The raw coordinates are still in the file (``x_raw``/``y_raw``), so
    nothing is lost and the clip can be re-run with a bigger sample.
    """
    gated = gated_valid(
        clip.x,
        clip.y,
        clip.visibility,
        clip.detected & ~clip.trainer_contact,
        min_visibility,
    )
    hold_like = hold_like_frames(clip.x, clip.y, gated, clip.joints)
    body = estimate_body_length(
        clip.joints, clip.x, clip.y, gated, percentile=percentile, min_frames=min_frames
    )
    if not body.usable:
        nothing = np.zeros_like(gated)
        return ProcessedClip(
            clip=clip,
            x=np.full(clip.x.shape, np.nan),
            y=np.full(clip.y.shape, np.nan),
            valid=nothing,
            filled=nothing.copy(),
            gated=gated,
            outliers=nothing.copy(),
            body_length=body,
            hold_like=hold_like,
        )

    times = clip.t_seconds
    after_outliers = remove_speed_outliers(times, clip.x, clip.y, gated, body.total, max_speed)
    filled_x, filled_y, bridged, filled = fill_gaps(
        times, clip.x, clip.y, after_outliers, max_gap_s
    )
    smooth_x, smooth_y = smooth_track(
        times,
        filled_x,
        filled_y,
        bridged,
        body.total,
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff,
    )
    return ProcessedClip(
        clip=clip,
        x=np.where(bridged, smooth_x, np.nan),
        y=np.where(bridged, smooth_y, np.nan),
        valid=bridged,
        filled=filled,
        gated=gated,
        outliers=after_outliers,
        body_length=body,
        hold_like=hold_like,
    )


def processed_table(processed: ProcessedClip) -> pd.DataFrame:
    """The processed clip as the input's long schema plus :data:`EXTRA_COLUMNS`.

    Every column of the input is passed through untouched, so the file still
    draws in :mod:`handstand.overlay` and still carries the model its own
    ``z``, ``visibility``, ``detected`` and ``trainer_contact``; ``x``/``y``
    become the processed position and NaN wherever there is none. The four added
    columns are in :data:`EXTRA_COLUMNS` order. Rows stay in the input's own
    order — :func:`read_clip` sorted them by frame and joint on the way in.
    """
    out = processed.clip.table.copy()
    out["x"] = processed.x.reshape(-1)
    out["y"] = processed.y.reshape(-1)
    out["x_raw"] = processed.clip.x.reshape(-1)
    out["y_raw"] = processed.clip.y.reshape(-1)
    out["valid"] = processed.valid.reshape(-1)
    out["filled"] = processed.filled.reshape(-1)
    return out


# --------------------------------------------------------------------------- #
# Measurements the sidecar and the run summary report
# --------------------------------------------------------------------------- #


def hold_like_frames(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    joints: Sequence[str] = CORE_JOINTS,
) -> np.ndarray:
    """Per frame: are the wrists below the ankles, i.e. is the athlete inverted?

    The same test the ``auto`` rotation mode uses to decide which way up the model
    should see a frame — mean wrist ``y`` greater than mean ankle ``y``, with ``y``
    growing downwards — applied to the *gated* raw coordinates, so the
    classification does not depend on the outlier, fill or smoothing decisions
    being measured. Each side is averaged over the joints of it that are visible,
    as long as there is at least one wrist and one ankle to average: a frame where
    only one wrist can be seen is still classifiable, and a frame where neither a
    wrist nor an ankle can is left unclassified rather than guessed at.
    """
    wrists = [
        column
        for column in (joint_columns(joints, name) for name in ("left_wrist", "right_wrist"))
        if column >= 0
    ]
    ankles = [
        column
        for column in (joint_columns(joints, name) for name in ("left_ankle", "right_ankle"))
        if column >= 0
    ]
    if not wrists or not ankles:
        return np.zeros(x.shape[0], dtype=bool)
    wrist_known, ankle_known = valid[:, wrists], valid[:, ankles]
    classifiable = wrist_known.any(axis=1) & ankle_known.any(axis=1)
    wrist_y = _mean_of_visible(y[:, wrists], wrist_known)
    ankle_y = _mean_of_visible(y[:, ankles], ankle_known)
    return classifiable & (wrist_y > ankle_y)


def _mean_of_visible(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """The mean of each frame's visible values; 0.0 for a frame with none.

    The invisible entries are summed as zero over a count that does not include
    them, so no all-NaN slice is ever taken.
    """
    return np.where(known, values, 0.0).sum(axis=1) / np.maximum(known.sum(axis=1), 1)


def median_jitter_l(
    t_seconds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    body_length: float,
) -> float:
    """Median frame-to-frame displacement in body lengths, over valid neighbours.

    Jitter, as opposed to movement: the *median* of the small displacements, which
    a real movement only inflates because most frames are not the moving ones.
    Samples must be valid in both frames, so the raw and the processed track are
    measured over exactly the same pairs of frames and the two numbers are
    comparable. NaN when the clip has no two neighbouring valid samples to
    measure.
    """
    if x.ndim != 2 or y.shape != x.shape or valid.shape != x.shape:
        raise ValueError("x, y and valid must have the same (frames, joints) shape")
    length = float(body_length)
    if not math.isfinite(length) or length <= 0.0:
        return math.nan
    pairs = valid[:-1] & valid[1:]
    if not pairs.any():
        return math.nan
    steps = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))[pairs] / length
    return float(np.median(steps))


def _columns_of(clip: ClipKeypoints, joints: Sequence[str]) -> np.ndarray:
    """The columns of ``joints`` this clip actually has, in that order.

    A source that does not report a joint (Apple Vision has no ``foot_index``)
    simply has fewer columns here; every measurement over these columns is
    skipped rather than padded.
    """
    return np.array(
        [column for column in (joint_columns(clip.joints, name) for name in joints) if column >= 0],
        dtype=int,
    )


@dataclasses.dataclass(frozen=True)
class ClipStats:
    """What one clip's sidecar records and the run's summary prints.

    Counts are kept as counts and percentages as properties, so the sidecar is a
    straight rendering of the dataclass and a reader can check one against the
    other by eye. The counts partition every sample of the clip three ways —

        total == measured + filled + unfilled

    with ``valid == measured + filled``, and the loss broken down by the rule
    that caused it: ``unfilled == gated_out + outliers``. A sample a rule removed
    is only counted as lost if the gap fill did not bridge it afterwards, so a
    short occlusion appears as a bridge rather than as a loss.
    """

    clip_id: str
    source: str
    rotate: str
    frame_count: int
    joint_count: int
    usable: bool
    unusable_reason: str
    body_length: float
    body_length_parts: dict[str, float]
    body_length_frames: dict[str, int]
    valid_samples: int
    total_samples: int
    tracked_valid_samples: int
    tracked_total_samples: int
    hold_like_frames: int
    hold_like_valid_samples: int
    hold_like_total_samples: int
    gated_out_samples: int
    outlier_samples: int
    unfilled_samples: int
    filled_samples: int
    jitter_raw_l: float
    jitter_processed_l: float
    runtime_seconds: float

    @staticmethod
    def _percent(part: int, total: int) -> float:
        return 0.0 if total <= 0 else 100.0 * part / total

    @property
    def measured_samples(self) -> int:
        """Samples that carry the model's own position rather than a bridge."""
        return self.valid_samples - self.filled_samples

    @property
    def valid_percent(self) -> float:
        """Share of all samples with a position."""
        return self._percent(self.valid_samples, self.total_samples)

    @property
    def tracked_valid_percent(self) -> float:
        """Share of the tracked joints' samples with a position."""
        return self._percent(self.tracked_valid_samples, self.tracked_total_samples)

    @property
    def hold_like_valid_percent(self) -> float:
        """Share of the tracked samples *in hold-like frames* that have a position."""
        return self._percent(self.hold_like_valid_samples, self.hold_like_total_samples)

    @property
    def jitter_reduction(self) -> float:
        """Jitter removed, as a fraction: ``0.7`` is 70 % of the raw jitter gone."""
        if not (math.isfinite(self.jitter_raw_l) and math.isfinite(self.jitter_processed_l)):
            return math.nan
        if self.jitter_raw_l <= 0.0:
            return math.nan
        return 1.0 - self.jitter_processed_l / self.jitter_raw_l

    def sidecar(self, source_parquet: str) -> dict[str, object]:
        """The JSON sidecar: provenance, the scale, the counts and the thresholds."""
        return {
            "clip_id": self.clip_id,
            "source": self.source,
            "rotate": self.rotate,
            "source_parquet": source_parquet,
            "frame_count": self.frame_count,
            "joint_count": self.joint_count,
            "usable": self.usable,
            "unusable_reason": self.unusable_reason or None,
            "body_length_px": None if math.isnan(self.body_length) else round(self.body_length, 3),
            "body_length_parts_px": {
                name: (None if math.isnan(value) else round(value, 3))
                for name, value in self.body_length_parts.items()
            },
            "body_length_frames": dict(self.body_length_frames),
            "valid_sample_count": self.valid_samples,
            "measured_sample_count": self.measured_samples,
            "total_sample_count": self.total_samples,
            "pct_valid_samples": round(self.valid_percent, 2),
            "tracked_valid_sample_count": self.tracked_valid_samples,
            "tracked_total_sample_count": self.tracked_total_samples,
            "pct_valid_tracked_samples": round(self.tracked_valid_percent, 2),
            "hold_like_frame_count": self.hold_like_frames,
            "pct_valid_tracked_samples_hold_like": round(self.hold_like_valid_percent, 2),
            "gated_out_sample_count": self.gated_out_samples,
            "outlier_sample_count": self.outlier_samples,
            "unfilled_sample_count": self.unfilled_samples,
            "filled_sample_count": self.filled_samples,
            "jitter_l_raw": _rounded(self.jitter_raw_l),
            "jitter_l_processed": _rounded(self.jitter_processed_l),
            "pct_jitter_reduction": _rounded(self.jitter_reduction),
            "parameters": {
                "min_visibility": MIN_VISIBILITY,
                "body_length_percentile": BODY_LENGTH_PERCENTILE,
                "max_speed_l_per_s": MAX_SPEED_L_PER_S,
                "max_gap_s": MAX_GAP_S,
                "one_euro": {"min_cutoff": MIN_CUTOFF, "beta": BETA, "d_cutoff": D_CUTOFF},
            },
            "runtime_seconds": round(self.runtime_seconds, 3),
        }

    def line(self) -> str:
        """The one progress line the CLI prints for this clip."""
        scale = "unusable" if not self.usable else f"L={self.body_length:.0f}px"
        return (
            f"clip  {self.clip_id} rotate={self.rotate} frames={self.frame_count} "
            f"joints={self.joint_count} {scale} valid={self.tracked_valid_percent:.1f}% "
            f"all={self.valid_percent:.1f}% "
            f"hold={self.hold_like_valid_percent:.1f}% ({self.hold_like_frames} frames) "
            f"gated_out={self.gated_out_samples} outliers={self.outlier_samples} "
            f"unfilled={self.unfilled_samples} filled={self.filled_samples} "
            f"jitter={self.jitter_raw_l:.4f}->{self.jitter_processed_l:.4f} L "
            f"runtime={self.runtime_seconds:.2f}s"
        )


def _rounded(value: float, digits: int = 5) -> float | None:
    """``value`` rounded, or ``None`` for a NaN — JSON has no NaN."""
    return None if not math.isfinite(value) else round(value, digits)


def measure(
    clip_id: str,
    source: str,
    rotate: str,
    processed: ProcessedClip,
    runtime_seconds: float,
) -> ClipStats:
    """Everything one clip's sidecar records, measured off the processed clip."""
    clip, body = processed.clip, processed.body_length
    tracked = _columns_of(clip, TRACKED_JOINTS)
    jitter = _columns_of(clip, HOLD_LIKE_JOINTS)
    measured = processed.valid & ~processed.filled
    hold = processed.hold_like
    # A sample a rule removed is only *lost* if the gap fill did not bridge it
    # afterwards, so the four counts below are disjoint and add up to the clip.
    lost_to_gate = ~processed.gated & ~processed.filled
    lost_to_speed = processed.gated & ~processed.outliers & ~processed.filled
    # The hold-like share is over the tracked joints only: a face or a finger is
    # not evidence about a hold, and the foot indexes are NaN on a source that
    # does not report them.
    hold_rows = np.repeat(hold[:, None], tracked.size, axis=1) if tracked.size else None
    return ClipStats(
        clip_id=clip_id,
        source=source,
        rotate=rotate,
        frame_count=clip.frames,
        joint_count=clip.joint_count,
        usable=body.usable,
        unusable_reason=body.reason,
        body_length=body.total,
        body_length_parts=body.parts(),
        body_length_frames=body.frames(),
        valid_samples=int(processed.valid.sum()),
        total_samples=clip.frames * clip.joint_count,
        tracked_valid_samples=int(processed.valid[:, tracked].sum()) if tracked.size else 0,
        tracked_total_samples=int(tracked.size * clip.frames),
        hold_like_frames=int(hold.sum()),
        hold_like_valid_samples=int((measured[:, tracked] & hold_rows).sum())
        if hold_rows is not None
        else 0,
        hold_like_total_samples=int(hold.sum()) * int(tracked.size),
        gated_out_samples=int(lost_to_gate.sum()),
        outlier_samples=int(lost_to_speed.sum()),
        unfilled_samples=int(processed.unfilled.sum()),
        filled_samples=int(processed.filled_samples()),
        jitter_raw_l=median_jitter_l(
            clip.t_seconds,
            clip.x[:, jitter],
            clip.y[:, jitter],
            measured[:, jitter],
            body.total,
        )
        if jitter.size
        else math.nan,
        jitter_processed_l=median_jitter_l(
            clip.t_seconds,
            processed.x[:, jitter],
            processed.y[:, jitter],
            measured[:, jitter],
            body.total,
        )
        if jitter.size
        else math.nan,
        runtime_seconds=runtime_seconds,
    )


def summary(reports: Sequence[ClipStats]) -> str:
    """The dataset-level answer, printed at the end of a run.

    The four questions a later stage needs answered before it can trust this
    output: how many clips are there and how many have a scale at all, how much of
    the keypoint data survived, how much of *that* survived in the frames the
    later stages care about (holds, which is where a phase detector looks), how
    much jitter the smoothing removed, and what the scale itself looks like
    across the dataset — the last one is the check that ``L`` is a body and not
    an accident of one clip's distance from the camera.
    """
    usable = [report for report in reports if report.usable]
    lines = [
        f"summary clips={len(reports)} usable={len(usable)} unusable={len(reports) - len(usable)}"
    ]
    if not reports:
        return "\n".join(lines)
    total = sum(report.total_samples for report in reports)
    valid = sum(report.valid_samples for report in reports)
    tracked_total = sum(report.tracked_total_samples for report in reports)
    tracked_valid = sum(report.tracked_valid_samples for report in reports)
    hold_total = sum(report.hold_like_total_samples for report in reports)
    hold_valid = sum(report.hold_like_valid_samples for report in reports)
    lines.append(
        f"  valid samples   {valid}/{total} ({_pct(valid, total)}) all joints, "
        f"{tracked_valid}/{tracked_total} ({_pct(tracked_valid, tracked_total)}) tracked"
    )
    lines.append(
        f"  valid in holds  {hold_valid}/{hold_total} ({_pct(hold_valid, hold_total)}) "
        f"tracked samples in {sum(r.hold_like_frames for r in reports)} hold-like frames"
    )
    lines.append(
        f"  dropped         gated_out={sum(r.gated_out_samples for r in reports)} "
        f"outliers={sum(r.outlier_samples for r in reports)} "
        f"unfilled={sum(r.unfilled_samples for r in reports)} "
        f"filled={sum(r.filled_samples for r in reports)}"
    )
    raw = [report.jitter_raw_l for report in reports if math.isfinite(report.jitter_raw_l)]
    smooth = [
        report.jitter_processed_l for report in reports if math.isfinite(report.jitter_processed_l)
    ]
    raw_median = float(np.median(raw)) if raw else math.nan
    smooth_median = float(np.median(smooth)) if smooth else math.nan
    if math.isfinite(raw_median) and math.isfinite(smooth_median):
        # A clip whose keypoints do not move at all has a jitter of zero, and no
        # percentage of that is a fraction — the two medians are printed anyway.
        less = (
            f" ({100.0 * (1.0 - smooth_median / raw_median):.0f}% less)"
            if raw_median > 0.0
            else " (n/a: the raw jitter is zero)"
        )
        lines.append(
            f"  jitter (L)      raw median {raw_median:.4f} -> processed {smooth_median:.4f}{less}"
        )
    lengths = np.array([report.body_length for report in usable], dtype=np.float64)
    if lengths.size:
        lines.append(
            f"  body length (px) min={lengths.min():.0f} p25={np.percentile(lengths, 25):.0f} "
            f"median={np.median(lengths):.0f} p75={np.percentile(lengths, 75):.0f} "
            f"max={lengths.max():.0f}"
        )
    unusable = [report for report in reports if not report.usable]
    if unusable:
        lines.append(
            f"  unusable clips  {', '.join(f'{r.clip_id} ({r.unusable_reason})' for r in unusable)}"
        )
    return "\n".join(lines)


def _pct(part: int, total: int) -> str:
    return f"{100.0 * part / total:.1f} %" if total else "n/a"


# --------------------------------------------------------------------------- #
# Locations, writing, one clip
# --------------------------------------------------------------------------- #


def input_dirname(source: str = DEFAULT_SOURCE) -> str:
    """Which directory under ``<data_dir>/keypoints/`` a ``--source`` value reads."""
    return athlete.output_dirname(source)


def output_dir(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pathlib.Path:
    """``<data_dir>/processed/<source>``, where that source's processed clips go."""
    return pathlib.Path(data) / PROCESSED_DIRNAME / source


def missing_input_message(rotate_mode: str, path: str | pathlib.Path, source: str) -> str:
    """The error shown when the athlete keypoints have not been generated yet.

    The command named is the one that produces them: the athlete selection for
    this source, in this rotation mode.
    """
    command = "uv run python -m handstand.athlete"
    if source != athlete.DEFAULT_SOURCE:
        command += f" --source {source}"
    return (
        f"no athlete keypoints for rotate mode {rotate_mode!r}: {path}\n"
        "generate them first with:\n"
        f"  cd pipeline && {command} --rotate {rotate_mode}"
    )


def available_clips(
    rotate_mode: str,
    in_root: str | pathlib.Path,
    clips: Sequence[str] | None = None,
    source: str = DEFAULT_SOURCE,
) -> list[str]:
    """The clip ids to process: ``clips`` as given, else every parquet found.

    Raises :class:`FileNotFoundError` naming the command that writes the input when
    there is no directory for this rotation mode at all.
    """
    if rotate_mode not in ROTATE_MODES:
        raise ValueError(f"unknown rotate mode {rotate_mode!r}; expected one of {ROTATE_MODES}")
    root = pathlib.Path(in_root) / rotate_mode
    if clips:
        return [str(clip) for clip in clips]
    if not root.is_dir():
        raise FileNotFoundError(missing_input_message(rotate_mode, root, source))
    return sorted(path.stem for path in root.glob("*.parquet"))


def clip_table(clip_id: str, rotate_mode: str, source: str) -> str:
    """The path of a clip's input, relative to ``<data_dir>``, for the sidecar."""
    return f"keypoints/{input_dirname(source)}/{rotate_mode}/{clip_id}.parquet"


@dataclasses.dataclass(frozen=True)
class ClipReport:
    """What one clip's run produced."""

    clip_id: str
    rotate: str
    stats: ClipStats | None
    parquet_path: pathlib.Path
    json_path: pathlib.Path
    skipped: bool = False

    def line(self) -> str:
        """The one progress line the CLI prints for this clip."""
        if self.skipped or self.stats is None:
            return f"skip  {self.clip_id} ({self.parquet_path} exists)"
        return self.stats.line()


def run_clip(
    clip_id: str,
    *,
    rotate_mode: str,
    in_root: str | pathlib.Path,
    out_root: str | pathlib.Path,
    overwrite: bool = False,
    source: str = DEFAULT_SOURCE,
) -> ClipReport:
    """Post-process one clip and write its parquet + JSON.

    Reads ``<in_root>/<rotate_mode>/<clip_id>.parquet`` — the athlete selection's
    output, in the shared schema — and writes
    ``<out_root>/<clip_id>.{parquet,json}``. A clip whose parquet is already there
    is skipped unless ``overwrite``, which is what makes a batch over 180 clips
    resumable.

    Both files are written atomically and the parquet is renamed **last**: its
    existence is what marks the clip as done, so an interrupted run never leaves
    a half-written trajectory behind for a later stage to read.
    """
    if rotate_mode not in ROTATE_MODES:
        raise ValueError(f"unknown rotate mode {rotate_mode!r}; expected one of {ROTATE_MODES}")
    athlete.output_dirname(source)  # rejects an unknown source before any file is touched
    in_path = pathlib.Path(in_root) / rotate_mode / f"{clip_id}.parquet"
    if not in_path.is_file():
        raise FileNotFoundError(
            missing_input_message(rotate_mode, pathlib.Path(in_root) / rotate_mode, source)
        )
    out_dir = pathlib.Path(out_root)
    parquet_path = out_dir / f"{clip_id}.parquet"
    json_path = out_dir / f"{clip_id}.json"
    if parquet_path.exists() and not overwrite:
        return ClipReport(
            clip_id=clip_id,
            rotate=rotate_mode,
            stats=None,
            parquet_path=parquet_path,
            json_path=json_path,
            skipped=True,
        )

    started = time.perf_counter()
    clip = read_clip(in_path)
    processed = process_clip(clip)
    table = processed_table(processed)
    runtime_seconds = time.perf_counter() - started
    stats = measure(clip_id, source, rotate_mode, processed, runtime_seconds)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_name(f".{json_path.name}.tmp")
    json_tmp.write_text(
        json.dumps(
            stats.sidecar(clip_table(clip_id, rotate_mode, source)), indent=2, sort_keys=True
        )
        + "\n"
    )
    json_tmp.replace(json_path)
    parquet_tmp = parquet_path.with_name(f".{parquet_path.name}.tmp")
    try:
        table.to_parquet(parquet_tmp, index=False)
        parquet_tmp.replace(parquet_path)
    finally:
        parquet_tmp.unlink(missing_ok=True)

    return ClipReport(
        clip_id=clip_id,
        rotate=rotate_mode,
        stats=stats,
        parquet_path=parquet_path,
        json_path=json_path,
        skipped=False,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The argument parser of ``python -m handstand.postprocess``."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.postprocess",
        description=(
            "Gate, de-spike, gap-fill and One-Euro smooth the athlete keypoints, "
            "and measure each clip's body length."
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--clips",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help="clip ids to process",
    )
    selection.add_argument(
        "--all",
        dest="all_clips",
        action="store_true",
        help="process every clip with athlete keypoints",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help=(
            "which pose model's athlete keypoints to read: "
            f"mediapipe is keypoints/{athlete.ATHLETE_OUTPUT_DIRNAME}, vision is "
            f"keypoints/{athlete.VISION_ATHLETE_OUTPUT_DIRNAME} "
            f"(default: {DEFAULT_SOURCE})"
        ),
    )
    parser.add_argument(
        "--rotate",
        choices=ROTATE_MODES,
        default=DEFAULT_ROTATE,
        help=(
            f"rotation mode of the keypoints to read (default: {DEFAULT_ROTATE}). The "
            "output path has no mode in it, so re-running a clip in another mode "
            "needs --overwrite"
        ),
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory holding keypoints/ (default: $HANDSTAND_DATA)",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batch and print the dataset-level summary. See the module docstring."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    root = pathlib.Path(args.data) if args.data is not None else data_dir()
    in_root = root / "keypoints" / input_dirname(args.source)
    out_root = output_dir(root, args.source)

    try:
        clip_ids = available_clips(
            args.rotate, in_root, None if args.all_clips else args.clips, args.source
        )
    except FileNotFoundError as error:
        print(f"postprocess: {error}")
        return 2
    if args.limit is not None:
        clip_ids = clip_ids[: args.limit]
    if not clip_ids:
        print("no clips to process")
        return 0

    print(
        f"source={args.source} rotate={args.rotate} clips={len(clip_ids)} "
        f"in={in_root} out={out_root}"
    )
    reports: list[ClipStats] = []
    failures = 0
    for clip_id in clip_ids:
        try:
            report = run_clip(
                clip_id,
                rotate_mode=args.rotate,
                in_root=in_root,
                out_root=out_root,
                overwrite=args.overwrite,
                source=args.source,
            )
        except Exception as error:  # one bad clip must not kill the whole batch
            failures += 1
            print(f"fail  {clip_id}: {error}")
            continue
        print(report.line(), flush=True)
        if report.stats is not None:
            reports.append(report.stats)
    print(summary(reports))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
