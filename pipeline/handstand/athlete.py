"""Pick the athlete out of the people MediaPipe reports, and flag trainer contact.

A multi-person run of the pose runner
(:mod:`handstand.pose_mediapipe --num-poses 3 --running-mode image
--min-detection 0.2 --min-presence 0.2`) writes one 33-row block per person per
frame::

    <data_dir>/keypoints/mediapipe_multi/<rotate>/<clip_id>.parquet

This module turns that into the **single-person schema** again, holding the
athlete's body only, so every later stage keeps working unchanged::

    <data_dir>/keypoints/mediapipe_athlete/<rotate>/<clip_id>.parquet
    <data_dir>/keypoints/mediapipe_athlete/<rotate>/<clip_id>.json

and adds three frame-level columns to it:

* ``athlete_score`` (float64) — the winner's score, 0..1; NaN when the frame is
  attributed to nobody.
* ``n_people`` (int64) — how many people were in the frame **after** dedup.
* ``trainer_contact`` (bool) — the trainer overlaps or touches the athlete: keep
  the keypoints, but do not score the frame.

The rules, in the order they are applied to a frame:

**1. Deduplicate** — two detections of the same body are one person. Two
detections are the same body when their hip midpoints are within
:data:`DEDUP_HIP_TOLERANCE` body lengths *and* their joint sets overlap (the
median joint-to-joint distance is under :data:`DEDUP_JOINT_TOLERANCE` body
lengths, over at least :data:`DEDUP_MIN_SHARED_JOINTS` shared joints). The one
with the higher mean visibility survives.

**2. Score and choose** — every remaining person is scored 0..1 from four
weighted terms, each of them 0..1:

* inversion, weight :data:`WEIGHT_INVERSION` — mean wrist y greater than mean
  ankle y, i.e. on their hands.
* support, weight :data:`WEIGHT_SUPPORT` — wrists near the lowest wrist/foot
  line of the frame, in this person's own shoulder-to-ankle lengths.
* continuity, weight :data:`WEIGHT_CONTINUITY` — hip midpoint close to the
  previously chosen athlete, in this person's own body lengths.
* visibility, weight :data:`WEIGHT_VISIBILITY` — mean visibility of the 12 main
  joints.

The weights add up to 1, so :data:`MIN_ATHLETE_SCORE` is directly comparable
with a probability-like number. The whole clip is swept **forward and backward**,
each sweep measuring continuity against the athlete *that* sweep chose in its
previous frame, and the higher-scoring sweep wins per frame. That is what covers
the kick-up before the first inverted frame, where nothing looks like a handstand
and only the frames after it say who the athlete is.

**3. Flag contact** — the chosen person is flagged ``trainer_contact`` when
another person's bounding box overlaps theirs by more than
:data:`CONTACT_MIN_IOU`, or when one of the athlete's limb joints sits closer to
the other person's joints than to the athlete's own centre line (their feet to
their head, which every limb joint is a hand's width away from): the two
skeletons overlap so much that the model stitched them into one. Such a frame is
written with the athlete's keypoints but must not be scored.

**4. Give up honestly** — a frame whose best score is below
:data:`MIN_ATHLETE_SCORE`, or where the two best candidates are within
:data:`AMBIGUITY_MARGIN` of each other, is written as *not detected* (NaN
coordinates, ``detected = false``) instead of guessing.

Body lengths are the person's own shoulder-midpoint to ankle-midpoint span in
display pixels (falling back to the diagonal of the usable joints when shoulders
or ankles are missing), so a long and a short body are judged on the same terms.
Joints count as visible at MediaPipe visibility
:data:`CONTACT_MIN_VISIBILITY` and above.

CLI::

    cd pipeline
    uv run python -m handstand.athlete --rotate auto
    uv run python -m handstand.athlete --rotate auto --clips 6508f9b355bd
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import time
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from handstand.paths import data_dir
from handstand.pose_mediapipe import (
    JOINT_INDEX,
    JOINT_NAMES,
    PARQUET_COLUMNS,
    PERSON_COLUMN,
    ROTATE_MODES,
    people_histogram,
)

__all__ = [
    "AMBIGUITY_MARGIN",
    "ANKLE_IDX",
    "ATHLETE_COLUMNS",
    "ATHLETE_OUTPUT_DIRNAME",
    "CONTACT_COLUMN",
    "CONTACT_MIN_IOU",
    "CONTACT_MIN_VISIBILITY",
    "DEDUP_HIP_TOLERANCE",
    "DEDUP_JOINT_TOLERANCE",
    "DEDUP_MIN_SHARED_JOINTS",
    "DEFAULT_ROTATE",
    "FEET_JOINTS",
    "FEET_JOINT_IDX",
    "HEAD_JOINTS",
    "HEAD_JOINT_IDX",
    "HIP_IDX",
    "LANDMARK_FIELDS",
    "LIMB_JOINTS",
    "LIMB_JOINT_IDX",
    "MAIN_JOINTS",
    "MAIN_JOINT_IDX",
    "MIN_ATHLETE_SCORE",
    "MIN_BODY_LENGTH_PIXELS",
    "MULTI_OUTPUT_DIRNAME",
    "N_PEOPLE_COLUMN",
    "SCORE_COLUMN",
    "SHOULDER_IDX",
    "SUPPORT_JOINTS",
    "SUPPORT_JOINT_IDX",
    "WEIGHT_CONTINUITY",
    "WEIGHT_INVERSION",
    "WEIGHT_SUPPORT",
    "WEIGHT_VISIBILITY",
    "WRIST_IDX",
    "ClipReport",
    "FrameChoice",
    "FramePeople",
    "Person",
    "athlete_table",
    "available_clips",
    "body_centre_line",
    "body_length",
    "box_iou",
    "bounding_box",
    "build_arg_parser",
    "choose_person",
    "continuity_score",
    "deduplicate",
    "input_dirname",
    "is_inverted",
    "load_frames",
    "main",
    "max_y",
    "measure_person",
    "mean_visibility",
    "mean_y",
    "midpoint",
    "missing_input_message",
    "mixed_skeleton",
    "nearest_joint_distance",
    "output_dirname",
    "person_score",
    "prepare_frame",
    "run_clip",
    "same_body",
    "select_athlete",
    "support_line_y",
    "support_score",
    "sweep",
    "trainer_contact",
    "visible_mask",
    "visible_midpoint",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Input root under ``<data_dir>/keypoints/``: the multi-person runner's output.
MULTI_OUTPUT_DIRNAME = "mediapipe_multi"
#: Output root under ``<data_dir>/keypoints/``. Its own directory, because the
#: single-person schema is what later stages read from ``mediapipe/`` and those
#: files must stay exactly as the plain runner wrote them.
ATHLETE_OUTPUT_DIRNAME = "mediapipe_athlete"

#: Frame-level columns this module adds to the single-person schema.
SCORE_COLUMN = "athlete_score"
N_PEOPLE_COLUMN = "n_people"
CONTACT_COLUMN = "trainer_contact"
#: :data:`handstand.pose_mediapipe.PARQUET_COLUMNS` plus the three above, in
#: write order. Every frame contributes 33 rows, exactly as the runner writes.
ATHLETE_COLUMNS: tuple[str, ...] = (
    *PARQUET_COLUMNS,
    SCORE_COLUMN,
    N_PEOPLE_COLUMN,
    CONTACT_COLUMN,
)

#: Score weights; they add up to 1, so a score is 0..1.
WEIGHT_INVERSION = 0.4
WEIGHT_SUPPORT = 0.2
WEIGHT_CONTINUITY = 0.3
WEIGHT_VISIBILITY = 0.1

#: Below this the winner is not convincing: the frame is written as not detected.
MIN_ATHLETE_SCORE = 0.35
#: Two candidates within this of each other are indistinguishable, so the frame
#: is written as not detected rather than resolved by a coin flip.
AMBIGUITY_MARGIN = 0.05

#: Two detections whose hip midpoints are this close (in body lengths) may be the
#: same body — but only if their joints also line up.
DEDUP_HIP_TOLERANCE = 0.15
#: ... and the median joint-to-joint distance under this (in body lengths) is
#: what makes the joint sets "overlap".
DEDUP_JOINT_TOLERANCE = 0.1
#: Fewer shared usable joints than this cannot tell two bodies apart at all.
DEDUP_MIN_SHARED_JOINTS = 4

#: Bounding boxes (from the visible joints) overlapping by more than this is
#: trainer contact.
CONTACT_MIN_IOU = 0.3
#: MediaPipe visibility at or above this counts as a visible joint: used for the
#: bounding boxes and for the "which joints can we trust" question.
CONTACT_MIN_VISIBILITY = 0.5

#: A body length is never below this many pixels, so normalising by it is safe.
MIN_BODY_LENGTH_PIXELS = 1.0

#: The 12 main joints: shoulders, elbows, wrists, hips, knees, ankles. Used for
#: the visibility term, so no face or finger score can carry a bad body.
MAIN_JOINTS: tuple[str, ...] = (
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
#: Joints that may be swapped for another body's when two people overlap: the
#: ends of the limbs, i.e. everything a bone hangs off except the trunk.
LIMB_JOINTS: tuple[str, ...] = (
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
#: Wrists and feet, whichever is lowest: where a handstand's hands and a
#: standing person's feet meet the floor.
SUPPORT_JOINTS: tuple[str, ...] = (
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
#: The ends of a person's centre line: the feet below, the head above. A
#: handstand's feet are up in the air, which does not matter — the line is the
#: body's own axis either way.
FEET_JOINTS: tuple[str, ...] = (
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
HEAD_JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
)


#: ``(x, y, z, visibility, presence)`` per landmark, as the runner writes them.
LANDMARK_FIELDS = 5

#: Rotation mode the CLI uses when ``--rotate`` is not given: the recommended one.
DEFAULT_ROTATE = "auto"


def _indices(joint_names: Sequence[str]) -> tuple[int, ...]:
    """Model row numbers of the named joints, in the order they were given."""
    return tuple(JOINT_INDEX[name] for name in joint_names)


SHOULDER_IDX = _indices(("left_shoulder", "right_shoulder"))
WRIST_IDX = _indices(("left_wrist", "right_wrist"))
HIP_IDX = _indices(("left_hip", "right_hip"))
ANKLE_IDX = _indices(("left_ankle", "right_ankle"))
MAIN_JOINT_IDX = _indices(MAIN_JOINTS)
LIMB_JOINT_IDX = _indices(LIMB_JOINTS)
SUPPORT_JOINT_IDX = _indices(SUPPORT_JOINTS)
FEET_JOINT_IDX = _indices(FEET_JOINTS)
HEAD_JOINT_IDX = _indices(HEAD_JOINTS)


# --------------------------------------------------------------------------- #
# Landmarks: the small measurements every rule is built from
# --------------------------------------------------------------------------- #


def _check_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Validate one ``(33, 5)`` landmark block and return it as float64."""
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape != (len(JOINT_NAMES), LANDMARK_FIELDS):
        raise ValueError(
            f"expected ({len(JOINT_NAMES)}, {LANDMARK_FIELDS}) landmarks, got {pts.shape}"
        )
    return pts


def midpoint(landmarks: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Mean position of the joints at ``indices``; NaN when none of them is usable.

    "Usable" means both coordinates are finite, so a half-reported joint cannot
    drag the midpoint off the body.
    """
    points = _check_landmarks(landmarks)[list(indices), :2]
    usable = points[np.all(np.isfinite(points), axis=1)]
    if not usable.size:
        return np.full(2, np.nan)
    return usable.mean(axis=0)


def visible_mask(
    landmarks: np.ndarray, min_visibility: float = CONTACT_MIN_VISIBILITY
) -> np.ndarray:
    """Which of the 33 joints are usable: finite coordinates and enough visibility.

    MediaPipe always reports all 33 landmarks, so a face or a foot the model did
    not really see still has a position — only its visibility score says so. This
    mask is what keeps such a joint out of the boxes, the centre line and the
    "which joints can we trust" questions.
    """
    pts = _check_landmarks(landmarks)
    scores = pts[:, 3]
    return (
        np.all(np.isfinite(pts[:, :2]), axis=1) & np.isfinite(scores) & (scores >= min_visibility)
    )


def visible_midpoint(landmarks: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Midpoint of the *visible* joints at ``indices``; NaN when none of them is.

    Used where a hallucinated joint would move the answer a long way (the ends
    of the centre line), as opposed to a joint that only has to be near a body.
    """
    pts = _check_landmarks(landmarks)
    mask = visible_mask(pts)
    points = pts[[index for index in indices if mask[index]], :2]
    if not points.size:
        return np.full(2, np.nan)
    return points.mean(axis=0)


def body_centre_line(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """This person's own centre line: the segment from their feet to their head.

    The body's axis, and it is the same line for a handstand and for a standing
    person, so every limb joint sits within a hand's width of it. A joint that
    is closer to somebody *else's* joints than to this line is not on this body
    any more — see :func:`mixed_skeleton`.

    Falls back to the ankles (feet not reported) and the shoulders (no visible
    head), and gives up with ``None`` when neither end can be placed, in which
    case the mixed-skeleton check simply does not apply.
    """
    feet = visible_midpoint(landmarks, FEET_JOINT_IDX)
    if not np.all(np.isfinite(feet)):
        feet = visible_midpoint(landmarks, ANKLE_IDX)
    head = visible_midpoint(landmarks, HEAD_JOINT_IDX)
    if not np.all(np.isfinite(head)):
        head = visible_midpoint(landmarks, SHOULDER_IDX)
    if np.all(np.isfinite(feet)) and np.all(np.isfinite(head)):
        return (feet, head)
    return None


def mean_y(landmarks: np.ndarray, indices: Sequence[int]) -> float:
    """Mean image y of the joints at ``indices``; NaN when none of them is usable."""
    points = _check_landmarks(landmarks)[list(indices), :2]
    usable = points[np.all(np.isfinite(points), axis=1)]
    if not usable.size:
        return float("nan")
    return float(usable[:, 1].mean())


def max_y(landmarks: np.ndarray, indices: Sequence[int]) -> float:
    """Lowest point (largest y) among the joints at ``indices``; NaN if none is."""
    points = _check_landmarks(landmarks)[list(indices), :2]
    usable = points[np.all(np.isfinite(points), axis=1)]
    if not usable.size:
        return float("nan")
    return float(usable[:, 1].max())


def mean_visibility(landmarks: np.ndarray, indices: Sequence[int]) -> float:
    """Mean MediaPipe visibility over the joints at ``indices``; NaN if unscored."""
    scores = _check_landmarks(landmarks)[list(indices), 3]
    usable = scores[np.isfinite(scores)]
    if not usable.size:
        return float("nan")
    return float(usable.mean())


def is_inverted(landmarks: np.ndarray) -> bool:
    """Is this person upside down — mean wrist y greater than mean ankle y?"""
    return mean_y(landmarks, WRIST_IDX) > mean_y(landmarks, ANKLE_IDX)


def body_length(landmarks: np.ndarray) -> float:
    """This body's length in pixels: the unit every distance in here is in.

    The shoulder-midpoint to ankle-midpoint span, which measures the same for a
    handstand and for a standing person. A body missing either falls back to the
    diagonal of the diagonal of its usable joints, and the result never drops
    below :data:`MIN_BODY_LENGTH_PIXELS`, so dividing by it stays finite.
    """
    pts = _check_landmarks(landmarks)
    shoulder = midpoint(pts, SHOULDER_IDX)
    ankle = midpoint(pts, ANKLE_IDX)
    length = _distance(shoulder, ankle)
    if not math.isfinite(length):
        usable = pts[np.all(np.isfinite(pts[:, :2]), axis=1), :2]
        if usable.size:
            span = usable.max(axis=0) - usable.min(axis=0)
            length = float(np.hypot(span[0], span[1]))
    if not math.isfinite(length) or length < MIN_BODY_LENGTH_PIXELS:
        return MIN_BODY_LENGTH_PIXELS
    return length


def bounding_box(
    landmarks: np.ndarray, min_visibility: float = CONTACT_MIN_VISIBILITY
) -> tuple[float, float, float, float] | None:
    """``(x0, y0, x1, y1)`` around the visible joints, or ``None`` if there are none.

    A joint counts as visible at ``min_visibility`` and above with finite
    coordinates (see :func:`visible_mask`), so a box never grows towards a limb
    the model did not see.
    """
    visible = _check_landmarks(landmarks)[:, :2][visible_mask(landmarks)]
    if not visible.size:
        return None
    return (
        float(visible[:, 0].min()),
        float(visible[:, 1].min()),
        float(visible[:, 0].max()),
        float(visible[:, 1].max()),
    )


def box_iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes; 0 when disjoint."""
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 < x0 or y1 < y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    """Euclidean distance between two points; NaN if either is not finite."""
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    if not np.all(np.isfinite(delta)):
        return float("nan")
    return float(np.hypot(delta[0], delta[1]))


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Distance from ``point`` to the segment ``start``-``end``."""
    span = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length_squared = float(span @ span)
    offset = np.asarray(point, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    if length_squared <= 0.0:
        return _distance(point, start)
    along = float(np.clip((offset @ span) / length_squared, 0.0, 1.0))
    return _distance(point, np.asarray(start, dtype=np.float64) + along * span)


# --------------------------------------------------------------------------- #
# One person
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True, eq=False)
class Person:
    """One person's landmarks in one frame, plus the measurements derived from them.

    Every field is measured once here so the rules below are pure arithmetic.
    ``length`` is this person's own body length: the distances that decide
    identity, support and continuity are all in body lengths, so a tall athlete
    and a short trainer are judged on the same terms.
    """

    landmarks: np.ndarray
    inverted: bool
    wrist_y: float
    support_y: float
    hip: np.ndarray
    centre_line: tuple[np.ndarray, np.ndarray] | None
    length: float
    visibility: float
    box: tuple[float, float, float, float] | None
    visible_joints: tuple[int, ...]


def measure_person(landmarks: np.ndarray) -> Person:
    """Measure one ``(33, 5)`` landmark block: everything :class:`Person` needs."""
    pts = _check_landmarks(landmarks)
    return Person(
        landmarks=pts,
        inverted=is_inverted(pts),
        wrist_y=mean_y(pts, WRIST_IDX),
        support_y=max_y(pts, SUPPORT_JOINT_IDX),
        hip=midpoint(pts, HIP_IDX),
        centre_line=body_centre_line(pts),
        length=body_length(pts),
        visibility=mean_visibility(pts, MAIN_JOINT_IDX),
        box=bounding_box(pts),
        visible_joints=tuple(int(index) for index in np.flatnonzero(visible_mask(pts))),
    )


# --------------------------------------------------------------------------- #
# Rule 1: two detections of one body are one person
# --------------------------------------------------------------------------- #


def same_body(first: Person, second: Person) -> bool:
    """Are these two detections of the same body?

    Two tests, both in body lengths (the larger of the two, so a half-reported
    twin is judged by the body that was seen): the hip midpoints must be within
    :data:`DEDUP_HIP_TOLERANCE`, and the joint sets must overlap — the median
    joint-to-joint distance over at least :data:`DEDUP_MIN_SHARED_JOINTS`
    joints reported by both must be under :data:`DEDUP_JOINT_TOLERANCE`. A
    missing hip or too few shared joints means "cannot tell", i.e. not the same.
    """
    if not (np.all(np.isfinite(first.hip)) and np.all(np.isfinite(second.hip))):
        return False
    scale = max(first.length, second.length)
    if _distance(first.hip, second.hip) > DEDUP_HIP_TOLERANCE * scale:
        return False
    shared = [index for index in first.visible_joints if index in second.visible_joints]
    if len(shared) < DEDUP_MIN_SHARED_JOINTS:
        return False
    distances = sorted(
        _distance(first.landmarks[index, :2], second.landmarks[index, :2]) for index in shared
    )
    middle = len(distances) // 2
    median = (
        distances[middle]
        if len(distances) % 2
        else (distances[middle - 1] + distances[middle]) / 2.0
    )
    return bool(median < DEDUP_JOINT_TOLERANCE * scale)


def deduplicate(people: Sequence[Person]) -> list[Person]:
    """Collapse duplicate detections of one body; the better-visible one survives.

    MediaPipe happily reports the same person twice when it is confident about
    two overlapping skeletons, and a doubled body would be scored (and judged in
    contact with) as if the trainer were standing on the athlete.
    """
    kept: list[Person] = []
    for person in people:
        twin = next((other for other in kept if same_body(other, person)), None)
        if twin is None:
            kept.append(person)
        elif person.visibility > twin.visibility:
            kept[kept.index(twin)] = person
    return kept


# --------------------------------------------------------------------------- #
# Rule 2: score the people, follow one through the clip
# --------------------------------------------------------------------------- #


def support_line_y(people: Sequence[Person]) -> float:
    """The lowest wrist or foot point of anyone in the frame; NaN if nobody has one.

    On a mat this is the floor: a handstand's hands and a standing trainer's
    feet both sit on it, and "close to this line" is what tells the athlete on
    their hands from somebody standing next to them.
    """
    ys = [person.support_y for person in people if math.isfinite(person.support_y)]
    return max(ys) if ys else float("nan")


def support_score(person: Person, support_line: float) -> float:
    """How close this person's hands are to the lowest hand/foot line: 1..0.

    Full score with the wrists on that line, nothing at one body length above
    it, and 0 when either measurement is missing.
    """
    if not (math.isfinite(person.wrist_y) and math.isfinite(support_line)):
        return 0.0
    return float(np.clip(1.0 - (support_line - person.wrist_y) / person.length, 0.0, 1.0))


def continuity_score(person: Person, previous_hip: np.ndarray | None) -> float:
    """How likely this person is the athlete followed so far: 1..0.

    Full score with the hips on the previously chosen athlete's hips, nothing at
    one body length away. Without a previously chosen athlete there is no
    history to continue, so everybody scores 1.0 and the other terms decide —
    the first frame of a sweep must not favour anyone.
    """
    if previous_hip is None or not np.all(np.isfinite(previous_hip)):
        return 1.0
    distance = _distance(person.hip, previous_hip)
    if not math.isfinite(distance):
        return 1.0
    return float(np.clip(1.0 - distance / person.length, 0.0, 1.0))


def person_score(person: Person, support_line: float, previous_hip: np.ndarray | None) -> float:
    """The athlete score of one person in one frame: 0..1, higher is more likely.

    The four weighted terms, see the module docstring. A missing visibility
    scores 0 for its term rather than NaN-ing the whole score.
    """
    visibility = person.visibility if math.isfinite(person.visibility) else 0.0
    return (
        (WEIGHT_INVERSION if person.inverted else 0.0)
        + WEIGHT_SUPPORT * support_score(person, support_line)
        + WEIGHT_CONTINUITY * continuity_score(person, previous_hip)
        + WEIGHT_VISIBILITY * visibility
    )


def choose_person(
    people: Sequence[Person], support_line: float, previous_hip: np.ndarray | None
) -> tuple[Person | None, float]:
    """The best-scoring person of a frame, or ``(None, score)`` when unusable.

    Two rules turn "highest score" into "nobody": a score below
    :data:`MIN_ATHLETE_SCORE` is not convincing, and two candidates within
    :data:`AMBIGUITY_MARGIN` of each other cannot be told apart. Either way the
    frame is left unattributed and the caller keeps the previous athlete.
    """
    if not people:
        return None, float("nan")
    scores = [person_score(person, support_line, previous_hip) for person in people]
    # Sorting by (-score, order) keeps MediaPipe's order for an exact tie, so
    # the choice is deterministic.
    ranked = sorted(range(len(people)), key=lambda position: (-scores[position], position))
    best = ranked[0]
    score = scores[best]
    runner_up = scores[ranked[1]] if len(ranked) > 1 else -math.inf
    if score < MIN_ATHLETE_SCORE or score - runner_up <= AMBIGUITY_MARGIN:
        return None, score
    return people[best], score


# --------------------------------------------------------------------------- #
# Rule 3: is the trainer on top of the athlete?
# --------------------------------------------------------------------------- #


def nearest_joint_distance(point: np.ndarray, other: Person) -> float:
    """Distance from ``point`` to the nearest visible joint of ``other``."""
    distances = [_distance(point, other.landmarks[index, :2]) for index in other.visible_joints]
    usable = [distance for distance in distances if math.isfinite(distance)]
    return min(usable) if usable else float("nan")


def mixed_skeleton(athlete: Person, other: Person) -> bool:
    """Is one of the athlete's limb joints nearer the other body than its own?

    When the trainer stands directly behind the athlete the two skeletons
    overlap almost completely and the model can stitch them into one that
    carries the athlete's upper body and the trainer's leg. A limb joint closer
    to the *other* person's joints than to the athlete's own centre line — their
    feet to their head, which every limb joint is a hand's width away from — is
    that stitch, and no amount of choosing can repair the frame: it has to be
    dropped. A body whose centre line cannot be placed is not checked.
    """
    if athlete.centre_line is None:
        return False
    line_start, line_end = athlete.centre_line
    for index in LIMB_JOINT_IDX:
        if index not in athlete.visible_joints:
            continue
        point = athlete.landmarks[index, :2]
        own_distance = _segment_distance(point, line_start, line_end)
        other_distance = nearest_joint_distance(point, other)
        if math.isfinite(other_distance) and other_distance < own_distance:
            return True
    return False


def trainer_contact(athlete: Person, others: Sequence[Person]) -> bool:
    """Does any other person in this frame overlap or touch the athlete?

    Either their bounding boxes of visible joints overlap by more than
    :data:`CONTACT_MIN_IOU`, or the athlete's skeleton looks stitched to theirs
    (:func:`mixed_skeleton`). A frame flagged here keeps the athlete's keypoints
    but must not be scored downstream.
    """
    for other in others:
        if athlete.box is not None and other.box is not None:
            if box_iou(athlete.box, other.box) > CONTACT_MIN_IOU:
                return True
        if mixed_skeleton(athlete, other):
            return True
    return False


# --------------------------------------------------------------------------- #
# Rule 2 + 3 over a whole clip
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True, eq=False)
class FramePeople:
    """One frame of the input, measured and deduplicated."""

    frame_idx: int
    t_ms: int
    rotated: bool
    people: tuple[Person, ...]
    support_line: float

    @property
    def n_people(self) -> int:
        """How many people this frame holds after dedup."""
        return len(self.people)


@dataclasses.dataclass(frozen=True, eq=False)
class FrameChoice:
    """What the selection made of one frame."""

    frame_idx: int
    person: Person | None
    score: float
    n_people: int
    contact: bool = False

    @property
    def chosen(self) -> bool:
        """Was a person attributed to this frame?"""
        return self.person is not None


def prepare_frame(frame_idx: int, t_ms: int, rotated: bool, landmarks: np.ndarray) -> FramePeople:
    """Measure a frame's ``(P, 33, 5)`` people, drop duplicates, find the mat line."""
    poses = np.asarray(landmarks, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (len(JOINT_NAMES), LANDMARK_FIELDS):
        raise ValueError(
            f"expected (P, {len(JOINT_NAMES)}, {LANDMARK_FIELDS}) poses, got {poses.shape}"
        )
    people = deduplicate([measure_person(pose) for pose in poses])
    return FramePeople(
        frame_idx=frame_idx,
        t_ms=t_ms,
        rotated=rotated,
        people=tuple(people),
        support_line=support_line_y(people),
    )


def sweep(frames: Sequence[FramePeople], order: Iterable[int]) -> list[FrameChoice]:
    """One pass over the frames, scoring against the athlete *this* pass chose.

    ``order`` is the sweep direction: ``range(n)`` forward, ``reversed`` backward.
    Continuity is measured against the last chosen athlete of this pass, which
    survives frames that were dropped, so a short loss of detection does not
    restart the identity. The result is one choice per frame, in frame order.
    """
    choices: list[FrameChoice | None] = [None] * len(frames)
    previous_hip: np.ndarray | None = None
    for position in order:
        frame = frames[position]
        person, score = choose_person(frame.people, frame.support_line, previous_hip)
        choices[position] = FrameChoice(
            frame_idx=frame.frame_idx,
            person=person,
            score=score,
            n_people=frame.n_people,
        )
        if person is not None:
            previous_hip = person.hip
    return [choice for choice in choices if choice is not None]  # every position is filled


def _higher(front: FrameChoice, back: FrameChoice) -> FrameChoice:
    """The sweep that scored this frame higher.

    A frame only one of the two passes could attribute goes to that pass. When
    the two scores are exactly equal, the pass that attributed somebody wins:
    an equal score with one pass holding a person and the other holding none is
    not a tie, it is an attribution.
    """
    if math.isnan(front.score) and math.isnan(back.score):
        return front
    if math.isnan(front.score):
        return back
    if math.isnan(back.score):
        return front
    if front.score > back.score:
        return front
    if back.score > front.score:
        return back
    return back if back.chosen else front


def select_athlete(frames: Sequence[FramePeople]) -> list[FrameChoice]:
    """Attribute every frame of a clip to one person, or to nobody.

    A forward and a backward sweep each follow the athlete through the clip by
    continuity, and the better-scoring sweep wins per frame: only the backward
    one reaches the kick-up *before* the first inverted frame, where nothing
    looks like a handstand yet. The winner is then checked against everybody
    else in its frame, and a frame where the trainer is on top of the athlete
    comes back with ``contact = True``.
    """
    forward = sweep(frames, range(len(frames)))
    backward = sweep(frames, reversed(range(len(frames))))
    chosen: list[FrameChoice] = []
    # Both sweeps already return one choice per frame in frame order, so the two
    # passes line up position for position.
    for frame, front, back in zip(frames, forward, backward, strict=True):
        best = _higher(front, back)
        others = [person for person in frame.people if person is not best.person]
        contact = best.person is not None and trainer_contact(best.person, others)
        chosen.append(dataclasses.replace(best, contact=contact))
    return chosen


# --------------------------------------------------------------------------- #
# Reading the multi-person parquet
# --------------------------------------------------------------------------- #


def load_frames(parquet_path: str | pathlib.Path) -> list[FramePeople]:
    """Read one multi-person parquet into measured :class:`FramePeople`, in order.

    The runner writes rows ordered by ``frame_idx``, then ``person_idx``, then
    landmark, but the loader does not trust that: it groups by ``frame_idx`` and
    sorts every block into model order itself, raising if a block is not a
    complete set of the 33 landmarks. A frame with nobody in it is the runner's
    ``person_idx = -1`` placeholder block; that is not a person, so the frame
    simply has no people.
    """
    path = pathlib.Path(parquet_path)
    table = pd.read_parquet(path)
    required = (*PARQUET_COLUMNS, PERSON_COLUMN)
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(
            f"{path}: multi-person keypoints are missing column(s) {missing}; "
            "see docs/keypoint_schema.md"
        )

    person_indices = table[PERSON_COLUMN].to_numpy(dtype=np.int64)
    values = table[["x", "y", "z", "visibility", "presence"]].to_numpy(dtype=np.float64)
    joints = np.array([JOINT_INDEX.get(str(name), -1) for name in table["joint"]])
    model_order = np.arange(len(JOINT_NAMES))

    frames: list[FramePeople] = []
    grouped = table.groupby("frame_idx", sort=True).indices
    for frame_idx in sorted(grouped):
        rows = np.asarray(grouped[frame_idx], dtype=np.int64)
        people_rows = rows[person_indices[rows] >= 0]
        # person-major, then model order, so one reshape() gives (P, 33, 5).
        order = np.lexsort((joints[people_rows], person_indices[people_rows]))
        block = values[people_rows][order]
        if block.size == 0:
            poses = np.empty((0, len(JOINT_NAMES), LANDMARK_FIELDS), dtype=np.float64)
        else:
            if block.shape[0] % len(JOINT_NAMES):
                raise ValueError(
                    f"{path}: frame {frame_idx} has {block.shape[0]} rows, "
                    f"not a whole number of {len(JOINT_NAMES)}-landmark blocks"
                )
            poses = block.reshape(-1, len(JOINT_NAMES), LANDMARK_FIELDS)
            found = joints[people_rows][order].reshape(-1, len(JOINT_NAMES))
            if not np.array_equal(found, np.tile(model_order, (poses.shape[0], 1))):
                raise ValueError(
                    f"{path}: frame {frame_idx} does not hold the 33 landmarks "
                    "once per person; re-run the keypoint extraction"
                )
        first = rows[0]
        frames.append(
            prepare_frame(
                frame_idx=int(frame_idx),
                t_ms=int(table["t_ms"].to_numpy(dtype=np.int64)[first]),
                rotated=bool(table["rotated"].to_numpy(dtype=bool)[first]),
                landmarks=poses,
            )
        )
    if not frames:
        raise ValueError(f"{path}: no frames in the keypoint parquet")
    return frames


# --------------------------------------------------------------------------- #
# Writing the single-person schema
# --------------------------------------------------------------------------- #


def athlete_table(frames: Sequence[FramePeople], choices: Sequence[FrameChoice]) -> pd.DataFrame:
    """The single-person schema plus the three athlete columns, 33 rows per frame.

    A frame with a chosen athlete carries that person's landmarks; a frame
    without one keeps the 33 NaN rows with ``detected = false`` the runner
    writes for a frame it missed, so a consumer can filter on ``detected``
    instead of re-indexing. ``athlete_score`` is NaN exactly when nobody was
    chosen, ``n_people`` is the count after dedup and ``trainer_contact`` is
    false whenever there is no athlete to be in contact with. ``rotated`` and
    ``t_ms`` describe the frame and are taken from the input unchanged.
    """
    if len(frames) != len(choices):
        raise ValueError(f"got {len(frames)} frames but {len(choices)} choices")

    landmarks = np.full((len(frames), len(JOINT_NAMES), LANDMARK_FIELDS), np.nan)
    scores = np.full(len(frames), np.nan)
    detected = np.zeros(len(frames), dtype=bool)
    for index, choice in enumerate(choices):
        if choice.person is not None:
            landmarks[index] = choice.person.landmarks
            scores[index] = choice.score
            detected[index] = True

    joints = np.asarray(JOINT_NAMES)
    per_frame = len(JOINT_NAMES)
    return pd.DataFrame(
        {
            "frame_idx": np.repeat(
                np.asarray([frame.frame_idx for frame in frames], dtype=np.int64), per_frame
            ),
            "t_ms": np.repeat(
                np.asarray([frame.t_ms for frame in frames], dtype=np.int64), per_frame
            ),
            "joint": np.tile(joints, len(frames)),
            "x": landmarks[..., 0].reshape(-1),
            "y": landmarks[..., 1].reshape(-1),
            "z": landmarks[..., 2].reshape(-1),
            "visibility": landmarks[..., 3].reshape(-1),
            "presence": landmarks[..., 4].reshape(-1),
            "rotated": np.repeat(
                np.asarray([frame.rotated for frame in frames], dtype=bool), per_frame
            ),
            "detected": np.repeat(detected, per_frame),
            SCORE_COLUMN: np.repeat(scores, per_frame),
            N_PEOPLE_COLUMN: np.repeat(
                np.asarray([choice.n_people for choice in choices], dtype=np.int64), per_frame
            ),
            CONTACT_COLUMN: np.repeat(
                np.asarray([choice.contact for choice in choices], dtype=bool), per_frame
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Per-clip run
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipReport:
    """What the athlete selection made of one clip."""

    clip_id: str
    rotate: str
    frame_count: int
    athlete_frames: int
    contact_frames: int
    dropped_frames: int
    runtime_seconds: float
    parquet_path: pathlib.Path
    json_path: pathlib.Path
    frames_by_people: dict[int, int] = dataclasses.field(default_factory=dict)
    skipped: bool = False

    def _percent(self, count: int) -> float:
        """``count`` as a percentage of the clip's frames."""
        if not self.frame_count:
            return 0.0
        return 100.0 * count / self.frame_count

    @property
    def athlete_percent(self) -> float:
        """Percentage of frames attributed to somebody."""
        return self._percent(self.athlete_frames)

    @property
    def contact_percent(self) -> float:
        """Percentage of frames flagged as trainer contact."""
        return self._percent(self.contact_frames)

    @property
    def dropped_percent(self) -> float:
        """Percentage of frames written as not detected."""
        return self._percent(self.dropped_frames)

    @property
    def people_summary(self) -> str:
        """``"0:2 1:150 2:80"`` — frames per number of people, after dedup."""
        return " ".join(
            f"{people}:{self.frames_by_people[people]}" for people in sorted(self.frames_by_people)
        )


def output_dirname() -> str:
    """Which directory under ``<data_dir>/keypoints/`` this module writes to."""
    return ATHLETE_OUTPUT_DIRNAME


def input_dirname() -> str:
    """Which directory under ``<data_dir>/keypoints/`` this module reads from."""
    return MULTI_OUTPUT_DIRNAME


def missing_input_message(rotate_mode: str, path: str | pathlib.Path) -> str:
    """The error shown when the multi-person keypoints have not been generated yet."""
    return (
        f"no multi-person keypoints for rotate mode {rotate_mode!r}: {path}\n"
        "generate them first with:\n"
        "  cd pipeline && uv run python -m handstand.pose_mediapipe "
        f"--rotate {rotate_mode} --num-poses 3 --running-mode image "
        "--min-detection 0.2 --min-presence 0.2"
    )


def run_clip(
    clip_id: str,
    *,
    rotate_mode: str,
    in_root: str | pathlib.Path,
    out_root: str | pathlib.Path,
    overwrite: bool = False,
) -> ClipReport:
    """Select the athlete in one clip and write its parquet + JSON.

    Reads ``<in_root>/<rotate_mode>/<clip_id>.parquet`` (the multi-person
    runner's output) and writes ``<out_root>/<rotate_mode>/<clip_id>.{parquet,json}``.
    Clips whose parquet already exists are skipped unless ``overwrite`` is true.

    Both files are written atomically and the parquet is renamed last: its
    existence is what marks a clip as done, so an interrupted run must never
    leave a partial file behind.
    """
    if rotate_mode not in ROTATE_MODES:
        raise ValueError(f"unknown rotate mode {rotate_mode!r}; expected one of {ROTATE_MODES}")
    in_path = pathlib.Path(in_root) / rotate_mode / f"{clip_id}.parquet"
    if not in_path.is_file():
        raise FileNotFoundError(
            missing_input_message(rotate_mode, pathlib.Path(in_root) / rotate_mode)
        )
    out_dir = pathlib.Path(out_root) / rotate_mode
    parquet_path = out_dir / f"{clip_id}.parquet"
    json_path = out_dir / f"{clip_id}.json"
    if parquet_path.exists() and not overwrite:
        return ClipReport(
            clip_id=clip_id,
            rotate=rotate_mode,
            frame_count=0,
            athlete_frames=0,
            contact_frames=0,
            dropped_frames=0,
            runtime_seconds=0.0,
            parquet_path=parquet_path,
            json_path=json_path,
            skipped=True,
        )

    started = time.perf_counter()
    frames = load_frames(in_path)
    choices = select_athlete(frames)
    table = athlete_table(frames, choices)
    runtime_seconds = time.perf_counter() - started

    frames_by_people = people_histogram([frame.n_people for frame in frames])
    athlete_frames = sum(choice.chosen for choice in choices)
    contact_frames = sum(choice.contact for choice in choices)
    frame_count = len(frames)
    scores = [choice.score for choice in choices if math.isfinite(choice.score)]
    sidecar = {
        "clip_id": clip_id,
        "rotate": rotate_mode,
        "source_parquet": f"{input_dirname()}/{rotate_mode}/{clip_id}.parquet",
        "frame_count": frame_count,
        "athlete_frame_count": athlete_frames,
        "contact_frame_count": contact_frames,
        "dropped_frame_count": frame_count - athlete_frames,
        "frames_by_people": {str(people): count for people, count in frames_by_people.items()},
        "mean_athlete_score": round(sum(scores) / len(scores), 4) if scores else None,
        "runtime_seconds": round(runtime_seconds, 3),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
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
        rotate=rotate_mode,
        frame_count=frame_count,
        athlete_frames=athlete_frames,
        contact_frames=contact_frames,
        dropped_frames=frame_count - athlete_frames,
        runtime_seconds=runtime_seconds,
        parquet_path=parquet_path,
        json_path=json_path,
        frames_by_people=frames_by_people,
    )


def available_clips(
    rotate_mode: str,
    in_root: str | pathlib.Path,
    clips: Sequence[str] | None = None,
) -> list[str]:
    """The clip ids to process: ``clips`` as given, else every parquet found.

    Raises :class:`FileNotFoundError` naming the command that produces the input
    when there is no multi-person directory for this rotation mode at all.
    """
    root = pathlib.Path(in_root) / rotate_mode
    if clips:
        return [str(clip) for clip in clips]
    if not root.is_dir():
        raise FileNotFoundError(missing_input_message(rotate_mode, root))
    return sorted(path.stem for path in root.glob("*.parquet"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m handstand.athlete",
        description=(
            "Pick the athlete out of the multi-person MediaPipe keypoints and "
            "flag the frames where the trainer overlaps or touches them."
        ),
    )
    parser.add_argument(
        "--rotate",
        choices=ROTATE_MODES,
        default=DEFAULT_ROTATE,
        help=(
            f"rotation mode of the keypoints to read, and of the output (default: {DEFAULT_ROTATE})"
        ),
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help="clip ids to process (default: every clip with multi-person keypoints)",
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
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory holding keypoints/ (default: $HANDSTAND_DATA)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    root = pathlib.Path(args.data) if args.data is not None else data_dir()
    in_root = root / "keypoints" / input_dirname()
    out_root = root / "keypoints" / output_dirname()

    try:
        clip_ids = available_clips(args.rotate, in_root, args.clips)
    except FileNotFoundError as error:
        print(f"athlete: {error}")
        return 2
    if args.limit is not None:
        clip_ids = clip_ids[: args.limit]
    if not clip_ids:
        print("no clips to process")
        return 0

    print(f"rotate={args.rotate} clips={len(clip_ids)} in={in_root} out={out_root}")
    failures = 0
    for clip_id in clip_ids:
        try:
            report = run_clip(
                clip_id,
                rotate_mode=args.rotate,
                in_root=in_root,
                out_root=out_root,
                overwrite=args.overwrite,
            )
        except Exception as error:  # one bad clip must not kill the whole batch
            failures += 1
            print(f"fail  {clip_id}: {error}")
            continue
        if report.skipped:
            print(f"skip  {clip_id} ({report.parquet_path} exists)")
            continue
        print(
            f"clip  {clip_id} rotate={report.rotate} frames={report.frame_count} "
            f"people[{report.people_summary}] athlete={report.athlete_percent:.1f}% "
            f"contact={report.contact_percent:.1f}% dropped={report.dropped_percent:.1f}% "
            f"runtime={report.runtime_seconds:.1f}s -> {report.parquet_path}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
