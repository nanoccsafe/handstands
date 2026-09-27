"""First-pass keypoint labels for the sampled frames, from a model outside the bake-off.

Labelling 300 frames joint by joint is a couple of hours of clicking, and every
click is the same work. This module runs **RTMPose** (the ``rtmlib`` package,
ONNX Runtime on the CPU) over every image
:mod:`handstand.frame_sampler` chose, turns each pose into a Label Studio
*pre-annotation*, and writes a review queue that puts the frames most likely to
be wrong at the top::

    <data_dir>/label_studio_prelabels.json   # one task per image, with "predictions"
    <data_dir>/labels/review_queue.csv       # worst-first, with the reason

The user then imports the tasks **with** their predictions into Label Studio
(#13), corrects what is wrong and exports as usual — ``handstand.labels`` imports
that export exactly as before.

Why RTMPose
-----------

The bias rule for chainlink #16: the bake-off contestants are **MediaPipe** and
**Apple Vision**, so the pre-labels must not come from either of them. Draw the
first-pass labels with a contestant and the corrected "ground truth" ends up
leaning towards whoever drew it, which is exactly the bias the bake-off exists
to measure. RTMPose is a third, independent model (a top-down pose estimator on
a YOLOX detector, trained on COCO-WholeBody), so what the labeler corrects is a
model's mistake rather than a contestant's — and where RTMPose and a contestant
disagree, the disagreement is itself the reason to look at that frame first.

The model
---------

:func:`make_model` builds ``rtmlib.Wholebody``, which reports the 133
**COCO-WholeBody** keypoints: 23 body-and-foot joints, 68 face, 42 hand. Only the
15 joints ``tools/labeling/label_studio_config.xml`` asks a labeler for are used;
the face and hand points are ignored. ``left_foot_index`` — MediaPipe's big toe,
which is what the config calls a toe — is COCO-WholeBody's ``left_big_toe``;
:data:`JOINT_MAP` is the whole translation, and a test pins it against rtmlib's
own ``coco133`` skeleton so a dependency bump cannot silently shift a joint.

Weights are downloaded on first use into ``pipeline/models/`` (git-ignored).
:func:`_resolve` looks there first, so a second run is offline.

Every frame is predicted **twice**: once as displayed and once rotated 180°, and
the result with the higher mean keypoint score wins (:func:`choose_orientation`).
An inverted body is what pose models are worst at, and the map-back is the same
exact formula the MediaPipe runner uses (:func:`handstand.rotation.inverse_rotate_points`),
so a rotated prediction lands in the same pixels as an upright one.

When RTMPose finds more than one person, the athlete is the one whose box best
overlaps the MediaPipe athlete of the same frame
(:func:`select_person`) — the selection ``handstand.athlete`` already made for
this very frame, so the two models are compared on the same body. Without a
reference the most confident person is taken and the frame is marked as having
no reference.

**Points RTMPose scored below** :data:`SCORE_THRESHOLD` are **not** written.
A guessed point is worse than no point: the labelling rule asks a labeler to
place nothing rather than guess, and a low-scoring pre-label would fight that.
Those joints are exactly what ``low_score_joints`` lists in the review queue.

The review queue
----------------

``max_disagreement`` is, per image, the **largest** distance between any two
models' points for the same joint — RTMPose, the MediaPipe athlete
(:data:`handstand.overlay.SOURCES`) and the Vision athlete when it is there —
divided by that frame's body length, so a tall athlete and a short one are
judged on the same terms. ``worst_joint`` names the joint that disagreement was
on. Rows are sorted by that disagreement plus
:data:`LOW_SCORE_WEIGHT` times the share of joints RTMPose was unsure about, so a
frame can reach the top either because two models put a joint in different
places or because the pre-labels are missing joints the labeler has to find
themselves.

CLI::

    cd pipeline
    uv run python -m handstand.prelabel
    uv run python -m handstand.prelabel --mode performance --limit 20
    uv run python -m handstand.prelabel --contact-sheet 6
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import pathlib
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import cv2
import numpy as np

from handstand import athlete as athlete_module
from handstand.labels import LABEL_JOINTS, load_manifest, percent_to_pixels
from handstand.overlay import (
    VISION_ATHLETE_SOURCE,
    FrameKeypoints,
    PanelKeypoints,
    draw_caption,
    draw_pose,
    load_panel,
    panel_parquet_path,
)
from handstand.paths import data_dir
from handstand.pose_mediapipe import JOINT_INDEX, JOINT_NAMES, apply_display_rotation
from handstand.rotation import inverse_rotate_points

__all__ = [
    "ATHLETE_SOURCE",
    "COCO133_BODY_JOINT_NAMES",
    "CONTACT_SHEET_COLUMNS",
    "CONTACT_SHEET_ROWS",
    "DEFAULT_CONTACT_SHEET",
    "DEFAULT_MODE",
    "DEFAULT_ROTATE",
    "DEFAULT_VISION_SOURCE",
    "JOINT_MAP",
    "LABEL_COLUMNS",
    "LOW_SCORE_WEIGHT",
    "MODES",
    "MODELS_DIRNAME",
    "MODEL_VERSION",
    "PERCENT_DIGITS",
    "REVIEW_FILENAME",
    "SCORE_THRESHOLD",
    "Detection",
    "ImagePrediction",
    "PanelCache",
    "PersonPose",
    "PrelabelReport",
    "build_arg_parser",
    "build_task",
    "choose_orientation",
    "contact_sheet",
    "image_uri",
    "keypoint_result",
    "landmarks_block",
    "load_manifest",
    "main",
    "make_model",
    "mean_score",
    "person_box",
    "pixels_to_percent",
    "predict_frame",
    "prelabel",
    "rotate_box_180",
    "run_model",
    "select_person",
    "summarise",
    "write_json",
    "write_review_queue",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Name of the models directory, relative to this file, not to the cwd.
MODELS_DIRNAME = "models"
#: ``pipeline/models/``, where RTMPose's ONNX weights are downloaded on first
#: use. Git-ignored, like the MediaPipe ``.task`` file next to it.
MODELS_DIR = pathlib.Path(__file__).resolve().parents[1] / MODELS_DIRNAME

#: ``rtmlib.Wholebody`` modes: the detector/pose pair and its input size.
MODES: tuple[str, ...] = ("lightweight", "balanced", "performance")
#: Default mode. ``balanced`` is rtmlib's own default and the one the weights
#: download message names; ``performance`` is the larger, slower pair.
DEFAULT_MODE = "balanced"

#: ``model_version`` written into every Label Studio prediction.
MODEL_VERSION = "rtmpose-prelabel"

#: The 23 body-and-foot keypoints of COCO-WholeBody's 133, in model order. The
#: other 110 are the 68 face and 42 hand points, which the labelling config
#: never asks for.
COCO133_BODY_JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)

#: ``handstand`` joint name -> COCO-WholeBody joint name, for the 15 joints the
#: labelling config asks for. The only entry that is not spelled the same on
#: both sides is the toe: MediaPipe calls it ``foot_index`` (a big toe) and
#: COCO-WholeBody calls it ``big_toe``.
JOINT_MAP: dict[str, str] = {
    "nose": "nose",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "left_foot_index": "left_big_toe",
    "right_foot_index": "right_big_toe",
}

#: Model row number of each labelled joint, derived from
#: :data:`COCO133_BODY_JOINT_NAMES` so the two cannot drift apart.
RTMPOSE_INDEX: dict[str, int] = {
    joint: COCO133_BODY_JOINT_NAMES.index(name) for joint, name in JOINT_MAP.items()
}

#: A point RTMPose scored below this is **not** written as a pre-label. The
#: labelling rule asks a labeler to place nothing rather than guess where a
#: hidden joint would be, and a low-scoring pre-label fights that rule: it puts
#: a plausible-looking dot where the model was guessing. :data:`SCORE_THRESHOLD`
#: is also the bar below which a point is ignored when the models are compared
#: against each other, so a joint nobody measured never counts as a disagreement.
#:
#: This is a constant rather than a flag on purpose: the number is not a
#: trade-off to tune but a translation of the labelling rule, and a run whose
#: pre-labels were written at one bar and reviewed at another would be
#: comparing two different questions.
SCORE_THRESHOLD = 0.3

#: Decimal places of the percentages written into the Label Studio tasks. Four
#: decimals of a percent is 0.04 of a pixel on a 368-pixel-wide frame, an order
#: of magnitude below the two the keypoint CSV keeps, so nothing is lost by the
#: pixels -> percent -> pixels round trip.
PERCENT_DIGITS = 4

#: How much a frame's share of unsure joints is worth against its disagreement,
#: in body lengths. 0.5 means a frame where RTMPose was unsure about *every*
#: joint is worth half a body length of two models pointing at different places.
LOW_SCORE_WEIGHT = 0.5

#: ``--rotate`` default: the rotation mode of the keypoint parquets this module
#: compares against, and the one the sampler used.
DEFAULT_ROTATE = "auto"

#: The keypoint source the athlete reference is read from.
ATHLETE_SOURCE = "athlete"
#: The second model, when its athlete selection has been generated.
DEFAULT_VISION_SOURCE = VISION_ATHLETE_SOURCE
#: The two sources compared against RTMPose, in the order they are read.
REFERENCE_SOURCES: tuple[str, ...] = (ATHLETE_SOURCE, DEFAULT_VISION_SOURCE)

#: The pre-annotation file, relative to the data directory.
PRELABELS_FILENAME = "label_studio_prelabels.json"

#: The review queue, relative to ``<data_dir>/labels/`` (``handstand.labels``'s
#: own directory, so the labelling outputs sit together).
REVIEW_FILENAME = "review_queue.csv"

#: Full CSV header of the review queue, in order. Part of the contract.
LABEL_COLUMNS: tuple[str, ...] = (
    "image",
    "clip_id",
    "frame_idx",
    "stratum",
    "max_disagreement",
    "worst_joint",
    "low_score_joints",
    "rotated_used",
)

#: Separator of the ``low_score_joints`` cell; ``;`` because a CSV cell is read
#: with a comma splitter in half the tools out there.
JOINT_SEPARATOR = ";"

#: Contact sheet shape: 3 columns by 2 rows, which is 6 frames.
CONTACT_SHEET_COLUMNS = 3
CONTACT_SHEET_ROWS = 2
#: Frames drawn when ``--contact-sheet`` is given without a number.
DEFAULT_CONTACT_SHEET = CONTACT_SHEET_COLUMNS * CONTACT_SHEET_ROWS

#: Lowercase booleans, as in ``catalogue.csv`` and the sampler manifest.
TRUE = "true"
FALSE = "false"

#: A body length that came out at the floor was never measured; see
#: :func:`handstand.athlete.body_length`.
UNMEASURED_BODY_LENGTH = athlete_module.MIN_BODY_LENGTH_PIXELS

#: Generated files are shared, human-read files, so they are not left private.
FILE_MODE = 0o644

#: Frames between progress lines of a long run.
PROGRESS_EVERY = 25


# --------------------------------------------------------------------------- #
# Percentages
# --------------------------------------------------------------------------- #


def pixels_to_percent(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Convert display-frame pixels into Label Studio's ``0..100`` percentages.

    The exact inverse of :func:`handstand.labels.percent_to_pixels`, and for the
    same reason: a percentage of the image is a fraction of ``size - 1``, because
    pixel *indices* run 0 to ``size - 1``. Round-tripping a prediction through
    this and back therefore returns the pixels it started from, which is what
    lets a pre-label be scored against the hand labels in one coordinate system.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"unknown display size: {width}x{height}")
    return (float(x) / (width - 1) * 100.0, float(y) / (height - 1) * 100.0)


def _percent(value: float) -> float:
    """One percentage, rounded to :data:`PERCENT_DIGITS` and inside 0..100."""
    return round(min(100.0, max(0.0, float(value))), PERCENT_DIGITS)


# --------------------------------------------------------------------------- #
# Keypoints of one person
# --------------------------------------------------------------------------- #


def _check_pose(joints: Mapping[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Validate and copy one person's joints, so a caller cannot mutate the original."""
    if not joints:
        raise ValueError("a pose needs at least one joint")
    unknown = sorted(set(joints) - set(JOINT_NAMES))
    if unknown:
        raise ValueError(f"not MediaPipe joint name(s): {', '.join(unknown)}")
    return {name: (float(xy[0]), float(xy[1])) for name, xy in joints.items()}


def _check_scores(scores: Mapping[str, float], joints: Mapping[str, Any]) -> dict[str, float]:
    """Validate and copy the scores of the same person.

    Every joint needs a score: a pose without one cannot be ranked against
    another, and a silently-missing score is how a low-confidence point ends up
    looking confident.
    """
    missing = sorted(set(joints) - set(scores))
    if missing:
        raise ValueError(f"no score for joint(s): {', '.join(missing)}")
    unknown = sorted(set(scores) - set(joints))
    if unknown:
        raise ValueError(f"score without a joint: {', '.join(unknown)}")
    return {name: float(value) for name, value in scores.items()}


@dataclasses.dataclass(frozen=True)
class PersonPose:
    """One person's labelled joints, in display-frame pixels, with RTMPose's scores.

    A joint is kept even when its score is below :data:`SCORE_THRESHOLD` — the
    point is where the model put it, and *that* is the evidence a low score
    carries. What is written out and what is compared is decided by the score
    at the point of use (:func:`keypoint_result`, :func:`person_box`).
    """

    joints: dict[str, tuple[float, float]]
    scores: dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", _check_pose(self.joints))
        object.__setattr__(self, "scores", _check_scores(self.scores, self.joints))

    @property
    def mean_score(self) -> float:
        """Mean score over the joints, or 0.0 when the scores are not finite.

        This is the number :func:`choose_orientation` compares: a pose the model
        was unsure about everywhere is a worse set of pre-labels, whatever it
        points at.
        """
        return mean_score(self.scores)

    @property
    def low_score_joints(self) -> tuple[str, ...]:
        """The joints RTMPose was unsure about, in :data:`LABEL_JOINTS` order."""
        return tuple(
            name
            for name in LABEL_JOINTS
            if name in self.scores and self.scores[name] < SCORE_THRESHOLD
        )

    def usable(self, name: str, min_score: float = SCORE_THRESHOLD) -> bool:
        """Is this joint measured well enough to be used as evidence?"""
        score = self.scores.get(name, float("nan"))
        if not math.isfinite(score) or score < min_score:
            return False
        x, y = self.joints[name]
        return math.isfinite(x) and math.isfinite(y)

    def point(self, name: str, min_score: float = SCORE_THRESHOLD) -> tuple[float, float] | None:
        """This joint's pixel position, or ``None`` when it may not be used."""
        return self.joints[name] if self.usable(name, min_score) else None

    def to_landmarks(self) -> np.ndarray:
        """The pose as the ``(33, 5)`` block the athlete module measures.

        The 18 MediaPipe joints this pre-label does not produce stay NaN, which is
        the same NaN a frame the model missed carries and every rule in
        :mod:`handstand.athlete` already drops.
        """
        return landmarks_block(self.joints, self.scores)


def mean_score(scores: Mapping[str, float]) -> float:
    """Mean of the finite scores; 0.0 when none of them is usable."""
    values = [value for value in scores.values() if math.isfinite(value)]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def landmarks_block(
    joints: Mapping[str, tuple[float, float]], scores: Mapping[str, float]
) -> np.ndarray:
    """A ``(33, 5)`` MediaPipe landmark block, NaN for every joint not given.

    The layout is the one the parquets use (``x, y, z, visibility, presence``),
    so the array can be handed straight to :mod:`handstand.athlete` and
    :func:`handstand.athlete.body_length` without a second convention.
    """
    block = np.full((len(JOINT_NAMES), athlete_module.LANDMARK_FIELDS), np.nan)
    for name, (x, y) in joints.items():
        index = JOINT_INDEX[name]
        block[index, 0] = float(x)
        block[index, 1] = float(y)
        block[index, 3] = float(scores.get(name, float("nan")))
    return block


def person_box(
    joints: Mapping[str, tuple[float, float]], scores: Mapping[str, float]
) -> tuple[float, float, float, float] | None:
    """``(x0, y0, x1, y1)`` around the joints scored at or above the threshold.

    Measured over the :data:`LABEL_JOINTS` only, so the RTMPose box and the
    MediaPipe athlete's box cover the same 15 joints and can be compared
    directly — measuring one over all 33 and the other over 15 would reward
    whichever model guessed more face landmarks.
    """
    points = [
        joints[name]
        for name in LABEL_JOINTS
        if name in joints
        and name in scores
        and math.isfinite(scores[name])
        and scores[name] >= SCORE_THRESHOLD
        and math.isfinite(joints[name][0])
        and math.isfinite(joints[name][1])
    ]
    if not points:
        return None
    array = np.asarray(points, dtype=np.float64)
    return (
        float(array[:, 0].min()),
        float(array[:, 1].min()),
        float(array[:, 0].max()),
        float(array[:, 1].max()),
    )


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class PoseModel(Protocol):
    """Structural type of ``rtmlib.Wholebody``, the model this module drives.

    ``__call__`` takes a BGR frame and returns ``(keypoints, scores)`` shaped
    ``(people, joints, 2)`` and ``(people, joints)``, in **display-frame pixels**
    — the pixels the model saw, which is the frame as it was handed in. Tests
    inject a stub with this shape, so nothing here needs the ONNX weights.
    """

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return every person's keypoints and their scores."""
        ...


def _resolve(url: str, models_dir: pathlib.Path) -> str:
    """A local ONNX file for ``url``, downloading it into ``models_dir`` if absent.

    The file name follows rtmlib's own convention, so a weight already in
    ``pipeline/models/`` is found without downloading anything and a second run
    needs no network.
    """
    stem = url.rsplit("/", 1)[-1].split(".", 1)[0]
    local = models_dir / f"{stem}.onnx"
    if local.is_file():
        return str(local)
    try:
        from rtmlib.tools.file import download_checkpoint
    except ImportError as error:  # pragma: no cover - rtmlib is a hard dependency
        raise RuntimeError(
            "rtmlib is required for the pre-labels; install it with `uv sync`"
        ) from error
    models_dir.mkdir(parents=True, exist_ok=True)
    return str(download_checkpoint(url, dst_dir=str(models_dir)))


def make_model(
    mode: str = DEFAULT_MODE,
    *,
    models_dir: str | pathlib.Path | None = None,
    backend: str = "onnxruntime",
    device: str = "cpu",
) -> PoseModel:
    """Build the RTMPose whole-body model, downloading its weights if needed.

    ``mode`` picks the detector/pose pair out of ``rtmlib.Wholebody.MODE``:
    ``lightweight``, ``balanced`` (the default) or ``performance``. Weights land
    in ``pipeline/models/``, which is git-ignored, and are found there on a
    second call instead of being downloaded again.

    ``backend``/``device`` are passed through to rtmlib; the defaults are ONNX
    Runtime on the CPU, which is what keeps this runnable on a workstation with
    no GPU.
    """
    try:
        from rtmlib import Wholebody
    except ImportError as error:
        raise RuntimeError(
            "rtmlib is required for the pre-labels; add it with `uv add rtmlib`"
        ) from error
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    destination = pathlib.Path(models_dir) if models_dir is not None else MODELS_DIR
    spec = Wholebody.MODE[mode]
    return Wholebody(
        det=_resolve(spec["det"], destination),
        det_input_size=spec["det_input_size"],
        pose=_resolve(spec["pose"], destination),
        pose_input_size=spec["pose_input_size"],
        mode=mode,
        backend=backend,
        device=device,
    )


def run_model(model: PoseModel, image: np.ndarray) -> tuple[list[PersonPose], int]:
    """Run the model on one frame; every person it found, in its own order.

    The raw model output is validated here and nowhere else, so everything
    downstream can assume one shape. A person whose keypoints are not all
    finite is kept with the NaNs — a half-seen body is still evidence, and the
    score decides whether a joint may be used, not whether the row exists.
    """
    raw_points, raw_scores = model(image)
    points = np.asarray(raw_points, dtype=np.float64)
    scores = np.asarray(raw_scores, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError(f"expected (people, joints, 2) keypoints, got {points.shape}")
    if scores.shape != points.shape[:2]:
        raise ValueError(
            f"expected ({points.shape[0]}, {points.shape[1]}) scores, got {scores.shape}"
        )
    if points.shape[0] and points.shape[1] <= max(RTMPOSE_INDEX.values()):
        raise ValueError(
            f"model reported {points.shape[1]} joints, but this module reads up to index "
            f"{max(RTMPOSE_INDEX.values())}; check the model and JOINT_MAP"
        )

    people: list[PersonPose] = []
    for index in range(points.shape[0]):
        joints = {
            joint: (float(points[index, row, 0]), float(points[index, row, 1]))
            for joint, row in RTMPOSE_INDEX.items()
        }
        per_joint = {joint: float(scores[index, row]) for joint, row in RTMPOSE_INDEX.items()}
        people.append(PersonPose(joints=joints, scores=per_joint))
    return people, points.shape[0]


def _map_back(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Undo the 180° the model saw, back into the frame as it is displayed."""
    return inverse_rotate_points(points, 180, width, height)


# --------------------------------------------------------------------------- #
# Choosing the athlete, and the orientation
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Detection:
    """The people found in one orientation, and which of them was taken."""

    people: tuple[PersonPose, ...]
    person_index: int
    athlete_reference: bool

    @property
    def pose(self) -> PersonPose | None:
        """The person that was taken, or ``None`` when nobody was found."""
        if not self.people or not 0 <= self.person_index < len(self.people):
            return None
        return self.people[self.person_index]

    @property
    def mean_score(self) -> float:
        """Mean score of the person that was taken; 0.0 when there is nobody."""
        pose = self.pose
        return 0.0 if pose is None else pose.mean_score

    @property
    def n_people(self) -> int:
        """How many people the model reported in this orientation."""
        return len(self.people)


def rotate_box_180(
    box: tuple[float, float, float, float] | None, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """A box turned 180°, which is what the rotated pass has to be matched against.

    A 180° turn swaps the two ends of both axes, so ``x0`` and ``x1`` exchange
    places and so do ``y0`` and ``y1``. ``None`` goes through: a frame with no
    reference has no rotated reference either.

    The reference is measured in the frame the labeler sees, while the rotated
    pass has to be matched in the frame the model saw, and those are the same
    picture turned over. Comparing a rotated pose against an unrotated reference
    would score the athlete's own body as a complete mismatch.
    """
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return (
        float(width - 1 - x1),  # the near x edge becomes the far one
        float(height - 1 - y1),  # and likewise for y: both orders flip
        float(width - 1 - x0),
        float(height - 1 - y0),
    )


def _into_display(detection: Detection, width: int, height: int) -> Detection:
    """A detection whose people are back in display-frame pixels.

    The rotated pass is *selected* in the frame the model saw and only mapped
    back afterwards, so a :class:`Detection` always speaks the frame the labeler
    is looking at and two of them can be compared. The choice of person, and
    whether the reference decided it, both carry over untouched: the map-back is
    the same 180° on every body, so it cannot change which one overlapped.
    """
    if not detection.people:
        return detection
    mapped = tuple(
        PersonPose(
            joints=_map_back_joints(pose.joints, width, height), scores=dict(pose.scores)
        )
        for pose in detection.people
    )
    return Detection(
        people=mapped,
        person_index=detection.person_index,
        athlete_reference=detection.athlete_reference,
    )


def _map_back_joints(
    joints: Mapping[str, tuple[float, float]], width: int, height: int
) -> dict[str, tuple[float, float]]:
    """One person's joints out of the rotated frame and back into the display one."""
    points = np.asarray(
        [joints[name] for name in LABEL_JOINTS], dtype=np.float64
    ).reshape(-1, 2)
    mapped = _map_back(points, width, height)
    return {
        name: (float(x), float(y)) for name, (x, y) in zip(LABEL_JOINTS, mapped, strict=True)
    }


def select_person(
    people: Sequence[PersonPose], reference_box: tuple[float, float, float, float] | None
) -> Detection:
    """Which of the people RTMPose found is the athlete.

    With a ``reference_box`` — the bounding box of the MediaPipe athlete of this
    very frame, so the two models are compared on the same body — the winner is
    the person whose box overlaps it most (:func:`handstand.athlete.box_iou`, the
    measure the athlete module itself uses). A reference that overlaps nobody at
    all is not evidence, so the most confident person is taken instead and
    ``athlete_reference`` is ``False``: the frame is then ranked on RTMPose's own
    confidence rather than pretended to be a reference match.

    Without a reference at all the most confident person is taken for the same
    reason. Ties keep the model's own order, so the choice is deterministic.
    """
    if not people:
        return Detection(people=(), person_index=-1, athlete_reference=False)

    if reference_box is not None:
        overlaps = []
        for person in people:
            box = person_box(person.joints, person.scores)
            overlaps.append(0.0 if box is None else athlete_module.box_iou(box, reference_box))
        best = max(range(len(people)), key=lambda index: (overlaps[index], -index))
        if overlaps[best] > 0.0:
            return Detection(
                people=tuple(people), person_index=best, athlete_reference=True
            )

    best_confident = max(range(len(people)), key=lambda index: (people[index].mean_score, -index))
    return Detection(people=tuple(people), person_index=best_confident, athlete_reference=False)


def choose_orientation(upright: Detection, rotated: Detection) -> tuple[Detection, bool]:
    """Keep whichever orientation RTMPose was more confident about.

    Returns the winner and whether it was the rotated one. An inverted body is
    what pose models are worst at, so a handstand usually wins rotated; but the
    decision is made on the score rather than on the rule, because the model
    sometimes copes with a hold upside down and sometimes does not. A tie goes
    to the upright result, so the same input always gives the same answer.
    """
    if rotated.mean_score > upright.mean_score:
        return rotated, True
    return upright, False


def predict_frame(
    model: PoseModel,
    image: np.ndarray,
    *,
    reference_box: tuple[float, float, float, float] | None = None,
) -> tuple[Detection, bool]:
    """Predict one frame: upright and rotated, athlete chosen, better one kept.

    ``image`` is a BGR frame in display orientation, as the sampler wrote it. The
    rotated pass is fed ``cv2.ROTATE_180`` of the same pixels and its result is
    mapped back through :func:`handstand.rotation.inverse_rotate_points` before
    anything is compared, so both passes end up speaking the frame's own
    coordinates.

    ``reference_box`` is the athlete's box in **display** pixels, and it is the
    only reference the caller supplies. Each pass is matched against the
    reference *as it appears in the frame that pass was given* — the upright one
    against the box, the rotated one against :func:`rotate_box_180` of it — and
    the rotated poses are only mapped home afterwards. Matching a rotated pose
    against an unrotated box would put the athlete's own body nowhere near it.
    """
    height, width = int(image.shape[0]), int(image.shape[1])
    rotated_image = apply_display_rotation(image, 180)

    upright_people, _ = run_model(model, image)
    upright = select_person(upright_people, reference_box)

    rotated_people, _ = run_model(model, rotated_image)
    rotated = _into_display(
        select_person(rotated_people, rotate_box_180(reference_box, width, height)),
        width,
        height,
    )
    return choose_orientation(upright, rotated)


# --------------------------------------------------------------------------- #
# The predictions of one image
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ImagePrediction:
    """What the pre-labeller settled on for one image."""

    image: str
    clip_id: str
    frame_idx: int
    stratum: str
    width: int
    height: int
    pose: PersonPose | None
    rotated_used: bool
    person_index: int
    athlete_reference: bool
    n_people: int
    runtime_seconds: float = 0.0

    @property
    def mean_score(self) -> float:
        """Mean score of the kept pose; 0.0 when nobody was found."""
        return 0.0 if self.pose is None else self.pose.mean_score

    @property
    def low_score_joints(self) -> tuple[str, ...]:
        """The joints the labeler has to place themselves."""
        return () if self.pose is None else self.pose.low_score_joints


def keypoint_result(
    joint: str, x: float, y: float, width: int, height: int
) -> dict[str, Any]:
    """One Label Studio keypoint result for one joint, in percentages.

    ``from_name``/``to_name`` are the ``name``/``toName`` of
    ``tools/labeling/label_studio_config.xml``'s ``KeyPointLabels`` tag, and the
    label travels in the plural ``keypointlabels`` list that Label Studio 1.23
    writes and :func:`handstand.labels.joint_labels` reads first. ``width`` and
    ``height`` are 1.0 because a keypoint is a point: the schema wants the size
    of the region a result covers, and a point covers one.
    """
    percent_x, percent_y = pixels_to_percent(x, y, width, height)
    return {
        "type": "keypoints",
        "from_name": "keypoints",
        "to_name": "image",
        "value": {
            "x": _percent(percent_x),
            "y": _percent(percent_y),
            "width": 1.0,
            "height": 1.0,
            "keypointlabels": [joint],
        },
    }


def build_task(image: str, uri: str, prediction: ImagePrediction) -> dict[str, Any]:
    """One Label Studio task for one image, carrying the pre-annotation.

    The task is the import shape Label Studio expects: a ``data.image`` it can
    load and a ``predictions`` entry, so the frames arrive **with** the
    pre-labels already drawn and the labeler only moves what is wrong. A frame
    RTMPose found nobody in still gets a task, with an empty result list — the
    labeler is asked for it like any other frame rather than it silently missing
    from the project.
    """
    results: list[dict[str, Any]] = []
    pose = prediction.pose
    if pose is not None:
        for joint in LABEL_JOINTS:
            point = pose.point(joint)
            if point is None:
                continue
            results.append(
                keypoint_result(joint, point[0], point[1], prediction.width, prediction.height)
            )
    return {
        "data": {"image": uri},
        "predictions": [
            {
                "model_version": MODEL_VERSION,
                "score": round(prediction.mean_score, 4),
                "result": results,
            }
        ],
    }


def image_uri(path: str | pathlib.Path, base: str | None = None) -> str:
    """The ``data.image`` Label Studio should load for one sampled frame.

    A local file becomes a ``file://`` URI, which is what Label Studio's local
    file serving reads. ``base`` overrides that: give an ``http(s)://`` prefix or
    a directory to have the frames served from somewhere else instead.
    """
    path = pathlib.Path(path)
    if base:
        return f"{base.rstrip('/')}/{path.name}"
    return path.as_uri()


# --------------------------------------------------------------------------- #
# The reference keypoints
# --------------------------------------------------------------------------- #


class PanelCache:
    """The athlete parquets, read once each and addressed by ``frame_idx``.

    The review queue needs the MediaPipe and Vision athlete of every sampled
    frame, and a parquet holds every frame of a clip, so loading the panel once
    per ``(source, clip)`` turns 300 lookups into 76 reads. A clip with no
    parquet at all — the Vision athlete has only been generated for a couple of
    clips so far — caches as ``None`` rather than retrying the filesystem.
    """

    def __init__(
        self,
        rotate: str = DEFAULT_ROTATE,
        *,
        data: str | pathlib.Path | None = None,
        sources: Sequence[str] = REFERENCE_SOURCES,
    ) -> None:
        self.rotate = rotate
        self.data = data
        self.sources = tuple(sources)
        self._panels: dict[tuple[str, str], PanelKeypoints | None] = {}

    def panel(self, source: str, clip_id: str) -> PanelKeypoints | None:
        """The whole panel for one clip, or ``None`` when it does not exist."""
        key = (source, clip_id)
        if key not in self._panels:
            path = panel_parquet_path(clip_id, self.rotate, self.data, source)
            self._panels[key] = load_panel(source, path) if path.is_file() else None
        return self._panels[key]

    def frame(self, source: str, clip_id: str, frame_idx: int) -> FrameKeypoints | None:
        """One frame's keypoints, or ``None`` when the clip or frame is absent."""
        panel = self.panel(source, clip_id)
        return None if panel is None else panel.frame(frame_idx)

    def box(self, source: str, clip_id: str, frame_idx: int) -> tuple[
        float, float, float, float
    ] | None:
        """One frame's box around the :data:`LABEL_JOINTS`, or ``None``.

        The box is measured over the 15 labelled joints rather than the whole
        keypoint set so it is comparable with :func:`person_box`.
        """
        frame = self.frame(source, clip_id, frame_idx)
        if frame is None or not frame.detected:
            return None
        joints = {name: frame.joints_xy[name] for name in LABEL_JOINTS if name in frame.joints_xy}
        scores = {
            name: frame.visibility.get(name, float("nan"))
            for name in joints
            if math.isfinite(frame.visibility.get(name, float("nan")))
            and frame.visibility[name] >= athlete_module.CONTACT_MIN_VISIBILITY
        }
        return person_box(joints, scores)


# --------------------------------------------------------------------------- #
# The review queue
# --------------------------------------------------------------------------- #


def _pair_distance(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    """Euclidean distance between two points in pixels."""
    return float(math.hypot(first[0] - second[0], first[1] - second[1]))


def _joint_disagreement(points: Sequence[tuple[float, float]]) -> float:
    """The largest distance between any two of a joint's points; NaN under two."""
    if len(points) < 2:
        return float("nan")
    return max(
        _pair_distance(points[first], points[second])
        for first in range(len(points))
        for second in range(first + 1, len(points))
    )


def _disagreement_row(
    prediction: ImagePrediction, cache: PanelCache
) -> tuple[float, str]:
    """``(max_disagreement, worst_joint)`` for one image, in body lengths.

    For every labelled joint the points RTMPose, the MediaPipe athlete and the
    Vision athlete each put on it are compared, and the largest pairwise gap
    wins; then the largest of those gaps across the joints, divided by this
    frame's body length, is the image's number. A joint only one model measured
    has no gap to report and is skipped, so a frame whose Vision keypoints are
    missing altogether is scored on MediaPipe alone rather than being called
    disagreeing with nobody.

    The body length is the MediaPipe athlete's own
    (:func:`handstand.athlete.body_length`), falling back to RTMPose's, so the
    unit is one athlete's own reach however tall or short they are.
    """
    pose = prediction.pose
    if pose is None:
        return 0.0, ""

    length = _body_length(prediction, cache)
    if length is None:
        return 0.0, ""

    frames = {
        source: cache.frame(source, prediction.clip_id, prediction.frame_idx)
        for source in REFERENCE_SOURCES
    }
    worst_distance = 0.0
    worst_joint = ""
    for joint in LABEL_JOINTS:
        points: list[tuple[float, float]] = []
        rtm = pose.point(joint)
        if rtm is not None:
            points.append(rtm)
        for frame in frames.values():
            if frame is None or not frame.detected:
                continue
            x, y = frame.joints_xy.get(joint, (float("nan"), float("nan")))
            score = frame.visibility.get(joint, float("nan"))
            if (
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(score)
                and score >= athlete_module.CONTACT_MIN_VISIBILITY
            ):
                points.append((float(x), float(y)))
        distance = _joint_disagreement(points)
        if not math.isfinite(distance):
            continue
        if distance > worst_distance:
            worst_distance = distance
            worst_joint = joint
    if not worst_joint:
        return 0.0, ""
    return worst_distance / length, worst_joint


def _body_length(prediction: ImagePrediction, cache: PanelCache) -> float | None:
    """This frame's body length in pixels, or ``None`` when it was never measured.

    The MediaPipe athlete is preferred because it is the model the frame is
    really about; RTMPose's own body is the fallback so a frame the MediaPipe
    athlete selection dropped is still comparable. A length that came back at
    :func:`handstand.athlete.body_length`'s floor was never measured, and
    dividing by it would report a disagreement of thousands of body lengths.
    """
    frame = cache.frame(ATHLETE_SOURCE, prediction.clip_id, prediction.frame_idx)
    if frame is not None and frame.detected:
        block = landmarks_block(frame.joints_xy, frame.visibility)
        length = athlete_module.body_length(block)
        if length > UNMEASURED_BODY_LENGTH:
            return length
    if prediction.pose is not None:
        length = athlete_module.body_length(prediction.pose.to_landmarks())
        if length > UNMEASURED_BODY_LENGTH:
            return length
    return None


def _review_score(max_disagreement: float, low_score_joints: Sequence[str]) -> float:
    """How likely this frame's pre-labels are to be wrong; higher is worse.

    The disagreement between the models, plus
    :data:`LOW_SCORE_WEIGHT` for every labelled joint RTMPose was unsure about.
    A frame reaches the top of the queue either because two models put a joint
    in different places — something is there and the models disagree about what
    — or because the pre-labels are missing joints the labeler has to find
    themselves.
    """
    return max_disagreement + LOW_SCORE_WEIGHT * len(low_score_joints) / len(LABEL_JOINTS)


def write_review_queue(
    rows: Sequence[Mapping[str, str]], path: str | pathlib.Path
) -> pathlib.Path:
    """Write the review queue worst-first, atomically.

    Sorting here rather than at the call site means the file on disk is the
    order the labeler works in, whatever order the frames were predicted in, and
    it is written to a temporary file beside it and renamed, so a reader never
    sees a half-written queue.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            -_review_score(
                _as_float(row.get("max_disagreement")), _split_joints(row.get("low_score_joints"))
            ),
            str(row.get("image", "")),
        ),
    )
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = pathlib.Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(LABEL_COLUMNS), lineterminator="\n", extrasaction="ignore"
            )
            writer.writeheader()
            for row in ordered:
                writer.writerow({column: row.get(column, "") for column in LABEL_COLUMNS})
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, FILE_MODE)
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


def _as_float(value: object) -> float:
    """Coerce a CSV cell to a finite float, 0.0 when it is not one."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _split_joints(value: object) -> list[str]:
    """Read a ``low_score_joints`` cell back into a list."""
    return [part for part in str(value or "").split(JOINT_SEPARATOR) if part]


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class PrelabelReport:
    """What one pre-labelling run produced."""

    tasks: list[dict[str, Any]]
    rows: list[dict[str, str]]
    images: int
    detected: int
    rotated: int
    with_reference: int
    failures: list[tuple[str, str]]
    prelabels_path: pathlib.Path
    review_path: pathlib.Path
    runtime_seconds: float
    mode: str = DEFAULT_MODE
    no_pose: int = 0
    with_disagreement: int = 0

    @property
    def rotated_percent(self) -> float:
        """Share of the images whose pre-labels came from the rotated pass."""
        return 0.0 if not self.images else 100.0 * self.rotated / self.images

    @property
    def detected_percent(self) -> float:
        """Share of the images RTMPose found somebody in."""
        return 0.0 if not self.images else 100.0 * self.detected / self.images

    @property
    def mean_low_score_joints(self) -> float:
        """Mean number of joints the labeler has to place, per image."""
        return 0.0 if not self.images else sum(
            len(_split_joints(row["low_score_joints"])) for row in self.rows
        ) / self.images


def write_json(payload: Any, path: str | pathlib.Path) -> pathlib.Path:
    """Write the Label Studio tasks as JSON, atomically.

    Written to a temporary file beside it and renamed, because a reader either
    sees the previous task list or the new one — never a truncated file that
    would import half the frames into Label Studio.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = pathlib.Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, FILE_MODE)
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


def prelabel(
    *,
    model: PoseModel,
    data: str | pathlib.Path | None = None,
    manifest: str | pathlib.Path | None = None,
    rotate: str = DEFAULT_ROTATE,
    limit: int | None = None,
    image_base: str | None = None,
    out_json: str | pathlib.Path | None = None,
    out_csv: str | pathlib.Path | None = None,
    read_image: Callable[[pathlib.Path], np.ndarray | None] | None = None,
    progress: Callable[[str], None] | None = None,
    mode: str = DEFAULT_MODE,
) -> PrelabelReport:
    """Predict every sampled frame and write the pre-labels and the review queue.

    ``model`` is anything with :class:`PoseModel`'s shape, so a test drives the
    whole run without the ONNX weights. ``mode`` is only recorded, so the report
    says which detector/pose pair produced it. ``read_image`` loads a JPEG and
    defaults to OpenCV; a frame that will not load is reported in
    :attr:`PrelabelReport.failures` and left out rather than aborting the run,
    so one damaged JPEG does not cost the other 299 pre-labels.

    The images are predicted in manifest order and the outputs are written only
    at the end, so a run either produces both files or produces neither.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    manifest_path = (
        pathlib.Path(manifest) if manifest is not None else root / "label_frames" / "manifest.csv"
    )
    frames_dir = manifest_path.parent
    prelabels_out = (
        pathlib.Path(out_json) if out_json is not None else root / PRELABELS_FILENAME
    )
    review_out = (
        pathlib.Path(out_csv) if out_csv is not None else root / "labels" / REVIEW_FILENAME
    )
    loader = read_image or _read_image
    say = progress or (lambda message: None)

    entries = [row for row in load_manifest(manifest_path) if str(row.get("image", "")).strip()]
    if limit is not None:
        entries = entries[: max(0, limit)]
    cache = PanelCache(rotate, data=root)

    started = time.perf_counter()
    tasks: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    predictions: list[ImagePrediction] = []
    failures: list[tuple[str, str]] = []
    detected = rotated = with_reference = with_disagreement = no_pose = 0

    for index, row in enumerate(entries, start=1):
        name = str(row["image"])
        path = frames_dir / name
        image = loader(path)
        if image is None:
            failures.append((name, "could not be read as an image"))
            continue
        height, width = int(image.shape[0]), int(image.shape[1])
        clip_id = str(row["clip_id"])
        frame_idx = int(row["frame_idx"])
        reference = cache.box(ATHLETE_SOURCE, clip_id, frame_idx)

        frame_started = time.perf_counter()
        try:
            detection, rotated_used = predict_frame(model, image, reference_box=reference)
        except (ValueError, RuntimeError) as error:
            failures.append((name, f"{type(error).__name__}: {error}"))
            continue

        pose = detection.pose
        prediction = ImagePrediction(
            image=name,
            clip_id=clip_id,
            frame_idx=frame_idx,
            stratum=str(row.get("stratum", "")),
            width=width,
            height=height,
            pose=pose,
            rotated_used=rotated_used,
            person_index=detection.person_index,
            athlete_reference=detection.athlete_reference,
            n_people=detection.n_people,
            runtime_seconds=time.perf_counter() - frame_started,
        )
        predictions.append(prediction)
        tasks.append(build_task(name, image_uri(path, image_base), prediction))
        detected += int(pose is not None)
        no_pose += int(pose is None)
        rotated += int(rotated_used)
        with_reference += int(detection.athlete_reference)

        disagreement, worst_joint = _disagreement_row(prediction, cache)
        with_disagreement += int(bool(worst_joint))
        rows.append(
            {
                "image": name,
                "clip_id": clip_id,
                "frame_idx": str(frame_idx),
                "stratum": prediction.stratum,
                "max_disagreement": f"{disagreement:.4f}",
                "worst_joint": worst_joint,
                "low_score_joints": JOINT_SEPARATOR.join(prediction.low_score_joints),
                "rotated_used": TRUE if rotated_used else FALSE,
            }
        )
        if index % PROGRESS_EVERY == 0 or index == len(entries):
            say(f"  {index}/{len(entries)} images, {time.perf_counter() - started:.0f}s")

    runtime = time.perf_counter() - started
    write_json(tasks, prelabels_out)
    write_review_queue(rows, review_out)
    return PrelabelReport(
        tasks=tasks,
        rows=rows,
        images=len(predictions),
        detected=detected,
        rotated=rotated,
        with_reference=with_reference,
        failures=failures,
        prelabels_path=prelabels_out,
        review_path=review_out,
        runtime_seconds=runtime,
        mode=mode,
        no_pose=no_pose,
        with_disagreement=with_disagreement,
    )


def _read_image(path: pathlib.Path) -> np.ndarray | None:
    """Read a sampled frame as BGR, or ``None`` when OpenCV will not read it."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if image is None else image


# --------------------------------------------------------------------------- #
# Contact sheet
# --------------------------------------------------------------------------- #


def contact_sheet(
    tasks: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, str]],
    out_path: str | pathlib.Path,
    frames_dir: str | pathlib.Path,
    *,
    read_image: Callable[[pathlib.Path], np.ndarray | None] | None = None,
    columns: int = CONTACT_SHEET_COLUMNS,
    rows_wanted: int = CONTACT_SHEET_ROWS,
) -> pathlib.Path | None:
    """Draw a grid of pre-labelled frames, worst-queue-first, as one JPEG.

    The frames come off the *sorted* review queue, so the sheet shows what the
    labeler sees first: the frames whose pre-labels most likely need correcting.
    Each tile carries the frame's own caption (its image name, whether the
    rotated pass was used, and its disagreement), which is what makes a
    surprising skeleton explainable.

    Returns ``None`` when there is nothing to draw, rather than writing an empty
    image nobody asked for.
    """
    if not tasks or not rows:
        return None
    loader = read_image or _read_image
    by_image = {str(task["data"]["image"]).rsplit("/", 1)[-1]: task for task in tasks}

    ordered = sorted(
        rows,
        key=lambda row: (
            -_review_score(
                _as_float(row.get("max_disagreement")), _split_joints(row.get("low_score_joints"))
            ),
            str(row.get("image", "")),
        ),
    )
    wanted = min(len(ordered), columns * rows_wanted)
    if wanted <= 0:
        return None

    tiles: list[np.ndarray] = []
    for row in ordered[:wanted]:
        name = str(row["image"])
        image = loader(pathlib.Path(frames_dir) / name)
        if image is None:
            continue
        joints: dict[str, tuple[float, float]] = {}
        scores: dict[str, float] = {}
        for result in by_image.get(name, {}).get("predictions", [{}])[0].get("result", []):
            value = result.get("value", {})
            labels = value.get("keypointlabels") or []
            if not labels or not math.isfinite(value.get("x", float("nan"))):
                continue
            joint = str(labels[0])
            joints[joint] = (
                _percent_to_pixels(value["x"], value["y"], image.shape[1], image.shape[0])
            )
            scores[joint] = 1.0
        if joints:
            image = draw_pose(image, joints, scores, min_visibility=0.0)
        tiles.append(
            draw_caption(
                image,
                [
                    name,
                    f"rotated={row.get('rotated_used', '')} "
                    f"disagree={row.get('max_disagreement', '')}",
                    f"worst={row.get('worst_joint', '') or '-'} "
                    f"unsure={len(_split_joints(row.get('low_score_joints')))}",
                ],
            )
        )

    if not tiles:
        return None
    sheet = _grid(tiles, columns, rows_wanted)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise OSError(f"could not write JPEG: {out_path}")
    os.chmod(out_path, FILE_MODE)
    return out_path


def _percent_to_pixels(
    percent_x: float, percent_y: float, width: int, height: int
) -> tuple[float, float]:
    """Percentages back to pixels, for drawing a prediction stored as percentages."""
    return percent_to_pixels(percent_x, percent_y, width, height)


def _grid(tiles: Sequence[np.ndarray], columns: int, rows_wanted: int) -> np.ndarray:
    """Lay tiles out in a grid, padding the ragged last row to the full width."""
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    padded = [
        cv2.copyMakeBorder(
            tile, 0, height - tile.shape[0], 0, width - tile.shape[1], cv2.BORDER_CONSTANT
        )
        for tile in tiles
    ]
    lines: list[np.ndarray] = []
    for start in range(0, len(padded), columns):
        chunk = padded[start : start + columns]
        while len(chunk) < columns:
            chunk.append(np.zeros_like(padded[0]))
        lines.append(np.hstack(chunk[:columns]))
    while len(lines) < rows_wanted:
        lines.append(np.zeros_like(lines[0]))
    return np.vstack(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.prelabel`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.prelabel",
        description=(
            "Pre-label the sampled frames with RTMPose for Label Studio, and write a "
            "review queue of the frames most likely to be wrong."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=DEFAULT_MODE,
        help=f"RTMPose detector/pose pair (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--rotate",
        default=DEFAULT_ROTATE,
        help=(
            "rotation mode of the keypoint parquets the models are compared against "
            f"(default: {DEFAULT_ROTATE})"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="predict at most N frames (default: every frame in the manifest)",
    )
    parser.add_argument(
        "--image-base",
        default=None,
        metavar="URL",
        help=(
            "serve the frames from here instead of the local files, e.g. "
            "http://localhost:8080/frames (default: file:// URIs, which need Label "
            "Studios local file serving)"
        ),
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory holding label_frames/ (default: $HANDSTAND_DATA)",
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=None,
        help="sampler manifest (default: <data_dir>/label_frames/manifest.csv)",
    )
    parser.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=None,
        help=f"pre-annotation file to write (default: <data_dir>/{PRELABELS_FILENAME})",
    )
    parser.add_argument(
        "--out-csv",
        type=pathlib.Path,
        default=None,
        help=f"review queue to write (default: <data_dir>/labels/{REVIEW_FILENAME})",
    )
    parser.add_argument(
        "--contact-sheet",
        type=int,
        nargs="?",
        const=DEFAULT_CONTACT_SHEET,
        default=0,
        metavar="N",
        help=(
            f"also draw a contact sheet of the N worst-queue frames "
            f"(default when given: {DEFAULT_CONTACT_SHEET})"
        ),
    )
    return parser


def summarise(report: PrelabelReport) -> str:
    """Multi-line summary of a run, for the CLI to print."""
    lines = [
        f"pre-labelled {report.images} frame(s) with RTMPose in {report.runtime_seconds:.0f}s",
        f"  model: {MODEL_VERSION} ({report.mode}); wrote {report.prelabels_path}",
        f"  found somebody in {report.detected}/{report.images} "
        f"({report.detected_percent:.1f} %), {report.no_pose} frame(s) left for the labeler",
        f"  rotated pass won {report.rotated}/{report.images} "
        f"({report.rotated_percent:.1f} %); athlete reference used in "
        f"{report.with_reference}",
        f"  {report.with_disagreement} frame(s) have a measurable disagreement; "
        f"{report.mean_low_score_joints:.1f} joint(s) per frame left for the labeler",
        f"  review queue: {report.review_path} (worst first)",
    ]
    lines += [f"  fail {name}: {reason}" for name, reason in report.failures]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.prelabel``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.contact_sheet < 0:
        parser.error("--contact-sheet must be >= 0")

    try:
        model = make_model(args.mode)
    except (RuntimeError, ValueError) as error:
        print(f"prelabel: {error}", file=sys.stderr)
        return 2

    try:
        report = prelabel(
            model=model,
            data=args.data,
            manifest=args.manifest,
            rotate=args.rotate,
            limit=args.limit,
            image_base=args.image_base,
            out_json=args.out_json,
            out_csv=args.out_csv,
            mode=args.mode,
            progress=lambda message: print(message, flush=True),
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"prelabel: {error}", file=sys.stderr)
        return 2

    if args.contact_sheet:
        root = pathlib.Path(args.data) if args.data is not None else data_dir()
        frames_dir = args.manifest.parent if args.manifest is not None else root / "label_frames"
        sheet = contact_sheet(
            report.tasks,
            report.rows,
            root / "overlays" / "prelabels_contact_sheet.jpg",
            frames_dir,
        )
        if sheet is not None:
            print(f"  contact sheet: {sheet}")

    print(summarise(report))
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
