"""Golden fixtures: synthetic clips whose Python answers Swift ports must match.

The Swift ports of the pipeline — post-process (#39), phases and features (#40),
the scorer (#41) — can only be trusted if they produce *the same numbers, frame
by frame*, as :mod:`handstand.postprocess`, :mod:`handstand.phases` and
:mod:`handstand.features` do. This module is what makes that checkable: it
generates small **synthetic** clips, runs the **real** Python stages over them in
memory, and writes input + expected output to one JSON file per case::

    swift/HandstandCore/Tests/HandstandCoreTests/Fixtures/golden/<case>.json

The repo is public (``CLAUDE.md``), so nothing derived from the real handstand
videos may ever be committed: every fixture written under ``swift/`` is
synthetic, generated from a seed, and carries ``meta.mode == "synthetic"``. The
same JSON *format* can be produced for a real clip with ``--real``, but only
into the git-ignored data directory — :func:`require_outside_repo` refuses any
output path inside the repository, which is what keeps real keypoints out of a
public tree by construction rather than by care.

CLI::

    cd pipeline
    uv run python -m handstand.golden                       # regenerate every case
    uv run python -m handstand.golden --only line_hold hand_step
    uv run python -m handstand.golden --real 6508f9b355bd   # local-only, never under swift/

What is in a fixture
--------------------

``meta``
    Which case this is, the generator version, the Python package versions that
    produced it, the display size, the body length ``L``, the joint names (in
    array order), the feature columns and the **tolerance per field** — see
    :data:`TOLERANCES`.
``input``
    One row per frame: ``frame_idx``, ``t_ms``, ``detected``,
    ``trainer_contact`` and ``joints`` keyed by joint name as ``[x, y,
    visibility]`` in display pixels. A frame the model did not detect carries
    ``[null, null, null]``, which is the schema's NaN (``docs/keypoint_schema.md``).
``expected``
    What the real stages answered: ``postprocess`` (per frame ``valid`` and
    ``filled`` as arrays in ``meta.joint_names`` order plus the processed
    ``joints``, and the body length parts), ``phases`` (per frame ``phase`` and
    ``hold_id``), ``features`` (per frame, ``meta.feature_columns``) and
    ``hold_summary`` (one row per hold, straight out of
    :func:`handstand.features.hold_rows`).

Every float is rounded to 1e-6 and every NaN is written as ``null``, so the file
is stable text: the same seed produces the same bytes
(:func:`fixture_text`). ``tests/test_golden.py`` reads the committed files back,
re-runs the pipeline from their ``input`` and checks the answers still match
their ``expected`` to the tolerances in ``meta``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import math
import pathlib
import sys
import zlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from handstand import athlete, features, phases, postprocess
from handstand.paths import data_dir

__all__ = [
    "CASES",
    "CASE_SPECS",
    "DEFAULT_OUT",
    "DEFAULT_SEED",
    "DISPLAY_HEIGHT",
    "DISPLAY_WIDTH",
    "FEATURE_COLUMNS",
    "GENERATOR_VERSION",
    "JOINTS",
    "MAX_FIXTURE_BYTES",
    "REAL_SUBDIR",
    "TOLERANCES",
    "CaseSpec",
    "Keyframe",
    "Mutation",
    "build_arg_parser",
    "build_fixture",
    "clip_from_frames",
    "default_out_dir",
    "fixture_text",
    "main",
    "read_fixture",
    "real_fixture",
    "recompute",
    "repo_root",
    "require_outside_repo",
    "run_pipeline",
    "write_fixture",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Bumped whenever the generator's output changes shape or meaning, so a Swift
#: port can say which version of the fixtures it was written against.
GENERATOR_VERSION = "1"

#: The default seed. Deterministic: the committed fixtures are regenerated from
#: it, and the test that proves it writes the same case twice and compares bytes.
DEFAULT_SEED = 20260925

#: The five synthetic cases, in the order the CLI writes them.
CASES: tuple[str, ...] = (
    "line_hold",
    "banana_hold",
    "hand_step",
    "gaps_and_noise",
    "trainer_contact",
)

#: The joints every fixture carries: :data:`handstand.postprocess.TRACKED_JOINTS`
#: in schema order — the nose, both shoulders, elbows, wrists, hips, knees,
#: ankles and foot indexes. These are the joints ``docs/keypoint_schema.md``
#: calls the shared schema and the only ones ``swift/HandstandCore``'s
#: ``Joint`` enum knows, so a fixture is decodable on the Swift side as it is;
#: the model's other 18 landmarks (eyes, ears, mouth, fingers, heels) feed no
#: stage being ported.
JOINTS: tuple[str, ...] = postprocess.TRACKED_JOINTS

#: The per-frame feature columns a fixture stores: every column of
#: :data:`handstand.features.FEATURES` except ``hand_width`` (not on the port's
#: list), then the centre-of-mass columns. ``com_complete`` is not stored either:
#: the list is what chainlink #25 names for the Swift side to port.
FEATURE_COLUMNS: tuple[str, ...] = (
    *(name for name in features.FEATURE_NAMES if name != "hand_width"),
    "com_u",
    "com_v",
    "com_forward",
    "facing_sign",
    "balance_zone",
)

#: How many decimals every float is written with, and therefore the most the
#: written value can differ from the one the pipeline computed.
ROUND_DIGITS = 6

#: Tolerance per field, stored in ``meta.tolerances``. A comparison walks a
#: value's dotted path (list indices dropped) and takes the longest key that
#: prefixes it; anything unmatched uses ``default``. Booleans, labels and
#: ``null``-ness are compared exactly whatever the tolerance says, because a
#: category is not a quantity.
#:
#: * pure geometry — the input coordinates and the body-length percentiles they
#:   are measured from — is exact to 1e-6, the writing precision;
#: * anything the One-Euro filter or the gap fill touched, and every feature
#:   measured off it, gets 1e-3: a port is allowed to disagree with Python in
#:   the last digits of a filtered number, not in its value.
TOLERANCES: dict[str, float] = {
    "default": 1e-6,
    "input.joints": 1e-6,
    "expected.postprocess.body_length": 1e-6,
    "expected.postprocess.frames.joints": 1e-3,
    "expected.postprocess.frames.valid": 0.0,
    "expected.postprocess.frames.filled": 0.0,
    "expected.phases.phase": 0.0,
    "expected.phases.hold_id": 0.0,
    "expected.features": 1e-3,
    "expected.hold_summary": 1e-3,
}

#: Display size every synthetic case is rendered in: the real clips are 1024x576
#: with a rotation of -90, i.e. 576x1024 as displayed, and every keypoint in the
#: schema is a pixel of *that* frame.
DISPLAY_WIDTH = 576
DISPLAY_HEIGHT = 1024

#: Where ``--real`` writes when ``--real-out`` is not given, relative to
#: ``handstand.paths.data_dir()``. Git-ignored like the rest of the data tree,
#: and refused if it resolves inside the repository.
REAL_SUBDIR = "golden_real"

#: How many frames of a real clip a ``--real`` fixture carries.
REAL_MAX_FRAMES = 300

#: The size the committed fixtures must stay under, in bytes. They are cloned
#: into every checkout and read over SSH by the Mac test runner, so the budget
#: is a real cost: about two megabytes for the five cases, checked by
#: ``tests/test_golden.py``.
MAX_FIXTURE_BYTES = 2 * 1024 * 1024

#: The ``source`` column of the hold rows: fixtures are not produced by either
#: pose model, so they get their own name rather than borrowing ``mediapipe``.
FIXTURE_SOURCE = "golden"

#: Where the committed synthetic fixtures go, relative to the repository root —
#: the default of ``--out``. ``default_out_dir`` is the absolute version of it.
DEFAULT_OUT = pathlib.Path("swift/HandstandCore/Tests/HandstandCoreTests/Fixtures/golden")


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #


def repo_root() -> pathlib.Path:
    """The repository root: the directory holding ``pipeline/`` and ``swift/``.

    Found by walking up from this file (the reliable answer when the package is
    installed editable, as ``uv sync`` installs it) and then up from the working
    directory, so a run from ``pipeline/`` or from the root finds the same tree.
    Raises :class:`RuntimeError` when neither search finds it, which is the
    honest answer for a checkout that is not one.
    """
    for start in (pathlib.Path(__file__).resolve(), pathlib.Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if _looks_like_root(candidate):
                return candidate
    raise RuntimeError("handstand.golden: cannot find the repository root")


def _looks_like_root(path: pathlib.Path) -> bool:
    """Is ``path`` a checkout of this repository?"""
    return (path / "pipeline" / "pyproject.toml").is_file() and (path / "swift").is_dir()


def default_out_dir() -> pathlib.Path:
    """Where the committed synthetic fixtures go, i.e. under ``swift/``."""
    return repo_root() / DEFAULT_OUT


def require_outside_repo(path: str | pathlib.Path) -> pathlib.Path:
    """Refuse a ``--real`` output path inside the repository; return it resolved.

    The repository is public: a fixture of a real clip holds keypoints of the
    real videos, and those never go in git. ``--real`` therefore accepts only a
    destination outside the checkout — the shared, git-ignored data directory by
    default — and this is the function that enforces it, called before anything
    is written. :class:`ValueError` names the offending path and the fix.
    """
    resolved = pathlib.Path(path).expanduser().resolve()
    try:
        root = repo_root()
    except RuntimeError as error:
        raise ValueError(f"refusing to write real fixtures: {error}") from error
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError(
            f"refusing to write real keypoint data inside the repository: {resolved}\n"
            f"the repo is public; choose a --real-out outside {root} "
            f"(the default is <data_dir>/{REAL_SUBDIR})"
        )
    return resolved


# --------------------------------------------------------------------------- #
# The stick figure
# --------------------------------------------------------------------------- #

#: Segment lengths of the stick figure, in display pixels. ``TORSO + THIGH +
#: SHIN`` is the body length ``L`` the whole fixture is scaled by — 300 px, which
#: is the order of magnitude a real clip measures (``meta.L`` records it and
#: ``meta.body_length_px`` records what :mod:`handstand.postprocess` measured).
TORSO = 105.0
THIGH = 100.0
SHIN = 95.0
ARM = 100.0
HEAD = 45.0
#: How far in front of the spine the nose sits, in pixels: the only facing cue
#: the schema has, and large enough that a hold's majority vote is not a coin toss.
NOSE_FORWARD = 25.0
#: Where along the shoulder→wrist line the elbow is drawn.
UPPER_ARM_FRACTION = 0.45
TOE = 14.0

#: The hip's position in every anchored pose: with the hands on the floor
#: (``y = 900``) the arm and the torso put it here, and with the feet on the
#: floor the legs do — so the plant is a fold at the hips rather than a stretch.
HIP_X = float(DISPLAY_WIDTH // 2)
HIP_Y = 695.0

#: The sideways offset of each side's joint from the body's midline, in pixels:
#: a side view sees the two sides nearly on top of each other, but not exactly —
#: and the ankles being further out than the hips is what gives
#: ``leg_separation`` a value instead of a hard zero.
SIDE_OFFSET: dict[str, float] = {
    "shoulder": 4.0,
    "elbow": 5.5,
    "wrist": 7.0,
    "hip": 3.0,
    "knee": 6.0,
    "ankle": 8.0,
    "foot_index": 8.0,
}

#: The detector's confidence per joint, constant over a synthetic clip: the near
#: side is the one a camera sees and the far side is partly occluded by the body.
VISIBILITY: dict[str, float] = {
    "nose": 0.9,
    **{f"left_{part}": 0.95 for part in SIDE_OFFSET},
    **{f"right_{part}": 0.72 for part in SIDE_OFFSET},
}

#: The sway of a held handstand: how far the body leans about the hands, the
#: period of the lean, and the seconds each end of a hold takes to ramp it in
#: and out (a sway switched on by a step would be a teleport, and the outlier
#: rule would quite reasonably delete it).
SWAY_DEG = 2.0
SWAY_PERIOD_S = 1.7
SWAY_RAMP_S = 0.3

#: Frame intervals of the synthetic clips, in milliseconds. Real clips are
#: variable frame rate, so the fixtures are too — but bounded, because the gap
#: and break thresholds of :mod:`handstand.postprocess` and
#: :mod:`handstand.phases` are stated in seconds and a fixture must land on the
#: same side of them whatever frame it is.
GAP_CYCLE_MS: tuple[int, ...] = (33, 36, 31, 34, 30, 37, 33, 32)


@dataclasses.dataclass(frozen=True)
class Pose:
    """Where the stick figure is, as one anchor and four directions.

    The body is a chain of fixed-length segments — hip → shoulder, hip → knee →
    ankle, shoulder → elbow → wrist — so a pose is the hip's pixel plus the
    *direction* of each of those segments, in degrees of the image frame (``y``
    grows downwards, ``0°`` points right, ``90°`` down). Segment lengths never
    change, only their directions: interpolating a pose rotates the body instead
    of stretching it, which is what keeps :func:`handstand.postprocess.estimate_body_length`
    measuring the same three spans in every frame.

    Angles are **unwrapped** across keyframes (``270, 450, …`` rather than
    ``270, 90, …``) so a linear interpolation sweeps the long way round the way
    the athlete moves: folding forward at the hips, not through a full spin.
    """

    hip: tuple[float, float]
    #: Hip → shoulder, degrees.
    sh: float
    #: Hip → ankle, degrees. The knee sits ``THIGH`` of the way along it.
    leg: float
    #: Shoulder → wrist, degrees.
    arm: float
    #: Ankle → foot index, degrees: toes forward when standing, pointed when up.
    foot: float
    #: Extra offset for the left wrist alone — the hand step of ``hand_step``.
    left_wrist_shift: tuple[float, float] = (0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class Keyframe:
    """One pose at one time: the timeline is a sequence of these.

    ``sway`` marks the keyframe that *ends* a swaying interval, so a hold is two
    keyframes a few seconds apart with ``sway`` on the second one and everything
    in between is that pose with the lean applied.
    """

    t: float
    pose: Pose
    sway: bool = False


@dataclasses.dataclass(frozen=True)
class Mutation:
    """What is done to some frames of a case after the figure is drawn.

    These are the model's own failures, in the schema's own terms: a stretch
    whose confidence falls below :data:`handstand.postprocess.MIN_VISIBILITY`, a
    stretch nobody was detected in, a frame the trainer walked into, and a joint
    that teleports. Each one is what a stage downstream has to survive.
    """

    #: ``low_visibility``, ``undetected``, ``contact`` or ``outlier``.
    kind: str
    #: When it happens, in seconds; the first frame at or after it is the first
    #: frame affected.
    at_s: float
    #: How many frames it covers (``outlier`` is always one).
    count: int = 1
    #: For ``outlier``: which joint jumps, and by how many pixels.
    joint: str = ""
    delta: float = 0.0


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    """One synthetic case: its timeline, what is wrong with it, how noisy it is."""

    keys: tuple[Keyframe, ...]
    mutations: tuple[Mutation, ...] = ()
    jitter_px: float = 0.0

    @property
    def duration_s(self) -> float:
        """How long the clip runs: the last keyframe's time."""
        return self.keys[-1].t


#: The four canonical poses. ``STAND`` has the feet on the floor and the arms
#: hanging, ``PLANTED`` is the same body folded at the hips with the hands on
#: the floor, ``HOLD`` is the same body inverted over those hands — so the
#: kick-up is the legs swinging from down to up *about the hips* and the hands
#: never move through it.
STAND = Pose(hip=(HIP_X, HIP_Y), sh=270.0, leg=90.0, arm=90.0, foot=15.0)
PLANTED = Pose(hip=(HIP_X, HIP_Y), sh=450.0, leg=90.0, arm=90.0, foot=15.0)
HOLD = Pose(hip=(HIP_X, HIP_Y), sh=450.0, leg=270.0, arm=90.0, foot=-90.0)
#: The arched body of ``banana_hold``: the hip is ahead of the shoulder→ankle
#: line, the shoulders sit behind the hands and the toes drift back — the shape
#: a judge calls a banana. The segment lengths are still exactly the four above;
#: only their directions differ.
BANANA = Pose(hip=(HIP_X + 45.0, 705.1), sh=475.4, leg=265.6, arm=90.0, foot=-90.0)
#: How far the stepped wrist moves, and the body length it is quoted in: 0.15 L
#: of a 300 px body, which is past :data:`handstand.phases.HAND_STEP_L` (0.1)
#: and therefore ends the hold it happens in.
STEP_PX = 45.0
STEP_L = STEP_PX / (TORSO + THIGH + SHIN)
#: How long the step takes. It has to be quick: :mod:`handstand.postprocess`
#: smooths before :mod:`handstand.phases` measures, and a step dawdled over
#: half a second comes out of the One-Euro filter as less than the 0.1 L the
#: hand-step rule looks for — the fixture would then contain a wrist that moved
#: and no hand step to show for it.
STEP_S = 0.15

#: The hold after one wrist has stepped :data:`STEP_PX` pixels along the floor.
HOLD_STEP = dataclasses.replace(HOLD, left_wrist_shift=(STEP_PX, 0.0))


def _story(
    hold_pose: Pose, hold_s: float, second_hold_s: float | None = None
) -> tuple[Keyframe, ...]:
    """One attempt as keyframes: stand, plant, pause, kick up, hold, come down, stand.

    The ``PLANTED`` pause between planting the hands and swinging the legs up is
    what gives ``kickup`` its own frames: :mod:`handstand.phases` only reads
    *kickup* while the hands are already still, and a hand takes a fifth of a
    second to stop being a moving hand after it lands. Without the pause the
    legs rise while the hands are still settling, the hold condition starts
    first, and the clip jumps straight from ``pre`` to ``hold``.

    ``second_hold_s`` is not ``None`` when the hold is cut in two by a hand step
    (:data:`STEP_S`): the same body, one wrist :data:`STEP_PX` pixels further
    along the mat, and a second stretch of holding after it.
    """
    keys: list[Keyframe] = [Keyframe(0.0, STAND)]
    clock = 0.0

    def move(seconds: float, pose: Pose, sway: bool = False) -> None:
        nonlocal clock
        clock += seconds
        keys.append(Keyframe(clock, pose, sway))

    move(0.40, STAND)  # standing, hands not yet down: `pre`
    move(0.30, PLANTED)  # the plant: hands travel from the hips to the floor
    move(0.25, PLANTED)  # planted and still, feet still down
    move(0.45, hold_pose)  # the kick-up: the legs swing from down to up
    move(hold_s, hold_pose, sway=True)  # the hold
    if second_hold_s is not None:
        move(STEP_S, HOLD_STEP)  # one wrist steps, 0.15 L along the floor
        move(second_hold_s, HOLD_STEP, sway=True)  # the hold picks up again
    move(0.45, PLANTED)  # the legs come back down: `exit`
    move(0.45, STAND)  # the hands leave the floor
    move(0.40, STAND)  # standing again, so `post` has some frames of its own
    return tuple(keys)


def _case_specs() -> dict[str, CaseSpec]:
    """The five cases, as timelines.

    Every case is the same story — stand, plant, kick up, hold, come down, stand
    up — with one thing varied, so a failure names itself: the first case is the
    control, and each of the others adds exactly one complication (an arch, a
    hand step, the detector's failures, a trainer).
    """
    gaps_keys = _story(HOLD, 2.40)
    # The mutations hang off the moment the hold proper begins — the end of the
    # kick-up — rather than off a magic number, so re-timing the story does not
    # quietly drop an occlusion out of the hold and into the exit. The first
    # keyframe that asks for sway ends the first swaying interval, which is the
    # hold itself; the interval *before* it is the kick-up.
    hold_start = gaps_keys[next(i for i, key in enumerate(gaps_keys) if key.sway) - 1].t
    return {
        # The control: a straight hold long enough to be scored, with the small
        # sway every held handstand has.
        "line_hold": CaseSpec(keys=_story(HOLD, 3.35)),
        # Same, but the body is arched through the hold.
        "banana_hold": CaseSpec(keys=_story(BANANA, 2.20)),
        # One wrist steps 0.15 L half-way through, which splits the hold in two.
        "hand_step": CaseSpec(keys=_story(HOLD, 1.05, second_hold_s=1.30)),
        # The detector's bad day: jitter everywhere, one short occlusion that
        # gets bridged, one long one that does not, one joint that teleports.
        "gaps_and_noise": CaseSpec(
            keys=gaps_keys,
            mutations=(
                Mutation("low_visibility", hold_start + 0.70, count=4),
                Mutation("outlier", hold_start + 1.20, joint="left_ankle", delta=180.0),
                Mutation("undetected", hold_start + 1.75, count=7),
            ),
            jitter_px=1.5,
        ),
        # A spotter in frame for a stretch of the hold: every joint of those
        # frames is gated out, and they are too long a gap to bridge.
        "trainer_contact": CaseSpec(
            keys=_story(HOLD, 2.40),
            mutations=(Mutation("contact", hold_start + 1.10, count=7),),
        ),
    }


#: The cases, keyed by name. ``hand_step``'s step really is 0.15 body lengths,
#: which is what :data:`handstand.phases.HAND_STEP_L` is measured against.
CASE_SPECS: dict[str, CaseSpec] = _case_specs()


def _vec(degrees: float, length: float) -> np.ndarray:
    """A ``length``-pixel offset in the image frame, ``degrees`` from +x (y down)."""
    radians = math.radians(degrees)
    return np.array([math.cos(radians) * length, math.sin(radians) * length])


def _rotate(point: np.ndarray, degrees: float) -> np.ndarray:
    """``point`` rotated about the origin, in the image frame."""
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.array([cosine * point[0] - sine * point[1], sine * point[0] + cosine * point[1]])


def joints_of(pose: Pose, sway_deg: float = 0.0) -> dict[str, tuple[float, float]]:
    """One pose as the fifteen tracked joints' display pixels.

    The chain is built from the hip out, then each joint is given its side's
    offset, then — ``sway_deg`` being a lean about the hands — everything is
    rotated about the wrist midpoint, which is the one point a handstand pivots
    on. The wrists themselves barely move under that (they are five pixels from
    the pivot), so a swaying hold still has planted hands.
    """
    hip = np.asarray(pose.hip, dtype=np.float64)
    shoulder = hip + _vec(pose.sh, TORSO)
    ankle = hip + _vec(pose.leg, THIGH + SHIN)
    knee = hip + _vec(pose.leg, THIGH)
    wrist = shoulder + _vec(pose.arm, ARM)
    elbow = shoulder + UPPER_ARM_FRACTION * (wrist - shoulder)
    nose = shoulder + _vec(pose.sh, HEAD) + np.array([NOSE_FORWARD, 0.0])
    toe = ankle + _vec(pose.foot, TOE)

    midline = {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "foot_index": toe,
    }
    points: dict[str, np.ndarray] = {"nose": nose}
    for part, base in midline.items():
        offset = SIDE_OFFSET[part]
        points[f"left_{part}"] = base + np.array([-offset, 0.0])
        points[f"right_{part}"] = base + np.array([offset, 0.0])
    points["left_wrist"] = points["left_wrist"] + np.asarray(pose.left_wrist_shift)

    if sway_deg:
        origin = (points["left_wrist"] + points["right_wrist"]) / 2.0
        points = {
            name: origin + _rotate(point - origin, sway_deg) for name, point in points.items()
        }
    return {name: (float(point[0]), float(point[1])) for name, point in points.items()}


def pose_at(keys: Sequence[Keyframe], t: float) -> Pose:
    """The pose at time ``t``: eased interpolation between the two keyframes around it.

    The easing is a smoothstep, so the figure is still at each keyframe: a
    hand plants without a jerk at the end of the plant and leaves without one at
    the start of the lift, which is what stops the smoothest stage of the
    pipeline from reading a corner as movement.
    """
    if t <= keys[0].t:
        return keys[0].pose
    if t >= keys[-1].t:
        return keys[-1].pose
    for first, second in zip(keys, keys[1:], strict=False):
        if first.t <= t < second.t:
            span = second.t - first.t
            if span <= 0.0:
                return second.pose
            fraction = (t - first.t) / span
            eased = fraction * fraction * (3.0 - 2.0 * fraction)
            return _lerp_pose(first.pose, second.pose, eased)
    return keys[-1].pose


def _lerp_pose(first: Pose, second: Pose, fraction: float) -> Pose:
    """Two poses mixed ``fraction`` of the way: the hip's pixel and the angles."""
    hip = tuple(
        a + (b - a) * fraction for a, b in zip(first.hip, second.hip, strict=True)
    )
    shift = tuple(
        a + (b - a) * fraction
        for a, b in zip(first.left_wrist_shift, second.left_wrist_shift, strict=True)
    )
    return Pose(
        hip=(hip[0], hip[1]),
        sh=first.sh + (second.sh - first.sh) * fraction,
        leg=first.leg + (second.leg - first.leg) * fraction,
        arm=first.arm + (second.arm - first.arm) * fraction,
        foot=first.foot + (second.foot - first.foot) * fraction,
        left_wrist_shift=(shift[0], shift[1]),
    )


def sway_envelope(keys: Sequence[Keyframe], t: float) -> float:
    """How much of the sway applies at ``t``: 0 outside the holds, ramped at their ends.

    The swaying intervals are the ones whose *ending* keyframe asked for it, and
    consecutive ones are merged into a single region. The hand step of
    ``hand_step`` is deliberately not one of them: the lean is zero across the
    step, so what changes in those frames is the wrist's own position and
    nothing else — the one movement that case is about is the only movement
    there is.
    """
    regions = _sway_regions(keys)
    if not regions:
        return 0.0
    envelope = 0.0
    for start, end in regions:
        if t < start or t > end:
            continue
        rise = min(max((t - start) / SWAY_RAMP_S, 0.0), 1.0)
        fall = min(max((end - t) / SWAY_RAMP_S, 0.0), 1.0)
        envelope = max(envelope, _smoothstep(rise) * _smoothstep(fall))
    return envelope


def _sway_regions(keys: Sequence[Keyframe]) -> list[tuple[float, float]]:
    """The ``(start, end)`` times the sway is switched on over, merged and sorted."""
    regions: list[tuple[float, float]] = []
    for first, second in zip(keys, keys[1:], strict=False):
        if second.sway:
            if regions and abs(regions[-1][1] - first.t) < 1e-9:
                regions[-1] = (regions[-1][0], second.t)
            else:
                regions.append((first.t, second.t))
    return regions


def _smoothstep(fraction: float) -> float:
    """The ramp both ends of a sway region use, flat where it meets ``0`` and ``1``."""
    clamped = min(max(fraction, 0.0), 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #


def frame_times(index: int, duration_s: float) -> np.ndarray:
    """Variable-frame-rate timestamps for one case, in milliseconds.

    The gap cycle is rotated by the case's index so no two cases share the same
    frame boundaries, and the frames run until the timeline's own duration — so
    a case's last frame is the last one *before* its last keyframe, never past it.
    """
    cycle = GAP_CYCLE_MS[index:] + GAP_CYCLE_MS[:index]
    limit_ms = int(round(duration_s * 1000.0))
    times: list[int] = []
    elapsed, step = 0, 0
    while elapsed < limit_ms:
        times.append(elapsed)
        elapsed += cycle[step % len(cycle)]
        step += 1
    return np.asarray(times, dtype=np.int64)


def generate_case(case: str, seed: int = DEFAULT_SEED) -> postprocess.ClipKeypoints:
    """One synthetic case as a clip the real pipeline can be run on.

    The figure is drawn frame by frame from the case's timeline, the seeded
    jitter of ``gaps_and_noise`` is added, the case's mutations are applied, and
    the coordinates are rounded to the writing precision *before* the pipeline
    ever sees them — so the committed ``input`` is byte-for-byte the input the
    committed ``expected`` was computed from.
    """
    if case not in CASE_SPECS:
        raise ValueError(f"unknown case {case!r}; expected one of {CASES}")
    spec = CASE_SPECS[case]
    times = frame_times(CASES.index(case), spec.duration_s)
    seconds = times.astype(np.float64) / 1000.0
    frames = times.size

    x = np.empty((frames, len(JOINTS)), dtype=np.float64)
    y = np.empty((frames, len(JOINTS)), dtype=np.float64)
    for index, t in enumerate(seconds):
        sway = (
            SWAY_DEG
            * math.sin(2.0 * math.pi * t / SWAY_PERIOD_S)
            * sway_envelope(spec.keys, float(t))
        )
        points = joints_of(pose_at(spec.keys, float(t)), sway)
        for column, name in enumerate(JOINTS):
            x[index, column], y[index, column] = points[name]

    if spec.jitter_px:
        rng = np.random.default_rng([seed, zlib.crc32(case.encode())])
        x = x + rng.normal(0.0, spec.jitter_px, x.shape)
        y = y + rng.normal(0.0, spec.jitter_px, y.shape)

    visibility = np.tile(
        np.array([VISIBILITY[name] for name in JOINTS], dtype=np.float64), (frames, 1)
    )
    detected = np.ones(frames, dtype=bool)
    contact = np.zeros(frames, dtype=bool)
    _apply_mutations(spec.mutations, times, x, y, visibility, detected, contact)

    clip = clip_of(times, x, y, visibility, detected, contact)
    return round_clip(clip)


def _apply_mutations(
    mutations: Sequence[Mutation],
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    visibility: np.ndarray,
    detected: np.ndarray,
    contact: np.ndarray,
) -> None:
    """Damage the frames a case's mutations name, in place.

    Each kind is one of the pipeline's own inputs: a confidence below
    :data:`handstand.postprocess.MIN_VISIBILITY`, a frame the model found
    nobody on (whose coordinates are then NaN, as ``docs/keypoint_schema.md``
    says they must be), a trainer-contact frame, and a joint that jumps further
    than :data:`handstand.postprocess.MAX_SPEED_L_PER_S` allows between two
    frames.
    """
    for mutation in mutations:
        start = int(np.searchsorted(times, int(round(mutation.at_s * 1000.0))))
        start = min(max(start, 0), times.size - 1)
        stop = min(start + max(mutation.count, 1), times.size)
        if mutation.kind == "low_visibility":
            visibility[start:stop] = 0.1
        elif mutation.kind == "undetected":
            detected[start:stop] = False
            x[start:stop] = np.nan
            y[start:stop] = np.nan
            visibility[start:stop] = np.nan
        elif mutation.kind == "contact":
            contact[start:stop] = True
        elif mutation.kind == "outlier":
            column = JOINTS.index(mutation.joint)
            x[start, column] += mutation.delta
        else:
            raise ValueError(f"unknown mutation kind {mutation.kind!r}")


def clip_of(
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    visibility: np.ndarray,
    detected: np.ndarray,
    contact: np.ndarray,
    joints: Sequence[str] = JOINTS,
) -> postprocess.ClipKeypoints:
    """Raw arrays as a :class:`handstand.postprocess.ClipKeypoints`.

    The long table built here is the athlete selection's own schema
    (``docs/keypoint_schema.md``, ``athlete.ATHLETE_COLUMNS``), so the clip
    answers to the real loader's rules: one row per ``(frame, joint)`` in model
    order, the frame's columns repeated on each of its rows.
    """
    names = tuple(joints)
    frames, per_frame = x.shape
    if y.shape != x.shape or visibility.shape != x.shape:
        raise ValueError("x, y and visibility must have the same (frames, joints) shape")
    if per_frame != len(names):
        raise ValueError(f"x has {per_frame} joint columns but {len(names)} joint names")
    if detected.shape != (frames,) or contact.shape != (frames,):
        raise ValueError("detected and contact must have one entry per frame")
    times = np.asarray(times, dtype=np.int64)
    if times.shape != (frames,):
        raise ValueError(f"times must have {frames} entries, got {times.shape}")

    flat = frames * per_frame
    repeat = np.repeat(np.arange(frames, dtype=np.int64), per_frame)
    tile = np.tile(np.asarray(names, dtype=str), frames)
    table = pd.DataFrame(
        {
            "frame_idx": repeat,
            "t_ms": np.repeat(times, per_frame),
            "joint": tile,
            "x": x.reshape(-1),
            "y": y.reshape(-1),
            "z": np.zeros(flat, dtype=np.float64),
            "visibility": visibility.reshape(-1),
            "presence": visibility.reshape(-1),
            "rotated": np.zeros(flat, dtype=bool),
            "detected": np.repeat(detected, per_frame),
            athlete.SCORE_COLUMN: np.repeat(
                np.where(detected, 0.9, np.nan), per_frame
            ),
            athlete.N_PEOPLE_COLUMN: np.repeat(
                np.where(contact, 2, 1).astype(np.int64), per_frame
            ),
            athlete.CONTACT_COLUMN: np.repeat(contact, per_frame),
            athlete.CONTACT_REASON_COLUMN: np.repeat(
                np.where(contact, athlete.REASON_BOX_IOU, athlete.REASON_NONE), per_frame
            ),
        }
    )
    return postprocess.ClipKeypoints(
        table=table,
        frame_idx=np.arange(frames, dtype=np.int64),
        t_ms=times,
        joints=names,
        x=x,
        y=y,
        visibility=visibility,
        detected=detected,
        trainer_contact=contact,
        rotated=np.zeros(frames, dtype=bool),
        has_contact_column=True,
    )


def round_clip(clip: postprocess.ClipKeypoints) -> postprocess.ClipKeypoints:
    """The clip with its coordinates at the writing precision.

    Rounding *before* the run rather than after it is what makes the round-trip
    test exact: the fixture stores ``round(v, 6)``, the JSON reads back the same
    ``float``, and the pipeline therefore sees the identical numbers both times.
    """
    return dataclasses.replace(
        clip,
        x=np.round(clip.x, ROUND_DIGITS),
        y=np.round(clip.y, ROUND_DIGITS),
        visibility=np.round(clip.visibility, ROUND_DIGITS),
    )


def slice_clip(clip: postprocess.ClipKeypoints, frames: int) -> postprocess.ClipKeypoints:
    """The first ``frames`` frames of a clip (all of them when it is shorter)."""
    kept = min(max(int(frames), 0), clip.frames)
    rows = kept * clip.joint_count
    return dataclasses.replace(
        clip,
        table=clip.table.iloc[:rows].reset_index(drop=True),
        frame_idx=clip.frame_idx[:kept].copy(),
        t_ms=clip.t_ms[:kept].copy(),
        x=clip.x[:kept],
        y=clip.y[:kept],
        visibility=clip.visibility[:kept],
        detected=clip.detected[:kept],
        trainer_contact=clip.trainer_contact[:kept],
        rotated=clip.rotated[:kept],
    )


def subset_clip(
    clip: postprocess.ClipKeypoints, joints: Sequence[str] = JOINTS
) -> postprocess.ClipKeypoints:
    """The clip restricted to ``joints``, in that order, table rows and all.

    A real parquet carries all 33 landmarks; a fixture carries the fifteen the
    Swift side has a name for. The rows are already sorted by model order, so
    filtering keeps every frame's rows aligned with the sliced arrays.
    """
    names = tuple(joints)
    missing = [name for name in names if name not in clip.joints]
    if missing:
        raise ValueError(f"{missing} are not joints of this clip ({clip.joints})")
    keep = [clip.joints.index(name) for name in names]
    mask = clip.table["joint"].isin(names).to_numpy()
    return dataclasses.replace(
        clip,
        table=clip.table[mask].reset_index(drop=True),
        joints=names,
        x=clip.x[:, keep],
        y=clip.y[:, keep],
        visibility=clip.visibility[:, keep],
    )


# --------------------------------------------------------------------------- #
# Running the real stages
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FixtureRun:
    """One clip's way through the three stages a fixture records."""

    clip: postprocess.ClipKeypoints
    processed: postprocess.ProcessedClip
    phases: phases.ClipPhases
    features: features.ClipFeatures

    @property
    def hold_count(self) -> int:
        """How many holds the phases stage found."""
        return len(self.features.hold_ids())

    @property
    def pct_valid(self) -> float:
        """Share of the processed samples that ended up with a position."""
        total = self.processed.valid.size
        return 100.0 * float(self.processed.valid.sum()) / total if total else 0.0


def run_pipeline(clip: postprocess.ClipKeypoints, case_id: str) -> FixtureRun:
    """The real stages, called as libraries, on one clip — no files anywhere.

    This is the same chain ``handstand.features``' own ``run_clip`` runs, minus
    the parquet it would write between each step: post-process, then phases from
    the processed trajectory, then features from the same trajectory and the
    phase labels. Nothing touches ``<data_dir>``; the input and the answer both
    stay in memory until :func:`write_fixture` says otherwise.
    """
    processed = postprocess.process_clip(clip)
    length = processed.body_length.total
    labels = phases.classify_clip(
        case_id,
        FIXTURE_SOURCE,
        clip.t_ms,
        processed.x,
        processed.y,
        processed.valid,
        clip.trainer_contact,
        length,
        clip.joints,
    )
    measured = features.extract_clip(
        case_id,
        FIXTURE_SOURCE,
        clip.t_ms,
        processed.x,
        processed.y,
        processed.valid,
        clip.joints,
        length,
        phase=labels.phase_codes,
        hold_id=labels.hold_id,
        trainer_contact=clip.trainer_contact,
    )
    return FixtureRun(clip=clip, processed=processed, phases=labels, features=measured)


# --------------------------------------------------------------------------- #
# The fixture document
# --------------------------------------------------------------------------- #


def _num(value: object) -> float | None:
    """One float at the writing precision: ``null`` for ``None`` and for NaN."""
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return None if not math.isfinite(number) else round(number, ROUND_DIGITS)


def _json_value(value: object) -> object:
    """Any pipeline value as JSON: bools stay bools, ints stay ints, NaN is ``null``."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _num(value)
    raise TypeError(f"cannot write {type(value).__name__} to a fixture")


def input_frames(clip: postprocess.ClipKeypoints) -> list[dict[str, object]]:
    """The clip as the fixture's ``input``: one row per frame, joints by name."""
    rows: list[dict[str, object]] = []
    for index in range(clip.frames):
        joints: dict[str, list[float | None]] = {}
        for column, name in enumerate(clip.joints):
            joints[name] = [
                _num(clip.x[index, column]),
                _num(clip.y[index, column]),
                _num(clip.visibility[index, column]),
            ]
        rows.append(
            {
                "frame_idx": int(clip.frame_idx[index]),
                "t_ms": int(clip.t_ms[index]),
                "detected": bool(clip.detected[index]),
                "trainer_contact": bool(clip.trainer_contact[index]),
                "joints": joints,
            }
        )
    return rows


def expected_output(run: FixtureRun) -> dict[str, object]:
    """The ``expected`` section: everything the three stages answered, per frame.

    The ``valid`` and ``filled`` arrays of a post-process frame are in
    ``meta.joint_names`` order, because they are per joint and a name-keyed
    object of them would be twice the size for no gain; ``joints`` stays keyed
    by name because that is the shape a person reads. Everything else is one
    row per frame in frame order, next to the input's own rows.
    """
    processed = run.processed
    body = processed.body_length
    post_frames: list[dict[str, object]] = []
    for index in range(run.clip.frames):
        post_frames.append(
            {
                "valid": [bool(value) for value in processed.valid[index]],
                "filled": [bool(value) for value in processed.filled[index]],
                "joints": {
                    name: [_num(processed.x[index, column]), _num(processed.y[index, column])]
                    for column, name in enumerate(run.clip.joints)
                },
            }
        )
    phase_rows = [
        {"phase": str(phase), "hold_id": int(hold)}
        for phase, hold in zip(run.phases.phase_codes, run.phases.hold_id, strict=True)
    ]
    feature_rows = [
        {name: _json_value(run.features.value(name)[index]) for name in FEATURE_COLUMNS}
        for index in range(run.clip.frames)
    ]
    hold_rows = [
        {key: _json_value(value) for key, value in row.items()}
        for row in features.hold_rows(run.features)
    ]
    return {
        "postprocess": {
            "body_length": {
                "usable": bool(body.usable),
                "reason": body.reason or None,
                "torso_px": _num(body.torso),
                "thigh_px": _num(body.thigh),
                "shin_px": _num(body.shin),
                "total_px": _num(body.total),
                "frames": {name: int(count) for name, count in body.frames().items()},
            },
            "frames": post_frames,
        },
        "phases": phase_rows,
        "features": feature_rows,
        "hold_summary": hold_rows,
    }


def packages() -> dict[str, str]:
    """The versions the fixture was produced with, recorded so a diff means something."""
    versions = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for name in ("handstand", "numpy", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _meta(
    case: str,
    mode: str,
    run: FixtureRun,
    *,
    seed: int | None,
    display: tuple[int, int] | None,
    nominal_l: float | None,
) -> dict[str, object]:
    """The ``meta`` section: provenance, scale, schema and the tolerances to compare at."""
    return {
        "case": case,
        "mode": mode,
        "generator": "handstand.golden",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "display": None if display is None else {"width": display[0], "height": display[1]},
        "L": _num(nominal_l) if nominal_l is not None else None,
        "body_length_px": _num(run.processed.body_length.total),
        "frame_count": int(run.clip.frames),
        "hold_count": int(run.hold_count),
        "joint_names": list(run.clip.joints),
        "feature_columns": list(FEATURE_COLUMNS),
        "packages": packages(),
        "tolerances": dict(TOLERANCES),
    }


def build_fixture(case: str, *, seed: int = DEFAULT_SEED) -> dict[str, object]:
    """One synthetic case as the whole fixture document: ``meta``, ``input``, ``expected``."""
    clip = generate_case(case, seed=seed)
    run = run_pipeline(clip, case)
    return {
        "meta": _meta(
            case,
            "synthetic",
            run,
            seed=seed,
            display=(DISPLAY_WIDTH, DISPLAY_HEIGHT),
            nominal_l=TORSO + THIGH + SHIN,
        ),
        "input": input_frames(clip),
        "expected": expected_output(run),
    }


def real_fixture(
    clip_id: str,
    *,
    data: str | pathlib.Path | None = None,
    rotate: str = postprocess.DEFAULT_ROTATE,
    source: str = postprocess.DEFAULT_SOURCE,
    frames: int = REAL_MAX_FRAMES,
) -> dict[str, object]:
    """The same JSON for a **real** clip — never for a repository path.

    Reads the athlete selection's parquet of ``clip_id`` (read-only, from the
    shared data directory), keeps its first ``frames`` frames and the fifteen
    tracked joints, and runs the identical chain over it. The document is marked
    ``mode: "real"``, which is how a test can tell the two apart without reading
    a single coordinate — the guard that keeps real keypoints out of ``swift/``.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    path = root / "keypoints" / postprocess.input_dirname(source) / rotate / f"{clip_id}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"no athlete keypoints for {clip_id!r} at {path}\n"
            "generate them first with:\n"
            f"  cd pipeline && uv run python -m handstand.athlete --rotate {rotate}"
        )
    clip = round_clip(subset_clip(slice_clip(postprocess.read_clip(path), frames), JOINTS))
    run = run_pipeline(clip, clip_id)
    return {
        "meta": _meta(
            clip_id,
            "real",
            run,
            seed=None,
            display=_sidecar_display(path),
            nominal_l=None,
        ),
        "input": input_frames(clip),
        "expected": expected_output(run),
    }


def _sidecar_display(parquet_path: pathlib.Path) -> tuple[int, int] | None:
    """The clip's display size, from its own sidecar or from the one it was read from.

    The athlete-selection sidecar records the selection rather than the frame
    size; the ``source_parquet`` it names is the runner's output, and *that*
    sidecar carries ``display_width``/``display_height``. ``None`` when neither
    has it: a fixture without a display size is still a fixture, and guessing
    one would be worse than saying nothing.
    """
    sidecar = parquet_path.with_suffix(".json")
    for candidate in (sidecar, *_runner_sidecars(parquet_path, sidecar)):
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            width, height = payload.get("display_width"), payload.get("display_height")
            if isinstance(width, int) and isinstance(height, int):
                return width, height
    return None


def _runner_sidecars(parquet_path: pathlib.Path, own: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """The sidecars of the run this parquet was read from, if it says which that was."""
    try:
        payload = json.loads(own.read_text())
    except (OSError, json.JSONDecodeError):
        return ()
    source = payload.get("source_parquet") if isinstance(payload, Mapping) else None
    if not isinstance(source, str) or len(parquet_path.parents) < 3:
        return ()
    relative = pathlib.Path(source.removeprefix("keypoints/"))
    return (parquet_path.parents[2] / relative.with_suffix(".json"),)


# --------------------------------------------------------------------------- #
# Round-tripping
# --------------------------------------------------------------------------- #


def clip_from_frames(
    frames: Sequence[Mapping[str, object]], joints: Sequence[str] = JOINTS
) -> postprocess.ClipKeypoints:
    """The fixture's ``input`` back into a clip: JSON is the source of truth here.

    This is the path the round-trip test takes — a reader that only ever sees the
    file must be able to rebuild exactly the clip the ``expected`` was computed
    from, so the file rather than the generator is what the parity check trusts.
    A joint named in ``joints`` but missing from a frame, or a name ``joints``
    does not know, is an error rather than a NaN: a silently padded clip would
    produce a plausible-looking wrong answer.
    """
    names = tuple(joints)
    frames_array = np.asarray(frames, dtype=object)
    count = frames_array.size
    x = np.full((count, len(names)), np.nan, dtype=np.float64)
    y = np.full((count, len(names)), np.nan, dtype=np.float64)
    visibility = np.full((count, len(names)), np.nan, dtype=np.float64)
    frame_idx = np.zeros(count, dtype=np.int64)
    t_ms = np.zeros(count, dtype=np.int64)
    detected = np.zeros(count, dtype=bool)
    contact = np.zeros(count, dtype=bool)

    for index, row in enumerate(frames_array):
        data = dict(row)  # type: ignore[arg-type]
        frame_idx[index] = int(data["frame_idx"])  # type: ignore[arg-type]
        t_ms[index] = int(data["t_ms"])  # type: ignore[arg-type]
        detected[index] = bool(data["detected"])  # type: ignore[arg-type]
        contact[index] = bool(data["trainer_contact"])  # type: ignore[arg-type]
        given = data["joints"]
        if not isinstance(given, Mapping):
            raise ValueError(f"frame {index}: joints must be an object of [x, y, visibility]")
        unknown = [name for name in given if name not in names]
        if unknown:
            raise ValueError(f"frame {index}: joint(s) {unknown} are not in {list(names)}")
        for column, name in enumerate(names):
            if name not in given:
                raise ValueError(f"frame {index}: no entry for joint {name!r}")
            values = list(given[name])  # type: ignore[arg-type]
            if len(values) != 3:
                raise ValueError(f"frame {index}, joint {name!r}: expected [x, y, visibility]")
            for target, value in zip((x, y, visibility), values, strict=True):
                target[index, column] = np.nan if value is None else float(value)
    return clip_of(t_ms, x, y, visibility, detected, contact, names)


def recompute(fixture: Mapping[str, object]) -> dict[str, object]:
    """Re-run the pipeline from a fixture's ``input`` and return its ``expected``."""
    meta = fixture["meta"]
    if not isinstance(meta, Mapping):
        raise ValueError("fixture has no meta object")
    case = str(meta.get("case", "fixture"))
    joint_names = meta.get("joint_names", JOINTS)
    if not isinstance(joint_names, list):
        raise ValueError("meta.joint_names must be an array")
    clip = clip_from_frames(fixture["input"], [str(name) for name in joint_names])  # type: ignore[arg-type]
    return expected_output(run_pipeline(clip, case))


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def fixture_text(fixture: Mapping[str, object]) -> str:
    """The fixture as JSON: objects indented, and one row of a list per line.

    A fixture is thousands of rows, so the rows are not broken up — one frame is
    one line, greppable and diffable — while the structure around them is
    indented a space at a time. ``json.loads`` of this is the fixture again;
    ``tests/test_golden.py`` checks that too.
    """
    return _dump(fixture, 0) + "\n"


def _dump(value: object, level: int) -> str:
    """One JSON value, recursively.

    The layout is the compromise a fixture needs: objects are indented a space
    at a time so the sections are readable, but a *row* — one frame, one hold —
    is written on a single line, and a list of numbers or booleans is written
    inline. A per-frame row is hundreds of tokens; broken across lines it is
    unreadable and a fifth bigger, on one line it is greppable.
    """
    pad = " " * level
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        items = [
            f'{pad} {json.dumps(str(key))}: {_dump(item, level + 1)}'
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(_is_scalar(item) for item in value):
            return "[" + ", ".join(_compact(item) for item in value) + "]"
        if all(isinstance(item, Mapping) for item in value):
            rows = ",\n".join(f"{pad} {_compact(item)}" for item in value)
            return "[\n" + rows + "\n" + pad + "]"
        items = [f"{pad} {_dump(item, level + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return _compact(value)


def _is_scalar(value: object) -> bool:
    """Is ``value`` written as a bare JSON token rather than a structure?"""
    return value is None or isinstance(value, (str, int, float, bool))


def _compact(value: object) -> str:
    """One scalar or one whole row as JSON on a single line, NaN forbidden."""
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def write_fixture(directory: str | pathlib.Path, fixture: Mapping[str, object]) -> pathlib.Path:
    """Write ``<case>.json`` into ``directory``, atomically, and return its path.

    Atomic because a fixture half-written is a fixture a Swift test will read:
    the temporary file is renamed only once its bytes are all there.
    """
    path = pathlib.Path(directory) / f"{fixture['meta']['case']}.json"  # type: ignore[index]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(fixture_text(fixture))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_fixture(path: str | pathlib.Path) -> dict[str, object]:
    """Read one fixture back. Raises on a file that is not JSON, as it should."""
    payload = json.loads(pathlib.Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: a fixture is a JSON object")
    return payload


def fixture_size(path: str | pathlib.Path) -> int:
    """The file's size in bytes — the number the 2 MB budget is checked against."""
    return pathlib.Path(path).stat().st_size


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The argument parser of ``python -m handstand.golden``."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.golden",
        description=(
            "Generate golden fixtures: synthetic clips through the real pipeline, "
            "for the Swift ports to be checked against frame by frame."
        ),
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help=(
            "where the committed synthetic fixtures go (default: "
            f"{DEFAULT_OUT.as_posix()} under the repository root)"
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=CASES,
        default=None,
        metavar="CASE",
        help=f"generate only these cases (default: all of {', '.join(CASES)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed of the synthetic generator (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--real",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help=(
            "local-only: write a fixture for these real clips instead of the "
            "synthetic cases, into --real-out (default: <data_dir>/golden_real); "
            "refused anywhere inside the repository, which is public"
        ),
    )
    parser.add_argument(
        "--real-out",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help=f"where --real writes (default: <data_dir>/{REAL_SUBDIR})",
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory holding keypoints/ (default: $HANDSTAND_DATA)",
    )
    parser.add_argument(
        "--rotate",
        default=postprocess.DEFAULT_ROTATE,
        choices=postprocess.ROTATE_MODES,
        help="rotation mode of the real keypoints --real reads",
    )
    parser.add_argument(
        "--source",
        choices=postprocess.SOURCES,
        default=postprocess.DEFAULT_SOURCE,
        help="which pose model's athlete keypoints --real reads",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=REAL_MAX_FRAMES,
        metavar="N",
        help=f"how many frames of each real clip to keep (default: {REAL_MAX_FRAMES})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the fixtures and print one summary line per clip. See the module docstring."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be >= 1")

    if args.real is not None:
        root = pathlib.Path(args.data) if args.data is not None else data_dir()
        target = args.real_out if args.real_out is not None else root / REAL_SUBDIR
        try:
            target = require_outside_repo(target)
        except ValueError as error:
            print(f"golden: {error}")
            return 2
        failures = 0
        for clip_id in args.real:
            try:
                fixture = real_fixture(
                    clip_id,
                    data=root,
                    rotate=args.rotate,
                    source=args.source,
                    frames=args.frames,
                )
                path = write_fixture(target, fixture)
            except (FileNotFoundError, ValueError, KeyError) as error:
                failures += 1
                print(f"fail  {clip_id}: {error}")
                continue
            print(_line(fixture, path), flush=True)
        return 1 if failures else 0

    target = args.out if args.out is not None else default_out_dir()
    cases = tuple(args.only) if args.only else CASES
    for case in cases:
        fixture = build_fixture(case, seed=args.seed)
        path = write_fixture(target, fixture)
        print(_line(fixture, path), flush=True)
    print(f"wrote {len(cases)} fixture(s) to {target}")
    return 0


def _line(fixture: Mapping[str, object], path: pathlib.Path) -> str:
    """The one summary line per fixture: frames, holds, how much survived, how big."""
    meta = fixture["meta"]
    expected = fixture["expected"]
    if not isinstance(meta, Mapping) or not isinstance(expected, Mapping):
        raise ValueError(f"{path}: not a fixture document")
    postprocess = expected["postprocess"]
    if not isinstance(postprocess, Mapping) or not isinstance(postprocess.get("frames"), list):
        raise ValueError(f"{path}: fixture has no post-process frames")
    samples = valid = 0
    for row in postprocess["frames"]:
        flags = row.get("valid") if isinstance(row, Mapping) else None
        if isinstance(flags, list):
            samples += len(flags)
            valid += sum(1 for flag in flags if flag)
    pct = 100.0 * valid / samples if samples else 0.0
    return (
        f"{meta['case']:<16} mode={meta['mode']} frames={meta['frame_count']} "
        f"holds={meta['hold_count']} valid={pct:.1f}% "
        f"{fixture_size(path) / 1024:.0f} KiB -> {path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
