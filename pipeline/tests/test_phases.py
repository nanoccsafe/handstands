"""Tests for :mod:`handstand.phases`.

Nothing here touches the real handstand videos. A clip is a synthetic
trajectory written in the **body frame** — the wrist midpoint is the origin, the
ankle midpoint is ``(u_ankle, v_ankle)`` body lengths away from it — and folded
into the same processed long parquet schema the pipeline writes, so the tests
exercise the real loader, the real thresholds, the real CSV and the real atomic
writes on input small enough to reason about exactly.

Writing the trajectory in the body frame is what makes these tests readable: a
held handstand is ``v_ankle = 0.75``, a person standing on the floor is
``v_ankle = -0.2``, and a hand step is a wrist that moves 0.3 of a body length.
The pixel scale (:data:`LENGTH`) is a known number, so a test can say "0.3 L" and
mean it.
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

from handstand import athlete, overlay
from handstand import phases as ph
from handstand import pose_mediapipe as pm

CLIP_ID = "abc123def456"
#: The clip's body length in pixels. Any positive number works; 300 keeps the
#: coordinates in a comfortable range and makes 0.1 L exactly 30 px.
LENGTH = 300.0
#: The frame interval of the synthetic clips, in milliseconds. Real clips are
#: variable frame rate, so the tests that care about time say so explicitly.
FRAME_MS = 33


# --------------------------------------------------------------------------- #
# A synthetic body
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Pose:
    """One frame of a synthetic trajectory, written in the body frame.

    ``u_ankle``/``v_ankle`` and ``v_hip`` are body lengths from the wrist
    midpoint, with ``v`` positive upwards, exactly as
    :func:`handstand.bodyframe.to_body_frame` defines them — so ``v_ankle=0.75``
    is a body standing on its hands and ``v_ankle=-0.2`` is a person standing on
    the floor. ``wrist_x``/``wrist_y`` are the wrist midpoint in *pixels*, which
    is how a hand step is written: the hands move and the body goes with them.
    """

    u_ankle: float = 0.0
    v_ankle: float = 0.0
    v_hip: float = 0.0
    wrist_x: float = 200.0
    wrist_y: float = 500.0
    #: How far apart the two wrists are, in body lengths. One wrist alone is a
    #: side-on clip, which the real data has plenty of.
    hand_span_l: float = 0.2
    #: The wrist that moves, as a fraction of the span: 0.0 is the left one
    #: moving, 1.0 the right one, 0.5 both. ``None`` moves neither.
    step_side: float | None = None
    #: An extra offset for one wrist only, in pixels, so a test can move a single
    #: hand the way a hand step does.
    wrist_offset_px: float = 0.0
    valid: bool = True
    contact: bool = False


def _frames(
    poses: Sequence[Pose], t_ms: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A pose list as the ``(t_ms, x, y, valid, contact)`` arrays of a clip.

    Only the six joints a phase is read off carry a position; the rest of the
    schema is parked on the hip midpoint, which is all a phase detector looks
    at and enough for the file to be a plausible processed parquet.
    """
    frames = len(poses)
    times = (
        np.arange(frames, dtype=np.int64) * FRAME_MS
        if t_ms is None
        else np.asarray(t_ms, dtype=np.int64)
    )
    if times.size != frames:
        raise ValueError("t_ms must hold one timestamp per frame")
    names = pm.JOINT_NAMES
    x = np.full((frames, len(names)), np.nan)
    y = np.full((frames, len(names)), np.nan)
    valid = np.zeros((frames, len(names)), dtype=bool)

    def column(joint: str) -> int:
        return names.index(joint)

    for index, pose in enumerate(poses):
        centre_x = pose.wrist_x
        centre_y = pose.wrist_y
        half = pose.hand_span_l * LENGTH / 2.0
        ankles = (centre_x + pose.u_ankle * LENGTH, centre_y - pose.v_ankle * LENGTH)
        hips = (centre_x, centre_y - pose.v_hip * LENGTH)
        for name, point in (
            ("left_wrist", (centre_x - half + pose.wrist_offset_px, centre_y)),
            ("right_wrist", (centre_x + half + pose.wrist_offset_px, centre_y)),
            ("left_ankle", (ankles[0] - 0.05 * LENGTH, ankles[1])),
            ("right_ankle", (ankles[0] + 0.05 * LENGTH, ankles[1])),
            ("left_hip", (hips[0] - 0.1 * LENGTH, hips[1])),
            ("right_hip", (hips[0] + 0.1 * LENGTH, hips[1])),
            ("left_shoulder", (hips[0] - 0.12 * LENGTH, hips[1] + 0.25 * LENGTH)),
            ("right_shoulder", (hips[0] + 0.12 * LENGTH, hips[1] + 0.25 * LENGTH)),
            ("left_knee", (hips[0] - 0.06 * LENGTH, (hips[1] + ankles[1]) / 2.0)),
            ("right_knee", (hips[0] + 0.06 * LENGTH, (hips[1] + ankles[1]) / 2.0)),
        ):
            x[index, column(name)] = point[0]
            y[index, column(name)] = point[1]
            valid[index, column(name)] = pose.valid
        for name in names:
            if not valid[index, column(name)]:
                x[index, column(name)] = hips[0]
                y[index, column(name)] = hips[1]
                valid[index, column(name)] = pose.valid
        if pose.step_side is not None and pose.step_side < 1.0:
            x[index, column("left_wrist")] -= pose.wrist_offset_px
    return times, x, y, valid, np.array([pose.contact for pose in poses], dtype=bool)


def classify(
    poses: Sequence[Pose], *, t_ms: Sequence[int] | None = None, **kwargs
) -> ph.ClipPhases:
    """Classify a pose list straight, with no file in the way."""
    times, x, y, valid, contact = _frames(poses, t_ms)
    return ph.classify_clip(
        CLIP_ID,
        "mediapipe",
        times,
        x,
        y,
        valid,
        contact,
        kwargs.pop("body_length", LENGTH),
        pm.JOINT_NAMES,
        **kwargs,
    )


def phases_of(result: ph.ClipPhases) -> list[str]:
    """The phase of each frame, as names, with runs collapsed for readability."""
    codes = [str(name) for name in result.phase_codes]
    return [code for index, code in enumerate(codes) if index == 0 or code != codes[index - 1]]


def codes_in(result: ph.ClipPhases, phase: str) -> list[int]:
    """The frame indices one phase was given."""
    return [int(index) for index, name in enumerate(result.phase_codes) if name == phase]


# --------------------------------------------------------------------------- #
# The trajectories the tests are about
# --------------------------------------------------------------------------- #

#: A body standing on the floor: the hands are down by the hips, so the ankle
#: midpoint is *below* the wrist midpoint and the body is not above its hands.
STANDING = Pose(v_ankle=-0.2, v_hip=-0.3, wrist_x=200.0, wrist_y=400.0)

#: A held handstand: the feet 0.75 body lengths above the hands, straight, the
#: hands planted and still.
HOLDING = Pose(u_ankle=0.02, v_ankle=0.75, v_hip=0.35, wrist_x=200.0, wrist_y=500.0)

#: Hands on the floor with the body above them: the moment before a kick-up. The
#: feet are still well below the :data:`handstand.phases.INVERTED_MIN` that would
#: make it a hold.
PLANTED = Pose(v_ankle=0.2, v_hip=0.15, wrist_x=200.0, wrist_y=500.0)

#: The top of a kick-up that never arrived: hands planted, legs swinging, feet
#: never 0.6 body lengths above the hands.
FAILED = Pose(v_ankle=0.45, v_hip=0.25, wrist_x=200.0, wrist_y=500.0)


def stand(frames: int) -> list[Pose]:
    """``frames`` of standing still on the floor."""
    return [dataclasses.replace(STANDING) for _ in range(frames)]


def hold(frames: int, **changes: object) -> list[Pose]:
    """``frames`` of a held handstand, the hands planted and not moving.

    ``changes`` are applied to every frame, which is how a test says "the hands
    are still planted, but over there": the body stays where the hands went.
    """
    return [dataclasses.replace(HOLDING, **changes) for _ in range(frames)]


def swing(start: Pose, end: Pose, frames: int) -> list[Pose]:
    """The body from ``start`` to ``end``, one frame at a time, linearly.

    A kick-up on the real thing is not linear, but the state machine only asks
    whether the feet are going up and whether the hands are the support, and a
    straight line between two poses answers both honestly. Two poses with the
    same wrist midpoint swing *around* the hands, which is what a kick-up does;
    two with different ones move the hands, which is what planting or standing
    up does.
    """
    if frames < 2:
        raise ValueError("a swing needs at least two frames")
    out: list[Pose] = []
    for index in range(frames):
        weight = index / (frames - 1)
        out.append(
            dataclasses.replace(
                start,
                u_ankle=start.u_ankle + weight * (end.u_ankle - start.u_ankle),
                v_ankle=start.v_ankle + weight * (end.v_ankle - start.v_ankle),
                v_hip=start.v_hip + weight * (end.v_hip - start.v_hip),
                wrist_x=start.wrist_x + weight * (end.wrist_x - start.wrist_x),
                wrist_y=start.wrist_y + weight * (end.wrist_y - start.wrist_y),
            )
        )
    return out


def attempt(hold_frames: int = 90) -> list[Pose]:
    """A whole attempt, in six moves.

    Stand still, put the hands on the floor, swing the legs up around them, hold,
    come back down, rest on the hands for a moment, and take the hands off the
    floor. The five phases of the issue are one of these, which is why every
    other test here is a variation on it.
    """
    return (
        stand(10)
        + swing(STANDING, PLANTED, 6)
        + swing(PLANTED, HOLDING, 12)
        + hold(hold_frames)
        + swing(HOLDING, FAILED, 12)
        + [dataclasses.replace(FAILED) for _ in range(8)]
        + swing(FAILED, STANDING, 8)
    )


def write_processed(
    root: pathlib.Path,
    poses: Sequence[Pose],
    *,
    clip_id: str = CLIP_ID,
    body_length: float = LENGTH,
    usable: bool = True,
    t_ms: Sequence[int] | None = None,
) -> pathlib.Path:
    """Write a synthetic processed parquet + sidecar, and return the parquet path.

    The parquet is the schema :mod:`handstand.postprocess` writes — the shared
    long one plus ``valid`` and ``filled`` — and the sidecar carries the body
    length, so the tests run against exactly what a real run reads.
    """
    times, x, y, valid, contact = _frames(poses, t_ms)
    rows = []
    for index in range(times.size):
        for joint in range(x.shape[1]):
            known = bool(valid[index, joint])
            rows.append(
                {
                    "frame_idx": index,
                    "t_ms": int(times[index]),
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


# --------------------------------------------------------------------------- #
# The signals
# --------------------------------------------------------------------------- #


def test_signals_read_a_held_handstand() -> None:
    result = classify(attempt())
    signals = result.signals
    holds = codes_in(result, "hold")
    middle = holds[len(holds) // 2]

    assert bool(signals.known.all())
    assert bool(signals.inverted[middle])
    assert signals.v_ankle_mid[middle] == pytest.approx(HOLDING.v_ankle, abs=0.01)
    assert signals.v_hip_mid[middle] == pytest.approx(HOLDING.v_hip, abs=0.01)
    # Straight up: the angle is the wrist->ankle vector's lean from vertical, and
    # this body leans a little over a degree.
    expected = math.degrees(math.atan2(0.02, 0.75))
    assert signals.body_angle_deg[middle] == pytest.approx(expected, abs=0.2)
    assert bool(signals.hands_down[middle])
    assert signals.wrist_speed_l_per_s[middle] == pytest.approx(0.0, abs=1e-9)
    assert signals.wrist_step_l[middle] == pytest.approx(0.0, abs=1e-9)
    assert not bool(signals.hand_step[middle])
    assert abs(float(signals.legs_velocity_l_per_s[middle])) < ph.LEG_VELOCITY_MIN_L_PER_S
    assert not bool(signals.legs_rising[middle])
    assert not bool(signals.legs_falling[middle])


def test_a_body_on_the_floor_is_not_inverted() -> None:
    signals = classify(stand(10)).signals

    assert not bool(signals.inverted.any())
    assert not bool(signals.hands_down.any())
    assert signals.v_ankle_mid[0] == pytest.approx(STANDING.v_ankle, abs=0.01)
    # The angle of a vector pointing *down* is more than 90 degrees from up.
    assert abs(float(signals.body_angle_deg[0])) > 90.0


def test_hands_low_needs_the_body_above_the_hands() -> None:
    # The feet above the hands but the hips not: the athlete is on their knees
    # with their hands on the floor, which is a kick-up, not a stand-up.
    kneeling = Pose(v_ankle=0.5, v_hip=0.02, wrist_y=500.0)
    assert bool(classify([kneeling] * 10).signals.hands_low[0]) is False
    # The hips above the hands and the feet below: an L-shape, hands planted.
    lying = Pose(v_ankle=-0.1, v_hip=0.2, wrist_y=500.0)
    assert bool(classify([lying] * 10).signals.hands_low[0]) is True


def test_the_legs_rising_and_falling_with_the_feet() -> None:
    up = swing(PLANTED, HOLDING, 20)
    rising = classify(up).signals
    assert bool(rising.legs_rising[-1])
    assert rising.legs_velocity_l_per_s[-1] > ph.LEG_VELOCITY_MIN_L_PER_S

    # The same body coming down: the same speed, the other sign.
    falling = classify(list(reversed(up))).signals
    assert bool(falling.legs_falling[-1])
    assert falling.legs_velocity_l_per_s[-1] < -ph.LEG_VELOCITY_MIN_L_PER_S


def test_a_moving_wrist_is_not_a_planted_hand() -> None:
    # The whole body slides sideways at 2 L/s: the hands are travelling, not
    # planted, whatever the geometry says.
    drifting = [
        dataclasses.replace(HOLDING, wrist_x=200.0 + index * 2.0 * LENGTH / 30.0)
        for index in range(15)
    ]
    signals = classify(drifting).signals
    assert signals.wrist_speed_l_per_s[-1] == pytest.approx(2.0, rel=0.05)
    # The window is the last 0.2 s, which is seven frames of 33 ms, so the step
    # is a little over 0.4 L: 0.5 L/s for a fifth of a second, plus a frame.
    assert signals.wrist_step_l[-1] == pytest.approx(2.0 * 0.231, rel=0.05)
    assert bool(signals.hand_step[-1])
    assert not bool(signals.hands_down[-1])


def test_a_speed_that_cannot_be_measured_is_not_a_moving_hand() -> None:
    """The first frames of a clip, and the frames after a gap, have no window."""
    signals = classify(hold(10)).signals
    assert math.isnan(float(signals.wrist_speed_l_per_s[0]))
    # NaN is not evidence of movement, so the hold condition can still pass.
    assert bool(signals.hands_down[0])


def test_one_visible_wrist_is_enough() -> None:
    """A side-on clip: the far wrist is never reported, and must not block a hold."""
    poses = [dataclasses.replace(HOLDING, hand_span_l=0.0) for _ in range(30)]
    result = classify(poses)
    assert int(result.signals.known.sum()) == len(poses)
    assert result.hold_count == 1


def test_a_frame_nobody_could_see_is_unknown_with_a_reason() -> None:
    poses = hold(4) + [dataclasses.replace(HOLDING, valid=False)] + hold(4)
    result = classify(poses)
    signals = result.signals

    assert not bool(signals.known[4])
    assert signals.unknown_reason[4] == ph.REASON_WRISTS
    assert str(result.phase_codes[4]) == "unknown"

    # ... and a frame the athlete selection flagged is unknown whatever its
    # keypoints say.
    contact = hold(4) + [dataclasses.replace(HOLDING, contact=True)] + hold(4)
    flagged = classify(contact)
    assert flagged.signals.unknown_reason[4] == ph.REASON_TRAINER
    assert str(flagged.phase_codes[4]) == "unknown"


def test_a_clip_with_no_body_length_is_all_unknown() -> None:
    result = classify(hold(30), body_length=math.nan)
    assert not result.usable
    assert "no body length" in result.unusable_reason
    assert set(result.phase_codes) == {"unknown"}
    assert result.hold_count == 0
    assert result.phase_counts()["hold"] == 0


def test_frame_signals_refuses_a_scale_it_cannot_measure_in() -> None:
    times, x, y, valid, contact = _frames(hold(3))
    with pytest.raises(ValueError, match="body_length"):
        ph.frame_signals(times, x, y, valid, contact, pm.JOINT_NAMES, 0.0)
    with pytest.raises(ValueError, match="joint columns"):
        ph.frame_signals(times, x, y, valid, contact, ("left_wrist",), LENGTH)
    with pytest.raises(ValueError, match="same \\(frames, joints\\) shape"):
        ph.frame_signals(times, x, y, valid[:, :3], contact, pm.JOINT_NAMES, LENGTH)
    with pytest.raises(ValueError, match="one entry per frame"):
        ph.frame_signals(times, x, y, valid, contact[:-1], pm.JOINT_NAMES, LENGTH)


# --------------------------------------------------------------------------- #
# window_motion and velocity
# --------------------------------------------------------------------------- #


def test_window_motion_measures_the_window_before_each_frame() -> None:
    times = np.array([0.0, 0.1, 0.25, 0.4])
    x = np.array([0.0, 0.5, 1.0, 1.5])
    y = np.zeros(4)
    known = np.ones(4, dtype=bool)

    step_l, speed = ph.window_motion(times, x, y, known, 2.0, 0.2)

    # Frame 2 is 0.25 s in; the last frame at or before 0.05 s is frame 0, so the
    # displacement is 1.0 px over 0.25 s.
    assert step_l[2] == pytest.approx(0.5)
    assert speed[2] == pytest.approx(2.0)
    # Frames 0 and 1 ask about a window that reaches back before the clip starts.
    assert math.isnan(float(step_l[0]))
    assert math.isnan(float(step_l[1]))
    # Frame 3 is 0.4 s in; the last frame at or before 0.2 s is frame 1, so the
    # window is 0.3 s long and holds the whole second pixel.
    assert step_l[3] == pytest.approx(0.5)
    assert speed[3] == pytest.approx(0.5 / 0.3)
    with pytest.raises(ValueError, match="window_s"):
        ph.window_motion(times, x, y, known, 2.0, 0.0)


def test_velocity_is_centred_and_needs_a_known_sample_on_each_side() -> None:
    times = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    values = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    known = np.ones(5, dtype=bool)

    velocity = ph.velocity_l_per_s(times, values, known, 0.1)
    assert velocity[2] == pytest.approx(10.0)
    # The two ends of the clip are measured against the one side they have: the
    # value is flat either side of frame 0 and of frame 4, so neither moves.
    assert velocity[0] == pytest.approx(0.0)
    assert velocity[-1] == pytest.approx(0.0)
    # A frame the module cannot see into has no velocity, and neither has the
    # frame compared against one.
    blind = known.copy()
    blind[2] = False
    blind[3] = False
    assert math.isnan(float(ph.velocity_l_per_s(times, values, blind, 0.1)[2]))
    assert math.isnan(float(ph.velocity_l_per_s(times, values, blind, 0.1)[3]))

    with pytest.raises(ValueError, match="span_s"):
        ph.velocity_l_per_s(times, values, known, -1.0)


# --------------------------------------------------------------------------- #
# Holds
# --------------------------------------------------------------------------- #


def test_a_run_shorter_than_the_minimum_is_not_a_hold() -> None:
    times = np.arange(13) * 0.05
    # Two runs of 0.1 s and 0.05 s, separated by a 0.35 s break: long enough that
    # the runs stay separate, short enough that a hold can sit inside it.
    holding = np.array([True] * 3 + [False] * 8 + [True] * 2)
    known = np.ones(13, dtype=bool)

    runs = ph.hold_runs(times, holding, known)

    assert runs == ()
    # ... and with a minimum of a twentieth of a second both count.
    kept = ph.hold_runs(times, holding, known, min_hold_s=0.05)
    assert [(run.start, run.end, run.hold_id) for run in kept] == [(0, 2, 0), (11, 12, 1)]
    with pytest.raises(ValueError, match="min_hold_s"):
        ph.hold_runs(times, holding, known, min_hold_s=-1.0)


def test_a_short_break_does_not_end_a_hold_and_a_long_one_does() -> None:
    times = np.arange(0, 40) * 0.05  # 2 s
    known = np.ones(40, dtype=bool)
    # Frames 10-14 fail for 0.2 s, well inside HOLD_BREAK_MAX_S.
    holding = np.ones(40, dtype=bool)
    holding[10:15] = False

    bridged = ph.hold_runs(times, holding, known)
    assert len(bridged) == 1
    assert (bridged[0].start, bridged[0].end) == (0, 39)
    assert bridged[0].hold_id == 0

    # Frames 20-27 fail for 0.4 s: past the window, so the hold ends there.
    holding = np.ones(40, dtype=bool)
    holding[10:15] = False
    holding[20:28] = False
    runs = ph.hold_runs(times, holding, known)
    assert [(run.start, run.end, run.hold_id) for run in runs] == [(0, 19, 0), (28, 39, 1)]


def test_an_event_ends_a_hold_wherever_it_happens() -> None:
    times = np.arange(0, 40) * 0.05
    known = np.ones(40, dtype=bool)
    holding = np.ones(40, dtype=bool)
    holding[12:14] = False
    event = np.zeros(40, dtype=bool)
    event[12] = True

    # Two failures in a row, well inside HOLD_BREAK_MAX_S, and a hand step on the
    # first of them: the run ends there rather than being bridged.
    runs = ph.hold_runs(times, holding, known, event)
    assert [(run.start, run.end, run.hold_id) for run in runs] == [(0, 11, 0), (14, 39, 1)]
    with pytest.raises(ValueError, match="event must have the same shape"):
        ph.hold_runs(times, holding, known, np.ones(5, dtype=bool))
    with pytest.raises(ValueError, match="max_break_s"):
        ph.hold_runs(times, holding, known, max_break_s=-0.1)


def test_a_gap_at_the_very_end_of_a_clip_is_not_part_of_the_hold() -> None:
    times = np.arange(0, 20) * 0.05
    known = np.ones(20, dtype=bool)
    holding = np.zeros(20, dtype=bool)
    holding[:12] = True

    runs = ph.hold_runs(times, holding, known, min_hold_s=0.1)
    assert [(run.start, run.end) for run in runs] == [(0, 11)]


# --------------------------------------------------------------------------- #
# The phase sequences
# --------------------------------------------------------------------------- #


def test_stand_kick_up_hold_exit_is_the_whole_sequence() -> None:
    """The trajectory the issue is named after: one hold, of about three seconds."""
    result = classify(attempt())

    assert phases_of(result) == ["pre", "kickup", "hold", "exit", "post"]
    assert result.hold_count == 1
    assert result.hold_durations_s() == (pytest.approx(3.23, abs=0.1),)
    assert result.longest_hold_s() == pytest.approx(3.23, abs=0.1)
    # Every frame of the hold, and only those frames, has a number.
    holds = codes_in(result, "hold")
    assert holds == list(range(25, 124))
    assert set(result.hold_id[holds].tolist()) == {0}
    outside = [index for index in range(result.frames) if index not in holds]
    assert all(int(result.hold_id[index]) == ph.NO_HOLD for index in outside)
    # The boundaries: the hold starts where the feet clear INVERTED_MIN, and ends
    # where they stop clearing it on the way down.
    assert result.phase_counts() == {
        "pre": 21,
        "kickup": 4,
        "hold": 99,
        "exit": 16,
        "post": 6,
        "unknown": 0,
    }


def test_a_hand_step_in_the_middle_of_a_hold_makes_two_holds() -> None:
    # The hands are put down 0.4 L to the right over four frames, and the body
    # stays over them where they landed.
    poses = hold(40)
    for offset in (0.1, 0.2, 0.3, 0.4):
        poses.append(dataclasses.replace(HOLDING, wrist_offset_px=offset * LENGTH))
    poses += hold(40, wrist_offset_px=0.4 * LENGTH)
    result = classify(poses)

    assert result.hold_count == 2
    assert [round(value, 2) for value in result.hold_durations_s()] == [1.32, 1.09]
    assert phases_of(result) == ["hold", "exit", "hold"]
    assert result.hold_id[40] == 0
    assert result.hold_id[50] == 1
    # The step is reported on the frames the wrists are moving, and the frames
    # after it are the exit the rules give: a hold that stops holding is an exit,
    # because the hands are still the support.
    assert [bool(result.signals.hand_step[index]) for index in range(41, 45)] == [True] * 4
    assert not bool(result.signals.hand_step[50])
    assert ph.EXIT in {int(result.phase[index]) for index in range(41, 51)}


def test_a_short_unknown_gap_inside_a_hold_does_not_split_it() -> None:
    poses = hold(20) + [dataclasses.replace(HOLDING, valid=False)] * 4 + hold(20)
    result = classify(poses)

    assert result.hold_count == 1
    assert result.hold_durations_s() == (pytest.approx(1.42, abs=0.05),)
    # The gap is inside the hold, and the hold's number is on it.
    gap = list(range(20, 24))
    assert [str(result.phase_codes[index]) for index in gap] == ["hold"] * 4
    assert {int(result.hold_id[index]) for index in gap} == {0}
    assert all(str(result.signals.unknown_reason[index]) for index in gap)
    # ... and the gap is under HOLD_BREAK_MAX_S, which is why it did not split it.
    assert 4 * FRAME_MS / 1000.0 <= ph.HOLD_BREAK_MAX_S


def test_a_long_unknown_gap_does_split_a_hold() -> None:
    poses = hold(20) + [dataclasses.replace(HOLDING, valid=False)] * 12 + hold(20)
    result = classify(poses)

    assert result.hold_count == 2
    assert [round(value, 2) for value in result.hold_durations_s()] == [0.63, 0.63]
    # The gap is its own segment: where the module could not look is an answer.
    assert str(result.phase_codes[20]) == "unknown"
    assert str(result.phase_codes[31]) == "unknown"
    assert str(result.phase_codes[32]) == "hold"
    assert 12 * FRAME_MS / 1000.0 > ph.HOLD_BREAK_MAX_S


def test_a_walk_with_the_hands_moving_yields_no_hold() -> None:
    """Hands and feet both travelling: the body passes through hold-like poses."""
    poses = []
    for index in range(120):
        travel = index * 1.2 * LENGTH / 30.0
        poses.append(
            dataclasses.replace(
                STANDING,
                wrist_x=200.0 + travel,
                wrist_y=400.0 - 120.0 * math.sin(index / 12.0),
                u_ankle=40.0 * math.sin(index / 5.0) / LENGTH,
                v_ankle=-0.2 + 1.0 * math.sin(index / 7.0) ** 2,
                v_hip=-0.3 + 0.9 * math.sin(index / 7.0) ** 2,
            )
        )
    result = classify(poses)

    assert result.hold_count == 0
    assert "hold" not in phases_of(result)
    # The trajectory does pass through inverted, upright poses — that is the trap
    # this test sets — and the hands moving is what saves it: the wrists travel at
    # 1.2 L/s, six times the speed at which they count as planted, so only the
    # first frames, which have no window to measure over, can read as planted.
    assert int(result.signals.inverted.sum()) > 10
    assert int(result.signals.hands_low.sum()) > 10
    assert int(result.signals.hands_down.sum()) <= 1
    assert int(result.signals.hand_step.sum()) > 100
    assert set(result.phase_codes) <= {"pre", "kickup", "exit", "post", "unknown"}


def test_a_failed_kick_up_never_inverted_yields_no_hold() -> None:
    """The athlete plants their hands, tries, and comes back down."""
    poses = stand(10) + swing(STANDING, PLANTED, 6) + swing(PLANTED, FAILED, 12) + stand(10)
    result = classify(poses)

    assert result.hold_count == 0
    assert not bool(result.signals.inverted.any())
    assert "hold" not in phases_of(result)
    # The hands are planted and still for the whole of the attempt, and the feet
    # never clear INVERTED_MIN: that is a kick-up that did not arrive.
    assert int(result.signals.hands_down.sum()) >= 5
    assert codes_in(result, "kickup")
    assert phases_of(result) == ["pre", "kickup", "post"]


def test_a_second_attempt_after_a_fall_is_a_second_hold() -> None:
    first = attempt(30)
    poses = first + first
    result = classify(poses)

    assert result.hold_count == 2
    assert [round(value, 2) for value in result.hold_durations_s()] == [1.25, 1.25]
    assert phases_of(result) == [
        "pre",
        "kickup",
        "hold",
        "exit",
        "post",
        "kickup",
        "hold",
        "exit",
        "post",
    ]


def test_a_clip_that_starts_in_a_hold_has_no_pre() -> None:
    result = classify(hold(30))
    assert phases_of(result) == ["hold"]
    assert result.hold_count == 1
    assert result.hold_durations_s() == (pytest.approx(0.9, abs=0.1),)


# --------------------------------------------------------------------------- #
# The output files
# --------------------------------------------------------------------------- #


def test_the_per_frame_table_carries_the_label_and_every_signal() -> None:
    poses = attempt(20) + [dataclasses.replace(HOLDING, contact=True)]
    result = classify(poses)
    table = ph.phase_table(result)

    assert list(table.columns) == [
        "frame_idx",
        "t_ms",
        "phase",
        "hold_id",
        "known",
        "unknown_reason",
        "v_ankle_mid",
        "u_ankle_mid",
        "v_hip_mid",
        "body_angle_deg",
        "inverted",
        "hands_low",
        "wrist_speed_l_per_s",
        "wrist_step_l",
        "hands_down",
        "hand_step",
        "legs_velocity_l_per_s",
        "legs_rising",
        "legs_falling",
    ]
    assert len(table) == result.frames
    assert table["frame_idx"].tolist() == list(range(result.frames))
    assert table["phase"].tolist() == [str(name) for name in result.phase_codes]
    assert table.loc[result.frames - 1, "phase"] == "unknown"
    assert table.loc[result.frames - 1, "unknown_reason"] == ph.REASON_TRAINER
    # The hold of this attempt, and the number it carries.
    holds = codes_in(result, "hold")
    assert table.loc[holds[0], "phase"] == "hold"
    assert int(table.loc[holds[0], "hold_id"]) == 0
    assert int(table.loc[holds[-1], "hold_id"]) == 0
    # The signals are the ones the answer was made of.
    assert table["inverted"].tolist() == result.signals.inverted.tolist()
    assert table["hands_down"].tolist() == result.signals.hands_down.tolist()


def test_the_segment_table_agrees_with_the_frames() -> None:
    result = classify(attempt(40))
    table = ph.segments_table([result])

    assert list(table.columns) == list(ph.SEGMENT_COLUMNS)
    assert table["clip_id"].eq(CLIP_ID).all()
    assert table["phase"].tolist() == ["pre", "kickup", "hold", "exit", "post"]
    # One row per phase run, and the rows tile the clip without a gap.
    assert table["start_frame"].tolist() == [0, 21, 25, 74, 90]
    assert table["end_frame"].tolist() == [20, 24, 73, 89, 95]
    assert (table["start_frame"].shift(-1).dropna() == table["end_frame"][:-1] + 1).all()
    # The hold row carries a number and the others do not.
    assert table["hold_id"].tolist() == [ph.NO_HOLD, ph.NO_HOLD, 0, ph.NO_HOLD, ph.NO_HOLD]
    for row in table.itertuples(index=False):
        assert row.duration_s == pytest.approx((row.end_ms - row.start_ms) / 1000.0, abs=1e-3)
    assert table.loc[2, "duration_s"] == pytest.approx(1.58, abs=0.05)


def test_two_holds_in_one_clip_are_two_segment_rows() -> None:
    poses = hold(20) + [dataclasses.replace(HOLDING, valid=False)] * 12 + hold(20)
    table = ph.segments_table([classify(poses)])
    holds = table[table["phase"] == "hold"]
    assert len(holds) == 2
    assert holds["hold_id"].tolist() == [0, 1]


def test_write_segments_replaces_only_the_clips_this_run_covered(tmp_path: pathlib.Path) -> None:
    path = tmp_path / ph.SEGMENTS_NAME
    other = classify(hold(30))
    dataclasses.replace(other, clip_id="other000000")  # the same answer, another clip
    ph.write_segments(path, ph.segments_table([dataclasses.replace(other, clip_id="other000000")]))

    mine = ph.segments_table([classify(attempt(20))])
    ph.write_segments(path, mine)

    written = pd.read_csv(path)
    assert sorted(written["clip_id"].unique()) == [CLIP_ID, "other000000"]
    assert len(written[written["clip_id"] == "other000000"]) == 1
    assert list(written.columns) == list(ph.SEGMENT_COLUMNS)
    # Sorted by clip, and no temporary file left behind.
    assert written["clip_id"].tolist() == sorted(written["clip_id"].tolist())
    assert [p.name for p in tmp_path.iterdir()] == [ph.SEGMENTS_NAME]


def test_write_segments_refuses_a_table_it_cannot_merge(tmp_path: pathlib.Path) -> None:
    path = tmp_path / ph.SEGMENTS_NAME
    path.write_text("clip_id,phase\nx,hold\n")
    with pytest.raises(ValueError, match="missing column"):
        ph.write_segments(path, ph.segments_table([classify(hold(20))]))


# --------------------------------------------------------------------------- #
# Reading, writing, one clip, the run
# --------------------------------------------------------------------------- #


def test_run_clip_writes_the_parquet_and_leaves_the_sidecar_alone(tmp_path: pathlib.Path) -> None:
    in_root = tmp_path / "processed" / "mediapipe"
    out_root = tmp_path / "phases" / "mediapipe"
    write_processed(in_root, attempt(20))

    report = ph.run_clip(CLIP_ID, in_root=in_root, out_root=out_root)

    assert not report.skipped
    assert report.parquet_path == out_root / f"{CLIP_ID}.parquet"
    assert report.parquet_path.is_file()
    assert report.stats.hold_count == 1
    assert report.stats.known_frames == report.result.frames
    assert "holds=1" in report.line()
    assert [p.name for p in out_root.iterdir()] == [f"{CLIP_ID}.parquet"]

    # The parquet reads back with everything the columns promise.
    table = ph.read_clip_phases(report.parquet_path)
    assert len(table) == report.result.frames
    assert table["phase"].tolist() == [str(name) for name in report.result.phase_codes]
    assert table["hold_id"].max() == 0

    # ... and re-running it is a no-op unless asked.
    again = ph.run_clip(CLIP_ID, in_root=in_root, out_root=out_root)
    assert again.skipped
    assert again.line().startswith("skip ")


def test_run_clip_writes_an_unusable_clip_as_all_unknown(tmp_path: pathlib.Path) -> None:
    in_root = tmp_path / "processed" / "mediapipe"
    out_root = tmp_path / "phases" / "mediapipe"
    write_processed(in_root, hold(20), usable=False)

    report = ph.run_clip(CLIP_ID, in_root=in_root, out_root=out_root)

    assert not report.stats.usable
    assert "no body length" in report.stats.unusable_reason
    assert set(ph.read_clip_phases(report.parquet_path)["phase"]) == {"unknown"}
    assert "unusable" in report.line()


def test_run_clip_refuses_a_clip_that_was_never_processed(tmp_path: pathlib.Path) -> None:
    in_root = tmp_path / "processed" / "mediapipe"
    with pytest.raises(FileNotFoundError, match="handstand.postprocess --all"):
        ph.run_clip(CLIP_ID, in_root=in_root, out_root=tmp_path / "out")


def test_reading_a_raw_keypoint_parquet_says_to_run_postprocess(tmp_path: pathlib.Path) -> None:
    path = write_processed(tmp_path, hold(4))
    table = pd.read_parquet(path).drop(columns=[ph.VALID_COLUMN])
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="not a processed parquet"):
        ph.read_processed(path)


def test_read_body_length_reports_why_a_clip_has_none(tmp_path: pathlib.Path) -> None:
    assert (
        ph.read_body_length(tmp_path / "missing.json")[0]
        != ph.read_body_length(tmp_path / "missing.json")[0]
    )  # NaN
    assert "unreadable" in ph.read_body_length(tmp_path / "missing.json")[1]

    path = write_processed(tmp_path, hold(4), usable=False)
    length, reason = ph.read_body_length(path.with_suffix(".json"))
    assert math.isnan(length)
    assert "no body length" in reason

    path = write_processed(tmp_path, hold(4), body_length=0.0)
    length, reason = ph.read_body_length(path.with_suffix(".json"))
    assert math.isnan(length)
    assert "body_length_px" in reason

    path = write_processed(tmp_path, hold(4))
    length, reason = ph.read_body_length(path.with_suffix(".json"))
    assert length == pytest.approx(LENGTH)
    assert reason == ""


def test_available_clips_and_the_output_locations(tmp_path: pathlib.Path) -> None:
    in_root = ph.input_dir(tmp_path, "vision")
    assert in_root == tmp_path / "processed" / "vision"
    assert ph.output_dir(tmp_path, "mediapipe") == tmp_path / "phases" / "mediapipe"

    with pytest.raises(FileNotFoundError, match="handstand.postprocess"):
        ph.available_clips(in_root)
    assert ph.available_clips(in_root, ["b", "a"]) == ["b", "a"]

    write_processed(in_root, hold(4), clip_id="a")
    write_processed(in_root, hold(4), clip_id="b")
    assert ph.available_clips(in_root) == ["a", "b"]


# --------------------------------------------------------------------------- #
# The run summary
# --------------------------------------------------------------------------- #


def test_summary_answers_the_dataset_questions() -> None:
    reports = [
        ph.measure(classify(attempt(30))),  # one 1.25 s hold
        ph.measure(classify(attempt(60))),  # one 2.58 s hold
        ph.measure(classify(stand(30))),  # no hold
        ph.measure(classify(hold(30), body_length=math.nan)),  # unusable
    ]
    text = ph.summary(reports)

    assert "clips=4" in text
    assert "usable=3" in text
    assert "with_hold=2" in text
    assert "no_hold=2" in text
    assert "median=1.75" in text
    assert "p25=" in text and "p75=" in text
    assert "hold time" in text
    assert f"no hold            {CLIP_ID}" in text
    assert "unusable clips" in text
    assert ph.summary([]) == "summary clips=0"
    assert "no hold" not in ph.summary([ph.measure(classify(attempt(30)))])


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def test_the_cli_labels_clips_and_writes_the_segments(tmp_path: pathlib.Path) -> None:
    in_root = tmp_path / "processed" / "mediapipe"
    out_root = tmp_path / "phases" / "mediapipe"
    write_processed(in_root, attempt(20), clip_id="aaa000000000")
    write_processed(in_root, hold(30), clip_id="bbb111111111")

    assert ph.main(["--clips", "aaa000000000", "bbb111111111", "--data", str(tmp_path)]) == 0

    segments = pd.read_csv(out_root / ph.SEGMENTS_NAME)
    assert sorted(segments["clip_id"].unique()) == ["aaa000000000", "bbb111111111"]
    assert list(segments[segments["clip_id"] == "bbb111111111"]["phase"]) == ["hold"]
    assert (out_root / "aaa000000000.parquet").is_file()
    assert (out_root / "bbb111111111.parquet").is_file()


def test_the_cli_needs_clips_or_all(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        ph.main(["--data", str(tmp_path)])
    with pytest.raises(SystemExit):
        ph.main(["--all", "--clips", CLIP_ID, "--data", str(tmp_path)])
    with pytest.raises(SystemExit):
        ph.main(["--clips", CLIP_ID, "--limit", "-1", "--data", str(tmp_path)])


def test_the_cli_says_so_when_nothing_has_been_processed(tmp_path: pathlib.Path, capsys) -> None:
    assert ph.main(["--all", "--data", str(tmp_path)]) == 2
    assert "handstand.postprocess" in capsys.readouterr().out


def test_the_cli_survives_a_clip_it_cannot_read(tmp_path: pathlib.Path, capsys) -> None:
    in_root = tmp_path / "processed" / "mediapipe"
    write_processed(in_root, hold(20), clip_id="aaa000000000")
    assert ph.main(["--clips", "aaa000000000", "gone", "--data", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "fail  gone" in out
    assert "clips=1" in out
    assert ph.summary([]) == "summary clips=0"


def test_the_cli_documents_the_documented_flags() -> None:
    parser = ph.build_arg_parser()
    args = parser.parse_args(["--all", "--source", "vision", "--limit", "3", "--overwrite"])
    assert args.all_clips
    assert args.source == "vision"
    assert args.limit == 3
    assert args.overwrite
    args = parser.parse_args(["--clips", "a", "b", "--render", "a", "--max-frames", "5"])
    assert args.clips == ["a", "b"]
    assert args.render == ["a"]
    assert args.max_frames == 5
    with pytest.raises(SystemExit):
        parser.parse_args(["--clips", "a", "--source", "nope"])
    assert ph.SOURCES == athlete.SOURCES


# --------------------------------------------------------------------------- #
# The overlay
# --------------------------------------------------------------------------- #


def test_phase_captions_name_the_phase_and_the_hold() -> None:
    result = classify(attempt(20))
    captions = ph.phase_captions(ph.phase_table(result))
    holds = codes_in(result, "hold")

    assert set(captions) == set(range(result.frames))
    assert captions[0] == ["phase pre"]
    assert captions[21] == ["phase kickup"]
    assert captions[holds[0]] == ["phase hold", "hold 0"]
    assert captions[holds[-1]] == ["phase hold", "hold 0"]
    assert captions[result.frames - 1] == ["phase post"]


def test_render_clip_names_the_output_and_asks_for_the_phases(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="handstand.phases --clips"):
        ph.render_clip(CLIP_ID, data=tmp_path)


def test_the_overlay_takes_the_phase_captions() -> None:
    assert "extra_captions" in overlay.render_overlay.__doc__
    assert "handstand.phases" in overlay.render_overlay.__doc__


def test_the_overlay_file_name_carries_the_phase() -> None:
    # The same naming rule the other overlays use: the panel label, and the source
    # when it is not the default, so two sources never overwrite each other.
    assert overlay.overlay_filename(CLIP_ID, [ph.PHASES_DIRNAME]) == f"{CLIP_ID}_phases.mp4"
    assert (
        overlay.overlay_filename(CLIP_ID, [ph.PHASES_DIRNAME], source="vision")
        == f"{CLIP_ID}_vision_phases.mp4"
    )
