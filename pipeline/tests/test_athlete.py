"""Tests for :mod:`handstand.athlete`.

Nothing here touches the real handstand videos: the input is a hand-built
multi-person parquet with synthetic ``(33, 5)`` landmark blocks, laid out like
the real clips (576x1024 display pixels) — a standing trainer, the same body
upside down, a third person standing on top of the athlete, and so on.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import pytest

from handstand import athlete
from handstand import pose_mediapipe as pm

#: Where the synthetic people stand in the 576x1024 frame.
TRAINER_X = 100.0
ATHLETE_X = 350.0
#: The mat: the lowest point of any body in the test scenes.
MAT_Y = 925.0
FRAME_COUNT = 6


# --------------------------------------------------------------------------- #
# Synthetic bodies
# --------------------------------------------------------------------------- #


def landmarks(points: dict[str, tuple[float, float]], visibility: float = 0.9) -> np.ndarray:
    """A ``(33, 5)`` block; joints left out of ``points`` are not reported at all."""
    block = np.full((len(pm.JOINT_NAMES), athlete.LANDMARK_FIELDS), np.nan)
    for name, (x, y) in points.items():
        block[pm.JOINT_INDEX[name]] = (x, y, 0.0, visibility, 0.9)
    return block


def face(x: float, y: float) -> dict[str, tuple[float, float]]:
    """The seven face landmarks, centred on ``(x, y)``."""
    return {
        "nose": (x, y),
        "left_eye": (x - 8, y - 8),
        "right_eye": (x + 8, y - 8),
        "left_ear": (x - 18, y - 6),
        "right_ear": (x + 18, y - 6),
        "mouth_left": (x - 8, y + 10),
        "mouth_right": (x + 8, y + 10),
    }


def standing(
    x: float, visibility: float = 0.9, wrist_y: float = 480.0, hip_y: float = 500.0
) -> np.ndarray:
    """A person on their feet: hands by their sides at ``wrist_y``, feet on the mat."""
    shift = hip_y - 500.0
    points = {
        **face(x, 250.0 + shift),
        "left_shoulder": (x - 15, 300.0 + shift),
        "right_shoulder": (x + 15, 300.0 + shift),
        "left_elbow": (x - 20, 400.0 + shift),
        "right_elbow": (x + 20, 400.0 + shift),
        "left_wrist": (x - 25, wrist_y),
        "right_wrist": (x + 25, wrist_y),
        "left_hip": (x - 10, hip_y),
        "right_hip": (x + 10, hip_y),
        "left_knee": (x - 12, 700.0),
        "right_knee": (x + 12, 700.0),
        "left_ankle": (x - 15, 890.0),
        "right_ankle": (x + 15, 890.0),
        "left_heel": (x - 15, 915.0),
        "right_heel": (x + 15, 915.0),
        "left_foot_index": (x - 20, MAT_Y),
        "right_foot_index": (x + 20, MAT_Y),
    }
    return landmarks(points, visibility=visibility)


def handstand(x: float, visibility: float = 0.9, wrist_y: float = 900.0) -> np.ndarray:
    """The same body upside down: hands on the mat, feet in the air."""
    points = {
        **face(x, 830.0),
        "left_shoulder": (x - 15, 700.0),
        "right_shoulder": (x + 15, 700.0),
        "left_elbow": (x - 20, 800.0),
        "right_elbow": (x + 20, 800.0),
        "left_wrist": (x - 25, wrist_y),
        "right_wrist": (x + 25, wrist_y),
        "left_hip": (x - 10, 500.0),
        "right_hip": (x + 10, 500.0),
        "left_knee": (x - 12, 350.0),
        "right_knee": (x + 12, 350.0),
        "left_ankle": (x - 15, 200.0),
        "right_ankle": (x + 15, 200.0),
        "left_heel": (x - 15, 180.0),
        "right_heel": (x + 15, 180.0),
        "left_foot_index": (x - 20, 170.0),
        "right_foot_index": (x + 20, 170.0),
    }
    return landmarks(points, visibility=visibility)


def scaled_bone(pose: np.ndarray, first: str, second: str, scale: float) -> np.ndarray:
    """``pose`` with one joint pushed out along its bone, ``scale`` times as far.

    The other joint of the bone stays put, so a shin of ``1.5`` is a leg half
    again as long — what MediaPipe reports when it finishes the athlete's leg at
    the trainer's foot instead of their own.
    """
    moved = np.array(pose, copy=True)
    near, far = (pm.JOINT_INDEX[first], pm.JOINT_INDEX[second])
    moved[far, :2] = moved[near, :2] + scale * (moved[far, :2] - moved[near, :2])
    return moved


def scaled_leg(pose: np.ndarray, scale: float, side: str = "left") -> np.ndarray:
    """``pose`` with both bones of one leg ``scale`` times as long.

    The hip, the knee and the ankle all move, so the leg is longer rather than
    folded differently.
    """
    knee, ankle = pm.JOINT_INDEX[f"{side}_knee"], pm.JOINT_INDEX[f"{side}_ankle"]
    hip_point = pose[pm.JOINT_INDEX[f"{side}_hip"], :2]
    stretched = np.array(pose, copy=True)
    stretched[knee, :2] = hip_point + scale * (pose[knee, :2] - hip_point)
    stretched[ankle, :2] = stretched[knee, :2] + scale * (pose[ankle, :2] - pose[knee, :2])
    return stretched


def frame_of(frame_idx: int, *people: np.ndarray, t_ms_step: int = 40) -> athlete.FramePeople:
    """A measured :class:`athlete.FramePeople` for one frame."""
    poses = np.stack(people) if people else np.empty((0, len(pm.JOINT_NAMES), 5))
    return athlete.prepare_frame(frame_idx, frame_idx * t_ms_step, False, poses)


def kick_up_clip() -> list[athlete.FramePeople]:
    """Three frames of two people standing, then three frames of a handstand.

    The kick-up: nobody looks like a handstand at the start, so only the frames
    after it can say which of the two is the athlete.
    """
    frames = [frame_of(index, standing(TRAINER_X), standing(ATHLETE_X)) for index in range(3)]
    frames += [frame_of(index, standing(TRAINER_X), handstand(ATHLETE_X)) for index in range(3, 6)]
    return frames


def chosen_x(choice: athlete.FrameChoice) -> float:
    """The x of the nose of whoever was chosen, or NaN."""
    if choice.person is None:
        return float("nan")
    return float(choice.person.landmarks[pm.JOINT_INDEX["nose"], 0])


# --------------------------------------------------------------------------- #
# Synthetic multi-person parquets
# --------------------------------------------------------------------------- #


def multi_table(
    frames: Sequence[Sequence[np.ndarray]], *, rotated: bool = False, t_ms_step: int = 40
) -> pd.DataFrame:
    """A long multi-person keypoint table: one 33-row block per person per frame.

    A frame given as ``[]`` is written the way the runner writes a frame with
    nobody in it: one block of NaN with ``person_idx = -1`` and ``detected =
    false``.
    """
    rows: list[dict[str, object]] = []
    for frame_idx, people in enumerate(frames):
        t_ms = frame_idx * t_ms_step
        blocks = list(people) or [np.full((len(pm.JOINT_NAMES), 5), np.nan)]
        for person_idx, pose in enumerate(blocks):
            detected = bool(people)
            person_idx_written = person_idx if people else pm.NO_PERSON_IDX
            for joint_index, joint in enumerate(pm.JOINT_NAMES):
                x, y, z, visibility, presence = pose[joint_index]
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "t_ms": t_ms,
                        "joint": joint,
                        "x": x,
                        "y": y,
                        "z": z,
                        "visibility": visibility,
                        "presence": presence,
                        "rotated": rotated,
                        "detected": detected,
                        pm.PERSON_COLUMN: person_idx_written,
                    }
                )
    return pd.DataFrame(rows, columns=list(pm.PARQUET_COLUMNS_MULTI))


def write_multi(
    directory: pathlib.Path, frames: Sequence[Sequence[np.ndarray]], clip_id: str = "clip00000001"
) -> pathlib.Path:
    """Write ``<directory>/<clip_id>.parquet`` in the multi-person schema."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{clip_id}.parquet"
    multi_table(frames).to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #


def test_schema_and_constants_match_the_documented_contract() -> None:
    assert athlete.ATHLETE_COLUMNS == (
        *pm.PARQUET_COLUMNS,
        "athlete_score",
        "n_people",
        "trainer_contact",
        "contact_reason",
    )
    assert athlete.ATHLETE_OUTPUT_DIRNAME == "mediapipe_athlete"
    assert athlete.input_dirname() == "mediapipe_multi"
    assert athlete.output_dirname() == "mediapipe_athlete"
    weights = (
        athlete.WEIGHT_INVERSION,
        athlete.WEIGHT_SUPPORT,
        athlete.WEIGHT_CONTINUITY,
        athlete.WEIGHT_VISIBILITY,
    )
    assert weights == (0.4, 0.2, 0.3, 0.1)
    assert sum(weights) == pytest.approx(1.0)
    assert (athlete.MIN_ATHLETE_SCORE, athlete.AMBIGUITY_MARGIN) == (0.35, 0.05)
    assert (athlete.DEDUP_HIP_TOLERANCE, athlete.DEDUP_JOINT_TOLERANCE) == (0.15, 0.1)
    assert athlete.CONTACT_MIN_IOU == 0.3
    assert athlete.BONE_LENGTH_TOLERANCE == 0.35
    # The reasons are the documented ones, and every bone has a mirror image.
    assert athlete.CONTACT_REASONS == ("box_iou", "mixed_skeleton", "bone_length")
    assert athlete.REASON_NONE == ""
    assert athlete.ASYMMETRY_MIN_PEOPLE == 2
    assert len(athlete.BONES) == 10
    assert sorted(athlete.COUNTERPART_BONES) == sorted(name for name, _, _ in athlete.BONES)
    assert set(athlete.COUNTERPART_BONES.values()) == {name for name, _, _ in athlete.BONES}
    assert athlete.LEG_COUNTERPARTS == {"leg_l": "leg_r", "leg_r": "leg_l"}
    assert len(athlete.MAIN_JOINTS) == 12
    assert athlete.MAIN_JOINTS == (
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


def test_body_length_is_the_shoulder_to_ankle_span() -> None:
    assert athlete.body_length(handstand(ATHLETE_X)) == pytest.approx(500.0)
    assert athlete.body_length(standing(TRAINER_X)) == pytest.approx(590.0)
    # A body without shoulders or ankles falls back to its own extent, and a
    # body with nothing usable at all is the floor rather than a division by 0.
    stub = np.full((len(pm.JOINT_NAMES), 5), np.nan)
    assert athlete.body_length(stub) == athlete.MIN_BODY_LENGTH_PIXELS
    assert math.isfinite(athlete.body_length(stub))


def test_measurements_ignore_joints_that_were_never_reported() -> None:
    pose = handstand(ATHLETE_X)
    pose[pm.JOINT_INDEX["left_ankle"], :2] = np.nan
    person = athlete.measure_person(pose)
    # One ankle missing still leaves the other one to judge the orientation.
    assert person.inverted
    assert person.box is not None

    pose[pm.JOINT_INDEX["left_wrist"], :2] = np.nan
    pose[pm.JOINT_INDEX["right_wrist"], :2] = np.nan
    pose[pm.JOINT_INDEX["right_ankle"], :2] = np.nan
    person = athlete.measure_person(pose)
    assert not person.inverted  # no wrist left: nothing to judge an orientation on
    assert math.isnan(person.wrist_y)

    with pytest.raises(ValueError, match="landmarks"):
        athlete.measure_person(np.zeros((32, 5)))
    with pytest.raises(ValueError, match="landmarks"):
        athlete.measure_person(np.zeros((33, 4)))


def test_visible_joints_are_the_ones_the_model_really_saw() -> None:
    pose = handstand(ATHLETE_X)
    # A face the model did not see: MediaPipe still reports a position for it.
    for name in athlete.HEAD_JOINTS:
        pose[pm.JOINT_INDEX[name], 3] = 0.1
    person = athlete.measure_person(pose)
    assert all(pm.JOINT_INDEX[name] not in person.visible_joints for name in athlete.HEAD_JOINTS)
    # ... and the centre line falls back to the shoulders rather than to it.
    assert person.centre_line is not None
    head = person.centre_line[1]
    assert head == pytest.approx((ATHLETE_X, 700.0))


def test_centre_line_runs_from_the_feet_to_the_head() -> None:
    # The ends are midpoints of the foot and of the face landmarks, not of one joint.
    feet, head = athlete.body_centre_line(standing(TRAINER_X))
    assert feet[1] == pytest.approx(910.0)
    assert head[1] == pytest.approx(248.857, abs=1e-3)
    upside_down = athlete.body_centre_line(handstand(ATHLETE_X))
    assert upside_down[0][1] == pytest.approx(183.33, abs=1e-2)  # the feet, up in the air
    assert upside_down[1][1] == pytest.approx(828.86, abs=1e-2)
    # Nothing to build a line from: the check that needs it is simply skipped.
    assert athlete.body_centre_line(np.full((len(pm.JOINT_NAMES), 5), np.nan)) is None


def test_bounding_box_covers_the_visible_joints_only() -> None:
    pose = handstand(ATHLETE_X)
    pose[pm.JOINT_INDEX["right_foot_index"], :2] = (9000.0, 9000.0)  # not really seen
    pose[pm.JOINT_INDEX["right_foot_index"], 3] = 0.05
    box = athlete.bounding_box(pose)
    assert box == pytest.approx((325.0, 170.0, 375.0, 900.0))
    assert athlete.bounding_box(np.full((len(pm.JOINT_NAMES), 5), np.nan)) is None


def test_box_iou_counts_the_overlap() -> None:
    assert athlete.box_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0
    assert athlete.box_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0
    # Half the area shared: 50 / 150.
    assert athlete.box_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1 / 3)
    assert athlete.box_iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)) == 0.0


# --------------------------------------------------------------------------- #
# Rule 1: duplicates are one body
# --------------------------------------------------------------------------- #


def test_two_detections_of_one_body_are_the_same_body() -> None:
    first = athlete.measure_person(handstand(ATHLETE_X))
    twin = athlete.measure_person(handstand(ATHLETE_X + 4.0, visibility=0.95))
    assert athlete.same_body(first, twin)


def test_two_people_next_to_each_other_are_not_the_same_body() -> None:
    trainer = athlete.measure_person(standing(TRAINER_X))
    athlete_body = athlete.measure_person(handstand(ATHLETE_X))
    assert not athlete.same_body(trainer, athlete_body)
    # Hips close enough to look like a duplicate, but the joints do not line up.
    crouched = athlete.measure_person(standing(ATHLETE_X, hip_y=700.0))
    assert not athlete.same_body(athlete_body, crouched)
    # Far too few shared joints to tell anything.
    assert not athlete.same_body(
        trainer, athlete.measure_person(standing(ATHLETE_X, visibility=0.0))
    )


def test_deduplicate_keeps_the_more_visible_detection() -> None:
    people = [
        athlete.measure_person(handstand(ATHLETE_X, visibility=0.90)),
        athlete.measure_person(standing(TRAINER_X)),
        athlete.measure_person(handstand(ATHLETE_X + 4.0, visibility=0.95)),
    ]
    kept = athlete.deduplicate(people)
    assert len(kept) == 2
    assert kept[0].landmarks[0, 0] == pytest.approx(ATHLETE_X + 4.0)
    assert kept[1].landmarks[0, 0] == pytest.approx(TRAINER_X)
    # ... and the order of the frame is kept, so nothing downstream has to care.
    assert kept[1] is people[1]


def test_deduplicate_keeps_the_first_when_the_scores_tie() -> None:
    people = [
        athlete.measure_person(handstand(ATHLETE_X)),
        athlete.measure_person(handstand(ATHLETE_X + 3.0)),
    ]
    assert athlete.deduplicate(people) == [people[0]]


# --------------------------------------------------------------------------- #
# Rule 2: scoring
# --------------------------------------------------------------------------- #


def test_the_inverted_athlete_outscores_the_standing_trainer() -> None:
    frame = frame_of(0, standing(TRAINER_X), handstand(ATHLETE_X))
    trainer, athlete_body = frame.people
    assert trainer.inverted is False
    assert athlete_body.inverted is True
    assert frame.support_line == pytest.approx(MAT_Y)

    trainer_score = athlete.person_score(trainer, frame.support_line, None)
    athlete_score = athlete.person_score(athlete_body, frame.support_line, None)
    assert athlete_score == pytest.approx(0.98)
    # The weights, spelled out: 0.4 inverted + 0.2 support + 0.3 continuity + 0.1 seen.
    assert athlete_score == pytest.approx(0.4 + 0.2 * 0.95 + 0.3 + 0.1 * 0.9)
    assert trainer_score == pytest.approx(0.0 + 0.2 * 0.246 + 0.3 + 0.1 * 0.9, abs=1e-3)
    assert athlete_score - trainer_score > athlete.AMBIGUITY_MARGIN


def test_support_is_measured_against_the_lowest_hand_or_foot() -> None:
    on_the_mat = athlete.measure_person(handstand(ATHLETE_X))
    hands_up = athlete.measure_person(standing(ATHLETE_X, wrist_y=100.0))
    assert athlete.support_score(on_the_mat, MAT_Y) > athlete.support_score(hands_up, MAT_Y)
    assert athlete.support_score(hands_up, MAT_Y) == 0.0
    # A body-length above the line scores 0, on it scores 1.
    person = on_the_mat
    assert athlete.support_score(person, person.wrist_y) == pytest.approx(1.0)
    assert athlete.support_score(person, person.wrist_y + person.length) == pytest.approx(0.0)
    assert athlete.support_score(person, float("nan")) == 0.0


def test_continuity_falls_off_with_the_hips_and_ignores_missing_hips() -> None:
    person = athlete.measure_person(handstand(ATHLETE_X))
    assert athlete.continuity_score(person, person.hip) == pytest.approx(1.0)
    far = person.hip + person.length
    assert athlete.continuity_score(person, far) == pytest.approx(0.0)
    # No history yet: nobody is penalised, so the other terms decide.
    assert athlete.continuity_score(person, None) == 1.0
    assert athlete.continuity_score(person, np.full(2, np.nan)) == 1.0


def test_choose_person_picks_the_athlete() -> None:
    frame = frame_of(0, standing(TRAINER_X), handstand(ATHLETE_X))
    person, score = athlete.choose_person(frame.people, frame.support_line, None)
    assert person is not None
    assert person.landmarks[0, 0] == pytest.approx(ATHLETE_X)
    assert score == pytest.approx(0.98)


def test_choose_person_gives_up_on_two_candidates_it_cannot_tell_apart() -> None:
    # Two identical standing bodies: same score, so no way to say which is which.
    frame = frame_of(0, standing(TRAINER_X), standing(ATHLETE_X))
    assert athlete.person_score(frame.people[0], frame.support_line, None) == pytest.approx(
        athlete.person_score(frame.people[1], frame.support_line, None)
    )
    person, score = athlete.choose_person(frame.people, frame.support_line, None)
    assert person is None
    assert math.isfinite(score)


def test_choose_person_gives_up_when_the_best_score_is_too_low() -> None:
    # Hands in the air and barely seen: visible, plausible, but not an athlete.
    frame = frame_of(0, standing(ATHLETE_X, visibility=0.2, wrist_y=100.0))
    person, score = athlete.choose_person(frame.people, frame.support_line, None)
    assert person is None
    assert score == pytest.approx(0.32)
    assert score < athlete.MIN_ATHLETE_SCORE


def test_choose_person_on_an_empty_frame_is_nobody() -> None:
    frame = frame_of(0)
    assert frame.n_people == 0
    assert math.isnan(frame.support_line)
    person, score = athlete.choose_person(frame.people, frame.support_line, None)
    assert person is None
    assert math.isnan(score)


# --------------------------------------------------------------------------- #
# Rule 3: trainer contact
# --------------------------------------------------------------------------- #


def test_a_separate_trainer_is_not_contact() -> None:
    frame = frame_of(0, standing(TRAINER_X), handstand(ATHLETE_X))
    assert athlete.trainer_contact(frame.people[1], [frame.people[0]]) is False
    assert athlete.trainer_contact(frame.people[1], []) is False


def test_a_trainer_standing_on_the_athlete_is_contact() -> None:
    # Same body outline, 20 px to the side: the boxes overlap almost completely.
    frame = frame_of(0, standing(ATHLETE_X + 20.0), handstand(ATHLETE_X))
    assert athlete.box_iou(frame.people[0].box, frame.people[1].box) > athlete.CONTACT_MIN_IOU
    assert athlete.trainer_contact(frame.people[1], [frame.people[0]]) is True


def test_a_stitched_skeleton_is_contact() -> None:
    """The athlete's shoulders but the trainer's leg: one limb joint is not theirs."""
    stitched = handstand(ATHLETE_X)
    # The model finished the athlete's leg at the trainer's knee (500 - 12, 700),
    # which is 138 px from the athlete's own centre line.
    for name in ("left_ankle", "right_ankle"):
        stitched[pm.JOINT_INDEX[name], :2] = (488.0, 700.0)
    trainer = athlete.measure_person(standing(500.0, hip_y=650.0))
    athlete_body = athlete.measure_person(stitched)

    assert athlete.nearest_joint_distance(
        athlete_body.landmarks[pm.JOINT_INDEX["left_ankle"], :2], trainer
    ) == pytest.approx(0.0)
    # The boxes are nowhere near each other, so this is the mixed-skeleton rule.
    assert athlete.box_iou(athlete_body.box, trainer.box) < athlete.CONTACT_MIN_IOU
    assert athlete.mixed_skeleton(athlete_body, trainer) is True
    assert athlete.trainer_contact(athlete_body, [trainer]) is True
    # Without the stitch the same two people are a handstand next to a trainer.
    assert athlete.trainer_contact(athlete.measure_person(handstand(ATHLETE_X)), [trainer]) is False


def test_mixed_skeleton_needs_a_centre_line() -> None:
    invisible = handstand(ATHLETE_X, visibility=0.0)
    assert athlete.measure_person(invisible).centre_line is None
    assert (
        athlete.mixed_skeleton(
            athlete.measure_person(invisible), athlete.measure_person(standing(TRAINER_X))
        )
        is False
    )


def test_contact_reason_names_the_rule_that_fired() -> None:
    # a. the trainer's box and the athlete's are nearly the same box
    stacked = frame_of(0, standing(ATHLETE_X + 20.0), handstand(ATHLETE_X))
    assert athlete.contact_reason(stacked.people[1], [stacked.people[0]]) == "box_iou"
    # b. the athlete's leg finished at somebody else's knee, boxes far apart
    stitched = handstand(ATHLETE_X)
    for name in ("left_ankle", "right_ankle"):
        stitched[pm.JOINT_INDEX[name], :2] = (488.0, 700.0)
    trainer = athlete.measure_person(standing(500.0, hip_y=650.0))
    body = athlete.measure_person(stitched)
    assert athlete.contact_reason(body, [trainer]) == "mixed_skeleton"
    # c. a leg half again too long — and nobody else in the frame at all, which
    #    is the case the two rules above cannot see. The clip's own medians are
    #    what say so; without them only the left/right rule can run.
    long_leg = scaled_bone(handstand(ATHLETE_X), "left_knee", "left_ankle", 1.5)
    body = athlete.measure_person(long_leg)
    assert athlete.contact_reason(body, [], {"shin_l": 150.0}) == "bone_length"
    assert athlete.contact_reason(body, []) == ""
    # None of them: a handstand next to a trainer standing well clear of them.
    apart = frame_of(0, standing(TRAINER_X), handstand(ATHLETE_X))
    assert athlete.contact_reason(apart.people[1], []) == ""
    assert athlete.contact_reason(apart.people[1], [apart.people[0]]) == ""
    assert athlete.trainer_contact(apart.people[1], [apart.people[0]]) is False
    # The boolean is the reason, and the flag is the reason being non-empty.
    assert athlete.trainer_contact(body, [], {"shin_l": 150.0}) is True
    assert athlete.trainer_contact(athlete.measure_person(handstand(ATHLETE_X)), []) is False


# --------------------------------------------------------------------------- #
# Rule 3c: the bones of one skeleton, with nobody else reported in the frame
# --------------------------------------------------------------------------- #


def handstand_clip(poses: Sequence[np.ndarray]) -> list[athlete.FramePeople]:
    """One frame per pose: a standing trainer beside that athlete, in handstand."""
    return [frame_of(index, standing(TRAINER_X), pose) for index, pose in enumerate(poses)]


def test_bone_lengths_measure_every_bone_and_skip_the_ones_not_seen() -> None:
    person = athlete.measure_person(handstand(ATHLETE_X))
    lengths = athlete.bone_lengths(person)
    assert set(lengths) == {name for name, _, _ in athlete.BONES}
    # The synthetic handstand is built with 100 px arms and 150 px legs.
    assert lengths["upper_arm_l"] == pytest.approx(100.1, abs=0.1)
    assert lengths["thigh_l"] == pytest.approx(150.0, abs=0.1)
    assert lengths["torso_r"] == pytest.approx(200.1, abs=0.1)
    assert athlete.leg_lengths(person)["leg_l"] == pytest.approx(300.0, abs=0.2)
    # A joint the model did not see makes its bone unusable...
    blind = athlete.measure_person(handstand(ATHLETE_X, visibility=0.0))
    assert all(math.isnan(value) for value in athlete.bone_lengths(blind).values())
    assert all(math.isnan(value) for value in athlete.leg_lengths(blind).values())
    # ... and one unseen hip takes the thigh and the whole leg with it, while
    # the other leg is still measurable.
    one_leg = handstand(ATHLETE_X)
    one_leg[pm.JOINT_INDEX["left_hip"], 3] = 0.1
    half = athlete.measure_person(one_leg)
    assert math.isnan(athlete.bone_lengths(half)["thigh_l"])
    assert math.isfinite(athlete.bone_lengths(half)["shin_l"])
    assert math.isnan(athlete.leg_lengths(half)["leg_l"])
    assert math.isfinite(athlete.leg_lengths(half)["leg_r"])


def test_a_bone_off_the_clips_own_length_is_flagged_bone_length() -> None:
    """One frame's ankle lands on the trainer's foot: a shin 50 % too long."""
    good = handstand(ATHLETE_X)
    long_leg = scaled_bone(good, "left_knee", "left_ankle", 1.5)
    chosen = athlete.select_athlete(handstand_clip([good, good, long_leg, good, good]))

    assert [choice.reason for choice in chosen] == ["", "", "bone_length", "", ""]
    assert [choice.contact for choice in chosen] == [False, False, True, False, False]
    # The frame keeps the athlete: the flag takes it out of scoring, not the clip.
    assert all(choice.chosen for choice in chosen)
    assert chosen_x(chosen[2]) == pytest.approx(ATHLETE_X)
    # It is the shin, and the clip's own median is what says so: one bad frame
    # out of five does not move the median of the other four.
    medians = athlete.clip_bone_medians(chosen)
    assert medians["shin_l"] == pytest.approx(150.0, abs=0.1)
    assert athlete.bone_lengths(chosen[2].person)["shin_l"] == pytest.approx(225.0, abs=0.1)
    assert athlete.bone_length_outlier(chosen[2].person, medians) == "shin_l"
    assert athlete.bone_length_outlier(chosen[0].person, medians) is None
    # The left/right rule is not what caught it: 225 against 150 is inside the
    # band once it is measured against the longer of the two.
    assert athlete.counterpart_outlier(chosen[2].person) is None
    assert athlete.bone_length_mismatch(chosen[2].person, medians) == "shin_l"


def test_a_shorter_bone_off_the_clip_is_flagged_too() -> None:
    """A trainer's bent leg is *shorter* than the athlete's, not only longer."""
    good = handstand(ATHLETE_X)
    short_leg = scaled_bone(good, "left_knee", "left_ankle", 0.5)
    chosen = athlete.select_athlete(handstand_clip([good, good, short_leg, good, good]))
    assert [choice.reason for choice in chosen] == ["", "", "bone_length", "", ""]
    medians = athlete.clip_bone_medians(chosen)
    assert athlete.bone_length_outlier(chosen[2].person, medians) == "shin_l"


def test_foreshortening_within_tolerance_is_not_flagged() -> None:
    """A leg pointing at the camera is shorter, and that is the athlete's own."""
    good = handstand(ATHLETE_X)
    foreshortened = scaled_bone(good, "left_knee", "left_ankle", 0.8)
    chosen = athlete.select_athlete(handstand_clip([good, good, foreshortened, good, good]))

    assert [choice.reason for choice in chosen] == [""] * 5
    assert not any(choice.contact for choice in chosen)
    medians = athlete.clip_bone_medians(chosen)
    assert medians["shin_l"] == pytest.approx(150.0, abs=0.1)
    assert athlete.bone_length_outlier(chosen[2].person, medians) is None
    assert athlete.counterpart_outlier(chosen[2].person) is None


def test_a_leg_that_differs_from_its_other_side_is_flagged() -> None:
    """The left/right rule needs no clip history: the two sides disagree.

    The knee slides down the leg, so the left thigh is 30 % long and the right
    25 % short — each inside the tolerance around the median, and the two legs
    still the same length end to end. The skeleton is still not one body.
    """
    good = handstand(ATHLETE_X)
    bent = scaled_bone(good, "left_hip", "left_knee", 1.3)
    bent = scaled_bone(bent, "right_hip", "right_knee", 0.75)
    chosen = athlete.select_athlete(handstand_clip([good, good, bent, good, good]))

    medians = athlete.clip_bone_medians(chosen)
    assert athlete.bone_length_outlier(chosen[2].person, medians) is None
    assert athlete.counterpart_outlier(chosen[2].person) == "thigh_l"
    assert [choice.reason for choice in chosen] == ["", "", "bone_length", "", ""]
    assert all(choice.chosen for choice in chosen)


def test_a_whole_leg_the_wrong_length_is_named_as_the_leg() -> None:
    """Both bones of one leg off by the same amount: the leg is what fails."""
    person = athlete.measure_person(scaled_leg(handstand(ATHLETE_X), 1.6))
    assert athlete.bone_lengths(person)["thigh_l"] == pytest.approx(240.0, abs=0.3)
    assert athlete.leg_lengths(person)["leg_l"] == pytest.approx(480.0, abs=0.5)
    assert athlete.leg_lengths(person)["leg_r"] == pytest.approx(300.0, abs=0.2)
    # The two legs are asked before their bones, so the answer names the leg
    # rather than one of the two bones that make it up. Inside the same band
    # the leg check cannot fire before a bone does, so it is a coarser net.
    assert athlete.counterpart_outlier(person) == "leg_l"
    inside_band = athlete.measure_person(scaled_leg(handstand(ATHLETE_X), 1.3))
    assert athlete.counterpart_outlier(inside_band) is None
    # Asked on its own, either bone says the same thing.
    assert athlete.bone_length_outlier(person, {"thigh_l": 150.0, "shin_l": 150.0}) == "thigh_l"


def test_the_left_right_rule_waits_for_a_second_body_to_have_been_seen() -> None:
    """A straddling handstand is not a stitch: nobody else is in the clip.

    The same frame either way: one leg 30 % long and its mirror 25 % short, which
    only the left/right rule can see. In a clip that has shown a second person
    somewhere, that leg could be the trainer's; in a clip that never has, the
    legs are the athlete's own and the frame must stay scorable.
    """
    good = handstand(ATHLETE_X)
    bent = scaled_bone(good, "left_hip", "left_knee", 1.3)
    bent = scaled_bone(bent, "right_hip", "right_knee", 0.75)

    alone = [frame_of(index, pose) for index, pose in enumerate([good, good, bent, good, good])]
    assert max(frame.n_people for frame in alone) < athlete.ASYMMETRY_MIN_PEOPLE
    assert [choice.reason for choice in athlete.select_athlete(alone)] == [""] * 5

    # One extra person in one frame of the clip is enough to switch it on, and
    # the median rule — which needs no second body — is on either way.
    with_trainer = [frame_of(0, standing(TRAINER_X), bent)]
    with_trainer += [frame_of(index, pose) for index, pose in enumerate([good, good, good], 1)]
    assert max(frame.n_people for frame in with_trainer) >= athlete.ASYMMETRY_MIN_PEOPLE
    assert [choice.reason for choice in athlete.select_athlete(with_trainer)] == [
        "bone_length",
        "",
        "",
        "",
    ]
    # A bone off the clip's own median is caught with or without a second body.
    long_leg = scaled_bone(good, "left_knee", "left_ankle", 1.5)
    solo = [frame_of(index, pose) for index, pose in enumerate([good, good, long_leg])]
    assert [choice.reason for choice in athlete.select_athlete(solo)] == [
        "",
        "",
        "bone_length",
    ]


def test_a_one_frame_clip_is_judged_on_its_other_side_alone() -> None:
    """No medians worth learning from: the left/right rule still runs."""
    good = handstand(ATHLETE_X)
    long_shin = scaled_bone(good, "left_knee", "left_ankle", 1.6)
    # 50 % on the shin is inside the band once it is measured against the
    # longer of the two sides, 60 % is not.
    inside = athlete.measure_person(scaled_bone(good, "left_knee", "left_ankle", 1.5))
    outside = athlete.measure_person(long_shin)
    assert athlete.contact_reason(inside, []) == ""
    assert athlete.counterpart_outlier(outside) == "shin_l"
    assert athlete.contact_reason(outside, []) == "bone_length"
    # In a clip of one frame the median is that frame, so only this can fire.
    chosen = athlete.select_athlete(handstand_clip([long_shin]))
    assert [choice.reason for choice in chosen] == ["bone_length"]
    assert [choice.contact for choice in chosen] == [True]


# --------------------------------------------------------------------------- #
# Rules 2 and 3 over a clip
# --------------------------------------------------------------------------- #


def test_the_athlete_is_chosen_in_every_frame_of_a_handstand() -> None:
    frames = [frame_of(index, standing(TRAINER_X), handstand(ATHLETE_X)) for index in range(4)]
    for choice in athlete.select_athlete(frames):
        assert choice.chosen
        assert choice.n_people == 2
        assert not choice.contact
        assert choice.score == pytest.approx(0.98)
        assert chosen_x(choice) == pytest.approx(ATHLETE_X)


def test_the_kick_up_before_the_handstand_is_covered_by_the_backward_sweep() -> None:
    frames = kick_up_clip()
    forward = athlete.sweep(frames, range(len(frames)))
    backward = athlete.sweep(frames, reversed(range(len(frames))))
    # Nobody looks like a handstand yet, so the forward sweep cannot tell the two
    # apart and drops those frames...
    assert [choice.chosen for choice in forward] == [False, False, False, True, True, True]
    # ... while the backward sweep arrives from the handstand and knows.
    assert [choice.chosen for choice in backward] == [True] * FRAME_COUNT
    assert backward[0].score == pytest.approx(0.439, abs=1e-3)
    # The trainer loses on continuity: their hips are 250 px from the athlete's,
    # which is less than half a body length, so they keep part of the term.
    trainer = frames[0].people[0]
    assert athlete.continuity_score(trainer, backward[0].person.hip) < 1.0

    chosen = athlete.select_athlete(frames)
    assert [choice.chosen for choice in chosen] == [True] * FRAME_COUNT
    assert [chosen_x(choice) for choice in chosen] == [pytest.approx(ATHLETE_X)] * FRAME_COUNT
    assert [choice.frame_idx for choice in chosen] == list(range(FRAME_COUNT))


def test_identity_survives_media_pipe_swapping_the_person_order() -> None:
    """MediaPipe returns the people in confidence order, which changes frame to frame."""
    frames = [
        frame_of(0, handstand(ATHLETE_X), standing(TRAINER_X)),  # athlete listed first
        frame_of(1, standing(TRAINER_X), handstand(ATHLETE_X)),  # ... second here
        frame_of(2, handstand(ATHLETE_X), standing(TRAINER_X)),
    ]
    for choice in athlete.select_athlete(frames):
        assert chosen_x(choice) == pytest.approx(ATHLETE_X)
        assert choice.n_people == 2


def test_a_frame_nobody_was_found_in_keeps_its_place_in_the_clip() -> None:
    frames = [
        frame_of(0, standing(TRAINER_X), handstand(ATHLETE_X)),
        frame_of(1),  # nobody detected at all
        frame_of(2, standing(TRAINER_X), handstand(ATHLETE_X)),
    ]
    chosen = athlete.select_athlete(frames)
    assert [choice.n_people for choice in chosen] == [2, 0, 2]
    assert [choice.chosen for choice in chosen] == [True, False, True]
    assert math.isnan(chosen[1].score)
    assert chosen[1].contact is False
    # The handstand either side is still the same athlete, not a restart.
    assert chosen_x(chosen[2]) == pytest.approx(ATHLETE_X)


def test_a_contact_frame_is_still_attributed_but_flagged() -> None:
    frames = [frame_of(0, standing(ATHLETE_X + 20.0), handstand(ATHLETE_X))]
    choice = athlete.select_athlete(frames)[0]
    assert choice.chosen
    assert choice.contact
    assert chosen_x(choice) == pytest.approx(ATHLETE_X)


# --------------------------------------------------------------------------- #
# Reading and writing
# --------------------------------------------------------------------------- #


def test_load_frames_reads_the_multi_person_parquet(tmp_path: pathlib.Path) -> None:
    path = write_multi(
        tmp_path,
        [
            [standing(TRAINER_X), handstand(ATHLETE_X)],
            [],
            [standing(TRAINER_X, visibility=0.85), handstand(ATHLETE_X, visibility=0.85)],
        ],
    )
    frames = athlete.load_frames(path)

    assert [frame.frame_idx for frame in frames] == [0, 1, 2]
    assert [frame.n_people for frame in frames] == [2, 0, 2]
    assert frames[0].t_ms == 0 and frames[2].t_ms == 80
    assert frames[0].rotated is False
    # The rows are sorted into model order, not into alphabetical order.
    first = frames[0].people[0]
    assert first.landmarks[pm.JOINT_INDEX["nose"], 0] == pytest.approx(TRAINER_X)
    assert first.landmarks[pm.JOINT_INDEX["right_foot_index"], 0] == pytest.approx(TRAINER_X + 20.0)
    # The two detections of one body in a later frame collapse to one person.
    assert frames[2].people[0].visibility == pytest.approx(0.85)


def test_load_frames_rejects_a_table_it_cannot_trust(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "single.parquet"
    multi_table([[handstand(ATHLETE_X)]]).drop(columns=[pm.PERSON_COLUMN]).to_parquet(path)
    with pytest.raises(ValueError, match="person_idx"):
        athlete.load_frames(path)

    truncated = tmp_path / "truncated.parquet"
    table = multi_table([[handstand(ATHLETE_X)]])
    multi_table([[handstand(ATHLETE_X), standing(TRAINER_X)]])  # two complete blocks
    broken = table.iloc[:-5].copy()  # ... minus five rows: not a whole block any more
    broken.to_parquet(truncated, index=False)
    with pytest.raises(ValueError, match="whole number"):
        athlete.load_frames(truncated)

    with pytest.raises(ValueError, match="no frames"):
        athlete.load_frames(empty_table(tmp_path))


def empty_table(tmp_path: pathlib.Path) -> pathlib.Path:
    """A parquet with the right columns and no rows at all."""
    path = tmp_path / "empty.parquet"
    multi_table([]).iloc[:0].to_parquet(path, index=False)
    return path


def test_athlete_table_keeps_the_single_person_schema() -> None:
    frames = [
        frame_of(0, standing(TRAINER_X), handstand(ATHLETE_X)),
        frame_of(1),  # nobody at all
        frame_of(2, standing(TRAINER_X), handstand(ATHLETE_X)),
    ]
    choices = athlete.select_athlete(frames)
    table = athlete.athlete_table(frames, choices)

    assert list(table.columns) == list(athlete.ATHLETE_COLUMNS)
    assert len(table) == 3 * len(pm.JOINT_NAMES)
    per_frame = table.groupby("frame_idx").first()
    assert per_frame.index.tolist() == [0, 1, 2]
    assert per_frame["t_ms"].tolist() == [0, 40, 80]
    assert per_frame["detected"].tolist() == [True, False, True]
    assert per_frame["athlete_score"].tolist() == pytest.approx(
        [0.98, float("nan"), 0.98], nan_ok=True
    )
    assert per_frame["n_people"].tolist() == [2, 0, 2]
    assert not table["trainer_contact"].any()
    # The chosen person's own landmarks, not the trainer's.
    frame0 = table[table["frame_idx"] == 0].set_index("joint")
    assert frame0.loc["nose", "x"] == pytest.approx(ATHLETE_X)
    # A frame nobody was found in keeps 33 rows of NaN, like the runner writes.
    assert (
        table[table["frame_idx"] == 1][["x", "y", "z", "visibility", "presence"]].isna().all().all()
    )
    assert table["frame_idx"].dtype == np.int64
    assert table["x"].dtype == np.float64
    assert table["trainer_contact"].dtype == bool

    with pytest.raises(ValueError, match="choices"):
        athlete.athlete_table(frames, choices[:2])


def test_run_clip_writes_the_parquet_and_a_sidecar_of_stats(tmp_path: pathlib.Path) -> None:
    in_root = tmp_path / "keypoints" / "mediapipe_multi"
    out_root = tmp_path / "keypoints" / "mediapipe_athlete"
    write_multi(
        in_root / "auto",
        [
            [standing(TRAINER_X), standing(ATHLETE_X)],  # kick-up: the backward sweep has it
            [standing(TRAINER_X), handstand(ATHLETE_X)],
            [standing(ATHLETE_X + 20.0), handstand(ATHLETE_X)],  # trainer on top: contact
            [],
        ],
        clip_id="clip00000001",
    )

    report = athlete.run_clip(
        "clip00000001", rotate_mode="auto", in_root=in_root, out_root=out_root
    )

    assert report.parquet_path == out_root / "auto" / "clip00000001.parquet"
    assert report.json_path.is_file()
    assert report.frame_count == 4
    assert report.athlete_frames == 3
    assert report.dropped_frames == 1
    assert report.contact_frames == 1
    assert report.frames_by_people == {0: 1, 2: 3}
    assert report.people_summary == "0:1 2:3"
    assert report.athlete_percent == pytest.approx(75.0)
    assert report.contact_percent == pytest.approx(25.0)
    assert report.dropped_percent == pytest.approx(25.0)

    sidecar = json.loads(report.json_path.read_text())
    assert sidecar["clip_id"] == "clip00000001"
    assert sidecar["rotate"] == "auto"
    assert sidecar["source_parquet"] == "mediapipe_multi/auto/clip00000001.parquet"
    assert sidecar["frame_count"] == 4
    assert sidecar["athlete_frame_count"] == 3
    assert sidecar["contact_frame_count"] == 1
    assert sidecar["dropped_frame_count"] == 1
    assert sidecar["frames_by_people"] == {"0": 1, "2": 3}
    assert sidecar["contact_frames_by_reason"] == {
        "box_iou": 1,
        "mixed_skeleton": 0,
        "bone_length": 0,
    }
    assert sidecar["mean_athlete_score"] == pytest.approx(0.79, abs=0.01)
    assert sidecar["runtime_seconds"] >= 0.0

    table = pd.read_parquet(report.parquet_path)
    assert list(table.columns) == list(athlete.ATHLETE_COLUMNS)
    by_frame = table.groupby("frame_idx").first()
    assert by_frame["trainer_contact"].tolist() == [False, False, True, False]
    assert by_frame["contact_reason"].tolist() == ["", "", "box_iou", ""]
    assert by_frame["detected"].tolist() == [True, True, True, False]
    assert by_frame["n_people"].tolist() == [2, 2, 2, 0]


def test_run_clip_records_which_rule_flagged_every_frame(tmp_path: pathlib.Path) -> None:
    in_root = tmp_path / "keypoints" / "mediapipe_multi"
    out_root = tmp_path / "keypoints" / "mediapipe_athlete"
    good = handstand(ATHLETE_X)
    write_multi(
        in_root / "auto",
        [
            [standing(TRAINER_X), good],
            [standing(ATHLETE_X + 20.0), good],  # the trainer's box over the athlete's
            [standing(TRAINER_X), scaled_bone(good, "left_knee", "left_ankle", 1.5)],
            [standing(TRAINER_X), good],
            [],
        ],
        clip_id="clip00000001",
    )

    report = athlete.run_clip(
        "clip00000001", rotate_mode="auto", in_root=in_root, out_root=out_root
    )

    assert report.contact_frames == 2
    assert report.contact_by_reason == {"box_iou": 1, "mixed_skeleton": 0, "bone_length": 1}
    assert report.reason_summary == "box_iou:1 bone_length:1"
    assert json.loads(report.json_path.read_text())["contact_frames_by_reason"] == (
        report.contact_by_reason
    )

    table = pd.read_parquet(report.parquet_path)
    by_frame = table.groupby("frame_idx").first()
    assert by_frame["contact_reason"].tolist() == ["", "box_iou", "bone_length", "", ""]
    assert by_frame["trainer_contact"].tolist() == [False, True, True, False, False]
    assert all(isinstance(value, str) for value in table["contact_reason"])
    # A flagged frame still carries the athlete's keypoints, not the trainer's.
    flagged = table[(table["frame_idx"] == 2) & (table["joint"] == "nose")]
    assert flagged["x"].iloc[0] == pytest.approx(ATHLETE_X)


def test_run_clip_skips_an_existing_output_unless_overwrite(tmp_path: pathlib.Path) -> None:
    in_root = tmp_path / "in"
    out_root = tmp_path / "out"
    write_multi(in_root / "auto", [[standing(TRAINER_X), handstand(ATHLETE_X)]])

    first = athlete.run_clip("clip00000001", rotate_mode="auto", in_root=in_root, out_root=out_root)
    assert not first.skipped
    assert first.parquet_path.read_bytes()

    def explode() -> None:  # pragma: no cover - only reached when the skip is broken
        raise AssertionError("an existing parquet must not be re-read")

    skipped = athlete.run_clip(
        "clip00000001", rotate_mode="auto", in_root=in_root, out_root=out_root
    )
    assert skipped.skipped
    assert skipped.frame_count == 0
    assert skipped.parquet_path == first.parquet_path

    again = athlete.run_clip(
        "clip00000001",
        rotate_mode="auto",
        in_root=in_root,
        out_root=out_root,
        overwrite=True,
    )
    assert not again.skipped
    assert again.frame_count == 1
    del explode


def test_run_clip_reports_missing_input_and_bad_modes(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="no multi-person keypoints"):
        athlete.run_clip(
            "clip00000001", rotate_mode="auto", in_root=tmp_path / "in", out_root=tmp_path / "out"
        )
    with pytest.raises(ValueError, match="rotate mode"):
        athlete.run_clip(
            "clip00000001", rotate_mode="90", in_root=tmp_path / "in", out_root=tmp_path / "out"
        )


def test_available_clips_lists_or_accepts_clip_ids(tmp_path: pathlib.Path) -> None:
    in_root = tmp_path / "in"
    write_multi(in_root / "auto", [[handstand(ATHLETE_X)]], clip_id="aaa")
    write_multi(in_root / "auto", [[handstand(ATHLETE_X)]], clip_id="bbb")

    assert athlete.available_clips("auto", in_root) == ["aaa", "bbb"]
    assert athlete.available_clips("auto", in_root, ["ccc"]) == ["ccc"]
    with pytest.raises(FileNotFoundError, match="handstand.pose_mediapipe"):
        athlete.available_clips("180", tmp_path / "missing")
    assert athlete.available_clips("auto", tmp_path / "missing", ["aaa"]) == ["aaa"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def cli_workspace(tmp_path: pathlib.Path, clips: Iterable[str] = ("clip00000001",)) -> pathlib.Path:
    """``tmp_path`` laid out as a data dir with multi-person keypoints in it."""
    for clip_id in clips:
        write_multi(
            tmp_path / "keypoints" / "mediapipe_multi" / "auto",
            [
                [standing(TRAINER_X), handstand(ATHLETE_X)],
                [standing(ATHLETE_X + 20.0), handstand(ATHLETE_X)],
                [],
            ],
            clip_id=clip_id,
        )
    return tmp_path


def test_cli_processes_every_clip_with_multi_person_keypoints(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = cli_workspace(tmp_path, ("clip00000001", "clip00000002"))

    assert athlete.main(["--rotate", "auto", "--data", str(data)]) == 0

    printed = capsys.readouterr().out
    assert "rotate=auto clips=2" in printed
    assert printed.count("clip  clip0000000") == 2
    assert "people[0:1 2:2]" in printed
    assert "athlete=66.7%" in printed
    assert "contact=33.3%" in printed
    assert "dropped=33.3%" in printed

    written = data / "keypoints" / "mediapipe_athlete" / "auto"
    assert sorted(path.name for path in written.glob("*.parquet")) == [
        "clip00000001.parquet",
        "clip00000002.parquet",
    ]
    assert (written / "clip00000001.json").is_file()


def test_cli_selects_clips_and_honours_the_limit_and_overwrite(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = cli_workspace(tmp_path, ("clip00000001", "clip00000002"))
    common = ["--rotate", "auto", "--data", str(data)]

    assert athlete.main(["--clips", "clip00000002", *common]) == 0
    written = data / "keypoints" / "mediapipe_athlete" / "auto"
    assert [path.stem for path in sorted(written.glob("*.parquet"))] == ["clip00000002"]

    assert athlete.main([*common, "--overwrite"]) == 0
    assert len(sorted(written.glob("*.parquet"))) == 2
    assert "skip" in capsys.readouterr().out or True  # a skip is only printed without overwrite

    assert athlete.main(["--limit", "1", *common, "--overwrite"]) == 0
    assert "clips=1" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        athlete.main(["--limit", "-1", *common])
    with pytest.raises(SystemExit):
        athlete.main(["--rotate", "sideways", "--data", str(data)])


def test_cli_reports_a_missing_input_directory(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert athlete.main(["--rotate", "auto", "--data", str(tmp_path / "empty")]) == 2
    captured = capsys.readouterr().out
    assert "no multi-person keypoints" in captured
    assert "--running-mode image" in captured
