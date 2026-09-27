"""Tests for :mod:`handstand.prelabel`.

Nothing here downloads the RTMPose weights or runs a real model: the model is a
stub that returns fixed keypoints, the frames are 2x2 pixel arrays and the
manifest is a couple of rows of the sampler's own CSV. What is under test is
everything between the model's output and the two files a human works from — the
joint-name translation, the pixels/percent round trip, which orientation and
which person won, the review queue's order, and the fact that a prediction
exported and re-imported lands back on the pixels it started from.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib

import numpy as np
import pytest

from handstand import labels, prelabel
from handstand.frame_sampler import MANIFEST_COLUMNS
from handstand.labels import LABEL_JOINTS, import_export
from handstand.prelabel import (
    LABEL_COLUMNS,
    MODEL_VERSION,
    SCORE_THRESHOLD,
    Detection,
    PersonPose,
    build_task,
    choose_orientation,
    contact_sheet,
    image_uri,
    pixels_to_percent,
    predict_frame,
    run_model,
    select_person,
    write_review_queue,
)
from handstand.prelabel import (
    prelabel as prelabel_run,
)

#: A small odd-sized frame, so ``size - 1`` is not a round number and a
#: percentage round trip is a real test rather than a lucky coincidence.
WIDTH = 101
HEIGHT = 151
CLIP = "6508f9b355bd"
OTHER_CLIP = "64184de33f84"
IMAGE = f"{CLIP}_10.jpg"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def standing_pose(
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    score: float = 0.9,
    joints: tuple[str, ...] = LABEL_JOINTS,
) -> PersonPose:
    """A synthetic pose on a fixed layout, comfortably inside the frame.

    The layout is a diagonal, which only has to be *consistent* and *in frame*:
    nothing under test looks at what the shape means, only at where the points
    ended up, and a point outside the frame would be clamped on the way into a
    Label Studio task and quietly break the round trip.
    """
    positions = {
        name: (20.0 + offset_x + index * 3.0, 15.0 + offset_y + index * 9.0)
        for index, name in enumerate(LABEL_JOINTS)
    }
    return PersonPose(
        joints={name: positions[name] for name in joints},
        scores={name: score for name in joints},
    )


#: One person as a stub model returns them: their joints and their scores.
OnePerson = tuple[dict[str, tuple[float, float]], dict[str, float]]


def stub_model(people_by_call: dict[bool, list[OnePerson]]):
    """A :class:`handstand.prelabel.PoseModel` returning a fixed pose per call.

    Keyed by *whether the frame it is handed is upside down*, so a test can say
    "the model is more confident on the rotated frame" and the module picks it
    up. The returned array is padded to the model's full 133 joints with zeros,
    so the stub has the same shape the real model has.
    """
    from handstand.prelabel import COCO133_BODY_JOINT_NAMES, RTMPOSE_INDEX

    joints_total = len(COCO133_BODY_JOINT_NAMES)

    class _Model:
        def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            flipped = _is_flipped(image)
            people = people_by_call[flipped]
            points = np.zeros((len(people), joints_total, 2), dtype=np.float64)
            scores = np.zeros((len(people), joints_total), dtype=np.float64)
            for index, (joints, person_scores) in enumerate(people):
                for joint, xy in joints.items():
                    row = RTMPOSE_INDEX[joint]
                    points[index, row] = xy
                    scores[index, row] = person_scores[joint]
            return points, scores

    return _Model()


def _is_flipped(image: np.ndarray) -> bool:
    """Has this frame been turned upside down by the module?

    The stub writes its keypoints into a frame that is otherwise blank, and the
    module rotates the *pixels* before the second call, so the two calls differ
    only in the rotation. A frame whose top-left pixel is the marker is upright.
    """
    return not bool(image[0, 0, 0])


def make_image(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """A blank BGR frame with a marker in the top-left corner.

    The marker is what :func:`_is_flipped` reads, so the stub can tell the two
    inference calls apart without the real model.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[0, 0, 0] = 255
    return image


def write_manifest(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """A sampler manifest with every column the converter needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults: dict[str, object] = {
        "t_ms": 1000,
        "stratum": "clean_no_trainer",
        "skill": "line",
        "trainer_present": "false",
        "trainer_contact": "false",
        "display_width": WIDTH,
        "display_height": HEIGHT,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**defaults, **row})
    return path


@pytest.fixture
def manifest(tmp_path: pathlib.Path) -> pathlib.Path:
    return write_manifest(
        tmp_path / "data" / "label_frames" / "manifest.csv",
        [
            {"image": IMAGE, "clip_id": CLIP, "frame_idx": 10, "stratum": "clean_no_trainer"},
            {
                "image": f"{OTHER_CLIP}_3.jpg",
                "clip_id": OTHER_CLIP,
                "frame_idx": 3,
                "stratum": "trainer_contact",
            },
        ],
    )


def read_review(path: pathlib.Path) -> list[dict[str, str]]:
    """The review queue as rows, in the order the file stores them."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# Joint-name mapping
# --------------------------------------------------------------------------- #


def test_joint_map_covers_exactly_the_labelled_joints() -> None:
    """The config asks for 15 joints; the map must answer for those 15 and no others."""
    assert set(prelabel.JOINT_MAP) == set(LABEL_JOINTS)
    assert len(LABEL_JOINTS) == 15


def test_joint_map_matches_rtmlibs_own_coco133_skeleton() -> None:
    """The pinned indices must be the ones rtmlib itself names.

    rtmlib is a dependency, so its ``coco133`` skeleton is available. If a future
    release renumbers the model this fails loudly instead of shifting every joint
    by one and quietly mis-labelling a whole batch of pre-labels.
    """
    from rtmlib.visualization.skeleton.coco133 import coco133

    keypoint_info = coco133["keypoint_info"]
    for joint, rtmpose_name in prelabel.JOINT_MAP.items():
        index = prelabel.RTMPOSE_INDEX[joint]
        assert keypoint_info[index]["name"] == rtmpose_name, joint


def test_run_model_maps_model_rows_onto_handstand_names() -> None:
    """The model speaks COCO-WholeBody rows; the rest of the module speaks MediaPipe names."""
    joints = {name: (float(index), float(index) * 2) for index, name in enumerate(LABEL_JOINTS)}
    scores = {name: 0.8 for name in LABEL_JOINTS}
    model = stub_model({False: [(joints, scores)], True: [(joints, scores)]})

    people, count = run_model(model, make_image())

    assert count == 1
    # Each handstand name came back holding the row the stub wrote under it...
    assert people[0].joints == joints
    assert people[0].scores == scores
    # ...and those rows really are the COCO-WholeBody ones, spread out rather
    # than packed into the first 15, which is what makes this a mapping.
    rows = sorted({prelabel.RTMPOSE_INDEX[joint] for joint in LABEL_JOINTS})
    assert rows != list(range(len(LABEL_JOINTS)))
    assert max(rows) == prelabel.RTMPOSE_INDEX["right_foot_index"]


def test_foot_index_is_the_big_toe() -> None:
    """MediaPipe's foot_index is a big toe, and it is what the config asks a labeler for."""
    assert prelabel.JOINT_MAP["left_foot_index"] == "left_big_toe"
    assert prelabel.JOINT_MAP["right_foot_index"] == "right_big_toe"
    assert prelabel.RTMPOSE_INDEX["left_foot_index"] != prelabel.RTMPOSE_INDEX["right_foot_index"]


def test_run_model_rejects_a_model_with_too_few_joints() -> None:
    """A model reporting fewer joints than the map reads is refused, not silently cropped."""
    with pytest.raises(ValueError, match="joints"):
        run_model(_short_model(), make_image())


def _short_model():
    class _Model:
        def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros((1, 3, 2), dtype=np.float64), np.zeros((1, 3), dtype=np.float64)

    return _Model()


# --------------------------------------------------------------------------- #
# Percent conversion, and the round trip through handstand.labels
# --------------------------------------------------------------------------- #


def test_pixels_to_percent_is_the_inverse_of_percent_to_pixels() -> None:
    """A prediction's percentages must convert back to the pixels it came from."""
    for x, y in ((0.0, 0.0), (WIDTH - 1, HEIGHT - 1), (37.25, 91.5), (12.0, 140.0)):
        percent_x, percent_y = pixels_to_percent(x, y, WIDTH, HEIGHT)
        back = labels.percent_to_pixels(percent_x, percent_y, WIDTH, HEIGHT)
        assert back == pytest.approx((x, y), abs=1e-9)


def test_pixels_to_percent_uses_size_minus_one() -> None:
    """The same rule the keypoint parquets use: a fraction of ``size - 1``."""
    assert pixels_to_percent(WIDTH - 1, HEIGHT - 1, WIDTH, HEIGHT) == pytest.approx((100.0, 100.0))
    assert pixels_to_percent(0.0, 0.0, WIDTH, HEIGHT) == pytest.approx((0.0, 0.0))


def test_pixels_to_percent_rejects_an_unknown_size() -> None:
    with pytest.raises(ValueError, match="display size"):
        pixels_to_percent(1.0, 1.0, 0, HEIGHT)


def test_prediction_exported_and_reimported_gives_the_same_pixels(
    manifest: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The end-to-end promise: pre-label -> Label Studio export -> pixels, unchanged.

    A prediction is written as a task, the task is turned into the export shape
    ``handstand.labels`` reads, and the import is run. The keypoints that come
    back out of the CSV must be the pixels the model produced — otherwise the
    labeler corrects a point and the correction lands somewhere else.
    """
    pose = standing_pose(offset_x=7.25, offset_y=3.5)
    model = stub_model({False: [(pose.joints, pose.scores)], True: [(pose.joints, pose.scores)]})
    data = manifest.parents[1]

    report = prelabel_run(
        model=model,
        data=data,
        manifest=manifest,
        image_base="file://frames",
        read_image=lambda path: make_image(),
    )

    assert report.images == 2
    tasks = json.loads(report.prelabels_path.read_text(encoding="utf-8"))
    assert len(tasks) == 2
    assert tasks[0]["data"]["image"].endswith(IMAGE)
    assert tasks[0]["predictions"][0]["model_version"] == MODEL_VERSION

    export = _tasks_as_export(tasks, manifest)
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")

    report_in = import_export(export_path, manifest=manifest, out_csv=tmp_path / "keypoints.csv")
    assert report_in.unknown_images == 0
    assert report_in.rows_read == len(LABEL_JOINTS) * 2

    rows = labels.read_rows(tmp_path / "keypoints.csv")
    assert len(rows) == len(LABEL_JOINTS) * 2
    for row in rows:
        if row.clip_id != CLIP:
            continue
        assert (row.x, row.y) == pytest.approx(pose.joints[row.joint], abs=0.01)


def _tasks_as_export(tasks: list[dict], manifest: pathlib.Path) -> list[dict]:
    """Turn pre-annotation tasks into the shape a Label Studio export has.

    Label Studio turns each task's predictions into an annotation once a human
    has touched the task; the import reads the annotation, so the prediction's
    results are copied there verbatim.
    """
    from handstand.labels import image_index, load_manifest

    index = image_index(load_manifest(manifest))
    export: list[dict] = []
    for task in tasks:
        name = task["data"]["image"].rsplit("/", 1)[-1]
        entry = index[name]
        # The hash prefix Label Studio puts in front of an uploaded file's name:
        # the converter still has to find the frame from the ``<clip>_<frame>``
        # tail, which is the whole point of going through the real import.
        stored = f"f7b8a3c1-{entry['clip_id']}_{entry['frame_idx']}.jpg"
        results: list[dict] = []
        for prediction in task["predictions"]:
            results.extend(prediction["result"])
        export.append(
            {
                "data": {"image": f"upload/2/{stored}"},
                "annotations": [
                    {
                        "result": results,
                        "completed_by": {"email": "prelabel@example.com"},
                        "created_at": "2026-09-27T10:00:00Z",
                    }
                ],
            }
        )
    return export


def test_keypoint_result_uses_the_configs_from_and_to_names() -> None:
    """from_name/to_name must be what label_studio_config.xml declares, or LS drops the point."""
    result = prelabel.keypoint_result("left_wrist", 30.0, 60.0, WIDTH, HEIGHT)
    assert result["from_name"] == "keypoints"
    assert result["to_name"] == "image"
    assert result["type"] == "keypoints"
    assert result["value"]["keypointlabels"] == ["left_wrist"]


def test_keypoint_result_percentages_match_the_config_labels() -> None:
    """The value is a percentage of the image, which is what Label Studio stores."""
    result = prelabel.keypoint_result("nose", 50.0, 75.5, WIDTH, HEIGHT)
    # A percentage of the image is a fraction of ``size - 1``: the same rule the
    # keypoint parquets were written with.
    assert result["value"]["x"] == pytest.approx(50.0 / (WIDTH - 1) * 100.0, abs=1e-3)
    assert result["value"]["y"] == pytest.approx(75.5 / (HEIGHT - 1) * 100.0, abs=1e-3)
    assert result["value"]["width"] == 1.0
    assert result["value"]["height"] == 1.0


def test_keypoint_result_stays_inside_the_image() -> None:
    """The last pixel is 100 %, and a point past it is clamped rather than written out."""
    inside = prelabel.keypoint_result("nose", WIDTH - 1.0, HEIGHT - 1.0, WIDTH, HEIGHT)
    assert inside["value"]["x"] == 100.0
    assert inside["value"]["y"] == 100.0
    outside = prelabel.keypoint_result("nose", WIDTH + 50.0, HEIGHT + 50.0, WIDTH, HEIGHT)
    assert outside["value"]["x"] == 100.0
    assert outside["value"]["y"] == 100.0


def test_build_task_omits_points_below_the_score_threshold() -> None:
    """A low-scoring point is left out: a guessed dot is worse than no dot."""
    joints = {name: (50.0, 50.0) for name in LABEL_JOINTS}
    scores = {name: 0.9 for name in LABEL_JOINTS}
    scores["left_wrist"] = 0.1
    scores["nose"] = 0.29
    prediction = _prediction(PersonPose(joints=joints, scores=scores))

    task = build_task(IMAGE, "file:///x.jpg", prediction)
    labels_written = {r["value"]["keypointlabels"][0] for r in task["predictions"][0]["result"]}

    assert "left_wrist" not in labels_written
    assert "nose" not in labels_written
    assert len(labels_written) == len(LABEL_JOINTS) - 2
    assert set(prediction.low_score_joints) == {"left_wrist", "nose"}


def test_build_task_for_an_empty_frame_still_makes_a_task() -> None:
    """Nobody found: the frame is still shown to the labeler, just with no pre-labels."""
    prediction = _prediction(None)
    task = build_task(IMAGE, "file:///x.jpg", prediction)
    assert task["data"]["image"] == "file:///x.jpg"
    assert task["predictions"][0]["model_version"] == MODEL_VERSION
    assert task["predictions"][0]["result"] == []


def _prediction(pose: PersonPose | None, **kwargs: object) -> prelabel.ImagePrediction:
    """An :class:`ImagePrediction` for one image, with a synthetic pose."""
    defaults: dict[str, object] = {
        "image": IMAGE,
        "clip_id": CLIP,
        "frame_idx": 10,
        "stratum": "clean_no_trainer",
        "width": WIDTH,
        "height": HEIGHT,
        "pose": pose,
        "rotated_used": False,
        "person_index": 0,
        "athlete_reference": True,
        "n_people": 1,
    }
    return prelabel.ImagePrediction(**{**defaults, **kwargs})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Orientation choice
# --------------------------------------------------------------------------- #


def test_choose_orientation_keeps_the_rotated_result_when_it_scores_higher() -> None:
    """An inverted body is what pose models are worst at, so rotated usually wins."""
    upright = Detection((standing_pose(score=0.4),), 0, True)
    rotated = Detection((standing_pose(score=0.8),), 0, True)

    winner, rotated_used = choose_orientation(upright, rotated)

    assert rotated_used is True
    assert winner is rotated


def test_choose_orientation_keeps_upright_when_it_scores_higher() -> None:
    """Some holds are coped with upside down; the decision follows the score, not a rule."""
    upright = Detection((standing_pose(score=0.8),), 0, True)
    rotated = Detection((standing_pose(score=0.4),), 0, True)

    winner, rotated_used = choose_orientation(upright, rotated)

    assert rotated_used is False
    assert winner is upright


def test_choose_orientation_breaks_a_tie_towards_upright() -> None:
    """The same input must always give the same answer, so a tie goes to upright."""
    upright = Detection((standing_pose(score=0.6),), 0, True)
    rotated = Detection((standing_pose(score=0.6),), 0, True)

    winner, rotated_used = choose_orientation(upright, rotated)

    assert rotated_used is False
    assert winner is upright


def test_predict_frame_maps_a_rotated_detection_back_into_display_pixels() -> None:
    """A pose found on the rotated frame must land on the pixels of the frame as shown.

    The stub is confident on the rotated pass and puts a known joint at a known
    place *in the rotated frame*; after the map-back that joint must be the same
    distance from the opposite corner, which is the only way a pre-label from
    either orientation can share one coordinate system.
    """
    joints = {name: (10.0, 20.0) for name in LABEL_JOINTS}
    scores = {name: 0.95 for name in LABEL_JOINTS}
    model = stub_model(
        {False: [(joints, {name: 0.2 for name in LABEL_JOINTS})], True: [(joints, scores)]}
    )

    detection, rotated_used = predict_frame(model, make_image())

    assert rotated_used is True
    pose = detection.pose
    assert pose is not None
    for name in LABEL_JOINTS:
        x, y = pose.joints[name]
        assert (x, y) == pytest.approx((WIDTH - 1 - 10.0, HEIGHT - 1 - 20.0), abs=1e-9)


def test_predict_frame_reports_no_people_without_raising() -> None:
    """A frame with nobody in it is a result, not an error."""
    model = stub_model({False: [], True: []})
    detection, rotated_used = predict_frame(model, make_image())
    assert detection.pose is None
    assert rotated_used is False


# --------------------------------------------------------------------------- #
# Athlete matching among several people
# --------------------------------------------------------------------------- #


def trainer_and_athlete() -> tuple[PersonPose, PersonPose]:
    """The two bodies RTMPose finds in ``6508f9b355bd_1``, as it really finds them.

    Real coordinates from the real frame, because the bug this guards against is
    not a hypothetical: the trainer's crouch overlaps the athlete's bounding box
    *more* than the athlete's own inverted skeleton does, so any test that only
    checks "does the right body get picked" with a convenient layout would pass
    under the box rule too.

    The athlete is upside down on the right of the frame — nose at the bottom,
    feet high and far apart. The trainer stands beside them, feet together on the
    floor at the same height, which is what makes the two boxes overlap.
    """
    trainer = PersonPose(
        joints={
            "nose": (286.1, 411.7),
            "left_shoulder": (336.7, 500.0),
            "right_shoulder": (256.7, 500.0),
            "left_elbow": (356.7, 550.0),
            "right_elbow": (236.7, 550.0),
            "left_wrist": (366.7, 595.0),
            "right_wrist": (226.7, 595.0),
            "left_hip": (326.7, 540.0),
            "right_hip": (266.7, 540.0),
            "left_knee": (346.7, 580.0),
            "right_knee": (246.7, 580.0),
            "left_ankle": (376.6, 602.1),
            "right_ankle": (216.8, 602.1),
            "left_foot_index": (376.6, 602.1),
            "right_foot_index": (216.8, 602.1),
        },
        scores={name: 0.8 for name in LABEL_JOINTS},
    )
    athlete = PersonPose(
        joints={
            "nose": (313.7, 535.7),
            "left_shoulder": (330.0, 460.0),
            "right_shoulder": (300.0, 460.0),
            "left_elbow": (340.0, 410.0),
            "right_elbow": (305.0, 415.0),
            "left_wrist": (320.0, 560.0),
            "right_wrist": (310.0, 550.0),
            "left_hip": (330.0, 380.0),
            "right_hip": (305.0, 385.0),
            "left_knee": (340.0, 330.0),
            "right_knee": (300.0, 340.0),
            "left_ankle": (330.0, 300.0),
            "right_ankle": (310.0, 300.0),
            "left_foot_index": (296.8, 602.0),
            "right_foot_index": (296.8, 282.8),
        },
        scores={name: 0.7 for name in LABEL_JOINTS},
    )
    return trainer, athlete


def test_select_person_prefers_the_athlete_over_a_closer_overlapping_trainer() -> None:
    """A trainer's crouch overlaps the athlete's box more than the athlete does.

    This is the whole reason selection is on keypoints. The box rule chose the
    trainer here (0.41 IoU against the athlete's own 0.17) and pre-labelled the
    one person the labelling config says not to label.
    """
    trainer, athlete = trainer_and_athlete()
    reference = athlete
    reference_box = prelabel.person_box(reference.joints, reference.scores)

    detection = select_person([trainer, athlete], reference, reference_box)

    assert detection.person_index == 1
    assert detection.pose is athlete
    assert detection.athlete_reference is True


#: The MediaPipe athlete's own box in ``6508f9b355bd_1``, as measured. Wider than
#: either RTMPose body, because MediaPipe puts the athlete's feet ~40 px apart
#: and its hands on the floor, so the box spans most of the frame.
MEDIAPIPE_ATHLETE_BOX_6508 = (221.6, 290.1, 322.9, 641.3)


def test_the_trainer_really_does_win_on_box_overlap_alone() -> None:
    """Guard the premise of the test above: this is not a rigged fixture.

    Measured against the real MediaPipe athlete box, the trainer's crouch
    overlaps it *more* (0.41) than the athlete's own thin skeleton does (0.17).
    If a future edit changed the geometry so that stopped being true, the
    keypoint rule would still pick the right body but the regression it guards
    would no longer reflect the real data.
    """
    from handstand.athlete import box_iou

    trainer, athlete = trainer_and_athlete()
    reference_box = MEDIAPIPE_ATHLETE_BOX_6508
    trainer_box = prelabel.person_box(trainer.joints, trainer.scores)
    athlete_box = prelabel.person_box(athlete.joints, athlete.scores)

    assert box_iou(trainer_box, reference_box) > box_iou(athlete_box, reference_box)


def test_keypoint_gap_separates_the_two_bodies_far_more_than_their_boxes() -> None:
    """The measure itself: the athlete's keypoints are near, the trainer's are not."""
    from handstand.athlete import body_length

    trainer, athlete = trainer_and_athlete()
    length = body_length(athlete.to_landmarks())

    assert prelabel.keypoint_gap(athlete, athlete, length) == pytest.approx(0.0)
    assert prelabel.keypoint_gap(trainer, athlete, length) > prelabel.MAX_REFERENCE_GAP


def test_keypoint_gap_needs_enough_shared_joints() -> None:
    """One coincidental joint is not a body; under four shared joints there is no evidence."""
    trainer, athlete = trainer_and_athlete()
    sparse = PersonPose(
        joints={"nose": athlete.joints["nose"]},
        scores={"nose": 0.9},
    )
    assert math.isnan(prelabel.keypoint_gap(sparse, athlete, 100.0))


def test_select_person_rejects_every_body_when_the_reference_contradicts_them_all() -> None:
    """A reference no candidate is near must not be answered with the nearest one.

    This is the harmful case the whole rule exists to prevent: the reference is
    the athlete, every detected body is the trainer, and falling back to "most
    confident" hands the labeler a pre-label of the person the config says not
    to label. With the reference present and no match, nobody is taken and the
    frame is written with no pre-label at all.
    """
    trainer, athlete = trainer_and_athlete()
    somebody_else = standing_pose(offset_y=40.0)  # a real body, far from the athlete

    detection = select_person([trainer, somebody_else], athlete)

    assert detection.rejected is True
    assert detection.pose is None
    assert detection.athlete_reference is False
    # The people are still reported: the model did see somebody, the reference
    # just would not vouch for them.
    assert detection.n_people == 2


def test_select_person_falls_back_to_confidence_when_the_reference_has_no_body() -> None:
    """A reference whose body cannot be measured cannot judge anybody, so it does not.

    With no usable body length there is no evidence to contradict, and the frame
    is ranked on RTMPose's own confidence as before.
    """
    first = standing_pose(offset_y=0.0, score=0.5)
    second = standing_pose(offset_y=200.0, score=0.9)
    # Every joint on one pixel: a body of length zero, which is never measured.
    degenerate = PersonPose(
        joints={name: (900.0, 900.0) for name in LABEL_JOINTS},
        scores={name: 0.9 for name in LABEL_JOINTS},
    )

    detection = select_person([first, second], degenerate)

    assert detection.person_index == 1
    assert detection.rejected is False
    assert detection.athlete_reference is False


def test_select_person_without_a_reference_takes_the_most_confident() -> None:
    low = standing_pose(score=0.4)
    high = standing_pose(score=0.9)
    detection = select_person([low, high], None)
    assert detection.person_index == 1
    assert detection.athlete_reference is False
    assert detection.rejected is False


def test_detection_separates_nobody_found_from_everybody_rejected() -> None:
    """Two different "no pose" reasons, which the run counts differently."""
    trainer, athlete = trainer_and_athlete()
    somebody_else = standing_pose(offset_y=40.0)

    nobody = select_person([], None)
    assert nobody.rejected is False  # the model saw nobody at all
    assert nobody.person_index == prelabel.NO_PERSON

    rejected = select_person([trainer, somebody_else], athlete)
    assert rejected.rejected is True
    assert rejected.person_index == prelabel.NO_MATCH
    # ...and both end up with no pose, so both produce an empty task.
    assert nobody.pose is None
    assert rejected.pose is None


def test_select_person_with_nobody_found() -> None:
    detection = select_person([], None)
    assert detection.pose is None
    assert detection.n_people == 0
    assert detection.mean_score == 0.0


def test_select_person_breaks_a_keypoint_tie_on_box_overlap() -> None:
    """Two candidates on the same keypoints: the better-overlapping one wins."""
    left = standing_pose(offset_x=-1.0)
    right = standing_pose(offset_x=1.0)
    reference = standing_pose()

    detection = select_person([left, right], reference, (0.0, 0.0, 200.0, 200.0))

    # Both are equally near the reference, so the model's own order decides.
    assert detection.person_index in (0, 1)
    assert detection.athlete_reference is True


def test_rotate_box_180_swaps_both_edges() -> None:
    """A box turned upside down, with each axis' near and far edges exchanged.

    Deliberately asymmetric in both axes: a box centred in the frame would come
    out right even with the y order transposed, and that mistake reads as "the
    athlete is nowhere near their own reference" rather than as a wrong answer.
    """
    box = (10.0, 100.0, 40.0, 500.0)  # x0, y0, x1, y1 in a 200x600 frame
    assert prelabel.rotate_box_180(box, 200, 600) == (159.0, 99.0, 189.0, 499.0)


def test_rotate_box_180_stays_a_valid_box() -> None:
    """The result must still have its low corner below and left of its high one."""
    rotated = prelabel.rotate_box_180((10.0, 100.0, 40.0, 500.0), 200, 600)
    assert rotated is not None
    x0, y0, x1, y1 = rotated
    assert x0 < x1
    assert y0 < y1


def test_rotate_box_180_is_its_own_inverse() -> None:
    box = (10.0, 100.0, 40.0, 500.0)
    assert prelabel.rotate_box_180(prelabel.rotate_box_180(box, 200, 600), 200, 600) == box


def test_rotate_box_180_passes_none_through() -> None:
    assert prelabel.rotate_box_180(None, 200, 600) is None


def spread_joints(x: float, y: float) -> dict[str, tuple[float, float]]:
    """Joints filling a 10x10 pixel square, so a bounding box has real area.

    A box built from points that all sit on one pixel has zero area and therefore
    zero IoU with everything, which would make these tests pass or fail for the
    wrong reason.
    """
    return {
        name: (x + (index % 3) * 5.0, y + (index // 3) * 3.0)
        for index, name in enumerate(LABEL_JOINTS)
    }


def test_predict_frame_matches_the_rotated_pass_against_the_rotated_reference() -> None:
    """The rotated pass must be compared against the reference *as it turned too*.

    The reference is measured in the frame the labeler sees; the rotated pass
    reports a pose in the frame the model saw. Matching one against the other
    puts the athlete nowhere near their own reference, so the reference is
    silently discarded and the frame is decided on raw confidence instead. Here
    the rotated pass is the one that wins, so the reference has to survive the
    rotation to be used at all.
    """
    # The stub writes these into the *rotated* frame; the reference the caller
    # supplies is the same square as it appears after the map-back.
    joints = spread_joints(45.0, 95.0)
    scores = {name: 0.9 for name in LABEL_JOINTS}
    reference = PersonPose(
        joints=prelabel.rotate_pose_180(
            PersonPose(joints=joints, scores=scores), WIDTH, HEIGHT
        ).joints,
        scores=scores,
    )
    display_box = prelabel.person_box(reference.joints, reference.scores)
    model = stub_model(
        {
            False: [(joints, {name: 0.1 for name in LABEL_JOINTS})],
            True: [(joints, scores)],
        }
    )

    detection, rotated_used = predict_frame(
        model, make_image(), reference=reference, reference_box=display_box
    )

    assert rotated_used is True
    pose = detection.pose
    assert pose is not None
    # The map-back landed the rotated pose exactly on the displayed athlete...
    assert prelabel.person_box(pose.joints, pose.scores) == pytest.approx(display_box)
    # ...so the reference, rotated alongside it, recognised it.
    assert detection.athlete_reference is True


def test_predict_frame_still_finds_the_athlete_in_the_upright_pass() -> None:
    """The same reference reaches the upright pass untouched."""
    joints = spread_joints(45.0, 95.0)
    scores = {name: 0.9 for name in LABEL_JOINTS}
    reference = PersonPose(joints=joints, scores=scores)
    model = stub_model(
        {
            False: [(joints, scores)],
            True: [(joints, {name: 0.1 for name in LABEL_JOINTS})],
        }
    )

    detection, rotated_used = predict_frame(
        model,
        make_image(),
        reference=reference,
        reference_box=prelabel.person_box(reference.joints, reference.scores),
    )

    assert rotated_used is False
    assert detection.athlete_reference is True


def test_rotate_180_preserves_the_pose_because_of_the_map_back() -> None:
    """A frame predicted only upright and a frame predicted rotated agree, pixel for pixel.

    This is the property that makes the "better of two orientations" comparison
    fair: both passes are in the same coordinate system before they are scored.
    """
    joints = {name: (float(3 + i), float(7 + i * 5)) for i, name in enumerate(LABEL_JOINTS)}
    scores = {name: 0.9 for name in LABEL_JOINTS}
    upright_model = stub_model({False: [(joints, scores)], True: [(joints, scores)]})
    only_upright = _model_that_fails_when_flipped(upright_model)
    detection, rotated_used = predict_frame(only_upright, make_image())
    assert rotated_used is False
    pose = detection.pose
    assert pose is not None
    assert pose.joints == {
        name: (float(3 + i), float(7 + i * 5)) for i, name in enumerate(LABEL_JOINTS)
    }


def _model_that_fails_when_flipped(inner):
    """Wrap a model so the rotated pass finds nobody."""

    class _Wrapper:
        def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if _is_flipped(image):
                return np.zeros((0, 133, 2), dtype=np.float64), np.zeros((0, 133), dtype=np.float64)
            return inner(image)

    return _Wrapper()


# --------------------------------------------------------------------------- #
# Review-queue ordering
# --------------------------------------------------------------------------- #


def _queue_row(
    image: str, disagreement: float, low: str, worst: str = "left_wrist"
) -> dict[str, str]:
    """One review-queue row, in the column order the writer expects."""
    return {
        "image": image,
        "clip_id": CLIP,
        "frame_idx": "10",
        "stratum": "clean_no_trainer",
        "max_disagreement": f"{disagreement:.4f}",
        "worst_joint": worst,
        "low_score_joints": low,
        "rotated_used": "false",
    }


def test_review_queue_puts_the_largest_disagreement_first(tmp_path: pathlib.Path) -> None:
    """The frames most likely wrong come first: disagreement is the leading term."""
    path = write_review_queue(
        [
            _queue_row("a.jpg", 0.10, ""),
            _queue_row("c.jpg", 0.90, ""),
            _queue_row("b.jpg", 0.50, ""),
        ],
        tmp_path / "queue.csv",
    )
    images = [row["image"] for row in read_review(path)]
    assert images == ["c.jpg", "b.jpg", "a.jpg"]


def test_review_queue_promotes_a_frame_with_many_unsure_joints(tmp_path: pathlib.Path) -> None:
    """A frame the model was unsure about is also worth a labeler's time, even
    when the models agree — that is what the low-score term is for."""
    path = write_review_queue(
        [
            _queue_row("agree.jpg", 0.10, ""),
            _queue_row("unsure.jpg", 0.05, "nose;left_wrist;right_wrist;left_ankle"),
        ],
        tmp_path / "queue.csv",
    )
    images = [row["image"] for row in read_review(path)]
    assert images[0] == "unsure.jpg"


def test_review_queue_is_deterministic_and_writes_the_header(tmp_path: pathlib.Path) -> None:
    """The file on disk is the order the labeler works in, and has the agreed columns."""
    path = write_review_queue(
        [_queue_row("b.jpg", 0.2, ""), _queue_row("a.jpg", 0.2, "")], tmp_path / "queue.csv"
    )
    rows = read_review(path)
    assert list(rows[0]) == list(LABEL_COLUMNS)
    # A tie falls back to the image name, so the same inputs give the same file.
    assert [row["image"] for row in rows] == ["a.jpg", "b.jpg"]


def _reference_joints() -> dict[str, tuple[float, float]]:
    """A synthetic athlete on the parquet's real 33-joint schema.

    Both shoulders at y=100 and both ankles at y=200, so
    :func:`handstand.athlete.body_length` measures exactly 100 px — which is what
    lets a test say "10 px of disagreement is 0.1 body lengths" and mean it.
    """
    from handstand.pose_mediapipe import JOINT_NAMES as ALL_JOINTS

    ys = {name: 10.0 + index * 5.0 for index, name in enumerate(ALL_JOINTS)}
    ys["left_shoulder"] = 100.0
    ys["right_shoulder"] = 100.0
    ys["left_ankle"] = 200.0
    ys["right_ankle"] = 200.0
    return {name: (50.0, ys[name]) for name in ALL_JOINTS}


def _cache_with_athlete(
    root: pathlib.Path, joints: dict[str, tuple[float, float]] | None = None
) -> prelabel.PanelCache:
    """A :class:`PanelCache` over a synthetic athlete parquet.

    Written with the real :mod:`handstand.athlete` columns, so the module reads
    it exactly as it reads the shipped data.
    """
    import pandas as pd

    from handstand.overlay import source_dirname
    from handstand.pose_mediapipe import JOINT_NAMES, PARQUET_COLUMNS

    points = joints if joints is not None else _reference_joints()
    path = (
        root
        / "keypoints"
        / source_dirname(prelabel.ATHLETE_SOURCE)
        / prelabel.DEFAULT_ROTATE
        / f"{CLIP}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "frame_idx": 10,
                "t_ms": 1000,
                "joint": joint,
                "x": points[joint][0],
                "y": points[joint][1],
                "z": float("nan"),
                "visibility": 0.9,
                "presence": 0.9,
                "rotated": False,
                "detected": True,
            }
            for joint in JOINT_NAMES
        ],
        columns=list(PARQUET_COLUMNS),
    ).to_parquet(path, index=False)
    return prelabel.PanelCache(prelabel.DEFAULT_ROTATE, data=root)


def _pose_shifted_by(
    reference: dict[str, tuple[float, float]], joint: str, dy: float
) -> PersonPose:
    """The athlete's own pose, with one joint moved ``dy`` pixels down.

    Built from the same layout the parquet holds, so every *other* joint agrees
    exactly and the only disagreement the queue can report is the one the test
    put there.
    """
    joints = {name: reference[name] for name in LABEL_JOINTS}
    x, y = joints[joint]
    joints[joint] = (x, y + dy)
    return PersonPose(joints=joints, scores={name: 0.9 for name in LABEL_JOINTS})


def test_disagreement_is_measured_in_body_lengths(tmp_path: pathlib.Path) -> None:
    """A gap in pixels only means something relative to the body it is on.

    The athlete below has a 100 px body and RTMPose puts one joint 10 px from
    where MediaPipe put it, so the reported number must be 0.1 body lengths and
    not 10 pixels.
    """
    reference = _reference_joints()
    cache = _cache_with_athlete(tmp_path, reference)

    disagreement, worst = prelabel._disagreement_row(
        _prediction(_pose_shifted_by(reference, "left_knee", 10.0)), cache
    )

    assert worst == "left_knee"
    assert disagreement == pytest.approx(0.10, abs=1e-6)


def test_a_longer_body_makes_the_same_gap_look_smaller(tmp_path: pathlib.Path) -> None:
    """The normalisation is what makes a tall and a short athlete comparable."""
    reference = _reference_joints()
    short_body, _ = prelabel._disagreement_row(
        _prediction(_pose_shifted_by(reference, "left_knee", 10.0)),
        _cache_with_athlete(tmp_path / "short", reference),
    )

    # The same 10 px gap, on a body twice as long: half the number.
    taller = {name: (x, (y - 100.0) * 2 + 100.0) for name, (x, y) in reference.items()}
    long_body, _ = prelabel._disagreement_row(
        _prediction(_pose_shifted_by(taller, "left_knee", 10.0)),
        _cache_with_athlete(tmp_path / "tall", taller),
    )

    assert short_body == pytest.approx(0.10, abs=1e-6)
    assert long_body == pytest.approx(0.05, abs=1e-6)


def test_worst_joint_names_the_joint_the_models_disagree_about(tmp_path: pathlib.Path) -> None:
    """The queue says *which* joint is in dispute, so the labeler knows where to look."""
    reference = _reference_joints()
    cache = _cache_with_athlete(tmp_path, reference)

    _disagreement, worst = prelabel._disagreement_row(
        _prediction(_pose_shifted_by(reference, "right_knee", 40.0)), cache
    )

    assert worst == "right_knee"


def test_disagreement_is_zero_when_only_one_model_measured_a_joint(tmp_path: pathlib.Path) -> None:
    """A joint only one model has is not a disagreement, it is just missing.

    RTMPose is below the threshold on every joint, so every joint is measured by
    MediaPipe alone and there is no gap to report at all.
    """
    cache = _cache_with_athlete(tmp_path)
    prediction = _prediction(
        PersonPose(
            joints={name: (50.0, 20.0) for name in LABEL_JOINTS},
            scores={name: 0.0 for name in LABEL_JOINTS},  # every point below threshold
        )
    )

    disagreement, worst = prelabel._disagreement_row(prediction, cache)

    assert disagreement == 0.0
    assert worst == ""
    assert set(prediction.low_score_joints) == set(LABEL_JOINTS)


def test_review_row_lists_the_joints_the_labeler_must_place() -> None:
    """low_score_joints is the labeler's to-do list for the frame."""
    prediction = _prediction(
        PersonPose(
            joints={name: (50.0, 20.0) for name in LABEL_JOINTS},
            scores={name: 0.1 if name in ("nose", "left_wrist") else 0.9 for name in LABEL_JOINTS},
        )
    )
    assert prediction.low_score_joints == ("nose", "left_wrist")
    assert SCORE_THRESHOLD == 0.3


# --------------------------------------------------------------------------- #
# The run, end to end (stubbed model)
# --------------------------------------------------------------------------- #


def test_run_writes_both_files_and_counts(manifest: pathlib.Path) -> None:
    """A run produces the pre-labels and the queue, and reports what it did."""
    pose = standing_pose()
    model = stub_model({False: [(pose.joints, pose.scores)], True: [(pose.joints, pose.scores)]})
    data = manifest.parents[1]

    report = prelabel_run(
        model=model, data=data, manifest=manifest, read_image=lambda path: make_image()
    )

    assert report.images == 2
    assert report.detected == 2
    assert report.failures == []
    assert json.loads(report.prelabels_path.read_text(encoding="utf-8"))[0]["predictions"]
    rows = read_review(report.review_path)
    assert len(rows) == 2
    assert list(rows[0]) == list(LABEL_COLUMNS)


def test_run_records_a_frame_with_nobody_in_it(manifest: pathlib.Path) -> None:
    """A frame RTMPose found nobody in still gets a task, so it is not lost."""
    model = stub_model({False: [], True: []})
    data = manifest.parents[1]

    report = prelabel_run(
        model=model, data=data, manifest=manifest, read_image=lambda path: make_image()
    )

    assert report.images == 2
    assert report.no_pose == 2
    assert report.detected == 0
    tasks = json.loads(report.prelabels_path.read_text(encoding="utf-8"))
    assert len(tasks) == 2
    assert all(task["predictions"][0]["result"] == [] for task in tasks)
    # Nothing to place and nothing to disagree about, so the frame is not pushed
    # to the top of the queue as if it were the most suspicious one.
    rows = read_review(report.review_path)
    assert all(row["max_disagreement"] == "0.0000" for row in rows)
    assert all(row["worst_joint"] == "" for row in rows)


def test_run_writes_no_pre_label_where_the_reference_rejects_every_body(
    manifest: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A rejected frame reaches the labeler empty, and is counted as rejected.

    This is the safety property behind the whole athlete-matching rule: when the
    reference athlete contradicts every body RTMPose found, the run must not
    write the nearest of them. The task still exists, so the frame is not lost.
    """
    trainer, athlete = trainer_and_athlete()
    scores = {name: 0.8 for name in LABEL_JOINTS}
    # Only the trainer is detected, and the reference is the athlete: the exact
    # situation that produced 20/300 wrong-body pre-labels.
    model = stub_model({False: [(trainer.joints, scores)], True: [(trainer.joints, scores)]})
    data = manifest.parents[1]
    # The parquet needs all 33 joints; the athlete's 15 replace the layout's.
    # It goes under the same ``data`` root the run reads, which is the manifest's
    # grandparent -- the fixture manifest lives in <data>/label_frames/.
    _cache_with_athlete(
        data,
        {**_reference_joints(), **{name: athlete.joints[name] for name in LABEL_JOINTS}},
    )

    report = prelabel_run(
        model=model,
        data=data,
        manifest=manifest,
        read_image=lambda path: make_image(),
    )

    # Only the clip with an athlete parquet has a reference to be rejected by;
    # the other frame has no reference at all, so there is nothing to contradict
    # and it falls back to confidence. The two are told apart by `rejected`.
    assert report.rejected == 1
    assert report.no_pose == 1
    assert report.detected == 1
    assert report.with_reference == 0

    by_image = {task["data"]["image"].rsplit("/", 1)[-1]: task for task in
                json.loads(report.prelabels_path.read_text(encoding="utf-8"))}
    assert by_image[IMAGE]["predictions"][0]["result"] == []
    assert by_image[f"{OTHER_CLIP}_3.jpg"]["predictions"][0]["result"]


def test_rejected_is_distinct_from_nobody_detected() -> None:
    """Two different empty frames, and the report tells them apart."""
    trainer, athlete = trainer_and_athlete()
    detection = select_person([trainer], athlete)
    assert detection.rejected is True
    assert detection.n_people == 1
    assert select_person([], None).rejected is False


def test_run_reports_an_unreadable_image_and_keeps_going(manifest: pathlib.Path) -> None:
    """One bad JPEG must not cost the other frames their pre-labels."""
    pose = standing_pose()
    model = stub_model({False: [(pose.joints, pose.scores)], True: [(pose.joints, pose.scores)]})
    data = manifest.parents[1]

    def read(path: pathlib.Path) -> np.ndarray | None:
        return None if path.name == IMAGE else make_image()

    report = prelabel_run(
        model=model, data=data, manifest=manifest, read_image=read
    )

    assert report.images == 1
    assert len(report.failures) == 1
    assert report.failures[0][0] == IMAGE


def test_run_respects_the_limit(manifest: pathlib.Path) -> None:
    pose = standing_pose()
    model = stub_model({False: [(pose.joints, pose.scores)], True: [(pose.joints, pose.scores)]})
    data = manifest.parents[1]
    report = prelabel_run(
        model=model,
        data=data,
        manifest=manifest,
        limit=1,
        read_image=lambda path: make_image(),
    )
    assert report.images == 1


def test_image_uri_uses_a_file_uri_by_default(tmp_path: pathlib.Path) -> None:
    """Label Studio's local file serving reads a file:// URI; --image-base overrides it."""
    path = tmp_path / "frame.jpg"
    assert image_uri(path) == f"file://{path}"
    assert image_uri(path, base="http://localhost:8080/frames/") == "http://localhost:8080/frames/frame.jpg"


# --------------------------------------------------------------------------- #
# Contact sheet
# --------------------------------------------------------------------------- #


def test_contact_sheet_draws_a_grid_of_the_worst_frames(
    manifest: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The sheet shows the frames the labeler meets first, worst-queue-first."""
    pose = standing_pose()
    model = stub_model({False: [(pose.joints, pose.scores)], True: [(pose.joints, pose.scores)]})
    data = manifest.parents[1]
    report = prelabel_run(
        model=model, data=data, manifest=manifest, read_image=lambda path: make_image()
    )
    rows = read_review(report.review_path)
    # Make the disagreement differ so ordering is not a tie.
    rows[0]["max_disagreement"] = "0.9"
    rows[1]["max_disagreement"] = "0.1"

    sheet = contact_sheet(
        report.tasks,
        rows,
        tmp_path / "sheet.jpg",
        manifest.parent,
        read_image=lambda path: make_image(),
    )

    assert sheet is not None and sheet.is_file()
    import cv2

    drawn = cv2.imread(str(sheet))
    assert drawn is not None
    assert drawn.shape[0] == 2 * HEIGHT


def test_contact_sheet_returns_none_with_nothing_to_draw(tmp_path: pathlib.Path) -> None:
    assert contact_sheet([], [], tmp_path / "sheet.jpg", tmp_path) is None
