"""Tests for :mod:`handstand.sides`, the left/right near-side measurement (chainlink #88).

Everything is hand-built in ``tmp_path``: no real labels, keypoints or phases
are read, so nothing here depends on the data directory.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping

import pandas as pd
import pytest

from handstand import sides

#: One joint's ``(x, y, z, visibility)`` in the athlete parquet.
Joint = tuple[float, float, float, float]

PARTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


def pose(
    nose_x: float = 10.0,
    *,
    nose_vis: float = 0.9,
    left_vis: float = 0.9,
    right_vis: float = 0.2,
    left_z: float = -0.3,
    right_z: float = 0.1,
) -> dict[str, Joint]:
    """An upright body facing right (nose 10 px right of the shoulders), not inverted."""
    joints: dict[str, Joint] = {"nose": (nose_x, 50.0, 0.0, nose_vis)}
    heights = {"shoulder": 40.0, "elbow": 30.0, "wrist": 20.0, "hip": 70.0, "knee": 85.0}
    heights["ankle"] = 100.0
    for part in PARTS:
        for side, vis, z in (("left", left_vis, left_z), ("right", right_vis, right_z)):
            joints[f"{side}_{part}"] = (0.0, heights[part], z, vis)
    return joints


def write_athlete(path: pathlib.Path, frames: Mapping[int, Mapping[str, Joint]]) -> None:
    """A long-format keypoint parquet with the athlete schema."""
    rows = []
    for frame, joints in sorted(frames.items()):
        for joint, (x, y, z, vis) in joints.items():
            rows.append(
                {
                    "frame_idx": frame,
                    "t_ms": frame * 40,
                    "joint": joint,
                    "x": x,
                    "y": y,
                    "z": z,
                    "visibility": vis,
                    "presence": 0.9,
                    "rotated": False,
                    "detected": True,
                    "athlete_score": 0.9,
                    "n_people": 1,
                    "trainer_contact": False,
                    "contact_reason": "",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def write_phases(path: pathlib.Path, holds: Mapping[int, int]) -> None:
    """A phase parquet: ``frame -> hold_id`` (``-1`` for no hold)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame_idx": list(holds),
            "t_ms": [frame * 40 for frame in holds],
            "phase": ["hold" if hold >= 0 else "none" for hold in holds.values()],
            "hold_id": list(holds.values()),
        }
    ).to_parquet(path)


def write_labels(root: pathlib.Path, rows: list[tuple[str, int, str, float, float]]) -> None:
    """``labels/keypoints.csv`` rows: ``(clip, frame, joint, x, y)``, all visible."""
    lines = ["clip_id,frame_idx,joint,x,y,visible,labeler,labeled_at"]
    lines += [
        f"{clip},{frame},{joint},{x},{y},true,me,2026-10-01" for clip, frame, joint, x, y in rows
    ]
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "keypoints.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gt_frame(frame: int, side: str, nose_x: float) -> list[tuple[str, int, str, float, float]]:
    """A labelled frame with one side's shoulder, wrist and ankle plus the nose."""
    return [
        ("c1", frame, "nose", nose_x, 50.0),
        ("c1", frame, f"{side}_shoulder", 0.0, 40.0),
        ("c1", frame, f"{side}_wrist", 0.0, 20.0),
        ("c1", frame, f"{side}_ankle", 0.0, 100.0),
    ]


# --------------------------------------------------------------------------- #
# near_side_from_nose
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("nose_x", "inverted", "expected"),
    [
        (10.0, False, "right"),  # facing right, upright: right side is near
        (10.0, True, "left"),  # facing right, inverted: flips to left
        (-10.0, False, "left"),  # facing left, upright
        (-10.0, True, "right"),  # facing left, inverted
    ],
)
def test_near_side_from_nose_covers_all_facing_and_inversion_combinations(
    nose_x: float, inverted: bool, expected: str
) -> None:
    assert sides.near_side_from_nose(nose_x, 0.0, inverted) == expected


def test_near_side_from_nose_is_none_for_missing_inputs() -> None:
    assert sides.near_side_from_nose(float("nan"), 0.0, False) is None
    assert sides.near_side_from_nose(10.0, float("nan"), False) is None
    assert sides.near_side_from_nose(None, 0.0, False) is None
    assert sides.near_side_from_nose(10.0, 0.0, None) is None


def test_near_side_from_nose_is_none_below_the_offset_threshold() -> None:
    assert sides.near_side_from_nose(1.0, 0.0, False) is None
    assert sides.near_side_from_nose(-1.5, 0.0, False) is None
    assert sides.near_side_from_nose(1.0, 0.0, False, min_offset_px=0.5) == "right"


# --------------------------------------------------------------------------- #
# claimed_near_side / near_side_from_z
# --------------------------------------------------------------------------- #


def test_claimed_near_side_picks_the_clearly_more_visible_side() -> None:
    left = {"shoulder": 0.9, "wrist": 0.8}
    right = {"shoulder": 0.2, "wrist": 0.3}
    assert sides.claimed_near_side(left, right) == "left"
    assert sides.claimed_near_side(right, left) == "right"


def test_claimed_near_side_is_none_within_the_margin() -> None:
    left = {"shoulder": 0.5, "hip": 0.5}
    right = {"shoulder": 0.45, "hip": 0.45}
    assert sides.claimed_near_side(left, right) is None


def test_claimed_near_side_ignores_parts_missing_on_one_side() -> None:
    left = {"shoulder": 0.9, "elbow": 0.9, "ankle": 0.0}
    right = {"shoulder": 0.2}
    assert sides.claimed_near_side(left, right) == "left"
    assert sides.claimed_near_side({"ankle": 0.9}, {"shoulder": 0.1}) is None


def test_near_side_from_z_takes_the_smaller_mean_z() -> None:
    assert sides.near_side_from_z({"shoulder": -0.3}, {"shoulder": 0.1}) == "left"
    assert sides.near_side_from_z({"shoulder": 0.2}, {"shoulder": -0.2}) == "right"


def test_near_side_from_z_is_none_within_the_margin_and_ignores_one_sided_parts() -> None:
    assert sides.near_side_from_z({"hip": 0.0}, {"hip": 0.03}) is None
    assert sides.near_side_from_z({"knee": -1.0}, {"hip": 0.5}) is None
    assert sides.near_side_from_z({"knee": -1.0, "hip": 0.0}, {"hip": 0.5}) == "left"


# --------------------------------------------------------------------------- #
# labelled_near_side / hold_vote
# --------------------------------------------------------------------------- #


def test_labelled_near_side_reads_the_only_labelled_side() -> None:
    only_left = {"left_shoulder": (0.0, 0.0), "left_ankle": (0.0, 1.0), "nose": (2.0, 2.0)}
    only_right = {"right_wrist": (0.0, 0.0)}
    assert sides.labelled_near_side(only_left) == "left"
    assert sides.labelled_near_side(only_right) == "right"


def test_labelled_near_side_is_none_when_both_or_neither_side_is_labelled() -> None:
    both = {"left_hip": (0.0, 0.0), "right_hip": (1.0, 0.0)}
    neither = {"nose": (0.0, 0.0), "left_eye": (1.0, 1.0)}
    assert sides.labelled_near_side(both) is None
    assert sides.labelled_near_side(neither) is None
    assert sides.labelled_near_side({}) is None


def test_hold_vote_takes_the_majority_of_non_none_votes() -> None:
    assert sides.hold_vote(["left", None, "right", "left"]) == "left"
    assert sides.hold_vote(["right"]) == "right"


def test_hold_vote_is_none_on_a_tie_or_without_votes() -> None:
    assert sides.hold_vote(["left", "right"]) is None
    assert sides.hold_vote(["left", None, "right", None]) is None
    assert sides.hold_vote([None, None]) is None
    assert sides.hold_vote([]) is None


# --------------------------------------------------------------------------- #
# measure_labels
# --------------------------------------------------------------------------- #


def build_labels_dataset(root: pathlib.Path) -> None:
    """Four labelled frames on clip c1 and one on a catalogue-missing clip c2.

    Frame 5: only the left side labelled, nose facing right -> rule says right
    (a disagreement with the label). Frame 6: only the right side labelled,
    nose facing right -> rule agrees. Frame 7: both sides labelled -> not
    testable. Frame 8: only the nose labelled -> not testable.
    """
    rows = (
        gt_frame(5, "left", nose_x=10.0)
        + gt_frame(6, "right", nose_x=10.0)
        + [
            *gt_frame(7, "left", nose_x=10.0),
            ("c1", 7, "right_shoulder", 0.0, 40.0),
            ("c1", 8, "nose", 10.0, 50.0),
            ("c2", 5, "nose", 10.0, 50.0),
            ("c2", 5, "left_shoulder", 0.0, 40.0),
        ]
    )
    write_labels(root, rows)
    (root / "catalogue.csv").write_text(
        "clip_id,filename,missing\nc1,a.mp4,false\nc2,b.mp4,true\n", encoding="utf-8"
    )
    write_athlete(
        root / "keypoints" / "mediapipe_athlete" / "best" / "c1.parquet",
        {frame: pose() for frame in range(5, 9)},
    )


def test_measure_labels_has_one_row_per_kept_frame_with_known_sides(tmp_path: pathlib.Path) -> None:
    build_labels_dataset(tmp_path)
    labels = sides.measure_labels(tmp_path)
    assert list(labels.columns) == list(sides.LABEL_COLUMNS)
    assert len(labels) == 4
    assert labels["frame_idx"].tolist() == [5, 6, 7, 8]
    assert set(labels["clip_id"]) == {"c1"}

    frame5 = labels.iloc[0]
    assert frame5["gt_side"] == "left"
    assert frame5["rule_side"] == "right"
    assert bool(frame5["testable"]) is True
    assert frame5["model_vis_side"] == "left"
    assert frame5["model_z_side"] == "left"

    frame6 = labels.iloc[1]
    assert frame6["gt_side"] == "right"
    assert frame6["rule_side"] == "right"

    assert labels["testable"].astype(bool).tolist() == [True, True, False, False]


def test_measure_labels_leaves_model_columns_empty_without_the_parquet(
    tmp_path: pathlib.Path,
) -> None:
    write_labels(tmp_path, gt_frame(5, "left", nose_x=10.0))
    labels = sides.measure_labels(tmp_path)
    assert len(labels) == 1
    assert pd.isna(labels.iloc[0]["model_vis_side"])
    assert pd.isna(labels.iloc[0]["model_z_side"])
    assert labels.iloc[0]["rule_side"] == "right"


# --------------------------------------------------------------------------- #
# measure_holds
# --------------------------------------------------------------------------- #


def build_holds_dataset(root: pathlib.Path) -> None:
    """Clip c1: holds 0 (frames 0-4) and 1 (frames 5-7), frames 8-9 unheld.

    Every frame faces right, so the rule votes right. The model's visibility and
    z both say left. Frame 6 has its nose hidden (visibility 0.2), so hold 1 has
    one frame without a rule vote. Clip c2 is catalogue-missing and skipped.
    """
    phases = {frame: 0 for frame in range(5)}
    phases.update({frame: 1 for frame in range(5, 8)})
    phases.update({8: -1, 9: -1})
    frames = {frame: pose() for frame in range(10)}
    frames[6] = pose(nose_vis=0.2)
    write_athlete(root / "keypoints" / "mediapipe_athlete" / "best" / "c1.parquet", frames)
    write_phases(root / "phases" / "mediapipe" / "c1.parquet", phases)
    write_athlete(root / "keypoints" / "mediapipe_athlete" / "best" / "c2.parquet", frames)
    write_phases(root / "phases" / "mediapipe" / "c2.parquet", phases)
    (root / "catalogue.csv").write_text(
        "clip_id,filename,missing\nc1,a.mp4,false\nc2,b.mp4,true\n", encoding="utf-8"
    )


def test_measure_holds_counts_votes_and_disagreements_per_hold(tmp_path: pathlib.Path) -> None:
    build_holds_dataset(tmp_path)
    holds = sides.measure_holds(tmp_path)
    assert list(holds.columns) == list(sides.HOLD_COLUMNS)
    assert holds[["clip_id", "hold_id"]].values.tolist() == [["c1", 0], ["c1", 1]]

    hold0, hold1 = holds.iloc[0], holds.iloc[1]
    assert hold0["n_frames"] == 5
    assert hold0["n_rule"] == 5
    assert hold0["rule_vote"] == "right"
    assert hold0["vis_vote"] == "left"
    assert hold0["z_vote"] == "left"
    assert hold0["frames_vis_disagree"] == 5
    assert hold0["frames_z_disagree"] == 5
    assert pd.isna(hold0["shape"])

    assert hold1["n_frames"] == 3
    assert hold1["n_rule"] == 2
    assert hold1["rule_vote"] == "right"
    assert hold1["frames_vis_disagree"] == 2


def test_measure_holds_passes_the_hold_shape_through(tmp_path: pathlib.Path) -> None:
    build_holds_dataset(tmp_path)
    (tmp_path / "labels").mkdir(exist_ok=True)
    (tmp_path / "labels" / "hold_shapes.csv").write_text(
        "clip_id,hold_id,shape,predicted_shape,agreed,labeled_at\nc1,0,line,line,true,2026-10-01\n",
        encoding="utf-8",
    )
    holds = sides.measure_holds(tmp_path)
    assert holds.iloc[0]["shape"] == "line"
    assert pd.isna(holds.iloc[1]["shape"])


def test_measure_holds_is_empty_without_phases(tmp_path: pathlib.Path) -> None:
    holds = sides.measure_holds(tmp_path)
    assert holds.empty
    assert list(holds.columns) == list(sides.HOLD_COLUMNS)


def test_measure_holds_has_no_nose_vote_when_the_nose_is_never_visible(
    tmp_path: pathlib.Path,
) -> None:
    frames = {0: pose(nose_vis=0.1), 1: pose(nose_vis=0.1)}
    write_athlete(tmp_path / "keypoints" / "mediapipe_athlete" / "best" / "c1.parquet", frames)
    write_phases(tmp_path / "phases" / "mediapipe" / "c1.parquet", {0: 0, 1: 0})
    holds = sides.measure_holds(tmp_path)
    assert pd.isna(holds.iloc[0]["rule_vote"])
    assert holds.iloc[0]["n_rule"] == 0


# --------------------------------------------------------------------------- #
# summary_text
# --------------------------------------------------------------------------- #


def test_summary_text_reports_agreement_and_disagreeing_frames(tmp_path: pathlib.Path) -> None:
    build_labels_dataset(tmp_path)
    build_holds_dataset(tmp_path)
    text = sides.summary_text(sides.measure_labels(tmp_path), sides.measure_holds(tmp_path))
    assert "testable (gt and rule both set): 2" in text
    assert "rule vs gt: 1/2" in text
    assert "model visibility vs gt: 1/2" in text
    assert "model z vs gt: 1/2" in text
    assert "(c1, 5)" in text
    assert "holds: 2" in text
    assert "holds with no nose vote: 0 of 2 (0.0%)" in text
    assert "rule_vote != vis_vote: 2 of 2 (100.0%)" in text


def test_summary_text_handles_empty_tables() -> None:
    text = sides.summary_text(
        pd.DataFrame(columns=list(sides.LABEL_COLUMNS)),
        pd.DataFrame(columns=list(sides.HOLD_COLUMNS)),
    )
    assert "rule vs gt: 0/0" in text
    assert "holds: 0" in text
    assert "none" in text


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_writes_three_reports(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_labels_dataset(tmp_path)
    build_holds_dataset(tmp_path)
    assert sides.main(["--data", str(tmp_path)]) == 0
    out = tmp_path / "reports" / "sides"
    assert sorted(p.name for p in out.iterdir()) == ["holds.csv", "labels.csv", "summary.txt"]
    assert len(pd.read_csv(out / "labels.csv")) == 4
    assert len(pd.read_csv(out / "holds.csv")) == 2
    assert "rule vs gt: 1/2" in capsys.readouterr().out


def test_cli_no_write_prints_but_writes_nothing(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_labels_dataset(tmp_path)
    build_holds_dataset(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert sides.main(["--data", str(tmp_path), "--no-write"]) == 0
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert "testable" in capsys.readouterr().out


def test_cli_fails_clearly_without_labels(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sides.main(["--data", str(tmp_path), "--no-write"]) != 0
    assert "keypoints.csv" in capsys.readouterr().err
