"""Tests for :mod:`handstand.orient_measure`.

The module measures rotation modes over a dataset, so the tests give it tiny
synthetic parquets: a two-frame scene with a person on their hands, the same
person upside down, a trainer standing next to them and a frame with nobody in
it. No real video, no real clip, and nothing that depends on a model.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from handstand import orient_measure as om
from handstand import pose_mediapipe as pm

FRAME_COUNT = 6
#: Where the synthetic bodies are, in a 576x1024 display frame.
UPRIGHT_WRIST_Y = 480.0
UPRIGHT_ANKLE_Y = 890.0
HANDSTAND_WRIST_Y = 900.0
HANDSTAND_ANKLE_Y = 200.0


def body(wrist_y: float, ankle_y: float, x: float = 300.0) -> np.ndarray:
    """A ``(33, 5)`` block with the wrists at ``wrist_y`` and the ankles at ``ankle_y``."""
    block = np.full((len(pm.JOINT_NAMES), 5), np.nan)
    for name, y in (
        ("left_wrist", wrist_y),
        ("right_wrist", wrist_y),
        ("left_ankle", ankle_y),
        ("right_ankle", ankle_y),
    ):
        block[pm.JOINT_INDEX[name]] = (x, y, 0.0, 0.9, 0.9)
    return block


def handstand_body(x: float = 300.0) -> np.ndarray:
    """A body on its hands: wrists on the mat, feet up."""
    return body(HANDSTAND_WRIST_Y, HANDSTAND_ANKLE_Y, x)


def upright_body(x: float = 300.0) -> np.ndarray:
    """A person on their feet: wrists by their sides, feet on the mat."""
    return body(UPRIGHT_WRIST_Y, UPRIGHT_ANKLE_Y, x)


def long_table(
    blocks: list[list[np.ndarray]],
    *,
    rotated: bool = False,
    t_ms_step: int = 40,
) -> pd.DataFrame:
    """The multi-person schema: one 33-row block per person per frame.

    A frame given as ``[]`` is written the way the runner writes a frame with
    nobody in it: one all-NaN block with ``person_idx = -1``, ``detected = false``.
    """
    rows: list[dict[str, object]] = []
    for frame_idx, people in enumerate(blocks):
        written = list(people) or [np.full((len(pm.JOINT_NAMES), 5), np.nan)]
        for person_idx, pose in enumerate(written):
            detected = bool(people)
            for joint_index, joint in enumerate(pm.JOINT_NAMES):
                x, y, z, visibility, presence = pose[joint_index]
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "t_ms": frame_idx * t_ms_step,
                        "joint": joint,
                        "x": x,
                        "y": y,
                        "z": z,
                        "visibility": visibility,
                        "presence": presence,
                        "rotated": rotated,
                        "detected": detected,
                        pm.PERSON_COLUMN: person_idx if people else pm.NO_PERSON_IDX,
                    }
                )
    return pd.DataFrame(rows, columns=list(pm.PARQUET_COLUMNS_MULTI))


def athlete_table(
    blocks: list[np.ndarray | None],
    *,
    rotated: bool = False,
    t_ms_step: int = 40,
) -> pd.DataFrame:
    """The athlete selection's schema: one body per frame, no ``person_idx``.

    A frame given as ``None`` is a frame nobody was found in: 33 all-NaN rows
    with ``detected = false``.
    """
    rows: list[dict[str, object]] = []
    for frame_idx, pose in enumerate(blocks):
        block = np.full((len(pm.JOINT_NAMES), 5), np.nan) if pose is None else pose
        for joint_index, joint in enumerate(pm.JOINT_NAMES):
            x, y, z, visibility, presence = block[joint_index]
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "t_ms": frame_idx * t_ms_step,
                    "joint": joint,
                    "x": x,
                    "y": y,
                    "z": z,
                    "visibility": visibility,
                    "presence": presence,
                    "rotated": rotated,
                    "detected": pose is not None,
                    "athlete_score": 0.9 if pose is not None else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=[*pm.PARQUET_COLUMNS, "athlete_score"])


# --------------------------------------------------------------------------- #
# Orientation flicker
# --------------------------------------------------------------------------- #


def test_orientation_segments_are_the_runs_of_one_orientation() -> None:
    assert om.orientation_segments([False] * 4) == [4]
    assert om.orientation_segments([True, False]) == [1, 1]
    assert om.orientation_segments([False, False, True, True, False]) == [2, 2, 1]
    assert om.orientation_segments([]) == []

    assert om.orientation_changes([False] * 4) == 0
    assert om.orientation_changes([False, True, False]) == 2
    assert om.orientation_changes([]) == 0


def test_short_segments_counts_the_runs_that_are_flicker() -> None:
    """One and two frame runs are flicker; a half-second run is a decision."""
    segments = [1, 2, 3, 8, 200]
    assert om.short_segments(segments) == 3
    assert om.short_segments(segments, max_length=1) == 1
    assert om.short_segments([]) == 0
    assert om.FLICKER_RUN_FRAMES == 3


def test_flicker_counts_the_switches_and_the_runs_of_a_set_of_clips() -> None:
    """The baseline flickers too, and that is what the numbers are against."""
    # 10 upright frames, 3 rotated, 7 upright again: the run lengths of one clip.
    one_clip = om.Flicker.of([10, 3, 7])
    assert (one_clip.switches, one_clip.run_count) == (2, 3)
    assert (one_clip.median_run, one_clip.p90_run, one_clip.max_run) == (7, 9, 10)
    assert one_clip.short_runs == 1
    assert "2 switches in 3 runs" in one_clip.describe()

    both = one_clip.add([1, 1, 4])
    assert (both.switches, both.run_count) == (4, 6)
    assert both.short_runs == 3
    assert om.Flicker().describe().startswith("0 switches in 0 runs")


def test_changed_runs_are_the_runs_where_two_modes_disagree() -> None:
    baseline = [False, False, False, True, True, True]
    #  Frame:  0  1  2  3  4  5
    #  base:   .  .  .  T  T  T
    #  cand:   .  .  T  T  .  T
    # They differ on frames 2 and 4, and only on those: the stretches where they
    # agree are not runs of changes.
    assert om.changed_runs(baseline, [False, False, True, True, False, True]) == [1, 1]
    # A decision changes a whole stretch: frames 1-2 and frames 4-5 here.
    assert om.changed_runs(baseline, [False, True, True, True, False, False]) == [2, 2]
    assert om.changed_runs(baseline, baseline) == []
    assert om.changed_runs([], []) == []
    with pytest.raises(ValueError, match="one reading per frame"):
        om.changed_runs(baseline, baseline[:2])


# --------------------------------------------------------------------------- #
# What a frame reads as
# --------------------------------------------------------------------------- #


def test_frame_reads_measures_the_athlete_pick() -> None:
    table = athlete_table(
        [handstand_body(), None, upright_body(), handstand_body(), handstand_body()],
        rotated=True,
    )
    reads = om.frame_reads(table)
    assert list(reads.columns) == [
        "frame_idx",
        "t_ms",
        "detected",
        "rotated",
        "wrist_y",
        "ankle_y",
        "inverted",
    ]
    assert reads["frame_idx"].tolist() == [0, 1, 2, 3, 4]
    assert reads["t_ms"].tolist() == [0, 40, 80, 120, 160]
    assert reads["detected"].tolist() == [True, False, True, True, True]
    assert reads["rotated"].all()
    assert reads["wrist_y"].tolist()[:1] == [HANDSTAND_WRIST_Y]
    assert reads["ankle_y"].tolist()[:1] == [HANDSTAND_ANKLE_Y]
    # The handstands read as handstands, the standing person does not, and the
    # frame nobody was found in is not "upright" so much as unreadable.
    assert reads["inverted"].tolist() == [True, False, False, True, True]
    assert np.isnan(reads["wrist_y"][1]) and np.isnan(reads["ankle_y"][1])


def test_frame_reads_averages_whichever_of_the_two_joints_is_there() -> None:
    """One wrist is enough to place the pair, exactly as a side-on clip has."""
    pose = handstand_body()
    pose[pm.JOINT_INDEX["right_wrist"]] = np.nan
    reads = om.frame_reads(athlete_table([pose]))
    assert reads["wrist_y"].iloc[0] == pytest.approx(HANDSTAND_WRIST_Y)
    assert reads["inverted"].iloc[0]

    pose[pm.JOINT_INDEX["left_wrist"]] = np.nan
    pose[pm.JOINT_INDEX["left_ankle"]] = np.nan
    reads = om.frame_reads(athlete_table([pose]))
    assert np.isnan(reads["wrist_y"].iloc[0])
    assert not reads["inverted"].iloc[0]


def test_frame_reads_measures_the_person_on_their_hands_in_a_multi_frame() -> None:
    """The rotation rules judge the person whose wrists are lowest in the image."""
    frames: list[list[np.ndarray]] = [[] for _ in range(FRAME_COUNT)]
    frames[0] = [upright_body(x=100.0), handstand_body(x=300.0)]
    frames[1] = [upright_body(x=100.0), handstand_body(x=300.0)]
    frames[2] = [handstand_body(x=100.0), handstand_body(x=300.0)]
    frames[3] = [upright_body(x=100.0), handstand_body(x=300.0)]
    frames[4] = []  # nobody in this frame at all
    reads = om.frame_reads(long_table(frames), pick="multi")
    # Frame 0 is measured on the handstand, not on the trainer next to it.
    assert reads["wrist_y"].iloc[0] == pytest.approx(HANDSTAND_WRIST_Y)
    assert reads["inverted"].tolist() == [True, True, True, True, False, False]
    assert reads["detected"].tolist() == [True, True, True, True, False, False]


def test_frame_reads_skips_a_multi_person_frame_whose_wrists_are_missing() -> None:
    """A person the model reported no wrists for cannot be the one on their hands."""
    trainer = upright_body(x=100.0)
    trainer[[pm.JOINT_INDEX[name] for name in pm.WRIST_JOINTS], 1] = np.nan
    frames: list[list[np.ndarray]] = [[trainer, handstand_body(x=300.0)]]
    frames += [[upright_body(x=100.0)]] * 2
    reads = om.frame_reads(long_table(frames), pick="multi")
    assert reads["wrist_y"].iloc[0] == pytest.approx(HANDSTAND_WRIST_Y)


def test_frame_reads_rejects_what_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="pick must be one of"):
        om.frame_reads(athlete_table([handstand_body()]), pick="everything")
    with pytest.raises(ValueError, match="frame_idx"):
        om.frame_reads(pd.DataFrame({"joint": ["nose"]}))


def test_inverted_share_counts_the_frames_the_mode_detected() -> None:
    reads = om.frame_reads(
        athlete_table([handstand_body(), upright_body(), None, handstand_body()])
    )
    assert om.inverted_share(reads) == pytest.approx(100.0 * 2 / 3)
    empty = om.frame_reads(athlete_table([None, None]))
    assert om.inverted_share(empty) == 0.0


# --------------------------------------------------------------------------- #
# One clip against another mode
# --------------------------------------------------------------------------- #


def test_compare_clip_counts_fixed_and_broken_on_the_shared_frames() -> None:
    #  auto reads frames 0 and 1 upside down, then reads the rest correctly;
    #  best reads the first frame as a handstand and gives up on frame 2.
    auto = om.frame_reads(
        athlete_table(
            [upright_body(), upright_body(), handstand_body(), handstand_body(), handstand_body()]
        )
    )
    best = om.frame_reads(
        athlete_table(
            [handstand_body(), upright_body(), upright_body(), handstand_body(), handstand_body()]
        )
    )
    compared = om.compare_clip("clip00000001", auto, best)
    assert compared.clip_id == "clip00000001"
    assert compared.compared_frames == 5
    assert compared.fixed == 1  # frame 0
    assert compared.broken == 1  # frame 2
    assert compared.baseline_inverted_percent == pytest.approx(60.0)
    assert compared.candidate_inverted_percent == pytest.approx(60.0)
    # They differ on frame 0 and on frame 2, in two single-frame runs.
    assert compared.changed_runs == (1, 1)
    assert compared.fixed_runs == (1,)
    assert compared.broken_runs == (1,)
    assert compared.moved_points == 0.0 and compared.unchanged
    assert compared.baseline_detected == compared.candidate_detected == 5
    assert compared.baseline_inverted == compared.candidate_inverted == 3


def test_compare_clip_only_judges_the_frames_both_modes_found_a_body_in() -> None:
    """A frame one mode missed is not charged to the mode that found the body."""
    auto = om.frame_reads(athlete_table([upright_body(), handstand_body()]))
    best = om.frame_reads(athlete_table([handstand_body(), handstand_body()]))
    compared = om.compare_clip("clip00000001", auto, best)
    assert compared.compared_frames == 2
    assert compared.fixed == 1 and compared.broken == 0
    assert compared.baseline_detected == 2 and compared.candidate_detected == 2

    missing = om.frame_reads(athlete_table([handstand_body(), None]))
    compared = om.compare_clip("clip00000001", auto, missing)
    assert compared.compared_frames == 1
    assert compared.fixed == 1  # the one shared frame, read the other way round
    assert compared.broken == 0
    assert compared.candidate_detected == 1
    assert compared.candidate_inverted_percent == pytest.approx(100.0)


def test_compare_clip_needs_both_modes_on_the_same_frames() -> None:
    auto = om.frame_reads(athlete_table([handstand_body()]))
    best = om.frame_reads(athlete_table([handstand_body(), handstand_body()]))
    with pytest.raises(ValueError, match="same frames"):
        om.compare_clip("clip00000001", auto, best)


def test_snapshot_rows_adds_the_clips_up() -> None:
    def comparison(clip_id: str, base: bool, cand: bool, runs: tuple[int, ...]):
        return om.ClipComparison(
            clip_id=clip_id,
            baseline_inverted_percent=100.0 * base,
            candidate_inverted_percent=100.0 * cand,
            compared_frames=3,
            fixed=2 if cand and not base else 0,
            broken=1 if base and not cand else 0,
            baseline_detected=3,
            candidate_detected=3,
            baseline_inverted=3 if base else 0,
            candidate_inverted=3 if cand else 0,
            changed_runs=runs,
            fixed_runs=(2,) if cand and not base else (),
            broken_runs=(1,) if base and not cand else (),
        )

    totals = om.snapshot_rows(
        [
            comparison("a", False, True, (1, 40)),
            comparison("b", True, False, (9,)),
            comparison("c", True, True, ()),
        ]
    )
    assert totals["clips"] == 3
    assert totals["fixed"] == 2 and totals["broken"] == 1
    assert (totals["better"], totals["worse"], totals["unchanged"]) == (1, 1, 1)
    # 6 of 9 frames read as a handstand by each mode, frame weighted.
    assert totals["baseline_inverted_percent"] == pytest.approx(200 / 3)
    assert totals["candidate_inverted_percent"] == pytest.approx(200 / 3)
    assert totals["changed_runs"] == 3
    assert totals["changed_runs_short"] == 1
    assert totals["changed_run_max"] == 40
    # A fix in a run of 2 and a loss in a run of 1, counted per direction.
    assert totals["fixed_runs"] == 1
    assert (totals["fixed_run_median"], totals["fixed_run_p90"], totals["fixed_run_max"]) == (
        2,
        2,
        2,
    )
    assert totals["fixed_runs_short"] == 1
    assert totals["broken_runs"] == 1
    assert totals["broken_run_max"] == 1
    assert totals["broken_runs_short"] == 1


# --------------------------------------------------------------------------- #
# The chain's own outputs
# --------------------------------------------------------------------------- #


def write_chain(data: pathlib.Path, *, with_report: bool = True) -> None:
    """A miniature workspace: two processed clips, two hold segments, a report."""
    processed = data / "processed" / "mediapipe"
    processed.mkdir(parents=True)
    (processed / "clip00000001.json").write_text(
        json.dumps({"frame_count": 300, "body_length_px": 210.0})
    )
    (processed / "clip00000002.json").write_text(
        json.dumps({"frame_count": 40, "body_length_px": None, "unusable_reason": "no body"})
    )
    phases = data / "phases" / "mediapipe"
    phases.mkdir(parents=True)
    (phases / "segments.csv").write_text(
        "clip_id,phase,hold_id,start_frame,end_frame,start_ms,end_ms,duration_s\n"
        "clip00000001,hold,0,10,60,400,2400,2.0\n"
        "clip00000001,pre,-1,61,299,2400,11960,0.0\n"
        "clip00000002,hold,0,0,30,0,1200,1.2\n"
    )
    if with_report:
        reports = data / "reports"
        reports.mkdir(parents=True)
        (reports / "trainer_report.csv").write_text(
            "clip_id,contact_frames,trainer_present\n"
            "clip00000001,42,true\n"
            "clip00000002,0,false\n"
        )


def test_chain_snapshot_reads_the_outputs_of_the_stages_on_disk(tmp_path: pathlib.Path) -> None:
    write_chain(tmp_path)
    snapshot = om.chain_snapshot(tmp_path)
    assert snapshot.usable_clips == 1
    assert snapshot.usable_frames == 300
    assert snapshot.total_frames == 340
    assert snapshot.hold_clips == 2
    assert snapshot.hold_seconds == pytest.approx(3.2)
    assert snapshot.trainer_clips == 1
    assert snapshot.contact_frames == 42


def test_chain_snapshot_says_so_when_there_is_nothing_to_read(tmp_path: pathlib.Path) -> None:
    snapshot = om.chain_snapshot(tmp_path)
    assert snapshot.usable_clips == 0
    assert snapshot.hold_clips == 0
    assert snapshot.hold_seconds == 0.0
    assert snapshot.trainer_clips is None
    assert snapshot.contact_frames is None


# --------------------------------------------------------------------------- #
# Report and CLI
# --------------------------------------------------------------------------- #


def test_render_report_quotes_the_numbers_it_was_given(tmp_path: pathlib.Path) -> None:
    write_chain(tmp_path)
    reads = om.frame_reads(
        athlete_table([handstand_body(), upright_body(), handstand_body()], rotated=True)
    )
    comparison = om.compare_clip("clip00000001", reads, reads)
    report = om.render_report(
        "auto",
        "best",
        "athlete",
        [comparison],
        om.Flicker.of([5, 1, 1, 1]),
        om.Flicker.of([9, 2]),
        om.chain_snapshot(tmp_path),
    )
    assert "best against auto (athlete keypoints, 1 clips)" in report
    assert "- `auto`: 3 switches in 4 runs" in report
    assert "3 of them 3 frames or less" in report
    assert "- `best`: 1 switches in 2 runs (median 6 frames, p90 8, max 9)" in report
    assert "frames fixed / broken against `auto`: **0 / 0** of 3 frames" in report
    assert "usable clips (a body length was measurable): **1**" in report
    assert "hold time: **3.2 s**" in report
    assert "clips with a trainer in them: **1** (42 contact frames)" in report
    # A clip the two modes read the same way is not a row in the moved table.
    assert "| clip |" not in report


def test_render_report_lists_the_clips_that_moved() -> None:
    auto = om.frame_reads(athlete_table([upright_body(), upright_body(), handstand_body()]))
    best = om.frame_reads(athlete_table([handstand_body(), upright_body(), handstand_body()]))
    report = om.render_report(
        "auto",
        "best",
        "athlete",
        [om.compare_clip("clip00000001", auto, best)],
        om.Flicker.of([4, 3]),
        om.Flicker.of([4, 3]),
        None,
    )
    assert "| `clip00000001` | 33.3 % | 66.7 % | 1 | 0 |" in report
    assert "What the chain made of it" not in report


def test_the_cli_measures_the_keypoints_it_is_pointed_at(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    write_chain(data)
    keypoints = data / "keypoints"
    for mode, blocks in (
        ("auto", [upright_body(), upright_body(), handstand_body()]),
        ("best", [handstand_body(), upright_body(), handstand_body()]),
    ):
        directory = keypoints / "mediapipe_athlete" / mode
        directory.mkdir(parents=True)
        athlete_table(blocks, rotated=mode == "best").to_parquet(
            directory / "clip00000001.parquet", index=False
        )
    assert om.main(["--data", str(data), "--clips", "clip00000001"]) == 0
    printed = capsys.readouterr().out
    assert "frames fixed / broken against `auto`: **1 / 0** of 3 frames" in printed
    assert "usable frames: **300** of 340" in printed


def test_the_cli_says_so_when_there_is_nothing_to_compare(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert om.main(["--data", str(tmp_path)]) == 1
    assert "missing keypoints" in capsys.readouterr().out


def test_the_cli_reports_a_clip_the_baseline_does_not_have(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keypoints = tmp_path / "keypoints" / "mediapipe_athlete"
    (keypoints / "auto").mkdir(parents=True)
    (keypoints / "best").mkdir(parents=True)
    athlete_table([handstand_body()]).to_parquet(keypoints / "best" / "clip1.parquet", index=False)
    assert om.main(["--data", str(tmp_path), "--no-chain"]) == 0
    assert "no `auto` keypoints for 1 clip(s): clip1" in capsys.readouterr().out


def test_the_cli_arguments_are_the_documented_ones() -> None:
    args = om.build_arg_parser().parse_args([])
    assert (args.baseline, args.candidate, args.pick) == ("auto", "best", "athlete")
    assert args.data is None and args.source == "mediapipe"
    assert args.clips is None and args.limit is None
    assert args.no_chain is False
    picked = om.build_arg_parser().parse_args(["--pick", "multi", "--limit", "2"])
    assert picked.pick == "multi" and picked.limit == 2
