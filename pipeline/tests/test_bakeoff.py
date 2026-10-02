"""Tests for :mod:`handstand.bakeoff`, the pose-model bake-off.

Nothing here touches the real labels, the real keypoints or the real videos:
every ground truth, person pool and confidence is written by hand in
display-frame pixels, so each assertion is arithmetic a reader can check with a
pencil — an error of 4 px against a torso of 20 px is 0.2 L by construction,
which is exactly where PCK@0.2 is allowed to sit.

The seven things the task asks for are all here: normalised error, PCK and
miss rate on hand-made frames; the left/right swap detector; the
human-reviewed filter (auto-accepted frames never score accuracy); the
missing-clip exclusion; oracle matching picking the closest person; the
clip-level bootstrap being deterministic with a seed; and the gate
recommendation on a synthetic error/confidence set.
"""

from __future__ import annotations

import math
import pathlib

import pandas as pd
import pytest

from handstand import bakeoff
from handstand.labels import KeypointRow

#: A side-view torso in display pixels: shoulders 20 px above the hips, so the
#: ground-truth torso length is ``L = 20`` and 0.2 L = 4 px, 0.5 L = 10 px.
SHOULDERS = ("left_shoulder", "right_shoulder")
HIPS = ("left_hip", "right_hip")
TORSO_L = 20.0


def frame_gt() -> dict[str, tuple[float, float]]:
    """Five visible joints of one frame, torso length exactly 20 px."""
    return {
        "left_shoulder": (0.0, 0.0),
        "right_shoulder": (10.0, 0.0),
        "left_hip": (0.0, 20.0),
        "right_hip": (10.0, 20.0),
        "left_wrist": (2.0, 5.0),
        "right_wrist": (18.0, 5.0),
        "nose": (5.0, -5.0),
    }


def frame_person() -> bakeoff.Person:
    """The model's answer: two exact joints, one 5 px off, two problems.

    ``right_hip`` is simply absent (the model did not output it),
    ``right_wrist`` carries confidence 0.2 (below any sane gate) and ``nose``
    carries a NaN confidence (the model refused to score it).
    """
    return {
        "left_shoulder": (0.0, 0.0, 0.9),
        "right_shoulder": (15.0, 0.0, 0.9),  # 5 px off = 0.25 L
        "left_hip": (0.0, 20.0, 0.9),
        "left_wrist": (6.0, 5.0, 0.9),  # 4 px off = 0.20 L exactly
        "right_wrist": (18.0, 5.0, 0.2),  # exact, but below gate 0.5
        "nose": (5.0, -5.0, float("nan")),  # exact, but unscored
    }


def records(person: bakeoff.Person | None) -> pd.DataFrame:
    """The record table one frame produces, with the frame's L attached."""
    rows = bakeoff.person_errors(frame_gt(), person, bakeoff.SCHEMA_JOINTS, TORSO_L)
    frame = pd.DataFrame(rows)
    frame["clip_id"] = "aaaaaaaaaaaa"
    frame["frame_idx"] = 0
    frame["L"] = TORSO_L
    return frame


# --------------------------------------------------------------------------- #
# Normalised error, PCK and miss rate
# --------------------------------------------------------------------------- #


def test_torso_length_uses_the_side_that_exists() -> None:
    """One labelled side is the midpoint in a side view; no hips is no L."""
    gt = frame_gt()
    assert bakeoff.torso_length(gt) == pytest.approx(TORSO_L)
    # A side view often labels one side only: the midpoint of one point is it.
    assert bakeoff.torso_length({k: gt[k] for k in ("left_shoulder", "left_hip")}) == (
        pytest.approx(TORSO_L)
    )
    # Shoulders without hips (or vice versa) cannot be normalised.
    assert bakeoff.torso_length({k: gt[k] for k in SHOULDERS}) is None
    assert bakeoff.torso_length({k: gt[k] for k in HIPS}) is None
    assert bakeoff.torso_length({}) is None


def test_normalised_error_scales_by_torso_length() -> None:
    """A 5 px miss on a 20 px torso is 0.25 L; no L is no error."""
    assert bakeoff.normalised_error((0.0, 0.0), (5.0, 0.0), TORSO_L) == pytest.approx(0.25)
    assert bakeoff.normalised_error((0.0, 0.0), (3.0, 4.0), TORSO_L) == pytest.approx(0.25)
    assert math.isnan(bakeoff.normalised_error((0.0, 0.0), (5.0, 0.0), None))


def test_pck_and_miss_rate_on_a_handmade_frame() -> None:
    """PCK counts misses against the score; the gate decides what counts.

    Seven visible joints; at gate 0.5 the below-gate wrist and the unscored
    nose join the absent hip as misses, and PCK@0.2 must not count them.
    """
    frame = records(frame_person())
    assert len(frame) == 7
    # 0.2 L boundary is inclusive: the left wrist sits exactly on it.
    assert bakeoff.pck(frame, gate=0.5, level=0.2) == pytest.approx(3 / 7)
    assert bakeoff.pck(frame, gate=0.5, level=0.5) == pytest.approx(4 / 7)
    assert bakeoff.miss_rate(frame, gate=0.5) == pytest.approx(3 / 7)
    summary = bakeoff.error_summary(frame, gate=0.5)
    assert summary["n_delivered"] == 4
    assert summary["mean_err"] == pytest.approx((0 + 0.25 + 0 + 0.20) / 4)
    assert summary["median_err"] == pytest.approx(0.10)

    # Lowering the gate to 0 recovers the low-confidence wrist: the unscored
    # (NaN) nose and the absent hip are misses at *any* gate.
    assert bakeoff.pck(frame, gate=0.0, level=0.2) == pytest.approx(4 / 7)
    assert bakeoff.miss_rate(frame, gate=0.0) == pytest.approx(2 / 7)


def test_a_frame_without_l_leaves_normalised_metrics_empty() -> None:
    """No torso GT, no normalised metric — and the caller counts the frame out."""
    frame = records(frame_person())
    frame["L"] = float("nan")
    assert math.isnan(bakeoff.pck(frame, gate=0.5, level=0.2))
    # The miss rate needs no normalisation, so the frame still counts there.
    assert bakeoff.miss_rate(frame, gate=0.5) == pytest.approx(3 / 7)


# --------------------------------------------------------------------------- #
# The swap detector
# --------------------------------------------------------------------------- #


def test_swap_detector_flags_crossed_wrists_only() -> None:
    """A real swap clears both bars; correct labels and close sides do not."""
    gt = {
        "left_wrist": (0.0, 100.0),
        "right_wrist": (40.0, 100.0),
        "left_ankle": (0.0, 0.0),
        "right_ankle": (40.0, 0.0),
    }
    length = 100.0
    correct: bakeoff.Person = {
        "left_wrist": (0.0, 100.0, 0.9),
        "right_wrist": (40.0, 100.0, 0.9),
        "left_ankle": (2.0, 0.0, 0.9),
        "right_ankle": (40.0, 0.0, 0.9),
    }
    swapped = dict(correct)
    swapped["left_wrist"] = (40.0, 100.0, 0.9)
    swapped["right_wrist"] = (0.0, 100.0, 0.9)

    clean = bakeoff.swap_report(gt, correct, length, gate=0.5)
    assert clean == {"wrists": False, "ankles": False}
    crossed = bakeoff.swap_report(gt, swapped, length, gate=0.5)
    assert crossed["wrists"] is True
    assert crossed["ankles"] is False  # only the wrists were crossed

    # A below-gate side is not evaluable: unknown, not fine.
    shy = dict(correct)
    shy["left_wrist"] = (0.0, 100.0, 0.3)
    assert bakeoff.swap_report(gt, shy, length, gate=0.5)["wrists"] is None


def test_swap_detector_refuses_sides_that_overlap() -> None:
    """In a side view the sides sit on top of each other: no verdict, no rate."""
    gt = {"left_wrist": (0.0, 100.0), "right_wrist": (5.0, 100.0)}
    person: bakeoff.Person = {
        "left_wrist": (5.0, 100.0, 0.9),
        "right_wrist": (0.0, 100.0, 0.9),
    }
    # 5 px separation < 0.15 L = 15 px, so the crossed pose proves nothing.
    assert bakeoff.swap_report(gt, person, length=100.0, gate=0.5)["wrists"] is None
    # No L at all is likewise not evaluable.
    assert bakeoff.swap_report(gt, person, length=None, gate=0.5)["wrists"] is None
    assert bakeoff.swap_report(gt, None, length=100.0, gate=0.5)["wrists"] is None


# --------------------------------------------------------------------------- #
# The filters: human-reviewed only, missing clips excluded
# --------------------------------------------------------------------------- #


def row(clip: str, frame: int, joint: str) -> KeypointRow:
    """One labelled joint, coordinates never read by the filters under test."""
    return KeypointRow(
        clip_id=clip,
        frame_idx=frame,
        joint=joint,
        x=1.0,
        y=2.0,
        visible=True,
        labeler="test",
        labeled_at="2026-10-02",
    )


def test_auto_accepted_frames_never_reach_the_accuracy_scope() -> None:
    """An auto-accepted image is agreement with RTMPose, never accuracy."""
    rows = [
        row("aaaaaaaaaaaa", 1, "left_wrist"),  # human-reviewed
        row("aaaaaaaaaaaa", 2, "left_wrist"),  # auto-accepted pre-label
        row("bbbbbbbbbbbb", 3, "left_wrist"),  # human-reviewed, other clip
    ]
    auto = {"aaaaaaaaaaaa_2.jpg"}
    frames = bakeoff.frames_table(rows, auto_images=auto, missing_clips=set())
    assert len(frames) == 3
    by_frame = frames.set_index(["clip_id", "frame_idx"])
    assert bool(by_frame.loc[("aaaaaaaaaaaa", 1), "human"]) is True
    assert bool(by_frame.loc[("aaaaaaaaaaaa", 2), "human"]) is False
    # keep is what every accuracy table draws from: human-reviewed only.
    assert set(frames.loc[frames["keep"], "frame_idx"]) == {1, 3}
    assert set(frames.loc[~frames["human"], "frame_idx"]) == {2}


def test_missing_catalogue_clips_are_excluded_everywhere() -> None:
    """Chainlink #49: the deleted duplicate must not be counted twice."""
    rows = [
        row("244275cd7059", 112, "left_wrist"),  # the duplicate clip
        row("aaaaaaaaaaaa", 1, "left_wrist"),
    ]
    frames = bakeoff.frames_table(
        rows, auto_images=set(), missing_clips={"244275cd7059"}
    )
    duplicate = frames[frames["clip_id"] == "244275cd7059"].iloc[0]
    assert bool(duplicate["missing_clip"]) is True
    assert bool(duplicate["keep"]) is False
    assert set(frames.loc[frames["keep"], "clip_id"]) == {"aaaaaaaaaaaa"}


def test_filters_read_their_csvs(tmp_path: pathlib.Path) -> None:
    """The auto-accepted list and the catalogue's missing flag are read as-is."""
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "auto_accepted.csv").write_text(
        "task_id,image,stratum\n1,cccccccccccc_4.jpg,clean_no_trainer\n",
        encoding="utf-8",
    )
    (tmp_path / "catalogue.csv").write_text(
        "clip_id,filename,missing\n"
        "dddddddddddd,deleted.mp4,true\n"
        "aaaaaaaaaaaa,present.mp4,false\n",
        encoding="utf-8",
    )
    assert bakeoff.load_auto_images(tmp_path) == {"cccccccccccc_4.jpg"}
    assert bakeoff.read_missing_clips(tmp_path) == {"dddddddddddd"}
    # A dataset without the files is simply an empty answer, not a crash.
    assert bakeoff.load_auto_images(tmp_path / "nowhere") == set()
    assert bakeoff.read_missing_clips(tmp_path / "nowhere") == set()


def test_flag_multi_person_clips_appends_and_stays_idempotent(
    tmp_path: pathlib.Path,
) -> None:
    """Chainlink #15: the catalogue itself says which clips held two people."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "trainer_report.csv").write_text(
        "clip_id,second_person_frames\n"
        "aaaaaaaaaaaa,12\n"
        "bbbbbbbbbbbb,0\n",
        encoding="utf-8",
    )
    catalogue_path = tmp_path / "catalogue.csv"
    catalogue_path.write_text(
        "clip_id,filename,notes,missing\n"
        "aaaaaaaaaaaa,a.mp4,first note,false\n"
        "bbbbbbbbbbbb,b.mp4,,false\n",
        encoding="utf-8",
    )
    changed = bakeoff.flag_multi_person_clips(tmp_path)
    assert changed == ["aaaaaaaaaaaa"]
    table = pd.read_csv(catalogue_path, dtype=str, keep_default_na=False)
    flagged = table.set_index("clip_id").loc["aaaaaaaaaaaa", "notes"]
    assert flagged == f"first note; {bakeoff.MULTI_PERSON_NOTE}"
    # The other clip is untouched...
    assert table.set_index("clip_id").loc["bbbbbbbbbbbb", "notes"] == ""
    # ...and flagging again changes nothing: the note is idempotent.
    assert bakeoff.flag_multi_person_clips(tmp_path) == []
    assert pd.read_csv(catalogue_path, dtype=str).set_index("clip_id").loc[
        "aaaaaaaaaaaa", "notes"
    ] == flagged
    # No trainer report means nothing to flag, not a crash.
    assert bakeoff.flag_multi_person_clips(tmp_path / "nowhere") == []


def test_annotate_frames_draws_the_selection_and_note_scopes() -> None:
    """Comment 26: choose on clean strata minus the MediaPipe contact mask."""
    rows = [
        row("aaaaaaaaaaaa", 1, "left_wrist"),  # clean, unmasked
        row("aaaaaaaaaaaa", 2, "left_wrist"),  # trainer-contact stratum
        row("aaaaaaaaaaaa", 3, "left_wrist"),  # clean, but the mask fires
        row("aaaaaaaaaaaa", 4, "left_wrist"),  # auto-accepted
    ]
    frames = bakeoff.frames_table(rows, auto_images={"aaaaaaaaaaaa_4.jpg"}, missing_clips=set())
    annotated = bakeoff.annotate_frames(
        frames,
        strata={
            ("aaaaaaaaaaaa", 1): "clean_no_trainer",
            ("aaaaaaaaaaaa", 2): "trainer_contact",
            ("aaaaaaaaaaaa", 3): "clean_trainer",
            ("aaaaaaaaaaaa", 4): "clean_no_trainer",
        },
        phases={"aaaaaaaaaaaa": {1: ("hold", 0), 3: ("pre", -1)}},
        shapes={("aaaaaaaaaaaa", 0): "line"},
        contact={"aaaaaaaaaaaa": {3: True}},
        multi_person_clips={"aaaaaaaaaaaa"},
    )
    by_frame = annotated.set_index(["clip_id", "frame_idx"])
    assert bool(by_frame.loc[("aaaaaaaaaaaa", 1), "in_selection"]) is True
    assert str(by_frame.loc[("aaaaaaaaaaaa", 1), "shape"]) == "line"
    # The trainer stratum is a note, never a selection criterion.
    assert bool(by_frame.loc[("aaaaaaaaaaaa", 2), "in_selection"]) is False
    assert bool(by_frame.loc[("aaaaaaaaaaaa", 2), "in_note"]) is True
    # The MediaPipe mask removes a clean frame from selection too — same mask
    # for every model — and moves it to the note.
    assert bool(by_frame.loc[("aaaaaaaaaaaa", 3), "in_selection"]) is False
    assert bool(by_frame.loc[("aaaaaaaaaaaa", 3), "in_note"]) is True
    assert bool(by_frame.loc[("aaaaaaaaaaaa", 3), "in_stratum"]) is False
    assert str(by_frame.loc[("aaaaaaaaaaaa", 3), "shape"]) == "none"  # hold_id -1
    # Only the auto frame is scored as agreement with RTMPose.
    assert set(annotated.loc[annotated["in_agreement"], "frame_idx"]) == {4}
    assert bool(by_frame.loc[("aaaaaaaaaaaa", 1), "multi_person"]) is True


# --------------------------------------------------------------------------- #
# Oracle matching
# --------------------------------------------------------------------------- #


def oracle_gt() -> dict[str, tuple[float, float]]:
    """Six visible joints — more than :data:`bakeoff.MIN_SHARED_JOINTS`."""
    return {
        "left_shoulder": (0.0, 0.0),
        "right_shoulder": (10.0, 0.0),
        "left_hip": (0.0, 20.0),
        "left_wrist": (2.0, 5.0),
        "right_wrist": (18.0, 5.0),
        "nose": (5.0, -5.0),
    }


def shifted(dx: float, dy: float, joints: tuple[str, ...]) -> bakeoff.Person:
    """A person whose listed joints sit ``dx, dy`` px off the ground truth."""
    gt = oracle_gt()
    return {name: (gt[name][0] + dx, gt[name][1] + dy, 0.9) for name in joints}


def test_oracle_picks_the_closest_person() -> None:
    """The oracle isolates keypoint accuracy from person choice."""
    gt = oracle_gt()
    far = shifted(60.0, 0.0, tuple(gt))
    close = shifted(2.0, 0.0, tuple(gt))
    pool: list[bakeoff.Person] = [far, close]
    assert bakeoff.oracle_person(pool, gt, tuple(gt)) is close
    # Ties keep the model's own order, so the pick is deterministic.
    twin = shifted(2.0, 0.0, tuple(gt))
    assert bakeoff.oracle_person([far, close, twin], gt, tuple(gt)) is close


def test_oracle_refuses_a_person_that_shares_too_little() -> None:
    """One lucky joint must not win the oracle; the bar is MIN_SHARED_JOINTS."""
    gt = oracle_gt()
    lucky = {"nose": (5.0, -5.0, 0.9)}  # perfect, but only one joint
    honest = shifted(30.0, 0.0, tuple(gt))
    assert bakeoff.oracle_person([lucky, honest], gt, tuple(gt)) is honest
    # Nobody qualifies and an empty pool both mean "no oracle answer".
    assert bakeoff.oracle_person([lucky], gt, tuple(gt)) is None
    assert bakeoff.oracle_person([], gt, tuple(gt)) is None
    # A frame with no visible GT joints cannot be judged at all.
    assert bakeoff.oracle_person([honest], {}, tuple(gt)) is None
    # The bar never exceeds how many joints the frame has.
    tiny = {"left_shoulder": (1.0, 1.0, 0.9), "left_hip": (1.0, 21.0, 0.9)}
    only_two = {name: gt[name] for name in ("left_shoulder", "left_hip")}
    assert bakeoff.oracle_person([tiny], only_two, tuple(gt)) is tiny


# --------------------------------------------------------------------------- #
# The clip-level bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_frame(hits: list[int], totals: list[int], clips: list[str]) -> pd.DataFrame:
    """A tiny record table for the bootstrap: one row per (clip, observation)."""
    return pd.DataFrame(
        {
            "clip_id": clips,
            "_hit": [float(value) for value in hits],
            "_total": [float(value) for value in totals],
        }
    )


def test_bootstrap_over_clips_is_deterministic_with_a_seed() -> None:
    """Same seed, same interval — the reviewer must be able to reproduce it."""
    frame = bootstrap_frame(
        hits=[1, 1, 0, 0, 1, 0],
        totals=[1, 1, 1, 1, 1, 1],
        clips=["a", "a", "b", "b", "c", "c"],
    )
    first = bakeoff.bootstrap_ratio_ci(frame, "_hit", "_total", samples=500, seed=7)
    second = bakeoff.bootstrap_ratio_ci(frame, "_hit", "_total", samples=500, seed=7)
    assert first == second
    low, high = first
    assert 0.0 <= low <= high <= 1.0
    # Clips move together: with one all-hit and one all-miss clip the resample
    # can land on either extreme, so the interval must span both.
    one_each = bootstrap_frame(
        hits=[1, 1, 0, 0], totals=[1, 1, 1, 1], clips=["a", "a", "b", "b"]
    )
    low, high = bakeoff.bootstrap_ratio_ci(one_each, "_hit", "_total", samples=500, seed=7)
    assert (low, high) == (0.0, 1.0)
    # Identical clips give a zero-width interval: nothing varies to resample.
    uniform = bootstrap_frame(hits=[1, 1], totals=[1, 1], clips=["a", "b"])
    assert bakeoff.bootstrap_ratio_ci(uniform, "_hit", "_total", samples=100, seed=7) == (
        1.0,
        1.0,
    )
    # No rows means no interval, not a fake 0.0.
    empty_low, empty_high = bakeoff.bootstrap_ratio_ci(
        uniform.iloc[:0], "_hit", "_total"
    )
    assert math.isnan(empty_low) and math.isnan(empty_high)


# --------------------------------------------------------------------------- #
# The visibility gate
# --------------------------------------------------------------------------- #


def synthetic_gate_rows() -> pd.DataFrame:
    """Ten confident-and-right joints and ten low-confidence-and-wrong ones."""
    good = pd.DataFrame(
        {
            "has_pred": [True] * 10,
            "err_l": [0.1] * 10,
            "conf": [0.9] * 10,
        }
    )
    bad = pd.DataFrame(
        {
            "has_pred": [True] * 10,
            "err_l": [0.5] * 10,
            "conf": [0.3] * 10,
        }
    )
    return pd.concat([good, bad], ignore_index=True)


def test_gate_sweep_tabulates_error_versus_confidence() -> None:
    """Each candidate gate reports how many joints it keeps and how precise."""
    sweep = bakeoff.gate_sweep(synthetic_gate_rows())
    assert len(sweep) == len(bakeoff.GATE_GRID)
    row0 = sweep.iloc[0]
    assert row0["gate"] == 0.0
    assert row0["n_kept"] == 20
    assert row0["within_02"] == pytest.approx(0.5)
    high = sweep[sweep["gate"] == 0.5].iloc[0]
    assert high["n_kept"] == 10  # the 0.3-confidence joints are gone
    assert high["within_02"] == pytest.approx(1.0)
    assert bool(high["meets_target"]) is True


def test_recommend_gate_picks_the_lowest_gate_meeting_the_target() -> None:
    """The rule is "lowest gate with >= 90 % precise", not "highest gate"."""
    sweep = bakeoff.gate_sweep(synthetic_gate_rows())
    gate, met = bakeoff.recommend_gate(sweep)
    # Gates up to 0.30 keep the bad joints (precision 0.5); 0.35 is the first
    # gate that keeps only the good ones.
    assert gate == pytest.approx(0.35)
    assert met is True


def test_recommend_gate_keeps_the_status_quo_when_the_target_is_unreachable() -> None:
    """A gate that buys no precision must not discard joints for nothing."""
    hopeless = pd.DataFrame(
        {
            "has_pred": [True] * 20,
            "err_l": [0.5] * 20,
            "conf": [value / 100 for value in range(20)],
        }
    )
    sweep = bakeoff.gate_sweep(hopeless)
    assert not sweep["meets_target"].any()
    gate, met = bakeoff.recommend_gate(sweep)
    assert gate == bakeoff.LEGACY_GATE
    assert met is False
    # An empty sweep (no evidence at all) also keeps the status quo.
    assert bakeoff.recommend_gate(pd.DataFrame(columns=sweep.columns)) == (
        bakeoff.LEGACY_GATE,
        False,
    )


# --------------------------------------------------------------------------- #
# RTMPose's pipeline person: inverted first, score second
# --------------------------------------------------------------------------- #


def standing(score: float) -> bakeoff.Person:
    """An upright body: ankles below the wrists (y grows downward)."""
    return {
        "left_wrist": (0.0, 50.0, score),
        "right_wrist": (10.0, 50.0, score),
        "left_ankle": (0.0, 200.0, score),
        "right_ankle": (10.0, 200.0, score),
    }


def handstand(score: float) -> bakeoff.Person:
    """An inverted body: wrists below the ankles, i.e. on the hands."""
    return {
        "left_wrist": (0.0, 200.0, score),
        "right_wrist": (10.0, 200.0, score),
        "left_ankle": (0.0, 50.0, score),
        "right_ankle": (10.0, 50.0, score),
    }


def test_rtmpose_pipeline_prefers_the_inverted_body_then_the_score() -> None:
    """The athlete is the inverted body; a standing trainer never wins."""
    trainer = standing(0.9)
    athlete = handstand(0.5)
    assert bakeoff.select_rtmpose_pipeline([trainer, athlete]) is athlete
    # With nobody inverted the best score wins, ties keep the model's order.
    first, second = standing(0.7), standing(0.7)
    assert bakeoff.select_rtmpose_pipeline([first, second]) is first
    # And no people at all is "no answer", not a crash.
    assert bakeoff.select_rtmpose_pipeline([]) is None


def test_is_inverted_needs_a_wrist_and_an_ankle() -> None:
    """An unknown body is never read as a claim the other body contradicts."""
    assert bakeoff.is_inverted({"left_wrist": (0.0, 10.0), "left_ankle": (0.0, 5.0)}) is True
    assert bakeoff.is_inverted({"left_wrist": (0.0, 5.0), "left_ankle": (0.0, 10.0)}) is False
    assert bakeoff.is_inverted({"left_wrist": (0.0, 5.0)}) is None
    assert bakeoff.is_inverted({}) is None
