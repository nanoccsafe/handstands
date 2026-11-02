"""Tests for :mod:`handstand.segments`.

Nothing here touches the real handstand data: the clips are synthetic
:class:`handstand.features.ClipFeatures` whose leg, knee and hip angles the test
picks, written to a features parquet under a tmp data directory when a test needs
the files.
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from handstand import features as ft
from handstand import hold_shapes, segments

CLIP_ID = "abc123def456"
OTHER_ID = "fedcba654321"
FRAME_MS = 100
UNMEASURED = hold_shapes.UNMEASURED

#: The three angles (leg separation, knee angle, hip angle) of each shape.
POSES = {
    "line": (0.0, 178.0, 176.0),
    "straddle": (60.0, 178.0, 176.0),
    "tuck": (0.0, 90.0, 176.0),
    "pike": (0.0, 178.0, 100.0),
    UNMEASURED: (math.nan, math.nan, math.nan),
}


def runs_input(spec: Sequence[tuple[str, int]]) -> tuple[list[int], list[str]]:
    """``t_ms`` and shapes for ``(shape, frames)`` blocks, 100 ms apart."""
    shapes = [shape for shape, count in spec for _ in range(count)]
    return [index * FRAME_MS for index in range(len(shapes))], shapes


def synthetic_clip(
    clip_id: str,
    frames: Sequence[tuple[str, int, bool]],
    hip_noise: bool = False,
) -> ft.ClipFeatures:
    """A clip from ``(shape, hold_id, valid)`` per frame, 100 ms apart (hold -1 is no hold)."""
    count = len(frames)
    values = {name: np.full(count, np.nan) for name in (*ft.FEATURE_NAMES, *ft.COM_COLUMNS)}
    for index, (shape, _hold, _valid) in enumerate(frames):
        leg, knee, hip = POSES[shape]
        values["leg_separation"][index] = leg
        values["knee_angle"][index] = knee
        # A varying hip so a median over the wrong frames is a different number.
        values["hip_angle"][index] = hip + (index % 4 if hip_noise and shape == "line" else 0)
    # The two non-float columns, as features.py carries them: a flag and words (NaN unknown).
    values["com_complete"] = np.array([index % 2 == 0 for index in range(count)], dtype=bool)
    zones = np.array(["under", "ok", "over"], dtype=object)
    values["balance_zone"] = np.array(
        [zones[index % 3] if index % 5 else math.nan for index in range(count)], dtype=object
    )
    return ft.ClipFeatures(
        clip_id=clip_id,
        source="mediapipe",
        frame_idx=np.arange(count, dtype=np.int64),
        t_ms=np.arange(count, dtype=np.int64) * FRAME_MS,
        phase=np.array([ft.HOLD_PHASE if hold >= 0 else "unknown" for _, hold, _ in frames]),
        hold_id=np.array([hold for _, hold, _ in frames], dtype=np.int64),
        measured=ft.BodyFeatures(
            values=values, valid=np.array([valid for _, _, valid in frames], dtype=bool)
        ),
    )


def two_hold_clip(clip_id: str = CLIP_ID) -> ft.ClipFeatures:
    """Hold 0: straddle (20 frames) then line (30); 5 frames of no hold; hold 1: all line."""
    frames: list[tuple[str, int, bool]] = []
    frames += [("straddle", 0, True)] * 20
    frames += [("line", 0, True)] * 30
    frames += [("line", -1, True)] * 5
    frames += [("line", 1, True)] * 12
    # Invalid frames look like a tuck but must not be measured or counted as one.
    frames += [("tuck", 1, False)] * 3
    frames += [("line", 1, True)] * 12
    return synthetic_clip(clip_id, frames, hip_noise=True)


def write_features(data: pathlib.Path, clip: ft.ClipFeatures) -> pathlib.Path:
    path = ft.output_dir(data) / f"{clip.clip_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    ft.feature_table(clip).to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# segment_runs
# --------------------------------------------------------------------------- #


def test_one_shape_is_one_segment() -> None:
    t_ms, shapes = runs_input([("line", 30)])

    assert segments.segment_runs(t_ms, shapes) == [("line", 0, 2900)]


def test_a_short_run_between_two_long_ones_is_absorbed() -> None:
    t_ms, shapes = runs_input([("line", 30), ("straddle", 5), ("line", 30)])

    assert segments.segment_runs(t_ms, shapes) == [("line", 0, 6400)]


def test_a_long_run_between_two_long_ones_makes_three_segments() -> None:
    t_ms, shapes = runs_input([("line", 30), ("straddle", 20), ("line", 30)])

    assert segments.segment_runs(t_ms, shapes) == [
        ("line", 0, 2900),
        ("straddle", 3000, 4900),
        ("line", 5000, 7900),
    ]


def test_unmeasured_frames_join_the_run_in_progress() -> None:
    t_ms, shapes = runs_input([("line", 10), (UNMEASURED, 5), ("straddle", 20)])

    assert segments.segment_runs(t_ms, shapes) == [("line", 0, 1400), ("straddle", 1500, 3400)]


def test_unmeasured_frames_between_equal_shapes_do_not_split_them() -> None:
    t_ms, shapes = runs_input([("line", 15), (UNMEASURED, 3), ("line", 15)])

    assert segments.segment_runs(t_ms, shapes) == [("line", 0, 3200)]


def test_leading_unmeasured_frames_join_the_first_run() -> None:
    t_ms, shapes = runs_input([(UNMEASURED, 3), ("line", 20), ("straddle", 20)])

    assert segments.segment_runs(t_ms, shapes) == [("line", 0, 2200), ("straddle", 2300, 4200)]


def test_a_hold_with_nothing_measured_is_one_unmeasured_segment() -> None:
    t_ms, shapes = runs_input([(UNMEASURED, 10)])

    assert segments.segment_runs(t_ms, shapes) == [(UNMEASURED, 0, 900)]


def test_no_frames_is_no_segments() -> None:
    assert segments.segment_runs([], []) == []


def test_a_tie_for_the_shortest_run_merges_the_first() -> None:
    # line 0-29, tuck 30-37, pike 38-45, straddle 46-75. The tuck (first of the
    # two 700 ms runs) joins the longer line; the pike then joins the line too.
    t_ms, shapes = runs_input([("line", 30), ("tuck", 8), ("pike", 8), ("straddle", 30)])

    assert segments.segment_runs(t_ms, shapes) == [("line", 0, 4500), ("straddle", 4600, 7500)]


def test_equal_neighbours_merge_into_the_previous_run() -> None:
    t_ms, shapes = runs_input([("line", 20), ("tuck", 5), ("straddle", 20)])

    assert segments.segment_runs(t_ms, shapes) == [("line", 0, 2400), ("straddle", 2500, 4400)]


def test_a_short_run_at_the_start_merges_into_the_next() -> None:
    t_ms, shapes = runs_input([("tuck", 5), ("line", 30)])

    assert segments.segment_runs(t_ms, shapes) == [("line", 0, 3400)]


def test_a_short_run_at_the_end_merges_into_the_previous() -> None:
    t_ms, shapes = runs_input([("line", 30), ("tuck", 5)])

    assert segments.segment_runs(t_ms, shapes) == [("line", 0, 3400)]


def test_the_minimum_is_a_parameter() -> None:
    t_ms, shapes = runs_input([("line", 30), ("straddle", 5), ("line", 30)])

    assert len(segments.segment_runs(t_ms, shapes, min_ms=300)) == 3


def test_segment_runs_refuses_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="entries"):
        segments.segment_runs([0, 100], ["line"])


# --------------------------------------------------------------------------- #
# frame_shapes
# --------------------------------------------------------------------------- #


def test_frame_shapes_agrees_with_predict_shape() -> None:
    names = ["tuck", "straddle", "pike", "line", UNMEASURED]
    leg, knee, hip = (np.array([POSES[name][axis] for name in names]) for axis in range(3))

    shapes = segments.frame_shapes(leg, knee, hip)

    assert shapes == names
    for name, (leg_value, knee_value, hip_value) in zip(names, map(POSES.get, names), strict=True):
        row = {
            "leg_separation_median": leg_value,
            "knee_angle_median": knee_value,
            "hip_angle_median": hip_value,
        }
        assert hold_shapes.predict_shape(row)[0] == name


# --------------------------------------------------------------------------- #
# clip_segments
# --------------------------------------------------------------------------- #


def test_clip_segments_splits_a_mixed_hold_and_keeps_a_single_one() -> None:
    rows = segments.clip_segments(two_hold_clip())

    assert [(row["hold_id"], row["segment_id"]) for row in rows] == [(0, 0), (0, 1), (1, 0)]
    assert [row["shape_guess"] for row in rows] == ["straddle", "line", "line"]
    assert [row["mixed"] for row in rows] == [True, True, False]
    straddle, line, single = rows
    assert (straddle["start_ms"], straddle["end_ms"], straddle["duration_s"]) == (0, 1900, 1.9)
    assert (line["start_ms"], line["end_ms"], line["duration_s"]) == (2000, 4900, 2.9)
    assert straddle["segment_frames"] == 20 and straddle["valid_frames"] == 20
    assert line["segment_frames"] == 30 and line["valid_frames"] == 30
    assert list(rows[0]) == list(segments.SEGMENT_COLUMNS)
    assert straddle["clip_id"] == CLIP_ID and straddle["source"] == "mediapipe"


def test_a_segment_is_summarised_over_only_its_own_frames() -> None:
    clip = two_hold_clip()
    line = segments.clip_segments(clip)[1]

    own = clip.value("hip_angle")[20:50]
    median, iqr = ft._spread(own)
    assert line["hip_angle_median"] == ft._rounded_scalar(median, 2)
    assert line["hip_angle_iqr"] == ft._rounded_scalar(iqr, 2)
    # The whole hold's hip median would include the straddle's different hip values.
    whole = ft._spread(clip.value("hip_angle")[:50])
    assert whole != (median, iqr)
    assert line["knee_angle_median"] == 178.0
    assert line["leg_separation_median"] == 0.0
    # A feature nothing measured stays NaN, as in the hold summary.
    assert math.isnan(line["off_hip_median"])


def test_invalid_frames_are_not_measured_or_shaped() -> None:
    rows = segments.clip_segments(two_hold_clip())
    single = rows[2]

    # 12 line + 3 invalid tuck-looking + 12 line frames: one line segment over the 27 frames.
    assert single["shape_guess"] == "line"
    assert single["segment_frames"] == 27
    assert single["valid_frames"] == 24
    assert single["knee_angle_median"] == 178.0


def test_a_hold_with_no_valid_frame_is_one_unmeasured_segment() -> None:
    clip = synthetic_clip(CLIP_ID, [("line", 0, False)] * 6)

    rows = segments.clip_segments(clip)

    assert len(rows) == 1
    assert rows[0]["shape_guess"] == UNMEASURED
    assert (rows[0]["start_ms"], rows[0]["end_ms"], rows[0]["mixed"]) == (0, 500, False)
    assert rows[0]["valid_frames"] == 0
    assert math.isnan(rows[0]["knee_angle_median"])


def test_a_clip_without_holds_has_no_segments() -> None:
    assert segments.clip_segments(synthetic_clip(CLIP_ID, [("line", -1, True)] * 5)) == []


# --------------------------------------------------------------------------- #
# build / read / CLI
# --------------------------------------------------------------------------- #


def test_build_and_read_round_trip(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    write_features(data, two_hold_clip(OTHER_ID))
    write_features(data, two_hold_clip(CLIP_ID))

    table = segments.build_segments(data)
    read = segments.read_segments(data)

    assert list(read.columns) == list(segments.SEGMENT_COLUMNS)
    assert read["clip_id"].tolist() == [CLIP_ID] * 3 + [OTHER_ID] * 3  # sorted by clip
    assert read["segment_id"].tolist() == [0, 1, 0, 0, 1, 0]
    assert read["mixed"].tolist() == [True, True, False] * 2
    assert len(table) == 6
    assert segments.segments_path(data) == ft.output_dir(data) / "hold_segments.csv"
    assert not list(ft.output_dir(data).glob(".*.tmp"))


def test_read_segments_is_empty_when_missing(tmp_path: pathlib.Path) -> None:
    table = segments.read_segments(tmp_path)

    assert table.empty
    assert list(table.columns) == list(segments.SEGMENT_COLUMNS)


def test_build_with_clips_keeps_the_other_clips_rows(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    write_features(data, two_hold_clip(CLIP_ID))
    write_features(data, two_hold_clip(OTHER_ID))
    segments.build_segments(data)

    # Re-segment one clip with a minimum so large its mixed hold collapses.
    segments.build_segments(data, clips=[CLIP_ID], min_ms=10_000)
    table = segments.read_segments(data)

    mine = table[table["clip_id"] == CLIP_ID]
    other = table[table["clip_id"] == OTHER_ID]
    assert len(mine) == 2 and not mine["mixed"].any()
    assert len(other) == 3 and other["mixed"].tolist() == [True, True, False]


def test_load_clip_restores_the_dtypes_the_features_writer_used(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    clip = two_hold_clip()
    write_features(data, clip)

    loaded = segments.load_clip(CLIP_ID, data)

    assert loaded.value("com_complete").dtype == bool
    assert loaded.value("balance_zone").dtype == object
    assert set(loaded.value("balance_zone")[~pd.isna(loaded.value("balance_zone"))]) == {
        "under",
        "ok",
        "over",
    }
    assert loaded.value("knee_angle").dtype == np.float64
    assert loaded.valid.tolist() == clip.valid.tolist()
    assert [row["shape_guess"] for row in segments.clip_segments(loaded)] == [
        "straddle",
        "line",
        "line",
    ]


def test_a_single_segment_hold_matches_the_hold_summary(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    clip = synthetic_clip(CLIP_ID, [("line", 0, True)] * 15 + [("line", 0, False)] * 2)
    # Non-trivial numbers in every feature, so equal medians are not equal NaNs.
    rng = np.random.default_rng(7)
    for name in ft.FEATURE_NAMES:
        if name not in ("leg_separation", "knee_angle", "hip_angle"):
            clip.measured.values[name] = rng.normal(size=clip.frames)
    write_features(data, clip)
    loaded = segments.load_clip(CLIP_ID, data)

    [row] = segments.clip_segments(loaded)
    [summary] = ft.hold_rows(loaded)

    assert row["mixed"] is False
    assert row["valid_frames"] == summary["valid_frames"]
    for feature in ft.FEATURES:
        for statistic in ("median", "iqr"):
            name = f"{feature.name}_{statistic}"
            assert row[name] == summary[name] or (
                math.isnan(row[name]) and math.isnan(summary[name])
            ), name


def test_a_bad_clip_is_reported_and_the_rest_are_built(tmp_path: pathlib.Path, capsys) -> None:
    data = tmp_path / "data"
    write_features(data, two_hold_clip(CLIP_ID))
    broken = ft.output_dir(data) / f"{OTHER_ID}.parquet"
    pd.DataFrame({"frame_idx": [0]}).to_parquet(broken, index=False)

    errors: list[str] = []
    table = segments.build_segments(data, errors=errors)

    assert set(table["clip_id"]) == {CLIP_ID}
    assert len(errors) == 1 and errors[0].startswith(f"{OTHER_ID}: ")
    assert errors[0] in capsys.readouterr().err

    errors = []
    segments.build_segments(data, clips=["nope"], errors=errors)
    assert len(errors) == 1 and errors[0].startswith("nope: ")
    # The earlier rows of the clip that was not rebuilt are kept.
    assert set(segments.read_segments(data)["clip_id"]) == {CLIP_ID}


def test_the_cli_writes_everything_else_and_exits_1_when_a_clip_fails(
    tmp_path: pathlib.Path, capsys
) -> None:
    data = tmp_path / "data"
    write_features(data, two_hold_clip(CLIP_ID))
    pd.DataFrame({"frame_idx": [0]}).to_parquet(
        ft.output_dir(data) / f"{OTHER_ID}.parquet", index=False
    )

    assert segments.main(["build", "--data", str(data)]) == 1

    assert set(segments.read_segments(data)["clip_id"]) == {CLIP_ID}
    assert len(list((data / "reports" / "segments").glob("*.txt"))) == 1
    assert f"{OTHER_ID}: " in capsys.readouterr().err


def test_line_segments_filters_shape_and_length() -> None:
    table = pd.DataFrame(
        {
            "shape_guess": ["line", "line", "straddle", "line"],
            "duration_s": [2.0, 1.9, 9.0, 5.0],
        }
    )

    assert segments.line_segments(table)["duration_s"].tolist() == [2.0, 5.0]
    assert segments.line_segments(table, min_s=1.0)["duration_s"].tolist() == [2.0, 1.9, 5.0]


def test_the_cli_writes_the_table_and_a_report(tmp_path: pathlib.Path, capsys) -> None:
    data = tmp_path / "data"
    write_features(data, two_hold_clip())

    assert segments.main(["build", "--data", str(data)]) == 0

    table = pd.read_csv(segments.segments_path(data))
    assert table["shape_guess"].tolist() == ["straddle", "line", "line"]
    reports = list((data / "reports" / "segments").glob("*.txt"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "holds: 2, mixed holds: 1" in text
    printed = capsys.readouterr().out
    assert str(segments.segments_path(data)) in printed
    assert str(reports[0]) in printed

    # A second build in the same second writes a second report rather than replacing it.
    assert segments.main(["build", "--data", str(data), "--clips", CLIP_ID]) == 0
    assert len(list((data / "reports" / "segments").glob("*.txt"))) == 2


def test_the_cli_min_ms_reaches_the_rule(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    write_features(data, two_hold_clip())

    assert segments.main(["build", "--data", str(data), "--min-ms", "10000"]) == 0

    assert not pd.read_csv(segments.segments_path(data))["mixed"].any()


def test_the_cli_show_prints_a_clips_segments(tmp_path: pathlib.Path, capsys) -> None:
    data = tmp_path / "data"
    write_features(data, two_hold_clip())
    segments.main(["build", "--data", str(data)])
    capsys.readouterr()

    assert segments.main(["show", CLIP_ID, "--data", str(data)]) == 0
    printed = capsys.readouterr().out
    assert f"{CLIP_ID}: 2 holds, 3 segments" in printed
    assert "h0 s0 straddle 0-1900 ms" in printed
    assert "h0 s1 line 2000-4900 ms" in printed

    assert segments.main(["show", "unknown", "--data", str(data)]) == 0
    assert "no segments for unknown" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def test_report_counts_line_seconds_and_lists_reviewed_mixed_holds() -> None:
    table = pd.DataFrame(segments.clip_segments(two_hold_clip()), columns=segments.SEGMENT_COLUMNS)
    labels = {(CLIP_ID, 0): "line", (CLIP_ID, 1): "line", (OTHER_ID, 0): "tuck"}

    text = segments.report(table, labels)

    assert "holds: 2, mixed holds: 1" in text
    assert "line: 2 segments" in text
    assert "straddle: 1 segments, 1.9s" in text
    assert "line seconds inside mixed holds: 2.9s in 1 segments (>= 2 s: 2.9s in 1)" in text
    assert "line: 1 / 1" in text
    assert f"{CLIP_ID} h0 label=line segments=straddle 1.9s, line 2.9s" in text
    assert "tuck" not in text  # a label for a hold the table does not have is ignored


def test_report_on_an_empty_table() -> None:
    text = segments.report(pd.DataFrame(columns=list(segments.SEGMENT_COLUMNS)), {})

    assert "holds: 0, mixed holds: 0" in text
    assert "line seconds inside mixed holds: 0.0s in 0 segments" in text
