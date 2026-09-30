"""Tests for :mod:`handstand.score`.

Nothing here touches the real handstand videos, and no reference number comes
from real data: every reference in this file is built inline from means and SDs
the test itself picked (#28 builds the real one out of the labelled good clips
of #27, and no real number belongs in this public repository before then). The
per-frame tables are built column by column, so a test can say "this feature is
two SDs off" and check the resulting deviation and score by hand.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from handstand import features as ft
from handstand import score

CLIP_ID = "abc000000000"
#: The frame interval of the synthetic clips, in milliseconds.
FRAME_MS = 33

#: The fourteen per-frame features behind the scores: SCORE_FEATURES without
#: the two spreads, which are computed from frames rather than read as a column.
MEDIAN_FEATURES = tuple(
    name for name in score.SCORE_FEATURES if name not in ("com_sway_sd", "hip_angle_sd")
)

#: What the reference of this file says a good hold shows, as ``(mean, sd)``
#: per feature — picked by the tests, one entry per :data:`score.SCORE_FEATURES`
#: name, every SD above the floor of its unit (lengths in body lengths, angles
#: in degrees).
GOOD: dict[str, tuple[float, float]] = {
    "off_shoulder": (0.0, 0.03),
    "off_hip": (0.0, 0.03),
    "off_knee": (0.0, 0.04),
    "off_ankle": (0.0, 0.05),
    "line_deviation": (0.05, 0.03),
    "body_angle": (0.0, 3.0),
    "shoulder_angle": (178.0, 4.0),
    "hip_angle": (176.0, 3.0),
    "knee_angle": (179.0, 2.0),
    "elbow_angle": (175.0, 4.0),
    "banana": (0.02, 0.03),
    "head": (18.0, 6.0),
    "leg_separation": (3.0, 4.0),
    "com_forward": (0.01, 0.02),
    "com_sway_sd": (0.02, 0.01),
    "hip_angle_sd": (1.0, 0.5),
}


# --------------------------------------------------------------------------- #
# Synthetic tables and references
# --------------------------------------------------------------------------- #


def spread(mean: float, sd: float, frames: int = 10) -> list[float]:
    """``frames`` values whose median is ``mean`` and whose population SD is ``sd``.

    Symmetric around the mean (so the median lands on it for an odd or even
    frame count alike) and scaled to unit spread first, which is how a test
    holds a hold's median and its sway apart: the CoM can sit exactly where the
    reference puts it and still wander exactly as much as the reference says.
    """
    if frames < 2:
        return [mean] * frames
    steps = np.arange(frames, dtype=float) - (frames - 1) / 2
    steps = steps / float(np.std(steps))
    return [float(mean + sd * step) for step in steps]


def _is_array(value: object) -> bool:
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def _column(value: object, frames: int) -> np.ndarray:
    """One column of the table: an array as given, a scalar broadcast to every frame."""
    if _is_array(value):
        column = np.asarray(value)
        if column.shape != (frames,):
            raise ValueError(f"expected {frames} values, got shape {column.shape}")
        return column
    return np.full(frames, value)


def frame_table(frames: int | None = None, **columns: object) -> pd.DataFrame:
    """A minimal per-frame features table for the scoring functions.

    Every column :func:`score.hold_values` reads defaults to a value a good
    hold would show (see :data:`GOOD`), so a test overrides only what it is
    about; ``frame_idx`` and ``t_ms`` are filled in when they are not given.
    Scalars are broadcast, arrays are taken as they are, and ``frames`` — when
    passed — is what every array must agree with.
    """
    defaults: dict[str, object] = {
        "phase": "hold",
        "hold_id": 0,
        "valid": True,
        "facing_sign": 1.0,
        **{name: GOOD[name][0] for name in MEDIAN_FEATURES},
    }
    defaults.update(columns)
    if frames is None:
        frames = max((len(value) for value in defaults.values() if _is_array(value)), default=10)
    data = {name: _column(value, frames) for name, value in defaults.items()}
    data.setdefault("frame_idx", np.arange(frames))
    data.setdefault("t_ms", np.arange(frames) * FRAME_MS)
    return pd.DataFrame(data)


def at_means_table(frames: int | None = None, **columns: object) -> pd.DataFrame:
    """A hold whose values all sit on the reference's means, spreads included.

    The two features that are spreads rather than positions get arrays built to
    the right median and the right population SD, so a hold scored from this
    table against :data:`GOOD` is exactly 100 — and a test can move one
    feature off it and know the rest of the score did not budge.
    """
    count = 10 if frames is None else frames
    defaults: dict[str, object] = {
        "com_forward": spread(GOOD["com_forward"][0], GOOD["com_sway_sd"][0], count),
        "hip_angle": spread(GOOD["hip_angle"][0], GOOD["hip_angle_sd"][0], count),
    }
    defaults.update(columns)
    return frame_table(count, **defaults)


def reference(
    features: Mapping[str, tuple[float, float]] | None = None,
    *,
    n_holds: int = 42,
    built_from: str = "the tests' inline reference",
) -> score.Reference:
    """A :class:`score.Reference` from ``(mean, sd)`` pairs; by default, all of GOOD."""
    entries = GOOD if features is None else features
    return score.Reference(
        features={name: (mean, sd, n_holds) for name, (mean, sd) in entries.items()},
        n_holds=n_holds,
        built_from=built_from,
    )


def payload(
    features: Mapping[str, tuple[float, float]] | None = None,
    **markers: object,
) -> dict[str, object]:
    """The JSON a reference file holds, ready for ``json.dumps``.

    ``features`` maps a scored name to ``(mean, sd)`` pairs; ``markers``
    replaces whole top-level keys (a wrong schema, version or signs marker).
    """
    entries = GOOD if features is None else features
    data: dict[str, object] = {
        "schema": score.REFERENCE_SCHEMA,
        "version": score.REFERENCE_VERSION,
        "signs": score.REFERENCE_SIGNS,
        "built_from": "the tests' inline reference",
        "n_holds": 42,
        "features": {
            name: {"mean": mean, "sd": sd, "n": 42} for name, (mean, sd) in entries.items()
        },
    }
    data.update(markers)
    return data


def write_reference(
    path: pathlib.Path,
    features: Mapping[str, tuple[float, float]] | None = None,
    **markers: object,
) -> pathlib.Path:
    """A reference file in ``path``, built from ``(mean, sd)`` pairs."""
    path.write_text(json.dumps(payload(features, **markers)), encoding="utf-8")
    return path


def write_reference_features(path: pathlib.Path, features_object: object) -> pathlib.Path:
    """A reference file whose ``features`` is written verbatim — for malformed ones."""
    data = payload()
    data["features"] = features_object
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_clip(data: pathlib.Path, clip_id: str = CLIP_ID, **columns: object) -> pathlib.Path:
    """One clip's features parquet under ``<data>/features/<source>/``, as #22 writes them."""
    table = at_means_table(**columns)
    # The columns read_clip_features insists on that scoring itself never reads.
    table["hand_width"] = 1.0
    table["com_u"] = 0.0
    table["com_v"] = 0.75
    table["com_complete"] = True
    table["balance_zone"] = "ok"
    table["side_view"] = True
    path = ft.output_dir(data) / f"{clip_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# The constants: groups, weights, floors, signs
# --------------------------------------------------------------------------- #


def test_the_group_weights_sum_to_one() -> None:
    assert sum(weight for _, weight, _ in score.GROUPS) == pytest.approx(1.0)


def test_every_scored_feature_is_in_exactly_one_group_but_the_hip_spread() -> None:
    covered = [name for _, _, members in score.GROUPS for name in members]
    assert set(covered) == set(score.SCORE_FEATURES) - {"hip_angle_sd"}
    assert len(covered) == len(set(covered))  # no feature counted twice
    assert len(score.SCORE_FEATURES) == 16
    # The hip spread is scored where it is discussed: the one-sided penalty.
    assert "hip_angle_sd" not in covered


def test_the_constants_the_issue_fixes_are_those_values() -> None:
    assert score.Z_CAP == 4.0
    assert score.SD_FLOOR_L == 0.01
    assert score.SD_FLOOR_DEG == 1.0
    assert score.HIP_PENALTY_WEIGHT == 0.10
    assert score.MIN_SCORE_FRAMES == 5
    assert score.SIGNED_BY_FACING == (
        "off_shoulder",
        "off_hip",
        "off_knee",
        "off_ankle",
        "body_angle",
    )


def test_the_sd_floor_each_feature_is_divided_by_follows_its_unit() -> None:
    for name in ("off_hip", "line_deviation", "banana", "com_forward", "com_sway_sd"):
        assert score._SD_FLOORS[name] == score.SD_FLOOR_L
    for name in ("body_angle", "shoulder_angle", "hip_angle", "knee_angle", "hip_angle_sd"):
        assert score._SD_FLOORS[name] == score.SD_FLOOR_DEG


# --------------------------------------------------------------------------- #
# Scoring one hold
# --------------------------------------------------------------------------- #


def test_a_hold_sitting_on_every_reference_mean_scores_100() -> None:
    result = score.score_hold(at_means_table(), 0, reference())

    assert result.score == 100.0
    assert result.reason == ""
    assert result.deviation == pytest.approx(0.0)
    # The spreads are built to their SDs in floating point, so the hip penalty
    # is a rounding error from zero rather than bit-exactly it.
    assert result.penalty == pytest.approx(0.0)
    assert result.missing_groups == ()
    assert set(result.groups) == {group for group, _, _ in score.GROUPS}
    for deviation in result.groups.values():
        assert deviation == pytest.approx(0.0)


def test_one_feature_two_sds_off_moves_only_its_group_by_the_right_amount() -> None:
    result = score.score_hold(at_means_table(shoulder_angle=186.0), 0, reference())

    # shoulder_angle 186 against 178 +/- 4 is z = 2, and shoulder is the whole
    # of its group, so d_shoulder = 2 and D = 0.20 * 2 / 1.00 = 0.4.
    assert result.z["shoulder_angle"] == pytest.approx(2.0)
    assert result.groups["shoulder"] == pytest.approx(2.0)
    for group, _, _ in score.GROUPS:
        if group != "shoulder":
            assert result.groups[group] == pytest.approx(0.0)
    assert result.deviation == pytest.approx(0.4)
    assert result.penalty == pytest.approx(0.0)
    # score = 100 * (1 - 0.4 / 4) = 90.
    assert result.score == 90.0
    assert result.top_faults[0] == ("shoulder_angle", 186.0, 178.0, pytest.approx(2.0))
    assert len(result.top_faults) == score.TOP_FAULTS


def test_a_z_beyond_the_cap_is_capped_and_the_score_cannot_go_negative() -> None:
    result = score.score_hold(at_means_table(head=618.0), 0, reference())  # 100 SDs off

    assert result.z["head"] == score.Z_CAP
    assert result.groups["head"] == pytest.approx(score.Z_CAP)
    # D = 0.05 * 4 = 0.2, so score = 100 * (1 - 0.2 / 4) = 95, not 0.
    assert result.deviation == pytest.approx(0.2)
    assert result.score == 95.0

    # And a hold far past the cap on *every* group stops at 0 rather than going
    # under: D of Z_CAP plus the hip penalty is past the whole scale.
    columns: dict[str, object] = {name: 1e9 for name in MEDIAN_FEATURES}
    columns["com_forward"] = spread(1e9, 100.0)
    columns["hip_angle"] = spread(1e9, 100.0)
    hopeless = score.score_hold(at_means_table(**columns), 0, reference())
    assert hopeless.score == 0.0


def test_the_sd_floor_divides_a_near_zero_reference_sd() -> None:
    ref = reference({"off_hip": (0.0, 0.001), "knee_angle": (179.0, 0.0)})
    result = score.score_hold(at_means_table(off_hip=0.02, knee_angle=179.5), 0, ref)

    # Without the floors these would be z = 20 (0.02 / 0.001) and infinite.
    assert result.z["off_hip"] == pytest.approx(0.02 / score.SD_FLOOR_L)
    assert result.z["knee_angle"] == pytest.approx(0.5 / score.SD_FLOOR_DEG)
    assert math.isfinite(result.deviation)


def test_the_facing_sign_flips_the_image_signed_features_before_scoring() -> None:
    filmed_left = at_means_table(facing_sign=1.0, off_hip=0.06, body_angle=6.0)
    filmed_right = at_means_table(facing_sign=-1.0, off_hip=-0.06, body_angle=-6.0)

    left = score.hold_values(filmed_left, 0)
    right = score.hold_values(filmed_right, 0)
    assert left["off_hip"] == pytest.approx(0.06)
    assert right["off_hip"] == pytest.approx(0.06)  # -0.06 * -1: the same pose
    assert left["body_angle"] == pytest.approx(right["body_angle"])

    # The unsigned features are not touched by the flip at all.
    assert right["line_deviation"] == GOOD["line_deviation"][0]

    left_result = score.score_hold(filmed_left, 0, reference())
    right_result = score.score_hold(filmed_right, 0, reference())
    assert left_result.score == right_result.score
    assert left_result.score is not None and left_result.score < 100.0


def test_a_nan_facing_sign_makes_the_five_values_nan() -> None:
    table = at_means_table(facing_sign=math.nan)
    values = score.hold_values(table, 0)

    for name in score.SIGNED_BY_FACING:
        assert math.isnan(values[name])
    # The features that were already athlete-signed or unsigned are untouched.
    assert values["line_deviation"] == GOOD["line_deviation"][0]
    assert values["banana"] == pytest.approx(GOOD["banana"][0])
    assert values["com_forward"] == pytest.approx(GOOD["com_forward"][0])


def test_the_stack_group_falls_back_to_its_other_feature_with_no_facing_sign() -> None:
    result = score.score_hold(at_means_table(facing_sign=math.nan), 0, reference())

    # Every signed feature of the stack is NaN, so the group is line_deviation alone.
    assert "off_hip" not in result.z
    assert result.groups["stack"] == pytest.approx(0.0)
    assert result.missing_groups == ()
    assert result.score == 100.0


def test_a_group_whose_features_are_all_unusable_goes_missing() -> None:
    unusable = (*score.SIGNED_BY_FACING, "line_deviation")
    without_stack = reference({name: good for name, good in GOOD.items() if name not in unusable})
    result = score.score_hold(at_means_table(facing_sign=math.nan), 0, without_stack)

    assert result.groups["stack"] is None
    assert result.missing_groups == ("stack",)
    # The other six groups still score: the weights renormalise over them.
    assert result.score == 100.0


def test_a_quieter_hip_than_the_reference_pays_no_penalty() -> None:
    # The hip angle never moves: hip_angle_sd is 0 against a reference of 1.
    result = score.score_hold(at_means_table(hip_angle=[176.0] * 10), 0, reference())

    assert result.z["hip_angle_sd"] == pytest.approx(-1.0)  # below the mean, one-sided
    assert result.penalty == 0.0
    assert result.deviation == pytest.approx(0.0)
    assert result.score == 100.0


def test_a_busier_hip_than_the_reference_pays_the_penalty() -> None:
    # hip_angle_sd of 2 against a reference of 1 +/- 0.5 is z = 1 (the angle
    # floor is 1 degree), so P = 0.10 * 1 and the score is 100 * (1 - 0.1/4).
    result = score.score_hold(at_means_table(hip_angle=spread(176.0, 2.0, 10)), 0, reference())

    assert result.penalty == pytest.approx(score.HIP_PENALTY_WEIGHT)
    assert result.deviation == pytest.approx(0.0)
    assert result.score == 97.5


def test_missing_groups_renormalise_the_weights() -> None:
    ref = reference({"shoulder_angle": GOOD["shoulder_angle"], "head": GOOD["head"]})
    result = score.score_hold(at_means_table(shoulder_angle=186.0), 0, ref)

    # Only shoulder and head can be compared: D = (0.20 * 2 + 0.05 * 0) / 0.25.
    assert result.missing_groups == ("stack", "hip", "com", "elbows", "knees_toes")
    assert result.deviation == pytest.approx(1.6)
    assert result.score == 60.0


def test_a_hold_with_nothing_to_compare_has_no_score() -> None:
    result = score.score_hold(at_means_table(), 0, reference({}))

    assert result.score is None
    assert result.reason == "nothing to compare"
    assert result.deviation == 0.0
    assert all(value is None for value in result.groups.values())
    assert result.missing_groups == tuple(group for group, _, _ in score.GROUPS)
    assert result.top_faults == ()


def test_a_hold_with_too_few_valid_frames_is_not_scored() -> None:
    result = score.score_hold(at_means_table(frames=4), 0, reference())

    assert result.score is None
    assert result.reason == "too few valid frames"
    assert result.hold_frames == 4
    assert result.valid_frames == 4
    assert result.values == {}
    assert result.z == {}

    # Invalid frames count for the hold's length but not for its score.
    sparsely = score.score_hold(
        at_means_table(frames=10, valid=[True] * 4 + [False] * 6), 0, reference()
    )
    assert sparsely.score is None
    assert sparsely.reason == "too few valid frames"
    assert sparsely.hold_frames == 10
    assert sparsely.valid_frames == 4


def test_invalid_frames_and_frames_outside_the_hold_are_ignored() -> None:
    table = at_means_table(
        16,
        hold_id=[0] * 12 + [-1] * 4,
        valid=[True] * 10 + [False] * 2 + [True] * 4,
        # Junk on exactly the frames that must not be read.
        off_hip=[0.0] * 10 + [999.0] * 6,
        com_forward=spread(GOOD["com_forward"][0], GOOD["com_sway_sd"][0], 10) + [50.0] * 6,
        hip_angle=spread(GOOD["hip_angle"][0], GOOD["hip_angle_sd"][0], 10) + [176.0] * 6,
    )
    values = score.hold_values(table, 0)

    assert values["off_hip"] == pytest.approx(0.0)  # the invalid frames' 999 never enters
    assert values["com_sway_sd"] == pytest.approx(GOOD["com_sway_sd"][0])
    assert values["hip_angle_sd"] == pytest.approx(GOOD["hip_angle_sd"][0])

    result = score.score_hold(table, 0, reference())
    assert result.hold_frames == 12  # both invalid frames belong to the hold
    assert result.valid_frames == 10
    assert result.hold_start_ms == 0
    assert result.hold_end_ms == 11 * FRAME_MS
    assert result.hold_duration_s == pytest.approx(11 * FRAME_MS / 1000.0)
    assert result.score == 100.0

    # The hold_id == -1 frames are not a hold: they are not scored at all.
    assert [held.hold_id for held in score.score_clip(table, reference())] == [0]


def test_the_spread_of_a_hold_is_the_population_sd_of_its_finite_values() -> None:
    two = frame_table(frames=2, com_forward=[0.0, 0.1], hip_angle=[170.0, 180.0])
    values = score.hold_values(two, 0)

    # ddof = 0: (0.05^2 + 0.05^2) / 2 = 0.05^2, not the sample's 0.0707.
    assert values["com_sway_sd"] == pytest.approx(0.05)
    assert values["hip_angle_sd"] == pytest.approx(5.0)

    # One finite value is not a spread, and NaN is never counted.
    one = frame_table(frames=3, com_forward=[0.0, 0.1, math.nan])
    assert score.hold_values(one, 0)["com_sway_sd"] == pytest.approx(0.05)
    lonely = frame_table(frames=2, com_forward=[0.0, math.nan])
    assert math.isnan(score.hold_values(lonely, 0)["com_sway_sd"])


def test_the_hold_rows_are_the_ones_the_hold_summary_writes() -> None:
    table = at_means_table(8, hold_id=[0] * 8, valid=[True] * 6 + [False] * 2)
    result = score.score_hold(table, 0, reference())

    assert result.hold_frames == 8
    assert result.valid_frames == 6
    assert result.hold_start_ms == 0
    assert result.hold_end_ms == 7 * FRAME_MS
    assert result.hold_duration_s == round(7 * FRAME_MS / 1000.0, 3)


# --------------------------------------------------------------------------- #
# clip_score
# --------------------------------------------------------------------------- #


def test_the_clips_score_is_its_longest_scored_hold() -> None:
    reference_ = reference()
    longest = score.score_clip(at_means_table(16, hold_id=[0] * 6 + [1] * 10), reference_)
    assert [held.hold_id for held in longest] == [0, 1]
    assert score.clip_score(longest).hold_id == 1  # 9 frames span beats 5

    # A longer hold nobody could score does not represent the clip.
    sparse = score.score_clip(
        at_means_table(16, hold_id=[0] * 6 + [1] * 10, valid=[True] * 6 + [False] * 10),
        reference_,
    )
    assert score.clip_score(sparse).hold_id == 0

    # Ties go to the earlier hold, like features.longest_hold_id.
    tied = score.score_clip(at_means_table(20, hold_id=[0] * 10 + [1] * 10), reference_)
    assert score.clip_score(tied).hold_id == 0

    # Nothing scored is no answer, not a zero.
    none = score.score_clip(
        at_means_table(8, hold_id=[0] * 4 + [1] * 4, valid=[False] * 8),
        reference_,
    )
    assert score.clip_score(none) is None
    assert score.clip_score([]) is None


# --------------------------------------------------------------------------- #
# The reference file
# --------------------------------------------------------------------------- #


def test_a_reference_file_round_trips(tmp_path: pathlib.Path) -> None:
    path = write_reference(tmp_path / "reference.json")
    loaded = score.load_reference(path)

    assert loaded.n_holds == 42
    assert loaded.built_from == "the tests' inline reference"
    assert loaded.features["hip_angle"] == (176.0, 3.0, 42)
    assert set(loaded.features) == set(GOOD)


def test_a_reference_without_every_feature_is_fine(tmp_path: pathlib.Path) -> None:
    path = write_reference(tmp_path / "reference.json", features={"head": GOOD["head"]})
    loaded = score.load_reference(path)

    assert set(loaded.features) == {"head"}
    # ... and the hold is then only scored where the reference has something to say.
    result = score.score_hold(at_means_table(), 0, loaded)
    assert result.groups["head"] == pytest.approx(0.0)
    assert result.groups["stack"] is None


def test_the_reference_markers_are_refused_when_they_are_wrong(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        score.load_reference(write_reference(tmp_path / "schema.json", schema="something-else"))
    with pytest.raises(ValueError, match="version"):
        score.load_reference(write_reference(tmp_path / "version.json", version=2))
    with pytest.raises(ValueError, match="signs"):
        score.load_reference(write_reference(tmp_path / "signs.json", signs="image"))


def test_a_reference_with_a_bad_number_is_refused(tmp_path: pathlib.Path) -> None:
    negative = {"hip_angle": {"mean": 176.2, "sd": -1.0, "n": 42}}
    with pytest.raises(ValueError, match="sd must be >= 0"):
        score.load_reference(write_reference_features(tmp_path / "negative.json", negative))

    not_a_number = {"hip_angle": {"mean": math.nan, "sd": 3.1, "n": 42}}
    with pytest.raises(ValueError, match="finite"):
        score.load_reference(write_reference_features(tmp_path / "nan.json", not_a_number))

    no_entry = {"hip_angle": {"mean": 176.2, "sd": 3.1}}
    with pytest.raises(ValueError, match=r"features\['hip_angle'\].n"):
        score.load_reference(write_reference_features(tmp_path / "count.json", no_entry))


def test_a_reference_with_a_feature_we_do_not_score_is_refused(tmp_path: pathlib.Path) -> None:
    path = write_reference(tmp_path / "extra.json", features={"hand_width": (1.0, 0.1)})
    with pytest.raises(ValueError, match="not a scored feature"):
        score.load_reference(path)


def test_a_reference_that_is_not_a_file_or_not_json_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="no reference file"):
        score.load_reference(tmp_path / "absent.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        score.load_reference(broken)


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def the_reference_file(tmp_path: pathlib.Path) -> pathlib.Path:
    return write_reference(tmp_path / "reference.json")


def test_the_cli_scores_every_clip_and_writes_scores_csv(
    tmp_path: pathlib.Path, capsys
) -> None:
    data = tmp_path / "data"
    # Two holds: the ten-frame one scores, the three-frame one cannot.
    write_clip(
        data,
        CLIP_ID,
        frames=13,
        hold_id=[0] * 10 + [1] * 3,
        com_forward=spread(GOOD["com_forward"][0], GOOD["com_sway_sd"][0], 10) + [0.01] * 3,
        hip_angle=spread(GOOD["hip_angle"][0], GOOD["hip_angle_sd"][0], 10) + [176.0] * 3,
    )
    reference_path = the_reference_file(tmp_path)

    assert (
        score.main(["--reference", str(reference_path), "--all", "--data", str(data)]) == 0
    )

    out = pd.read_csv(score.output_dir(data) / score.SCORES_NAME)
    assert out["clip_id"].tolist() == [CLIP_ID, CLIP_ID]
    assert out["hold_id"].tolist() == [0, 1]
    assert out.loc[0, "score"] == 100.0
    assert out.loc[0, "valid_frames"] == 10
    assert out.loc[0, "dev_stack"] == pytest.approx(0.0)
    assert out.loc[0, "value_off_hip"] == pytest.approx(0.0)
    assert out.loc[0, "z_off_hip"] == pytest.approx(0.0)
    assert out.loc[0, "top_faults"]  # the three faults are written out as text
    # The hold too short to score keeps its row, and says why.
    assert math.isnan(out.loc[1, "score"])
    assert out.loc[1, "reason"] == "too few valid frames"
    assert out.loc[1, "hold_frames"] == 3

    printed = capsys.readouterr().out
    assert f"scores {score.output_dir(data) / score.SCORES_NAME}" in printed
    assert "clips=1 holds=2 scored=1 unscored=1" in printed
    assert "not scored: too few valid frames=1" in printed
    assert "score min=100.0 median=100.0 max=100.0" in printed
    assert "best:" in printed and "worst:" in printed


def test_the_cli_scores_only_the_clips_it_was_given(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    write_clip(data, CLIP_ID)
    write_clip(data, "def456000000", frames=12, hold_id=[0] * 12)
    reference_path = the_reference_file(tmp_path)

    assert (
        score.main(
            ["--reference", str(reference_path), "--clip", CLIP_ID, "--data", str(data)]
        )
        == 0
    )
    assert [p.name for p in score.output_dir(data).glob("*.csv")] == [score.SCORES_NAME]
    out = pd.read_csv(score.output_dir(data) / score.SCORES_NAME)
    assert out["clip_id"].unique().tolist() == [CLIP_ID]


def test_the_cli_refuses_a_missing_or_invalid_reference(tmp_path: pathlib.Path, capsys) -> None:
    data = tmp_path / "data"
    write_clip(data, CLIP_ID)  # features exist, and are still not read: no yardstick, no score

    assert (
        score.main(
            ["--reference", str(tmp_path / "absent.json"), "--all", "--data", str(data)]
        )
        == 2
    )
    out = capsys.readouterr().out
    assert score.NO_REFERENCE_MESSAGE == (
        "the reference comes from chainlink #28; pass --reference PATH"
    )
    assert score.NO_REFERENCE_MESSAGE in out

    invalid = write_reference(tmp_path / "invalid.json", version=9)
    assert score.main(["--reference", str(invalid), "--all", "--data", str(data)]) == 2
    out = capsys.readouterr().out
    assert "version" in out
    assert score.NO_REFERENCE_MESSAGE in out


def test_the_cli_says_so_when_a_clip_has_no_features(tmp_path: pathlib.Path, capsys) -> None:
    reference_path = the_reference_file(tmp_path)
    empty = tmp_path / "empty"

    assert score.main(["--reference", str(reference_path), "--all", "--data", str(empty)]) == 1
    out = capsys.readouterr().out
    assert "no features" in out
    assert "generate them with: cd pipeline && uv run python -m handstand.features --all" in out

    # An explicit clip that is not there gets the same answer.
    assert (
        score.main(["--reference", str(reference_path), "--clip", "gone", "--data", str(empty)])
        == 1
    )
    out = capsys.readouterr().out
    assert "gone" in out
    assert "handstand.features --all" in out


def test_the_cli_needs_a_reference_and_exactly_one_selection(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):  # no reference at all
        score.main(["--all", "--data", str(tmp_path)])
    with pytest.raises(SystemExit):  # no selection
        score.main(["--reference", "x.json", "--data", str(tmp_path)])
    with pytest.raises(SystemExit):  # both --all and --clip
        score.main(["--reference", "x.json", "--all", "--clip", CLIP_ID, "--data", str(tmp_path)])
    with pytest.raises(SystemExit):  # a source that does not exist
        score.main(
            ["--reference", "x.json", "--all", "--source", "nope", "--data", str(tmp_path)]
        )
