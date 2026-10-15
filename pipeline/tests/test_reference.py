"""Tests for :mod:`handstand.reference`.

Nothing here touches the real handstand videos or the real reviewed labels: every
reference is built inline or from tiny synthetic clips written under ``tmp_path``
(#28's real reference comes out of the user's own data, and no reference number
belongs in this public repository). The synthetic parquet is written through the
same helpers :mod:`test_score` uses, so ``read_clip_features`` reads it exactly as
it reads a real one.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib

import pytest
from test_score import GOOD, write_clip

from handstand import features as ft
from handstand import hold_shapes, reference, reference_review, score

# test_score builds a per-frame table with every column the scoring functions
# read, and writes it through read_clip_features' own contract; reusing its
# ``write_clip``/``GOOD`` keeps one description of "a features parquet".
LABELS_HEADER = ("clip_id", "hold_id", "reference", "wall", "labeled_at")


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #


def write_labels(data: pathlib.Path, rows: list[tuple[str, int, str]]) -> pathlib.Path:
    """A ``reference_review.csv`` with ``(clip_id, hold_id, reference)`` rows."""
    path = reference_review.labels_path(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(LABELS_HEADER)
        for clip_id, hold_id, choice in rows:
            writer.writerow([clip_id, hold_id, choice, "false", "2026-10-15T00:00:00Z"])
    return path


def write_shapes(data: pathlib.Path, rows: list[tuple[str, int, str]]) -> pathlib.Path:
    """A ``hold_shapes.csv`` with ``(clip_id, hold_id, shape)`` rows."""
    path = hold_shapes.labels_path(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(hold_shapes.LABEL_COLUMNS)
        for clip_id, hold_id, shape in rows:
            writer.writerow([clip_id, hold_id, shape, shape, "true", "2026-10-15T00:00:00Z"])
    return path


def good_clips(data: pathlib.Path, clip_ids: list[str]) -> None:
    """One ten-frame, at-the-means clip per id, so a hold scores."""
    for clip_id in clip_ids:
        write_clip(data, clip_id)


# --------------------------------------------------------------------------- #
# build_reference
# --------------------------------------------------------------------------- #


def test_build_reference_uses_sample_sd_and_ignores_nan() -> None:
    # Two scored features over six holds; the third is finite in only four.
    hip = [176.0, 177.0, 178.0, 179.0, 180.0, 181.0]
    head = [10.0, 20.0, 30.0, 40.0, 50.0, math.nan]
    banana = [0.01, 0.02, 0.03, 0.04, math.nan, math.nan]
    rows = [
        {"clip_id": "c", "hold_id": i, "hip_angle": hip[i], "head": head[i], "banana": banana[i]}
        for i in range(6)
    ]

    ref = reference.build_reference(rows, "six synthetic holds")

    assert ref["schema"] == score.REFERENCE_SCHEMA
    assert ref["version"] == score.REFERENCE_VERSION
    assert ref["signs"] == score.REFERENCE_SIGNS
    assert ref["built_from"] == "six synthetic holds"
    assert ref["n_holds"] == 6

    hip_stats = ref["features"]["hip_angle"]
    assert hip_stats["mean"] == pytest.approx(178.5, abs=1e-6)
    assert hip_stats["sd"] == pytest.approx(math.sqrt(17.5 / 5), abs=1e-6)  # ddof = 1
    assert hip_stats["n"] == 6

    # The NaN is dropped: five finite values, the sample SD of those five.
    head_stats = ref["features"]["head"]
    assert head_stats["mean"] == pytest.approx(30.0)
    assert head_stats["sd"] == pytest.approx(math.sqrt(1000.0 / 4))  # ddof = 1, not /5
    assert head_stats["n"] == 5

    # Four finite holds is under MIN_FEATURE_HOLDS = 5: left out entirely.
    assert reference.MIN_FEATURE_HOLDS == 5
    assert "banana" not in ref["features"]


def test_build_reference_leaves_out_a_feature_under_min_n() -> None:
    rows = [{"hip_angle": 176.0, "head": 20.0} for _ in range(6)]
    ref = reference.build_reference(rows, "text", min_n=7)

    assert ref["features"] == {}  # six holds cannot satisfy min_n 7
    assert ref["n_holds"] == 6


# --------------------------------------------------------------------------- #
# write_reference
# --------------------------------------------------------------------------- #


def test_write_reference_round_trips_through_score_loader(tmp_path: pathlib.Path) -> None:
    rows = [{"hip_angle": 176.0 + i} for i in range(6)]
    ref = reference.build_reference(rows, "the tests' inline reference")
    path = reference.write_reference(ref, tmp_path / "reference-line.json")

    loaded = score.load_reference(path)
    assert loaded.built_from == "the tests' inline reference"
    assert loaded.n_holds == 6
    assert loaded.features["hip_angle"] == pytest.approx((178.5, math.sqrt(17.5 / 5), 6))
    # No key the format does not know: the loader refuses any that is not a
    # scored feature, so this passing is the assertion.
    assert set(loaded.features) <= set(score.SCORE_FEATURES)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema"] == score.REFERENCE_SCHEMA
    assert on_disk["version"] == score.REFERENCE_VERSION
    assert on_disk["signs"] == "athlete"
    assert set(on_disk["features"]) <= set(score.SCORE_FEATURES)


# --------------------------------------------------------------------------- #
# collect_hold_values
# --------------------------------------------------------------------------- #


def test_collect_hold_values_reads_good_holds_and_notes_the_rest(
    tmp_path: pathlib.Path,
) -> None:
    data = tmp_path / "data"
    write_clip(data, "clipGood")  # ten valid frames on hold 0
    write_clip(data, "clipShort", frames=3)  # three valid frames: under the floor
    write_clip(data, "clipOther", hold_id=2)  # has hold 2, not hold 9

    holds = {
        ("clipGood", "0"),  # the id may arrive as text; int() is applied
        ("clipShort", 0),
        ("clipOther", "9"),
        ("clipGone", 0),
    }
    rows, notes = reference.collect_hold_values(holds, data)

    assert [(row["clip_id"], row["hold_id"]) for row in rows] == [("clipGood", 0)]
    good = rows[0]
    assert set(score.SCORE_FEATURES) <= set(good)
    assert good["hip_angle"] == pytest.approx(GOOD["hip_angle"][0])
    assert good["com_sway_sd"] == pytest.approx(GOOD["com_sway_sd"][0])
    assert good["hip_angle_sd"] == pytest.approx(GOOD["hip_angle_sd"][0])

    assert "clipShort h0: too few valid frames" in notes
    assert "clipOther h9: hold id not in features" in notes
    assert "clipGone h0: features file missing" in notes
    assert len(notes) == 3


# --------------------------------------------------------------------------- #
# built_from
# --------------------------------------------------------------------------- #


def test_built_from_text_names_no_clip(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    write_labels(data, [("0123456789ab", 0, "yes")])
    rows = [{"clip_id": "0123456789ab", "hold_id": 0} for _ in range(3)]
    rows[2]["clip_id"] = "ffffffffff"

    text = reference.built_from_text(rows, data)

    assert "0123456789ab" not in text
    assert "ffffffffff" not in text
    assert "3 reviewed line holds from 2 clips" in text
    assert "reference review" in text


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def test_the_cli_refuses_fewer_than_min_holds(tmp_path: pathlib.Path, capsys) -> None:
    data = tmp_path / "data"
    good_clips(data, ["clipA", "clipB"])
    write_labels(data, [("clipA", 0, "yes"), ("clipB", 0, "yes")])

    assert reference.main(["build", "--data", str(data), "--min-holds", "5"]) == 2
    assert not reference.default_output(data).is_file()
    assert "only 2 usable hold(s)" in capsys.readouterr().err


def test_the_cli_builds_and_installs(tmp_path: pathlib.Path, capsys) -> None:
    data = tmp_path / "data"
    repo = tmp_path / "repo"
    clip_ids = [f"clip{i:08d}" for i in range(6)]
    good_clips(data, clip_ids)
    write_labels(data, [(clip_id, 0, "yes") for clip_id in clip_ids])

    assert (
        reference.main(
            ["build", "--data", str(data), "--min-holds", "5", "--install"], repo_dir=repo
        )
        == 0
    )

    out = reference.default_output(data)
    assert out.is_file()
    loaded = score.load_reference(out)
    assert loaded.n_holds == 6
    assert loaded.features  # features written
    assert json.loads(out.read_text(encoding="utf-8"))["signs"] == "athlete"

    installed = reference.install_path(repo)
    assert installed == repo / "ios" / "LocalResources" / "reference-line.json"
    assert installed.read_bytes() == out.read_bytes()

    reports = list((data / "reports" / "reference").glob("*.txt"))
    assert len(reports) == 1
    assert "reference holds LOO median" in reports[0].read_text(encoding="utf-8")

    printed = capsys.readouterr().out
    assert f"reference {out}" in printed
    assert "n_holds=6" in printed
    assert f"installed {installed}" in printed


def test_the_cli_built_from_leaks_no_clip_id(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    clip_ids = ["abc000000001", "def000000002"]
    good_clips(data, clip_ids)
    write_labels(data, [(clip_id, 0, "yes") for clip_id in clip_ids])

    assert reference.main(["build", "--data", str(data), "--min-holds", "2"]) == 0

    built_from = json.loads(reference.default_output(data).read_text(encoding="utf-8"))[
        "built_from"
    ]
    for clip_id in clip_ids:
        assert clip_id not in built_from


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


def test_evaluate_scores_reference_holds_and_other_lines(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    clip_ids = [f"clip{i:08d}" for i in range(6)]
    good_clips(data, clip_ids + ["otherclip"])
    # The six reference holds, plus one line hold that did not build the reference.
    write_shapes(
        data,
        [(clip_id, 0, "line") for clip_id in clip_ids] + [("otherclip", 0, "line")],
    )
    rows = [{"clip_id": clip_id, "hold_id": 0, **score.hold_values(
        ft.read_clip_features(ft.output_dir(data) / f"{clip_id}.parquet"), 0
    )} for clip_id in clip_ids]
    ref = reference.build_reference(rows, "the tests' inline reference")

    report = reference.evaluate(rows, ref, data)

    assert "features:" in report
    assert "leave-one-out of the reference holds:" in report
    assert "n=6" in report  # six LOO scores
    # One non-reference line hold scored against the full reference.
    assert "other line holds against the full reference:" in report
    assert "n=1" in report
    assert "reference holds LOO median" in report
