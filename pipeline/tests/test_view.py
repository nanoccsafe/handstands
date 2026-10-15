"""Tests for :mod:`handstand.view`.

Nothing here touches the real handstand videos: the processed parquet and the
sidecar are written from separations the test picks, and the features parquet
comes out of :func:`test_score.write_clip`, so a hold's width is a number the
test can work out by hand. The two widths that decide a view — shoulders and
hips — are set directly, which is the whole measurement.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from test_score import write_clip

from handstand import features as ft
from handstand import reference_review, view

CLIP_ID = "abc123def456"
FRAME_MS = 33
#: The joint names a synthetic processed clip carries, in the order they are
#: written. Only the four the detector reads, so the schema is exactly the two
#: joint pairs.
JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


# --------------------------------------------------------------------------- #
# Synthetic files
# --------------------------------------------------------------------------- #


def write_processed(
    root: pathlib.Path,
    clip_id: str = CLIP_ID,
    *,
    shoulders: Sequence[float],
    hips: Sequence[float],
    length: float = 200.0,
    usable: bool = True,
) -> pathlib.Path:
    """A synthetic processed parquet + sidecar, with the two widths per frame.

    ``shoulders`` and ``hips`` are the pixel separations of each pair on each
    frame; a NaN hides both joints of that pair on that frame. The two joints of
    a pair straddle x=200, so their distance is exactly the number given.
    """
    frames = len(shoulders)
    assert len(hips) == frames
    rows = []
    for index in range(frames):
        for part, width, y in (
            ("shoulder", shoulders[index], 100.0),
            ("hip", hips[index], 220.0),
        ):
            known = math.isfinite(float(width))
            for side, direction in (("left", -0.5), ("right", 0.5)):
                rows.append(
                    {
                        "frame_idx": index,
                        "t_ms": index * FRAME_MS,
                        "joint": f"{side}_{part}",
                        "x": 200.0 + direction * width if known else math.nan,
                        "y": y if known else math.nan,
                        "z": math.nan,
                        "visibility": 0.9 if known else math.nan,
                        "presence": 0.9 if known else math.nan,
                        "rotated": False,
                        "detected": known,
                        "valid": known,
                    }
                )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{clip_id}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    sidecar = {
        "clip_id": clip_id,
        "source": "mediapipe",
        "usable": usable,
        "body_length_px": length if usable else None,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return path


def write_catalogue(data: pathlib.Path, angles: dict[str, str]) -> pathlib.Path:
    """A ``catalogue.csv`` carrying ``clip_id`` and ``camera_angle`` only."""
    path = reference_review.catalogue_path(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("clip_id", "camera_angle"), lineterminator="\n"
        )
        writer.writeheader()
        for clip_id, angle in angles.items():
            writer.writerow({"clip_id": clip_id, "camera_angle": angle})
    return path


# --------------------------------------------------------------------------- #
# view_width, frame by frame
# --------------------------------------------------------------------------- #


def test_view_width_is_the_mean_pair_over_the_body_length() -> None:
    # Shoulders 10 px apart, hips 6: the mean width is 8, over L=200 that is 0.04.
    x = np.array([[0.0, 10.0, 0.0, 6.0]])
    y = np.zeros_like(x)
    valid = np.ones_like(x, dtype=bool)

    widths = view.view_width(x, y, valid, JOINTS, 200.0)

    assert widths[0] == pytest.approx(0.04)


def test_view_width_uses_one_pair_when_only_one_is_valid() -> None:
    x = np.array([[0.0, 10.0, 0.0, 6.0]])
    y = np.zeros_like(x)
    # Only the shoulders are valid: the hips must not be read as a phantom 0.
    valid = np.array([[True, True, False, False]])

    widths = view.view_width(x, y, valid, JOINTS, 200.0)

    assert widths[0] == pytest.approx(10.0 / 200.0)


def test_view_width_is_nan_without_a_pair_or_a_body_length() -> None:
    x = np.array([[0.0, 10.0, 0.0, 6.0]])
    y = np.zeros_like(x)

    neither = view.view_width(x, y, np.zeros_like(x, dtype=bool), JOINTS, 200.0)
    assert math.isnan(neither[0])

    one_shoulder = np.array([[True, False, False, True]])
    half_pair = view.view_width(x, y, one_shoulder, JOINTS, 200.0)
    assert math.isnan(half_pair[0])

    no_length = view.view_width(x, y, np.ones_like(x, dtype=bool), JOINTS, math.nan)
    assert math.isnan(no_length[0])


def test_view_width_ignores_a_joint_the_clip_does_not_report() -> None:
    x = np.array([[0.0, 10.0]])
    y = np.zeros_like(x)
    valid = np.ones_like(x, dtype=bool)

    widths = view.view_width(x, y, valid, ("left_shoulder", "right_shoulder"), 200.0)

    assert widths[0] == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #


def test_classify_reads_the_thresholds() -> None:
    assert view.SIDE_MAX == 0.035
    assert view.NOT_SIDE_MIN == 0.042

    assert view.classify(np.full(20, 0.02))[1] == "side"
    assert view.classify(np.full(20, 0.05))[1] == "not_side"
    assert view.classify(np.full(20, 0.038))[1] == "unsure"


def test_classify_leaves_the_boundaries_and_a_short_hold_unsure() -> None:
    assert view.classify(np.full(20, view.SIDE_MAX))[1] == "unsure"
    assert view.classify(np.full(20, view.NOT_SIDE_MIN))[1] == "not_side"

    short = view.classify(np.full(view.MIN_VIEW_FRAMES - 5, 0.02))
    assert short[1] == "unsure"
    assert short[0] == pytest.approx(0.02)

    assert math.isnan(view.classify(np.full(20, math.nan))[0])


# --------------------------------------------------------------------------- #
# clip_hold_views and build_hold_views
# --------------------------------------------------------------------------- #


def test_clip_hold_views_classifies_each_hold(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    # Twenty side-like frames, then twenty front-like ones, in one clip.
    write_processed(
        ft.input_dir(data),
        CLIP_ID,
        shoulders=[4.0] * 20 + [20.0] * 20,
        hips=[6.0] * 20 + [16.0] * 20,
        length=200.0,
    )
    write_clip(data, CLIP_ID, frames=40, hold_id=[0] * 20 + [1] * 20)

    rows = view.clip_hold_views(CLIP_ID, data)

    by_hold = {row["hold_id"]: row for row in rows}
    assert sorted(by_hold) == [0, 1]
    assert by_hold[0]["view"] == "side"
    assert by_hold[0]["view_frames"] == 20
    assert by_hold[0]["view_width"] == pytest.approx(0.025)
    assert by_hold[1]["view"] == "not_side"
    assert by_hold[1]["view_width"] == pytest.approx(0.09)
    assert all(set(row) == set(view.HOLD_VIEW_COLUMNS) for row in rows)


def test_clip_hold_views_counts_only_the_hold_and_valid_frames(
    tmp_path: pathlib.Path,
) -> None:
    data = tmp_path / "data"
    write_processed(
        ft.input_dir(data),
        CLIP_ID,
        shoulders=[4.0] * 40,
        hips=[6.0] * 40,
        length=200.0,
    )
    # Hold 0 is twenty frames, the first ten of them invalid: ten finite widths.
    write_clip(
        data,
        CLIP_ID,
        frames=40,
        hold_id=[0] * 20 + [1] * 20,
        valid=[False] * 10 + [True] * 30,
    )

    rows = {row["hold_id"]: row for row in view.clip_hold_views(CLIP_ID, data)}

    assert rows[0]["view_frames"] == 10
    assert rows[0]["view"] == "unsure"  # under MIN_VIEW_FRAMES
    assert rows[1]["view_frames"] == 20
    assert rows[1]["view"] == "side"


def test_clip_hold_views_refuses_mismatched_frames(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    write_processed(ft.input_dir(data), CLIP_ID, shoulders=[4.0] * 20, hips=[6.0] * 20)
    write_clip(data, CLIP_ID, frames=10)

    with pytest.raises(ValueError, match="processed has"):
        view.clip_hold_views(CLIP_ID, data)


def test_build_hold_views_sorts_and_skips_a_clip_without_processed(
    tmp_path: pathlib.Path, capsys
) -> None:
    data = tmp_path / "data"
    write_processed(ft.input_dir(data), "clipA", shoulders=[4.0] * 20, hips=[6.0] * 20)
    write_processed(ft.input_dir(data), "clipB", shoulders=[20.0] * 20, hips=[16.0] * 20)
    write_clip(data, "clipA", frames=20)
    write_clip(data, "clipB", frames=20)
    write_clip(data, "clipGone", frames=20)  # features outlive their processed input

    table = view.build_hold_views(data)

    assert table["clip_id"].tolist() == ["clipA", "clipB"]
    assert table["view"].tolist() == ["side", "not_side"]
    assert "clipGone" in capsys.readouterr().out


def test_read_hold_views_round_trips_and_is_empty_without_the_file(
    tmp_path: pathlib.Path,
) -> None:
    data = tmp_path / "data"
    assert view.read_hold_views(data) == {}

    table = pd.DataFrame(
        [{"clip_id": "clipA", "hold_id": 0, "view_frames": 20, "view_width": 0.02, "view": "side"}],
        columns=list(view.HOLD_VIEW_COLUMNS),
    )
    view.write_hold_views(view.hold_views_path(data), table)

    assert view.read_hold_views(data) == {("clipA", 0): "side"}


# --------------------------------------------------------------------------- #
# validation_report
# --------------------------------------------------------------------------- #


def test_validation_report_counts_and_lists_mismatches() -> None:
    table = pd.DataFrame(
        [
            {"hold_id": 0, "clip_id": "s1", "view_frames": 20, "view_width": 0.02, "view": "side"},
            {
                "hold_id": 0,
                "clip_id": "s2",
                "view_frames": 20,
                "view_width": 0.05,
                "view": "not_side",
            },
            {"hold_id": 0, "clip_id": "f1", "view_frames": 20, "view_width": 0.02, "view": "side"},
            {"hold_id": 0, "clip_id": "e1", "view_frames": 20, "view_width": 0.02, "view": "side"},
        ],
        columns=list(view.HOLD_VIEW_COLUMNS),
    )
    angles = {"s1": "side", "s2": "side", "f1": "front", "e1": ""}

    report = view.validation_report(table, angles)

    assert "s2 h0 label=side detected=not_side width=0.0500" in report
    assert "f1 h0 label=front detected=side width=0.0200" in report
    # The empty-label clip is in the totals but not in the label counts.
    assert "detected side: 3 holds in 3 clips (of 4 holds)" in report

    counts = {}
    for line in report.splitlines():
        fields = line.split()
        if fields and fields[0] in ("side", "front", "other"):
            counts[fields[0]] = fields[1:4]
    assert counts["side"] == ["1", "1", "0"]
    assert counts["front"] == ["1", "0", "0"]
    assert counts["other"] == ["0", "0", "0"]


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def test_the_cli_writes_the_table_and_a_report(tmp_path: pathlib.Path, capsys) -> None:
    data = tmp_path / "data"
    write_processed(ft.input_dir(data), CLIP_ID, shoulders=[4.0] * 20, hips=[6.0] * 20)
    write_clip(data, CLIP_ID, frames=20)
    write_catalogue(data, {CLIP_ID: "side"})

    assert view.main(["build", "--data", str(data)]) == 0

    out = pd.read_csv(view.hold_views_path(data))
    assert out["view"].tolist() == ["side"]
    assert list(out.columns) == list(view.HOLD_VIEW_COLUMNS)

    reports = list((data / "reports" / "views").glob("*.txt"))
    assert len(reports) == 1
    assert "detected side: 1 holds in 1 clips (of 1 holds)" in reports[0].read_text(
        encoding="utf-8"
    )

    printed = capsys.readouterr().out
    assert str(view.hold_views_path(data)) in printed
    assert str(reports[0]) in printed
