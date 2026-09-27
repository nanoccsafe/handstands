"""Tests for :mod:`handstand.vision_import`.

Nothing here touches a real clip: the CSVs are written into ``tmp_path`` in the
exact shape ``swift/VisionPose`` writes them (a frame per person, plus the
all-empty block of a frame with nobody in it), and the run manifests are
one-field dicts.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from handstand import pose_mediapipe as pm
from handstand import vision_import as vi

JOINTS = vi.JOINT_NAMES


# --------------------------------------------------------------------------- #
# Building runner-shaped CSVs
# --------------------------------------------------------------------------- #


def csv_row(
    frame_idx: int,
    person_idx: int,
    joint: str,
    x: float | None,
    y: float | None,
    confidence: float | None,
    *,
    t_ms: int | None = None,
    rotated: bool = False,
    detected: bool = True,
) -> dict[str, object]:
    """One CSV line, in the column order the runner writes."""
    return {
        "frame_idx": frame_idx,
        "t_ms": frame_idx * 33 if t_ms is None else t_ms,
        "person_idx": person_idx,
        "joint": joint,
        "x": "" if x is None else f"{x:.4f}",
        "y": "" if y is None else f"{y:.4f}",
        "confidence": "" if confidence is None else f"{confidence:.6f}",
        "rotated": "true" if rotated else "false",
        "detected": "true" if detected else "false",
    }


def person_block(
    frame_idx: int,
    person_idx: int,
    *,
    wrists: tuple[float, float] = (300.0, 310.0),
    ankles: tuple[float, float] = (200.0, 210.0),
    rotated: bool = False,
    offset: float = 0.0,
    omit: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """A full 19-joint block for one person.

    Wrists and ankles at the given ``y`` (which is what the lowest-wrist rule
    reads); everything else in the middle of the frame. ``omit`` drops joints,
    which is how a person who cannot be ranked is built.
    """
    rows = []
    for joint in JOINTS:
        if joint in omit:
            continue
        if joint == "left_wrist":
            x, y = wrists[0] + offset, wrists[1]
        elif joint == "right_wrist":
            x, y = wrists[1] + offset, wrists[1]
        elif joint == "left_ankle":
            x, y = ankles[0] + offset, ankles[0]
        elif joint == "right_ankle":
            x, y = ankles[1] + offset, ankles[1]
        else:
            x, y = 280.0 + offset, 250.0
        rows.append(
            csv_row(
                frame_idx, person_idx, joint, x, y, 0.75,
                rotated=rotated,
            )
        )
    return rows


def empty_block(frame_idx: int, *, rotated: bool = False) -> list[dict[str, object]]:
    """The placeholder block of a frame nobody was detected in."""
    return [
        csv_row(
            frame_idx, pm.NO_PERSON_IDX, joint, None, None, None,
            rotated=rotated, detected=False,
        )
        for joint in JOINTS
    ]


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    pd.DataFrame(rows, columns=list(vi.CSV_COLUMNS)).to_csv(path, index=False)
    return path


def write_manifest(path: pathlib.Path, **overrides: object) -> pathlib.Path:
    manifest = {
        "clip_id": path.stem,
        "source_file": "WhatsApp Video 2026-07-06 at 7.09.23 PM.mp4",
        "model": "apple-vision-body-pose",
        "vision_revision": 1,
        "macos_version": "26.7.0",
        "rotate": "auto",
        "display_width": 576,
        "display_height": 1024,
        "frame_count": 0,
        "detected_frame_count": 0,
        "rotated_frame_count": 0,
        "runtime_seconds": 2.7,
    }
    manifest.update(overrides)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


@pytest.fixture()
def clip_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Three frames of two people and one of nobody.

    * frame 0: two people, the second one's wrists lower — the one on their hands
    * frame 1: two people, the *first* one's wrists lower
    * frame 2: nobody, and the frame was rotated
    """
    rows: list[dict[str, object]] = []
    rows += person_block(0, 0, wrists=(300.0, 500.0), offset=0.0)
    rows += person_block(0, 1, wrists=(300.0, 900.0), offset=20.0)
    rows += person_block(1, 0, wrists=(300.0, 950.0), offset=0.0)
    rows += person_block(1, 1, wrists=(300.0, 400.0), offset=20.0)
    rows += empty_block(2, rotated=True)
    path = write_csv(tmp_path / "1a2b3c4d5e6f.csv", rows)
    return write_manifest(path, frame_count=3, detected_frame_count=2, rotated_frame_count=1)


# --------------------------------------------------------------------------- #
# The joint vocabulary
# --------------------------------------------------------------------------- #


def test_vision_has_nineteen_joints() -> None:
    assert len(JOINTS) == 19
    assert len(set(JOINTS)) == 19


def test_the_seventeen_shared_joints_are_mediapipes_names() -> None:
    """The whole point of the mapping: a bake-off is a join on the joint name."""
    shared = vi.MEDIAPIPE_SHARED_JOINTS
    assert len(shared) == 17
    assert set(shared) <= set(pm.JOINT_NAMES)
    assert set(shared) | set(vi.VISION_ONLY_JOINTS) == set(JOINTS)


def test_neck_and_root_are_the_only_vision_only_joints() -> None:
    """They are Vision's shoulder and hip midpoints; MediaPipe has no such name."""
    assert vi.VISION_ONLY_JOINTS == ("neck", "root")
    for joint in vi.VISION_ONLY_JOINTS:
        assert joint not in pm.JOINT_NAMES


def test_the_wrist_joints_are_mediapipes() -> None:
    assert vi.WRIST_JOINTS == pm.WRIST_JOINTS
    assert set(vi.WRIST_JOINTS) <= set(JOINTS)


def test_the_output_roots_are_their_own() -> None:
    """A Vision run must never be able to overwrite a MediaPipe parquet."""
    assert vi.SINGLE_OUTPUT_DIRNAME == "vision"
    assert vi.MULTI_OUTPUT_DIRNAME == "vision_multi"
    assert vi.RAW_OUTPUT_DIRNAME == "vision_raw"
    assert "mediapipe" not in (
        vi.SINGLE_OUTPUT_DIRNAME,
        vi.MULTI_OUTPUT_DIRNAME,
        vi.RAW_OUTPUT_DIRNAME,
    )


# --------------------------------------------------------------------------- #
# Reading the CSV
# --------------------------------------------------------------------------- #


def test_reading_gives_the_parquet_dtypes(clip_csv: pathlib.Path) -> None:
    frame = vi.read_csv(clip_csv)
    assert list(frame.columns) == list(vi.CSV_COLUMNS)
    assert frame["frame_idx"].dtype == np.int64
    assert frame["t_ms"].dtype == np.int64
    assert frame["person_idx"].dtype == np.int64
    assert frame["x"].dtype == np.float64
    assert frame["y"].dtype == np.float64
    assert frame["confidence"].dtype == np.float64
    assert frame["rotated"].dtype == bool
    assert frame["detected"].dtype == bool
    # Frames 0 and 1 have two people, frame 2 has nobody.
    assert len(frame) == 2 * 19 + 2 * 19 + 19


def test_empty_fields_become_nan(clip_csv: pathlib.Path) -> None:
    """The placeholder block of an empty frame has no coordinates at all."""
    frame = vi.read_csv(clip_csv)
    empty = frame[frame["frame_idx"] == 2]
    assert len(empty) == 19
    assert empty["x"].isna().all()
    assert empty["y"].isna().all()
    assert empty["confidence"].isna().all()
    assert (~empty["detected"]).all()
    assert (empty["person_idx"] == pm.NO_PERSON_IDX).all()


def test_a_missing_column_is_an_error(tmp_path: pathlib.Path) -> None:
    rows = person_block(0, 0)
    columns = [column for column in vi.CSV_COLUMNS if column != "confidence"]
    pd.DataFrame([{key: row[key] for key in columns} for row in rows]).to_csv(
        tmp_path / "clip.csv", index=False
    )
    with pytest.raises(ValueError, match="confidence"):
        vi.read_csv(tmp_path / "clip.csv")


def test_an_empty_csv_is_an_error(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "clip.csv"
    pd.DataFrame(columns=list(vi.CSV_COLUMNS)).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no rows"):
        vi.read_csv(path)


def test_parse_bool_is_strict() -> None:
    assert vi.parse_bool("true", "c") is True
    assert vi.parse_bool("false", "c") is False
    assert vi.parse_bool("True", "c") is True
    assert vi.parse_bool("1", "c") is True
    assert vi.parse_bool("0", "c") is False
    assert vi.parse_bool(True, "c") is True
    with pytest.raises(ValueError, match="expected true or false"):
        vi.parse_bool("maybe", "c")


# --------------------------------------------------------------------------- #
# The multi-person table
# --------------------------------------------------------------------------- #


def test_the_multi_table_is_the_mediapipe_multi_schema(clip_csv: pathlib.Path) -> None:
    table = vi.to_multi_table(vi.read_csv(clip_csv))
    assert list(table.columns) == list(pm.PARQUET_COLUMNS_MULTI)
    assert table["frame_idx"].dtype == np.int64
    assert table[pm.PERSON_COLUMN].dtype == np.int64
    assert table["rotated"].dtype == bool
    assert table["detected"].dtype == bool
    assert table["x"].dtype == np.float64
    # 3 frames: two people, two people, nobody.
    assert len(table) == 2 * 19 + 2 * 19 + 19


def test_every_person_gets_a_block_with_their_person_idx(clip_csv: pathlib.Path) -> None:
    table = vi.to_multi_table(vi.read_csv(clip_csv))
    frame0 = table[table["frame_idx"] == 0]
    assert sorted(frame0[pm.PERSON_COLUMN].unique()) == [0, 1]
    assert frame0.groupby(pm.PERSON_COLUMN).size().eq(19).all()
    # The two people are genuinely different bodies.
    nose = frame0[frame0["joint"] == "nose"].set_index(pm.PERSON_COLUMN)["x"]
    assert nose[0] == pytest.approx(280.0)
    assert nose[1] == pytest.approx(300.0)


def test_an_empty_frame_keeps_one_placeholder_block(clip_csv: pathlib.Path) -> None:
    table = vi.to_multi_table(vi.read_csv(clip_csv))
    frame2 = table[table["frame_idx"] == 2]
    assert len(frame2) == 19
    assert (frame2[pm.PERSON_COLUMN] == pm.NO_PERSON_IDX).all()
    assert (~frame2["detected"]).all()
    assert frame2["x"].isna().all()
    # `rotated` still describes the frame.
    assert frame2["rotated"].all()


def test_rows_are_ordered_by_frame_then_person(clip_csv: pathlib.Path) -> None:
    table = vi.to_multi_table(vi.read_csv(clip_csv))
    keys = list(zip(table["frame_idx"], table[pm.PERSON_COLUMN], strict=True))
    assert keys == sorted(keys)


def test_confidence_becomes_visibility_and_z_and_presence_are_nan(
    clip_csv: pathlib.Path,
) -> None:
    table = vi.to_multi_table(vi.read_csv(clip_csv))
    assert table["visibility"].to_numpy()[0] == pytest.approx(0.75)
    assert table["z"].isna().all()
    assert table["presence"].isna().all()


# --------------------------------------------------------------------------- #
# The lowest-wrist person
# --------------------------------------------------------------------------- #


def test_mean_wrist_y_needs_both_wrists() -> None:
    assert vi.mean_wrist_y([300.0, 400.0]) == pytest.approx(350.0)
    assert np.isnan(vi.mean_wrist_y([300.0, float("nan")]))
    assert np.isnan(vi.mean_wrist_y([]))


def test_the_lowest_wrists_win() -> None:
    assert vi.lowest_wrist_person_index([100.0, 900.0, 400.0]) == 1
    assert vi.lowest_wrist_person_index([900.0, 100.0]) == 0


def test_an_unrankable_person_is_skipped() -> None:
    """A body with one wrist found must not be ranked on half of itself."""
    assert vi.lowest_wrist_person_index([float("nan"), 900.0]) == 1
    assert vi.lowest_wrist_person_index([float("nan"), 900.0, 400.0]) == 1


def test_the_first_person_is_used_when_nobody_can_be_ranked() -> None:
    assert vi.lowest_wrist_person_index([float("nan"), float("nan")]) == 0
    assert vi.lowest_wrist_person_index([500.0]) == 0


def test_there_has_to_be_somebody() -> None:
    with pytest.raises(ValueError, match="at least one person"):
        vi.lowest_wrist_person_index([])


def test_the_rule_agrees_with_the_mediapipe_runner() -> None:
    """The two runners must pick the same body, or the bake-off compares
    different people on different frames."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        people = int(rng.integers(1, 4))
        scores = rng.choice(
            [np.nan, *rng.uniform(0.0, 1000.0, size=people).tolist()], size=people
        )
        poses = np.full((people, len(pm.JOINT_NAMES), 2), np.nan)
        for index, score in enumerate(scores):
            if np.isfinite(score):
                poses[index, pm.JOINT_INDEX["left_wrist"], 1] = score
                poses[index, pm.JOINT_INDEX["right_wrist"], 1] = score
        winner = pm.lowest_wrist_pose(poses)
        expected = next(
            index
            for index in range(people)
            if np.array_equal(poses[index], winner, equal_nan=True)
        )
        assert vi.lowest_wrist_person_index(scores.tolist()) == expected


# --------------------------------------------------------------------------- #
# The single-person table
# --------------------------------------------------------------------------- #


def test_the_single_table_is_the_mediapipe_single_schema(clip_csv: pathlib.Path) -> None:
    table = vi.to_single_table(vi.read_csv(clip_csv))
    assert list(table.columns) == list(pm.PARQUET_COLUMNS)
    assert pm.PERSON_COLUMN not in table.columns
    assert len(table) == 3 * 19


def test_the_lowest_wrist_person_is_the_one_kept(clip_csv: pathlib.Path) -> None:
    table = vi.to_single_table(vi.read_csv(clip_csv))
    # Frame 0: person 1's wrists are at y=900, person 0's at 500. Person 1 is
    # also the one offset by 20 px, so its x carries that offset too.
    frame0 = table[table["frame_idx"] == 0].set_index("joint")
    assert frame0.loc["left_wrist", "y"] == pytest.approx(900.0)
    assert frame0.loc["left_wrist", "x"] == pytest.approx(320.0)
    # Frame 1: the other way round.
    frame1 = table[table["frame_idx"] == 1].set_index("joint")
    assert frame1.loc["left_wrist", "y"] == pytest.approx(950.0)


def test_an_empty_frame_stays_empty_in_the_single_table(clip_csv: pathlib.Path) -> None:
    table = vi.to_single_table(vi.read_csv(clip_csv))
    frame2 = table[table["frame_idx"] == 2]
    assert len(frame2) == 19
    assert (~frame2["detected"]).all()
    assert frame2["x"].isna().all()
    assert frame2["rotated"].all()


def test_a_person_without_both_wrists_is_never_chosen_over_one_that_has(
    tmp_path: pathlib.Path,
) -> None:
    rows = (
        person_block(0, 0, wrists=(300.0, 200.0), omit=("right_wrist",))
        + person_block(0, 1, wrists=(300.0, 800.0))
    )
    table = vi.to_single_table(vi.read_csv(write_csv(tmp_path / "c.csv", rows)))
    assert table.set_index("joint").loc["left_wrist", "y"] == pytest.approx(800.0)
    # The unrankable person is still written out, just without that joint.
    assert len(table) == 19


def test_a_frame_where_nobody_can_be_ranked_keeps_the_first_person(
    tmp_path: pathlib.Path,
) -> None:
    rows = (
        person_block(0, 0, wrists=(300.0, 200.0), omit=("right_wrist",), offset=5.0)
        + person_block(0, 1, wrists=(300.0, 900.0), omit=("left_wrist",), offset=0.0)
    )
    table = vi.to_single_table(vi.read_csv(write_csv(tmp_path / "c.csv", rows)))
    assert table.set_index("joint").loc["nose", "x"] == pytest.approx(285.0)


# --------------------------------------------------------------------------- #
# run_clip
# --------------------------------------------------------------------------- #


def test_run_clip_writes_both_parquets_and_both_sidecars(
    clip_csv: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    report = vi.run_clip(
        clip_csv, clip_id="1a2b3c4d5e6f", rotate="auto", out_root=tmp_path / "keypoints"
    )
    assert (
        report.multi_parquet_path
        == tmp_path / "keypoints/vision_multi/auto/1a2b3c4d5e6f.parquet"
    )
    assert report.single_parquet_path == tmp_path / "keypoints/vision/auto/1a2b3c4d5e6f.parquet"
    assert report.multi_parquet_path.is_file()
    assert report.single_parquet_path.is_file()
    assert report.multi_json_path.is_file()
    assert report.single_json_path.is_file()
    assert not report.skipped
    assert report.frame_count == 3
    assert report.detected_frames == 2
    assert report.rotated_frames == 1
    assert report.selected_frames == 2
    assert report.frames_by_people == {0: 1, 2: 2}
    assert report.detected_percent == pytest.approx(200 / 3)
    assert report.rotated_percent == pytest.approx(100 / 3)
    assert report.people_summary == "0:1 2:2"


def test_the_parquets_carry_the_mediapipe_columns(
    clip_csv: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    vi.run_clip(clip_csv, clip_id="1a2b3c4d5e6f", rotate="auto", out_root=tmp_path / "keypoints")
    multi = pd.read_parquet(tmp_path / "keypoints/vision_multi/auto/1a2b3c4d5e6f.parquet")
    single = pd.read_parquet(tmp_path / "keypoints/vision/auto/1a2b3c4d5e6f.parquet")
    assert list(multi.columns) == list(pm.PARQUET_COLUMNS_MULTI)
    assert list(single.columns) == list(pm.PARQUET_COLUMNS)
    assert multi["frame_idx"].nunique() == single["frame_idx"].nunique() == 3
    # Byte-for-byte the same joint vocabulary as MediaPipe's own files.
    assert set(multi["joint"]) == set(JOINTS)


def test_the_sidecar_records_the_mac_and_the_model(
    clip_csv: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    report = vi.run_clip(
        clip_csv, clip_id="1a2b3c4d5e6f", rotate="auto", out_root=tmp_path / "keypoints"
    )
    sidecar = json.loads(report.multi_json_path.read_text())
    assert sidecar["model"] == "apple-vision-body-pose"
    assert sidecar["macos_version"] == "26.7.0"
    assert sidecar["rotate"] == "auto"
    assert sidecar["frame_count"] == 3
    assert sidecar["display_width"] == 576
    assert sidecar["display_height"] == 1024
    assert sidecar["runtime_seconds"] == 2.7
    assert sidecar["clip_id"] == "1a2b3c4d5e6f"
    assert sidecar["source_file"].endswith(".mp4")
    assert sidecar["vision_revision"] == 1
    assert sum(sidecar["frames_by_people"].values()) == 3
    # The single-person sidecar says how the one person was chosen.
    single = json.loads(report.single_json_path.read_text())
    assert single["selection"] == "lowest_wrist"
    assert single["selected_frame_count"] == 2


def test_a_missing_manifest_still_imports(tmp_path: pathlib.Path) -> None:
    """The manifest is nice to have; its absence must not lose the keypoints."""
    path = write_csv(tmp_path / "clip.csv", person_block(0, 0))
    report = vi.run_clip(path, clip_id="clip", rotate="none", out_root=tmp_path / "keypoints")
    sidecar = json.loads(report.multi_json_path.read_text())
    assert sidecar["macos_version"] == "unknown"
    assert sidecar["runtime_seconds"] is None
    assert sidecar["display_width"] is None
    assert report.frame_count == 1


def test_a_manifest_from_another_run_is_refused(tmp_path: pathlib.Path) -> None:
    """A CSV and a manifest with different frame counts are a mix-up, not a run."""
    path = write_manifest(write_csv(tmp_path / "clip.csv", person_block(0, 0)), frame_count=99)
    with pytest.raises(ValueError, match="different runs"):
        vi.run_clip(path, clip_id="clip", rotate="auto", out_root=tmp_path / "keypoints")


def test_a_mismatched_frame_count_leaves_no_parquet(tmp_path: pathlib.Path) -> None:
    path = write_manifest(write_csv(tmp_path / "clip.csv", person_block(0, 0)), frame_count=99)
    out = tmp_path / "keypoints"
    with pytest.raises(ValueError):
        vi.run_clip(path, clip_id="clip", rotate="auto", out_root=out)
    assert not (out / "vision_multi/auto/clip.parquet").exists()
    assert not (out / "vision/auto/clip.parquet").exists()


def test_an_existing_parquet_is_skipped(clip_csv: pathlib.Path, tmp_path: pathlib.Path) -> None:
    out = tmp_path / "keypoints"
    first = vi.run_clip(clip_csv, clip_id="1a2b3c4d5e6f", rotate="auto", out_root=out)
    assert not first.skipped
    second = vi.run_clip(clip_csv, clip_id="1a2b3c4d5e6f", rotate="auto", out_root=out)
    assert second.skipped
    assert second.frame_count == 0
    third = vi.run_clip(
        clip_csv, clip_id="1a2b3c4d5e6f", rotate="auto", out_root=out, overwrite=True
    )
    assert not third.skipped
    assert third.frame_count == 3


def test_a_missing_csv_is_an_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="no runner CSV"):
        vi.run_clip(
            tmp_path / "nope.csv", clip_id="x", rotate="auto", out_root=tmp_path / "keypoints"
        )


def test_load_manifest(clip_csv: pathlib.Path, tmp_path: pathlib.Path) -> None:
    assert vi.load_manifest(clip_csv)["clip_id"] == "1a2b3c4d5e6f"
    assert vi.load_manifest(tmp_path / "nothing.csv") == {}
    broken = tmp_path / "broken.csv"
    broken.with_suffix(".json").write_text("{not json")
    with pytest.raises(ValueError, match="broken.json"):
        vi.load_manifest(broken)


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def test_collect_clips(tmp_path: pathlib.Path) -> None:
    raw = tmp_path / "vision_raw/auto"
    raw.mkdir(parents=True)
    assert vi.collect_clips(None, raw, "auto") == []
    # A bare id, with and without the suffix, and an explicit path.
    path = write_csv(raw / "clip.csv", person_block(0, 0))
    assert vi.collect_clips(None, raw, "auto") == [path]
    assert vi.collect_clips(["clip"], raw, "auto") == [path]
    assert vi.collect_clips(["clip.csv"], raw, "auto") == [path]
    assert vi.collect_clips([str(path)], raw, "auto") == [path]
    with pytest.raises(FileNotFoundError, match="run_vision.sh"):
        vi.collect_clips(["missing"], raw, "auto")


def test_main_imports_a_directory(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "keypoints/vision_raw/auto"
    raw.mkdir(parents=True)
    write_manifest(write_csv(raw / "1a2b3c4d5e6f.csv", person_block(0, 0)), frame_count=1)
    monkeypatch.setenv("HANDSTAND_DATA", str(tmp_path))
    assert vi.main(["--rotate", "auto"]) == 0
    assert (tmp_path / "keypoints/vision/auto/1a2b3c4d5e6f.parquet").is_file()
    assert (tmp_path / "keypoints/vision_multi/auto/1a2b3c4d5e6f.parquet").is_file()


def test_main_defaults_to_every_mode_present(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for rotate in ("auto", "none"):
        raw = tmp_path / f"keypoints/vision_raw/{rotate}"
        raw.mkdir(parents=True)
        write_manifest(
            write_csv(raw / "clip.csv", person_block(0, 0)),
            frame_count=1,
            rotate=rotate,
        )
    monkeypatch.setenv("HANDSTAND_DATA", str(tmp_path))
    assert vi.main([]) == 0
    for rotate in ("auto", "none"):
        assert (tmp_path / f"keypoints/vision/{rotate}/clip.parquet").is_file()


def test_main_says_so_when_nothing_has_been_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HANDSTAND_DATA", str(tmp_path))
    assert vi.main([]) == 0
    assert "run tools/mac/run_vision.sh" in capsys.readouterr().out


def test_main_keeps_going_after_a_bad_clip(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "keypoints/vision_raw/auto"
    raw.mkdir(parents=True)
    # A manifest claiming the wrong frame count: this clip must fail...
    write_manifest(write_csv(raw / "aaa.csv", person_block(0, 0)), frame_count=7)
    # ...and this one must still be imported.
    write_manifest(write_csv(raw / "bbb.csv", person_block(0, 0)), frame_count=1)
    monkeypatch.setenv("HANDSTAND_DATA", str(tmp_path))
    assert vi.main(["--rotate", "auto"]) == 1
    output = capsys.readouterr().out
    assert "fail  auto/aaa" in output
    assert "clip  auto/bbb" in output
    assert (tmp_path / "keypoints/vision/auto/bbb.parquet").is_file()


def test_the_dataclass_is_frozen() -> None:
    """Reports are values, not builders."""
    assert dataclasses.fields(vi.ClipReport)
    with pytest.raises(dataclasses.FrozenInstanceError):
        vi.ClipReport(
            clip_id="c", rotate="auto", csv_path=pathlib.Path("c.csv"), frame_count=0,
            detected_frames=0, rotated_frames=0, frames_by_people={},
            multi_parquet_path=pathlib.Path("m"), multi_json_path=pathlib.Path("mj"),
            single_parquet_path=pathlib.Path("s"), single_json_path=pathlib.Path("sj"),
        ).frame_count = 1
