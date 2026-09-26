"""Tests for :mod:`handstand.trainer_report`.

Nothing here touches the real handstand videos: the input is a hand-built athlete
parquet — the single-person schema plus ``n_people``, ``trainer_contact`` and
``detected`` repeated on each of a frame's 33 rows — with the frame-level columns
set to whatever the test is about. The report reads those columns and the frame
timestamps, so the coordinates in here are filler.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from handstand import athlete
from handstand import pose_mediapipe as pm
from handstand import trainer_report as tr

#: One frame of the athlete schema, as (n_people, trainer_contact, detected).
Frame = tuple[int, bool, bool]
#: One clip's worth: the frame-level columns in order, plus the timestamp step.
Clip = tuple[Sequence[Frame], int]


# --------------------------------------------------------------------------- #
# Synthetic athlete parquets
# --------------------------------------------------------------------------- #


def athlete_table(clip: Clip) -> pd.DataFrame:
    """An athlete parquet for ``clip``: 33 rows per frame, the frame columns repeated."""
    frames, t_ms_step = clip
    joints = np.asarray(pm.JOINT_NAMES)
    rows: list[dict[str, object]] = []
    for frame_idx, (n_people, contact, detected) in enumerate(frames):
        for joint_idx, joint in enumerate(joints):
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "t_ms": frame_idx * t_ms_step,
                    "joint": joint,
                    "x": float(joint_idx),
                    "y": 0.0,
                    "z": 0.0,
                    "visibility": 0.9,
                    "presence": 0.9,
                    "rotated": False,
                    "detected": detected,
                    athlete.SCORE_COLUMN: 0.9 if detected else float("nan"),
                    athlete.N_PEOPLE_COLUMN: n_people,
                    athlete.CONTACT_COLUMN: contact,
                    athlete.CONTACT_REASON_COLUMN: "box_iou" if contact else "",
                }
            )
    return pd.DataFrame(rows, columns=list(athlete.ATHLETE_COLUMNS))


def write_athlete(
    directory: pathlib.Path,
    clip: Clip,
    clip_id: str = "clip00000001",
    sidecar: dict[str, object] | None = None,
) -> pathlib.Path:
    """Write ``<directory>/<clip_id>.parquet``, plus a sidecar when one is given."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{clip_id}.parquet"
    athlete_table(clip).to_parquet(path, index=False)
    if sidecar is not None:
        path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def alone(frames: int = 20) -> list[Frame]:
    """``frames`` frames of a handstand on their own: one person, no contact."""
    return [(1, False, True)] * frames


def with_second_person(frames: int, *, contact: bool = False) -> list[Frame]:
    """``frames`` frames, every one of them with a second person in it."""
    return [(2, contact, True)] * frames


#: The catalogue's real column order, trimmed to the columns this report reads
#: plus ``missing``: the report must not depend on a column it does not use.
CATALOGUE_HEADER = [
    "clip_id",
    "filename",
    "duration_s",
    "camera_angle",
    "full_body",
    "skill",
    "outcome",
    "hold_s",
    "good_clip",
    "notes",
    "missing",
]


def write_catalogue(path: pathlib.Path, rows: Sequence[dict[str, str]]) -> pathlib.Path:
    """A catalogue with the real column order, filled in from ``rows``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOGUE_HEADER, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CATALOGUE_HEADER})
    return path


# --------------------------------------------------------------------------- #
# The small measurements
# --------------------------------------------------------------------------- #


def test_percentage_of_a_whole_and_of_nothing() -> None:
    assert tr.percentage(1, 4) == 25.0
    assert tr.percentage(3, 3) == 100.0
    # A clip with no frames has no percentage; the report must not stop on it.
    assert tr.percentage(0, 0) == 0.0


def test_frame_durations_come_from_the_timestamps() -> None:
    # Evenly spaced: every frame is one step, the last one included.
    assert tr.frame_durations_ms([0, 40, 80, 120]).tolist() == [40.0, 40.0, 40.0, 40.0]
    # Variable frame rate: the gap to the next timestamp is what a frame lasted,
    # and the last frame falls back to the median gap (50 of 33 and 67).
    assert tr.frame_durations_ms([0, 33, 100]).tolist() == [33.0, 67.0, 50.0]
    # The last frame has no timestamp after it, so it takes the median gap, and so
    # does a pair the runner had to bump to keep the timestamps increasing.
    assert tr.frame_durations_ms([0, 0, 40]).tolist() == [40.0, 40.0, 40.0]
    assert tr.frame_durations_ms([5]).tolist() == [0.0]
    assert tr.frame_durations_ms([]).size == 0


def test_longest_run_is_a_stretch_of_frames() -> None:
    assert tr.longest_run([False, True, True, True, False, True]) == 3
    assert tr.longest_run([True]) == 1
    assert tr.longest_run([False, False]) == 0
    assert tr.longest_run([]) == 0


def test_longest_run_in_seconds_times_the_clip_s_own_frames() -> None:
    flags = [False, True, True, True, False, True]
    durations = tr.frame_durations_ms([0, 40, 80, 120, 160, 200])
    assert tr.longest_run_seconds(flags, durations) == pytest.approx(0.12)
    # No contact at all is 0.0 s, not 0 frames.
    assert tr.longest_run_seconds([False, False], durations) == 0.0
    # Timestamps with nothing to say about the frames: a frame is a millisecond.
    assert tr.longest_run_seconds([True, True], []) == pytest.approx(0.002)


def test_the_longest_run_is_the_one_that_lasted_longest() -> None:
    # Four 10 ms frames of contact, then a single 100 ms one. The run with the
    # most frames is not the run that lasted longest, and the column is in
    # seconds: reporting the first as "the longest run" would understate it.
    flags = [True] * 4 + [False] + [True] + [False] * 4
    durations = [10.0] * 4 + [40.0] + [100.0] + [10.0] * 4

    assert tr.longest_run(flags) == 4  # the frame count is a different question
    assert tr.longest_run_seconds(flags, durations) == pytest.approx(0.1)
    # An evenly spaced clip cannot tell the two apart, which is why this only
    # shows up on the variable-frame-rate ones.
    assert tr.longest_run_seconds(flags, [40.0] * len(flags)) == pytest.approx(0.16)
    # Two separate runs: each is measured on its own, never summed.
    assert tr.longest_run_seconds([True, True, False, True], [10.0, 10.0, 5.0, 90.0]) == (
        pytest.approx(0.09)
    )


def test_flagged_seconds_counts_every_flagged_frame() -> None:
    durations = tr.frame_durations_ms([0, 40, 80, 120])
    assert tr.flagged_seconds([True, False, True, True], durations) == pytest.approx(0.12)
    assert tr.flagged_seconds([False, False, False, False], durations) == 0.0
    # Fewer durations than flags cannot be timed, so one millisecond each.
    assert tr.flagged_seconds([True, True, True], [40.0]) == pytest.approx(0.003)


# --------------------------------------------------------------------------- #
# Per-clip metrics
# --------------------------------------------------------------------------- #


def test_per_clip_metrics_of_a_quiet_clip() -> None:
    frames = alone(20)
    frames[3] = (0, False, False)  # a frame nobody was found in
    table = athlete_table((frames, 40))

    metrics = tr.clip_metrics(table, clip_id="clip00000001", filename="a clip.mp4")

    assert metrics.frames == 20
    assert metrics.fps == pytest.approx(25.0)
    assert metrics.seconds == pytest.approx(0.8)
    assert metrics.second_person_frames == 0
    assert metrics.pct_frames_second_person == 0.0
    assert metrics.second_person_seconds == 0.0
    assert metrics.contact_frames == 0
    assert metrics.pct_trainer_contact == 0.0
    assert metrics.longest_contact_run_s == 0.0
    assert metrics.trainer_present is False
    # The one frame the selection could not attribute to anybody.
    assert metrics.athlete_frames == 19
    assert metrics.dropped_frames == 1
    assert metrics.pct_dropped == pytest.approx(5.0)
    assert metrics.pct_athlete == pytest.approx(95.0)
    assert metrics.scorable_frames == 19
    assert metrics.error == ""


def test_percentages_and_counts_agree_with_each_other() -> None:
    frames = alone(10)
    frames[2] = (2, True, True)
    frames[6] = (3, False, True)  # three people is still a second person
    frames[7] = (1, False, False)  # and this one is dropped
    metrics = tr.clip_metrics(athlete_table((frames, 40)), clip_id="clip00000001")

    assert (metrics.second_person_frames, metrics.contact_frames) == (2, 1)
    assert metrics.pct_frames_second_person == pytest.approx(20.0)
    assert metrics.pct_trainer_contact == pytest.approx(10.0)
    assert metrics.pct_dropped == pytest.approx(10.0)
    assert metrics.pct_athlete == pytest.approx(90.0)
    assert metrics.scorable_frames == metrics.athlete_frames - metrics.contact_frames == 8


def test_the_longest_contact_run_is_a_stretch_and_not_a_total() -> None:
    frames = alone(20)
    for index in (2, 3, 4, 8, 15):
        frames[index] = (1, True, True)
    metrics = tr.clip_metrics(athlete_table((frames, 40)), clip_id="clip00000001")

    assert metrics.contact_frames == 5
    assert metrics.pct_trainer_contact == pytest.approx(25.0)
    # Five frames of contact, but never more than three of them in a row.
    assert metrics.longest_contact_run_s == pytest.approx(0.12)


def test_a_one_frame_second_person_is_a_ghost_and_not_a_trainer() -> None:
    frames = alone(30)
    frames[7] = (2, False, True)
    metrics = tr.clip_metrics(athlete_table((frames, 33)), clip_id="clip00000001")

    assert metrics.second_person_frames == 1
    assert metrics.pct_frames_second_person == pytest.approx(100 / 30, abs=0.01)
    assert metrics.second_person_seconds == pytest.approx(0.033, abs=0.001)
    # Under half a second in total: a phantom detection, not somebody standing there.
    assert metrics.trainer_present is False
    # A dropped frame is a different question again.
    assert metrics.dropped_frames == 0


def test_the_persistence_threshold_is_half_a_second() -> None:
    """12 flagged frames at 25 fps is 0.48 s and 13 is 0.52 s: the boundary is exact."""

    def measured(count: int) -> tr.ClipMetrics:
        frames = alone(20)
        for index in range(count):
            frames[index] = (2, False, True)
        return tr.clip_metrics(athlete_table((frames, 40)), clip_id="clip00000001")

    under = measured(12)
    assert under.second_person_seconds == pytest.approx(0.48)
    assert under.trainer_present is False

    at = measured(13)
    assert at.second_person_seconds == pytest.approx(0.52)
    assert at.trainer_present is True
    assert at.second_person_frames == 13
    assert tr.SECOND_PERSON_MIN_SECONDS == 0.5


def test_a_trainer_who_came_and_went_still_counts() -> None:
    """Two quarter-second visits are a trainer: the question is about the total."""
    frames = alone(40)
    for index in [*range(0, 7), *range(30, 37)]:
        frames[index] = (2, False, True)
    metrics = tr.clip_metrics(athlete_table((frames, 40)), clip_id="clip00000001")

    assert metrics.longest_contact_run_s == 0.0  # no contact at all
    assert metrics.second_person_seconds == pytest.approx(0.56)
    assert metrics.trainer_present is True


def test_the_frame_columns_are_reduced_to_one_row_per_frame() -> None:
    frames = tr.frame_flags(athlete_table((alone(5), 40)))

    assert frames.frames == 5
    assert frames.frame_idx.tolist() == [0, 1, 2, 3, 4]
    assert frames.t_ms.tolist() == [0, 40, 80, 120, 160]
    assert frames.n_people.tolist() == [1] * 5
    assert frames.fps == pytest.approx(25.0)
    assert frames.seconds == pytest.approx(0.2)
    assert not frames.second_person.any()
    assert not frames.dropped.any()
    assert not frames.contact.any()
    # A parquet written in the wrong order still comes out in time order.
    shuffled = athlete_table((alone(5), 40)).iloc[::-1]
    assert tr.frame_flags(shuffled).frame_idx.tolist() == [0, 1, 2, 3, 4]


def test_metrics_reject_a_parquet_they_cannot_read() -> None:
    table = athlete_table((alone(3), 40))
    with pytest.raises(ValueError, match="missing column"):
        tr.frame_flags(table.drop(columns=[athlete.N_PEOPLE_COLUMN]))
    with pytest.raises(ValueError, match="no frames"):
        tr.frame_flags(table.iloc[:0])
    with pytest.raises(ValueError, match="missing column"):
        # The plain single-person schema: nothing to report a trainer from.
        tr.clip_metrics(table[list(pm.PARQUET_COLUMNS)], clip_id="clip00000001")


# --------------------------------------------------------------------------- #
# Reading one clip off disk
# --------------------------------------------------------------------------- #


def test_measure_clip_counts_from_the_parquet_and_reasons_from_the_sidecar(
    tmp_path: pathlib.Path,
) -> None:
    frames = alone(20)
    frames[4] = (2, True, True)
    path = write_athlete(
        tmp_path,
        (frames, 40),
        sidecar={
            "clip_id": "clip00000001",
            "frame_count": 20,
            "contact_frame_count": 1,
            "contact_frames_by_reason": {"box_iou": 1, "mixed_skeleton": 0, "bone_length": 0},
            "runtime_seconds": 1.5,
        },
    )

    metrics = tr.measure_clip(
        path,
        clip_id="clip00000001",
        filename="a clip.mp4",
        notes="trainer came in",
        sidecars=[path.with_suffix(".json")],
    )

    assert metrics.frames == 20
    assert metrics.contact_frames == 1  # counted from the parquet...
    assert metrics.reason_counts == {"box_iou": 1, "mixed_skeleton": 0, "bone_length": 0}
    assert metrics.generate_seconds == pytest.approx(1.5)
    assert metrics.notes == "trainer came in"
    # One second-person frame on its own is a ghost, not a trainer.
    assert metrics.trainer_present is False
    # A rule the report does not know about is still counted, not dropped.
    assert metrics.contact_by_reason["box_iou"] == 1


def test_measure_clip_sums_the_runtime_of_both_stages(tmp_path: pathlib.Path) -> None:
    path = write_athlete(tmp_path, (alone(5), 40))
    multi_sidecar = tmp_path / "keypoints" / pm.MULTI_OUTPUT_DIRNAME / "auto" / "clip00000001.json"
    multi_sidecar.parent.mkdir(parents=True, exist_ok=True)
    multi_sidecar.write_text(json.dumps({"runtime_seconds": 9.0}), encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps({"contact_frames_by_reason": {"bone_length": 2}, "runtime_seconds": 1.0}),
        encoding="utf-8",
    )

    metrics = tr.measure_clip(
        path, clip_id="clip00000001", sidecars=[path.with_suffix(".json"), multi_sidecar]
    )

    assert metrics.generate_seconds == pytest.approx(10.0)
    assert metrics.reason_counts == {"box_iou": 0, "mixed_skeleton": 0, "bone_length": 2}


def test_measure_clip_survives_a_missing_or_broken_sidecar(tmp_path: pathlib.Path) -> None:
    path = write_athlete(tmp_path, (alone(5), 40))

    metrics = tr.measure_clip(path, clip_id="clip00000001")  # no sidecar at all
    assert metrics.frames == 5
    assert metrics.generate_seconds == 0.0
    assert metrics.reason_counts == {"box_iou": 0, "mixed_skeleton": 0, "bone_length": 0}

    path.with_suffix(".json").write_text("{not json", encoding="utf-8")
    assert tr.sidecar_runtime(path.with_suffix(".json")) == 0.0
    assert tr.measure_clip(path, clip_id="clip00000001").frames == 5
    assert tr.sidecar_runtime(tmp_path / "nothing.json") == 0.0


def test_a_failed_clip_is_a_row_with_the_error_in_it() -> None:
    metrics = tr.ClipMetrics.failed("clip00000001", "RuntimeError: boom", filename="a.mp4")

    assert metrics.error == "RuntimeError: boom"
    assert metrics.filename == "a.mp4"
    assert metrics.frames == 0
    assert metrics.pct_trainer_contact == 0.0
    assert metrics.trainer_present is False
    assert metrics.scorable_frames == 0


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #


def test_read_catalogue_takes_clip_id_filename_and_notes(tmp_path: pathlib.Path) -> None:
    path = write_catalogue(
        tmp_path / "catalogue.csv",
        [
            {"clip_id": "aaa", "filename": "a.mp4", "notes": "trainer spots every hold"},
            {"clip_id": "bbb", "filename": "b.mp4", "notes": ""},
            {"clip_id": "", "filename": "orphan.mp4", "notes": "no id"},
        ],
    )

    entries = tr.read_catalogue(path)

    # The row with no clip_id cannot be matched to a keypoint file, so it is skipped.
    assert [(entry.clip_id, entry.filename, entry.notes) for entry in entries] == [
        ("aaa", "a.mp4", "trainer spots every hold"),
        ("bbb", "b.mp4", ""),
    ]


def test_read_catalogue_says_how_to_build_a_missing_one(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="handstand.catalogue"):
        tr.read_catalogue(tmp_path / "catalogue.csv")
    assert tr.read_catalogue(write_catalogue(tmp_path / "empty.csv", [])) == []


# --------------------------------------------------------------------------- #
# Generating what is missing
# --------------------------------------------------------------------------- #


def write_multi_sidecar(parquet: pathlib.Path, **fields: object) -> None:
    """A sidecar beside a multi-person parquet, as the runner writes it."""
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.touch()
    parquet.with_suffix(".json").write_text(json.dumps(fields), encoding="utf-8")


def test_a_stale_multi_person_run_is_recognised(tmp_path: pathlib.Path) -> None:
    multi_dir = tmp_path / pm.MULTI_OUTPUT_DIRNAME / "auto"
    parquet = multi_dir / "clip00000001.parquet"

    # Not there at all: missing, not stale — it gets generated either way.
    assert tr._stale_multi_person(parquet) is False

    # The runner's own defaults record nothing, and they report one person a frame.
    write_multi_sidecar(parquet, clip_id="clip00000001")
    assert tr._stale_multi_person(parquet) is True

    write_multi_sidecar(parquet, num_poses=3, **dataclasses.asdict(pm.DEFAULT_DETECTOR_SETTINGS))
    assert tr._stale_multi_person(parquet) is True  # VIDEO mode, thresholds 0.5

    write_multi_sidecar(parquet, num_poses=3, **dataclasses.asdict(tr.DETECTOR_SETTINGS))
    assert tr._stale_multi_person(parquet) is False

    # One person per frame is not this report's input, whatever the mode.
    write_multi_sidecar(parquet, num_poses=1, **dataclasses.asdict(tr.DETECTOR_SETTINGS))
    assert tr._stale_multi_person(parquet) is True

    # Nothing to trust, nothing kept.
    parquet.with_suffix(".json").write_text("", encoding="utf-8")
    assert tr._stale_multi_person(parquet) is True


def test_generate_clip_runs_both_stages_with_the_recommended_settings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_multi(video_path: pathlib.Path, **kwargs: object) -> None:
        calls.append(("multi", {"video": pathlib.Path(video_path), **kwargs}))

    def fake_athlete(clip_id: str, **kwargs: object) -> None:
        calls.append(("athlete", {"clip_id": clip_id, **kwargs}))

    monkeypatch.setattr(tr.pose_mediapipe, "run_clip", fake_multi)
    monkeypatch.setattr(tr.athlete, "run_clip", fake_athlete)

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "a.mp4").touch()
    entry = tr.CatalogueEntry(clip_id="clip00000001", filename="a.mp4")
    keypoints = tmp_path / "keypoints"

    assert tr.generate_clip(entry, keypoints_root=keypoints, videos=videos) == ""

    assert [name for name, _ in calls] == ["multi", "athlete"]
    multi = calls[0][1]
    assert multi["clip_id"] == "clip00000001"
    assert multi["rotate_mode"] == tr.DEFAULT_ROTATE
    assert multi["num_poses"] == 3
    assert multi["settings"] == tr.DETECTOR_SETTINGS
    assert multi["settings"].running_mode == "image"
    assert multi["settings"].min_detection == 0.2
    assert multi["settings"].min_presence == 0.2
    assert pathlib.Path(multi["out_root"]) == keypoints / pm.MULTI_OUTPUT_DIRNAME
    # Nothing exists yet, so this is a fresh run and not a refresh.
    assert multi["overwrite"] is False
    selection = calls[1][1]
    assert pathlib.Path(selection["in_root"]) == keypoints / pm.MULTI_OUTPUT_DIRNAME
    assert pathlib.Path(selection["out_root"]) == keypoints / athlete.ATHLETE_OUTPUT_DIRNAME
    assert selection["overwrite"] is False


def test_generate_clip_refreshes_a_clip_whose_settings_are_wrong(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(
        tr.pose_mediapipe, "run_clip", lambda path, **kw: seen.append(kw["overwrite"])
    )
    monkeypatch.setattr(tr.athlete, "run_clip", lambda clip_id, **kw: seen.append(kw["overwrite"]))

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "a.mp4").touch()
    keypoints = tmp_path / "keypoints"
    write_multi_sidecar(
        keypoints / pm.MULTI_OUTPUT_DIRNAME / "auto" / "clip00000001.parquet", num_poses=1
    )
    entry = tr.CatalogueEntry(clip_id="clip00000001", filename="a.mp4")

    assert tr.generate_clip(entry, keypoints_root=keypoints, videos=videos) == ""
    assert seen == [True, True]  # both stages, or the selection would read a stale input

    write_multi_sidecar(
        keypoints / pm.MULTI_OUTPUT_DIRNAME / "auto" / "clip00000001.parquet",
        num_poses=3,
        **dataclasses.asdict(tr.DETECTOR_SETTINGS),
    )
    assert tr.generate_clip(entry, keypoints_root=keypoints, videos=videos) == ""
    assert seen == [True, True, False, False]  # resumable: the second run skips both


def test_generate_clip_reports_a_missing_video_and_a_stage_that_raises(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(tr.pose_mediapipe, "run_clip", lambda path, **kw: calls.append("multi"))
    monkeypatch.setattr(tr.athlete, "run_clip", lambda clip_id, **kw: calls.append("athlete"))

    missing = tr.generate_clip(
        tr.CatalogueEntry(clip_id="clip00000001", filename="gone.mp4"),
        keypoints_root=tmp_path / "keypoints",
        videos=tmp_path / "videos",
    )
    assert missing.startswith("missing video:")
    assert calls == []

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "a.mp4").touch()

    def explode(path: pathlib.Path, **kwargs: object) -> None:
        raise RuntimeError("could not open video")

    monkeypatch.setattr(tr.pose_mediapipe, "run_clip", explode)
    failure = tr.generate_clip(
        tr.CatalogueEntry(clip_id="clip00000001", filename="a.mp4"),
        keypoints_root=tmp_path / "keypoints",
        videos=videos,
    )
    assert failure == "RuntimeError: could not open video"
    assert calls == []  # the selection never ran on a missing input


def test_a_clip_that_raises_is_recorded_and_the_batch_carries_on(
    tmp_path: pathlib.Path,
) -> None:
    """The failure lands in the error column; every other clip is still attempted."""
    entries = [
        tr.CatalogueEntry(clip_id="aaa", filename="a.mp4"),
        tr.CatalogueEntry(clip_id="boom", filename="b.mp4"),
        tr.CatalogueEntry(clip_id="ccc", filename="c.mp4"),
    ]
    attempted: list[str] = []
    lines: list[str] = []

    def generate(entry: tr.CatalogueEntry, **kwargs: object) -> str:
        attempted.append(entry.clip_id)
        if entry.clip_id == "boom":
            raise RuntimeError("decoder exploded")
        return ""

    failures = tr.generate_batch(
        entries,
        keypoints_root=tmp_path / "keypoints",
        videos=tmp_path / "videos",
        progress=lines.append,
        generate=generate,
    )

    assert attempted == ["aaa", "boom", "ccc"]  # nothing stopped the batch
    assert failures == {"boom": "RuntimeError: decoder exploded"}
    assert len(lines) == 3
    assert "[   2/3] fail  boom" in lines[1]
    assert "decoder exploded" in lines[1]
    assert lines[0].startswith("[   1/3] ok    aaa")
    assert lines[2].startswith("[   3/3] ok    ccc")


def test_a_generator_that_returns_an_error_is_recorded_too(tmp_path: pathlib.Path) -> None:
    failures = tr.generate_batch(
        [tr.CatalogueEntry(clip_id="aaa", filename="a.mp4")],
        keypoints_root=tmp_path / "keypoints",
        videos=tmp_path / "videos",
        progress=lambda message: None,
        generate=lambda entry, **kwargs: "missing video: nope.mp4",
    )
    assert failures == {"aaa": "missing video: nope.mp4"}


def test_generate_batch_passes_the_rotation_mode_and_the_roots(tmp_path: pathlib.Path) -> None:
    seen: list[dict[str, object]] = []

    def generate(entry: tr.CatalogueEntry, **kwargs: object) -> str:
        seen.append(kwargs)
        return ""

    tr.generate_batch(
        [tr.CatalogueEntry(clip_id="aaa", filename="a.mp4")],
        keypoints_root=tmp_path / "keypoints",
        videos=tmp_path / "videos",
        rotate="180",
        progress=lambda message: None,
        generate=generate,
    )

    assert seen[0]["rotate"] == "180"
    assert pathlib.Path(str(seen[0]["keypoints_root"])) == tmp_path / "keypoints"
    assert pathlib.Path(str(seen[0]["videos"])) == tmp_path / "videos"
    assert seen[0]["overwrite"] is False


# --------------------------------------------------------------------------- #
# The CSV
# --------------------------------------------------------------------------- #


def test_the_csv_columns_are_the_documented_ones() -> None:
    # The columns the report is asked for, by name.
    for column in (
        "frames",
        "fps",
        "pct_frames_second_person",
        "pct_trainer_contact",
        "pct_dropped",
        "longest_contact_run_s",
        "trainer_present",
        "notes",
        "error",
    ):
        assert column in tr.REPORT_COLUMNS
    assert len(set(tr.REPORT_COLUMNS)) == len(tr.REPORT_COLUMNS)
    assert tr.REPORTS_DIRNAME == "reports"
    assert (tr.CSV_FILENAME, tr.MARKDOWN_FILENAME) == (
        "trainer_report.csv",
        "trainer_report.md",
    )


def test_the_metrics_table_is_one_row_per_clip_in_catalogue_order() -> None:
    first = tr.clip_metrics(athlete_table((alone(10), 40)), clip_id="aaa", notes="one")
    second = tr.clip_metrics(athlete_table((alone(10), 40)), clip_id="bbb", notes="two")

    table = tr.metrics_table([first, second])

    assert list(table.columns) == list(tr.REPORT_COLUMNS)
    assert table["clip_id"].tolist() == ["aaa", "bbb"]
    assert table["notes"].tolist() == ["one", "two"]
    assert table["frames"].tolist() == [10, 10]
    assert table["fps"].tolist() == pytest.approx([25.0, 25.0])
    assert table["trainer_present"].tolist() == ["false", "false"]  # as in catalogue.csv
    for column in tr._INT_COLUMNS:
        assert table[column].dtype == np.int64, column
    for column in tr._FLOAT_COLUMNS:
        assert table[column].dtype == np.float64, column

    # A clip nothing could be measured on is still a row.
    failed = tr.metrics_table([tr.ClipMetrics.failed("ccc", "RuntimeError: boom")])
    assert failed["error"].tolist() == ["RuntimeError: boom"]
    assert failed["frames"].tolist() == [0]
    assert failed["contact_box_iou_frames"].tolist() == [0]


def test_write_report_writes_both_files(tmp_path: pathlib.Path) -> None:
    metrics = [
        tr.clip_metrics(athlete_table((alone(10), 40)), clip_id="aaa", filename="a.mp4"),
        tr.ClipMetrics.failed("bbb", "RuntimeError: boom", filename="b.mp4"),
    ]

    csv_path, markdown_path = tr.write_report(metrics, data=tmp_path, report_seconds=1.5)

    assert csv_path == tmp_path / "reports" / "trainer_report.csv"
    assert markdown_path == tmp_path / "reports" / "trainer_report.md"
    header, *rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert header.split(",") == list(tr.REPORT_COLUMNS)
    assert len(rows) == 2
    assert rows[0].startswith("aaa,")
    assert rows[1].startswith("bbb,")
    assert "RuntimeError: boom" in rows[1]
    assert "clips with a trainer" in markdown_path.read_text(encoding="utf-8")
    # No temp file is left behind.
    assert sorted(path.name for path in (tmp_path / "reports").iterdir()) == [
        "trainer_report.csv",
        "trainer_report.md",
    ]


# --------------------------------------------------------------------------- #
# The summary
# --------------------------------------------------------------------------- #


def test_the_histogram_has_a_bucket_per_ten_percent() -> None:
    histogram = tr.contact_histogram([0.0, 5.0, 9.9, 10.0, 55.0, 99.9, 100.0])

    assert [label for label, _ in histogram] == [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]
    assert [count for _, count in histogram] == [3, 1, 0, 0, 0, 1, 0, 0, 0, 2]
    # 100 % lands in the top bucket rather than off the end of the table.
    assert histogram[-1] == ("90-100%", 2)
    # A clip with no usable percentage is not counted anywhere.
    assert tr.contact_histogram([float("nan")]) == tr.contact_histogram([])
    assert [count for _, count in tr.contact_histogram([])] == [0] * 10


def test_the_summary_reports_the_headline_numbers() -> None:
    clean = tr.clip_metrics(athlete_table((alone(100), 40)), clip_id="aaa", filename="a.mp4")
    busy = tr.clip_metrics(
        athlete_table((with_second_person(60, contact=True), 40)),
        clip_id="bbb",
        filename="b.mp4",
        notes="hands-on",
    )
    failed = tr.ClipMetrics.failed("ccc", "RuntimeError: boom", filename="c.mp4")

    text = tr.summarise([clean, busy, failed], report_seconds=2.0)

    assert "Measured 2 of 3 catalogue clips" in text
    assert "160 frames" in text
    # 1 of the 2 measured clips has a trainer in it.
    assert "**1 of 2 (50.0 %)**" in text
    assert "clips with no trainer detected" in text
    assert "60 of 160 (37.5 %)" in text  # frames with a second person, and in contact
    assert "0 of 160 (0.0 %)" in text  # frames dropped
    assert "## Contact per clip" in text
    assert "```" in text  # the histogram is a code block, not a paragraph
    assert "60-70% |" in text
    assert "0-10% |" in text
    assert "## The 10 most affected clips" in text
    assert "`bbb`" in text and "hands-on" in text
    assert "`ccc`: RuntimeError: boom" in text
    assert "- this report, reading the parquets and writing these two files: 2.0 s" in text
    assert "could not be measured" in text


def test_the_most_affected_clips_are_ranked_by_contact_then_by_run() -> None:
    scattered = with_second_person(60)
    for index in range(0, 60, 2):  # 30 frames of contact, never two in a row
        scattered[index] = (1, True, True)
    steady = with_second_person(60)
    for index in range(30):  # the same 30 frames, in one unbroken stretch
        steady[index] = (1, True, True)

    ranked = tr._most_affected(
        [
            tr.clip_metrics(athlete_table((scattered, 40)), clip_id="scattered"),
            tr.clip_metrics(athlete_table((steady, 40)), clip_id="steady"),
            tr.clip_metrics(athlete_table((alone(20), 40)), clip_id="quiet"),
        ]
    )

    # Both are 50 % of their clip in contact, so the unbroken stretch decides.
    assert [clip.clip_id for clip in ranked] == ["steady", "scattered", "quiet"]
    assert [clip.pct_trainer_contact for clip in ranked] == [50.0, 50.0, 0.0]
    assert ranked[0].longest_contact_run_s == pytest.approx(1.2)
    assert ranked[1].longest_contact_run_s == pytest.approx(0.04)
    assert len(tr._most_affected([tr.ClipMetrics.failed("x", "boom")])) == 0


def test_the_runtimes_are_reported_apart() -> None:
    metrics = [tr.clip_metrics(athlete_table((alone(10), 40)), clip_id="aaa")]

    # A report-only run generated nothing, so it says nothing about generating:
    # only the keypoints the sidecars recorded, and its own two seconds.
    text = tr.summarise(metrics, report_seconds=2.0)
    assert "generation in this run" not in text
    assert "this report, reading the parquets and writing these two files: 2.0 s" in text
    assert "total for this run: 2.0 s" in text

    # A --generate run says how long it spent generating, and totals it up. The
    # two are four orders of magnitude apart, so a single "this report" number
    # for the whole run would read as though reading the parquets took as long
    # as detecting poses in them.
    text = tr.summarise(metrics, generate_seconds=600.0, report_seconds=2.0)
    assert "generation in this run: 10.0 min" in text
    assert "this report, reading the parquets and writing these two files: 2.0 s" in text
    assert "total for this run: 10.0 min" in text


def test_the_summary_of_nothing_says_so() -> None:
    text = tr.summarise([tr.ClipMetrics.failed("aaa", "RuntimeError: boom")])

    assert "Measured 0 of 1 catalogue clips" in text
    assert "_No clip could be measured._" in text
    assert "_Every clip in the catalogue was measured._" not in text
    assert tr.summarise([]).startswith("# Trainer presence across the dataset")
    assert "Measured 0 of 0 catalogue clips" in tr.summarise([])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def report_workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    """A data dir: a catalogue of two clips, keypoints for one of them only."""
    write_catalogue(
        tmp_path / "catalogue.csv",
        [
            {"clip_id": "clip00000001", "filename": "a.mp4", "notes": "clean"},
            {"clip_id": "clip00000002", "filename": "b.mp4", "notes": "trainer"},
        ],
    )
    write_athlete(
        tmp_path / "keypoints" / athlete.output_dirname() / "auto",
        (with_second_person(20, contact=True), 40),
        "clip00000001",
    )
    return tmp_path


def read_report_csv(data: pathlib.Path) -> list[dict[str, str]]:
    """The rows of the written report, as a dict per clip."""
    with (data / "reports" / "trainer_report.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_cli_reports_on_existing_outputs(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = report_workspace(tmp_path)

    assert tr.main(["--data", str(data)]) == 1  # the second clip has no keypoints

    printed = capsys.readouterr().out
    assert "clips=2" in printed
    assert "generate=no" in printed
    assert (data / "reports" / "trainer_report.csv").is_file()
    assert (data / "reports" / "trainer_report.md").is_file()

    rows = read_report_csv(data)
    # The clip with no keypoints is a row with the error in it, not a missing row.
    assert [row["clip_id"] for row in rows] == ["clip00000001", "clip00000002"]
    assert rows[0]["trainer_present"] == "true"
    assert rows[0]["pct_trainer_contact"] == "100.0"
    assert rows[0]["longest_contact_run_s"] == "0.8"
    assert rows[0]["notes"] == "clean"
    assert rows[0]["error"] == ""
    assert rows[1]["error"].startswith("FileNotFoundError")
    assert "re-run with --generate" in rows[1]["error"]
    assert "FileNotFoundError" in printed  # the summary lists it too
    # A report-only run generated nothing, so its summary must not claim to have.
    markdown = (data / "reports" / "trainer_report.md").read_text(encoding="utf-8")
    assert "generation in this run" not in markdown
    assert "total for this run: " in markdown


def test_the_cli_reports_a_missing_catalogue(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tr.main(["--data", str(tmp_path / "empty")]) == 2
    assert "handstand.catalogue" in capsys.readouterr().out


def test_the_cli_selects_clips_and_honours_the_limit(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = report_workspace(tmp_path)

    assert tr.main(["--data", str(data), "--clips", "clip00000001"]) == 0
    assert "clips=1" in capsys.readouterr().out
    assert [row["clip_id"] for row in read_report_csv(data)] == ["clip00000001"]

    assert tr.main(["--data", str(data), "--limit", "1"]) == 0
    assert len(read_report_csv(data)) == 1

    with pytest.raises(SystemExit):
        tr.main(["--data", str(data), "--clips", "nope"])
    with pytest.raises(SystemExit):
        tr.main(["--data", str(data), "--limit", "-1"])
    with pytest.raises(SystemExit):
        tr.main(["--data", str(data), "--rotate", "sideways"])


def test_the_cli_generates_before_reporting(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = report_workspace(tmp_path)
    generated: list[str] = []

    def fake_generate_batch(
        entries: Sequence[tr.CatalogueEntry], **kwargs: object
    ) -> dict[str, str]:
        generated.extend(entry.clip_id for entry in entries)
        return {}

    monkeypatch.setattr(tr, "generate_batch", fake_generate_batch)

    assert tr.main(["--data", str(data), "--generate"]) == 1  # clip 2 still has no keypoints

    printed = capsys.readouterr().out
    assert generated == ["clip00000001", "clip00000002"]
    assert "generate=yes" in printed
    assert "generated 2 of 2 clips, 0 failed" in printed
    # The generation it did and the report it wrote are timed apart, so the
    # summary does not bill reading 180 parquets for the time spent detecting.
    markdown = (data / "reports" / "trainer_report.md").read_text(encoding="utf-8")
    assert "generation in this run: " in markdown
    assert "this report, reading the parquets" in markdown


def test_the_cli_arguments_are_the_documented_ones() -> None:
    args = tr.build_arg_parser().parse_args(["--generate", "--limit", "3", "--rotate", "180"])

    assert args.generate is True
    assert args.overwrite is False
    assert (args.limit, args.rotate, args.clips) == (3, "180", None)
    assert args.data is None and args.videos is None
    assert pathlib.Path(args.model) == pm.DEFAULT_MODEL_PATH
    assert tr.DEFAULT_ROTATE == "auto"
    assert tr.DETECTOR_SETTINGS == pm.DetectorSettings(
        running_mode="image", min_detection=0.2, min_presence=0.2
    )


def test_the_cli_reports_an_unreadable_parquet_as_one_error_row(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt parquet is one row with an error, not a report that stops."""
    data = report_workspace(tmp_path)
    keypoints = data / "keypoints" / athlete.output_dirname() / "auto"
    (keypoints / "clip00000001.parquet").write_bytes(b"not a parquet")
    write_athlete(
        keypoints,
        (alone(5), 40),
        "clip00000002",
        sidecar={"contact_frames_by_reason": {"box_iou": 2}, "runtime_seconds": 0.5},
    )

    assert tr.main(["--data", str(data)]) == 1

    rows = read_report_csv(data)
    assert rows[0]["error"] != ""
    assert rows[1]["error"] == ""
    assert rows[1]["frames"] == "5"
    assert rows[1]["contact_box_iou_frames"] == "2"
    assert rows[1]["generate_seconds"] == "0.5"
    markdown = (data / "reports" / "trainer_report.md").read_text(encoding="utf-8")
    assert "Measured 1 of 2 catalogue clips" in markdown
    assert "## Failures" in markdown
    assert "clip00000001" in markdown
    capsys.readouterr()


def test_frame_rate_survives_a_clip_whose_timestamps_are_useless() -> None:
    """One frame, no gap to measure: 0 fps, and no division by zero in the summary."""
    metrics = tr.clip_metrics(athlete_table(([(1, False, True)], 40)), clip_id="aaa")

    assert metrics.fps == 0.0
    assert metrics.seconds == 0.0
    assert math.isfinite(metrics.pct_trainer_contact)
    assert "trainer" in tr.summarise([metrics])
