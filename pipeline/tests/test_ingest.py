"""Tests for :mod:`handstand.ingest`: the weekly inbox run (#86).

Everything here is synthetic: ``data``/``videos`` directories built in
``tmp_path``, tiny generated clips (an MJPEG file renamed to ``.mp4`` — the
decoder reads the bytes, not the suffix), ffprobe injected as a stub and the
model steps stubbed, so no MediaPipe weights are ever downloaded. The one real
step exercised end-to-end is ``hold_shapes.prelabel``, which only decodes the
generated clip.
"""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest

from handstand import athlete, features, frame_sampler, hold_shapes, ingest
from handstand import catalogue as catalogue_module

#: The generated clip: 20 frames at 30 fps (633 ms), grey level = index * 12.
WIDTH = 64
HEIGHT = 96
FPS = 30
FRAMES = 20
FRAME_STEP = 12

#: A plausible ffprobe report for the generated clips — enough for build_row.
PROBE_INFO: dict[str, Any] = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "mjpeg",
            "width": WIDTH,
            "height": HEIGHT,
            "avg_frame_rate": f"{FPS}/1",
            "r_frame_rate": f"{FPS}/1",
        }
    ],
    "format": {"duration": "0.667", "bit_rate": "800000"},
}

HOLD_COLUMNS = (
    "clip_id",
    "source",
    "hold_id",
    "hold_frames",
    "valid_frames",
    "hold_start_ms",
    "hold_end_ms",
    "hold_duration_s",
    "leg_separation_median",
    "knee_angle_median",
    "hip_angle_median",
)

#: The two hold rows a 2-hold clip gets: a line (0-400 ms) and a tuck
#: (300-500 ms), inside the 667 ms the generated clip spans.
LINE_HOLD = {
    "source": "mediapipe",
    "hold_id": 0,
    "hold_frames": 12,
    "valid_frames": 12,
    "hold_start_ms": 0,
    "hold_end_ms": 400,
    "hold_duration_s": 0.4,
    "leg_separation_median": 5.0,
    "knee_angle_median": 176.0,
    "hip_angle_median": 172.0,
}
TUCK_HOLD = {
    **LINE_HOLD,
    "hold_id": 1,
    "hold_start_ms": 300,
    "hold_end_ms": 500,
    "hold_duration_s": 0.2,
    "leg_separation_median": 8.0,
    "knee_angle_median": 100.0,
    "hip_angle_median": 170.0,
}


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #


def write_clip(
    directory: pathlib.Path, name: str, *, seed: int = 0, frames: int = FRAMES
) -> pathlib.Path:
    """A tiny generated clip at ``directory/name``; ``seed`` changes the bytes.

    Written as MJPEG into a temp ``.avi`` and renamed: OpenCV decodes by
    content, and two seeds produce two different files while the same file
    copied elsewhere keeps its hash — the property the duplicate tests are
    about.
    """
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory / f".generated_{seed}_{frames}.avi"
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), f"cv2.VideoWriter could not open {temp}"
    for index in range(frames):
        value = (index * FRAME_STEP + seed) % 256
        writer.write(np.full((HEIGHT, WIDTH, 3), value, np.uint8))
    writer.release()
    target = directory / name
    shutil.move(str(temp), str(target))
    return target


def write_hold_summary(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """The hold summary in the schema :func:`handstand.features` writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HOLD_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_catalogue(path: pathlib.Path, rows: list[dict[str, str]]) -> pathlib.Path:
    """A full-width catalogue; ``missing`` defaults to false."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(catalogue_module.COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            full = {column: "" for column in catalogue_module.COLUMNS}
            full[catalogue_module.MISSING_COLUMN] = "false"
            full.update(row)
            writer.writerow(full)
    return path


def write_labels(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """A reviewed ``hold_shapes.csv``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(hold_shapes.LABEL_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def fake_probe(path: pathlib.Path) -> Mapping[str, Any]:
    """ffprobe without the subprocess: :data:`PROBE_INFO` for any file."""
    return PROBE_INFO


def no_probe(path: pathlib.Path) -> Mapping[str, Any]:
    """A probe that must never run — a dry run reads nothing through ffprobe."""
    raise AssertionError(f"a dry run must not probe, but probed {path}")


def quiet(_message: str) -> None:
    """Progress sink for the tests."""


def stub_model_steps() -> dict[str, ingest.StepRunner]:
    """Everything except the real ``hold_shapes.prelabel``: no model runs."""
    return {step: (lambda clips: {}) for step in ingest.STEPS if step != "prelabel"}


def recorder(calls: list[tuple[str, list[str]]]) -> dict[str, ingest.StepRunner]:
    """Step runners that only record ``(step, clips)`` — nothing is executed."""

    def make(step: str) -> ingest.StepRunner:
        def run(clips: Sequence[str]) -> Mapping[str, str]:
            calls.append((step, list(clips)))
            return {}

        return run

    return {step: make(step) for step in ingest.STEPS}


def candidate(frame_idx: int) -> frame_sampler.FrameCandidate:
    """One sampler frame candidate, for :func:`ingest.pick_evenly`."""
    return frame_sampler.FrameCandidate(
        clip_id="aaaaaaaaaaaa",
        frame_idx=frame_idx,
        t_ms=frame_idx * 33,
        clip_t_start_ms=0,
        clip_t_end_ms=1000,
        stratum="clean_no_trainer",
        inverted=True,
        session_date="2026-10-02",
        skill="line",
        trainer_present=False,
        trainer_contact=False,
        display_width=WIDTH,
        display_height=HEIGHT,
    )


def run(
    data: pathlib.Path,
    videos: pathlib.Path,
    *,
    probe: Any = fake_probe,
    steps: dict[str, ingest.StepRunner] | None = None,
    **kwargs: Any,
) -> ingest.IngestReport:
    """One ingest run with the model steps stubbed and quiet progress."""
    return ingest.run_ingest(
        data=data,
        videos=videos,
        probe=probe,
        step_overrides=steps if steps is not None else stub_model_steps(),
        progress=quiet,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# The rename rule
# --------------------------------------------------------------------------- #


def test_whatsapp_target_turns_the_suffix_into_a_take() -> None:
    # The binding rule (#86): '(N)' becomes ' take{N+1}', space or no space.
    assert (
        ingest.whatsapp_target("WhatsApp Video 2026-10-02 at 12.47.45 PM(1).mp4")
        == "WhatsApp Video 2026-10-02 at 12.47.45 PM take2.mp4"
    )
    assert (
        ingest.whatsapp_target("WhatsApp Video 2026-10-02 at 12.47.45 PM (1).mp4")
        == "WhatsApp Video 2026-10-02 at 12.47.45 PM take2.mp4"
    )
    assert ingest.whatsapp_target("clip(2).mp4") == "clip take3.mp4"
    # No suffix: the file keeps its name (it is take 1 implicitly).
    assert ingest.whatsapp_target("clip.mp4") == "clip.mp4"
    assert (
        ingest.whatsapp_target("WhatsApp Video 2026-10-02 at 12.47.45 PM.mp4")
        == "WhatsApp Video 2026-10-02 at 12.47.45 PM.mp4"
    )
    # Not a WhatsApp numbering at the end: left alone.
    assert ingest.whatsapp_target("notes (draft).mp4") == "notes (draft).mp4"


# --------------------------------------------------------------------------- #
# Inbox scan, duplicates, renames, clashes, --keep, --dry-run
# --------------------------------------------------------------------------- #


def test_non_mp4_files_are_reported_and_left_alone(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    write_clip(inbox, "WhatsApp Video 2026-10-02 at 3.00.00 PM.mp4", seed=1)
    jpeg = inbox / "WhatsApp Image 2026-10-02 at 3.01.00 PM.jpeg"
    jpeg.write_bytes(b"\xff\xd8 not really a jpeg \xff\xd9")

    report = run(data, videos, today="2026-10-02")

    assert jpeg.is_file(), "a WhatsApp .jpeg must be left in the inbox"
    assert len(report.others) == 1
    assert len(report.placed) == 1
    assert "left alone: WhatsApp Image 2026-10-02 at 3.01.00 PM.jpeg" in report.text
    assert "scan: 1 mp4 clip(s), 1 other" in report.text


def test_a_duplicate_by_content_stays_in_the_inbox(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    # A clip that is already catalogued ...
    existing = write_clip(videos, "WhatsApp Video 2026-09-01 at 10.00.00 AM.mp4", seed=5)
    clip_id = catalogue_module.file_clip_id(existing)
    write_catalogue(data / "catalogue.csv", [{"clip_id": clip_id, "filename": existing.name}])
    # ... and its byte-identical '(1)' export, dropped into the inbox.
    inbox.mkdir(parents=True)
    duplicate = inbox / "WhatsApp Video 2026-10-02 at 12.47.45 PM(1).mp4"
    shutil.copyfile(existing, duplicate)
    new_clip = write_clip(inbox, "WhatsApp Video 2026-10-02 at 1.00.00 PM.mp4", seed=9)
    new_clip_id = catalogue_module.file_clip_id(new_clip)

    report = run(data, videos, today="2026-10-02")

    # Decided by content, never by the '(1)' name.
    assert len(report.duplicates) == 1
    path, duplicate_id, reason = report.duplicates[0]
    assert path == duplicate
    assert duplicate_id == clip_id
    assert reason == "already in the catalogue"
    assert duplicate.is_file(), "a duplicate is left in the inbox"
    assert not (videos / "WhatsApp Video 2026-10-02 at 12.47.45 PM take2.mp4").exists()
    assert "duplicate(s): 1" in report.text
    assert "already in the catalogue" in report.text
    # The genuinely new clip was still processed.
    assert list(report.placed) == [new_clip_id]


def test_two_identical_new_files_are_one_clip_and_one_duplicate(
    tmp_path: pathlib.Path,
) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    first = write_clip(inbox, "WhatsApp Video 2026-10-02 at 2.00.00 PM.mp4", seed=4)
    twin = inbox / "WhatsApp Video 2026-10-02 at 2.00.00 PM(1).mp4"
    shutil.copyfile(first, twin)

    report = run(data, videos, today="2026-10-02")

    assert len(report.placed) == 1
    assert len(report.duplicates) == 1
    _path, _clip_id, reason = report.duplicates[0]
    assert "in this inbox" in reason
    # Exactly one file moved in, the twin stayed behind.
    assert len([p for p in videos.glob("*.mp4")]) == 1
    assert len(list(inbox.glob("*.mp4"))) == 1


def test_a_take2_rename_is_catalogued_and_processed(tmp_path, capsys) -> None:
    """The pair 12.47.45 PM.mp4 / (1).mp4: different content, two clips."""
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    plain = write_clip(inbox, "WhatsApp Video 2026-10-02 at 12.47.45 PM.mp4", seed=1)
    twin = write_clip(inbox, "WhatsApp Video 2026-10-02 at 12.47.45 PM(1).mp4", seed=2)
    plain_id = catalogue_module.file_clip_id(plain)
    twin_id = catalogue_module.file_clip_id(twin)
    assert plain_id != twin_id
    calls: list[tuple[str, list[str]]] = []

    code = ingest.main(
        ["--data", str(data), "--videos", str(videos)],
        probe=fake_probe,
        step_overrides=recorder(calls),
        today="2026-10-02",
    )

    assert code == 0
    out = capsys.readouterr().out
    assert (
        "WhatsApp Video 2026-10-02 at 12.47.45 PM(1).mp4 -> "
        "WhatsApp Video 2026-10-02 at 12.47.45 PM take2.mp4"
    ) in out
    # Both files are in videos/, the '(1)' one under its take2 name.
    assert (videos / "WhatsApp Video 2026-10-02 at 12.47.45 PM.mp4").is_file()
    assert (videos / "WhatsApp Video 2026-10-02 at 12.47.45 PM take2.mp4").is_file()
    assert list(inbox.iterdir()) == []

    rows = catalogue_module.read_rows(data / "catalogue.csv")
    assert len(rows) == 2
    by_name = {row["filename"]: row for row in rows}
    take2 = by_name["WhatsApp Video 2026-10-02 at 12.47.45 PM take2.mp4"]
    # The timestamp parser reads ' takeN' names (#86): recorded_at stays right.
    assert take2["session_date"] == "2026-10-02"
    assert take2["recorded_at"] == "2026-10-02T12:47:45"
    assert take2["width"] == str(WIDTH)
    for row in rows:
        for column in catalogue_module.ANNOTATION_COLUMNS:
            assert row[column] == "", f"{column} must stay empty for a human"

    # Every step ran, in order, over both new clips (step-major).
    assert [step for step, _ in calls] == list(ingest.STEPS)
    expected_ids = {plain_id, twin_id}
    for _step, clips in calls:
        assert set(clips) == expected_ids

    # The report was written to data/reports/ingest/<date>.txt.
    report_path = data / "reports" / "ingest" / "2026-10-02.txt"
    assert report_path.is_file()
    assert report_path.read_text(encoding="utf-8").startswith("weekly ingest 2026-10-02")


def test_a_name_clash_gets_the_ingest_suffix(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    name = "WhatsApp Video 2026-10-02 at 12.47.45 PM.mp4"
    # A different file already sits at the target name.
    occupant = write_clip(videos, name, seed=3)
    occupant_id = catalogue_module.file_clip_id(occupant)
    incoming = write_clip(inbox, name, seed=4)
    incoming_id = catalogue_module.file_clip_id(incoming)

    report = run(data, videos, today="2026-10-02")

    # The occupant was never overwritten ...
    assert catalogue_module.file_clip_id(occupant) == occupant_id
    # ... the incoming file took the smallest free ' (ingest N)'.
    target = videos / "WhatsApp Video 2026-10-02 at 12.47.45 PM (ingest 1).mp4"
    assert target.is_file()
    assert catalogue_module.file_clip_id(target) == incoming_id
    assert list(inbox.iterdir()) == []
    assert " (ingest 1).mp4" in report.text
    assert str(target) in [str(path) for path in report.placed.values()]


def test_keep_copies_the_files(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    source = write_clip(inbox, "WhatsApp Video 2026-10-02 at 4.00.00 PM.mp4", seed=6)
    before = catalogue_module.file_clip_id(source)

    report = run(data, videos, keep=True, today="2026-10-02")

    assert source.is_file(), "--keep must leave the inbox file where it was"
    target = videos / "WhatsApp Video 2026-10-02 at 4.00.00 PM.mp4"
    assert target.is_file()
    assert catalogue_module.file_clip_id(target) == before
    assert "mode: copy (--keep)" in report.text


def test_dry_run_prints_everything_and_writes_nothing(tmp_path, capsys) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    plain = write_clip(inbox, "WhatsApp Video 2026-10-02 at 12.47.45 PM.mp4", seed=1)
    twin = write_clip(inbox, "WhatsApp Video 2026-10-02 at 12.47.45 PM(1).mp4", seed=2)
    jpeg = inbox / "WhatsApp Image 2026-10-02 at 5.00.00 PM.jpeg"
    jpeg.write_bytes(b"\xff\xd8 jpg \xff\xd9")
    inbox_before = sorted(p.name for p in inbox.iterdir())

    code = ingest.main(
        ["--dry-run", "--data", str(data), "--videos", str(videos)],
        probe=no_probe,
        today="2026-10-02",
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "DRY RUN: nothing was written" in out
    assert "scan: 2 mp4 clip(s), 1 other" in out
    assert "left alone: WhatsApp Image 2026-10-02 at 5.00.00 PM.jpeg" in out
    assert "duplicate(s): 0" in out
    # The take2 rename is shown without being done ...
    assert "12.47.45 PM(1).mp4 -> WhatsApp Video 2026-10-02 at 12.47.45 PM take2.mp4" in out
    # ... and so is what would run.
    assert "would run pose_mediapipe, athlete, postprocess, phases, features, prelabel" in out
    assert "would append 2 row(s)" in out
    # Nothing, anywhere: no catalogue, no report, no moves, no inbox mkdir.
    assert not (data / "catalogue.csv").exists()
    assert not (data / "reports").exists()
    assert sorted(p.name for p in inbox.iterdir()) == inbox_before
    assert plain.is_file() and twin.is_file() and jpeg.is_file()
    assert list(videos.glob("*.mp4")) == []


# --------------------------------------------------------------------------- #
# Step order and per-clip failure isolation
# --------------------------------------------------------------------------- #


def test_pipeline_runs_every_step_in_order() -> None:
    calls: list[tuple[str, list[str]]] = []
    report = ingest.run_pipeline(["a", "b"], recorder(calls), progress=quiet)

    assert [step for step, _ in calls] == list(ingest.STEPS)
    for _step, clips in calls:
        assert clips == ["a", "b"]
    assert report.done == ["a", "b"]
    assert report.failures == {}


def test_a_clip_failing_a_step_drops_out_and_the_others_continue() -> None:
    calls: list[tuple[str, list[str]]] = []
    runners = recorder(calls)

    def pose(clips: Sequence[str]) -> Mapping[str, str]:
        calls.append(("pose_mediapipe", list(clips)))
        return {"a": "boom"} if "a" in clips else {}

    runners["pose_mediapipe"] = pose
    report = ingest.run_pipeline(["a", "b"], runners, progress=quiet)

    assert report.failures == {"pose_mediapipe": {"a": "boom"}}
    assert report.done == ["b"]
    assert report.failed == {"a": "pose_mediapipe"}
    # 'a' ran its first step only; 'b' ran them all.
    after_pose = [step for step, _ in calls if step != "pose_mediapipe"]
    assert after_pose == ["athlete", "postprocess", "phases", "features", "prelabel"]
    for step, clips in calls:
        if step == "pose_mediapipe":
            assert clips == ["a", "b"]
        else:
            assert clips == ["b"]


def test_a_step_wrapper_records_an_exception_and_keeps_going() -> None:
    ran: list[str] = []

    def call(clip_id: str) -> None:
        ran.append(clip_id)
        if clip_id == "a":
            raise ValueError("undecodable")

    failed = ingest._per_clip(call)(["a", "b"])

    assert failed == {"a": "ValueError: undecodable"}
    assert ran == ["a", "b"]


def test_a_failed_step_is_reported_in_the_summary(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    write_clip(inbox, "WhatsApp Video 2026-10-02 at 6.00.00 PM.mp4", seed=8)

    def failing_phases(clips: Sequence[str]) -> Mapping[str, str]:
        return {clip_id: "boom" for clip_id in clips}

    steps = {**stub_model_steps(), "phases": failing_phases}
    report = run(data, videos, steps=steps, today="2026-10-02")

    assert not report.ok
    clip_id = next(iter(report.placed))
    assert report.failures["phases"] == {clip_id: "boom"}
    assert "failures: 1 (per step)" in report.text
    assert f"phases: {clip_id}: boom" in report.text
    # The failure did not stop the file move or the catalogue.
    assert len(report.placed) == 1
    assert report.catalogue_added == 1


# --------------------------------------------------------------------------- #
# The review queue: holds per clip, queued tasks, reviewed holds
# --------------------------------------------------------------------------- #


def test_a_two_hold_clip_gives_two_queued_holds(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    clip = write_clip(inbox, "WhatsApp Video 2026-10-02 at 7.00.00 PM.mp4", seed=10)
    clip_id = catalogue_module.file_clip_id(clip)
    write_hold_summary(
        data / "features" / "mediapipe" / "hold_summary.csv",
        [{**LINE_HOLD, "clip_id": clip_id}, {**TUCK_HOLD, "clip_id": clip_id}],
    )

    report = run(data, videos, today="2026-10-02")

    assert report.ok, report.failures
    assert report.queued == 2
    assert report.queued_per_shape == {"line": 1, "tuck": 1}
    assert report.holds_per_clip == {clip_id: 2}
    assert report.todo == 2
    assert report.import_file is not None and report.import_file.is_file()
    tasks = json.loads(report.import_file.read_text(encoding="utf-8"))
    assert len(tasks) == 2
    assert {task["data"]["hold_id"] for task in tasks} == {0, 1}
    # #87: the image URIs are the local-files form Label Studio can serve.
    for task in tasks:
        assert task["data"]["image"].startswith("/data/local-files/?d=label_frames_holds/")
    manifest = data / "label_frames_holds" / "manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert "queued this run: 2 hold(s) per predicted shape: line=1 tuck=1" in report.text
    assert "review: 2 new hold(s) to label — import file:" in report.text


def test_reviewed_holds_are_not_re_queued(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    clip = write_clip(inbox, "WhatsApp Video 2026-10-02 at 8.00.00 PM.mp4", seed=11)
    clip_id = catalogue_module.file_clip_id(clip)
    write_hold_summary(
        data / "features" / "mediapipe" / "hold_summary.csv",
        [{**LINE_HOLD, "clip_id": clip_id}, {**TUCK_HOLD, "clip_id": clip_id}],
    )
    write_labels(
        data / "labels" / "hold_shapes.csv",
        [{"clip_id": clip_id, "hold_id": 0, "shape": "line"}],
    )

    report = run(data, videos, today="2026-10-02")

    # Hold 0 was reviewed: only hold 1 is queued, and the to-do agrees.
    assert report.queued == 1
    assert report.todo == 1
    tasks = json.loads(report.import_file.read_text(encoding="utf-8"))
    assert [task["data"]["hold_id"] for task in tasks] == [1]
    with (data / "label_frames_holds" / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["hold_id"] for row in rows] == ["1"]
    assert "queued this run: 1 hold(s)" in report.text
    assert "review: 1 new hold(s) to label" in report.text


def test_the_summary_reports_the_trainer_and_the_clips_without_holds(
    tmp_path: pathlib.Path,
) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    first = write_clip(inbox, "WhatsApp Video 2026-10-02 at 9.00.00 AM.mp4", seed=12)
    second = write_clip(inbox, "WhatsApp Video 2026-10-02 at 9.00.01 AM.mp4", seed=13)
    with_id_first = catalogue_module.file_clip_id(first)
    with_id_second = catalogue_module.file_clip_id(second)
    # Only the first clip has a hold; the second's sidecar says a trainer was in.
    write_hold_summary(
        data / "features" / "mediapipe" / "hold_summary.csv",
        [{**LINE_HOLD, "clip_id": with_id_first}],
    )
    athlete_dir = data / "keypoints" / athlete.output_dirname(features.DEFAULT_SOURCE) / "best"
    athlete_dir.mkdir(parents=True, exist_ok=True)
    (athlete_dir / f"{with_id_first}.json").write_text(
        json.dumps({"frames_by_people": {"1": 20}, "contact_frames": 0}), encoding="utf-8"
    )
    (athlete_dir / f"{with_id_second}.json").write_text(
        json.dumps({"frames_by_people": {"2": 10}, "contact_frames": 0}), encoding="utf-8"
    )

    report = run(data, videos, today="2026-10-02")

    assert report.trainer_present == [with_id_second]
    assert report.no_hold == [with_id_second]
    assert f"trainer present: {with_id_second}" in report.text
    assert f"no hold: {with_id_second}" in report.text


# --------------------------------------------------------------------------- #
# The catalogue is appended to, never rewritten
# --------------------------------------------------------------------------- #


def test_the_catalogue_keeps_every_existing_byte(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    inbox = videos / "inbox"
    catalogue_path = data / "catalogue.csv"
    old = catalogue_module.normalise_row(
        {
            "clip_id": "0123456789ab",
            "filename": "WhatsApp Video 2026-09-01 at 10.00.00 AM.mp4",
            "session_date": "2026-09-01",
            "recorded_at": "2026-09-01T10:00:00",
            "duration_s": "6.667",
        }
    )
    old["skill"] = "line"
    old["notes"] = 'typed by hand, with a comma, and "quotes"'
    old["good_clip"] = "1"
    catalogue_module.write_rows([old], catalogue_path)
    before = catalogue_path.read_bytes()
    new_file = write_clip(inbox, "WhatsApp Video 2026-10-02 at 10.00.00 AM.mp4", seed=14)
    new_file_id = catalogue_module.file_clip_id(new_file)

    report = run(data, videos, today="2026-10-02")

    after = catalogue_path.read_bytes()
    # Byte-for-byte: the old file is a prefix of the new one, one row appended.
    assert after.startswith(before)
    assert len(after.decode("utf-8").splitlines()) == 3
    rows = catalogue_module.read_rows(catalogue_path)
    assert len(rows) == 2
    assert rows[0] == old, "the hand-annotated row must not shift by a byte"
    appended = rows[1]
    assert appended["clip_id"] == new_file_id
    assert appended["filename"] == new_file.name
    for column in catalogue_module.ANNOTATION_COLUMNS:
        assert appended[column] == ""
    assert report.catalogue_added == 1


def test_update_hold_summary_keeps_the_other_clips_rows(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "hold_summary.csv"
    columns = list(features.HOLD_SUMMARY_COLUMNS)
    existing = pd.DataFrame(
        [
            {"clip_id": "oldclip00001", "source": "mediapipe", "hold_id": 0},
            {"clip_id": "oldclip00001", "source": "mediapipe", "hold_id": 1},
            {"clip_id": "oldclip00002", "source": "mediapipe", "hold_id": 0},
        ],
        columns=columns,
    )
    existing.to_csv(path, index=False)
    fresh = pd.DataFrame(
        [{"clip_id": "newclip00001", "source": "mediapipe", "hold_id": 0}],
        columns=columns,
    )

    ingest.update_hold_summary(path, fresh)

    back = pd.read_csv(path)
    counts = Counter(back["clip_id"].astype(str))
    assert counts == {"oldclip00001": 2, "oldclip00002": 1, "newclip00001": 1}


# --------------------------------------------------------------------------- #
# Label Studio: only with --label-studio, and only over the API
# --------------------------------------------------------------------------- #


class FakeLabelStudio:
    """The whole HTTP conversation, recorded; answers like Label Studio does."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    def __call__(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: Mapping[str, str],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        self.calls.append((method, url, body, dict(headers)))
        if url.endswith("/user/login/"):
            return 200, [("Set-Cookie", "csrftoken=tok123; Path=/")], b""
        if url.endswith("/user-login/"):
            return (
                200,
                [
                    ("Set-Cookie", "sessionid=sess42; Path=/"),
                    ("Set-Cookie", "csrftoken=tok123; Path=/"),
                ],
                b"ok",
            )
        if "/api/projects?" in url:
            payload = json.dumps({"projects": [{"id": 3, "title": ingest.PROJECT_TITLE}]}).encode()
            return 200, [], payload
        if url.endswith("/import"):
            return 201, [], b'{"tasks_imported": 2}'
        raise AssertionError(f"unexpected request: {method} {url}")


def two_hold_workspace(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A data/videos pair whose one clip has two holds waiting to be queued."""
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    clip = write_clip(videos / "inbox", "WhatsApp Video 2026-10-02 at 11.00.00 AM.mp4", seed=15)
    clip_id = catalogue_module.file_clip_id(clip)
    write_hold_summary(
        data / "features" / "mediapipe" / "hold_summary.csv",
        [{**LINE_HOLD, "clip_id": clip_id}, {**TUCK_HOLD, "clip_id": clip_id}],
    )
    return data, videos


def test_the_label_studio_api_is_only_used_with_the_flag(tmp_path: pathlib.Path) -> None:
    # Without the flag: no HTTP at all, the import JSON is written instead.
    data_a, videos_a = two_hold_workspace(tmp_path / "a")
    http_a = FakeLabelStudio()
    without_flag = run(data_a, videos_a, label_studio=False, http=http_a, today="2026-10-02")

    assert http_a.calls == []
    assert without_flag.import_file is not None and without_flag.import_file.is_file()
    assert "import file:" in without_flag.text

    # With the flag: session login, project lookup by title, then the import.
    data_b, videos_b = two_hold_workspace(tmp_path / "b")
    credentials = tmp_path / "label-studio.env"
    credentials.write_text(
        "LS_USER=tester@example.com\nLS_PASSWORD=sekret-please\n", encoding="utf-8"
    )
    http_b = FakeLabelStudio()
    with_flag = run(
        data_b,
        videos_b,
        label_studio=True,
        http=http_b,
        env_path=credentials,
        today="2026-10-02",
    )

    assert with_flag.ok, with_flag.failures
    methods = [method for method, _url, _body, _headers in http_b.calls]
    urls = [url for _method, url, _body, _headers in http_b.calls]
    assert methods == ["GET", "POST", "GET", "POST"]
    assert urls[0].endswith("/user/login/")
    assert urls[1].endswith("/user-login/")
    assert urls[2] == f"http://localhost:8080/api/projects?title={ingest.PROJECT_TITLE}"
    assert urls[3] == "http://localhost:8080/api/projects/3/import"
    # The imported payload is exactly the two new tasks ...
    body = http_b.calls[3][2]
    assert body is not None and len(json.loads(body)) == 2
    # ... carrying the session and the CSRF token, as the API requires.
    import_headers = http_b.calls[3][3]
    assert import_headers["Cookie"].startswith("sessionid=sess42") or (
        "sessionid=sess42" in import_headers["Cookie"]
    )
    assert import_headers["X-CSRFToken"] == "tok123"
    # The credentials never reach the summary.
    assert "sekret-please" not in with_flag.text
    assert "tester@example.com" not in with_flag.text
    assert "imported 2 new task(s) into 'handstand-hold-shapes' (project 3)" in with_flag.text
    assert with_flag.import_file is None, "with the flag the tasks go to the API"


# --------------------------------------------------------------------------- #
# --sample-frames: the frame picker
# --------------------------------------------------------------------------- #


def test_pick_evenly_spreads_the_picks_over_the_clip() -> None:
    frames = [candidate(index) for index in range(10)]
    assert [frame.frame_idx for frame in ingest.pick_evenly(frames, 4)] == [0, 3, 6, 9]
    assert [frame.frame_idx for frame in ingest.pick_evenly(frames, 1)] == [5]
    # Fewer candidates than asked for: take them all, in frame order.
    shuffled = [candidate(index) for index in (7, 2, 5)]
    assert [frame.frame_idx for frame in ingest.pick_evenly(shuffled, 9)] == [2, 5, 7]
    assert ingest.pick_evenly(frames, 0) == []
    assert ingest.pick_evenly([], 5) == []


def test_sample_frames_reports_clips_without_keypoints(tmp_path: pathlib.Path) -> None:
    """The sampler cannot work before the keypoints exist: reported, not raised."""
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    clip = write_clip(videos / "inbox", "WhatsApp Video 2026-10-02 at 12.00.00 PM.mp4", seed=16)
    clip_id = catalogue_module.file_clip_id(clip)
    write_hold_summary(
        data / "features" / "mediapipe" / "hold_summary.csv",
        [{**LINE_HOLD, "clip_id": clip_id}],
    )

    report = run(data, videos, sample_frames=2, today="2026-10-02")

    assert "no keypoints" in report.failures["sample_frames"][clip_id]
    assert not report.ok
    assert "sampled frames: 0 of 2/clip requested" in report.text
    assert not (data / "label_frames").exists()


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


def test_credentials_are_read_from_the_env_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "label-studio.env"
    path.write_text(
        '# where the API login lives\nLS_USER = someone\nLS_PASSWORD="p4ss"\n\n',
        encoding="utf-8",
    )
    assert ingest.read_ls_credentials(path) == ("someone", "p4ss")

    missing = tmp_path / "nope.env"
    with pytest.raises(ingest.LabelStudioError) as error:
        ingest.read_ls_credentials(missing)
    assert "LS_USER" in str(error.value)
    assert "LS_PASSWORD" in str(error.value)

    incomplete = tmp_path / "incomplete.env"
    incomplete.write_text("LS_USER=only-a-user\n", encoding="utf-8")
    with pytest.raises(ingest.LabelStudioError):
        ingest.read_ls_credentials(incomplete)
