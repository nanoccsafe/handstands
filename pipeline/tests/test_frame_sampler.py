"""Tests for :mod:`handstand.frame_sampler`.

Nothing here touches the real handstand videos: the data directory is built in
``tmp_path`` out of tiny synthetic parquets in the documented schema, and both
the video decoder and the JPEG writer are stubs, so the test asserts the *sampling*
and never a codec.
"""

from __future__ import annotations

import collections
import csv
import pathlib
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import pytest

from handstand import frame_sampler
from handstand.frame_sampler import MANIFEST_COLUMNS, STRATA, FrameCandidate, image_name
from handstand.pose_mediapipe import JOINT_INDEX, JOINT_NAMES, FramePacket

#: A small display size, in the shape the real clips are: portrait.
WIDTH = 64
HEIGHT = 96
#: One frame every 33 ms, the rate the WhatsApp clips run at.
FRAME_MS = 33
#: Frames per synthetic clip: ~4 s, long enough for four time bins.
CLIP_FRAMES = 120


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #


def synthetic_table(
    *,
    frames: int = CLIP_FRAMES,
    contact_from: int = 0,
    contact_to: int = 0,
    inverted: bool = True,
    detected: bool = True,
) -> pd.DataFrame:
    """A keypoint parquet in the athlete schema: 33 rows per frame.

    ``inverted`` puts the wrists lower in the image than the ankles, which is
    what MediaPipe's own rule calls a hold; the kick-up case is the other way
    round. ``contact_from``/``contact_to`` mark a half-open frame range as
    trainer contact.
    """
    rows = []
    for frame_idx in range(frames):
        t_ms = frame_idx * FRAME_MS
        contact = contact_from <= frame_idx < contact_to
        for joint in JOINT_NAMES:
            y = 10.0
            if joint in ("left_wrist", "right_wrist"):
                y = 80.0 if inverted else 20.0
            elif joint in ("left_ankle", "right_ankle"):
                y = 20.0 if inverted else 80.0
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "t_ms": t_ms,
                    "joint": joint,
                    "x": float(JOINT_INDEX[joint]),
                    "y": y,
                    "z": 0.0,
                    "visibility": 0.9,
                    "presence": 0.9,
                    "rotated": False,
                    "detected": detected,
                    "athlete_score": 0.9,
                    "n_people": 2 if contact else 1,
                    "trainer_contact": contact,
                    "contact_reason": "box_iou" if contact else "",
                }
            )
    return pd.DataFrame(rows)


def write_parquet(path: pathlib.Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)


def video_name(clip_id: str) -> str:
    """The file name the catalogue gives a clip, and so the one in the videos dir."""
    return f"WhatsApp Video {clip_id}.mp4"


def make_catalogue(
    path: pathlib.Path,
    rows: Sequence[tuple[str, str, str]],
) -> None:
    """Write a catalogue: ``(clip_id, session_date, skill)`` per clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "filename",
                "session_date",
                "display_width",
                "display_height",
                "skill",
                "missing",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for clip_id, session_date, skill in rows:
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "filename": video_name(clip_id),
                    "session_date": session_date,
                    "display_width": WIDTH,
                    "display_height": HEIGHT,
                    "skill": skill,
                    "missing": "false",
                }
            )


def make_trainer_report(path: pathlib.Path, rows: Sequence[tuple[str, bool]]) -> None:
    """Write a trainer report: ``(clip_id, trainer_present)`` per clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["clip_id", "trainer_present", "pct_trainer_contact"],
            lineterminator="\n",
        )
        writer.writeheader()
        for clip_id, trainer_present in rows:
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "trainer_present": "true" if trainer_present else "false",
                    "pct_trainer_contact": "0.0",
                }
            )


def make_videos(directory: pathlib.Path, clip_ids: Sequence[str]) -> None:
    """Touch a file per clip: the resolver only ever stats them, never decodes."""
    directory.mkdir(parents=True, exist_ok=True)
    for clip_id in clip_ids:
        (directory / video_name(clip_id)).write_bytes(b"not a real video")


# --------------------------------------------------------------------------- #
# Stubs for the two IO seams
# --------------------------------------------------------------------------- #


class StubVideo:
    """What :class:`handstand.pose_mediapipe.DisplayVideo` exposes, minus the codec."""

    def __init__(self, path: pathlib.Path, frames: int, width: int = WIDTH, height: int = HEIGHT):
        self.path = path
        self.frames = frames
        self.display_width = width
        self.display_height = height

    def __iter__(self) -> Iterator[FramePacket]:
        for frame_idx in range(self.frames):
            yield FramePacket(
                frame_idx=frame_idx,
                t_ms=frame_idx * FRAME_MS,
                frame=np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8),
            )

    def __enter__(self) -> StubVideo:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def stub_writer() -> tuple[list[pathlib.Path], object]:
    """A frame writer that records what it was asked to write and writes 3 bytes."""
    written: list[pathlib.Path] = []

    def write(frame: np.ndarray, path: pathlib.Path) -> None:
        written.append(pathlib.Path(path))
        pathlib.Path(path).write_bytes(b"\xff\xd8\xff")

    return written, write


def opener_for(frames: int = CLIP_FRAMES, width: int = WIDTH, height: int = HEIGHT):
    """An ``open_video`` stub decoding a fixed number of upright frames."""

    def open_video(path: object) -> StubVideo:
        return StubVideo(pathlib.Path(str(path)), frames, width, height)  # type: ignore[arg-type]

    return open_video


def resolver_for(videos: pathlib.Path, missing: Sequence[str] = ()):
    """A ``clip_id`` -> path resolver with per-clip clips that have no video."""

    def resolve(clip_id: str) -> pathlib.Path | None:
        if clip_id in missing:
            return None
        return videos / video_name(clip_id)

    return resolve


def make_resolver_returning(videos: pathlib.Path):
    def resolve(clip_id: str) -> pathlib.Path | None:
        return videos / video_name(clip_id)

    return resolve


# --------------------------------------------------------------------------- #
# A whole synthetic workspace
# --------------------------------------------------------------------------- #


def build_workspace(
    tmp_path: pathlib.Path,
    *,
    clips: Sequence[tuple[str, str, str, bool]] = (),
    frames: int = CLIP_FRAMES,
    contact_frames: int = 0,
    inverted: bool = True,
) -> tuple[pathlib.Path, pathlib.Path]:
    """A data directory and a videos directory full of synthetic clips.

    Each clip is ``(clip_id, session_date, skill, trainer_present)``. The clip id
    is a hex-looking string, because the converter and the image names assume it.
    """
    data = tmp_path / "data"
    videos = tmp_path / "videos"
    if not clips:
        clips = tuple(
            (f"{index:012x}", f"2026-07-{(index % 20) + 1:02d}", "line", index % 2 == 0)
            for index in range(8)
        )
    make_catalogue(
        data / "catalogue.csv",
        [(clip_id, session_date, skill) for clip_id, session_date, skill, _ in clips],
    )
    make_trainer_report(
        data / "reports" / "trainer_report.csv",
        [(clip_id, trainer_present) for clip_id, _, _, trainer_present in clips],
    )
    for clip_id, _, _, _ in clips:
        write_parquet(
            data / "keypoints" / "mediapipe_athlete" / "auto" / f"{clip_id}.parquet",
            synthetic_table(frames=frames, contact_to=contact_frames, inverted=inverted),
        )
    make_videos(videos, [clip_id for clip_id, _, _, _ in clips])
    return data, videos


def sample(tmp_path: pathlib.Path, **kwargs) -> frame_sampler.SampleReport:
    """Run the sampler over a synthetic workspace with stubbed video and JPEG IO."""
    data, videos = build_workspace(tmp_path, **kwargs)
    written: list[pathlib.Path] = []

    def write(frame: np.ndarray, path: pathlib.Path) -> None:
        written.append(pathlib.Path(path))
        pathlib.Path(path).write_bytes(b"\xff\xd8\xff")

    return frame_sampler.sample_frames(
        12,
        data=data,
        videos=videos,
        out_dir=tmp_path / "out",
        resolve_video=make_resolver_returning(videos),
        open_video=opener_for(),
        write_frame=write,
    )


def candidate(
    clip_id: str,
    frame_idx: int,
    *,
    stratum: str = "clean_no_trainer",
    inverted: bool = True,
    session_date: str = "2026-07-01",
    t_ms: int | None = None,
    skill: str = "line",
    trainer_present: bool = False,
) -> FrameCandidate:
    """One hand-built candidate, so the planner can be tested without any files."""
    return FrameCandidate(
        clip_id=clip_id,
        frame_idx=frame_idx,
        t_ms=frame_idx * FRAME_MS if t_ms is None else t_ms,
        clip_t_start_ms=0,
        clip_t_end_ms=CLIP_FRAMES * FRAME_MS,
        stratum=stratum,
        inverted=inverted,
        session_date=session_date,
        skill=skill,
        trainer_present=trainer_present,
        trainer_contact=stratum == "trainer_contact",
        display_width=WIDTH,
        display_height=HEIGHT,
    )


def candidate_pool(
    *,
    clips: int = 24,
    frames: int = CLIP_FRAMES,
    inverted: Sequence[bool] | None = None,
) -> list[FrameCandidate]:
    """A pool covering all three strata, spread over ``clips`` clips and sessions.

    24 clips is what makes 60 frames reachable: the per-clip cap of four bounds
    the whole sample at ``4 * clips``.
    """
    pool: list[FrameCandidate] = []
    for index in range(clips):
        clip_id = f"{index:012x}"
        session = f"2026-07-{index + 1:02d}"
        for frame_idx in range(frames):
            # A frame belongs to exactly one stratum, as it does in a real
            # parquet: contact frames are frames, not frames *and* contact.
            if 40 <= frame_idx < 50:
                stratum = "trainer_contact"
            elif frame_idx < 20:
                stratum = "clean_trainer"
            else:
                stratum = "clean_no_trainer"
            pool.append(
                candidate(
                    clip_id,
                    frame_idx,
                    stratum=stratum,
                    trainer_present=stratum != "clean_no_trainer",
                    session_date=session,
                )
            )
    if inverted is not None:
        keep = [frame for index, frame in enumerate(pool) if inverted[index % len(inverted)]]
        pool = keep
    return pool


# --------------------------------------------------------------------------- #
# Quotas
# --------------------------------------------------------------------------- #


def test_quotas_split_n_into_the_documented_strata() -> None:
    assert frame_sampler.stratum_quotas(300) == {
        "clean_no_trainer": 180,
        "clean_trainer": 75,
        "trainer_contact": 45,
    }


def test_leftover_frames_go_to_the_largest_stratum_first() -> None:
    assert sum(frame_sampler.stratum_quotas(10).values()) == 10
    assert frame_sampler.stratum_quotas(10)["clean_no_trainer"] == 7
    assert frame_sampler.stratum_quotas(7) == {
        "clean_no_trainer": 5,
        "clean_trainer": 1,
        "trainer_contact": 1,
    }


def test_quota_of_zero_is_honoured() -> None:
    quotas = frame_sampler.stratum_quotas(10, shares=(1.0, 0.0, 0.0))
    assert quotas == {"clean_no_trainer": 10, "clean_trainer": 0, "trainer_contact": 0}
    assert frame_sampler.stratum_quotas(0) == dict.fromkeys(STRATA, 0)


def test_negative_n_is_refused() -> None:
    with pytest.raises(ValueError):
        frame_sampler.stratum_quotas(-1)


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def test_plan_fills_each_stratum_to_its_quota() -> None:
    chosen, quotas, _ = frame_sampler.plan(candidate_pool(), 60)
    counts = collections.Counter(frame.stratum for frame in chosen)
    assert len(chosen) == 60
    for stratum in STRATA:
        assert counts[stratum] == quotas[stratum]


def test_plan_never_exceeds_the_per_clip_cap_or_the_minimum_spacing() -> None:
    chosen, _, _ = frame_sampler.plan(candidate_pool(), 60, n_per_clip=4, min_spacing_s=0.5)
    per_clip = collections.Counter(frame.clip_id for frame in chosen)
    assert max(per_clip.values()) <= 4
    times: dict[str, list[int]] = collections.defaultdict(list)
    for frame in chosen:
        times[frame.clip_id].append(frame.t_ms)
    for clip_times in times.values():
        gaps = [b - a for a, b in zip(sorted(clip_times), sorted(clip_times)[1:], strict=False)]
        assert all(gap >= 500 for gap in gaps), gaps


def test_plan_spreads_over_sessions() -> None:
    pool = candidate_pool(clips=12)
    chosen, _, _ = frame_sampler.plan(pool, 48)
    sessions = {frame.session_date for frame in chosen}
    assert len(sessions) == 12


def test_plan_spreads_inside_a_clip_across_the_timeline() -> None:
    pool = candidate_pool(clips=1, frames=CLIP_FRAMES)
    chosen, _, _ = frame_sampler.plan(pool, 4, quotas={"clean_no_trainer": 4})
    times = sorted(frame.t_ms for frame in chosen)
    assert len(times) == 4
    assert all(later - earlier >= 500 for earlier, later in zip(times, times[1:], strict=False))
    # One pick per quarter of the clip, not four picks from the first second.
    for quarter in range(4):
        low = quarter * CLIP_FRAMES * FRAME_MS / 4
        high = (quarter + 1) * CLIP_FRAMES * FRAME_MS / 4
        assert sum(1 for t in times if low <= t < high) == 1, (times, quarter)


def test_plan_is_deterministic_for_a_seed() -> None:
    pool = candidate_pool()
    first, _, _ = frame_sampler.plan(pool, 60, seed=7)
    again, _, _ = frame_sampler.plan(pool, 60, seed=7)
    other, _, _ = frame_sampler.plan(pool, 60, seed=8)
    assert [frame.key for frame in first] == [frame.key for frame in again]
    assert [frame.key for frame in first] != [frame.key for frame in other]


def test_plan_keeps_at_least_seventy_percent_inverted() -> None:
    pool = candidate_pool()
    chosen, _, target = frame_sampler.plan(pool, 60, inverted_fraction=0.7)
    inverted = sum(1 for frame in chosen if frame.inverted)
    assert target == 42
    assert inverted >= 42


def test_plan_reports_the_hold_share_it_could_not_reach() -> None:
    # Every fourth frame is a hold and the planner wants all of them: it can only
    # have what the pool has, and it says so by asking for less.
    pool = [candidate("000000000000", index, inverted=index % 4 == 0) for index in range(120)]
    chosen, _, target = frame_sampler.plan(pool, 40, inverted_fraction=0.7)
    inverted = sum(1 for frame in chosen if frame.inverted)
    assert inverted <= 40
    assert inverted >= min(target, sum(1 for frame in pool if frame.inverted))


def test_swapping_for_holds_never_changes_the_strata() -> None:
    pool = candidate_pool()
    without_holds = [
        candidate(frame.clip_id, frame.frame_idx, stratum=frame.stratum, inverted=False)
        for frame in pool
    ]
    # The same pool twice, once with holds to swap in: the strata are a question
    # about difficulty and the hold share is a question about pose, so the second
    # run may only differ in how many of the same frames are holds.
    with_holds, quotas, _ = frame_sampler.plan(pool, 60, inverted_fraction=0.7)
    without, _, _ = frame_sampler.plan(without_holds, 60, inverted_fraction=0.7)
    assert collections.Counter(f.stratum for f in with_holds) == quotas
    assert collections.Counter(f.stratum for f in without) == quotas
    assert not any(f.inverted for f in without)
    assert sum(1 for f in with_holds if f.inverted) >= 42


def test_plan_of_an_empty_pool_is_empty() -> None:
    chosen, quotas, target = frame_sampler.plan([], 10)
    assert chosen == []
    assert quotas == frame_sampler.stratum_quotas(10)
    assert target == 0


def test_plan_refuses_impossible_arguments() -> None:
    pool = candidate_pool()
    for kwargs in (
        {"n_per_clip": 0},
        {"min_spacing_s": -1.0},
        {"inverted_fraction": 1.5},
    ):
        with pytest.raises(ValueError):
            frame_sampler.plan(pool, 10, **kwargs)


# --------------------------------------------------------------------------- #
# The pool: skills, walk, the empty catalogue
# --------------------------------------------------------------------------- #


def test_walk_clips_are_never_sampled(tmp_path: pathlib.Path) -> None:
    clips = (
        ("000000000001", "2026-07-01", "line", False),
        ("000000000002", "2026-07-02", "walk", True),
        ("000000000003", "2026-07-03", "line", True),
    )
    data, videos = build_workspace(tmp_path, clips=clips)
    report = sample(tmp_path, clips=clips)
    assert report.pool.clips_excluded_walk == 1
    assert {row["clip_id"] for row in report.rows} == {"000000000001", "000000000003"}


def test_only_the_requested_skills_are_sampled(tmp_path: pathlib.Path) -> None:
    clips = (
        ("000000000001", "2026-07-01", "line", False),
        ("000000000002", "2026-07-02", "press_up", False),
        ("000000000003", "2026-07-03", "line", True),
    )
    data, videos = build_workspace(tmp_path, clips=clips)
    report = sample(tmp_path, clips=clips)
    assert report.pool.skill_filter_active
    assert report.pool.clips_excluded_by_skill == 1
    assert {row["skill"] for row in report.rows} == {"line"}
    assert {row["clip_id"] for row in report.rows} == {"000000000001", "000000000003"}


def test_a_second_skill_can_be_requested(tmp_path: pathlib.Path) -> None:
    clips = (
        ("000000000001", "2026-07-01", "line", False),
        ("000000000002", "2026-07-02", "press_up", False),
    )
    build_workspace(tmp_path, clips=clips)
    report = frame_sampler.sample_frames(
        4,
        data=tmp_path / "data",
        out_dir=tmp_path / "out",
        skills=["press_up"],
        resolve_video=resolver_for(tmp_path / "videos"),
        open_video=opener_for(),
        write_frame=lambda frame, path: pathlib.Path(path).write_bytes(b"x"),
    )
    assert {row["clip_id"] for row in report.rows} == {"000000000002"}


def test_an_empty_skill_column_uses_every_clip_and_says_so(tmp_path: pathlib.Path) -> None:
    clips = (
        ("000000000001", "2026-07-01", "", False),
        ("000000000002", "2026-07-02", "", True),
    )
    report = sample(tmp_path, clips=clips)
    assert not report.pool.skill_filter_active
    assert {row["skill"] for row in report.rows} == {frame_sampler.UNLABELLED_SKILL}
    assert "OFF" in frame_sampler.summarise(report)
    assert any("no skill values" in note for note in report.pool.notes)


def test_a_partly_filled_skill_column_drops_the_blank_clips(tmp_path: pathlib.Path) -> None:
    clips = (
        ("000000000001", "2026-07-01", "line", False),
        ("000000000002", "2026-07-02", "", False),
    )
    report = sample(tmp_path, clips=clips)
    assert report.pool.skill_filter_active
    assert report.pool.clips_unlabelled_skill == 1
    assert {row["clip_id"] for row in report.rows} == {"000000000001"}


def test_a_missing_catalogue_yields_no_candidates(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data"
    (data / "keypoints" / "mediapipe_athlete" / "auto").mkdir(parents=True)
    write_parquet(
        data / "keypoints" / "mediapipe_athlete" / "auto" / "00000000000a.parquet",
        synthetic_table(frames=10),
    )
    with pytest.raises(RuntimeError, match="no frame can be sampled"):
        frame_sampler.sample_frames(4, data=data, out_dir=tmp_path / "out")


def test_clips_without_keypoints_or_video_are_counted_not_sampled(tmp_path: pathlib.Path) -> None:
    data, videos = build_workspace(
        tmp_path,
        clips=(
            ("000000000001", "2026-07-01", "line", False),
            ("000000000002", "2026-07-02", "line", False),
        ),
    )
    (data / "keypoints" / "mediapipe_athlete" / "auto" / "000000000002.parquet").unlink()
    pool = frame_sampler.build_pool(data, resolve_video=make_resolver_returning(videos))
    assert pool.clips_considered == 2
    assert pool.clips_with_keypoints == 1
    assert pool.clip_ids == ["000000000001"]


def test_a_clip_whose_video_is_missing_never_enters_the_pool(tmp_path: pathlib.Path) -> None:
    data, _ = build_workspace(
        tmp_path,
        clips=(
            ("000000000001", "2026-07-01", "line", False),
            ("000000000002", "2026-07-02", "line", False),
        ),
    )
    resolver = resolver_for(tmp_path / "videos", ["000000000002"])
    pool = frame_sampler.build_pool(data, resolve_video=resolver)
    assert pool.clip_ids == ["000000000001"]


# --------------------------------------------------------------------------- #
# Reading the parquets
# --------------------------------------------------------------------------- #


def test_strata_come_from_the_contact_flags(tmp_path: pathlib.Path) -> None:
    clips = (("000000000001", "2026-07-01", "line", True),)
    data, videos = build_workspace(tmp_path, clips=clips, frames=60, contact_frames=10)
    pool = frame_sampler.build_pool(data, resolve_video=make_resolver_returning(videos))
    assert {frame.stratum for frame in pool.candidates} == {"clean_trainer", "trainer_contact"}
    contact = [f for f in pool.candidates if f.stratum == "trainer_contact"]
    assert all(frame.trainer_contact for frame in contact)
    assert all(0 <= frame.frame_idx < 10 for frame in contact)


def test_undetected_frames_never_enter_the_pool(tmp_path: pathlib.Path) -> None:
    clips = (("000000000001", "2026-07-01", "line", False),)
    data, videos = build_workspace(tmp_path, clips=clips, frames=20)
    write_parquet(
        data / "keypoints" / "mediapipe_athlete" / "auto" / "000000000001.parquet",
        synthetic_table(frames=20, detected=False),
    )
    pool = frame_sampler.build_pool(data, resolve_video=make_resolver_returning(videos))
    assert pool.candidates == []
    assert pool.clips_considered == 1


def test_a_kick_up_frame_is_not_a_hold(tmp_path: pathlib.Path) -> None:
    clips = (("000000000001", "2026-07-01", "line", False),)
    data, videos = build_workspace(tmp_path, clips=clips, frames=10, inverted=False)
    pool = frame_sampler.build_pool(data, resolve_video=make_resolver_returning(videos))
    assert pool.candidates
    assert not any(frame.inverted for frame in pool.candidates)


def test_a_missing_trainer_report_leaves_the_trainer_strata_empty(tmp_path: pathlib.Path) -> None:
    clips = (("000000000001", "2026-07-01", "line", True),)
    data, videos = build_workspace(tmp_path, clips=clips, frames=20)
    (data / "reports" / "trainer_report.csv").unlink()
    pool = frame_sampler.build_pool(data, resolve_video=make_resolver_returning(videos))
    assert not pool.trainer_report_known
    assert {frame.stratum for frame in pool.candidates} == {"clean_no_trainer"}
    assert any("no trainer report" in note for note in pool.notes)


def test_keypoints_without_a_contact_column_are_flagged(tmp_path: pathlib.Path) -> None:
    clips = (("000000000001", "2026-07-01", "line", False),)
    data, videos = build_workspace(tmp_path, clips=clips, frames=10)
    table = synthetic_table(frames=10).drop(columns=["trainer_contact", "contact_reason"])
    write_parquet(data / "keypoints" / "mediapipe_athlete" / "auto" / "000000000001.parquet", table)
    pool = frame_sampler.build_pool(data, resolve_video=make_resolver_returning(videos))
    assert not pool.contact_column_known
    assert {frame.stratum for frame in pool.candidates} == {"clean_no_trainer"}


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def mixed_workspace(tmp_path: pathlib.Path, *, clips: int = 10, frames: int = 60):
    """A workspace shaped like the real one: contact only where a trainer is.

    Even-numbered clips have no trainer and are all clean; odd-numbered ones have
    a trainer, whose contact frames are the first second of the clip. Ten clips is
    what it takes to fill a 12-frame sample to its quota.
    """
    setup = tuple(
        (f"{index:012x}", f"2026-07-{index + 1:02d}", "line", index % 2 == 1)
        for index in range(clips)
    )
    data, videos = build_workspace(tmp_path, clips=setup, frames=frames)
    for clip_id, _, _, trainer_present in setup:
        if trainer_present:
            write_parquet(
                data / "keypoints" / "mediapipe_athlete" / "auto" / f"{clip_id}.parquet",
                synthetic_table(frames=frames, contact_to=20),
            )
    return data, videos


def test_a_run_writes_one_jpeg_and_one_manifest_row_per_frame(tmp_path: pathlib.Path) -> None:
    written: list[pathlib.Path] = []

    def write(frame: np.ndarray, path: pathlib.Path) -> None:
        written.append(pathlib.Path(path))
        pathlib.Path(path).write_bytes(b"\xff\xd8\xff")

    data, videos = mixed_workspace(tmp_path)
    report = frame_sampler.sample_frames(
        12,
        data=data,
        out_dir=tmp_path / "out",
        resolve_video=make_resolver_returning(videos),
        open_video=opener_for(frames=60),
        write_frame=write,
    )
    assert report.written == 12
    assert len(written) == 12
    assert {path.name for path in written} == {row["image"] for row in report.rows}
    for path in written:
        assert path.parent == tmp_path / "out"
        assert path.name == image_name(path.name.split("_")[0], int(path.stem.split("_")[1]))

    with report.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == MANIFEST_COLUMNS
    assert len(rows) == 12
    assert {int(row["display_width"]) for row in rows} == {WIDTH}
    assert {int(row["display_height"]) for row in rows} == {HEIGHT}
    assert {row["stratum"] for row in rows} == set(STRATA)
    assert {row["trainer_present"] for row in rows} == {"true", "false"}
    assert {row["trainer_contact"] for row in rows} == {"true", "false"}
    assert {int(row["frame_idx"]) for row in rows} >= {6, 22, 38}
    assert sum(1 for row in rows if row["trainer_contact"] == "true") > 0
    assert report.inverted >= report.target_inverted
    assert report.inverted_percent >= 70.0


def test_the_manifest_timestamps_come_from_the_parquet(tmp_path: pathlib.Path) -> None:
    data, videos = mixed_workspace(tmp_path)
    report = frame_sampler.sample_frames(
        6,
        data=data,
        out_dir=tmp_path / "out",
        resolve_video=make_resolver_returning(videos),
        open_video=opener_for(frames=60),
        write_frame=lambda frame, path: pathlib.Path(path).write_bytes(b"x"),
    )
    assert report.written
    for row in report.rows:
        assert int(row["t_ms"]) == int(row["frame_idx"]) * FRAME_MS


def test_a_clip_that_cannot_be_decoded_is_reported_and_dropped(tmp_path: pathlib.Path) -> None:
    clips = (
        ("000000000000", "2026-07-01", "line", False),
        ("000000000001", "2026-07-02", "line", True),
    )
    data, videos = build_workspace(tmp_path, clips=clips, frames=60)

    def open_video(path: object) -> StubVideo:
        if "000000000001" in pathlib.Path(str(path)).name:  # type: ignore[arg-type]
            raise OSError("decoder exploded")
        return StubVideo(pathlib.Path(str(path)), frames=60)  # type: ignore[arg-type]

    report = frame_sampler.sample_frames(
        12,
        data=data,
        out_dir=tmp_path / "out",
        resolve_video=make_resolver_returning(videos),
        open_video=open_video,
        write_frame=lambda frame, path: pathlib.Path(path).write_bytes(b"x"),
    )
    # Only the clip that decoded is in the manifest, and the other one is named.
    assert [clip_id for clip_id, _ in report.failures] == ["000000000001"]
    assert "OSError" in report.failures[0][1]
    assert {row["clip_id"] for row in report.rows} == {"000000000000"}
    assert report.written == len(report.rows) < report.requested
    assert "fail clip 000000000001" in frame_sampler.summarise(report)


def test_a_frame_the_video_never_reaches_is_a_failure(tmp_path: pathlib.Path) -> None:
    clips = (("000000000001", "2026-07-01", "line", False),)
    data, videos = build_workspace(tmp_path, clips=clips, frames=40)
    report = frame_sampler.sample_frames(
        4,
        data=data,
        out_dir=tmp_path / "out",
        resolve_video=make_resolver_returning(videos),
        open_video=opener_for(frames=5),  # the clip is 40 frames in the parquet
        write_frame=lambda frame, path: pathlib.Path(path).write_bytes(b"x"),
    )
    assert report.written == 0
    assert report.failures and report.failures[0][0] == "000000000001"
    assert "ends before frame" in report.failures[0][1]


def test_every_manifest_row_points_at_a_written_jpeg(tmp_path: pathlib.Path) -> None:
    data, videos = mixed_workspace(tmp_path)
    report = frame_sampler.sample_frames(
        12,
        data=data,
        out_dir=tmp_path / "out",
        resolve_video=make_resolver_returning(videos),
        open_video=opener_for(frames=60),
        write_frame=lambda frame, path: pathlib.Path(path).write_bytes(b"x"),
    )
    assert len(report.rows) == 12
    for row in report.rows:
        assert (report.out_dir / row["image"]).is_file()


def test_stale_images_of_an_earlier_run_are_reported_not_deleted(
    tmp_path: pathlib.Path,
) -> None:
    data, videos = mixed_workspace(tmp_path)
    out = tmp_path / "out"
    out.mkdir(parents=True)
    old = out / "000000000000_1.jpg"
    old.write_bytes(b"from an earlier, larger run")

    def run(n: int) -> frame_sampler.SampleReport:
        return frame_sampler.sample_frames(
            n,
            data=data,
            out_dir=out,
            resolve_video=make_resolver_returning(videos),
            open_video=opener_for(frames=60),
            write_frame=lambda frame, path: pathlib.Path(path).write_bytes(b"x"),
        )

    report = run(12)
    assert report.stale_images == ["000000000000_1.jpg"]
    assert old.is_file()  # reported, never removed
    assert "earlier run" in frame_sampler.summarise(report)
    assert run(12).stale_images == ["000000000000_1.jpg"]


def test_the_manifest_is_rewritten_not_appended(tmp_path: pathlib.Path) -> None:
    data, videos = mixed_workspace(tmp_path)

    def run() -> frame_sampler.SampleReport:
        return frame_sampler.sample_frames(
            12,
            data=data,
            out_dir=tmp_path / "out",
            resolve_video=make_resolver_returning(videos),
            open_video=opener_for(frames=60),
            write_frame=lambda frame, path: pathlib.Path(path).write_bytes(b"x"),
        )

    first, again = run(), run()
    assert first.manifest == again.manifest
    assert first.rows == again.rows
    assert again.manifest.read_text(encoding="utf-8").count("\n") == 13  # header plus 12 rows


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def test_main_writes_the_frames_and_prints_the_summary(
    tmp_path: pathlib.Path, capsys, monkeypatch
) -> None:
    data, videos = mixed_workspace(tmp_path)
    monkeypatch.setattr(
        frame_sampler, "DisplayVideo", lambda path: StubVideo(pathlib.Path(str(path)), 60)
    )
    monkeypatch.setattr(
        frame_sampler, "write_jpeg", lambda frame, path: pathlib.Path(path).write_bytes(b"x")
    )
    code = frame_sampler.main(
        [
            "--n",
            "12",
            "--data",
            str(data),
            "--videos",
            str(videos),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "wrote 12 of 12 requested frames" in out
    assert "strata: clean_no_trainer=8/8 clean_trainer=3/3 trainer_contact=1/1" in out
    assert "skill filter: active (line)" in out
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 12


def test_main_reports_a_directory_without_keypoints(tmp_path: pathlib.Path, capsys) -> None:
    (tmp_path / "data").mkdir()
    assert frame_sampler.main(["--n", "4", "--data", str(tmp_path / "data")]) == 2
    assert "no frame can be sampled" in capsys.readouterr().err


def test_main_rejects_a_negative_n() -> None:
    with pytest.raises(SystemExit):
        frame_sampler.main(["--n", "-1"])
