"""Tests for :mod:`handstand.catalogue`: filename parsing, rotation, merge, one real clip.

No test here reads the real videos in the workspace. The merge tests probe
synthetic one-pixel files through a fake ``probe``; the single integration test
generates its own clip with ffmpeg into ``tmp_path``.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

from handstand import catalogue
from handstand.catalogue import FilenameStamp

# --------------------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4",
            FilenameStamp("2026-07-22", "2026-07-22T19:09:23"),
        ),
        (
            "WhatsApp Video 2026-07-14 at 10.13.39 AM.mp4",
            FilenameStamp("2026-07-14", "2026-07-14T10:13:39"),
        ),
        # 12-hour clock edge cases: midnight and noon are the ones that go wrong.
        ("clip 2026-01-02 at 12.00.00 AM.mp4", FilenameStamp("2026-01-02", "2026-01-02T00:00:00")),
        ("clip 2026-01-02 at 12.30.00 AM.mp4", FilenameStamp("2026-01-02", "2026-01-02T00:30:00")),
        ("clip 2026-01-02 at 12.02.37 PM.mp4", FilenameStamp("2026-01-02", "2026-01-02T12:02:37")),
        ("clip 2026-01-02 at 12.59.59 PM.mp4", FilenameStamp("2026-01-02", "2026-01-02T12:59:59")),
        ("clip 2026-01-02 at 11.59.59 PM.mp4", FilenameStamp("2026-01-02", "2026-01-02T23:59:59")),
        # The (1) suffix iOS appends to duplicate downloads does not disturb the stamp.
        (
            "WhatsApp Video 2026-07-22 at 7.09.23 PM (1).mp4",
            FilenameStamp("2026-07-22", "2026-07-22T19:09:23"),
        ),
        (
            "WhatsApp Video 2026-07-22 at 7.09.23 PM (12).mp4",
            FilenameStamp("2026-07-22", "2026-07-22T19:09:23"),
        ),
        # Leading zeroes and a two-digit hour.
        ("VID_20260722_070923.mp4", FilenameStamp("", "")),
        ("clip 2026-07-22 at 07.09.23 PM.mp4", FilenameStamp("2026-07-22", "2026-07-22T19:09:23")),
        # No match, or an impossible clock reading: empty, never a guess.
        ("IMG_4821.mp4", FilenameStamp("", "")),
        ("handstand practice.mp4", FilenameStamp("", "")),
        ("2026-07-22.mp4", FilenameStamp("", "")),
        ("clip 2026-07-22 at 7.09 PM.mp4", FilenameStamp("", "")),
        ("clip 2026-07-22 at 7.09.23.mp4", FilenameStamp("", "")),
        ("clip 2026-07-22 at 13.09.23 PM.mp4", FilenameStamp("", "")),
        ("clip 2026-07-22 at 0.09.23 AM.mp4", FilenameStamp("", "")),
        ("clip 2026-02-30 at 7.09.23 PM.mp4", FilenameStamp("", "")),
        ("clip 2026-13-01 at 7.09.23 PM.mp4", FilenameStamp("", "")),
    ],
)
def test_parse_filename(filename: str, expected: FilenameStamp) -> None:
    assert catalogue.parse_filename(filename) == expected


def test_parse_filename_ignores_a_directory_component() -> None:
    assert catalogue.parse_filename("/videos/WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4") == (
        FilenameStamp("2026-07-22", "2026-07-22T19:09:23")
    )


def test_parse_filename_separates_date_and_datetime() -> None:
    stamp = catalogue.parse_filename("WhatsApp Video 2026-09-21 at 1.26.59 PM.mp4")
    assert stamp.session_date == "2026-09-21"
    assert stamp.recorded_at == "2026-09-21T13:26:59"
    assert stamp.recorded_at.startswith(stamp.session_date)


# --------------------------------------------------------------------------------------
# Rotation -> display dimensions
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (0, (1024, 576)),
        (90, (576, 1024)),
        (-90, (576, 1024)),
        (180, (1024, 576)),
        (-180, (1024, 576)),
        (270, (576, 1024)),
        (-270, (576, 1024)),
    ],
)
def test_display_dimensions(rotation: int, expected: tuple[int, int]) -> None:
    assert catalogue.display_dimensions(1024, 576, rotation) == expected


def test_display_dimensions_of_an_already_portrait_clip() -> None:
    assert catalogue.display_dimensions(1080, 1920, 0) == (1080, 1920)
    assert catalogue.display_dimensions(1080, 1920, 180) == (1080, 1920)


# --------------------------------------------------------------------------------------
# Frame rate
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30/1", 30.0),
        ("29280/979", pytest.approx(29.908069458631257)),
        ("25/1", 25.0),
        ("0/0", None),
        ("N/A", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_frame_rate(value: object, expected: object) -> None:
    assert catalogue.parse_frame_rate(value) == expected


def test_is_variable_frame_rate_uses_a_half_percent_tolerance() -> None:
    assert not catalogue.is_variable_frame_rate(30.0, 30.0)
    assert not catalogue.is_variable_frame_rate(29.9, 30.0)  # 0.33% off
    assert catalogue.is_variable_frame_rate(29.7, 30.0)  # 1.00% off
    assert catalogue.is_variable_frame_rate(30.3, 30.0)


def test_is_variable_frame_rate_of_unknown_rates() -> None:
    assert not catalogue.is_variable_frame_rate(None, 30.0)
    assert not catalogue.is_variable_frame_rate(30.0, None)
    assert not catalogue.is_variable_frame_rate(30.0, 0.0)


# --------------------------------------------------------------------------------------
# Clip ids
# --------------------------------------------------------------------------------------


def test_clip_id_is_content_addressed_and_survives_a_rename(tmp_path: pathlib.Path) -> None:
    original = tmp_path / "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4"
    original.write_bytes(b"handstand bytes")
    clip_id = catalogue.file_clip_id(original)
    renamed = tmp_path / "whatever-i-called-it.mp4"
    original.rename(renamed)

    assert len(clip_id) == catalogue.CLIP_ID_LENGTH
    assert all(char in "0123456789abcdef" for char in clip_id)
    assert catalogue.file_clip_id(renamed) == clip_id


def test_clip_id_differs_for_different_content(tmp_path: pathlib.Path) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert catalogue.file_clip_id(first) != catalogue.file_clip_id(second)


# --------------------------------------------------------------------------------------
# Fake ffprobe
# --------------------------------------------------------------------------------------


def ffprobe_report(
    *,
    width: int = 1024,
    height: int = 576,
    rotation: int | None = -90,
    avg_frame_rate: str = "29280/979",
    r_frame_rate: str = "30/1",
    duration: str = "8.158333",
    bit_rate: str = "1511497",
    codec_name: str = "h264",
) -> dict[str, Any]:
    """A slice of the real ffprobe JSON for one of the workspace clips."""
    stream: dict[str, Any] = {
        "index": 0,
        "codec_name": codec_name,
        "codec_type": "video",
        "width": width,
        "height": height,
        "avg_frame_rate": avg_frame_rate,
        "r_frame_rate": r_frame_rate,
    }
    if rotation is not None:
        stream["side_data_list"] = [
            {"side_data_type": "Display Matrix", "rotation": rotation},
        ]
    return {
        "streams": [stream],
        "format": {
            "filename": "ignored.mp4",
            "duration": duration,
            "bit_rate": bit_rate,
        },
    }


def fake_probe(
    report: Mapping[str, Any] | None = None,
    reports: Mapping[str, Mapping[str, Any]] | None = None,
) -> catalogue.Probe:
    """Return a stand-in for :func:`probe_media`.

    With ``reports`` each file gets its own report, keyed by filename; with
    ``report`` every file gets that one. With neither, the default landscape clip.
    """

    def probe(path: pathlib.Path) -> Mapping[str, Any]:
        if reports is not None:
            return reports[path.name]
        if report is not None:
            return report
        return ffprobe_report()

    return probe


# --------------------------------------------------------------------------------------
# Row building
# --------------------------------------------------------------------------------------


def test_build_row_reads_the_ffprobe_report(tmp_path: pathlib.Path) -> None:
    clip = tmp_path / "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4"
    clip.write_bytes(b"clip bytes")

    row = catalogue.build_row(clip, fake_probe())

    assert row["filename"] == "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4"
    assert row["clip_id"] == catalogue.file_clip_id(clip)
    assert row["session_date"] == "2026-07-22"
    assert row["recorded_at"] == "2026-07-22T19:09:23"
    assert row["duration_s"] == "8.158"
    assert row["fps_avg"] == "29.908"
    assert row["fps_nominal"] == "30.000"
    assert row["is_vfr"] == "false"  # 0.31% drift is under the 0.5% tolerance
    assert row["width"] == "1024"
    assert row["height"] == "576"
    assert row["rotation"] == "-90"
    assert row["display_width"] == "576"  # rotated: swapped
    assert row["display_height"] == "1024"
    assert row["bitrate_kbps"] == "1511"
    assert row["codec"] == "h264"


def test_build_row_flags_a_really_drifting_clip_as_vfr(tmp_path: pathlib.Path) -> None:
    clip = tmp_path / "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4"
    clip.write_bytes(b"clip bytes")

    row = catalogue.build_row(clip, fake_probe(ffprobe_report(avg_frame_rate="250/9")))

    assert row["is_vfr"] == "true"


def test_build_row_without_a_rotation_flag(tmp_path: pathlib.Path) -> None:
    clip = tmp_path / "landscape.mp4"
    clip.write_bytes(b"clip bytes")

    row = catalogue.build_row(clip, fake_probe(ffprobe_report(rotation=None)))

    assert row["rotation"] == "0"
    assert (row["display_width"], row["display_height"]) == ("1024", "576")


def test_build_row_uses_the_first_video_stream(tmp_path: pathlib.Path) -> None:
    clip = tmp_path / "with_audio.mp4"
    clip.write_bytes(b"clip bytes")
    report = ffprobe_report()
    report["streams"].insert(0, {"codec_type": "audio", "codec_name": "aac"})

    row = catalogue.build_row(clip, fake_probe(report))

    assert row["codec"] == "h264"
    assert row["width"] == "1024"


def test_build_row_of_an_undated_clip_leaves_the_stamp_empty(tmp_path: pathlib.Path) -> None:
    clip = tmp_path / "IMG_4821.mp4"
    clip.write_bytes(b"clip bytes")

    row = catalogue.build_row(clip, fake_probe())

    assert row["session_date"] == ""
    assert row["recorded_at"] == ""


# --------------------------------------------------------------------------------------
# Finding clips
# --------------------------------------------------------------------------------------


def test_find_clips_ignores_images_and_directories(tmp_path: pathlib.Path) -> None:
    (tmp_path / "b second.mp4").write_bytes(b"b")
    (tmp_path / "a first.mp4").write_bytes(b"a")
    (tmp_path / "WhatsApp Image 2026-08-10 at 5.51.11 PM.jpeg").write_bytes(b"jpeg")
    (tmp_path / "notes.txt").write_bytes(b"txt")
    nested = tmp_path / "nested.mp4"
    nested.mkdir()

    assert [path.name for path in catalogue.find_clips(tmp_path)] == ["a first.mp4", "b second.mp4"]


def test_find_clips_of_a_missing_directory(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        catalogue.find_clips(tmp_path / "absent")


# --------------------------------------------------------------------------------------
# Merging with a catalogue written earlier
# --------------------------------------------------------------------------------------


def write_catalogue(out: pathlib.Path, frame: pd.DataFrame) -> None:
    """Write a pre-existing catalogue the way a spreadsheet round-trip would."""
    frame.to_csv(out, index=False)


def clip_id_of(tmp_path: pathlib.Path, name: str, payload: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return catalogue.file_clip_id(path)


def is_empty(value: object) -> bool:
    """True for a blank annotation cell, however pandas chose to represent it."""
    return value is None or value == "" or pd.isna(value)


def test_merge_keeps_annotations_refreshes_metadata_and_flags_what_vanished(
    tmp_path: pathlib.Path,
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    out = tmp_path / "catalogue.csv"
    stamp = "2026-07-22 at 7.09.23 PM"
    keep = clip_id_of(videos, f"WhatsApp Video {stamp}.mp4", b"keep")
    also = clip_id_of(videos, "WhatsApp Video 2026-07-23 at 8.00.00 PM.mp4", b"also")
    gone = clip_id_of(tmp_path, "WhatsApp Video 2026-07-24 at 9.00.00 PM.mp4", b"gone")

    # A catalogue from an earlier run: stale metadata, hand-typed annotations, and the
    # `missing` column absent because nothing was missing that day.
    write_catalogue(
        out,
        pd.DataFrame(
            [
                {
                    "clip_id": keep,
                    "filename": f"WhatsApp Video {stamp}.mp4",
                    "duration_s": 99.0,
                    "codec": "",
                    "camera_angle": "low",
                    "full_body": "yes",
                    "skill": "wall",
                    "outcome": "held",
                    "hold_s": 12,
                    "good_clip": "y",
                    "notes": "best of the session",
                },
                {
                    "clip_id": also,
                    "filename": "WhatsApp Video 2026-07-23 at 8.00.00 PM.mp4",
                    "camera_angle": "front",
                    "notes": "",
                },
                {
                    "clip_id": gone,
                    "filename": "WhatsApp Video 2026-07-24 at 9.00.00 PM.mp4",
                    "camera_angle": "back",
                    "notes": "clip on the deleted phone",
                },
            ]
        ),
    )

    result = catalogue.build_catalogue(videos, out, probe=fake_probe())
    assert result.failures == []

    frame = pd.read_csv(out)
    assert list(frame.columns) == list(catalogue.COLUMNS)
    by_id = {row["clip_id"]: row for _, row in frame.iterrows()}
    assert set(by_id) == {keep, also, gone}
    assert len(frame) == 3

    # Annotations survive the re-run...
    kept = by_id[keep]
    assert kept["camera_angle"] == "low"
    assert kept["full_body"] == "yes"
    assert kept["skill"] == "wall"
    assert kept["outcome"] == "held"
    assert kept["hold_s"] == 12
    assert kept["good_clip"] == "y"
    assert kept["notes"] == "best of the session"
    assert by_id[also]["camera_angle"] == "front"
    assert is_empty(by_id[also]["notes"])

    # ...while the metadata is refreshed from the files themselves.
    assert float(kept["duration_s"]) == pytest.approx(8.158)
    assert kept["codec"] == "h264"
    assert float(kept["fps_nominal"]) == 30.0
    assert int(kept["rotation"]) == -90
    assert int(kept["display_width"]) == 576
    assert int(kept["display_height"]) == 1024

    # The deleted clip keeps its row and its notes, flagged as missing.
    vanished = by_id[gone]
    assert str(vanished["missing"]).lower() == "true"
    assert vanished["notes"] == "clip on the deleted phone"
    assert vanished["camera_angle"] == "back"

    # Clips that are really there are not flagged.
    assert str(kept["missing"]).lower() == "false"
    assert str(by_id[also]["missing"]).lower() == "false"

    # A new clip appears with empty annotation cells and the full column set.
    fresh = videos / "WhatsApp Video 2026-07-25 at 6.15.00 PM.mp4"
    fresh.write_bytes(b"brand new")
    catalogue.build_catalogue(videos, out, probe=fake_probe())
    frame = pd.read_csv(out)
    new_row = frame[frame["filename"] == fresh.name].iloc[0]
    assert new_row["clip_id"] == catalogue.file_clip_id(fresh)
    assert str(new_row["missing"]).lower() == "false"
    for column in catalogue.ANNOTATION_COLUMNS:
        assert is_empty(new_row[column]), column
    assert set(frame["filename"]) == {
        f"WhatsApp Video {stamp}.mp4",
        "WhatsApp Video 2026-07-23 at 8.00.00 PM.mp4",
        "WhatsApp Video 2026-07-24 at 9.00.00 PM.mp4",
        fresh.name,
    }


def test_merge_adds_clips_when_no_catalogue_exists_yet(tmp_path: pathlib.Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    out = tmp_path / "nested" / "catalogue.csv"
    clip_id_of(videos, "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4", b"a")
    clip_id_of(videos, "WhatsApp Video 2026-07-22 at 7.09.24 PM.mp4", b"b")

    result = catalogue.build_catalogue(videos, out, probe=fake_probe())

    assert out.exists()  # the data dir is created on demand
    assert len(result.rows) == 2
    for row in result.rows:
        assert all(row[column] == "" for column in catalogue.ANNOTATION_COLUMNS)
        assert row["missing"] == "false"


def test_merge_keeps_a_row_whose_clip_id_was_lost(tmp_path: pathlib.Path) -> None:
    fresh = [catalogue.normalise_row({"clip_id": "abc", "filename": "a.mp4"})]

    merged = catalogue.merge_rows(fresh, [{"filename": "orphan.mp4", "notes": "typed by hand"}])

    assert [row["filename"] for row in merged] == ["a.mp4", "orphan.mp4"]
    assert merged[1]["notes"] == "typed by hand"
    assert merged[1]["missing"] == "true"


def test_merge_of_nothing_is_nothing() -> None:
    assert catalogue.merge_rows([], []) == []


def test_rows_sort_by_recording_time_then_filename(tmp_path: pathlib.Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in (
        "WhatsApp Video 2026-07-22 at 7.09.24 PM.mp4",
        "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4",
        "WhatsApp Video 2026-07-22 at 7.09.23 AM.mp4",
        "IMG_4821.mp4",
    ):
        (videos / name).write_bytes(name.encode())

    out = tmp_path / "catalogue.csv"
    catalogue.build_catalogue(videos, out, probe=fake_probe())

    frame = pd.read_csv(out)
    # Undated first (empty timestamp), then chronological; the two 7.09.23 PM files
    # are told apart by filename.
    assert list(frame["filename"]) == [
        "IMG_4821.mp4",
        "WhatsApp Video 2026-07-22 at 7.09.23 AM.mp4",
        "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4",
        "WhatsApp Video 2026-07-22 at 7.09.24 PM.mp4",
    ]


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def test_write_rows_uses_the_full_header_even_for_one_row(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "catalogue.csv"
    row = catalogue.normalise_row({"clip_id": "abc", "filename": "a.mp4", "notes": "hi"})

    catalogue.write_rows([row], out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(catalogue.COLUMNS)
    assert len(lines) == 2
    assert catalogue.read_rows(out) == [row]


def test_write_rows_leaves_no_temporary_file_behind(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "catalogue.csv"

    catalogue.write_rows([catalogue.normalise_row({"clip_id": "abc"})], out)

    assert [path.name for path in tmp_path.iterdir()] == ["catalogue.csv"]


def test_write_rows_keeps_the_previous_catalogue_when_a_row_is_bad(
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "catalogue.csv"
    catalogue.write_rows([catalogue.normalise_row({"clip_id": "abc", "notes": "typed"})], out)

    with pytest.raises(ValueError, match="fieldnames"):
        catalogue.write_rows([{"clip_id": "abc", "stray": "column"}], out)

    assert [path.name for path in tmp_path.iterdir()] == ["catalogue.csv"]
    assert pd.read_csv(out)["notes"].iloc[0] == "typed"


def test_write_rows_leaves_a_readable_catalogue(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "catalogue.csv"

    catalogue.write_rows([catalogue.normalise_row({"clip_id": "abc"})], out)

    assert out.stat().st_mode & 0o777 == catalogue.FILE_MODE


def test_read_rows_of_a_missing_catalogue(tmp_path: pathlib.Path) -> None:
    assert catalogue.read_rows(tmp_path / "absent.csv") == []


# --------------------------------------------------------------------------------------
# A clip ffprobe cannot read
# --------------------------------------------------------------------------------------


def test_a_clip_ffprobe_cannot_read_still_gets_a_row(tmp_path: pathlib.Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    good = clip_id_of(videos, "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4", b"good")
    bad = clip_id_of(videos, "WhatsApp Video 2026-07-23 at 7.09.23 PM.mp4", b"corrupt")

    def probe(path: pathlib.Path) -> Mapping[str, Any]:
        if path.name.startswith("WhatsApp Video 2026-07-23"):
            raise subprocess.CalledProcessError(1, "ffprobe")
        return ffprobe_report()

    result = catalogue.build_catalogue(videos, tmp_path / "catalogue.csv", probe=probe)

    assert [name for name, _ in result.failures] == ["WhatsApp Video 2026-07-23 at 7.09.23 PM.mp4"]
    rows = {row["clip_id"]: row for row in result.rows}
    assert rows[good]["codec"] == "h264"
    assert rows[bad]["filename"] == "WhatsApp Video 2026-07-23 at 7.09.23 PM.mp4"
    assert rows[bad]["recorded_at"] == "2026-07-23T19:09:23"
    assert rows[bad]["codec"] == ""
    assert rows[bad]["missing"] == "false"


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def test_main_honours_videos_and_out_flags(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    clip_id_of(videos, "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4", b"a")
    out = tmp_path / "elsewhere" / "clips.csv"
    monkeypatch.setattr(catalogue, "probe_media", fake_probe())

    assert catalogue.main(["--videos", str(videos), "--out", str(out)]) == 0

    assert len(pd.read_csv(out)) == 1
    printed = capsys.readouterr().out
    assert "wrote 1 rows" in printed
    assert "variable frame rate: 0" in printed
    assert "rotated: 1" in printed


def test_main_defaults_to_the_workspace_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    clip_id_of(videos, "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4", b"a")
    monkeypatch.setenv("HANDSTAND_VIDEOS", str(videos))
    monkeypatch.setenv("HANDSTAND_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(catalogue, "probe_media", fake_probe())

    assert catalogue.main([]) == 0

    assert (tmp_path / "data" / "catalogue.csv").exists()
    assert str(tmp_path / "videos") in capsys.readouterr().out


def test_main_reports_a_missing_videos_directory_without_a_traceback(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "catalogue.csv"

    assert catalogue.main(["--videos", str(tmp_path / "absent"), "--out", str(out)]) == 2

    assert not out.exists()
    assert str(tmp_path / "absent") in capsys.readouterr().err


def test_summarise_survives_a_hand_edited_rotation_cell() -> None:
    rows = [
        catalogue.normalise_row({"clip_id": "a", "filename": "a.mp4", "rotation": "-90"}),
        catalogue.normalise_row({"clip_id": "b", "filename": "b.mp4", "rotation": "sideways"}),
    ]

    summary = catalogue.summarise(catalogue.Catalogue(rows=rows, failures=[]))

    assert "rotated: 1" in summary


# --------------------------------------------------------------------------------------
# Integration: a real clip, probed by the real ffprobe
# --------------------------------------------------------------------------------------

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg/ffprobe are not installed",
)


@requires_ffmpeg
def test_catalogue_a_generated_clip_end_to_end(tmp_path: pathlib.Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    clip = videos / "WhatsApp Video 2026-07-22 at 7.09.23 PM.mp4"
    subprocess.run(
        [
            str(FFMPEG),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    # A still image in the same folder must not become a row.
    (videos / "WhatsApp Image 2026-08-10 at 5.51.11 PM.jpeg").write_bytes(b"jpeg")
    out = tmp_path / "data" / "catalogue.csv"

    result = catalogue.build_catalogue(videos, out)
    assert result.failures == []
    assert len(result.rows) == 1

    row = result.rows[0]
    assert row["clip_id"] == catalogue.file_clip_id(clip)
    assert len(row["clip_id"]) == 12
    assert row["recorded_at"] == "2026-07-22T19:09:23"
    assert row["duration_s"] == "1.000"
    assert row["fps_avg"] == "10.000"
    assert row["fps_nominal"] == "10.000"
    assert row["is_vfr"] == "false"
    assert (row["width"], row["height"]) == ("64", "48")
    assert (row["display_width"], row["display_height"]) == ("64", "48")
    assert row["rotation"] == "0"
    assert row["codec"] == "h264"
    assert int(row["bitrate_kbps"]) > 0
    assert all(row[column] == "" for column in catalogue.ANNOTATION_COLUMNS)

    # A re-run with the same folder is idempotent, and the sheet reopens cleanly.
    first = out.read_bytes()
    catalogue.build_catalogue(videos, out)
    assert out.read_bytes() == first

    frame = pd.read_csv(out)
    assert list(frame.columns) == list(catalogue.COLUMNS)
    assert all(isinstance(value, str) for value in row.values())
