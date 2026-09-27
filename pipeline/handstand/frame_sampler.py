"""Pick a stratified sample of frames for a human to label joints on.

The pose-model bake-off needs ground truth: frames a person has marked the
athlete's joints on by hand, so MediaPipe (and later Apple Vision, YOLO-pose)
can be scored against them. This module chooses *which* frames to show the
labeler, writes them as JPEGs and leaves a manifest saying exactly what each one
is::

    <data_dir>/label_frames/<clip_id>_<frame_idx>.jpg
    <data_dir>/label_frames/manifest.csv

Every JPEG is decoded with :class:`handstand.pose_mediapipe.DisplayVideo`, so it
is upright (display orientation) and its ``frame_idx`` is the same one the
keypoint parquets index, which means a hand-labelled point and a MediaPipe point
on the same frame live in the same pixel coordinates without any conversion.

What gets sampled
-----------------

The clip pool comes from ``<data_dir>/catalogue.csv``: only clips whose ``skill``
is one of ``--skills`` (``line`` by default) are used, and ``walk`` is *always*
excluded — a walking clip has no handstand in it to measure. The skill column is
filled in by hand (#64), so while it is still empty **every** clip is used and
the manifest says so: the ``skill`` column then reads :data:`UNLABELLED_SKILL`.
The same happens when the catalogue is missing altogether.

The frames themselves come from the athlete keypoints
(``<data_dir>/keypoints/mediapipe_athlete/<mode>/<clip_id>.parquet``, #68/#70),
which carry a per-frame ``detected`` and ``trainer_contact`` flag, and from
``<data_dir>/reports/trainer_report.csv`` (#69/#70), which says per clip whether
a trainer was ever there. Those two give three strata:

``clean_no_trainer``
    A detected frame, no contact, from a clip with no trainer in it. The easy
    case: the whole body is visible and nothing is touching it. 60 %.
``clean_trainer``
    A detected, contact-free frame from a clip where a trainer *was* present.
    The body is still clear, but the labeler has to ignore the second person. 25 %.
``trainer_contact``
    A frame the athlete selection flagged as contact — the hard case, where the
    joints a person would place are exactly the joints in dispute. 15 %.

Within a stratum the pick is spread rather than sampled at random: clips are
taken round-robin over session dates, so every session day contributes, and
inside a clip the frames come from equal-width time bins across the whole clip
(at most :data:`N_PER_CLIP` of them, none closer than :data:`MIN_SPACING_S` to
another pick from the same clip). The result covers the dataset instead of a few
clips sampled over and over.

Finally at least :data:`DEFAULT_INVERTED_FRACTION` of the frames have the mean
wrist y *below* the mean ankle y in the image (an actual hold); the rest are
kick-ups and exits. The rule is MediaPipe's own, via
:func:`handstand.pose_mediapipe.is_inverted`, so "inverted" means here exactly
what it means everywhere else in the pipeline. When the strata leave too few
holds, non-hold frames are swapped for hold frames of the same stratum.

Everything is deterministic: the same ``--seed`` picks the same frames, so a
re-run after new keypoints can be compared like for like.

CLI::

    cd pipeline
    uv run python -m handstand.frame_sampler --n 300
    uv run python -m handstand.frame_sampler --n 60 --skills line press_up
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import pathlib
import random
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import pandas as pd

from handstand.catalogue import MISSING_COLUMN
from handstand.overlay import SOURCES, keypoints_root, source_dirname, video_for_clip
from handstand.paths import data_dir
from handstand.pose_mediapipe import JOINT_NAMES, ROTATE_MODES, DisplayVideo, is_inverted

__all__ = [
    "CONTACT_COLUMN",
    "DEFAULT_INVERTED_FRACTION",
    "DEFAULT_N",
    "DEFAULT_ROTATE",
    "DEFAULT_SEED",
    "DEFAULT_SKILLS",
    "DEFAULT_SOURCE",
    "DETECTED_COLUMN",
    "EXCLUDED_SKILL",
    "FILE_MODE",
    "INVERTED_COLUMN",
    "JPEG_QUALITY",
    "LABEL_FRAMES_DIRNAME",
    "MANIFEST_COLUMNS",
    "MANIFEST_NAME",
    "MIN_SPACING_S",
    "N_PER_CLIP",
    "STRATA",
    "STRATUM_SHARES",
    "TRAINER_REPORT_NAME",
    "UNLABELLED_SKILL",
    "ClipInfo",
    "FrameCandidate",
    "Pool",
    "SampleReport",
    "build_arg_parser",
    "build_pool",
    "image_name",
    "label_frames_dir",
    "load_clip_frames",
    "main",
    "manifest_path",
    "normalise_skill",
    "plan",
    "read_catalogue",
    "read_trainer_present",
    "sample_frames",
    "stratum_of",
    "stratum_quotas",
    "summarise",
    "write_frames",
    "write_jpeg",
    "write_manifest",
]

#: Name of the output directory inside the data directory.
LABEL_FRAMES_DIRNAME = "label_frames"
#: Name of the manifest inside :data:`LABEL_FRAMES_DIRNAME`.
MANIFEST_NAME = "manifest.csv"

#: The two reports this module reads out of ``<data_dir>/reports``.
REPORTS_DIRNAME = "reports"
TRAINER_REPORT_NAME = "trainer_report.csv"
#: The catalogue, whose skill column decides the clip pool.
CATALOGUE_NAME = "catalogue.csv"

#: The three strata, in the order they are reported. See the module docstring.
STRATA: tuple[str, ...] = ("clean_no_trainer", "clean_trainer", "trainer_contact")

#: Default share of ``--n`` per stratum, in :data:`STRATA` order.
STRATUM_SHARES: tuple[float, float, float] = (0.60, 0.25, 0.15)

#: Frames a clip may contribute, so one long clip cannot fill the whole sample.
N_PER_CLIP = 4
#: Two picks from the same clip are at least this far apart.
MIN_SPACING_S = 0.5
#: Share of the sample that must be a hold (wrists below the ankles in the image).
DEFAULT_INVERTED_FRACTION = 0.70

#: Frames to sample by default.
DEFAULT_N = 300
#: Seed of the sampler; the same seed always picks the same frames.
DEFAULT_SEED = 0
#: Skills sampled by default.
DEFAULT_SKILLS: tuple[str, ...] = ("line",)
#: Keypoints sampled by default. The athlete parquets are the ones carrying the
#: ``trainer_contact`` flag the third stratum needs.
DEFAULT_SOURCE = "athlete"
DEFAULT_ROTATE = "auto"

#: The skill that is never sampled, whatever ``--skills`` says.
EXCLUDED_SKILL = "walk"

#: What the manifest's ``skill`` column says while the catalogue has no skills.
UNLABELLED_SKILL = "unlabelled"

#: Columns read from a keypoint parquet.
DETECTED_COLUMN = "detected"
CONTACT_COLUMN = "trainer_contact"
#: Written to the manifest as ``inverted``; kept separate from the parquet's
#: ``rotated`` flag, which says whether the *model* saw a rotated frame.
INVERTED_COLUMN = "inverted"

#: Quality of the written JPEGs. High: these are looked at closely, joint by joint.
JPEG_QUALITY = 90
#: The frames are shared, human-watched files, so they are not left private.
FILE_MODE = 0o644

#: Lowercase booleans, as in ``catalogue.csv``.
TRUE = "true"
FALSE = "false"

#: Full CSV header, in order. Part of the contract: the converter reads it back.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "image",
    "clip_id",
    "frame_idx",
    "t_ms",
    "stratum",
    "skill",
    "trainer_present",
    "trainer_contact",
    "display_width",
    "display_height",
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _as_bool(value: object) -> bool:
    """Coerce a parquet cell to a bool; ``NaN`` or a blank cell is False.

    A missing cell must not read as a phantom ``True``: ``bool(float("nan"))`` is
    ``True`` in Python, which would put every dropped frame in the contact stratum.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() == TRUE
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number != 0.0


def _as_int(value: object) -> int:
    """Coerce a CSV cell to an int, 0 when it is blank or has been typed into junk."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return int(number) if math.isfinite(number) else 0


def normalise_skill(skill: str) -> str:
    """A skill as a comparable token: lower case, spaces folded to underscores.

    The skill column is typed by hand into a spreadsheet, so ``"Line "`` and
    ``"straight line"`` have to compare equal to what ``--skills`` says.
    """
    return " ".join(str(skill).split()).lower().replace(" ", "_")


def label_frames_dir(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_frames``, where the JPEGs and the manifest go."""
    root = pathlib.Path(data) if data is not None else data_dir()
    return root / LABEL_FRAMES_DIRNAME


def manifest_path(data: str | pathlib.Path | None = None) -> pathlib.Path:
    """``<data_dir>/label_frames/manifest.csv``."""
    return label_frames_dir(data) / MANIFEST_NAME


def image_name(clip_id: str, frame_idx: int) -> str:
    """File name of one sampled frame: ``<clip_id>_<frame_idx>.jpg``."""
    return f"{clip_id}_{frame_idx}.jpg"


def write_jpeg(frame: np.ndarray, path: str | pathlib.Path, quality: int = JPEG_QUALITY) -> None:
    """Write one BGR frame as a JPEG, refusing to fail quietly.

    ``cv2.imwrite`` returns ``False`` for a path it cannot write, and the file
    mode is then set explicitly because these frames are opened by everyone
    sharing the workspace.
    """
    path = pathlib.Path(path)
    if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
        raise OSError(f"could not write JPEG: {path}")
    os.chmod(path, FILE_MODE)


def _format_bool(value: bool) -> str:
    """A CSV flag, lowercase as in the catalogue sheet."""
    return TRUE if value else FALSE


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipInfo:
    """The catalogue columns this sampler needs, one row per clip."""

    clip_id: str
    filename: str
    session_date: str
    display_width: int
    display_height: int
    skill: str


@dataclasses.dataclass(frozen=True)
class FrameCandidate:
    """One frame that may be sampled, with everything the manifest will say about it.

    ``clip_t_start_ms``/``clip_t_end_ms`` are the clip's *whole* timeline, not
    this stratum's: the time bins that spread the picks inside a clip are cut
    from the clip, so a frame at 8 s stays at 8 s even when the frames before it
    were not sampled.
    """

    clip_id: str
    frame_idx: int
    t_ms: int
    clip_t_start_ms: int
    clip_t_end_ms: int
    stratum: str
    inverted: bool
    session_date: str
    skill: str
    trainer_present: bool
    trainer_contact: bool
    display_width: int
    display_height: int

    @property
    def key(self) -> tuple[str, int]:
        """The identity of a frame: one clip, one frame index."""
        return (self.clip_id, self.frame_idx)

    def sort_key(self) -> tuple[str, int, int]:
        """Deterministic manifest order: by clip, then by time along the clip."""
        return (self.clip_id, self.t_ms, self.frame_idx)


@dataclasses.dataclass(frozen=True)
class Pool:
    """Every frame that may be sampled, plus why the pool looks the way it does.

    The counts are what the CLI prints, so a run that quietly fell back to
    "every clip" says so instead of looking like an ordinary one.
    """

    candidates: list[FrameCandidate] = dataclasses.field(default_factory=list)
    skills: tuple[str, ...] = DEFAULT_SKILLS
    skill_filter_active: bool = True
    clips_considered: int = 0
    clips_with_keypoints: int = 0
    clips_with_video: int = 0
    clips_missing_file: int = 0
    clips_excluded_by_skill: int = 0
    clips_excluded_walk: int = 0
    clips_unlabelled_skill: int = 0
    trainer_report_known: bool = True
    contact_column_known: bool = True
    notes: tuple[str, ...] = ()

    @property
    def clip_ids(self) -> list[str]:
        """Distinct clips contributing at least one candidate frame."""
        return sorted({candidate.clip_id for candidate in self.candidates})


@dataclasses.dataclass(frozen=True)
class SampleReport:
    """What one sampling run produced."""

    requested: int
    rows: list[dict[str, Any]]
    quotas: dict[str, int]
    counts: dict[str, int]
    inverted: int
    clips: int
    sessions: int
    out_dir: pathlib.Path
    manifest: pathlib.Path
    pool: Pool
    failures: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    size_mismatches: int = 0
    target_inverted: int = 0
    #: JPEGs left in the output directory by an earlier, larger run.
    stale_images: list[str] = dataclasses.field(default_factory=list)

    @property
    def written(self) -> int:
        """Frames actually written: below ``requested`` when clips failed."""
        return len(self.rows)

    @property
    def inverted_percent(self) -> float:
        """Share of the written frames that are a hold."""
        if not self.rows:
            return 0.0
        return 100.0 * self.inverted / len(self.rows)


def stratum_of(*, detected: bool, trainer_contact: bool, trainer_present: bool) -> str | None:
    """Which stratum a frame belongs to, or ``None`` when it is not a candidate.

    A frame nobody was found in cannot be labelled and never enters the pool.
    """
    if not detected:
        return None
    if trainer_contact:
        return "trainer_contact"
    return "clean_trainer" if trainer_present else "clean_no_trainer"


# --------------------------------------------------------------------------- #
# Reading the inputs
# --------------------------------------------------------------------------- #


def read_catalogue(path: str | pathlib.Path) -> list[ClipInfo]:
    """Read the catalogue columns the sampler needs; a missing file is an empty one.

    Rows flagged ``missing`` are dropped: their video is gone, so the frame could
    not be decoded anyway and counting them would only skew the report.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    clips: list[ClipInfo] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = (row.get("clip_id") or "").strip()
            if not clip_id or (row.get(MISSING_COLUMN) or "").strip().lower() == TRUE:
                continue
            clips.append(
                ClipInfo(
                    clip_id=clip_id,
                    filename=(row.get("filename") or "").strip(),
                    session_date=(row.get("session_date") or "").strip(),
                    display_width=_as_int(row.get("display_width")),
                    display_height=_as_int(row.get("display_height")),
                    skill=normalise_skill(row.get("skill") or ""),
                )
            )
    return clips


def read_trainer_present(path: str | pathlib.Path) -> dict[str, bool]:
    """Per clip ``trainer_present`` from the trainer report; a missing file is empty.

    A clip the report does not mention reads as trainer-free, which is the
    answer that lets the clean strata be filled; the run reports that the report
    was missing so the split is not mistaken for a measured one.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return {}
    found: dict[str, bool] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = (row.get("clip_id") or "").strip()
            if not clip_id:
                continue
            found[clip_id] = (row.get("trainer_present") or "").strip().lower() == TRUE
    return found


def frame_is_inverted(x_by_joint: Mapping[str, float], y_by_joint: Mapping[str, float]) -> bool:
    """MediaPipe's inversion rule for one frame, from that frame's joint columns.

    A joint missing from the frame reads as ``NaN``, which
    :func:`handstand.pose_mediapipe.is_inverted` treats as "cannot tell", i.e.
    not a hold — the same answer it gives for a frame the model half-saw.
    """
    points = np.array(
        [[x_by_joint.get(joint, np.nan), y_by_joint.get(joint, np.nan)] for joint in JOINT_NAMES],
        dtype=np.float64,
    )
    return is_inverted(points)


def load_clip_frames(
    parquet_path: str | pathlib.Path,
    clip: ClipInfo,
    trainer_present: bool,
) -> tuple[list[FrameCandidate], bool]:
    """Every candidate frame of one clip, from its keypoint parquet.

    Returns the candidates and whether the parquet carried a ``trainer_contact``
    column: the plain single-person parquets do not, and a run over them can
    never fill the ``trainer_contact`` stratum.
    """
    table = pd.read_parquet(parquet_path)
    has_contact = CONTACT_COLUMN in table.columns
    per_frame = table.drop_duplicates(subset="frame_idx").sort_values("frame_idx")
    joints = list(JOINT_NAMES)
    wide_x = table.pivot(index="frame_idx", columns="joint", values="x").reindex(columns=joints)
    wide_y = table.pivot(index="frame_idx", columns="joint", values="y").reindex(columns=joints)

    times = per_frame["t_ms"].tolist()
    start_ms = int(times[0]) if times else 0
    end_ms = int(times[-1]) if times else 0

    candidates: list[FrameCandidate] = []
    for _, row in per_frame.iterrows():
        frame_idx = int(row["frame_idx"])
        contact = _as_bool(row[CONTACT_COLUMN]) if has_contact else False
        stratum = stratum_of(
            detected=_as_bool(row[DETECTED_COLUMN]),
            trainer_contact=contact,
            trainer_present=trainer_present,
        )
        if stratum is None:
            continue
        candidates.append(
            FrameCandidate(
                clip_id=clip.clip_id,
                frame_idx=frame_idx,
                t_ms=int(row["t_ms"]),
                clip_t_start_ms=start_ms,
                clip_t_end_ms=max(end_ms, int(row["t_ms"])),
                stratum=stratum,
                inverted=frame_is_inverted(
                    dict(wide_x.loc[frame_idx]), dict(wide_y.loc[frame_idx])
                ),
                session_date=clip.session_date,
                skill=clip.skill,
                trainer_present=trainer_present,
                trainer_contact=contact,
                display_width=clip.display_width,
                display_height=clip.display_height,
            )
        )
    return candidates, has_contact


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #


def _default_resolver(
    data: pathlib.Path, videos: pathlib.Path | None
) -> Callable[[str], pathlib.Path | None]:
    """A ``clip_id`` -> video-path resolver that reports "no video" as ``None``."""

    def resolve(clip_id: str) -> pathlib.Path | None:
        try:
            return video_for_clip(clip_id, videos, data)
        except (FileNotFoundError, ValueError):
            return None

    return resolve


def build_pool(
    data: str | pathlib.Path | None = None,
    *,
    skills: Sequence[str] = DEFAULT_SKILLS,
    source: str = DEFAULT_SOURCE,
    rotate: str = DEFAULT_ROTATE,
    videos: str | pathlib.Path | None = None,
    resolve_video: Callable[[str], pathlib.Path | None] | None = None,
) -> Pool:
    """Collect every frame that may be sampled, and say how the pool was cut down.

    A clip is skipped when its skill is not requested, when its skill is
    ``walk``, when its keypoint parquet is missing and when its video cannot be
    resolved. While **no** clip carries a skill the filter is switched off
    entirely (the catalogue is not filled in yet) and the manifest says so.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    wanted_skills = tuple(normalise_skill(skill) for skill in skills)
    clips = read_catalogue(root / CATALOGUE_NAME)
    report_path = root / REPORTS_DIRNAME / TRAINER_REPORT_NAME
    trainer_present = read_trainer_present(report_path)
    resolve = resolve_video or _default_resolver(root, pathlib.Path(videos) if videos else None)

    notes: list[str] = []
    skill_filter_active = any(clip.skill for clip in clips)
    if not skill_filter_active:
        notes.append(
            "the catalogue carries no skill values, so the skill filter is off and every clip "
            f"is sampled; the manifest's skill column reads {UNLABELLED_SKILL!r}"
        )
    if not report_path.is_file():
        notes.append(
            f"no trainer report at {report_path}: every clip counts as trainer-free, so the "
            "clean_trainer and trainer_contact strata stay empty"
        )

    candidates: list[FrameCandidate] = []
    considered = 0
    with_keypoints = 0
    with_video = 0
    missing_file = 0
    excluded_skill = 0
    excluded_walk = 0
    unlabelled = 0
    contact_known = True

    for clip in clips:
        if clip.skill == EXCLUDED_SKILL:
            excluded_walk += 1
            continue
        if skill_filter_active and clip.skill not in wanted_skills:
            excluded_skill += 1
            unlabelled += int(not clip.skill)
            continue
        considered += 1
        parquet = keypoints_root(root, source) / rotate / f"{clip.clip_id}.parquet"
        if not parquet.is_file():
            continue
        with_keypoints += 1
        if resolve(clip.clip_id) is None:
            missing_file += 1
            continue
        clip_candidates, has_contact = load_clip_frames(
            parquet, clip, trainer_present.get(clip.clip_id, False)
        )
        contact_known = contact_known and has_contact
        if not clip_candidates:
            with_video -= 1
            continue
        with_video += 1
        candidates.extend(clip_candidates)

    if not contact_known:
        notes.append(
            f"the {source_dirname(source)} parquets have no {CONTACT_COLUMN} column, so the "
            "trainer_contact stratum cannot be filled"
        )
    if unlabelled:
        notes.append(
            f"{unlabelled} clip(s) have an empty skill and were left out because the skill "
            f"filter is on (skills: {', '.join(wanted_skills)})"
        )

    return Pool(
        candidates=candidates,
        skills=wanted_skills,
        skill_filter_active=skill_filter_active,
        clips_considered=considered,
        clips_with_keypoints=with_keypoints,
        clips_with_video=with_video,
        clips_missing_file=missing_file,
        clips_excluded_by_skill=excluded_skill,
        clips_excluded_walk=excluded_walk,
        clips_unlabelled_skill=unlabelled,
        trainer_report_known=report_path.is_file(),
        contact_column_known=contact_known,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Picking
# --------------------------------------------------------------------------- #


def stratum_quotas(n: int, shares: Sequence[float] = STRATUM_SHARES) -> dict[str, int]:
    """Split ``n`` frames over the strata: 60 % / 25 % / 15 % by default.

    Each stratum gets ``floor(n * share)`` and the frames left over go to the
    largest shares first, so the quotas always add up to ``n`` exactly. The
    shares are read positionally from :data:`STRATA`, and a share of 0 for a
    stratum is honoured (a run over keypoints with no contact column asks for
    none).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if len(shares) != len(STRATA):
        raise ValueError(f"expected {len(STRATA)} shares, got {len(shares)}")
    quotas = {
        stratum: int(math.floor(n * share)) for stratum, share in zip(STRATA, shares, strict=True)
    }
    leftover = n - sum(quotas.values())
    by_share = sorted(STRATA, key=lambda name: (-shares[STRATA.index(name)], STRATA.index(name)))
    for stratum in by_share:
        if leftover <= 0:
            break
        quotas[stratum] += 1
        leftover -= 1
    return quotas


def _clip_order(clip_sessions: Mapping[str, str], *, seed: int) -> list[str]:
    """Clips in the order they get their first pick: round-robin over session dates.

    Sorting by clip id alone would hand the whole quota to whichever session day
    sorts first; dealing round-robin means every session day contributes before
    any of them contributes twice. The seed shuffles both the sessions and the
    clips inside one, so another seed picks a different sample of the same shape.
    """
    sessions: dict[str, list[str]] = {}
    for clip_id, session_date in clip_sessions.items():
        sessions.setdefault(session_date, []).append(clip_id)
    rng = random.Random(f"frame_sampler:{seed}")
    order_of_sessions = sorted(sessions)
    rng.shuffle(order_of_sessions)
    for session in order_of_sessions:
        rng.shuffle(sessions[session])
    order: list[str] = []
    for position in range(max((len(clips) for clips in sessions.values()), default=0)):
        for session in order_of_sessions:
            clips = sessions[session]
            if position < len(clips):
                order.append(clips[position])
    return order


def _in_bin_order(
    frames: Sequence[FrameCandidate],
    *,
    n_per_clip: int,
) -> list[FrameCandidate]:
    """Order a clip's frames so the first one is in its first time bin.

    The clip's whole timeline is cut into ``n_per_clip`` equal-width bins and
    every frame is emitted bin by bin, each bin's frames ordered by distance to
    its centre. The list is not capped: it is an *order* to take frames from, and
    the caller stops when the clip's budget of :data:`N_PER_CLIP` frames is spent.
    That way a frame that is too close to one already taken is skipped in favour
    of the next-nearest frame in the same bin rather than in favour of the next
    bin, which is what keeps a clip's picks spread over its whole timeline.
    """
    if not frames or n_per_clip <= 0:
        return []
    start = min(frame.clip_t_start_ms for frame in frames)
    end = max(frame.clip_t_end_ms for frame in frames)
    span = max(end - start, 1)
    bins: list[list[FrameCandidate]] = [[] for _ in range(n_per_clip)]
    for frame in sorted(frames, key=lambda item: item.sort_key()):
        index = min(int((frame.t_ms - start) * n_per_clip / span), n_per_clip - 1)
        bins[index].append(frame)
    ordered: list[FrameCandidate] = []
    for index, bin_frames in enumerate(bins):
        centre = start + span * (index + 0.5) / n_per_clip
        ordered += sorted(bin_frames, key=lambda frame: (abs(frame.t_ms - centre), frame.frame_idx))
    return ordered


def _select(
    pool: Sequence[FrameCandidate],
    quotas: Mapping[str, int],
    n: int,
    *,
    seed: int,
    n_per_clip: int,
    min_spacing_ms: int,
) -> list[FrameCandidate]:
    """Fill every stratum's quota, sharing one per-clip budget between them.

    :data:`N_PER_CLIP` is a budget for the *clip*, not for a clip-and-stratum: a
    clip that offered four clean frames and two contact frames gives four frames
    in total, dealt round-robin over the strata (2 / 1 / 1 for the default
    60/25/15 split). Clips are visited in the session order of
    :func:`_clip_order`, each one taking its frames from the bin order of
    :func:`_in_bin_order` and skipping any that is closer to an already picked
    frame than :data:`MIN_SPACING_S`. A stratum whose quota is spent is not dealt
    from any more, so one stratum cannot starve the next.
    """
    by_clip: dict[str, dict[str, list[FrameCandidate]]] = {}
    for frame in pool:
        by_clip.setdefault(frame.clip_id, {}).setdefault(frame.stratum, []).append(frame)
    if not by_clip:
        return []
    sessions: dict[str, str] = {}
    for clip_id, frames in by_clip.items():
        for stratum_frames in frames.values():
            if stratum_frames:
                sessions[clip_id] = stratum_frames[0].session_date
                break
    remaining = {stratum: int(quotas.get(stratum, 0)) for stratum in STRATA}
    chosen: list[FrameCandidate] = []
    for clip_id in _clip_order(sessions, seed=seed):
        if sum(remaining.values()) <= 0 or len(chosen) >= n:
            break
        budget = n_per_clip
        ordered = {
            stratum: _in_bin_order(frames, n_per_clip=n_per_clip)
            for stratum, frames in by_clip[clip_id].items()
        }
        cursors = {stratum: 0 for stratum in STRATA}
        while budget > 0 and any(
            remaining[stratum] > 0 and cursors[stratum] < len(ordered.get(stratum, []))
            for stratum in STRATA
        ):
            for stratum in STRATA:
                picks = ordered.get(stratum, [])
                if budget <= 0 or remaining[stratum] <= 0 or cursors[stratum] >= len(picks):
                    continue
                pick = picks[cursors[stratum]]
                cursors[stratum] += 1
                if not _spacing_ok(pick, chosen, min_spacing_ms):
                    continue
                chosen.append(pick)
                remaining[stratum] -= 1
                budget -= 1
    return chosen


def _count_inverted(frames: Sequence[FrameCandidate]) -> int:
    """How many of these frames are a hold."""
    return sum(1 for frame in frames if frame.inverted)


def _spacing_ok(
    candidate: FrameCandidate,
    chosen: Sequence[FrameCandidate],
    min_spacing_ms: int,
    *,
    ignore: FrameCandidate | None = None,
) -> bool:
    """Is ``candidate`` far enough from every pick of the same clip already made?"""
    return all(
        abs(candidate.t_ms - other.t_ms) >= min_spacing_ms
        for other in chosen
        if other.clip_id == candidate.clip_id and other is not ignore
    )


def _replacement(
    victim: FrameCandidate,
    chosen: Sequence[FrameCandidate],
    spare: Mapping[str, Sequence[FrameCandidate]],
    counts: Mapping[str, int],
    *,
    n_per_clip: int,
    min_spacing_ms: int,
) -> FrameCandidate | None:
    """The hold frame that should take ``victim``'s place in the sample.

    The same clip and stratum first, closest in time: that leaves the clip's
    spread intact. Failing that, another clip of the same stratum that is under
    the per-clip cap, preferring the same session date, so a swap does not skew
    either split.
    """
    same_clip = [
        frame for frame in spare.get(victim.clip_id, ()) if frame.stratum == victim.stratum
    ]
    by_distance = sorted(
        same_clip, key=lambda frame: (abs(frame.t_ms - victim.t_ms), frame.frame_idx)
    )
    for option in by_distance:
        if _spacing_ok(option, chosen, min_spacing_ms, ignore=victim):
            return option
    alternatives: list[tuple[int, int, str, int, FrameCandidate]] = []
    for clip_id, frames in spare.items():
        if clip_id == victim.clip_id or counts.get(clip_id, 0) >= n_per_clip:
            continue
        for frame in frames:
            if frame.stratum != victim.stratum:
                continue
            same_session = 0 if frame.session_date == victim.session_date else 1
            alternatives.append(
                (same_session, abs(frame.t_ms - victim.t_ms), clip_id, frame.frame_idx, frame)
            )
    for _, _, _, _, frame in sorted(alternatives, key=lambda item: item[:4]):
        if _spacing_ok(frame, chosen, min_spacing_ms, ignore=victim):
            return frame
    return None


def _enforce_inverted(
    chosen: Sequence[FrameCandidate],
    pool: Sequence[FrameCandidate],
    *,
    target: int,
    n_per_clip: int,
    min_spacing_ms: int,
) -> list[FrameCandidate]:
    """Swap non-hold frames for hold frames until ``target`` of them are holds.

    The strata are picked first and the hold share second, because the strata
    answer "how hard is this frame" and the hold share answers "is it a
    handstand at all" — two different questions. Every swap keeps the stratum,
    so the counts do not drift.
    """
    sample = list(chosen)
    if _count_inverted(sample) >= target:
        return sample
    used = {frame.key for frame in sample}
    counts: dict[str, int] = {}
    for frame in sample:
        counts[frame.clip_id] = counts.get(frame.clip_id, 0) + 1
    spare: dict[str, list[FrameCandidate]] = {}
    for frame in pool:
        if frame.inverted and frame.key not in used:
            spare.setdefault(frame.clip_id, []).append(frame)
    for frames in spare.values():
        frames.sort(key=lambda frame: frame.sort_key())

    improved = True
    while improved and _count_inverted(sample) < target:
        improved = False
        for position, victim in enumerate(sample):
            if victim.inverted:
                continue
            replacement = _replacement(
                victim, sample, spare, counts, n_per_clip=n_per_clip, min_spacing_ms=min_spacing_ms
            )
            if replacement is None:
                continue
            sample[position] = replacement
            spare[replacement.clip_id].remove(replacement)
            improved = True
            break
    return sample


def plan(
    candidates: Sequence[FrameCandidate],
    n: int = DEFAULT_N,
    *,
    seed: int = DEFAULT_SEED,
    quotas: Mapping[str, int] | None = None,
    n_per_clip: int = N_PER_CLIP,
    min_spacing_s: float = MIN_SPACING_S,
    inverted_fraction: float = DEFAULT_INVERTED_FRACTION,
) -> tuple[list[FrameCandidate], dict[str, int], int]:
    """Choose the frames to label: the pure half of :func:`sample_frames`.

    Returns the sample (ordered by clip and time), the quota per stratum and the
    number of holds aimed for. Nothing is read or written here, which is what
    makes the whole spread reproducible in a test.
    """
    if n_per_clip < 1:
        raise ValueError(f"n_per_clip must be >= 1, got {n_per_clip}")
    if min_spacing_s < 0.0:
        raise ValueError(f"min_spacing_s must be >= 0, got {min_spacing_s}")
    if not 0.0 <= inverted_fraction <= 1.0:
        raise ValueError(f"inverted_fraction must be between 0 and 1, got {inverted_fraction}")
    if n <= 0 or not candidates:
        return [], dict(quotas) if quotas is not None else stratum_quotas(n), 0

    share = dict(quotas) if quotas is not None else stratum_quotas(n)
    min_spacing_ms = int(round(min_spacing_s * 1000))
    chosen = _select(
        candidates,
        share,
        n,
        seed=seed,
        n_per_clip=n_per_clip,
        min_spacing_ms=min_spacing_ms,
    )
    target = min(len(chosen), math.ceil(inverted_fraction * min(n, len(chosen))))
    chosen = _enforce_inverted(
        chosen,
        candidates,
        target=target,
        n_per_clip=n_per_clip,
        min_spacing_ms=min_spacing_ms,
    )
    chosen.sort(key=lambda frame: frame.sort_key())
    return chosen, share, target


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_frames(
    video_path: str | pathlib.Path,
    wanted: Mapping[int, FrameCandidate],
    out_dir: str | pathlib.Path,
    *,
    open_video: Callable[[Any], Any] | None = None,
    write_frame: Callable[[np.ndarray, pathlib.Path], None] | None = None,
) -> dict[int, tuple[int, int]]:
    """Decode one clip in display orientation and write the wanted frames.

    The clip is decoded once, in order, and only the wanted frames are written.
    Every written frame's decoded size is returned, so the manifest can carry the
    size of the image the labeler actually sees. A frame the keypoints claim but
    the decoder never reaches is an error rather than a silent drop: the labeler
    would otherwise be handed a manifest row with no image.

    ``open_video`` and ``write_frame`` default to the real reader and the real
    JPEG writer, looked up at call time so a test can replace either one.
    """
    read = open_video or DisplayVideo
    write = write_frame or write_jpeg
    remaining = dict(wanted)
    destination = pathlib.Path(out_dir)
    sizes: dict[int, tuple[int, int]] = {}
    with read(video_path) as video:
        for packet in video:
            candidate = remaining.pop(packet.frame_idx, None)
            if candidate is not None:
                name = image_name(candidate.clip_id, candidate.frame_idx)
                write(packet.frame, destination / name)
                sizes[candidate.frame_idx] = (
                    int(packet.frame.shape[1]),
                    int(packet.frame.shape[0]),
                )
            if not remaining:
                break
    if remaining:
        raise RuntimeError(
            f"{pathlib.Path(video_path).name}: the video ends before frame(s) "
            f"{sorted(remaining)}, which the keypoints claim exist"
        )
    return sizes


def _manifest_row(
    candidate: FrameCandidate,
    size: tuple[int, int],
    *,
    skill: str,
) -> dict[str, Any]:
    """One manifest row: what this frame is, and how big the image is.

    The size is the decoded frame's, not the catalogue's: the JPEG is what the
    labeler places points on, so the percentages Label Studio exports are
    fractions of *that*.
    """
    return {
        "image": image_name(candidate.clip_id, candidate.frame_idx),
        "clip_id": candidate.clip_id,
        "frame_idx": candidate.frame_idx,
        "t_ms": candidate.t_ms,
        "stratum": candidate.stratum,
        "skill": skill,
        "trainer_present": _format_bool(candidate.trainer_present),
        "trainer_contact": _format_bool(candidate.trainer_contact),
        "display_width": size[0],
        "display_height": size[1],
    }


def write_manifest(rows: Sequence[Mapping[str, Any]], path: str | pathlib.Path) -> pathlib.Path:
    """Write the manifest atomically: temp file beside it, then rename.

    A reader either sees the previous manifest or the new one, so an interrupted
    re-run never leaves a half-written work list for the labeler.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = pathlib.Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(MANIFEST_COLUMNS),
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, FILE_MODE)
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


def stale_images(destination: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """JPEGs in the output directory the new manifest does not mention.

    Re-running with a smaller ``--n`` leaves the frames it no longer wants
    behind, and a labeler importing the whole folder would then be shown images
    that the manifest says nothing about. They are reported rather than deleted:
    the directory is shared, and a sampler run has no business removing files it
    did not write in this run.
    """
    wanted = {str(row["image"]) for row in rows}
    return sorted(path.name for path in destination.glob("*.jpg") if path.name not in wanted)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def sample_frames(
    n: int = DEFAULT_N,
    *,
    seed: int = DEFAULT_SEED,
    skills: Sequence[str] = DEFAULT_SKILLS,
    data: str | pathlib.Path | None = None,
    videos: str | pathlib.Path | None = None,
    out_dir: str | pathlib.Path | None = None,
    source: str = DEFAULT_SOURCE,
    rotate: str = DEFAULT_ROTATE,
    n_per_clip: int = N_PER_CLIP,
    min_spacing_s: float = MIN_SPACING_S,
    inverted_fraction: float = DEFAULT_INVERTED_FRACTION,
    resolve_video: Callable[[str], pathlib.Path | None] | None = None,
    open_video: Callable[[Any], Any] | None = None,
    write_frame: Callable[[np.ndarray, pathlib.Path], None] | None = None,
) -> SampleReport:
    """Pick, decode and write the frames to label, then write the manifest.

    The library-level entry point behind the CLI. Raises :class:`RuntimeError`
    when the inputs cannot produce a single frame — no catalogue, no keypoints,
    nothing left after the skill filter — because an empty manifest is never what
    the caller wanted. A clip that fails while being decoded is reported in
    :attr:`SampleReport.failures` and left out of the manifest, so every row of
    the manifest points at a JPEG that exists.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    destination = pathlib.Path(out_dir) if out_dir is not None else label_frames_dir(root)
    pool = build_pool(
        root,
        skills=skills,
        source=source,
        rotate=rotate,
        videos=videos,
        resolve_video=resolve_video,
    )
    if not pool.candidates:
        raise RuntimeError(
            f"no frame can be sampled from {root}: {pool.clips_considered} clip(s) considered, "
            f"{pool.clips_with_keypoints} with keypoints, {pool.clips_with_video} with a video. "
            "Generate the keypoints first:\n"
            "  uv run python -m handstand.pose_mediapipe --rotate "
            f"{rotate} --num-poses 3 --running-mode image --min-detection 0.2 --min-presence 0.2\n"
            f"  uv run python -m handstand.athlete --rotate {rotate}"
        )

    chosen, quotas, target = plan(
        pool.candidates,
        n,
        seed=seed,
        n_per_clip=n_per_clip,
        min_spacing_s=min_spacing_s,
        inverted_fraction=inverted_fraction,
    )
    destination.mkdir(parents=True, exist_ok=True)
    fallback_skill = UNLABELLED_SKILL if not pool.skill_filter_active else ""

    by_clip: dict[str, list[FrameCandidate]] = {}
    for frame in chosen:
        by_clip.setdefault(frame.clip_id, []).append(frame)

    resolve = resolve_video or _default_resolver(root, pathlib.Path(videos) if videos else None)
    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    mismatches = 0
    inverted = 0
    sessions: set[str] = set()
    for clip_id in sorted(by_clip):
        video_path = resolve(clip_id)
        if video_path is None:
            failures.append((clip_id, "no video found"))
            continue
        try:
            sizes = write_frames(
                video_path,
                {frame.frame_idx: frame for frame in by_clip[clip_id]},
                destination,
                open_video=open_video,
                write_frame=write_frame,
            )
        except (OSError, RuntimeError, ValueError) as error:
            failures.append((clip_id, f"{type(error).__name__}: {error}"))
            continue
        for frame in by_clip[clip_id]:
            size = sizes[frame.frame_idx]
            if (frame.display_width, frame.display_height) not in ((0, 0), size):
                mismatches += 1
            inverted += int(frame.inverted)
            sessions.add(frame.session_date)
            rows.append(_manifest_row(frame, size, skill=fallback_skill or frame.skill))

    rows.sort(key=lambda row: (str(row["clip_id"]), int(row["frame_idx"])))
    counts = {stratum: 0 for stratum in STRATA}
    for row in rows:
        counts[str(row["stratum"])] = counts.get(str(row["stratum"]), 0) + 1
    return SampleReport(
        requested=n,
        rows=rows,
        quotas=quotas,
        counts=counts,
        inverted=inverted,
        clips=len({str(row["clip_id"]) for row in rows}),
        sessions=len(sessions),
        out_dir=destination,
        manifest=write_manifest(rows, destination / MANIFEST_NAME),
        pool=pool,
        failures=failures,
        size_mismatches=mismatches,
        target_inverted=target,
        stale_images=stale_images(destination, rows),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.frame_sampler`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.frame_sampler",
        description="Sample frames for keypoint labelling and write them as JPEGs.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        metavar="N",
        help=f"how many frames to sample (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="N",
        help=f"sampling seed; the same seed always picks the same frames (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--skills",
        nargs="+",
        default=list(DEFAULT_SKILLS),
        metavar="SKILL",
        help=(
            "catalogue skills to sample, e.g. line press_up (default: "
            f"{' '.join(DEFAULT_SKILLS)}); ignored while the catalogue carries no skills at "
            f"all, and {EXCLUDED_SKILL!r} is never sampled"
        ),
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help=(
            "which keypoints to sample: athlete carries the trainer-contact flag the strata "
            f"need (default: {DEFAULT_SOURCE})"
        ),
    )
    parser.add_argument(
        "--rotate",
        choices=ROTATE_MODES,
        default=DEFAULT_ROTATE,
        help=f"rotation mode of the keypoints to read (default: {DEFAULT_ROTATE})",
    )
    parser.add_argument(
        "--n-per-clip",
        type=int,
        default=N_PER_CLIP,
        metavar="N",
        help=f"most frames sampled from one clip (default: {N_PER_CLIP})",
    )
    parser.add_argument(
        "--min-spacing-s",
        type=float,
        default=MIN_SPACING_S,
        metavar="SECONDS",
        help=f"closest two frames of one clip may be (default: {MIN_SPACING_S})",
    )
    parser.add_argument(
        "--inverted-fraction",
        type=float,
        default=DEFAULT_INVERTED_FRACTION,
        metavar="F",
        help=(
            "share of the sample that must be a hold, wrists below the ankles in the image "
            f"(default: {DEFAULT_INVERTED_FRACTION})"
        ),
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help=(
            "data directory holding catalogue.csv, reports/ and keypoints/ "
            "(default: $HANDSTAND_DATA)"
        ),
    )
    parser.add_argument(
        "--videos",
        type=pathlib.Path,
        default=None,
        help="videos directory (default: $HANDSTAND_VIDEOS, else the workspace videos dir)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=f"output directory (default: <data_dir>/{LABEL_FRAMES_DIRNAME})",
    )
    return parser


def summarise(report: SampleReport) -> str:
    """Multi-line summary of a run, for the CLI to print."""
    pool = report.pool
    skill_line = (
        f"  skill filter: active ({', '.join(pool.skills)}), {pool.clips_excluded_by_skill} "
        f"clip(s) excluded, {pool.clips_excluded_walk} walk clip(s) never sampled"
        if pool.skill_filter_active
        else f"  skill filter: OFF, the catalogue carries no skills so every clip is used "
        f"({pool.clips_excluded_walk} walk clip(s) never sampled)"
    )
    lines = [
        f"wrote {report.written} of {report.requested} requested frames from "
        f"{report.clips} clip(s) over {report.sessions} session(s) -> {report.out_dir}",
        f"  manifest: {report.manifest}",
        "  strata: "
        + " ".join(
            f"{stratum}={report.counts.get(stratum, 0)}/{report.quotas.get(stratum, 0)}"
            for stratum in STRATA
        ),
        f"  inverted (hold): {report.inverted} of {report.written} "
        f"({report.inverted_percent:.1f} %), target {report.target_inverted}",
        f"  pool: {len(pool.candidates)} candidate frames from {pool.clips_with_video} clip(s) "
        f"of {pool.clips_considered} considered",
        skill_line,
        f"  display size mismatches against the catalogue: {report.size_mismatches}",
    ]
    lines += [f"  note: {note}" for note in pool.notes]
    if report.stale_images:
        lines.append(
            f"  note: {len(report.stale_images)} JPEG(s) in {report.out_dir} are left from an "
            "earlier run and are not in this manifest; delete them or import only the manifest's "
            "images, or the labeler will be shown frames this run did not choose"
        )
    lines += [f"  fail clip {clip_id}: {reason}" for clip_id, reason in report.failures]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.frame_sampler``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.n < 0:
        parser.error("--n must be >= 0")
    if args.n_per_clip < 1:
        parser.error("--n-per-clip must be >= 1")
    if args.min_spacing_s < 0.0:
        parser.error("--min-spacing-s must be >= 0")
    if not 0.0 <= args.inverted_fraction <= 1.0:
        parser.error("--inverted-fraction must be between 0 and 1")

    try:
        report = sample_frames(
            args.n,
            seed=args.seed,
            skills=args.skills,
            data=args.data,
            videos=args.videos,
            out_dir=args.out,
            source=args.source,
            rotate=args.rotate,
            n_per_clip=args.n_per_clip,
            min_spacing_s=args.min_spacing_s,
            inverted_fraction=args.inverted_fraction,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"frame_sampler: {error}", file=sys.stderr)
        return 2
    print(summarise(report))
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
