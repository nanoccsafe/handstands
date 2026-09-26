"""Measure how much of the dataset the trainer is in, clip by clip and over the whole.

The athlete selection (:mod:`handstand.athlete`) already says, per frame, whether
another person was in it (``n_people``) and whether they were on top of the athlete
(``trainer_contact``). This module turns those frame-level answers into the
numbers a dataset plan needs: **how much of the data is usable as it stands, how
much has to go through the athlete-selection path, and how much is lost to trainer
contact**. The answer for the whole dataset is a row per clip plus a summary::

    <data_dir>/reports/trainer_report.csv   # one row per catalogue clip
    <data_dir>/reports/trainer_report.md    # the summary, in markdown

Both are regenerated output, never committed: the CSV is a table of the frame-level
columns of ``keypoints/mediapipe_athlete/`` and the markdown is a rendering of the
CSV, so both can be thrown away and rebuilt from the parquets at any time.

Generating the input is a **separate step** (``--generate``), because it is the
expensive one: a multi-person run over the whole dataset takes tens of minutes,
while the report on outputs that already exist takes seconds. ``--generate``
writes exactly what the two CLIs write —

    uv run python -m handstand.pose_mediapipe --rotate auto --num-poses 3 \\
        --running-mode image --min-detection 0.2 --min-presence 0.2
    uv run python -m handstand.athlete --rotate auto

— by calling the two ``run_clip`` functions those CLIs call, one clip at a time, so
a run that is interrupted half way is resumed by simply running it again: both
stages skip a clip whose parquet already exists, which is also what makes the
report re-runnable over a partially generated dataset. One clip that raises is
recorded in the CSV's ``error`` column and the batch carries on, because a report
that stops at the first bad clip is not a report.

What each row says:

``frames``, ``fps``
    The clip's decoded frame count, and its frame rate measured from the
    ``t_ms`` column (the median gap, not ``1/fps_avg``: the WhatsApp clips are
    variable frame rate, and every duration below is taken from the timestamps
    rather than from a nominal rate).
``pct_frames_second_person``
    Share of frames in which MediaPipe reported a second person, **after** dedup
    (see :func:`handstand.athlete.deduplicate`), so a doubled detection of the
    athlete is not counted as a trainer.
``second_person_seconds``, ``trainer_present``
    How long a second person was there in total, and whether that adds up to at
    least :data:`SECOND_PERSON_MIN_SECONDS`. The threshold is what keeps a
    one-frame phantom detection — which happens in almost every clip — from
    turning a clean clip into a "has a trainer" one.
``pct_trainer_contact``, ``longest_contact_run_s``
    Share of frames flagged as contact, and the longest unbroken stretch of
    contact in seconds. The run is the number that says whether the clip is a
    quick spotter or a hands-on session; the percentage alone flattens a clip
    that is clean for 30 s and then touched for 1 s against one that is never
    left alone.
``pct_dropped``
    Share of frames the selection left unattributed (no athlete could be chosen),
    i.e. data that is not usable for anything, trainer or not.
``notes``
    The catalogue's own ``notes`` column, carried through untouched so this report
    can be lined up against the hand annotation of the same clips later.

One thing ``--generate`` refuses to skip: a clip whose multi-person keypoints
exist but were made with *other* detector settings (see
:func:`_stale_multi_person`). The default VIDEO-mode run reports one person per
frame even with the trainer standing there, so leaving such a file in place would
report every trainer in that clip as absent.

CLI::

    cd pipeline
    uv run python -m handstand.trainer_report                # report on existing outputs
    uv run python -m handstand.trainer_report --generate     # generate what is missing, then report
    uv run python -m handstand.trainer_report --limit 5      # a first look
    uv run python -m handstand.trainer_report --clips 6508f9b355bd
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import math
import pathlib
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from handstand import athlete, pose_mediapipe
from handstand.paths import data_dir, videos_dir

__all__ = [
    "CATALOGUE_FILENAME",
    "CSV_FILENAME",
    "DEFAULT_ROTATE",
    "DETECTOR_SETTINGS",
    "HISTOGRAM_BUCKET_PERCENT",
    "MARKDOWN_FILENAME",
    "MOST_AFFECTED_CLIPS",
    "NUM_POSES",
    "REASON_COLUMNS",
    "REPORTS_DIRNAME",
    "REPORT_COLUMNS",
    "SECOND_PERSON_MIN_PEOPLE",
    "SECOND_PERSON_MIN_SECONDS",
    "CatalogueEntry",
    "ClipMetrics",
    "FrameFlags",
    "build_arg_parser",
    "clip_metrics",
    "contact_histogram",
    "flagged_seconds",
    "frame_durations_ms",
    "frame_flags",
    "generate_batch",
    "generate_clip",
    "longest_run",
    "longest_run_seconds",
    "main",
    "measure_clip",
    "metrics_table",
    "percentage",
    "read_catalogue",
    "reports_dir",
    "sidecar_runtime",
    "summarise",
    "write_report",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The catalogue the report is built over: ``<data_dir>/catalogue.csv``.
CATALOGUE_FILENAME = "catalogue.csv"
#: Where the two report files go, relative to ``<data_dir>/``.
REPORTS_DIRNAME = "reports"
CSV_FILENAME = "trainer_report.csv"
MARKDOWN_FILENAME = "trainer_report.md"

#: Rotation mode the report is about: the recommended one.
DEFAULT_ROTATE = athlete.DEFAULT_ROTATE

#: People per frame the multi-person run asks MediaPipe for.
NUM_POSES = 3
#: The recommended detector settings for finding a second body: the detector runs
#: on **every** frame (IMAGE mode) and the two score thresholds are lowered, which
#: is what a multi-person run needs. ``min_tracking`` stays on MediaPipe's own
#: default, as in the documented command.
DETECTOR_SETTINGS = pose_mediapipe.DetectorSettings(
    running_mode="image",
    min_detection=0.2,
    min_presence=0.2,
)

#: A second person has to be in the clip for this many seconds in total before the
#: clip counts as having a trainer in it. "In total" is what keeps a one-frame
#: phantom detection — which happens in almost every clip — from turning a clean
#: clip into a "has a trainer" one, and it lets a trainer who walks past twice
#: count as one. At 30 fps a single frame is 0.03 s.
SECOND_PERSON_MIN_SECONDS = 0.5
#: How many people in a frame means somebody else is in it: one is the athlete, two
#: or more is at least one other body. This is the ``n_people`` column *after* the
#: athlete selection's dedup, so a doubled detection of the athlete is not a trainer.
SECOND_PERSON_MIN_PEOPLE = 2

#: Width of one bucket of the contact histogram, in percentage points.
HISTOGRAM_BUCKET_PERCENT = 10.0
#: How many clips the summary's "most affected" table lists.
MOST_AFFECTED_CLIPS = 10
#: Decimal places every float column of the CSV is rounded to.
FLOAT_DIGITS = 3
#: Decimal places every percentage of the markdown is printed with.
PERCENT_DIGITS = 1
#: Bar width of the histogram, in block characters.
HISTOGRAM_BAR_WIDTH = 40

#: One CSV column per contact rule, named after it: which rule flagged the frames
#: says a lot about *how* the trainer is in the clip.
REASON_COLUMNS: dict[str, str] = {
    reason: f"contact_{reason}_frames" for reason in athlete.CONTACT_REASONS
}

#: Full CSV header, in order. Part of the contract: a column added here is a
#: column every reader of the sheet can rely on.
REPORT_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "filename",
    "rotate",
    "frames",
    "fps",
    "second_person_frames",
    "pct_frames_second_person",
    "second_person_seconds",
    "contact_frames",
    "pct_trainer_contact",
    "longest_contact_run_s",
    "trainer_present",
    "athlete_frames",
    "pct_athlete",
    "dropped_frames",
    "pct_dropped",
    *REASON_COLUMNS.values(),
    "generate_seconds",
    "notes",
    "error",
)

#: CSV columns holding whole numbers.
_INT_COLUMNS = (
    "frames",
    "second_person_frames",
    "contact_frames",
    "athlete_frames",
    "dropped_frames",
    *REASON_COLUMNS.values(),
)
#: CSV columns holding floats, rounded to :data:`FLOAT_DIGITS`.
_FLOAT_COLUMNS = (
    "fps",
    "pct_frames_second_person",
    "second_person_seconds",
    "pct_trainer_contact",
    "longest_contact_run_s",
    "pct_athlete",
    "pct_dropped",
    "generate_seconds",
)
#: The one boolean column, written the way ``catalogue.csv`` writes its flags.
_BOOL_COLUMNS = ("trainer_present",)
#: The catalogue's own annotation column, carried into the report untouched.
NOTES_COLUMN = "notes"

#: Lowercase booleans in the CSV, as in the catalogue sheet, so the two sheets
#: read the same way in a spreadsheet.
TRUE = "true"
FALSE = "false"


# --------------------------------------------------------------------------- #
# Small measurements
# --------------------------------------------------------------------------- #


def percentage(count: int, total: int) -> float:
    """``count`` as a percentage of ``total``; 0.0 for an empty clip.

    A clip with no frames, and a clip whose only frame failed to be measured, both
    report 0 % rather than raising: one bad clip must not stop the report.
    """
    return 100.0 * count / total if total else 0.0


def frame_durations_ms(t_ms: Sequence[int]) -> np.ndarray:
    """How long each frame of a clip is held, in milliseconds, one per timestamp.

    The gap to the next timestamp is what a decoder's timestamps mean, and it is
    the only frame rate available for a variable-frame-rate clip. The last frame
    has no timestamp after it, so it takes the median gap; a non-increasing pair
    (which the runner's own 1 ms bump for duplicate timestamps can produce) takes
    it too, rather than a negative duration. An empty clip gets an empty array.
    """
    stamps = np.asarray(list(t_ms), dtype=np.float64)
    if stamps.size == 0:
        return np.empty(0, dtype=np.float64)
    gaps = np.diff(stamps)
    positive = gaps[gaps > 0.0]
    typical = float(np.median(positive)) if positive.size else 0.0
    durations = np.empty(stamps.size, dtype=np.float64)
    if stamps.size > 1:
        durations[:-1] = np.where(gaps > 0.0, gaps, typical)
    durations[-1] = typical
    return durations


def _longest_run_span(flags: Sequence[bool]) -> tuple[int, int]:
    """``(start, length)`` of the first longest stretch of true flags; length 0 if none.

    Plain iteration rather than a clever array trick: the answer is a stretch of
    frames, and a frame-at-a-time scan is clearer here than a vectorised one.
    """
    best_start = 0
    best_length = 0
    start = 0
    current = 0
    for index, flag in enumerate(flags):
        if flag:
            if current == 0:
                start = index
            current += 1
            if current > best_length:
                best_start, best_length = start, current
        else:
            current = 0
    return best_start, best_length


def longest_run(flags: Sequence[bool]) -> int:
    """The longest stretch of consecutive true flags, in items; 0 when none are."""
    return _longest_run_span(flags)[1]


def longest_run_seconds(flags: Sequence[bool], durations_ms: Sequence[float]) -> float:
    """The stretch of consecutive true flags that lasted longest, in seconds.

    The run is the one that covers the most *time*, not the one with the most
    frames: these clips are variable frame rate, so a run of four 10 ms frames is
    shorter on screen than a run of one 100 ms frame, and a column measured in
    seconds that reported the first as "the longest run" would understate the
    trainer's longest touch. On an evenly spaced clip the two agree, and both
    agree with :func:`longest_run`'s frame count.

    A run of nothing is 0.0 s, not 0 frames. Fewer durations than flags cannot be
    timed at all, so it falls back to counting the frames as a millisecond each;
    more durations than flags is harmless, the first one per frame being the one
    that counts.
    """
    durations = np.asarray(list(durations_ms), dtype=np.float64)
    if durations.size < len(flags):
        return longest_run(flags) / 1000.0
    best = current = 0.0
    for flag, duration in zip(flags, durations[: len(flags)], strict=True):
        if flag:
            current += duration
            best = max(best, current)
        else:
            current = 0.0
    return best / 1000.0


def flagged_seconds(flags: Sequence[bool], durations_ms: Sequence[float]) -> float:
    """How much of the clip the true flags cover, in seconds (not necessarily in a row)."""
    mask = np.asarray(list(flags), dtype=bool)
    durations = np.asarray(list(durations_ms), dtype=np.float64)
    if durations.size != mask.size:
        return float(mask.sum()) / 1000.0
    return float(durations[mask].sum()) / 1000.0


# --------------------------------------------------------------------------- #
# One clip's frames
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FrameFlags:
    """One entry per frame of an athlete parquet: the frame-level columns only.

    The parquet holds 33 rows per frame with ``n_people``, ``trainer_contact`` and
    ``detected`` repeated on every one of them, which is what makes this a
    reduction rather than a scan. The timestamps come along because every
    duration in the report is measured from them.
    """

    frame_idx: np.ndarray
    t_ms: np.ndarray
    n_people: np.ndarray
    detected: np.ndarray
    contact: np.ndarray
    durations_ms: np.ndarray

    @property
    def frames(self) -> int:
        """How many frames the clip has."""
        return int(self.frame_idx.size)

    @property
    def second_person(self) -> np.ndarray:
        """Per frame: was there a second person in it, after dedup?"""
        return self.n_people >= SECOND_PERSON_MIN_PEOPLE

    @property
    def dropped(self) -> np.ndarray:
        """Per frame: was the frame written as not detected?"""
        return ~self.detected

    @property
    def fps(self) -> float:
        """The clip's frame rate, as the median gap between its timestamps.

        Variable frame rate is the norm here (the clips come off WhatsApp), so the
        median is the honest summary and the mean is not. A clip whose timestamps
        carry no positive gap has no rate to report and says 0.0.
        """
        positive = self.durations_ms[self.durations_ms > 0.0]
        if not positive.size:
            return 0.0
        return 1000.0 / float(np.median(positive))

    @property
    def seconds(self) -> float:
        """How much video the frames cover, in seconds."""
        return float(self.durations_ms.sum()) / 1000.0


def frame_flags(table: pd.DataFrame) -> FrameFlags:
    """Reduce the 33 rows per frame of an athlete parquet to one entry per frame.

    The first row of each frame carries that frame's ``n_people``,
    ``trainer_contact`` and ``detected`` (they are repeated on all 33 rows), and
    the frames are sorted by index first, so a parquet that is not in frame order
    still comes out in time order. A table without the frame-level columns is a
    single-person parquet or the wrong file, and raises rather than being read as
    "no trainer anywhere".
    """
    required = (
        "frame_idx",
        "t_ms",
        "detected",
        athlete.N_PEOPLE_COLUMN,
        athlete.CONTACT_COLUMN,
    )
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(
            f"athlete keypoints are missing column(s) {missing}; "
            "re-run `python -m handstand.athlete` (see docs/keypoint_schema.md)"
        )
    per_frame = (
        table.sort_values("frame_idx", kind="stable").groupby("frame_idx", sort=True).head(1)
    )
    if per_frame.empty:
        raise ValueError("athlete keypoints: no frames in the parquet")
    t_ms = per_frame["t_ms"].to_numpy(dtype=np.int64)
    return FrameFlags(
        frame_idx=per_frame["frame_idx"].to_numpy(dtype=np.int64),
        t_ms=t_ms,
        n_people=per_frame[athlete.N_PEOPLE_COLUMN].to_numpy(dtype=np.int64),
        detected=per_frame["detected"].to_numpy(dtype=bool),
        contact=per_frame[athlete.CONTACT_COLUMN].to_numpy(dtype=bool),
        durations_ms=frame_durations_ms(t_ms),
    )


# --------------------------------------------------------------------------- #
# One clip's numbers
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipMetrics:
    """Everything the report says about one clip.

    The counts are kept as counts and the percentages as properties, so the CSV
    is a straight rendering of the dataclass and a reader can check
    ``pct_trainer_contact`` against ``contact_frames / frames`` by eye.
    """

    clip_id: str
    filename: str = ""
    rotate: str = DEFAULT_ROTATE
    frames: int = 0
    fps: float = 0.0
    second_person_frames: int = 0
    second_person_seconds: float = 0.0
    contact_frames: int = 0
    longest_contact_run_s: float = 0.0
    athlete_frames: int = 0
    dropped_frames: int = 0
    contact_by_reason: dict[str, int] = dataclasses.field(default_factory=dict)
    trainer_present: bool = False
    generate_seconds: float = 0.0
    notes: str = ""
    error: str = ""

    @classmethod
    def failed(
        cls,
        clip_id: str,
        error: str,
        *,
        filename: str = "",
        rotate: str = DEFAULT_ROTATE,
        notes: str = "",
    ) -> ClipMetrics:
        """A clip nothing could be measured on: the error, and zeros everywhere else.

        The row is still written, so the report covers every clip in the
        catalogue and a failure is visible in the sheet instead of being a clip
        that silently is not there.
        """
        return cls(clip_id=clip_id, filename=filename, rotate=rotate, notes=notes, error=error)

    @property
    def seconds(self) -> float:
        """How much video this clip covers, in seconds: its frames at its own rate."""
        return self.frames / self.fps if self.fps > 0.0 else 0.0

    @property
    def pct_frames_second_person(self) -> float:
        """Share of frames a second person was in, after dedup."""
        return percentage(self.second_person_frames, self.frames)

    @property
    def pct_trainer_contact(self) -> float:
        """Share of frames flagged as trainer contact."""
        return percentage(self.contact_frames, self.frames)

    @property
    def pct_athlete(self) -> float:
        """Share of frames an athlete was attributed to."""
        return percentage(self.athlete_frames, self.frames)

    @property
    def pct_dropped(self) -> float:
        """Share of frames written as not detected by the athlete selection."""
        return percentage(self.dropped_frames, self.frames)

    @property
    def reason_counts(self) -> dict[str, int]:
        """Flagged frames per contact rule, every rule in
        :data:`handstand.athlete.CONTACT_REASONS` present."""
        return {
            reason: int(self.contact_by_reason.get(reason, 0)) for reason in athlete.CONTACT_REASONS
        }

    @property
    def scorable_frames(self) -> int:
        """Frames an athlete was attributed to *and* which are not in contact.

        The frames that can actually be scored: everything else in the clip is
        either lost to the trainer or never attributed to anybody.
        """
        return self.athlete_frames - self.contact_frames


def clip_metrics(
    table: pd.DataFrame,
    *,
    clip_id: str,
    filename: str = "",
    rotate: str = DEFAULT_ROTATE,
    notes: str = "",
    contact_by_reason: Mapping[str, int] | None = None,
    generate_seconds: float = 0.0,
) -> ClipMetrics:
    """Measure one clip from its athlete parquet.

    ``trainer_present`` is the question "was a second person in this clip", and
    the answer it gives is a duration, not a frame count: somebody is in the clip
    when a second person is there for at least :data:`SECOND_PERSON_MIN_SECONDS`
    in total, however that is split up. A phantom detection in a single frame
    lasts a fraction of a second and is not a trainer; a trainer who walks past
    twice for a second each is.
    """
    frames = frame_flags(table)
    second_person = frames.second_person
    second_person_s = flagged_seconds(second_person, frames.durations_ms)
    return ClipMetrics(
        clip_id=clip_id,
        filename=filename,
        rotate=rotate,
        frames=frames.frames,
        fps=frames.fps,
        second_person_frames=int(second_person.sum()),
        second_person_seconds=second_person_s,
        contact_frames=int(frames.contact.sum()),
        longest_contact_run_s=longest_run_seconds(frames.contact, frames.durations_ms),
        athlete_frames=int(frames.detected.sum()),
        dropped_frames=int(frames.dropped.sum()),
        contact_by_reason=dict(contact_by_reason or {}),
        trainer_present=second_person_s >= SECOND_PERSON_MIN_SECONDS,
        generate_seconds=generate_seconds,
        notes=notes,
    )


def _read_sidecar(path: pathlib.Path) -> dict[str, object]:
    """One sidecar's contents, or an empty mapping when it is missing or unreadable.

    The parquets are the measurement; the sidecars only add provenance (the
    runtimes, the reason counts), so a sidecar that has gone missing must not
    fail a clip whose numbers are all there.
    """
    if not path.is_file():
        return {}
    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return contents if isinstance(contents, dict) else {}


def sidecar_runtime(*paths: pathlib.Path) -> float:
    """Total ``runtime_seconds`` over the sidecars that are there; 0.0 for the rest.

    The runtimes of the stages that produced the keypoints, summed, so the report
    can say what the whole dataset cost to generate however it was generated —
    one batch or twenty resumed ones.
    """
    total = 0.0
    for path in paths:
        value = _read_sidecar(path).get("runtime_seconds")
        if isinstance(value, (int, float)) and math.isfinite(value):
            total += float(value)
    return total


def measure_clip(
    parquet_path: pathlib.Path,
    *,
    clip_id: str,
    filename: str = "",
    rotate: str = DEFAULT_ROTATE,
    notes: str = "",
    sidecars: Sequence[pathlib.Path] = (),
) -> ClipMetrics:
    """Measure one clip from its athlete parquet, with the sidecars for provenance.

    ``sidecars`` are the JSON files that came with it — the athlete selection's
    own and the multi-person run's — and only two things are read out of them: the
    flagged frames per contact rule and the runtimes. Every count in the returned
    metrics is computed from the parquet, so the two can be compared.
    """
    table = pd.read_parquet(parquet_path)
    stage_sidecars = list(sidecars) or [parquet_path.with_suffix(".json")]
    return clip_metrics(
        table,
        clip_id=clip_id,
        filename=filename,
        rotate=rotate,
        notes=notes,
        contact_by_reason=_read_sidecar(stage_sidecars[0]).get("contact_frames_by_reason"),
        generate_seconds=sidecar_runtime(*stage_sidecars),
    )


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class CatalogueEntry:
    """One row of ``catalogue.csv``, as far as this report cares.

    Only three columns: the id every keypoint file is named after, the video the
    clip came from (``--generate`` needs it), and the human's ``notes`` — carried
    through untouched so the report can be lined up against the hand annotation.
    """

    clip_id: str
    filename: str
    notes: str = ""


def read_catalogue(path: str | pathlib.Path) -> list[CatalogueEntry]:
    """Read the catalogue's ``clip_id``, ``filename`` and ``notes`` columns, in file order.

    A row with no ``clip_id`` cannot be matched to a keypoint file, so it is
    skipped rather than reported under an empty id. A missing file raises
    :class:`FileNotFoundError` naming the command that writes it: without the
    catalogue there is no list of clips to report on.
    """
    catalogue = pathlib.Path(path)
    if not catalogue.is_file():
        raise FileNotFoundError(
            f"no catalogue at {catalogue}; build it first with:\n"
            "  cd pipeline && uv run python -m handstand.catalogue"
        )
    entries: list[CatalogueEntry] = []
    with catalogue.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = (row.get("clip_id") or "").strip()
            if not clip_id:
                continue
            entries.append(
                CatalogueEntry(
                    clip_id=clip_id,
                    filename=(row.get("filename") or "").strip(),
                    notes=(row.get(NOTES_COLUMN) or "").strip(),
                )
            )
    return entries


# --------------------------------------------------------------------------- #
# Step 1: generate what is missing
# --------------------------------------------------------------------------- #


def _print_line(message: str) -> None:
    """Print one line of progress and flush it.

    A batch of this length is watched through a redirected log, and Python's
    block buffering would hold the last few clips back until the buffer filled.
    """
    print(message, flush=True)


def _stale_multi_person(parquet_path: pathlib.Path) -> bool:
    """Was this clip's multi-person keypoints produced with other detector settings?

    A run on MediaPipe's own defaults — VIDEO mode, both thresholds at 0.5 — keeps
    reporting one person per frame even with the trainer standing right there, so a
    clip left over from such a run would measure as "no trainer in this one, ever"
    and the report would quietly understate the whole dataset. The sidecar records
    the settings exactly when they differ from those defaults, which is how a
    mismatched file is recognised; a sidecar that is missing or unreadable counts
    as mismatched too, because nothing about it can be trusted.

    A parquet that is not there at all is *missing*, not stale: it is generated
    either way, and saying ``False`` keeps the two cases from being confused.
    """
    if not parquet_path.is_file():
        return False
    sidecar = _read_sidecar(parquet_path.with_suffix(".json"))
    poses = int(sidecar.get("num_poses", pose_mediapipe.DEFAULT_NUM_POSES) or 0)
    if poses != NUM_POSES:
        return True
    fields = dataclasses.asdict(pose_mediapipe.DEFAULT_DETECTOR_SETTINGS)
    recorded = {name: sidecar[name] for name in fields if name in sidecar}
    if not recorded:  # a run on the defaults records none of the four
        return not DETECTOR_SETTINGS.is_default
    return pose_mediapipe.DetectorSettings(**recorded) != DETECTOR_SETTINGS


def generate_clip(
    entry: CatalogueEntry,
    *,
    keypoints_root: pathlib.Path,
    videos: pathlib.Path,
    rotate: str = DEFAULT_ROTATE,
    model_path: str | pathlib.Path = pose_mediapipe.DEFAULT_MODEL_PATH,
    overwrite: bool = False,
) -> str:
    """Generate the multi-person keypoints and the athlete selection for one clip.

    The two stages are the ones the two CLIs run —
    :func:`handstand.pose_mediapipe.run_clip` with :data:`NUM_POSES` people per
    frame in :data:`DETECTOR_SETTINGS`, then :func:`handstand.athlete.run_clip` —
    so this writes exactly what::

        uv run python -m handstand.pose_mediapipe --rotate auto --num-poses 3 \\
            --running-mode image --min-detection 0.2 --min-presence 0.2 --clips VIDEO
        uv run python -m handstand.athlete --rotate auto --clips CLIP_ID

    write, in one process instead of two and one clip at a time. Both stages skip
    a clip whose parquet already exists unless ``overwrite``, which is what makes
    an interrupted batch resumable — with one exception, :func:`_stale_multi_person`:
    a clip whose keypoints were made with *other* detector settings is re-run,
    because the report this module writes is a statement about the dataset measured
    this way, and a single-person run reports every trainer in a clip as absent.

    Returns ``""`` when the clip is ready and the error text when it is not;
    nothing is raised, so the caller can carry on with the next clip.
    """
    video_path = videos / entry.filename
    if not entry.filename or not video_path.is_file():
        return f"missing video: {video_path}"
    multi_dir = keypoints_root / pose_mediapipe.output_dirname(NUM_POSES) / rotate
    refresh = overwrite or _stale_multi_person(multi_dir / f"{entry.clip_id}.parquet")
    try:
        pose_mediapipe.run_clip(
            video_path,
            clip_id=entry.clip_id,
            rotate_mode=rotate,
            out_root=keypoints_root / pose_mediapipe.output_dirname(NUM_POSES),
            landmarker_factory=pose_mediapipe.make_landmarker_factory(
                model_path, NUM_POSES, DETECTOR_SETTINGS
            ),
            overwrite=refresh,
            num_poses=NUM_POSES,
            settings=DETECTOR_SETTINGS,
        )
        athlete.run_clip(
            entry.clip_id,
            rotate_mode=rotate,
            in_root=keypoints_root / pose_mediapipe.MULTI_OUTPUT_DIRNAME,
            out_root=keypoints_root / athlete.output_dirname(),
            overwrite=refresh,
        )
    except Exception as error:  # one clip must not stop the batch
        return f"{type(error).__name__}: {error}"
    return ""


def generate_batch(
    entries: Sequence[CatalogueEntry],
    *,
    keypoints_root: pathlib.Path,
    videos: pathlib.Path,
    rotate: str = DEFAULT_ROTATE,
    model_path: str | pathlib.Path = pose_mediapipe.DEFAULT_MODEL_PATH,
    overwrite: bool = False,
    progress: Callable[[str], None] = _print_line,
    generate: Callable[..., str] = generate_clip,
) -> dict[str, str]:
    """Make sure every catalogue clip has both stages generated; collect the failures.

    Returns ``{clip_id: error}`` for the clips that failed and nothing for the rest.
    A clip that raises is reported and the batch moves on: the point of the run is
    to cover the dataset, and a single undecodable video must not cost the other
    179 clips their report.
    """
    failures: dict[str, str] = {}
    total = len(entries)
    for position, entry in enumerate(entries, start=1):
        started = time.perf_counter()
        try:
            error = generate(
                entry,
                keypoints_root=keypoints_root,
                videos=videos,
                rotate=rotate,
                model_path=model_path,
                overwrite=overwrite,
            )
        except Exception as unexpected:  # a generator that raises is still one clip
            error = f"{type(unexpected).__name__}: {unexpected}"
        elapsed = time.perf_counter() - started
        if error:
            failures[entry.clip_id] = error
            progress(f"[{position:>4}/{total}] fail  {entry.clip_id} ({elapsed:.1f}s): {error}")
        else:
            progress(f"[{position:>4}/{total}] ok    {entry.clip_id} ({elapsed:.1f}s)")
    return failures


# --------------------------------------------------------------------------- #
# Step 2: the CSV
# --------------------------------------------------------------------------- #


def _row(metrics: ClipMetrics) -> dict[str, object]:
    """One CSV row of a clip: the counts, the percentages, the reasons, the error."""
    row: dict[str, object] = {
        "clip_id": metrics.clip_id,
        "filename": metrics.filename,
        "rotate": metrics.rotate,
        "frames": metrics.frames,
        "fps": round(metrics.fps, FLOAT_DIGITS),
        "second_person_frames": metrics.second_person_frames,
        "pct_frames_second_person": round(metrics.pct_frames_second_person, FLOAT_DIGITS),
        "second_person_seconds": round(metrics.second_person_seconds, FLOAT_DIGITS),
        "contact_frames": metrics.contact_frames,
        "pct_trainer_contact": round(metrics.pct_trainer_contact, FLOAT_DIGITS),
        "longest_contact_run_s": round(metrics.longest_contact_run_s, FLOAT_DIGITS),
        "trainer_present": TRUE if metrics.trainer_present else FALSE,
        "athlete_frames": metrics.athlete_frames,
        "pct_athlete": round(metrics.pct_athlete, FLOAT_DIGITS),
        "dropped_frames": metrics.dropped_frames,
        "pct_dropped": round(metrics.pct_dropped, FLOAT_DIGITS),
        "generate_seconds": round(metrics.generate_seconds, FLOAT_DIGITS),
        NOTES_COLUMN: metrics.notes,
        "error": metrics.error,
    }
    for reason, count in metrics.reason_counts.items():
        row[REASON_COLUMNS[reason]] = count
    return row


def metrics_table(metrics: Sequence[ClipMetrics]) -> pd.DataFrame:
    """The per-clip table, in :data:`REPORT_COLUMNS` order, one row per clip.

    The rows keep the catalogue's order, so the CSV lines up with
    ``catalogue.csv`` and with the hand annotation next to it.
    """
    return pd.DataFrame([_row(clip) for clip in metrics], columns=list(REPORT_COLUMNS))


def reports_dir(data: str | pathlib.Path) -> pathlib.Path:
    """Where the two report files go: ``<data_dir>/reports``."""
    return pathlib.Path(data) / REPORTS_DIRNAME


def _write_atomic(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temp file beside it, then rename.

    Both report files are regenerated from the parquets, so an interrupted write
    costs a re-run; renaming still means a reader never sees half a file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_report(
    metrics: Sequence[ClipMetrics],
    *,
    data: str | pathlib.Path,
    generate_seconds: float = 0.0,
    report_seconds: float = 0.0,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write ``trainer_report.csv`` and ``trainer_report.md`` under ``<data_dir>/reports``.

    Returns the two paths, in that order. Neither is committed: both are a
    rendering of the parquets and are rebuilt by re-running the report.
    ``generate_seconds`` and ``report_seconds`` are the two halves of this run's
    runtime, timed separately so the summary can say which is which — they are
    different by four orders of magnitude, and calling the whole of a
    ``--generate`` run "this report" would read as though reading the parquets
    took as long as detecting poses in them.
    """
    out_dir = reports_dir(data)
    csv_path = out_dir / CSV_FILENAME
    markdown_path = out_dir / MARKDOWN_FILENAME
    buffer = io.StringIO()
    metrics_table(metrics).to_csv(buffer, index=False, lineterminator="\n")
    _write_atomic(csv_path, buffer.getvalue())
    _write_atomic(
        markdown_path,
        summarise(metrics, generate_seconds=generate_seconds, report_seconds=report_seconds),
    )
    return csv_path, markdown_path


# --------------------------------------------------------------------------- #
# Step 3: the summary
# --------------------------------------------------------------------------- #


def contact_histogram(
    percents: Sequence[float], *, width: float = HISTOGRAM_BUCKET_PERCENT
) -> list[tuple[str, int]]:
    """Clips per bucket of ``pct_trainer_contact``, as ``(label, clips)``, in order.

    The buckets are ``0-10 %``, ``10-20 %``, ... and the top one is closed at
    100 %, so a clip that is in contact in every frame lands in ``90-100 %``
    rather than off the end of the table. Every bucket is listed, including the
    empty ones, so the shape of the histogram does not depend on the data.
    """
    edges = np.linspace(0.0, 100.0, int(round(100.0 / width)) + 1)
    counts = [0] * (edges.size - 1)
    for percent in percents:
        if not math.isfinite(percent) or percent < 0.0:
            continue
        counts[min(int(percent // width), len(counts) - 1)] += 1
    return [
        (f"{low:.0f}-{high:.0f}%", count)
        for low, high, count in zip(edges[:-1], edges[1:], counts, strict=True)
    ]


def _bar(count: int, longest: int, *, width: int = HISTOGRAM_BAR_WIDTH) -> str:
    """A bar of block characters, scaled so the largest bucket fills ``width``."""
    if longest <= 0 or count <= 0:
        return ""
    return "█" * max(1, round(width * count / longest))


def _fmt_int(value: int) -> str:
    """A whole number with thousands separators."""
    return f"{value:,}"


def _fmt_pct(value: float) -> str:
    """A percentage, to :data:`PERCENT_DIGITS` decimals."""
    return f"{value:.{PERCENT_DIGITS}f} %"


def _fmt_count(part: int, total: int) -> str:
    """``"1,234 of 5,678 (21.7 %)"`` — the shape every headline number takes."""
    return f"{_fmt_int(part)} of {_fmt_int(total)} ({_fmt_pct(percentage(part, total))})"


def _fmt_duration(seconds: float) -> str:
    """A duration in seconds, or in minutes once it is more than a minute of it."""
    if seconds >= 120.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.1f} s"


def _most_affected(
    metrics: Sequence[ClipMetrics], count: int = MOST_AFFECTED_CLIPS
) -> list[ClipMetrics]:
    """The clips with the most contact, worst first.

    Ranked by the share of frames in contact, then by the longest unbroken stretch
    of it (a clip that is in contact in more frames but only ever in one-frame
    touches is the less affected one), then by clip id so the order is stable
    between runs.
    """
    usable = [clip for clip in metrics if not clip.error]
    return sorted(
        usable,
        key=lambda clip: (-clip.pct_trainer_contact, -clip.longest_contact_run_s, clip.clip_id),
    )[:count]


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A markdown table: the header row, a rule, then the rows."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def summarise(
    metrics: Sequence[ClipMetrics], *, generate_seconds: float = 0.0, report_seconds: float = 0.0
) -> str:
    """The markdown summary: the headline numbers, the worst clips, the histogram.

    The runtimes are three different numbers and are kept apart: what the
    keypoints cost, summed from the runtimes the sidecars recorded however many
    batches produced them; what *this* run spent in ``--generate`` (nothing, on a
    report-only run); and what this run spent reading the parquets and writing
    the two files. Their sum is the total for the run. Clips that failed carry no
    numbers, so they are listed rather than counted into any percentage — every
    share below is "of the clips that were measured".
    """
    usable = [clip for clip in metrics if not clip.error]
    failed = [clip for clip in metrics if clip.error]
    with_trainer = [clip for clip in usable if clip.trainer_present]
    clean = len(usable) - len(with_trainer)
    frames = sum(clip.frames for clip in usable)
    second_person = sum(clip.second_person_frames for clip in usable)
    contact = sum(clip.contact_frames for clip in usable)
    dropped = sum(clip.dropped_frames for clip in usable)
    scorable = sum(clip.scorable_frames for clip in usable)
    video_s = sum(clip.seconds for clip in usable)
    keypoints_s = sum(clip.generate_seconds for clip in usable)
    reasons = {
        reason: sum(clip.reason_counts[reason] for clip in usable)
        for reason in athlete.CONTACT_REASONS
    }
    rotate = metrics[0].rotate if metrics else DEFAULT_ROTATE
    histogram = contact_histogram([clip.pct_trainer_contact for clip in usable])
    longest = max((count for _, count in histogram), default=0)
    threshold = _fmt_duration(SECOND_PERSON_MIN_SECONDS)

    out: list[str] = [
        "# Trainer presence across the dataset",
        "",
        f"Measured {_fmt_int(len(usable))} of {_fmt_int(len(metrics))} catalogue clips: "
        f"{_fmt_int(frames)} frames, {_fmt_duration(video_s)} of video"
        + (f". {_fmt_int(len(failed))} clips carry no numbers (see *Failures*)" if failed else "")
        + ".",
        "",
        f"Keypoints from `handstand.pose_mediapipe --rotate {rotate} --num-poses {NUM_POSES} "
        f"--running-mode {DETECTOR_SETTINGS.running_mode} "
        f"--min-detection {DETECTOR_SETTINGS.min_detection} "
        f"--min-presence {DETECTOR_SETTINGS.min_presence}` and "
        f"`handstand.athlete --rotate {rotate}`. Durations are measured from the frame "
        "timestamps (these clips are variable frame rate).",
        "",
        "## Headline",
        "",
        _markdown_table(
            ("measure", "value"),
            (
                (
                    f"clips with a trainer in them (`trainer_present`: a second person for "
                    f"{threshold} or more in total)",
                    f"**{_fmt_count(len(with_trainer), len(usable))}**",
                ),
                ("clips with no trainer detected", _fmt_count(clean, len(usable))),
                ("frames with a second person (after dedup)", _fmt_count(second_person, frames)),
                ("frames flagged trainer contact", _fmt_count(contact, frames)),
                ("frames dropped by the athlete selection", _fmt_count(dropped, frames)),
                ("frames usable as-is (attributed, not in contact)", _fmt_count(scorable, frames)),
            ),
        ),
        "",
        "## What that means for the dataset",
        "",
        _markdown_table(
            ("clips", "count", "share"),
            (
                (
                    "usable as-is: no trainer detected, the plain single-person keypoints "
                    "are enough",
                    _fmt_int(clean),
                    _fmt_pct(percentage(clean, len(usable))),
                ),
                (
                    f"need the athlete-selection path: a trainer is in the clip ({threshold}+)",
                    _fmt_int(len(with_trainer)),
                    _fmt_pct(percentage(len(with_trainer), len(usable))),
                ),
                ("could not be measured", _fmt_int(len(failed)), "-"),
            ),
        ),
        "",
        _markdown_table(
            ("frames", "count", "share", "what happens to them"),
            (
                (
                    "attributed to the athlete, not in contact",
                    _fmt_int(scorable),
                    _fmt_pct(percentage(scorable, frames)),
                    "scored",
                ),
                (
                    "attributed, but in trainer contact",
                    _fmt_int(contact),
                    _fmt_pct(percentage(contact, frames)),
                    "lost: keypoints kept, never scored",
                ),
                (
                    "dropped: no athlete could be chosen",
                    _fmt_int(dropped),
                    _fmt_pct(percentage(dropped, frames)),
                    "lost: no athlete in them",
                ),
                (
                    "total",
                    _fmt_int(frames),
                    "100.0 %",
                    "",
                ),
            ),
        ),
        "",
        "## Contact per clip",
        "",
        f"`pct_trainer_contact` of each of the {_fmt_int(len(usable))} measured clips, "
        f"in {HISTOGRAM_BUCKET_PERCENT:.0f} % buckets:",
        "",
        "```",
    ]
    out += [
        f"{label} | {_bar(count, longest):<{HISTOGRAM_BAR_WIDTH}} {count}"
        for label, count in histogram
    ]
    out += [
        "```",
        "",
        "Flagged frames per contact rule: "
        + ", ".join(f"`{reason}` {_fmt_int(reasons[reason])}" for reason in athlete.CONTACT_REASONS)
        + ".",
        "",
        f"## The {MOST_AFFECTED_CLIPS} most affected clips",
        "",
    ]
    worst = _most_affected(usable)
    out.append(
        _markdown_table(
            ("clip_id", "frames", "second person", "contact", "longest run", "dropped", "notes"),
            [
                (
                    f"`{clip.clip_id}`",
                    _fmt_int(clip.frames),
                    _fmt_pct(clip.pct_frames_second_person),
                    _fmt_pct(clip.pct_trainer_contact),
                    f"{clip.longest_contact_run_s:.1f} s",
                    _fmt_pct(clip.pct_dropped),
                    clip.notes or "-",
                )
                for clip in worst
            ],
        )
        if worst
        else "_No clip could be measured._"
    )
    out += [
        "",
        "## Runtime",
        "",
        f"- keypoints and athlete selection, as recorded in the sidecars: "
        f"{_fmt_duration(keypoints_s)}",
    ]
    if generate_seconds > 0.0:
        out.append(f"- generation in this run: {_fmt_duration(generate_seconds)}")
    out += [
        f"- this report, reading the parquets and writing these two files: "
        f"{_fmt_duration(report_seconds)}",
        f"- total for this run: {_fmt_duration(generate_seconds + report_seconds)}",
        "",
        "## Failures",
        "",
    ]
    out += (
        [f"- `{clip.clip_id}`: {clip.error}" for clip in failed]
        if failed
        else ["_Every clip in the catalogue was measured._"]
    )
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.trainer_report`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.trainer_report",
        description=(
            "Report how much of the dataset the trainer is in, clip by clip, and "
            "over the whole dataset."
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "generate the multi-person keypoints and the athlete selection for "
            "every catalogue clip that is missing them first (minutes, not "
            "seconds; resumable, and separate from the report so the report can "
            "be re-run on its own)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-generate clips whose parquet already exists (with --generate)",
    )
    parser.add_argument(
        "--rotate",
        choices=pose_mediapipe.ROTATE_MODES,
        default=DEFAULT_ROTATE,
        help=f"rotation mode to read and write (default: {DEFAULT_ROTATE})",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help="clip ids to work on (default: every clip in the catalogue)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="work on at most N clips",
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory holding catalogue.csv and keypoints/ (default: $HANDSTAND_DATA)",
    )
    parser.add_argument(
        "--videos",
        type=pathlib.Path,
        default=None,
        help="videos directory, read only (default: $HANDSTAND_VIDEOS)",
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=pose_mediapipe.DEFAULT_MODEL_PATH,
        help=f"pose landmarker .task file (default: {pose_mediapipe.DEFAULT_MODEL_PATH})",
    )
    return parser


def _selected(entries: Sequence[CatalogueEntry], args: argparse.Namespace) -> list[CatalogueEntry]:
    """The catalogue rows this run works on, after ``--clips`` and ``--limit``.

    A ``--clips`` id that is not in the catalogue is a typo, and silently
    reporting on nothing would hide it, so it raises :class:`ValueError` and the
    CLI turns that into a usage error.
    """
    chosen = list(entries)
    if args.clips:
        wanted = set(args.clips)
        unknown = sorted(wanted - {entry.clip_id for entry in chosen})
        if unknown:
            raise ValueError(f"not in the catalogue: {', '.join(unknown)}")
        chosen = [entry for entry in chosen if entry.clip_id in wanted]
    if args.limit is not None:
        chosen = chosen[: args.limit]
    return chosen


def _measure(
    entries: Sequence[CatalogueEntry], *, data: pathlib.Path, rotate: str
) -> list[ClipMetrics]:
    """Measure every clip, turning a per-clip failure into an error row.

    A parquet that cannot be read, or a clip nobody generated, is one row with an
    error in it: the report still covers the whole catalogue, and the sheet says
    which clips are missing numbers and why.
    """
    keypoints = data / "keypoints"
    measured: list[ClipMetrics] = []
    for entry in entries:
        athlete_dir = keypoints / athlete.output_dirname() / rotate
        parquet_path = athlete_dir / f"{entry.clip_id}.parquet"
        sidecars = (
            athlete_dir / f"{entry.clip_id}.json",
            keypoints / pose_mediapipe.MULTI_OUTPUT_DIRNAME / rotate / f"{entry.clip_id}.json",
        )
        try:
            if not parquet_path.is_file():
                raise FileNotFoundError(
                    f"no athlete keypoints at {parquet_path} (re-run with --generate)"
                )
            metrics = measure_clip(
                parquet_path,
                clip_id=entry.clip_id,
                filename=entry.filename,
                rotate=rotate,
                notes=entry.notes,
                sidecars=sidecars,
            )
        except Exception as error:  # one clip must not stop the report
            metrics = ClipMetrics.failed(
                entry.clip_id,
                f"{type(error).__name__}: {error}",
                filename=entry.filename,
                rotate=rotate,
                notes=entry.notes,
            )
        measured.append(metrics)
    return measured


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.trainer_report``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    data = (args.data or data_dir()).expanduser()
    videos = (args.videos or videos_dir()).expanduser()
    try:
        entries = read_catalogue(data / CATALOGUE_FILENAME)
    except FileNotFoundError as error:
        print(f"trainer_report: {error}")
        return 2
    try:
        entries = _selected(entries, args)
    except ValueError as error:
        parser.error(str(error))
    if not entries:
        print("no clips to report on")
        return 0
    print(
        f"clips={len(entries)} data={data} rotate={args.rotate} "
        f"generate={'yes' if args.generate else 'no'}",
        flush=True,
    )

    failures: dict[str, str] = {}
    generate_seconds = 0.0
    if args.generate:
        started = time.perf_counter()
        failures = generate_batch(
            entries,
            keypoints_root=data / "keypoints",
            videos=videos,
            rotate=args.rotate,
            model_path=args.model,
            overwrite=args.overwrite,
        )
        generate_seconds = time.perf_counter() - started
        print(
            f"generated {len(entries) - len(failures)} of {len(entries)} clips, "
            f"{len(failures)} failed, {_fmt_duration(generate_seconds)}",
            flush=True,
        )

    report_started = time.perf_counter()
    metrics = _measure(entries, data=data, rotate=args.rotate)
    csv_path, markdown_path = write_report(
        metrics, data=data, generate_seconds=generate_seconds, report_seconds=0.0
    )
    report_seconds = time.perf_counter() - report_started
    # The summary quotes the runtimes, so the clock has to have stopped before it
    # can be rendered: it is written once more with the numbers in it. Only the
    # markdown quotes them, so the CSV is left as ``write_report`` wrote it.
    _write_atomic(
        markdown_path,
        summarise(metrics, generate_seconds=generate_seconds, report_seconds=report_seconds),
    )
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")
    print(
        summarise(metrics, generate_seconds=generate_seconds, report_seconds=report_seconds)
        .rstrip("\n")
    )
    return 1 if failures or any(clip.error for clip in metrics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
