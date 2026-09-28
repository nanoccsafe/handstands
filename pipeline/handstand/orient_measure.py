"""Measure what a rotation mode changed, over every clip of the dataset.

``--rotate best`` is chosen by :mod:`handstand.pose_mediapipe` and the claim
made for it is a measurement, so the measurement has to be runnable and the
numbers in ``docs/keypoint_schema.md`` have to come out of it rather than out of
a notebook. This module is that measurement:

* **flicker** — how many times a clip switches orientation, and how long the
  runs of one orientation are. A body does not turn upside down between two
  frames, so a mode whose changes come in runs of one or two frames is not
  deciding anything, it is following the model's confidence frame by frame
  (chainlink #79, lead review);
* **inverted share** — the share of frames per clip whose skeleton reads as a
  handstand (mean wrist ``y`` greater than mean ankle ``y``), per mode;
* **fixed / broken** — against a baseline mode, over the frames both modes
  detected: a frame the candidate reads as a handstand and the baseline does
  not is *fixed*, the other way round is *broken*, and the runs of changed
  frames say whether the changes are decisions or flickers;
* **the chain** — usable clips and frames, clips with at least one hold, hold
  time and trainer presence, read from the outputs of ``handstand.postprocess``,
  ``handstand.phases`` and ``handstand.trainer_report`` on disk.

Nothing here runs a model and nothing is written: it reads the parquets and
sidecars the pipeline already produced.

Definitions, so two people reading the same output count the same frames:

* the **athlete pick** (default) reads ``keypoints/mediapipe_athlete/<mode>/``,
  one body per frame — the skeleton everything downstream uses. ``--pick multi``
  reads ``keypoints/mediapipe_multi/<mode>/`` and takes, per frame, the person
  whose wrists sit lowest in the image (:func:`handstand.pose_mediapipe.lowest_wrist_pose`);
* the **inverted share** of a mode is over the frames that mode detected, so a
  mode is not charged for the frames it found nobody in;
* **fixed** and **broken** are over the frames *both* modes detected, and a
  frame neither could judge is counted as "not inverted" by both and therefore
  never counts as either.

CLI::

    cd pipeline
    uv run python -m handstand.orient_measure                       # best vs auto
    uv run python -m handstand.orient_measure --pick multi
    uv run python -m handstand.orient_measure --baseline 180 --candidate best
    uv run python -m handstand.orient_measure --clips 438c3693d6d7 --no-chain
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from handstand.paths import data_dir
from handstand.pose_mediapipe import (
    ANKLE_JOINTS,
    JOINT_INDEX,
    JOINT_NAMES,
    NO_PERSON_IDX,
    PERSON_COLUMN,
    RECOMMENDED_ROTATE,
    WRIST_JOINTS,
)

__all__ = [
    "DEFAULT_BASELINE",
    "PICKS",
    "ChainSnapshot",
    "ClipComparison",
    "Flicker",
    "build_arg_parser",
    "chain_snapshot",
    "changed_runs",
    "compare_clip",
    "frame_reads",
    "inverted_share",
    "main",
    "orientation_changes",
    "orientation_segments",
    "read_clip",
    "render_report",
    "short_segments",
    "snapshot_rows",
]

#: The mode the dataset was first measured with, and the one every claim about
#: a new mode is a change *from*.
DEFAULT_BASELINE = "auto"
#: Which keypoints a frame's reading comes from.
PICKS: tuple[str, ...] = ("athlete", "multi")
#: The keypoints directory each pick reads, under ``keypoints/``.
PICK_DIRNAME: dict[str, str] = {"athlete": "mediapipe_athlete", "multi": "mediapipe_multi"}
#: A run of one orientation this many frames long or shorter is a flicker, not a
#: decision: at the clips' 30 fps it is under a tenth of a second.
FLICKER_RUN_FRAMES = 3

_LANDMARK_FIELDS = ["x", "y", "z", "visibility", "presence"]
_WRIST_ROWS = [JOINT_INDEX[name] for name in WRIST_JOINTS]
_ANKLE_ROWS = [JOINT_INDEX[name] for name in ANKLE_JOINTS]


# --------------------------------------------------------------------------- #
# What a frame of a clip reads as
# --------------------------------------------------------------------------- #


def _lowest_wrist_person(poses: np.ndarray) -> np.ndarray:
    """Of a frame's ``(P, 33, 5)`` poses, the one whose wrists sit lowest.

    Same rule as :func:`handstand.pose_mediapipe.lowest_wrist_pose` — the
    largest mean wrist ``y``, ignoring the people whose wrists are not both
    visible, and the first person when nobody can be ranked — written on the
    padded grid :func:`frame_reads` builds, where a missing person is all-NaN.
    """
    wrists = poses[:, _WRIST_ROWS, 1]
    means = _nanmean_rows(wrists)
    rankable = np.flatnonzero(np.isfinite(means))
    if rankable.size == 0:
        return poses[0]
    return poses[int(rankable[np.nanargmax(means[rankable])])]


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    """Mean over the last axis, skipping NaN, NaN where the whole row is.

    ``np.nanmean`` raises a ``RuntimeWarning`` on an all-NaN slice, and a frame
    with nobody in it is normal rather than exceptional here, so the mean is
    written out.
    """
    usable = np.isfinite(values)
    counts = usable.sum(axis=-1)
    totals = np.where(usable, values, 0.0).sum(axis=-1)
    return np.divide(
        totals, counts, out=np.full(totals.shape, np.nan, dtype=np.float64), where=counts > 0
    )


def frame_reads(table: pd.DataFrame, pick: str = "athlete") -> pd.DataFrame:
    """One row per frame: what the model read there, and which way round.

    ``table`` is a long keypoints parquet — the athlete selection's schema (one
    body per frame) or the multi-person schema, in which case each frame is
    reduced to the person whose wrists sit lowest (:func:`_lowest_wrist_person`).
    The result has one row per frame in frame order with:

    ``frame_idx``, ``t_ms``
        copied from the first row of the frame;
    ``detected``
        whether anybody was found in the frame at all;
    ``rotated``
        the orientation the frame was read in, i.e. the run's ``rotated`` flag;
    ``wrist_y``, ``ankle_y``
        mean ``y`` of each pair, over whichever of the two joints is present
        (NaN when neither is);
    ``inverted``
        mean wrist ``y`` greater than mean ankle ``y``, i.e. a body on its
        hands — and ``False`` where either is missing, because a frame with no
        readable wrists cannot be judged (the same rule
        :func:`handstand.pose_mediapipe.is_inverted` uses).
    """
    if pick not in PICKS:
        raise ValueError(f"pick must be one of {PICKS}, got {pick!r}")
    if "frame_idx" not in table.columns or "joint" not in table.columns:
        raise ValueError("expected a long keypoints table with frame_idx and joint")

    has_people = PERSON_COLUMN in table.columns
    people = (
        table[PERSON_COLUMN].to_numpy()
        if has_people
        else np.zeros(len(table), dtype=np.int64)
    )
    frame_of_row = table["frame_idx"].to_numpy()
    frames = np.unique(frame_of_row)
    frame_index = {frame: index for index, frame in enumerate(frames)}
    max_people = int(people.max()) + 1 if has_people else 1
    rows_of = np.array([JOINT_INDEX[name] for name in table["joint"].to_numpy()])
    frame_slot = np.array([frame_index[frame] for frame in frame_of_row])
    person_slot = (
        np.where(people == NO_PERSON_IDX, 0, people).astype(np.int64)
        if has_people
        else np.zeros(len(table), dtype=np.int64)
    )

    # A (frames, people, 33, 5) grid of the frame's landmarks, NaN where a frame
    # had fewer people than the widest frame in the clip.
    grid = np.full((frames.size, max_people, len(JOINT_NAMES), len(_LANDMARK_FIELDS)), np.nan)
    grid[frame_slot, person_slot, rows_of] = table[_LANDMARK_FIELDS].to_numpy(
        dtype=np.float64
    )

    if max_people > 1:
        picked = np.stack([_lowest_wrist_person(poses) for poses in grid])
    else:
        picked = grid[:, 0]
    wrist_y = _nanmean_rows(picked[:, _WRIST_ROWS, 1])
    ankle_y = _nanmean_rows(picked[:, _ANKLE_ROWS, 1])
    readable = np.isfinite(wrist_y) & np.isfinite(ankle_y)

    per_frame = table.groupby("frame_idx", sort=True)
    return pd.DataFrame(
        {
            "frame_idx": frames,
            "t_ms": per_frame["t_ms"].first().to_numpy(),
            "detected": per_frame["detected"].max().to_numpy().astype(bool),
            "rotated": per_frame["rotated"].max().to_numpy().astype(bool),
            "wrist_y": wrist_y,
            "ankle_y": ankle_y,
            "inverted": readable & (wrist_y > ankle_y),
        }
    )


def read_clip(root: pathlib.Path, clip_id: str, pick: str = "athlete") -> pd.DataFrame:
    """:func:`frame_reads` of ``<root>/<clip_id>.parquet``."""
    return frame_reads(pd.read_parquet(pathlib.Path(root) / f"{clip_id}.parquet"), pick)


# --------------------------------------------------------------------------- #
# Orientation flicker
# --------------------------------------------------------------------------- #


def _run_lengths(flags: np.ndarray, only_set: bool) -> list[int]:
    """Lengths of the runs of a boolean array, of the set runs only if asked."""
    edges = np.flatnonzero(np.diff(flags)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [flags.size]])
    lengths = (stops - starts).astype(int)
    if not only_set:
        return [int(length) for length in lengths]
    return [int(length) for length, flag in zip(lengths, flags[starts], strict=True) if flag]


def orientation_segments(rotated: Sequence[bool]) -> list[int]:
    """Lengths in frames of the runs of one orientation in a ``rotated`` column.

    A clip read entirely upright is one segment; a clip that switches twice is
    three. The shortest segments are the flicker: a body that is on its hands
    for one frame and upright for the next has not been read, it has been
    guessed twice.
    """
    flags = np.asarray(rotated, dtype=bool).reshape(-1)
    if flags.size == 0:
        return []
    return _run_lengths(flags, only_set=False)


def orientation_changes(rotated: Sequence[bool]) -> int:
    """How many times the orientation switches along a ``rotated`` column."""
    segments = orientation_segments(rotated)
    return max(len(segments) - 1, 0)


def short_segments(segments: Sequence[int], max_length: int = FLICKER_RUN_FRAMES) -> int:
    """How many runs of one orientation are ``max_length`` frames or shorter."""
    return sum(1 for length in segments if length <= max_length)


@dataclasses.dataclass(frozen=True)
class Flicker:
    """How often a mode switches orientation, and how long its runs are.

    ``auto`` switches orientation whenever the previous frame's skeleton says it
    should, which the model makes it do thousands of times over a dataset;
    ``best`` decides over the whole clip. A :class:`Flicker` counts a set of
    clips: :meth:`of` takes one clip's :func:`orientation_segments` and
    :meth:`add` pools another clip's in.
    """

    #: How many times the orientation switches, over all the clips.
    switches: int = 0
    #: Lengths in frames of the runs of one orientation, over all the clips.
    runs: tuple[int, ...] = ()

    @property
    def run_count(self) -> int:
        """How many runs of one orientation there are in total."""
        return len(self.runs)

    @property
    def short_runs(self) -> int:
        """Runs of :data:`FLICKER_RUN_FRAMES` frames or less, i.e. the flicker."""
        return short_segments(self.runs)

    @property
    def median_run(self) -> int:
        """Median run length in frames, 0 when there is no run."""
        return int(round(_percentile(self.runs, 50.0))) if self.runs else 0

    @property
    def p90_run(self) -> int:
        """90th percentile of the run lengths in frames, 0 when there is none."""
        return int(round(_percentile(self.runs, 90.0))) if self.runs else 0

    @property
    def max_run(self) -> int:
        """Longest run in frames, 0 when there is none."""
        return max(self.runs, default=0)

    def add(self, runs: Sequence[int]) -> Flicker:
        """The same counts with one clip's :func:`orientation_segments` added."""
        return Flicker(
            switches=self.switches + max(len(runs) - 1, 0),
            runs=(*self.runs, *runs),
        )

    @classmethod
    def of(cls, segments: Sequence[int]) -> Flicker:
        """The counts of one clip's :func:`orientation_segments`."""
        return cls().add(segments)

    def describe(self) -> str:
        """One line: switches, runs, and the flicker among them."""
        return (
            f"{self.switches} switches in {self.run_count} runs"
            f" (median {self.median_run} frames, p90 {self.p90_run},"
            f" max {self.max_run}); {self.short_runs} of them"
            f" {FLICKER_RUN_FRAMES} frames or less"
        )


def changed_runs(baseline: Sequence[bool], candidate: Sequence[bool]) -> list[int]:
    """Lengths of the runs of frames where two modes read the clip differently.

    The inputs are two ``inverted`` columns over the **same** frames. A run of
    one or two frames is the flicker the temporal rule exists to remove; a run
    of half a second or more is a decision it took. Only the runs where the two
    modes actually differ are returned, so the length of the agreeing stretch
    around them is not in the list.
    """
    left = np.asarray(baseline, dtype=bool).reshape(-1)
    right = np.asarray(candidate, dtype=bool).reshape(-1)
    if left.size != right.size:
        raise ValueError(
            f"expected one reading per frame in both, got {left.size} and {right.size}"
        )
    if left.size == 0:
        return []
    return _run_lengths(left != right, only_set=True)


# --------------------------------------------------------------------------- #
# One clip against another mode
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClipComparison:
    """What one clip's candidate mode changed against its baseline mode."""

    clip_id: str
    #: Share of frames the baseline read as a handstand, over the frames it detected.
    baseline_inverted_percent: float
    #: The same for the candidate.
    candidate_inverted_percent: float
    #: Frames only both modes detected, the ones a change is counted on.
    compared_frames: int
    #: Of those, read as a handstand by the candidate and not by the baseline.
    fixed: int
    #: Of those, read as a handstand by the baseline and not by the candidate.
    broken: int
    #: Frames each mode detected, and how many of them read as a handstand:
    #: the totals the dataset-wide share is made of.
    baseline_detected: int = 0
    candidate_detected: int = 0
    baseline_inverted: int = 0
    candidate_inverted: int = 0
    #: Lengths of the runs of frames the two modes read differently.
    changed_runs: tuple[int, ...] = ()
    #: Lengths of the runs of frames each of those changed, split by direction,
    #: because a fix and a loss are not the same thing to look at.
    fixed_runs: tuple[int, ...] = ()
    broken_runs: tuple[int, ...] = ()

    @property
    def moved_points(self) -> float:
        """Change in the inverted share, in percentage points."""
        return self.candidate_inverted_percent - self.baseline_inverted_percent

    @property
    def unchanged(self) -> bool:
        """Do the two modes read the clip the same way?"""
        return self.moved_points == 0.0


def inverted_share(reads: pd.DataFrame) -> float:
    """Percentage of the frames ``reads`` detected that read as a handstand."""
    detected = reads[reads["detected"]]
    if detected.empty:
        return 0.0
    return 100.0 * float(detected["inverted"].mean())


def compare_clip(
    clip_id: str,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> ClipComparison:
    """Compare two modes' readings of one clip, frame by frame.

    ``baseline`` and ``candidate`` are two :func:`frame_reads` tables over the
    same frames. Only the frames **both** modes detected are compared: a frame
    one of them found nobody in says nothing about which way round the body is,
    and counting it would charge the mode that finds bodies for it.
    """
    if not baseline["frame_idx"].equals(candidate["frame_idx"]):
        raise ValueError(f"{clip_id}: the two modes do not cover the same frames")
    both = (baseline["detected"] & candidate["detected"]).to_numpy()
    base_inverted = (baseline["inverted"] & baseline["detected"]).to_numpy()
    cand_inverted = (candidate["inverted"] & candidate["detected"]).to_numpy()
    fixed = ~base_inverted & cand_inverted & both
    broken = base_inverted & ~cand_inverted & both
    return ClipComparison(
        clip_id=clip_id,
        baseline_inverted_percent=inverted_share(baseline),
        candidate_inverted_percent=inverted_share(candidate),
        compared_frames=int(both.sum()),
        fixed=int(fixed.sum()),
        broken=int(broken.sum()),
        baseline_detected=int(baseline["detected"].sum()),
        candidate_detected=int(candidate["detected"].sum()),
        baseline_inverted=int(base_inverted.sum()),
        candidate_inverted=int(cand_inverted.sum()),
        changed_runs=tuple(changed_runs(base_inverted[both], cand_inverted[both])),
        fixed_runs=tuple(_run_lengths(fixed[both], only_set=True)),
        broken_runs=tuple(_run_lengths(broken[both], only_set=True)),
    )


# --------------------------------------------------------------------------- #
# What the chain made of the keypoints
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ChainSnapshot:
    """What the stages after the keypoints produced, read off their outputs."""

    #: Clips ``handstand.postprocess`` could measure a body length on.
    usable_clips: int
    #: Frames of those clips, i.e. every frame of the dataset except the
    #: unusable clips' — the "usable frames" of the pipeline's own reports.
    usable_frames: int
    #: Frames of every clip the athlete selection wrote.
    total_frames: int
    #: Clips ``handstand.phases`` found at least one hold in.
    hold_clips: int
    #: Seconds of hold over all of them.
    hold_seconds: float
    #: Clips ``handstand.trainer_report`` found a trainer in, or None when the
    #: report is not there.
    trainer_clips: int | None = None
    #: Frames the same report flagged trainer contact, or None.
    contact_frames: int | None = None


def chain_snapshot(
    root: pathlib.Path | None = None,
    source: str = "mediapipe",
    report_csv: pathlib.Path | None = None,
) -> ChainSnapshot:
    """Usable clips and frames, holds and trainer presence, from what is on disk.

    Reads ``<data_dir>/processed/<source>/*.json`` for the usable clips,
    ``<data_dir>/phases/<source>/segments.csv`` for the holds and
    ``<data_dir>/reports/trainer_report.csv`` for trainer presence — the three
    outputs the chain writes, so nothing is re-computed here and a number in the
    docs is a number the pipeline can be asked for again.
    """
    base = pathlib.Path(root) if root is not None else data_dir()
    usable_clips = 0
    usable_frames = 0
    total_frames = 0
    for sidecar in sorted((base / "processed" / source).glob("*.json")):
        report = json.loads(sidecar.read_text())
        total_frames += int(report["frame_count"])
        if report.get("body_length_px"):
            usable_clips += 1
            usable_frames += int(report["frame_count"])

    segments_path = base / "phases" / source / "segments.csv"
    hold_clips = 0
    hold_seconds = 0.0
    if segments_path.is_file():
        segments = pd.read_csv(segments_path)
        holds = segments[segments["phase"] == "hold"]
        hold_clips = int(holds["clip_id"].nunique())
        hold_seconds = float(holds["duration_s"].sum())

    trainer_clips = None
    contact_frames = None
    path = report_csv if report_csv is not None else base / "reports" / "trainer_report.csv"
    if path.is_file():
        report = pd.read_csv(path)
        trainer_clips = int(report["trainer_present"].fillna(False).astype(bool).sum())
        contact_frames = int(report["contact_frames"].fillna(0).sum())

    return ChainSnapshot(
        usable_clips=usable_clips,
        usable_frames=usable_frames,
        total_frames=total_frames,
        hold_clips=hold_clips,
        hold_seconds=hold_seconds,
        trainer_clips=trainer_clips,
        contact_frames=contact_frames,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _percentile(values: Sequence[int], percentile: float) -> float:
    """``percentile`` of a sequence, or NaN when it is empty."""
    if not len(values):
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def snapshot_rows(rows: Sequence[ClipComparison]) -> dict[str, float]:
    """The dataset totals of a set of clips: shares, fixed, broken, runs.

    The one place the numbers ``docs/keypoint_schema.md`` quotes are put
    together, so they are computed once and from the same frames. The shares
    are **frame weighted** — every detected frame counts once, whatever clip it
    is in — because that is how the pipeline's own reports count.
    """
    changed = [length for row in rows for length in row.changed_runs]
    fixed_runs = [length for row in rows for length in row.fixed_runs]
    broken_runs = [length for row in rows for length in row.broken_runs]
    baseline_detected = sum(row.baseline_detected for row in rows)
    candidate_detected = sum(row.candidate_detected for row in rows)
    totals: dict[str, float] = {
        "clips": len(rows),
        "compared_frames": sum(row.compared_frames for row in rows),
        "fixed": sum(row.fixed for row in rows),
        "broken": sum(row.broken for row in rows),
        "better": sum(1 for row in rows if row.moved_points > 0.0),
        "worse": sum(1 for row in rows if row.moved_points < 0.0),
        "unchanged": sum(1 for row in rows if row.unchanged),
        "moved_ten_points": sum(1 for row in rows if abs(row.moved_points) > 10.0),
        "baseline_detected": baseline_detected,
        "candidate_detected": candidate_detected,
        "baseline_inverted_percent": _share(
            sum(row.baseline_inverted for row in rows), baseline_detected
        ),
        "candidate_inverted_percent": _share(
            sum(row.candidate_inverted for row in rows), candidate_detected
        ),
        "changed_runs": len(changed),
        "changed_run_median": _percentile(changed, 50.0),
        "changed_run_p90": _percentile(changed, 90.0),
        "changed_run_max": max(changed, default=0),
        "changed_runs_short": sum(1 for length in changed if length <= FLICKER_RUN_FRAMES),
    }
    for name, lengths in (("fixed", fixed_runs), ("broken", broken_runs)):
        totals[f"{name}_runs"] = len(lengths)
        totals[f"{name}_run_median"] = _run_stat(lengths, 50.0)
        totals[f"{name}_run_p90"] = _run_stat(lengths, 90.0)
        totals[f"{name}_run_max"] = max(lengths, default=0)
        totals[f"{name}_runs_short"] = sum(
            1 for length in lengths if length <= FLICKER_RUN_FRAMES
        )
    return totals


def _run_stat(lengths: Sequence[int], percentile: float) -> int:
    """A run-length percentile in whole frames, 0 when there are no runs."""
    if not len(lengths):
        return 0
    return int(round(_percentile(lengths, percentile)))


def _share(numerator: int, denominator: int) -> float:
    """``numerator`` of ``denominator`` as a percentage, 0.0 when there are none."""
    return 100.0 * numerator / denominator if denominator else 0.0


def render_report(
    baseline: str,
    candidate: str,
    pick: str,
    comparisons: Sequence[ClipComparison],
    baseline_flicker: Flicker,
    candidate_flicker: Flicker,
    snapshot: ChainSnapshot | None,
) -> str:
    """The report :func:`main` prints, as markdown."""
    totals = snapshot_rows(comparisons)
    lines = [
        f"# {candidate} against {baseline} ({pick} keypoints, {totals['clips']} clips)",
        "",
        "## Orientation flicker",
        "",
        f"- `{baseline}`: {baseline_flicker.describe()}",
        f"- `{candidate}`: {candidate_flicker.describe()}",
        "",
        "## What the model read",
        "",
        f"- frames read as a handstand:"
        f" {totals['baseline_inverted_percent']:.1f} % of"
        f" {int(totals['baseline_detected'])} detected (`{baseline}`) ->"
        f" {totals['candidate_inverted_percent']:.1f} % of"
        f" {int(totals['candidate_detected'])} detected (`{candidate}`)",
        f"- frames fixed / broken against `{baseline}`:"
        f" **{totals['fixed']} / {totals['broken']}**"
        f" of {totals['compared_frames']} frames both modes detected",
        f"- clips better / worse / unchanged: {totals['better']} / {totals['worse']}"
        f" / {totals['unchanged']}; moving more than 10 points:"
        f" {totals['moved_ten_points']}",
        f"- runs of **fixed** frames: {int(totals['fixed_runs'])}"
        f" (median {totals['fixed_run_median']:.0f}, p90 {totals['fixed_run_p90']:.0f},"
        f" max {int(totals['fixed_run_max'])},"
        f" {100.0 * totals['fixed_runs_short'] / max(totals['fixed_runs'], 1):.1f} %"
        f" of them {FLICKER_RUN_FRAMES} frames or less)",
        f"- runs of **broken** frames: {int(totals['broken_runs'])}"
        f" (median {totals['broken_run_median']:.0f}, p90 {totals['broken_run_p90']:.0f},"
        f" max {int(totals['broken_run_max'])},"
        f" {100.0 * totals['broken_runs_short'] / max(totals['broken_runs'], 1):.1f} %"
        f" of them {FLICKER_RUN_FRAMES} frames or less)",
        f"- runs of changed frames, both directions: {totals['changed_runs']}"
        f" (median {totals['changed_run_median']:.0f}"
        f", p90 {totals['changed_run_p90']:.0f}"
        f", max {totals['changed_run_max']}),"
        f" {100.0 * totals['changed_runs_short'] / max(totals['changed_runs'], 1):.1f} %"
        f" of them {FLICKER_RUN_FRAMES} frames or less — a short run here is"
        " usually the *baseline* flipping, not the candidate: it counts every"
        " frame where the two modes read a clip differently, and"
        f" `{baseline}` follows the previous frame",
    ]
    moved = sorted(
        (row for row in comparisons if row.moved_points),
        key=lambda row: -abs(row.moved_points),
    )
    if moved:
        lines += [
            "",
            "## Clips that moved",
            "",
            "| clip | " + f"`{baseline}` | `{candidate}` | fixed | broken |",
            "|---|---|---|---|---|",
        ]
        for row in moved:
            lines.append(
                f"| `{row.clip_id}` | {row.baseline_inverted_percent:.1f} %"
                f" | {row.candidate_inverted_percent:.1f} %"
                f" | {row.fixed} | {row.broken} |"
            )
    if snapshot is not None:
        lines += [
            "",
            "## What the chain made of it (on disk now)",
            "",
            f"- usable clips (a body length was measurable): **{snapshot.usable_clips}**",
            f"- usable frames: **{snapshot.usable_frames}** of {snapshot.total_frames}",
            f"- clips with at least one hold: **{snapshot.hold_clips}**",
            f"- hold time: **{snapshot.hold_seconds:.1f} s**",
        ]
        if snapshot.trainer_clips is not None:
            lines.append(
                f"- clips with a trainer in them: **{snapshot.trainer_clips}**"
                f" ({snapshot.contact_frames} contact frames)"
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m handstand.orient_measure",
        description="Measure what a rotation mode changed over the dataset's keypoints.",
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="mode to compare against")
    parser.add_argument(
        "--candidate",
        default=RECOMMENDED_ROTATE,
        help=f"mode under test (default: {RECOMMENDED_ROTATE})",
    )
    parser.add_argument(
        "--pick",
        choices=PICKS,
        default="athlete",
        help="athlete (one body per frame, the default) or multi (every person)",
    )
    parser.add_argument("--data", type=pathlib.Path, default=None, help="data directory")
    parser.add_argument("--source", default="mediapipe", help="source name (default: mediapipe)")
    parser.add_argument("--clips", nargs="+", default=None, metavar="CLIP_ID")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument(
        "--no-chain",
        action="store_true",
        help="skip the processed/phases/trainer_report part of the report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    base = pathlib.Path(args.data) if args.data is not None else data_dir()
    dirname = PICK_DIRNAME[args.pick]
    baseline_dir = base / "keypoints" / dirname / args.baseline
    candidate_dir = base / "keypoints" / dirname / args.candidate
    if not baseline_dir.is_dir() or not candidate_dir.is_dir():
        print(f"missing keypoints: {baseline_dir} or {candidate_dir}")
        return 1

    clip_ids = args.clips or sorted(path.stem for path in candidate_dir.glob("*.parquet"))
    if args.limit is not None:
        clip_ids = clip_ids[: args.limit]

    comparisons: list[ClipComparison] = []
    baseline_flicker = Flicker()
    candidate_flicker = Flicker()
    missing: list[str] = []
    for clip_id in clip_ids:
        if not (baseline_dir / f"{clip_id}.parquet").is_file():
            missing.append(clip_id)
            continue
        candidate = read_clip(candidate_dir, clip_id, args.pick)
        baseline = read_clip(baseline_dir, clip_id, args.pick)
        comparisons.append(compare_clip(clip_id, baseline, candidate))
        baseline_flicker = baseline_flicker.add(
            orientation_segments(baseline["rotated"].to_numpy())
        )
        candidate_flicker = candidate_flicker.add(
            orientation_segments(candidate["rotated"].to_numpy())
        )
    if missing:
        print(
            f"no `{args.baseline}` keypoints for {len(missing)} clip(s): "
            f"{', '.join(missing[:5])}"
        )

    snapshot = None if args.no_chain else chain_snapshot(base, args.source)
    report = render_report(
        args.baseline,
        args.candidate,
        args.pick,
        comparisons,
        baseline_flicker,
        candidate_flicker,
        snapshot,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
