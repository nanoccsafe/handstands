"""Pose-model bake-off: measure the candidates against the human labels (chainlink #16).

Three candidates are compared on the frames a human actually labelled:

* **mediapipe** — the pipeline's own athlete selection
  (``keypoints/mediapipe_athlete/best``); the oracle pool is the multi-person
  parquet ``keypoints/mediapipe_multi/best``.
* **vision** — Apple Vision, the on-device candidate, rotate ``auto``
  (``keypoints/vision_athlete/auto``; oracle pool ``vision_multi/auto``).
* **rtmpose** — rtmlib *balanced*, the reference upper bound, run fresh on the
  label-frame JPEGs with :func:`handstand.prelabel.make_model` +
  ``predict_frame`` **without a reference** (both orientations, the best score
  wins). It is not an on-device candidate.

Every model is reported **two ways** (the task's person matching):

``pipeline``
    the model's own choice: the athlete parquet's person for mediapipe and
    vision; for RTMPose the *inverted* person with the best score, else the
    best score — in a handstand the athlete is the inverted body, the trainer
    never is.
``oracle``
    among all people the model detected in the frame, the one with the
    smallest mean keypoint error to the ground truth. This isolates keypoint
    accuracy from person choice. The decision is made on ``pipeline``;
    ``oracle`` explains the gap.

Ground-truth rules (chainlink #16 comments, binding):

* ACCURACY (error, PCK, miss rate) is computed on **human-reviewed frames
  only**. The frames listed in ``labels/auto_accepted.csv`` carry unreviewed
  RTMPose pre-labels; they are reported separately as *agreement with RTMPose*
  — never as accuracy, and RTMPose itself is not scored there (it would be
  compared with itself).
* Clips whose catalogue row has ``missing=true`` are excluded everywhere
  (the duplicate 244275cd7059 must not be counted twice).
* ``trainer_contact`` is decided **once** by the MediaPipe multi-person mask
  (the ``trainer_contact`` column of the MediaPipe athlete parquet) and applied
  to every model's keypoints; the app model is chosen on the clean strata
  (``clean_no_trainer`` + ``clean_trainer``), and trainer-contact frames are
  reported only as a note.
* MediaPipe is never used as a reference for another model's correctness; the
  human labels are the only reference.

Metrics, over the 13 schema joints Vision also has (foot indexes are reported
separately for mediapipe/rtmpose only):

* ``L`` = ground-truth torso length, shoulder midpoint to hip midpoint. A frame
  with no GT shoulder or no GT hip is excluded from the normalised metrics and
  counted in the summary; side views label one side at a time and the two sides
  sit on top of each other, so each midpoint uses the shoulders/hips that exist.
* mean / median normalised error per joint group, PCK@0.2 and PCK@0.5 (a
  missed joint counts as a failure), miss rate (no output, or output below the
  backend's confidence gate), left/right swap rate on wrists and ankles,
  breakdowns per stratum, per phase, per shape, plus the multi-person split and
  the headline subset: **HOLD frames of LINE holds in the clean strata** — the
  app's use case, and what the decision is made on.
* 95 % intervals by bootstrap **over clips** (frames in a clip are correlated);
  the sample is small and the summary says so.

Per-backend visibility gate: error vs confidence is tabulated on the
human-reviewed pipeline joints and the lowest gate at which >= 90 % of the kept
joints fall within 0.2 L is recommended;
:func:`handstand.postprocess.process_clip` then runs over every non-missing
catalogue clip at ``MIN_VISIBILITY = 0.5`` (the value tuned on MediaPipe) and
at the recommended gate to count unusable clips (no body length). RTMPose has
no full-clip keypoints, so its gate is reported and its unusable-clip count is
marked not applicable.

CLI::

    cd pipeline
    uv run python -m handstand.bakeoff
    uv run python -m handstand.bakeoff --skip-rtmpose
    uv run python -m handstand.bakeoff --models mediapipe vision

Outputs land in ``<data_dir>/reports/bakeoff/*.csv`` (git-ignored) and a
summary is printed; ``docs/bakeoff.md`` holds the written decision.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import sys
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from handstand import postprocess
from handstand.hold_shapes import load_hold_shapes
from handstand.labels import KeypointRow
from handstand.labels import read_rows as read_label_rows
from handstand.paths import data_dir

__all__ = [
    "AGREEMENT_SCOPE",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CLEAN_STRATA",
    "FOOT_JOINTS",
    "GATE_GRID",
    "GATE_TARGET",
    "GROUP_ORDER",
    "JOINT_GROUPS",
    "LEGACY_GATE",
    "MAIN_JOINTS",
    "MEASUREMENT_MODELS",
    "MIN_SHARED_JOINTS",
    "MODES",
    "MODELS",
    "MULTI_PERSON_NOTE",
    "PIPELINE_MODE",
    "SCHEMA_JOINTS",
    "SELECTION_SCOPE",
    "SWAP_MIN_SEPARATION_L",
    "SWAP_SCORE_MARGIN",
    "SWAP_STRONG",
    "TRAINER_NOTE_SCOPE",
    "BakeoffResult",
    "annotate_frames",
    "bootstrap_ratio_ci",
    "build_arg_parser",
    "decide",
    "flag_multi_person_clips",
    "frames_table",
    "gate_sweep",
    "is_inverted",
    "load_auto_images",
    "load_frames",
    "load_pipeline_persons",
    "load_pools",
    "main",
    "miss_rate",
    "normalised_error",
    "oracle_person",
    "pck",
    "person_errors",
    "read_missing_clips",
    "read_multi_person_clips",
    "read_strata",
    "recommend_gate",
    "run",
    "run_rtmpose",
    "select_rtmpose_pipeline",
    "summary",
    "swap_report",
    "torso_length",
    "unusable_clip_counts",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The models the bake-off compares, in report order.
MODELS: tuple[str, ...] = ("mediapipe", "vision", "rtmpose")
#: Person-matching modes, in report order. The decision uses ``pipeline``.
MODES: tuple[str, ...] = ("pipeline", "oracle")
PIPELINE_MODE = "pipeline"
ORACLE_MODE = "oracle"

#: Keypoint roots under ``<data_dir>/keypoints/``: (single, multi, rotate) per
#: model. RTMPose has no parquets; it is predicted from the label JPEGs.
MODEL_DIRS: dict[str, tuple[str, str, str]] = {
    "mediapipe": ("mediapipe_athlete", "mediapipe_multi", "best"),
    "vision": ("vision_athlete", "vision_multi", "auto"),
}
#: Models whose keypoints exist over full clips (the gate's unusable-clip count
#: needs them). RTMPose runs on label frames only.
MEASUREMENT_MODELS: tuple[str, ...] = ("mediapipe", "vision")

#: The 13 schema joints every model is scored on — :data:`postprocess.CORE_JOINTS`,
#: the joints Vision also has.
SCHEMA_JOINTS: tuple[str, ...] = postprocess.CORE_JOINTS
#: MediaPipe's and RTMPose's two big toes; reported separately, never mixed
#: into the cross-model headline (Vision has no such joint).
FOOT_JOINTS: tuple[str, ...] = postprocess.FOOT_INDEX_JOINTS
#: What the headline, PCK and the gate sweep are computed over.
MAIN_JOINTS: tuple[str, ...] = SCHEMA_JOINTS
#: Which models produce which joints: vision stops at the ankle by design.
MODEL_JOINTS: dict[str, tuple[str, ...]] = {
    "mediapipe": (*SCHEMA_JOINTS, *FOOT_JOINTS),
    "vision": SCHEMA_JOINTS,
    "rtmpose": (*SCHEMA_JOINTS, *FOOT_JOINTS),
}

#: Joint groups, in the order the task lists them.
JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "wrists": ("left_wrist", "right_wrist"),
    "elbows": ("left_elbow", "right_elbow"),
    "shoulders": ("left_shoulder", "right_shoulder"),
    "hips": ("left_hip", "right_hip"),
    "knees": ("left_knee", "right_knee"),
    "ankles": ("left_ankle", "right_ankle"),
    "nose": ("nose",),
}
GROUP_ORDER: tuple[str, ...] = (
    "wrists",
    "elbows",
    "shoulders",
    "hips",
    "knees",
    "ankles",
    "nose",
    "foot_index",
)
_GROUP_OF: dict[str, str] = {
    joint: group for group, joints in JOINT_GROUPS.items() for joint in joints
}
_GROUP_OF.update({joint: "foot_index" for joint in FOOT_JOINTS})

#: The strata the app model is chosen on; the trainer stratum is a note.
CLEAN_STRATA: tuple[str, ...] = ("clean_no_trainer", "clean_trainer")
TRAINER_STRATUM = "trainer_contact"

#: What the summary table calls its slices.
SELECTION_SCOPE = "selection"
TRAINER_NOTE_SCOPE = "trainer_note"
AGREEMENT_SCOPE = "agreement_rtmpose"

#: PCK thresholds, in units of L.
PCK_LEVELS: tuple[float, ...] = (0.2, 0.5)

#: Candidate gates for the per-backend sweep.
GATE_GRID: tuple[float, ...] = tuple(
    float(round(value, 2)) for value in np.arange(0.0, 0.951, 0.05)
)
#: Recommend the lowest gate whose kept joints are at least this precise.
GATE_TARGET = 0.90
#: The pipeline's historical gate, also reported for comparison.
LEGACY_GATE = float(postprocess.MIN_VISIBILITY)

#: Fewest joints a person must share with the ground truth before their
#: keypoints can be judged — the same bar
#: :func:`handstand.prelabel.select_person` sets. Never more than the frame
#: itself has.
MIN_SHARED_JOINTS = 4

#: A wrists/ankles pair is only judged swappable when the two ground-truth
#: points are at least this far apart, in units of L: in a side view the sides
#: sit on top of each other, and there "swapped" would mean nothing.
SWAP_MIN_SEPARATION_L = 0.15
#: ... and the crossed assignment must beat the direct one by this margin, in
#: units of L, so noise on a near-symmetric pose cannot flap the rate.
SWAP_SCORE_MARGIN = 0.05
#: ... and by this share of (direct + cross): a genuine swap scores cross ~ 0
#: against direct ~ 2 x separation, which clears the bar easily.
SWAP_STRONG = 0.5

#: Bootstrap defaults: resample whole clips, frames in a clip move together.
BOOTSTRAP_SEED = 16
BOOTSTRAP_SAMPLES = 1000


# --------------------------------------------------------------------------- #
# Ground truth and frame metadata
# --------------------------------------------------------------------------- #


def read_missing_clips(data: str | pathlib.Path | None = None) -> set[str]:
    """Clip ids whose catalogue row says the video file is gone.

    Chainlink #49: the WhatsApp "(1)" duplicate 244275cd7059 is one of them and
    must not be counted twice, so every table drops these clips wholesale.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    path = root / "catalogue.csv"
    if not path.is_file():
        return set()
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "missing" not in table.columns:
        return set()
    flag = table["missing"].str.strip().str.lower().isin({"true", "1", "yes"})
    return set(table.loc[flag, "clip_id"])


def load_auto_images(data: str | pathlib.Path | None = None) -> set[str]:
    """The ``<clip>_<frame>.jpg`` names of the auto-accepted label frames.

    Those frames carry unreviewed RTMPose pre-labels (chainlink #47): they are
    reported as agreement with RTMPose and never as accuracy. A missing file is
    an empty set — a dataset with no auto frames simply has none.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    path = root / "labels" / "auto_accepted.csv"
    if not path.is_file():
        return set()
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "image" not in table.columns:
        return set()
    return set(table["image"].map(str))


def read_strata(data: str | pathlib.Path | None = None) -> dict[tuple[str, int], str]:
    """``(clip_id, frame_idx) -> stratum`` from the review queue."""
    root = pathlib.Path(data) if data is not None else data_dir()
    path = root / "labels" / "review_queue.csv"
    if not path.is_file():
        return {}
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"clip_id", "frame_idx", "stratum"} <= set(table.columns):
        return {}
    return {
        (str(row["clip_id"]), int(row["frame_idx"])): str(row["stratum"])
        for _, row in table.iterrows()
        if row["clip_id"] and row["stratum"]
    }


def read_multi_person_clips(data: str | pathlib.Path | None = None) -> set[str]:
    """Clips the trainer report saw a second person in (chainlink #15/#69)."""
    root = pathlib.Path(data) if data is not None else data_dir()
    path = root / "reports" / "trainer_report.csv"
    if not path.is_file():
        return set()
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"clip_id", "second_person_frames"} <= set(table.columns):
        return set()
    counts = pd.to_numeric(table["second_person_frames"], errors="coerce").fillna(0.0)
    return set(table.loc[counts > 0, "clip_id"])


#: What the catalogue note says about a clip a second person appeared in
#: (chainlink #15: flag multi-person clips so later stages can find them).
MULTI_PERSON_NOTE = "second person present"


def flag_multi_person_clips(data: str | pathlib.Path | None = None) -> list[str]:
    """Append :data:`MULTI_PERSON_NOTE` to the catalogue notes of those clips.

    Chainlink #15 asks for multi-person clips to be findable *in the
    catalogue itself*, not only in the trainer report, so the next stage that
    needs them reads one file. The note is appended (existing notes are kept),
    is idempotent — a clip already flagged is left untouched — and the file is
    written atomically by :func:`handstand.catalogue.write_rows`. Returns the
    clip ids that changed; an empty list when there is nothing to flag.
    """
    from handstand import catalogue

    root = pathlib.Path(data) if data is not None else data_dir()
    path = root / "catalogue.csv"
    if not path.is_file():
        return []
    multi_person = read_multi_person_clips(root)
    if not multi_person:
        return []
    rows = catalogue.read_rows(path)
    changed: list[str] = []
    for row in rows:
        clip = str(row.get("clip_id", ""))
        if clip not in multi_person:
            continue
        notes = str(row.get("notes", "") or "")
        if MULTI_PERSON_NOTE in notes:
            continue
        row["notes"] = f"{notes}; {MULTI_PERSON_NOTE}" if notes else MULTI_PERSON_NOTE
        changed.append(clip)
    if changed:
        catalogue.write_rows(rows, path)
    return changed


def frames_table(
    rows: Sequence[KeypointRow],
    *,
    auto_images: set[str],
    missing_clips: set[str],
) -> pd.DataFrame:
    """One row per labelled ``(clip, frame)`` with the review tags attached.

    ``human`` is false for an auto-accepted image, ``missing_clip`` true for a
    catalogue row whose video is gone, and ``keep`` is what every accuracy
    table draws from: human-reviewed **and** not on a missing clip. Pure, so
    both filters are unit-testable without touching the real labels.
    """
    columns = ["clip_id", "frame_idx", "image", "n_joints", "human", "missing_clip", "keep"]
    frame_rows: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        key = (row.clip_id, row.frame_idx)
        entry = frame_rows.setdefault(
            key,
            {
                "clip_id": row.clip_id,
                "frame_idx": row.frame_idx,
                "image": f"{row.clip_id}_{row.frame_idx}.jpg",
                "n_joints": 0,
            },
        )
        entry["n_joints"] = int(entry["n_joints"]) + 1
    if not frame_rows:
        return pd.DataFrame(columns=columns)
    frames = pd.DataFrame(frame_rows.values())
    frames["human"] = ~frames["image"].isin(auto_images)
    frames["missing_clip"] = frames["clip_id"].isin(missing_clips)
    frames["keep"] = frames["human"] & ~frames["missing_clip"]
    return frames.sort_values(["clip_id", "frame_idx"], ignore_index=True)


def annotate_frames(
    frames: pd.DataFrame,
    *,
    strata: Mapping[tuple[str, int], str],
    phases: Mapping[str, Mapping[int, tuple[str, int]]],
    shapes: Mapping[tuple[str, int | str], str],
    contact: Mapping[str, Mapping[int, bool]],
    multi_person_clips: set[str],
) -> pd.DataFrame:
    """Add stratum, phase, hold shape, the trainer mask and the report scopes.

    The mask is the **MediaPipe** multi-person ``trainer_contact`` flag, taken
    once and applied to every model's keypoints (chainlink #16 comments 25/26):
    ``in_selection`` is the clean single-person set the app model is chosen on,
    ``in_note`` the trainer-contact set reported only as a note, ``in_stratum``
    the mask-applied set the per-stratum table draws from, and ``in_agreement``
    the auto-accepted frames reported as agreement with RTMPose.
    """
    out = frames.copy()
    clip_ids = [str(clip) for clip in out.get("clip_id", [])]
    frame_indices = [int(frame) for frame in out.get("frame_idx", [])]
    pairs = list(zip(clip_ids, frame_indices, strict=True))
    out["stratum"] = [strata.get(pair, "") for pair in pairs]
    phase_values: list[str] = []
    hold_ids: list[int] = []
    contact_values: list[bool] = []
    for clip, frame in pairs:
        phase_frame = phases.get(clip, {}).get(frame)
        if phase_frame is None:
            phase_values.append("unknown")
            hold_ids.append(-1)
        else:
            phase_values.append(phase_frame[0])
            hold_ids.append(int(phase_frame[1]))
        contact_values.append(bool(contact.get(clip, {}).get(frame, False)))
    out["phase"] = phase_values
    out["hold_id"] = hold_ids
    out["mp_trainer_contact"] = contact_values
    out["multi_person"] = out["clip_id"].isin(multi_person_clips)
    out["shape"] = [
        "none" if hold < 0 else str(shapes.get((clip, hold), "unlabelled"))
        for clip, hold in zip(clip_ids, hold_ids, strict=True)
    ]
    clean = out["stratum"].isin(CLEAN_STRATA)
    out["in_selection"] = out["keep"] & clean & ~out["mp_trainer_contact"]
    out["in_note"] = out["keep"] & (
        (out["stratum"] == TRAINER_STRATUM) | out["mp_trainer_contact"]
    )
    out["in_stratum"] = out["keep"] & ~out["mp_trainer_contact"]
    out["in_agreement"] = ~out["human"] & ~out["missing_clip"]
    return out


def torso_length(gt: Mapping[str, tuple[float, float]]) -> float | None:
    """Ground-truth torso length L: shoulder midpoint to hip midpoint.

    Midpoints use the shoulders (hips) that exist: side views are labelled one
    side at a time and the two sides sit on top of each other, so one labelled
    shoulder *is* the midpoint there. A frame with no GT shoulder or no GT hip
    cannot be normalised and returns ``None`` — the caller counts it out of the
    normalised metrics.
    """
    shoulders = [gt[name] for name in ("left_shoulder", "right_shoulder") if name in gt]
    hips = [gt[name] for name in ("left_hip", "right_hip") if name in gt]
    if not shoulders or not hips:
        return None
    shoulder = np.mean(np.asarray(shoulders, dtype=np.float64), axis=0)
    hip = np.mean(np.asarray(hips, dtype=np.float64), axis=0)
    length = float(np.hypot(*(shoulder - hip)))
    return length if math.isfinite(length) and length > 0.0 else None


def _load_phases(
    root: pathlib.Path, clips: Sequence[str]
) -> dict[str, dict[int, tuple[str, int]]]:
    """``clip -> frame -> (phase, hold_id)`` from the phase parquets."""
    out: dict[str, dict[int, tuple[str, int]]] = {}
    for clip in clips:
        path = root / "phases" / "mediapipe" / f"{clip}.parquet"
        if not path.is_file():
            continue
        table = pd.read_parquet(path, columns=["frame_idx", "phase", "hold_id"])
        out[clip] = {
            int(row.frame_idx): (str(row.phase), int(row.hold_id))
            for row in table.itertuples(index=False)
        }
    return out


def _load_contact(root: pathlib.Path, clips: Sequence[str]) -> dict[str, dict[int, bool]]:
    """``clip -> frame -> trainer_contact`` from the MediaPipe athlete parquet.

    This is the one mask: the MediaPipe multi-person run's answer, applied to
    every model's accuracy, so Vision's own contact flags can never quietly
    reshape the comparison (chainlink #16 comment 25).
    """
    out: dict[str, dict[int, bool]] = {}
    for clip in clips:
        path = root / "keypoints" / "mediapipe_athlete" / "best" / f"{clip}.parquet"
        if not path.is_file():
            continue
        table = pd.read_parquet(path, columns=["frame_idx", "trainer_contact"])
        per_frame = table.groupby("frame_idx", sort=False)["trainer_contact"].any()
        out[clip] = {int(frame): bool(flag) for frame, flag in per_frame.items()}
    return out


def load_frames(data: str | pathlib.Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Everything frame-level: the frame table and the raw ground truth.

    Returns ``(frames, context)``: ``frames`` carries review tags, strata,
    phases, shapes, the MediaPipe contact mask, L and the report scopes;
    ``context`` holds ``gt_visible`` keyed by ``(clip, frame)`` — the joints a
    human placed and called visible, which are what every metric scores.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    rows = read_label_rows(root / "labels" / "keypoints.csv")
    frames = frames_table(
        rows,
        auto_images=load_auto_images(root),
        missing_clips=read_missing_clips(root),
    )
    gt_present: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
    gt_visible: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
    for row in rows:
        key = (row.clip_id, row.frame_idx)
        gt_present.setdefault(key, {})[row.joint] = (row.x, row.y)
        if row.visible:
            gt_visible.setdefault(key, {})[row.joint] = (row.x, row.y)
    clips = sorted({str(clip) for clip in frames.get("clip_id", pd.Series(dtype=str))})
    frames = annotate_frames(
        frames,
        strata=read_strata(root),
        phases=_load_phases(root, clips),
        shapes=load_hold_shapes(root),
        contact=_load_contact(root, clips),
        multi_person_clips=read_multi_person_clips(root),
    )
    clip_ids = [str(clip) for clip in frames.get("clip_id", [])]
    frame_indices = [int(frame) for frame in frames.get("frame_idx", [])]
    frames["L"] = [
        torso_length(gt_present.get((clip, frame), {}))
        for clip, frame in zip(clip_ids, frame_indices, strict=True)
    ]
    return frames, {"gt_visible": gt_visible}


# --------------------------------------------------------------------------- #
# Model outputs
# --------------------------------------------------------------------------- #

#: A person as ``joint -> (x, y, confidence)`` in display-frame pixels.
Person = dict[str, tuple[float, float, float]]
#: One clip's per-frame keys the bake-off asks about.
FrameKey = tuple[str, int]


def _wanted(frames: pd.DataFrame) -> dict[str, list[int]]:
    """``clip -> frame indices`` for every row of the frame table."""
    wanted: dict[str, list[int]] = {}
    for clip, frame in zip(frames.clip_id, frames.frame_idx, strict=True):
        wanted.setdefault(str(clip), []).append(int(frame))
    return wanted


def load_pipeline_persons(
    root: pathlib.Path,
    model: str,
    frames: pd.DataFrame,
) -> dict[FrameKey, Person | None]:
    """The model's own person per label frame, from its athlete parquet.

    ``detected = false`` becomes ``None``: the selection gave up on that frame
    and every ground-truth joint is a miss, which is exactly what the pipeline
    would deliver there.
    """
    single_dir, _multi_dir, rotate = MODEL_DIRS[model]
    persons: dict[FrameKey, Person | None] = {}
    for clip, frame_list in _wanted(frames).items():
        persons.update({(clip, frame): None for frame in frame_list})
        path = root / "keypoints" / single_dir / rotate / f"{clip}.parquet"
        if not path.is_file():
            continue
        table = pd.read_parquet(
            path, columns=["frame_idx", "joint", "x", "y", "visibility", "detected"]
        )
        table = table[table["frame_idx"].isin(frame_list)]
        for frame, block in table.groupby("frame_idx", sort=True):
            if not bool(block["detected"].iloc[0]):
                continue
            persons[(clip, int(frame))] = {
                str(row.joint): (float(row.x), float(row.y), float(row.visibility))
                for row in block.itertuples(index=False)
            }
    return persons


def load_pools(
    root: pathlib.Path,
    model: str,
    frames: pd.DataFrame,
) -> dict[FrameKey, list[Person]]:
    """All people the model detected per label frame, for the oracle match."""
    _single_dir, multi_dir, rotate = MODEL_DIRS[model]
    pools: dict[FrameKey, list[Person]] = {}
    for clip, frame_list in _wanted(frames).items():
        pools.update({(clip, frame): [] for frame in frame_list})
        path = root / "keypoints" / multi_dir / rotate / f"{clip}.parquet"
        if not path.is_file():
            continue
        table = pd.read_parquet(
            path, columns=["frame_idx", "person_idx", "joint", "x", "y", "visibility"]
        )
        table = table[(table["frame_idx"].isin(frame_list)) & (table["person_idx"] >= 0)]
        for (frame, _person_idx), block in table.groupby(["frame_idx", "person_idx"], sort=True):
            pools[(clip, int(frame))].append(
                {
                    str(row.joint): (float(row.x), float(row.y), float(row.visibility))
                    for row in block.itertuples(index=False)
                }
            )
    return pools


def is_inverted(joints: Mapping[str, tuple[float, float]]) -> bool | None:
    """Is this body on its hands in the display frame?

    Wrists below the ankles (y grows downward). ``None`` when the pose cannot
    be judged — no usable wrist or no usable ankle — so an unknown body is
    never read as a claim the other body contradicts.
    """
    wrists = [
        joints[name][1]
        for name in ("left_wrist", "right_wrist")
        if name in joints and math.isfinite(joints[name][1])
    ]
    ankles = [
        joints[name][1]
        for name in ("left_ankle", "right_ankle")
        if name in joints and math.isfinite(joints[name][1])
    ]
    if not wrists or not ankles:
        return None
    return sum(wrists) / len(wrists) > sum(ankles) / len(ankles)


def _mean_score(person: Person) -> float:
    """A person's mean confidence — RTMPose's own ``mean_score`` rule."""
    values = [score for _x, _y, score in person.values() if math.isfinite(score)]
    return float(np.mean(values)) if values else 0.0


def select_rtmpose_pipeline(pool: Sequence[Person]) -> Person | None:
    """RTMPose's pipeline person: the inverted body with the best score.

    ...else the best score. The pool is what ``predict_frame`` kept for the
    frame (its better-scoring orientation, mapped back into display pixels);
    preferring the inverted body is what separates the athlete from a standing
    trainer when both are detected.
    """
    if not pool:
        return None
    inverted = [
        index
        for index, person in enumerate(pool)
        if is_inverted({joint: (xy[0], xy[1]) for joint, xy in person.items()}) is True
    ]
    candidates = inverted if inverted else list(range(len(pool)))
    best = max(candidates, key=lambda index: (_mean_score(pool[index]), -index))
    return pool[best]


def run_rtmpose(
    root: pathlib.Path,
    frames: pd.DataFrame,
) -> tuple[dict[FrameKey, list[Person]], int]:
    """Run RTMPose fresh on **all** the label JPEGs, both orientations, best score.

    ``make_model`` + ``predict_frame`` **without a reference** (chainlink #16
    comment 46: MediaPipe is never the referee), on every labelled frame the
    task names — accuracy only ever reads the human-reviewed ones, and the
    auto-accepted ones are never scored against RTMPose (that would be RTMPose
    compared with itself), but the run covers the whole label set as the task
    asks. Returns the person pools and the number of JPEGs that were missing.
    The oracle pool is the orientation ``predict_frame`` kept — the model's own
    answer to "which way up is this frame".
    """
    import cv2

    from handstand.prelabel import make_model, predict_frame

    wanted = _wanted(frames)
    if not wanted:
        return {}, 0
    model = make_model("balanced")
    pools: dict[FrameKey, list[Person]] = {}
    missing_files = 0
    for clip, frame_list in wanted.items():
        for frame in frame_list:
            path = root / "label_frames" / f"{clip}_{frame}.jpg"
            image = None if not path.is_file() else cv2.imread(str(path))
            if image is None:
                missing_files += 1
                continue
            detection, _rotated = predict_frame(model, image)
            pools[(clip, frame)] = [
                {
                    joint: (float(x), float(y), float(pose.scores.get(joint, float("nan"))))
                    for joint, (x, y) in pose.joints.items()
                }
                for pose in detection.people
            ]
    return pools, missing_files


# --------------------------------------------------------------------------- #
# Matching and metrics
# --------------------------------------------------------------------------- #


def oracle_person(
    pool: Sequence[Person],
    gt: Mapping[str, tuple[float, float]],
    joints: Sequence[str],
    min_shared: int = MIN_SHARED_JOINTS,
) -> Person | None:
    """The person whose keypoints are closest to the ground truth.

    Mean pixel error over the ground-truth-visible joints both share; a person
    sharing fewer than ``min_shared`` joints cannot be judged (a body with one
    lucky joint must not win the oracle), and the bar never exceeds how many
    joints the frame actually has. Ties keep the model's own order.
    """
    wanted = [joint for joint in joints if joint in gt]
    if not wanted:
        return None
    required = min(min_shared, len(wanted))
    best: Person | None = None
    best_error = math.inf
    for person in pool:
        shared = [
            (gt[joint], person[joint][0], person[joint][1])
            for joint in wanted
            if joint in person
            and math.isfinite(person[joint][0])
            and math.isfinite(person[joint][1])
        ]
        if len(shared) < required:
            continue
        error = float(np.mean([math.hypot(px - gx, py - gy) for (gx, gy), px, py in shared]))
        if error < best_error:
            best_error = error
            best = person
    return best


def normalised_error(
    gt_xy: tuple[float, float],
    pred_xy: tuple[float, float],
    length: float | None,
) -> float:
    """``||pred - gt|| / L``; NaN when there is no L to normalise by."""
    if length is None or not math.isfinite(length):
        return float("nan")
    error = math.hypot(pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1])
    return error / length


def person_errors(
    gt: Mapping[str, tuple[float, float]],
    person: Person | None,
    joints: Sequence[str],
    length: float | None,
) -> list[dict[str, object]]:
    """One record per ground-truth-visible joint of ``joints``.

    Records carry the model's confidence and normalised error but no gate: the
    gate is a per-backend decision made *from* these records, so it is applied
    when metrics are computed, not when the evidence is collected.
    """
    records: list[dict[str, object]] = []
    for joint in joints:
        if joint not in gt:
            continue  # not GT-visible: out of every denominator
        gt_xy = gt[joint]
        person_joint = person.get(joint) if person else None
        has_pred = bool(
            person_joint is not None
            and math.isfinite(person_joint[0])
            and math.isfinite(person_joint[1])
        )
        conf = float(person_joint[2]) if person_joint is not None else float("nan")
        error = (
            normalised_error(gt_xy, (person_joint[0], person_joint[1]), length)
            if has_pred
            else float("nan")
        )
        records.append(
            {
                "joint": joint,
                "group": _GROUP_OF.get(joint, "other"),
                "gt_x": float(gt_xy[0]),
                "gt_y": float(gt_xy[1]),
                "conf": conf,
                "has_pred": has_pred,
                "err_l": error,
            }
        )
    return records


def _delivered(frame: pd.DataFrame, gate: float) -> pd.Series:
    """Joints the backend actually delivers: output at or above its gate.

    A NaN confidence fails the comparison — the same answer
    :func:`handstand.postprocess.gated_valid` gives: a joint the model refused
    to score is not a position.
    """
    return frame["has_pred"] & (frame["conf"] >= gate)


def _normalised(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows a normalised metric may score: frames that have an L.

    A frame without GT shoulders or hips is excluded from PCK and error and
    counted in the summary; the miss rate needs no normalisation and keeps it.
    """
    return frame[frame["L"].notna()] if "L" in frame.columns else frame


def pck(frame: pd.DataFrame, gate: float, level: float) -> float:
    """PCK@``level``: share of GT-visible joints within ``level * L``.

    The denominator is every GT-visible joint of a frame that has an L; a
    missed or below-gate joint cannot be within the threshold and counts
    against the score, which is what stops a model from inflating PCK by
    refusing to answer.
    """
    normalised = _normalised(frame)
    if normalised.empty:
        return float("nan")
    delivered = _delivered(normalised, gate)
    hits = delivered & (normalised["err_l"] <= level)
    return float(hits.sum() / len(normalised))


def miss_rate(frame: pd.DataFrame, gate: float) -> float:
    """Share of GT-visible joints the model did not deliver.

    No output, no finite coordinates, or output below the backend's confidence
    gate — all three are "the pipeline drops this joint".
    """
    if frame.empty:
        return float("nan")
    return float((~_delivered(frame, gate)).sum() / len(frame))


def error_summary(frame: pd.DataFrame, gate: float) -> dict[str, float]:
    """Mean / median normalised error over the delivered joints, plus n."""
    delivered = frame[_delivered(frame, gate) & frame["err_l"].notna()]
    values = delivered["err_l"].to_numpy(dtype=np.float64)
    return {
        "mean_err": float(np.mean(values)) if values.size else float("nan"),
        "median_err": float(np.median(values)) if values.size else float("nan"),
        "n_delivered": int(values.size),
    }


def swap_report(
    gt: Mapping[str, tuple[float, float]],
    person: Person | None,
    length: float | None,
    gate: float,
) -> dict[str, bool | None]:
    """Left/right swap evidence for one frame: wrists and ankles.

    A pair is *evaluable* only when the ground truth shows both sides far
    enough apart to mean anything (``SWAP_MIN_SEPARATION_L`` — the sides
    overlap in a side view), the model delivers both sides at the gate, and
    there is an L. A pair counts as *swapped* when the crossed assignment beats
    the direct one both by ``SWAP_SCORE_MARGIN`` of L and by ``SWAP_STRONG``
    of their sum: a genuine swap scores cross ~ 0 against direct ~ 2 x
    separation and clears both bars, while noise on a symmetric pose clears
    neither. ``None`` means "not evaluable", which is not the same as "fine".
    """
    result: dict[str, bool | None] = {"wrists": None, "ankles": None}
    if length is None or person is None:
        return result
    for label, left, right in (
        ("wrists", "left_wrist", "right_wrist"),
        ("ankles", "left_ankle", "right_ankle"),
    ):
        if left not in gt or right not in gt:
            continue
        gl, gr = gt[left], gt[right]
        separation = math.hypot(gr[0] - gl[0], gr[1] - gl[1])
        if separation < SWAP_MIN_SEPARATION_L * length:
            continue
        pl, pr = person.get(left), person.get(right)
        if pl is None or pr is None or pl[2] < gate or pr[2] < gate:
            continue
        if not all(math.isfinite(value) for value in (pl[0], pl[1], pr[0], pr[1])):
            continue
        direct = math.hypot(pl[0] - gl[0], pl[1] - gl[1]) + math.hypot(
            pr[0] - gr[0], pr[1] - gr[1]
        )
        cross = math.hypot(pl[0] - gr[0], pl[1] - gr[1]) + math.hypot(
            pr[0] - gl[0], pr[1] - gl[1]
        )
        strong = (direct - cross) > SWAP_STRONG * (direct + cross)
        margin = (direct - cross) > SWAP_SCORE_MARGIN * length
        result[label] = bool(strong and margin)
    return result


# --------------------------------------------------------------------------- #
# The visibility gate
# --------------------------------------------------------------------------- #


def gate_sweep(rows: pd.DataFrame) -> pd.DataFrame:
    """Error vs confidence, tabulated per candidate gate (one backend).

    ``rows`` are the human-reviewed pipeline joints of the selection set that
    have an L; each output row says how many joints a gate keeps and how
    precise those kept joints are — the evidence the recommendation is made
    from. Joints the model never output are not this table's business: no gate
    can recover them, they are misses either way.
    """
    evidence = rows[rows["has_pred"] & rows["err_l"].notna()]
    total = len(evidence)
    table: list[dict[str, object]] = []
    for gate in GATE_GRID:
        kept = evidence[evidence["conf"] >= gate]
        precise = int((kept["err_l"] <= 0.2).sum()) if not kept.empty else 0
        precision = precise / len(kept) if not kept.empty else float("nan")
        table.append(
            {
                "gate": gate,
                "n_kept": int(len(kept)),
                "n_total": total,
                "kept_fraction": len(kept) / total if total else float("nan"),
                "within_02": precision,
                "meets_target": bool(not kept.empty and precision >= GATE_TARGET),
            }
        )
    return pd.DataFrame(table)


def recommend_gate(sweep: pd.DataFrame) -> tuple[float, bool]:
    """The lowest gate that keeps >= ``GATE_TARGET`` of joints within 0.2 L.

    Returns ``(gate, target_met)``. When **no** gate reaches the target the
    target is reported as unmet and the gate stays at ``MIN_VISIBILITY``:
    raising a gate that buys no precision only discards good joints (mediapipe's
    confidence is flat against its errors — 0.76 precise at gate 0, 0.79 at
    0.95 — so 0.95 would drop 12 % of the joints and nine usable clips for two
    points of precision). The sweep table keeps the evidence either way.
    """
    if sweep.empty:
        return LEGACY_GATE, False
    meeting = sweep[sweep["meets_target"]]
    if not meeting.empty:
        return float(meeting["gate"].min()), True
    return LEGACY_GATE, False


def unusable_clip_counts(
    data: str | pathlib.Path | None,
    gates: Mapping[str, float],
) -> pd.DataFrame:
    """Clips :func:`handstand.postprocess.process_clip` gives no body length to.

    Every non-missing catalogue clip, at ``MIN_VISIBILITY = 0.5`` (the value
    tuned on MediaPipe) and at the backend's recommended gate — the count the
    visibility-gate section of the task asks for. RTMPose has no full-clip
    keypoints, so it gets a not-applicable row rather than a made-up one.
    """
    root = pathlib.Path(data) if data is not None else data_dir()
    catalogue = root / "catalogue.csv"
    clips: list[str] = []
    if catalogue.is_file():
        table = pd.read_csv(catalogue, dtype=str, keep_default_na=False)
        clips = [str(clip) for clip in table.get("clip_id", [])]
    missing = read_missing_clips(root)
    clips = [clip for clip in clips if clip not in missing]
    rows: list[dict[str, object]] = []
    for model in MEASUREMENT_MODELS:
        single_dir, _multi, rotate = MODEL_DIRS[model]
        recommended = gates.get(model, LEGACY_GATE)
        for label, gate in ((f"{LEGACY_GATE:g}", LEGACY_GATE), ("recommended", recommended)):
            unusable = 0
            absent = 0
            for clip in clips:
                path = root / "keypoints" / single_dir / rotate / f"{clip}.parquet"
                if not path.is_file():
                    absent += 1
                    continue
                try:
                    clip_data = postprocess.read_clip(path)
                    processed = postprocess.process_clip(clip_data, min_visibility=gate)
                except (ValueError, OSError):
                    unusable += 1
                    continue
                if not processed.body_length.usable:
                    unusable += 1
            rows.append(
                {
                    "model": model,
                    "gate_setting": label,
                    "gate": gate,
                    "n_clips": len(clips),
                    "n_missing_keypoints": absent,
                    "n_unusable": unusable,
                }
            )
    rows.append(
        {
            "model": "rtmpose",
            "gate_setting": "recommended",
            "gate": gates.get("rtmpose", LEGACY_GATE),
            "n_clips": len(clips),
            "n_missing_keypoints": len(clips),
            "n_unusable": float("nan"),
            "note": "no full-clip RTMPose keypoints; not an on-device backend",
        }
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_ratio_ci(
    frame: pd.DataFrame,
    numer: str,
    denom: str,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95 % interval for ``sum(numer) / sum(denom)`` by resampling **clips**.

    Rows of one clip move together — frames in a clip are correlated, so
    resampling frames would fake a sample the data does not have. The RNG is
    seeded, so the same table always reports the same interval.
    """
    if frame.empty or samples <= 0:
        return float("nan"), float("nan")
    grouped = frame.groupby("clip_id", sort=True)[[numer, denom]].sum()
    if grouped.empty:
        return float("nan"), float("nan")
    numer_by_clip = grouped[numer].to_numpy(dtype=np.float64)
    denom_by_clip = grouped[denom].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(samples, len(grouped)))
    totals_num = numer_by_clip[draws].sum(axis=1)
    totals_den = denom_by_clip[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(totals_den > 0, totals_num / totals_den, np.nan)
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _metric_row(
    frame: pd.DataFrame,
    gate: float,
    *,
    samples: int,
    seed: int,
    full_ci: bool,
) -> dict[str, object]:
    """One aggregate row: PCK, miss, error and the clip-bootstrap intervals."""
    row: dict[str, object] = {
        "n_joints": int(len(frame)),
        "pck02": pck(frame, gate, 0.2),
        "pck05": pck(frame, gate, 0.5),
        "miss_rate": miss_rate(frame, gate),
        **error_summary(frame, gate),
    }
    row["pck02_lo"], row["pck02_hi"] = _pck_ci(frame, gate, 0.2, samples, seed)
    if full_ci:
        row["pck05_lo"], row["pck05_hi"] = _pck_ci(frame, gate, 0.5, samples, seed)
        work = frame.copy()
        work["_hit"] = (~_delivered(work, gate)).astype(float)
        work["_total"] = 1.0
        row["miss_lo"], row["miss_hi"] = bootstrap_ratio_ci(
            work, "_hit", "_total", samples=samples, seed=seed
        )
        delivered = frame[_delivered(frame, gate) & frame["err_l"].notna()]
        if delivered.empty:
            row["mean_err_lo"] = row["mean_err_hi"] = float("nan")
        else:
            work = delivered.copy()
            work["_num"] = work["err_l"]
            work["_den"] = 1.0
            row["mean_err_lo"], row["mean_err_hi"] = bootstrap_ratio_ci(
                work, "_num", "_den", samples=samples, seed=seed
            )
    return row


def _pck_ci(
    frame: pd.DataFrame,
    gate: float,
    level: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap interval for one PCK, built as hit/total counts per clip."""
    work = _normalised(frame).copy()
    if work.empty:
        return float("nan"), float("nan")
    delivered = _delivered(work, gate)
    work["_hit"] = (delivered & (work["err_l"] <= level)).astype(float)
    work["_total"] = 1.0
    return bootstrap_ratio_ci(work, "_hit", "_total", samples=samples, seed=seed)


def _aggregate(
    records: pd.DataFrame,
    by: Sequence[str],
    *,
    gate_of: Mapping[str, float],
    samples: int,
    seed: int,
    full_ci: bool = False,
) -> pd.DataFrame:
    """Aggregate already-scoped ``records`` by model/mode/``by``.

    Each model is scored at its own gate — the gate it would ship with — which
    is the fair comparison the per-backend gate exists for.
    """
    if records.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (model, mode), group in records.groupby(["model", "mode"], sort=False):
        gate = gate_of.get(model, LEGACY_GATE)
        grouped = group.groupby(by, sort=False) if by else [(None, group)]
        for key, part in grouped:
            row: dict[str, object] = {"model": model, "mode": mode, "gate": gate}
            if by and key is not None:
                key_values = key if isinstance(key, tuple) else (key,)
                row.update(dict(zip(by, key_values, strict=True)))
            row.update(_metric_row(part, gate, samples=samples, seed=seed, full_ci=full_ci))
            frames_seen = part[["clip_id", "frame_idx"]].drop_duplicates()
            row["n_frames"] = int(len(frames_seen))
            row["n_clips"] = int(part["clip_id"].nunique())
            rows.append(row)
    return pd.DataFrame(rows)


def _headline_mask(records: pd.DataFrame) -> pd.Series:
    """HOLD frames of LINE holds in the clean strata — the app's use case."""
    return (
        records["in_selection"]
        & (records["phase"] == "hold")
        & (records["shape"] == "line")
    )


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #


def decide(
    headline: pd.DataFrame,
    gates: Mapping[str, float],
    gate_met: Mapping[str, bool],
    unusable: pd.DataFrame,
    headline_shapes: pd.DataFrame,
) -> str:
    """The decision rule of the task, applied to the headline numbers.

    Higher headline PCK@0.2 (pipeline matching, each backend at its own
    recommended gate) wins **unless the clip-bootstrap 95 % intervals overlap**
    — then Vision is preferred (built into iOS, no model download) and the
    result is called out as not decisive. The same comparison is repeated at
    the common ``0.5`` gate as a robustness check, the recommended gate and its
    unusable-clip counts are stated, the wrists/ankles rule can open chainlink
    #17, and RTMPose's number is the headroom.
    """
    lines: list[str] = []
    pipeline = headline[
        (headline["mode"] == PIPELINE_MODE) & (headline["gate_setting"] == "recommended")
    ]
    by_model = {str(row.model): row for row in pipeline.itertuples(index=False)}

    def interval(model: str) -> tuple[float, float, float]:
        row = by_model.get(model)
        if row is None:
            return float("nan"), float("nan"), float("nan")
        return float(row.pck02), float(row.pck02_lo), float(row.pck02_hi)

    mp_pck, mp_lo, mp_hi = interval("mediapipe")
    vision_pck, vision_lo, vision_hi = interval("vision")
    rtm_pck = interval("rtmpose")[0]

    overlap = bool(
        math.isfinite(mp_lo)
        and math.isfinite(vision_lo)
        and not (mp_hi < vision_lo or vision_hi < mp_lo)
    )
    if not math.isfinite(mp_pck) or not math.isfinite(vision_pck):
        lines.append("DECISION: cannot decide — a headline number is missing.")
    elif overlap:
        lines.append(
            "DECISION: prefer vision (built into iOS, no model download); the "
            "headline PCK@0.2 intervals overlap, so the comparison is NOT "
            f"decisive (mediapipe {mp_pck:.3f} [{mp_lo:.3f}, {mp_hi:.3f}] vs "
            f"vision {vision_pck:.3f} [{vision_lo:.3f}, {vision_hi:.3f}])."
        )
    elif vision_pck > mp_pck:
        lines.append(
            f"DECISION: vision wins on headline PCK@0.2 "
            f"({vision_pck:.3f} [{vision_lo:.3f}, {vision_hi:.3f}] vs mediapipe "
            f"{mp_pck:.3f} [{mp_lo:.3f}, {mp_hi:.3f}]), intervals disjoint."
        )
    else:
        lines.append(
            f"DECISION: mediapipe wins on headline PCK@0.2 "
            f"({mp_pck:.3f} [{mp_lo:.3f}, {mp_hi:.3f}] vs vision "
            f"{vision_pck:.3f} [{vision_lo:.3f}, {vision_hi:.3f}]), intervals disjoint."
        )

    # Robustness: the same comparison with both backends at the one gate the
    # pipeline ships today, so the verdict cannot be blamed on the gate choice.
    legacy = headline[
        (headline["mode"] == PIPELINE_MODE)
        & (headline["gate_setting"] == f"{LEGACY_GATE:g}")
    ]
    legacy_by_model = {str(row.model): row for row in legacy.itertuples(index=False)}
    if "mediapipe" in legacy_by_model and "vision" in legacy_by_model:
        mp_row = legacy_by_model["mediapipe"]
        vision_row = legacy_by_model["vision"]
        same_winner = (mp_row.pck02 > vision_row.pck02) == (mp_pck > vision_pck)
        disjoint = bool(
            mp_row.pck02_hi < vision_row.pck02_lo
            or vision_row.pck02_hi < mp_row.pck02_lo
        )
        if not same_winner:
            verdict = "the winner DIFFERS from the per-backend-gate verdict"
        elif disjoint:
            verdict = "intervals disjoint, winner unchanged"
        else:
            verdict = (
                "the winner is unchanged, but the intervals overlap, so at this "
                "gate alone the gap is not decisive"
            )
        lines.append(
            f"ROBUSTNESS: at the common gate {LEGACY_GATE:g}/{LEGACY_GATE:g} the headline is "
            f"mediapipe {mp_row.pck02:.3f} [{mp_row.pck02_lo:.3f}, {mp_row.pck02_hi:.3f}] vs "
            f"vision {vision_row.pck02:.3f} [{vision_row.pck02_lo:.3f}, "
            f"{vision_row.pck02_hi:.3f}] — {verdict}."
        )

    for model in MEASUREMENT_MODELS:
        setting = unusable[
            (unusable["model"] == model) & (unusable["gate_setting"] == "recommended")
        ]
        legacy_row = unusable[
            (unusable["model"] == model) & (unusable["gate_setting"] == f"{LEGACY_GATE:g}")
        ]
        if setting.empty:
            continue
        legacy_count = (
            int(legacy_row["n_unusable"].iloc[0]) if not legacy_row.empty else float("nan")
        )
        met = "90% target met" if gate_met.get(model, False) else "90% target NOT met at any gate"
        lines.append(
            f"GATE {model}: recommended {gates.get(model, LEGACY_GATE):.2f} ({met}); unusable "
            f"clips at {LEGACY_GATE:g} = {legacy_count}, at the recommended gate = "
            f"{int(setting['n_unusable'].iloc[0])}."
        )
    if math.isfinite(rtm_pck):
        best_on_device = max(
            (value for value in (mp_pck, vision_pck) if math.isfinite(value)), default=0.0
        )
        lines.append(
            f"HEADROOM: rtmpose (reference upper bound) headline PCK@0.2 = {rtm_pck:.3f} "
            f"vs best on-device {best_on_device:.3f} — that gap is what better "
            "on-device keypoints or a fine-tuned model could recover."
        )
    worst = headline_shapes[
        (headline_shapes["mode"] == PIPELINE_MODE)
        & headline_shapes["group"].isin(("wrists", "ankles"))
    ]
    per_model = worst.groupby("model")["pck02"].min()
    low = [
        model
        for model in MEASUREMENT_MODELS
        if model in per_model.index and float(per_model[model]) < 0.6
    ]
    if len(low) == len(MEASUREMENT_MODELS):
        detail = ", ".join(
            f"{model} {float(per_model[model]):.3f}"
            for model in MEASUREMENT_MODELS
            if model in per_model.index
        )
        lines.append(
            "NEXT: both on-device models are below 0.6 PCK@0.2 on wrists or ankles "
            f"in the headline subset ({detail}) — open chainlink #17 (fine-tune): "
            "those are the joints a hold is judged by, and no visibility gate fixes "
            "a keypoint the model put in the wrong place."
        )
    else:
        detail = ", ".join(
            f"{model} {float(per_model[model]):.3f}"
            for model in MEASUREMENT_MODELS
            if model in per_model.index
        )
        lines.append(
            "NEXT: the #17 rule (both on-device models below 0.6 PCK@0.2 on wrists "
            f"or ankles in the headline) does not fire — worst group per model: "
            f"{detail}. Re-check after more human review; the sample is small."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class BakeoffResult:
    """What one bake-off run produced."""

    frames: pd.DataFrame
    records: pd.DataFrame
    headline: pd.DataFrame
    headline_shapes: pd.DataFrame
    gates: dict[str, float]
    gate_met: dict[str, bool]
    sweeps: dict[str, pd.DataFrame]
    unusable: pd.DataFrame
    swaps: pd.DataFrame
    missing_rtmpose_frames: int
    written: list[pathlib.Path]
    samples: int = BOOTSTRAP_SAMPLES
    seed: int = BOOTSTRAP_SEED

    def summary(self) -> str:
        """The printed report: sample sizes, headline, gates, decision."""
        return summary(self)


def _build_records(
    frames: pd.DataFrame,
    gt_visible: Mapping[FrameKey, Mapping[str, tuple[float, float]]],
    persons: Mapping[FrameKey, Person | None],
    pools: Mapping[FrameKey, list[Person]],
    *,
    model: str,
    mode: str,
) -> pd.DataFrame:
    """Every record of one model x mode over the frames it is scored on.

    Accuracy scopes (selection, stratum, note) need kept frames; the agreement
    scope needs the auto-accepted frames too — but only for the models the
    agreement compares, so RTMPose is never scored against its own pre-labels.
    """
    joints = MODEL_JOINTS[model]
    rows: list[dict[str, object]] = []
    for frame in frames.itertuples(index=False):
        scored = bool(frame.keep) or (
            bool(frame.in_agreement) and model in MEASUREMENT_MODELS
        )
        if not scored:
            continue
        key = (str(frame.clip_id), int(frame.frame_idx))
        gt = gt_visible.get(key, {})
        pool = pools.get(key, [])
        if mode == PIPELINE_MODE:
            person = (
                select_rtmpose_pipeline(pool)
                if model == "rtmpose"
                else persons.get(key)
            )
        else:
            person = oracle_person(pool, gt, joints)
        length = float(frame.L) if frame.L is not None and math.isfinite(float(frame.L)) else None
        for record in person_errors(gt, person, joints, length):
            record.update(
                {
                    "model": model,
                    "mode": mode,
                    "clip_id": str(frame.clip_id),
                    "frame_idx": int(frame.frame_idx),
                    "stratum": str(frame.stratum),
                    "phase": str(frame.phase),
                    "shape": str(frame.shape),
                    "multi_person": bool(frame.multi_person),
                    "in_selection": bool(frame.in_selection),
                    "in_note": bool(frame.in_note),
                    "in_stratum": bool(frame.in_stratum),
                    "in_agreement": bool(frame.in_agreement),
                    "L": length,
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _swap_table(
    frames: pd.DataFrame,
    gt_visible: Mapping[FrameKey, Mapping[str, tuple[float, float]]],
    persons_by_model: Mapping[str, Mapping[FrameKey, Person | None]],
    pools_by_model: Mapping[str, Mapping[FrameKey, list[Person]]],
    *,
    gate_of: Mapping[str, float],
) -> pd.DataFrame:
    """Frame-level left/right swap rates over the selection set, per model."""
    selection = frames[frames["in_selection"]]
    rows: list[dict[str, object]] = []
    for model in MODELS:
        if model not in pools_by_model:
            continue
        gate = gate_of.get(model, LEGACY_GATE)
        persons = persons_by_model[model]
        pools = pools_by_model[model]
        for mode in MODES:
            evaluable = 0
            swapped = 0
            for frame in selection.itertuples(index=False):
                key = (str(frame.clip_id), int(frame.frame_idx))
                gt = gt_visible.get(key, {})
                if mode == PIPELINE_MODE:
                    person = (
                        select_rtmpose_pipeline(pools.get(key, []))
                        if model == "rtmpose"
                        else persons.get(key)
                    )
                else:
                    person = oracle_person(pools.get(key, []), gt, MODEL_JOINTS[model])
                if person is None:
                    continue
                length = (
                    float(frame.L)
                    if frame.L is not None and math.isfinite(float(frame.L))
                    else None
                )
                flags = [
                    flag
                    for flag in swap_report(gt, person, length, gate).values()
                    if flag is not None
                ]
                if flags:
                    evaluable += 1
                    swapped += int(any(flags))
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "gate": gate,
                    "n_frames_evaluable": evaluable,
                    "n_frames_swapped": swapped,
                    "swap_rate": swapped / evaluable if evaluable else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def run(
    data: str | pathlib.Path | None = None,
    *,
    models: Sequence[str] = MODELS,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    write: bool = True,
) -> BakeoffResult:
    """Run the whole bake-off: load, predict, score, gate, write, decide."""
    root = pathlib.Path(data) if data is not None else data_dir()
    wanted = [model for model in MODELS if model in models]
    if not wanted:
        raise ValueError("no models to score")

    frames, context = load_frames(root)
    gt_visible: dict[FrameKey, Mapping[str, tuple[float, float]]] = context["gt_visible"]

    persons_by_model: dict[str, dict[FrameKey, Person | None]] = {}
    pools_by_model: dict[str, dict[FrameKey, list[Person]]] = {}
    missing_rtmpose = 0
    for model in wanted:
        if model == "rtmpose":
            pools, missing_rtmpose = run_rtmpose(root, frames)
            pools_by_model[model] = pools
            persons_by_model[model] = {
                key: select_rtmpose_pipeline(pool) for key, pool in pools.items()
            }
            continue
        persons_by_model[model] = load_pipeline_persons(root, model, frames)
        pools_by_model[model] = load_pools(root, model, frames)

    records = pd.concat(
        [
            table
            for model in wanted
            for mode in MODES
            if not (
                table := _build_records(
                    frames,
                    gt_visible,
                    persons_by_model[model],
                    pools_by_model[model],
                    model=model,
                    mode=mode,
                )
            ).empty
        ],
        ignore_index=True,
    )

    # The gate: human-reviewed pipeline joints, mask applied, per backend.
    gates: dict[str, float] = {}
    gate_met: dict[str, bool] = {}
    sweeps: dict[str, pd.DataFrame] = {}
    for model in wanted:
        evidence = records[
            (records["model"] == model)
            & (records["mode"] == PIPELINE_MODE)
            & records["in_stratum"]
            & records["joint"].isin(MAIN_JOINTS)
            & records["L"].notna()
        ]
        sweep = gate_sweep(evidence)
        gates[model], gate_met[model] = recommend_gate(sweep)
        sweeps[model] = sweep

    selection_all = records[records["in_selection"]]
    headline_records = records[_headline_mask(records)]
    headline_main = headline_records[headline_records["joint"].isin(MAIN_JOINTS)]

    headline = _aggregate(
        headline_main,
        [],
        gate_of=gates,
        samples=samples,
        seed=seed,
        full_ci=True,
    )
    legacy = _aggregate(
        headline_main,
        [],
        gate_of={model: LEGACY_GATE for model in wanted},
        samples=samples,
        seed=seed,
        full_ci=False,
    )
    raw = _aggregate(
        headline_main,
        [],
        gate_of={model: 0.0 for model in wanted},
        samples=samples,
        seed=seed,
        full_ci=False,
    )
    for table_frame, label in (
        (headline, "recommended"),
        (legacy, f"{LEGACY_GATE:g}"),
        (raw, "0"),
    ):
        if table_frame.empty:
            continue
        table_frame.insert(0, "scope", SELECTION_SCOPE)
        table_frame["gate_setting"] = label
    headline = pd.concat([headline, legacy, raw], ignore_index=True)

    def table(
        scope: pd.DataFrame,
        by: Sequence[str],
        *,
        full_ci: bool = False,
    ) -> pd.DataFrame:
        return _aggregate(
            scope, by, gate_of=gates, samples=samples, seed=seed, full_ci=full_ci
        )

    headline_shapes = table(headline_records, ["group"])
    if not headline_shapes.empty:
        headline_shapes.insert(0, "scope", SELECTION_SCOPE)
    per_group = table(
        selection_all[selection_all["joint"].isin(MAIN_JOINTS)], ["group"]
    )
    per_joint = table(selection_all, ["joint", "group"])
    per_stratum = table(
        records[records["in_stratum"] & records["joint"].isin(MAIN_JOINTS)],
        ["stratum"],
    )
    per_phase = table(
        selection_all[selection_all["joint"].isin(MAIN_JOINTS)], ["phase"]
    )
    per_shape = table(selection_all[selection_all["joint"].isin(MAIN_JOINTS)], ["shape"])
    multi_person = table(
        selection_all[selection_all["joint"].isin(MAIN_JOINTS)], ["multi_person"]
    )
    trainer_note = table(
        records[records["in_note"] & records["joint"].isin(MAIN_JOINTS)],
        ["stratum"],
    )
    if not trainer_note.empty:
        trainer_note.insert(0, "scope", TRAINER_NOTE_SCOPE)
    agreement = table(
        records[records["in_agreement"] & records["joint"].isin(MAIN_JOINTS)],
        [],
    )
    if not agreement.empty:
        agreement.insert(0, "scope", AGREEMENT_SCOPE)
    swaps = _swap_table(
        frames, gt_visible, persons_by_model, pools_by_model, gate_of=gates
    )
    unusable = unusable_clip_counts(root, gates)

    written: list[pathlib.Path] = []
    if write:
        out_dir = root / "reports" / "bakeoff"
        out_dir.mkdir(parents=True, exist_ok=True)
        tables: dict[str, pd.DataFrame] = {
            "headline": headline,
            "headline_by_group": headline_shapes,
            "per_group": per_group,
            "per_joint": per_joint,
            "per_stratum": per_stratum,
            "per_phase": per_phase,
            "per_shape": per_shape,
            "multi_person": multi_person,
            "trainer_note": trainer_note,
            "agreement_auto": agreement,
            "swap": swaps,
            "unusable_clips": unusable,
        }
        for model, sweep in sweeps.items():
            table_sweep = sweep.copy()
            table_sweep.insert(0, "model", model)
            table_sweep["recommended"] = table_sweep["gate"] == gates[model]
            table_sweep["target_met"] = gate_met[model]
            tables[f"gate_{model}"] = table_sweep
        for name, table_frame in tables.items():
            if table_frame.empty:
                continue
            path = out_dir / f"{name}.csv"
            table_frame.round(6).to_csv(path, index=False)
            written.append(path)

    return BakeoffResult(
        frames=frames,
        records=records,
        headline=headline,
        headline_shapes=headline_shapes,
        gates=gates,
        gate_met=gate_met,
        sweeps=sweeps,
        unusable=unusable,
        swaps=swaps,
        missing_rtmpose_frames=missing_rtmpose,
        written=written,
        samples=samples,
        seed=seed,
    )


def summary(result: BakeoffResult) -> str:
    """The printed report: sample sizes, headline, gates, decision."""
    frames = result.frames
    lines: list[str] = []
    n_total = int(len(frames))
    n_human = int(frames["keep"].sum())
    n_auto = int(frames["in_agreement"].sum())
    n_selection = int(frames["in_selection"].sum())
    n_note = int(frames["in_note"].sum())
    headline_frames = frames[
        frames["in_selection"] & (frames["phase"] == "hold") & (frames["shape"] == "line")
    ]
    n_headline = int(len(headline_frames))
    no_torso = int((frames["keep"] & frames["L"].isna()).sum())
    lines.append(
        f"samples: {n_total} labelled frames; {n_human} human-reviewed kept for accuracy; "
        f"{n_auto} auto-accepted reported as agreement with RTMPose only; "
        f"{n_selection} selection frames (clean strata, MediaPipe contact mask applied), "
        f"{n_note} trainer-contact note frames"
    )
    lines.append(
        "human frames per stratum: "
        f"{frames[frames['keep']]['stratum'].value_counts().sort_index().to_dict()}"
    )
    lines.append(
        "auto-accepted frames per stratum: "
        f"{frames[frames['in_agreement']]['stratum'].value_counts().sort_index().to_dict()}"
    )
    lines.append(
        f"headline subset: {n_headline} hold frames of line holds on "
        f"{headline_frames['clip_id'].nunique()} clips — SMALL SAMPLE: "
        f"95% intervals bootstrap over clips, {result.samples} draws, "
        f"seed {result.seed}"
    )
    lines.append(
        f"frames excluded from normalised metrics for missing torso GT: {no_torso}"
    )
    if result.missing_rtmpose_frames:
        lines.append(
            f"rtmpose: {result.missing_rtmpose_frames} label JPEG(s) missing and skipped"
        )

    headline = result.headline
    shown = headline[
        (headline["mode"] == PIPELINE_MODE) & (headline["gate_setting"] == "recommended")
    ]
    lines.append("")
    lines.append("headline (hold + line + clean strata), pipeline matching, own gate:")
    for row in shown.itertuples(index=False):
        lines.append(
            f"  {row.model:<10} PCK@0.2 {row.pck02:.3f} [{row.pck02_lo:.3f}, {row.pck02_hi:.3f}]"
            f"  PCK@0.5 {row.pck05:.3f}  mean {row.mean_err:.3f}  median {row.median_err:.3f}"
            f"  miss {row.miss_rate:.3f}  n={row.n_joints} joints / {row.n_frames} frames"
        )
    oracle = headline[
        (headline["mode"] == ORACLE_MODE) & (headline["gate_setting"] == "recommended")
    ]
    lines.append("oracle matching (person choice given for free):")
    for row in oracle.itertuples(index=False):
        lines.append(
            f"  {row.model:<10} PCK@0.2 {row.pck02:.3f} [{row.pck02_lo:.3f}, {row.pck02_hi:.3f}]"
            f"  mean {row.mean_err:.3f}  miss {row.miss_rate:.3f}"
        )
    lines.append("")

    gates_line = ", ".join(
        f"{model}={gate:.2f}{'*' if not result.gate_met.get(model, True) else ''}"
        for model, gate in sorted(result.gates.items())
    )
    lines.append(
        f"recommended visibility gates: {gates_line} "
        f"(* = 90% target not reached at any gate); pipeline MIN_VISIBILITY={LEGACY_GATE:g}"
    )
    for row in result.unusable.itertuples(index=False):
        if row.model == "rtmpose":
            lines.append("  rtmpose unusable clips: n/a (no full-clip keypoints)")
            continue
        lines.append(
            f"  {row.model}: {int(row.n_unusable)}/{int(row.n_clips)} clips unusable "
            f"(no body length) at gate {row.gate:g} ({row.gate_setting})"
        )
    if not result.swaps.empty:
        swap_line = ", ".join(
            f"{row.model}/{row.mode} {row.swap_rate:.2f} ({row.n_frames_swapped}/"
            f"{row.n_frames_evaluable})"
            for row in result.swaps.itertuples(index=False)
            if row.mode == PIPELINE_MODE
        )
        lines.append(f"L/R swap rate on evaluable selection frames: {swap_line}")
    lines.append("")
    lines.append(
        decide(
            result.headline,
            result.gates,
            result.gate_met,
            result.unusable,
            result.headline_shapes,
        )
    )
    if result.written:
        lines.append(f"csv: {len(result.written)} tables under {result.written[0].parent}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.bakeoff`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.bakeoff",
        description="Compare MediaPipe, Apple Vision and RTMPose against the human labels.",
    )
    parser.add_argument(
        "--skip-rtmpose",
        action="store_true",
        help="do not run the RTMPose reference (saves the fresh inference pass)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=None,
        metavar="MODEL",
        help=f"models to score (default: {' '.join(MODELS)})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
        metavar="N",
        help=f"bootstrap resamples over clips (default: {BOOTSTRAP_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BOOTSTRAP_SEED,
        metavar="N",
        help=f"bootstrap seed (default: {BOOTSTRAP_SEED})",
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory (default: $HANDSTAND_DATA)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="compute and print, but write no CSVs",
    )
    parser.add_argument(
        "--flag-catalogue",
        action="store_true",
        help=(
            "before scoring, append the multi-person note to the catalogue "
            "notes of every clip the trainer report saw a second person in "
            "(chainlink #15); idempotent"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.bakeoff``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.samples < 0:
        parser.error("--samples must be >= 0")
    models = list(args.models or MODELS)
    if args.skip_rtmpose and "rtmpose" in models:
        models.remove("rtmpose")
    if not models:
        parser.error("no models left to score")
    if args.flag_catalogue:
        flagged = flag_multi_person_clips(args.data)
        print(
            f"catalogue: flagged {len(flagged)} multi-person clip(s)"
            if flagged
            else "catalogue: nothing to flag"
        )
    try:
        result = run(
            args.data,
            models=models,
            samples=args.samples,
            seed=args.seed,
            write=not args.no_write,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"bakeoff: {error}", file=sys.stderr)
        return 2
    print(summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
