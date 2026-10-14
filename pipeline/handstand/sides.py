"""Left/right near-side check from the nose in side views (chainlink #88).

Stage 1 is measurement only. In a side view the athlete's nose shows which way
the body faces; with the inversion that decides which body side is nearer the
camera::

    facing_right = nose x > shoulder-midpoint x          (image pixels)
    near side    = "right" if facing_right XOR inverted else "left"

Pose models sometimes label the near side as the wrong side. This module
measures how often each model's own near-side claim (visibility or MediaPipe
z) disagrees with that rule and with the human labels, so the lead can decide
whether a correction step is worth building. Nothing here changes a keypoint:
``postprocess``, ``features``, ``com``, ``score``, ``bakeoff`` and the Swift
code are untouched.

CLI::

    python -m handstand.sides [--data PATH] [--no-write]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import Counter
from collections.abc import Mapping, Sequence

import pandas as pd

from handstand import bakeoff, postprocess
from handstand.hold_shapes import load_hold_shapes
from handstand.paths import data_dir

__all__ = [
    "claimed_near_side",
    "hold_vote",
    "labelled_near_side",
    "main",
    "measure_holds",
    "measure_labels",
    "near_side_from_nose",
    "near_side_from_z",
    "summary_text",
]

#: The body parts each side is compared on, as the model names them with a
#: ``left_``/``right_`` prefix.
SIDE_PARTS: tuple[str, ...] = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")

#: Visibility a model joint needs before the rule may read it (holds only).
MIN_RULE_VISIBILITY = 0.5

#: Frames closer than this many pixels in x have no clear facing direction.
MIN_OFFSET_PX = 2.0
#: Visibility gap between sides below which the model makes no claim.
VISIBILITY_MARGIN = 0.1
#: Mean z gap between sides below which MediaPipe makes no claim.
Z_MARGIN = 0.05

LABEL_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "frame_idx",
    "gt_side",
    "rule_side",
    "model_vis_side",
    "model_z_side",
    "testable",
)
HOLD_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "hold_id",
    "n_frames",
    "n_rule",
    "rule_vote",
    "vis_vote",
    "z_vote",
    "frames_vis_disagree",
    "frames_z_disagree",
    "shape",
)

#: The athlete keypoint parquets the model checks read, relative to the data dir.
ATHLETE_DIR = ("keypoints", "mediapipe_athlete", "best")
#: The MediaPipe phase parquets that say which frames belong to a hold.
PHASE_DIR = ("phases", "mediapipe")
REPORT_DIR = ("reports", "sides")


def _finite(value: object) -> bool:
    """Is ``value`` a real number (not ``None``, NaN or infinity)?"""
    return isinstance(value, (int, float)) and math.isfinite(value)


def near_side_from_nose(
    nose_x: float | None,
    shoulder_mid_x: float | None,
    inverted: bool | None,
    *,
    min_offset_px: float = MIN_OFFSET_PX,
) -> str | None:
    """The near side by the nose rule, or ``None`` when it cannot be decided.

    ``None`` when either x is missing or NaN, when ``inverted`` is unknown, or
    when the nose sits within ``min_offset_px`` of the shoulder midpoint.
    """
    if not (_finite(nose_x) and _finite(shoulder_mid_x)) or inverted is None:
        return None
    offset = float(nose_x) - float(shoulder_mid_x)
    if abs(offset) < min_offset_px:
        return None
    facing_right = offset > 0
    return "right" if facing_right != bool(inverted) else "left"


def _side_means(
    left: Mapping[str, float], right: Mapping[str, float]
) -> tuple[float, float] | None:
    """Mean of each side over the parts both sides report, or ``None``."""
    shared = [
        part for part in left if part in right and _finite(left[part]) and _finite(right[part])
    ]
    if not shared:
        return None
    left_mean = sum(float(left[part]) for part in shared) / len(shared)
    right_mean = sum(float(right[part]) for part in shared) / len(shared)
    return left_mean, right_mean


def claimed_near_side(
    left: Mapping[str, float],
    right: Mapping[str, float],
    margin: float = VISIBILITY_MARGIN,
) -> str | None:
    """The side the model treats as near: the one with the higher visibility.

    ``None`` when no part is reported on both sides, or when the mean
    visibilities differ by less than ``margin``.
    """
    means = _side_means(left, right)
    if means is None:
        return None
    left_mean, right_mean = means
    if abs(left_mean - right_mean) < margin:
        return None
    return "left" if left_mean > right_mean else "right"


def near_side_from_z(
    left_z: Mapping[str, float],
    right_z: Mapping[str, float],
    margin: float = Z_MARGIN,
) -> str | None:
    """The MediaPipe z fallback: the side with the smaller mean z is nearer.

    ``None`` when no part is shared or the mean z values differ by less than
    ``margin``.
    """
    means = _side_means(left_z, right_z)
    if means is None:
        return None
    left_mean, right_mean = means
    if abs(left_mean - right_mean) < margin:
        return None
    return "left" if left_mean < right_mean else "right"


def labelled_near_side(gt: Mapping[str, tuple[float, float]]) -> str | None:
    """The side a human labelled as near, from the visible joints of one frame.

    Only one side is labelled in a side view. ``None`` when both sides or
    neither side have labelled parts: that frame is not testable.
    """
    left_count = sum(f"left_{part}" in gt for part in SIDE_PARTS)
    right_count = sum(f"right_{part}" in gt for part in SIDE_PARTS)
    if left_count and not right_count:
        return "left"
    if right_count and not left_count:
        return "right"
    return None


def hold_vote(sides: Sequence[str | None]) -> str | None:
    """The majority of the non-``None`` votes; ``None`` on a tie or no votes."""
    counts = Counter(side for side in sides if side is not None)
    ranked = counts.most_common()
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _shoulder_mid_x(joints: Mapping[str, tuple[float, float]]) -> float | None:
    """Mean x of the shoulders that exist in ``joints``, or ``None``."""
    xs = [
        float(joints[name][0])
        for name in ("left_shoulder", "right_shoulder")
        if name in joints and _finite(joints[name][0])
    ]
    return sum(xs) / len(xs) if xs else None


def _rule_side(joints: Mapping[str, tuple[float, float]]) -> str | None:
    """The nose rule applied to one body given as ``joint -> (x, y)``."""
    nose = joints.get("nose")
    nose_x = float(nose[0]) if nose is not None and _finite(nose[0]) else None
    return near_side_from_nose(nose_x, _shoulder_mid_x(joints), bakeoff.is_inverted(joints))


def _model_joints(
    table: pd.DataFrame,
    joints: Sequence[str],
    frames: Sequence[int] | None = None,
) -> dict[int, dict[str, tuple[float, float, float, float]]]:
    """``frame -> joint -> (x, y, z, visibility)`` from a keypoint long table."""
    wanted = set(joints)
    keep_frames = None if frames is None else set(frames)
    out: dict[int, dict[str, tuple[float, float, float, float]]] = {}
    for row in table.itertuples(index=False):
        name = str(row.joint)
        frame = int(row.frame_idx)
        if name not in wanted or (keep_frames is not None and frame not in keep_frames):
            continue
        out.setdefault(frame, {})[name] = (
            float(row.x),
            float(row.y),
            float(row.z),
            float(row.visibility),
        )
    return out


def _model_joint_names() -> tuple[str, ...]:
    """Every joint the model-side checks read: the nose and both sides' parts."""
    return (
        "nose",
        *(f"{side}_{part}" for side in ("left", "right") for part in SIDE_PARTS),
    )


def _side_values(
    joints: Mapping[str, tuple[float, float, float, float]],
    field: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """The left and right part values of one frame, ``field`` 2 = z, 3 = visibility."""
    left = {part: joints[f"left_{part}"][field] for part in SIDE_PARTS if f"left_{part}" in joints}
    right = {
        part: joints[f"right_{part}"][field] for part in SIDE_PARTS if f"right_{part}" in joints
    }
    return left, right


def _vis_and_z_sides(
    joints: Mapping[str, tuple[float, float, float, float]],
) -> tuple[str | None, str | None]:
    """The model's visibility claim and its z claim for one frame."""
    left_vis, right_vis = _side_values(joints, 3)
    left_z, right_z = _side_values(joints, 2)
    return claimed_near_side(left_vis, right_vis), near_side_from_z(left_z, right_z)


def _read_athlete(data: pathlib.Path, clip: str) -> pd.DataFrame | None:
    """The athlete keypoint table of ``clip``, or ``None`` when the file is absent."""
    path = data.joinpath(*ATHLETE_DIR, f"{clip}.parquet")
    if not path.is_file():
        return None
    return postprocess.read_clip(path).table


def _resolve(data: str | pathlib.Path | None) -> pathlib.Path:
    """The data directory: the argument, else ``$HANDSTAND_DATA``."""
    return pathlib.Path(data) if data is not None else data_dir()


def _none_if_unset(value: object) -> object:
    """``None`` for a missing value (``None`` or NaN), the value otherwise."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def measure_labels(data: str | pathlib.Path | None = None) -> pd.DataFrame:
    """One row per human-reviewed labelled frame: ground truth, rule and models.

    ``gt_side`` is the human label, ``rule_side`` the nose rule on the ground
    truth, ``model_vis_side``/``model_z_side`` the athlete parquet's own claims
    at that frame. A model column is ``None`` when its frame or clip file is
    missing, and ``testable`` is true when both ``gt_side`` and ``rule_side``
    are set.
    """
    root = _resolve(data)
    frames, context = bakeoff.load_frames(root)
    gt_visible = context["gt_visible"]
    kept = frames[frames["keep"].astype(bool)]
    model_cache: dict[str, dict[int, dict[str, tuple[float, float, float, float]]] | None] = {}
    rows: list[dict[str, object]] = []
    for clip, frame in zip(kept["clip_id"], kept["frame_idx"], strict=True):
        clip_id = str(clip)
        frame_idx = int(frame)
        gt = gt_visible.get((clip_id, frame_idx), {})
        gt_side = labelled_near_side(gt)
        rule_side = _rule_side(gt)
        if clip_id not in model_cache:
            table = _read_athlete(root, clip_id)
            model_cache[clip_id] = (
                None if table is None else _model_joints(table, _model_joint_names())
            )
        model_frames = model_cache[clip_id]
        joints = None if model_frames is None else model_frames.get(frame_idx)
        vis_side, z_side = (None, None) if joints is None else _vis_and_z_sides(joints)
        rows.append(
            {
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "gt_side": gt_side,
                "rule_side": rule_side,
                "model_vis_side": vis_side,
                "model_z_side": z_side,
                "testable": gt_side is not None and rule_side is not None,
            }
        )
    return pd.DataFrame(rows, columns=list(LABEL_COLUMNS))


def _hold_frames(phases: pd.DataFrame) -> dict[int, list[int]]:
    """``hold_id -> frame indices`` for the frames that belong to a hold."""
    held = phases[phases["hold_id"] >= 0]
    out: dict[int, list[int]] = {}
    for hold, frame in zip(held["hold_id"], held["frame_idx"], strict=True):
        out.setdefault(int(hold), []).append(int(frame))
    return out


def _hold_row(
    clip: str,
    hold: int,
    frames: Sequence[int],
    model: Mapping[int, dict[str, tuple[float, float, float, float]]],
    shape: str | None,
) -> dict[str, object]:
    """One hold's votes: each frame is read from the model, visible joints only."""
    rule_sides: list[str | None] = []
    vis_sides: list[str | None] = []
    z_sides: list[str | None] = []
    for frame in frames:
        joints = model.get(frame, {})
        visible = {
            name: (x, y)
            for name, (x, y, _z, vis) in joints.items()
            if vis >= MIN_RULE_VISIBILITY and _finite(x) and _finite(y)
        }
        rule_sides.append(_rule_side(visible))
        vis_side, z_side = _vis_and_z_sides(joints) if joints else (None, None)
        vis_sides.append(vis_side)
        z_sides.append(z_side)
    return {
        "clip_id": clip,
        "hold_id": hold,
        "n_frames": len(frames),
        "n_rule": sum(side is not None for side in rule_sides),
        "rule_vote": hold_vote(rule_sides),
        "vis_vote": hold_vote(vis_sides),
        "z_vote": hold_vote(z_sides),
        "frames_vis_disagree": _count_disagree(rule_sides, vis_sides),
        "frames_z_disagree": _count_disagree(rule_sides, z_sides),
        "shape": shape,
    }


def _count_disagree(
    left: Sequence[str | None],
    right: Sequence[str | None],
) -> int:
    """Frames where both sides are set and differ."""
    return sum(
        1 for a, b in zip(left, right, strict=True) if a is not None and b is not None and a != b
    )


def measure_holds(data: str | pathlib.Path | None = None) -> pd.DataFrame:
    """One row per ``(clip_id, hold_id)`` over the dataset, with the model votes.

    Clips come from the MediaPipe phase parquets; catalogue-missing clips and
    clips without an athlete parquet are skipped. Only frames with
    ``hold_id >= 0`` vote, and the rule reads each frame's joints with a
    visibility of at least :data:`MIN_RULE_VISIBILITY`.
    """
    root = _resolve(data)
    missing = bakeoff.read_missing_clips(root)
    shapes = load_hold_shapes(root)
    names = _model_joint_names()
    rows: list[dict[str, object]] = []
    phase_dir = root.joinpath(*PHASE_DIR)
    if not phase_dir.is_dir():
        return pd.DataFrame(rows, columns=list(HOLD_COLUMNS))
    for phase_path in sorted(phase_dir.glob("*.parquet")):
        clip = phase_path.stem
        if clip in missing:
            continue
        table = _read_athlete(root, clip)
        if table is None:
            continue
        phases = pd.read_parquet(phase_path, columns=["frame_idx", "hold_id"])
        holds = _hold_frames(phases)
        if not holds:
            continue
        all_frames = [frame for frames in holds.values() for frame in frames]
        model = _model_joints(table, names, all_frames)
        for hold in sorted(holds):
            rows.append(_hold_row(clip, hold, holds[hold], model, shapes.get((clip, hold))))
    return pd.DataFrame(rows, columns=list(HOLD_COLUMNS))


def _pct(part: int, total: int) -> str:
    """``part`` of ``total`` as a percentage, or a dash when there is nothing."""
    return "n/a" if total == 0 else f"{100.0 * part / total:.1f}%"


def _agreement(labels: pd.DataFrame, column: str) -> tuple[int, int]:
    """``(agreeing, compared)`` for one model column against the human label."""
    both = labels[labels[column].notna() & labels["gt_side"].notna()]
    agree = int((both[column] == both["gt_side"]).sum())
    return agree, len(both)


def _vote_differs(a: object, b: object) -> bool:
    """Do two votes differ? A missing vote counts as its own value."""
    return _none_if_unset(a) != _none_if_unset(b)


def _hold_block(holds: pd.DataFrame) -> list[str]:
    """The hold lines of the summary for one set of holds."""
    total = len(holds)
    no_nose = sum(_none_if_unset(v) is None for v in holds["rule_vote"])
    vis_diff = sum(
        _vote_differs(a, b) for a, b in zip(holds["rule_vote"], holds["vis_vote"], strict=True)
    )
    z_diff = sum(
        _vote_differs(a, b) for a, b in zip(holds["rule_vote"], holds["z_vote"], strict=True)
    )
    return [
        f"  holds: {total}",
        f"  holds with no nose vote: {no_nose} of {total} ({_pct(no_nose, total)})",
        f"  rule_vote != vis_vote: {vis_diff} of {total} ({_pct(vis_diff, total)})",
        f"  rule_vote != z_vote: {z_diff} of {total} ({_pct(z_diff, total)})",
    ]


def summary_text(labels_df: pd.DataFrame, holds_df: pd.DataFrame) -> str:
    """The plain-text report: label agreement, then the hold votes."""
    testable = labels_df[labels_df["testable"].astype(bool)]
    rule_agree = int((testable["rule_side"] == testable["gt_side"]).sum())
    vis_agree, vis_total = _agreement(labels_df, "model_vis_side")
    z_agree, z_total = _agreement(labels_df, "model_z_side")
    disagree = testable[testable["rule_side"] != testable["gt_side"]]
    lines = [
        "Labels (human-reviewed, kept frames)",
        f"  labelled frames: {len(labels_df)}",
        f"  testable (gt and rule both set): {len(testable)}",
        f"  rule vs gt: {rule_agree}/{len(testable)}",
        f"  model visibility vs gt: {vis_agree}/{vis_total}",
        f"  model z vs gt: {z_agree}/{z_total}",
        "  rule disagrees with gt at: "
        + (
            ", ".join(
                f"({c}, {f})"
                for c, f in zip(disagree["clip_id"], disagree["frame_idx"], strict=True)
            )
            or "none"
        ),
        "",
        "Holds (hold_id >= 0, whole dataset)",
        *_hold_block(holds_df),
        "Line holds",
        *_hold_block(holds_df[holds_df["shape"] == "line"]),
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """The ``python -m handstand.sides`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.sides",
        description="Measure how often the models' near side disagrees with the nose rule.",
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
        help="print the summary only; write no CSVs or summary.txt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m handstand.sides``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = _resolve(args.data)
    if not (root / "labels" / "keypoints.csv").is_file():
        print(
            f"sides: {root / 'labels' / 'keypoints.csv'} is missing; nothing to measure",
            file=sys.stderr,
        )
        return 2
    labels = measure_labels(root)
    holds = measure_holds(root)
    text = summary_text(labels, holds)
    print(text, end="")
    if not args.no_write:
        out = root.joinpath(*REPORT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        labels.to_csv(out / "labels.csv", index=False)
        holds.to_csv(out / "holds.csv", index=False)
        (out / "summary.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
