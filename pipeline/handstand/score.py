"""Score every hold of a clip against a reference of good holds: #29's weighted z-score.

A number for a handstand only means something next to a yardstick, so this stage
compares each hold of a clip with a **reference file** — the mean and standard
deviation of every feature over holds a human labelled good (#27 collects those
clips, #28 builds the reference). Nothing is bundled here: ``--reference`` is a
required argument, the tests build theirs inline, and no reference number is
committed to the public repository before it comes out of the labelled set.

Input and output
----------------

Input is the per-frame parquet :mod:`handstand.features` writes (read with
:func:`handstand.features.read_clip_features`, found through
:func:`handstand.features.output_dir` — no path is spelled out here)::

    <data_dir>/features/<source>/<clip_id>.parquet
    --reference PATH                        # handstand-reference, schema v1

Output is one CSV row per hold, plus a summary on stdout::

    <data_dir>/scores/<source>/scores.csv

CLI::

    cd pipeline
    uv run python -m handstand.score --reference REFERENCE.json --all
    uv run python -m handstand.score --reference REFERENCE.json --clip 057c9e6c96af

What a hold is made of
----------------------

Over the hold's measurable frames (``hold_id == h & valid``), each scored value
of :data:`SCORE_FEATURES` is the median over the finite values of one of the
fourteen per-frame features, plus ``com_sway_sd`` and ``hip_angle_sd`` — the
**population** SDs (``ddof = 0``, NaN under two frames) of ``com_forward`` and
``hip_angle``. The hold frame counts and times
(``hold_frames``/``valid_frames``/``hold_start_ms``/``hold_end_ms``/
``hold_duration_s``) are computed exactly the way
:func:`handstand.features.hold_rows` computes them. A hold with fewer than
:data:`MIN_SCORE_FRAMES` valid frames is not scored at all: there is no honest
median in four frames of a body nobody could see.

Five of the features — :data:`SIGNED_BY_FACING` — are written by #22/#23 in
**image** coordinates (+ = screen right), because that is what the geometry can
measure; the reference and the athlete's own body are not, so each is multiplied
by that frame's ``facing_sign`` first. A NaN sign makes that frame's value NaN,
which drops it out of the median like any other frame nobody measured.
``banana`` and ``com_forward`` are already athlete-signed, and the rest are
unsigned magnitudes.

The score
---------

Per feature, ``z = (value - mean) / max(sd, SD_FLOOR)`` (the floor keeps a
reference with a near-zero SD from turning ordinary noise into an infinite
score), then ``|z|`` is capped at :data:`Z_CAP`. Each group of :data:`GROUPS`
takes the mean capped ``|z|`` of the features in it that have both a finite hold
value and a reference entry — a group with none is *missing*. The deviation is
the weight-averaged group deviation over the groups that are there::

    D = sum(w_g * d_g) / sum(w_g)          over the groups that are not missing

and the score is::

    score = round(100 * max(0, 1 - (D + P) / Z_CAP), 1)

so 100 means every feature sits on its reference mean and 0 means an average of
:data:`Z_CAP` SDs away. ``P`` is the one-sided hip-correction penalty
(:data:`HIP_PENALTY_WEIGHT` times ``max(0, z(hip_angle_sd))``): a hip busier
than the reference's is a hip strategy paying for the balance, a quieter one is
not a fault. With every group missing the answer is no score at all —
``nothing to compare`` — and :meth:`HoldScore.top_faults` names the three
features carrying the largest share of the deviation, for the app's feedback.

Who owns what: #28 builds the real reference from the labelled good clips, #30
tunes :data:`GROUPS` and the constants against real data, and #41 ports this
module to Swift. The format is documented in ``docs/scoring.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import statistics
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from handstand import features
from handstand.paths import data_dir
from handstand.postprocess import DEFAULT_SOURCE, SOURCES

__all__ = [
    "GROUPS",
    "HIP_PENALTY_WEIGHT",
    "MIN_SCORE_FRAMES",
    "NO_REFERENCE_MESSAGE",
    "REFERENCE_SCHEMA",
    "REFERENCE_SIGNS",
    "REFERENCE_VERSION",
    "SCORES_COLUMNS",
    "SCORES_DIRNAME",
    "SCORES_NAME",
    "SCORE_FEATURES",
    "SD_FLOOR_DEG",
    "SD_FLOOR_L",
    "SIGNED_BY_FACING",
    "TOP_FAULTS",
    "Z_CAP",
    "HoldScore",
    "Reference",
    "build_arg_parser",
    "clip_score",
    "hold_values",
    "load_reference",
    "main",
    "output_dir",
    "score_clip",
    "score_hold",
    "scores_table",
    "summary",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Output root under ``<data_dir>/``, one directory per source exactly as
#: ``features/`` and ``phases/`` have one, so one source's scores never
#: overwrite another's.
SCORES_DIRNAME = "scores"

#: The per-clip hold scores of one source, one row per hold.
SCORES_NAME = "scores.csv"

#: What the CLI prints when the reference is missing or unreadable, pointing at
#: the stage that builds one — this stage never guesses a yardstick.
NO_REFERENCE_MESSAGE = "the reference comes from chainlink #28; pass --reference PATH"

#: The reference file's markers, validated by :func:`load_reference`: the name
#: identifies the file, ``version`` says which reader it was written for, and
#: ``signs`` says the numbers are already athlete-signed, which is what makes
#: them comparable with :func:`hold_values` at all.
REFERENCE_SCHEMA = "handstand-reference"
REFERENCE_VERSION = 1
REFERENCE_SIGNS = "athlete"

#: The five features #22/#23 write in **image** coordinates (+ = screen right),
#: converted to athlete-signed values by multiplying with that frame's
#: ``facing_sign`` before anything is summarised or scored. The geometry can
#: only measure "to the right in the picture", which is why the same pose
#: filmed from the other side reports the opposite number; the reference of
#: good holds is about the athlete's body, not about which side of it the
#: camera stood on, so without this flip a mirrored recording would look like
#: an opposite fault. ``banana`` and ``com_forward`` are already signed this
#: way by the features stage, and the remaining features are unsigned
#: magnitudes, so neither needs it.
SIGNED_BY_FACING: tuple[str, ...] = (
    "off_shoulder",
    "off_hip",
    "off_knee",
    "off_ankle",
    "body_angle",
)

#: The values a hold is scored on: the fourteen per-frame features (medians
#: over the hold) plus the two spreads. Not scored: ``hand_width``, which the
#: side view cannot trust, and everything else the parquet carries.
SCORE_FEATURES: tuple[str, ...] = (
    "off_shoulder",
    "off_hip",
    "off_knee",
    "off_ankle",
    "body_angle",
    "line_deviation",
    "shoulder_angle",
    "hip_angle",
    "knee_angle",
    "elbow_angle",
    "banana",
    "head",
    "leg_separation",
    "com_forward",
    "com_sway_sd",
    "hip_angle_sd",
)

#: The two scored values that are not medians of a column: they are spreads of
#: one, computed by :func:`hold_values` from the frames it already holds.
_DERIVED: tuple[str, ...] = ("com_sway_sd", "hip_angle_sd")

#: The fourteen features whose hold value is a median, in :data:`SCORE_FEATURES`
#: order: every scored feature but the two spreads.
_MEDIAN_FEATURES: tuple[str, ...] = tuple(
    name for name in SCORE_FEATURES if name not in _DERIVED
)

#: The groups a hold is judged in, as ``(name, weight, features)``, weights
#: summing to 1.0. A group is what a judge would say out loud ("the stack is
#: off"), and its deviation is the mean capped |z| over the features of it that
#: could be compared — so one wild feature cannot carry a group of six. The
#: weights are #30's to tune against real data; #29 only fixes that they start
#: here and sum to one.
GROUPS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    (
        "stack",
        0.25,
        ("off_shoulder", "off_hip", "off_knee", "off_ankle", "line_deviation", "body_angle"),
    ),
    # Open shoulders: the one piece of style advice every judge gives.
    ("shoulder", 0.20, ("shoulder_angle",)),
    # The line-vs-pike difference, and the curve that comes with it.
    ("hip", 0.20, ("hip_angle", "banana")),
    # Where the centre of mass sat and how far it wandered: balance, not shape.
    ("com", 0.15, ("com_forward", "com_sway_sd")),
    # A bent arm is a severe fault, which is why it speaks louder than its
    # share of the body.
    ("elbows", 0.10, ("elbow_angle",)),
    ("knees_toes", 0.05, ("knee_angle", "leg_separation")),
    ("head", 0.05, ("head",)),
)

#: Which group each feature is scored in — ``(group, weight, group size)`` —
#: which is exactly what :data:`HoldScore.top_faults` ranks by: a feature's
#: share of the deviation is ``weight * |z| / size``.
_GROUP_FOR: Mapping[str, tuple[str, float, int]] = {
    name: (group, weight, len(members))
    for group, weight, members in GROUPS
    for name in members
}

#: A hold with fewer valid frames than this is not scored: a median of four
#: frames of a body mostly out of view is a number, not a measurement.
MIN_SCORE_FRAMES = 5

#: The cap on |z|, and the yardstick the score is cut from: ``D + P`` of one
#: :data:`Z_CAP` scores 0, so four SDs away is "as wrong as a hold gets" and a
#: single absurd feature cannot drag an otherwise good hold to zero by itself.
Z_CAP = 4.0

#: The smallest reference SD used for a length feature, in body lengths: below
#: a hundredth of a body the spread between good holds is measurement noise,
#: and dividing by it would turn that noise into an infinite z.
SD_FLOOR_L = 0.01

#: The same floor for angles, in degrees: a reference whose good holds agree
#: within a degree cannot promise a tenth of one.
SD_FLOOR_DEG = 1.0

#: How much of a z of the hip's own sway the one-sided hip penalty takes: a hip
#: busier than the reference's is paying for the balance out of the score,
#: at most ``HIP_PENALTY_WEIGHT * Z_CAP`` (0.4) points of deviation.
HIP_PENALTY_WEIGHT = 0.10

#: How many faults :meth:`HoldScore.top_faults` reports: three is what a person
#: can act on, and the rest is in the ``z`` columns of the CSV.
TOP_FAULTS = 3

#: Which SD floor each scored feature is divided by: lengths in body lengths
#: (:data:`SD_FLOOR_L`), angles in degrees (:data:`SD_FLOOR_DEG`). The units
#: come from :data:`handstand.features.FEATURES` for the thirteen per-frame
#: features in it, and are spelled out for the three columns that are not —
#: both CoM numbers are in body lengths like every other length here.
_SD_FLOORS: Mapping[str, float] = {
    **{
        feature.name: SD_FLOOR_L if feature.unit == "L" else SD_FLOOR_DEG
        for feature in features.FEATURES
        if feature.name in SCORE_FEATURES
    },
    "com_forward": SD_FLOOR_L,
    "com_sway_sd": SD_FLOOR_L,
    "hip_angle_sd": SD_FLOOR_DEG,
}


# --------------------------------------------------------------------------- #
# The reference file
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Reference:
    """What a good hold looks like: per feature ``(mean, sd, n)`` plus provenance.

    ``features`` is keyed by the :data:`SCORE_FEATURES` a reference file
    carried — a feature left out is one that simply is not scored against it,
    never a zero — and ``n_holds``/``built_from`` say how many holds the
    numbers came from and which ones, so a reader can tell a reference built
    from a dozen clips apart from one built from the whole labelled set.
    """

    features: Mapping[str, tuple[float, float, int]]
    n_holds: int
    built_from: str


def load_reference(path: str | pathlib.Path) -> Reference:
    """Read and validate a reference file; ``docs/scoring.md`` is its schema.

    :class:`FileNotFoundError` when ``path`` is not a file, :class:`ValueError`
    naming the problem when anything else is off: the schema, version or signs
    marker, a mean or sd that is not a finite number, a negative sd, a count
    that is not a non-negative integer, or a feature name outside
    :data:`SCORE_FEATURES`. Features the file does not carry are fine — they
    just are not scored — which is what lets #28 ship a partial reference.
    """
    file = pathlib.Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"no reference file: {file}")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{file}: not valid JSON ({error})") from error
    return _build_reference(data, str(file))


def _build_reference(data: object, origin: str) -> Reference:
    """The parsed reference behind :func:`load_reference`, refusing to guess."""
    if not isinstance(data, dict):
        raise ValueError(f"{origin}: a reference must be a JSON object, got {data!r}")
    schema = data.get("schema")
    if schema != REFERENCE_SCHEMA:
        raise ValueError(f"{origin}: schema is {schema!r}, expected {REFERENCE_SCHEMA!r}")
    version = data.get("version")
    if version != REFERENCE_VERSION:
        raise ValueError(f"{origin}: version is {version!r}, expected {REFERENCE_VERSION}")
    signs = data.get("signs")
    if signs != REFERENCE_SIGNS:
        raise ValueError(
            f"{origin}: signs are {signs!r}, expected {REFERENCE_SIGNS!r} — the hold values "
            "this module scores are athlete-signed"
        )
    built_from = data.get("built_from")
    if not isinstance(built_from, str):
        raise ValueError(f"{origin}: built_from must be free text, got {built_from!r}")
    n_holds = _count(data.get("n_holds"), f"{origin}: n_holds")
    entries = data.get("features")
    if not isinstance(entries, dict):
        raise ValueError(f"{origin}: features must be an object, got {entries!r}")
    parsed: dict[str, tuple[float, float, int]] = {}
    for name, entry in entries.items():
        if name not in SCORE_FEATURES:
            raise ValueError(
                f"{origin}: {name!r} is not a scored feature (the keys are SCORE_FEATURES)"
            )
        parsed[name] = _feature_stats(entry, f"{origin}: features[{name!r}]")
    return Reference(features=parsed, n_holds=n_holds, built_from=built_from)


def _feature_stats(entry: object, origin: str) -> tuple[float, float, int]:
    """One feature's ``(mean, sd, n)``, with every part checked by name."""
    if not isinstance(entry, dict):
        raise ValueError(f"{origin} must be an object with mean, sd and n, got {entry!r}")
    mean = _finite(entry.get("mean"), f"{origin}.mean")
    sd = _finite(entry.get("sd"), f"{origin}.sd")
    if sd < 0.0:
        raise ValueError(f"{origin}.sd must be >= 0, got {sd}")
    return mean, sd, _count(entry.get("n"), f"{origin}.n")


def _finite(value: object, origin: str) -> float:
    """``value`` as a finite float, or :class:`ValueError` naming ``origin``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{origin} must be a finite number, got {value!r}")
    return float(value)


def _count(value: object, origin: str) -> int:
    """``value`` as a non-negative whole number, or :class:`ValueError`."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{origin} must be a non-negative integer, got {value!r}")
    return int(value)


# --------------------------------------------------------------------------- #
# The values a hold is scored on
# --------------------------------------------------------------------------- #


def _median_finite(values: np.ndarray) -> float:
    """The median of the finite values, NaN when there are none (nanmedian)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.median(finite))


def _sd_finite(values: np.ndarray) -> float:
    """The **population** SD of the finite values, NaN under two of them.

    The hold's frames are the whole population a judge saw, not a sample of a
    bigger one, so there is no N-1 correction — the same answer
    :func:`handstand.features.hold_stability` gives ``com_sway_sd``.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return math.nan
    return float(np.std(finite, ddof=0))


def hold_values(table: pd.DataFrame, hold_id: int) -> dict[str, float]:
    """One hold's athlete-signed values, keyed by :data:`SCORE_FEATURES`.

    The medians are taken over the hold's measurable frames (``hold_id == h``
    and ``valid``) and the two spreads over the same frames' finite values, all
    in the units the reference speaks: :data:`SIGNED_BY_FACING` is multiplied by
    each frame's ``facing_sign`` first (a NaN sign making that frame's value
    NaN), and a feature nobody could measure comes back as NaN rather than as
    a number out of nothing.

    ``table`` is a per-frame features table — what
    :func:`handstand.features.read_clip_features` returns — so it must carry
    ``hold_id``, ``valid``, ``facing_sign``, ``com_forward``, ``hip_angle`` and
    every feature in :data:`SCORE_FEATURES`.
    """
    mask = (table["hold_id"].to_numpy() == hold_id) & table["valid"].to_numpy(dtype=bool)
    facing = table["facing_sign"].to_numpy(dtype=float)
    values: dict[str, float] = {}
    for name in _MEDIAN_FEATURES:
        column = table[name].to_numpy(dtype=float)
        if name in SIGNED_BY_FACING:
            column = column * facing
        values[name] = _median_finite(column[mask])
    values["com_sway_sd"] = _sd_finite(table["com_forward"].to_numpy(dtype=float)[mask])
    values["hip_angle_sd"] = _sd_finite(table["hip_angle"].to_numpy(dtype=float)[mask])
    return values


# --------------------------------------------------------------------------- #
# Scoring one hold
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class HoldScore:
    """One hold's score, and everything the score was made of.

    ``hold_frames`` counts every frame the hold spans while ``valid_frames``
    counts the ones the values could be measured on, and the two times and the
    duration are first frame to last — the same row
    :func:`handstand.features.hold_rows` writes, so a score and a hold summary
    line up. ``score`` is ``None`` with a ``reason`` when the hold could not be
    scored; ``deviation`` (D) and ``penalty`` (P) are what it would be scored
    from, ``groups`` the per-group deviations (``None`` where the group is
    missing), ``values`` and ``z`` the hold's values and their capped z-scores,
    ``top_faults`` the three worst features as ``(feature, value, mean, z)``,
    and ``missing_groups`` which groups had nothing to compare.
    """

    hold_id: int
    hold_frames: int
    valid_frames: int
    hold_start_ms: int
    hold_end_ms: int
    hold_duration_s: float
    #: ``None`` exactly when ``reason`` says why there is no score.
    score: float | None
    #: "" for a scored hold, else why it is not ("too few valid frames",
    #: "nothing to compare").
    reason: str
    deviation: float
    penalty: float
    groups: dict[str, float | None]
    values: dict[str, float]
    z: dict[str, float]
    top_faults: tuple[tuple[str, float, float, float], ...]
    missing_groups: tuple[str, ...]


def score_hold(table: pd.DataFrame, hold_id: int, reference: Reference) -> HoldScore:
    """Score one hold of a clip against ``reference``, or say why it is not scored.

    Three answers are possible: a score out of 100; ``nothing to compare``,
    when the reference carries no feature of any group that this hold could
    offer a value for; and ``too few valid frames``, below
    :data:`MIN_SCORE_FRAMES`, which is decided before anything is measured.
    """
    hold_ids = table["hold_id"].to_numpy()
    t_ms = table["t_ms"].to_numpy()
    valid = table["valid"].to_numpy(dtype=bool)
    rows = np.flatnonzero(hold_ids == hold_id)
    if rows.size:
        hold_frames = int(rows.size)
        valid_frames = int(valid[rows].sum())
        hold_start_ms = int(t_ms[rows[0]])
        hold_end_ms = int(t_ms[rows[-1]])
        hold_duration_s = round((hold_end_ms - hold_start_ms) / 1000.0, 3)
    else:
        hold_frames, valid_frames = 0, 0
        hold_start_ms = hold_end_ms = 0
        hold_duration_s = 0.0

    fields = (hold_id, hold_frames, valid_frames, hold_start_ms, hold_end_ms, hold_duration_s)
    if valid_frames < MIN_SCORE_FRAMES:
        return HoldScore(
            *fields,
            score=None,
            reason="too few valid frames",
            deviation=0.0,
            penalty=0.0,
            groups={group: None for group, _, _ in GROUPS},
            values={},
            z={},
            top_faults=(),
            missing_groups=tuple(group for group, _, _ in GROUPS),
        )

    values = hold_values(table, hold_id)
    z: dict[str, float] = {}
    for name in SCORE_FEATURES:
        entry = reference.features.get(name)
        if entry is None or not math.isfinite(values[name]):
            continue
        mean, sd, _ = entry
        raw = (values[name] - mean) / max(sd, _SD_FLOORS[name])
        # The z the score is made of: signed, but never beyond the cap, so one
        # broken feature is worth at most Z_CAP like any other.
        z[name] = float(np.clip(raw, -Z_CAP, Z_CAP))

    groups: dict[str, float | None] = {}
    missing: list[str] = []
    weighted, weights = 0.0, 0.0
    for group, weight, members in GROUPS:
        comparable = [z[name] for name in members if name in z]
        if not comparable:
            groups[group] = None
            missing.append(group)
            continue
        deviation = float(np.mean(np.abs(comparable)))
        groups[group] = deviation
        weighted += weight * deviation
        weights += weight

    hip_sway = z.get("hip_angle_sd")
    penalty = HIP_PENALTY_WEIGHT * max(0.0, hip_sway) if hip_sway is not None else 0.0

    if weights > 0.0:
        deviation = weighted / weights
        score = round(100.0 * max(0.0, 1.0 - (deviation + penalty) / Z_CAP), 1)
        reason = ""
    else:
        deviation, score = 0.0, None
        reason = "nothing to compare"

    return HoldScore(
        *fields,
        score=score,
        reason=reason,
        deviation=deviation,
        penalty=penalty,
        groups=groups,
        values=values,
        z=z,
        top_faults=_top_faults(values, z, reference),
        missing_groups=tuple(missing),
    )


def _top_faults(
    values: Mapping[str, float],
    z: Mapping[str, float],
    reference: Reference,
) -> tuple[tuple[str, float, float, float], ...]:
    """The ``TOP_FAULTS`` features carrying the largest share of the deviation.

    A feature's share of the score is its group's weight times its capped |z|,
    over the number of features the group spreads that weight across — so a
    two-SD hip speaks louder than a two-SD knee. Ties keep
    :data:`SCORE_FEATURES` order, and ``hip_angle_sd`` is never one: no group
    takes it, it is the penalty's own input.
    """
    ranked = sorted(
        (name for name in z if name in _GROUP_FOR),
        key=lambda name: -(_GROUP_FOR[name][1] * abs(z[name]) / _GROUP_FOR[name][2]),
    )
    return tuple(
        (name, values[name], reference.features[name][0], z[name]) for name in ranked[:TOP_FAULTS]
    )


def score_clip(table: pd.DataFrame, reference: Reference) -> list[HoldScore]:
    """Score every hold of a clip, in ``hold_id`` order.

    Frames outside a hold (``hold_id == -1``) are not holds and are skipped:
    a warm-up, a bail and a hand step are all one hold each or none.
    """
    hold_ids = sorted({int(value) for value in table["hold_id"].to_numpy() if int(value) >= 0})
    return [score_hold(table, hold_id, reference) for hold_id in hold_ids]


def clip_score(scores: list[HoldScore]) -> HoldScore | None:
    """The hold a clip is represented by: its longest **scored** hold.

    A short hold is a hop on the hands and its median says more about balance
    than about form, so the clip's score is the longest hold anyone could
    actually score, ties going to the earlier one — the same rule
    :meth:`handstand.features.ClipFeatures.longest_hold_id` picks a clip's
    hold by. ``None`` when no hold of the clip was scored.
    """
    scored = [hold for hold in scores if hold.score is not None]
    if not scored:
        return None
    return max(scored, key=lambda hold: hold.hold_duration_s)


# --------------------------------------------------------------------------- #
# The CSV and the summary
# --------------------------------------------------------------------------- #

#: The scores CSV's columns, in write order: the clip and the hold row from
#: :func:`handstand.features.hold_rows`, the score and what it was made of
#: (deviation, penalty, one column per group), then every value and every z
#: under ``value_``/``z_`` prefixes, and the faults as text.
SCORES_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "hold_id",
    "hold_frames",
    "valid_frames",
    "hold_start_ms",
    "hold_end_ms",
    "hold_duration_s",
    "score",
    "reason",
    "deviation",
    "penalty",
    *(f"dev_{group}" for group, _, _ in GROUPS),
    *(f"value_{name}" for name in SCORE_FEATURES),
    *(f"z_{name}" for name in SCORE_FEATURES),
    "top_faults",
)


def _faults_text(hold: HoldScore) -> str:
    """The top faults as one CSV cell: ``feature z=+2.00 value=0.06 mean=0.00; ...``."""
    return "; ".join(
        f"{name} z={z:+.2f} value={value:.4g} mean={mean:.4g}"
        for name, value, mean, z in hold.top_faults
    )


def scores_table(results: Sequence[tuple[str, HoldScore]]) -> pd.DataFrame:
    """The scored holds as the ``scores.csv`` rows, in clip and hold order."""
    rows: list[dict[str, object]] = []
    for clip_id, hold in results:
        row: dict[str, object] = {
            "clip_id": clip_id,
            "hold_id": hold.hold_id,
            "hold_frames": hold.hold_frames,
            "valid_frames": hold.valid_frames,
            "hold_start_ms": hold.hold_start_ms,
            "hold_end_ms": hold.hold_end_ms,
            "hold_duration_s": hold.hold_duration_s,
            "score": hold.score,
            "reason": hold.reason,
            "deviation": hold.deviation,
            "penalty": hold.penalty,
        }
        for group, _, _ in GROUPS:
            row[f"dev_{group}"] = hold.groups.get(group)
        for name in SCORE_FEATURES:
            row[f"value_{name}"] = hold.values.get(name, math.nan)
            row[f"z_{name}"] = hold.z.get(name, math.nan)
        row["top_faults"] = _faults_text(hold)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(SCORES_COLUMNS))


#: How many holds each ranking lists: a list a person reads is short, the
#: whole run is in the CSV.
LISTED = 5


def summary(results: Sequence[tuple[str, HoldScore]]) -> str:
    """The block the CLI prints: how much was scored, and the extremes.

    The five best and five worst scored holds (ties to the earlier clip and
    hold, like everything else here) are what a person looks at first; the
    whole run is one filter away in ``scores.csv``.
    """
    scored = [(clip_id, hold) for clip_id, hold in results if hold.score is not None]
    reasons: dict[str, int] = {}
    for _, hold in results:
        if hold.score is None:
            reasons[hold.reason] = reasons.get(hold.reason, 0) + 1
    clips = len({clip_id for clip_id, _ in results})
    lines = [
        f"clips={clips} holds={len(results)} scored={len(scored)} "
        f"unscored={len(results) - len(scored)}"
    ]
    if reasons:
        lines.append(
            "not scored: "
            + ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
        )
    if scored:
        scores = sorted(float(hold.score) for _, hold in scored)
        lines.append(
            f"score min={scores[0]:.1f} median={statistics.median(scores):.1f} max={scores[-1]:.1f}"
        )
        ordered = sorted(scored, key=lambda item: (-float(item[1].score), item[0], item[1].hold_id))
        lines.append("best:")
        lines.extend(_ranking_lines(ordered[:LISTED]))
        lines.append("worst:")
        lines.extend(_ranking_lines(list(reversed(ordered))[:LISTED]))
    return "\n".join(lines)


def _ranking_lines(ranked: Sequence[tuple[str, HoldScore]]) -> list[str]:
    """One indented line per hold of a ranking, naming its top fault."""
    lines = []
    for clip_id, hold in ranked:
        fault = hold.top_faults[0][0] if hold.top_faults else "-"
        lines.append(f"  {clip_id} hold={hold.hold_id} score={hold.score:.1f} top={fault}")
    return lines


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def output_dir(data: str | pathlib.Path, source: str = DEFAULT_SOURCE) -> pathlib.Path:
    """``<data_dir>/scores/<source>``, where that source's scores go."""
    return pathlib.Path(data) / SCORES_DIRNAME / source


def features_hint(source: str = DEFAULT_SOURCE) -> str:
    """The error shown when a clip's features do not exist yet, in #22's words."""
    command = "uv run python -m handstand.features --all"
    if source != DEFAULT_SOURCE:
        command += f" --source {source}"
    return f"generate them with: cd pipeline && {command}"


def build_arg_parser() -> argparse.ArgumentParser:
    """The argument parser of ``python -m handstand.score``."""
    parser = argparse.ArgumentParser(
        prog="python -m handstand.score",
        description="Score every hold of a clip against a reference of good holds.",
    )
    parser.add_argument(
        "--reference",
        type=pathlib.Path,
        required=True,
        metavar="PATH",
        help=(
            "JSON reference of good holds (handstand-reference, schema v1); "
            "chainlink #28 builds the real one"
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--clip",
        nargs="+",
        metavar="CLIP_ID",
        default=None,
        help="clip ids to score",
    )
    selection.add_argument(
        "--all",
        dest="all_clips",
        action="store_true",
        help="score every clip with features",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help=f"which features to read (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="data directory holding features/ and scores/ (default: $HANDSTAND_DATA)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Score the requested clips, write ``scores.csv`` and print the summary.

    Exit codes: 2 when the reference is missing or invalid (the yardstick comes
    from #28, and nothing here can stand in for it), 1 when a clip has no
    features to score, 0 otherwise.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = pathlib.Path(args.data) if args.data is not None else data_dir()

    try:
        reference = load_reference(args.reference)
    except (OSError, ValueError) as error:
        print(f"score: {error}")
        print(NO_REFERENCE_MESSAGE)
        return 2

    in_root = features.output_dir(root, args.source)
    clips = None if args.all_clips else args.clip
    try:
        clip_ids = features.available_clips(in_root, clips, args.source)
    except FileNotFoundError:
        # No features directory at all: there is nothing to score yet.
        clip_ids = []
    absent = [clip_id for clip_id in clip_ids if not (in_root / f"{clip_id}.parquet").is_file()]
    if absent or not clip_ids:
        wanted = ", ".join(absent) if absent else str(in_root)
        print(f"score: no features for {wanted}")
        print(features_hint(args.source))
        return 1

    out_root = output_dir(root, args.source)
    results: list[tuple[str, HoldScore]] = []
    failures = 0
    for clip_id in clip_ids:
        try:
            table = features.read_clip_features(in_root / f"{clip_id}.parquet")
        except Exception as error:  # one unreadable clip must not kill the batch
            failures += 1
            print(f"fail  {clip_id}: {error}")
            continue
        holds = score_clip(table, reference)
        results.extend((clip_id, hold) for hold in holds)
        print(f"score {clip_id} holds={len(holds)}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / SCORES_NAME
    scores_table(results).to_csv(path, index=False)
    print(f"scores {path} ({len(clip_ids)} clip(s), {len(results)} hold(s))")
    print(summary(results))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
