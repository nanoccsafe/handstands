# Scoring

How a hold becomes a number: `handstand.score` (#29) compares each hold of a
clip against a **reference** — the mean and standard deviation of every feature
over holds a human labelled good — and reports one score out of 100, plus the
values, z-scores and worst faults it was made of.

Two other issues own the parts this one deliberately does not decide:

- **#28 builds the real reference** from the reviewed side-view line holds; see
  [Building the reference](#building-the-reference). Nothing here ships one:
  `--reference` is a required argument, there is no bundled default, and no
  reference number is committed to this public repository before it comes out
  of the labelled set. The tests build theirs inline.
- **#30 tunes the weights.** The groups and their starting weights live at the
  top of `pipeline/handstand/score.py`, each with a comment saying what it is
  for, and they are the only thing that is expected to change.

The Swift port is #41: `swift/HandstandCore/Sources/HandstandCore/Scorer.swift`
mirrors this document and `pipeline/handstand/score.py` — same constants, same
groups, same NaN rules, same rounding — so the score on the device is the score
the pipeline writes.

## Input

The per-frame features `handstand.features` writes (#22/#23), found through
`features.output_dir` / `features.read_clip_features`:

```
<data_dir>/features/<source>/<clip_id>.parquet
```

Columns read: `t_ms`, `hold_id` (`-1` = not in a hold), `valid`, the 14
features, `com_forward` and `facing_sign`.

### The values a hold is scored on

Over a hold's *measurable* frames (`hold_id == h & valid`):

- the **median** over the finite values (`np.nanmedian` semantics; NaN when
  there are none) of `off_shoulder`, `off_hip`, `off_knee`, `off_ankle`,
  `body_angle`, `line_deviation`, `shoulder_angle`, `hip_angle`, `banana`,
  `elbow_angle`, `knee_angle`, `leg_separation`, `head` and `com_forward` —
  these fourteen, athlete-signed where signing applies (below), are
  `SCORE_FEATURES`' first fourteen entries;
- `com_sway_sd` — the **population** SD (`ddof = 0`) of the finite
  `com_forward` values, NaN under two frames;
- `hip_angle_sd` — the population SD of the finite `hip_angle` values, NaN
  under two frames.

The hold's `hold_frames`, `valid_frames`, `hold_start_ms`, `hold_end_ms` and
`hold_duration_s` are computed exactly the way `features.hold_rows` computes
them (all frames of the hold, first frame to last, duration in seconds), so a
score row lines up with a hold-summary row.

A hold with `valid_frames < MIN_SCORE_FRAMES` (5) is **not scored**: score
`None`, reason `"too few valid frames"`.

## Signs

`off_shoulder`, `off_hip`, `off_knee`, `off_ankle` and `body_angle` are
written in **image** coordinates (+ = screen right) by the features stage —
that is what the geometry can measure — so before anything is summarised or
scored each is multiplied by that frame's `facing_sign`. Without it the same
pose filmed from the other side of the athlete would report the opposite
number and look like the opposite fault. The five names are the module
constant `SIGNED_BY_FACING`.

A frame with a NaN `facing_sign` gives NaN for those five, which simply drops
the frame out of the median like any other frame nobody measured; if the whole
hold's sign is unknown, the stack group falls back to its unsigned feature
(`line_deviation`) or goes missing. `banana` and `com_forward` are already
athlete-signed by the features stage; everything else is an unsigned
magnitude.

## The formula

Per scored feature, with the reference's `(mean, sd)`:

```
z = (value - mean) / max(sd, SD_FLOOR)        SD_FLOOR = 0.01 L for lengths,
                                              1.0 deg for angles
z capped at |z| <= Z_CAP = 4.0
```

Each **group** of the table below takes the mean capped `|z|` over its
features that have *both* a finite hold value and a reference entry; a group
with none is `missing` (recorded in `missing_groups`). The deviation is the
weight-averaged group deviation over the groups that are there — so missing
groups renormalise the weights:

```
D = sum(w_g * d_g) / sum(w_g)                 over the groups that are not missing
```

With every group missing: score `None`, reason `"nothing to compare"`.

**Hip-correction penalty** (one-sided):

```
P = HIP_PENALTY_WEIGHT (0.10) * max(0, z(hip_angle_sd))     z capped at Z_CAP
```

Only a hip *busier* than the reference is penalised — a hip balancing more
than the reference suggests a hip strategy — and it is 0 when `hip_angle_sd`
or its reference entry is missing.

**The score:**

```
score = round(100 * max(0, 1 - (D + P) / Z_CAP), 1)
```

100 means every feature sits at the reference mean; 0 means an average of
`Z_CAP` SDs off.

**Top faults:** the three features with the largest
`w_g * capped |z| / (number of features in that group)`, reported as
`(feature, athlete-signed value, reference mean, z)` — what the app will show
as feedback later. `hip_angle_sd` is in no group (it is the penalty alone),
so it is never a top fault.

### Groups and weights

The weights sum to 1.0 (`GROUPS` in `handstand/score.py`; #30 tunes them):

| group | weight | features |
|---|---|---|
| `stack` | 0.25 | `off_shoulder`, `off_hip`, `off_knee`, `off_ankle`, `line_deviation`, `body_angle` |
| `shoulder` | 0.20 | `shoulder_angle` |
| `hip` | 0.20 | `hip_angle`, `banana` |
| `com` | 0.15 | `com_forward`, `com_sway_sd` |
| `elbows` | 0.10 | `elbow_angle` |
| `knees_toes` | 0.05 | `knee_angle`, `leg_separation` |
| `head` | 0.05 | `head` |

`SCORE_FEATURES` is these fifteen plus `hip_angle_sd`, which only feeds the
penalty.

## The reference file

JSON, schema v1 — this is the whole format:

```json
{"schema": "handstand-reference", "version": 1, "signs": "athlete",
 "built_from": "free text: which holds, when", "n_holds": 42,
 "features": {
"hip_angle": {
 "mean": 176.2, "sd": 3.1, "n": 42}, "...": {}}}
```

- `features` is keyed by exactly the `SCORE_FEATURES` names (the fourteen
  medians above plus `com_sway_sd` and `hip_angle_sd`); a key outside them is
  refused.
- `load_reference(path)` validates the `schema`/`version`/`signs` markers and
  that every `mean`/`sd` is a finite number with `sd >= 0`, raising
  `ValueError` naming the problem (a missing file raises
  `FileNotFoundError`).
- **Missing features are allowed**: they just are not scored — the hold is
  compared where the reference has something to say and its group goes
  missing otherwise.
- `signs: "athlete"` is what makes the numbers comparable with the hold
  values above; a reference in image coordinates would score mirrored clips
  backwards.

## Building the reference

The real reference is built from the user's own reviewed data — **never
committed** to this public repository, and no reference number appears before it
comes out of the labelled set:

```sh
cd pipeline
uv run python -m handstand.reference build [--data DIR] [--source SRC]
    [--out PATH] [--min-holds N] [--install] [--any-view] [--labels-only]
```

`handstand.reference` reads:

- `data/labels/reference_review.csv` (#27) for the holds the user marked
  reference-worthy and not against a wall;
- `data/catalogue.csv`'s `camera_angle` to keep **side-view holds only** by
  default: the side-view features (hip angle, leg separation, knee) are only
  honest from the side, and the review records the camera nowhere. `camera_angle`
  is a person's answer until view detection (#74) exists; a clip counts as side
  view when the cell is `"side"` after stripping and lower-casing. A hold in a
  `front` or angled clip, or in a clip whose cell is empty, is skipped with a
  note and counted in the report header. `--any-view` builds from every reviewed
  hold regardless of the camera, and a missing catalogue is an error unless it is
  given;
- `data/features/<source>/<clip_id>.parquet` (#22/#23) for one value per hold —
  the same athlete-signed `score.hold_values` medians and spreads a hold is
  scored on.

For a clip whose `camera_angle` cell is **empty** the build falls back to the
detected view: run

```sh
uv run python -m handstand.view build [--data DIR] [--source SRC]
```

first, which writes `data/features/<source>/hold_views.csv` (one row per hold)
and prints a validation report at `data/reports/views/`. The hold is kept when
that table calls it `side`; otherwise it is skipped with a note naming the
detected view (`none` when there is no row). The hand label always wins, so a
`front` clip is dropped however the detector reads it. With no `hold_views.csv`
the build prints a note and only hand-labelled side clips count, and
`--labels-only` asks for exactly that even when the table exists. The
`other line holds in side-view clips` line of the sanity report uses the same
rule per hold, so it compares side views with side views.

### In the app

The app scores with the same rule and the same constants
(`swift/HandstandCore/Sources/HandstandCore/ViewDetector.swift`, the port of
`handstand.view`): every hold's camera view is measured on device from the
shoulder and hip separations, and `Analysis.sideClipScore` is the clip's
longest scored hold that was not filmed from the front or at an angle. A hold
whose median lands in the band between `SIDE_MAX` (0.035) and `NOT_SIDE_MIN`
(0.042), or with fewer than `MIN_VIEW_FRAMES` (15) measurable frames, is
`unsure` and still scores — the band is not evidence either way. A take whose
**every** scored hold is `not_side` shows no number at all: the panel and the
History row say `No score: film from the side for a score` instead, and an
`unsure` view is shown with a "the score assumes a side view" footnote.

Each **hold** contributes one value per feature, so every hold weighs the same
and no long hold outvotes a short one; frames are never pooled. Per feature it
writes the mean, the **sample** SD (`ddof = 1`) and `n` over the finite hold
values, and a feature finite in fewer than `MIN_FEATURE_HOLDS` (5) holds is left
out — missing features are allowed by the format. It writes
`<data_dir>/references/reference-line.json` by default, prints `n_holds` and the
features written, and writes a sanity report to
`<data_dir>/reports/reference/<YYYY-MM-DD_HHMMSS>.txt`: the per-feature table, a
leave-one-out of the reference holds, every other line hold scored against the
full reference, and the subset of those in side-view clips (the only comparable
ones). `--min-holds` (default 10) refuses a reference built from too few holds,
and `--install` copies the file to `ios/LocalResources/reference-line.json`, the
name the app's `ReferenceLoader` looks for.

## CLI

```sh
cd pipeline
uv run python -m handstand.score --reference PATH (--clip ID ... | --all) [--data DIR] [--source SRC]
```

Writes `<data_dir>/scores/<source>/scores.csv`: one row per hold with
`clip_id`, the hold fields, `score`, `reason`, `deviation`, `penalty`, the
group deviations (`dev_<group>`), every value (`value_<feature>`) and z-score
(`z_<feature>`), and `top_faults` as text.

Prints a short summary: clips, holds scored and not scored (by reason), the
score min/median/max, and the 5 best and 5 worst holds.

Exit codes:

- **2** — bad or missing reference file, with
  `the reference comes from chainlink #28; pass --reference PATH`;
- **1** — no features exist for a clip, with the same hint features.py uses:
  `generate them with: cd pipeline && uv run python -m handstand.features --all`.

## Tests

`pipeline/tests/test_score.py` builds every table and reference inline (small
synthetic frames, no real data): a hold at the reference means scores 100.0,
one feature 2 SDs off moves only its group by the hand-computed amount, the
cap and the floors, the facing-sign flip and its NaN, the one-sided hip
penalty, weight renormalisation, the two "not scored" reasons, the population
SD, `clip_score`, reference validation and the CLI's CSV and exit codes.
