# Golden fixtures

One JSON file per case: a **synthetic** clip, the real Python pipeline's answer
to it, and the tolerance a Swift port may differ by. They are the parity checks
for the Swift ports of post-process (chainlink #39), phases + features (#40)
and the scorer (#41): a port is right when it reproduces `expected`, frame by
frame, inside `meta.tolerances`.

Regenerate them from the repository root:

```sh
cd pipeline
uv run python -m handstand.golden                    # every case
uv run python -m handstand.golden --only line_hold   # one of them
uv run pytest tests/test_golden.py                   # what the tests check
```

The generator (`pipeline/handstand/golden.py`) draws a kinematic stick figure
from a timeline — stand, plant, kick up, hold, come down, stand up — and runs
the **real** `handstand.postprocess`, `handstand.phases` and
`handstand.features` over it in memory, calling them as libraries. No file ever
touches the shared data directory, and the coordinates are rounded to the
writing precision *before* the pipeline sees them, so the committed `input` is
byte-for-byte the input the committed `expected` was computed from.

## These files are synthetic. Always.

The repository is public and the real videos' keypoints never go in git. Every
fixture here is generated from a seed (`meta.seed`) and carries
`meta.mode == "synthetic"`; a test asserts it. The same format can be produced
for a **real** clip with

```sh
cd pipeline
uv run python -m handstand.golden --real 6508f9b355bd
```

but `--real` only ever writes outside the checkout — into
`<data_dir>/golden_real/` (git-ignored) by default — and
`golden.require_outside_repo` refuses any `--real` output path inside the
repository, before anything is written. Real fixtures are never written under
`swift/`.

## Format

```json
{
 "meta":    { … },
 "input":   [ {frame}, … ],
 "expected": {
  "postprocess": { "body_length": {…}, "frames": [{frame}, …] },
  "phases":      [ {"phase": "hold", "hold_id": 0}, … ],
  "features":    [ {"off_shoulder": 0.0, …}, … ],
  "hold_summary":[ { … } ]
 }
}
```

Objects are indented one space at a time; a **row** — one frame, one hold — is
one line, and a list of numbers or booleans is inline, so a file stays greppable
and diffable. Every float is rounded to 1e-6 and every NaN is written `null`
(JSON has no NaN). The row arrays are all the same length as `input`, one entry
per frame, in frame order.

### `meta`

| key | meaning |
|---|---|
| `case`, `mode` | the case's name; `"synthetic"` here, `"real"` for a `--real` export |
| `generator`, `generator_version` | `handstand.golden` and the layout version this file follows |
| `seed` | the synthetic seed (`null` for a real clip) |
| `display`, `L` | the display size the pixels are in (576x1024), and the generator's nominal body length: torso 105 + thigh 100 + shin 95 px |
| `body_length_px` | what `handstand.postprocess` actually measured — the sum of the three 90th-percentile parts; the real `L` every value below is in |
| `frame_count`, `hold_count` | rows, and how many holds the phases stage found |
| `joint_names` | the joints, **in the order every array below uses** |
| `feature_columns` | the columns of each `expected.features` row, in order |
| `packages` | python / handstand / numpy / pandas versions the file was produced with |
| `tolerances` | the tolerance to compare each field at (below) |

`joint_names` is `handstand.postprocess.TRACKED_JOINTS`: the nose, both
shoulders, elbows, wrists, hips, knees, ankles and foot indexes — the fifteen
names `swift/HandstandCore`'s `Joint` enum knows. The model's other eighteen
landmarks (eyes, ears, mouth, fingers, heels) feed no stage being ported, so
they are not in a fixture.

### `input`

One object per frame:

| key | meaning |
|---|---|
| `frame_idx`, `t_ms` | the frame and its timestamp: strictly increasing, and deliberately **variable** frame rate around 30 fps, because every threshold in the pipeline is stated in seconds |
| `detected` | did the model find a pose; `false` frames carry `[null, null, null]` for every joint, which is the schema's NaN |
| `trainer_contact` | did the trainer overlap or touch the athlete (the athlete selection's flag) |
| `joints` | `{name: [x, y, visibility]}` in display pixels, `y` down, `docs/keypoint_schema.md` |

### `expected`

What the three real stages answered.

* **`postprocess.body_length`** — `usable`, the `reason` when it is not, the
  three parts (`torso_px`, `thigh_px`, `shin_px`), their sum (`total_px`) and
  how many frames each part was measurable on (`frames`).
* **`postprocess.frames`** — one row per frame:
  * `valid`, `filled`: **arrays in `meta.joint_names` order**, one boolean per
    joint. `valid` is "has a position at all"; `filled` is "the position is a
    bridge interpolated across a gap, not a measurement". They are arrays
    rather than one boolean per frame because gating, outlier removal and gap
    filling act on a *joint*, not on a frame;
  * `joints`: `{name: [x, y]}`, the processed position, `null` where `valid` is
    false.
* **`phases`** — `{phase, hold_id}` per frame: `phase` is one of `pre`,
  `kickup`, `hold`, `exit`, `post`, `unknown`; `hold_id` counts from 0 and is
  `-1` outside a hold. Frames bridged into a hold (a trainer in the way) keep
  the hold's phase and number.
* **`features`** — one row per frame holding `meta.feature_columns`: the
  stacking offsets (`off_*`), `line_deviation`, `body_angle`, the four joint
  angles, `banana`, `head`, `leg_separation`, then `com_u`, `com_v`,
  `com_forward`, `facing_sign` and `balance_zone` (a string, or `null`).
  Measured on the processed trajectory, so they carry the One-Euro filter's
  numbers, and NaN where the frame could not support them.
* **`hold_summary`** — one object per hold, straight out of
  `handstand.features.hold_rows`: `clip_id`, `source`, `hold_id`, the hold's
  frame counts and times, then `{feature}_median` and `{feature}_iqr` for every
  feature and the balance columns (`com_forward_median`, `com_sway_sd`,
  `com_sway_range`, `com_speed_rms`, `hip_angle_sd`, `shoulder_angle_sd`,
  `pct_over`, `pct_under`).

### `tolerances`

The absolute tolerance per field, as `meta.tolerances`; a key is the dotted path
of the field it covers and the longest match wins (`default` is the fallback):

| key | value | why |
|---|---|---|
| `input.joints` | 1e-6 | the generator's own coordinates |
| `expected.postprocess.body_length` | 1e-6 | percentiles of those coordinates: pure geometry |
| `expected.postprocess.frames.joints` | 1e-3 | gap-filled and One-Euro-filtered positions |
| `expected.postprocess.frames.valid`, `.filled` | 0 | booleans are exact |
| `expected.phases.*` | 0 | a label is not a quantity |
| `expected.features` | 1e-3 | measured off filtered positions |
| `expected.hold_summary` | 1e-3 | medians and IQRs of the same |

A comparison walks a value's path (list indices such as `frames.12` are
dropped), takes the longest key that prefixes it, and compares within it —
except that booleans, strings and `null` are always compared exactly: a category
rounded is a different category.

## The cases

| case | what it is for |
|---|---|
| `line_hold` | the control: stand → kick-up → a ~4 s straight hold with a small CoM sway → exit → post |
| `banana_hold` | the same with an arched body: hips ahead of the shoulder–ankle line, `banana` and `line_deviation` past `BANANA_TOLERANCE_L` |
| `hand_step` | one wrist steps 0.15 L along the mat mid-hold: the phases stage ends the hold and starts a second one |
| `gaps_and_noise` | jitter everywhere, a 4-frame occlusion that gets bridged, a 7-frame one that does not, and one joint that teleports (gap fill, outlier removal, gating, One-Euro) |
| `trainer_contact` | 7 frames flagged `trainer_contact`: gated out of the processed trajectory and too long a gap to bridge |

## Reading one in Swift

`Tests/HandstandCoreTests/GoldenFixtureTests.swift` loads every file through
`Bundle.module` (the test target copies this whole folder as a resource) and
checks that it decodes and that the row arrays line up. The ports will decode
the same shape — `meta.joint_names` is the key that says which joint each
element of a `valid` array belongs to.
