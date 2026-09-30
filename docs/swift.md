# The Swift side

The workstation pipeline is Python, but the iOS app (chainlink #44) and anything
on-device has to run the same maths without Python around. So the pure pieces of
`pipeline/handstand/` have a Swift port, and the rule everything here hangs on is:

> **Swift mirrors Python behaviour.** Same formulas, same conventions, same joint
> names, same numbers on the same input.

Parity fixtures — golden inputs and outputs taken from the real clips — come in
chainlink #25 (the Python side) and #42 (the Swift tests that read them). Until
they land, the guarantee is that the Swift tests mirror the Python test cases one
for one, with the expectations hard-coded in both.

## Where things are

| path | what |
|---|---|
| `swift/HandstandCore/` | SwiftPM package: the port of the pipeline (this issue) |
| `swift/VisionPose/` | the Apple Vision keypoint runner (chainlink #15) — a separate package, untouched by this work |
| `ios/` | the app (chainlink #44) that will import `HandstandCore` |
| `tools/mac/swift_test.sh` | build and test a package on the Mac mini, from Linux |
| `docs/swift.md` | this note |

`HandstandCore` is one library product with two targets and **no third-party
dependencies**:

```
swift/HandstandCore/
├── Package.swift                       swift-tools-version 6.0; iOS 17, macOS 14
├── Sources/HandstandCore/
│   ├── Joint.swift                     the 15 tracked joints, raw values = Python names
│   ├── Keypoint.swift                  Keypoint, PoseFrame (+ midpoint / isValid, JSON Codable)
│   ├── Rotation.swift                  handstand.rotation
│   ├── PostProcess.swift               handstand.postprocess (#39)
│   ├── Phases.swift                    handstand.phases — the segmenter, phases only (#40)
│   ├── CentreOfMass.swift              handstand.com — the segment model and balance (#81)
│   ├── Features.swift                  handstand.features — per-frame features + hold rows (#81)
│   ├── BodyFrame.swift                 handstand.bodyframe
│   └── FramingCheck.swift              the Record screen's live framing guide (#46)
└── Tests/HandstandCoreTests/
    ├── JointTests.swift                raw values == handstand.postprocess.TRACKED_JOINTS
    ├── RotationTests.swift             mirrors pipeline/tests/test_rotation.py
    ├── PostProcessTests.swift          golden parity + the postprocess step tests (#39)
    ├── PhasesTests.swift               golden parity + the phase segmenter tests (#40)
    ├── CentreOfMassTests.swift         mirrors pipeline/tests/test_com.py (#81)
    ├── FeaturesTests.swift             golden parity + the feature tests (#81)
    ├── BodyFrameTests.swift            mirrors the body-frame cases in test_postprocess.py
    ├── FramingCheckTests.swift         whole body in frame / cut off / too small / alone (#46)
    ├── PoseFrameTests.swift            midpoint / isValid
    ├── FixtureTests.swift              reads the fixture through Bundle.module
    ├── GoldenFixtures.swift            the golden fixture schema, shared by the parity tests
    ├── GoldenFixtureTests.swift        the fixtures decode and their rows line up (#25)
    ├── Fixtures/tiny_frames.json       three hand-written PoseFrames
    └── Fixtures/golden/                the five synthetic parity cases + README.md
```

Conventions the sources keep, so the package builds for **both iOS and macOS**:

* every type is `Sendable` with value semantics (structs and enums only);
* no UIKit or AppKit imports — nothing here draws or views anything;
* the library target does no I/O: it is pure functions over numbers, which is
  what makes `swift test` a complete check of it;
* coordinates are display pixels with `y` down, as `docs/keypoint_schema.md`
  says; the body frame is the exception by design (`u` right, `v` **up**, both
  divided by the body length `L`).

### The pieces

* `Joint` — the 15 joints of the shared schema (`handstand.postprocess.TRACKED_JOINTS`:
  the nose, both shoulders, elbows, wrists, hips, knees, ankles and foot
  indexes). The raw values are the Python spellings from
  `handstand.pose_mediapipe.JOINT_NAMES`, so `"left_wrist"` is the same string
  on both sides.
* `Keypoint` / `PoseFrame` — one joint (`x`, `y`, `visibility`) and one frame
  (`frameIndex`, `tMs`, `joints`). `PoseFrame.midpoint(_:_:)` is the midpoint of
  two joints of that frame (`nil` when one is missing; `visibility` is the
  smaller of the two) — the wrist pair of which is the body frame's origin.
  `PoseFrame.isValid(minVisibility:)` is the strict gate: all 15 joints present
  and every one above the threshold.
  `PoseFrame` is Codable *by hand* as a JSON object keyed by joint name; a
  synthesized `[Joint: Keypoint]` conformance would decode as a JSON array of
  alternating keys and values, which no fixture would ever be written as.
* `Rotation` — `handstand.rotation.rotate_points` /
  `inverse_rotate_points` / `rotated_frame_size`. Angles are counter-clockwise
  in image coordinates, 90° and 270° swap the frame size, and 180° is the index
  flip `(width - 1 - x, height - 1 - y)`.
* `PostProcess` — `handstand.postprocess`: the five-step keypoint
  post-process, in the section below (chainlink #39).
* `PhaseSegmenter` — `handstand.phases`: one label (`pre`, `kickup`, `hold`,
  `exit`, `post`, `unknown`) and one hold number per frame, in the section
  below (chainlink #40).
* `CentreOfMass` — `handstand.com`: where the body's centre of mass sits over
  the hands, which way the athlete faces and which side of the base of support
  it is on, in the section below (chainlink #81).
* `Features` — `handstand.features`: the 14 per-frame features, the balance
  columns and the per-hold summary, in the section below (chainlink #81).
* `BodyFrame` — `handstand.bodyframe.to_body_frame` /
  `body_frame_points` / `midpoint`: origin at the wrist midpoint, `u` to the
  right, `v` up, divided by `L`. A `body_length` that is not a positive finite
  number throws, exactly where Python raises its `ValueError`.
* `FramingCheck` — the one piece here with no Python twin: the Record
  screen's live framing guide (chainlink #46). It takes the people one frame's
  detector saw, as joints in **normalised** coordinates (0…1, origin top-left,
  `y` down), and returns `noPerson` / `multiplePeople` / `tooSmall` /
  `partlyOutOfFrame(edges:missing:)` / `ok` plus the sentence the screen
  shows. The rules — a confident joint, the 0.04 edge margin, the 0.35
  minimum body height, the six-joint second-person threshold — are constants
  at the top of the file, and `FramingCheckTests` pins each verdict and each
  message down. `ios/HandstandApp/Capture/` is what feeds it.

Python's `numpy` broadcasting has no place to land in a typed, dependency-free
package, so batches are `[Keypoint]` in and out; where an error type replaces a
`ValueError`, the message text is kept.

## PostProcess (chainlink #39)

`Sources/HandstandCore/PostProcess.swift` mirrors
`pipeline/handstand/postprocess.py`: the five steps `process_clip` runs over a
clip, in the same order and with the same edge cases, over
`[PostProcessInputFrame]` → `ProcessedClip` (`joints`, `valid` and `filled` per
frame, plus the `BodyLength` sidecar). The entry point is
`PostProcess.process(_:config:)`; the helpers (`gatedValid`,
`estimateBodyLength`, `removeSpeedOutliers`, `fillGaps`, `smoothTrack`,
`OneEuroFilter`, `percentileOf`) are `internal`, so `@testable import` reaches
them the way the Python tests reach the module functions.

1. **Gating** — a frame with no pose or with `trainerContact` contributes
   nothing, and a joint whose coordinates are not finite, or whose visibility
   is below `minVisibility`, is not a position (a `NaN` visibility fails the
   comparison, as in Python).
2. **Body length** — `L` = torso + thigh + shin, each the percentile of its
   per-frame span over the frames where *that* span could be measured, the
   thigh and shin taken as the **longer** leg of the frame. Fewer than
   `minBodyFrames` measurable frames — or a total below
   `handstand.athlete.MIN_BODY_LENGTH_PIXELS` — makes the clip unusable, with
   Python's reason string in `BodyLength.reason`.
3. **Outliers** — a joint moving faster than `maxSpeedLPerS` body lengths per
   second between two consecutive valid samples is invalid in the *later* one;
   the earlier sample stays the reference.
4. **Gap fill** — a run of invalid samples is interpolated linearly **in time**
   when its two ends are at most `maxGapS` seconds apart, and marked
   `filled`; further apart, or at a track's head or tail, it stays invalid.
5. **Smoothing** — a One-Euro filter per joint coordinate, in body lengths,
   with the real `dt` from `t_ms`, restarted after every invalid run.

The constants, each the Python module constant of the same name:

| Python | value | Swift default |
|---|---|---|
| `MIN_VISIBILITY` | 0.5 | `PostProcessConfig.minVisibility` |
| `BODY_LENGTH_PERCENTILE` | 90.0 | `bodyLengthPercentile` |
| `MIN_BODY_FRAMES` | 10 | `minBodyFrames` |
| `MAX_SPEED_L_PER_S` | 8.0 | `maxSpeedLPerS` |
| `MAX_GAP_S` | 0.2 | `maxGapS` |
| `MIN_CUTOFF`, `BETA`, `D_CUTOFF` | 1.0, 0.3, 1.0 | `minCutoff`, `beta`, `dCutoff` |
| `MIN_SAMPLE_DT` | 1e-3 | `minSampleDt` |
| `athlete.MIN_BODY_LENGTH_PIXELS` | 1.0 | `PostProcessConstants.minBodyLengthPixels` |

Three rules the parity check is really about: the percentile interpolates the
way numpy's default `"linear"` method does (index `(n - 1) * p / 100`, lerped
between the two neighbours), every time comparison uses `t_ms` converted to
seconds (the clips are variable frame rate — nothing assumes a fixed fps), and
an unusable clip comes back with every joint invalid and no positions, exactly
as Python writes it. `hold_like_frames`, the `ClipStats` summary, the parquet /
JSON writing and the CLI are **not** ported.

`PostProcessTests.testEveryGoldenFixtureMatchesThePythonPostProcess` runs all
five golden fixtures of chainlink #25 through the port and compares them with
`expected.postprocess` at `meta.tolerances` (body length 1e-6 px, positions
1e-3 px, `valid`/`filled` exact); the other tests mirror the unit cases of
`pipeline/tests/test_postprocess.py` on tiny hand-made inputs.

## PhaseSegmenter (chainlink #40)

`Sources/HandstandCore/Phases.swift` mirrors the phase half of
`pipeline/handstand/phases.py`: **every** frame gets a label and the frames of
each hold get a number, because every stage after this one needs to know which
part of a clip it is looking at. The entry point is
`PhaseSegmenter.classify(tMs:processed:trainerContact:config:)` → `ClipPhases`
(`phase` and `holdId` per frame, plus the `FrameSignals` they were made from
and the `[HoldRun]s` it found). The input is exactly what Python's
`classify_clip` receives: the trajectory from `PostProcess.process`, the
**raw** trainer-contact flags (the post-process gates them away; a phase still
has to say *why* a frame is unknown) and `bodyLength.totalPx` as the scale.
The helpers (`midpoint`, `largest`, `bodyFrameUV`, `windowMotion`,
`velocityLPerS`, `frameSignals`, `holdCondition`, `holdRuns`, `assignPhases`)
are `internal` statics of `PhaseSegmenter`, so `@testable import` reaches them
the way the Python tests reach the module functions.

What it does, in the order the pipeline runs it:

1. **Signals** — every measurement taken in the body frame
   (`BodyFrame.toBodyFrame`, origin at the wrist midpoint, units of `L`, so
   one threshold means the same thing on a 350 px athlete and a 600 px one):
   `inverted`, `bodyAngleDeg`, `handsLow`, the wrist step and speed over
   `handStillWindowS`, `handsDown`, `handStep`, and the ankle midpoint's
   `legsVelocityLPerS` over `legVelocityWindowS`. A frame the module cannot
   see into — trainer contact, or a wrist, ankle or hip with no position — is
   `known == false`, carries the reason string Python writes, and every
   boolean of it is `false`.
2. **Hold runs** — `holdCondition` (inverted and straight, hands planted, no
   hand step) collapsed into runs: an unknown frame or a frame where the
   geometry merely failed bridges the run while the whole break is at most
   `holdBreakMaxS`, a **hand step ends the run wherever it happens**, and only
   runs at least `minHoldS` long count as holds — numbered from 0 in time
   order.
3. **State machine** — `assignPhases` labels every frame: `pre` becomes
   `kickup` when the hands go down and the legs rise, a qualified run is
   `hold`, a hold that stops holding is `exit`, and once the body is no longer
   above the hands every state but `pre` becomes `post`. Frames nobody could
   see are `unknown` and never move the state on; a gap *inside* a hold keeps
   the hold's phase and number.

The constants, each the Python module constant of the same name, on
`PhaseConfig` (`.init()` is the Python configuration):

| Python | value | Swift default |
|---|---|---|
| `NO_HOLD` | -1 | `PhaseConfig.noHold`, `PhaseSegmenter.noHold` |
| `INVERTED_MIN` | 0.6 | `PhaseConfig.invertedMin` |
| `HOLD_MAX_ANGLE` | 35.0 | `holdMaxAngle` |
| `MIN_HOLD_S` | 0.3 | `minHoldS` |
| `HAND_STILL_L_PER_S` | 0.3 | `handStillLPerS` |
| `HAND_STILL_WINDOW_S` | 0.2 | `handStillWindowS` |
| `HAND_STEP_L` | 0.1 | `handStepL` |
| `HANDS_LOW_MIN_V` | 0.1 | `handsLowMinV` |
| `LEG_VELOCITY_MIN_L_PER_S` | 0.1 | `legVelocityMinLPerS` |
| `LEG_VELOCITY_WINDOW_S` | 0.1 | `legVelocityWindowS` |
| `HOLD_BREAK_MAX_S` | 0.3 | `holdBreakMaxS` |
| `REASON_TRAINER`, `REASON_WRISTS`, `REASON_ANKLES`, `REASON_HIPS` | the four `unknown` reasons | `reasonTrainer`, `reasonWrists`, `reasonAnkles`, `reasonHips` |
| `WRIST_JOINTS`, `ANKLE_JOINTS`, `HIP_JOINTS` | the joints each signal is read off | `wristJoints`, `ankleJoints`, `hipJoints` |
| `_TIME_EPS` | 1e-9 | `timeEps` |

`PHASES` is not a config value: it *is* the declaration order of the `Phase`
enum — `pre`, `kickup`, `hold`, `exit`, `post`, `unknown`.

Three rules the parity check is really about: every window and duration is
measured against `tMs` converted to **seconds** (these clips are variable
frame rate — nothing assumes a fixed fps, and `np.searchsorted` finds each
window by timestamp), a `NaN` comparison is false *and* a `NaN` wrist speed is
not evidence that the hands moved (which is what lets a trainer's occlusion
pass through a hold instead of ending it), and a clip with no body length
comes back with every frame `unknown`, `hold_id` -1 and Python's
`"no body length (… px)"` reason — Swift's description of a `Double` agrees
with Python's `repr` on every value that path produces. The features, the
centre of mass and the hold summary are chainlink #81's, in the two sections
below; the parquet / CSV writing, the segments table, the `ClipStats` summary,
the overlay and the CLI are **not** ported.

`PhasesTests.testEveryGoldenFixtureMatchesThePythonPhases` runs all five
golden fixtures of chainlink #25 through `PostProcess.process` and the
segmenter and compares `phase` and `hold_id` with `expected.phases` **exactly**
(that tolerance is 0), plus `holdCount` with `meta.hold_count`; a mismatch
names the case, the frame, `t_ms`, both answers and that frame's signals. The
other tests mirror the unit cases of `pipeline/tests/test_phases.py` on tiny
hand-made trajectories written in the body frame (the wrist midpoint is the
origin, `LENGTH` = 300 px).

## CentreOfMass (chainlink #81)

`Sources/HandstandCore/CentreOfMass.swift` mirrors
`pipeline/handstand/com.py`: where the body's centre of mass is over the
hands, which way the athlete faces, and which side of the base of support it
sits on. The input is one clip's joints **already in the body frame** — the
`BodyTrack` that `Features.bodyFrameTrack` builds, origin at the wrist
midpoint, `u` right, `v` up, units of `L` — and the answer is in the same
units, so "the CoM is at 0.02 L" is a distance a judge could see.

1. **Segment model** — `centreOfMass(uv:joints:)` is the mass-weighted mean of
   Winter's twelve segments: head + neck and trunk on the midline, and the
   upper arm, forearm + hand, thigh, shank and foot on **both** sides. The
   helpers (`points`, `point`, `midpoint`, `sideMean`, `along`, `headPoints`,
   `segmentPoints`, `midlinePoints`) are `internal` statics, so `@testable
   import` reaches them the way the Python tests reach the module functions;
   a shape Python refuses with a `ValueError` throws the same message text.
2. **Which way is forward** — `facingSign(shoulderMid:hipMid:nose:)` reads the
   nose's side of the shoulder→hip torso line (`+1` towards `+u`; `NaN`
   without a nose, on the line, or with no line at all), and
   `majorityPerHold(_:holdId:)` decides it **per hold**: every frame of a hold
   takes the sign most of its frames voted for, frames outside a hold keep
   their own, and a tie — or no finite vote — keeps its frames as they were.
3. **Balance** — `comForward` is `com_u × facing_sign` (the CoM in anatomical
   rather than image coordinates), and `balanceZone` cuts it at the base of
   support: `under` behind the heel of the hand, `over` in front of the
   fingertips, `ok` in between, `nil` where there is no CoM.

The constants, each the Python module constant of the same name:

| Python | value | Swift |
|---|---|---|
| `MASS_HEAD_NECK`, `MASS_TRUNK` | 0.081, 0.497 | `CentreOfMass.massHeadNeck`, `.massTrunk` |
| `MASS_UPPER_ARM`, `MASS_FOREARM_HAND` | 0.028, 0.022 | `massUpperArm`, `massForearmHand` |
| `MASS_THIGH`, `MASS_SHANK`, `MASS_FOOT` | 0.100, 0.0465, 0.0145 | `massThigh`, `massShank`, `massFoot` |
| `COM_HEAD_NECK`, `COM_TRUNK` | 1.0, 0.5 | `comHeadNeck`, `comTrunk` |
| `COM_UPPER_ARM`, `COM_FOREARM_HAND` | 0.436, 0.682 | `comUpperArm`, `comForearmHand` |
| `COM_THIGH`, `COM_SHANK`, `COM_FOOT` | 0.433, 0.433, 0.5 | `comThigh`, `comShank`, `comFoot` |
| `HEAD_FALLBACK` | 0.5 | `headFallback` |
| `BASE_BACK`, `BASE_FRONT` | 0.03, 0.06 | `CentreOfMass.baseBack`, `.baseFront` (public) |
| `SEGMENT_SOURCE`, `ZONE_NAMES` | Winter (2009); `under`/`ok`/`over` | `segmentSource`, `zoneNames` |

Three rules the parity check is really about, all in the Python docstrings: a
segment missing on **one** side is measured on the other side's *coordinates*
(a side view overlaps the two sides, so nothing is mirrored) while one missing
on **both** is dropped and the rest renormalised with `complete` false; the
head's CoM **is** the nose, or 0.5 of the way from the shoulder midpoint
towards the spine extended past the shoulder when there is no nose — an
estimated head keeps the frame complete; and a frame with nothing left has no
CoM: `NaN`, not the mean of nothing.

`CentreOfMassTests` mirrors `pipeline/tests/test_com.py` case for case (13
tests, the same hand-written poses and the same pencil-checkable expectations).

## Features (chainlink #81)

`Sources/HandstandCore/Features.swift` mirrors the measurement half of
`pipeline/handstand/features.py`: every per-frame number the scorer will read,
and one row per hold. The entry points are
`Features.extract(tMs:processed:phases:trainerContact:)` → `ClipFeatures` and
`Features.holdRows(_:)` → `[HoldSummaryRow]`; the helpers (`bodyFrameTrack`,
`pixelMidpoint`, `bodyFeatures`, `groupMidpoint`, `sideMean`, `sideAngle`,
`bothSides`, `angleBetween`, `angleAt`, `rowMax`, `facingSign`, `banana`,
`head`, `distance`, `ratio`, `unusableFeatures`, `applyHoldFacing`, `spread`,
`holdStability`, `median`, `populationSD`, `roundedScalar`) are `internal`.

What it does, in the order the pipeline runs it:

1. **Body frame** — `bodyFrameTrack` is `BodyFrame.toBodyFrame` over a whole
   clip, one frame at a time, with the origin the wrist midpoint of *each*
   frame (one visible wrist is enough; none is a frame whose body frame was
   never placed, so every joint comes back `NaN` rather than a position
   measured against the image's own origin).
2. **Geometry** — `bodyFeatures` measures `Features.all`, in Python's order:
   the four stacking offsets (`off_shoulder`…`off_ankle`), `line_deviation`
   (the worst of them, `NaN` only when all four are unmeasured), `body_angle`,
   the four interior joint angles averaged over the sides, `banana` (signed
   towards the nose's side of the shoulder→ankle line), `head`, then the two
   features that *are* a measurement between the two sides and so need both:
   `leg_separation` and `hand_width`.
3. **Centre of mass and balance** — `CentreOfMass.centreOfMass` on the same
   track, the per-frame `facing_sign`, `com_forward` and `balance_zone`.
4. **Hold facing** — `applyHoldFacing` replaces the per-frame facing with the
   hold's majority vote and recomputes `com_forward` and `balance_zone`; the
   CoM itself is not touched.
5. **Hold summary** — `holdRows` gives every feature's median and IQR over the
   hold's *measurable* frames, the hold's frame counts, times and duration,
   and `holdStability`'s eight balance columns.

The constants, each the Python module constant of the same name; they are
statics rather than a config struct, and `Features.all` **is** `FEATURES`:

| Python | value | Swift |
|---|---|---|
| `MIN_SEGMENT_L` | 1e-9 | `Features.minSegmentL` |
| `LINE_TOLERANCE_L`, `BODY_ANGLE_TOL_DEG` | 0.1 L, 10° | `lineToleranceL`, `bodyAngleTolDeg` |
| `PIKE_HIP_DEG`, `OPEN_SHOULDER_DEG` | 165°, 160° | `pikeHipDeg`, `openShoulderDeg` |
| `STRAIGHT_KNEE_DEG`, `BENT_ELBOW_DEG` | 165°, 160° | `straightKneeDeg`, `bentElbowDeg` |
| `BANANA_TOLERANCE_L`, `HEAD_FLEXION_DEG`, `SPLIT_DEG` | 0.1 L, 60°, 30° | `bananaToleranceL`, `headFlexionDeg`, `splitDeg` |
| `_LENGTH_DIGITS`, `_ANGLE_DIGITS` | 5, 2 | `lengthDigits`, `angleDigits` |
| `FEATURES` | 14 specs (name, unit, target, digits) | `Features.all: [FeatureSpec]` |
| `COM_COLUMNS` | `com_u`, `com_v`, `facing_sign`, `com_forward` | `comColumns` (the other two are `ClipFeatures.comComplete` / `.balanceZone`) |
| `HOLD_STABILITY` | 8 balance columns + their digits | `balanceStats: [BalanceStat]`, `holdStabilityNames` |
| `HOLD_SUMMARY_COLUMNS` | the hold table's columns | `holdSummaryColumns` (minus `clip_id` and `source`, which Swift does not carry) |

Three rules the parity check is really about: a feature the frame could not
support is **`NaN`, never a number made of nothing**, and `valid` is what a
later stage filters on (a trainer in front of the camera makes the frame
invalid but leaves its numbers in the table); a left/right pair is the mean of
both where both are and the one that is where only one is, except
`leg_separation` and `hand_width`, which need both; and the hold summary
rounds like Python's `round(x, d)` — half to even on the binary value — with
`NaN` left as `NaN`. The tables (`feature_table`), the parquet / CSV writing,
the `ClipStats` summary, the TOLERANCES fault list, the plots and the CLI are
**not** ported.

`FeaturesTests.testEveryGoldenFixtureMatchesThePythonFeatures` runs all five
golden fixtures of chainlink #25 through `PostProcess.process`,
`PhaseSegmenter.classify` and `Features.extract`, and compares every column of
`expected.features` and every numeric key of `expected.hold_summary` at
`meta.tolerances` (1e-3), with `null` ⇔ `NaN`, `balance_zone` and the hold's
integer fields exact; a mismatch names the case, the frame or hold, the column
and both values. The other tests mirror the core of
`pipeline/tests/test_features.py` on poses written by hand in the body frame.

## Running the tests

Swift cannot build on Linux, so the tests run on the Mac mini. From the repo
root:

```bash
tools/mac/swift_test.sh                          # HandstandCore, into wt/i38
tools/mac/swift_test.sh HandstandCore wt/i38      # the same, spelled out
tools/mac/swift_test.sh VisionPose wt/i15         # the other package
```

What it does:

1. `rsync -a --delete` the repo's `swift/` directory to
   `macmini:<remote-dir>/swift/`, excluding `.build` and `.swiftpm` so the
   remote build cache survives (`--delete` keeps the remote an exact copy
   otherwise);
2. `ssh macmini 'cd <remote-dir>/swift/<package> && swift build && swift test'`;
3. print the test summary lines (`Executed … tests`, `Test Suite 'All tests' …`);
4. exit non-zero if the rsync, the build or the tests failed.

The Mac is `macmini` by default; override it with `HANDSTAND_MAC_HOST` if your
SSH config names it differently.

To check the package still builds for iOS (the app's platform, which macOS
tests do not exercise):

```bash
ssh macmini 'cd wt/i38/swift/HandstandCore && xcodebuild -scheme HandstandCore -destination "generic/platform=iOS Simulator" build'
```

Nothing on the Python side changes for a Swift port: `cd pipeline && uv run
pytest` stays green, and stays the source of truth.

## Adding the next port (chainlink #39–#42)

1. Read the Python first — `pipeline/handstand/postprocess.py`, `phases.py`,
   `features.py` — and copy the formula, the constants and the error
   behaviour, not a paraphrase of them.
2. One source file per Python module under `Sources/HandstandCore/`, types
   `Sendable`, no UI imports, no dependencies.
3. Mirror the Python test file under `Tests/HandstandCoreTests/`, hard-coding
   the same expectations, and run `tools/mac/swift_test.sh`.
4. When #25's fixtures exist, add them under `Tests/HandstandCoreTests/Fixtures/`
   and read them through `Bundle.module` (the directory is declared with
   `.copy("Fixtures")` in `Package.swift`), exactly like `tiny_frames.json`
   today.
