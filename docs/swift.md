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
│   ├── BodyFrame.swift                 handstand.bodyframe
│   └── FramingCheck.swift              the Record screen's live framing guide (#46)
└── Tests/HandstandCoreTests/
    ├── JointTests.swift                raw values == handstand.postprocess.TRACKED_JOINTS
    ├── RotationTests.swift             mirrors pipeline/tests/test_rotation.py
    ├── BodyFrameTests.swift            mirrors the body-frame cases in test_postprocess.py
    ├── FramingCheckTests.swift         whole body in frame / cut off / too small / alone (#46)
    ├── PoseFrameTests.swift            midpoint / isValid
    ├── FixtureTests.swift              reads the fixture through Bundle.module
    └── Fixtures/tiny_frames.json       three hand-written PoseFrames
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
