import XCTest

@testable import HandstandCore

/// The Swift mirror of the phase segmenter port (chainlink #40): the golden
/// fixtures of chainlink #25 run through `PostProcess.process` and
/// `PhaseSegmenter.classify`, compared with `expected.phases` **exactly**
/// (`meta.tolerances` is 0 for both columns), plus the unit cases of
/// `pipeline/tests/test_phases.py` on tiny hand-made trajectories — the same
/// cases, the same expectations.
///
/// Nothing here reads a real video. The fixtures are the synthetic cases
/// `pipeline/handstand/golden.py` writes; every other input is a handful of
/// numbers built in the test itself, written in the **body frame** the way the
/// Python tests write it: the wrist midpoint is the origin, the ankle midpoint
/// is `(uAnkle, vAnkle)` body lengths away, and `LENGTH` is a known pixel scale
/// so a test can say "0.3 L" and mean it.
final class PhasesTests: XCTestCase {
    // MARK: - A synthetic body, as `pipeline/tests/test_phases.py` builds it

    /// The clip's body length in pixels — `LENGTH` in Python. 300 keeps the
    /// coordinates in a comfortable range and makes 0.1 L exactly 30 px.
    private static let length = 300.0
    /// The frame interval of the synthetic clips, in milliseconds — `FRAME_MS`.
    private static let frameMs = 33

    /// One frame of a synthetic trajectory, written in the body frame — the
    /// same `Pose` the Python tests build. `uAnkle`/`vAnkle` and `vHip` are
    /// body lengths from the wrist midpoint with `v` positive upwards, so
    /// `vAnkle = 0.75` is a body standing on its hands and `vAnkle = -0.2` is
    /// a person standing on the floor; `wristX`/`wristY` are the wrist
    /// midpoint in *pixels*, which is how a hand step is written.
    private struct Pose {
        var uAnkle = 0.0
        var vAnkle = 0.0
        var vHip = 0.0
        var wristX = 200.0
        var wristY = 500.0
        /// How far apart the two wrists are, in body lengths. Zero is a
        /// side-on clip whose far wrist the model never reports.
        var handSpanL = 0.2
        /// An extra offset for the wrists, in pixels: a hand step is written
        /// by putting the hands down somewhere else.
        var wristOffsetPx = 0.0
        var valid = true
        var contact = false
    }

    /// A body standing on the floor: the hands are down by the hips, so the
    /// ankle midpoint is *below* the wrist midpoint and the body is not above
    /// its hands — `STANDING`.
    private static let standing = Pose(
        uAnkle: 0.0, vAnkle: -0.2, vHip: -0.3, wristX: 200.0, wristY: 400.0)
    /// A held handstand: the feet 0.75 body lengths above the hands, straight,
    /// the hands planted and still — `HOLDING`.
    private static let holding = Pose(
        uAnkle: 0.02, vAnkle: 0.75, vHip: 0.35, wristX: 200.0, wristY: 500.0)
    /// Hands on the floor with the body above them: the moment before a
    /// kick-up — `PLANTED`.
    private static let planted = Pose(
        uAnkle: 0.0, vAnkle: 0.2, vHip: 0.15, wristX: 200.0, wristY: 500.0)
    /// The top of a kick-up that never arrived: hands planted, legs swinging,
    /// feet never 0.6 body lengths above the hands — `FAILED`.
    private static let failed = Pose(
        uAnkle: 0.0, vAnkle: 0.45, vHip: 0.25, wristX: 200.0, wristY: 500.0)

    /// One clip as `classify` sees it: the processed trajectory, the clock and
    /// the **raw** trainer-contact flags.
    private struct SyntheticClip {
        var tMs: [Int]
        var clip: ProcessedClip
        var trainerContact: [Bool]
    }

    /// The pose list as the processed clip Python's `_frames` would write:
    /// only the joints a phase is read off carry a position (the rest are
    /// parked on the hip midpoint), a `valid == false` frame carries no
    /// positions at all, `tMs` is `frameMs` apart unless the test says
    /// otherwise, and `bodyLength` is the sidecar's scale — `NaN` is what the
    /// post-process hands back for a clip it could not measure one on.
    private func makeClip(
        _ poses: [Pose],
        tMs: [Int]? = nil,
        bodyLength: Double = PhasesTests.length
    ) -> SyntheticClip {
        let frames = poses.count
        let times = tMs ?? (0..<frames).map { $0 * PhasesTests.frameMs }
        precondition(times.count == frames, "t_ms must hold one timestamp per frame")
        let scale = PhasesTests.length

        var processed: [ProcessedFrame] = []
        processed.reserveCapacity(frames)
        for pose in poses {
            guard pose.valid else {
                processed.append(ProcessedFrame(joints: [:], valid: [], filled: []))
                continue
            }
            let centreX = pose.wristX
            let centreY = pose.wristY
            let half = pose.handSpanL * scale / 2.0
            let ankleMidX = centreX + pose.uAnkle * scale
            let ankleY = centreY - pose.vAnkle * scale
            let hipX = centreX
            let hipY = centreY - pose.vHip * scale
            var joints: [Joint: Point2] = [
                .leftWrist: Point2(x: centreX - half + pose.wristOffsetPx, y: centreY),
                .rightWrist: Point2(x: centreX + half + pose.wristOffsetPx, y: centreY),
                .leftAnkle: Point2(x: ankleMidX - 0.05 * scale, y: ankleY),
                .rightAnkle: Point2(x: ankleMidX + 0.05 * scale, y: ankleY),
                .leftHip: Point2(x: hipX - 0.1 * scale, y: hipY),
                .rightHip: Point2(x: hipX + 0.1 * scale, y: hipY),
                .leftShoulder: Point2(x: hipX - 0.12 * scale, y: hipY + 0.25 * scale),
                .rightShoulder: Point2(x: hipX + 0.12 * scale, y: hipY + 0.25 * scale),
                .leftKnee: Point2(x: hipX - 0.06 * scale, y: (hipY + ankleY) / 2.0),
                .rightKnee: Point2(x: hipX + 0.06 * scale, y: (hipY + ankleY) / 2.0),
            ]
            // The rest of the schema is parked on the hip midpoint, which is
            // all a phase detector looks at and enough for the frame to carry
            // a complete schema.
            for joint in Joint.allCases where joints[joint] == nil {
                joints[joint] = Point2(x: hipX, y: hipY)
            }
            processed.append(
                ProcessedFrame(joints: joints, valid: Set(Joint.allCases), filled: [])
            )
        }

        let body: BodyLength
        if bodyLength.isNaN {
            body = BodyLength(
                reason: "torso was measurable on 3 frame(s), need 10",
                torsoPx: nil, thighPx: nil, shinPx: nil,
                frames: ["torso": 3, "thigh": 3, "shin": 3]
            )
        } else {
            // The parts only have to add up to `bodyLength` exactly — the
            // phases never read them, and `totalPx` is the scale Python is
            // handed. A scale of 0.0 is a *usable* sidecar with a zero total,
            // which is the other way `classify_clip` refuses a clip.
            body = BodyLength(
                reason: nil, torsoPx: bodyLength, thighPx: 0, shinPx: 0, frames: [:]
            )
        }
        return SyntheticClip(
            tMs: times,
            clip: ProcessedClip(frames: processed, bodyLength: body),
            trainerContact: poses.map(\.contact)
        )
    }

    /// Classify a pose list straight, with no file in the way — Python's
    /// `classify(poses, **kwargs)`.
    private func classify(
        _ poses: [Pose],
        tMs: [Int]? = nil,
        bodyLength: Double = PhasesTests.length
    ) -> ClipPhases {
        let built = makeClip(poses, tMs: tMs, bodyLength: bodyLength)
        return PhaseSegmenter.classify(
            tMs: built.tMs, processed: built.clip, trainerContact: built.trainerContact
        )
    }

    /// `frames` of standing still on the floor — `stand`.
    private func stand(_ frames: Int) -> [Pose] {
        [Pose](repeating: PhasesTests.standing, count: frames)
    }

    /// `frames` of a held handstand, the hands planted and not moving —
    /// `hold`.
    private func hold(_ frames: Int, wristOffsetPx: Double = 0.0) -> [Pose] {
        (0..<frames).map { _ in
            var pose = PhasesTests.holding
            pose.wristOffsetPx = wristOffsetPx
            return pose
        }
    }

    /// The body from `start` to `end`, one frame at a time, linearly —
    /// `swing`.
    private func swing(_ start: Pose, _ end: Pose, _ frames: Int) -> [Pose] {
        precondition(frames >= 2, "a swing needs at least two frames")
        return (0..<frames).map { index in
            let weight = Double(index) / Double(frames - 1)
            var pose = start
            pose.uAnkle = start.uAnkle + weight * (end.uAnkle - start.uAnkle)
            pose.vAnkle = start.vAnkle + weight * (end.vAnkle - start.vAnkle)
            pose.vHip = start.vHip + weight * (end.vHip - start.vHip)
            pose.wristX = start.wristX + weight * (end.wristX - start.wristX)
            pose.wristY = start.wristY + weight * (end.wristY - start.wristY)
            return pose
        }
    }

    /// A whole attempt, in six moves — `attempt`: stand still, put the hands
    /// on the floor, swing the legs up around them, hold, come back down, rest
    /// on the hands for a moment, and take the hands off the floor.
    private func attempt(_ holdFrames: Int = 90) -> [Pose] {
        stand(10)
            + swing(PhasesTests.standing, PhasesTests.planted, 6)
            + swing(PhasesTests.planted, PhasesTests.holding, 12)
            + hold(holdFrames)
            + swing(PhasesTests.holding, PhasesTests.failed, 12)
            + [Pose](repeating: PhasesTests.failed, count: 8)
            + swing(PhasesTests.failed, PhasesTests.standing, 8)
    }

    /// `count` copies of one pose — Python's `[pose] * count`.
    private func repeating(_ pose: Pose, count: Int) -> [Pose] {
        [Pose](repeating: pose, count: count)
    }

    /// The phase of each frame as names, with runs collapsed for readability —
    /// Python's `phases_of`.
    private func phasesOf(_ result: ClipPhases) -> [String] {
        var out: [String] = []
        for (index, phase) in result.phase.enumerated() {
            if index == 0 || phase != result.phase[index - 1] {
                out.append(phase.rawValue)
            }
        }
        return out
    }

    /// The frame indices one phase was given — Python's `codes_in`.
    private func framesIn(_ result: ClipPhases, _ phase: Phase) -> [Int] {
        result.phase.indices.filter { result.phase[$0] == phase }
    }

    /// How many frames one phase got.
    private func frameCount(_ result: ClipPhases, of phase: Phase) -> Int {
        result.phase.lazy.filter { $0 == phase }.count
    }

    // MARK: - Parity against the golden fixtures of chainlink #25

    /// Every committed fixture, decoded, run through the Swift post-process
    /// and then `PhaseSegmenter.classify`, compared with `expected.phases`
    /// frame by frame at **zero** tolerance, and the hold count with
    /// `meta.hold_count` — by the shared `GoldenComparison.phases`, so a
    /// mismatch names the case, the frame, `t_ms`, both answers and that
    /// frame's signals. The comparison itself lives in
    /// `GoldenComparison.swift`, shared with `AnalyzerTests` and
    /// `RealParityTests` (chainlink #42).
    func testEveryGoldenFixtureMatchesThePythonPhases() throws {
        let urls = GoldenFixtures.urls()
        XCTAssertGreaterThanOrEqual(urls.count, 5, "the five committed cases")
        for url in urls {
            let fixture = try GoldenFixtures.load(url)
            let processed = PostProcess.process(GoldenFixtures.inputFrames(from: fixture))
            let result = PhaseSegmenter.classify(
                tMs: fixture.input.map(\.tMs),
                processed: processed,
                trainerContact: fixture.input.map(\.trainerContact)
            )
            for mismatch in GoldenComparison.phases(result, fixture: fixture) {
                XCTFail(mismatch)
            }
        }
    }

    // MARK: - The constants

    /// Every constant of `pipeline/handstand/phases.py` at its Python value.
    func testTheConstantsAreThePythonOnes() {
        let config = PhaseConfig()
        XCTAssertEqual(config.noHold, -1)  // NO_HOLD
        XCTAssertEqual(config.invertedMin, 0.6)  // INVERTED_MIN
        XCTAssertEqual(config.holdMaxAngle, 35.0)  // HOLD_MAX_ANGLE
        XCTAssertEqual(config.minHoldS, 0.3)  // MIN_HOLD_S
        XCTAssertEqual(config.handStillLPerS, 0.3)  // HAND_STILL_L_PER_S
        XCTAssertEqual(config.handStillWindowS, 0.2)  // HAND_STILL_WINDOW_S
        XCTAssertEqual(config.handStepL, 0.1)  // HAND_STEP_L
        XCTAssertEqual(config.handsLowMinV, 0.1)  // HANDS_LOW_MIN_V
        XCTAssertEqual(config.legVelocityMinLPerS, 0.1)  // LEG_VELOCITY_MIN_L_PER_S
        XCTAssertEqual(config.legVelocityWindowS, 0.1)  // LEG_VELOCITY_WINDOW_S
        XCTAssertEqual(config.holdBreakMaxS, 0.3)  // HOLD_BREAK_MAX_S
        XCTAssertEqual(config.reasonTrainer, "trainer_contact")  // REASON_TRAINER
        XCTAssertEqual(config.reasonWrists, "no_visible_wrist")  // REASON_WRISTS
        XCTAssertEqual(config.reasonAnkles, "no_visible_ankle")  // REASON_ANKLES
        XCTAssertEqual(config.reasonHips, "no_visible_hip")  // REASON_HIPS
        XCTAssertEqual(config.wristJoints, [.leftWrist, .rightWrist])  // WRIST_JOINTS
        XCTAssertEqual(config.ankleJoints, [.leftAnkle, .rightAnkle])  // ANKLE_JOINTS
        XCTAssertEqual(config.hipJoints, [.leftHip, .rightHip])  // HIP_JOINTS
        XCTAssertEqual(config.timeEps, 1e-9)  // _TIME_EPS

        XCTAssertEqual(PhaseSegmenter.noHold, -1)  // NO_HOLD
        XCTAssertEqual(
            Phase.allCases.map(\.rawValue),
            ["pre", "kickup", "hold", "exit", "post", "unknown"])  // PHASES
    }

    // MARK: - The signals

    func testSignalsReadAHeldHandstand() {
        let result = classify(attempt())
        let signals = result.signals
        let holds = framesIn(result, .hold)
        XCTAssertFalse(holds.isEmpty)
        let middle = holds[holds.count / 2]

        XCTAssertTrue(signals.known.allSatisfy { $0 })
        XCTAssertTrue(signals.inverted[middle])
        XCTAssertEqual(signals.vAnkleMid[middle], 0.75, accuracy: 0.01)
        XCTAssertEqual(signals.vHipMid[middle], 0.35, accuracy: 0.01)
        // Straight up: the angle is the wrist->ankle vector's lean from
        // vertical, and this body leans a little over a degree.
        let expectedAngle = atan2(0.02, 0.75) * (180.0 / Double.pi)
        XCTAssertEqual(signals.bodyAngleDeg[middle], expectedAngle, accuracy: 0.2)
        XCTAssertTrue(signals.handsDown[middle])
        XCTAssertEqual(signals.wristSpeedLPerS[middle], 0.0, accuracy: 1e-9)
        XCTAssertEqual(signals.wristStepL[middle], 0.0, accuracy: 1e-9)
        XCTAssertFalse(signals.handStep[middle])
        XCTAssertLessThan(abs(signals.legsVelocityLPerS[middle]), 0.1)
        XCTAssertFalse(signals.legsRising[middle])
        XCTAssertFalse(signals.legsFalling[middle])
    }

    func testABodyOnTheFloorIsNotInverted() {
        let signals = classify(stand(10)).signals

        XCTAssertFalse(signals.inverted.contains(true))
        XCTAssertFalse(signals.handsDown.contains(true))
        XCTAssertEqual(signals.vAnkleMid[0], -0.2, accuracy: 0.01)
        // The angle of a vector pointing *down* is more than 90 degrees from up.
        XCTAssertGreaterThan(abs(signals.bodyAngleDeg[0]), 90.0)
    }

    func testHandsLowNeedsTheBodyAboveTheHands() {
        // The feet above the hands but the hips not: the athlete is on their
        // knees with their hands on the floor, which is a kick-up, not a
        // stand-up.
        var kneeling = PhasesTests.holding
        kneeling.vAnkle = 0.5
        kneeling.vHip = 0.02
        XCTAssertFalse(classify(repeating(kneeling, count: 10)).signals.handsLow[0])

        // The hips above the hands and the feet below: an L-shape, hands planted.
        var lying = PhasesTests.holding
        lying.vAnkle = -0.1
        lying.vHip = 0.2
        XCTAssertTrue(classify(repeating(lying, count: 10)).signals.handsLow[0])
    }

    func testTheLegsRisingAndFallingWithTheFeet() {
        let up = swing(PhasesTests.planted, PhasesTests.holding, 20)
        let rising = classify(up).signals
        XCTAssertTrue(rising.legsRising[19])
        XCTAssertGreaterThan(rising.legsVelocityLPerS[19], 0.1)

        // The same body coming down: the same speed, the other sign.
        let falling = classify(Array(up.reversed())).signals
        XCTAssertTrue(falling.legsFalling[19])
        XCTAssertLessThan(falling.legsVelocityLPerS[19], -0.1)
    }

    func testAMovingWristIsNotAPlantedHand() {
        // The whole body slides sideways at 2 L/s: the hands are travelling,
        // not planted, whatever the geometry says.
        let drifting = (0..<15).map { index -> Pose in
            var pose = PhasesTests.holding
            pose.wristX = 200.0 + Double(index) * 2.0 * PhasesTests.length / 30.0
            return pose
        }
        let signals = classify(drifting).signals
        XCTAssertEqual(signals.wristSpeedLPerS[14], 2.0, accuracy: 0.1)
        // The window is the last 0.2 s, which is seven frames of 33 ms, so the
        // step is a little over 0.4 L: 0.5 L/s for a fifth of a second, plus a
        // frame.
        XCTAssertEqual(signals.wristStepL[14], 2.0 * 0.231, accuracy: 0.05 * 2.0 * 0.231)
        XCTAssertTrue(signals.handStep[14])
        XCTAssertFalse(signals.handsDown[14])
    }

    func testASpeedThatCannotBeMeasuredIsNotAMovingHand() {
        // The first frames of a clip, and the frames after a gap, have no
        // window. NaN is not evidence of movement, so the hold condition can
        // still pass.
        let signals = classify(hold(10)).signals
        XCTAssertTrue(signals.wristSpeedLPerS[0].isNaN)
        XCTAssertTrue(signals.handsDown[0])
    }

    func testOneVisibleWristIsEnough() {
        // A side-on clip: the two wrists report the same point, which is the
        // degenerate form of "the far wrist was never seen".
        let poses = (0..<30).map { _ -> Pose in
            var pose = PhasesTests.holding
            pose.handSpanL = 0.0
            return pose
        }
        let result = classify(poses)
        XCTAssertTrue(result.signals.known.allSatisfy { $0 })
        XCTAssertEqual(result.holdCount, 1)
    }

    // MARK: - window_motion and velocity

    func testWindowMotionMeasuresTheWindowBeforeEachFrame() {
        let times = [0.0, 0.1, 0.25, 0.4]
        let x = [0.0, 0.5, 1.0, 1.5]
        let y = [0.0, 0.0, 0.0, 0.0]
        let known = [Bool](repeating: true, count: 4)

        let motion = PhaseSegmenter.windowMotion(
            tSeconds: times, x: x, y: y, known: known, bodyLength: 2.0, windowS: 0.2)

        // Frame 2 is 0.25 s in; the last frame at or before 0.05 s is frame 0,
        // so the displacement is 1.0 px over 0.25 s.
        XCTAssertEqual(motion.stepL[2], 0.5, accuracy: 1e-12)
        XCTAssertEqual(motion.speedLPerS[2], 2.0, accuracy: 1e-12)
        // Frames 0 and 1 ask about a window that reaches back before the clip
        // starts.
        XCTAssertTrue(motion.stepL[0].isNaN)
        XCTAssertTrue(motion.stepL[1].isNaN)
        // Frame 3 is 0.4 s in; the last frame at or before 0.2 s is frame 1,
        // so the window is 0.3 s long and holds the whole second pixel.
        XCTAssertEqual(motion.stepL[3], 0.5, accuracy: 1e-12)
        XCTAssertEqual(motion.speedLPerS[3], 0.5 / 0.3, accuracy: 1e-12)
    }

    func testVelocityIsCentredAndNeedsAKnownSampleOnEachSide() {
        let times = [0.0, 0.05, 0.1, 0.15, 0.2]
        let values = [0.0, 0.0, 1.0, 1.0, 1.0]
        let known = [Bool](repeating: true, count: 5)

        let velocity = PhaseSegmenter.velocityLPerS(
            tSeconds: times, values: values, known: known, spanS: 0.1)
        XCTAssertEqual(velocity[2], 10.0, accuracy: 1e-9)
        // The two ends of the clip are measured against the one side they
        // have: the value is flat either side of frame 0 and of frame 4, so
        // neither moves.
        XCTAssertEqual(velocity[0], 0.0, accuracy: 1e-9)
        XCTAssertEqual(velocity[4], 0.0, accuracy: 1e-9)

        // A frame the module cannot see into has no velocity, and neither has
        // the frame compared against one.
        var blind = known
        blind[2] = false
        blind[3] = false
        let afterBlind = PhaseSegmenter.velocityLPerS(
            tSeconds: times, values: values, known: blind, spanS: 0.1)
        XCTAssertTrue(afterBlind[2].isNaN)
        XCTAssertTrue(afterBlind[3].isNaN)
    }

    func testUnevenTmSpacingIsMeasuredInSeconds() {
        // Twelve frames that are not a fixed step apart: every window is found
        // by timestamp, and the hold's duration comes from `tMs` and not from
        // a frame count.
        let times = [0, 30, 70, 100, 140, 170, 210, 250, 290, 320, 360, 400]
        let result = classify(hold(12), tMs: times)

        XCTAssertEqual(result.holdCount, 1)
        XCTAssertEqual(result.holdDurationsS().count, 1)
        XCTAssertEqual(result.holdDurationsS()[0], 0.4, accuracy: 1e-9)
        XCTAssertEqual(phasesOf(result), ["hold"])
    }

    // MARK: - Holds

    func testARunShorterThanTheMinimumIsNotAHold() {
        let times = (0..<13).map { Double($0) * 0.05 }
        // Two runs of 0.1 s and 0.05 s, separated by a 0.35 s break: long
        // enough that the runs stay separate, short enough that a hold can sit
        // inside it.
        var holding = [Bool](repeating: false, count: 13)
        for index in 0..<3 { holding[index] = true }
        for index in 11..<13 { holding[index] = true }
        let known = [Bool](repeating: true, count: 13)

        XCTAssertEqual(
            PhaseSegmenter.holdRuns(tSeconds: times, holding: holding, known: known), [])
        // ... and with a minimum of a twentieth of a second both count.
        let kept = PhaseSegmenter.holdRuns(
            tSeconds: times, holding: holding, known: known, minHoldS: 0.05)
        XCTAssertEqual(
            kept,
            [HoldRun(start: 0, end: 2, holdId: 0), HoldRun(start: 11, end: 12, holdId: 1)])
    }

    func testAShortBreakDoesNotEndAHoldAndALongOneDoes() {
        let times = (0..<40).map { Double($0) * 0.05 }  // 2 s
        let known = [Bool](repeating: true, count: 40)

        // Frames 10-14 fail for 0.2 s, well inside HOLD_BREAK_MAX_S.
        var holding = [Bool](repeating: true, count: 40)
        for index in 10..<15 { holding[index] = false }
        let bridged = PhaseSegmenter.holdRuns(tSeconds: times, holding: holding, known: known)
        XCTAssertEqual(bridged.count, 1)
        XCTAssertEqual(bridged[0].start, 0)
        XCTAssertEqual(bridged[0].end, 39)
        XCTAssertEqual(bridged[0].holdId, 0)

        // Frames 20-27 fail for 0.4 s: past the window, so the hold ends there.
        holding = [Bool](repeating: true, count: 40)
        for index in 10..<15 { holding[index] = false }
        for index in 20..<28 { holding[index] = false }
        let runs = PhaseSegmenter.holdRuns(tSeconds: times, holding: holding, known: known)
        XCTAssertEqual(
            runs,
            [HoldRun(start: 0, end: 19, holdId: 0), HoldRun(start: 28, end: 39, holdId: 1)])
    }

    func testAnEventEndsAHoldWhereverItHappens() {
        let times = (0..<40).map { Double($0) * 0.05 }
        let known = [Bool](repeating: true, count: 40)
        var holding = [Bool](repeating: true, count: 40)
        holding[12] = false
        holding[13] = false
        var event = [Bool](repeating: false, count: 40)
        event[12] = true

        // Two failures in a row, well inside HOLD_BREAK_MAX_S, and a hand step
        // on the first of them: the run ends there rather than being bridged.
        let runs = PhaseSegmenter.holdRuns(
            tSeconds: times, holding: holding, known: known, event: event)
        XCTAssertEqual(
            runs,
            [HoldRun(start: 0, end: 11, holdId: 0), HoldRun(start: 14, end: 39, holdId: 1)])
    }

    func testAGapAtTheVeryEndOfAClipIsNotPartOfTheHold() {
        let times = (0..<20).map { Double($0) * 0.05 }
        let known = [Bool](repeating: true, count: 20)
        var holding = [Bool](repeating: false, count: 20)
        for index in 0..<12 { holding[index] = true }

        let runs = PhaseSegmenter.holdRuns(
            tSeconds: times, holding: holding, known: known, minHoldS: 0.1)
        XCTAssertEqual(runs, [HoldRun(start: 0, end: 11, holdId: 0)])
    }

    // MARK: - The phase sequences

    func testStandKickUpHoldExitIsTheWholeSequence() {
        // The trajectory the issue is named after: one hold, about three
        // seconds long.
        let result = classify(attempt())

        XCTAssertEqual(phasesOf(result), ["pre", "kickup", "hold", "exit", "post"])
        XCTAssertEqual(result.holdCount, 1)
        XCTAssertEqual(result.holdDurationsS().count, 1)
        XCTAssertEqual(result.holdDurationsS()[0], 3.23, accuracy: 0.1)

        // Every frame of the hold, and only those frames, has a number.
        let holds = framesIn(result, .hold)
        XCTAssertEqual(holds, Array(25..<124))
        XCTAssertTrue(holds.allSatisfy { result.holdId[$0] == 0 })
        let holdSet = Set(holds)
        for index in 0..<result.phase.count where !holdSet.contains(index) {
            XCTAssertEqual(result.holdId[index], PhaseSegmenter.noHold)
        }

        // The boundaries: the hold starts where the feet clear INVERTED_MIN,
        // and ends where they stop clearing it on the way down.
        XCTAssertEqual(frameCount(result, of: .pre), 21)
        XCTAssertEqual(frameCount(result, of: .kickup), 4)
        XCTAssertEqual(frameCount(result, of: .hold), 99)
        XCTAssertEqual(frameCount(result, of: .exit), 16)
        XCTAssertEqual(frameCount(result, of: .post), 6)
        XCTAssertEqual(frameCount(result, of: .unknown), 0)
    }

    func testAHandStepInTheMiddleOfAHoldMakesTwoHolds() {
        // The hands are put down 0.4 L to the right over four frames, and the
        // body stays over them where they landed.
        var poses = hold(40)
        for offset in [0.1, 0.2, 0.3, 0.4] {
            var pose = PhasesTests.holding
            pose.wristOffsetPx = offset * PhasesTests.length
            poses.append(pose)
        }
        poses += hold(40, wristOffsetPx: 0.4 * PhasesTests.length)
        let result = classify(poses)

        XCTAssertEqual(result.holdCount, 2)
        let durations = result.holdDurationsS()
        XCTAssertEqual(durations.count, 2)
        XCTAssertEqual(durations[0], 1.32, accuracy: 0.005)
        XCTAssertEqual(durations[1], 1.09, accuracy: 0.005)
        XCTAssertEqual(phasesOf(result), ["hold", "exit", "hold"])
        XCTAssertEqual(result.holdId[40], 0)
        XCTAssertEqual(result.holdId[50], 1)
        // The step is reported on the frames the wrists are moving, and the
        // frames after it are the exit the rules give: a hold that stops
        // holding is an exit, because the hands are still the support.
        for index in 41..<45 {
            XCTAssertTrue(result.signals.handStep[index], "frame \(index) hand_step")
        }
        XCTAssertFalse(result.signals.handStep[50])
        XCTAssertTrue((41...50).contains { result.phase[$0] == .exit })
    }

    func testAShortUnknownGapInsideAHoldDoesNotSplitIt() {
        var invisible = PhasesTests.holding
        invisible.valid = false
        let poses = hold(20) + [Pose](repeating: invisible, count: 4) + hold(20)
        let result = classify(poses)

        XCTAssertEqual(result.holdCount, 1)
        XCTAssertEqual(result.holdDurationsS()[0], 1.42, accuracy: 0.05)
        // The gap is inside the hold, and the hold's number is on it.
        for index in 20..<24 {
            XCTAssertEqual(result.phase[index], .hold, "frame \(index)")
            XCTAssertEqual(result.holdId[index], 0)
            XCTAssertFalse(result.signals.unknownReason[index].isEmpty)
        }
        // ... and the gap is under HOLD_BREAK_MAX_S, which is why it did not
        // split it.
        XCTAssertLessThanOrEqual(
            Double(4 * PhasesTests.frameMs) / 1000.0, PhaseConfig().holdBreakMaxS)
    }

    func testALongUnknownGapDoesSplitAHold() {
        var invisible = PhasesTests.holding
        invisible.valid = false
        let poses = hold(20) + [Pose](repeating: invisible, count: 12) + hold(20)
        let result = classify(poses)

        XCTAssertEqual(result.holdCount, 2)
        let durations = result.holdDurationsS()
        XCTAssertEqual(durations.count, 2)
        XCTAssertEqual(durations[0], 0.63, accuracy: 0.005)
        XCTAssertEqual(durations[1], 0.63, accuracy: 0.005)
        // The gap is where the module could not look, which is an answer.
        XCTAssertEqual(result.phase[20], .unknown)
        XCTAssertEqual(result.phase[31], .unknown)
        XCTAssertEqual(result.phase[32], .hold)
        XCTAssertGreaterThan(
            Double(12 * PhasesTests.frameMs) / 1000.0, PhaseConfig().holdBreakMaxS)
    }

    func testATrainerContactFrameIsUnknownWithThatReason() {
        var invisible = PhasesTests.holding
        invisible.valid = false
        let poses = hold(4) + [invisible] + hold(4)
        let result = classify(poses)
        let config = PhaseConfig()

        XCTAssertFalse(result.signals.known[4])
        XCTAssertEqual(result.signals.unknownReason[4], config.reasonWrists)
        XCTAssertEqual(result.phase[4], .unknown)

        // ... and a frame the athlete selection flagged is unknown whatever
        // its keypoints say.
        var flagged = PhasesTests.holding
        flagged.contact = true
        let contactPoses = hold(4) + [flagged] + hold(4)
        let flaggedResult = classify(contactPoses)
        XCTAssertEqual(flaggedResult.signals.unknownReason[4], config.reasonTrainer)
        XCTAssertEqual(flaggedResult.phase[4], .unknown)
    }

    func testAClipWithNoBodyLengthIsAllUnknown() {
        let result = classify(hold(30), bodyLength: .nan)

        XCTAssertFalse(result.usable)
        // Python's `f"no body length ({body_length!r} px)"` with `nan`.
        XCTAssertEqual(result.unusableReason, "no body length (nan px)")
        XCTAssertTrue(result.phase.allSatisfy { $0 == .unknown })
        XCTAssertEqual(phasesOf(result), ["unknown"])
        XCTAssertEqual(result.holdCount, 0)
        XCTAssertEqual(result.holdId, [Int](repeating: PhaseSegmenter.noHold, count: 30))
        XCTAssertFalse(result.signals.known.contains(true))
        XCTAssertTrue(result.signals.unknownReason.allSatisfy {
            $0 == "no body length (nan px)"
        })
        XCTAssertTrue(result.signals.vAnkleMid.allSatisfy { $0.isNaN })
        XCTAssertTrue(result.signals.handsDown.allSatisfy { !$0 })
    }

    func testAnUnusableBodyLengthIsWrittenWithPythonsRepr() {
        // The other refusal path: a usable sidecar whose total is zero. Python
        // spells the float with `repr`, and Swift's description of a Double
        // agrees on every value this path produces.
        let result = classify(hold(4), bodyLength: 0.0)

        XCTAssertFalse(result.usable)
        XCTAssertEqual(result.unusableReason, "no body length (0.0 px)")
        XCTAssertTrue(result.phase.allSatisfy { $0 == .unknown })
        XCTAssertEqual(result.holdId, [Int](repeating: -1, count: 4))
    }

    func testAFailedKickUpNeverInvertedYieldsNoHold() {
        // The athlete plants their hands, tries, and comes back down.
        let poses = stand(10) + swing(PhasesTests.standing, PhasesTests.planted, 6)
            + swing(PhasesTests.planted, PhasesTests.failed, 12) + stand(10)
        let result = classify(poses)

        XCTAssertEqual(result.holdCount, 0)
        XCTAssertFalse(result.signals.inverted.contains(true))
        XCTAssertFalse(phasesOf(result).contains("hold"))
        // The hands are planted and still for the whole of the attempt, and
        // the feet never clear INVERTED_MIN: that is a kick-up that did not
        // arrive.
        XCTAssertGreaterThanOrEqual(
            result.signals.handsDown.filter { $0 }.count, 5)
        XCTAssertFalse(framesIn(result, .kickup).isEmpty)
        XCTAssertEqual(phasesOf(result), ["pre", "kickup", "post"])
    }

    func testASecondAttemptAfterAFallIsASecondHold() {
        let first = attempt(30)
        let result = classify(first + first)

        XCTAssertEqual(result.holdCount, 2)
        let durations = result.holdDurationsS()
        XCTAssertEqual(durations.count, 2)
        XCTAssertEqual(durations[0], 1.25, accuracy: 0.005)
        XCTAssertEqual(durations[1], 1.25, accuracy: 0.005)
        XCTAssertEqual(
            phasesOf(result),
            ["pre", "kickup", "hold", "exit", "post", "kickup", "hold", "exit", "post"])
    }

    func testAClipThatStartsInAHoldHasNoPre() {
        let result = classify(hold(30))

        XCTAssertEqual(phasesOf(result), ["hold"])
        XCTAssertEqual(result.holdCount, 1)
        XCTAssertEqual(result.holdDurationsS()[0], 0.9, accuracy: 0.1)
    }
}
