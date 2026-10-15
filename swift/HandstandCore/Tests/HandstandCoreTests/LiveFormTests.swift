import XCTest

@testable import HandstandCore

/// What the live form detector says, frame by frame (chainlink #97): the red
/// hysteresis, the hold gate, the once-per-attempt rule and its re-arm, the
/// spacing between phrases — and that a live banana is the offline `banana`
/// measured in the live body length, not a second formula.
///
/// Everything here is synthetic (a hand-placed body in display pixels), so no
/// camera, video or model is involved.
final class LiveFormTests: XCTestCase {
    // MARK: - Synthetic frames

    /// A hold-like body, built from its own numbers: wrists on the floor at
    /// the bottom, ankles straight above them, and the hip midpoint slid
    /// sideways by `hipOffsetX` — with the nose on the right, a positive
    /// offset is the belly side (an arch) and a negative one the back (a
    /// hollow). `nose` off drops the nose so the sign is unknown.
    private enum Synthetic {
        static func hold(
            hipOffsetX: Double = 0,
            nose: Bool = true,
            noseX: Double = 575,
            visibility: Double = 0.9
        ) -> [Joint: Keypoint] {
            var joints: [Joint: Keypoint] = [
                .leftWrist: Keypoint(x: 500, y: 1850, visibility: visibility),
                .rightWrist: Keypoint(x: 580, y: 1850, visibility: visibility),
                .leftShoulder: Keypoint(x: 510, y: 1550, visibility: visibility),
                .rightShoulder: Keypoint(x: 570, y: 1550, visibility: visibility),
                .leftHip: Keypoint(x: 520 + hipOffsetX, y: 1000, visibility: visibility),
                .rightHip: Keypoint(x: 560 + hipOffsetX, y: 1000, visibility: visibility),
                .leftKnee: Keypoint(x: 525 + hipOffsetX, y: 725, visibility: visibility),
                .rightKnee: Keypoint(x: 555 + hipOffsetX, y: 725, visibility: visibility),
                .leftAnkle: Keypoint(x: 520, y: 450, visibility: visibility),
                .rightAnkle: Keypoint(x: 560, y: 450, visibility: visibility),
                .leftFootIndex: Keypoint(x: 525, y: 380, visibility: visibility),
                .rightFootIndex: Keypoint(x: 555, y: 380, visibility: visibility),
            ]
            if nose {
                joints[.nose] = Keypoint(x: noseX, y: 1400, visibility: visibility)
            }
            return joints
        }

        /// A red arch: the hip slid towards the nose's side.
        static func arch(visibility: Double = 0.9) -> [Joint: Keypoint] {
            hold(hipOffsetX: 200, visibility: visibility)
        }

        /// A red hollow: the hip slid away from the nose's side.
        static func hollow(visibility: Double = 0.9) -> [Joint: Keypoint] {
            hold(hipOffsetX: -200, visibility: visibility)
        }

        /// A red arch with no nose: the sign is unknown.
        static func bent(visibility: Double = 0.9) -> [Joint: Keypoint] {
            hold(hipOffsetX: 200, nose: false, visibility: visibility)
        }

        /// The same body standing on its feet: nothing is inverted, so the
        /// gate never opens however red the banana would be.
        static func standingArch(visibility: Double = 0.9) -> [Joint: Keypoint] {
            [
                .leftWrist: Keypoint(x: 500, y: 450, visibility: visibility),
                .rightWrist: Keypoint(x: 580, y: 450, visibility: visibility),
                .leftShoulder: Keypoint(x: 510, y: 750, visibility: visibility),
                .rightShoulder: Keypoint(x: 570, y: 750, visibility: visibility),
                .leftHip: Keypoint(x: 720, y: 1050, visibility: visibility),
                .rightHip: Keypoint(x: 760, y: 1050, visibility: visibility),
                .leftKnee: Keypoint(x: 725, y: 1400, visibility: visibility),
                .rightKnee: Keypoint(x: 755, y: 1400, visibility: visibility),
                .leftAnkle: Keypoint(x: 520, y: 1850, visibility: visibility),
                .rightAnkle: Keypoint(x: 560, y: 1850, visibility: visibility),
                .leftFootIndex: Keypoint(x: 525, y: 1920, visibility: visibility),
                .rightFootIndex: Keypoint(x: 555, y: 1920, visibility: visibility),
                .nose: Keypoint(x: 575, y: 800, visibility: visibility),
            ]
        }
    }

    /// Feed one stride of frames to the detector and collect its events.
    private func feed(
        _ detector: inout LiveFormDetector,
        from first: Int,
        through last: Int,
        by step: Int,
        joints: (Int) -> [Joint: Keypoint]
    ) -> [LiveFormEvent] {
        var events = [LiveFormEvent]()
        var tMs = first
        while tMs <= last {
            events += detector.update(tMs: tMs, joints: joints(tMs))
            tMs += step
        }
        return events
    }

    // MARK: - The config

    func testTheConfigDefaults() {
        let config = LiveFormConfig()
        XCTAssertEqual(config.redHoldS, 1.0)
        XCTAssertEqual(config.gapS, 0.3)
        XCTAssertEqual(config.rearmS, 2.0)
        XCTAssertEqual(config.minSpacingS, 5.0)
        XCTAssertEqual(config.gateS, 0.5)
        XCTAssertEqual(config.gateMaxAngleDeg, 35.0)
        XCTAssertEqual(config.minVisibility, 0.5)
    }

    // MARK: - The red hysteresis

    func testACleanLineSaysNothing() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 3000, by: 100) { _ in
            Synthetic.hold()
        }
        XCTAssertEqual(events, [], "a straight line has no fault to correct")
        XCTAssertTrue(detector.isInAttempt, "the gate was open the whole time")
    }

    func testARedArchHeldLongEnoughSaysItOnce() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 1200, by: 100) { _ in Synthetic.arch() }

        XCTAssertEqual(events, [.arch(tMs: 1000)], "1.2 s of red earns one phrase, at 1.0 s")
    }

    func testARedArchHeldTooBrieflySaysNothing() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 800, by: 100) { _ in Synthetic.arch() }

        XCTAssertEqual(events, [], "0.8 s of red is below the 1.0 s hold")
    }

    func testAShortClearGapDoesNotResetTheRed() {
        var detector = LiveFormDetector()
        // 0.7 s red, then a 0.2 s clear blink, then red again: the blink is
        // inside gapS, so the red run still began at 0 and fires at 1.0 s.
        var events = feed(&detector, from: 0, through: 700, by: 100) { _ in Synthetic.arch() }
        events += feed(&detector, from: 800, through: 900, by: 100) { _ in Synthetic.hold() }
        events += feed(&detector, from: 1000, through: 1500, by: 100) { _ in Synthetic.arch() }

        XCTAssertEqual(events, [.arch(tMs: 1000)], "a blink is not a recovery")
    }

    // MARK: - Once per attempt, and the re-arm

    func testASecondArchInTheSameAttemptSaysNothing() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 8000, by: 100) { _ in Synthetic.arch() }

        XCTAssertEqual(events, [.arch(tMs: 1000)], "one phrase per direction per attempt")
    }

    func testAClearedFaultReArmsAfterTwoSecondsAndFiveSecondsSpacing() {
        var detector = LiveFormDetector()
        var events = feed(&detector, from: 0, through: 1000, by: 100) { _ in Synthetic.arch() }
        // The fault is clear for ~2.9 s: long enough to re-arm at 3.1 s.
        events += feed(&detector, from: 1100, through: 4000, by: 100) { _ in Synthetic.hold() }
        // Red again: the 1.0 s hold is met at 5.1 s, but the global 5 s spacing
        // (last phrase 1.0 s) is only met at 6.0 s — so the second phrase is
        // 6.0 s, not 5.1 s.
        events += feed(&detector, from: 4100, through: 6200, by: 100) { _ in Synthetic.arch() }

        XCTAssertEqual(
            events, [.arch(tMs: 1000), .arch(tMs: 6000)],
            "re-armed after the clear, but spaced five seconds from the first phrase")
    }

    func testAHoldAttemptEndsAfterOneSecondWithoutTheGate() {
        var detector = LiveFormDetector()
        // A red arch for 0.9 s (no phrase yet)…
        _ = feed(&detector, from: 0, through: 900, by: 100) { _ in Synthetic.arch() }
        XCTAssertTrue(detector.isInAttempt)
        // …then standing frames: the gate is false, and 1.0 s of it ends the
        // attempt, so a red arch that arrives later starts from scratch.
        _ = feed(&detector, from: 1000, through: 2000, by: 100) { _ in Synthetic.standingArch() }
        XCTAssertFalse(detector.isInAttempt, "1.0 s without the gate ends the attempt")
    }

    // MARK: - The geometry gates

    func testNotInvertedSaysNothing() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in
            Synthetic.standingArch()
        }
        XCTAssertEqual(events, [], "a standing body is never in a hold")
        XCTAssertFalse(detector.isInAttempt)
    }

    func testNoNoseMeansTheSignIsUnknownAndTheCueIsBent() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 1200, by: 100) { _ in Synthetic.bent() }

        XCTAssertEqual(events, [.bent(tMs: 1000)], "a red without a nose is directionless")
    }

    func testANegativeSignedBananaIsAHollow() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 1200, by: 100) { _ in Synthetic.hollow() }

        XCTAssertEqual(events, [.hollow(tMs: 1000)])
    }

    // MARK: - The red test

    func testRedIsTheStressDiagramBand() {
        XCTAssertFalse(LiveFormDetector.isRed(0.0))
        XCTAssertFalse(LiveFormDetector.isRed(0.13), "still amber")
        XCTAssertTrue(LiveFormDetector.isRed(0.14), "the band flips to red at exactly 0.14 L")
        XCTAssertTrue(LiveFormDetector.isRed(0.5))
        XCTAssertFalse(LiveFormDetector.isRed(.nan), "an unmeasured banana is not a fault")
    }

    // MARK: - The offline agreement

    /// The live banana is `Features.banana`'s own body measured in the live
    /// body length — same maths, one helper.
    func testTheLiveBananaIsTheOfflineBananaInTheLiveBodyLength() throws {
        let joints = Synthetic.arch()
        let config = LiveFormConfig()
        let live = try XCTUnwrap(LiveFormDetector.banana(joints: joints, config: config))

        // Rebuild the same three midpoints and the same live body length, and
        // run them through the offline `Features.banana` arrays.
        let shoulder = try XCTUnwrap(midpoint(joints, .leftShoulder, .rightShoulder))
        let hip = try XCTUnwrap(midpoint(joints, .leftHip, .rightHip))
        let ankle = try XCTUnwrap(midpoint(joints, .leftAnkle, .rightAnkle))
        let nose = try XCTUnwrap(joints[.nose])
        let length = hypot(shoulder.x - hip.x, shoulder.y - hip.y)
            + hypot(hip.x - ankle.x, hip.y - ankle.y)
        func bodyPoint(_ point: Keypoint) -> BodyPoint {
            BodyPoint(u: point.x / length, v: -point.y / length)
        }
        let offline = Features.banana(
            shoulderMid: [bodyPoint(shoulder)],
            hipMid: [bodyPoint(hip)],
            ankleMid: [bodyPoint(ankle)],
            nose: [bodyPoint(nose)]
        )[0]

        XCTAssertEqual(live.signed, offline, accuracy: 1e-12)
        XCTAssertEqual(live.magnitude, abs(offline), accuracy: 1e-12)
        XCTAssertGreaterThan(live.signed, 0, "an arch is a positive banana")
    }

    private func midpoint(
        _ joints: [Joint: Keypoint], _ left: Joint, _ right: Joint
    ) -> Keypoint? {
        guard let a = joints[left], let b = joints[right] else { return nil }
        return Keypoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, visibility: min(a.visibility, b.visibility))
    }
}
