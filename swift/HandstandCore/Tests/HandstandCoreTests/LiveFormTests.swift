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
    /// the bottom, ankles straight above them, and the **right** hip (and its
    /// knee and foot) slid sideways by `hipOffsetX` — with the nose on the
    /// right, a positive offset is the belly side (an arch) and a negative one
    /// the back (a hollow).
    ///
    /// Only one side moves on purpose: `hipAngleDeg` measures the *left* side
    /// (both sides are equally confident, so it keeps the first), which stays
    /// straight — this is a red *banana* with a fine hip, the shape the
    /// "biggest fault first" rule must still call `.arch` / `.hollow`.
    /// `Synthetic.pike` is the hip fault instead. `nose` off drops the nose so
    /// the sign is unknown.
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
                .leftHip: Keypoint(x: 520, y: 1000, visibility: visibility),
                .rightHip: Keypoint(x: 560 + hipOffsetX, y: 1000, visibility: visibility),
                .leftKnee: Keypoint(x: 525, y: 725, visibility: visibility),
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

        /// A red arch: the hip slid towards the nose's side, the measured hip
        /// still fine.
        static func arch(visibility: Double = 0.9) -> [Joint: Keypoint] {
            hold(hipOffsetX: 400, visibility: visibility)
        }

        /// A red hollow: the hip slid away from the nose's side.
        static func hollow(visibility: Double = 0.9) -> [Joint: Keypoint] {
            hold(hipOffsetX: -400, visibility: visibility)
        }

        /// A red arch with no nose: the sign is unknown.
        static func bent(visibility: Double = 0.9) -> [Joint: Keypoint] {
            hold(hipOffsetX: 400, nose: false, visibility: visibility)
        }

        /// A piked hold: the shoulders swung to `hipDeg` from the hip→ankle
        /// direction, symmetric about the body's axis — a real pike whose
        /// banana stays below red (`pike(hipDeg: 150)` is red at the hip and
        /// amber at the banana, the hip fault alone).
        static func pike(hipDeg: Double, visibility: Double = 0.9) -> [Joint: Keypoint] {
            let hipMid = (x: 540.0, y: 1000.0)
            let reach = 550.0
            let radians = hipDeg * Double.pi / 180.0
            let shoulderMid = (
                x: hipMid.x + reach * sin(radians),
                y: hipMid.y - reach * cos(radians)
            )
            var joints = hold(visibility: visibility)
            joints[.leftShoulder] = Keypoint(
                x: shoulderMid.x - 30, y: shoulderMid.y, visibility: visibility)
            joints[.rightShoulder] = Keypoint(
                x: shoulderMid.x + 30, y: shoulderMid.y, visibility: visibility)
            joints[.nose] = Keypoint(
                x: shoulderMid.x, y: shoulderMid.y + 60, visibility: visibility)
            return joints
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
        XCTAssertEqual(config.repeatS, 2.0, "coach pace: a phrase every two seconds")
        XCTAssertEqual(config.minSpacingS, 2.0)
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

    func testARedArchHeldLongEnoughSaysIt() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 1200, by: 100) { _ in Synthetic.arch() }

        XCTAssertEqual(events, [.arch(tMs: 1000)], "1.2 s of red earns a phrase, at 1.0 s")
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

    // MARK: - Coach-mode repeats (chainlink #100)

    func testARedFaultRepeatsEveryTwoSecondsWhileRed() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 8000, by: 100) { _ in Synthetic.arch() }

        XCTAssertEqual(
            events,
            [.arch(tMs: 1000), .arch(tMs: 3000), .arch(tMs: 5000), .arch(tMs: 7000)],
            "coach pace: a phrase every two seconds while the fault stays red")
    }

    func testAClearedFaultStartsAFreshHold() {
        var detector = LiveFormDetector()
        var events = feed(&detector, from: 0, through: 1000, by: 100) { _ in Synthetic.arch() }
        // Clear for ~3 s: the red run is over, so the next red earns a fresh
        // 1.0 s hold before it speaks.
        events += feed(&detector, from: 1100, through: 4000, by: 100) { _ in Synthetic.hold() }
        events += feed(&detector, from: 4100, through: 6200, by: 100) { _ in Synthetic.arch() }

        XCTAssertEqual(
            events, [.arch(tMs: 1000), .arch(tMs: 5100)],
            "after the clear, the next phrase waits out the fresh 1 s hold")
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

    // MARK: - The hip fault (chainlink #100)

    func testHipRedIsTheStressDiagramBand() {
        for deg in [150.0, 160.0, 162.0, 170.0] {
            let severity = StressDiagram.valueSeverity(deg, feature: "hip_angle", reference: nil)
            XCTAssertEqual(
                LiveFormDetector.isHipRed(deg),
                severity.map { Severity.colourBand($0) == .bad } ?? false,
                "\(deg)° follows the stress diagram, not a copied threshold")
        }
        XCTAssertTrue(LiveFormDetector.isHipRed(150))
        XCTAssertTrue(LiveFormDetector.isHipRed(160), "160° is still in the red band")
        XCTAssertFalse(LiveFormDetector.isHipRed(162), "162° is above the red band")
        XCTAssertFalse(LiveFormDetector.isHipRed(170), "an open hip is no fault")
        XCTAssertFalse(LiveFormDetector.isHipRed(.nan), "an unmeasured hip is not a fault")
    }

    func testAPikedHipHeldLongEnoughSaysHipsOverShoulders() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 1200, by: 100) { _ in
            Synthetic.pike(hipDeg: 150)
        }
        XCTAssertEqual(events, [.hips(tMs: 1000)], "a red hip earns the pike phrase at 1.0 s")
    }

    func testAPikedHipRepeatsEveryTwoSecondsWhileRed() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 3300, by: 100) { _ in
            Synthetic.pike(hipDeg: 150)
        }
        XCTAssertEqual(
            events, [.hips(tMs: 1000), .hips(tMs: 3000)],
            "the hip phrase repeats at coach pace while the hips stay piked")
    }

    func testAnOpenHipSaysNothing() {
        var detector = LiveFormDetector()
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in
            Synthetic.pike(hipDeg: 170)
        }
        XCTAssertEqual(events, [], "170° at the hip is no fault")
    }

    func testTheBiggestFaultWinsOnAFrame() {
        // 0.2 L is a banana severity of 0.8; 160° is a hip severity of 0.55.
        let arch = LiveBanana(magnitude: 0.2, signed: 0.2)
        XCTAssertEqual(
            LiveFormDetector.chosenFault(banana: arch, hipAngleDeg: 160), .arch,
            "an arch at least as severe as the hip keeps 'Ribs in.'")
        XCTAssertEqual(
            LiveFormDetector.chosenFault(banana: arch, hipAngleDeg: 140), .hips,
            "a much worse hip wins over the arch")
        XCTAssertEqual(
            LiveFormDetector.chosenFault(
                banana: LiveBanana(magnitude: 0.2, signed: -0.2), hipAngleDeg: 150), .hips,
            "a hollow banana loses to a red hip")
        XCTAssertEqual(
            LiveFormDetector.chosenFault(
                banana: LiveBanana(magnitude: 0.2, signed: -0.2), hipAngleDeg: 170), .hollow,
            "a hollow banana with a fine hip keeps its own fault")
        XCTAssertNil(
            LiveFormDetector.chosenFault(
                banana: LiveBanana(magnitude: 0.05, signed: 0.05), hipAngleDeg: 175),
            "nothing red is no fault")
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
