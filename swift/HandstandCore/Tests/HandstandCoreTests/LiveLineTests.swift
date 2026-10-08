import XCTest

import HandstandCore

/// What the live line detector does with one frame at a time (chainlink #91):
/// the hysteresis, the geometry gates, the timestamps — and the one check that
/// ties the live cue to the analysis it must never contradict: every
/// `lineAchieved` on the golden fixtures falls inside a hold the offline
/// `PhaseSegmenter` finds on the same input.
///
/// Everything here is synthetic (`Synthetic.frame`, a hand-placed straight
/// body in display pixels) except the last test, which reads the committed
/// `Fixtures/golden/*.json` — no camera, no video, no model.
final class LiveLineTests: XCTestCase {
    // MARK: - Synthetic frames

    /// Side-on keypoints of a body, built from the shape's own numbers:
    /// hands on the floor at `wristY`, the legs straight above them, and the
    /// hips opened or closed by `hipDeg` before the whole body is leaned by
    /// `leanDeg` around the wrists.
    private enum Synthetic {
        /// A straight line: wrist midpoint (540, 1850), hip midpoint 700 px
        /// above it, ankle midpoint 1400 px above that — a body angle of 0°
        /// and a hip angle of `hipDeg`.
        static func frame(
            leanDeg: Double = 0,
            hipDeg: Double = 180,
            visibility: Double = 0.9
        ) -> [Joint: Keypoint] {
            let wristMid = (x: 540.0, y: 1850.0)
            let hipMid = (x: 540.0, y: 1150.0)
            let ankleMid = (x: 540.0, y: 450.0)

            // The shoulder sits `shoulderReach` from the hip, at `hipDeg`
            // from the hip→ankle direction: 180° puts it straight below the
            // hip (a line), 140° folds it towards the legs (a pike).
            let reach = 500.0
            let radians = hipDeg * Double.pi / 180.0
            let shoulderMid = (
                x: hipMid.x + reach * sin(radians),
                y: hipMid.y - reach * cos(radians)
            )

            var joints: [Joint: Keypoint] = [
                .leftWrist: Keypoint(x: 500, y: 1850, visibility: visibility),
                .rightWrist: Keypoint(x: 580, y: 1850, visibility: visibility),
                .leftHip: Keypoint(x: 520, y: 1150, visibility: visibility),
                .rightHip: Keypoint(x: 560, y: 1150, visibility: visibility),
                .leftAnkle: Keypoint(x: 520, y: 450, visibility: visibility),
                .rightAnkle: Keypoint(x: 560, y: 450, visibility: visibility),
                .leftShoulder: Keypoint(x: shoulderMid.x - 30, y: shoulderMid.y, visibility: visibility),
                .rightShoulder: Keypoint(x: shoulderMid.x + 30, y: shoulderMid.y, visibility: visibility),
                .leftKnee: Keypoint(x: 525, y: 800, visibility: visibility),
                .rightKnee: Keypoint(x: 555, y: 800, visibility: visibility),
                .leftFootIndex: Keypoint(x: 525, y: 380, visibility: visibility),
                .rightFootIndex: Keypoint(x: 555, y: 380, visibility: visibility),
                .nose: Keypoint(x: shoulderMid.x, y: shoulderMid.y + 60, visibility: visibility),
            ]
            if leanDeg != 0 {
                joints = rotate(joints, by: leanDeg, around: wristMid)
            }
            return joints
        }

        /// An athlete standing on their feet: nothing is inverted, so it is
        /// never a line however straight they stand.
        static func standing(visibility: Double = 0.9) -> [Joint: Keypoint] {
            [
                .leftWrist: Keypoint(x: 500, y: 700, visibility: visibility),
                .rightWrist: Keypoint(x: 580, y: 700, visibility: visibility),
                .leftShoulder: Keypoint(x: 510, y: 500, visibility: visibility),
                .rightShoulder: Keypoint(x: 570, y: 500, visibility: visibility),
                .leftHip: Keypoint(x: 520, y: 1150, visibility: visibility),
                .rightHip: Keypoint(x: 560, y: 1150, visibility: visibility),
                .leftKnee: Keypoint(x: 525, y: 1500, visibility: visibility),
                .rightKnee: Keypoint(x: 555, y: 1500, visibility: visibility),
                .leftAnkle: Keypoint(x: 520, y: 1850, visibility: visibility),
                .rightAnkle: Keypoint(x: 560, y: 1850, visibility: visibility),
                .leftFootIndex: Keypoint(x: 525, y: 1890, visibility: visibility),
                .rightFootIndex: Keypoint(x: 555, y: 1890, visibility: visibility),
                .nose: Keypoint(x: 540, y: 440, visibility: visibility),
            ]
        }

        /// Every joint turned around `origin` by `degrees` — a rigid turn, so
        /// the hip angle survives it and only the body angle changes.
        private static func rotate(
            _ joints: [Joint: Keypoint],
            by degrees: Double,
            around origin: (x: Double, y: Double)
        ) -> [Joint: Keypoint] {
            let radians = degrees * Double.pi / 180.0
            let cosine = cos(radians)
            let sine = sin(radians)
            return joints.mapValues { point in
                let dx = point.x - origin.x
                let dy = point.y - origin.y
                return Keypoint(
                    x: origin.x + dx * cosine - dy * sine,
                    y: origin.y + dx * sine + dy * cosine,
                    visibility: point.visibility
                )
            }
        }
    }

    /// Feed a stride of line frames (or standing frames) to a fresh detector
    /// and collect everything it said.
    private func feed(
        _ detector: inout LiveLineDetector,
        from first: Int,
        through last: Int,
        by step: Int,
        joints: (Int) -> [Joint: Keypoint]
    ) -> [LiveEvent] {
        var events = [LiveEvent]()
        var tMs = first
        while tMs <= last {
            events += detector.update(tMs: tMs, joints: joints(tMs))
            tMs += step
        }
        return events
    }

    // MARK: - The hysteresis

    func testACleanLineAnnouncesItselfOnceAfterTheOnset() {
        var detector = LiveLineDetector()
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in Synthetic.frame() }

        XCTAssertEqual(events, [.lineAchieved(tMs: 500)], "one line, one cue, at 0.5 s")
        XCTAssertTrue(detector.isInLine)
        XCTAssertTrue(detector.isInverted)
    }

    func testAShortFlickerSaysNothingAtAll() {
        var detector = LiveLineDetector()
        var events = feed(&detector, from: 0, through: 300, by: 100) { _ in Synthetic.frame() }
        events += feed(&detector, from: 400, through: 2000, by: 100) { _ in Synthetic.standing() }

        XCTAssertEqual(events, [], "0.3 s of line is below the 0.5 s onset: no cue, no loss")
        XCTAssertFalse(detector.isInLine)
        XCTAssertFalse(detector.isInverted)
    }

    func testABriefGapKeepsTheLineAndALongOneEndsIt() {
        var detector = LiveLineDetector()
        var events = feed(&detector, from: 0, through: 1000, by: 100) { _ in Synthetic.frame() }
        // 300 ms without a line — inside the 0.4 s loss window.
        events += feed(&detector, from: 1100, through: 1300, by: 100) { _ in Synthetic.standing() }
        XCTAssertTrue(detector.isInLine, "a gap shorter than lossS is not a loss")

        events += feed(&detector, from: 1400, through: 1500, by: 100) { _ in Synthetic.frame() }
        // 600 ms without a line — the line is over.
        events += feed(&detector, from: 1600, through: 2100, by: 100) { _ in Synthetic.standing() }

        XCTAssertEqual(
            events,
            [.lineAchieved(tMs: 500), .lineLost(tMs: 2000, heldS: 1.5)],
            "the loss is reported once, 0.4 s after the line's last frame"
        )
        XCTAssertFalse(detector.isInLine)
    }

    func testASecondLineIsANewLineAndSpeaksAgain() {
        var detector = LiveLineDetector()
        var events = feed(&detector, from: 0, through: 1000, by: 100) { _ in Synthetic.frame() }
        events += feed(&detector, from: 1100, through: 2000, by: 100) { _ in Synthetic.standing() }
        events += feed(&detector, from: 2100, through: 3000, by: 100) { _ in Synthetic.frame() }

        XCTAssertEqual(
            events,
            [.lineAchieved(tMs: 500), .lineLost(tMs: 1500, heldS: 1.0), .lineAchieved(tMs: 2600)]
        )
    }

    // MARK: - The geometry gates

    func testAPikeIsNotALine() {
        var detector = LiveLineDetector()
        // Straight up (body angle 0°) but only 140° at the hip.
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in
            Synthetic.frame(hipDeg: 140)
        }

        XCTAssertEqual(events, [], "140° at the hip is a pike, not a line")
        XCTAssertFalse(detector.isInLine)
        XCTAssertTrue(detector.isInverted, "a pike is still upside down — just not a line")
    }

    func testALeanIsNotALine() {
        var detector = LiveLineDetector()
        // Perfectly open at the hip (the rigid turn keeps 180°) but leaning
        // 25° off vertical — outside the 15° body angle.
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in
            Synthetic.frame(leanDeg: 25)
        }

        XCTAssertEqual(events, [], "25° of lean is a handstand walk, not a held line")
        XCTAssertFalse(detector.isInLine)
    }

    func testAStandingBodyIsNeverInvertedAndNeverALine() {
        var detector = LiveLineDetector()
        let events = feed(&detector, from: 0, through: 1000, by: 100) { _ in Synthetic.standing() }

        XCTAssertEqual(events, [])
        XCTAssertFalse(detector.isInverted)
        XCTAssertFalse(detector.isInLine)
    }

    func testAnkleJointsMustBeConfidentForALine() {
        // Ankles too sure of themselves: `isInverted` falls back to the hips
        // (they say "upside down"), but the line needs confident ankles, so
        // no cue is ever earned.
        var detector = LiveLineDetector()
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in
            var joints = Synthetic.frame()
            joints[.leftAnkle]?.visibility = 0.2
            joints[.rightAnkle]?.visibility = 0.2
            return joints
        }

        XCTAssertEqual(events, [])
        _ = detector.update(tMs: 2100, joints: Synthetic.frame())
        XCTAssertTrue(detector.isInverted, "the hips are still there to judge by")
        XCTAssertFalse(detector.isInLine)
    }

    // MARK: - Time

    func testOnsetIsMeasuredInTimestampsNotFrameCounts() {
        // Variable frame rate: seven frames, 780 ms of line. The cue comes at
        // the first frame at or past 500 ms — a frame *count* would have said
        // something different at either end.
        var vfr = LiveLineDetector()
        var events = [LiveEvent]()
        for tMs in [0, 40, 90, 160, 260, 420, 610, 780] {
            events += vfr.update(tMs: tMs, joints: Synthetic.frame())
        }
        XCTAssertEqual(events, [.lineAchieved(tMs: 610)])

        // Forty-one frames of the same shape in 0.4 s: plenty of frames, not
        // enough time, no cue.
        var fast = LiveLineDetector()
        let fastEvents = feed(&fast, from: 0, through: 400, by: 10) { _ in Synthetic.frame() }
        XCTAssertEqual(fastEvents, [])
    }

    func testTimeMarksFireEveryFiveSecondsOfLineTime() {
        var detector = LiveLineDetector(timeMarksEverySeconds: 5)
        let events = feed(&detector, from: 0, through: 11_000, by: 100) { _ in Synthetic.frame() }

        XCTAssertEqual(
            events,
            [
                .lineAchieved(tMs: 500),
                .timeMark(tMs: 5000, seconds: 5),
                .timeMark(tMs: 10_000, seconds: 10),
            ],
            "marks count line time from the line's own first frame"
        )
    }

    func testNoTimeMarksWhenTheIntervalIsOff() {
        var detector = LiveLineDetector(timeMarksEverySeconds: nil)
        let events = feed(&detector, from: 0, through: 11_000, by: 100) { _ in Synthetic.frame() }

        XCTAssertEqual(events, [.lineAchieved(tMs: 500)])
    }

    // MARK: - The offline agreement

    /// The whole point of the feature: a live cue must never fire where the
    /// analysis afterwards will say "no hold". Every `lineAchieved` on every
    /// golden fixture's **input** must land inside a hold the offline
    /// `PhaseSegmenter` finds when it is run on that same input.
    func testEveryLiveCueFallsInsideAnOfflineHoldOnTheGoldenFixtures() throws {
        let urls = GoldenFixtures.urls()
        XCTAssertFalse(urls.isEmpty, "Fixtures/golden/*.json is missing from Bundle.module")

        var cuesByCase = [String: Int]()
        for url in urls {
            let name = url.deletingPathExtension().lastPathComponent
            let fixture = try GoldenFixtures.load(url)
            let frames = GoldenFixtures.inputFrames(from: fixture)
            let tMs = frames.map(\.tMs)

            // The offline answer, made from the same input the live detector
            // is fed: post-process, then the phase segmenter — the first two
            // stages of `Analyzer.analyze`.
            let processed = PostProcess.process(frames, config: PostProcessConfig())
            let phases = PhaseSegmenter.classify(
                tMs: tMs,
                processed: processed,
                trainerContact: frames.map(\.trainerContact)
            )

            var detector = LiveLineDetector()
            var cues = 0
            for (index, frame) in frames.enumerated() {
                for event in detector.update(tMs: frame.tMs, joints: frame.joints) {
                    guard case .lineAchieved(let tMs) = event else { continue }
                    cues += 1
                    XCTAssertEqual(tMs, frame.tMs, "\(name): the cue's timestamp is the frame's")
                    XCTAssertNotEqual(
                        phases.holdId[index], PhaseSegmenter.noHold,
                        "\(name): the live cue at \(tMs) ms falls outside every offline hold"
                    )
                }
            }
            cuesByCase[name] = cues
        }

        // Not vacuous: the clean case really does cue, and the arched-back
        // case really does not (its hips never open to 160°).
        XCTAssertEqual(cuesByCase["line_hold"], 1, "line_hold must produce exactly one cue")
        XCTAssertEqual(cuesByCase["banana_hold"], 0, "banana_hold must never cue")
        for (name, cues) in cuesByCase.sorted(by: { $0.key < $1.key }) where name != "banana_hold" {
            XCTAssertGreaterThanOrEqual(cues, 1, "\(name): expected at least one live cue")
        }
    }
}
