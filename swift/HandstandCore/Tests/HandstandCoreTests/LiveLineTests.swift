import XCTest

@testable import HandstandCore

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

        /// A frame that passes every *line* gate but is a red banana: the two
        /// sides' midpoints bend (the right hip is far to the right, so the
        /// hip midpoint sits off the shoulder→ankle line) while the left side
        /// stays straight and confident, which is the side `hipAngleDeg`
        /// picks. Chainlink #97: this must never be a line however straight
        /// the wrist→ankle vector and the left side look.
        static func bananaRedLine() -> [Joint: Keypoint] {
            [
                .leftWrist: Keypoint(x: 500, y: 1850, visibility: 0.9),
                .rightWrist: Keypoint(x: 580, y: 1850, visibility: 0.6),
                .leftShoulder: Keypoint(x: 500, y: 1550, visibility: 0.9),
                .rightShoulder: Keypoint(x: 580, y: 1550, visibility: 0.6),
                .leftHip: Keypoint(x: 500, y: 1000, visibility: 0.9),
                .rightHip: Keypoint(x: 920, y: 1000, visibility: 0.6),
                .leftKnee: Keypoint(x: 500, y: 725, visibility: 0.9),
                .rightKnee: Keypoint(x: 920, y: 725, visibility: 0.6),
                .leftAnkle: Keypoint(x: 500, y: 450, visibility: 0.9),
                .rightAnkle: Keypoint(x: 580, y: 450, visibility: 0.6),
                .leftFootIndex: Keypoint(x: 500, y: 380, visibility: 0.9),
                .rightFootIndex: Keypoint(x: 580, y: 380, visibility: 0.6),
                .nose: Keypoint(x: 500, y: 1600, visibility: 0.9),
            ]
        }

        /// A frame that passes every *line* gate except a controlled hip: the
        /// wrists, hips and ankles of `frame()`, and a single confident
        /// shoulder placed so `hipAngleDeg` measures **exactly** `hipDeg`.
        /// The line gate's 160° and the hip-red band just above it overlap
        /// (chainlink #100 review), so the measured value has to be exact
        /// rather than `frame()`'s ±30 px shoulder fudge — both shoulders sit
        /// on the same point, so the left side is the one measured.
        static func hipExactly(_ hipDeg: Double) -> [Joint: Keypoint] {
            let leftHip = (x: 520.0, y: 1150.0)
            let reach = 500.0
            let radians = hipDeg * Double.pi / 180.0
            let shoulder = (
                x: leftHip.x + reach * sin(radians),
                y: leftHip.y - reach * cos(radians)
            )
            return [
                .leftWrist: Keypoint(x: 500, y: 1850, visibility: 0.9),
                .rightWrist: Keypoint(x: 580, y: 1850, visibility: 0.9),
                .leftShoulder: Keypoint(x: shoulder.x, y: shoulder.y, visibility: 0.9),
                .rightShoulder: Keypoint(x: shoulder.x, y: shoulder.y, visibility: 0.9),
                .leftHip: Keypoint(x: 520, y: 1150, visibility: 0.9),
                .rightHip: Keypoint(x: 560, y: 1150, visibility: 0.9),
                .leftAnkle: Keypoint(x: 520, y: 450, visibility: 0.9),
                .rightAnkle: Keypoint(x: 560, y: 450, visibility: 0.9),
                .nose: Keypoint(x: shoulder.x, y: shoulder.y + 60, visibility: 0.9),
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

        XCTAssertEqual(
            events, [.up(tMs: 500), .lineAchieved(tMs: 500)],
            "up is acknowledged first, then one line cue, both at 0.5 s")
        XCTAssertTrue(detector.isInLine)
        XCTAssertTrue(detector.isUpNow)
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
            [.up(tMs: 500), .lineAchieved(tMs: 500), .lineLost(tMs: 2000, heldS: 1.5)],
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
            [
                .up(tMs: 500), .lineAchieved(tMs: 500),
                .lineLost(tMs: 1500, heldS: 1.0), .lineAchieved(tMs: 2600),
            ]
        )
    }

    // MARK: - The geometry gates

    func testAPikeIsNotALine() {
        var detector = LiveLineDetector()
        // Straight up (body angle 0°) but only 140° at the hip.
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in
            Synthetic.frame(hipDeg: 140)
        }

        XCTAssertEqual(events, [.up(tMs: 500)], "140° at the hip is a pike, not a line — but still up")
        XCTAssertFalse(detector.isInLine)
        XCTAssertTrue(detector.isUpNow, "a pike is still up — it earns the acknowledgement")
        XCTAssertTrue(detector.isInverted, "a pike is still upside down — just not a line")
    }

    func testALeanIsNotALine() {
        var detector = LiveLineDetector()
        // Perfectly open at the hip (the rigid turn keeps 180°) but leaning
        // 25° off vertical — outside the 15° body angle.
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in
            Synthetic.frame(leanDeg: 25)
        }

        XCTAssertEqual(events, [.up(tMs: 500)], "25° of lean is a handstand walk, not a held line — but still up")
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

    /// Chainlink #97: a red banana is not a line, even when the wrist→ankle
    /// vector is vertical and the visible side's hip is straight — the very
    /// case the app used to cheer with "Line. Hold it.".
    func testARedBananaIsNeverALine() {
        var detector = LiveLineDetector()
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in
            Synthetic.bananaRedLine()
        }

        XCTAssertEqual(events, [.up(tMs: 500)], "a banana the report will call red is never a line — but still up")
        XCTAssertFalse(detector.isInLine)
        XCTAssertTrue(detector.isInverted, "it is still upside down — just not a line")
    }

    /// Chainlink #100 review: the line gate (hip ≥ 160°) and the hip-red band
    /// (red at about 161° and below) overlap, so a 160.5° frame used to be both
    /// a line *and* a piked-hip fault. `isLine` now rejects a red hip, exactly
    /// as it already rejected a red banana; an open hip is still a line.
    func testARedHipIsNotALineButAnOpenHipStillIs() {
        // The isolation is the hip alone: 160.5° is red, its banana is not, so
        // without the hip check this exact frame is a line.
        let pikedFrame = Synthetic.hipExactly(160.5)
        XCTAssertTrue(LiveFormDetector.isHipRed(160.5), "160.5° is a red hip")
        XCTAssertFalse(
            LiveFormDetector.isBananaRed(
                joints: pikedFrame, minVisibility: LiveLineConfig().minVisibility),
            "…and its banana is not red, so only the hip can reject it")

        var piked = LiveLineDetector()
        let pikedEvents = feed(&piked, from: 0, through: 2000, by: 100) { _ in
            Synthetic.hipExactly(160.5)
        }
        XCTAssertEqual(
            pikedEvents, [.up(tMs: 500)],
            "160.5° is a red hip — not a line, but still up")
        XCTAssertFalse(piked.isInLine)

        var open = LiveLineDetector()
        let openEvents = feed(&open, from: 0, through: 2000, by: 100) { _ in
            Synthetic.hipExactly(170)
        }
        XCTAssertEqual(
            openEvents, [.up(tMs: 500), .lineAchieved(tMs: 500)],
            "170° at the hip is open — still a line")
        XCTAssertTrue(open.isInLine)
    }

    // MARK: - The "Up." acknowledgement (chainlink #100)

    /// A piked hold: 140° at the hip is a pike, so the *line* gates fail and
    /// only `.up` can fire — which keeps these assertions about the
    /// acknowledgement alone.
    private func upFrame() -> [Joint: Keypoint] { Synthetic.frame(hipDeg: 140) }

    func testAnUpOffTheGroundIsAcknowledgedOnceAfterTheOnset() {
        var detector = LiveLineDetector()
        let events = feed(&detector, from: 0, through: 1200, by: 100) { _ in upFrame() }

        XCTAssertEqual(events, [.up(tMs: 500)], "0.6 s up earns exactly one acknowledgement")
        XCTAssertTrue(detector.isUpNow)
    }

    func testAShortUpSaysNothing() {
        var detector = LiveLineDetector()
        let events = feed(&detector, from: 0, through: 400, by: 100) { _ in upFrame() }

        XCTAssertEqual(events, [], "0.4 s up is below the 0.5 s onset")
        XCTAssertFalse(detector.isUpNow)
    }

    func testStandingIsNeverUp() {
        var detector = LiveLineDetector()
        let events = feed(&detector, from: 0, through: 2000, by: 100) { _ in Synthetic.standing() }

        XCTAssertEqual(events, [], "wrists below the hips is never up")
        XCTAssertFalse(detector.isUpNow)
    }

    func testAShortDipDoesNotReFireUpButALongOneEarnsASecond() {
        var detector = LiveLineDetector()
        // Up, then a 0.5 s dip — inside upEndS (1 s), so the stretch survives.
        var events = feed(&detector, from: 0, through: 600, by: 100) { _ in upFrame() }
        events += feed(&detector, from: 700, through: 1100, by: 100) { _ in Synthetic.standing() }
        events += feed(&detector, from: 1200, through: 1800, by: 100) { _ in upFrame() }
        XCTAssertEqual(events, [.up(tMs: 500)], "a dip shorter than upEndS does not re-fire")
        XCTAssertTrue(detector.isUpNow)

        // A full 1.2 s down ends the stretch, so coming back up is a new one.
        var again = LiveLineDetector()
        var more = feed(&again, from: 0, through: 600, by: 100) { _ in upFrame() }
        more += feed(&again, from: 700, through: 1900, by: 100) { _ in Synthetic.standing() }
        XCTAssertFalse(again.isUpNow, "1.2 s down ends the stretch")
        more += feed(&again, from: 2000, through: 2600, by: 100) { _ in upFrame() }
        XCTAssertEqual(more, [.up(tMs: 500), .up(tMs: 2500)])
    }

    func testUpPrecedesLineAchievedOnTheSameFrame() {
        var detector = LiveLineDetector()
        let events = feed(&detector, from: 0, through: 800, by: 100) { _ in Synthetic.frame() }

        XCTAssertEqual(
            events, [.up(tMs: 500), .lineAchieved(tMs: 500)],
            "the acknowledgement is spoken before the line cue")
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
        XCTAssertEqual(events, [.up(tMs: 610), .lineAchieved(tMs: 610)])

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
                .up(tMs: 500),
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

        XCTAssertEqual(events, [.up(tMs: 500), .lineAchieved(tMs: 500)])
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
