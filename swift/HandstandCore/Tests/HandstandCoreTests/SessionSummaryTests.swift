import XCTest

@testable import HandstandCore

/// The summary view's answers (chainlink #49): the heat strip's bins and
/// their colours, the worst moment (edges skipped, ties to the earlier
/// frame), and the coaching cues in both modes — plus the proof that
/// `featureSeverities` is the one severity implementation the diagram draws
/// from.
///
/// Everything here is built from synthetic frames in the test itself (the
/// way `StressDiagramTests`/`FeaturesTests` do their unit cases): no
/// fixtures, no real videos, no views.
final class SessionSummaryTests: XCTestCase {
    // MARK: - The synthetic clip

    /// The clip's body length in display pixels — `StressDiagramTests.length`.
    private static let length = 300.0
    /// Where the wrist midpoint sits in the display frame.
    private static let wristX = 200.0
    private static let wristY = 500.0

    /// Where each joint sits in the display frame: the same column-up the
    /// stress-diagram tests stand their athletes on, so `frame(_:…)` has
    /// positions to draw.
    private static func station(_ joint: Joint) -> Point2 {
        switch joint {
        case .nose: return Point2(x: wristX - 10, y: wristY - 140)
        case .leftWrist: return Point2(x: wristX - 15, y: wristY)
        case .rightWrist: return Point2(x: wristX + 15, y: wristY)
        case .leftElbow, .rightElbow:
            let dx = joint == .leftElbow ? -10.0 : 10.0
            return Point2(x: wristX + dx, y: wristY - 60)
        case .leftShoulder, .rightShoulder:
            let dx = joint == .leftShoulder ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 120)
        case .leftHip, .rightHip:
            let dx = joint == .leftHip ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 220)
        case .leftKnee, .rightKnee:
            let dx = joint == .leftKnee ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 320)
        case .leftAnkle, .rightAnkle:
            let dx = joint == .leftAnkle ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 400)
        case .leftFootIndex, .rightFootIndex:
            let dx = joint == .leftFootIndex ? -5.0 : 5.0
            return Point2(x: wristX + dx, y: wristY - 420)
        }
    }

    /// A clip of `frames` frames, 100 ms apart (0…900 for the default
    /// ten), every joint at its station, every feature `NaN` until a test
    /// overrides it — and `facing_sign` 1.
    ///
    /// `phaseList` / `holdIds` set the per-frame phases and hold numbers
    /// (default: one hold, numbered 0, over the whole clip);
    /// `invalidFrames` makes whole frames unmeasured (`features.valid`);
    /// `unusable` marks the clip as one nobody could measure.
    private func makeAnalysis(
        frames: Int = 10,
        stepMs: Int = 100,
        phase: Phase = .hold,
        phaseList: [Phase]? = nil,
        holdIds: [Int]? = nil,
        overrides: [String: [Double]] = [:],
        invalidFrames: Set<Int> = [],
        unusable: Bool = false,
        holdScores: [HoldScore] = []
    ) -> Analysis {
        precondition(frames >= 1, "at least one frame")
        let stamps = (0..<frames).map { $0 * stepMs }
        let phases = phaseList ?? [Phase](repeating: phase, count: frames)
        let ids =
            holdIds
            ?? [Int](repeating: phase == .hold ? 0 : PhaseSegmenter.noHold, count: frames)

        var processedFrames: [ProcessedFrame] = []
        for _ in 0..<frames {
            var joints: [Joint: Point2] = [:]
            var validJoints: Set<Joint> = []
            for joint in Joint.allCases {
                joints[joint] = Self.station(joint)
                validJoints.insert(joint)
            }
            processedFrames.append(
                ProcessedFrame(joints: joints, valid: validJoints, filled: []))
        }
        let processed = ProcessedClip(
            frames: processedFrames,
            bodyLength: BodyLength(
                reason: nil, torsoPx: Self.length, thighPx: 0, shinPx: 0, frames: [:])
        )

        let clipPhases = ClipPhases(
            tMs: stamps,
            phase: phases,
            holdId: ids,
            signals: PhaseSegmenter.unusableSignals(frames: frames, reason: ""),
            runs: [],
            usable: !unusable,
            unusableReason: unusable ? "no body length" : ""
        )

        var values: [String: [Double]] = [:]
        for name in Features.featureNames + Features.comColumns {
            values[name] = [Double](repeating: .nan, count: frames)
        }
        values["facing_sign"] = [Double](repeating: 1.0, count: frames)
        for (name, column) in overrides {
            precondition(
                column.count == frames, "expected \(frames) values for \(name), got \(column.count)")
            values[name] = column
        }

        let features = ClipFeatures(
            tMs: stamps,
            phase: phases,
            holdId: ids,
            values: values,
            balanceZone: [BalanceZone?](repeating: nil, count: frames),
            comComplete: [Bool](repeating: true, count: frames),
            valid: (0..<frames).map { !invalidFrames.contains($0) },
            unusableReason: unusable ? "no body length" : ""
        )

        return Analysis(
            processed: processed,
            phases: clipPhases,
            features: features,
            holdScores: holdScores,
            clipScore: nil
        )
    }

    /// A reference carrying only the named features at the given
    /// `(mean, sd)` — the tests' own numbers, never a real one.
    private func scoreReference(
        _ entries: [String: (mean: Double, sd: Double)]
    ) -> ScoreReference {
        ScoreReference(
            features: entries.mapValues { ScoreReference.Stat(mean: $0.mean, sd: $0.sd, n: 42) },
            nHolds: 42,
            builtFrom: "SessionSummaryTests' inline reference"
        )
    }

    /// A scored hold carrying whatever `topFaults` the test wants.
    private func holdScore(
        holdId: Int = 0, topFaults: [HoldScore.TopFault] = []
    ) -> HoldScore {
        HoldScore(
            holdId: holdId,
            holdFrames: 10,
            validFrames: 10,
            holdStartMs: 0,
            holdEndMs: 900,
            holdDurationS: 0.9,
            score: 60,
            reason: "",
            deviation: 0.4,
            penalty: 0,
            groups: [:],
            values: [:],
            z: [:],
            topFaults: topFaults,
            missingGroups: []
        )
    }

    private func column(_ value: Double, _ count: Int) -> [Double] {
        [Double](repeating: value, count: count)
    }

    // MARK: - The heat strip

    /// The bins tile the clip exactly and each carries the max severity of
    /// the frames inside it: 0 (on target) for the first bin, 0.8 for the
    /// one with a hip 10° past its target, 1.0 for the one with 15°.
    func testBinsCoverTheClipAndCarryTheMaxSeverityPerBin() throws {
        let hip = [172, 172, 172, 155, 172, 172, 172, 150, 172, 172].map(Double.init)
        let analysis = makeAnalysis(overrides: ["hip_angle": hip])

        let strip = SessionSummary.heatStrip(analysis: analysis, reference: nil, bins: 3)

        XCTAssertEqual(strip.count, 3)
        // The bins cover [0, 900] with no gaps and no overlaps.
        XCTAssertEqual(strip.map(\.startMs), [0, 300, 600])
        XCTAssertEqual(strip.map(\.endMs), [300, 600, 900])

        // Bin 0's frames all pass their target: severity 0, green.
        XCTAssertEqual(try XCTUnwrap(strip[0].severity), 0, accuracy: 1e-9)
        XCTAssertEqual(strip[0].band, .ok)
        // Bin 1 holds the 155° frame: 0.3 + (165 − 155)/20 = 0.8, bad.
        XCTAssertEqual(try XCTUnwrap(strip[1].severity), 0.8, accuracy: 1e-9)
        XCTAssertEqual(strip[1].band, .bad)
        // Bin 2 holds the 150° frame: 0.3 + 15/20 = 1.05, capped at 1.
        XCTAssertEqual(try XCTUnwrap(strip[2].severity), 1.0, accuracy: 1e-9)
        XCTAssertEqual(strip[2].band, .bad)

        XCTAssertTrue(strip.allSatisfy(\.inHold), "every frame of this clip is held")
    }

    /// The default is 120 bins covering the clip — and with far fewer
    /// frames than bins, the bins that caught no frame are `.neutral`
    /// rather than invented numbers.
    func testMoreBinsThanFramesLeavesTheEmptyBinsNeutral() throws {
        let analysis = makeAnalysis(overrides: ["hip_angle": column(140, 10)])

        let strip = SessionSummary.heatStrip(analysis: analysis, reference: nil)

        XCTAssertEqual(strip.count, 120)
        XCTAssertEqual(strip.first?.startMs, 0)
        XCTAssertEqual(strip.last?.endMs, 900)
        let empty = strip.filter { $0.severity == nil }
        XCTAssertGreaterThan(empty.count, 0, "ten frames cannot fill 120 bins")
        XCTAssertTrue(empty.allSatisfy { $0.band == .neutral && !$0.inHold })
        // The frames that *are* somewhere make their bins red: hip 140° is
        // a full severity (0.3 + 25/20, capped at 1).
        let filled = strip.filter { $0.severity != nil }
        XCTAssertFalse(filled.isEmpty)
        XCTAssertTrue(filled.allSatisfy { $0.band == .bad })
    }

    /// Frames outside a hold are not judged: their bins are grey and
    /// carry no severity, while the held stretch of the same clip keeps
    /// its colours.
    func testBinsOutsideAHoldAreNeutral() throws {
        let phases = [Phase](repeating: .pre, count: 6) + [Phase](repeating: .hold, count: 4)
        let ids = [Int](repeating: PhaseSegmenter.noHold, count: 6) + [Int](repeating: 0, count: 4)
        let analysis = makeAnalysis(
            phaseList: phases, holdIds: ids, overrides: ["hip_angle": column(140, 10)])

        let strip = SessionSummary.heatStrip(analysis: analysis, reference: nil, bins: 3)

        XCTAssertEqual(strip.count, 3)
        for bin in strip.prefix(2) {
            XCTAssertNil(bin.severity, "a warm-up is not a fault")
            XCTAssertEqual(bin.band, .neutral)
            XCTAssertFalse(bin.inHold)
        }
        XCTAssertEqual(try XCTUnwrap(strip[2].severity), 1.0, accuracy: 1e-9)
        XCTAssertEqual(strip[2].band, .bad)
        XCTAssertTrue(strip[2].inHold, "frames 6…9 are the hold")
    }

    // MARK: - The worst moment

    /// The two worst frames of the clip sit in the hold's first and last
    /// 0.3 s — the entry and the exit — so they are skipped, and the worst
    /// frame of the *held* form wins instead.
    func testWorstMomentSkipsTheHoldEdges() throws {
        // Frames 0 and 9 (t = 0 ms and 900 ms) are the worst numbers of
        // the clip (severity 1.0) but sit outside the edges; frame 5
        // (t = 500 ms, severity 0.8) is the worst frame actually held.
        let hip = [140, 172, 172, 172, 172, 155, 172, 172, 172, 140].map(Double.init)
        let analysis = makeAnalysis(overrides: ["hip_angle": hip])

        let worst = try XCTUnwrap(SessionSummary.worstMoment(analysis: analysis, reference: nil))

        XCTAssertEqual(worst.frameIndex, 5)
        XCTAssertEqual(worst.tMs, 500)
        XCTAssertEqual(worst.severity, 0.8, accuracy: 1e-9)
        XCTAssertEqual(worst.holdId, 0)
        XCTAssertEqual(worst.features, ["hip_angle"], "0.8 is over the 0.5 cut")
    }

    /// Two frames equally bad: the earlier one is the worst moment, so
    /// the card never flickers between them.
    func testWorstMomentTiesGoToTheEarlierFrame() throws {
        let hip = [172, 172, 172, 155, 172, 172, 155, 172, 172, 172].map(Double.init)
        let analysis = makeAnalysis(overrides: ["hip_angle": hip])

        let worst = try XCTUnwrap(SessionSummary.worstMoment(analysis: analysis, reference: nil))

        XCTAssertEqual(worst.frameIndex, 3, "frames 3 and 6 tie at 0.8; 3 is earlier")
        XCTAssertEqual(worst.tMs, 300)
        XCTAssertEqual(worst.severity, 0.8, accuracy: 1e-9)
    }

    /// Nothing held → no worst moment, however bad the numbers are.
    func testWorstMomentIsNilWithoutAHold() throws {
        let analysis = makeAnalysis(phase: .pre, overrides: ["hip_angle": column(140, 10)])

        XCTAssertNil(SessionSummary.worstMoment(analysis: analysis, reference: nil))
    }

    /// A hold with nothing wrong (or only zeros) has no worst moment:
    /// severity must be *above* zero to be worth a card.
    func testWorstMomentIsNilWhenEveryFrameIsOnTarget() throws {
        let analysis = makeAnalysis(overrides: ["hip_angle": column(172, 10)])

        XCTAssertNil(SessionSummary.worstMoment(analysis: analysis, reference: nil))
    }

    // MARK: - Coaching cues without a reference

    /// Without a reference the candidates are the longest hold's medians
    /// that fail `FeatureTolerances`: a 140° hip is piked, and the cue
    /// says so with the one text table's wording.
    func testCuesWithoutAReferenceComeFromFailingMedians() throws {
        let analysis = makeAnalysis(overrides: ["hip_angle": column(140, 10)])

        let cues = CoachingCues.cues(analysis: analysis, reference: nil)

        XCTAssertEqual(cues.count, 1)
        XCTAssertEqual(cues[0].feature, "hip_angle")
        XCTAssertEqual(cues[0].severity, 1.0, accuracy: 1e-9)
        XCTAssertEqual(cues[0].text, CoachingCues.texts["hip_angle"])
    }

    /// The `banana` cue picks its direction from the sign of the hold's
    /// median: positive is the arch, negative the hollow (banana is
    /// athlete-signed already).
    func testBananaSignPicksTheText() throws {
        let arching = makeAnalysis(overrides: ["banana": column(0.2, 10)])
        let archingCues = CoachingCues.cues(analysis: arching, reference: nil)
        XCTAssertEqual(archingCues.map(\.feature), ["banana_pos"])
        XCTAssertEqual(archingCues[0].text, CoachingCues.texts["banana_pos"])

        let hollowing = makeAnalysis(overrides: ["banana": column(-0.2, 10)])
        let hollowingCues = CoachingCues.cues(analysis: hollowing, reference: nil)
        XCTAssertEqual(hollowingCues.map(\.feature), ["banana_neg"])
        XCTAssertEqual(hollowingCues[0].text, CoachingCues.texts["banana_neg"])
    }

    /// More time past the fingertips than the balance cue's 30 %, and the
    /// cue appears — *after* the shape faults, because 0.4 is its
    /// severity, and says the finger-press fix.
    func testBalanceCuesComeFromTheHoldsPercentagesAndSortAt04() throws {
        let over = makeAnalysis(
            overrides: ["hip_angle": column(140, 10), "com_forward": column(1.0, 10)])

        let cues = CoachingCues.cues(analysis: over, reference: nil)

        XCTAssertEqual(cues.map(\.feature), ["hip_angle", "balance_over"])
        XCTAssertEqual(cues[0].severity, 1.0, accuracy: 1e-9)
        XCTAssertEqual(cues[1].severity, 0.4, accuracy: 1e-9)
        XCTAssertEqual(cues[1].text, CoachingCues.texts["balance_over"])

        // The other side: behind the heel of the hand all hold long.
        let under = makeAnalysis(overrides: ["com_forward": column(-1.0, 10)])
        let underCues = CoachingCues.cues(analysis: under, reference: nil)
        XCTAssertEqual(underCues.map(\.feature), ["balance_under"])
        XCTAssertEqual(underCues[0].text, CoachingCues.texts["balance_under"])
    }

    /// A hold where nothing is over the limits is not an empty section:
    /// it is the one "none" cue.
    func testACleanHoldGetsTheNoneCue() throws {
        let analysis = makeAnalysis(overrides: ["hip_angle": column(172, 10)])

        let cues = CoachingCues.cues(analysis: analysis, reference: nil)

        XCTAssertEqual(cues.count, 1)
        XCTAssertEqual(cues[0].feature, "none")
        XCTAssertEqual(cues[0].severity, 0, accuracy: 1e-9)
        XCTAssertEqual(cues[0].text, CoachingCues.texts["none"])
    }

    /// Five failing medians of decreasing severity: the cue list is
    /// ordered worst first and cut to `max` (three by default, and `max`
    /// is obeyed when it is smaller).
    func testAtMostThreeCuesInSeverityOrder() throws {
        let analysis = makeAnalysis(
            overrides: [
                "hip_angle": column(140, 10),  // 1.0
                "head": column(70, 10),  // 0.8
                "knee_angle": column(160, 10),  // 0.55
                "shoulder_angle": column(159, 10),  // 0.35
                "elbow_angle": column(159, 10),  // 0.35
            ])

        let cues = CoachingCues.cues(analysis: analysis, reference: nil)

        XCTAssertEqual(cues.map(\.feature), ["hip_angle", "head", "knee_angle"])
        for (actual, expected) in zip(cues.map(\.severity), [1.0, 0.8, 0.55]) {
            XCTAssertEqual(actual, expected, accuracy: 1e-9)
        }

        let two = CoachingCues.cues(analysis: analysis, reference: nil, max: 2)
        XCTAssertEqual(two.map(\.feature), ["hip_angle", "head"])
    }

    // MARK: - Coaching cues with a reference

    /// With a reference the candidates are the longest hold's
    /// `topFaults`, kept when `|z| ≥ 1.5` — a z of −2 is as much a fault
    /// as +2 (the severity is a magnitude either way) — ordered by
    /// `min(|z|, Z_CAP)/Z_CAP`.
    func testCuesWithAReferenceComeFromTopFaults() throws {
        let faults = [
            HoldScore.TopFault(feature: "hip_angle", value: 140, mean: 175, z: 3.0),
            HoldScore.TopFault(feature: "knee_angle", value: 160, mean: 168, z: 1.0),
            HoldScore.TopFault(feature: "elbow_angle", value: 159, mean: 172, z: -2.0),
        ]
        let analysis = makeAnalysis(holdScores: [holdScore(topFaults: faults)])
        let reference = scoreReference([:])

        let cues = CoachingCues.cues(analysis: analysis, reference: reference)

        XCTAssertEqual(cues.map(\.feature), ["hip_angle", "elbow_angle"])
        XCTAssertEqual(cues[0].severity, 3.0 / 4.0, accuracy: 1e-9)
        XCTAssertEqual(cues[1].severity, 2.0 / 4.0, accuracy: 1e-9)
        XCTAssertEqual(cues[0].text, CoachingCues.texts["hip_angle"])
        XCTAssertEqual(cues[1].text, CoachingCues.texts["elbow_angle"])
    }

    /// The balance cue speaks in both modes: the percentages come from
    /// the hold's row, not from the reference.
    func testTheBalanceCueAppliesWithAReferenceToo() throws {
        let analysis = makeAnalysis(
            overrides: ["com_forward": column(1.0, 10)],
            holdScores: [holdScore(topFaults: [])])
        let reference = scoreReference([:])

        let cues = CoachingCues.cues(analysis: analysis, reference: reference)

        XCTAssertEqual(cues.map(\.feature), ["balance_over"])
        XCTAssertEqual(cues[0].severity, 0.4, accuracy: 1e-9)
    }

    // MARK: - No hold, unusable clip

    /// No hold → no cues, not a "none" cue: there is no hold to have
    /// been solid.
    func testNoHoldGivesNoCues() {
        let analysis = makeAnalysis(phase: .pre, overrides: ["hip_angle": column(140, 10)])

        XCTAssertTrue(CoachingCues.cues(analysis: analysis, reference: nil).isEmpty)
    }

    /// A clip nobody could measure answers nothing at all, even if it
    /// somehow has hold-shaped rows: there is no measured hold to coach.
    func testAnUnusableClipGivesNoCues() {
        let analysis = makeAnalysis(
            overrides: ["hip_angle": column(140, 10)], unusable: true)

        XCTAssertTrue(CoachingCues.cues(analysis: analysis, reference: nil).isEmpty)
    }

    // MARK: - One severity implementation

    /// The proof there is no second copy of the severity rules: every
    /// joint of `StressDiagram.frame` equals the max over
    /// `featureSeverities` of the joint's own features, in both modes —
    /// so what the overlay draws and what the summary reads are the same
    /// numbers by construction.
    func testFeatureSeveritiesIsWhatTheFrameDrawsFrom() throws {
        let analysis = makeAnalysis(
            overrides: [
                "hip_angle": column(140, 10),
                "line_deviation": column(0.2, 10),
                "head": column(70, 10),
            ])
        let refs: [ScoreReference?] = [nil, scoreReference(["hip_angle": (175, 5)])]

        for reference in refs {
            for i in 0..<10 {
                let severities = StressDiagram.featureSeverities(
                    i, analysis: analysis, reference: reference)
                let frame = StressDiagram.frame(
                    i, analysis: analysis, reference: reference, height: 640)
                for joint in frame.joints {
                    let expected = (StressDiagram.jointFeatures[joint.joint] ?? [])
                        .compactMap { severities[$0] }
                        .max()
                    XCTAssertEqual(
                        joint.severity, expected,
                        "\(joint.joint.rawValue) of frame \(i) disagrees with featureSeverities")
                }
                XCTAssertEqual(
                    StressDiagram.frameSeverity(i, analysis: analysis, reference: reference),
                    severities.values.max(),
                    "the frame severity is the max over the features")
            }
        }
    }

    /// Outside a hold there are no severities at all — not a zero, not a
    /// red: an empty dictionary and a `nil` frame severity.
    func testFeatureSeveritiesIsEmptyOutsideAHold() throws {
        let analysis = makeAnalysis(phase: .pre, overrides: ["hip_angle": column(140, 10)])

        XCTAssertTrue(
            StressDiagram.featureSeverities(0, analysis: analysis, reference: nil).isEmpty)
        XCTAssertNil(StressDiagram.frameSeverity(0, analysis: analysis, reference: nil))
    }

    /// …and the same on a frame the features could not measure, however
    /// bad its values are.
    func testFeatureSeveritiesIsNilOnAnUnmeasuredFrame() throws {
        let analysis = makeAnalysis(
            overrides: ["hip_angle": column(140, 10)], invalidFrames: [0])

        XCTAssertTrue(
            StressDiagram.featureSeverities(0, analysis: analysis, reference: nil).isEmpty)
        XCTAssertNil(StressDiagram.frameSeverity(0, analysis: analysis, reference: nil))
        XCTAssertFalse(
            StressDiagram.featureSeverities(1, analysis: analysis, reference: nil).isEmpty,
            "the other frames are still measured")
    }
}
