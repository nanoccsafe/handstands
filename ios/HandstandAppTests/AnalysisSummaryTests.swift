import XCTest

@testable import HandstandApp
import HandstandCore

/// `AnalysisSummary.make` (chainlink #47): the facts the analysis screens
/// read off an `Analysis`.
///
/// The clip is built by hand — a static, inverted body written as
/// `PostProcessInputFrame`s — so no video, no model and no real keypoints
/// are anywhere near the test (the repo is public). The body is one
/// straight line from wrists to ankles, arms down and still, which is
/// exactly the hold `PhaseSegmenter` looks for, so the summary has a hold
/// to count and a duration to time.
final class AnalysisSummaryTests: XCTestCase {
    // MARK: - The synthetic clip

    /// How many frames the clip has, at 33 ms apart (the ~30 fps the app
    /// analyses at): 59 steps × 33 ms = 1.947 s of holding.
    private static let frameCount = 60
    private static let stepMs = 33

    /// One frame of a handstand held perfectly still: y grows downwards, the
    /// hands are at y = 300 and the feet at y = 40 — ankles well above the
    /// wrists (`invertedMin`), hips above the hands (`handsLowMinV`), the
    /// whole body on one vertical line (`bodyAngle` 0°), and nothing moving
    /// (no wrist speed, no hand step). Every joint of the 15-joint schema is
    /// present at visibility 0.9, so every frame is measurable.
    private static func holdingFrame(tMs: Int) -> PostProcessInputFrame {
        func point(_ y: Double, x: Double = 96) -> Keypoint {
            Keypoint(x: x, y: y, visibility: 0.9)
        }
        let joints: [Joint: Keypoint] = [
            // The nose sits off the shoulder→hip line, so the facing sign
            // has a side to read (x = 106 against the body's x = 96).
            .nose: Keypoint(x: 106, y: 195, visibility: 0.9),
            .leftShoulder: point(210, x: 92),
            .rightShoulder: point(210, x: 100),
            .leftElbow: point(250, x: 92),
            .rightElbow: point(250, x: 100),
            .leftWrist: point(300, x: 92),
            .rightWrist: point(300, x: 100),
            .leftHip: point(150, x: 92),
            .rightHip: point(150, x: 100),
            .leftKnee: point(100, x: 92),
            .rightKnee: point(100, x: 100),
            .leftAnkle: point(55, x: 92),
            .rightAnkle: point(55, x: 100),
            .leftFootIndex: point(40, x: 92),
            .rightFootIndex: point(40, x: 100),
        ]
        return PostProcessInputFrame(
            tMs: tMs, detected: true, trainerContact: false, joints: joints)
    }

    /// The whole clip: `frameCount` copies of the hold above, on a clock
    /// 33 ms apart.
    private static func holdingClip() -> [PostProcessInputFrame] {
        (0..<frameCount).map { index in
            holdingFrame(tMs: index * stepMs)
        }
    }

    // MARK: - No reference

    /// Without a reference the analysis still finds the hold: the summary
    /// counts it and times it from `analysis.phases`, while the score is
    /// `nil` and `hasReference` says why (the screen's "No score yet (no
    /// reference)").
    func testWithoutAReferenceTheSummaryCountsHoldsAndHasNoScore() throws {
        let frames = Self.holdingClip()
        let analysis = Analyzer.analyze(frames, reference: nil)

        let summary = AnalysisSummary.make(from: analysis, hasReference: false)

        // The clip really holds: one run, first frame to last.
        XCTAssertEqual(analysis.phases.holdCount, 1, "the synthetic clip is exactly one hold")
        XCTAssertEqual(summary.holdCount, analysis.phases.holdCount)
        XCTAssertEqual(
            summary.longestHoldS, analysis.phases.holdDurationsS().max() ?? -1,
            accuracy: 1e-9, "the longest hold is the phases' own duration")
        XCTAssertEqual(summary.longestHoldS, 1.947, accuracy: 0.001, "59 steps of 33 ms")
        XCTAssertEqual(summary.frames, frames.count)

        // No reference, no scores — never a score against nothing.
        XCTAssertNil(analysis.clipScore)
        XCTAssertNil(summary.clipScore)
        XCTAssertTrue(summary.topFaults.isEmpty)
        XCTAssertFalse(summary.hasReference)
    }

    // MARK: - With the reference

    /// The same clip scored against the (synthetic, inlined) parity
    /// reference: a score is present, and so are its ranked faults.
    func testWithTheParityReferenceAScoreIsPresent() throws {
        let reference = try ScoreReference.decode(ParityReference.data)
        let frames = Self.holdingClip()
        let analysis = Analyzer.analyze(frames, reference: reference)

        let summary = AnalysisSummary.make(from: analysis, hasReference: true)

        XCTAssertEqual(summary.holdCount, analysis.phases.holdCount)
        XCTAssertNotNil(
            analysis.clipScore, "the hold has more than the 5 valid frames scoring needs")
        XCTAssertEqual(summary.clipScore, analysis.clipScore?.score)
        XCTAssertTrue(summary.hasReference)
        XCTAssertFalse(summary.topFaults.isEmpty, "a scored hold has ranked faults")
        XCTAssertLessThanOrEqual(summary.topFaults.count, Scorer.topFaults, "at most three")
        // The faults are the scorer's feature keys — what FaultLabel turns
        // into words for the screen.
        for name in summary.topFaults {
            XCTAssertFalse(FaultLabel.text(for: name).isEmpty, "\(name) can be said")
        }
    }

    // MARK: - An empty analysis

    /// A video nobody could be seen in: no holds, no score, and `0` for the
    /// longest hold rather than an absence — the analysis finished.
    func testAClipWithNoHoldSummarisesAsZero() throws {
        let frames = (0..<10).map { index in
            PostProcessInputFrame(
                tMs: index * 33, detected: false, trainerContact: false, joints: [:])
        }
        let analysis = Analyzer.analyze(frames, reference: nil)

        let summary = AnalysisSummary.make(from: analysis, hasReference: false)

        XCTAssertEqual(summary.holdCount, 0)
        XCTAssertEqual(summary.longestHoldS, 0)
        XCTAssertNil(summary.clipScore)
        XCTAssertEqual(summary.frames, frames.count)
    }
}
