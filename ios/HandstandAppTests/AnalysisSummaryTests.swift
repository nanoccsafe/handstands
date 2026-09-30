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

        let summary = AnalysisSummary.make(from: analysis, frames: frames, hasReference: false)

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

        // Measurable: nothing to explain away, and every frame had a person.
        XCTAssertNil(summary.unusableReason)
        XCTAssertEqual(summary.detectedFrames, frames.count)
    }

    // MARK: - With the reference

    /// The same clip scored against the (synthetic, inlined) parity
    /// reference: a score is present, and so are its ranked faults.
    func testWithTheParityReferenceAScoreIsPresent() throws {
        let reference = try ScoreReference.decode(ParityReference.data)
        let frames = Self.holdingClip()
        let analysis = Analyzer.analyze(frames, reference: reference)

        let summary = AnalysisSummary.make(from: analysis, frames: frames, hasReference: true)

        XCTAssertEqual(summary.holdCount, analysis.phases.holdCount)
        XCTAssertNotNil(
            analysis.clipScore, "the hold has more than the 5 valid frames scoring needs")
        XCTAssertEqual(summary.clipScore, analysis.clipScore?.score)
        XCTAssertTrue(summary.hasReference)
        XCTAssertNil(summary.unusableReason, "this clip was measured")
        XCTAssertEqual(summary.detectedFrames, frames.count)
        XCTAssertFalse(summary.topFaults.isEmpty, "a scored hold has ranked faults")
        XCTAssertLessThanOrEqual(summary.topFaults.count, Scorer.topFaults, "at most three")
        // The faults are the scorer's feature keys — what FaultLabel turns
        // into words for the screen.
        for name in summary.topFaults {
            XCTAssertFalse(FaultLabel.text(for: name).isEmpty, "\(name) can be said")
        }
    }

    // MARK: - Clips that could not be measured

    /// A video nobody could be seen in: the analysis finishes with no
    /// holds, no score and `0` for the longest hold — and, because there
    /// was no body to measure, says *why*: the screens must not show
    /// "Holds: 0" as if the athlete had failed to hold.
    func testAClipWithNoPersonSummarisesAsZeroAndSaysWhyItCannotBeMeasured() throws {
        let frames = (0..<10).map { index in
            PostProcessInputFrame(
                tMs: index * 33, detected: false, trainerContact: false, joints: [:])
        }
        let analysis = Analyzer.analyze(frames, reference: nil)

        let summary = AnalysisSummary.make(from: analysis, frames: frames, hasReference: false)

        XCTAssertEqual(summary.holdCount, 0)
        XCTAssertEqual(summary.longestHoldS, 0)
        XCTAssertNil(summary.clipScore)
        XCTAssertEqual(summary.frames, frames.count)

        // Nobody was found, and the clip says why it could not be measured.
        XCTAssertEqual(summary.detectedFrames, 0)
        XCTAssertNotNil(summary.unusableReason)
        XCTAssertEqual(summary.unusableReason, analysis.processed.bodyLength.reason)
    }

    /// A person the model finds but cannot measure — legs below
    /// `MIN_VISIBILITY`, exactly the real clip that triggered this: Vision
    /// saw a body in every frame, the ankles were never confident enough to
    /// measure a body length with. The summary must carry the reason (and
    /// zero holds, never a hold count read as "did not hold") while still
    /// counting the frames that had a person in them.
    func testAClipTheAppCannotMeasureSaysWhyAndCountsTheDetectedFrames() throws {
        let frames = (0..<Self.frameCount).map { index -> PostProcessInputFrame in
            var frame = Self.holdingFrame(tMs: index * Self.stepMs)
            // Ankles and knees under the 0.5 gate: no shin, no thigh → no L.
            for joint in [Joint.leftAnkle, .rightAnkle, .leftKnee, .rightKnee] {
                frame.joints[joint]?.visibility = 0.3
            }
            return frame
        }
        let analysis = Analyzer.analyze(frames, reference: nil)

        let summary = AnalysisSummary.make(from: analysis, frames: frames, hasReference: false)

        XCTAssertFalse(analysis.processed.usable, "no body length can be measured")
        XCTAssertNotNil(summary.unusableReason)
        XCTAssertEqual(summary.unusableReason, analysis.processed.bodyLength.reason)
        XCTAssertEqual(summary.holdCount, 0, "an unmeasurable clip has no holds")
        XCTAssertNil(summary.clipScore)
        XCTAssertEqual(summary.detectedFrames, frames.count, "the person was found every frame")
    }

    /// `detectedFrames` counts the frames the model found *a person* in —
    /// which is not the same as the frames the pipeline could measure, so
    /// it is read off the input frames, not off the analysis.
    func testDetectedFramesCountsTheFramesWithAPerson() throws {
        // Every second frame has nobody in it: 30 of 60. The 30 detected
        // ones still measure a body length (MIN_BODY_FRAMES is 10), so the
        // clip itself stays measurable.
        let frames = Self.holdingClip().enumerated().map { index, frame -> PostProcessInputFrame in
            var frame = frame
            frame.detected = index % 2 == 0
            return frame
        }
        let analysis = Analyzer.analyze(frames, reference: nil)

        let summary = AnalysisSummary.make(from: analysis, frames: frames, hasReference: false)

        XCTAssertEqual(summary.detectedFrames, 30)
        XCTAssertEqual(summary.frames, frames.count)
        XCTAssertNil(summary.unusableReason, "the clip was measured")
        XCTAssertEqual(summary.holdCount, analysis.phases.holdCount)
    }
}
