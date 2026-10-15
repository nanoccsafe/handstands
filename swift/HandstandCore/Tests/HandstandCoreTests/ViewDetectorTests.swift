import XCTest

@testable import HandstandCore

/// `ViewDetector` (chainlink #102): the Swift port of `handstand.view`'s
/// measurement — the per-frame `view_width`, the median classification and the
/// per-hold views — plus the `Analysis` accessors the app scores through
/// (`sideClipScore`, `notSideOnly`).
///
/// Everything is built here from a handful of numbers: the frames carry only
/// the two joints pairs the detector reads, at the pixel separations the test
/// picks, so a `view_width` is a ratio the test can work out by hand. No
/// fixture, no video and no reference is anywhere near it.
final class ViewDetectorTests: XCTestCase {
    // MARK: - The synthetic clip

    /// One processed frame with the two joint pairs at the given pixel
    /// separations. The pairs straddle `x = 200` and share a `y`, so a pair's
    /// distance is exactly the number given. A `nil` width leaves the pair out
    /// of the frame entirely; `isValid == false` leaves its joints in
    /// `frame.joints` but out of `frame.valid`, which is the case the
    /// detector has to skip.
    private func frame(
        shoulderWidth: Double?,
        hipWidth: Double?,
        shoulderValid: Bool = true,
        hipValid: Bool = true
    ) -> ProcessedFrame {
        var joints: [Joint: Point2] = [:]
        var valid: Set<Joint> = []
        func add(_ left: Joint, _ right: Joint, _ width: Double?, _ isValid: Bool) {
            guard let width else { return }
            joints[left] = Point2(x: 200 - width / 2.0, y: 100)
            joints[right] = Point2(x: 200 + width / 2.0, y: 100)
            if isValid {
                valid.insert(left)
                valid.insert(right)
            }
        }
        add(.leftShoulder, .rightShoulder, shoulderWidth, shoulderValid)
        add(.leftHip, .rightHip, hipWidth, hipValid)
        return ProcessedFrame(joints: joints, valid: valid, filled: [])
    }

    /// A body length whose `totalPx` is exactly `total` — the torso carries
    /// the whole sum so the parts add to it without rounding. `nil` is an
    /// unusable clip.
    private func bodyLength(_ total: Double?) -> BodyLength {
        guard let total else {
            return BodyLength(
                reason: "no body length", torsoPx: nil, thighPx: nil, shinPx: nil, frames: [:])
        }
        return BodyLength(reason: nil, torsoPx: total, thighPx: 0, shinPx: 0, frames: [:])
    }

    /// A processed clip of `frames` at one body length.
    private func clip(_ frames: [ProcessedFrame], length: Double?) -> ProcessedClip {
        ProcessedClip(frames: frames, bodyLength: bodyLength(length))
    }

    /// A features table that carries only what `holdViews` reads: one hold
    /// number and one validity flag per frame, everything else `NaN`.
    private func features(holdIds: [Int], valid: [Bool]) -> ClipFeatures {
        let frames = holdIds.count
        var values: [String: [Double]] = [:]
        for name in Features.featureNames + Features.comColumns {
            values[name] = [Double](repeating: .nan, count: frames)
        }
        values["facing_sign"] = [Double](repeating: 1.0, count: frames)
        return ClipFeatures(
            tMs: (0..<frames).map { $0 * 33 },
            phase: holdIds.map { $0 >= 0 ? Phase.hold : Phase.pre },
            holdId: holdIds,
            values: values,
            balanceZone: [BalanceZone?](repeating: nil, count: frames),
            comComplete: [Bool](repeating: true, count: frames),
            valid: valid,
            unusableReason: ""
        )
    }

    /// One scored hold, at whatever score and duration the test wants.
    private func holdScore(holdId: Int, score: Double?, durationS: Double) -> HoldScore {
        HoldScore(
            holdId: holdId,
            holdFrames: 10,
            validFrames: 10,
            holdStartMs: 0,
            holdEndMs: Int(durationS * 1000),
            holdDurationS: durationS,
            score: score,
            reason: "",
            deviation: 0,
            penalty: 0,
            groups: [:],
            values: [:],
            z: [:],
            topFaults: [],
            missingGroups: []
        )
    }

    /// An `Analysis` carrying exactly the scores and views a test gives it —
    /// the rest is an empty-but-usable clip, because the accessors read the
    /// scores and the views and nothing else.
    private func analysis(holdScores: [HoldScore], holdViews: [HoldView]) -> Analysis {
        let processed = clip([], length: 200)
        let phases = ClipPhases(
            tMs: [],
            phase: [],
            holdId: [],
            signals: PhaseSegmenter.unusableSignals(frames: 0, reason: ""),
            runs: [],
            usable: true,
            unusableReason: ""
        )
        let features = ClipFeatures(
            tMs: [],
            phase: [],
            holdId: [],
            values: [:],
            balanceZone: [],
            comComplete: [],
            valid: [],
            unusableReason: ""
        )
        return Analysis(
            processed: processed,
            phases: phases,
            features: features,
            holdScores: holdScores,
            clipScore: nil,
            holdViews: holdViews
        )
    }

    // MARK: - The constants

    /// Pinned against the Python module (chainlink #74): the app scores with
    /// the same two thresholds and the same minimum frame count.
    func testTheConstantsAreThePythonOnes() {
        // = pipeline/handstand/view.py
        XCTAssertEqual(ViewDetector.sideMax, 0.035)
        XCTAssertEqual(ViewDetector.notSideMin, 0.042)
        XCTAssertEqual(ViewDetector.minViewFrames, 15)
        XCTAssertEqual(CameraView.side.rawValue, "side")
        XCTAssertEqual(CameraView.notSide.rawValue, "not_side")
        XCTAssertEqual(CameraView.unsure.rawValue, "unsure")
    }

    // MARK: - viewWidths

    /// The frame's `view_width` is the mean of the two pair widths over the
    /// body length: (10 + 6) / 2 / 200 = 0.04.
    func testViewWidthIsTheMeanOfBothPairWidthsOverBodyLength() {
        let processed = clip([frame(shoulderWidth: 10, hipWidth: 6)], length: 200)

        let widths = ViewDetector.viewWidths(processed)

        XCTAssertEqual(widths.count, 1)
        XCTAssertEqual(widths[0], 0.04, accuracy: 1e-12)
    }

    /// One pair valid is one measurement: 10 / 200 = 0.05, with the hip joints
    /// present in the frame but not in `valid`.
    func testViewWidthUsesTheOnePairThatIsValid() {
        let processed = clip(
            [frame(shoulderWidth: 10, hipWidth: 6, hipValid: false)], length: 200)

        let widths = ViewDetector.viewWidths(processed)

        XCTAssertEqual(widths[0], 0.05, accuracy: 1e-12)
    }

    /// Neither pair valid is no `view_width` at all.
    func testViewWidthIsNaNWhenNeitherPairIsValid() {
        let processed = clip(
            [frame(
                shoulderWidth: 10, hipWidth: 6,
                shoulderValid: false, hipValid: false)], length: 200)

        XCTAssertTrue(ViewDetector.viewWidths(processed)[0].isNaN)
    }

    /// No body length is no ratio, however wide the shoulders are.
    func testViewWidthIsNaNWithoutABodyLength() {
        let processed = clip([frame(shoulderWidth: 10, hipWidth: 6)], length: nil)

        XCTAssertTrue(ViewDetector.viewWidths(processed)[0].isNaN)
    }

    // MARK: - classify

    /// The threshold rule, case by case: under 0.035 side, at or over 0.042
    /// not side, the band between unsure — and under 15 finite frames always
    /// unsure.
    func testClassifyThresholds() {
        func classify(_ value: Double, count: Int = 20) -> CameraView {
            ViewDetector.classify([Double](repeating: value, count: count)).view
        }

        XCTAssertEqual(classify(0.02), .side)
        XCTAssertEqual(classify(0.05), .notSide)
        XCTAssertEqual(classify(0.038), .unsure)
        XCTAssertEqual(classify(0.02, count: 10), .unsure, "under MIN_VIEW_FRAMES")
        XCTAssertEqual(classify(0.035), .unsure, "SIDE_MAX is exclusive")
        XCTAssertEqual(classify(0.042), .notSide, "NOT_SIDE_MIN is inclusive")
    }

    /// An even count takes the mean of the two middle values, as numpy does:
    /// ten 0.01s and ten 0.03s give a median of 0.02.
    func testClassifyMedianOfAnEvenCountIsTheMeanOfTheTwoMiddle() {
        let widths = [Double](repeating: 0.01, count: 10) + [Double](repeating: 0.03, count: 10)

        let (median, view) = ViewDetector.classify(widths)

        XCTAssertEqual(median, 0.02, accuracy: 1e-12)
        XCTAssertEqual(view, .side)
    }

    /// No finite value is a `NaN` median and an unsure view.
    func testClassifyWithNothingToMeasureIsUnsureAndNaN() {
        let (median, view) = ViewDetector.classify([.nan, .nan])

        XCTAssertTrue(median.isNaN)
        XCTAssertEqual(view, .unsure)
    }

    // MARK: - holdViews

    /// Two holds, one narrow (side) and one wide (notSide): one view each, in
    /// `hold_id` order, with the frame count and the hold's own median.
    func testHoldViewsClassifiesEachHold() {
        var frames: [ProcessedFrame] = []
        var holdIds: [Int] = []
        var valid: [Bool] = []
        for _ in 0..<20 {
            frames.append(frame(shoulderWidth: 4, hipWidth: 4))  // 0.02, side
            holdIds.append(0)
            valid.append(true)
        }
        for _ in 0..<20 {
            frames.append(frame(shoulderWidth: 20, hipWidth: 20))  // 0.1, notSide
            holdIds.append(1)
            valid.append(true)
        }
        let processed = clip(frames, length: 200)

        let views = ViewDetector.holdViews(
            processed: processed, features: features(holdIds: holdIds, valid: valid))

        XCTAssertEqual(views.map(\.holdId), [0, 1])
        XCTAssertEqual(views[0].view, .side)
        XCTAssertEqual(views[0].viewFrames, 20)
        XCTAssertEqual(views[0].viewWidth, 0.02, accuracy: 1e-12)
        XCTAssertEqual(views[1].view, .notSide)
        XCTAssertEqual(views[1].viewFrames, 20)
        XCTAssertEqual(views[1].viewWidth, 0.1, accuracy: 1e-12)
    }

    /// Frames the features table marks invalid are not measured: a hold of ten
    /// narrow valid frames and ten wide invalid ones reads 0.02 over ten
    /// frames — unsure, because ten is under `minViewFrames` — not the 0.11
    /// the invalid frames would drag it to.
    func testHoldViewsIgnoresInvalidFrames() {
        var frames: [ProcessedFrame] = []
        var holdIds: [Int] = []
        var valid: [Bool] = []
        for index in 0..<20 {
            let wide = index >= 10
            frames.append(frame(shoulderWidth: wide ? 40 : 4, hipWidth: wide ? 40 : 4))
            holdIds.append(0)
            valid.append(!wide)  // only the narrow, valid ten count
        }
        let processed = clip(frames, length: 200)

        let views = ViewDetector.holdViews(
            processed: processed, features: features(holdIds: holdIds, valid: valid))

        XCTAssertEqual(views.count, 1)
        XCTAssertEqual(views[0].viewFrames, 10)
        XCTAssertEqual(views[0].viewWidth, 0.02, accuracy: 1e-12)
        XCTAssertEqual(views[0].view, .unsure)
    }

    /// A frames table the processed clip does not line up with is not evidence
    /// of anything: `[]`, not a crash.
    func testHoldViewsReturnsEmptyOnAMismatchedLength() {
        let processed = clip([frame(shoulderWidth: 4, hipWidth: 4)], length: 200)
        let mismatched = features(holdIds: [0, 0], valid: [true, true])

        XCTAssertTrue(
            ViewDetector.holdViews(processed: processed, features: mismatched).isEmpty)
    }

    // MARK: - sideClipScore and notSideOnly

    /// The longest scored hold is a notSide one, the shorter is a side one:
    /// the clip's side score is the shorter hold, and it is not a notSide-only
    /// clip.
    func testSideClipScoreIsTheLongestScoredSideHold() {
        let long = holdScore(holdId: 0, score: 80, durationS: 5.0)
        let short = holdScore(holdId: 1, score: 60, durationS: 2.0)
        let views = [
            HoldView(holdId: 0, viewFrames: 20, viewWidth: 0.05, view: .notSide),
            HoldView(holdId: 1, viewFrames: 20, viewWidth: 0.02, view: .side),
        ]
        let analysis = analysis(holdScores: [long, short], holdViews: views)

        XCTAssertEqual(analysis.sideClipScore?.holdId, 1)
        XCTAssertFalse(analysis.notSideOnly)
        XCTAssertEqual(analysis.view(ofHold: 0), .notSide)
        XCTAssertEqual(analysis.view(ofHold: 1), .side)
        XCTAssertNil(analysis.view(ofHold: 99))
    }

    /// Every scored hold is notSide: there is no side score at all, and the
    /// clip is one the app must refuse to score.
    func testNotSideOnlyWhenEveryScoredHoldIsNotSide() {
        let first = holdScore(holdId: 0, score: 80, durationS: 5.0)
        let second = holdScore(holdId: 1, score: 60, durationS: 2.0)
        let views = [
            HoldView(holdId: 0, viewFrames: 20, viewWidth: 0.05, view: .notSide),
            HoldView(holdId: 1, viewFrames: 20, viewWidth: 0.06, view: .notSide),
        ]
        let analysis = analysis(holdScores: [first, second], holdViews: views)

        XCTAssertNil(analysis.sideClipScore)
        XCTAssertTrue(analysis.notSideOnly)
    }

    /// Nothing scored is not a camera verdict: no score, and not
    /// notSide-only.
    func testNoScoresIsNotNotSideOnly() {
        let unscored = holdScore(holdId: 0, score: nil, durationS: 5.0)
        let views = [HoldView(holdId: 0, viewFrames: 20, viewWidth: 0.05, view: .notSide)]
        let withUnscored = analysis(holdScores: [unscored], holdViews: views)

        XCTAssertFalse(withUnscored.notSideOnly)
        XCTAssertNil(withUnscored.sideClipScore)

        let empty = analysis(holdScores: [], holdViews: [])
        XCTAssertFalse(empty.notSideOnly)
        XCTAssertNil(empty.sideClipScore)
    }

    /// A hold the clip carries no view for counts as allowed (the pre-#102
    /// `Analysis`), so its score is still the side score.
    func testAHoldWithoutADetectedViewCountsAsAllowed() {
        let hold = holdScore(holdId: 0, score: 80, durationS: 5.0)
        let analysis = analysis(holdScores: [hold], holdViews: [])

        XCTAssertEqual(analysis.sideClipScore?.holdId, 0)
        XCTAssertFalse(analysis.notSideOnly)
    }
}
