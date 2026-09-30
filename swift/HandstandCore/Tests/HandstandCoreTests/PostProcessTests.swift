import XCTest

@testable import HandstandCore

/// The Swift mirror of the post-process port (chainlink #39): the golden
/// fixtures of chainlink #25 run through `PostProcess.process` and compared
/// with `expected.postprocess` at `meta.tolerances`, plus one unit test per
/// step of `pipeline/handstand/postprocess.py` on tiny hand-made inputs —
/// the same cases `pipeline/tests/test_postprocess.py` pins down, with the
/// same expectations.
///
/// Nothing here reads a real video: the fixtures are the synthetic cases
/// `pipeline/handstand/golden.py` writes, and every other input is a handful
/// of numbers built in the test itself.
final class PostProcessTests: XCTestCase {
    // MARK: - A synthetic body, as `pipeline/tests/test_postprocess.py` builds it

    /// A side-on inverted handstand body, `frames` long.
    ///
    /// `y` grows downwards from the hip midpoint at 0: the shoulders are
    /// `torso` below it (towards the hands), the knees `thigh` and the ankles
    /// `thigh + shin` above it (towards the feet), and the wrists rest on the
    /// floor `handDrop` below the shoulders. So the spans `LENGTH_PARTS`
    /// measures are exactly `torso`, `thigh` and `shin`, and the body length
    /// is their sum — which is what lets a test assert numbers instead of
    /// "it did not crash".
    private func skeleton(
        _ frames: Int = 40,
        torso: Double = 100,
        thigh: Double = 120,
        shin: Double = 120,
        handDrop: Double = 90,
        tMs: [Int]? = nil,
        visibility: Double = 0.95
    ) -> [PostProcessInputFrame] {
        let times = tMs ?? (0..<frames).map { $0 * 33 }
        precondition(times.count == frames, "t_ms must hold one timestamp per frame")
        let floorY = torso + handDrop
        return (0..<frames).map { index in
            var joints: [Joint: Keypoint] = [:]
            let pairs: [(Joint, Joint, Double)] = [
                (.leftHip, .rightHip, 0),
                (.leftShoulder, .rightShoulder, torso),
                (.leftKnee, .rightKnee, -thigh),
                (.leftAnkle, .rightAnkle, -(thigh + shin)),
                (.leftWrist, .rightWrist, floorY),
                (.leftElbow, .rightElbow, (torso + floorY) / 2.0),
                (.leftFootIndex, .rightFootIndex, -(thigh + shin) + 8),
            ]
            for (left, right, y) in pairs {
                joints[left] = Keypoint(x: -8.0, y: y, visibility: visibility)
                joints[right] = Keypoint(x: 8.0, y: y, visibility: visibility)
            }
            joints[.nose] = Keypoint(x: 0.0, y: torso + 40, visibility: visibility)
            return PostProcessInputFrame(
                tMs: times[index],
                detected: true,
                trainerContact: false,
                joints: joints
            )
        }
    }

    // MARK: - Parity against the golden fixtures of chainlink #25

    /// Every committed fixture, decoded, run through `PostProcess.process`
    /// and compared with `expected.postprocess` by the shared
    /// `GoldenComparison.postprocess`: the body length at
    /// `meta.tolerances["expected.postprocess.body_length"]`, `valid` and
    /// `filled` exactly (booleans are categories, not quantities), and the
    /// joint positions at `meta.tolerances["expected.postprocess.frames.joints"]`.
    /// The comparison itself lives in `GoldenComparison.swift`, shared with
    /// `AnalyzerTests` and `RealParityTests` (chainlink #42).
    func testEveryGoldenFixtureMatchesThePythonPostProcess() throws {
        let urls = GoldenFixtures.urls()
        XCTAssertGreaterThanOrEqual(urls.count, 5, "the five committed cases")
        for url in urls {
            let fixture = try GoldenFixtures.load(url)
            let processed = PostProcess.process(GoldenFixtures.inputFrames(from: fixture))
            for mismatch in GoldenComparison.postprocess(processed, fixture: fixture) {
                XCTFail(mismatch)
            }
        }
    }

    // MARK: - Step 1: gating

    func testGatingKeepsAConfidentSample() {
        let valid = PostProcess.gatedValid(
            x: [[1.0, 2.0]], y: [[1.0, 2.0]], visibility: [[0.9, 0.5]], frameValid: [true])
        XCTAssertEqual(valid, [[true, true]])
    }

    func testGatingDropsAFrameTheModelFoundNobodyIn() {
        let valid = PostProcess.gatedValid(
            x: [[1.0], [2.0]], y: [[1.0], [2.0]], visibility: [[0.9], [0.9]],
            frameValid: [true, false]
        )
        XCTAssertEqual(valid, [[true], [false]])
    }

    func testGatingDropsAJointBelowTheVisibilityThreshold() {
        // 0.49 is below MIN_VISIBILITY, 0.5 is not, and a NaN visibility —
        // a joint the model refused to score — fails the comparison too.
        let valid = PostProcess.gatedValid(
            x: [[1.0, 1.0, 1.0]], y: [[1.0, 1.0, 1.0]],
            visibility: [[0.49, 0.5, .nan]],
            frameValid: [true]
        )
        XCTAssertEqual(valid, [[false, true, false]])
    }

    func testGatingDropsAJointWithNoCoordinates() {
        let valid = PostProcess.gatedValid(
            x: [[1.0, .nan]], y: [[1.0, 2.0]], visibility: [[0.9, 0.9]], frameValid: [true])
        XCTAssertEqual(valid, [[true, false]])
    }

    func testALowVisibilityJointIsGatedOutOfTheWholeClip() {
        var frames = skeleton(40)
        for index in frames.indices {
            frames[index].joints[.nose] = Keypoint(x: 0.0, y: 140, visibility: 0.4)
        }
        let clip = PostProcess.process(frames)
        for (index, row) in clip.frames.enumerated() {
            XCTAssertFalse(row.valid.contains(.nose), "frame \(index): nose is below 0.5")
            XCTAssertFalse(row.filled.contains(.nose), "frame \(index): a never-seen joint is not bridged")
            XCTAssertTrue(row.valid.contains(.leftWrist), "frame \(index): the rest survives")
        }
    }

    func testAFrameTheModelFoundNobodyInIsGatedOut() {
        var frames = skeleton(40)
        for index in 10..<30 { frames[index].detected = false }
        let clip = PostProcess.process(frames)
        // 20 frames at 33 ms is ~0.66 s between the samples that bracket the
        // run — far past MAX_GAP_S, so nothing bridges it.
        for index in 10..<30 {
            XCTAssertTrue(
                clip.frames[index].valid.isEmpty, "frame \(index) has no pose but a position")
            XCTAssertTrue(
                clip.frames[index].joints.isEmpty, "frame \(index) has no pose but a position")
        }
        XCTAssertTrue(clip.frames[9].valid.contains(.leftWrist))
        XCTAssertTrue(clip.frames[30].valid.contains(.leftWrist))
    }

    func testTrainerContactFramesAreGatedOut() {
        // A contact frame is two bodies' keypoints stitched together: all of it goes.
        var frames = skeleton(40)
        for index in 10..<30 { frames[index].trainerContact = true }
        let clip = PostProcess.process(frames)
        for index in 10..<30 {
            XCTAssertTrue(
                clip.frames[index].valid.isEmpty, "frame \(index) is trainer contact but is valid")
            XCTAssertFalse(
                clip.frames[index].filled.contains(.leftWrist),
                "frame \(index) is trainer contact but is bridged")
        }
        XCTAssertTrue(clip.frames[9].valid.contains(.leftWrist))
        XCTAssertTrue(clip.frames[30].valid.contains(.leftWrist))
    }

    func testAShortTrainerContactBlipIsBridgedAndMarkedFilled() {
        // Four frames is 0.13 s of occlusion, inside MAX_GAP_S: the bridge
        // comes back, marked `filled` precisely so a later stage can refuse it.
        var frames = skeleton(40)
        for index in 10..<14 { frames[index].trainerContact = true }
        let clip = PostProcess.process(frames)
        for index in 10..<14 {
            XCTAssertEqual(
                clip.frames[index].valid, Set(Joint.allCases), "frame \(index) valid")
            XCTAssertEqual(
                clip.frames[index].filled, Set(Joint.allCases), "frame \(index) filled")
        }
        XCTAssertFalse(clip.frames[9].filled.contains(.leftWrist))
        XCTAssertFalse(clip.frames[14].filled.contains(.leftWrist))
    }

    // MARK: - Step 2: body length

    func testBodyLengthIsTheSumOfItsThreeParts() {
        let clip = PostProcess.process(skeleton(40, torso: 100, thigh: 120, shin: 90))
        let body = clip.bodyLength
        XCTAssertTrue(clip.usable)
        XCTAssertNil(body.reason)
        XCTAssertEqual(body.torsoPx ?? .nan, 100, accuracy: 1e-9, "torso")
        XCTAssertEqual(body.thighPx ?? .nan, 120, accuracy: 1e-9, "thigh")
        XCTAssertEqual(body.shinPx ?? .nan, 90, accuracy: 1e-9, "shin")
        XCTAssertEqual(body.totalPx ?? .nan, 310, accuracy: 1e-9, "total")
        XCTAssertEqual(
            body.frames, ["torso": 40, "thigh": 40, "shin": 40], "measurable on every frame")
    }

    func testThePercentileInterpolatesLikeNumpyLinear() {
        // numpy's default "linear" method: index (n - 1) * p / 100, lerped.
        let between = PostProcess.percentileOf([0.0, 10.0], p: 90)
        XCTAssertEqual(between.value, 9, accuracy: 1e-12, "0.9 of the way from 0 to 10")
        XCTAssertEqual(between.count, 2)

        let exact = PostProcess.percentileOf((0...10).map { Double($0) * 10 }, p: 90)
        XCTAssertEqual(exact.value, 90, accuracy: 1e-12, "(11 - 1) * 0.9 = 9, sample 9")
        XCTAssertEqual(exact.count, 11)

        let median = PostProcess.percentileOf([1.0, 3.0], p: 50)
        XCTAssertEqual(median.value, 2, accuracy: 1e-12, "the midpoint is between the samples")
        XCTAssertEqual(median.count, 2)

        // NaN samples are dropped, not counted…
        let withNaN = PostProcess.percentileOf([1.0, .nan, 3.0], p: 90)
        XCTAssertEqual(withNaN.value, 2.8, accuracy: 1e-12, "index 0.9 of two samples")
        XCTAssertEqual(withNaN.count, 2)

        // …and with nothing finite there is no percentile at all.
        let empty = PostProcess.percentileOf([.nan], p: 90)
        XCTAssertTrue(empty.value.isNaN, "no finite sample means no value")
        XCTAssertEqual(empty.count, 0)
    }

    func testFewerThanTenFramesIsUnusableAndEveryJointIsInvalid() {
        let clip = PostProcess.process(skeleton(5))
        XCTAssertFalse(clip.usable, "MIN_BODY_FRAMES is 10")
        XCTAssertEqual(
            clip.bodyLength.reason, "torso was measurable on 5 frame(s), need 10",
            "the reason Python records")
        XCTAssertEqual(
            clip.bodyLength.frames, ["torso": 5, "thigh": 5, "shin": 5],
            "an unusable clip still records what it could have measured on")
        XCTAssertNil(clip.bodyLength.totalPx)
        XCTAssertEqual(clip.frames.count, 5, "one row per input frame")
        for (index, row) in clip.frames.enumerated() {
            XCTAssertTrue(row.joints.isEmpty, "frame \(index) carries a position")
            XCTAssertTrue(row.valid.isEmpty, "frame \(index) is valid")
            XCTAssertTrue(row.filled.isEmpty, "frame \(index) is filled")
        }
    }

    // MARK: - Step 3: outliers

    func testAJointThatJumpsIsInvalidInThatFrameOnly() {
        let frames = 10
        let t = (0..<frames).map { Double($0) / 30.0 }
        // 500 px in a body length of 100 px, in 1/30 s: 150 L/s, not a travel.
        let x = (0..<frames).map { index -> [Double] in [index == 4 ? 500.0 : 0.0] }
        let y = [[Double]](repeating: [0.0], count: frames)
        let valid = [[Bool]](repeating: [true], count: frames)
        let kept = PostProcess.removeSpeedOutliers(
            tSeconds: t, x: x, y: y, valid: valid, bodyLength: 100)

        XCTAssertFalse(kept[4][0], "the jump is not a measurement")
        XCTAssertEqual(kept.filter { $0[0] }.count, 9, "only the spike is dropped")
        // The earlier sample stays the reference, so the joint is not dragged
        // invalid along with the spike.
        XCTAssertTrue(
            kept[5...].allSatisfy { $0[0] }, "the frames after the spike keep their positions")
    }

    func testTheSpeedLimitIsMeasuredInBodyLengths() {
        // The same 50 px jump is 60 L/s for a 25 px body and 1.5 L/s for a 1000 px one.
        let frames = 30
        let t = (0..<frames).map { Double($0) / 30.0 }
        var values = [Double](repeating: 0.0, count: frames)
        values[10] = 50.0
        let x = values.map { [$0] }
        let y = [[Double]](repeating: [0.0], count: frames)
        let valid = [[Bool]](repeating: [true], count: frames)
        let small = PostProcess.removeSpeedOutliers(
            tSeconds: t, x: x, y: y, valid: valid, bodyLength: 25)
        let large = PostProcess.removeSpeedOutliers(
            tSeconds: t, x: x, y: y, valid: valid, bodyLength: 1000)
        XCTAssertFalse(small[10][0], "60 L/s is a teleport")
        XCTAssertTrue(large[10][0], "1.5 L/s is a movement")
    }

    func testTheSpeedLimitUsesTheRealFrameTimes() {
        // Variable frame rate: the same 40 px step is a jump at 30 fps and a
        // drift at 5 fps, because the limit is per second, not per frame.
        let x = [[0.0], [40.0], [0.0], [0.0]]
        let y = [[0.0], [0.0], [0.0], [0.0]]
        let valid = [[Bool]](repeating: [true], count: 4)
        let fast = PostProcess.removeSpeedOutliers(
            tSeconds: [0.0, 0.033, 0.066, 0.099], x: x, y: y, valid: valid, bodyLength: 100)
        let slow = PostProcess.removeSpeedOutliers(
            tSeconds: [0.0, 0.2, 0.4, 0.6], x: x, y: y, valid: valid, bodyLength: 100)
        XCTAssertFalse(fast[1][0], "12 L/s at 30 fps")
        XCTAssertTrue(slow[1][0], "2 L/s at 5 fps")
    }

    func testADuplicateTimestampIsNotEvidenceOfAJump() {
        let x = [[0.0], [90.0], [0.0]]
        let y = [[0.0], [0.0], [0.0]]
        let valid = [[Bool]](repeating: [true], count: 3)
        let kept = PostProcess.removeSpeedOutliers(
            tSeconds: [0.0, 0.0, 0.1], x: x, y: y, valid: valid, bodyLength: 100)
        XCTAssertEqual(kept[1], [true], "there is no speed to measure across a duplicate time")
    }

    func testAJumpInTheDataIsRemovedAndBridgedEndToEnd() throws {
        var frames = skeleton(40)
        let wrist = try XCTUnwrap(frames[20].joints[.leftWrist])
        frames[20].joints[.leftWrist] = Keypoint(
            x: wrist.x + 400, y: wrist.y, visibility: wrist.visibility)
        let clip = PostProcess.process(frames)

        // The model really did report a position on frame 20 — it is just not
        // a travel, so the outlier is dropped and the one-frame hole bridged.
        XCTAssertTrue(clip.frames[20].valid.contains(.leftWrist), "the frame keeps a position")
        XCTAssertTrue(clip.frames[20].filled.contains(.leftWrist), "…as a bridge, not a measurement")
        XCTAssertTrue(clip.frames[19].valid.contains(.leftWrist))
        XCTAssertFalse(clip.frames[19].filled.contains(.leftWrist), "the neighbours are measured")
        XCTAssertFalse(clip.frames[21].filled.contains(.leftWrist))
    }

    // MARK: - Step 4: gap fill

    func testAShortGapIsInterpolatedAndMarkedFilled() {
        // 0.1 s between the two samples that bracket the gap: inside MAX_GAP_S.
        let result = PostProcess.fillGaps(
            tSeconds: [0.0, 0.1, 0.2],
            x: [[0.0], [0.0], [200.0]],
            y: [[0.0], [0.0], [0.0]],
            valid: [[true], [false], [true]]
        )
        XCTAssertEqual(result.valid[1], [true], "the bridge is a position")
        XCTAssertEqual(result.filled[1], [true], "…marked as a bridge")
        XCTAssertEqual(result.x[1][0], 100, accuracy: 1e-12, "half way in time")
        XCTAssertEqual(result.filled[0], [false], "the measurements stay measurements")
        XCTAssertEqual(result.filled[2], [false], "the measurements stay measurements")
    }

    func testALongGapIsLeftInvalid() {
        // 0.3 s between the two samples that bracket the run: past MAX_GAP_S,
        // where there is no honest straight line to draw.
        let result = PostProcess.fillGaps(
            tSeconds: [0.0, 0.05, 0.1, 0.2, 0.3],
            x: [[0.0], [1.0], [2.0], [3.0], [300.0]],
            y: [[0.0], [0.0], [0.0], [0.0], [0.0]],
            valid: [[true], [false], [false], [false], [true]]
        )
        XCTAssertEqual(
            result.valid.map { $0[0] }, [true, false, false, false, true], "no bridge appears")
        XCTAssertEqual(
            result.filled.map { $0[0] }, [false, false, false, false, false], "nothing is filled")
        XCTAssertTrue(result.x[1][0].isNaN, "an unfilled sample has no position")
        XCTAssertTrue(result.x[2][0].isNaN, "an unfilled sample has no position")
        XCTAssertTrue(result.x[3][0].isNaN, "an unfilled sample has no position")
    }

    func testTheGapBoundaryIsTheConstant() {
        // Exactly MAX_GAP_S between the two samples is filled…
        let at = PostProcess.fillGaps(
            tSeconds: [0.0, 0.1, 0.2],
            x: [[0.0], [0.0], [200.0]],
            y: [[0.0], [0.0], [0.0]],
            valid: [[true], [false], [true]]
        )
        XCTAssertEqual(at.filled[1], [true])
        // …one millisecond more is not.
        let past = PostProcess.fillGaps(
            tSeconds: [0.0, 0.1, 0.201],
            x: [[0.0], [0.0], [200.0]],
            y: [[0.0], [0.0], [0.0]],
            valid: [[true], [false], [true]]
        )
        XCTAssertEqual(past.valid[1], [false])
        XCTAssertEqual(past.filled[1], [false])
        XCTAssertTrue(past.x[1][0].isNaN)
    }

    func testFillGapsInterpolatesInTimeNotInFrames() {
        // Uneven t_ms — 0, 30, 60, 100, 130, 200 ms. The middle sample sits
        // 100 ms in, so it is half the gap in time even though it is the
        // fourth of five missing frames.
        let result = PostProcess.fillGaps(
            tSeconds: [0.0, 0.03, 0.06, 0.1, 0.13, 0.2],
            x: [[0.0], [0.0], [0.0], [0.0], [0.0], [200.0]],
            y: [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            valid: [[true], [false], [false], [false], [false], [true]]
        )
        XCTAssertEqual(result.x[3][0], 100, accuracy: 1e-9, "100 ms in, 100 ms out")
        XCTAssertEqual(result.x[1][0], 30, accuracy: 1e-9, "30 ms in, 170 ms out")
        XCTAssertEqual(
            result.filled.map { $0[0] }, [false, true, true, true, true, false])
    }

    func testATrackDoesNotStartOrEndWithAnInvention() {
        // Nothing to interpolate between means nothing is written: the head
        // and tail of a track have no far side to lean on.
        let result = PostProcess.fillGaps(
            tSeconds: [0.0, 0.033, 0.066, 0.099],
            x: [[5.0], [6.0], [7.0], [8.0]],
            y: [[5.0], [6.0], [7.0], [8.0]],
            valid: [[false], [false], [true], [true]]
        )
        XCTAssertEqual(result.valid.map { $0[0] }, [false, false, true, true])
        XCTAssertEqual(
            result.filled.map { $0[0] }, [false, false, false, false], "no invented frames")
        XCTAssertTrue(result.x[0][0].isNaN, "a track does not start with a guess")
        XCTAssertTrue(result.x[1][0].isNaN, "a track does not start with a guess")
    }

    func testUnevenFrameTimingIsHandledEndToEnd() throws {
        // Deliberately irregular clock: 0, 33, 66, 99, 249, 282, 315, 348,
        // 648, 681, 714, 747 ms, with the nose unseen on frames 4…6 (a
        // 249 ms hole, past the limit) and on frames 9…10 (99 ms, inside it).
        // Every threshold is in seconds, so frame counts never decide.
        let tMs = [0, 33, 66, 99, 249, 282, 315, 348, 648, 681, 714, 747]
        var frames = skeleton(12, tMs: tMs)
        for index in [4, 5, 6, 9, 10] {
            frames[index].joints[.nose] = Keypoint(x: 0.0, y: 140, visibility: 0.4)
        }
        let clip = PostProcess.process(frames)

        // The long hole stays a hole…
        for index in [4, 5, 6] {
            XCTAssertFalse(
                clip.frames[index].valid.contains(.nose), "frame \(index): a 249 ms gap is not bridged")
            XCTAssertFalse(
                clip.frames[index].filled.contains(.nose), "frame \(index): a 249 ms gap is not filled")
        }
        // …the short one is bridged, at the interpolated position.
        for index in [9, 10] {
            XCTAssertTrue(
                clip.frames[index].valid.contains(.nose), "frame \(index): a 99 ms gap is bridged")
            XCTAssertTrue(
                clip.frames[index].filled.contains(.nose), "frame \(index): a 99 ms gap is filled")
            let point = try XCTUnwrap(clip.frames[index].joints[.nose])
            XCTAssertEqual(point.y, 140, accuracy: 1e-6, "frame \(index): between two equal ends")
        }
        // The other joints never saw a gap.
        for (index, row) in clip.frames.enumerated() {
            XCTAssertTrue(row.valid.contains(.leftWrist), "frame \(index): the rest is untouched")
        }
    }

    // MARK: - Step 5: One-Euro smoothing

    func testTheFirstSamplePassesThroughAndAResetStartsAgain() {
        var filter = OneEuroFilter()
        XCTAssertEqual(filter(4.0, at: 0.0), 4.0, accuracy: 0, "the first sample seeds the filter")
        let smoothed = filter(9.0, at: 0.1)
        XCTAssertGreaterThan(smoothed, 4.0, "it moves towards the new sample")
        XCTAssertLessThan(smoothed, 9.0, "without jumping to it")
        filter.reset()
        XCTAssertEqual(filter(-1.0, at: 0.2), -1.0, accuracy: 0, "a reset forgets everything")
    }

    func testTheOneEuroFilterOnAConstantSignalReturnsThatConstant() {
        var filter = OneEuroFilter()
        for index in 0..<60 {
            let value = filter(250.0, at: Double(index) / 30.0)
            XCTAssertEqual(value, 250.0, accuracy: 1e-9, "sample \(index) must not drift")
        }
    }

    func testSmoothTrackKeepsAStillJointWhereItIs() {
        // Smoothing must not shrink a body: a constant track stays constant.
        let frames = 60
        let t = (0..<frames).map { Double($0) / 30.0 }
        let x = [[Double]](repeating: [250.0], count: frames)
        let valid = [[Bool]](repeating: [true], count: frames)
        let smooth = PostProcess.smoothTrack(
            tSeconds: t, x: x, y: x, valid: valid, bodyLength: 100)
        for index in 0..<frames {
            XCTAssertEqual(smooth.x[index][0], 250, accuracy: 1e-6, "x at \(index)")
            XCTAssertEqual(smooth.y[index][0], 250, accuracy: 1e-6, "y at \(index)")
        }
    }

    func testSmoothTrackRestartsAfterAGap() {
        // Across a gap the filter must not remember: the first sample back is
        // itself, not dragged towards the run that came before.
        let frames = 20
        let t = (0..<frames).map { Double($0) / 30.0 }
        var x = [[Double]](repeating: [0.0], count: frames)
        for index in 8..<frames { x[index] = [5.0] }
        var valid = [[Bool]](repeating: [true], count: frames)
        valid[8] = [false]
        valid[9] = [false]
        let smooth = PostProcess.smoothTrack(
            tSeconds: t, x: x, y: x, valid: valid, bodyLength: 100)

        XCTAssertTrue(smooth.x[8][0].isNaN, "an invalid sample has no position")
        XCTAssertTrue(smooth.x[9][0].isNaN, "an invalid sample has no position")
        XCTAssertEqual(smooth.x[10][0], 5.0, accuracy: 1e-9, "the first sample back is itself")
    }

    // MARK: - The constants

    func testTheConstantsAreThePythonOnes() {
        let config = PostProcessConfig()
        XCTAssertEqual(config.minVisibility, 0.5)  // MIN_VISIBILITY
        XCTAssertEqual(config.bodyLengthPercentile, 90.0)  // BODY_LENGTH_PERCENTILE
        XCTAssertEqual(config.minBodyFrames, 10)  // MIN_BODY_FRAMES
        XCTAssertEqual(config.maxSpeedLPerS, 8.0)  // MAX_SPEED_L_PER_S
        XCTAssertEqual(config.maxGapS, 0.2)  // MAX_GAP_S
        XCTAssertEqual(config.minCutoff, 1.0)  // MIN_CUTOFF
        XCTAssertEqual(config.beta, 0.3)  // BETA
        XCTAssertEqual(config.dCutoff, 1.0)  // D_CUTOFF
        XCTAssertEqual(config.minSampleDt, 1e-3)  // MIN_SAMPLE_DT

        // The helpers default to the same numbers, so a test that calls one
        // directly runs Python's configuration.
        XCTAssertEqual(PostProcessConstants.minVisibility, config.minVisibility)
        XCTAssertEqual(PostProcessConstants.bodyLengthPercentile, config.bodyLengthPercentile)
        XCTAssertEqual(PostProcessConstants.minBodyFrames, config.minBodyFrames)
        XCTAssertEqual(PostProcessConstants.maxSpeedLPerS, config.maxSpeedLPerS)
        XCTAssertEqual(PostProcessConstants.maxGapS, config.maxGapS)
        XCTAssertEqual(PostProcessConstants.minCutoff, config.minCutoff)
        XCTAssertEqual(PostProcessConstants.beta, config.beta)
        XCTAssertEqual(PostProcessConstants.dCutoff, config.dCutoff)
        XCTAssertEqual(PostProcessConstants.minSampleDt, config.minSampleDt)
        XCTAssertEqual(
            PostProcessConstants.minBodyLengthPixels, 1.0,
            "handstand.athlete.MIN_BODY_LENGTH_PIXELS")

        // The geometry the body length is built from, spelled as Python spells it.
        XCTAssertEqual(
            PostProcessConstants.torsoEnds.flatMap { [$0.0.rawValue, $0.1.rawValue] },
            ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            "TORSO_ENDS"
        )
        XCTAssertEqual(
            PostProcessConstants.legSegments.map(\.name), ["thigh", "shin"], "LEG_SEGMENTS")
        XCTAssertEqual(PostProcessConstants.legSides, ["left", "right"], "LEG_SIDES")
        XCTAssertEqual(
            PostProcessConstants.lengthParts, ["torso", "thigh", "shin"], "LENGTH_PARTS")
    }
}
