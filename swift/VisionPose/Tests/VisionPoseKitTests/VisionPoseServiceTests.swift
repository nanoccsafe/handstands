// `VisionPoseService` — the per-frame pipeline the macOS runner and the iOS
// app share (chainlink #82).
//
// Every test drives the service through a `FakeDetector` with canned points
// and a blank pixel buffer: what is under test is the *order* of the steps
// (rotate? detect, y flip, map-back, choose, update, convert) and the values
// they produce, not Vision's accuracy.

import CoreVideo
import XCTest

import HandstandCore
import VisionPoseCore
import VisionPoseKit

final class VisionPoseServiceTests: XCTestCase {
    // MARK: - Fixtures

    private func size(of buffer: CVPixelBuffer) -> PixelSize {
        PixelSize(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
    }

    /// Wrists high in the display frame, ankles low — a standing person, not
    /// inverted. (Normalised y is measured from the *bottom*.)
    private func standing(confidence: Double = 0.9) -> FakeDetector.Person {
        person(
            [
                .nose: (x: 0.5, y: 0.8),
                .leftWrist: (x: 0.4, y: 0.9),
                .rightWrist: (x: 0.6, y: 0.9),
                .leftAnkle: (x: 0.45, y: 0.1),
                .rightAnkle: (x: 0.55, y: 0.1),
            ],
            confidence: confidence
        )
    }

    /// Wrists below the ankles in the display frame: a handstand, which the
    /// `--rotate auto` rule reads as "rotate the next frame".
    private func handstand() -> FakeDetector.Person {
        person([
            .nose: (x: 0.5, y: 0.3),
            .leftWrist: (x: 0.4, y: 0.1),
            .rightWrist: (x: 0.6, y: 0.1),
            .leftAnkle: (x: 0.45, y: 0.9),
            .rightAnkle: (x: 0.55, y: 0.9),
        ])
    }

    // MARK: - Identity

    func testBackendNameIsVision() {
        let service: PoseService = VisionPoseService()
        XCTAssertEqual(service.backendName, "vision")
    }

    // MARK: - The y flip

    /// One upright person: Vision's bottom-left normalised coordinates become
    /// the pipeline's top-left pixels — `y = (1 - y) * (height - 1)`.
    func testAnUprightPersonMapsThroughTheYFlip() throws {
        let buffer = try makePixelBuffer(width: 64, height: 48)
        let frameSize = size(of: buffer)
        let detector = FakeDetector(frames: [
            [person([.nose: (x: 0.5, y: 0.25)], confidence: 0.75)],
        ])
        let service = VisionPoseService(rotate: .none, detector: detector)

        let frame = try service.process(buffer, tMs: 42)

        let expected = try CoordinateMath.normalizedToPixels(
            NormalizedPoint(x: 0.5, y: 0.25),
            in: frameSize
        )
        let nose = try XCTUnwrap(frame.joints[.nose])
        XCTAssertEqual(nose.x, expected.x, accuracy: 1e-9)
        XCTAssertEqual(nose.y, expected.y, accuracy: 1e-9)
        // The flip, spelled out: 0.25 down from the *bottom* is 0.75 down
        // from the top.
        XCTAssertEqual(nose.x, 0.5 * Double(frameSize.width - 1), accuracy: 1e-9)
        XCTAssertEqual(nose.y, 0.75 * Double(frameSize.height - 1), accuracy: 1e-9)
        // Vision's confidence is the shared schema's visibility.
        XCTAssertEqual(nose.visibility, 0.75, accuracy: 1e-12)
        // The frame was handed over upright, as `rotate: .none` promises.
        XCTAssertEqual(detector.rotatedCalls, [false])
        XCTAssertTrue(frame.detected)
    }

    // MARK: - The half turn

    /// `.halfTurn`: Vision sees the frame upside down, so its points come back
    /// through `mapBackHalfTurn` — checked against the exact maths the runner
    /// used to do inline, with the same `CoordinateMath` functions.
    func testARotatedFrameMapsBackThroughTheHalfTurn() throws {
        let buffer = try makePixelBuffer(width: 64, height: 48)
        let frameSize = size(of: buffer)
        let raw: [VisionJoint: (x: Double, y: Double)] = [
            .leftWrist: (x: 0.2, y: 0.75),
            .rightAnkle: (x: 0.9, y: 0.1),
        ]
        let detector = FakeDetector(frames: [[person(raw)]])
        let service = VisionPoseService(rotate: .halfTurn, detector: detector)

        let result = try service.processAll(buffer, tMs: 0)

        XCTAssertTrue(result.rotated)
        XCTAssertEqual(detector.rotatedCalls, [true])
        let mapped = try XCTUnwrap(result.people.first)
        for (joint, point) in raw {
            let flipped = try CoordinateMath.normalizedToPixels(
                NormalizedPoint(x: point.x, y: point.y),
                in: frameSize
            )
            let expected = try CoordinateMath.mapBackHalfTurn(flipped, in: frameSize)
            let found = try XCTUnwrap(mapped.points[joint])
            XCTAssertEqual(found.point.x, expected.x, accuracy: 1e-9)
            XCTAssertEqual(found.point.y, expected.y, accuracy: 1e-9)
            // ...and the map-back really happened: the result is not merely
            // the y flip (it would only coincide at the frame's centre).
            XCTAssertEqual(found.point.x, Double(frameSize.width - 1) - flipped.x, accuracy: 1e-9)
        }
    }

    // MARK: - The auto rule

    /// The `.auto` sequence: never rotated on the first frame, rotated after a
    /// handstand, unchanged by a frame with nobody in it, back to square one
    /// after `reset()`.
    func testAutoRotationFollowsThePreviousFrameAndResetStartsOver() throws {
        let buffer = try makePixelBuffer()
        let detector = FakeDetector(frames: [
            [standing()],  // frame 1: judged after it is seen
            [handstand()],  // frame 2: judged from frame 1
            [],  // frame 3: nobody — says nothing about the orientation
            [handstand()],  // frame 4: judged from frame 3, i.e. not at all
        ])
        let service = VisionPoseService(rotate: .auto, detector: detector)

        let frame1 = try service.processAll(buffer, tMs: 0)
        let frame2 = try service.processAll(buffer, tMs: 1)
        let frame3 = try service.processAll(buffer, tMs: 2)
        let frame4 = try service.processAll(buffer, tMs: 3)

        // The first frame of a clip is never rotated.
        XCTAssertFalse(frame1.rotated)
        // Frame 1 was a standing body, so frame 2 runs upright too...
        XCTAssertFalse(frame2.rotated)
        // ...but frame 2 was a handstand, so frame 3 is handed over upside
        // down...
        XCTAssertTrue(frame3.rotated)
        // ...and frame 3 had nobody in it, so the decision stands.
        XCTAssertTrue(frame4.rotated)

        XCTAssertEqual(detector.rotatedCalls, [false, false, true, true])

        // `reset()` — a new clip starts over, whatever the last frame said.
        service.reset()
        let next = try service.processAll(buffer, tMs: 4)
        XCTAssertFalse(next.rotated)
        XCTAssertEqual(detector.rotatedCalls, [false, false, true, true, false])
    }

    /// `.none` never consults the state, however upside down the bodies are.
    func testNoneNeverRotates() throws {
        let buffer = try makePixelBuffer()
        let detector = FakeDetector(frames: [[handstand()], [], [handstand()]])
        let service = VisionPoseService(rotate: .none, detector: detector)

        for index in 0..<3 {
            let result = try service.processAll(buffer, tMs: index)
            XCTAssertFalse(result.rotated, "frame \(index)")
        }
        XCTAssertEqual(detector.rotatedCalls, [false, false, false])
    }

    /// `.halfTurn` rotates every frame, whatever the bodies say.
    func testHalfTurnAlwaysRotates() throws {
        let buffer = try makePixelBuffer()
        let detector = FakeDetector(frames: [[standing()], [], [standing()]])
        let service = VisionPoseService(rotate: .halfTurn, detector: detector)

        for index in 0..<3 {
            let result = try service.processAll(buffer, tMs: index)
            XCTAssertTrue(result.rotated, "frame \(index)")
        }
        XCTAssertEqual(detector.rotatedCalls, [true, true, true])
    }

    // MARK: - Choosing a person

    /// Two people: the one whose wrists sit lowest in the display frame — the
    /// one most likely to be on their hands — is the one reported, and the
    /// other is still in `people` for the runner's per-person rows.
    func testTheLowestWristPersonIsChosen() throws {
        let buffer = try makePixelBuffer(width: 64, height: 48)
        let frameSize = size(of: buffer)
        let bystander = person([
            .nose: (x: 0.1, y: 0.5),
            .leftWrist: (x: 0.05, y: 0.95),
            .rightWrist: (x: 0.15, y: 0.95),
        ])
        let athlete = person([
            .nose: (x: 0.9, y: 0.4),
            .leftWrist: (x: 0.85, y: 0.05),
            .rightWrist: (x: 0.95, y: 0.05),
        ])
        let detector = FakeDetector(frames: [[bystander, athlete]])
        let service = VisionPoseService(rotate: .none, detector: detector)

        let result = try service.processAll(buffer, tMs: 7)

        XCTAssertEqual(result.people.count, 2)
        // The bystander's wrists are high in the display frame, the athlete's
        // low — so the athlete's nose is the one that reaches the schema.
        let nose = try XCTUnwrap(result.frame.joints[.nose])
        XCTAssertEqual(nose.x, 0.9 * Double(frameSize.width - 1), accuracy: 1e-9)
        XCTAssertEqual(nose.y, 0.6 * Double(frameSize.height - 1), accuracy: 1e-9)
        // Both people keep their own mapped points, in Vision's order.
        let first = try XCTUnwrap(result.people.first)
        XCTAssertEqual(
            try XCTUnwrap(first.points[.nose]).point.x,
            0.1 * Double(frameSize.width - 1),
            accuracy: 1e-9
        )
    }

    // MARK: - The shared schema

    /// A joint Vision did not report is absent (no zero, no placeholder), and
    /// the schema's `foot_index` pair is absent even when Vision reported
    /// everything it has: Vision has no such joint.
    func testAMissingJointIsAbsentAndThereIsNoFootIndex() throws {
        let buffer = try makePixelBuffer()
        // Everything Vision reported here, including the three joints the
        // shared schema does not track (an eye, `neck`, `root`).
        let detector = FakeDetector(frames: [[
            person(
                [
                    .nose: (x: 0.5, y: 0.5),
                    .leftEye: (x: 0.45, y: 0.55),
                    .neck: (x: 0.5, y: 0.4),
                    .root: (x: 0.5, y: 0.3),
                ],
                confidence: 0.8
            ),
        ]])
        let service = VisionPoseService(rotate: .none, detector: detector)

        let result = try service.processAll(buffer, tMs: 0)

        let joints = result.frame.joints
        XCTAssertEqual(joints.count, 1)
        XCTAssertNotNil(joints[.nose])
        // Not reported by the detector -> absent.
        XCTAssertNil(joints[.leftShoulder])
        XCTAssertNil(joints[.leftAnkle])
        // Vision has no foot_index at all -> absent, whatever else was found.
        XCTAssertNil(joints[.leftFootIndex])
        XCTAssertNil(joints[.rightFootIndex])
        // The runner, though, still gets every joint Vision *did* report: the
        // schema's gap is not the detector's.
        let found = try XCTUnwrap(result.people.first)
        XCTAssertNotNil(found.points[.leftEye])
        XCTAssertNotNil(found.points[.neck])
        XCTAssertNotNil(found.points[.root])
    }

    /// Vision's confidence travels as the shared schema's `visibility`.
    func testVisibilityEqualsTheConfidence() throws {
        let buffer = try makePixelBuffer()
        let detector = FakeDetector(frames: [
            [person([.nose: (x: 0.5, y: 0.5), .leftWrist: (x: 0.4, y: 0.6)], confidence: 0.42)],
        ])
        let service = VisionPoseService(rotate: .none, detector: detector)

        let frame = try service.process(buffer, tMs: 0)

        XCTAssertEqual(try XCTUnwrap(frame.joints[.nose]).visibility, 0.42, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(frame.joints[.leftWrist]).visibility, 0.42, accuracy: 1e-12)
    }

    // MARK: - Nobody, and the frame's own facts

    func testNobodyFoundIsDetectedFalseWithEmptyJoints() throws {
        let buffer = try makePixelBuffer()
        let detector = FakeDetector(frames: [[]])
        let service = VisionPoseService(rotate: .auto, detector: detector)

        let result = try service.processAll(buffer, tMs: 99)

        XCTAssertFalse(result.frame.detected)
        XCTAssertTrue(result.frame.joints.isEmpty)
        XCTAssertTrue(result.people.isEmpty)
        XCTAssertFalse(result.rotated)
    }

    func testTMsPassesThroughAndTrainerContactIsAlwaysFalse() throws {
        let buffer = try makePixelBuffer()
        let detector = FakeDetector(frames: [[standing()], []])
        let service = VisionPoseService(rotate: .auto, detector: detector)

        let found = try service.process(buffer, tMs: 1234)
        let missed = try service.process(buffer, tMs: 5678)

        XCTAssertEqual(found.tMs, 1234)
        XCTAssertEqual(missed.tMs, 5678)
        // One athlete on the device: there is no trainer logic to set it.
        XCTAssertFalse(found.trainerContact)
        XCTAssertFalse(missed.trainerContact)
        XCTAssertTrue(found.detected)
        XCTAssertFalse(missed.detected)
    }
}
