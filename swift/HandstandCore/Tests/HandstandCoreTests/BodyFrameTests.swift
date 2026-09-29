import XCTest

@testable import HandstandCore

/// The Swift mirror of the body-frame cases in
/// `pipeline/tests/test_postprocess.py` (the `handstand.bodyframe` section):
/// same numbers, same expectations.
final class BodyFrameTests: XCTestCase {
    private func keypoint(_ x: Double, _ y: Double, visibility: Double = 1.0) -> Keypoint {
        Keypoint(x: x, y: y, visibility: visibility)
    }

    private func assertUV(
        _ actual: (u: Double, v: Double),
        u: Double,
        v: Double,
        accuracy: Double = 1e-12,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.u, u, accuracy: accuracy, "u", file: file, line: line)
        XCTAssertEqual(actual.v, v, accuracy: accuracy, "v", file: file, line: line)
    }

    func testToBodyFramePutsTheWristMidpointAtTheOrigin() throws {
        let at = try BodyFrame.toBodyFrame(x: 120, y: 300, wristMidX: 120, wristMidY: 300, bodyLength: 200)
        assertUV(at, u: 0, v: 0, accuracy: 0)
    }

    func testToBodyFrameVIsPositiveAboveTheHands() throws {
        // A handstand is upside down, so "up" has to be flipped to mean
        // "away from the floor".
        let above = try BodyFrame.toBodyFrame(x: 120, y: 200, wristMidX: 120, wristMidY: 300, bodyLength: 200)
        let below = try BodyFrame.toBodyFrame(x: 120, y: 400, wristMidX: 120, wristMidY: 300, bodyLength: 200)
        XCTAssertGreaterThan(above.v, 0)
        XCTAssertLessThan(below.v, 0)
        assertUV(above, u: 0, v: 0.5)
        assertUV(below, u: 0, v: -0.5)
    }

    func testToBodyFrameUIsPositiveToTheRight() throws {
        let right = try BodyFrame.toBodyFrame(x: 220, y: 300, wristMidX: 120, wristMidY: 300, bodyLength: 200)
        let left = try BodyFrame.toBodyFrame(x: 20, y: 300, wristMidX: 120, wristMidY: 300, bodyLength: 200)
        assertUV(right, u: 0.5, v: 0)
        assertUV(left, u: -0.5, v: 0)
    }

    func testToBodyFrameIsOneBodyLengthUp() throws {
        let up = try BodyFrame.toBodyFrame(x: 120, y: 100, wristMidX: 120, wristMidY: 300, bodyLength: 200)
        assertUV(up, u: 0, v: 1)
    }

    func testBodyFramePointsMatchesTheScalarCalls() throws {
        // The Python array case, frame by frame.
        let frameA = [keypoint(120, 300), keypoint(320, 200)]
        let frameB = [keypoint(120, 300), keypoint(120, 100)]

        let a = try BodyFrame.bodyFramePoints(frameA, wristMid: (120, 300), bodyLength: 200)
        XCTAssertEqual(a.count, 2)
        assertUV(a[0], u: 0, v: 0, accuracy: 0)
        assertUV(a[1], u: 1, v: 0.5)

        let b = try BodyFrame.bodyFramePoints(frameB, wristMid: (120, 300), bodyLength: 200)
        assertUV(b[0], u: 0, v: 0, accuracy: 0)
        assertUV(b[1], u: 0, v: 1)

        // ...and each entry equals the scalar conversion of that point.
        for (points, batch) in [(frameA, a), (frameB, b)] {
            for (point, uv) in zip(points, batch) {
                let scalar = try BodyFrame.toBodyFrame(
                    x: point.x,
                    y: point.y,
                    wristMidX: 120,
                    wristMidY: 300,
                    bodyLength: 200
                )
                assertUV(uv, u: scalar.u, v: scalar.v, accuracy: 0)
            }
        }
    }

    func testToBodyFrameRejectsAUselessScale() {
        for bad in [0.0, -1.0, Double.nan, Double.infinity] {
            XCTAssertThrowsError(
                try BodyFrame.toBodyFrame(x: 1, y: 1, wristMidX: 0, wristMidY: 0, bodyLength: bad),
                "bodyLength \(bad) should throw"
            ) { error in
                guard let bodyError = error as? BodyFrame.BodyFrameError else {
                    return XCTFail("expected invalidBodyLength, got \(error)")
                }
                guard case .invalidBodyLength = bodyError else {
                    return XCTFail("expected invalidBodyLength, got \(bodyError)")
                }
                XCTAssertTrue("\(bodyError)".contains("body_length"), "\(bodyError)")
            }
        }
        // The batch form checks the scale up front too, even with no points.
        XCTAssertThrowsError(
            try BodyFrame.bodyFramePoints([], wristMid: (0, 0), bodyLength: 0)
        ) { error in
            XCTAssertNotNil(error as? BodyFrame.BodyFrameError)
        }
    }

    func testWristMidpointIsTheMeanOfTheTwoWrists() {
        let midpoint = BodyFrame.midpoint(keypoint(10, 20, visibility: 0.4), keypoint(30, 40, visibility: 0.9))
        XCTAssertEqual(midpoint.x, 20, accuracy: 0)
        XCTAssertEqual(midpoint.y, 30, accuracy: 0)
        // A midpoint is only as visible as its least visible half.
        XCTAssertEqual(midpoint.visibility, 0.4)
    }

    func testBodyFrameErrorMatchesThePythonMessage() {
        XCTAssertEqual(
            "\(BodyFrame.BodyFrameError.invalidBodyLength(0))",
            "body_length must be a positive, finite number, got 0.0"
        )
    }
}
