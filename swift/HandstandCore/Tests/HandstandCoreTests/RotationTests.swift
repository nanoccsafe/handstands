import XCTest

@testable import HandstandCore

/// The Swift mirror of `pipeline/tests/test_rotation.py`: same frames, same
/// points, same expectations — the parity the port is judged by until the
/// fixtures of chainlink #25/#42 replace "same expectations" with "same data".
final class RotationTests: XCTestCase {
    /// A 4x3 frame: width 4 (x in 0..3), height 3 (y in 0..2).
    private let width = 4
    private let height = 3

    /// The Python test's POINTS: four corners and an interior point.
    private var corners: [Keypoint] {
        [
            keypoint(0, 0),  // top-left
            keypoint(3, 0),  // top-right
            keypoint(0, 2),  // bottom-left
            keypoint(3, 2),  // bottom-right
            keypoint(1, 1),  // interior
        ]
    }

    private func keypoint(_ x: Double, _ y: Double, visibility: Double = 1.0) -> Keypoint {
        Keypoint(x: x, y: y, visibility: visibility)
    }

    private func assertPoints(
        _ actual: [Keypoint],
        _ expected: [Keypoint],
        accuracy: Double = 1e-9,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, "point count", file: file, line: line)
        guard actual.count == expected.count else { return }
        for (got, want) in zip(actual, expected) {
            XCTAssertEqual(got.x, want.x, accuracy: accuracy, "x", file: file, line: line)
            XCTAssertEqual(got.y, want.y, accuracy: accuracy, "y", file: file, line: line)
        }
    }

    /// Deterministic pseudo-random points, so a failure is reproducible the
    /// way `np.random.default_rng(angle)` makes the Python ones reproducible.
    private func randomPoints(seed: UInt64, count: Int = 50) -> [Keypoint] {
        var rng = SplitMix64(seed: seed)
        return (0..<count).map { _ in
            keypoint(
                Double.random(in: 0..<1, using: &rng) * Double(width - 1),
                Double.random(in: 0..<1, using: &rng) * Double(height - 1)
            )
        }
    }

    // ----------------------------------------------------------------- //

    func test180IsTheIndexFlipFormula() throws {
        // x = W - 1 - x_r, y = H - 1 - y_r on known points.
        let rotated = try Rotation.rotate(corners, angleDegrees: 180, width: width, height: height)
        let expected = [
            keypoint(Double(width - 1 - 0), Double(height - 1 - 0)),
            keypoint(Double(width - 1 - 3), Double(height - 1 - 0)),
            keypoint(Double(width - 1 - 0), Double(height - 1 - 2)),
            keypoint(Double(width - 1 - 3), Double(height - 1 - 2)),
            keypoint(Double(width - 1 - 1), Double(height - 1 - 1)),
        ]
        assertPoints(rotated, expected, accuracy: 0)
        assertPoints(
            rotated,
            [keypoint(3, 2), keypoint(0, 2), keypoint(3, 0), keypoint(0, 0), keypoint(2, 1)],
            accuracy: 0
        )
        let back = try Rotation.inverseRotate(rotated, angleDegrees: 180, width: width, height: height)
        assertPoints(back, corners, accuracy: 0)
    }

    func testCardinalRoundTrips() throws {
        for angle in [0.0, 90.0, 180.0, 270.0] {
            let points = randomPoints(seed: UInt64(angle))
            let size = try Rotation.rotatedFrameSize(angleDegrees: angle, width: width, height: height)

            let rotated = try Rotation.rotate(points, angleDegrees: angle, width: width, height: height)
            XCTAssertEqual(rotated.count, points.count)
            // The frame size swaps for 90/270; the point array itself keeps its count.
            let expectedSize: (width: Int, height: Int) = (angle == 0 || angle == 180)
                ? (width, height)
                : (height, width)
            XCTAssertEqual(size.width, expectedSize.width)
            XCTAssertEqual(size.height, expectedSize.height)

            let back = try Rotation.inverseRotate(rotated, angleDegrees: angle, width: width, height: height)
            assertPoints(back, points)

            // Rotating twice equals one 2*angle rotation (frame size swapped in between).
            let twice = try Rotation.rotate(rotated, angleDegrees: angle, width: size.width, height: size.height)
            let expectedTwice = try Rotation.rotate(
                points,
                angleDegrees: (2 * angle).truncatingRemainder(dividingBy: 360),
                width: width,
                height: height
            )
            assertPoints(twice, expectedTwice)
        }
    }

    func test90CounterClockwiseMapsKnownPoints() throws {
        // 90° CCW is what cv2.ROTATE_90_COUNTERCLOCKWISE does to pixels.
        let points = [keypoint(0, 0), keypoint(3, 0), keypoint(1, 1)]
        let rotated = try Rotation.rotate(points, angleDegrees: 90, width: width, height: height)
        // (x, y) -> (y, W - 1 - x) in the (HEIGHT, WIDTH) rotated frame.
        assertPoints(rotated, [keypoint(0, 3), keypoint(0, 0), keypoint(1, 2)], accuracy: 0)
        let size = try Rotation.rotatedFrameSize(angleDegrees: 90, width: width, height: height)
        XCTAssertEqual(size.width, height)
        XCTAssertEqual(size.height, width)
    }

    func test270CounterClockwiseMapsKnownPoints() throws {
        let points = [keypoint(0, 0), keypoint(0, 2), keypoint(1, 1)]
        let rotated = try Rotation.rotate(points, angleDegrees: 270, width: width, height: height)
        // (x, y) -> (H - 1 - y, x) in the (HEIGHT, WIDTH) rotated frame.
        assertPoints(rotated, [keypoint(2, 0), keypoint(0, 0), keypoint(1, 1)], accuracy: 0)
        let size = try Rotation.rotatedFrameSize(angleDegrees: 270, width: width, height: height)
        XCTAssertEqual(size.width, height)
        XCTAssertEqual(size.height, width)
    }

    func test90And270UndoEachOther() throws {
        let points = corners
        let first = try Rotation.rotate(points, angleDegrees: 90, width: width, height: height)
        assertPoints(
            try Rotation.rotate(first, angleDegrees: 270, width: height, height: width),
            points,
            accuracy: 0
        )
        let second = try Rotation.rotate(points, angleDegrees: 270, width: width, height: height)
        assertPoints(
            try Rotation.rotate(second, angleDegrees: 90, width: height, height: width),
            points,
            accuracy: 0
        )
        // The explicit inverse helper is exactly the opposite angle.
        assertPoints(
            try Rotation.inverseRotate(points, angleDegrees: 90, width: width, height: height),
            try Rotation.rotate(points, angleDegrees: 270, width: height, height: width),
            accuracy: 0
        )
        assertPoints(
            try Rotation.inverseRotate(points, angleDegrees: 270, width: width, height: height),
            try Rotation.rotate(points, angleDegrees: 90, width: height, height: width),
            accuracy: 0
        )
    }

    func testZeroRotationIsIdentityAndKeepsSize() throws {
        assertPoints(try Rotation.rotate(corners, angleDegrees: 0, width: width, height: height), corners, accuracy: 0)
        let zero = try Rotation.rotatedFrameSize(angleDegrees: 0, width: width, height: height)
        XCTAssertEqual(zero.width, width)
        XCTAssertEqual(zero.height, height)
        let half = try Rotation.rotatedFrameSize(angleDegrees: 180, width: width, height: height)
        XCTAssertEqual(half.width, width)
        XCTAssertEqual(half.height, height)
    }

    func testNegativeAndLargeAnglesNormalise() throws {
        assertPoints(
            try Rotation.rotate(corners, angleDegrees: -180, width: width, height: height),
            try Rotation.rotate(corners, angleDegrees: 180, width: width, height: height),
            accuracy: 0
        )
        assertPoints(
            try Rotation.rotate(corners, angleDegrees: 450, width: width, height: height),
            try Rotation.rotate(corners, angleDegrees: 90, width: width, height: height),
            accuracy: 0
        )
        assertPoints(
            try Rotation.inverseRotate(corners, angleDegrees: -90, width: width, height: height),
            try Rotation.inverseRotate(corners, angleDegrees: 270, width: width, height: height),
            accuracy: 0
        )
    }

    func testNonCardinalAngleRotatesAboutTheCentreAndRoundTrips() throws {
        let points = [keypoint(0, 0), keypoint(3, 2), keypoint(1.5, 1)]
        let angle = 45.0
        let rotated = try Rotation.rotate(points, angleDegrees: angle, width: width, height: height)
        let size = try Rotation.rotatedFrameSize(angleDegrees: angle, width: width, height: height)
        XCTAssertEqual(size.width, width)
        XCTAssertEqual(size.height, height)

        let centre = [keypoint((Double(width) - 1) / 2, (Double(height) - 1) / 2)]
        assertPoints(
            try Rotation.rotate(centre, angleDegrees: angle, width: width, height: height),
            centre,
            accuracy: 1e-12
        )
        XCTAssertFalse(allClose(rotated, points, accuracy: 1e-6))
        assertPoints(
            try Rotation.inverseRotate(rotated, angleDegrees: angle, width: width, height: height),
            points
        )

        // Four eighth-turns are a half turn, i.e. the exact 180° branch.
        var fourTimes = points
        for _ in 0..<4 {
            fourTimes = try Rotation.rotate(fourTimes, angleDegrees: angle, width: width, height: height)
        }
        assertPoints(
            fourTimes,
            try Rotation.rotate(points, angleDegrees: 180, width: width, height: height)
        )
    }

    func testRejectsBadFrameSize() {
        XCTAssertThrowsError(
            try Rotation.rotate(corners, angleDegrees: 180, width: 0, height: height)
        ) { error in
            XCTAssertEqual(error as? RotationError, .nonPositiveFrameSize(width: 0, height: height))
            XCTAssertTrue("\(error)".contains("positive"))
        }
        XCTAssertThrowsError(
            try Rotation.rotatedFrameSize(angleDegrees: 90, width: width, height: 0)
        ) { error in
            XCTAssertEqual(error as? RotationError, .nonPositiveFrameSize(width: width, height: 0))
        }
    }

    func testVisibilityRidesAlong() throws {
        let point = keypoint(1, 1, visibility: 0.42)
        let rotated = try Rotation.rotate([point], angleDegrees: 180, width: width, height: height)
        XCTAssertEqual(rotated.first?.visibility, 0.42)
    }

    // ----------------------------------------------------------------- //

    private func allClose(_ a: [Keypoint], _ b: [Keypoint], accuracy: Double) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy { abs($0.x - $1.x) <= accuracy && abs($0.y - $1.y) <= accuracy }
    }
}

/// SplitMix64: a tiny deterministic generator, so the "random" round-trip
/// points are the same on every run and on the Mac.
private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
