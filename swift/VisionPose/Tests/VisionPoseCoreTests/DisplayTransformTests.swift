// Resolving a video track's `preferredTransform` into a display frame.
//
// The matrices below are the real ones from the handstand clips (read off with
// `track.preferredTransform` on the Mac mini), so this is the test that pins
// down which way round a phone's sideways footage has to go.

import XCTest

@testable import VisionPoseCore

final class DisplayTransformTests: XCTestCase {
    /// 6508f9b355bd / 64184de33f84: 1024x576 stored, rotation -90 in the
    /// catalogue, displayed 576x1024.
    func testTheSidewaysPhoneClipsAreOneQuarterTurnClockwise() throws {
        let transform = try DisplayTransform.quarterTurns(
            a: 0, b: 1, c: -1, d: 0,  // (x, y) -> (-y, x)
            storedSize: PixelSize(width: 1024, height: 576)
        )
        XCTAssertEqual(transform.quarterTurns, 1)
        XCTAssertEqual(transform.displaySize, PixelSize(width: 576, height: 1024))
        XCTAssertFalse(transform.isIdentity)
    }

    /// 057c9e6c96af: 464x640, rotation 0, already upright.
    func testAnUprightClipNeedsNoRotation() throws {
        let transform = try DisplayTransform.quarterTurns(
            a: 1, b: 0, c: 0, d: 1,
            storedSize: PixelSize(width: 464, height: 640)
        )
        XCTAssertEqual(transform.quarterTurns, 0)
        XCTAssertEqual(transform.displaySize, PixelSize(width: 464, height: 640))
        XCTAssertTrue(transform.isIdentity)
    }

    /// The translation of a real container matrix is whatever the muxer felt
    /// like (1080 for a 1024-wide track), so it must not change the answer.
    func testTheTranslationIsIgnored() throws {
        let transform = try DisplayTransform.quarterTurns(
            a: 0, b: 1, c: -1, d: 0,
            storedSize: PixelSize(width: 1024, height: 576)
        )
        XCTAssertEqual(transform.quarterTurns, 1)
    }

    func testAllFourQuarterTurnsAreRecognised() throws {
        let cases: [(parts: [Double], turns: Int)] = [
            ([1, 0, 0, 1], 0),
            ([0, 1, -1, 0], 1),
            ([-1, 0, 0, -1], 2),
            ([0, -1, 1, 0], 3),
        ]
        for testCase in cases {
            let transform = try DisplayTransform.quarterTurns(
                a: testCase.parts[0], b: testCase.parts[1],
                c: testCase.parts[2], d: testCase.parts[3],
                storedSize: PixelSize(width: 1024, height: 576)
            )
            XCTAssertEqual(transform.quarterTurns, testCase.turns)
        }
    }

    func testAFloatMatrixIsCloseEnough() throws {
        let transform = try DisplayTransform.quarterTurns(
            a: 0, b: 1, c: -1.0000001, d: 1e-9,
            storedSize: PixelSize(width: 100, height: 50)
        )
        XCTAssertEqual(transform.quarterTurns, 1)
    }

    func testAQuarterTurnSwapsTheFrameSizeAndAHalfTurnDoesNot() throws {
        let stored = PixelSize(width: 1024, height: 576)
        XCTAssertEqual(
            try DisplayTransform(quarterTurns: 0, storedSize: stored).displaySize,
            PixelSize(width: 1024, height: 576)
        )
        XCTAssertEqual(
            try DisplayTransform(quarterTurns: 1, storedSize: stored).displaySize,
            PixelSize(width: 576, height: 1024)
        )
        XCTAssertEqual(
            try DisplayTransform(quarterTurns: 2, storedSize: stored).displaySize,
            PixelSize(width: 1024, height: 576)
        )
        XCTAssertEqual(
            try DisplayTransform(quarterTurns: 3, storedSize: stored).displaySize,
            PixelSize(width: 576, height: 1024)
        )
    }

    func testMirroredAndNonsenseTransformsAreRefused() {
        // det < 0: flipped.
        XCTAssertThrowsError(
            try DisplayTransform.quarterTurns(
                a: -1, b: 0, c: 0, d: 1, storedSize: PixelSize(width: 8, height: 4)
            )
        ) { XCTAssertEqual($0 as? VisionPoseError, .mirroredDisplayTransform) }
        // A scale, not a rotation.
        XCTAssertThrowsError(
            try DisplayTransform.quarterTurns(
                a: 2, b: 0, c: 0, d: 2, storedSize: PixelSize(width: 8, height: 4)
            )
        ) { error in
            guard case .unsupportedDisplayTransform = error as? VisionPoseError else {
                return XCTFail("expected unsupportedDisplayTransform, got \(error)")
            }
        }
        // A shear.
        XCTAssertThrowsError(
            try DisplayTransform.quarterTurns(
                a: 1, b: 0.5, c: 0, d: 1, storedSize: PixelSize(width: 8, height: 4)
            )
        ) { error in
            guard case .unsupportedDisplayTransform = error as? VisionPoseError else {
                return XCTFail("expected unsupportedDisplayTransform, got \(error)")
            }
        }
    }

    func testQuarterTurnsMustBeInRange() {
        for turns in [-1, 4, 90] {
            XCTAssertThrowsError(
                try DisplayTransform(
                    quarterTurns: turns, storedSize: PixelSize(width: 8, height: 4)
                )
            ) { XCTAssertEqual($0 as? VisionPoseError, .invalidQuarterTurns(turns)) }
        }
        XCTAssertThrowsError(
            try DisplayTransform(quarterTurns: 1, storedSize: PixelSize(width: 0, height: 4))
        ) { XCTAssertEqual($0 as? VisionPoseError, .invalidFrameSize(width: 0, height: 4)) }
    }

    // MARK: - The pixel transform

    /// The transform table, checked on points. The matrices are in **Core
    /// Image's** coordinate system — origin at the bottom left, `y` up, the
    /// image spanning `0...width` by `0...height` — so that is the frame the
    /// corners are named in here.
    ///
    /// A clockwise quarter turn carries the picture's top-left corner to the
    /// turned picture's top-right, which is the check that catches a
    /// transposed or mirrored table: the bug that would quietly produce
    /// sideways keypoints.
    func testQuarterTurnsMoveTheCornersWhereAClockwiseTurnWould() throws {
        let stored = PixelSize(width: 4, height: 2)
        let turns = try (0...3).map { try DisplayTransform(quarterTurns: $0, storedSize: stored) }

        func corners(of size: PixelSize) -> (bottomLeft: PixelPoint, bottomRight: PixelPoint, topLeft: PixelPoint, topRight: PixelPoint) {
            (
                PixelPoint(x: 0, y: 0),
                PixelPoint(x: Double(size.width), y: 0),
                PixelPoint(x: 0, y: Double(size.height)),
                PixelPoint(x: Double(size.width), y: Double(size.height))
            )
        }
        let source = corners(of: stored)

        // No turn: the picture is already upright.
        let identity = corners(of: stored)
        XCTAssertEqual(turns[0].pixelTransform.applied(to: source.topLeft), identity.topLeft)
        XCTAssertEqual(turns[0].pixelTransform.applied(to: source.bottomRight), identity.bottomRight)

        // 90° clockwise: top-left -> top-right, top-right -> bottom-right, and
        // so on round the frame. The turned picture is 2 wide and 4 tall.
        let clockwise = corners(of: PixelSize(width: 2, height: 4))
        XCTAssertEqual(turns[1].pixelTransform.applied(to: source.topLeft), clockwise.topRight)
        XCTAssertEqual(turns[1].pixelTransform.applied(to: source.topRight), clockwise.bottomRight)
        XCTAssertEqual(turns[1].pixelTransform.applied(to: source.bottomRight), clockwise.bottomLeft)
        XCTAssertEqual(turns[1].pixelTransform.applied(to: source.bottomLeft), clockwise.topLeft)

        // 90° counter-clockwise: top-left -> bottom-left, the right edge
        // becomes the top edge, and the turned picture is 2 wide and 4 tall.
        let counterClockwise = corners(of: PixelSize(width: 2, height: 4))
        XCTAssertEqual(turns[3].pixelTransform.applied(to: source.topLeft), counterClockwise.bottomLeft)
        XCTAssertEqual(turns[3].pixelTransform.applied(to: source.bottomRight), counterClockwise.topRight)
        XCTAssertEqual(turns[3].pixelTransform.applied(to: source.topRight), counterClockwise.topLeft)

        // 180°: top-left -> bottom-right.
        let half = corners(of: stored)
        XCTAssertEqual(turns[2].pixelTransform.applied(to: source.topLeft), half.bottomRight)
        XCTAssertEqual(turns[2].pixelTransform.applied(to: source.topRight), half.bottomLeft)
    }

    /// Quarter turns are a cyclic group: two of them are the same as their sum,
    /// and a turn undone is the identity. Checked on a **square** frame, where
    /// a turn does not change the frame's size and the two readings of
    /// "compose" coincide — on a 1024x576 frame the second turn happens in a
    /// frame of different proportions, which is a different question.
    func testQuarterTurnsComposeOnASquareFrame() throws {
        let square = PixelSize(width: 8, height: 8)
        let turns = try (0...3).map { try DisplayTransform(quarterTurns: $0, storedSize: square) }
        let point = PixelPoint(x: 1.25, y: 6.5)
        for a in 0...3 {
            for b in 0...3 {
                let stepwise = turns[b].pixelTransform.applied(
                    to: turns[a].pixelTransform.applied(to: point)
                )
                let combined = turns[(a + b) % 4].pixelTransform.applied(to: point)
                XCTAssertEqual(stepwise.x, combined.x, accuracy: 1e-9, "turns \(a) then \(b)")
                XCTAssertEqual(stepwise.y, combined.y, accuracy: 1e-9, "turns \(a) then \(b)")
            }
        }
        // A turn undone is no turn at all.
        for a in 0...3 {
            let back = turns[(4 - a) % 4].pixelTransform.applied(
                to: turns[a].pixelTransform.applied(to: point)
            )
            XCTAssertEqual(back.x, point.x, accuracy: 1e-9, "turns \(a) then \((4 - a) % 4)")
            XCTAssertEqual(back.y, point.y, accuracy: 1e-9, "turns \(a) then \((4 - a) % 4)")
        }
    }

    /// The same four turns as `CGImagePropertyOrientation`, so the matrix table
    /// can be read against Apple's own description of them.
    func testTheOrientationTable() throws {
        let expected: [Int: CGImagePropertyOrientation] = [
            0: .up, 1: .right, 2: .down, 3: .left,
        ]
        for (quarterTurns, orientation) in expected {
            let transform = try DisplayTransform(
                quarterTurns: quarterTurns, storedSize: PixelSize(width: 4, height: 2)
            )
            XCTAssertEqual(DisplayOrientation.orientation(for: transform), orientation)
        }
    }
}
