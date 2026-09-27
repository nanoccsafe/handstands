// Coordinate conversion and the 180° map-back.
//
// The invariants here are the ones the whole schema rests on: a normalised
// point and its 180°-rotated counterpart must come out `size - 1` apart, and
// turning a point and mapping it back must be the identity.

import XCTest

@testable import VisionPoseCore

final class GeometryTests: XCTestCase {
    func testNormalizedToPixelsFlipsTheBottomLeftOrigin() {
        let size = PixelSize(width: 576, height: 1024)
        // (0, 0) is the bottom-left corner in Vision, the top-left in pixels.
        let bottomLeft = try! CoordinateMath.normalizedToPixels(
            NormalizedPoint(x: 0, y: 0), in: size
        )
        XCTAssertEqual(bottomLeft.x, 0)
        XCTAssertEqual(bottomLeft.y, 1023)
        // ...and the other way round.
        let topRight = try! CoordinateMath.normalizedToPixels(
            NormalizedPoint(x: 1, y: 1), in: size
        )
        XCTAssertEqual(topRight.x, 575)
        XCTAssertEqual(topRight.y, 0)
    }

    func testNormalizedToPixelsUsesSizeMinusOne() {
        let size = PixelSize(width: 10, height: 20)
        let middle = try! CoordinateMath.normalizedToPixels(
            NormalizedPoint(x: 0.5, y: 0.5), in: size
        )
        // size - 1, not size: this is what makes the half turn an exact inverse.
        XCTAssertEqual(middle.x, 4.5, accuracy: 1e-12)
        XCTAssertEqual(middle.y, 9.5, accuracy: 1e-12)
    }

    func testNormalizedToPixelsAgreesWithMediaPipesNormalisationForAnUprightPoint() {
        // MediaPipe has a top-left origin, so its (x, y) and Vision's (x, 1 - y)
        // are the same point; the conversion must agree with
        // `pose_mediapipe.normalized_to_pixels` to the last bit.
        let size = PixelSize(width: 464, height: 640)
        let visionTop = try! CoordinateMath.normalizedToPixels(
            NormalizedPoint(x: 0.25, y: 0.75), in: size
        )
        let mediaPipeTop = (0.25 * 463.0, 0.25 * 639.0)
        XCTAssertEqual(visionTop.x, mediaPipeTop.0, accuracy: 1e-12)
        XCTAssertEqual(visionTop.y, mediaPipeTop.1, accuracy: 1e-12)
    }

    func testHalfTurnMirrorsAboutTheFrameCentre() {
        let size = PixelSize(width: 576, height: 1024)
        let turned = try! CoordinateMath.rotateHalfTurn(PixelPoint(x: 0, y: 0), in: size)
        XCTAssertEqual(turned.x, 575)
        XCTAssertEqual(turned.y, 1023)
    }

    func testHalfTurnMapBackIsAnExactRoundTrip() {
        let size = PixelSize(width: 576, height: 1024)
        for point in [
            PixelPoint(x: 0, y: 0),
            PixelPoint(x: 575, y: 1023),
            PixelPoint(x: 310.5, y: 551.25),
            PixelPoint(x: 0.5, y: 1023.5),
        ] {
            let turned = try! CoordinateMath.rotateHalfTurn(point, in: size)
            let back = try! CoordinateMath.mapBackHalfTurn(turned, in: size)
            XCTAssertEqual(back.x, point.x, accuracy: 1e-12)
            XCTAssertEqual(back.y, point.y, accuracy: 1e-12)
        }
    }

    /// A point and the same point on the model-facing 180°-rotated frame are
    /// the same physical pixel, so mapping the second one back has to land on
    /// the first. This is the `none`/`180` agreement the schema doc promises.
    func testARotatedFramePointMapsBackOntoTheSamePixel() {
        let size = PixelSize(width: 464, height: 640)
        let display = PixelPoint(x: 231.5, y: 320)
        // The same physical pixel in the 180°-rotated frame, and the normalised
        // coordinates Vision would report for it *there* — its origin is the
        // bottom left, hence the `1 -`.
        let inRotatedFrame = try! CoordinateMath.rotateHalfTurn(display, in: size)
        let asReported = try! CoordinateMath.normalizedToPixels(
            NormalizedPoint(
                x: inRotatedFrame.x / 463.0,
                y: 1 - inRotatedFrame.y / 639.0
            ),
            in: size
        )
        // ...which is that point in the rotated frame again, pixel for pixel.
        XCTAssertEqual(asReported.x, inRotatedFrame.x, accuracy: 1e-9)
        XCTAssertEqual(asReported.y, inRotatedFrame.y, accuracy: 1e-9)
        let mappedBack = try! CoordinateMath.mapBackHalfTurn(asReported, in: size)
        XCTAssertEqual(mappedBack.x, display.x, accuracy: 1e-9)
        XCTAssertEqual(mappedBack.y, display.y, accuracy: 1e-9)
    }

    func testAFrameSizeMustBePositive() {
        for size in [PixelSize(width: 0, height: 10), PixelSize(width: 10, height: 0)] {
            XCTAssertThrowsError(
                try CoordinateMath.normalizedToPixels(NormalizedPoint(x: 0.5, y: 0.5), in: size)
            ) { error in
                XCTAssertEqual(
                    error as? VisionPoseError,
                    .invalidFrameSize(width: size.width, height: size.height)
                )
            }
            XCTAssertThrowsError(try CoordinateMath.rotateHalfTurn(PixelPoint(x: 1, y: 1), in: size))
            XCTAssertThrowsError(try CoordinateMath.mapBackHalfTurn(PixelPoint(x: 1, y: 1), in: size))
        }
    }
}
