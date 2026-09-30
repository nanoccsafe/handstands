import CoreGraphics
import XCTest

@testable import HandstandApp
import HandstandCore

/// `OverlayGeometry` (chainlink #48): where the picture sits in the view
/// and where a display-pixel point lands on it. Pure maths — no player, no
/// view, nothing but numbers — which is the point: the layer plays the
/// video aspect-fitted, and this is the same fit, so the overlay and the
/// picture cannot drift apart.
final class OverlayGeometryTests: XCTestCase {
    /// A 720 × 1280 (portrait) video in a 400 × 200 (landscape) view: the
    /// height is what binds, so the picture is letterboxed sideways.
    func testAPortraitVideoInALandscapeViewLetterboxesHorizontally() {
        let video = CGSize(width: 720, height: 1280)
        let view = CGSize(width: 400, height: 200)

        let rect = OverlayGeometry.videoRect(videoSize: video, in: view)

        // min(400/720, 200/1280) = 0.15625, so 112.5 × 200, centred.
        XCTAssertEqual(rect.width, 112.5, accuracy: 1e-9)
        XCTAssertEqual(rect.height, 200, accuracy: 1e-9)
        XCTAssertEqual(rect.origin.x, (400 - 112.5) / 2, accuracy: 1e-9)
        XCTAssertEqual(rect.origin.y, 0, accuracy: 1e-9)
        XCTAssertEqual(
            rect.width / rect.height, video.width / video.height, accuracy: 1e-9,
            "the aspect ratio is kept")
    }

    /// The other way round: a 1280 × 720 (landscape) video in a 200 × 400
    /// (portrait) view is pillarboxed — the width binds this time.
    func testALandscapeVideoInAPortraitViewLetterboxesVertically() {
        let video = CGSize(width: 1280, height: 720)
        let view = CGSize(width: 200, height: 400)

        let rect = OverlayGeometry.videoRect(videoSize: video, in: view)

        XCTAssertEqual(rect.width, 200, accuracy: 1e-9)
        XCTAssertEqual(rect.height, 112.5, accuracy: 1e-9)
        XCTAssertEqual(rect.origin.x, 0, accuracy: 1e-9)
        XCTAssertEqual(rect.origin.y, (400 - 112.5) / 2, accuracy: 1e-9)
        XCTAssertEqual(
            rect.width / rect.height, video.width / video.height, accuracy: 1e-9)
    }

    /// The point that cannot be misplaced: the video's centre maps to the
    /// rect's centre, in both orientations — and the corners map to the
    /// corners, which is what pins the linear mapping down completely.
    func testACentrePointMapsToTheCentreInBothOrientations() {
        let cases: [(video: CGSize, view: CGSize)] = [
            (CGSize(width: 720, height: 1280), CGSize(width: 400, height: 200)),
            (CGSize(width: 1280, height: 720), CGSize(width: 200, height: 400)),
        ]
        for testCase in cases {
            let rect = OverlayGeometry.videoRect(videoSize: testCase.video, in: testCase.view)
            let centre = OverlayGeometry.viewPoint(
                Point2(x: testCase.video.width / 2, y: testCase.video.height / 2),
                videoSize: testCase.video, rect: rect)
            XCTAssertEqual(centre.x, rect.midX, accuracy: 1e-9, "\(testCase.video)")
            XCTAssertEqual(centre.y, rect.midY, accuracy: 1e-9, "\(testCase.video)")

            let origin = OverlayGeometry.viewPoint(
                Point2(x: 0, y: 0), videoSize: testCase.video, rect: rect)
            XCTAssertEqual(origin.x, rect.origin.x, accuracy: 1e-9)
            XCTAssertEqual(origin.y, rect.origin.y, accuracy: 1e-9)

            let far = OverlayGeometry.viewPoint(
                Point2(x: testCase.video.width, y: testCase.video.height),
                videoSize: testCase.video, rect: rect)
            XCTAssertEqual(far.x, rect.maxX, accuracy: 1e-9)
            XCTAssertEqual(far.y, rect.maxY, accuracy: 1e-9)
        }
    }

    /// An unreadable movie (0 × 0) or a view with no size yet draws
    /// nothing — a zero rect, never NaN.
    func testADegenerateSizeDrawsNothing() {
        let empty = OverlayGeometry.videoRect(
            videoSize: CGSize(width: 0, height: 0), in: CGSize(width: 400, height: 200))
        XCTAssertEqual(empty, .zero)

        let rect = OverlayGeometry.videoRect(
            videoSize: CGSize(width: 720, height: 1280), in: CGSize(width: 400, height: 200))
        let point = OverlayGeometry.viewPoint(
            Point2(x: 10, y: 20), videoSize: .zero, rect: rect)
        XCTAssertEqual(point, .zero)
    }
}
