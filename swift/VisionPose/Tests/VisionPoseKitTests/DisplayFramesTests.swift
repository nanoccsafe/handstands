// `DisplayFrames` — the display-orientation conversion the runner used to keep
// to itself (chainlink #82).
//
// What matters here is the contract: `nil` means "the decoded frame is already
// upright, use it as it is", and anything else is a buffer of the transform's
// display size carrying the same picture, rotated.

import CoreVideo
import XCTest

import VisionPoseCore
import VisionPoseKit

final class DisplayFramesTests: XCTestCase {
    private func transform(_ quarterTurns: Int, width: Int, height: Int) throws -> DisplayTransform {
        try DisplayTransform(
            quarterTurns: quarterTurns,
            storedSize: PixelSize(width: width, height: height)
        )
    }

    /// An upright track needs no conversion at all — and no copy: the caller
    /// hands the decoder's own buffer to the model.
    func testAnUprightTrackNeedsNoConversion() throws {
        let buffer = try makePixelBuffer(width: 64, height: 48)
        let upright = try transform(0, width: 64, height: 48)
        XCTAssertNil(try DisplayFrames.displayBuffer(from: buffer, transform: upright))
    }

    /// A sideways track comes back at its *display* size: 64x48 stored with a
    /// quarter turn is a 48x64 frame, the size a viewer sees.
    func testASidewaysTrackComesBackAtItsDisplaySize() throws {
        let buffer = try makePixelBuffer(width: 64, height: 48)
        let sideways = try transform(1, width: 64, height: 48)

        let display = try XCTUnwrap(DisplayFrames.displayBuffer(from: buffer, transform: sideways))

        XCTAssertEqual(CVPixelBufferGetWidth(display), 48)
        XCTAssertEqual(CVPixelBufferGetHeight(display), 64)
    }

    /// The pixels really move: a frame whose halves are painted red and blue
    /// comes out with the halves swapped after a half turn.
    func testAHalfTurnSwapsTheHalves() throws {
        let buffer = try makePixelBuffer(width: 64, height: 48)
        paintHalvesRedAndBlue(buffer)
        let halfTurn = try transform(2, width: 64, height: 48)

        let display = try XCTUnwrap(DisplayFrames.displayBuffer(from: buffer, transform: halfTurn))

        XCTAssertEqual(CVPixelBufferGetWidth(display), 64)
        XCTAssertEqual(CVPixelBufferGetHeight(display), 48)
        // Stored left (red) is display right, stored right (blue) is display
        // left — whatever the colour management does to the exact bytes, the
        // red-dominant half cannot be on the same side it started.
        let displayLeft = try XCTUnwrap(redAndBlue(display, x: 8, y: 24))
        let displayRight = try XCTUnwrap(redAndBlue(display, x: 55, y: 24))
        XCTAssertGreaterThan(displayLeft.blue, displayLeft.red)
        XCTAssertGreaterThan(displayRight.red, displayRight.blue)
    }

    /// Two conversions in a row share the cached `CIContext` without stepping
    /// on each other — the cache is the only mutable state in the type.
    func testRepeatedConversionsWork() throws {
        let buffer = try makePixelBuffer(width: 64, height: 48)
        let upright = try transform(0, width: 64, height: 48)
        let halfTurn = try transform(2, width: 64, height: 48)
        for _ in 0..<3 {
            XCTAssertNil(try DisplayFrames.displayBuffer(from: buffer, transform: upright))
            XCTAssertNotNil(try DisplayFrames.displayBuffer(from: buffer, transform: halfTurn))
        }
    }
}
