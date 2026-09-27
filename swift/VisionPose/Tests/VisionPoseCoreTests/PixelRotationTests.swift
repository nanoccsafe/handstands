// The display rotation, checked against real pixels.
//
// ``DisplayTransformTests`` pins the matrix tables down arithmetically; this
// pins down what Core Image does with them. A transposed or mirrored table
// would still satisfy the arithmetic and would still pass a
// compare-to-MediaPipe eyeball check on one clip, so it is worth rendering a
// marked frame and looking at where the mark ended up.
//
// The marker is a single white pixel in the **top-left** of the stored frame,
// and each quarter turn has to carry it to the corner a real clockwise turn
// would.

import CoreImage
import CoreVideo
import XCTest

@testable import VisionPoseCore

final class PixelRotationTests: XCTestCase {
    private let storedWidth = 4
    private let storedHeight = 2

    /// A `width x height` BGRA frame with one white pixel at (row, column).
    private func makeBuffer(width: Int, height: Int, markRow: Int, markColumn: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &buffer
        )
        guard status == kCVReturnSuccess, let buffer else {
            throw RunnerTestError.allocationFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw RunnerTestError.allocationFailed
        }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = base.assumingMemoryBound(to: UInt8.self)
        for row in 0..<height {
            for column in 0..<width {
                let offset = row * stride + column * 4
                let isMark = row == markRow && column == markColumn
                let value: UInt8 = isMark ? 255 : 0
                // BGRA, all four channels equal so the mark is plain white.
                for channel in 0..<4 { rowBytes[offset + channel] = value }
            }
        }
        return buffer
    }

    /// Where a pixel is, as `(row, column)` with row 0 at the top — a small
    /// struct rather than a tuple so it can be compared.
    private struct Marker: Equatable {
        let row: Int
        let column: Int
    }

    /// Where the white pixel ended up, and how many white pixels there are
    /// (more than one would mean the transform is a shear).
    private func markedPixel(in buffer: CVPixelBuffer) throws -> [Marker] {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw RunnerTestError.allocationFailed
        }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let width = CVPixelBufferGetWidth(buffer)
        let rowBytes = base.assumingMemoryBound(to: UInt8.self)
        var marks: [Marker] = []
        for row in 0..<height {
            for column in 0..<width {
                if rowBytes[row * stride + column * 4] > 127 {
                    marks.append(Marker(row: row, column: column))
                }
            }
        }
        return marks
    }

    private func render(_ transform: DisplayTransform, from buffer: CVPixelBuffer) throws -> CVPixelBuffer {
        let affine = transform.pixelTransform
        let image = CIImage(cvPixelBuffer: buffer).transformed(
            by: CGAffineTransform(
                a: affine.a, b: affine.b, c: affine.c,
                d: affine.d, tx: affine.tx, ty: affine.ty
            )
        )
        // The turned image has to come out exactly the size the transform says.
        XCTAssertEqual(Int(image.extent.width), transform.displaySize.width)
        XCTAssertEqual(Int(image.extent.height), transform.displaySize.height)
        XCTAssertEqual(image.extent.origin.x, 0)
        XCTAssertEqual(image.extent.origin.y, 0)
        var out: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            transform.displaySize.width,
            transform.displaySize.height,
            kCVPixelFormatType_32BGRA,
            nil,
            &out
        )
        guard status == kCVReturnSuccess, let out else { throw RunnerTestError.allocationFailed }
        CIContext().render(image, to: out)
        return out
    }

    private func markedCorner(quarterTurns: Int) throws -> Marker {
        let stored = try makeBuffer(
            width: storedWidth, height: storedHeight, markRow: 0, markColumn: 0
        )
        let transform = try DisplayTransform(
            quarterTurns: quarterTurns,
            storedSize: PixelSize(width: storedWidth, height: storedHeight)
        )
        let rendered = try render(transform, from: stored)
        let marks = try markedPixel(in: rendered)
        XCTAssertEqual(
            marks.count, 1,
            "quarter turn \(quarterTurns): expected one marked pixel, got \(marks)"
        )
        return marks[0]
    }

    func testNoRotationLeavesTheMarkWhereItIs() throws {
        XCTAssertEqual(try markedCorner(quarterTurns: 0), Marker(row: 0, column: 0))
    }

    /// Turned 90° clockwise, the top-left corner of the frame ends up at the
    /// top-right of the turned frame.
    func testAClockwiseTurnMovesTheMarkToTheTopRight() throws {
        XCTAssertEqual(try markedCorner(quarterTurns: 1), Marker(row: 0, column: 1))
    }

    func testAHalfTurnMovesTheMarkToTheBottomRight() throws {
        XCTAssertEqual(try markedCorner(quarterTurns: 2), Marker(row: 1, column: 3))
    }

    /// Turned 90° counter-clockwise, the top-left corner of the frame ends up
    /// at the bottom-left of the turned frame — which is 2 wide and 4 tall, so
    /// that is the last of its four rows.
    func testACounterClockwiseTurnMovesTheMarkToTheBottomLeft() throws {
        XCTAssertEqual(try markedCorner(quarterTurns: 3), Marker(row: 3, column: 0))
    }

    /// The real clip shape: 1024x576 stored, 576x1024 displayed.
    func testTheSidewaysPhoneClipGeometry() throws {
        let stored = try makeBuffer(
            width: 1024, height: 576, markRow: 0, markColumn: 0
        )
        let transform = try DisplayTransform(quarterTurns: 1, storedSize: PixelSize(width: 1024, height: 576))
        let rendered = try render(transform, from: stored)
        XCTAssertEqual(CVPixelBufferGetWidth(rendered), 576)
        XCTAssertEqual(CVPixelBufferGetHeight(rendered), 1024)
        // Top-left of a 1024x576 frame -> top-right of a 576x1024 one.
        XCTAssertEqual(try markedPixel(in: rendered), [Marker(row: 0, column: 575)])
    }
}

private enum RunnerTestError: Error {
    case allocationFailed
}
