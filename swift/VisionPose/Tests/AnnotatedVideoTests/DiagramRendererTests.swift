import CoreGraphics
import XCTest

import AnnotatedVideo
import HandstandCore

// --------------------------------------------------------------------------- #
// The renderer itself (chainlink #50), unit-tested without a video: a small
// bitmap context, the flip the contract asks the caller for, and pixels read
// back. The key case is the one the export leans on — a **bad** joint lands
// as red on its own pixel — plus the radius rule and the strip.
//
// The bitmap is RGBA (`byteOrder32Big` + `premultipliedLast`) so the bytes
// read back are `(r, g, b, a)` in that order, and the background is filled
// mid-grey first: what is drawn must differ from it, what is not must not.
// --------------------------------------------------------------------------- #

final class DiagramRendererTests: XCTestCase {
    // MARK: - The bitmap

    private static let background = 128  // 0.5 × 255, filled before drawing

    /// Draws into a fresh `width × height` RGBA bitmap over a mid-grey
    /// background, applying the flip `DiagramRenderer`'s contract asks the
    /// caller for: **display pixels, origin top-left** — so the pixel at
    /// display `(x, y)` is byte row `y`.
    private func bitmap(
        width: Int = 64, height: Int = 64,
        _ draw: (CGContext) throws -> Void
    ) throws -> [UInt8] {
        var data = [UInt8](repeating: 0, count: width * height * 4)
        try data.withUnsafeMutableBytes { raw in
            let context = try XCTUnwrap(
                CGContext(
                    data: raw.baseAddress,
                    width: width, height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: width * 4,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                        | CGImageAlphaInfo.premultipliedLast.rawValue
                ),
                "the bitmap context is made")
            context.setFillColor(
                red: CGFloat(Self.background) / 255.0,
                green: CGFloat(Self.background) / 255.0,
                blue: CGFloat(Self.background) / 255.0,
                alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.translateBy(x: 0, y: CGFloat(height))
            context.scaleBy(x: 1, y: -1)
            try draw(context)
        }
        return data
    }

    /// One pixel of the bitmap, `(r, g, b)`.
    private func pixel(_ data: [UInt8], width: Int, x: Int, y: Int) -> (r: Int, g: Int, b: Int) {
        let at = y * width * 4 + x * 4
        return (Int(data[at]), Int(data[at + 1]), Int(data[at + 2]))
    }

    // MARK: - The diagram

    /// The key check: a joint the diagram judged **bad** is red at its own
    /// pixel — not blended away, not somewhere else.
    func testABadJointIsDrawnRedAtItsOwnPixel() throws {
        let size = 64
        let data = try bitmap(width: size, height: size) { context in
            let frame = DiagramFrame(
                tMs: 0,
                inHold: true,
                joints: [
                    DiagramJoint(
                        joint: .leftHip, point: Point2(x: 32, y: 32),
                        severity: 1, band: .bad)
                ],
                bones: [],
                stackLine: nil,
                stackBand: .neutral,
                com: nil,
                comFloor: nil,
                balanceZone: nil,
                ideal: nil
            )
            DiagramRenderer.draw(frame, in: context, scale: 1, style: .standard)
        }

        let at = pixel(data, width: size, x: 32, y: 32)
        XCTAssertGreaterThan(at.r, 200, "red is the bad band's red")
        XCTAssertLessThan(at.g, 150)
        XCTAssertLessThan(at.b, 150)

        // The background around it is untouched grey.
        let far = pixel(data, width: size, x: 5, y: 5)
        XCTAssertLessThanOrEqual(abs(far.r - Self.background), 3)
        XCTAssertLessThanOrEqual(abs(far.g - Self.background), 3)
        XCTAssertLessThanOrEqual(abs(far.b - Self.background), 3)
    }

    /// A bone is drawn as a line between its two joints' positions — a
    /// mid-grey bitmap otherwise, so the line's pixels are the only ones
    /// that moved.
    func testABoneIsDrawnBetweenItsJoints() throws {
        let size = 64
        let data = try bitmap(width: size, height: size) { context in
            let joints = [
                DiagramJoint(joint: .leftShoulder, point: Point2(x: 32, y: 20),
                    severity: 0, band: .ok),
                DiagramJoint(joint: .leftElbow, point: Point2(x: 32, y: 44),
                    severity: 0, band: .ok),
            ]
            let frame = DiagramFrame(
                tMs: 0,
                inHold: true,
                joints: joints,
                bones: [DiagramBone(from: .leftShoulder, to: .leftElbow, band: .ok, severity: 0)],
                stackLine: nil,
                stackBand: .neutral,
                com: nil,
                comFloor: nil,
                balanceZone: nil,
                ideal: nil
            )
            DiagramRenderer.draw(frame, in: context, scale: 1, style: .standard)
        }

        let on = pixel(data, width: size, x: 32, y: 32)
        XCTAssertGreaterThan(on.g, on.r, "the bone is green where it runs")
        let off = pixel(data, width: size, x: 10, y: 32)
        XCTAssertLessThanOrEqual(abs(off.r - Self.background), 3, "nothing is drawn away from the bone")
        XCTAssertLessThanOrEqual(abs(off.g - Self.background), 3)
        XCTAssertLessThanOrEqual(abs(off.b - Self.background), 3)
    }

    // MARK: - The radius rule

    func testTheRadiusRuleIsFourPlusEightTimesSeverity() {
        let style = DiagramStyle.standard
        XCTAssertEqual(style.jointRadius(severity: nil), 4, "a joint with no severity")
        XCTAssertEqual(style.jointRadius(severity: 0), 4, "on target")
        XCTAssertEqual(style.jointRadius(severity: 0.5), 8, "half as bad")
        XCTAssertEqual(style.jointRadius(severity: 1), 12, "as bad as it gets")
        XCTAssertEqual(style.jointRadius(severity: 7), 12, "clamped to 1")
        XCTAssertEqual(style.jointRadius(severity: .nan), 4, "not a number is not a severity")
    }

    // MARK: - The heat strip

    func testTheStripDrawsItsBinsAndAWhitePlayhead() throws {
        let bins = [
            HeatBin(startMs: 0, endMs: 50, severity: 0, band: .ok, inHold: true),
            HeatBin(startMs: 50, endMs: 100, severity: 0.6, band: .bad, inHold: true),
        ]
        let width = 100
        let data = try bitmap(width: width, height: 8) { context in
            DiagramRenderer.drawHeatStrip(
                bins, playheadMs: 75,
                in: CGRect(x: 0, y: 0, width: width, height: 8),
                ctx: context, style: .standard)
        }

        let good = pixel(data, width: width, x: 20, y: 4)
        XCTAssertGreaterThan(good.g, good.r, "the first bin is the ok band's green")
        XCTAssertGreaterThan(good.g, good.b)

        let bad = pixel(data, width: width, x: 60, y: 4)
        XCTAssertGreaterThan(bad.r, 200, "the second bin is the bad band's red")
        XCTAssertLessThan(bad.g, 150)

        let playhead = pixel(data, width: width, x: 75, y: 4)
        XCTAssertGreaterThan(playhead.r, 240, "the playhead is white")
        XCTAssertGreaterThan(playhead.g, 240)
        XCTAssertGreaterThan(playhead.b, 240)
    }

    func testAnEmptyStripIsNeutralGrey() throws {
        let width = 64
        let data = try bitmap(width: width, height: 8) { context in
            DiagramRenderer.drawHeatStrip(
                [], playheadMs: 0,
                in: CGRect(x: 0, y: 0, width: width, height: 8),
                ctx: context, style: .standard)
        }
        for x in [0, 20, 40, width - 1] {
            let at = pixel(data, width: width, x: x, y: 4)
            // The style's grey: #8E8E93 → (142, 142, 147).
            XCTAssertLessThanOrEqual(abs(at.r - 142), 6, "x=\(x) is the neutral grey")
            XCTAssertLessThanOrEqual(abs(at.g - 142), 6)
            XCTAssertLessThanOrEqual(abs(at.b - 147), 6)
        }
    }
}
