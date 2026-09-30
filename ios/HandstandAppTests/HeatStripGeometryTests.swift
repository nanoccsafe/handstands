import XCTest

@testable import HandstandApp

/// `HeatStripGeometry` (chainlink #49): the heat strip's x ↔ time mapping.
/// Pure numbers, no view — the strip draws its bins with
/// `fraction(tMs:…)` and reads taps back with `tMs(fraction:…)`, so both
/// directions have to agree with the clip's own span.
final class HeatStripGeometryTests: XCTestCase {
    /// A clip's span as the strip sees it: first frame at 0 ms, last at
    /// 1000 ms (middle and ends land on whole numbers).
    private let firstMs = 0
    private let lastMs = 1000

    func testZeroMiddleAndFullWidthMapToTheClipEnds() {
        XCTAssertEqual(
            HeatStripGeometry.tMs(fraction: 0, firstMs: firstMs, lastMs: lastMs), 0)
        XCTAssertEqual(
            HeatStripGeometry.tMs(fraction: 0.5, firstMs: firstMs, lastMs: lastMs), 500)
        XCTAssertEqual(
            HeatStripGeometry.tMs(fraction: 1, firstMs: firstMs, lastMs: lastMs), 1000)
    }

    /// The same three points asked the other way round — where on the
    /// strip a time is.
    func testTheClipEndsAndTheMiddleAreAtTheEdgesAndCentreOfTheStrip() {
        XCTAssertEqual(
            HeatStripGeometry.fraction(tMs: 0, firstMs: firstMs, lastMs: lastMs), 0)
        XCTAssertEqual(
            HeatStripGeometry.fraction(tMs: 500, firstMs: firstMs, lastMs: lastMs), 0.5)
        XCTAssertEqual(
            HeatStripGeometry.fraction(tMs: 1000, firstMs: firstMs, lastMs: lastMs), 1)
    }

    /// Round trip: any time → its place on the strip → back is the time
    /// again (to the millisecond the mapping is rounded to), and any
    /// fraction → time → place is the fraction again.
    func testTheRoundTripHolds() {
        for tMs in stride(from: 0, through: 1000, by: 7) {
            let fraction = HeatStripGeometry.fraction(
                tMs: tMs, firstMs: firstMs, lastMs: lastMs)
            let back = HeatStripGeometry.tMs(
                fraction: fraction, firstMs: firstMs, lastMs: lastMs)
            XCTAssertEqual(back, tMs)
        }
        for fraction in stride(from: 0.0, through: 1.0, by: 0.041) {
            let tMs = HeatStripGeometry.tMs(
                fraction: fraction, firstMs: firstMs, lastMs: lastMs)
            let back = HeatStripGeometry.fraction(
                tMs: tMs, firstMs: firstMs, lastMs: lastMs)
            XCTAssertEqual(back, fraction, accuracy: 0.5 / Double(lastMs - firstMs))
        }
    }

    /// A clip that does not start at zero maps the strip's ends to *its*
    /// own timestamps, not to zero.
    func testAClipStartingLaterMapsItsOwnTimestamps() {
        let first = 1_500
        let last = 2_500
        XCTAssertEqual(HeatStripGeometry.tMs(fraction: 0, firstMs: first, lastMs: last), 1_500)
        XCTAssertEqual(HeatStripGeometry.tMs(fraction: 1, firstMs: first, lastMs: last), 2_500)
        XCTAssertEqual(HeatStripGeometry.fraction(tMs: 2_000, firstMs: first, lastMs: last), 0.5)
    }

    /// A finger that slides off the strip clamps to the clip's ends
    /// rather than seeking past the movie, and a time outside the clip
    /// clamps back into it.
    func testOutOfRangeFractionsAndTimesClamp() {
        XCTAssertEqual(
            HeatStripGeometry.tMs(fraction: -0.3, firstMs: firstMs, lastMs: lastMs), 0)
        XCTAssertEqual(
            HeatStripGeometry.tMs(fraction: 1.4, firstMs: firstMs, lastMs: lastMs), 1000)
        XCTAssertEqual(
            HeatStripGeometry.tMs(fraction: .nan, firstMs: firstMs, lastMs: lastMs), 0)
        XCTAssertEqual(
            HeatStripGeometry.fraction(tMs: -50, firstMs: firstMs, lastMs: lastMs), 0)
        XCTAssertEqual(
            HeatStripGeometry.fraction(tMs: 9999, firstMs: firstMs, lastMs: lastMs), 1)
    }

    /// A clip with no span (a single frame) answers that frame's
    /// timestamp and the strip's start — no division by zero anywhere.
    func testAClipWithNoSpanIsItsOneTimestamp() {
        XCTAssertEqual(HeatStripGeometry.tMs(fraction: 0.7, firstMs: 400, lastMs: 400), 400)
        XCTAssertEqual(HeatStripGeometry.fraction(tMs: 400, firstMs: 400, lastMs: 400), 0)
    }
}
