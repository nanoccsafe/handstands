import XCTest

@testable import HandstandApp

/// The clock the Record screen shows while the red light is on — the live
/// estimate only; the finished take's length comes from the file itself and
/// is `VideoInfoFormatter.duration`'s business.
final class CaptureFormatterTests: XCTestCase {
    func testElapsedReadsAsMinutesSecondsAndTenths() {
        XCTAssertEqual(CaptureFormatter.elapsed(0), "0:00.0")
        XCTAssertEqual(CaptureFormatter.elapsed(7.24), "0:07.2")
        XCTAssertEqual(CaptureFormatter.elapsed(10), "0:10.0")
        XCTAssertEqual(CaptureFormatter.elapsed(65.4), "1:05.4")
        XCTAssertEqual(CaptureFormatter.elapsed(3599.4), "59:59.4")
    }

    func testNonsenseElapsedReadsAsZeroInsteadOfTrapping() {
        XCTAssertEqual(CaptureFormatter.elapsed(-1), "0:00.0")
        XCTAssertEqual(CaptureFormatter.elapsed(.nan), "0:00.0")
        XCTAssertEqual(CaptureFormatter.elapsed(.infinity), "0:00.0")
        XCTAssertEqual(CaptureFormatter.elapsed(1e300), "0:00.0")
    }
}
