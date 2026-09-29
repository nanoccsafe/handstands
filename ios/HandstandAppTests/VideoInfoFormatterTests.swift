import XCTest

@testable import HandstandApp

/// The view-model half of the video screen: the strings it shows, tested
/// without a view, a picker or a video file anywhere near them.
final class VideoInfoFormatterTests: XCTestCase {
    func testDurationReadsAsClockTime() {
        XCTAssertEqual(VideoInfoFormatter.duration(0), "0:00")
        XCTAssertEqual(VideoInfoFormatter.duration(12.4), "0:12")
        XCTAssertEqual(VideoInfoFormatter.duration(65.4), "1:05")
        XCTAssertEqual(VideoInfoFormatter.duration(3599.4), "59:59")
        XCTAssertEqual(VideoInfoFormatter.duration(3661), "1:01:01")
    }

    func testNonsenseDurationReadsAsZeroInsteadOfTrapping() {
        XCTAssertEqual(VideoInfoFormatter.duration(-1), "0:00")
        XCTAssertEqual(VideoInfoFormatter.duration(.nan), "0:00")
        XCTAssertEqual(VideoInfoFormatter.duration(.infinity), "0:00")
        XCTAssertEqual(VideoInfoFormatter.duration(1e300), "0:00")
    }

    func testPixelSizeIsWidthByHeight() {
        XCTAssertEqual(VideoInfoFormatter.pixelSize(width: 1920, height: 1080), "1920 × 1080")
        XCTAssertEqual(VideoInfoFormatter.pixelSize(width: 1080, height: 1920), "1080 × 1920")
    }
}
