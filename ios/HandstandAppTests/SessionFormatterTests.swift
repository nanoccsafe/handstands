import XCTest

@testable import HandstandApp

/// `SessionFormatter` (chainlink #51): the two strings on a History row
/// that `VideoInfoFormatter` does not own — when a take happened, and what
/// it scored. Fixed time zone here so the expected strings are exact.
final class SessionFormatterTests: XCTestCase {
    private let utc = TimeZone(identifier: "UTC")!

    private var afternoon: Date {
        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 28
        components.hour = 14
        components.minute = 30
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = utc
        return calendar.date(from: components)!
    }

    func testDateTimeIsDateAndTimeInAFixedFormat() {
        XCTAssertEqual(SessionFormatter.dateTime(afternoon, timeZone: utc), "28 Sep 2026, 14:30")
        // A single-digit day keeps no padding, and the time is 24-hour.
        XCTAssertEqual(
            SessionFormatter.dateTime(afternoon.addingTimeInterval(-27 * 3600), timeZone: utc),
            "27 Sep 2026, 11:30"
        )
    }

    func testScoreIsTwoDecimals() {
        XCTAssertEqual(SessionFormatter.score(0.87), "0.87")
        XCTAssertEqual(SessionFormatter.score(1), "1.00")
        XCTAssertEqual(SessionFormatter.score(0), "0.00")
    }

    func testAScoreThatIsNotARealNumberReadsAsADash() {
        // `nil` is "not analysed yet" — the views say so in words; this is
        // only what a broken number would show instead of "nan".
        XCTAssertEqual(SessionFormatter.score(nil), "—")
        XCTAssertEqual(SessionFormatter.score(.nan), "—")
        XCTAssertEqual(SessionFormatter.score(.infinity), "—")
    }
}
