import XCTest

@testable import HandstandCore

/// The hold type the recording picker offers (chainlink #66): the raw values
/// are a persisted format, Line is the only available hold in the MVP, and
/// every stored string — however old or bogus — resolves to something usable.
final class HoldTypeTests: XCTestCase {
    func testTheRawValuesAreExactlyTheseStringsAndNeverChange() {
        XCTAssertEqual(
            HoldType.allCases.map(\.rawValue),
            ["line", "tuck", "straddle", "stag", "diamond", "one_arm"]
        )
        XCTAssertEqual(HoldType.oneArm.rawValue, "one_arm")
    }

    func testOnlyLineIsAvailableInTheMVP() {
        XCTAssertEqual(HoldType.available, [.line])
        XCTAssertTrue(HoldType.line.isAvailable)
        for hold in HoldType.allCases where hold != .line {
            XCTAssertFalse(hold.isAvailable, hold.rawValue)
        }
    }

    func testDefaultIsLine() {
        XCTAssertEqual(HoldType.default, .line)
    }

    func testResolvedFallsBackToLineForUnknownAndUnavailableValues() {
        XCTAssertEqual(HoldType.resolved(nil), .line)
        XCTAssertEqual(HoldType.resolved("bogus"), .line)
        // Known, but not recordable yet — still Line.
        XCTAssertEqual(HoldType.resolved("tuck"), .line)
        XCTAssertEqual(HoldType.resolved("one_arm"), .line)
        XCTAssertEqual(HoldType.resolved("line"), .line)
    }

    func testCodableRoundTrip() throws {
        let data = try JSONEncoder().encode([HoldType.line])
        XCTAssertEqual(String(data: data, encoding: .utf8), #"["line"]"#)
        let decoded: [HoldType] = try JSONDecoder().decode([HoldType].self, from: data)
        XCTAssertEqual(decoded, [.line])
    }

    func testReferenceResourceNameForLine() {
        XCTAssertEqual(HoldType.line.referenceResourceName, "reference-line")
    }

    func testDisplayNames() {
        XCTAssertEqual(
            HoldType.allCases.map(\.displayName),
            ["Line", "Tuck", "Straddle", "Stag", "Diamond", "One arm"]
        )
    }
}
