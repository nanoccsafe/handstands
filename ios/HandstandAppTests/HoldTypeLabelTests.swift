import XCTest

@testable import HandstandApp
import HandstandCore

/// The hold picker's list labels (chainlink #66): available holds read as
/// their display name, every other hold is listed but says "coming soon" —
/// the user sees what is planned without being able to choose it.
final class HoldTypeLabelTests: XCTestCase {
    func testAnAvailableHoldIsJustItsDisplayName() {
        XCTAssertEqual(HoldTypeLabel.text(for: .line), "Line")
    }

    func testEveryUnavailableHoldSaysComingSoon() {
        XCTAssertEqual(HoldTypeLabel.text(for: .tuck), "Tuck — coming soon")
        for hold in HoldType.allCases where hold.isAvailable == false {
            XCTAssertEqual(
                HoldTypeLabel.text(for: hold),
                "\(hold.displayName) — coming soon",
                hold.rawValue
            )
        }
    }
}
