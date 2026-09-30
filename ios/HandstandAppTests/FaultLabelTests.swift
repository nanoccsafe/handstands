import XCTest

@testable import HandstandApp
import HandstandCore

/// `FaultLabel` (chainlink #47): the scorer's feature keys said in words
/// for the results panel — `"hip_angle"` is a column name, "Hip angle" is
/// what a person reads.
final class FaultLabelTests: XCTestCase {
    func testFeatureKeysBecomeWords() {
        XCTAssertEqual(FaultLabel.text(for: "hip_angle"), "Hip angle")
        XCTAssertEqual(FaultLabel.text(for: "line_deviation"), "Line deviation")
        XCTAssertEqual(FaultLabel.text(for: "com_sway_sd"), "Com sway sd")
        XCTAssertEqual(FaultLabel.text(for: "off_shoulder"), "Off shoulder")
        XCTAssertEqual(FaultLabel.text(for: "elbow_angle"), "Elbow angle")
    }

    func testEveryScoredFeatureCanBeSaid() {
        for name in Scorer.scoreFeatures {
            let text = FaultLabel.text(for: name)
            XCTAssertFalse(text.isEmpty, "\(name) has a display name")
            XCTAssertFalse(
                text.contains("_"), "\(name) is spelled with words, not underscores")
        }
    }

    func testAnEmptyNameStaysEmpty() {
        XCTAssertEqual(FaultLabel.text(for: ""), "")
    }
}
