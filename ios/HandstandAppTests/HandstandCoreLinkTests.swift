import HandstandCore
import XCTest

/// The app links `HandstandCore` as a local package dependency (see
/// ios/project.yml), so this test bundle can import it — that import is half
/// the proof; the numbers below are the other half, and they are the same
/// cases the package's own tests pin down in swift/HandstandCore.
final class HandstandCoreLinkTests: XCTestCase {
    func testTrackedJointCountIsTheSharedSchema() {
        XCTAssertEqual(Joint.allCases.count, 15)
        XCTAssertEqual(Joint.leftWrist.rawValue, "left_wrist")
        XCTAssertEqual(Joint.nose.rawValue, "nose")
    }

    func testKnownBodyFrameConversion() throws {
        // One body length above the wrist midpoint: u = 0, v = +1 — the same
        // sample BodyFrameTests checks in swift/HandstandCore.
        let uv = try BodyFrame.toBodyFrame(
            x: 120, y: 100, wristMidX: 120, wristMidY: 300, bodyLength: 200
        )
        XCTAssertEqual(uv.u, 0, accuracy: 1e-12)
        XCTAssertEqual(uv.v, 1, accuracy: 1e-12)
    }
}
