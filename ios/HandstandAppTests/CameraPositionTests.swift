import XCTest

@testable import HandstandApp

/// The camera choice (chainlink #95): what a fresh install records with, how
/// the stored key resolves, and the pure fallback that picks a device when
/// the chosen camera is not there. No camera needed — the selection takes the
/// available positions as an argument, so the front/back/two/neither cases
/// are all testable here.
final class CameraPositionTests: XCTestCase {
    private let suiteName = "CameraPositionTests"
    private var defaults: UserDefaults!

    override func setUpWithError() throws {
        defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
    }

    override func tearDownWithError() throws {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
    }

    // MARK: - The stored choice

    func testAFreshInstallUsesTheBackCamera() {
        XCTAssertEqual(CameraSettingsStorage.read(defaults), .back)
        XCTAssertEqual(CameraSettingsStorage.defaultPosition, .back)
        XCTAssertEqual(CameraPosition.default, .back)
    }

    func testTheChoiceRoundTripsThroughTheStoredKey() {
        defaults.set(CameraPosition.front.rawValue, forKey: CameraSettingsStorage.positionKey)
        XCTAssertEqual(CameraSettingsStorage.read(defaults), .front)

        defaults.set(CameraPosition.back.rawValue, forKey: CameraSettingsStorage.positionKey)
        XCTAssertEqual(CameraSettingsStorage.read(defaults), .back)
    }

    func testAnUnknownStoredValueReadsAsBack() {
        defaults.set("sideways", forKey: CameraSettingsStorage.positionKey)
        XCTAssertEqual(CameraSettingsStorage.read(defaults), .back)
        XCTAssertEqual(CameraSettingsStorage.resolved(nil), .back)
        XCTAssertEqual(CameraSettingsStorage.resolved(""), .back)
    }

    func testTheStoredKeyIsCameraPosition() {
        XCTAssertEqual(CameraSettingsStorage.positionKey, "cameraPosition")
    }

    // MARK: - Choosing a device

    func testBackPreferredAndPresentUsesBack() {
        XCTAssertEqual(
            CaptureService.resolvedPosition(preferred: .back, available: [.back, .front]),
            .back
        )
    }

    func testFrontPreferredAndPresentUsesFront() {
        XCTAssertEqual(
            CaptureService.resolvedPosition(preferred: .front, available: [.back, .front]),
            .front
        )
    }

    func testAPreferredCameraThatIsMissingFallsBackToTheOther() {
        XCTAssertEqual(
            CaptureService.resolvedPosition(preferred: .front, available: [.back]),
            .back
        )
        XCTAssertEqual(
            CaptureService.resolvedPosition(preferred: .back, available: [.front]),
            .front
        )
    }

    func testNoCameraAtAllResolvesToNil() {
        XCTAssertNil(CaptureService.resolvedPosition(preferred: .back, available: []))
        XCTAssertNil(CaptureService.resolvedPosition(preferred: .front, available: []))
    }

    func testTheOppositeIsTheOtherCamera() {
        XCTAssertEqual(CameraPosition.back.opposite, .front)
        XCTAssertEqual(CameraPosition.front.opposite, .back)
    }
}
