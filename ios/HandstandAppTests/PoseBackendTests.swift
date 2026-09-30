import XCTest

@testable import HandstandApp
import VisionPoseKit

/// The pose backend (chainlink #82): Vision is the one that exists, and it
/// hands back the `VisionPoseKit` service the post-recording analysis will
/// run; MediaPipe is a placeholder until #45.
final class PoseBackendTests: XCTestCase {
    func testVisionMakesAService() throws {
        let service = try XCTUnwrap(PoseBackend.vision.makeService())
        XCTAssertTrue(service is VisionPoseService)
        XCTAssertEqual(service.backendName, "vision")
        XCTAssertTrue(PoseBackend.vision.isAvailable)
    }

    func testMediaPipeGivesNilUntilItsBackendExists() {
        XCTAssertNil(PoseBackend.mediapipe.makeService())
        XCTAssertFalse(PoseBackend.mediapipe.isAvailable)
    }

    func testNamesAndOrder() {
        XCTAssertEqual(PoseBackend.allCases.map(\.rawValue), ["vision", "mediapipe"])
        XCTAssertEqual(PoseBackend.vision.displayName, "Apple Vision")
        XCTAssertEqual(PoseBackend.mediapipe.displayName, "MediaPipe")
    }
}
