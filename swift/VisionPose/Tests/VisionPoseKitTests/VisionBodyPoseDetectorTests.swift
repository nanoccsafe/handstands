// The one test that runs the real Vision model — over a frame no person could
// ever be in.
//
// Everything else in VisionPoseKitTests drives the service through
// `FakeDetector`, because the repo is public and a test image of a body is
// not something to check in. What the real detector has to prove is narrower:
// it runs, it does not throw, and a blank frame is a frame with nobody in it.

import CoreVideo
import XCTest

import HandstandCore
import VisionPoseKit

final class VisionBodyPoseDetectorTests: XCTestCase {
    func testABlackFrameHasNoPersonAndDoesNotThrow() throws {
        let blank = try makePixelBuffer(width: 64, height: 64)

        let detector = VisionBodyPoseDetector()
        let people = try detector.detect(blank, rotated: false)
        XCTAssertTrue(people.isEmpty)

        // ...and the same frame upside down: the orientation is a request
        // setting, not a reason to fail.
        let peopleRotated = try detector.detect(blank, rotated: true)
        XCTAssertTrue(peopleRotated.isEmpty)

        // The whole service over the same blank frame: nobody found, nothing
        // to map.
        let service = VisionPoseService(rotate: .auto, detector: VisionBodyPoseDetector())
        let frame = try service.process(blank, tMs: 0)
        XCTAssertFalse(frame.detected)
        XCTAssertTrue(frame.joints.isEmpty)
        XCTAssertEqual(frame.tMs, 0)
        XCTAssertFalse(frame.trainerContact)
    }
}
