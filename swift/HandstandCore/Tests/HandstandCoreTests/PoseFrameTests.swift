import XCTest

@testable import HandstandCore

/// `PoseFrame`'s two helpers: the midpoint of a joint pair (the body frame's
/// origin among them) and the strict whole-frame validity gate.
final class PoseFrameTests: XCTestCase {
    /// A frame carrying the whole tracked schema, every joint at
    /// `visibility`, spread across a body-length-ish layout.
    private func fullFrame(visibilities: (Joint) -> Double = { _ in 0.9 }) -> PoseFrame {
        var joints: [Joint: Keypoint] = [:]
        for (index, joint) in Joint.allCases.enumerated() {
            joints[joint] = Keypoint(
                x: Double(index) * 10,
                y: Double(index) * 10,
                visibility: visibilities(joint)
            )
        }
        return PoseFrame(frameIndex: 7, tMs: 233, joints: joints)
    }

    func testMidpointIsTheMeanOfTheTwoJoints() throws {
        let joints: [Joint: Keypoint] = [
            .leftWrist: Keypoint(x: 70, y: 200, visibility: 0.9),
            .rightWrist: Keypoint(x: 130, y: 200, visibility: 0.8),
        ]
        let frame = PoseFrame(frameIndex: 0, tMs: 0, joints: joints)

        let midpoint = try XCTUnwrap(frame.midpoint(.leftWrist, .rightWrist))
        XCTAssertEqual(midpoint.x, 100, accuracy: 0)
        XCTAssertEqual(midpoint.y, 200, accuracy: 0)
        XCTAssertEqual(midpoint.visibility, 0.8, "the less visible side gates the midpoint")
    }

    func testMidpointIsNilWhenAJointIsMissing() {
        let frame = PoseFrame(
            frameIndex: 0,
            tMs: 0,
            joints: [.leftWrist: Keypoint(x: 70, y: 200, visibility: 0.9)]
        )
        XCTAssertNil(frame.midpoint(.leftWrist, .rightWrist))
        XCTAssertNil(frame.midpoint(.leftHip, .rightHip))
        XCTAssertNotNil(frame.midpoint(.leftWrist, .leftWrist))
    }

    func testIsValidNeedsTheWholeTrackedSchema() {
        let full = fullFrame()
        XCTAssertEqual(full.joints.count, Joint.allCases.count)
        XCTAssertTrue(full.isValid(minVisibility: 0.9))
        XCTAssertFalse(full.isValid(minVisibility: 0.91))

        // One joint short of the schema is not a valid frame, however visible
        // the rest are.
        var partial = full
        partial.joints[.rightFootIndex] = nil
        XCTAssertFalse(partial.isValid(minVisibility: 0))
    }

    func testIsValidRejectsALowVisibilityJoint() {
        let frame = fullFrame { $0 == .leftAnkle ? 0.2 : 0.9 }
        XCTAssertFalse(frame.isValid(minVisibility: 0.5))
        XCTAssertTrue(frame.isValid(minVisibility: 0.2))
    }

    func testIsValidRejectsNaNVisibility() {
        let frame = fullFrame { _ in Double.nan }
        XCTAssertFalse(frame.isValid(minVisibility: 0))
    }

    func testFrameIdentityFieldsSurvive() {
        let frame = fullFrame()
        XCTAssertEqual(frame.frameIndex, 7)
        XCTAssertEqual(frame.tMs, 233)
    }
}
