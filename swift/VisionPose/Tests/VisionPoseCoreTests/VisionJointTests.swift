// Joint names: 19 Vision joints, 17 of which MediaPipe also has.

import Vision
import XCTest

@testable import VisionPoseCore

final class VisionJointTests: XCTestCase {
    func testVisionReportsNineteenJoints() {
        XCTAssertEqual(VisionJoint.allCases.count, 19)
        XCTAssertEqual(VisionJoint.columnNames.count, 19)
    }

    /// The 17 joints both runners have, spelled MediaPipe's way. If a name
    /// drifts the bake-off's join silently drops the joint, so it is pinned.
    func testTheSharedJointsUseMediaPipesNames() {
        let expected: Set<String> = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        let shared = Set(VisionJoint.allCases.filter { !$0.isVisionOnly }.map(\.columnName))
        XCTAssertEqual(shared, expected)
        XCTAssertEqual(shared.count, 17)
    }

    /// `neck` and `root` are the only two joints MediaPipe has no name for.
    func testTheVisionOnlyJoints() {
        let visionOnly = VisionJoint.allCases.filter(\.isVisionOnly)
        XCTAssertEqual(visionOnly.map(\.columnName), ["neck", "root"])
    }

    func testColumnNamesAreUnique() {
        let names = VisionJoint.columnNames
        XCTAssertEqual(Set(names).count, names.count)
    }

    /// Every joint has to reach Vision's own enumeration, and the mapping has
    /// to be one to one — a duplicate there would mean two of our rows reading
    /// the same point, or one joint silently going missing. It is an exhaustive
    /// switch, so a joint added to the enum without a mapping is a build error.
    func testTheVisionBridgeIsOneToOne() {
        var seen: [VNHumanBodyPoseObservation.JointName: VisionJoint] = [:]
        for joint in VisionJoint.allCases {
            let name = joint.visionJointName
            XCTAssertNil(
                seen[name],
                "\(joint) and \(String(describing: seen[name])) both map to \(name)"
            )
            seen[name] = joint
        }
        XCTAssertEqual(seen.count, VisionJoint.allCases.count)
    }

    /// The joints the auto rule reads, matching `pose_mediapipe`'s.
    func testTheAutoRuleJoints() {
        XCTAssertEqual(
            VisionJoint.wristJoints.map(\.columnName), ["left_wrist", "right_wrist"]
        )
        XCTAssertEqual(
            VisionJoint.ankleJoints.map(\.columnName), ["left_ankle", "right_ankle"]
        )
    }
}
