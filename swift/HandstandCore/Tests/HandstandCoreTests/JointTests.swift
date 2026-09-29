import XCTest

@testable import HandstandCore

/// `Joint`'s raw values are the Python joint names — the test that keeps them
/// that way. The expected lists are hard-coded: they are the contract with
/// `pipeline/handstand/`, not something derived from either side at run time.
final class JointTests: XCTestCase {
    /// `handstand.postprocess.TRACKED_JOINTS`: `CORE_JOINTS` followed by
    /// `FOOT_INDEX_JOINTS`, in that order.
    private let expectedTracked = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_foot_index",
        "right_foot_index",
    ]

    /// `handstand.pose_mediapipe.JOINT_NAMES`, all 33, in model order.
    private let pythonJointNames = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]

    func testRawValuesAreThePythonTrackedJointNamesInOrder() {
        XCTAssertEqual(Joint.allCases.map(\.rawValue), expectedTracked)
    }

    func testEveryRawValueIsASpellingFromThePythonJointNames() {
        for joint in Joint.allCases {
            XCTAssertTrue(
                pythonJointNames.contains(joint.rawValue),
                "\(joint.rawValue) is not in pose_mediapipe.JOINT_NAMES"
            )
        }
    }

    func testTheTrackedSchemaIsFifteenJoints() {
        // The "15 joints of the shared schema" the sidecar counts.
        XCTAssertEqual(Joint.allCases.count, 15)
    }

    func testRawValueRoundTrips() {
        for joint in Joint.allCases {
            XCTAssertEqual(Joint(rawValue: joint.rawValue), joint)
        }
        XCTAssertNil(Joint(rawValue: "left_eye"))
        XCTAssertNil(Joint(rawValue: "neck"))
    }
}
