/// The joints the shared schema is measured over — MediaPipe's landmark names,
/// spelled exactly as `pipeline/handstand/pose_mediapipe.py` spells them
/// (`JOINT_NAMES`), so a Swift raw value and a Python name are the same string.
///
/// The 15 cases are `handstand.postprocess.TRACKED_JOINTS`: the nose, both
/// shoulders, elbows, wrists, hips, knees and ankles (`CORE_JOINTS`) plus the
/// two foot indexes (`FOOT_INDEX_JOINTS`). Declaration order is that order, and
/// `allCases` follows it, so `Joint.allCases.map(\.rawValue)` is the Python
/// tuple element for element.
///
/// The other 18 MediaPipe landmarks (eyes, ears, mouth, pinkies, thumbs,
/// heels) are deliberately absent: the body length, the body frame and every
/// number the report shows are built from these 15 only.
public enum Joint: String, CaseIterable, Sendable {
    case nose
    case leftShoulder = "left_shoulder"
    case rightShoulder = "right_shoulder"
    case leftElbow = "left_elbow"
    case rightElbow = "right_elbow"
    case leftWrist = "left_wrist"
    case rightWrist = "right_wrist"
    case leftHip = "left_hip"
    case rightHip = "right_hip"
    case leftKnee = "left_knee"
    case rightKnee = "right_knee"
    case leftAnkle = "left_ankle"
    case rightAnkle = "right_ankle"
    case leftFootIndex = "left_foot_index"
    case rightFootIndex = "right_foot_index"
}
