// The 19 joints Vision's body-pose model reports, and the names they are
// written under.
//
// The pipeline's joint vocabulary is `handstand.pose_mediapipe.JOINT_NAMES`
// (MediaPipe's 33 landmarks). Vision has a coarser skeleton: 19 joints, of
// which 17 have a MediaPipe counterpart and two (`neck`, `root`) do not exist
// in it at all. Everything here is about that reconciliation:
//
// * ``VisionJoint/columnName`` is the name written to the CSV. The 17 shared
//   joints are spelled exactly as MediaPipe spells them, so the two runners'
//   parquets can be compared joint by joint; `neck` and `root` keep Vision's
//   own names and are the only two rows a MediaPipe join will drop.
// * ``VisionJoint/visionJointName`` bridges to Vision's own enum. Keeping the
//   switch in the core means an SDK rename is a compile error here rather than
//   a silently empty joint at run time.

import Foundation
import Vision

/// One joint of Vision's body-pose skeleton.
public enum VisionJoint: String, CaseIterable, Equatable, Sendable {
    case nose
    case leftEye
    case rightEye
    case leftEar
    case rightEar
    case leftShoulder
    case rightShoulder
    case neck
    case leftElbow
    case rightElbow
    case leftWrist
    case rightWrist
    case root
    case leftHip
    case rightHip
    case leftKnee
    case rightKnee
    case leftAnkle
    case rightAnkle

    /// The name this joint is written under in the CSV and, after
    /// `handstand.vision_import`, in the parquet.
    ///
    /// MediaPipe's spelling for the 17 joints it also has (`left_shoulder`,
    /// `right_ankle`, ...) so that a frame-by-frame comparison is a join on the
    /// name; `neck` and `root` are Vision's alone.
    public var columnName: String {
        switch self {
        case .nose: return "nose"
        case .leftEye: return "left_eye"
        case .rightEye: return "right_eye"
        case .leftEar: return "left_ear"
        case .rightEar: return "right_ear"
        case .leftShoulder: return "left_shoulder"
        case .rightShoulder: return "right_shoulder"
        case .neck: return "neck"
        case .leftElbow: return "left_elbow"
        case .rightElbow: return "right_elbow"
        case .leftWrist: return "left_wrist"
        case .rightWrist: return "right_wrist"
        case .leftHip: return "left_hip"
        case .rightHip: return "right_hip"
        case .leftKnee: return "left_knee"
        case .rightKnee: return "right_knee"
        case .leftAnkle: return "left_ankle"
        case .rightAnkle: return "right_ankle"
        case .root: return "root"
        }
    }

    /// Is this a joint MediaPipe does not have?
    ///
    /// Only ``neck`` and ``root``: Vision's shoulder midpoint and hip
    /// midpoint, which MediaPipe's 33 landmarks spell out with several
    /// landmarks each and therefore has no single name for.
    public var isVisionOnly: Bool {
        self == .neck || self == .root
    }

    /// The same joint in Vision's own enumeration.
    public var visionJointName: VNHumanBodyPoseObservation.JointName {
        switch self {
        case .nose: return .nose
        case .leftEye: return .leftEye
        case .rightEye: return .rightEye
        case .leftEar: return .leftEar
        case .rightEar: return .rightEar
        case .leftShoulder: return .leftShoulder
        case .rightShoulder: return .rightShoulder
        case .neck: return .neck
        case .leftElbow: return .leftElbow
        case .rightElbow: return .rightElbow
        case .leftWrist: return .leftWrist
        case .rightWrist: return .rightWrist
        case .leftHip: return .leftHip
        case .rightHip: return .rightHip
        case .leftKnee: return .leftKnee
        case .rightKnee: return .rightKnee
        case .leftAnkle: return .leftAnkle
        case .rightAnkle: return .rightAnkle
        case .root: return .root
        }
    }

    /// The joints the `--rotate auto` rule looks at: mean wrist y against mean
    /// ankle y. The same four joints as `pose_mediapipe.WRIST_JOINTS` /
    /// `ANKLE_JOINTS`.
    public static let wristJoints: [VisionJoint] = [.leftWrist, .rightWrist]
    public static let ankleJoints: [VisionJoint] = [.leftAnkle, .rightAnkle]

    /// Every column name, in joint order — the row order of the CSV.
    public static var columnNames: [String] { allCases.map(\.columnName) }
}
