// The `--rotate auto` rule, in the same form as the MediaPipe runner's.
//
// Pose models are worst at inverted bodies, so a frame is turned 180° before
// inference when the body looks upside down. The judgement is made from the
// *previous* frame — the current one has not been seen yet — and from the
// person whose wrists are lowest in the image, i.e. whoever is most likely the
// one on their hands:
//
//     mean y of both wrists > mean y of both ankles  =>  rotate
//
// which is exactly `pose_mediapipe.is_inverted` / `lowest_wrist_pose`. Keeping
// the two implementations in step matters: the bake-off (chainlink #16) only
// means anything if the same frames were rotated for both.

import Foundation

/// One person's joints, in **display-frame** pixels.
public struct DisplayPose: Equatable, Sendable {
    /// The joints Vision reported for this person. A joint Vision did not
    /// report is simply absent.
    public var points: [VisionJoint: PixelPoint]

    public init(points: [VisionJoint: PixelPoint]) {
        self.points = points
    }
}

/// Which frame to feed to the model next.
public struct AutoRotation: Equatable, Sendable {
    /// Is the next frame rotated 180°? The first frame never is.
    public private(set) var rotateNextFrame: Bool

    public init(rotateNextFrame: Bool = false) {
        self.rotateNextFrame = rotateNextFrame
    }

    /// Judge the next frame from the people just seen.
    ///
    /// A frame with nobody in it keeps the previous decision — the model has
    /// said nothing about the orientation, and flipping every frame of a
    /// one-frame detection dropout would be the wrong answer.
    public mutating func update(with people: [DisplayPose]) {
        guard let chosen = Self.chosenPose(people) else { return }
        rotateNextFrame = Self.isInverted(chosen)
    }

    /// The person the rule is judged from: the one whose wrists sit lowest.
    ///
    /// People whose wrists are not both visible cannot be ranked and are
    /// skipped; when nobody can be ranked the first person is used, so a
    /// single-person frame is judged exactly as it always was. `nil` means
    /// nobody was in the frame.
    public static func chosenPose(_ people: [DisplayPose]) -> DisplayPose? {
        guard let first = people.first else { return nil }
        guard people.count > 1 else { return first }
        let scores = people.map { meanWristY($0) }
        let rankable = scores.indices.filter { scores[$0].isFinite }
        guard let best = rankable.max(by: { scores[$0] < scores[$1] }) else { return first }
        return people[best]
    }

    /// Does this body look upside down? Mean wrist y greater (lower in the
    /// image) than mean ankle y.
    ///
    /// A pose missing either wrist or either ankle cannot be judged, and counts
    /// as "not inverted" — the same as `pose_mediapipe.is_inverted`, which
    /// returns `False` for non-finite coordinates.
    public static func isInverted(_ pose: DisplayPose) -> Bool {
        guard let wrists = meanY(pose, VisionJoint.wristJoints),
            let ankles = meanY(pose, VisionJoint.ankleJoints)
        else { return false }
        return wrists > ankles
    }

    /// Mean y of both wrists, or `nil` when either wrist is missing. Bigger
    /// means lower in the image, which is what sorts the people.
    public static func meanWristY(_ pose: DisplayPose) -> Double {
        meanY(pose, VisionJoint.wristJoints) ?? .nan
    }

    /// Mean y of the given joints, or `nil` if any of them is missing.
    public static func meanY(_ pose: DisplayPose, _ joints: [VisionJoint]) -> Double? {
        var total = 0.0
        for joint in joints {
            guard let point = pose.points[joint], point.x.isFinite, point.y.isFinite else {
                return nil
            }
            total += point.y
        }
        return total / Double(joints.count)
    }
}
