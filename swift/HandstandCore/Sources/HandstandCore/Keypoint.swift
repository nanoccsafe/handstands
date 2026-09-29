/// One joint in one frame: display pixels, `y` down, as `docs/keypoint_schema.md`
/// defines them. `visibility` is the detector's score in `0...1`, the same
/// column the parquet carries.
public struct Keypoint: Sendable, Equatable, Codable {
    /// Horizontal position in display pixels; rightwards.
    public var x: Double
    /// Vertical position in display pixels; downwards.
    public var y: Double
    /// Detector confidence in `0...1` (MediaPipe's `visibility`).
    public var visibility: Double

    public init(x: Double, y: Double, visibility: Double) {
        self.x = x
        self.y = y
        self.visibility = visibility
    }
}

/// One frame of a clip: which frame, when it was shown to the model, and the
/// keypoints that were found. Value semantics, `Sendable`, and Codable as a
/// JSON object keyed by the joint names — the shape the parity fixtures of
/// chainlink #25/#42 are written in:
///
/// ```json
/// {"frameIndex": 0, "tMs": 0, "joints": {"nose": {"x": 1.0, "y": 2.0, "visibility": 0.9}}}
/// ```
///
/// The dictionary is Codable *by hand* rather than by synthesis: a synthesized
/// `[Joint: Keypoint]` Codable encodes as a JSON array of alternating keys and
/// values, which no fixture on the Python side would ever produce.
public struct PoseFrame: Sendable, Equatable {
    /// Index of the frame in the clip, `0`-based.
    public var frameIndex: Int
    /// Presentation timestamp of the frame, milliseconds from the start of the
    /// clip — the same `tMs` the parquet carries.
    public var tMs: Int
    /// The joints found in this frame. A frame whose source did not report a
    /// joint (Apple Vision has no `foot_index`) simply has no entry for it.
    public var joints: [Joint: Keypoint]

    public init(frameIndex: Int, tMs: Int, joints: [Joint: Keypoint]) {
        self.frameIndex = frameIndex
        self.tMs = tMs
        self.joints = joints
    }

    /// The midpoint of two joints of this frame, in display pixels — the
    /// shoulder and hip pairs the body length is built from, and the wrist
    /// pair that is the body frame's origin (`handstand.bodyframe.wrist_midpoint`).
    ///
    /// `nil` when either joint is missing from the frame: a midpoint of one
    /// side would be a side, not a midpoint. Its `visibility` is the *smaller*
    /// of the two, so a midpoint is only as visible as its least visible half.
    public func midpoint(_ first: Joint, _ second: Joint) -> Keypoint? {
        guard let a = joints[first], let b = joints[second] else { return nil }
        return BodyFrame.midpoint(a, b)
    }

    /// Whether this frame carries the whole tracked schema and every joint
    /// clears `minVisibility` — the strict frame-level gate the tests and the
    /// later parity fixtures use.
    ///
    /// A joint that is missing, or whose `visibility` is below the threshold
    /// (a `NaN` fails the comparison, as it should), makes the frame invalid.
    /// Per-joint gating downstream reads `Keypoint.visibility` directly; this
    /// is the all-or-nothing "is this frame usable at all" test.
    public func isValid(minVisibility: Double) -> Bool {
        guard joints.count == Joint.allCases.count else { return false }
        return joints.values.allSatisfy { $0.visibility >= minVisibility }
    }
}

extension PoseFrame: Codable {
    private enum CodingKeys: String, CodingKey {
        case frameIndex
        case tMs
        case joints
    }

    /// Coding key for one joint: whatever the JSON object's key is, checked
    /// against `Joint(rawValue:)` on the way in.
    private struct JointKey: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }

        init(_ jointName: String) {
            stringValue = jointName
        }

        init?(stringValue: String) {
            self.stringValue = stringValue
        }

        init?(intValue: Int) {
            return nil
        }
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let frameIndex = try container.decode(Int.self, forKey: .frameIndex)
        let tMs = try container.decode(Int.self, forKey: .tMs)
        let jointContainer = try container.nestedContainer(keyedBy: JointKey.self, forKey: .joints)
        var joints: [Joint: Keypoint] = [:]
        for key in jointContainer.allKeys {
            guard let joint = Joint(rawValue: key.stringValue) else {
                throw DecodingError.dataCorruptedError(
                    forKey: key,
                    in: jointContainer,
                    debugDescription: "unknown joint '\(key.stringValue)'"
                )
            }
            joints[joint] = try jointContainer.decode(Keypoint.self, forKey: key)
        }
        self.frameIndex = frameIndex
        self.tMs = tMs
        self.joints = joints
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(frameIndex, forKey: .frameIndex)
        try container.encode(tMs, forKey: .tMs)
        var jointContainer = container.nestedContainer(keyedBy: JointKey.self, forKey: .joints)
        for (joint, keypoint) in joints {
            try jointContainer.encode(keypoint, forKey: JointKey(joint.rawValue))
        }
    }
}
