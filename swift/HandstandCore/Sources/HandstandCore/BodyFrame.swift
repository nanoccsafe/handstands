/// The body frame: the coordinate system every later measurement is taken in.
///
/// A port of `pipeline/handstand/bodyframe.py`, down to the formulas:
///
/// * the **origin** is the *wrist midpoint* — where a handstand's hands meet
///   the floor, and the only point of an inverted athlete that stays put;
/// * `u` runs to the right, like the display frame's `x`;
/// * `v` runs **up**, i.e. display `y` with the sign flipped, because a
///   handstand is upside down: a positive `v` always means "further from the
///   floor";
/// * both are divided by the clip's body length `L`, so a 500 px tall athlete
///   and a 350 px one are measured in the same numbers.
///
/// So the wrist midpoint itself is `(0, 0)`, a point one body length above it
/// is `(0, 1)` and a point one body length to the right of it is `(1, 0)`.
public enum BodyFrame {
    /// Something `toBodyFrame` refuses: a scale the body frame cannot be
    /// measured in.
    public enum BodyFrameError: Error, Equatable, Sendable {
        /// Zero, negative, `NaN` or infinite `bodyLength`, matching
        /// `handstand.bodyframe`'s "body_length must be a positive, finite
        /// number" `ValueError`.
        case invalidBodyLength(Double)
    }

    /// The mean of two keypoints — `handstand.bodyframe.midpoint`, which the
    /// shoulder and hip midpoints and `wrist_midpoint` (the body frame's
    /// origin) are both built from.
    ///
    /// `visibility` is the *smaller* of the two: a midpoint is only as
    /// visible as its least visible half, which is what a later gate wants.
    public static func midpoint(_ first: Keypoint, _ second: Keypoint) -> Keypoint {
        Keypoint(
            x: (first.x + second.x) / 2.0,
            y: (first.y + second.y) / 2.0,
            visibility: Swift.min(first.visibility, second.visibility)
        )
    }

    /// One display-frame point in the body frame: `(u, v)`, measured from the
    /// wrist midpoint, in units of `bodyLength` — `handstand.bodyframe.to_body_frame`.
    ///
    /// `u` is `x - wristMidX` and `v` is `wristMidY - y`, both divided by
    /// `bodyLength`. Throws `BodyFrameError.invalidBodyLength` for a scale
    /// that is not a positive finite number, so the mistake surfaces in the
    /// stage that made it instead of silently becoming `NaN` downstream.
    public static func toBodyFrame(
        x: Double,
        y: Double,
        wristMidX: Double,
        wristMidY: Double,
        bodyLength: Double
    ) throws -> (u: Double, v: Double) {
        let scale = try checkedLength(bodyLength)
        return ((x - wristMidX) / scale, (wristMidY - y) / scale)
    }

    /// A whole batch of display-frame points in the body frame —
    /// `handstand.bodyframe.body_frame_points`, for one clip's joints or a
    /// trajectory. The scale is checked once, up front.
    public static func bodyFramePoints(
        _ points: [Keypoint],
        wristMid: (x: Double, y: Double),
        bodyLength: Double
    ) throws -> [(u: Double, v: Double)] {
        let scale = try checkedLength(bodyLength)
        return points.map { point in
            ((point.x - wristMid.x) / scale, (wristMid.y - point.y) / scale)
        }
    }

    /// `bodyLength` as a usable scale: finite and above zero, or a
    /// `BodyFrameError.invalidBodyLength`.
    static func checkedLength(_ bodyLength: Double) throws -> Double {
        guard bodyLength.isFinite, bodyLength > 0 else {
            throw BodyFrameError.invalidBodyLength(bodyLength)
        }
        return bodyLength
    }
}

extension BodyFrame.BodyFrameError: CustomStringConvertible {
    public var description: String {
        switch self {
        case let .invalidBodyLength(value):
            return "body_length must be a positive, finite number, got \(value)"
        }
    }
}
