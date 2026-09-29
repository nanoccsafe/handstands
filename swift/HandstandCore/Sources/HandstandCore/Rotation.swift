import Foundation

/// Something the rotation maths refuses: today only a frame that cannot exist.
public enum RotationError: Error, Equatable, Sendable {
    /// `width` or `height` below 1, as `handstand.rotation`'s "frame size must
    /// be positive" `ValueError`.
    case nonPositiveFrameSize(width: Int, height: Int)
}

extension RotationError: CustomStringConvertible {
    public var description: String {
        switch self {
        case let .nonPositiveFrameSize(width, height):
            return "frame size must be positive, got \(width)x\(height)"
        }
    }
}

/// Port of `pipeline/handstand/rotation.py`: mapping keypoints between frame
/// orientations.
///
/// Conventions, identical to the Python:
///
/// * Angles are degrees **counter-clockwise** in image coordinates (`x` right,
///   `y` down), i.e. `angleDegrees: 90` does what `cv2.ROTATE_90_COUNTERCLOCKWISE`
///   does to the pixels.
/// * `width`/`height` always describe the frame the *input* points live in.
///   90° and 270° swap the frame size (`rotatedFrameSize`); other angles keep it.
/// * A 180° rotation of the frame flips both indices: `(x, y)` lands on
///   `(width - 1 - x, height - 1 - y)` of the rotated frame.
/// * Angles are normalised into `[0, 360)` the way Python's `%` does it, so
///   `-180` and `180`, `450` and `90` are the same rotation.
/// * `visibility` rides along untouched: it describes the detection, not the
///   orientation of the frame it was detected in.
public enum Rotation {
    /// `(width, height)` of the frame `rotate` produces: 90° and 270° swap the
    /// axes, every other angle keeps them. The counterpart of
    /// `handstand.rotation.rotated_frame_size`.
    public static func rotatedFrameSize(
        angleDegrees: Double,
        width: Int,
        height: Int
    ) throws -> (width: Int, height: Int) {
        guard width >= 1, height >= 1 else {
            throw RotationError.nonPositiveFrameSize(width: width, height: height)
        }
        let angle = normalized(angleDegrees)
        if angle == 90.0 || angle == 270.0 {
            return (height, width)
        }
        return (width, height)
    }

    /// Map points from a `width x height` frame into the coordinate system of
    /// the rotated frame — `handstand.rotation.rotate_points`.
    ///
    /// 90° multiples map pixel indices exactly (integer maths in a float
    /// costume); any other angle rotates about the frame centre and keeps the
    /// frame size, exactly as the Python does.
    public static func rotate(
        _ points: [Keypoint],
        angleDegrees: Double,
        width: Int,
        height: Int
    ) throws -> [Keypoint] {
        guard width >= 1, height >= 1 else {
            throw RotationError.nonPositiveFrameSize(width: width, height: height)
        }
        let angle = normalized(angleDegrees)
        let w = Double(width)
        let h = Double(height)

        return points.map { point in
            let x = point.x
            let y = point.y
            let rotated: (x: Double, y: Double)
            switch angle {
            case 0.0:
                rotated = (x, y)
            case 90.0:
                rotated = (y, (w - 1) - x)
            case 180.0:
                rotated = ((w - 1) - x, (h - 1) - y)
            case 270.0:
                rotated = ((h - 1) - y, x)
            default:
                // Non-cardinal angles: about the frame centre, same frame size.
                let theta = angle * Double.pi / 180.0
                let c = cos(theta)
                let s = sin(theta)
                let centreX = (w - 1) / 2.0
                let centreY = (h - 1) / 2.0
                let deltaX = x - centreX
                let deltaY = y - centreY
                rotated = (
                    centreX + deltaX * c + deltaY * s,
                    centreY - deltaX * s + deltaY * c
                )
            }
            return Keypoint(x: rotated.x, y: rotated.y, visibility: point.visibility)
        }
    }

    /// Undo `rotate` — `handstand.rotation.inverse_rotate_points`.
    ///
    /// `points` must be in the frame `rotate(..., angleDegrees, width, height)`
    /// produced; the result is back in the original `width x height` frame.
    /// `width`/`height` always refer to the *original* frame, so rotating and
    /// inverse-rotating is the identity for every supported angle.
    public static func inverseRotate(
        _ points: [Keypoint],
        angleDegrees: Double,
        width: Int,
        height: Int
    ) throws -> [Keypoint] {
        let angle = normalized(angleDegrees)
        let size = try rotatedFrameSize(angleDegrees: angle, width: width, height: height)
        return try rotate(points, angleDegrees: -angle, width: size.width, height: size.height)
    }

    /// The angle into `[0, 360)`, with Python's `%` sign rules: `-90` becomes
    /// `270`, `-180` becomes `180`.
    static func normalized(_ angleDegrees: Double) -> Double {
        let angle = angleDegrees.truncatingRemainder(dividingBy: 360.0)
        return angle < 0 ? angle + 360.0 : angle
    }
}
