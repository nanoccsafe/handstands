// Point and frame geometry shared by the runner.
//
// Two coordinate systems are in play and mixing them up is the one mistake
// that would silently produce sideways or upside-down keypoints:
//
// * **Normalised** — what Vision reports. `x` runs 0...1 left to right, `y` runs
//   0...1 **bottom to top** (the origin is the bottom-left corner, not the
//   top-left one like every image library).
// * **Pixels** — what the pipeline stores: display-frame pixel indices with the
//   origin at the **top** left, `x` right, `y` down, `0 <= x <= width - 1`.
//
// `CoordinateMath` is the only place the conversion between them happens, and
// it is the Swift twin of `normalized_to_pixels` +
// `handstand.rotation.inverse_rotate_points` in the Python pipeline. Both use
// `size - 1` so that a point and its mirrored normalised coordinates land on
// the same pixel.

import Foundation

/// A point in Vision's normalised space: `0...1`, origin **bottom**-left.
public struct NormalizedPoint: Equatable, Sendable {
    public var x: Double
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

/// A pixel index in a frame, origin **top**-left, `y` growing downwards.
public struct PixelPoint: Equatable, Sendable {
    public var x: Double
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

/// A frame's size in pixels. Always the *display* size unless stated otherwise.
public struct PixelSize: Equatable, Sendable, CustomStringConvertible {
    public var width: Int
    public var height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    public var description: String { "\(width)x\(height)" }
}

/// Everything that can go wrong in the pure part of the runner.
public enum VisionPoseError: Error, Equatable, CustomStringConvertible {
    /// A frame of zero (or negative) size was handed in.
    case invalidFrameSize(width: Int, height: Int)
    /// A quarter turn outside 0...3 was asked for.
    case invalidQuarterTurns(Int)
    /// The video's `preferredTransform` is not a plain 90° multiple, so the
    /// display orientation cannot be resolved by rotating.
    case unsupportedDisplayTransform(a: Double, b: Double, c: Double, d: Double)
    /// The video's `preferredTransform` mirrors the image. Phone footage is
    /// never mirrored and a mirrored frame would need a flip as well as a
    /// rotation, so this is refused rather than guessed at.
    case mirroredDisplayTransform

    public var description: String {
        switch self {
        case .invalidFrameSize(let width, let height):
            return "frame size must be positive, got \(width)x\(height)"
        case .invalidQuarterTurns(let turns):
            return "quarter turns must be 0...3, got \(turns)"
        case .unsupportedDisplayTransform(let a, let b, let c, let d):
            return """
                unsupported video rotation: the transform's linear part is \
                (a: \(a), b: \(b), c: \(c), d: \(d)), which is not a 90° multiple
                """
        case .mirroredDisplayTransform:
            return "mirrored video rotation is not supported"
        }
    }
}

/// Normalised <-> pixel conversions and the 180° map-back.
public enum CoordinateMath {
    /// Vision's normalised point -> pixel indices of a `size` frame.
    ///
    /// `x_px = x * (width - 1)` and `y_px = (1 - y) * (height - 1)`: the `1 - y`
    /// is the flip from Vision's bottom-left origin to the pipeline's top-left
    /// one, and the `size - 1` puts the frame edges on pixel indices `0` and
    /// `size - 1` — the same indexing the rotation formula below uses, so the
    /// two are exact inverses of each other.
    public static func normalizedToPixels(_ point: NormalizedPoint, in size: PixelSize) throws -> PixelPoint {
        try validate(size)
        return PixelPoint(
            x: point.x * Double(size.width - 1),
            y: (1 - point.y) * Double(size.height - 1)
        )
    }

    /// A display-frame point -> the same point in the frame turned 180°.
    ///
    /// A half turn flips both axes, so `x` and `y` are mirrored about the frame
    /// centre. This is `handstand.rotation.rotate_points(..., 180, ...)`; it is
    /// its own inverse.
    public static func rotateHalfTurn(_ point: PixelPoint, in size: PixelSize) throws -> PixelPoint {
        try validate(size)
        return PixelPoint(
            x: Double(size.width - 1) - point.x,
            y: Double(size.height - 1) - point.y
        )
    }

    /// A 180°-rotated-frame point -> the display frame, i.e. the map-back.
    ///
    /// The twin of ``rotateHalfTurn(_:in:)`` and the reason rotated
    /// coordinates never leave the runner: the pipeline only ever stores
    /// display pixels. It is its own inverse, so
    /// `mapBackHalfTurn(rotateHalfTurn(p)) == p` for every point and size.
    public static func mapBackHalfTurn(_ point: PixelPoint, in size: PixelSize) throws -> PixelPoint {
        try validate(size)
        return PixelPoint(
            x: Double(size.width - 1) - point.x,
            y: Double(size.height - 1) - point.y
        )
    }

    private static func validate(_ size: PixelSize) throws {
        guard size.width >= 1, size.height >= 1 else {
            throw VisionPoseError.invalidFrameSize(width: size.width, height: size.height)
        }
    }
}
