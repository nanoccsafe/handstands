// Turning the frames a video track stores into the frames a person sees.
//
// A phone held sideways writes landscape frames plus a `preferredTransform`
// matrix; the runner has to apply it before inference, so a 1024x576 clip with
// rotation -90 is decoded as 576x1024 — exactly what OpenCV does on the Linux
// side, which is what keeps the two sets of keypoints in the same coordinates.
//
// `DisplayTransform` resolves that matrix into a quarter turn plus the pixel
// transform that performs it, and `DisplayOrientation` is the same rotation
// expressed the way Vision's `CGImagePropertyOrientation` wants it (unused by
// the pixel-buffer path, but it is the reading a reviewer will check the
// matrix against, and it is testable without decoding anything).

import Foundation
import ImageIO

/// A 2x3 affine matrix, `x' = a*x + c*y + tx`, `y' = b*x + d*y + ty`.
///
/// The same convention as `CGAffineTransform`, spelled out as plain numbers so
/// the tables in this file can be unit tested on their own.
public struct AffineTransform2D: Equatable, Sendable {
    public var a: Double
    public var b: Double
    public var c: Double
    public var d: Double
    public var tx: Double
    public var ty: Double

    public init(a: Double, b: Double, c: Double, d: Double, tx: Double, ty: Double) {
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.tx = tx
        self.ty = ty
    }

    /// Where the matrix sends a point.
    public func applied(to point: PixelPoint) -> PixelPoint {
        PixelPoint(
            x: a * point.x + c * point.y + tx,
            y: b * point.x + d * point.y + ty
        )
    }
}

/// How a video track's stored frames relate to its display (upright) frames.
public struct DisplayTransform: Equatable, Sendable {
    /// Quarter turns clockwise: 0 = as stored, 1 = 90° CW, 2 = 180°, 3 = 90° CCW.
    public let quarterTurns: Int
    /// Size of the frames the video track stores.
    public let storedSize: PixelSize
    /// Size of the frames a person sees, and therefore of every coordinate written.
    public let displaySize: PixelSize

    /// - Throws: ``VisionPoseError/invalidQuarterTurns(_:)`` for a turn outside
    ///   `0...3` and ``VisionPoseError/invalidFrameSize(width:height:)`` for a
    ///   non-positive frame.
    public init(quarterTurns: Int, storedSize: PixelSize) throws {
        guard (0...3).contains(quarterTurns) else {
            throw VisionPoseError.invalidQuarterTurns(quarterTurns)
        }
        guard storedSize.width >= 1, storedSize.height >= 1 else {
            throw VisionPoseError.invalidFrameSize(width: storedSize.width, height: storedSize.height)
        }
        self.quarterTurns = quarterTurns
        self.storedSize = storedSize
        // A quarter turn swaps the axes, the half turn does not — the same rule
        // as `handstand.rotation.rotated_frame_size`.
        self.displaySize = (quarterTurns % 2 == 0)
            ? storedSize
            : PixelSize(width: storedSize.height, height: storedSize.width)
    }

    /// Is the stored frame already upright? Then it can be handed to Vision as
    /// it comes out of the decoder, with no copy.
    public var isIdentity: Bool { quarterTurns == 0 }

    /// The transform that turns a stored frame into a display frame, in the
    /// coordinate system `CIImage(cvPixelBuffer:)` works in: origin at the
    /// **bottom** left, `y` growing **upwards**, the image spanning
    /// `0...width` by `0...height`.
    ///
    /// Note that this is not the same as "rotate the image by N quarter turns"
    /// in that space — Core Image already reads the buffer upright, so a
    /// visual 90° clockwise turn is the matrix that sends `(x, y)` to
    /// `(y, width - x)` rather than the textbook counter-clockwise one.
    public var pixelTransform: AffineTransform2D {
        let width = Double(storedSize.width)
        let height = Double(storedSize.height)
        switch quarterTurns {
        case 0:
            return AffineTransform2D(a: 1, b: 0, c: 0, d: 1, tx: 0, ty: 0)
        case 1:
            return AffineTransform2D(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: width)
        case 2:
            return AffineTransform2D(a: -1, b: 0, c: 0, d: -1, tx: width, ty: height)
        default:
            return AffineTransform2D(a: 0, b: 1, c: -1, d: 0, tx: height, ty: 0)
        }
    }

    /// Resolve a video track's `preferredTransform` into a quarter turn.
    ///
    /// `a, b, c, d` are the matrix's linear part in `CGAffineTransform`'s
    /// parameter order, so the matrix maps `(x, y)` to `(a*x + c*y, b*x + d*y)`.
    /// Only the four axis-aligned rotations of the 90° multiples are accepted:
    /// the frame size can only be resolved from those, and a matrix that is
    /// really a scale or a shear is a bug in the container, not a pose.
    ///
    /// - Throws: ``VisionPoseError/mirroredDisplayTransform`` when the
    ///   determinant is negative, and
    ///   ``VisionPoseError/unsupportedDisplayTransform(a:b:c:d:)`` for anything
    ///   that is not a 90° multiple.
    public static func quarterTurns(
        a: Double,
        b: Double,
        c: Double,
        d: Double,
        storedSize: PixelSize
    ) throws -> DisplayTransform {
        // The linear parts of the four quarter turns, in clockwise order.
        let rotations: [(Int, [Double])] = [
            (0, [1, 0, 0, 1]),
            (1, [0, 1, -1, 0]),
            (2, [-1, 0, 0, -1]),
            (3, [0, -1, 1, 0]),
        ]
        let parts = [a, b, c, d]
        for (turns, expected) in rotations
        where zip(parts, expected).allSatisfy({ abs($0 - $1) < Self.linearPartTolerance }) {
            return try DisplayTransform(quarterTurns: turns, storedSize: storedSize)
        }
        // A negative determinant means the image is flipped, which is a
        // different operation and is not something phone footage does.
        if a * d - b * c < 0 {
            throw VisionPoseError.mirroredDisplayTransform
        }
        throw VisionPoseError.unsupportedDisplayTransform(a: a, b: b, c: c, d: d)
    }

    /// How much the linear part may differ from a quarter turn and still count
    /// as one. The matrix comes out of the container as floats, so it is never
    /// exactly 0 or 1; 1e-6 is far below anything a real matrix deviates by.
    static let linearPartTolerance = 1e-6
}

/// The same rotation as a `CGImagePropertyOrientation`.
///
/// Documented and tested next to the matrix tables so the two readings of
/// `preferredTransform` can be checked against each other: `.right` is the
/// 90° clockwise turn these phone clips carry, which is the reading Apple's own
/// header gives ("to correct an image with right orientation for display,
/// rotate it 90° clockwise").
public enum DisplayOrientation {
    /// The four non-mirrored `CGImagePropertyOrientation` cases, in clockwise
    /// quarter turns: 0 = `.up`, 1 = `.right`, 2 = `.down`, 3 = `.left`.
    public static let clockwiseQuarterTurns: [CGImagePropertyOrientation] = [
        .up, .right, .down, .left,
    ]

    /// The orientation that turns a stored frame upright, or `nil` when the
    /// transform mirrors (a case ``DisplayTransform`` refuses).
    public static func orientation(for transform: DisplayTransform) -> CGImagePropertyOrientation? {
        clockwiseQuarterTurns[transform.quarterTurns]
    }
}
