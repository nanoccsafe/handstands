// Display orientation, which is the caller's job.
//
// A phone-held-sideways clip stores landscape frames plus a
// `preferredTransform`; a viewer sees the rotation of it. Who applies that
// rotation decides where the keypoints live, so the rule is simple:
//
// * the **caller** decodes and turns the stored frames into display frames —
//   the runner does it while reading the track, the app will do it over the
//   frames it feeds the analysis of chainlink #47;
// * ``PoseService`` only ever sees display frames, and every coordinate it
//   returns is in that frame's pixels.
//
// This is the runner's `DisplayOrienter`, moved out so both callers share it:
// `nil` when the track is already upright (use the decoded buffer as it is),
// a freshly rendered buffer of `transform.displaySize` otherwise.

import CoreImage
import CoreVideo
import Foundation
import VisionPoseCore

/// Turning decoded frames into display-orientation frames.
public enum DisplayFrames {
    /// One `CIContext` for every frame of a clip: building one per frame would
    /// cost milliseconds each, on a clip with thousands of frames.
    ///
    /// `nonisolated(unsafe)` is the price of a process-wide cache under Swift 6
    /// strict concurrency; `lock` (a `Sendable` `NSLock`) is what actually makes
    /// it safe — no two threads ever touch the cache at once, and the context
    /// itself is only ever handed to `render(_:to:)`.
    private nonisolated(unsafe) static var cachedContext: CIContext?
    private static let lock = NSLock()

    private static func context() -> CIContext {
        lock.lock()
        defer { lock.unlock() }
        if let cachedContext { return cachedContext }
        // Working in a known colour space keeps the render from shifting the
        // colours; sRGB is what the pixel buffers are.
        let created = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
        cachedContext = created
        return created
    }

    /// The runner's display-orientation conversion: a decoded buffer + the
    /// track's `preferredTransform` → a display-oriented buffer.
    ///
    /// `nil` when the stored frame is already upright, in which case the
    /// decoder's own buffer is used as it is — no copy, no render.
    ///
    /// - Throws: ``PoseServiceError/pixelBufferAllocation(width:height:)``
    ///   when the display-sized buffer cannot be made.
    public static func displayBuffer(from pixelBuffer: CVPixelBuffer, transform: DisplayTransform)
        throws -> CVPixelBuffer?
    {
        if transform.isIdentity { return nil }
        let matrix = transform.pixelTransform
        let image = CIImage(cvPixelBuffer: pixelBuffer).transformed(
            by: CGAffineTransform(
                a: matrix.a, b: matrix.b, c: matrix.c,
                d: matrix.d, tx: matrix.tx, ty: matrix.ty
            )
        )
        let displaySize = transform.displaySize
        var rotated: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            displaySize.width,
            displaySize.height,
            kCVPixelFormatType_32BGRA,
            nil,
            &rotated
        )
        guard status == kCVReturnSuccess, let rotated else {
            throw PoseServiceError.pixelBufferAllocation(
                width: displaySize.width,
                height: displaySize.height
            )
        }
        context().render(image, to: rotated)
        return rotated
    }
}
