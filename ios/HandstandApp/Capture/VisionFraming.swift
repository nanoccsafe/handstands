import CoreImage
import Foundation
import HandstandCore
import Vision

/// The shared schema's joint names, spelled out: `Vision` defines its own
/// `Joint` type (iOS 18+), so in this file — the one that imports both — the
/// schema's would otherwise be ambiguous for type lookup.
typealias SchemaJoint = HandstandCore.Joint

/// The framing guide's Vision half: one frame in, one `FramingStatus` out.
///
/// Two things happen here that would otherwise be invisible bugs:
///
/// * **Vision's coordinates are upside down.** `VNRecognizedPoint` is
///   normalised with the origin at the *bottom* left; the guide (and
///   `HandstandCore`) reads normalised coordinates with the origin at the
///   *top* left, like every image the user sees. `displayPoint` is the flip.
/// * **Vision reads an upside-down body badly**, and a handstand frame is
///   exactly that, so every frame is asked for twice — once `.up`, once
///   `.down` — and the pass that came back more sure of more joints is the
///   one kept. The `.down` pass reports its points in the rotated image's
///   space, so its points are mapped back through the same half turn
///   `swift/VisionPose`'s runner maps back (`CoordinateMath.mapBackHalfTurn`).
///
/// Everything runs on the engine's analysis queue — never the main actor.
enum VisionFraming {
    /// Vision joint names the shared schema knows. Vision's own `neck`,
    /// `root`, eyes and ears have no `Joint` case, and Vision has no
    /// `foot_index`, so those are dropped here rather than guessed at
    /// (`docs/keypoint_schema.md` explains the 15-joint schema).
    static let jointMap: [VNHumanBodyPoseObservation.JointName: SchemaJoint] = [
        .nose: .nose,
        .leftShoulder: .leftShoulder,
        .rightShoulder: .rightShoulder,
        .leftElbow: .leftElbow,
        .rightElbow: .rightElbow,
        .leftWrist: .leftWrist,
        .rightWrist: .rightWrist,
        .leftHip: .leftHip,
        .rightHip: .rightHip,
        .leftKnee: .leftKnee,
        .rightKnee: .rightKnee,
        .leftAnkle: .leftAnkle,
        .rightAnkle: .rightAnkle,
    ]

    /// One Vision point → display normalised coordinates.
    ///
    /// - Parameters:
    ///   - x: Vision's `x`, `0...1`, left to right *of the image Vision was
    ///     shown*.
    ///   - y: Vision's `y`, `0...1`, bottom to top of that image.
    ///   - flipped: `true` when the request ran with orientation `.down`.
    ///     Vision then reports in the 180°-rotated image's space, so both
    ///     axes are turned back before the origin flip.
    /// - Returns: the same point in the guide's space — origin top left,
    ///   `y` down, of the buffer as it was delivered.
    static func displayPoint(x: Double, y: Double, flipped: Bool) -> (x: Double, y: Double) {
        let imageX = flipped ? 1 - x : x
        let imageY = flipped ? 1 - y : y
        return (imageX, 1 - imageY)
    }

    /// How many joints a result is sure of — the score the two orientations
    /// are compared with ("keep the result with more confident joints").
    static func confidentJointCount(_ people: [[SchemaJoint: Keypoint]]) -> Int {
        people.reduce(0) { total, person in
            total + person.values.filter { $0.visibility >= FramingCheck.confidentThreshold }.count
        }
    }
}

/// Runs `VNDetectHumanBodyPoseRequest` on a frame, twice, and turns the
/// better pass into a `FramingStatus`.
///
/// A struct on purpose: it owns no state that survives one frame except the
/// `CIContext` (which is explicitly thread-safe to share, and here is only
/// ever used from the analysis queue anyway).
struct FramingAnalyzer {
    /// The longest side a frame is scaled down to before Vision sees it.
    /// `1080 × 1920` becomes `360 × 640` — a ninth of the pixels for a
    /// decision made five times a second, at no cost to the normalised
    /// coordinates the verdict is built from.
    static let maxDimension: CGFloat = 640

    private let context = CIContext(options: [.cacheIntermediates: false])

    /// The verdict for one camera frame, or `nil` when the frame could not
    /// be prepared at all — the caller keeps the previous status rather than
    /// flashing red at a glitch.
    func analyse(_ pixelBuffer: CVPixelBuffer) -> FramingStatus? {
        guard let frame = downscaled(pixelBuffer, maxDimension: Self.maxDimension) else {
            return nil
        }
        let upright = detect(in: frame, orientation: .up)
        let inverted = detect(in: frame, orientation: .down)
        let people = VisionFraming.confidentJointCount(inverted) > VisionFraming.confidentJointCount(upright)
            ? inverted
            : upright
        return FramingCheck.status(people: people)
    }

    /// A copy of `source` scaled so its longest side is `maxDimension`
    /// (unchanged when it is already smaller), as a buffer Vision can read.
    func downscaled(_ source: CVPixelBuffer, maxDimension: CGFloat) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        guard width > 0, height > 0 else { return nil }
        let longest = max(width, height)
        guard longest > 0, CGFloat(longest) > maxDimension else { return source }

        let scale = maxDimension / CGFloat(longest)
        let targetWidth = max(1, Int((CGFloat(width) * scale).rounded()))
        let targetHeight = max(1, Int((CGFloat(height) * scale).rounded()))

        var destination: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        let result = CVPixelBufferCreate(
            kCFAllocatorDefault,
            targetWidth,
            targetHeight,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &destination
        )
        guard result == kCVReturnSuccess, let destination else { return nil }

        let bounds = CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight)
        let image = CIImage(cvPixelBuffer: source)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .cropped(to: bounds)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        context.render(image, to: destination, bounds: bounds, colorSpace: colorSpace)
        return destination
    }

    /// Vision's answer for one orientation, as display-space people.
    private func detect(
        in buffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> [[SchemaJoint: Keypoint]] {
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: orientation, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return []
        }
        return (request.results ?? []).map { observation in
            let recognized = (try? observation.recognizedPoints(.all)) ?? [:]
            var joints: [SchemaJoint: Keypoint] = [:]
            for (name, point) in recognized {
                guard let joint = VisionFraming.jointMap[name] else { continue }
                let display = VisionFraming.displayPoint(
                    x: Double(point.location.x),
                    y: Double(point.location.y),
                    flipped: orientation == .down
                )
                joints[joint] = Keypoint(x: display.x, y: display.y, visibility: Double(point.confidence))
            }
            return joints
        }
    }
}
