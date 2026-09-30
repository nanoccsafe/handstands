// The one pose API the app and the runner share.
//
// A `PoseService` is "one clip of video in, frames of the shared schema out":
// the caller decodes (and turns into display orientation — see
// ``DisplayFrames``), the service runs the model, maps the points and decides
// the `--rotate auto` state, and hands back `PostProcessInputFrame`, the input
// `HandstandCore.Analyzer.analyze` takes. The app (chainlink #47) and the
// macOS runner (`vision-pose`) run the *same* class, so what the bake-off
// compares is what the phone would run.
//
// The backend is injectable (``BodyPoseDetecting``) so the tests can feed
// canned points and no image of a person is ever needed.

import CoreVideo
import Foundation
import HandstandCore

/// A pose backend: one frame of **display** orientation in, one frame of the
/// shared schema out.
public protocol PoseService: AnyObject {
    /// The backend's identity: `"vision"` here, `"mediapipe"` for the backend
    /// chainlink #45 adds. The manifest and any future UI say which one ran.
    var backendName: String { get }

    /// Call before each new clip: resets the auto-rotation state, so clip two
    /// starts with the same "the first frame is not rotated" as clip one.
    func reset()

    /// One frame, ALREADY in display orientation (the orientation a person
    /// watching the video sees), in, one frame of the shared schema out.
    ///
    /// Rotation of the *stored* frames is the caller's job (``DisplayFrames``),
    /// not this method's: the frame handed in is what a viewer would see, and
    /// every coordinate that comes back is in that frame's pixels.
    func process(_ displayFrame: CVPixelBuffer, tMs: Int) throws -> PostProcessInputFrame
}

/// Everything that can go wrong in the kit: the Vision call itself, and the
/// buffer allocation of ``DisplayFrames``.
public enum PoseServiceError: Error, Equatable, CustomStringConvertible {
    /// Vision refused to run the request or to hand back a person's joints.
    /// The runner turns this into its own per-frame error, so the message it
    /// exits with names the frame.
    case visionFailed(String)
    /// The display-oriented buffer of a rotated frame could not be allocated.
    case pixelBufferAllocation(width: Int, height: Int)

    public var description: String {
        switch self {
        case .visionFailed(let message):
            return "could not read the body-pose points: \(message)"
        case .pixelBufferAllocation(let width, let height):
            return "could not allocate a \(width)x\(height) pixel buffer"
        }
    }
}
