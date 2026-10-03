import CoreVideo
import Foundation
import MediaPipeTasksVision

// --------------------------------------------------------------------------- #
// The real MediaPipe landmarker (chainlink #45), behind
// `MediaPipeLandmarkDetecting` — the only file in the app that imports the
// `MediaPipeTasksVision` pod, so everything else (the clip extractor, its
// tests, the parity harness) compiles and runs without it.
//
// The settings are `pipeline/handstand/pose_mediapipe.py`'s, one for one:
// `pose_landmarker_full.task`, VIDEO running mode with millisecond
// timestamps, MediaPipe's own 0.5 confidences, `numPoses = 1` (the app
// assumes one athlete — no trainer logic on the device). Two instances are
// built per clip — the upright pass and the rotated pass — so a tracker that
// saw one orientation never carries that state into the other, the same
// reason Python's `landmarker_factory()` is called twice for `--rotate best`.
// --------------------------------------------------------------------------- #

/// One VIDEO-mode `PoseLandmarker` over `pose_landmarker_full.task`.
///
/// `@unchecked Sendable` because the protocol requires it and the guarantee
/// is structural: one pass owns one instance and calls it sequentially from
/// the extraction's task — nothing ever shares it (the same shape as
/// `VideoFrameReader`'s own `@unchecked Sendable`).
final class MediaPipeLandmarker: MediaPipeLandmarkDetecting, @unchecked Sendable {
    private let landmarker: PoseLandmarker

    /// - Throws: the pod's error when the model cannot be opened (a missing
    ///   or unreadable `.task` file).
    init(modelURL: URL) throws {
        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelURL.path
        // MediaPipe's own defaults, spelled out because they *are* the
        // pipeline's settings: VIDEO mode (the detector tracks between
        // frames), 0.5 at every gate (`DEFAULT_MIN_*_CONFIDENCE` in
        // pose_mediapipe.py) and a single body per frame.
        options.runningMode = .video
        options.numPoses = 1
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        landmarker = try PoseLandmarker(options: options)
    }

    func detectForVideo(_ frame: CVPixelBuffer, tMs: Int) throws -> [MediaPipeLandmark]? {
        let image = try MPImage(pixelBuffer: frame)
        let result = try landmarker.detect(videoFrame: image, timestampInMilliseconds: tMs)
        // `numPoses = 1`: at most one pose, and an empty list is "nobody" —
        // the `nil` the protocol spells that with. The pose carries all 33
        // landmarks; the extractor re-checks the count exactly as Python's
        // `_detect_pose` does.
        guard let first = result.landmarks.first else { return nil }
        return first.map { landmark in
            MediaPipeLandmark(
                x: Double(landmark.x),
                y: Double(landmark.y),
                z: Double(landmark.z),
                visibility: Self.score(landmark.visibility),
                presence: Self.score(landmark.presence)
            )
        }
    }

    /// `NormalizedLandmark.visibility`/`presence` arrive as `NSNumber?`; one
    /// overload exists for the importers that bridge them to `Double?`
    /// directly — whichever the compiler sees, "no score" reads `nil`.
    private static func score(_ value: NSNumber?) -> Double? { value?.doubleValue }
    private static func score(_ value: Double?) -> Double? { value }
}
