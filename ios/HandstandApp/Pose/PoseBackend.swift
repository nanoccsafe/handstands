import Foundation
import HandstandCore
import VisionPoseKit

/// Which pose backend runs the detection.
///
/// The choice belongs to the bake-off (chainlink #16, docs/bakeoff.md), so
/// the app never hard-codes a backend further down: it asks for one.
/// MediaPipe won that bake-off — PCK@0.2 on clean line holds 0.76 vs 0.43 at
/// the common gate, and at the 0.5 gate Vision leaves 84/179 clips
/// unmeasurable against MediaPipe's 3 — so `preferred` is **MediaPipe
/// whenever its model is in the bundle**, and Apple Vision (built into iOS,
/// needs no download) is the fallback that always works.
///
/// There is no UI for this enum: `#47` asks `PoseBackend.preferred` for its
/// extraction and gets the backend that is actually available — MediaPipe's
/// `makeService()` hands back `MediaPipePoseService`, whose `extract` is the
/// clip-level run (chainlink #45), Vision's hands back `VisionPoseService`.
enum PoseBackend: String, CaseIterable {
    /// Apple Vision, through `VisionPoseKit`.
    case vision
    /// MediaPipe (chainlink #45): available when the model was fetched.
    case mediapipe

    /// The bundled `pose_landmarker_full.task`, or `nil` when the build has
    /// none — the model is never committed (`*.task` is git-ignored);
    /// `tools/ios/fetch_mediapipe_model.sh` puts it in `LocalResources`
    /// before a build. Looked up the three ways Xcode may have copied that
    /// optional folder (flattened to the bundle root, `models/`, or
    /// `LocalResources/models/`), the same double-take `ReferenceLoader`
    /// takes for the scoring reference.
    static var mediapipeModelURL: URL? {
        Bundle.main.url(forResource: "pose_landmarker_full", withExtension: "task")
            ?? Bundle.main.url(
                forResource: "pose_landmarker_full", withExtension: "task", subdirectory: "models")
            ?? Bundle.main.url(
                forResource: "pose_landmarker_full", withExtension: "task",
                subdirectory: "LocalResources/models")
    }

    /// Can this backend produce a service right now? MediaPipe's answer is
    /// exactly "the model file is in the bundle" (`modelURL` injectable so
    /// the tests can ask with and without one); Vision needs nothing.
    func isAvailable(modelURL: URL?) -> Bool {
        switch self {
        case .vision:
            return true
        case .mediapipe:
            guard let modelURL else { return false }
            return FileManager.default.fileExists(atPath: modelURL.path)
        }
    }

    /// Can this backend produce a service right now, in this build?
    var isAvailable: Bool { isAvailable(modelURL: Self.mediapipeModelURL) }

    /// The backend the analysis runs: MediaPipe when its model is bundled —
    /// chainlink #45's default, per the bake-off — Vision otherwise, so a
    /// build without the fetched model still analyses.
    static func preferred(modelURL: URL?) -> PoseBackend {
        mediapipe.isAvailable(modelURL: modelURL) ? .mediapipe : .vision
    }

    /// The backend the analysis runs, in this build.
    static var preferred: PoseBackend { preferred(modelURL: mediapipeModelURL) }

    /// What the backend is called in a log line or a future settings row.
    var displayName: String {
        switch self {
        case .vision: return "Apple Vision"
        case .mediapipe: return "MediaPipe"
        }
    }

    /// The version this backend's runs write to `Session.analysisVersion`
    /// and the pose cache's `analysis_version` (chainlink #48): a cache
    /// written by the old `"vision-1"` builds fails the version check the
    /// moment MediaPipe is the preferred backend, so every old pose cache is
    /// recomputed by the backend that now decides it — never replayed.
    var analysisVersion: String {
        switch self {
        case .vision: return "vision-1"
        case .mediapipe: return "mediapipe-1"
        }
    }

    /// The `PostProcessConfig.minVisibility` gate this backend's confidences
    /// are calibrated against — the gate is **per backend**, so a later
    /// Vision recalibration changes exactly this one number here. Both are
    /// 0.5 for now: MediaPipe's is the bake-off's gate (docs/bakeoff.md),
    /// Vision's is unchanged.
    var minVisibility: Double {
        switch self {
        case .vision: return 0.5
        case .mediapipe: return 0.5
        }
    }

    /// The post-process configuration an analysis with this backend runs
    /// with: `PostProcessConfig()` with this backend's gate filled in.
    var postProcessConfig: PostProcessConfig {
        var config = PostProcessConfig()
        config.minVisibility = minVisibility
        return config
    }

    /// A fresh service for one clip, or `nil` when the backend is not
    /// available — the caller falls back to an available one rather than
    /// assuming every case works.
    func makeService() -> PoseService? {
        switch self {
        case .vision:
            return VisionPoseService()
        case .mediapipe:
            guard let modelURL = Self.mediapipeModelURL, isAvailable(modelURL: modelURL) else {
                return nil
            }
            return MediaPipePoseService(modelURL: modelURL)
        }
    }
}
