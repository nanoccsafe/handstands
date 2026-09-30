import Foundation
import VisionPoseKit

/// Which pose backend runs the detection.
///
/// The choice itself belongs to the bake-off (chainlink #16), so the app never
/// hard-codes a backend further down: it asks for one. Until the bake-off
/// decides, only one exists — Apple Vision, which is built into iOS, needs no
/// download, and is the backend chainlink #82 put behind `PoseService`
/// (chainlink #47 runs it after recording; the analysis UI comes later).
///
/// There is no UI for this enum: `#47` asks with
/// `PoseBackend.vision.makeService()` and gets a `VisionPoseService`. When
/// MediaPipe lands (#45, after the bake-off) its case becomes available and
/// the same call hands back that backend instead — nothing else changes.
enum PoseBackend: String, CaseIterable {
    /// Apple Vision, through `VisionPoseKit`.
    case vision
    /// MediaPipe (chainlink #45): not written yet, so never available.
    case mediapipe

    /// Can this backend produce a service right now?
    var isAvailable: Bool {
        switch self {
        case .vision: return true
        case .mediapipe: return false
        }
    }

    /// What the backend is called in a log line or a future settings row.
    var displayName: String {
        switch self {
        case .vision: return "Apple Vision"
        case .mediapipe: return "MediaPipe"
        }
    }

    /// A fresh service for one clip, or `nil` when the backend is not
    /// available yet — the caller falls back to an available one rather than
    /// assuming every case works.
    func makeService() -> PoseService? {
        guard isAvailable else { return nil }
        switch self {
        case .vision: return VisionPoseService()
        case .mediapipe: return nil  // unreachable: guarded above
        }
    }
}
