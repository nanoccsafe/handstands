import Foundation
import HandstandCore

// --------------------------------------------------------------------------- #
// The pose cache (#48): the frames Vision saw, written next to the movie so
// reopening a session never runs Vision again.
//
// Analysing a take costs seconds of decode + inference; *showing* the
// overlay again costs only `Analyzer.analyze`, which is milliseconds. So the
// run that extracted the frames writes them once — `<basename>.pose.json`
// beside the `.mov`, local to the phone exactly like the video itself — and
// the session screen reads them back on appear.
//
// The document is deliberately the *input* of the pipeline, not its output:
// what Vision answered, in display pixels, with the same `tMs` clock the
// movie plays on. The analysis is therefore always recomputed from these
// frames (with whatever reference the build has), never frozen into a file
// that would go stale when #28 lands a new reference.
//
//     {"schema": 1, "backend": "vision", "analysis_version": "vision-1",
//      "max_fps": 30,
//      "frames": [{"t_ms": 0, "detected": true,
//                  "joints": {"nose": [x, y, visibility], …}}, …]}
//
// `trainer_contact` is not a column: the device backends never report one
// (there is no trainer behind the camera of a phone recording), so every
// cached frame reads back `false`, which is exactly what it was written
// with. A cache from another schema or another analysis version is not
// read at all — `read` answers `nil` and the caller falls back to the real
// run — because a frame written by a different pipeline version is a
// different question, not an answer.
// --------------------------------------------------------------------------- #

/// Reading and writing a movie's `<basename>.pose.json` next to it.
///
/// `@MainActor` because the version it stamps comes from
/// `AnalysisService` (and every caller — the analysis run, the session
/// screen, `SessionStore.delete` — is on the main actor anyway).
@MainActor
enum PoseCache {
    /// The document's schema version — bumped when the *shape* changes.
    static let schema = 1

    /// Which pose backend wrote the file (`PoseBackend.preferred.rawValue`):
    /// `"mediapipe"` for runs of the default backend (chainlink #45),
    /// `"vision"` while Vision is the fallback. `read` keys on
    /// `analysis_version`, which follows the backend the same way — a cache
    /// written by the old `"vision-1"` builds is not read once MediaPipe
    /// decides the frames.
    static var backend: String { PoseBackend.preferred.rawValue }

    /// The cache for `movie`: same folder, same stem, `.pose.json`.
    static func cacheURL(for movie: URL) -> URL {
        movie.deletingPathExtension().appendingPathExtension("pose.json")
    }

    /// Writes `frames` next to `movie`, atomically: a crash mid-write leaves
    /// either the old file or none, never half a document.
    ///
    /// A joint whose coordinates (or score) are not finite is **not**
    /// written: the schema spells "the model did not see this joint" as "no
    /// entry", and JSON has no NaN to spell it with. A joint name this
    /// schema does not know is skipped the same way on the way back in.
    static func write(frames: [PostProcessInputFrame], for movie: URL) throws {
        let document = Document(
            schema: PoseCache.schema,
            backend: PoseCache.backend,
            analysisVersion: AnalysisService.analysisVersion,
            maxFps: Int(AnalysisService.maxFps),
            frames: frames.map { frame in
                var joints: [String: [Double]] = [:]
                for (joint, keypoint) in frame.joints where keypoint.x.isFinite
                    && keypoint.y.isFinite && keypoint.visibility.isFinite
                {
                    joints[joint.rawValue] = [keypoint.x, keypoint.y, keypoint.visibility]
                }
                return Document.Frame(
                    tMs: frame.tMs, detected: frame.detected, joints: joints)
            }
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(document).write(
            to: PoseCache.cacheURL(for: movie), options: .atomic)
    }

    /// The cached frames of `movie`, or `nil` — no file, unreadable bytes,
    /// a `schema` this build does not know, an `analysis_version` written by
    /// another pipeline, or a frame entry that does not parse. `nil` means
    /// "run the extraction again", never "no frames in this take".
    static func read(for movie: URL) -> [PostProcessInputFrame]? {
        guard let data = try? Data(contentsOf: PoseCache.cacheURL(for: movie)),
            let document = try? JSONDecoder().decode(Document.self, from: data),
            document.schema == PoseCache.schema,
            document.analysisVersion == AnalysisService.analysisVersion
        else {
            return nil
        }
        return document.frames.map { frame in
            var joints: [Joint: Keypoint] = [:]
            for (name, value) in frame.joints {
                guard let joint = Joint(rawValue: name), value.count == 3,
                    value[0].isFinite, value[1].isFinite, value[2].isFinite
                else { continue }
                joints[joint] = Keypoint(x: value[0], y: value[1], visibility: value[2])
            }
            return PostProcessInputFrame(
                tMs: frame.tMs, detected: frame.detected, trainerContact: false, joints: joints)
        }
    }

    // MARK: - The document

    /// The JSON shape above, with the snake_case keys the file is spelled
    /// in. `Codable` rather than `JSONSerialization`: the types are the
    /// check, and a malformed document fails the decode (hence `nil`).
    private struct Document: Codable {
        var schema: Int
        var backend: String
        var analysisVersion: String
        var maxFps: Int
        var frames: [Frame]

        enum CodingKeys: String, CodingKey {
            case schema
            case backend
            case frames
            case analysisVersion = "analysis_version"
            case maxFps = "max_fps"
        }

        struct Frame: Codable {
            var tMs: Int
            var detected: Bool
            var joints: [String: [Double]]

            enum CodingKeys: String, CodingKey {
                case detected
                case joints
                case tMs = "t_ms"
            }
        }
    }
}
