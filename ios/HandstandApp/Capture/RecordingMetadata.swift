import Foundation
import HandstandCore

/// What a recording was: the hold type chosen when the take started, when it
/// was taken and which build took it — plus, since live mode stage 1
/// (chainlink #91), what the phone said and how fast it was thinking while
/// the camera rolled. A small JSON **sidecar next to the movie** —
/// `20260928-143059.mov` → `20260928-143059.json` in the same Recordings
/// folder (`RecordingFile.swift`) — read by `SessionStore` (#51) when it
/// turns a recording into a History row, and by analysis (#47), which needs
/// `hold_type` to pick the matching scoring reference.
///
/// ```json
/// {"schema": 1, "hold_type": "line", "recorded_at": "2026-09-28T14:30:59Z",
///  "app_version": "0.1.0", "live_fps_target": 10.0, "live_fps_achieved": 9.7,
///  "avg_inference_ms": 41.2, "recording_frames_dropped": 0,
///  "cues_spoken": [{"t_ms": 1240, "cue": "line_hold"}]}
/// ```
///
/// Keys are snake_case, dates are ISO-8601 in UTC and the hold is
/// `HoldType`'s raw value — the same persisted strings the picker stores.
/// The five live keys are **optional** (schema stays 1): a take from a build
/// without the pose model, or from before this issue, reads back exactly as
/// it always did, with the live fields absent rather than zero.
struct RecordingMetadata: Codable, Equatable, Sendable {
    /// Bumped only if the format ever changes incompatibly.
    var schema: Int = 1
    /// The hold selected when recording *started* (never a later tap).
    var holdType: HoldType
    /// When the take started, in UTC.
    var recordedAt: Date
    /// The app's marketing version, e.g. "0.1.0".
    var appVersion: String
    /// The live rate in force at the end of the take, fps — halved by the
    /// performance guard when it had to be (chainlink #91).
    var liveFpsTarget: Double?
    /// The live frames actually analysed over the take's length, fps.
    var liveFpsAchieved: Double?
    /// What one live inference cost on average, milliseconds.
    var avgInferenceMs: Double?
    /// Frames the capture output dropped from the recording — `0` when the
    /// take lost none, which is the number the guard exists to keep.
    var recordingFramesDropped: Int?
    /// What was said during the take, in time order — empty means absent.
    var cuesSpoken: [SpokenCue]?

    private enum CodingKeys: String, CodingKey {
        case schema
        case holdType = "hold_type"
        case recordedAt = "recorded_at"
        case appVersion = "app_version"
        case liveFpsTarget = "live_fps_target"
        case liveFpsAchieved = "live_fps_achieved"
        case avgInferenceMs = "avg_inference_ms"
        case recordingFramesDropped = "recording_frames_dropped"
        case cuesSpoken = "cues_spoken"
    }

    init(
        schema: Int = 1,
        holdType: HoldType,
        recordedAt: Date,
        appVersion: String,
        liveFpsTarget: Double? = nil,
        liveFpsAchieved: Double? = nil,
        avgInferenceMs: Double? = nil,
        recordingFramesDropped: Int? = nil,
        cuesSpoken: [SpokenCue]? = nil
    ) {
        self.schema = schema
        self.holdType = holdType
        self.recordedAt = recordedAt
        self.appVersion = appVersion
        self.liveFpsTarget = liveFpsTarget
        self.liveFpsAchieved = liveFpsAchieved
        self.avgInferenceMs = avgInferenceMs
        self.recordingFramesDropped = recordingFramesDropped
        self.cuesSpoken = cuesSpoken
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schema = try container.decodeIfPresent(Int.self, forKey: .schema) ?? 1
        // An unknown (or missing) hold_type must never break the read:
        // `resolved` falls back to `.default`.
        holdType = HoldType.resolved(try? container.decode(String.self, forKey: .holdType))
        recordedAt = try container.decode(Date.self, forKey: .recordedAt)
        appVersion = try container.decodeIfPresent(String.self, forKey: .appVersion) ?? ""
        // The live fields (#91) are optional in both directions: absent from a
        // sidecar written before them, and left out when a take never saw a
        // live frame.
        liveFpsTarget = try container.decodeIfPresent(Double.self, forKey: .liveFpsTarget)
        liveFpsAchieved = try container.decodeIfPresent(Double.self, forKey: .liveFpsAchieved)
        avgInferenceMs = try container.decodeIfPresent(Double.self, forKey: .avgInferenceMs)
        recordingFramesDropped =
            try container.decodeIfPresent(Int.self, forKey: .recordingFramesDropped)
        cuesSpoken = try container.decodeIfPresent([SpokenCue].self, forKey: .cuesSpoken)
    }

    // MARK: - The sidecar file

    /// The sidecar for a movie: same folder, same stem, `.json` instead of
    /// the movie's extension.
    static func sidecarURL(for movie: URL) -> URL {
        movie.deletingPathExtension().appendingPathExtension("json")
    }

    /// The marketing version written into sidecars (`0.1.0`), read from the
    /// bundle; the fallback matches `MARKETING_VERSION` in `ios/project.yml`.
    static var currentAppVersion: String {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "0.1.0"
    }

    /// Writes the sidecar next to `movie`, atomically: a crash mid-write
    /// leaves either the old file or none, never half a JSON document.
    func write(for movie: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(self).write(to: Self.sidecarURL(for: movie), options: .atomic)
    }

    /// The sidecar for `movie`, or `nil` when there is none. An unreadable
    /// document throws; an unknown `hold_type` inside it reads back as
    /// `HoldType.resolved(...)`, never a crash.
    static func read(for movie: URL) throws -> RecordingMetadata? {
        let url = sidecarURL(for: movie)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(RecordingMetadata.self, from: Data(contentsOf: url))
    }
}

/// One cue spoken during a take (chainlink #91): when, and which — the
/// `cues_spoken` rows of the sidecar, the tuning data for "how much feedback
/// is too much".
///
/// ```json
/// {"t_ms": 1240, "cue": "line_hold"}
/// ```
///
/// `t_ms` is milliseconds into the recording (the capture clock relative to
/// the take's first live frame), `cue` the stable identifier `CueText`
/// spells — greppable, and never the spoken words, which the user may reword.
struct SpokenCue: Codable, Equatable, Sendable {
    /// Milliseconds from the start of the take to the cue.
    var tMs: Int
    /// The cue's identifier: `line_hold`, `time_mark_5`, `step_back`, …
    var cue: String

    private enum CodingKeys: String, CodingKey {
        case tMs = "t_ms"
        case cue
    }

    init(tMs: Int, cue: String) {
        self.tMs = tMs
        self.cue = cue
    }
}
