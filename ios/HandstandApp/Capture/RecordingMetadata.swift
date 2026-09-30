import Foundation
import HandstandCore

/// What a recording was: the hold type chosen when the take started, when it
/// was taken and which build took it. A small JSON **sidecar next to the
/// movie** — `20260928-143059.mov` → `20260928-143059.json` in the same
/// Recordings folder (`RecordingFile.swift`) — because there is no session
/// store yet (#51 comes later). Analysis (#47) reads `hold_type` to pick the
/// matching scoring reference.
///
/// ```json
/// {"schema": 1, "hold_type": "line", "recorded_at": "2026-09-28T14:30:59Z",
///  "app_version": "0.1.0"}
/// ```
///
/// Keys are snake_case, dates are ISO-8601 in UTC and the hold is
/// `HoldType`'s raw value — the same persisted strings the picker stores.
struct RecordingMetadata: Codable, Equatable, Sendable {
    /// Bumped only if the format ever changes incompatibly.
    var schema: Int = 1
    /// The hold selected when recording *started* (never a later tap).
    var holdType: HoldType
    /// When the take started, in UTC.
    var recordedAt: Date
    /// The app's marketing version, e.g. "0.1.0".
    var appVersion: String

    private enum CodingKeys: String, CodingKey {
        case schema
        case holdType = "hold_type"
        case recordedAt = "recorded_at"
        case appVersion = "app_version"
    }

    init(schema: Int = 1, holdType: HoldType, recordedAt: Date, appVersion: String) {
        self.schema = schema
        self.holdType = holdType
        self.recordedAt = recordedAt
        self.appVersion = appVersion
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schema = try container.decodeIfPresent(Int.self, forKey: .schema) ?? 1
        // An unknown (or missing) hold_type must never break the read:
        // `resolved` falls back to `.default`.
        holdType = HoldType.resolved(try? container.decode(String.self, forKey: .holdType))
        recordedAt = try container.decode(Date.self, forKey: .recordedAt)
        appVersion = try container.decodeIfPresent(String.self, forKey: .appVersion) ?? ""
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
