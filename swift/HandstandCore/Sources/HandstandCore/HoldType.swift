import Foundation

/// The kind of hold the athlete is training — the *only* thing the user tells
/// the app before a recording (chainlink #66). Wall, hand steps, camera angle
/// and the phases are detected automatically (#72), so nothing else is asked.
///
/// - Important: The raw values are a **persisted format**. They are written to
///   the sidecar next to every recording (`RecordingMetadata`) and remembered
///   in `@AppStorage("holdType")`, so existing files must keep reading
///   forever — **never rename a case**; add new ones instead.
public enum HoldType: String, CaseIterable, Codable, Sendable, Identifiable {
    case line, tuck, straddle, stag, diamond, oneArm = "one_arm"

    public var id: String { rawValue }

    /// What the picker and the done screen call it.
    public var displayName: String {
        switch self {
        case .line: "Line"
        case .tuck: "Tuck"
        case .straddle: "Straddle"
        case .stag: "Stag"
        case .diamond: "Diamond"
        case .oneArm: "One arm"
        }
    }

    /// Only Line is recordable in the MVP (chainlink #66); the rest are shown
    /// in the picker disabled, as "coming soon".
    public var isAvailable: Bool { self == .line }

    /// The hold a fresh install (and any unreadable stored value) starts with.
    public static let `default`: HoldType = .line

    /// The holds the picker can actually be set to.
    public static var available: [HoldType] { allCases.filter(\.isAvailable) }

    /// Resource name of the scoring reference for this hold, e.g.
    /// "reference-line". No reference is bundled yet (the real one comes from
    /// chainlink #28), so this is just the name.
    public var referenceResourceName: String { "reference-\(rawValue)" }

    /// A stored raw value → a usable hold: unknown **or unavailable** values
    /// fall back to `.default`, so a sidecar written by a future build (or a
    /// corrupted `@AppStorage` value) reads as Line instead of failing.
    public static func resolved(_ raw: String?) -> HoldType {
        guard let raw, let hold = HoldType(rawValue: raw), hold.isAvailable else {
            return .default
        }
        return hold
    }
}
