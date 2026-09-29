import HandstandCore
import SwiftUI

/// What colour the framing border is for a verdict, and that colour as a
/// SwiftUI `Color`.
///
/// The mapping is its own type so the tests can assert it without SwiftUI's
/// `Color` in the way: green says "record this", amber "fix the framing
/// first", red "there is nobody (or more than one body) to frame".
enum FramingTone: Equatable, Sendable {
    case green
    case amber
    case red
}

enum FramingDisplay {
    /// The border colour for a framing verdict:
    ///
    /// * `.ok` → green — the whole body is in frame;
    /// * `.tooSmall` and `.partlyOutOfFrame` → amber — one athlete, wrong
    ///   framing; the message says which way;
    /// * `.noPerson` and `.multiplePeople` → red — no single body to record.
    static func tone(for status: FramingStatus) -> FramingTone {
        switch status {
        case .ok:
            return .green
        case .tooSmall, .partlyOutOfFrame:
            return .amber
        case .noPerson, .multiplePeople:
            return .red
        }
    }
}

extension FramingTone {
    /// The colour itself. Amber is a real amber (`#FFBF00`), not the system
    /// yellow, so it reads as "adjust" rather than "warning/error".
    var color: Color {
        switch self {
        case .green:
            return .green
        case .amber:
            return Color(red: 1.0, green: 0.75, blue: 0.0)
        case .red:
            return .red
        }
    }
}
