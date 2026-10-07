import Foundation
import HandstandCore

/// Why a take had no hold, as the app says it (chainlink #93) — one case
/// per reason the diagnostics can name, **in the order they are worth
/// saying**: the framing first, because on a phone the take that shows no
/// hold is usually the one that was out of frame, and "your hands were not
/// visible" would be naming the symptom rather than the cause.
///
/// The raw values are the strings the `.diagnostics.json` file stores in
/// `dominant_reason`, so a file pulled off the phone and this enum always
/// speak the same words.
enum NoHoldReason: String, CaseIterable, Sendable {
    /// Nothing was detected in most of the take: nobody in the picture.
    case notInFrame = "not_in_frame"
    /// A person was found, a wrist/ankle/hip was not, and the visible body
    /// touched a frame border — the take was cut off at an edge.
    case outOfFrame = "out_of_frame"
    /// Something or someone was in front of the camera for most of the
    /// take (never a phone recording as such; the reason exists because the
    /// segmenter has it).
    case trainerContact = "trainer_contact"
    case noVisibleWrist = "no_visible_wrist"
    case noVisibleAnkle = "no_visible_ankle"
    case noVisibleHip = "no_visible_hip"
    /// Everything *was* visible: the athlete simply was not upside down
    /// long enough for the shortest hold (`PhaseConfig.minHoldS`).
    case notInverted = "not_inverted"
}

/// The "No hold found" explanation (chainlink #93): one short sentence in
/// plain words saying what went wrong, then **one** setup tip — the whole
/// message, so a take that found nothing is never just a zero.
///
/// The words live in one table each (``sentence`` and ``tip(for:)``), so
/// the user can reword them in one place.
struct NoHoldExplanation: Equatable, Sendable {
    let reason: NoHoldReason
    /// The frame edges the body was cut off at — only `.outOfFrame` names
    /// them, already in `Edge.allCases` order so the same take always
    /// reads the same sentence.
    var edges: [Edge]

    init(reason: NoHoldReason, edges: [Edge] = []) {
        self.reason = reason
        self.edges = edges
    }

    /// The reason in plain words — exactly one sentence.
    var sentence: String {
        switch reason {
        case .notInFrame:
            return "You were out of frame for most of the take."
        case .outOfFrame:
            guard !edges.isEmpty else {
                return "You were partly out of frame for most of the take."
            }
            let names = edges.map(\.rawValue)
            return "You were partly out of frame (\(Self.list(names)) edges) for most of the take."
        case .trainerContact:
            return "Something was in front of the camera for most of the take."
        case .noVisibleWrist:
            return "Your hands weren't visible for most of the take."
        case .noVisibleAnkle:
            return "Your feet weren't visible for most of the take."
        case .noVisibleHip:
            return "Your hips weren't visible for most of the take."
        case .notInverted:
            let seconds = PhaseConfig().minHoldS
            return String(
                format: "No one was upside down long enough (%.1f s minimum).", seconds)
        }
    }

    /// The one short setup tip the message ends with — a small table: where
    /// to put the camera when the take did not fit in the picture, and how
    /// to light the room when the model simply could not see well enough.
    var tip: String { Self.tip(for: reason) }

    /// The whole message: the sentence, then the tip.
    var text: String { "\(sentence) \(tip)" }

    /// The tip table itself, one entry per *kind* of reason, so the tips
    /// stay few and stay the same wherever they are needed.
    static func tip(for reason: NoHoldReason) -> String {
        switch reason {
        case .notInFrame, .outOfFrame, .notInverted:
            return positionTip
        case .noVisibleWrist, .noVisibleAnkle, .noVisibleHip:
            return lightTip
        case .trainerContact:
            return clearViewTip
        }
    }

    /// Far enough and low enough that a whole body fits in the picture both
    /// standing and upside down — the distance/height tip (chainlink #93).
    static let positionTip =
        "Place the phone about 3 m away at hip height, so your whole body fits upside down."

    /// Even light and an uncluttered background — the contrast tip, for a
    /// model that could not read the joints (chainlink #93).
    static let lightTip =
        "Stand in even light against a plain background, so the camera can see you clearly."

    /// Nothing but the athlete in the way (chainlink #93's third case).
    static let clearViewTip = "Keep the camera's view clear — only you should be in it."

    /// "top", "top and right", "top, bottom and right" — the join the
    /// sentences are built from, spelled the same way the framing guide's
    /// own messages spell it.
    private static func list(_ items: [String]) -> String {
        switch items.count {
        case 0: return ""
        case 1: return items[0]
        case 2: return "\(items[0]) and \(items[1])"
        default:
            return items.dropLast().joined(separator: ", ") + " and " + items[items.count - 1]
        }
    }
}
