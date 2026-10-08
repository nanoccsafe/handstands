import Foundation
import HandstandCore

// --------------------------------------------------------------------------- #
// What each cue *reads* (chainlink #91) — one table behind both voices the app
// has: `AVSpeechSynthesizer` says it, the Record screen's banner shows it, and
// the sidecar's `cues_spoken` names it. Rewording a phrase is one edit here.
//
// The words are the ones the user asked for (2026-10-06): short, encouraging,
// and sparse — "how much is too much" is to be tested at the gym, so every
// phrase is optional and nothing here ever corrects form mid-hold.
// --------------------------------------------------------------------------- #

enum CueText {
    /// The phrase for a cue — the table `docs/ios.md` quotes and the user
    /// rewords. Every `Cue` case has exactly one.
    static func phrase(for cue: Cue) -> String {
        switch cue {
        case .lineHold:
            return "Line. Hold it."
        case .timeMark(let seconds):
            return numberWord(seconds)
        case .framing(let hint):
            return hintPhrase(hint)
        case .ready:
            return "Ready."
        }
    }

    /// The cue's name as the sidecar's `cues_spoken[].cue` spells it — stable
    /// snake_case, so the tuning numbers can be grepped and grouped.
    static func identifier(for cue: Cue) -> String {
        switch cue {
        case .lineHold:
            return "line_hold"
        case .timeMark(let seconds):
            return "time_mark_\(seconds)"
        case .framing(let hint):
            return hintIdentifier(hint)
        case .ready:
            return "ready"
        }
    }

    // MARK: - The pieces

    private static func hintPhrase(_ hint: FramingHint) -> String {
        switch hint {
        case .stepBack: return "Step back."
        case .comeCloser: return "Come closer."
        case .moveToMiddle: return "Move to the middle."
        case .onlyYou: return "Only you in the frame."
        case .raisePhone: return "Raise the phone."
        }
    }

    private static func hintIdentifier(_ hint: FramingHint) -> String {
        switch hint {
        case .stepBack: return "step_back"
        case .comeCloser: return "come_closer"
        case .moveToMiddle: return "move_to_middle"
        case .onlyYou: return "only_you"
        case .raisePhone: return "raise_phone"
        }
    }

    /// "Five", "Ten", "Fifteen", "Twenty-one", … — the time marks read as
    /// words because a number spoken aloud in a gym is a jumble. Out of
    /// range (there is no 101-second hold worth a word) falls back to the
    /// digits, which the synthesizer reads correctly anyway.
    static func numberWord(_ value: Int) -> String {
        guard value > 0 else { return "\(value)" }
        let ones = [
            "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
            "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
            "Seventeen", "Eighteen", "Nineteen",
        ]
        let tens = [
            "", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty",
            "Ninety",
        ]
        if value < ones.count { return ones[value] }
        if value < 100 {
            // Only the first word is capitalised in a compound: a time mark
            // stands alone ("Twenty"), but inside it the unit is not a fresh
            // sentence ("Twenty-one").
            let rest = value % 10
            return rest == 0 ? tens[value / 10] : tens[value / 10] + "-" + ones[rest].lowercased()
        }
        return "\(value)"
    }
}
