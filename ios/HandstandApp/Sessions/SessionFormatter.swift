import Foundation

/// The strings the History screens show that are neither durations nor
/// pixel sizes (`VideoInfoFormatter` owns those): when a take happened and
/// what it scored. Kept out of the views so the tests can pin the formats
/// down without a view in the way.
enum SessionFormatter {
    /// `28 Sep 2026, 14:30` — a list of takes is read by date first, so
    /// date and time always appear together, in a fixed POSIX format that
    /// does not shift with the device language. The time zone is injected
    /// for the tests; the app uses the phone's.
    static func dateTime(_ date: Date, timeZone: TimeZone = .current) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "d MMM yyyy, HH:mm"
        return formatter.string(from: date)
    }

    /// A clip score to two decimals. Anything that is not a real number
    /// (and the `nil` of a take that has not been analysed) reads "—"
    /// rather than "nan" or "-0.00"; `analysisStatus` is what turns the
    /// `nil` case into words.
    static func score(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "—" }
        return String(format: "%.2f", value)
    }

    /// What the History row and the detail screen's Recording section say
    /// about a take's analysis — one function, so the two screens can never
    /// disagree (chainlink #96). The order matters: a note means the run
    /// could not measure the clip at all (chainlink #47), a score is the one
    /// number worth showing, `analyzedAt == nil` is the only "not analysed
    /// yet", and anything left was analysed but produced no score — a take
    /// with no hold that could be scored, or a build with no reference
    /// (chainlink #28). The hold count is the detail the row otherwise
    /// loses: "Analysed · 2 holds · no score" says more than "no score".
    static func analysisStatus(
        analyzedAt: Date?,
        clipScore: Double?,
        analysisNote: String?,
        holdCount: Int?
    ) -> String {
        if analysisNote != nil { return "Couldn't measure" }
        if let clipScore { return score(clipScore) }
        if analyzedAt == nil { return "Not analysed yet" }
        if let holdCount, holdCount > 0 {
            let noun = holdCount == 1 ? "hold" : "holds"
            return "Analysed · \(holdCount) \(noun) · no score"
        }
        return "Analysed · no score"
    }

    /// A playback position in the clip's own milliseconds as `0:04.2` —
    /// minutes, padded seconds and a tenth, the way the worst-moment card
    /// (chainlink #49) says when the worst frame was. A time that is not
    /// a whole number of milliseconds (negative, absurdly large) reads
    /// `0:00.0` rather than something odd.
    static func position(_ tMs: Int) -> String {
        guard tMs >= 0, tMs < 100_000_000 else { return "0:00.0" }
        let total = tMs / 1000
        let tenths = (tMs % 1000) / 100
        return String(format: "%d:%02d.%d", total / 60, total % 60, tenths)
    }
}
