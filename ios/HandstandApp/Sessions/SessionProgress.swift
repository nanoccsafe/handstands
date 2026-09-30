import Foundation

/// One session as pure numbers — the input `SessionProgress` works on, so
/// the progress maths is testable without a database, a view or a video.
struct SessionSnapshot: Equatable {
    var recordedAt: Date
    var durationS: Double
    /// The clip score, `nil` while the take has not been analysed
    /// (chainlink #47).
    var clipScore: Double?
}

/// What the History header shows (chainlink #51): how much has been
/// recorded, how much of it was this week, and how the analysed takes
/// scored. Plain values, no SwiftData — `summary(of:now:calendar:)` takes
/// the clock and the calendar from outside, so "this week" is pinned down
/// by a fixed date and a fixed calendar in the tests instead of whatever
/// the phone happens to think today.
struct SessionProgress: Equatable {
    var total: Int
    /// Sessions recorded in `calendar`'s current week around `now`.
    var thisWeek: Int
    var totalRecordedS: Double
    /// Sessions that have a score. A take without one counts towards
    /// `total` only — it has not been analysed yet.
    var analysed: Int
    var bestScore: Double?
    /// The last up to 10 analysed scores, oldest → newest.
    var recentScores: [Double]

    static func summary(of sessions: [SessionSnapshot], now: Date, calendar: Calendar) -> SessionProgress {
        // "This week" is the calendar's week, not a rolling seven days:
        // `compare(_:to:toGranularity:)` puts `now` and the session in the
        // same week-of-year under `calendar`'s own first weekday.
        let thisWeek = sessions.filter {
            calendar.compare($0.recordedAt, to: now, toGranularity: .weekOfYear) == .orderedSame
        }.count

        // Scores in take order: `bestScore` reads across all of them, while
        // `recentScores` keeps only the newest ten (still oldest → newest,
        // so a chart can draw them left to right).
        let scores = sessions
            .filter { $0.clipScore != nil }
            .sorted { $0.recordedAt < $1.recordedAt }
            .compactMap(\.clipScore)

        return SessionProgress(
            total: sessions.count,
            thisWeek: thisWeek,
            totalRecordedS: sessions.reduce(0) { $0 + $1.durationS },
            analysed: scores.count,
            bestScore: scores.max(),
            recentScores: Array(scores.suffix(10))
        )
    }
}
