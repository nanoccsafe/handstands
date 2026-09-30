import XCTest

@testable import HandstandApp

/// `SessionProgress` (chainlink #51): the numbers above the History list.
/// Pure logic over plain snapshots, with the clock and the calendar handed
/// in — so "this week" is a fixed date against a fixed calendar here, never
/// whatever day the test happens to run on.
final class SessionProgressTests: XCTestCase {
    private let utc = TimeZone(identifier: "UTC")!

    /// Weeks start on Monday (`firstWeekday = 2`), so Sunday belongs to the
    /// week before — pinned here rather than inherited from the device.
    private var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = utc
        calendar.firstWeekday = 2
        return calendar
    }

    /// A date in UTC, built from components so the intent is readable.
    private func date(
        _ year: Int, _ month: Int, _ day: Int,
        _ hour: Int = 12, _ minute: Int = 0
    ) -> Date {
        var components = DateComponents()
        components.year = year
        components.month = month
        components.day = day
        components.hour = hour
        components.minute = minute
        return calendar.date(from: components)!
    }

    private func snapshot(
        at recordedAt: Date,
        durationS: Double = 10,
        score: Double? = nil
    ) -> SessionSnapshot {
        SessionSnapshot(recordedAt: recordedAt, durationS: durationS, clipScore: score)
    }

    private var now: Date { date(2026, 9, 30, 12) } // a Wednesday

    // MARK: - The totals

    func testTotalsCountEverySessionAndSumTheirDuration() {
        let sessions = [
            snapshot(at: date(2026, 9, 30, 9), durationS: 12.4),
            snapshot(at: date(2026, 9, 29, 18), durationS: 45),
            snapshot(at: date(2026, 9, 1, 8), durationS: 2.6, score: 0.8)
        ]

        let progress = SessionProgress.summary(of: sessions, now: now, calendar: calendar)

        XCTAssertEqual(progress.total, 3)
        XCTAssertEqual(progress.totalRecordedS, 60, accuracy: 0.0001)
        XCTAssertEqual(progress.thisWeek, 2, "30 and 29 September are this week, 1 September is not")
    }

    func testAnEmptyListIsAllZeroesAndNoBestScore() {
        let progress = SessionProgress.summary(of: [], now: now, calendar: calendar)

        XCTAssertEqual(progress, SessionProgress(
            total: 0,
            thisWeek: 0,
            totalRecordedS: 0,
            analysed: 0,
            bestScore: nil,
            recentScores: []
        ))
    }

    // MARK: - This week

    func testThisWeekFollowsTheInjectedCalendarAndDate() {
        // The calendar's week around Wednesday 30 September 2026 runs
        // Monday 28 → Sunday 4 October; Sunday 27 belongs to the week
        // before, and a rolling seven days would wrongly include it.
        let sessions = [
            snapshot(at: date(2026, 9, 28, 7)),   // Monday — in
            snapshot(at: date(2026, 9, 30, 23, 59)), // Wednesday — in
            snapshot(at: date(2026, 9, 27, 23, 59)), // Sunday — out
            snapshot(at: date(2026, 9, 21, 12)),  // the Monday before — out
            snapshot(at: date(2026, 10, 4, 12))   // the Sunday after — in
        ]

        let progress = SessionProgress.summary(of: sessions, now: now, calendar: calendar)

        XCTAssertEqual(progress.thisWeek, 3)
        XCTAssertEqual(progress.total, 5, "this week never changes the total")
    }

    func testASessionOnAnotherDayOfTheSameWeekStillCounts() {
        let progress = SessionProgress.summary(
            of: [snapshot(at: date(2026, 9, 28, 6))],
            now: now,
            calendar: calendar
        )
        XCTAssertEqual(progress.thisWeek, 1)
    }

    // MARK: - Scores

    func testSessionsWithoutAScoreAreNotAnalysed() {
        let sessions = [
            snapshot(at: date(2026, 9, 30), durationS: 10, score: 0.9),
            snapshot(at: date(2026, 9, 29), durationS: 10), // not analysed yet
            snapshot(at: date(2026, 9, 28), durationS: 10, score: 0.4)
        ]

        let progress = SessionProgress.summary(of: sessions, now: now, calendar: calendar)

        XCTAssertEqual(progress.total, 3, "an unanalysed take still counts as a session")
        XCTAssertEqual(progress.analysed, 2)
        XCTAssertEqual(progress.bestScore, 0.9)
        XCTAssertEqual(progress.recentScores, [0.4, 0.9], "oldest → newest")
    }

    func testNoAnalysedSessionsMeansNoBestScoreAndNoScores() {
        let progress = SessionProgress.summary(
            of: [snapshot(at: date(2026, 9, 30)), snapshot(at: date(2026, 9, 29))],
            now: now,
            calendar: calendar
        )

        XCTAssertEqual(progress.total, 2)
        XCTAssertEqual(progress.analysed, 0)
        XCTAssertNil(progress.bestScore)
        XCTAssertEqual(progress.recentScores, [])
    }

    func testBestScoreIsTheHighestAcrossAllAnalysedSessions() {
        let sessions = [
            snapshot(at: date(2026, 9, 30), score: 0.6),
            snapshot(at: date(2026, 9, 20), score: 0.95),
            snapshot(at: date(2026, 9, 10), score: 0.2)
        ]

        let progress = SessionProgress.summary(of: sessions, now: now, calendar: calendar)

        XCTAssertEqual(progress.bestScore, 0.95, "the best is read across all of them, not just the recent ten")
    }

    func testRecentScoresKeepTheNewestTenOldestFirst() {
        // 12 analysed takes, in date order, scored 0.01, 0.02 … 0.12.
        let sessions = (1...12).map { index in
            snapshot(at: date(2026, 9, index), score: Double(index) / 100)
        }

        let progress = SessionProgress.summary(of: sessions, now: now, calendar: calendar)

        XCTAssertEqual(progress.analysed, 12)
        XCTAssertEqual(progress.recentScores.count, 10, "capped at the last ten")
        XCTAssertEqual(
            progress.recentScores,
            [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
            "the newest ten, still oldest → newest"
        )
    }

    func testRecentScoresCountOnlyAnalysedSessionsTowardsTheTen() {
        // Ten scores among many unanalysed takes: the cap counts scores,
        // not sessions, and the unanalysed ones never enter the list.
        var sessions: [SessionSnapshot] = (1...15).map { index in
            snapshot(at: date(2026, 9, index), durationS: 5)
        }
        for index in [1, 3, 5, 7, 9, 11, 13] {
            sessions[index - 1].clipScore = Double(index) / 100
        }
        for index in [2, 4, 6, 8, 10] {
            sessions[index - 1].clipScore = 0.99
        }

        let progress = SessionProgress.summary(of: sessions, now: now, calendar: calendar)

        XCTAssertEqual(progress.total, 15)
        XCTAssertEqual(progress.analysed, 12)
        XCTAssertEqual(progress.recentScores.count, 10)
        XCTAssertEqual(progress.recentScores.last, 0.13, "the newest score is the last element")
        XCTAssertFalse(progress.recentScores.contains(0), "nil scores never become 0")
    }
}
