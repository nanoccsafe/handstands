import XCTest

@testable import HandstandApp

/// The performance guard of live mode (chainlink #91), in its two pure
/// halves: the rate that gets halved (and where it stops), and the window
/// that decides to halve it. The recording must never drop a frame because
/// of live mode, and this is the arithmetic that keeps that promise.
final class LiveModeGuardTests: XCTestCase {
    // MARK: - The rate

    func testTheRateStartsAtTenAndHalvesDownToFourFps() {
        let rate = LiveRateController()
        XCTAssertEqual(rate.targetFps, 10, "live mode starts at 10 fps")
        XCTAssertEqual(rate.intervalS, 0.1, accuracy: 1e-9)

        XCTAssertTrue(rate.halve(reason: "test"), "10 halves")
        XCTAssertEqual(rate.targetFps, 5)
        XCTAssertTrue(rate.halve(reason: "test"), "5 halves onto the floor")
        XCTAssertEqual(rate.targetFps, 4, "the floor is 4 fps")
        XCTAssertFalse(rate.halve(reason: "test"), "already at the floor")
        XCTAssertEqual(rate.targetFps, 4, "the floor holds")
    }

    func testARateBelowTheFloorIsClamped() {
        XCTAssertEqual(LiveRateController(fps: 1).targetFps, 4)
        XCTAssertEqual(LiveRateController(fps: 60).targetFps, 60)
    }

    // MARK: - The window

    func testNoVerdictBeforeAFullWindowOfEvidence() {
        var window = InferenceGuard(windowS: 2.0, budgetMs: 80)

        var tMs = 0
        while tMs < 2_000 {
            XCTAssertNil(
                window.record(tMs: tMs, ms: 100),
                "over budget, but only \(tMs) ms of evidence — the cold start must not decide")
            tMs += 100
        }
        XCTAssertNotNil(
            window.record(tMs: 2_000, ms: 100),
            "a full 2 s over 80 ms is the guard's verdict"
        )
        // The window emptied itself: the next verdict needs fresh evidence.
        XCTAssertNil(window.record(tMs: 2_100, ms: 100))
        XCTAssertNil(window.record(tMs: 2_200, ms: 100))
    }

    func testAWindowWithinBudgetNeverVerdicts() {
        var window = InferenceGuard(windowS: 2.0, budgetMs: 80)

        var tMs = 0
        while tMs <= 10_000 {
            XCTAssertNil(window.record(tMs: tMs, ms: 40), "40 ms is within budget at \(tMs)")
            tMs += 100
        }
    }

    func testOneColdFrameIsDilutedByTheRestOfTheWindow() {
        var window = InferenceGuard(windowS: 2.0, budgetMs: 80)

        // The first inference of a session pays for whatever the model has to
        // warm up; a window of fast frames around it must not read that one
        // frame as a trend.
        XCTAssertNil(window.record(tMs: 0, ms: 600))
        for tMs in stride(from: 100, through: 6_000, by: 100) {
            XCTAssertNil(window.record(tMs: tMs, ms: 40), "600 ms once, then 40 ms, is not a slow camera")
        }
    }

    func testFramesOutsideTheWindowDoNotVote() {
        var window = InferenceGuard(windowS: 2.0, budgetMs: 80)

        // A fast first two seconds…
        for tMs in stride(from: 0, through: 1_900, by: 100) {
            XCTAssertNil(window.record(tMs: tMs, ms: 40), "a fast first two seconds")
        }
        // …then a gap, then slow frames: the fast ones are long outside the
        // window now, so they cannot hold the average down. One frame is
        // still not evidence…
        XCTAssertNil(window.record(tMs: 6_000, ms: 400))
        // …two slow frames in the window are.
        XCTAssertNotNil(window.record(tMs: 6_100, ms: 400))
    }
}
