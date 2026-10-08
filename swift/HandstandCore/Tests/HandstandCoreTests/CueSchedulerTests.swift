import XCTest

import HandstandCore

/// What the cue scheduler says, tick by tick (chainlink #91): the line cue and
/// its once-per-line rule, the time marks behind their toggle, the framing
/// hints and their three throttles, "Ready." on the framing edge — and the
/// global debounce that keeps the whole thing sparse.
///
/// No voice, no view, no camera: every test is a handful of `decide` calls.
final class CueSchedulerTests: XCTestCase {
    // MARK: - Helpers

    /// One tick, with the framing verdict the Record screen would have shown.
    @discardableResult
    private func tick(
        _ scheduler: inout CueScheduler,
        tMs: Int,
        events: [LiveEvent] = [],
        framing: FramingStatus? = nil,
        inverted: Bool = false,
        recording: Bool = false
    ) -> Cue? {
        scheduler.decide(
            tMs: tMs, events: events, framing: framing, inverted: inverted, recording: recording)
    }

    // MARK: - The settings

    func testFreshInstallDefaults() {
        XCTAssertEqual(
            CueSettings(),
            CueSettings(voiceCues: true, timeMarks: false, framingHints: true),
            "voice on, time marks off, framing hints on"
        )
    }

    // MARK: - The line cue

    func testLineHoldSpeaksOncePerLine() {
        var scheduler = CueScheduler()

        XCTAssertEqual(tick(&scheduler, tMs: 0, events: [.lineAchieved(tMs: 0)], inverted: true), .lineHold)
        XCTAssertNil(tick(&scheduler, tMs: 500, inverted: true), "one line, one cue")
        XCTAssertNil(tick(&scheduler, tMs: 900, events: [.lineLost(tMs: 900, heldS: 0.9)]))

        // A second line, well clear of the debounce, speaks again.
        XCTAssertEqual(
            tick(&scheduler, tMs: 5000, events: [.lineAchieved(tMs: 5000)], inverted: true),
            .lineHold
        )
        XCTAssertEqual(scheduler.lastCue, .lineHold)
    }

    func testLineHoldWaitsForItsOneSecondGapInsteadOfBeingDropped() {
        var scheduler = CueScheduler()

        // A framing hint first…
        XCTAssertEqual(
            tick(&scheduler, tMs: 1000, framing: .tooSmall), .framing(.comeCloser))
        // …then a line only 0.5 s later: held back, not thrown away…
        XCTAssertNil(
            tick(&scheduler, tMs: 1500, events: [.lineAchieved(tMs: 1500)], inverted: true),
            "the line cue may follow anything after 1 s, not before")
        // …and spoken at the first tick past it.
        XCTAssertEqual(tick(&scheduler, tMs: 2100, inverted: true), .lineHold)
    }

    // MARK: - Time marks

    func testTimeMarksOnlySpeakWhenTheirToggleIsOn() {
        var off = CueScheduler()
        XCTAssertNil(tick(&off, tMs: 100, events: [.timeMark(tMs: 100, seconds: 5)]))

        var on = CueScheduler()
        on.settings.timeMarks = true
        XCTAssertEqual(
            tick(&on, tMs: 100, events: [.timeMark(tMs: 100, seconds: 5)]),
            .timeMark(5)
        )
    }

    func testATimeMarkWaitsOutTheGlobalDebounce() {
        var scheduler = CueScheduler()
        scheduler.settings.timeMarks = true

        XCTAssertEqual(tick(&scheduler, tMs: 0, events: [.lineAchieved(tMs: 0)], inverted: true), .lineHold)
        XCTAssertNil(tick(&scheduler, tMs: 500, events: [.timeMark(tMs: 500, seconds: 1)]))
        XCTAssertEqual(tick(&scheduler, tMs: 3500, events: [.timeMark(tMs: 3500, seconds: 5)]), .timeMark(5))
    }

    // MARK: - The toggles

    func testNothingIsSpokenWhenVoiceIsOffExceptFramingHints() {
        var scheduler = CueScheduler()
        scheduler.settings.voiceCues = false

        XCTAssertNil(tick(&scheduler, tMs: 0, events: [.lineAchieved(tMs: 0)], inverted: true))
        XCTAssertNil(tick(&scheduler, tMs: 100, events: [.timeMark(tMs: 100, seconds: 5)]))
        XCTAssertNil(tick(&scheduler, tMs: 200, events: [.lineLost(tMs: 200, heldS: 0.2)]))

        // The "Ready." edge is silent too…
        XCTAssertNil(tick(&scheduler, tMs: 3000, framing: .noPerson))
        XCTAssertNil(tick(&scheduler, tMs: 4000, framing: .ok))
        // …but framing hints have their own toggle and still speak.
        XCTAssertEqual(tick(&scheduler, tMs: 5000, framing: .tooSmall), .framing(.comeCloser))
    }

    func testNothingIsSpokenWhenFramingHintsAreOffEither() {
        var scheduler = CueScheduler()
        scheduler.settings.voiceCues = false
        scheduler.settings.framingHints = false

        XCTAssertNil(tick(&scheduler, tMs: 0, events: [.lineAchieved(tMs: 0)], inverted: true))
        XCTAssertNil(tick(&scheduler, tMs: 1000, framing: .noPerson))
        XCTAssertNil(tick(&scheduler, tMs: 2000, framing: .ok))
        XCTAssertNil(tick(&scheduler, tMs: 3000, framing: .tooSmall))
    }

    // MARK: - The framing hints

    func testHintsStayQuietWhileUpsideDownAndInALine() {
        var scheduler = CueScheduler()

        // Upside down: no hint whatever the border says.
        XCTAssertNil(tick(&scheduler, tMs: 0, framing: .tooSmall, inverted: true))
        // In an announced line: still quiet, even seconds later.
        XCTAssertEqual(
            tick(&scheduler, tMs: 100, events: [.lineAchieved(tMs: 100)], framing: .ok, inverted: true),
            .lineHold
        )
        XCTAssertNil(tick(&scheduler, tMs: 5000, framing: .tooSmall, inverted: false))
        // Coming down starts a 1.5 s cooldown…
        XCTAssertNil(tick(&scheduler, tMs: 5100, events: [.lineLost(tMs: 5100, heldS: 5)], framing: .tooSmall))
        // …and after it the hint may speak.
        XCTAssertEqual(tick(&scheduler, tMs: 9000, framing: .tooSmall), .framing(.comeCloser))
    }

    func testAtMostOneFramingHintEveryFourSeconds() {
        var scheduler = CueScheduler()

        XCTAssertEqual(tick(&scheduler, tMs: 1000, framing: .tooSmall), .framing(.comeCloser))
        XCTAssertNil(
            tick(&scheduler, tMs: 4000, framing: .partlyOutOfFrame(edges: [.bottom], missing: [])),
            "3 s after the last hint — the gap is 4 s")
        XCTAssertEqual(
            tick(&scheduler, tMs: 5000, framing: .partlyOutOfFrame(edges: [.bottom], missing: [])),
            .framing(.stepBack)
        )
        XCTAssertEqual(
            tick(&scheduler, tMs: 9000, framing: .partlyOutOfFrame(edges: [.top], missing: [])),
            .framing(.raisePhone)
        )
    }

    func testTheSameHintIsNotRepeatedWithinEightSeconds() {
        var scheduler = CueScheduler()

        XCTAssertEqual(tick(&scheduler, tMs: 1000, framing: .tooSmall), .framing(.comeCloser))
        // A line and its end pass in between — the cooldown clock restarts
        // when the athlete comes down at 5.5 s.
        XCTAssertEqual(
            tick(&scheduler, tMs: 5000, events: [.lineAchieved(tMs: 5000)], inverted: true),
            .lineHold
        )
        XCTAssertNil(tick(&scheduler, tMs: 5500, events: [.lineLost(tMs: 5500, heldS: 0.7)]))
        // 7.5 s after the *last* hint of this wording — under 8 s, so no.
        XCTAssertNil(tick(&scheduler, tMs: 8500, framing: .tooSmall))
        // Past 8 s it may say it again.
        XCTAssertEqual(tick(&scheduler, tMs: 9500, framing: .tooSmall), .framing(.comeCloser))
    }

    func testHintsArePreAttemptOnly() {
        var scheduler = CueScheduler()
        XCTAssertNil(tick(&scheduler, tMs: 0, framing: .tooSmall, recording: true),
                     "no framing hints while the red light is on")
        XCTAssertEqual(tick(&scheduler, tMs: 1000, framing: .tooSmall, recording: false), .framing(.comeCloser))
    }

    // MARK: - "Ready."

    func testReadySpeaksOncePerNonOkToOkTransition() {
        var scheduler = CueScheduler()

        XCTAssertNil(tick(&scheduler, tMs: 0, framing: .noPerson), "the first verdict is not a transition")
        XCTAssertEqual(tick(&scheduler, tMs: 500, framing: .ok), .ready)
        XCTAssertNil(tick(&scheduler, tMs: 4500, framing: .ok), "still ok — no new transition")
        XCTAssertNil(tick(&scheduler, tMs: 5000, framing: .noPerson))
        XCTAssertEqual(tick(&scheduler, tMs: 9000, framing: .ok), .ready)
    }

    func testReadyOnlyWhileUprightAndOnlyBeforeTheAttempt() {
        var upsideDown = CueScheduler()
        XCTAssertNil(tick(&upsideDown, tMs: 0, framing: .noPerson))
        XCTAssertNil(tick(&upsideDown, tMs: 1000, framing: .ok, inverted: true))

        var recording = CueScheduler()
        XCTAssertNil(tick(&recording, tMs: 0, framing: .noPerson))
        XCTAssertNil(tick(&recording, tMs: 1000, framing: .ok, recording: true))

        var ready = CueScheduler()
        XCTAssertNil(tick(&ready, tMs: 0, framing: .noPerson))
        XCTAssertEqual(tick(&ready, tMs: 1000, framing: .ok), .ready)
    }

    // MARK: - The debounce

    func testAtLeastThreeSecondsBetweenAnyTwoCuesButOneSecondForTheLineCue() {
        var scheduler = CueScheduler()

        XCTAssertEqual(tick(&scheduler, tMs: 1000, framing: .tooSmall), .framing(.comeCloser))
        // An edge 1 s later: dropped (it would be stale by the time it spoke).
        XCTAssertNil(tick(&scheduler, tMs: 1500, framing: .noPerson))
        XCTAssertNil(tick(&scheduler, tMs: 2000, framing: .ok), "only 1 s after the last cue")
        // The same transition, once the debounce is spent, speaks.
        XCTAssertNil(tick(&scheduler, tMs: 2500, framing: .noPerson))
        XCTAssertEqual(tick(&scheduler, tMs: 6000, framing: .ok), .ready)

        // The line cue is the exception: 1 s after "Ready." is enough.
        XCTAssertEqual(
            tick(&scheduler, tMs: 7000, events: [.lineAchieved(tMs: 7000)], inverted: true),
            .lineHold
        )
    }

    // MARK: - The mapping

    func testTheFramingHintTable() {
        XCTAssertEqual(CueScheduler.hint(for: .tooSmall, upright: true), .comeCloser)
        XCTAssertEqual(CueScheduler.hint(for: .tooSmall, upright: false), .comeCloser)
        XCTAssertEqual(CueScheduler.hint(for: .multiplePeople, upright: true), .onlyYou)
        XCTAssertNil(CueScheduler.hint(for: .noPerson, upright: true), "nobody to coach yet")
        XCTAssertNil(CueScheduler.hint(for: .ok, upright: true), "nothing to fix")

        XCTAssertEqual(
            CueScheduler.hint(for: .partlyOutOfFrame(edges: [.bottom], missing: []), upright: true),
            .stepBack)
        XCTAssertEqual(
            CueScheduler.hint(for: .partlyOutOfFrame(edges: [.top], missing: []), upright: true),
            .raisePhone, "clipped at the top while standing: the phone is too low")
        XCTAssertEqual(
            CueScheduler.hint(for: .partlyOutOfFrame(edges: [.top], missing: []), upright: false),
            .stepBack)
        XCTAssertEqual(
            CueScheduler.hint(for: .partlyOutOfFrame(edges: [.left], missing: []), upright: true),
            .moveToMiddle)
        XCTAssertEqual(
            CueScheduler.hint(for: .partlyOutOfFrame(edges: [.right], missing: []), upright: true),
            .moveToMiddle)
        XCTAssertEqual(
            CueScheduler.hint(for: .partlyOutOfFrame(edges: [.top, .left], missing: []), upright: true),
            .moveToMiddle, "any sideways edge is a sideways fix")
        XCTAssertEqual(
            CueScheduler.hint(for: .partlyOutOfFrame(edges: [], missing: [.leftWrist]),
                              upright: true),
            .stepBack, "a joint nobody can see reads like the border's own message")
    }
}
