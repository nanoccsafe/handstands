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
        formEvents: [LiveFormEvent] = [],
        framing: FramingStatus? = nil,
        inverted: Bool = false,
        up: Bool = true,
        recording: Bool = false
    ) -> Cue? {
        scheduler.decide(
            tMs: tMs, events: events, formEvents: formEvents, framing: framing,
            inverted: inverted, up: up, recording: recording)
    }

    // MARK: - The settings

    func testFreshInstallDefaults() {
        XCTAssertEqual(
            CueSettings(),
            CueSettings(voiceCues: true, timeMarks: false, framingHints: true, formCues: true),
            "voice on, time marks off, framing hints on, form cues on"
        )
    }

    // MARK: - The form cue (chainlink #97)

    func testAFormEventSpeaksThroughTheCoachGap() {
        var scheduler = CueScheduler()

        // A hint holds the clock; a form event 1 s later waits it out…
        XCTAssertEqual(tick(&scheduler, tMs: 0, framing: .tooSmall), .framing(.comeCloser))
        XCTAssertNil(
            tick(&scheduler, tMs: 1000, formEvents: [.arch(tMs: 1000)]),
            "the coach gap is 2 s")
        // …and speaks at the first tick past it.
        XCTAssertEqual(
            tick(&scheduler, tMs: 2000), .form(.arch),
            "the form cue waited its turn rather than being dropped")
    }

    func testFormCuesAreTwoSecondsApartAtCoachPace() {
        var scheduler = CueScheduler()
        XCTAssertEqual(
            tick(&scheduler, tMs: 0, formEvents: [.hips(tMs: 0)], inverted: true), .form(.hips))
        XCTAssertNil(
            tick(&scheduler, tMs: 1000, formEvents: [.hips(tMs: 1000)], inverted: true),
            "1 s after the last form cue is too soon")
        XCTAssertEqual(
            tick(&scheduler, tMs: 2000, formEvents: [.hips(tMs: 2000)], inverted: true),
            .form(.hips))
    }

    func testEachFaultSpeaksItsOwnCue() {
        for (event, fault) in [
            (LiveFormEvent.arch(tMs: 0), FormFault.arch),
            (LiveFormEvent.hollow(tMs: 0), FormFault.hollow),
            (LiveFormEvent.bent(tMs: 0), FormFault.bent),
            (LiveFormEvent.hips(tMs: 0), FormFault.hips),
        ] {
            var scheduler = CueScheduler()
            XCTAssertEqual(
                tick(&scheduler, tMs: 1000, formEvents: [event], inverted: true), .form(fault))
            XCTAssertEqual(scheduler.lastCue, .form(fault))
        }
    }

    func testNoFormCueWhenItsToggleIsOff() {
        var scheduler = CueScheduler()
        scheduler.settings.formCues = false

        XCTAssertNil(tick(&scheduler, tMs: 1000, formEvents: [.arch(tMs: 1000)], inverted: true))
    }

    func testNoFormCueWhenVoiceIsOff() {
        var scheduler = CueScheduler()
        scheduler.settings.voiceCues = false

        XCTAssertNil(tick(&scheduler, tMs: 1000, formEvents: [.bent(tMs: 1000)], inverted: true))
    }

    func testAFramingHintOutranksAWaitingFormCue() {
        var scheduler = CueScheduler()

        XCTAssertEqual(
            tick(&scheduler, tMs: 0, framing: .tooSmall), .framing(.comeCloser))
        // The form cue arrives while the debounce is unspent…
        XCTAssertNil(
            tick(&scheduler, tMs: 1000, formEvents: [.arch(tMs: 1000)], framing: .tooSmall))
        // …and at the first tick both are due, the hint is spoken and the form
        // cue stays pending for the next one.
        XCTAssertEqual(
            tick(&scheduler, tMs: 5000, framing: .multiplePeople), .framing(.onlyYou),
            "a framing hint outranks the form cue")
        XCTAssertEqual(tick(&scheduler, tMs: 9000), .form(.arch))
    }

    // MARK: - Form cue ordering (chainlink #100 review)

    /// A correction must never arrive before the acknowledgement of the same
    /// up stretch: while the athlete is not (yet) *up*, form events are
    /// ignored, not queued.
    func testAFormEventBeforeUpIsNeverSpoken() {
        var scheduler = CueScheduler()

        // Up is not acknowledged yet (the 0.5 s onset has not elapsed): the
        // correction is out of order…
        XCTAssertNil(
            tick(&scheduler, tMs: 1000, formEvents: [.hips(tMs: 1000)], inverted: true, up: false))
        // …and it is still never spoken once the coach gap has passed.
        XCTAssertNil(
            tick(&scheduler, tMs: 4000, inverted: true, up: false),
            "an event ignored while not up is not replayed later")
        XCTAssertNil(scheduler.lastCue)
    }

    func testAFormEventAfterUpSpeaksAfterTheAcknowledgement() {
        var scheduler = CueScheduler()

        // "Up." first…
        XCTAssertEqual(
            tick(&scheduler, tMs: 0, events: [.up(tMs: 0)], inverted: true, up: true), .up)
        // …then the same correction, which waits out the coach gap behind it.
        XCTAssertNil(
            tick(&scheduler, tMs: 1000, formEvents: [.hips(tMs: 1000)], inverted: true, up: true),
            "the form cue follows Up. by formGapS")
        XCTAssertEqual(
            tick(&scheduler, tMs: 2000, inverted: true, up: true), .form(.hips),
            "and speaks once the gap is spent")
    }

    func testAPendingFormCueIsDroppedWhenTheAthleteIsNotUp() {
        var scheduler = CueScheduler()

        // A hint holds the clock, then a form event arrives while up…
        XCTAssertEqual(tick(&scheduler, tMs: 0, framing: .tooSmall), .framing(.comeCloser))
        XCTAssertNil(tick(&scheduler, tMs: 1000, formEvents: [.arch(tMs: 1000)], up: true))
        // …and the athlete comes back down before the coach gap is spent: the
        // waiting correction is dropped, not spoken into an empty frame.
        XCTAssertNil(tick(&scheduler, tMs: 2500, up: false))
        XCTAssertNil(tick(&scheduler, tMs: 6000, up: true), "the dropped form cue never speaks")
    }

    /// A line achieved while a form cue waits clears it: "Line. Hold it." is
    /// never followed by a stale correction to a shape the athlete has left.
    func testALineAchievedDropsAWaitingFormCue() {
        var scheduler = CueScheduler()

        // A hint holds the clock; a form cue queues 1 s later…
        XCTAssertEqual(tick(&scheduler, tMs: 0, framing: .tooSmall), .framing(.comeCloser))
        XCTAssertNil(tick(&scheduler, tMs: 1000, formEvents: [.arch(tMs: 1000)], up: true))
        // …and the line is achieved on the next tick: only the line cue speaks.
        XCTAssertEqual(
            tick(&scheduler, tMs: 1100, events: [.lineAchieved(tMs: 1100)], up: true), .lineHold)
        XCTAssertNil(
            tick(&scheduler, tMs: 3000, up: true),
            "the stale form cue does not follow Line. Hold it.")
    }

    /// Form events inside an achieved line are ignored, not queued — and a
    /// fresh one after `lineLost` speaks again.
    func testAFormEventInsideALineIsIgnoredAndANewOneAfterLineLostSpeaks() {
        var scheduler = CueScheduler()

        XCTAssertEqual(
            tick(&scheduler, tMs: 0, events: [.lineAchieved(tMs: 0)], inverted: true, up: true),
            .lineHold)
        // A fault arrives while the line is held: it is not spoken…
        XCTAssertNil(
            tick(&scheduler, tMs: 2000, formEvents: [.arch(tMs: 2000)], inverted: true, up: true))
        // …and not queued, so it never surfaces after the line ends.
        XCTAssertNil(
            tick(&scheduler, tMs: 3000, events: [.lineLost(tMs: 3000, heldS: 3)], up: true))
        XCTAssertNil(tick(&scheduler, tMs: 5000, up: true), "the in-line fault was not queued")
        // A fresh fault after the line, with the coach gap spent, speaks.
        XCTAssertEqual(
            tick(&scheduler, tMs: 5500, formEvents: [.hips(tMs: 5500)], inverted: true, up: true),
            .form(.hips))
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

    // MARK: - The "Up." acknowledgement (chainlink #100)

    func testUpSpeaksThenLineHoldASecondLater() {
        var scheduler = CueScheduler()

        XCTAssertEqual(
            tick(&scheduler, tMs: 0, events: [.up(tMs: 0)], inverted: true), .up)
        XCTAssertEqual(
            tick(&scheduler, tMs: 1000, events: [.lineAchieved(tMs: 1000)], inverted: true),
            .lineHold)
    }

    func testAPendingUpIsDroppedWhenALineIsPending() {
        var scheduler = CueScheduler()

        // A hint first, so the next cue must wait.
        XCTAssertEqual(tick(&scheduler, tMs: 0, framing: .tooSmall), .framing(.comeCloser))
        // The acknowledgement is queued…
        XCTAssertNil(tick(&scheduler, tMs: 500, events: [.up(tMs: 500)], inverted: true))
        // …then a line arrives: the line cue supersedes the acknowledgement.
        XCTAssertNil(tick(&scheduler, tMs: 800, events: [.lineAchieved(tMs: 800)], inverted: true))
        XCTAssertEqual(tick(&scheduler, tMs: 1500, inverted: true), .lineHold)
        XCTAssertNil(tick(&scheduler, tMs: 2000, inverted: true), "the dropped Up. never speaks")
    }

    func testNoUpWhenVoiceIsOff() {
        var scheduler = CueScheduler()
        scheduler.settings.voiceCues = false

        XCTAssertNil(tick(&scheduler, tMs: 1000, events: [.up(tMs: 1000)], inverted: true))
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
