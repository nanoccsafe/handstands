import HandstandCore
import XCTest

@testable import HandstandApp

/// The voice the wiring test hears — an order of cues, nothing spoken.
///
/// Declared at file scope rather than nested: a type nested in a `@MainActor`
/// test class would inherit the isolation and could not satisfy the
/// nonisolated `CueSpeaking` requirement.
private final class FakeSpeaker: CueSpeaking, @unchecked Sendable {
    private(set) var spoken: [Cue] = []

    func speak(_ cue: Cue) {
        spoken.append(cue)
    }
}

/// The wiring between the live detector, the scheduler and the voice
/// (chainlink #91): cues reach the speaker in order, the debounce is
/// respected, the banner twins them, and the take's stats pick them up — all
/// driven through `CaptureService.liveUpdate` with fake frames, so no camera,
/// model or microphone is involved.
@MainActor
final class CaptureServiceCueTests: XCTestCase {
    /// One fake live frame: a capture timestamp, whatever the detector
    /// noticed, and what the inference cost.
    private func frame(
        tMs: Int,
        events: [LiveEvent] = [],
        inverted: Bool = false,
        inferenceMs: Double = 40,
        fpsTarget: Double = 10
    ) -> LiveFrame {
        LiveFrame(
            tMs: tMs,
            events: events,
            inverted: inverted,
            inferenceMs: inferenceMs,
            fpsTarget: fpsTarget
        )
    }

    private func makeService() -> (CaptureService, FakeSpeaker) {
        let service = CaptureService()
        let speaker = FakeSpeaker()
        service.speaker = speaker
        return (service, speaker)
    }

    // MARK: - Order and debounce

    func testCuesReachTheSpeakerInOrderWithTheDebounceRespected() {
        let (service, speaker) = makeService()

        // Before the attempt, the framing hint is the first thing said.
        service.liveUpdate(frame(tMs: 1_000), framing: .tooSmall, recording: false)
        XCTAssertEqual(speaker.spoken, [.framing(.comeCloser)])

        // A line only 0.5 s later waits out the line cue's 1 s gap — held,
        // not dropped…
        service.liveUpdate(
            frame(tMs: 1_500, events: [.lineAchieved(tMs: 1_500)], inverted: true),
            framing: .tooSmall,
            recording: true
        )
        XCTAssertEqual(speaker.spoken, [.framing(.comeCloser)])

        // …and spoken at the first tick past it, in order.
        service.liveUpdate(frame(tMs: 2_100, inverted: true), framing: .tooSmall, recording: true)
        XCTAssertEqual(speaker.spoken, [.framing(.comeCloser), .lineHold])

        // The banner twins the last cue, and the take's stats saw the two
        // frames that arrived while the red light was on (the hint was
        // before it).
        XCTAssertEqual(service.cueBanner, .lineHold)
        XCTAssertEqual(service.liveStats.liveFrames, 2)
        XCTAssertEqual(service.liveStats.firstLiveTMs, 1_500)
        XCTAssertEqual(service.liveStats.cues.map(\.cue), ["line_hold"])
        XCTAssertEqual(service.liveStats.cues.first?.tMs, 600, "the cue is 0.6 s into the take")
    }

    func testThreeSecondsBetweenCuesThatAreNotTheLineCue() {
        let (service, speaker) = makeService()

        service.liveUpdate(frame(tMs: 0), framing: .noPerson, recording: false)
        service.liveUpdate(frame(tMs: 1_000), framing: .tooSmall, recording: false)
        XCTAssertEqual(speaker.spoken, [.framing(.comeCloser)], "the first hint is immediate")

        // "Ready." 1.5 s later is stale: dropped, not queued.
        service.liveUpdate(frame(tMs: 2_500), framing: .ok, recording: false)
        XCTAssertEqual(speaker.spoken, [.framing(.comeCloser)])

        // The same transition once the debounce is spent, speaks.
        service.liveUpdate(frame(tMs: 3_000), framing: .noPerson, recording: false)
        service.liveUpdate(frame(tMs: 7_000), framing: .ok, recording: false)
        XCTAssertEqual(speaker.spoken, [.framing(.comeCloser), .ready])
    }

    // MARK: - The toggles

    func testNoFramingHintsWhileTheRedLightIsOn() {
        let (service, speaker) = makeService()

        service.liveUpdate(frame(tMs: 0), framing: .tooSmall, recording: true)
        XCTAssertTrue(speaker.spoken.isEmpty, "the border already says it mid-take")
        XCTAssertTrue(service.liveStats.cues.isEmpty)
    }

    func testVoiceCuesOffMeansNoAttemptCuesButTheHintToggleStillSpeaks() {
        let (service, speaker) = makeService()
        service.cueSettings = CueSettings(voiceCues: false, timeMarks: false, framingHints: true)

        service.liveUpdate(
            frame(tMs: 1_000, events: [.lineAchieved(tMs: 1_000)], inverted: true),
            framing: nil,
            recording: true
        )
        service.liveUpdate(
            frame(tMs: 2_000, events: [.lineLost(tMs: 2_000, heldS: 1.0)], inverted: false),
            framing: nil,
            recording: true
        )
        service.liveUpdate(frame(tMs: 6_000), framing: nil, recording: true)
        XCTAssertTrue(speaker.spoken.isEmpty, "voice cues off: no line cue at all")

        // Framing hints have their own toggle, and it is on — 7 s after the
        // athlete came down, well clear of the 1.5 s cooldown.
        service.liveUpdate(frame(tMs: 9_000), framing: .tooSmall, recording: false)
        XCTAssertEqual(speaker.spoken, [.framing(.comeCloser)])
    }

    func testTimeMarksOnlyWhenTheirToggleIsOn() {
        let (service, speaker) = makeService()

        service.liveUpdate(
            frame(tMs: 5_000, events: [.timeMark(tMs: 5_000, seconds: 5)], inverted: true),
            framing: nil,
            recording: true
        )
        XCTAssertTrue(speaker.spoken.isEmpty, "time marks are off by default")

        service.cueSettings = CueSettings(voiceCues: true, timeMarks: true, framingHints: true)
        service.liveUpdate(
            frame(tMs: 10_000, events: [.timeMark(tMs: 10_000, seconds: 10)], inverted: true),
            framing: nil,
            recording: true
        )
        XCTAssertEqual(speaker.spoken, [.timeMark(10)])
    }

    // MARK: - What the take keeps

    func testLiveStatsCountOnlyTheTakeAndTheDrops() {
        let (service, _) = makeService()

        service.liveUpdate(frame(tMs: 100, inferenceMs: 30), framing: nil, recording: false)
        service.liveUpdate(frame(tMs: 200, inferenceMs: 50), framing: nil, recording: true)
        service.liveUpdate(frame(tMs: 300, inferenceMs: 70), framing: nil, recording: true)
        service.recordingFrameDropped()
        service.recordingFrameDropped()

        XCTAssertEqual(service.liveStats.liveFrames, 2, "pre-attempt frames are not the take's")
        XCTAssertEqual(service.liveStats.inferenceMsTotal, 120, accuracy: 1e-9)
        XCTAssertEqual(service.liveStats.fpsTarget, 10)
        XCTAssertEqual(service.liveStats.droppedFrames, 2)
        XCTAssertEqual(service.liveStats.firstLiveTMs, 200)
    }

    func testTheBannerIsReplacedByTheNextCue() {
        let (service, _) = makeService()

        service.liveUpdate(frame(tMs: 1_000), framing: .tooSmall, recording: false)
        XCTAssertEqual(service.cueBanner, .framing(.comeCloser))
        service.liveUpdate(frame(tMs: 5_000), framing: .partlyOutOfFrame(edges: [.left], missing: []), recording: false)
        XCTAssertEqual(service.cueBanner, .framing(.moveToMiddle), "the newer cue owns the banner")
    }
}
