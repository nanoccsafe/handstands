import XCTest

@testable import HandstandApp
import HandstandCore

/// The analysis diagnostics and the "No hold found" explanation (chainlink
/// #93): the few KB written beside a recording, the out-of-frame reason
/// read off the keypoints, the dominant-reason choice and the words the
/// screen shows. No video, no model, no view: every clip here is built by
/// hand out of `PostProcessInputFrame`s, as the analysis tests do, in a
/// 200 × 400 picture (3% of it is 6 px sideways, 12 px up and down).
final class AnalysisDiagnosticsTests: XCTestCase {
    // MARK: - The synthetic clips

    /// How many frames a clip has, at 33 ms apart (the ~30 fps the app
    /// analyses at).
    private static let frameCount = 60
    private static let stepMs = 33
    /// The picture every clip below stands in.
    private static let size = FrameSize(width: 200, height: 400)

    /// One frame of a person **standing**: whole, confident, well inside
    /// the picture, and — crucially — never upside down, so the phase
    /// segmenter finds no hold whatever happens to the wrists. The base for
    /// every "why was there no hold" case: a take that could have been
    /// measured, and was not held.
    private static func standingFrame(tMs: Int) -> PostProcessInputFrame {
        func point(_ y: Double, x: Double) -> Keypoint {
            Keypoint(x: x, y: y, visibility: 0.9)
        }
        let joints: [Joint: Keypoint] = [
            .nose: point(40, x: 106),
            .leftShoulder: point(75, x: 92),
            .rightShoulder: point(75, x: 100),
            .leftElbow: point(115, x: 92),
            .rightElbow: point(115, x: 100),
            .leftWrist: point(155, x: 92),
            .rightWrist: point(155, x: 100),
            .leftHip: point(175, x: 92),
            .rightHip: point(175, x: 100),
            .leftKnee: point(245, x: 92),
            .rightKnee: point(245, x: 100),
            .leftAnkle: point(315, x: 92),
            .rightAnkle: point(315, x: 100),
            .leftFootIndex: point(345, x: 92),
            .rightFootIndex: point(345, x: 100),
        ]
        return PostProcessInputFrame(
            tMs: tMs, detected: true, trainerContact: false, joints: joints)
    }

    /// One frame of a person **holding** a handstand: hands at the bottom,
    /// feet above them, still and straight — exactly the hold
    /// `PhaseSegmenter` looks for, so a clip of these has a hold to count
    /// (the same body `AnalysisSummaryTests` builds, in the same pixels).
    private static func holdingFrame(tMs: Int) -> PostProcessInputFrame {
        func point(_ y: Double, x: Double) -> Keypoint {
            Keypoint(x: x, y: y, visibility: 0.9)
        }
        let joints: [Joint: Keypoint] = [
            .nose: Keypoint(x: 106, y: 195, visibility: 0.9),
            .leftShoulder: point(210, x: 92),
            .rightShoulder: point(210, x: 100),
            .leftElbow: point(250, x: 92),
            .rightElbow: point(250, x: 100),
            .leftWrist: point(300, x: 92),
            .rightWrist: point(300, x: 100),
            .leftHip: point(150, x: 92),
            .rightHip: point(150, x: 100),
            .leftKnee: point(100, x: 92),
            .rightKnee: point(100, x: 100),
            .leftAnkle: point(55, x: 92),
            .rightAnkle: point(55, x: 100),
            .leftFootIndex: point(40, x: 92),
            .rightFootIndex: point(40, x: 100),
        ]
        return PostProcessInputFrame(
            tMs: tMs, detected: true, trainerContact: false, joints: joints)
    }

    /// A frame nobody was found in — an empty picture.
    private static func emptyFrame(tMs: Int) -> PostProcessInputFrame {
        PostProcessInputFrame(tMs: tMs, detected: false, trainerContact: false, joints: [:])
    }

    /// The same standing body with **both wrists gone**: the phase
    /// segmenter can only report `no_visible_wrist`, which is the whole
    /// point — the diagnosis of *why* is what this task adds.
    private static func withoutWrists(_ frame: PostProcessInputFrame) -> PostProcessInputFrame {
        var frame = frame
        frame.joints[.leftWrist] = nil
        frame.joints[.rightWrist] = nil
        return frame
    }

    /// The same body pushed into the **top right corner**: still detected,
    /// still missing its wrists, but now the visible body touches two
    /// borders — a take cut off by the picture, not by the model.
    private static func offTheEdge(_ frame: PostProcessInputFrame) -> PostProcessInputFrame {
        var frame = withoutWrists(frame)
        frame.joints = frame.joints.mapValues { point in
            Keypoint(x: point.x + 100, y: point.y - 30, visibility: point.visibility)
        }
        return frame
    }

    /// `frameCount` frames: `body(index, tMs)` decides what each one is.
    private static func clip(
        _ body: (Int, Int) -> PostProcessInputFrame
    ) -> [PostProcessInputFrame] {
        (0..<frameCount).map { index in body(index, index * stepMs) }
    }

    /// Every frame standing, as above.
    private static func standingClip() -> [PostProcessInputFrame] {
        clip { _, tMs in standingFrame(tMs: tMs) }
    }

    /// The movie's own numbers — 60 frames of a two-second, 60 fps take.
    private static let video = VideoInfo(
        duration: 1.95, width: 200, height: 400, frameRate: 59.9)

    /// The document for `frames`, with the fixed bits the tests do not care
    /// about spelled once.
    private func make(
        _ frames: [PostProcessInputFrame]
    ) -> (diagnostics: AnalysisDiagnostics, analysis: Analysis) {
        let analysis = Analyzer.analyze(frames, reference: nil)
        let diagnostics = AnalysisDiagnostics.make(
            analysis: analysis,
            frames: frames,
            video: Self.video,
            wallTimeS: 6.42,
            appVersion: "0.1.0",
            backend: "mediapipe",
            analysisVersion: "mediapipe-1"
        )
        return (diagnostics, analysis)
    }

    /// The tally the dominant reason is read from — the same pass
    /// `AnalysisDiagnostics.make` and the summary both take.
    private func tallyFor(
        _ frames: [PostProcessInputFrame]
    ) -> AnalysisDiagnostics.Tally {
        AnalysisDiagnostics.Tally(
            analysis: Analyzer.analyze(frames, reference: nil),
            frames: frames,
            size: Self.size)
    }

    /// A temp folder with an empty `.mov` in it; the caller removes it.
    private func makeMovie() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnalysisDiagnosticsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let movie = directory.appendingPathComponent("20260928-143059.mov")
        try Data().write(to: movie)
        return movie
    }

    private func remove(_ movie: URL) {
        try? FileManager.default.removeItem(at: movie.deletingLastPathComponent())
    }

    /// A number out of a JSON object, whatever shape `JSONSerialization`
    /// gave it (an integer in the file is an `NSNumber` either way).
    private func number(_ object: [String: Any], _ key: String) throws -> Double {
        try XCTUnwrap((object[key] as? NSNumber)?.doubleValue, "no number \(key)")
    }

    // MARK: - The file

    func testTheDiagnosticsFileSitsBesideTheMovieUnderItsOwnName() {
        let movie = URL(fileURLWithPath: "/Recordings/20260928-143059.mov")
        let url = AnalysisDiagnostics.url(for: movie)

        XCTAssertEqual(url.lastPathComponent, "20260928-143059.diagnostics.json")
        XCTAssertEqual(url.deletingLastPathComponent().path, "/Recordings")
    }

    /// The round trip: what was written is what reads back, and the file is
    /// a few KB of *counts* — no keypoints, no per-frame rows.
    func testWhatWasWrittenReadsBackExactly() throws {
        let movie = try makeMovie()
        defer { remove(movie) }
        let frames = Self.clip { index, tMs in
            index < 36 ? Self.standingFrame(tMs: tMs) : Self.emptyFrame(tMs: tMs)
        }
        let (written, _) = make(frames)

        try AnalysisDiagnostics.write(written, for: movie)

        XCTAssertEqual(AnalysisDiagnostics.read(for: movie), written)
        XCTAssertEqual(written.framesTotal, Self.frameCount)
        XCTAssertEqual(written.framesWithPerson, 36)
        XCTAssertEqual(written.framesWithPersonPct, 60, accuracy: 0.01)
        XCTAssertEqual(written.video.durationS, Self.video.duration, accuracy: 0.001)
        XCTAssertEqual(written.video.fps, Self.video.frameRate, accuracy: 0.01)
        XCTAssertEqual(written.video.width, Self.video.width, "the movie's own numbers")
        XCTAssertEqual(written.video.height, Self.video.height)
        XCTAssertEqual(written.appVersion, "0.1.0")
        XCTAssertEqual(written.backend, "mediapipe")
        XCTAssertEqual(written.analysisVersion, "mediapipe-1")
        XCTAssertEqual(written.analysisWallTimeS, 6.42, accuracy: 0.01)
        XCTAssertEqual(written.usable, true)
        XCTAssertEqual(written.unusableReason, "")
        XCTAssertEqual(written.holdCount, 0)
        XCTAssertTrue(written.holdDurationsS.isEmpty)
    }

    /// The document chainlink #93 specifies, spelled out: schema, versions,
    /// the video, the frame counts and their percentage, the two histograms
    /// and the holds — snake_case, because the lead reads these with `jq`.
    func testTheDocumentIsTheShapeTheTaskSpecifies() throws {
        let movie = try makeMovie()
        defer { remove(movie) }
        let frames = Self.clip { index, tMs in
            index < 40 ? Self.offTheEdge(Self.standingFrame(tMs: tMs)) : Self.standingFrame(
                tMs: tMs)
        }
        try AnalysisDiagnostics.write(make(frames).diagnostics, for: movie)

        let data = try Data(contentsOf: AnalysisDiagnostics.url(for: movie))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(try number(object, "schema"), 1)
        XCTAssertEqual(object["app_version"] as? String, "0.1.0")
        XCTAssertEqual(object["backend"] as? String, "mediapipe")
        XCTAssertEqual(object["analysis_version"] as? String, "mediapipe-1")
        XCTAssertEqual(try number(object, "analysis_wall_time_s"), 6.42, accuracy: 0.01)
        XCTAssertEqual(try number(object, "analysed_fps"), 30.3, accuracy: 0.01)
        XCTAssertEqual(try number(object, "frames_total"), 60)
        XCTAssertEqual(try number(object, "frames_with_person"), 60)
        XCTAssertEqual(try number(object, "frames_with_person_pct"), 100, accuracy: 0.01)

        let video = try XCTUnwrap(object["video"] as? [String: Any])
        XCTAssertEqual(try number(video, "duration_s"), 1.95, accuracy: 0.001)
        XCTAssertEqual(try number(video, "fps"), 59.9, accuracy: 0.01)
        XCTAssertEqual(try number(video, "width"), 200)
        XCTAssertEqual(try number(video, "height"), 400)

        let reasons = try XCTUnwrap(object["unknown_reasons"] as? [String: Any])
        XCTAssertEqual(try number(reasons, "no_visible_wrist"), 40)

        let outOfFrame = try XCTUnwrap(object["out_of_frame"] as? [String: Any])
        XCTAssertEqual(try number(outOfFrame, "partly"), 40)
        XCTAssertEqual(try number(outOfFrame, "not_in_frame"), 0)
        let edges = try XCTUnwrap(outOfFrame["edges"] as? [String: Any])
        XCTAssertEqual(try number(edges, "top"), 40)
        XCTAssertEqual(try number(edges, "right"), 40)

        XCTAssertEqual(object["dominant_reason"] as? String, "out_of_frame")
        XCTAssertEqual(try number(object, "hold_count"), 0)
        XCTAssertEqual(object["hold_durations_s"] as? [Double], [Double]())
        XCTAssertEqual(object["usable"] as? Bool, true)
        XCTAssertEqual(object["unusable_reason"] as? String, "")
    }

    /// No file, unreadable bytes, or a schema a later build wrote: `nil`,
    /// never a document this build would misunderstand.
    func testAMissingOrForeignFileReadsAsNothing() throws {
        let movie = try makeMovie()
        defer { remove(movie) }

        XCTAssertNil(AnalysisDiagnostics.read(for: movie), "no file is no diagnostics")

        try Data("{ not json at all".utf8).write(to: AnalysisDiagnostics.url(for: movie))
        XCTAssertNil(AnalysisDiagnostics.read(for: movie))

        try Data(#"{"schema": 2, "frames_total": 10}"#.utf8).write(
            to: AnalysisDiagnostics.url(for: movie))
        XCTAssertNil(AnalysisDiagnostics.read(for: movie), "a foreign schema is a foreign question")
    }

    /// "Not too much bloat": a thousand frames cost no more bytes than
    /// sixty — the file carries counts, never the frames themselves.
    func testTheFileStaysSmallWhateverTheTake() throws {
        let movie = try makeMovie()
        defer { remove(movie) }
        let frames = (0..<1000).map { index in
            Self.standingFrame(tMs: index * Self.stepMs)
        }

        try AnalysisDiagnostics.write(make(frames).diagnostics, for: movie)

        let attributes = try FileManager.default.attributesOfItem(
            atPath: AnalysisDiagnostics.url(for: movie).path)
        let bytes = try XCTUnwrap(attributes[.size] as? Int)
        XCTAssertLessThan(bytes, 4096, "a diagnostics file is a few KB, not a second take")
    }

    // MARK: - The out-of-frame reason

    /// The framing verdict, one case at a time: what "partly out of frame"
    /// is made of (a missing wrist/ankle/hip *and* a body touching a
    /// border) and what it is not.
    func testFramingIsMissingJointsAndABorderTogether() {
        let standing = Self.standingFrame(tMs: 0)

        // Nobody at all.
        XCTAssertEqual(Framing(frame: Self.emptyFrame(tMs: 0), phaseReason: "no_visible_wrist", size: Self.size), .notInFrame)
        // Both wrists gone and the body in the corner: cut off.
        XCTAssertEqual(
            Framing(
                frame: Self.offTheEdge(standing), phaseReason: "no_visible_wrist",
                size: Self.size),
            .outOfFrame([.top, .right]))
        // Both wrists gone but the body in the middle of the picture: the
        // model lost them, the picture did not take them.
        XCTAssertEqual(
            Framing(
                frame: Self.withoutWrists(standing), phaseReason: "no_visible_wrist",
                size: Self.size),
            .inFrame)
        // A whole body at an edge is still in the picture — the reason is
        // what decides it, not the border alone.
        XCTAssertEqual(
            Framing(frame: standing, phaseReason: "", size: Self.size), .inFrame)
        // A lost trainer is not the framing either.
        XCTAssertEqual(
            Framing(frame: standing, phaseReason: "trainer_contact", size: Self.size),
            .inFrame)
        // No size, no border: only "nobody detected" is claimed.
        XCTAssertEqual(
            Framing(
                frame: Self.offTheEdge(standing), phaseReason: "no_visible_wrist", size: nil),
            .inFrame)
    }

    /// 3% of the picture is the line: a joint on this side of it touches
    /// that edge, one past it does not.
    func testTheBorderIsThreePercentOfTheFrame() {
        let size = FrameSize(width: 200, height: 400)

        XCTAssertEqual(size.edges(x: 100, y: 0), [.top])
        XCTAssertEqual(size.edges(x: 100, y: 12), [.top], "3% of 400 is 12")
        XCTAssertTrue(size.edges(x: 100, y: 13).isEmpty, "just inside the top")
        XCTAssertEqual(size.edges(x: 100, y: 400), [.bottom])
        XCTAssertEqual(size.edges(x: 0, y: 200), [.left])
        XCTAssertEqual(size.edges(x: 6, y: 200), [.left], "3% of 200 is 6")
        XCTAssertTrue(size.edges(x: 7, y: 200).isEmpty, "just inside the left")
        XCTAssertEqual(size.edges(x: 200, y: 200), [.right])
        // A body clipped by two edges names both, in `Edge.allCases` order.
        XCTAssertEqual(size.edges(x: 200, y: 0), [.top, .right])
        // A size the analysis never had claims nothing.
        XCTAssertTrue(FrameSize(width: 0, height: 0).edges(x: 0, y: 0).isEmpty)
    }

    /// The counts the document carries: the segmenter's own histogram
    /// verbatim, the out-of-frame one beside it, and the dominant reason
    /// the framing wins.
    func testTheCountsAreTheOnesTheTaskLists() throws {
        let frames = Self.clip { index, tMs in
            index < 40 ? Self.offTheEdge(Self.standingFrame(tMs: tMs)) : Self.standingFrame(
                tMs: tMs)
        }

        let (diagnostics, _) = make(frames)

        // The segmenter's own words, unmodified: 40 frames lost their
        // wrists, 20 were fine.
        XCTAssertEqual(diagnostics.unknownReasons, ["no_visible_wrist": 40])
        // …and the framing read off the keypoints beside them.
        XCTAssertEqual(diagnostics.outOfFrame.partly, 40)
        XCTAssertEqual(diagnostics.outOfFrame.notInFrame, 0)
        XCTAssertEqual(diagnostics.outOfFrame.edges, ["top": 40, "right": 40])
        // The exclusive buckets the dominant reason is picked from: the 40
        // frames count as out-of-frame, not as the missing wrist.
        let tally = tallyFor(frames)
        XCTAssertEqual(tally.buckets, ["out_of_frame": 40])
        XCTAssertEqual(tally.dominant, "out_of_frame")
        XCTAssertEqual(tally.dominantEdges, [.top, .right])
        XCTAssertEqual(diagnostics.dominantReason, "out_of_frame")
    }

    // MARK: - The dominant-reason choice

    /// Chainlink #93's own case: a take where most frames had nobody in
    /// them — the handstand happened outside the picture. The message is
    /// about the framing, not about the wrists the segmenter reported.
    func testATakeNobodyWasInReadsAsOutOfFrame() throws {
        let frames = Self.clip { index, tMs in
            index < 24 ? Self.standingFrame(tMs: tMs) : Self.emptyFrame(tMs: tMs)
        }

        let (diagnostics, _) = make(frames)
        let tally = tallyFor(frames)

        XCTAssertEqual(tally.notInFrame, 36)
        XCTAssertEqual(tally.buckets, ["not_in_frame": 36], "nobody detected is nobody in frame")
        XCTAssertEqual(tally.noHoldReason(frameCount: frames.count), .notInFrame)
        XCTAssertEqual(diagnostics.dominantReason, "not_in_frame")
        XCTAssertEqual(diagnostics.outOfFrame.notInFrame, 36)
        XCTAssertEqual(diagnostics.framesWithPerson, 24)
        XCTAssertEqual(diagnostics.framesWithPersonPct, 40, accuracy: 0.01)
        XCTAssertEqual(
            diagnostics.noHoldExplanation,
            "You were out of frame for most of the take. " + NoHoldExplanation.positionTip)
    }

    /// The preference the task asks for: when the framing explains more
    /// frames than the missing joint it causes, the message is about the
    /// framing — with the edges it was cut off at, and the distance tip.
    func testTheFramingOutranksTheMissingJointItCauses() throws {
        let frames = Self.clip { index, tMs -> PostProcessInputFrame in
            let standing = Self.standingFrame(tMs: tMs)
            if index < 30 { return Self.offTheEdge(standing) }
            if index < 55 { return Self.withoutWrists(standing) }
            return standing
        }

        let (diagnostics, _) = make(frames)
        let tally = tallyFor(frames)

        // The segmenter only ever sees wrists it could not find.
        XCTAssertEqual(diagnostics.unknownReasons, ["no_visible_wrist": 55])
        // The exclusive buckets put the framing first…
        XCTAssertEqual(tally.buckets, ["out_of_frame": 30, "no_visible_wrist": 25])
        XCTAssertEqual(tally.dominant, "out_of_frame")
        XCTAssertEqual(tally.dominantEdges, [.top, .right])
        XCTAssertEqual(tally.noHoldReason(frameCount: frames.count), .outOfFrame)
        // …and the message says so, edges and all, ending in the tip.
        let message = try XCTUnwrap(diagnostics.noHoldExplanation)
        XCTAssertEqual(
            message,
            "You were partly out of frame (top and right edges) for most of the take. "
                + NoHoldExplanation.positionTip)
        XCTAssertTrue(message.contains(NoHoldExplanation.positionTip), "it ends with a tip")
    }

    /// Mostly visible, wrists lost in some frames: the framing explains
    /// 25 of 60 and the missing joint 25 — a take whose wrists the model
    /// simply did not read says so, and points at the light.
    func testHandsNobodyCouldSeeSaySo() throws {
        let frames = Self.clip { index, tMs -> PostProcessInputFrame in
            let standing = Self.standingFrame(tMs: tMs)
            return index < 40 ? Self.withoutWrists(standing) : standing
        }

        let (diagnostics, _) = make(frames)
        let tally = tallyFor(frames)

        XCTAssertEqual(tally.buckets, ["no_visible_wrist": 40])
        XCTAssertEqual(tally.noHoldReason(frameCount: frames.count), .noVisibleWrist)
        XCTAssertEqual(diagnostics.dominantReason, "no_visible_wrist")
        let message = try XCTUnwrap(diagnostics.noHoldExplanation)
        XCTAssertEqual(
            message,
            "Your hands weren't visible for most of the take. " + NoHoldExplanation.lightTip)
    }

    /// A take seen well enough is not "…for most of the take": a few frames
    /// nobody was in does not make the framing the story, the missing hold
    /// does.
    func testAMostlyVisibleTakeIsExplainedAsNeverUpsideDown() throws {
        let frames = Self.clip { index, tMs in
            index < 10 ? Self.emptyFrame(tMs: tMs) : Self.standingFrame(tMs: tMs)
        }

        let tally = tallyFor(frames)

        XCTAssertEqual(tally.notInFrame, 10, "the frames are still counted")
        XCTAssertEqual(tally.noHoldReason(frameCount: frames.count), .notInverted)
    }

    /// Nothing went wrong at all: every frame seen, nobody upside down —
    /// which is exactly what it says.
    func testATakeNothingWasWrongWithSaysItWasNeverUpsideDown() throws {
        let (diagnostics, _) = make(Self.standingClip())
        let tally = tallyFor(Self.standingClip())

        XCTAssertTrue(tally.buckets.isEmpty, "every frame was seen")
        XCTAssertEqual(tally.dominant, "")
        XCTAssertEqual(tally.noHoldReason(frameCount: Self.frameCount), .notInverted)
        XCTAssertEqual(diagnostics.unknownReasons, [:])
        XCTAssertEqual(diagnostics.outOfFrame.partly, 0)
        XCTAssertEqual(diagnostics.outOfFrame.notInFrame, 0)
        XCTAssertEqual(diagnostics.dominantReason, "not_inverted")
        XCTAssertEqual(
            diagnostics.noHoldExplanation,
            "No one was upside down long enough (0.3 s minimum). "
                + NoHoldExplanation.positionTip)
    }

    // MARK: - The reason table

    /// Every reason says one sentence and ends in one tip — the table
    /// chainlink #93 asks for, small enough to reword in one place.
    func testEveryReasonHasOneSentenceAndOneTip() {
        for reason in NoHoldReason.allCases {
            let explanation = NoHoldExplanation(
                reason: reason, edges: reason == .outOfFrame ? [.top, .right] : [])

            XCTAssertFalse(explanation.sentence.isEmpty, "\(reason)")
            XCTAssertTrue(
                explanation.sentence.hasSuffix("."), "\(reason) is one finished sentence")
            XCTAssertFalse(explanation.tip.isEmpty, "\(reason) ends in a tip")
            XCTAssertEqual(
                explanation.text, "\(explanation.sentence) \(explanation.tip)",
                "\(reason) is the sentence then the tip, once")
        }
        // The tip table has three entries for the seven reasons: where to
        // put the camera, how to light it, and keep it clear.
        let tips = Set(NoHoldReason.allCases.map(NoHoldExplanation.tip(for:)))
        XCTAssertEqual(tips, [
            NoHoldExplanation.positionTip,
            NoHoldExplanation.lightTip,
            NoHoldExplanation.clearViewTip,
        ])
        XCTAssertEqual(
            NoHoldExplanation(reason: .outOfFrame, edges: [.top, .right]).sentence,
            "You were partly out of frame (top and right edges) for most of the take.")
        XCTAssertEqual(
            NoHoldExplanation(reason: .notInverted).sentence,
            "No one was upside down long enough (0.3 s minimum).")
    }

    // MARK: - What the screens show

    /// The summary the analysis panel renders: a measurable take with no
    /// hold carries the explanation, in full.
    func testTheSummaryExplainsAMeasurableTakeWithNoHold() throws {
        let frames = Self.clip { index, tMs in
            index < 24 ? Self.standingFrame(tMs: tMs) : Self.emptyFrame(tMs: tMs)
        }
        let analysis = Analyzer.analyze(frames, reference: nil)

        let summary = AnalysisSummary.make(
            from: analysis, frames: frames, hasReference: false, frameSize: Self.size)

        XCTAssertNil(summary.unusableReason, "the clip was measured")
        XCTAssertEqual(summary.holdCount, 0)
        XCTAssertEqual(
            summary.noHoldExplanation,
            "You were out of frame for most of the take. " + NoHoldExplanation.positionTip)
    }

    /// A take with a hold has nothing to explain.
    func testTheSummarySaysNothingWhenThereIsAHold() throws {
        let frames = (0..<Self.frameCount).map { index in
            Self.holdingFrame(tMs: index * Self.stepMs)
        }
        let analysis = Analyzer.analyze(frames, reference: nil)

        let summary = AnalysisSummary.make(
            from: analysis, frames: frames, hasReference: false, frameSize: Self.size)

        XCTAssertEqual(analysis.phases.holdCount, 1, "the synthetic clip is exactly one hold")
        XCTAssertNil(summary.noHoldExplanation, "there is a hold to show")
    }

    /// A clip the app could not measure is not "no hold found" — that
    /// screen says it could not measure.
    func testTheSummarySaysNothingAboutAClipItCouldNotMeasure() throws {
        let frames = (0..<10).map { index in Self.emptyFrame(tMs: index * Self.stepMs) }
        let analysis = Analyzer.analyze(frames, reference: nil)

        let summary = AnalysisSummary.make(
            from: analysis, frames: frames, hasReference: false, frameSize: Self.size)

        XCTAssertNotNil(summary.unusableReason, "nobody to measure a body length from")
        XCTAssertEqual(summary.holdCount, 0)
        XCTAssertNil(summary.noHoldExplanation, "that screen has its own words")
    }

    /// A hold found and a clip measured both leave the document's own
    /// `dominant_reason` empty: it is a reason *for no hold*.
    func testTheDiagnosticsKeepTheirReasonOnlyWhenThereWasNone() throws {
        let holding = (0..<Self.frameCount).map { index in
            Self.holdingFrame(tMs: index * Self.stepMs)
        }

        let (withHold, _) = make(holding)
        XCTAssertEqual(withHold.holdCount, 1)
        XCTAssertEqual(withHold.dominantReason, "")
        XCTAssertNil(withHold.noHoldExplanation)

        let nobody = (0..<10).map { index in Self.emptyFrame(tMs: index * Self.stepMs) }
        let (unusable, analysis) = make(nobody)
        XCTAssertFalse(unusable.usable, "no body length in a clip with no body")
        XCTAssertEqual(
            unusable.unusableReason,
            AnalysisSummary.unusableReason(of: analysis),
            "the file says exactly what the screens say")
        XCTAssertEqual(unusable.dominantReason, "", "an unmeasurable clip is explained elsewhere")
        XCTAssertNil(unusable.noHoldExplanation)
    }

    /// The rate the pipeline was fed at, read off the frames' own clock:
    /// 59 steps of 33 ms, which is the ~30 fps the extraction caps at.
    func testAnalysedFpsComesFromTheFramesThemselves() {
        XCTAssertEqual(AnalysisDiagnostics.analysedFps(Self.standingClip()), 30.3, accuracy: 0.01)
        XCTAssertEqual(AnalysisDiagnostics.analysedFps([]), 0)
        XCTAssertEqual(
            AnalysisDiagnostics.analysedFps([Self.standingFrame(tMs: 0)]), 0,
            "one frame has no span to measure")
    }
}
