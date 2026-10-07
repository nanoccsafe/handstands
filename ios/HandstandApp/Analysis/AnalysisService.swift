import Foundation
import HandstandCore
import Observation
import VisionPoseKit

// --------------------------------------------------------------------------- #
// One video, analysed (chainlink #47): the app's first end-to-end run —
// VideoFrameSource reads the frames, `PoseBackend.preferred`'s extractor runs
// the model over them (MediaPipe's clip-level `MediaPipeClipExtractor` when
// its model is bundled — chainlink #45 —, `VideoPoseExtractor` with Apple
// Vision otherwise), `Analyzer.analyze` does the pipeline, `AnalysisSummary`
// is what the screen shows and `SessionStore.recordAnalysis` what History
// remembers.
//
// Everything here is on the main actor; the heavy half (decode + model + the
// pipeline) runs in a `Task.detached`, so the progress bar moves, the Cancel
// button answers and the screen never blocks. `state` is the only thing this
// class owns, and it is written only by the run that still owns it — see
// `generation`.
// --------------------------------------------------------------------------- #

/// What the analysis of one video is doing right now.
@MainActor
@Observable
final class AnalysisService {
    /// Where the run is: nothing yet, working (with 0…1 progress), the
    /// summary to show, or a readable reason it failed. `Equatable` so the
    /// views (and the tests) can compare states directly.
    enum State: Equatable {
        case idle
        case running(progress: Double)
        case finished(AnalysisSummary)
        case failed(String)
    }

    /// The version written to `Session.analysisVersion` by this build's
    /// scoring run — one definition, so every row says which algorithm wrote
    /// it, and old scores can be told apart from new ones. It follows the
    /// backend that ran: `"mediapipe-1"` once the model is bundled,
    /// `"vision-1"` while Vision is the fallback (`PoseBackend.analysisVersion`),
    /// which is what makes old pose caches (#48) recompute instead of replay.
    static var analysisVersion: String { PoseBackend.preferred.analysisVersion }

    /// The cap on frames fed to the model: the app analyses after recording,
    /// and 30 fps is every frame a normal recording has. Doubles as the
    /// estimate `estimatedFrameCount()` uses for the progress bar.
    static let maxFps = 30.0

    private(set) var state: State = .idle

    /// Which run owns `state` (and the right to save a result). Every new
    /// run and every `cancel()` bumps it, so a run that was cancelled or
    /// replaced can neither write the state it left behind nor save a row
    /// nobody asked for: cancelling saves nothing.
    private var generation = 0
    /// The frame extraction of the run in flight — what `cancel()` cancels.
    /// Cancelling a finished task is a no-op, so it is simply overwritten.
    private var extraction: Task<[PostProcessInputFrame], Error>?

    /// Analyses `movie` for `holdType`, off the main actor, updating
    /// `state` as it goes.
    ///
    /// When a `session` and its `store` are given the result is recorded on
    /// that row (`analyzedAt`, score, holds, version) and the run's
    /// diagnostics are written beside the movie as
    /// `<basename>.diagnostics.json` (chainlink #93) — a few KB, local
    /// only, removed with the recording. A video picked from Photos passes
    /// `nil` for both, because it is not in History and has nothing to sit
    /// beside. A video with nobody in it finishes with `holdCount` 0 rather
    /// than failing; only a video that cannot be read (or a model failure)
    /// ends in `.failed`.
    func analyse(movie: URL, holdType: HoldType, session: Session?, store: SessionStore?) async {
        // One run at a time: the previous one stops and loses the right to
        // write anything.
        cancel()
        let mine = generation
        state = .running(progress: 0)
        // The whole run's wall time — what the diagnostics (chainlink #93)
        // record against the "30 s of video in 5–10 s" baseline.
        let startedAt = Date()

        do {
            // Which backend this run is: decided once, so the extraction,
            // the cache version and the saved row all agree. MediaPipe when
            // its model is bundled (chainlink #45's default, per the
            // bake-off), Vision otherwise.
            let backend = PoseBackend.preferred
            // The movie's own numbers (length, frame rate, frame size):
            // the diagnostics record them and "out of frame" is measured
            // against the size. `nil` costs only those — the extraction
            // below is about to fail on the same unreadable file anyway.
            let info = try? await VideoInfoReader.read(url: movie)
            let frameSize = info.map { FrameSize(width: $0.width, height: $0.height) }
            // 1. The reference for this hold, if the app was built with one
            //    (`ReferenceLoader`; the real file is never in the repo).
            let reference = ReferenceLoader.load(for: holdType)
            // 2./3. Every kept frame through the pose backend, with progress.
            let frames = try await extract(movie: movie, backend: backend, generation: mine)
            // A *recording's* frames go to the pose cache beside the movie
            // (chainlink #48) the moment Vision is done with them: the next
            // time the session screen opens it reads them back and never
            // runs Vision again. A picked video has no row in History, so it
            // has no cache either — and a run that was cancelled or replaced
            // while it extracted writes nothing.
            if session != nil, generation == mine {
                try? PoseCache.write(frames: frames, for: movie)
            }
            // 4. The whole pipeline, also off the main actor: the segmenter
            //    and the scorer are CPU work, not something to do between
            //    two redraws. The config carries this backend's
            //    `minVisibility` gate (`PoseBackend`, per backend).
            let config = backend.postProcessConfig
            let analysis = await Task.detached {
                Analyzer.analyze(frames, reference: reference, config: config)
            }.value
            // 5. What the screen shows. The frame size is what lets the
            //    summary say *why* there was no hold (chainlink #93): a
            //    missing joint only reads as "out of frame" against the
            //    picture's own edges.
            let summary = AnalysisSummary.make(
                from: analysis, frames: frames, hasReference: reference != nil,
                frameSize: frameSize
            )
            // Cancelled (or a newer run started) while the pipeline ran: not
            // ours to report, and nothing may be saved.
            guard generation == mine else { return }
            // 6. The row's analysis columns. A clip the app could not measure
            //    (`unusableReason`) is recorded as "analysed, no result": no
            //    score, no holds, and the reason in `note` — "Holds: 0" would
            //    read as "you did not hold" when the truth is "I could not
            //    measure you".
            if let session, let store {
                let unusable = summary.unusableReason
                try store.recordAnalysis(
                    for: session,
                    score: unusable == nil ? summary.clipScore : nil,
                    holdCount: unusable == nil ? summary.holdCount : 0,
                    longestHoldS: unusable == nil ? summary.longestHoldS : 0,
                    version: backend.analysisVersion,
                    note: unusable
                )
                // 7. The diagnostics beside the recording (chainlink #93): a
                //    few KB of numbers — the wall time, the video, the frame
                //    counts, the unknown-reason histogram, the out-of-frame
                //    edges and the holds. Local to the phone, never sent
                //    anywhere, removed by `SessionStore.delete`. Only a
                //    *recording* gets one: a video picked from Photos is a
                //    temporary copy with no History row to sit beside.
                if let info {
                    let diagnostics = AnalysisDiagnostics.make(
                        analysis: analysis,
                        frames: frames,
                        video: info,
                        wallTimeS: Date().timeIntervalSince(startedAt),
                        appVersion: RecordingMetadata.currentAppVersion,
                        backend: backend.rawValue,
                        analysisVersion: backend.analysisVersion
                    )
                    try? AnalysisDiagnostics.write(diagnostics, for: movie)
                }
            }
            state = .finished(summary)
        } catch is CancellationError {
            // `cancel()` already set `.idle`; only write it if this run is
            // still the current one (it is — cancelling bumped the
            // generation, so this branch is the cancelled run itself).
            if generation == mine {
                state = .idle
            }
        } catch {
            if generation == mine {
                state = .failed(Self.readableMessage(error))
            }
        }
    }

    /// Stops the run in flight: no more frames, no result, no saved row —
    /// back to `.idle`.
    func cancel() {
        generation += 1
        extraction?.cancel()
        extraction = nil
        state = .idle
    }

    // MARK: - Private

    /// The extraction task of this run, awaited here: `analyse` stays on the
    /// main actor, the work inside the task does not.
    private func extract(
        movie: URL, backend: PoseBackend, generation mine: Int
    ) async throws -> [PostProcessInputFrame] {
        // Copied out before the task: `Self.maxFps` is main-actor isolated
        // like everything on this class, the number itself is not.
        let maxFps = Self.maxFps
        let task = Task.detached { () async throws -> [PostProcessInputFrame] in
            let source = VideoFrameSource(url: movie, maxFps: maxFps)
            // Called on the extraction's executor; hop home to show it,
            // and only if this run is still the one on screen.
            let show: @Sendable (Double) -> Void = { progress in
                Task { @MainActor in
                    guard self.generation == mine, case .running = self.state else { return }
                    self.state = .running(progress: progress)
                }
            }
            // MediaPipe cannot answer frame by frame — the orientation is
            // chosen over the whole clip (#45) — so it runs its own
            // clip-level extractor; Vision keeps the per-frame
            // `VideoPoseExtractor` it has always had. Both are created
            // inside the task: one clip's worth of state, never sent
            // anywhere.
            switch backend {
            case .mediapipe:
                guard let modelURL = PoseBackend.mediapipeModelURL else {
                    throw AnalysisError.noPoseBackend
                }
                return try await MediaPipeClipExtractor.extract(
                    source: source, modelURL: modelURL, progress: show)
            case .vision:
                guard let service = backend.makeService() else {
                    throw AnalysisError.noPoseBackend
                }
                return try await VideoPoseExtractor.extract(
                    source, service: service, progress: show)
            }
        }
        extraction = task
        return try await task.value
    }

    /// A failure as a sentence a person can read: the kit's own errors in
    /// their own words, everything else through `localizedDescription`.
    private static func readableMessage(_ error: Error) -> String {
        switch error {
        case let error as VideoFrameSourceError: return error.description
        case let error as PoseServiceError: return error.description
        case let error as MediaPipeExtractorError: return error.description
        case let error as AnalysisError: return error.description
        case let error as ScoreReferenceError: return error.description
        default: return error.localizedDescription
        }
    }
}

/// What can go wrong around the pipeline — never the maths itself, whose
/// stages refuse their bad input by precondition rather than by throwing.
enum AnalysisError: Error, CustomStringConvertible {
    /// The pose backend this run picked is not available in this build —
    /// `PoseBackend.preferred` fell through to a backend whose `makeService`
    /// or model file disappeared between the choice and the run (chainlink
    /// #45 made MediaPipe the default whenever its model is bundled; Vision
    /// is the always-available fallback).
    case noPoseBackend

    var description: String {
        switch self {
        case .noPoseBackend:
            return "the pose model is not available in this build"
        }
    }
}

/// What one analysis says on screen (and what a session row remembers): the
/// facts read off an `Analysis`, pure — so the tests can build one from a
/// handful of synthetic frames without a video, a view or a model.
struct AnalysisSummary: Equatable {
    /// How many frames went through the pipeline.
    var frames: Int
    /// How many holds the phase segmenter found.
    var holdCount: Int
    /// The longest of them, in seconds.
    var longestHoldS: Double
    /// The clip's score; `nil` without a reference (or when no hold could
    /// be scored against it).
    var clipScore: Double?
    /// The scored hold's worst faults, worst first, as the scorer spells
    /// them (`"hip_angle"`) — at most `Scorer.topFaults` of them.
    var topFaults: [String]
    /// Was a scoring reference available for this run?
    var hasReference: Bool
    /// Why the analysis could not measure the clip at all; `nil` when it
    /// could. Set from `Analysis` when the post-process has no body length
    /// (or the segmenter says the clip is unusable) — "Holds: 0" would read
    /// as "you did not hold", which is not what an unmeasurable clip means.
    var unusableReason: String?
    /// How many of the analysed frames had a person in them (`detected`) —
    /// "Person found in X of Y frames" on the unmeasurable-clip screen.
    var detectedFrames: Int
    /// What to tell the user when the take had no hold (chainlink #93):
    /// the dominant reason in plain words, then one setup tip — "You were
    /// partly out of frame (top and right edges) for most of the take.
    /// Place the phone about 3 m away at hip height, …". `nil` when there
    /// is a hold to show, and when the clip could not be measured (that
    /// screen has its own words: this is the *measurable* take that simply
    /// found nothing).
    var noHoldExplanation: String?

    /// `summary` from the pipeline's own answer.
    ///
    /// Holds come from `analysis.phases` (the hold runs), never from the
    /// scores: a clip analysed with no reference still has its holds
    /// counted and its longest hold timed. The score is `clipScore`'s
    /// number — the longest *scored* hold, as `Scorer.clipScore` picks it —
    /// and the faults are that hold's `topFaults`.
    ///
    /// `frames` are the frames that went *into* the pipeline: `detected` is
    /// their flag, not something the `Analysis` carries.
    ///
    /// - Parameter frameSize: the pixels the model saw those keypoints in
    ///   (`nil` when the movie's size could not be read, which costs only
    ///   the out-of-frame half of the no-hold explanation).
    static func make(
        from analysis: Analysis,
        frames: [PostProcessInputFrame],
        hasReference: Bool,
        frameSize: FrameSize? = nil
    ) -> AnalysisSummary {
        let durations = analysis.phases.holdDurationsS()
        let unusable = unusableReason(of: analysis)
        return AnalysisSummary(
            frames: analysis.phases.tMs.count,
            holdCount: analysis.phases.holdCount,
            longestHoldS: durations.max() ?? 0,
            clipScore: analysis.clipScore?.score,
            topFaults: (analysis.clipScore?.topFaults ?? [])
                .prefix(Scorer.topFaults)
                .map(\.feature),
            hasReference: hasReference,
            unusableReason: unusable,
            detectedFrames: frames.filter(\.detected).count,
            noHoldExplanation: noHoldMessage(
                of: analysis, frames: frames, size: frameSize, unusable: unusable)
        )
    }

    /// The "No hold found" message, or `nil` when there is nothing to
    /// explain (chainlink #93) — a hold was found, or the clip could not
    /// be measured and says that instead. The reason comes from one pass
    /// over the frames (`AnalysisDiagnostics.Tally`), the same pass the
    /// diagnostics file counts with, so the two can never disagree.
    private static func noHoldMessage(
        of analysis: Analysis,
        frames: [PostProcessInputFrame],
        size: FrameSize?,
        unusable: String?
    ) -> String? {
        guard unusable == nil, analysis.phases.holdCount == 0 else { return nil }
        return AnalysisDiagnostics.Tally(analysis: analysis, frames: frames, size: size)
            .noHoldMessage(frameCount: frames.count)
    }

    /// Why the clip could not be measured, or `nil` when it could — the
    /// post-process's reason first (no body length, e.g. Vision's leg
    /// confidences under `MIN_VISIBILITY`), the segmenter's otherwise.
    /// Spelled once: the diagnostics file (chainlink #93) records the same
    /// `usable` / `unusable_reason` pair the screens show.
    static func unusableReason(of analysis: Analysis) -> String? {
        if !analysis.processed.usable {
            return analysis.processed.bodyLength.reason
        }
        if !analysis.phases.usable {
            return analysis.phases.unusableReason.isEmpty ? nil : analysis.phases.unusableReason
        }
        return nil
    }
}
