import Foundation
import HandstandCore
import Observation
import VisionPoseKit

// --------------------------------------------------------------------------- #
// One video, analysed (chainlink #47): the app's first end-to-end run —
// VideoFrameSource reads the frames, PoseBackend.vision runs Apple Vision on
// each one, `Analyzer.analyze` does the pipeline, `AnalysisSummary` is what
// the screen shows and `SessionStore.recordAnalysis` what History remembers.
//
// Everything here is on the main actor; the heavy half (decode + Vision + the
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
    /// it, and old scores can be told apart from new ones.
    static let analysisVersion = "vision-1"

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
    /// that row (`analyzedAt`, score, holds, version) — a video picked from
    /// Photos passes `nil` for both, because it is not in History. A video
    /// with nobody in it finishes with `holdCount` 0 rather than failing;
    /// only a video that cannot be read (or a model failure) ends in
    /// `.failed`.
    func analyse(movie: URL, holdType: HoldType, session: Session?, store: SessionStore?) async {
        // One run at a time: the previous one stops and loses the right to
        // write anything.
        cancel()
        let mine = generation
        state = .running(progress: 0)

        do {
            // 1. The reference for this hold, if the app was built with one
            //    (`ReferenceLoader`; the real file is never in the repo).
            let reference = ReferenceLoader.load(for: holdType)
            // 2./3. Every kept frame through the pose backend, with progress.
            let frames = try await extract(movie: movie, generation: mine)
            // 4. The whole pipeline, also off the main actor: the segmenter
            //    and the scorer are CPU work, not something to do between
            //    two redraws.
            let analysis = await Task.detached {
                Analyzer.analyze(frames, reference: reference)
            }.value
            // 5. What the screen shows.
            let summary = AnalysisSummary.make(from: analysis, hasReference: reference != nil)
            // Cancelled (or a newer run started) while the pipeline ran: not
            // ours to report, and nothing may be saved.
            guard generation == mine else { return }
            // 6. The row's analysis columns.
            if let session, let store {
                try store.recordAnalysis(
                    for: session,
                    score: summary.clipScore,
                    holdCount: summary.holdCount,
                    longestHoldS: summary.longestHoldS,
                    version: Self.analysisVersion
                )
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
    private func extract(movie: URL, generation mine: Int) async throws -> [PostProcessInputFrame] {
        // Copied out before the task: `Self.maxFps` is main-actor isolated
        // like everything on this class, the number itself is not.
        let maxFps = Self.maxFps
        let task = Task.detached { () async throws -> [PostProcessInputFrame] in
            // The service is created *inside* the task: one clip's worth of
            // state (the auto-rotation), never sent anywhere.
            guard let service = PoseBackend.vision.makeService() else {
                throw AnalysisError.noPoseBackend
            }
            let source = VideoFrameSource(url: movie, maxFps: maxFps)
            return try await VideoPoseExtractor.extract(source, service: service) { progress in
                // Called on the extraction's executor; hop home to show it,
                // and only if this run is still the one on screen.
                Task { @MainActor in
                    guard self.generation == mine, case .running = self.state else { return }
                    self.state = .running(progress: progress)
                }
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
        case let error as AnalysisError: return error.description
        case let error as ScoreReferenceError: return error.description
        default: return error.localizedDescription
        }
    }
}

/// What can go wrong around the pipeline — never the maths itself, whose
/// stages refuse their bad input by precondition rather than by throwing.
enum AnalysisError: Error, CustomStringConvertible {
    /// The pose backend asked for is not available in this build
    /// (`PoseBackend.vision` is always available; MediaPipe, #45, is not
    /// yet).
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

    /// `summary` from the pipeline's own answer.
    ///
    /// Holds come from `analysis.phases` (the hold runs), never from the
    /// scores: a clip analysed with no reference still has its holds
    /// counted and its longest hold timed. The score is `clipScore`'s
    /// number — the longest *scored* hold, as `Scorer.clipScore` picks it —
    /// and the faults are that hold's `topFaults`.
    static func make(from analysis: Analysis, hasReference: Bool) -> AnalysisSummary {
        let durations = analysis.phases.holdDurationsS()
        return AnalysisSummary(
            frames: analysis.phases.tMs.count,
            holdCount: analysis.phases.holdCount,
            longestHoldS: durations.max() ?? 0,
            clipScore: analysis.clipScore?.score,
            topFaults: (analysis.clipScore?.topFaults ?? [])
                .prefix(Scorer.topFaults)
                .map(\.feature),
            hasReference: hasReference
        )
    }
}
