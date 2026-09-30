import AnnotatedVideo
import Foundation
import HandstandCore
import Observation
import VisionPoseKit

// --------------------------------------------------------------------------- #
// The annotated video export (chainlink #50), the analysis run's shape: one
// state, one generation, the heavy half in a `Task` — so the progress bar
// moves, Cancel answers and the screen never blocks.
//
// Everything here is on the main actor; `AnnotatedExport.run` is the work
// (decode → CoreGraphics → AVAssetWriter) and it runs off it, reporting
// progress through a closure that hops home. Like `AnalysisService`, a run
// that was cancelled or replaced can neither write the state it left behind
// nor pretend it finished.
//
// The file itself is the export's business: a cancelled or failed run
// deletes its own partial file, so `state` is all this class owns.
// --------------------------------------------------------------------------- #

/// What the export of one session is doing right now.
@MainActor
@Observable
final class ExportService {
    /// Where the export is: nothing yet, working (with 0…1 progress), the
    /// finished movie to share, or a readable reason it failed. `Equatable`
    /// so the views (and the tests) can compare states directly.
    enum State: Equatable {
        case idle
        case running(progress: Double)
        case finished(URL)
        case failed(String)
    }

    private(set) var state: State = .idle

    /// Which run owns `state`. Every new run and every `cancel()` bumps it,
    /// so a run that was cancelled or replaced can neither report progress
    /// nor claim it finished.
    private var generation = 0
    /// The export of the run in flight — what `cancel()` cancels.
    /// Cancelling a finished task is a no-op, so it is simply overwritten.
    private var task: Task<Void, Never>?

    /// The finished movie this run wrote, if there is one — the URL the
    /// share link shows and the screen deletes when it goes away
    /// (chainlink #50). Every other state holds no file: a cancelled or
    /// failed run deletes its own.
    var finishedURL: URL? {
        if case .finished(let url) = state { return url }
        return nil
    }

    /// Exports `analysis` over `movie` into `output`, off the main actor,
    /// updating `state` as it goes. Returns when the run is over — finished
    /// (`.finished(output)`), cancelled (`.idle`) or failed (`.failed`, with
    /// a sentence a person can read).
    ///
    /// The output file is **overwritten**, and a run that stops early leaves
    /// no file at all: `AnnotatedExport.run` deletes its own partial movie,
    /// which is what makes Cancel a button that costs nothing.
    func export(
        movie: URL, analysis: Analysis, reference: ScoreReference?, to output: URL
    ) async {
        // One run at a time: the previous one stops and loses the right to
        // write anything.
        cancel()
        let mine = generation
        state = .running(progress: 0)

        let run = AnnotatedExport(
            source: movie, analysis: analysis, reference: reference, output: output)
        let runTask = Task {
            do {
                try await run.run { progress in
                    // Called on the export's executor; hop home to show it,
                    // and only if this run is still the one on screen.
                    Task { @MainActor in
                        guard self.generation == mine, case .running = self.state else {
                            return
                        }
                        self.state = .running(progress: progress)
                    }
                }
                guard self.generation == mine else { return }
                self.state = .finished(output)
            } catch is CancellationError {
                // `cancel()` already put `.idle` back; only a cancel nobody
                // asked this class for lands here with the generation still
                // this run's own.
                if self.generation == mine {
                    self.state = .idle
                }
            } catch {
                if self.generation == mine {
                    self.state = .failed(Self.readableMessage(error))
                }
            }
        }
        task = runTask
        await runTask.value
    }

    /// Stops the run in flight: no more progress, no finished movie — back
    /// to `.idle`. The export deletes its partial file as it unwinds.
    func cancel() {
        generation += 1
        task?.cancel()
        task = nil
        state = .idle
    }

    // MARK: - Private

    /// A failure as a sentence a person can read: the export's and the
    /// reader's own errors in their own words, everything else through
    /// `localizedDescription`.
    private static func readableMessage(_ error: Error) -> String {
        switch error {
        case let error as ExportError: return error.description
        case let error as VideoFrameSourceError: return error.description
        default: return error.localizedDescription
        }
    }
}
