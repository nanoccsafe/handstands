import Foundation
import SwiftUI

/// The analysis controls and results, shared by the session detail screen
/// and the picked-video screen (chainlink #47): the button that starts a
/// run, the progress with its Cancel while it runs, the summary when it is
/// done, and a readable reason when it failed.
///
/// The panel owns no policy — the screen decides which movie, hold and row
/// a run uses (`onStart`) and what the button says (`startLabel`, "Analyse"
/// or "Analyse again"). It reads `service.state` and renders it, so both
/// screens show exactly the same run the same way.
struct AnalysisPanel: View {
    /// The service the screen owns; its `state` is what this shows.
    let service: AnalysisService
    /// What the start button says.
    let startLabel: String
    /// Starts a run — called on the main actor, from the button.
    let onStart: () -> Void

    var body: some View {
        switch service.state {
        case .idle:
            startButton
        case .running(let progress):
            running(progress)
        case .finished(let summary):
            results(summary)
            // Done is not the end: the run can be repeated from here (the
            // button the task's checklist taps — "Analyse again" once the
            // row has been analysed), without leaving and re-opening the
            // screen.
            startButton
        case .failed(let message):
            Text(message)
                .font(.footnote)
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
            Button("Try again", action: onStart)
                .buttonStyle(.bordered)
        }
    }

    // MARK: - The states

    /// The button that starts a run — the same control in `.idle` and under
    /// a finished run's results, so the label the screen chose ("Analyse" /
    /// "Analyse again") reads the same wherever it appears.
    private var startButton: some View {
        Button(startLabel, action: onStart)
            .buttonStyle(.borderedProminent)
    }

    /// Working: how far (a percentage a person can read off a glance) and a
    /// way out. The screen stays awake while this is on (`isIdleTimerDisabled`
    /// is the screens' job, not this one's).
    private func running(_ progress: Double) -> some View {
        VStack(spacing: 10) {
            ProgressView(value: progress) {
                Text("Analysing… \(Int((progress * 100).rounded()))%")
                    .font(.subheadline.monospacedDigit())
            }
            Button("Cancel", role: .destructive) {
                service.cancel()
            }
        }
    }

    /// Done: the facts of the run — holds, the longest one, the score (or
    /// why there is none) and the worst faults, in words. A clip that could
    /// not be measured says so instead: "Holds: 0" would read as "you did
    /// not hold", which is not what an unmeasurable clip means.
    ///
    /// A take that *was* measurable but found nothing says **No hold
    /// found** and why, in plain words plus one setup tip (chainlink #93) —
    /// never just a zero.
    private func results(_ summary: AnalysisSummary) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if summary.unusableReason != nil {
                unmeasurable(summary)
            } else {
                LabeledContent("Holds", value: "\(summary.holdCount)")
                LabeledContent(
                    "Longest hold",
                    value: String(format: "%.1f s", summary.longestHoldS)
                )
                if summary.holdCount == 0, let explanation = summary.noHoldExplanation {
                    // The dominant reason the analysis could not see into
                    // (out of frame, hands not visible, never upside down …)
                    // in the words of `NoHoldExplanation`, tip included.
                    Text("No hold found")
                        .font(.subheadline.weight(.semibold))
                    Text(explanation)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if let score = summary.clipScore {
                    LabeledContent("Score", value: SessionFormatter.score(score))
                } else {
                    // Without a reference there is nothing to score against —
                    // say so, rather than showing a dash with no explanation.
                    Text(
                        summary.hasReference
                            ? "No score: this take could not be scored"
                            : "No score yet (no reference)"
                    )
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                }
                if !summary.topFaults.isEmpty {
                    Text(
                        "Top faults: "
                            + summary.topFaults.map(FaultLabel.text(for:)).joined(separator: ", ")
                    )
                    .font(.subheadline)
                }
            }
        }
    }

    /// What the screen says when the app could not measure the clip: why,
    /// how to fix the framing, and how many frames had a person in them —
    /// never a hold count, which would read as "you did not hold".
    private func unmeasurable(_ summary: AnalysisSummary) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Couldn't measure your body in this video.")
                .font(.subheadline)
            Text("Make sure your whole body is in frame and well lit, side-on to the camera.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text(
                summary.detectedFrames == 0
                    ? "No person found in this video."
                    : "Person found in \(summary.detectedFrames) of \(summary.frames) frames"
            )
            .font(.footnote)
            .foregroundStyle(.secondary)
        }
    }
}
