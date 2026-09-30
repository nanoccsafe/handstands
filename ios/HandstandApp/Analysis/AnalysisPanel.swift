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
            Button(startLabel, action: onStart)
                .buttonStyle(.borderedProminent)
        case .running(let progress):
            running(progress)
        case .finished(let summary):
            results(summary)
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
    /// why there is none) and the worst faults, in words.
    private func results(_ summary: AnalysisSummary) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent("Holds", value: "\(summary.holdCount)")
            LabeledContent(
                "Longest hold",
                value: String(format: "%.1f s", summary.longestHoldS)
            )
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
