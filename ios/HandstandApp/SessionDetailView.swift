import AVKit
import HandstandCore
import SwiftUI
import UIKit

/// One recording, played (chainlink #51): the movie the row points at,
/// exactly the facts the list shows beside it, the analysis of chainlink #47
/// (read the frames, run Apple Vision + the pipeline, save the result back
/// to this row), and the one destructive action in the app — deleting takes
/// the video off the phone too, after the same confirmation the list uses.
@MainActor
struct SessionDetailView: View {
    let session: Session
    /// The store the row came from: it owns the Recordings folder, so it is
    /// what `delete` and `movieURL(for:)` need.
    let store: SessionStore

    @Environment(\.dismiss) private var dismiss
    @State private var confirmingDelete = false
    @State private var deletionError: String?
    /// The player for this row's movie, made once — body would otherwise
    /// build a fresh `AVPlayer` (and restart the video) on every redraw.
    @State private var player: AVPlayer?
    /// The analysis of this row: button, progress, results. Owned here, so
    /// leaving the screen stops a run in flight (see `onDisappear`).
    @State private var service = AnalysisService()

    /// Is a run in flight — the state the screen has to stay awake for.
    private var isAnalysing: Bool {
        if case .running = service.state { return true }
        return false
    }

    var body: some View {
        List {
            Section {
                if let player {
                    VideoPlayer(player: player)
                        .frame(height: 260)
                        .listRowInsets(EdgeInsets())
                } else {
                    ProgressView()
                        .frame(height: 260)
                        .listRowInsets(EdgeInsets())
                }
            }

            Section("Recording") {
                LabeledContent("Recorded", value: SessionFormatter.dateTime(session.recordedAt))
                LabeledContent("Hold", value: session.holdType.displayName)
                LabeledContent("Duration", value: VideoInfoFormatter.duration(session.durationS))
                LabeledContent(
                    "Frame",
                    value: VideoInfoFormatter.pixelSize(width: session.width, height: session.height)
                )
                if let score = session.clipScore {
                    LabeledContent("Score", value: SessionFormatter.score(score))
                } else if session.analyzedAt == nil {
                    Text("Not analysed yet")
                        .foregroundStyle(.secondary)
                }
                // Analysed but scoreless (no reference in the build, or no
                // hold that could be scored): the Analysis section below
                // says so in words — "Not analysed yet" would be wrong.
            }

            Section("Analysis") {
                // The first end-to-end run in the app (chainlink #47):
                // frames -> Apple Vision -> Analyzer -> this row.
                AnalysisPanel(
                    service: service,
                    startLabel: session.analyzedAt == nil ? "Analyse" : "Analyse again",
                    onStart: analyse
                )
            }

            Section {
                Button("Delete recording", role: .destructive) {
                    confirmingDelete = true
                }
            }
        }
        .navigationTitle(SessionFormatter.dateTime(session.recordedAt))
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            // One player per screen, for this screen's file.
            if player == nil {
                player = AVPlayer(url: store.movieURL(for: session))
            }
        }
        .onDisappear {
            // Leaving the screen takes the sound with it, the analysis with
            // it (a run nobody is watching must not keep the CPU hot — and
            // Delete may be what sent this screen away), and the "stay
            // awake" with it.
            player?.pause()
            service.cancel()
            UIApplication.shared.isIdleTimerDisabled = false
        }
        .onChange(of: isAnalysing) { _, analysing in
            // The run takes seconds and the screen dims in ten: keep it
            // awake while it runs, and only while it runs — restored the
            // moment it finishes, fails or is cancelled.
            UIApplication.shared.isIdleTimerDisabled = analysing
        }
        .confirmationDialog(
            "Delete this recording? The video is removed from the phone.",
            isPresented: $confirmingDelete,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) { delete() }
            Button("Cancel", role: .cancel) {}
        }
        .alert(
            "Could not delete the recording",
            isPresented: Binding(
                get: { deletionError != nil },
                set: { if !$0 { deletionError = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(deletionError ?? "")
        }
    }

    private func delete() {
        // A run in flight must not write its result to a row that is about
        // to go: cancelling bumps the generation, so its save is refused
        // no matter when it wakes up.
        service.cancel()
        do {
            try store.delete(session)
            dismiss()
        } catch {
            // The row and the video are still there (delete only removes
            // the record once the files are gone), so say what went wrong.
            deletionError = error.localizedDescription
        }
    }

    /// The "Analyse" button: this row's movie, this row's hold, saved back
    /// to this row. `AnalysisService` stops anything already running, so
    /// every tap means one run.
    private func analyse() {
        Task {
            await service.analyse(
                movie: store.movieURL(for: session),
                holdType: session.holdType,
                session: session,
                store: store
            )
        }
    }
}
