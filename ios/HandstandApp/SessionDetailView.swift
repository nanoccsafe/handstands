import AVKit
import HandstandCore
import SwiftUI

/// One recording, played (chainlink #51): the movie the row points at,
/// exactly the facts the list shows beside it, and the one destructive
/// action in the app — deleting takes the video off the phone too, after
/// the same confirmation the list uses.
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
                } else {
                    Text("Not analysed yet")
                        .foregroundStyle(.secondary)
                }
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
            // Leaving the screen takes the sound with it.
            player?.pause()
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
        do {
            try store.delete(session)
            dismiss()
        } catch {
            // The row and the video are still there (delete only removes
            // the record once the files are gone), so say what went wrong.
            deletionError = error.localizedDescription
        }
    }
}
