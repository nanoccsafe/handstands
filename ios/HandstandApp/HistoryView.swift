import HandstandCore
import SwiftData
import SwiftUI

/// The History screen (chainlink #51): every recording on the phone, newest
/// first, headed by what has been done — how many takes, how many of them
/// this week, the total time, and the best score once analysis (#47) has
/// produced one. The list is a `@Query` over the local container
/// (`cloudKitDatabase: .none`), so a row deleted here or in the detail view
/// disappears everywhere; `.task` runs `reconcile()` first, which is what
/// brings in recordings taken before this screen existed.
@MainActor
struct HistoryView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \Session.recordedAt, order: .reverse) private var sessions: [Session]

    @State private var store: SessionStore?
    @State private var loaded = false
    @State private var loadError: String?
    @State private var pendingDelete: Session?
    @State private var deletionError: String?

    /// The header's numbers, computed from whatever the query just returned.
    private var progress: SessionProgress {
        SessionProgress.summary(
            of: sessions.map {
                SessionSnapshot(recordedAt: $0.recordedAt, durationS: $0.durationS, clipScore: $0.clipScore)
            },
            now: Date(),
            calendar: .current
        )
    }

    private var deletePresented: Binding<Bool> {
        Binding(
            get: { pendingDelete != nil },
            set: { presented in
                if !presented { pendingDelete = nil }
            }
        )
    }

    var body: some View {
        Group {
            if !loaded {
                ProgressView("Loading…")
            } else if sessions.isEmpty, let loadError {
                // Nothing to show *and* the folder could not be read: say
                // why instead of claiming there are no recordings.
                ContentUnavailableView {
                    Label("History unavailable", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(loadError)
                }
            } else if sessions.isEmpty {
                ContentUnavailableView {
                    Label("No recordings yet", systemImage: "video.badge.clock")
                } description: {
                    Text("Recordings you take are listed here, on this phone.")
                }
            } else {
                list
            }
        }
        .navigationTitle("History")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        // One confirmation for every delete on this screen: the swipe only
        // *offers* the delete, the dialog is what carries it out.
        .confirmationDialog(
            "Delete this recording? The video is removed from the phone.",
            isPresented: deletePresented,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) { deletePending() }
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

    // MARK: - The list

    private var list: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 4) {
                    Text(summaryLine)
                        .font(.subheadline.weight(.medium))
                    Text(
                        progress.bestScore.map { "Best score \(SessionFormatter.score($0))" }
                            ?? "Scores appear once analysis is available"
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    // A folder that could not be fully read (reconcile
                    // failed) must not hide the rows that *are* there — the
                    // note goes under the numbers instead.
                    if let loadError {
                        Text(loadError)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
                .padding(.vertical, 4)
            }

            Section {
                ForEach(sessions) { session in
                    if let store {
                        NavigationLink {
                            SessionDetailView(session: session, store: store)
                        } label: {
                            row(session)
                        }
                    } else {
                        row(session)
                    }
                }
                .onDelete { offsets in
                    pendingDelete = offsets.first.map { sessions[$0] }
                }
            }
        }
    }

    /// "3 sessions · 1 this week · total time 0:45".
    private var summaryLine: String {
        let numbers = progress
        let noun = numbers.total == 1 ? "session" : "sessions"
        return "\(numbers.total) \(noun) · \(numbers.thisWeek) this week"
            + " · total time \(VideoInfoFormatter.duration(numbers.totalRecordedS))"
    }

    private func row(_ session: Session) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(SessionFormatter.dateTime(session.recordedAt))
                    .font(.headline)
                Text(session.holdType.displayName)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text(VideoInfoFormatter.duration(session.durationS))
                    .font(.subheadline.monospacedDigit())
                Text(session.clipScore.map { SessionFormatter.score($0) } ?? "Not analysed yet")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .combine)
    }

    // MARK: - Loading and deleting

    /// Opens the store (once) and reconciles it with the Recordings folder
    /// every time the screen appears.
    private func load() async {
        if store == nil {
            do {
                let directory = try RecordingFile.directory()
                store = SessionStore(context: context, recordingsDirectory: directory)
            } catch {
                loadError = "The Recordings folder could not be opened: \(error.localizedDescription)"
                loaded = true
                return
            }
        }
        do {
            _ = try await store?.reconcile()
            loadError = nil
        } catch {
            // The list still shows what the database already knows; the
            // header simply may not include a file just copied in.
            loadError = "The Recordings folder could not be read: \(error.localizedDescription)"
        }
        loaded = true
    }

    private func deletePending() {
        defer { pendingDelete = nil }
        guard let session = pendingDelete, let store else { return }
        do {
            try store.delete(session)
        } catch {
            deletionError = error.localizedDescription
        }
    }
}
