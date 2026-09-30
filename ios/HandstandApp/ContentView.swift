import SwiftUI

/// The home screen: the app title, the one-line summary of what it is for,
/// and the things it can do. Capture (chainlink #46) leads to the live
/// camera screen, History (#51) to what has been recorded, and analysis
/// (#47) still leads to a placeholder.
struct ContentView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Text("Record a handstand or pick a video to analyse.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                NavigationLink {
                    RecordView()
                } label: {
                    Label("Record", systemImage: "video.fill")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .frame(maxWidth: .infinity)

                // The one assumption the whole pipeline makes: a single
                // athlete, framed by a stationary phone.
                Text("Only you in the frame, phone on a tripod.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                NavigationLink {
                    AnalyseVideoView()
                } label: {
                    Label("Analyse a video", systemImage: "film")
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .frame(maxWidth: .infinity)

                // Everything recorded, newest first (chainlink #51) —
                // local SwiftData, no network behind it.
                NavigationLink {
                    HistoryView()
                } label: {
                    Label("History", systemImage: "clock.arrow.circlepath")
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .frame(maxWidth: .infinity)

                Spacer()
            }
            .padding(.horizontal, 24)
            .navigationTitle("Handstand")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink("About") {
                        AboutView()
                    }
                }
            }
        }
    }
}
