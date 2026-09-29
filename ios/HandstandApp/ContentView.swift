import SwiftUI

/// The home screen: the app title, the one-line summary of what it is for,
/// and the two things it can do. Capture (chainlink #46) and analysis
/// (chainlink #47) replace the placeholders this screen leads to.
struct ContentView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Text("Record a handstand or pick a video to analyse.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                NavigationLink {
                    RecordPlaceholderView()
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
