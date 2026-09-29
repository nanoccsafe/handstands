import SwiftUI

/// Where "Record" leads until capture lands (chainlink #46). The button and
/// the navigation are real; only the camera screen behind them is missing,
/// so the placeholder says so instead of pretending.
struct RecordPlaceholderView: View {
    var body: some View {
        ContentUnavailableView {
            Label("Record", systemImage: "video")
        } description: {
            Text("Recording comes in a later version of the app.")
        }
        .navigationTitle("Record")
    }
}
