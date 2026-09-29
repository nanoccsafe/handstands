import SwiftUI

/// The screen when camera access was refused: a sentence saying what the
/// camera is for and a button into Settings — the app cannot grant itself
/// the permission, only iOS can.
struct CameraAccessNeededView: View {
    let openSettings: () -> Void

    var body: some View {
        ContentUnavailableView {
            Label("Camera access needed", systemImage: "camera.fill")
        } description: {
            Text(
                "The app uses the camera to record your attempt and to check "
                + "that your whole body is in frame. Allow camera access in "
                + "Settings to start."
            )
        } actions: {
            Button("Open Settings", action: openSettings)
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
        }
    }
}
