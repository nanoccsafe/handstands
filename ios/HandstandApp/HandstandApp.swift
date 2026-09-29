import SwiftUI

/// The app entry point: one window, one navigation stack (`ContentView`), and
/// nothing that leaves the phone — no networking, no analytics, recordings
/// stay on the device.
@main
struct HandstandApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
