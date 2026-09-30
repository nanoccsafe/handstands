import SwiftData
import SwiftUI

/// The app entry point: one window, one navigation stack (`ContentView`), and
/// nothing that leaves the phone — no networking, no analytics, recordings
/// stay on the device.
@main
struct HandstandApp: App {
    /// The local History store (chainlink #51): just `Session`, in the app's
    /// own SwiftData store. `cloudKitDatabase: .none` is the promise that
    /// nothing in it is ever mirrored to iCloud or anywhere else — the rows
    /// only point at movies in `Application Support/Recordings`.
    ///
    /// Built here rather than through the scene modifier's `for:inMemory:`
    /// convenience so the CloudKit-off configuration is explicit; a store
    /// that cannot be opened means no History, so the app does not start
    /// without it.
    let container: ModelContainer = {
        do {
            return try ModelContainer(
                for: Session.self,
                configurations: ModelConfiguration(cloudKitDatabase: .none)
            )
        } catch {
            fatalError("The History store could not be opened: \(error.localizedDescription)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(container)
    }
}
