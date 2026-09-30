import Foundation
import HandstandCore

/// The words the hold picker shows for a hold: just its display name when the
/// MVP can record it ("Line"), the name plus "— coming soon" for the ones it
/// only lists ("Tuck — coming soon"), so the user sees what is planned.
/// Kept out of SwiftUI, next to `CaptureFormatter`, so the tests can pin the
/// strings down without a view in the way (chainlink #66).
enum HoldTypeLabel {
    static func text(for hold: HoldType) -> String {
        hold.isAvailable ? hold.displayName : "\(hold.displayName) — coming soon"
    }
}
