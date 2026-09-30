import Foundation

/// The fault names as they are said out loud: the scorer speaks feature
/// keys (`"hip_angle"`, `"com_sway_sd"`) because those are the pipeline's
/// column names, but the screen says words — "Hip angle". Kept out of
/// SwiftUI, next to the other formatters, so the tests can pin the strings
/// down without a view in the way (chainlink #47).
enum FaultLabel {
    /// `"hip_angle"` → `"Hip angle"`, `"com_sway_sd"` → `"Com sway sd"`:
    /// the parts joined with spaces, the *first* letter capitalised and the
    /// rest left as the scorer spells it — only the first word is a name.
    /// An empty name stays empty rather than becoming a stray capital.
    static func text(for feature: String) -> String {
        let words = feature.split(separator: "_").joined(separator: " ")
        guard let head = words.first else { return "" }
        return String(head).uppercased() + String(words.dropFirst())
    }
}
