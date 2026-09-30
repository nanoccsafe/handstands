import Foundation
import HandstandCore

// --------------------------------------------------------------------------- #
// The scoring reference (chainlink #28's output), loaded from the bundle.
//
// The real reference is built from the user's own labelled data, so it is
// **never** committed: the user drops their `reference-line.json` into
// `ios/LocalResources/` (git-ignored, an optional resource folder in
// `project.yml`) before building, and the app finds it in the bundle. A
// build without one — the normal state of this public repo — simply has no
// reference: the analysis still finds phases, holds and features, and the
// score reads "No score yet (no reference)".
// --------------------------------------------------------------------------- #

/// Finding and decoding the scoring reference for a hold. Never throws and
/// never crashes: no file is "no reference", and so is an invalid one (the
/// reason is logged once, so a broken file is findable without a debugger).
enum ReferenceLoader {
    /// The reference for `hold` in the app bundle, or `nil` when there is
    /// none (or it does not decode).
    ///
    /// Looks for `<hold.referenceResourceName>.json` — `reference-line.json`
    /// for Line — at the bundle's root and, when the resource folder was
    /// copied as a folder, inside `LocalResources/`: which of the two the
    /// build produces depends on how XcodeGen saw the optional folder, and
    /// the answer must not.
    static func load(for hold: HoldType, bundle: Bundle = .main) -> ScoreReference? {
        let name = hold.referenceResourceName
        guard
            let url = bundle.url(forResource: name, withExtension: "json")
                ?? bundle.url(forResource: name, withExtension: "json", subdirectory: "LocalResources"),
            let data = try? Data(contentsOf: url)
        else {
            return nil
        }
        return load(data: data)
    }

    /// Reference bytes → a validated reference, or `nil` (with the reason
    /// logged once) when the bytes are not a usable reference.
    static func load(data: Data) -> ScoreReference? {
        do {
            return try ScoreReference.decode(data)
        } catch {
            logOnce(error)
            return nil
        }
    }

    // MARK: - Logging

    /// The lock over `logged`: one message is logged for one reason, not
    /// once per screen that happens to ask. `nonisolated(unsafe)` is the
    /// price of a process-wide set under Swift 6 strict concurrency — the
    /// lock is what makes it safe.
    private static let lock = NSLock()
    private nonisolated(unsafe) static var logged: Set<String> = []

    private static func logOnce(_ error: Error) {
        let text = "\(error)"
        lock.lock()
        let firstTime = logged.insert(text).inserted
        lock.unlock()
        guard firstTime else { return }
        NSLog("Handstand: no usable scoring reference: %@", text)
    }
}
