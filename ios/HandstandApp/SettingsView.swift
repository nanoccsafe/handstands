import SwiftUI
import HandstandCore

// --------------------------------------------------------------------------- #
// Where the live cues are turned on and off (chainlink #91) — three toggles
// behind a gear on the home screen, stored with `@AppStorage`, and the one line
// of philosophy the issue asks for under them.
//
// The keys and the defaults live in `CueSettingsStorage` rather than in the
// view, so `RecordView` (which reads them every tick), the settings screen and
// the tests all agree on one spelling — and so a test can round-trip a
// `UserDefaults` without a view in the way.
// --------------------------------------------------------------------------- #

/// The `@AppStorage` keys behind ``CueSettings``, and how to read them as a
/// `CueSettings` — the single source of the defaults a fresh install shows.
enum CueSettingsStorage {
    static let voiceCuesKey = "cueVoiceCues"
    static let timeMarksKey = "cueTimeMarks"
    static let framingHintsKey = "cueFramingHints"

    /// What a fresh install shows: voice on, time marks off, framing hints on
    /// — `CueSettings()`'s own defaults, spelled once for the tests.
    static let defaults = CueSettings()

    /// The one line of philosophy the issue asks for under the toggles —
    /// one string so the test can pin the exact words.
    static let philosophy =
        "Live cues only confirm; the analysis afterwards gives the critique."

    /// The three toggles out of a defaults store, each falling back to its
    /// default when the key has never been written (a key written as `false`
    /// must read as `false`, so presence is checked, not truthiness).
    static func read(_ defaults: UserDefaults) -> CueSettings {
        CueSettings(
            voiceCues: bool(voiceCuesKey, fallback: Self.defaults.voiceCues, in: defaults),
            timeMarks: bool(timeMarksKey, fallback: Self.defaults.timeMarks, in: defaults),
            framingHints: bool(framingHintsKey, fallback: Self.defaults.framingHints, in: defaults)
        )
    }

    private static func bool(_ key: String, fallback: Bool, in defaults: UserDefaults) -> Bool {
        defaults.object(forKey: key) == nil ? fallback : defaults.bool(forKey: key)
    }
}

/// The settings screen: the three live-cue toggles and the philosophy line
/// under them (chainlink #91).
struct SettingsView: View {
    @AppStorage(CueSettingsStorage.voiceCuesKey)
    private var voiceCues = CueSettingsStorage.defaults.voiceCues
    @AppStorage(CueSettingsStorage.timeMarksKey)
    private var timeMarks = CueSettingsStorage.defaults.timeMarks
    @AppStorage(CueSettingsStorage.framingHintsKey)
    private var framingHints = CueSettingsStorage.defaults.framingHints

    var body: some View {
        Form {
            Section {
                Toggle("Voice cues", isOn: $voiceCues)
                Toggle("Time marks", isOn: $timeMarks)
                Toggle("Framing hints", isOn: $framingHints)
            } footer: {
                Text(CueSettingsStorage.philosophy)
            }
        }
        .navigationTitle("Settings")
    }
}
