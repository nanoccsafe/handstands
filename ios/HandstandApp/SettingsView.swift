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
    static let formCuesKey = "cueFormCues"

    /// What a fresh install shows: voice on, time marks off, framing hints on,
    /// form cues on (the user asked for them, 2026-10-14) — `CueSettings()`'s
    /// own defaults, spelled once for the tests.
    static let defaults = CueSettings()

    /// The one line of philosophy the issue asks for under the toggles —
    /// one string so the test can pin the exact words.
    static let philosophy =
        "Live cues only confirm; the analysis afterwards gives the critique."

    /// The toggles out of a defaults store, each falling back to its default
    /// when the key has never been written (a key written as `false` must read
    /// as `false`, so presence is checked, not truthiness).
    static func read(_ defaults: UserDefaults) -> CueSettings {
        CueSettings(
            voiceCues: bool(voiceCuesKey, fallback: Self.defaults.voiceCues, in: defaults),
            timeMarks: bool(timeMarksKey, fallback: Self.defaults.timeMarks, in: defaults),
            framingHints: bool(framingHintsKey, fallback: Self.defaults.framingHints, in: defaults),
            formCues: bool(formCuesKey, fallback: Self.defaults.formCues, in: defaults)
        )
    }

    private static func bool(_ key: String, fallback: Bool, in defaults: UserDefaults) -> Bool {
        defaults.object(forKey: key) == nil ? fallback : defaults.bool(forKey: key)
    }
}

/// The `@AppStorage` key behind the camera choice (chainlink #95), placed
/// next to the cue keys and read the same way: one spelling shared by the
/// flip button, `CaptureService` and the tests.
///
/// The stored value is `CameraPosition`'s raw value (`"back"`/`"front"`); a
/// fresh install — and any value this build does not know — means the back
/// camera.
enum CameraSettingsStorage {
    static let positionKey = "cameraPosition"

    /// What a fresh install records with, and what an unknown stored value
    /// falls back to.
    static let defaultPosition: CameraPosition = .default

    /// The stored choice out of a defaults store.
    static func read(_ defaults: UserDefaults) -> CameraPosition {
        resolved(defaults.string(forKey: positionKey))
    }

    /// The choice a raw stored string means: a known value stays itself,
    /// anything else (including `nil`) is the default.
    static func resolved(_ raw: String?) -> CameraPosition {
        raw.flatMap(CameraPosition.init(rawValue:)) ?? defaultPosition
    }
}

/// The settings screen: the live-cue toggles and the philosophy line under
/// them (chainlink #91, #97).
struct SettingsView: View {
    @AppStorage(CueSettingsStorage.voiceCuesKey)
    private var voiceCues = CueSettingsStorage.defaults.voiceCues
    @AppStorage(CueSettingsStorage.timeMarksKey)
    private var timeMarks = CueSettingsStorage.defaults.timeMarks
    @AppStorage(CueSettingsStorage.framingHintsKey)
    private var framingHints = CueSettingsStorage.defaults.framingHints
    @AppStorage(CueSettingsStorage.formCuesKey)
    private var formCues = CueSettingsStorage.defaults.formCues

    var body: some View {
        Form {
            Section {
                Toggle("Voice cues", isOn: $voiceCues)
                Toggle("Time marks", isOn: $timeMarks)
                Toggle("Framing hints", isOn: $framingHints)
                Toggle("Form cues (say when the line breaks)", isOn: $formCues)
            } footer: {
                Text(CueSettingsStorage.philosophy)
            }
        }
        .navigationTitle("Settings")
    }
}
