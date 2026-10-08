import HandstandCore
import XCTest

@testable import HandstandApp

/// The gear on the home screen (chainlink #91): what a fresh install shows,
/// that the three toggles round-trip through `UserDefaults`, and that the
/// three keys are three keys — the settings `SettingsView` renders and
/// `RecordView` reads are these, and nothing else.
final class SettingsDefaultsTests: XCTestCase {
    private let suiteName = "SettingsDefaultsTests"
    private var defaults: UserDefaults!

    override func setUpWithError() throws {
        defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
    }

    override func tearDownWithError() throws {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
    }

    func testAFreshInstallShowsVoiceOnTimeMarksOffAndHintsOn() {
        XCTAssertEqual(
            CueSettingsStorage.read(defaults),
            CueSettings(voiceCues: true, timeMarks: false, framingHints: true)
        )
        // The same defaults the core's own `CueSettings()` starts with — one
        // definition, not two that can drift.
        XCTAssertEqual(CueSettingsStorage.defaults, CueSettings())
        XCTAssertEqual(CueSettings(), CueSettings(voiceCues: true, timeMarks: false, framingHints: true))
    }

    func testTheThreeTogglesRoundTrip() {
        let everythingOn = CueSettings(voiceCues: true, timeMarks: true, framingHints: true)
        defaults.set(everythingOn.voiceCues, forKey: CueSettingsStorage.voiceCuesKey)
        defaults.set(everythingOn.timeMarks, forKey: CueSettingsStorage.timeMarksKey)
        defaults.set(everythingOn.framingHints, forKey: CueSettingsStorage.framingHintsKey)
        XCTAssertEqual(CueSettingsStorage.read(defaults), everythingOn)

        let everythingOff = CueSettings(voiceCues: false, timeMarks: false, framingHints: false)
        defaults.set(everythingOff.voiceCues, forKey: CueSettingsStorage.voiceCuesKey)
        defaults.set(everythingOff.timeMarks, forKey: CueSettingsStorage.timeMarksKey)
        defaults.set(everythingOff.framingHints, forKey: CueSettingsStorage.framingHintsKey)
        XCTAssertEqual(CueSettingsStorage.read(defaults), everythingOff)
    }

    func testAStoredFalseIsFalseNotTheDefault() {
        // `bool(forKey:)` reads a missing key as false too, which would make
        // voice cues' default indistinguishable from "switched off" — the
        // read checks presence instead, which is what this pins down.
        defaults.set(false, forKey: CueSettingsStorage.voiceCuesKey)
        XCTAssertEqual(CueSettingsStorage.read(defaults).voiceCues, false)
        XCTAssertEqual(CueSettingsStorage.read(defaults).timeMarks, false, "never written: the default")
        XCTAssertEqual(CueSettingsStorage.read(defaults).framingHints, true, "never written: the default")
    }

    func testTheThreeKeysAreThreeKeys() {
        let keys = [
            CueSettingsStorage.voiceCuesKey,
            CueSettingsStorage.timeMarksKey,
            CueSettingsStorage.framingHintsKey,
        ]
        XCTAssertEqual(Set(keys).count, 3, "three distinct @AppStorage keys")
    }

    func testThePhilosophyLineUnderTheToggles() {
        XCTAssertEqual(
            CueSettingsStorage.philosophy,
            "Live cues only confirm; the analysis afterwards gives the critique."
        )
    }

    /// The screen itself, built from the same keys and defaults: evaluating
    /// its body is what forces the three `@AppStorage` reads the toggles are
    /// rendered from.
    @MainActor
    func testTheScreenBuildsFromTheSameStorage() {
        let screen = SettingsView()
        _ = screen.body
    }
}
