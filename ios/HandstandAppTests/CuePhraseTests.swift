import HandstandCore
import XCTest

@testable import HandstandApp

/// The one table of words (chainlink #91): what the voice says and what the
/// banner shows for every cue the scheduler can produce — the table
/// `docs/ios.md` quotes and the user rewords.
final class CuePhraseTests: XCTestCase {
    /// Every cue, spelled out: the table cannot miss a case, because the
    /// switch behind `phrase(for:)` cannot either — and if a case is added
    /// without words, this list is what fails.
    private let everyCue: [Cue] = [
        .lineHold,
        .timeMark(5),
        .timeMark(10),
        .framing(.stepBack),
        .framing(.comeCloser),
        .framing(.moveToMiddle),
        .framing(.onlyYou),
        .framing(.raisePhone),
        .ready,
    ]

    func testThePhraseTable() {
        XCTAssertEqual(CueText.phrase(for: .lineHold), "Line. Hold it.")
        XCTAssertEqual(CueText.phrase(for: .timeMark(5)), "Five")
        XCTAssertEqual(CueText.phrase(for: .timeMark(10)), "Ten")
        XCTAssertEqual(CueText.phrase(for: .framing(.stepBack)), "Step back.")
        XCTAssertEqual(CueText.phrase(for: .framing(.comeCloser)), "Come closer.")
        XCTAssertEqual(CueText.phrase(for: .framing(.moveToMiddle)), "Move to the middle.")
        XCTAssertEqual(CueText.phrase(for: .framing(.onlyYou)), "Only you in the frame.")
        XCTAssertEqual(CueText.phrase(for: .framing(.raisePhone)), "Raise the phone.")
        XCTAssertEqual(CueText.phrase(for: .ready), "Ready.")
    }

    func testEveryCueHasNonEmptyWords() {
        for cue in everyCue {
            XCTAssertFalse(CueText.phrase(for: cue).isEmpty, "\(cue)")
            XCTAssertFalse(CueText.identifier(for: cue).isEmpty, "\(cue)")
        }
    }

    func testTimeMarksReadAsWords() {
        XCTAssertEqual(CueText.phrase(for: .timeMark(1)), "One")
        XCTAssertEqual(CueText.phrase(for: .timeMark(15)), "Fifteen")
        XCTAssertEqual(CueText.phrase(for: .timeMark(20)), "Twenty")
        XCTAssertEqual(CueText.phrase(for: .timeMark(21)), "Twenty-one")
        XCTAssertEqual(CueText.phrase(for: .timeMark(60)), "Sixty")
        XCTAssertEqual(CueText.phrase(for: .timeMark(101)), "101", "out of range reads as digits")
    }

    func testIdentifiersAreStableSnakeCase() {
        XCTAssertEqual(CueText.identifier(for: .lineHold), "line_hold")
        XCTAssertEqual(CueText.identifier(for: .timeMark(5)), "time_mark_5")
        XCTAssertEqual(CueText.identifier(for: .framing(.stepBack)), "step_back")
        XCTAssertEqual(CueText.identifier(for: .framing(.comeCloser)), "come_closer")
        XCTAssertEqual(CueText.identifier(for: .framing(.moveToMiddle)), "move_to_middle")
        XCTAssertEqual(CueText.identifier(for: .framing(.onlyYou)), "only_you")
        XCTAssertEqual(CueText.identifier(for: .framing(.raisePhone)), "raise_phone")
        XCTAssertEqual(CueText.identifier(for: .ready), "ready")

        // Every identifier is one greppable token — the sidecar is read with
        // `jq` as often as with this app.
        for cue in everyCue {
            let id = CueText.identifier(for: cue)
            XCTAssertFalse(id.contains(" "), "\(id)")
            XCTAssertFalse(id.contains("."), "\(id)")
        }
    }
}
