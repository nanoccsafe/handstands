import HandstandCore
import XCTest

@testable import HandstandApp

/// The border colour of the framing guide: what the Record screen paints for
/// each verdict `HandstandCore` can reach. The colour itself lives in SwiftUI
/// (`FramingTone.color`); the mapping is what matters and what is tested.
final class FramingDisplayTests: XCTestCase {
    func testWholeBodyInFrameIsGreen() {
        XCTAssertEqual(FramingDisplay.tone(for: .ok), .green)
    }

    func testWrongFramingIsAmber() {
        XCTAssertEqual(FramingDisplay.tone(for: .tooSmall), .amber)
        XCTAssertEqual(
            FramingDisplay.tone(for: .partlyOutOfFrame(edges: [.bottom], missing: [.leftAnkle])),
            .amber
        )
    }

    func testNoSingleBodyToFrameIsRed() {
        XCTAssertEqual(FramingDisplay.tone(for: .noPerson), .red)
        XCTAssertEqual(FramingDisplay.tone(for: .multiplePeople), .red)
    }

    func testEveryVerdictTheGuideCanReachHasAColour() {
        let everyStatus: [FramingStatus] = [
            .noPerson,
            .multiplePeople,
            .tooSmall,
            .partlyOutOfFrame(edges: [], missing: []),
            .ok,
        ]
        for status in everyStatus {
            let tone = FramingDisplay.tone(for: status)
            XCTAssertTrue(
                [FramingTone.green, .amber, .red].contains(tone),
                "no colour for \(status)"
            )
        }
        // Green is the only "go" colour: nothing but a whole, lone, big
        // enough body may turn it.
        let goSignals: [FramingStatus] = [
            .ok,
            .tooSmall,
            .noPerson,
            .multiplePeople,
            .partlyOutOfFrame(edges: [.top], missing: []),
        ]
        XCTAssertEqual(goSignals.filter { FramingDisplay.tone(for: $0) == .green }, [.ok])
    }
}
