import XCTest

@testable import HandstandApp
import HandstandCore

/// `AnalysisService` (chainlink #47): the state machine around the run —
/// idle at rest, `.failed` with a readable sentence when the video cannot
/// be read, `.idle` again after a cancel.
///
/// The happy path needs a real video and Vision (the phone checklist in
/// docs/ios.md covers it); what can be pinned down without one is exactly
/// this: it starts idle, it never traps, and its failure message is a
/// person's sentence rather than a type name.
@MainActor
final class AnalysisServiceTests: XCTestCase {
    func testItStartsIdle() {
        XCTAssertEqual(AnalysisService().state, .idle)
    }

    func testAnalysingAFileThatIsNotThereFailsWithAReadableMessage() async {
        let service = AnalysisService()
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("no-such-clip-\(UUID().uuidString).mov")

        await service.analyse(movie: missing, holdType: .line, session: nil, store: nil)

        guard case .failed(let message) = service.state else {
            return XCTFail("expected .failed, got \(service.state)")
        }
        XCTAssertFalse(
            message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "the failure says something")
        XCTAssertFalse(message.contains("VideoFrameSourceError"), "not a type name")
        XCTAssertFalse(message.contains("Error)"), "not a raw debug description")
    }

    func testCancelFromIdleStaysIdle() {
        let service = AnalysisService()
        service.cancel()
        XCTAssertEqual(service.state, .idle)
    }

    /// A second run supersedes the first: `analyse` stops what was going
    /// (the service cancels it at the door) and starts over, so the state
    /// is always some run's state, never a mix.
    func testASecondRunStartsFromTheTop() async {
        let service = AnalysisService()
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("no-such-clip-\(UUID().uuidString).mov")

        await service.analyse(movie: missing, holdType: .line, session: nil, store: nil)
        let first = service.state
        await service.analyse(movie: missing, holdType: .line, session: nil, store: nil)

        guard case .failed = first else {
            return XCTFail("the first run reports its failure, got \(first)")
        }
        guard case .failed = service.state else {
            return XCTFail("the second run reports its own, got \(service.state)")
        }
    }
}
