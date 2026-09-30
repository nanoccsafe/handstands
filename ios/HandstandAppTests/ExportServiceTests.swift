import XCTest

@testable import HandstandApp
import HandstandCore

/// `ExportService` (chainlink #50): the state machine around the export —
/// idle at rest, `.failed` with a readable sentence when the video cannot
/// be read, `.idle` again after a cancel, and never a file left behind.
///
/// The happy path needs a real video and a writer; that is what
/// `AnnotatedVideoTests` covers on the Mac. What the screen reads off this
/// class is pinned here, without a video, a view or a model of a person.
@MainActor
final class ExportServiceTests: XCTestCase {
    /// An analysis of a clip with no person in it — the export refuses the
    /// *movie* here, not the analysis, so this is all the run needs.
    private func analysis() -> Analysis {
        let frames = (0..<5).map { index in
            PostProcessInputFrame(
                tMs: index * 33, detected: false, trainerContact: false, joints: [:])
        }
        return Analyzer.analyze(frames, reference: nil)
    }

    private func missingMovie() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("no-such-clip-\(UUID().uuidString).mov")
    }

    private func outputURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("no-such-export-\(UUID().uuidString).mp4")
    }

    func testItStartsIdle() {
        XCTAssertEqual(ExportService().state, .idle)
    }

    func testExportingAFileThatIsNotThereFailsWithAReadableMessage() async {
        let service = ExportService()
        let output = outputURL()

        await service.export(
            movie: missingMovie(), analysis: analysis(), reference: nil, to: output)

        guard case .failed(let message) = service.state else {
            return XCTFail("expected .failed, got \(service.state)")
        }
        XCTAssertFalse(
            message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "the failure says something")
        XCTAssertFalse(message.contains("VideoFrameSourceError"), "not a type name")
        XCTAssertFalse(message.contains("CocoaError"), "not a type name")
        XCTAssertFalse(message.contains("Error)"), "not a raw debug description")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: output.path),
            "a failed export writes no file")
    }

    func testCancelFromIdleStaysIdle() {
        let service = ExportService()
        service.cancel()
        XCTAssertEqual(service.state, .idle)
        XCTAssertNil(service.finishedURL, "an idle export holds no file")
    }
}
