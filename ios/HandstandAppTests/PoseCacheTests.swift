import XCTest

@testable import HandstandApp
import HandstandCore

/// The pose cache (chainlink #48): frames written beside the movie so
/// reopening a session never runs Vision again, and read back only when
/// they were written by *this* pipeline. No real video: the movie is an
/// empty file with the right name, because nothing here ever plays it.
@MainActor
final class PoseCacheTests: XCTestCase {
    /// A temp folder with an empty `.mov` in it; the caller removes the
    /// folder when it is done.
    private func makeMovie() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PoseCacheTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let movie = directory.appendingPathComponent("20260928-143059.mov")
        try Data().write(to: movie)
        return movie
    }

    /// Two frames: one with a nose and a wrist, one nobody was found in —
    /// the two shapes a real run writes.
    private func frames() -> [PostProcessInputFrame] {
        [
            PostProcessInputFrame(
                tMs: 0,
                detected: true,
                trainerContact: false,
                joints: [
                    .nose: Keypoint(x: 100.5, y: 200.25, visibility: 0.9),
                    .leftWrist: Keypoint(x: 80, y: 480, visibility: 1),
                ]
            ),
            PostProcessInputFrame(tMs: 33, detected: false, trainerContact: false, joints: [:]),
        ]
    }

    private func remove(_ movie: URL) {
        try? FileManager.default.removeItem(at: movie.deletingLastPathComponent())
    }

    /// The whole round trip: what went in is what comes out.
    func testARoundTripKeepsEveryFrame() throws {
        let movie = try makeMovie()
        defer { remove(movie) }

        try PoseCache.write(frames: frames(), for: movie)
        let read = try XCTUnwrap(PoseCache.read(for: movie))

        XCTAssertEqual(read, frames())
        XCTAssertEqual(read[0].tMs, 0)
        XCTAssertEqual(read[0].detected, true)
        XCTAssertEqual(read[0].joints[.nose], Keypoint(x: 100.5, y: 200.25, visibility: 0.9))
        XCTAssertNil(read[0].joints[.rightWrist], "a joint that was not seen has no entry")
        XCTAssertEqual(read[1].tMs, 33)
        XCTAssertFalse(read[1].detected)
        XCTAssertTrue(read[1].joints.isEmpty)
        // The device never reports a trainer: the schema has no column for
        // it, and the read side says so explicitly.
        XCTAssertFalse(read[0].trainerContact)
    }

    /// The file is the document chainlink #48 spells — same keys, same
    /// values — sitting beside the movie under `<basename>.pose.json`.
    func testTheFileIsTheDocumentTheTaskSpecifies() throws {
        let movie = try makeMovie()
        defer { remove(movie) }

        try PoseCache.write(frames: frames(), for: movie)

        let url = PoseCache.cacheURL(for: movie)
        XCTAssertEqual(
            url.lastPathComponent, "20260928-143059.pose.json",
            "same folder, same stem, .pose.json")
        XCTAssertEqual(url.deletingLastPathComponent().path, movie.deletingLastPathComponent().path)

        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data)
        let object = try XCTUnwrap(json as? [String: Any])
        XCTAssertEqual(object["schema"] as? Int, 1)
        XCTAssertEqual(object["backend"] as? String, "vision")
        XCTAssertEqual(object["analysis_version"] as? String, AnalysisService.analysisVersion)
        XCTAssertEqual(object["max_fps"] as? Int, 30)

        let written = try XCTUnwrap(object["frames"] as? [[String: Any]])
        XCTAssertEqual(written.count, 2)
        XCTAssertEqual(written[0]["t_ms"] as? Int, 0)
        XCTAssertEqual(written[0]["detected"] as? Bool, true)
        let joints = try XCTUnwrap(written[0]["joints"] as? [String: [Double]])
        XCTAssertEqual(joints["nose"] ?? [], [100.5, 200.25, 0.9])
        XCTAssertEqual(joints["left_wrist"] ?? [], [80, 480, 1])
        XCTAssertNil(joints["right_wrist"])
        XCTAssertEqual(written[1]["t_ms"] as? Int, 33)
        XCTAssertEqual(written[1]["detected"] as? Bool, false)
        let emptyJoints = try XCTUnwrap(written[1]["joints"] as? [String: [Double]])
        XCTAssertTrue(emptyJoints.isEmpty)
    }

    /// No file: not an error, just "no cache" — the caller runs the
    /// extraction instead.
    func testAMissingFileReadsAsNoCache() throws {
        let movie = try makeMovie()
        defer { remove(movie) }

        XCTAssertNil(PoseCache.read(for: movie))
    }

    func testBadJSONReadsAsNoCache() throws {
        let movie = try makeMovie()
        defer { remove(movie) }
        try Data("{ not json at all".utf8).write(to: PoseCache.cacheURL(for: movie))

        XCTAssertNil(PoseCache.read(for: movie))
    }

    /// A document that parses but was written by another shape of pipeline
    /// (schema) or another run of it (analysis version) is a different
    /// question, not an answer: `nil`, and the run happens again.
    func testASchemaOrVersionMismatchReadsAsNoCache() throws {
        let movie = try makeMovie()
        defer { remove(movie) }

        let documents = [
            #"{"schema": 2, "backend": "vision", "analysis_version": "vision-1", "max_fps": 30, "frames": []}"#,
            #"{"schema": 1, "backend": "vision", "analysis_version": "vision-0", "max_fps": 30, "frames": []}"#,
        ]
        for document in documents {
            try Data(document.utf8).write(to: PoseCache.cacheURL(for: movie))
            XCTAssertNil(PoseCache.read(for: movie), document)
        }
    }

    /// Writing twice overwrites rather than appending — "Analyse again"
    /// replaces the cache instead of stacking a second clip under it.
    func testWritingTwiceLeavesOneDocument() throws {
        let movie = try makeMovie()
        defer { remove(movie) }

        try PoseCache.write(frames: frames(), for: movie)
        try PoseCache.write(frames: [frames()[1]], for: movie)

        let read = try XCTUnwrap(PoseCache.read(for: movie))
        XCTAssertEqual(read, [frames()[1]])
    }
}
