import XCTest

@testable import HandstandApp
import HandstandCore

/// The JSON sidecar saved next to every recording (chainlink #66): same
/// stem as the movie, snake_case keys, an ISO-8601 UTC date and the hold's
/// raw value — and reads that never crash, whatever an old or edited file
/// says. No camera needed: a zero-byte `.mov` stands in for the movie.
final class RecordingMetadataTests: XCTestCase {
    private var directory: URL!
    private var movie: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("RecordingMetadataTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        movie = directory.appendingPathComponent("20260928-143059.mov")
        try Data().write(to: movie)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    /// A whole second in UTC — `iso8601` writes no fractional digits, so a
    /// round trip of this date is exact.
    private var afternoon: Date {
        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 28
        components.hour = 14
        components.minute = 30
        components.second = 59
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar.date(from: components)!
    }

    private func sample() -> RecordingMetadata {
        RecordingMetadata(holdType: .line, recordedAt: afternoon, appVersion: "0.1.0")
    }

    /// The sidecar on disk, parsed the way any reader would.
    private func sidecarJSON() throws -> [String: Any] {
        let data = try Data(contentsOf: RecordingMetadata.sidecarURL(for: movie))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    // MARK: - The file

    func testSidecarURLSwapsMovForJsonInTheSameFolder() {
        let sidecar = RecordingMetadata.sidecarURL(for: movie)
        XCTAssertEqual(sidecar.lastPathComponent, "20260928-143059.json")
        XCTAssertEqual(sidecar.deletingPathExtension().lastPathComponent, "20260928-143059")
        XCTAssertEqual(sidecar.deletingLastPathComponent().path, movie.deletingLastPathComponent().path)
    }

    func testWriteThenReadRoundTrips() throws {
        try sample().write(for: movie)
        let read = try RecordingMetadata.read(for: movie)
        XCTAssertEqual(read, sample())
        XCTAssertEqual(read?.holdType, .line)
        XCTAssertEqual(read?.appVersion, "0.1.0")
        XCTAssertTrue(FileManager.default.fileExists(atPath: movie.path), "the video is untouched")
    }

    func testReadWithoutASidecarIsNil() throws {
        XCTAssertNil(try RecordingMetadata.read(for: movie))
    }

    // MARK: - The document

    func testJSONUsesSnakeCaseKeysAndTheLineRawValue() throws {
        try sample().write(for: movie)
        let object = try sidecarJSON()
        XCTAssertEqual(
            Set(object.keys),
            ["schema", "hold_type", "recorded_at", "app_version"]
        )
        XCTAssertEqual(object["schema"] as? Int, 1)
        XCTAssertEqual(object["hold_type"] as? String, "line")
        XCTAssertEqual(object["app_version"] as? String, "0.1.0")
    }

    func testTheDateIsISO8601InUTC() throws {
        try sample().write(for: movie)
        let object = try sidecarJSON()
        XCTAssertEqual(object["recorded_at"] as? String, "2026-09-28T14:30:59Z")
    }

    func testAnUnknownHoldTypeReadsBackAsLine() throws {
        // A sidecar from a future build — or an edited file — reads as the
        // default instead of failing the whole document.
        for raw in ["bogus", "tuck", "one_arm", ""] {
            let json = """
            {"schema":1,"hold_type":"\(raw)","recorded_at":"2026-09-28T14:30:59Z","app_version":"0.1.0"}
            """
            try Data(json.utf8).write(to: RecordingMetadata.sidecarURL(for: movie))
            XCTAssertEqual(try RecordingMetadata.read(for: movie)?.holdType, .line, "hold_type: \(raw)")
        }
        // …and a known, available one stays itself.
        let line = #"{"schema":1,"hold_type":"line","recorded_at":"2026-09-28T14:30:59Z","app_version":"0.1.0"}"#
        try Data(line.utf8).write(to: RecordingMetadata.sidecarURL(for: movie))
        XCTAssertEqual(try RecordingMetadata.read(for: movie)?.holdType, .line)
    }

    // MARK: - The live stats (chainlink #91)

    /// The five optional fields live mode writes beside a take, and the round
    /// trip they make: numbers stay numbers, the cues stay in order, and the
    /// schema is still 1.
    func testTheLiveStatsRoundTrip() throws {
        var metadata = sample()
        metadata.liveFpsTarget = 5.0
        metadata.liveFpsAchieved = 4.8
        metadata.avgInferenceMs = 96.3
        metadata.recordingFramesDropped = 0
        metadata.cuesSpoken = [
            SpokenCue(tMs: 1_240, cue: "line_hold"),
            SpokenCue(tMs: 6_020, cue: "time_mark_5"),
            SpokenCue(tMs: 11_040, cue: "time_mark_10"),
        ]
        try metadata.write(for: movie)

        let read = try XCTUnwrap(RecordingMetadata.read(for: movie))
        XCTAssertEqual(read, metadata)
        XCTAssertEqual(read.schema, 1, "the schema does not move for optional fields")
        XCTAssertEqual(read.liveFpsTarget, 5.0)
        XCTAssertEqual(read.liveFpsAchieved, 4.8)
        XCTAssertEqual(read.avgInferenceMs, 96.3)
        XCTAssertEqual(read.recordingFramesDropped, 0)
        XCTAssertEqual(read.cuesSpoken?.map(\.cue), ["line_hold", "time_mark_5", "time_mark_10"])

        let object = try sidecarJSON()
        XCTAssertEqual(
            Set(object.keys),
            [
                "schema", "hold_type", "recorded_at", "app_version",
                "live_fps_target", "live_fps_achieved", "avg_inference_ms",
                "recording_frames_dropped", "cues_spoken",
            ]
        )
        XCTAssertEqual(object["live_fps_target"] as? Double, 5.0)
        XCTAssertEqual(object["live_fps_achieved"] as? Double, 4.8)
        XCTAssertEqual(object["avg_inference_ms"] as? Double, 96.3)
        XCTAssertEqual(object["recording_frames_dropped"] as? Int, 0)
        let cues = try XCTUnwrap(object["cues_spoken"] as? [[String: Any]])
        XCTAssertEqual(cues.count, 3)
        XCTAssertEqual(cues.first?["t_ms"] as? Int, 1_240)
        XCTAssertEqual(cues.first?["cue"] as? String, "line_hold")
    }

    /// A take live mode never touched (no model in the bundle, a simulator
    /// build) writes the sidecar it always wrote: the fields are absent, not
    /// zero — and an *old* sidecar reads back the same way.
    func testTheLiveFieldsAreAbsentWhenThereWereNone() throws {
        try sample().write(for: movie)

        let object = try sidecarJSON()
        XCTAssertNil(object["live_fps_target"])
        XCTAssertNil(object["live_fps_achieved"])
        XCTAssertNil(object["avg_inference_ms"])
        XCTAssertNil(object["recording_frames_dropped"])
        XCTAssertNil(object["cues_spoken"])

        let read = try XCTUnwrap(RecordingMetadata.read(for: movie))
        XCTAssertNil(read.liveFpsTarget)
        XCTAssertNil(read.cuesSpoken)
        XCTAssertEqual(read, sample(), "the four original fields are unchanged")
    }

    /// One field present without the others — a take that dropped frames but
    /// had no live pose (or stopped before its first frame) — must still read.
    func testTheLiveFieldsAreIndependentlyOptional() throws {
        let json = """
        {"schema":1,"hold_type":"line","recorded_at":"2026-09-28T14:30:59Z",
         "app_version":"0.1.0","recording_frames_dropped":3}
        """
        try Data(json.utf8).write(to: RecordingMetadata.sidecarURL(for: movie))
        let read = try XCTUnwrap(RecordingMetadata.read(for: movie))
        XCTAssertEqual(read.recordingFramesDropped, 3)
        XCTAssertNil(read.liveFpsTarget)
        XCTAssertNil(read.cuesSpoken)
    }
}
