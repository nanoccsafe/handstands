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
}
