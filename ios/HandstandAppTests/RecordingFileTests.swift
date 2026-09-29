import XCTest

@testable import HandstandApp

/// The recording filename generator: a sortable local timestamp, and the
/// collision rule for two takes that start in the same second. Nothing here
/// needs a camera — `RecordingFile` never touches one.
final class RecordingFileTests: XCTestCase {
    private let utc = TimeZone(identifier: "UTC")!

    private var gregorianUTC: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = utc
        return calendar
    }

    func testFilenameIsAPlaceableTimestamp() {
        XCTAssertEqual(
            RecordingFile.filename(for: Date(timeIntervalSince1970: 0), timeZone: utc),
            "19700101-000000.mov"
        )

        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 28
        components.hour = 14
        components.minute = 30
        components.second = 59
        let afternoon = gregorianUTC.date(from: components)!
        XCTAssertEqual(RecordingFile.filename(for: afternoon, timeZone: utc), "20260928-143059.mov")
    }

    func testFilenameIsWrittenInTheGivenTimeZone() {
        // The same instant, read on each side of the date line: the name
        // follows the clock the user is standing next to.
        let epoch = Date(timeIntervalSince1970: 0)
        XCTAssertEqual(
            RecordingFile.filename(for: epoch, timeZone: TimeZone(identifier: "Asia/Tokyo")!),
            "19700101-090000.mov"
        )
        XCTAssertEqual(
            RecordingFile.filename(for: epoch, timeZone: TimeZone(identifier: "America/Los_Angeles")!),
            "19691231-160000.mov"
        )
    }

    func testATakeInTheSameSecondGetsANumberInsteadOfOverwriting() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("RecordingFileTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let first = RecordingFile.unique(in: directory, filename: "19700101-000000.mov")
        XCTAssertEqual(first.lastPathComponent, "19700101-000000.mov")
        // The file the writer would have made…
        XCTAssertTrue(FileManager.default.createFile(atPath: first.path, contents: Data()))

        let second = RecordingFile.unique(in: directory, filename: "19700101-000000.mov")
        XCTAssertEqual(second.lastPathComponent, "19700101-000000-1.mov")
        XCTAssertTrue(FileManager.default.createFile(atPath: second.path, contents: Data()))

        let third = RecordingFile.unique(in: directory, filename: "19700101-000000.mov")
        XCTAssertEqual(third.lastPathComponent, "19700101-000000-2.mov")
        XCTAssertNotEqual(third, second)
    }
}
