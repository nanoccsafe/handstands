import Foundation

/// Where a recording is written: `Application Support/Recordings/<yyyyMMdd-HHmmss>.mov`.
///
/// The name is a sortable local timestamp, so the list of attempts on the
/// phone reads in the order they were taken; the uniquing step keeps a second
/// attempt started in the same second from quietly overwriting the first.
/// Name generation is separate from the file system on purpose — the tests
/// pin the format and the collision rule down without writing anything.
enum RecordingFile {
    /// The folder under Application Support. It is created on first use and
    /// deliberately **not** marked excluded from backup: recordings are the
    /// user's footage, so they ride along in an iCloud backup like any other
    /// app document (`URLResourceValues.isExcludedFromBackup` is never set).
    static let folderName = "Recordings"
    static let pathExtension = "mov"

    /// `20260928-143059.mov` for a given instant in a given time zone.
    static func filename(for date: Date, timeZone: TimeZone = .current) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "\(formatter.string(from: date)).\(pathExtension)"
    }

    /// The recording folder, created (with its parents) if it does not exist.
    static func directory(fileManager: FileManager = .default) throws -> URL {
        let base = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = base.appendingPathComponent(folderName, isDirectory: true)
        if !fileManager.fileExists(atPath: directory.path) {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        return directory
    }

    /// An unused URL for `date` inside the recording folder.
    ///
    /// The folder is created; the file itself is not — `AVAssetWriter` makes
    /// it. Collisions within one second become `name-1.mov`, `name-2.mov`, …
    static func url(
        date: Date = Date(),
        timeZone: TimeZone = .current,
        fileManager: FileManager = .default
    ) throws -> URL {
        let directory = try directory(fileManager: fileManager)
        return unique(
            in: directory,
            filename: filename(for: date, timeZone: timeZone),
            fileManager: fileManager
        )
    }

    /// The collision rule on its own: the first name in `directory` that is
    /// not already taken (`foo.mov`, `foo-1.mov`, `foo-2.mov`, …).
    static func unique(in directory: URL, filename: String, fileManager: FileManager = .default) -> URL {
        let stem = URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent
        var candidate = directory.appendingPathComponent(stem).appendingPathExtension(pathExtension)
        var suffix = 1
        while fileManager.fileExists(atPath: candidate.path) {
            candidate = directory
                .appendingPathComponent("\(stem)-\(suffix)")
                .appendingPathExtension(pathExtension)
            suffix += 1
        }
        return candidate
    }
}
