import Foundation
import HandstandCore
import SwiftData

/// History's half of the store (chainlink #51): a thin wrapper over a
/// `ModelContext` that keeps the `Recordings` folder and the sessions in
/// step. The folder stays the source of truth — `reconcile()` is what makes
/// the database catch up with recordings made before this feature, or with
/// a row that was never written because a take failed at the last moment.
///
/// Local only: the container behind `context` is opened with
/// `cloudKitDatabase: .none` (`HandstandApp.swift`) and nothing here talks
/// to the network. `delete` never touches a file outside the Recordings
/// folder, whatever a stored filename says — except the session's annotated
/// export in Caches (chainlink #50), which is built from the movie's *name*
/// and so can only be the file the export button wrote.
@MainActor
final class SessionStore {
    /// How `reconcile()` reads a movie's duration and size. Injected so the
    /// tests can answer with a plain value instead of opening a real movie.
    typealias InfoReader = @Sendable (URL) async throws -> VideoInfo

    private let context: ModelContext
    private let recordingsDirectory: URL
    private let fileManager: FileManager
    private let readInfo: InfoReader
    /// Where `<basename>-annotated.mp4` exports (chainlink #50) are cached —
    /// the app's Caches directory unless a test injects one, so `delete`
    /// removes an export from the same place the export button wrote it.
    private let annotatedDirectory: URL

    /// - Parameters:
    ///   - context: the History database (one container for the app).
    ///   - recordingsDirectory: where the movies live —
    ///     `RecordingFile.directory()`.
    ///   - fileManager: injectable for the tests.
    ///   - readInfo: reads duration and frame size; defaults to
    ///     `VideoInfoReader` (AVFoundation).
    ///   - annotatedDirectory: where annotated exports are cached; defaults
    ///     to the app's Caches directory.
    init(
        context: ModelContext,
        recordingsDirectory: URL,
        fileManager: FileManager = .default,
        readInfo: @escaping InfoReader = { try await VideoInfoReader.read(url: $0) },
        annotatedDirectory: URL? = nil
    ) {
        self.context = context
        self.recordingsDirectory = recordingsDirectory.standardizedFileURL
        self.fileManager = fileManager
        self.readInfo = readInfo
        self.annotatedDirectory = annotatedDirectory
            ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? recordingsDirectory.standardizedFileURL
    }

    // MARK: - Reading

    /// Every session, newest first — the order of the History list.
    func sessions() throws -> [Session] {
        try context.fetch(
            FetchDescriptor<Session>(sortBy: [SortDescriptor(\.recordedAt, order: .reverse)])
        )
    }

    /// The movie a session points at: the folder joined with the stored
    /// **relative** filename. The folder may have moved (a new install);
    /// the name never did.
    func movieURL(for session: Session) -> URL {
        recordingsDirectory.appendingPathComponent(session.movieFilename)
    }

    /// Where this session's annotated export (chainlink #50) is cached —
    /// `<basename>-annotated.mp4` in the Caches directory (a test's own
    /// directory). Not a recording: it is never in the Recordings folder
    /// and never becomes a History row.
    func annotatedExportURL(for session: Session) -> URL {
        Self.annotatedExportURL(for: movieURL(for: session), in: annotatedDirectory)
    }

    /// The export's name, spelled once so the export button and `delete`
    /// can never disagree: the movie's basename plus `-annotated.mp4`. The
    /// basename is a *name*, so joining it to `directory` cannot leave it.
    static func annotatedExportURL(for movie: URL, in directory: URL) -> URL {
        let basename = movie.deletingPathExtension().lastPathComponent
        return directory.appendingPathComponent("\(basename)-annotated.mp4")
    }

    // MARK: - Writing

    /// Records `movie` as a session — **idempotent**: the same filename
    /// returns the row that already exists instead of adding a second one,
    /// so a recording that is both saved by the recorder and picked up by
    /// `reconcile()` still counts once.
    ///
    /// The stored name is relative to the Recordings folder; the hold and
    /// the date come from `metadata` (the sidecar) when there is one, and
    /// from the file itself when there is not.
    func add(movie: URL, metadata: RecordingMetadata?, info: VideoInfo) throws -> Session {
        let filename = relativeFilename(for: movie)
        if let existing = try session(filename: filename) {
            return existing
        }
        let session = Session(
            movieFilename: filename,
            recordedAt: metadata?.recordedAt ?? creationDate(of: movie) ?? Date(),
            holdTypeRaw: (metadata?.holdType ?? HoldType.default).rawValue,
            durationS: info.duration,
            width: info.width,
            height: info.height
        )
        context.insert(session)
        try context.save()
        return session
    }

    /// Deletes a session **and its video**: the `.mov`, its `.json` sidecar,
    /// its `.pose.json` pose cache (chainlink #48), its `.diagnostics.json`
    /// analysis diagnostics (chainlink #93), its leftover annotated export
    /// in Caches (chainlink #50) and the row. A missing sidecar, cache,
    /// diagnostics or export is not an error — the movie goes and the row
    /// goes with it. Files outside the Recordings folder are never touched:
    /// only URLs that resolve inside it are removed, so a filename that
    /// somehow points elsewhere costs the file nothing (the row still goes,
    /// otherwise History would keep listing a take the app refuses to open).
    /// The one file built outside the folder is the annotated export, whose
    /// URL is a *name* joined to the fixed exports directory — it cannot
    /// point at anything the app did not write.
    func delete(_ session: Session) throws {
        let movie = movieURL(for: session)
        if isInsideRecordingsFolder(movie) {
            let sidecar = RecordingMetadata.sidecarURL(for: movie)
            if isInsideRecordingsFolder(sidecar), fileManager.fileExists(atPath: sidecar.path) {
                try fileManager.removeItem(at: sidecar)
            }
            let cache = PoseCache.cacheURL(for: movie)
            if isInsideRecordingsFolder(cache), fileManager.fileExists(atPath: cache.path) {
                try fileManager.removeItem(at: cache)
            }
            // The analysis diagnostics (chainlink #93): a few KB beside the
            // recording, and a file of numbers about a take nobody can play
            // is a file nobody would ever read again.
            let diagnostics = AnalysisDiagnostics.url(for: movie)
            if isInsideRecordingsFolder(diagnostics),
                fileManager.fileExists(atPath: diagnostics.path)
            {
                try fileManager.removeItem(at: diagnostics)
            }
            if fileManager.fileExists(atPath: movie.path) {
                try fileManager.removeItem(at: movie)
            }
        }
        // A leftover `<basename>-annotated.mp4` in Caches: nobody can watch
        // an export of a recording that is gone. It is a cache rather than
        // the recording, so a file that refuses to go must not keep the row
        // alive — the deletion succeeds either way.
        let annotated = Self.annotatedExportURL(for: movie, in: annotatedDirectory)
        if fileManager.fileExists(atPath: annotated.path) {
            try? fileManager.removeItem(at: annotated)
        }
        context.delete(session)
        try context.save()
    }

    /// Makes the database match the folder, so recordings made before this
    /// feature (or while a take failed to be recorded) appear too:
    ///
    /// - a `.mov` with no session gets one — hold type and date from its
    ///   sidecar if it has one, otherwise Line and the file's creation date;
    ///   duration and frame size from `readInfo`;
    /// - a session whose `.mov` is gone is removed;
    /// - anything that is not a `.mov` (sidecars, `.pose.json` pose caches,
    ///   junk) is ignored.
    ///
    /// Runs every time the History screen appears.
    func reconcile() async throws -> (added: Int, removed: Int) {
        var added = 0
        var removed = 0

        let known = try sessions()
        var seen = Set(known.map(\.movieFilename))

        for movie in try movieFiles() {
            let filename = relativeFilename(for: movie)
            guard !seen.contains(filename) else { continue }
            seen.insert(filename)

            // What the take *was* comes from the sidecar; what it *is* comes
            // from the file. An unreadable movie still becomes a row (it
            // exists, and hiding it would hide the recording) — it just
            // reads as 0:00 until something can open it, so one bad file
            // never costs the user the rest of their History.
            let metadata = try? RecordingMetadata.read(for: movie)
            let info: VideoInfo
            do {
                info = try await readInfo(movie)
            } catch {
                info = VideoInfo(duration: 0, width: 0, height: 0)
            }
            context.insert(
                Session(
                    movieFilename: filename,
                    recordedAt: metadata?.recordedAt ?? creationDate(of: movie) ?? Date(),
                    holdTypeRaw: (metadata?.holdType ?? HoldType.default).rawValue,
                    durationS: info.duration,
                    width: info.width,
                    height: info.height
                )
            )
            added += 1
        }

        for session in known where !fileManager.fileExists(atPath: movieURL(for: session).path) {
            context.delete(session)
            removed += 1
        }

        try context.save()
        return (added, removed)
    }

    /// Writes the analysis of one take (chainlink #47's entry point, tested
    /// now): when it was scored, how it scored, and which run did it. A
    /// `nil` score is allowed — it records "analysed, no score", and
    /// `SessionProgress` keeps it out of the numbers.
    ///
    /// `note` is why the clip could not be measured (an unusable analysis:
    /// hold count 0, score `nil`), `nil` for one that was measured — it is
    /// written every time, so a later good run clears an earlier note.
    func recordAnalysis(
        for session: Session,
        score: Double?,
        holdCount: Int,
        longestHoldS: Double,
        version: String,
        note: String? = nil
    ) throws {
        session.analyzedAt = Date()
        session.clipScore = score
        session.holdCount = holdCount
        session.longestHoldS = longestHoldS
        session.analysisVersion = version
        session.analysisNote = note
        try context.save()
    }

    // MARK: - Private

    /// The `.mov` files in the folder, in a stable (name) order so two runs
    /// over the same folder add the same rows in the same order.
    private func movieFiles() throws -> [URL] {
        try fileManager.contentsOfDirectory(
            at: recordingsDirectory,
            includingPropertiesForKeys: [.isRegularFileKey, .creationDateKey],
            options: [.skipsHiddenFiles]
        )
        .filter { $0.pathExtension.lowercased() == RecordingFile.pathExtension }
        .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func session(filename: String) throws -> Session? {
        try context.fetch(
            FetchDescriptor<Session>(predicate: #Predicate { $0.movieFilename == filename })
        ).first
    }

    /// The movie's name relative to the Recordings folder — never the
    /// absolute path (the app container changes between installs). A URL
    /// from elsewhere collapses to its last path component rather than
    /// escaping the folder.
    private func relativeFilename(for movie: URL) -> String {
        let directory = recordingsDirectory.path
        let path = movie.standardizedFileURL.path
        if path.hasPrefix(directory + "/") {
            return String(path.dropFirst(directory.count + 1))
        }
        return movie.lastPathComponent
    }

    private func isInsideRecordingsFolder(_ url: URL) -> Bool {
        url.standardizedFileURL.path.hasPrefix(recordingsDirectory.path + "/")
    }

    private func creationDate(of url: URL) -> Date? {
        (try? url.resourceValues(forKeys: [.creationDateKey]))?.creationDate
    }
}
