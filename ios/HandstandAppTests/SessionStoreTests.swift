import SwiftData
import XCTest

@testable import HandstandApp
import HandstandCore

/// `SessionStore` (chainlink #51): the History database as a mirror of the
/// Recordings folder — one row per movie, newest first, `reconcile()`
/// catching up with files made before the feature existed, and `delete`
/// taking the video and its sidecar with the row. No real movie is ever
/// opened: `add` is handed its `VideoInfo` and `reconcile()` an injected
/// reader, so a zero-byte `.mov` stands in for footage.
@MainActor
final class SessionStoreTests: XCTestCase {
    /// A temp Recordings folder with an in-memory SwiftData store over it.
    /// The returned container keeps the store alive for the test; the
    /// caller removes the folder when it is done. `annotatedDirectory` is
    /// where the store looks for annotated exports (chainlink #50) — a test
    /// passes its own folder so nothing touches the app's real Caches.
    private func makeSUT(
        readInfo: SessionStore.InfoReader? = nil,
        annotatedDirectory: URL? = nil
    ) throws -> (
        store: SessionStore, container: ModelContainer, directory: URL
    ) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("SessionStoreTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let container = try ModelContainer(
            for: Session.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        let store: SessionStore
        if let readInfo {
            store = SessionStore(
                context: ModelContext(container),
                recordingsDirectory: directory,
                readInfo: readInfo,
                annotatedDirectory: annotatedDirectory
            )
        } else {
            store = SessionStore(
                context: ModelContext(container),
                recordingsDirectory: directory,
                annotatedDirectory: annotatedDirectory
            )
        }
        return (store, container, directory)
    }

    /// A reader that answers every movie with one fixed `VideoInfo` — the
    /// tests never touch AVFoundation.
    private func fixedReader(_ info: VideoInfo) -> SessionStore.InfoReader {
        { _ in info }
    }

    /// An empty `.mov` where the writer would have left one.
    private func makeMovie(named name: String, in directory: URL) throws -> URL {
        let movie = directory.appendingPathComponent(name)
        try Data().write(to: movie)
        return movie
    }

    private func metadata(recordedAt: Date, hold: HoldType = .line) -> RecordingMetadata {
        RecordingMetadata(holdType: hold, recordedAt: recordedAt, appVersion: "0.1.0")
    }

    private func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    // MARK: - add

    func testAddIsIdempotentForTheSameFilename() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let take = metadata(recordedAt: Date(timeIntervalSince1970: 1_790_000_000))

        let first = try sut.store.add(
            movie: movie,
            metadata: take,
            info: VideoInfo(duration: 12.4, width: 1080, height: 1920)
        )
        // The same file a second time — what a repeat pass would do — must
        // return the row that already exists, not create another one.
        let second = try sut.store.add(
            movie: movie,
            metadata: take,
            info: VideoInfo(duration: 99, width: 1, height: 1)
        )

        XCTAssertEqual(first.persistentModelID, second.persistentModelID)
        let sessions = try sut.store.sessions()
        XCTAssertEqual(sessions.count, 1)
        XCTAssertEqual(sessions[0].durationS, 12.4, "the first add wins, the row is not rewritten")
        XCTAssertEqual(sessions[0].movieFilename, "20260928-143059.mov")
    }

    func testAddStoresTheFilenameRelativeToTheFolderNeverAnAbsolutePath() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        // A movie handed over as an absolute path must still be stored as
        // just its name: the container path changes between installs.
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)

        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 5, width: 720, height: 1280)
        )

        XCTAssertEqual(session.movieFilename, "20260928-143059.mov")
        XCTAssertFalse(session.movieFilename.hasPrefix("/"))
    }

    func testSessionsAreSortedNewestFirst() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let older = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let newer = try makeMovie(named: "20260929-080000.mov", in: sut.directory)
        let info = VideoInfo(duration: 10, width: 1080, height: 1920)

        _ = try sut.store.add(
            movie: older,
            metadata: metadata(recordedAt: Date(timeIntervalSince1970: 1_790_000_000)),
            info: info
        )
        _ = try sut.store.add(
            movie: newer,
            metadata: metadata(recordedAt: Date(timeIntervalSince1970: 1_790_100_000)),
            info: info
        )

        XCTAssertEqual(
            try sut.store.sessions().map(\.movieFilename),
            ["20260929-080000.mov", "20260928-143059.mov"]
        )
    }

    func testMovieURLJoinsTheFolderAndTheFilename() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)

        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 5, width: 720, height: 1280)
        )

        let url = sut.store.movieURL(for: session)
        XCTAssertEqual(url.path, sut.directory.appendingPathComponent("20260928-143059.mov").path)
        XCTAssertEqual(url.lastPathComponent, "20260928-143059.mov")
        XCTAssertEqual(url.deletingLastPathComponent().path, sut.directory.path)
    }

    // MARK: - delete

    func testDeleteRemovesTheMovieTheSidecarAndTheRecord() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let take = metadata(recordedAt: Date(timeIntervalSince1970: 1_790_000_000))
        try take.write(for: movie)
        let session = try sut.store.add(
            movie: movie,
            metadata: take,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )

        try sut.store.delete(session)

        XCTAssertFalse(exists(movie), "the video is removed from the phone")
        XCTAssertFalse(exists(RecordingMetadata.sidecarURL(for: movie)), "the sidecar goes with it")
        XCTAssertEqual(try sut.store.sessions().count, 0)
    }

    func testDeleteWithoutASidecarJustRemovesTheMovie() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )
        XCTAssertFalse(exists(RecordingMetadata.sidecarURL(for: movie)))

        try sut.store.delete(session)

        XCTAssertFalse(exists(movie))
        XCTAssertEqual(try sut.store.sessions().count, 0)
    }

    func testDeleteRemovesTheRecordEvenWhenTheMovieIsAlreadyGone() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )
        try FileManager.default.removeItem(at: movie)

        try sut.store.delete(session)

        XCTAssertEqual(try sut.store.sessions().count, 0)
    }

    // MARK: - reconcile

    func testReconcileAddsAMovieWithASidecarUsingItsHoldAndDate() async throws {
        let sut = try makeSUT(
            readInfo: fixedReader(VideoInfo(duration: 8.5, width: 720, height: 1280))
        )
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        // Whole seconds: the sidecar's ISO-8601 date has no fractional part.
        let taken = Date(timeIntervalSince1970: 1_790_000_000)
        try metadata(recordedAt: taken).write(for: movie)

        let result = try await sut.store.reconcile()

        XCTAssertEqual(result.added, 1)
        XCTAssertEqual(result.removed, 0)
        let sessions = try sut.store.sessions()
        XCTAssertEqual(sessions.count, 1)
        XCTAssertEqual(sessions[0].movieFilename, "20260928-143059.mov")
        XCTAssertEqual(sessions[0].holdType, .line)
        // The date proves the sidecar was read: it is hours away from the
        // file's creation date (Line is the only recordable hold, so the
        // hold alone could not tell sidecar from no-sidecar apart).
        XCTAssertEqual(sessions[0].recordedAt.timeIntervalSince(taken), 0, accuracy: 0.5)
        XCTAssertEqual(sessions[0].durationS, 8.5)
        XCTAssertEqual(sessions[0].width, 720)
        XCTAssertEqual(sessions[0].height, 1280)
        XCTAssertNil(sessions[0].clipScore, "no analysis exists yet (#47)")

        // A second pass over the same folder adds nothing.
        let again = try await sut.store.reconcile()
        XCTAssertEqual(again.added, 0)
        XCTAssertEqual(again.removed, 0)
        XCTAssertEqual(try sut.store.sessions().count, 1)
    }

    func testReconcileAddsAMovieWithNoSidecarAsLineOnItsFileDate() async throws {
        let sut = try makeSUT(
            readInfo: fixedReader(VideoInfo(duration: 3, width: 1080, height: 1920))
        )
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let values = try movie.resourceValues(forKeys: [.creationDateKey])
        let created = try XCTUnwrap(values.creationDate)

        let result = try await sut.store.reconcile()

        XCTAssertEqual(result.added, 1)
        let sessions = try sut.store.sessions()
        XCTAssertEqual(sessions.count, 1)
        XCTAssertEqual(sessions[0].holdType, .line, "no sidecar → the default hold")
        XCTAssertEqual(
            sessions[0].recordedAt.timeIntervalSince(created), 0, accuracy: 2,
            "no sidecar → the file's own creation date"
        )
    }

    func testReconcileRemovesASessionWhoseMovieIsGone() async throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )
        XCTAssertEqual(session.movieFilename, "20260928-143059.mov")
        // The video disappeared (deleted from Files, an interrupted copy…):
        // the row must not survive it.
        try FileManager.default.removeItem(at: movie)

        let result = try await sut.store.reconcile()

        XCTAssertEqual(result.removed, 1)
        XCTAssertEqual(result.added, 0)
        XCTAssertEqual(try sut.store.sessions().count, 0)
    }

    func testReconcileIgnoresNonMovFiles() async throws {
        let sut = try makeSUT(
            readInfo: fixedReader(VideoInfo(duration: 8.5, width: 720, height: 1280))
        )
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        // Notes, an orphan sidecar and a directory that happens to end in
        // .mov are not recordings.
        try Data("not a video".utf8).write(to: sut.directory.appendingPathComponent("notes.txt"))
        try Data("{}".utf8).write(
            to: sut.directory.appendingPathComponent("20260928-143059.json")
        )
        try FileManager.default.createDirectory(
            at: sut.directory.appendingPathComponent("folder.mov", isDirectory: true),
            withIntermediateDirectories: true
        )

        let result = try await sut.store.reconcile()

        XCTAssertEqual(result.added, 0)
        XCTAssertEqual(result.removed, 0)
        XCTAssertEqual(try sut.store.sessions().count, 0)
    }

    func testReconcileKeepsRecordingWhatItCannotReadInsteadOfHidingIt() async throws {
        // A movie the reader refuses (corrupt, not really a video) still
        // gets its row — one bad file must not cost the user their History.
        let sut = try makeSUT(
            readInfo: { _ in throw VideoInfoReader.ReadError.notAVideo }
        )
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        _ = try makeMovie(named: "20260928-143059.mov", in: sut.directory)

        let result = try await sut.store.reconcile()

        XCTAssertEqual(result.added, 1)
        let sessions = try sut.store.sessions()
        XCTAssertEqual(sessions.count, 1)
        XCTAssertEqual(sessions[0].movieFilename, "20260928-143059.mov")
        XCTAssertEqual(sessions[0].durationS, 0)
    }

    // MARK: - recordAnalysis

    func testRecordAnalysisSetsTheAnalysisFields() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )
        XCTAssertNil(session.clipScore, "a fresh recording is not analysed")

        try sut.store.recordAnalysis(
            for: session,
            score: 0.87,
            holdCount: 3,
            longestHoldS: 4.2,
            version: "analysis-1"
        )

        XCTAssertEqual(session.clipScore, 0.87)
        XCTAssertEqual(session.holdCount, 3)
        XCTAssertEqual(session.longestHoldS, 4.2)
        XCTAssertEqual(session.analysisVersion, "analysis-1")
        XCTAssertNil(session.analysisNote, "a measured clip has no note")
        XCTAssertNotNil(session.analyzedAt)
        // Analysis changes the row, never the file list.
        XCTAssertEqual(try sut.store.sessions().count, 1)
        XCTAssertTrue(exists(movie))
    }

    func testRecordAnalysisAcceptsANilScore() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )

        try sut.store.recordAnalysis(
            for: session,
            score: nil,
            holdCount: 2,
            longestHoldS: 1.5,
            version: "analysis-1"
        )

        XCTAssertNotNil(session.analyzedAt)
        XCTAssertNil(session.clipScore, "analysed without a score stays score-less")
        XCTAssertEqual(session.holdCount, 2)
    }

    /// An unmeasurable analysis is still an analysis: `recordAnalysis` is
    /// called with hold count 0, no score, and the reason in `note` — which
    /// is what the detail screen and the History row read as "Couldn't
    /// measure" instead of a hold count of 0.
    func testRecordAnalysisStoresTheNoteOfAnUnmeasurableClip() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )
        XCTAssertNil(session.analysisNote, "a fresh recording has no note")

        try sut.store.recordAnalysis(
            for: session,
            score: nil,
            holdCount: 0,
            longestHoldS: 0,
            version: "vision-1",
            note: "thigh was measurable on 0 frame(s), need 10"
        )

        XCTAssertEqual(session.analysisNote, "thigh was measurable on 0 frame(s), need 10")
        XCTAssertNil(session.clipScore)
        XCTAssertEqual(session.holdCount, 0)
        XCTAssertEqual(session.longestHoldS, 0)
        XCTAssertNotNil(session.analyzedAt, "it was analysed — just not measured")
        XCTAssertEqual(session.analysisVersion, "vision-1")
    }

    /// A later, measurable run overwrites the note: the row shows the
    /// numbers then, not the old failure.
    func testRecordAnalysisWithoutANoteClearsAnEarlierOne() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )
        try sut.store.recordAnalysis(
            for: session, score: nil, holdCount: 0, longestHoldS: 0,
            version: "vision-1", note: "no body length"
        )

        try sut.store.recordAnalysis(
            for: session, score: 0.87, holdCount: 2, longestHoldS: 4.2, version: "vision-1"
        )

        XCTAssertNil(session.analysisNote)
        XCTAssertEqual(session.clipScore, 0.87)
        XCTAssertEqual(session.holdCount, 2)
    }

    // MARK: - The pose cache (chainlink #48)

    /// Deleting a recording takes its `.pose.json` with the movie: a cache
    /// for a film nobody can play is a file nobody would ever read again.
    func testDeleteRemovesThePoseCache() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )
        try PoseCache.write(frames: [], for: movie)
        let cache = PoseCache.cacheURL(for: movie)
        XCTAssertTrue(exists(cache), "the cache sits beside the movie")

        try sut.store.delete(session)

        XCTAssertFalse(exists(cache), "the pose cache goes with the recording")
        XCTAssertFalse(exists(movie))
        XCTAssertEqual(try sut.store.sessions().count, 0)
    }

    /// A `.pose.json` beside nothing is not a recording: `reconcile()` must
    /// not give it a History row (it filters on `.mov`, and this pins the
    /// cache's own suffix down as one of the ignored files).
    func testReconcileIgnoresThePoseCacheFile() async throws {
        let sut = try makeSUT(
            readInfo: fixedReader(VideoInfo(duration: 8.5, width: 720, height: 1280))
        )
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        try Data(#"{"schema": 1, "frames": []}"#.utf8).write(
            to: sut.directory.appendingPathComponent("20260928-143059.pose.json")
        )

        let result = try await sut.store.reconcile()

        XCTAssertEqual(result.added, 0)
        XCTAssertEqual(result.removed, 0)
        XCTAssertEqual(try sut.store.sessions().count, 0)
    }

    // MARK: - The analysis diagnostics (chainlink #93)

    /// A diagnostics document for a test: the store never reads the
    /// numbers, only the file's name and existence, so this spells the
    /// shape out rather than running an analysis to get one.
    private func diagnostics() -> AnalysisDiagnostics {
        AnalysisDiagnostics(
            schema: AnalysisDiagnostics.schema,
            appVersion: "0.1.0",
            backend: "vision",
            analysisVersion: "vision-1",
            analysisWallTimeS: 1.5,
            video: AnalysisDiagnostics.Video(
                durationS: 10, fps: 59.9, width: 1080, height: 1920),
            analysedFps: 30,
            framesTotal: 300,
            framesWithPerson: 210,
            framesWithPersonPct: 70,
            unknownReasons: ["no_visible_wrist": 40],
            outOfFrame: AnalysisDiagnostics.OutOfFrame(
                partly: 40, notInFrame: 0, edges: ["top": 40]),
            dominantReason: "out_of_frame",
            holdCount: 1,
            holdDurationsS: [1.5],
            usable: true,
            unusableReason: ""
        )
    }

    /// Deleting a recording takes its `.diagnostics.json` with it: a file
    /// of numbers about a take nobody can play is a file nobody would ever
    /// read again.
    func testDeleteRemovesTheDiagnosticsFile() throws {
        let sut = try makeSUT()
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )
        try AnalysisDiagnostics.write(diagnostics(), for: movie)
        let file = AnalysisDiagnostics.url(for: movie)
        XCTAssertEqual(file.lastPathComponent, "20260928-143059.diagnostics.json")
        XCTAssertTrue(exists(file), "the diagnostics sit beside the movie")

        try sut.store.delete(session)

        XCTAssertFalse(exists(file), "the diagnostics go with the recording")
        XCTAssertFalse(exists(movie))
        XCTAssertEqual(try sut.store.sessions().count, 0)
    }

    /// A `.diagnostics.json` beside nothing is not a recording, and one
    /// beside a recording is not a second one: `reconcile()` filters on the
    /// movie's own extension, so the file is never a History row of its
    /// own and never a reason to add or remove one.
    func testReconcileIgnoresTheDiagnosticsFile() async throws {
        let sut = try makeSUT(
            readInfo: fixedReader(VideoInfo(duration: 8.5, width: 720, height: 1280))
        )
        defer { try? FileManager.default.removeItem(at: sut.directory) }
        try Data(#"{"schema": 1, "frames_total": 10}"#.utf8).write(
            to: sut.directory.appendingPathComponent("20260928-143059.diagnostics.json")
        )

        let result = try await sut.store.reconcile()

        XCTAssertEqual(result.added, 0)
        XCTAssertEqual(result.removed, 0)
        XCTAssertEqual(try sut.store.sessions().count, 0)
    }

    // MARK: - The annotated export (chainlink #50)

    /// The export's name, spelled once for the export button and for
    /// `delete`: `<basename>-annotated.mp4` in the exports directory (the
    /// app's Caches; a test's own folder here).
    func testTheAnnotatedExportIsNamedAfterTheMovie() throws {
        let caches = FileManager.default.temporaryDirectory
            .appendingPathComponent("SessionStoreTests-caches-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: caches, withIntermediateDirectories: true)
        let sut = try makeSUT(annotatedDirectory: caches)
        defer {
            try? FileManager.default.removeItem(at: sut.directory)
            try? FileManager.default.removeItem(at: caches)
        }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 5, width: 720, height: 1280)
        )

        let export = sut.store.annotatedExportURL(for: session)

        XCTAssertEqual(export.lastPathComponent, "20260928-143059-annotated.mp4")
        XCTAssertEqual(
            export.deletingLastPathComponent().path, caches.path,
            "exports are cached outside the Recordings folder")
        XCTAssertEqual(
            export.path, SessionStore.annotatedExportURL(for: movie, in: caches).path,
            "the static spelling `delete` uses agrees with the store's")
    }

    /// Deleting a recording takes its leftover annotated export with it:
    /// nobody can watch an export of a film that is gone, and Caches is not
    /// a place anything reads back.
    func testDeleteRemovesTheAnnotatedExportInCaches() throws {
        let caches = FileManager.default.temporaryDirectory
            .appendingPathComponent("SessionStoreTests-caches-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: caches, withIntermediateDirectories: true)
        let sut = try makeSUT(annotatedDirectory: caches)
        defer {
            try? FileManager.default.removeItem(at: sut.directory)
            try? FileManager.default.removeItem(at: caches)
        }
        let movie = try makeMovie(named: "20260928-143059.mov", in: sut.directory)
        let session = try sut.store.add(
            movie: movie,
            metadata: nil,
            info: VideoInfo(duration: 10, width: 1080, height: 1920)
        )
        let export = sut.store.annotatedExportURL(for: session)
        try Data("an annotated movie".utf8).write(to: export)
        XCTAssertTrue(exists(export), "the export sits in the caches folder")

        try sut.store.delete(session)

        XCTAssertFalse(exists(export), "the leftover export goes with the recording")
        XCTAssertFalse(exists(movie))
        XCTAssertEqual(try sut.store.sessions().count, 0)
    }
}
