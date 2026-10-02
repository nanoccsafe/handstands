import CoreVideo
import XCTest

@testable import HandstandApp
import HandstandCore
import VisionPoseKit

/// `MediaPipeClipExtractor` (chainlink #45): the two passes, the orientation
/// choice and the map-back of the rotated pass — with an **injectable
/// landmarker protocol**, so fake landmark sequences stand in for the model:
/// no model file, no video, no network, the same code path the analysis runs.
///
/// What is checked here is the extractor's own contract:
/// * the chosen orientation is the one `OrientationChooser` (Python parity,
///   OrientationChooserTests) says, and the chosen pass's keypoints are the
///   ones emitted — upright in display pixels, rotated mapped back;
/// * the rotated pass really receives 180°-turned frames;
/// * progress crosses **both** passes (0 → 0.5 → 1);
/// * a wrong-shaped answer and a misaligned second pass are refused.
final class MediaPipeClipExtractorTests: XCTestCase {
    // MARK: - Fakes

    /// Scripted answers, one per frame, consumed in call order; records the
    /// top-left pixel and timestamp of every frame it is handed, so a test
    /// can see *what* the pass was fed (upright vs turned, and in which
    /// order).
    private final class ScriptedLandmarker: MediaPipeLandmarkDetecting,
        @unchecked Sendable
    {
        private let lock = NSLock()
        private var answers: [[MediaPipeLandmark]?]
        private(set) var received: [(tMs: Int, topLeftB: UInt8, topLeftG: UInt8)] = []

        init(_ answers: [[MediaPipeLandmark]?]) {
            self.answers = answers
        }

        func detectForVideo(_ frame: CVPixelBuffer, tMs: Int) throws -> [MediaPipeLandmark]? {
            let pixel = Self.topLeft(of: frame)
            lock.lock()
            defer { lock.unlock() }
            received.append((tMs, pixel.b, pixel.g))
            guard !answers.isEmpty else {
                throw MediaPipeExtractorError.passMismatch("fake ran out of scripted frames")
            }
            return answers.removeFirst()
        }

        static func topLeft(of buffer: CVPixelBuffer) -> (b: UInt8, g: UInt8) {
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { return (0, 0) }
            let bytes = base.assumingMemoryBound(to: UInt8.self)
            return (bytes[0], bytes[1])
        }
    }

    /// Which pass a `frames()` call belongs to: the first call is the
    /// upright pass, the second (and any later) the rotated one.
    private final class PassCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var calls = 0
        func next() -> Int {
            lock.lock()
            defer { lock.unlock() }
            calls += 1
            return calls
        }
    }

    /// Collects progress values across the extraction's executor.
    private final class ProgressLog: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: [Double] = []
        func append(_ value: Double) {
            lock.lock()
            stored.append(value)
            lock.unlock()
        }
        var values: [Double] {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }
    }

    /// A clip of `tMs` frames, re-readable (the extractor reads it twice);
    /// `secondPassTMs` lets a test make the passes disagree, `estimates:
    /// false` models a clip whose length cannot be estimated.
    private struct FakeFrames: ClipFrameSource {
        let tMs: [Int]
        let secondPassTMs: [Int]?
        let estimates: Bool
        let counter = PassCounter()

        init(_ tMs: [Int], secondPassTMs: [Int]? = nil, estimates: Bool = true) {
            self.tMs = tMs
            self.secondPassTMs = secondPassTMs
            self.estimates = estimates
        }

        func estimatedFrameCount() async throws -> Int {
            guard estimates else {
                throw MediaPipeExtractorError.passMismatch("no estimate available")
            }
            return tMs.count
        }

        func frames() -> AsyncThrowingStream<(buffer: CVPixelBuffer, tMs: Int), Error> {
            let pass = counter.next()
            let times = pass > 1 ? (secondPassTMs ?? tMs) : tMs
            return AsyncThrowingStream { continuation in
                for t in times {
                    continuation.yield((
                        buffer: MediaPipeClipExtractorTests.makeFrame(), tMs: t
                    ))
                }
                continuation.finish()
            }
        }
    }

    /// A 64×48 32BGRA frame whose pixels spell their own position: byte
    /// `B` = x, `G` = y — so a 180° turn is visible in the bytes.
    private static func makeFrame(width: Int = 64, height: Int = 48) -> CVPixelBuffer {
        var created: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &created)
        precondition(status == kCVReturnSuccess && created != nil, "test buffer")
        let buffer = created!
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let bytes = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<height {
            for x in 0..<width {
                let pixel = bytes + y * rowBytes + x * 4
                pixel[0] = UInt8(x % 256)  // B
                pixel[1] = UInt8(y % 256)  // G
                pixel[2] = 0  // R
                pixel[3] = 255  // A
            }
        }
        return buffer
    }

    /// One pose of 33 landmarks: every landmark at `(x, y)` with the same
    /// visibility — enough to tell the passes apart (coordinates) and to
    /// feed the score (visibility).
    private static func pose(x: Double, y: Double, visibility: Double?) -> [MediaPipeLandmark] {
        (0..<33).map { _ in
            MediaPipeLandmark(x: x, y: y, z: 0, visibility: visibility, presence: visibility)
        }
    }

    /// The two scripted landmarkers through the extractor's injected factory.
    private func run(
        upright: [[MediaPipeLandmark]?],
        rotated: [[MediaPipeLandmark]?],
        frames: FakeFrames,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> MediaPipeClipExtractor.Report {
        let up = ScriptedLandmarker(upright)
        let ro = ScriptedLandmarker(rotated)
        return try await MediaPipeClipExtractor.run(
            source: frames,
            makeLandmarkers: {
                MediaPipeClipExtractor.Landmarkers(upright: up, rotated: ro)
            },
            progress: progress
        )
    }

    // MARK: - The chosen orientation and the chosen pass's keypoints

    func testTheUprightPassWinsWhenItIsTheSurerOne() async throws {
        // Upright surer the whole clip -> every frame keeps the upright pass.
        let report = try await run(
            upright: (0..<5).map { _ in Self.pose(x: 0.25, y: 0.5, visibility: 0.9) },
            rotated: (0..<5).map { _ in Self.pose(x: 0.25, y: 0.5, visibility: 0.6) },
            frames: FakeFrames([0, 100, 200, 300, 400])
        )
        XCTAssertEqual(report.rotated, [false, false, false, false, false])
        XCTAssertEqual(report.frames.count, 5)
        for frame in report.frames {
            XCTAssertTrue(frame.detected)
            XCTAssertFalse(frame.trainerContact)
            // Upright pass: normalised × (size − 1), no map-back (64×48).
            let wrist = try XCTUnwrap(frame.joints[.leftWrist])
            XCTAssertEqual(wrist.x, 0.25 * 63, accuracy: 1e-9)
            XCTAssertEqual(wrist.y, 0.5 * 47, accuracy: 1e-9)
            XCTAssertEqual(wrist.visibility, 0.9, accuracy: 1e-12)
        }
        // The 15-joint schema, mapped from the 33-landmark pose.
        XCTAssertEqual(report.frames[0].joints.count, 15)
    }

    func testTheRotatedPassWinsWhenItIsTheSurerOneAndIsMappedBack() async throws {
        let report = try await run(
            upright: (0..<5).map { _ in Self.pose(x: 0.25, y: 0.5, visibility: 0.4) },
            rotated: (0..<5).map { _ in Self.pose(x: 0.25, y: 0.5, visibility: 0.95) },
            frames: FakeFrames([0, 100, 200, 300, 400])
        )
        XCTAssertEqual(report.rotated, [true, true, true, true, true])
        for frame in report.frames {
            XCTAssertTrue(frame.detected)
            let wrist = try XCTUnwrap(frame.joints[.leftWrist])
            // The rotated pass saw (0.25, 0.5) of the *turned* frame; the
            // map-back is `size − 1 − x·(size − 1)`:
            XCTAssertEqual(wrist.x, 63 - 0.25 * 63, accuracy: 1e-9)
            XCTAssertEqual(wrist.y, 47 - 0.5 * 47, accuracy: 1e-9)
            XCTAssertEqual(wrist.visibility, 0.95, accuracy: 1e-12)
        }
    }

    func testASustainedSwitchChangesTheChoiceWhereTheHysteresisSays() async throws {
        // 20 frames @100 ms: the upright pass is surer for the first second,
        // the rotated pass after that. The smoothed margin (±250 ms window)
        // first clears the 0.05 band at frame 10 (+0.06), and 300 ms later —
        // frame 13 — the hold is done: the switch is the chooser's, frame
        // for frame, the same maths OrientationChooserTests pins to Python.
        let times = (0..<20).map { $0 * 100 }
        let upright = (0..<20).map { $0 < 10 ? 0.9 : 0.6 }
        let rotated = (0..<20).map { $0 < 10 ? 0.6 : 0.9 }
        let report = try await run(
            upright: upright.map { Self.pose(x: 0.1, y: 0.5, visibility: $0) },
            rotated: rotated.map { Self.pose(x: 0.1, y: 0.5, visibility: $0) },
            frames: FakeFrames(times)
        )
        XCTAssertEqual(
            report.rotated,
            [Bool](repeating: false, count: 13) + [Bool](repeating: true, count: 7),
            "the switch is the chooser's, frame for frame")
        // …and the *keypoints* follow the choice: upright coordinates before
        // the switch, the mapped-back ones after.
        let uprightX = 0.1 * 63
        let rotatedX = 63 - 0.1 * 63
        for (index, frame) in report.frames.enumerated() {
            let x = try XCTUnwrap(frame.joints[.leftWrist]).x
            XCTAssertEqual(
                x, index < 13 ? uprightX : rotatedX, accuracy: 1e-9, "frame \(index)")
        }
    }

    func testAFrameNobodyWasFoundInIsNotDetectedAndMovesNothing() async throws {
        // Both passes answer "nobody" for every frame: no margin anywhere,
        // the clip stays in the opening orientation (upright), and every
        // frame is `detected == false` with no joints — Python writes the
        // same empty block.
        let report = try await run(
            upright: (0..<5).map { _ in nil as [MediaPipeLandmark]? },
            rotated: (0..<5).map { _ in nil as [MediaPipeLandmark]? },
            frames: FakeFrames([0, 100, 200, 300, 400])
        )
        XCTAssertEqual(report.rotated, [false, false, false, false, false])
        for frame in report.frames {
            XCTAssertFalse(frame.detected)
            XCTAssertTrue(frame.joints.isEmpty)
        }
    }

    func testOnePassMissingOneFrameKeepsTheOrientation() async throws {
        // The upright pass loses frame 2 — a frame with no margin is
        // ignorable: the clip opens upright and never switches, and only
        // frame 2 of the *chosen* (upright) pass reads as undetected.
        var upright: [[MediaPipeLandmark]?] = (0..<5).map { _ in
            Self.pose(x: 0.1, y: 0.5, visibility: 0.9)
        }
        upright[2] = nil
        let report = try await run(
            upright: upright,
            rotated: (0..<5).map { _ in Self.pose(x: 0.1, y: 0.5, visibility: 0.6) },
            frames: FakeFrames([0, 100, 200, 300, 400])
        )
        XCTAssertEqual(report.rotated, [false, false, false, false, false])
        XCTAssertFalse(report.frames[2].detected, "the chosen pass found nobody there")
        XCTAssertTrue(report.frames[0].detected)
    }

    // MARK: - The rotated pass really gets turned frames

    func testTheRotatedPassReceivesFramesTurned180Exactly() async throws {
        let upright = ScriptedLandmarker(
            (0..<3).map { _ in Self.pose(x: 0.1, y: 0.1, visibility: 0.9) })
        let rotated = ScriptedLandmarker(
            (0..<3).map { _ in Self.pose(x: 0.1, y: 0.1, visibility: 0.6) })
        _ = try await MediaPipeClipExtractor.run(
            source: FakeFrames([0, 100, 200]),
            makeLandmarkers: { .init(upright: upright, rotated: rotated) }
        )
        // The upright pass sees the pixel at (0, 0): B = 0, G = 0.
        XCTAssertEqual(upright.received.first?.topLeftB, 0)
        XCTAssertEqual(upright.received.first?.topLeftG, 0)
        // The rotated pass sees what was (63, 47) at its (0, 0) — a pure
        // pixel permutation, exactly `width-1-x, height-1-y`.
        XCTAssertEqual(rotated.received.first?.topLeftB, 63)
        XCTAssertEqual(rotated.received.first?.topLeftG, 47)
        // Both passes saw every frame, in order.
        XCTAssertEqual(upright.received.map(\.tMs), [0, 100, 200])
        XCTAssertEqual(rotated.received.map(\.tMs), [0, 100, 200])
    }

    // MARK: - Progress

    func testProgressCrossesBothPasses() async throws {
        let reported = ProgressLog()
        _ = try await run(
            upright: (0..<4).map { _ in Self.pose(x: 0.1, y: 0.5, visibility: 0.9) },
            rotated: (0..<4).map { _ in Self.pose(x: 0.1, y: 0.5, visibility: 0.6) },
            frames: FakeFrames([0, 100, 200, 300]),
            progress: { reported.append($0) }
        )
        let values = reported.values
        XCTAssertEqual(values.first ?? .nan, 0, "the run announces itself before frame one")
        XCTAssertEqual(values.last ?? .nan, 1, "…and finishes at 1, after both passes")
        XCTAssertTrue(
            values.contains { $0 > 0 && $0 < 0.25 },
            "the first pass reports inside 0…0.5: \(values)")
        XCTAssertTrue(
            values.contains { $0 > 0.5 && $0 < 1 },
            "the second pass reports inside 0.5…1: \(values)")
        XCTAssertTrue(values.contains(0.5), "the milestone between the passes: \(values)")
        XCTAssertEqual(values, values.sorted(), "progress never goes backwards")
    }

    func testAnUnestimableClipStillAnnouncesStartMiddleAndEnd() async throws {
        let reported = ProgressLog()
        _ = try await run(
            upright: (0..<3).map { _ in Self.pose(x: 0.1, y: 0.5, visibility: 0.9) },
            rotated: (0..<3).map { _ in Self.pose(x: 0.1, y: 0.5, visibility: 0.6) },
            frames: FakeFrames([0, 100, 200], estimates: false),
            progress: { reported.append($0) }
        )
        XCTAssertEqual(reported.values, [0, 0.5, 1], "only the milestones, as documented")
    }

    // MARK: - Refusals

    func testAPoseThatIsNotThirtyThreeLandmarksIsRefused() async throws {
        let short = (0..<12).map { _ in MediaPipeLandmark(x: 0, y: 0, z: 0, visibility: 1) }
        do {
            _ = try await run(
                upright: [short],
                rotated: [nil],
                frames: FakeFrames([0])
            )
            XCTFail("expected wrongLandmarkCount")
        } catch let error as MediaPipeExtractorError {
            XCTAssertEqual(error, .wrongLandmarkCount(12))
        }
    }

    func testASecondPassThatSawDifferentFramesIsRefused() async throws {
        do {
            _ = try await run(
                upright: (0..<3).map { _ in Self.pose(x: 0.1, y: 0.5, visibility: 0.9) },
                rotated: (0..<3).map { _ in Self.pose(x: 0.1, y: 0.5, visibility: 0.6) },
                frames: FakeFrames([0, 100, 200], secondPassTMs: [0, 100, 999])
            )
            XCTFail("expected passMismatch")
        } catch let error as MediaPipeExtractorError {
            guard case .passMismatch = error else {
                return XCTFail("expected passMismatch, got \(error)")
            }
        }
    }

    // MARK: - The score

    func testTheScoreIsTheMeanVisibilityOfTheTwelveMainJointsOnly() {
        // Every landmark scored 0.8 except the main joints' 0.5: the mean
        // must come out of the 12 main joints alone.
        var pose = (0..<33).map { _ in
            MediaPipeLandmark(x: 0, y: 0, z: 0, visibility: 0.8, presence: 0.8)
        }
        for index in OrientationChooser.mainJointIndices {
            pose[index] = MediaPipeLandmark(x: 0, y: 0, z: 0, visibility: 0.5, presence: 0.5)
        }
        XCTAssertEqual(MediaPipeClipExtractor.score(of: pose), 0.5, accuracy: 1e-12)

        // Scores that do not exist are skipped, not counted as zero…
        pose[OrientationChooser.mainJointIndices[0]] = MediaPipeLandmark(
            x: 0, y: 0, z: 0, visibility: nil, presence: nil)
        XCTAssertEqual(MediaPipeClipExtractor.score(of: pose), 0.5, accuracy: 1e-12)

        // …and with nobody, or not one score, the frame has no score at all.
        XCTAssertTrue(MediaPipeClipExtractor.score(of: nil).isNaN)
        let unscored = (0..<33).map { _ in MediaPipeLandmark(x: 0, y: 0, z: 0) }
        XCTAssertTrue(MediaPipeClipExtractor.score(of: unscored).isNaN)
    }

    // MARK: - The PoseService face of the backend

    func testTheServiceSaysItIsClipLevelRatherThanGuessingPerFrame() throws {
        let service = MediaPipePoseService(modelURL: URL(fileURLWithPath: "/tmp/model.task"))
        XCTAssertEqual(service.backendName, "mediapipe")
        service.reset()  // nothing to reset — must not trap
        XCTAssertThrowsError(
            try service.process(Self.makeFrame(), tMs: 0)
        ) { error in
            guard let error = error as? PoseServiceError,
                case .clipLevelOnly = error
            else {
                return XCTFail("expected clipLevelOnly, got \(error)")
            }
            XCTAssertTrue(
                error.description.contains("whole clip"),
                "the refusal says why and what to run instead: \(error.description)")
        }
    }
}
