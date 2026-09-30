// `VideoFrameSource` / `VideoPoseExtractor` — the app's read of a video
// (chainlink #47), checked against a clip written by the test itself.
//
// Every test writes a small synthetic movie with `AVAssetWriter`: 60 solid
// black frames at 60 fps, 64×128, no person in any of them (the repo is
// public — no real footage and no real keypoints are ever read here). What is
// under test is the *reading*: how many frames come out, their `tMs`, the
// display size of a rotated track, and how `VideoPoseExtractor` drives a
// `PoseService` over them.

import AVFoundation
import CoreVideo
import XCTest

import HandstandCore
import VisionPoseKit

final class VideoFrameSourceTests: XCTestCase {
    // MARK: - The clip every test reads

    private static let frameCount = 60
    private static let fps = 60

    /// Why a test clip could not be written — a test-environment failure,
    /// never a path the sources take.
    private enum WriterError: Error {
        case cannotAddInput
        case failed(String)
    }

    /// 60 black frames at 60 fps, 64×128, in a fresh temp directory. The
    /// caller removes the directory when the test is done.
    private func writeVideo(transform: CGAffineTransform? = nil) async throws -> (url: URL, directory: URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("VideoFrameSourceTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("clip.mov")
        do {
            try await writeFrames(
                to: url,
                count: Self.frameCount,
                fps: Self.fps,
                width: 64,
                height: 128,
                transform: transform
            )
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
        return (url, directory)
    }

    /// The `AVAssetWriter` half: one solid-colour buffer per frame, at
    /// `1/fps` second apart. `transform` is set on the input, the way a
    /// sideways phone clip carries its `preferredTransform`.
    private func writeFrames(
        to url: URL,
        count: Int,
        fps: Int,
        width: Int,
        height: Int,
        transform: CGAffineTransform?
    ) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
            ]
        )
        input.expectsMediaDataInRealTime = false
        if let transform {
            input.transform = transform
        }
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )
        guard writer.canAdd(input) else { throw WriterError.cannotAddInput }
        writer.add(input)
        guard writer.startWriting() else {
            throw WriterError.failed(writer.error?.localizedDescription ?? "startWriting failed")
        }
        writer.startSession(atSourceTime: .zero)

        for index in 0..<count {
            // The writer drains its queue asynchronously; wait for room, and
            // give up (with its reason) if it failed instead of spinning.
            while !input.isReadyForMoreMediaData {
                if writer.status == .failed {
                    throw WriterError.failed(
                        writer.error?.localizedDescription ?? "failed between frames")
                }
                try await Task.sleep(nanoseconds: 1_000_000)
            }
            // Black: a solid colour with no person in it.
            let buffer = try makePixelBuffer(width: width, height: height)
            let time = CMTime(value: CMTimeValue(index), timescale: CMTimeScale(fps))
            guard adaptor.append(buffer, withPresentationTime: time) else {
                throw WriterError.failed(writer.error?.localizedDescription ?? "append failed")
            }
        }
        input.markAsFinished()
        await withCheckedContinuation { continuation in
            writer.finishWriting {
                continuation.resume()
            }
        }
        if writer.status == .failed {
            throw WriterError.failed(writer.error?.localizedDescription ?? "writing failed")
        }
    }

    /// Every kept frame of `source`, in order.
    private func collect(_ source: VideoFrameSource) async throws -> [(buffer: CVPixelBuffer, tMs: Int)] {
        var frames: [(buffer: CVPixelBuffer, tMs: Int)] = []
        for try await frame in source.frames() {
            frames.append(frame)
        }
        return frames
    }

    /// The `maxFps` keep rule, spelled again in the test rather than shared
    /// with the code: a frame is kept when it is at least
    /// `1000/maxFps - 0.5` ms after the last kept one, the first always is.
    private func kept(tMs: [Int], maxFps: Double) -> [Int] {
        guard maxFps > 0 else { return tMs }
        var out: [Int] = []
        for stamp in tMs {
            if let last = out.last, Double(stamp) < Double(last) + (1000.0 / maxFps - 0.5) {
                continue
            }
            out.append(stamp)
        }
        return out
    }

    // MARK: - Reading every frame

    func testMaxFpsZeroKeepsEveryFrameInIncreasingTimeOrder() async throws {
        let (url, directory) = try await writeVideo()
        defer { try? FileManager.default.removeItem(at: directory) }

        let frames = try await collect(VideoFrameSource(url: url, maxFps: 0))
        let stamps = frames.map(\.tMs)

        XCTAssertEqual(frames.count, Self.frameCount, "maxFps 0 keeps every frame")
        for (previous, next) in zip(stamps, stamps.dropFirst()) {
            XCTAssertGreaterThan(next, previous, "presentation time only moves forward")
        }
        // An upright clip is not copied: the frames arrive at their stored
        // size, 64×128.
        let first = frames[0].buffer
        XCTAssertEqual(CVPixelBufferGetWidth(first), 64)
        XCTAssertEqual(CVPixelBufferGetHeight(first), 128)
        // The runner's rounding of 60 fps timestamps: frame 1 is 16.67 ms
        // → 17, frame 2 is 33.33 ms → 33.
        XCTAssertEqual(Array(stamps.prefix(3)), [0, 17, 33])
    }

    // MARK: - The 30 fps cap

    func testMaxFps30KeepsAboutHalfWithIncreasingTimeOrder() async throws {
        let (url, directory) = try await writeVideo()
        defer { try? FileManager.default.removeItem(at: directory) }

        let all = try await collect(VideoFrameSource(url: url, maxFps: 0))
        let capped = try await collect(VideoFrameSource(url: url, maxFps: 30))
        let stamps = capped.map(\.tMs)

        XCTAssertEqual(
            stamps, kept(tMs: all.map(\.tMs), maxFps: 30),
            "the cap keeps exactly the frames the rule keeps")
        XCTAssertGreaterThanOrEqual(capped.count, all.count / 2 - 2, "about half of a 60 fps clip")
        XCTAssertLessThanOrEqual(capped.count, all.count / 2 + 2, "about half of a 60 fps clip")
        for (previous, next) in zip(stamps, stamps.dropFirst()) {
            XCTAssertGreaterThan(next, previous)
            XCTAssertGreaterThanOrEqual(
                Double(next), Double(previous) + 1000.0 / 30 - 0.5,
                "a kept frame is at least one 30 fps interval after the last one")
        }
    }

    // MARK: - A rotated track

    /// A track that stores its frames sideways (a `preferredTransform` with a
    /// quarter turn) comes out in display orientation: the stored 64×128
    /// frames are handed over turned, 128×64 — the size a viewer sees and the
    /// size every keypoint must be in.
    func testAQuarterTurnTrackIsReadInDisplayOrientation() async throws {
        let (url, directory) = try await writeVideo(
            transform: CGAffineTransform(rotationAngle: .pi / 2))
        defer { try? FileManager.default.removeItem(at: directory) }

        // If the writer refused to keep the transform there is no rotation to
        // check — say so rather than asserting the wrong size.
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw XCTSkip("the written clip has no track to read a transform off")
        }
        let preferred = try await track.load(.preferredTransform)
        guard preferred != .identity else {
            throw XCTSkip(
                "AVAssetWriter did not keep the quarter-turn transform; there is no "
                    + "rotated track to check the display size against")
        }

        let frames = try await collect(VideoFrameSource(url: url, maxFps: 0))

        XCTAssertFalse(frames.isEmpty)
        let first = frames[0].buffer
        XCTAssertEqual(
            CVPixelBufferGetWidth(first), 128, "the stored height is the display width")
        XCTAssertEqual(CVPixelBufferGetHeight(first), 64, "the stored width is the display height")
    }

    // MARK: - VideoPoseExtractor

    func testExtractResetsOnceAndProcessesEveryKeptFrame() async throws {
        let (url, directory) = try await writeVideo()
        defer { try? FileManager.default.removeItem(at: directory) }

        let capped = try await collect(VideoFrameSource(url: url, maxFps: 30))
        let service = CountingPoseService()
        let reported = ProgressRecorder()

        let frames = try await VideoPoseExtractor.extract(
            VideoFrameSource(url: url, maxFps: 30),
            service: service
        ) { progress in
            reported.record(progress)
        }

        XCTAssertEqual(service.resetCount, 1, "one clip, one reset")
        XCTAssertEqual(frames.count, capped.count, "one process per kept frame")
        XCTAssertEqual(
            service.processedTMs, capped.map(\.tMs),
            "the service saw every kept frame, with its own timestamps")
        XCTAssertEqual(frames.map(\.tMs), service.processedTMs)
        let progress = reported.values
        XCTAssertEqual(progress.first ?? .nan, 0, "progress starts at 0")
        XCTAssertEqual(progress.last ?? .nan, 1, "progress ends at 1")
        for (previous, next) in zip(progress, progress.dropFirst()) {
            XCTAssertLessThanOrEqual(previous, next, "progress only moves forward")
            XCTAssertGreaterThanOrEqual(previous, 0)
            XCTAssertLessThanOrEqual(next, 1, "progress stays inside 0...1")
        }
    }

    func testCancellationStopsEarly() async throws {
        let (url, directory) = try await writeVideo()
        defer { try? FileManager.default.removeItem(at: directory) }
        let service = CountingPoseService()

        let task = Task {
            try await VideoPoseExtractor.extract(
                VideoFrameSource(url: url, maxFps: 0),
                service: service
            ) { _ in }
        }
        // Cancel the moment the run starts: whatever frame it had reached,
        // it must stop there and say so.
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("a cancelled extract must throw")
        } catch is CancellationError {
            // The expected answer.
        }
        XCTAssertLessThan(
            service.processedTMs.count, Self.frameCount,
            "no inference past the cancellation")
    }

    // MARK: - The progress estimate

    func testEstimatedFrameCountFollowsDurationAndTheCap() async throws {
        let (url, directory) = try await writeVideo()
        defer { try? FileManager.default.removeItem(at: directory) }

        // A one-second clip: 60 fps in full, about 30 frames at the app's cap.
        let every = try await VideoFrameSource(url: url, maxFps: 0).estimatedFrameCount()
        let capped = try await VideoFrameSource(url: url, maxFps: 30).estimatedFrameCount()
        XCTAssertGreaterThanOrEqual(every, Self.frameCount - 2)
        XCTAssertLessThanOrEqual(every, Self.frameCount + 2)
        XCTAssertGreaterThanOrEqual(capped, Self.frameCount / 2 - 2)
        XCTAssertLessThanOrEqual(capped, Self.frameCount / 2 + 2)
    }

    // MARK: - The display size

    /// `displaySize()` is the reader's own measurement — the same track find
    /// and `preferredTransform` → quarter-turn resolution `open()` takes,
    /// factored out so chainlink #50's export can configure its writer
    /// *before* it reads a frame. It must agree with the frames `frames()`
    /// then delivers, upright clip and quarter-turn clip alike.
    func testDisplaySizeAgreesWithTheFramesItWouldRead() async throws {
        let (upright, uprightDirectory) = try await writeVideo()
        defer { try? FileManager.default.removeItem(at: uprightDirectory) }

        let size = try await VideoFrameSource(url: upright, maxFps: 0).displaySize()
        XCTAssertEqual(size, CGSize(width: 64, height: 128), "an upright clip reads as stored")
        let uprightFrames = try await collect(VideoFrameSource(url: upright, maxFps: 0))
        XCTAssertEqual(CVPixelBufferGetWidth(uprightFrames[0].buffer), Int(size.width))
        XCTAssertEqual(CVPixelBufferGetHeight(uprightFrames[0].buffer), Int(size.height))

        let (sideways, sidewaysDirectory) = try await writeVideo(
            transform: CGAffineTransform(rotationAngle: .pi / 2))
        defer { try? FileManager.default.removeItem(at: sidewaysDirectory) }
        let asset = AVURLAsset(url: sideways)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw XCTSkip("the written clip has no track to read a transform off")
        }
        let preferred = try await track.load(.preferredTransform)
        guard preferred != .identity else {
            throw XCTSkip("AVAssetWriter did not keep the quarter-turn transform")
        }

        let display = try await VideoFrameSource(url: sideways, maxFps: 0).displaySize()
        XCTAssertEqual(
            display, CGSize(width: 128, height: 64),
            "the stored height is the display width")
        let sidewaysFrames = try await collect(VideoFrameSource(url: sideways, maxFps: 0))
        XCTAssertEqual(CVPixelBufferGetWidth(sidewaysFrames[0].buffer), Int(display.width))
        XCTAssertEqual(CVPixelBufferGetHeight(sidewaysFrames[0].buffer), Int(display.height))
    }
}

/// A `PoseService` that only counts: no Vision, no model, no image of a
/// person — just how often `reset()` and `process(_:)` were called and with
/// which timestamps.
///
/// `@unchecked Sendable` because the test creates it, hands it to
/// `VideoPoseExtractor` (which runs it on whatever executor the extraction
/// is on) and reads it back afterwards: one task at a time, with the `await`
/// in between, so the mutation is never concurrent.
private final class CountingPoseService: PoseService, @unchecked Sendable {
    private(set) var resetCount = 0
    private(set) var processedTMs: [Int] = []

    var backendName: String { "counting" }

    func reset() {
        resetCount += 1
    }

    func process(_ displayFrame: CVPixelBuffer, tMs: Int) throws -> PostProcessInputFrame {
        processedTMs.append(tMs)
        return PostProcessInputFrame(
            tMs: tMs, detected: false, trainerContact: false, joints: [:])
    }
}

/// The progress values of one extraction, in call order — a class because
/// the `@Sendable` progress closure may not capture a mutable local.
///
/// `@unchecked Sendable` for the same reason as the counting service: the
/// calls all happen inside one extraction task, and the test reads the
/// values after the `await`.
private final class ProgressRecorder: @unchecked Sendable {
    private(set) var values: [Double] = []

    func record(_ value: Double) {
        values.append(value)
    }
}
