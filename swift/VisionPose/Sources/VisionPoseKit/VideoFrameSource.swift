// Reading a video frame by frame — the runner's reading loop, made pullable.
//
// The app (chainlink #47) has to hand the *same* frames to the pipeline that
// the macOS runner (`vision-pose`) hands to its CSV: the same track, the same
// `kCVPixelFormatType_32BGRA` decode, the same `preferredTransform` → display
// orientation conversion (`DisplayFrames`) and the same `tMs` rounding. So the
// decode below copies `RunVisionPose.run`'s reading loop step for step and
// only changes *who decides when the next frame happens*:
//
// * the runner decodes ahead in a `while` loop, because it consumes each
//   buffer before asking for the next one;
// * this side **pulls**: one `next()` from the consumer is one frame decoded.
//   Apple Vision on a phone is slower than the decoder, so pushing frames
//   would either pile the whole clip up in memory (a 10 s 1080p clip is ~2.5
//   GB of pixel buffers) or hand the consumer a buffer the decoder already
//   recycled. Pulling does neither: at most one frame is alive at a time, and
//   the buffer a caller holds is never touched again until it asks.
//
// `frames()` is an `AsyncThrowingStream` because that is the shape both
// callers want: `for try await` over the frames, an error when the video
// cannot be read, `nil`-finished when it ends.

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import HandstandCore
import VisionPoseCore

/// Something that went wrong while opening or decoding a video, in the
/// runner's own words (`RunnerError`'s, minus the parts only a CSV writer
/// has) so a failure on the phone reads like a failure on the Mac.
public enum VideoFrameSourceError: Error, Equatable, CustomStringConvertible {
    /// The file exists but has no video track.
    case noVideoTrack(String)
    /// `AVAssetReader.startReading` refused; the associated value is its
    /// error message.
    case cannotStartReading(String)
    /// The reader failed mid-clip; the associated value is its error message.
    case readFailed(String)
    /// A sample with no pixel buffer in it — the frame cannot be shown or
    /// measured.
    case noPixelBuffer(frame: Int)
    /// The video decoded nothing at all: not a clip, or an unreadable one.
    case noFramesDecoded(String)

    public var description: String {
        switch self {
        case .noVideoTrack(let path):
            return "no video track in \(path)"
        case .cannotStartReading(let message):
            return "could not start reading: \(message)"
        case .readFailed(let message):
            return "reading failed: \(message)"
        case .noPixelBuffer(let frame):
            return "frame \(frame) has no pixel buffer"
        case .noFramesDecoded(let path):
            return "no frames decoded from \(path)"
        }
    }
}

/// One video, read the way the runner reads it: display-orientation frames
/// with their `tMs`, at most `maxFps` of them per second.
///
/// Nothing here runs until `frames()` is iterated (or `estimatedFrameCount()`
/// is asked), so making a source costs nothing — the analysis screen can hold
/// one before the user presses Analyse.
public struct VideoFrameSource: Sendable {
    /// The movie to read.
    public let url: URL
    /// Keep at most this many frames per second of video; `<= 0` keeps every
    /// frame the decoder produces (what the parity check against the runner's
    /// CSV wants).
    public let maxFps: Double

    /// - Parameters:
    ///   - url: the movie file.
    ///   - maxFps: the frame cap; see ``maxFps``.
    public init(url: URL, maxFps: Double = 30) {
        self.url = url
        self.maxFps = maxFps
    }

    /// Frames in display orientation with their `tMs`, keeping at most
    /// `maxFps`: a frame is kept when
    /// `tMs >= lastKeptTMs + 1000/maxFps - 0.5` (the first frame is always
    /// kept; `maxFps <= 0` keeps every frame).
    ///
    /// The timestamps are the presentation timestamps rounded to whole
    /// milliseconds, exactly as `vision-pose` writes them, and the pixels are
    /// exactly what the runner would feed the model — that is what makes
    /// `maxFps 0` a frame-for-frame match with its CSV.
    ///
    /// The stream finishes after the last frame; a video with no frames (or
    /// no video track, or a transform that is not a plain quarter turn) makes
    /// it throw instead. Cancelling the consuming task stops the read at the
    /// next frame.
    public func frames() -> AsyncThrowingStream<(buffer: CVPixelBuffer, tMs: Int), Error> {
        let reader = VideoFrameReader(url: url, maxFps: maxFps)
        return AsyncThrowingStream(unfolding: {
            try await reader.nextFrame()
        })
    }

    /// How many frames ``frames()`` is expected to keep — the clip's
    /// duration × `min(nominalFrameRate, maxFps)`, for the progress bar.
    ///
    /// An estimate, never a promise: a track that reports no frame rate is
    /// read as 30 fps (only the progress bar looks at this), and a clip with
    /// no duration counts as nothing.
    public func estimatedFrameCount() async throws -> Int {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoFrameSourceError.noVideoTrack(url.path)
        }
        let duration = try await asset.load(.duration)
        let nominal = Double(try await track.load(.nominalFrameRate))
        let fps: Double
        if nominal > 0 {
            fps = maxFps > 0 ? Swift.min(nominal, maxFps) : nominal
        } else {
            fps = maxFps > 0 ? maxFps : 30
        }
        let seconds = Swift.max(0, CMTimeGetSeconds(duration))
        guard seconds.isFinite else { return 0 }
        return Int((seconds * fps).rounded())
    }
}

/// One consumer's walk over one video: the runner's reading loop, one frame
/// per ``VideoFrameReader/nextFrame()``.
///
/// `@unchecked Sendable` because the stream's unfolding closure may be
/// `@Sendable`: the state is touched by exactly one consumer, one frame at a
/// time (an `AsyncThrowingStream` has a single iteration), never concurrently.
final class VideoFrameReader: @unchecked Sendable {
    private let url: URL
    private let maxFps: Double

    /// The decoder, once ``open()`` has set it up.
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var displayTransform: DisplayTransform?
    /// `tMs` of the last frame that was kept — the clock the `maxFps` rule
    /// measures from, and (for `maxFps <= 0`) simply "a frame has come out".
    private var lastKeptTMs: Int?
    private var opened = false
    /// How many samples have been decoded, for `noPixelBuffer(frame:)`.
    private var decoded = 0

    init(url: URL, maxFps: Double) {
        self.url = url
        self.maxFps = maxFps
    }

    /// The next kept frame in display orientation, or `nil` when the video
    /// has no more.
    func nextFrame() async throws -> (buffer: CVPixelBuffer, tMs: Int)? {
        try Task.checkCancellation()
        if !opened {
            try await open()
            opened = true
        }
        guard let output, let displayTransform else {
            // Unreachable: `open()` either sets both or throws.
            return nil
        }
        while let sampleBuffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let stored = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                throw VideoFrameSourceError.noPixelBuffer(frame: decoded)
            }
            // The runner's rounding, to the millisecond, from the sample's
            // presentation timestamp.
            let tMs = Int(
                (CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)) * 1000)
                    .rounded()
            )
            decoded += 1
            guard keep(tMs) else { continue }
            // `nil` means the frame is already upright, in which case the
            // decoder's own buffer goes to the consumer — same as the runner.
            let display = try DisplayFrames.displayBuffer(from: stored, transform: displayTransform)
                ?? stored
            return (display, tMs)
        }
        if reader?.status == .failed {
            throw VideoFrameSourceError.readFailed(
                reader?.error?.localizedDescription ?? "unknown")
        }
        guard lastKeptTMs != nil else {
            // The first frame is always kept, so "nothing kept" is "nothing
            // decoded" — the runner refuses an empty clip the same way.
            throw VideoFrameSourceError.noFramesDecoded(url.path)
        }
        return nil
    }

    /// The `maxFps` rule: keep a frame when it is at least
    /// `1000/maxFps - 0.5` ms after the last one that was kept; the first
    /// frame is always kept, and `maxFps <= 0` keeps everything.
    private func keep(_ tMs: Int) -> Bool {
        if maxFps > 0, let last = lastKeptTMs,
            Double(tMs) < Double(last) + (1000.0 / maxFps - 0.5)
        {
            return false
        }
        lastKeptTMs = tMs
        return true
    }

    /// Everything the runner does before its `while` loop: find the track,
    /// resolve its `preferredTransform` to a quarter turn, and start an
    /// `AVAssetReader` that decodes 32BGRA.
    private func open() async throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CocoaError(.fileNoSuchFile)
        }
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoFrameSourceError.noVideoTrack(url.path)
        }
        let naturalSize = try await track.load(.naturalSize)
        let preferred = try await track.load(.preferredTransform)
        let storedSize = PixelSize(
            width: Int(naturalSize.width.rounded()),
            height: Int(naturalSize.height.rounded())
        )
        // Throws rather than guessing when the container's transform is not a
        // plain quarter turn: sideways keypoints would be silent.
        let displayTransform = try DisplayTransform.quarterTurns(
            a: preferred.a, b: preferred.b, c: preferred.c, d: preferred.d,
            storedSize: storedSize
        )

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        )
        // The runner leaves this `false` and finishes with each buffer before
        // decoding the next one. Here the buffer crosses to the consumer,
        // which may still hold it while the next decode runs — so this side
        // pays for the copy that keeps its memory its own. Same pixels, same
        // keypoints, one memcpy per frame.
        output.alwaysCopiesSampleData = true
        reader.add(output)
        guard reader.startReading() else {
            throw VideoFrameSourceError.cannotStartReading(
                reader.error?.localizedDescription ?? "unknown")
        }
        self.reader = reader
        self.output = output
        self.displayTransform = displayTransform
    }
}

/// The app's half of the runner's loop: every kept frame of `source` through
/// `service`, with progress — the call chain `AnalysisService` (chainlink
/// #47) runs off the main actor.
public enum VideoPoseExtractor {
    /// Runs `service` (after `service.reset()`) over every kept frame, in
    /// time order, and returns the frames of the shared schema.
    ///
    /// - Parameters:
    ///   - source: the video to read.
    ///   - service: the pose backend, reset first — one clip, one
    ///     auto-rotation state, exactly like the runner's one service per
    ///     clip.
    ///   - progress: called from the background with 0…1: `0` before the
    ///     first frame, the frames seen against the estimated total while
    ///     they go, `1` when the video is done. A clip whose length cannot be
    ///     estimated only gets the two ends.
    /// - Throws: ``VideoFrameSourceError`` (or the asset's own error) when
    ///   the video cannot be read, ``PoseServiceError`` when the model fails
    ///   on a frame, and `CancellationError` at the first frame once the
    ///   task is cancelled — nothing is ever inferred past that point.
    public static func extract(
        _ source: VideoFrameSource,
        service: PoseService,
        progress: @Sendable (Double) -> Void
    ) async throws -> [PostProcessInputFrame] {
        service.reset()
        progress(0)
        // The estimate only feeds the progress bar; a video it cannot open
        // makes `frames()` throw with the real reason a moment later.
        let estimated = (try? await source.estimatedFrameCount()) ?? 0

        var frames: [PostProcessInputFrame] = []
        var seen = 0
        for try await frame in source.frames() {
            try Task.checkCancellation()
            frames.append(try service.process(frame.buffer, tMs: frame.tMs))
            seen += 1
            if estimated > 0 {
                progress(Swift.min(1.0, Double(seen) / Double(estimated)))
            }
        }
        // The stream can finish silently when the consumer was cancelled
        // between frames; the caller asked to stop, so it must stop.
        try Task.checkCancellation()
        progress(1.0)
        return frames
    }
}
