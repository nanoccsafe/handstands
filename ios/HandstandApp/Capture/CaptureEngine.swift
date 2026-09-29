import AVFoundation
import CoreMedia
import Foundation
import HandstandCore

/// The AVFoundation half of the Record screen: the session, the explicit
/// format choice, the sample buffers, the writer, and the scheduling of the
/// framing checks.
///
/// Everything runs on `queue` (which is also the session's sample-buffer
/// queue) or on `analysisQueue` (Vision) — never the main actor. The main
/// actor only ever sees this object through the three `@Sendable` callbacks,
/// which hop themselves; that is what makes the `@unchecked Sendable` below
/// honest: every stored property is confined to `queue` except
/// `lastAnalysisTime`, which is confined to `analysisQueue`.
///
/// The same buffers do both jobs: each frame is appended to the file (when
/// recording) and, at most ``analysisInterval`` apart, handed to the framing
/// guide — which is why this is a data output rather than
/// `AVCaptureMovieFileOutput`, whose frames nothing else can see.
final class CaptureEngine: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    /// The capture session the preview layer displays and the frames come
    /// from. Configured on `queue`, read by the preview on the main actor —
    /// handing an unstarted session to a preview layer and starting it
    /// afterwards is the sequence Apple's own sample uses.
    let session = AVCaptureSession()

    /// At most five framing checks per second: a check is two Vision passes
    /// on a downscaled frame, and the border does not need to flicker faster
    /// than a person can read it.
    private let analysisInterval: TimeInterval = 0.2

    private let queue = DispatchQueue(label: "handstand.capture.session")
    private let analysisQueue = DispatchQueue(label: "handstand.capture.framing")
    private let device: AVCaptureDevice
    private let analyzer = FramingAnalyzer()

    private var configured = false
    private var videoOutput: AVCaptureVideoDataOutput?
    private var frameRate = 30
    private var writer: RecordingWriter?
    private var pendingURL: URL?

    /// Analysis-queue only: the presentation time of the last frame the
    /// guide was asked about.
    private var lastAnalysisTime = CMTime.invalid

    private let onFraming: @Sendable (FramingStatus) -> Void
    private let onRecordingFinished: @Sendable (Result<URL, any Error>) -> Void
    private let onFailure: @Sendable (String) -> Void

    init(
        device: AVCaptureDevice,
        onFraming: @escaping @Sendable (FramingStatus) -> Void,
        onRecordingFinished: @escaping @Sendable (Result<URL, any Error>) -> Void,
        onFailure: @escaping @Sendable (String) -> Void
    ) {
        self.device = device
        self.onFraming = onFraming
        self.onRecordingFinished = onRecordingFinished
        self.onFailure = onFailure
        super.init()
    }

    // MARK: - Session lifecycle

    /// Configures the session (once) and starts it. Safe to call from the
    /// main actor; the work happens on `queue` because `startRunning()`
    /// blocks.
    func start() {
        queue.async { [self] in
            guard configure() else { return }
            if !session.isRunning {
                session.startRunning()
            }
        }
    }

    /// Stops the session, finishing any recording in progress first so a
    /// half-made file is either completed or cleaned up, never left open.
    func stop() {
        queue.async { [self] in
            endRecording()
            if session.isRunning {
                session.stopRunning()
            }
        }
    }

    /// Begin a new file: the writer is created from the *first frame's*
    /// buffer, so its pixel size is the size the camera actually delivers
    /// (portrait 1080 × 1920 on the iPhone this app is set up for).
    func startRecording(url: URL) {
        queue.async { [self] in
            pendingURL = url
        }
    }

    /// Stop and finish the current file. The result arrives through
    /// `onRecordingFinished`.
    func stopRecording() {
        queue.async { [self] in
            endRecording()
        }
    }

    // MARK: - Configuration

    /// One pass over the session: preset, input, an explicit format and
    /// frame duration, the data output and its portrait connection.
    /// Returns `false` (and reports through `onFailure`) when something
    /// essential is missing.
    private func configure() -> Bool {
        guard !configured else { return true }
        configured = true

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // 1920 × 1080 at 60 fps when the sensor has it, 30 otherwise —
        // chosen by looking at the device's formats rather than hoping a
        // preset lands on one. `.inputPriority` is the preset that says
        // "the session does not manage the format; the client does", which
        // is what makes setting `activeFormat` ourselves the supported thing
        // to do rather than a fight with the session.
        let formats1080 = device.formats.filter { format in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return dimensions.width == 1920 && dimensions.height == 1080
        }
        var chosenFormat: AVCaptureDevice.Format?
        if let best = Self.bestFormat(in: formats1080), session.canSetSessionPreset(.inputPriority) {
            session.sessionPreset = .inputPriority
            chosenFormat = best
        } else if session.canSetSessionPreset(.hd1920x1080) {
            session.sessionPreset = .hd1920x1080
        }

        guard let input = try? AVCaptureDeviceInput(device: device) else {
            onFailure("The camera could not be opened.")
            return false
        }
        guard session.canAddInput(input) else {
            onFailure("The camera cannot be used for recording.")
            return false
        }
        session.addInput(input)

        do {
            try device.lockForConfiguration()
        } catch {
            onFailure("The camera could not be configured: \(error.localizedDescription)")
            return false
        }
        defer { device.unlockForConfiguration() }

        let format = chosenFormat ?? device.activeFormat
        if let chosenFormat {
            device.activeFormat = chosenFormat
        }
        // A frame duration outside the format's supported ranges is an
        // Objective-C exception the app would not survive, so the rate is
        // clamped against those ranges first — 60 when supported, 30 if not.
        let rate = Self.frameRate(forTarget: 60, in: format)
        let duration = CMTime(value: 1, timescale: CMTimeScale(rate))
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        frameRate = rate

        let output = AVCaptureVideoDataOutput()
        // The framing check runs on its own queue, so the delegate queue
        // only appends to the file — fast enough that a frame which arrives
        // while the previous one is still being handled is a bug, not a
        // backlog, and is dropped rather than buffered.
        output.alwaysDiscardsLateVideoFrames = true
        // 420f (YCbCr) rather than BGRA: a third of the bytes from sensor to
        // encoder, and Vision, Core Image and AVAssetWriter all read it.
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            onFailure("The camera's frames could not be captured.")
            return false
        }
        session.addOutput(output)
        videoOutput = output

        // Portrait: the buffers the guide reads and the file records are
        // turned upright here, so the recording is 1080 × 1920 and the
        // normalised coordinates the guide reasons in are the ones the user
        // sees. Falls back to unrotated (landscape) buffers on a device
        // whose data-output connection cannot rotate.
        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        return true
    }

    /// The format worth using among the 1920 × 1080 ones: one that can do
    /// 60 fps first (the rate this app records at), then one that can do 30.
    /// Never a slo-mo format whose *lowest* rate is above 60 — that would
    /// record an attempt that plays back in slow motion — and never the
    /// highest-ceiling format for its own sake.
    static func bestFormat(in formats: [AVCaptureDevice.Format]) -> AVCaptureDevice.Format? {
        for rate in [60, 30] {
            if let format = formats.first(where: { supports(rate, in: $0) }) {
                return format
            }
        }
        return formats.first
    }

    /// Whether some frame-rate range of `format` contains `rate`.
    static func supports(_ rate: Int, in format: AVCaptureDevice.Format) -> Bool {
        format.videoSupportedFrameRateRanges.contains {
            Double(rate) >= $0.minFrameRate && Double(rate) <= $0.maxFrameRate
        }
    }

    /// The frame rate to actually set: `target` when the format supports it,
    /// else 30 (the fallback the task asks for), else the lowest ceiling the
    /// format offers so the duration is still legal.
    static func frameRate(forTarget target: Int, in format: AVCaptureDevice.Format) -> Int {
        if supports(target, in: format) { return target }
        if target > 30, supports(30, in: format) { return 30 }
        // A frame duration outside every supported range is an
        // Objective-C exception the app would not survive, so this is the
        // last resort, not the plan.
        let ranges = format.videoSupportedFrameRateRanges
        let ceiling = ranges.map { Int($0.maxFrameRate.rounded()) }.min() ?? 30
        return max(1, ceiling)
    }

    // MARK: - Sample buffers

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // 1. Into the file, if one is open. This is the whole job of the
        //    delegate queue: it must never wait for Vision.
        if pendingURL != nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            makeWriter(from: pixelBuffer)
        }
        writer?.append(sampleBuffer)

        // 2. To the framing guide, at most five times a second. The hand-off
        //    keeps the buffer alive; the throttle itself lives on the
        //    analysis queue, where the timestamps it compares are read in
        //    order.
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let frame = FramingFrame(
            buffer: pixelBuffer,
            time: CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        )
        analysisQueue.async { [weak self] in
            self?.analyse(frame)
        }
    }

    /// Analysis-queue only: run the check if this frame is far enough past
    /// the last one that was checked.
    private func analyse(_ frame: FramingFrame) {
        if lastAnalysisTime.isValid,
           CMTimeGetSeconds(CMTimeSubtract(frame.time, lastAnalysisTime)) < analysisInterval {
            return
        }
        lastAnalysisTime = frame.time
        guard let status = analyzer.analyse(frame.buffer) else { return }
        onFraming(status)
    }

    // MARK: - Recording

    private func makeWriter(from pixelBuffer: CVPixelBuffer) {
        guard let url = pendingURL else { return }
        pendingURL = nil
        do {
            writer = try RecordingWriter(
                url: url,
                width: CVPixelBufferGetWidth(pixelBuffer),
                height: CVPixelBufferGetHeight(pixelBuffer),
                frameRate: frameRate
            )
        } catch {
            writer = nil
            onRecordingFinished(.failure(error))
        }
    }

    private func endRecording() {
        guard let writer else {
            // Stop arrived before the first frame did: there is no file to
            // keep, and pretending otherwise would show a "done" screen for
            // something that does not exist.
            if let url = pendingURL {
                pendingURL = nil
                try? FileManager.default.removeItem(at: url)
                onRecordingFinished(.failure(RecordingWriter.WriterError.noFrames))
            }
            return
        }
        pendingURL = nil
        self.writer = nil
        writer.finish { [onRecordingFinished] result in
            onRecordingFinished(result)
        }
    }
}

/// One frame's hand-off to the analysis queue: the buffer (immutable once
/// the camera has delivered it, and read only by the analysis queue from
/// here on) and the timestamp the throttle compares.
private struct FramingFrame: @unchecked Sendable {
    let buffer: CVPixelBuffer
    let time: CMTime
}
