import AVFoundation
import CoreGraphics
import CoreText
import CoreVideo
import Foundation
import HandstandCore
import VisionPoseKit

// --------------------------------------------------------------------------- #
// The annotated video (chainlink #50): the source movie with the stress
// diagram drawn on every frame and the heat strip along the bottom, written
// as an H.264 `.mp4` the user shares from the phone — and then it stays on
// the phone: nothing here uploads anywhere.
//
// The shape of the run is `AVAssetReader → CoreGraphics → AVAssetWriter`,
// deliberately **not** `AVVideoCompositionCoreAnimationTool`: the skeleton
// changes every frame, and drawing each frame directly is deterministic and
// testable on macOS (the tests in `AnnotatedVideoTests` do exactly that).
//
// Reading is `VideoFrameSource` with `maxFps 0` — every source frame, in
// display orientation, with the clip's own millisecond `tMs` — so the export
// sees exactly the frames the analysis saw (chainlink #47's steps), writes
// every one of them (the output's frame timing is the source's), and picks
// the diagram for a frame with `StressDiagram.frameIndex(atMs:in:)`: the
// latest analysed frame at or before it.
//
// The output is upright by construction: the reader turns the source's
// `preferredTransform` into display pixels, and the writer's input carries no
// transform at all, so what comes out is the display size with an identity
// transform on the track.
// --------------------------------------------------------------------------- #

/// One export: `source` + `analysis` → an annotated `output` movie.
public struct AnnotatedExport: Sendable {
    /// The movie to annotate.
    public let source: URL
    /// What to draw — the same `Analysis` the overlay on screen draws.
    public let analysis: Analysis
    /// The reference the analysis was severed against, `nil` for the
    /// built-in tolerances (the overlay's own choice).
    public let reference: ScoreReference?
    /// Where the annotated movie is written; overwritten if it exists.
    public let output: URL

    /// The palette and sizes — `.standard`, i.e. the on-screen overlay's.
    private let style = DiagramStyle.standard
    /// The day stamped in the footer; `Date()` for a real export, injectable
    /// so a test's footer is deterministic.
    private let date: Date
    /// The heat strip's bin count — the screen's own (chainlink #49), capped
    /// at the number of analysed frames so a short clip's strip has no bins
    /// nothing ever fell into.
    private static let stripBins = 120

    /// - Parameters:
    ///   - source: the movie to annotate (any orientation; it is read in
    ///     display orientation and written upright).
    ///   - analysis: what to draw over it. An analysis the pipeline could
    ///     not stand behind does **not** fail the export: the video still
    ///     goes out, with no skeleton and a neutral strip.
    ///   - reference: the scoring reference the colours are severed against.
    ///   - output: the `.mp4` to write, replaced if it is already there.
    public init(source: URL, analysis: Analysis, reference: ScoreReference?, output: URL) {
        self.init(
            source: source, analysis: analysis, reference: reference,
            output: output, date: Date())
    }

    init(
        source: URL, analysis: Analysis, reference: ScoreReference?, output: URL, date: Date
    ) {
        self.source = source
        self.analysis = analysis
        self.reference = reference
        self.output = output
        self.date = date
    }

    /// Writes the annotated movie, reporting 0…1 along the way.
    ///
    /// Every source frame is written (none capped), with the source's own
    /// timing, and its audio passed through when the container can carry it
    /// unchanged. The output is **overwritten** if it exists.
    ///
    /// On `Task` cancellation the run stops, the partial file is deleted and
    /// `CancellationError` is thrown — a cancelled export never leaves half a
    /// movie behind. The same cleanup runs for any other failure, so
    /// `output` either does not exist or is a finished, playable movie.
    ///
    /// - Parameter progress: called from the background, 0…1: `0` before the
    ///   first frame, the frames written against the estimate as they go, and
    ///   `1` only when the file is finished.
    public func run(progress: @Sendable (Double) -> Void) async throws {
        progress(0)
        try Task.checkCancellation()
        // Overwrite: a leftover from an earlier export is replaced, never
        // appended to.
        if FileManager.default.fileExists(atPath: output.path) {
            try FileManager.default.removeItem(at: output)
        }
        do {
            try await export(progress: progress)
            try Task.checkCancellation()
        } catch {
            try? FileManager.default.removeItem(at: output)
            if Task.isCancelled {
                throw CancellationError()
            }
            throw error
        }
        progress(1.0)
    }

    // MARK: - The run

    /// Everything `run` does between the two cleanups: set the writer up,
    /// walk the source's frames, draw and append each one, pass the audio
    /// along, finish the file. Any failure throws; `run` deletes the file.
    private func export(progress: @Sendable (Double) -> Void) async throws {
        let asset = AVURLAsset(url: source)
        // The display size is the same measurement the *reader* makes (the
        // source's `preferredTransform` resolved to a quarter turn), taken
        // before any frame is read so the writer can be configured with it.
        let displaySize = try await VideoFrameSource(url: source, maxFps: 0).displaySize()
        let width = Int(displaySize.width.rounded())
        let height = Int(displaySize.height.rounded())
        guard width > 0, height > 0 else {
            throw ExportError.badDisplaySize("\(width)×\(height)")
        }
        let canvas = makeCanvas(width: width, height: height)

        let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
        // Whatever happens below, a writer still holding the file open must
        // let go of it before `run` cleans up (the audio reader too).
        var audio: AudioTap?
        defer {
            if writer.status == .writing {
                writer.cancelWriting()
            }
            if let audio, audio.reader.status == .reading {
                audio.reader.cancelReading()
            }
        }

        let videoInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
            ]
        )
        videoInput.expectsMediaDataInRealTime = false
        // No `transform` on purpose: the frames below are already display
        // pixels, so the output track is upright (identity), whatever the
        // source's `preferredTransform` was.
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )
        guard writer.canAdd(videoInput) else { throw ExportError.cannotAddInput("video") }
        writer.add(videoInput)

        // The source's audio, passed through unchanged when the container
        // can carry it as it is (AAC into `.mp4` — the phone's own format).
        // Anything else, and a track that cannot be read, costs the export
        // its sound rather than the export itself: the video still goes out,
        // which is what a passthrough is for instead of a transcode.
        if let audioTrack = try? await asset.loadTracks(withMediaType: .audio).first {
            audio = Self.audioPassthrough(asset: asset, track: audioTrack, writer: writer)
        }

        guard writer.startWriting() else {
            throw ExportError.writeFailed(
                writer.error?.localizedDescription ?? "startWriting failed")
        }
        writer.startSession(atSourceTime: .zero)

        // 1. Every source frame, at the source's own timestamps.
        let clip = VideoFrameSource(url: source, maxFps: 0)
        let estimated = (try? await clip.estimatedFrameCount()) ?? 0
        var seen = 0
        for try await frame in clip.frames() {
            try Task.checkCancellation()
            let presentation = CMTime(value: CMTimeValue(frame.tMs), timescale: 1000)
            // Audio first, up to this frame's time: both inputs then see
            // their samples in time order, interleaved the way the
            // container wants them.
            try await drainAudio(audio, upTo: presentation)

            let diagram = diagram(atMs: frame.tMs, height: height)
            let buffer = try makeBuffer(adaptor: adaptor, width: width, height: height)
            try render(
                frame.buffer, into: buffer, diagram: diagram,
                playheadMs: frame.tMs, canvas: canvas
            )

            while !videoInput.isReadyForMoreMediaData {
                if writer.status == .failed {
                    throw ExportError.writeFailed(
                        writer.error?.localizedDescription ?? "failed between frames")
                }
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            guard adaptor.append(buffer, withPresentationTime: presentation) else {
                throw ExportError.writeFailed(
                    writer.error?.localizedDescription ?? "append failed")
            }
            seen += 1
            if estimated > 0 {
                progress(Swift.min(1.0, Double(seen) / Double(estimated)))
            }
        }
        try Task.checkCancellation()

        // 2. Whatever audio trails the last frame, then finish both inputs
        //    and the file itself.
        try await drainAudio(audio, upTo: nil)
        audio?.input.markAsFinished()
        videoInput.markAsFinished()
        await withCheckedContinuation { continuation in
            writer.finishWriting {
                continuation.resume()
            }
        }
        if writer.status == .failed {
            throw ExportError.writeFailed(
                writer.error?.localizedDescription ?? "finishWriting failed")
        }
    }

    // MARK: - What goes on each frame

    /// What every frame's drawing needs and none of them changes: the scale,
    /// the strip's rect, the strip's bins and the footer's sentence.
    private struct Canvas {
        var scale: Double
        var stripRect: CGRect
        var bins: [HeatBin]
        var footer: String
    }

    /// The strip along the bottom (3 % of the height, full width), the
    /// diagram's scale (`videoHeight / 1000`) and the footer's text — all
    /// computed once for the whole export, never per frame.
    private func makeCanvas(width: Int, height: Int) -> Canvas {
        let stripHeight = Double(height) * style.stripFraction
        let bins: [HeatBin]
        if usable {
            bins = SessionSummary.heatStrip(
                analysis: analysis, reference: reference,
                bins: Swift.min(Self.stripBins, Swift.max(analysis.features.tMs.count, 1))
            )
        } else {
            bins = []
        }
        return Canvas(
            scale: Double(height) / 1000.0,
            stripRect: CGRect(
                x: 0, y: Double(height) - stripHeight,
                width: Double(width), height: stripHeight
            ),
            bins: bins,
            footer: footerText()
        )
    }

    /// The diagram for the source frame at `tMs`: the latest analysed frame
    /// at or before it (`StressDiagram.frameIndex(atMs:in:)`, the overlay's
    /// own rule), `nil` before the analysis's first frame — and `nil`
    /// throughout when the analysis is not one the pipeline stands behind.
    private func diagram(atMs tMs: Int, height: Int) -> DiagramFrame? {
        guard usable else { return nil }
        guard let index = StressDiagram.frameIndex(atMs: tMs, in: analysis.features.tMs)
        else { return nil }
        // `frameIndex` indexes `features.tMs`, which `usable` has pinned to
        // `processed.frames`, so `frame(_:…)`'s precondition holds.
        return StressDiagram.frame(
            index, analysis: analysis, reference: reference, height: Double(height))
    }

    /// Is `analysis` one the export can draw from: frames to look up, and a
    /// pipeline that stood behind them (`features` and `phases` both usable).
    ///
    /// An unusable analysis is not an error — the export still writes the
    /// movie, with no skeleton and a neutral strip, which is what the screen
    /// shows for a clip it could not measure (chainlink #50).
    private var usable: Bool {
        !analysis.features.tMs.isEmpty
            && analysis.features.tMs.count == analysis.processed.frames.count
            && analysis.features.usable
            && analysis.phases.usable
    }

    /// "Handstand · 2026-09-30", with the clip's score when it has one —
    /// the footer's two facts, drawn bottom-left above the strip.
    private func footerText() -> String {
        var parts = ["Handstand · \(Self.dateString(date))"]
        if let score = analysis.clipScore?.score, score.isFinite {
            parts.append("Score \(String(format: "%.0f", score))")
        }
        return parts.joined(separator: " · ")
    }

    private static func dateString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }

    // MARK: - Drawing one frame

    /// One frame: the source's pixels, the diagram over them, the strip
    /// along the bottom and the footer above it — all in display pixels with
    /// the origin **top-left**, which is the context flipped once, here
    /// (`DiagramRenderer`'s contract).
    private func render(
        _ frame: CVPixelBuffer, into destination: CVPixelBuffer, diagram: DiagramFrame?,
        playheadMs: Int, canvas: Canvas
    ) throws {
        CVPixelBufferLockBaseAddress(frame, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
        CVPixelBufferLockBaseAddress(destination, [])
        defer { CVPixelBufferUnlockBaseAddress(destination, []) }

        let width = CVPixelBufferGetWidth(destination)
        let height = CVPixelBufferGetHeight(destination)
        guard CVPixelBufferGetWidth(frame) == width, CVPixelBufferGetHeight(frame) == height
        else {
            throw ExportError.sizeMismatch(
                source: "\(CVPixelBufferGetWidth(frame))×\(CVPixelBufferGetHeight(frame))",
                output: "\(width)×\(height)")
        }
        guard let from = CVPixelBufferGetBaseAddress(frame),
            let into = CVPixelBufferGetBaseAddress(destination)
        else {
            throw ExportError.pixelBufferUnavailable
        }

        // 1. The picture itself: both buffers are display-sized 32BGRA, so
        //    the copy is one memcpy per row — cheapest thing that leaves the
        //    decoder's buffer untouched.
        let fromRow = CVPixelBufferGetBytesPerRow(frame)
        let intoRow = CVPixelBufferGetBytesPerRow(destination)
        let rowBytes = Swift.min(width * 4, Swift.min(fromRow, intoRow))
        let source = from.assumingMemoryBound(to: UInt8.self)
        let target = into.assumingMemoryBound(to: UInt8.self)
        for row in 0..<height {
            target.advanced(by: row * intoRow)
                .update(from: source.advanced(by: row * fromRow), count: rowBytes)
        }

        // 2. The context: sRGB (where `DiagramStyle`'s components live) in
        //    the buffer's own BGRA layout, then the flip that puts the
        //    origin top-left — display `y` grows downwards, like the view's.
        guard let ctx = CGContext(
            data: into,
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: intoRow,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
                | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else {
            throw ExportError.contextFailed
        }
        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1, y: -1)

        if let diagram {
            DiagramRenderer.draw(diagram, in: ctx, scale: canvas.scale, style: style)
        }
        DiagramRenderer.drawHeatStrip(
            canvas.bins, playheadMs: playheadMs, in: canvas.stripRect, ctx: ctx,
            style: style)
        Self.drawFooter(
            canvas.footer, in: ctx, scale: canvas.scale, style: style,
            above: canvas.stripRect.minY)
    }

    /// The footer line, drawn with CoreText in the bottom-left, its baseline
    /// `footerMargin × scale` above the strip. CoreText's text matrix is
    /// flipped for the y-down context `render` made — the standard recipe
    /// for drawing a line in a flipped space.
    private static func drawFooter(
        _ text: String, in ctx: CGContext, scale: Double, style: DiagramStyle,
        above stripTop: CGFloat
    ) {
        guard !text.isEmpty else { return }
        let size = style.footerFontSize * Swift.max(scale, 0.01)
        let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String):
                DiagramStyle.Colour.white.cgColor,
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: text, attributes: attributes))

        ctx.saveGState()
        defer { ctx.restoreGState() }
        ctx.textMatrix = CGAffineTransform(scaleX: 1, y: -1)
        ctx.textPosition = CGPoint(
            x: style.footerMargin * scale,
            y: stripTop - style.footerMargin * scale)
        CTLineDraw(line, ctx)
    }

    // MARK: - Buffers

    /// A display-sized 32BGRA buffer to draw the next frame into: the
    /// adaptor's pool when it has one (it recycles the encoded frames),
    /// a fresh buffer otherwise.
    private func makeBuffer(
        adaptor: AVAssetWriterInputPixelBufferAdaptor, width: Int, height: Int
    ) throws -> CVPixelBuffer {
        if let pool = adaptor.pixelBufferPool {
            var pooled: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pooled)
            if let pooled { return pooled }
        }
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        var created: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            attributes as CFDictionary, &created)
        guard status == kCVReturnSuccess, let created else {
            throw ExportError.pixelBufferUnavailable
        }
        return created
    }

    // MARK: - Audio

    /// One source audio track set up to be copied straight through: the
    /// reader over the track, the writer input carrying the source's own
    /// format, and the sample that was read to learn that format, held
    /// until its time comes. A class because it holds the reader, the output
    /// and that pending sample together across the whole run.
    ///
    /// `@unchecked Sendable` for the same reason `VideoFrameReader` is: one
    /// tap belongs to exactly one `export` run, which creates it, passes it
    /// down that run's own async calls and never shares it — no two tasks
    /// ever touch it, so the mutable state is safe even though the compiler
    /// cannot see that.
    private final class AudioTap: @unchecked Sendable {
        let reader: AVAssetReader
        let output: AVAssetReaderTrackOutput
        let input: AVAssetWriterInput
        /// The last sample read but not yet appended, when its time is still
        /// ahead of the frame the export is on.
        var pending: CMSampleBuffer?

        init(
            reader: AVAssetReader, output: AVAssetReaderTrackOutput,
            input: AVAssetWriterInput, pending: CMSampleBuffer?
        ) {
            self.reader = reader
            self.output = output
            self.input = input
            self.pending = pending
        }
    }

    /// The `.mp4` passthrough's format: AAC (`'mp4a'`), which is what a
    /// phone records. Anything else — LPCM in a `.mov`, say — answers "no
    /// tap" rather than being transcoded: this export copies the source's
    /// sound, it does not re-encode it.
    private static func isPassthroughFormat(_ format: CMFormatDescription) -> Bool {
        CMFormatDescriptionGetMediaSubType(format) == 0x6D70_3461  // 'mp4a'
    }

    private static func audioPassthrough(
        asset: AVAsset, track: AVAssetTrack, writer: AVAssetWriter
    ) -> AudioTap? {
        guard let reader = try? AVAssetReader(asset: asset) else { return nil }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        guard reader.canAdd(output) else { return nil }
        reader.add(output)
        guard reader.startReading(),
            let first = output.copyNextSampleBuffer(),
            let format = CMSampleBufferGetFormatDescription(first),
            isPassthroughFormat(format)
        else {
            reader.cancelReading()
            return nil
        }
        let input = AVAssetWriterInput(
            mediaType: .audio, outputSettings: nil, sourceFormatHint: format)
        input.expectsMediaDataInRealTime = false
        guard writer.canAdd(input) else {
            reader.cancelReading()
            return nil
        }
        writer.add(input)
        return AudioTap(reader: reader, output: output, input: input, pending: first)
    }

    /// Appends the held and following audio samples whose time has come —
    /// everything at or before `limit` (`nil` drains the track). Called
    /// before each video frame so the two inputs advance interleaved in
    /// time, and once more after the last frame for whatever trails it.
    private func drainAudio(_ tap: AudioTap?, upTo limit: CMTime?) async throws {
        guard let tap else { return }
        while true {
            let sample: CMSampleBuffer
            if let held = tap.pending {
                tap.pending = nil
                sample = held
            } else if let next = tap.output.copyNextSampleBuffer() {
                sample = next
            } else {
                if tap.reader.status == .failed {
                    throw ExportError.writeFailed(
                        tap.reader.error?.localizedDescription ?? "audio read failed")
                }
                break
            }
            if let limit,
                CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sample), limit) > 0
            {
                tap.pending = sample
                break
            }
            while !tap.input.isReadyForMoreMediaData {
                if tap.reader.status == .failed {
                    throw ExportError.writeFailed(
                        tap.reader.error?.localizedDescription ?? "audio read failed")
                }
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            if !tap.input.append(sample) {
                throw ExportError.writeFailed("audio append failed")
            }
        }
    }
}

/// What went wrong around the export — never the drawing or the maths,
/// whose stages refuse their bad input by precondition instead.
public enum ExportError: Error, CustomStringConvertible {
    /// The source has no usable display size (no video track, or a
    /// transform that is not a plain quarter turn — `VideoFrameSource`'s own
    /// errors usually arrive instead).
    case badDisplaySize(String)
    /// The writer refuses an input it cannot encode with.
    case cannotAddInput(String)
    /// A source frame and the output buffer disagree about the frame size.
    case sizeMismatch(source: String, output: String)
    /// A pixel buffer could not be made (or has no address to draw into).
    case pixelBufferUnavailable
    /// The drawing context could not be created over the frame buffer.
    case contextFailed
    /// The writer (or the audio reader) failed, with its own message.
    case writeFailed(String)

    public var description: String {
        switch self {
        case .badDisplaySize(let size):
            return "source has no display size (\(size))"
        case .cannotAddInput(let media):
            return "the writer cannot add the \(media) input"
        case .sizeMismatch(let source, let output):
            return "frame size \(source) does not match the output \(output)"
        case .pixelBufferUnavailable:
            return "no pixel buffer to draw into"
        case .contextFailed:
            return "could not open the frame for drawing"
        case .writeFailed(let message):
            return "writing failed: \(message)"
        }
    }
}
