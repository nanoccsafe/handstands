import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import XCTest

import HandstandCore

// --------------------------------------------------------------------------- #
// What every AnnotatedVideo test reads, writes and draws over (chainlink #50).
//
// The repo is public, so no real footage and no real keypoints are ever near
// these tests: the clips are solid mid-grey frames written with
// `AVAssetWriter` here, and the analysis is built from hand-made
// `PostProcessInputFrame`s the way `StressDiagramTests` builds its own — a
// clean inverted body at fixed pixel positions inside the 180 × 320 display
// frame, held still for the whole clip, which is exactly the hold
// `PhaseSegmenter` looks for.
//
// Mid-grey is the point of the pixels: the source background is
// `(128, 128, 128)`, so anything the export *draws* — a bone, the strip —
// is visibly **not** grey, and anything it must leave alone (a pixel far
// from the skeleton) still is.
// --------------------------------------------------------------------------- #

enum SyntheticClip {
    // MARK: - The clip

    /// The **display** frame: what the analysis is written in, and the size
    /// the export must keep — upright, whatever the source's transform says.
    static let displayWidth = 180
    static let displayHeight = 320
    /// How many frames every test clip has, at `fps` — 1.5 s of holding.
    static let frameCount = 45
    static let fps = 30
    /// The background every source frame is filled with: mid-grey.
    static let background: UInt8 = 128

    /// The clip's frame timestamps — the very rounding `VideoFrameSource`
    /// does (`round(seconds × 1000)`), so the analysis's clock and the
    /// frames the export reads are the same clock, frame for frame.
    static func tMs(frame: Int) -> Int {
        Int((Double(frame) / Double(fps) * 1000.0).rounded())
    }

    /// Why a test clip could not be written — a test-environment failure,
    /// never a path the sources take.
    enum WriterError: Error {
        case cannotAddInput
        case failed(String)
        case noBuffer
    }

    /// A fresh temp directory holding `name`: solid mid-grey frames at
    /// `fps`, `frameCount` of them. `storedWidth`/`storedHeight` are the
    /// **stored** (pre-transform) size; pass a quarter turn with a landscape
    /// size to make the sideways-phone variant, whose display size is the
    /// same 180 × 320. The caller removes the directory when done.
    static func write(
        named name: String = "clip.mov",
        storedWidth: Int? = nil,
        storedHeight: Int? = nil,
        transform: CGAffineTransform? = nil
    ) async throws -> (url: URL, directory: URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AnnotatedVideoTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(name)
        do {
            try await writeFrames(
                to: url,
                width: storedWidth ?? displayWidth,
                height: storedHeight ?? displayHeight,
                transform: transform
            )
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
        return (url, directory)
    }

    /// The `AVAssetWriter` half: one solid mid-grey buffer per frame, at
    /// `1/fps` second apart. `transform` is set on the input, the way a
    /// sideways phone clip carries its `preferredTransform`.
    private static func writeFrames(
        to url: URL, width: Int, height: Int, transform: CGAffineTransform?
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

        for index in 0..<frameCount {
            while !input.isReadyForMoreMediaData {
                if writer.status == .failed {
                    throw WriterError.failed(
                        writer.error?.localizedDescription ?? "failed between frames")
                }
                try await Task.sleep(nanoseconds: 1_000_000)
            }
            let buffer = try makeBuffer(width: width, height: height)
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
            throw WriterError.failed(writer.error?.localizedDescription ?? "finishWriting failed")
        }
    }

    /// One solid mid-grey 32BGRA frame.
    private static func makeBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &buffer)
        guard status == kCVReturnSuccess, let buffer else { throw WriterError.noBuffer }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw WriterError.noBuffer }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        for row in 0..<height {
            for column in 0..<width {
                let at = row * rowBytes + column * 4
                pixels[at] = background
                pixels[at + 1] = background
                pixels[at + 2] = background
                pixels[at + 3] = 255
            }
        }
        return buffer
    }

    // MARK: - Reading a movie back

    /// One decoded frame, copied out of the reader: BGRA bytes and their
    /// geometry, so a test can ask any pixel what colour it is.
    struct Frame {
        var bytes: [UInt8]
        var bytesPerRow: Int
        var width: Int
        var height: Int

        init(copyOf buffer: CVPixelBuffer) {
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            self.width = CVPixelBufferGetWidth(buffer)
            self.height = CVPixelBufferGetHeight(buffer)
            self.bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            let count = bytesPerRow * height
            self.bytes = [UInt8](repeating: 0, count: count)
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                bytes.withUnsafeMutableBytes { raw in
                    raw.copyMemory(from: UnsafeRawBufferPointer(start: base, count: raw.count))
                }
            }
        }

        /// The pixel at `(x, y)` as `(r, g, b)` — display coordinates, the
        /// origin top-left, exactly the space the diagram is drawn in.
        func pixel(x: Int, y: Int) -> (r: Int, g: Int, b: Int) {
            precondition(x >= 0 && x < width && y >= 0 && y < height, "pixel outside the frame")
            let at = y * bytesPerRow + x * 4
            // kCVPixelFormatType_32BGRA: bytes are B, G, R, A.
            return (Int(bytes[at + 2]), Int(bytes[at + 1]), Int(bytes[at]))
        }
    }

    /// Every frame of `url`, decoded as stored — the test reads the *export*
    /// with this (its frames are already display orientation, so the pixels
    /// are straight to ask about) and counts the source's frames with it too.
    static func read(_ url: URL) async throws -> [Frame] {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw WriterError.failed("no video track in \(url.lastPathComponent)")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        )
        output.alwaysCopiesSampleData = true
        reader.add(output)
        guard reader.startReading() else {
            throw WriterError.failed(reader.error?.localizedDescription ?? "startReading failed")
        }
        var frames: [Frame] = []
        while let sample = output.copyNextSampleBuffer() {
            guard let image = CMSampleBufferGetImageBuffer(sample) else { continue }
            frames.append(Frame(copyOf: image))
        }
        if reader.status == .failed {
            throw WriterError.failed(reader.error?.localizedDescription ?? "read failed")
        }
        return frames
    }

    /// The facts a test checks the output against: the **stored** size, the
    /// track's transform and the movie's duration.
    struct Info {
        var naturalSize: CGSize
        var transform: CGAffineTransform
        var duration: CMTime
    }

    static func info(_ url: URL) async throws -> Info {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw WriterError.failed("no video track in \(url.lastPathComponent)")
        }
        return Info(
            naturalSize: try await track.load(.naturalSize),
            transform: try await track.load(.preferredTransform),
            duration: try await asset.load(.duration)
        )
    }

    // MARK: - The analysis

    /// One frame of a handstand held perfectly still, in the 180 × 320
    /// display frame: y grows downwards, the hands are at y = 300 and the
    /// feet at y = 40 — ankles well above the wrists (`invertedMin`), hips
    /// above the hands (`handsLowMinV`), the whole body on one vertical line
    /// (`bodyAngle` 0°) and nothing moving. Every joint of the 15-joint
    /// schema is present at visibility 0.9, so every frame is measurable.
    ///
    /// The same body `AnalysisSummaryTests` builds its clip from — a static
    /// inverted body is exactly the hold `PhaseSegmenter` looks for.
    static func holdingFrame(tMs: Int) -> PostProcessInputFrame {
        func point(_ y: Double, x: Double = 96) -> Keypoint {
            Keypoint(x: x, y: y, visibility: 0.9)
        }
        let joints: [Joint: Keypoint] = [
            // The nose sits off the shoulder→hip line, so the facing sign
            // has a side to read (x = 106 against the body's x = 96).
            .nose: Keypoint(x: 106, y: 195, visibility: 0.9),
            .leftShoulder: point(210, x: 92),
            .rightShoulder: point(210, x: 100),
            .leftElbow: point(250, x: 92),
            .rightElbow: point(250, x: 100),
            .leftWrist: point(300, x: 92),
            .rightWrist: point(300, x: 100),
            .leftHip: point(150, x: 92),
            .rightHip: point(150, x: 100),
            .leftKnee: point(100, x: 92),
            .rightKnee: point(100, x: 100),
            .leftAnkle: point(55, x: 92),
            .rightAnkle: point(55, x: 100),
            .leftFootIndex: point(40, x: 92),
            .rightFootIndex: point(40, x: 100),
        ]
        return PostProcessInputFrame(
            tMs: tMs, detected: true, trainerContact: false, joints: joints)
    }

    /// The whole clip's analysis: `frameCount` copies of the hold above, on
    /// the export's own clock, through the real pipeline (`Analyzer`) — no
    /// fixture, no real video.
    static func holdingAnalysis(reference: ScoreReference? = nil) -> Analysis {
        let frames = (0..<frameCount).map { holdingFrame(tMs: tMs(frame: $0)) }
        return Analyzer.analyze(frames, reference: reference)
    }

    /// An analysis nobody could stand behind: a clip with no person in any
    /// frame, so there is no body length, the phases are unusable and the
    /// features are too. The export must still work — no skeleton, a neutral
    /// strip (chainlink #50).
    static func nobodyAnalysis() -> Analysis {
        let frames = (0..<frameCount).map { index in
            PostProcessInputFrame(
                tMs: tMs(frame: index), detected: false, trainerContact: false, joints: [:])
        }
        return Analyzer.analyze(frames, reference: nil)
    }

    // MARK: - Reading pixels back

    /// Is this pixel the background the source was filled with (within
    /// `tolerance`, for the encoder's own rounding)? "Not background grey"
    /// is therefore the test for "something was drawn here".
    static func isBackground(_ pixel: (r: Int, g: Int, b: Int), tolerance: Int = 14) -> Bool {
        let grey = Int(background)
        return abs(pixel.r - grey) <= tolerance
            && abs(pixel.g - grey) <= tolerance
            && abs(pixel.b - grey) <= tolerance
    }

    /// Is this pixel the style's **neutral** grey — `#8E8E93`, what
    /// `DiagramStyle` fills a strip that has nothing to show with. Not the
    /// background: a neutral strip is a *drawn* grey bar over the picture,
    /// which is exactly what an unusable analysis exports.
    static func isNeutralGrey(_ pixel: (r: Int, g: Int, b: Int), tolerance: Int = 10) -> Bool {
        abs(pixel.r - 142) <= tolerance
            && abs(pixel.g - 142) <= tolerance
            && abs(pixel.b - 147) <= tolerance
    }
}

/// The progress values of one export, in call order — a class because the
/// `@Sendable` progress closure may not capture a mutable local.
///
/// `@unchecked Sendable` for the same reason `VideoFrameSourceTests`'s
/// recorder is: the calls all happen inside one export task, and the test
/// reads the values after the `await`.
final class ProgressRecorder: @unchecked Sendable {
    private(set) var values: [Double] = []

    func record(_ value: Double) {
        values.append(value)
    }
}
