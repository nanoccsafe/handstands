import AVFoundation
import Foundation

/// Writes the camera's video sample buffers to a `.mov` with `AVAssetWriter`.
///
/// `AVCaptureMovieFileOutput` cannot be used because its frames never pass
/// through a data output, and the framing guide has to see *the same* buffers
/// that end up in the file (chainlink #46).
///
/// Queue confinement: an instance is created, appended to and finished on the
/// capture engine's queue and touched nowhere else, which is what makes the
/// `@unchecked Sendable` below honest — the completion block handed to
/// `AVAssetWriter` is the one piece that outlives a `finish` call, and it
/// only reads values captured before it was created.
final class RecordingWriter: @unchecked Sendable {
    enum WriterError: Error, Equatable {
        /// The asset writer refused the output URL (it already exists, or is
        /// not writable).
        case cannotStart(String)
        /// Stopped before a single frame was appended — nothing to keep.
        case noFrames
    }

    /// The file being written.
    let url: URL

    private let assetWriter: AVAssetWriter
    private let input: AVAssetWriterInput
    private var sessionStarted = false
    private var rejected = false

    /// Prepares `url` to receive `width × height` H.264 video at
    /// `frameRate`. Nothing is written until the first frame arrives, so the
    /// file's timeline starts at that frame's presentation time stamp and a
    /// short attempt costs no leading gap.
    init(url: URL, width: Int, height: Int, frameRate: Int) throws {
        self.url = url
        do {
            assetWriter = try AVAssetWriter(outputURL: url, fileType: .mov)
        } catch {
            throw WriterError.cannotStart(error.localizedDescription)
        }

        // H.264 in a .mov: encodes everywhere this app runs, including the
        // simulator, and is what `AVAssetWriter` is happiest to append to in
        // real time. (HEVC would halve the bytes on the phone, but only where
        // an encoder exists — the availability check would be guesswork.)
        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: 12_000_000,
            AVVideoExpectedSourceFrameRateKey: frameRate,
            AVVideoMaxKeyFrameIntervalKey: frameRate * 2,
        ]
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression,
        ]
        input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        // Real time, not "encode as fast as you can": frames arrive when the
        // sensor produces them.
        input.expectsMediaDataInRealTime = true
        guard assetWriter.canAdd(input) else {
            throw WriterError.cannotStart("the writer refuses a \(width)×\(height) h264 input")
        }
        assetWriter.add(input)
    }

    /// Appends one camera frame. Called for every frame while recording;
    /// frames the encoder is not ready for are dropped (the encoder catches
    /// up — `expectsMediaDataInRealTime` says they are not worth blocking
    /// the sensor for).
    func append(_ sampleBuffer: CMSampleBuffer) {
        guard !rejected else { return }
        if !sessionStarted {
            guard assetWriter.startWriting() else {
                rejected = true
                return
            }
            assetWriter.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
            sessionStarted = true
        }
        guard assetWriter.status == .writing else {
            rejected = true
            return
        }
        if input.isReadyForMoreMediaData {
            rejected = !input.append(sampleBuffer)
        }
    }

    /// Stops the file and hands the finished URL (or the failure) to
    /// `completion`, which the engine hops to the main actor with.
    func finish(_ completion: @escaping @Sendable (Result<URL, any Error>) -> Void) {
        guard sessionStarted, !rejected else {
            if assetWriter.status == .writing || assetWriter.status == .unknown {
                assetWriter.cancelWriting()
            }
            completion(.failure(WriterError.noFrames))
            return
        }
        input.markAsFinished()
        let writer = assetWriter
        let fileURL = url
        writer.finishWriting {
            if let error = writer.error {
                completion(.failure(error))
            } else {
                completion(.success(fileURL))
            }
        }
    }
}
