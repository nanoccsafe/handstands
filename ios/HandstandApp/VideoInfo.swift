import AVFoundation
import CoreTransferable
import Foundation
import UniformTypeIdentifiers

/// What the app shows after a video is picked: how long it is, how big its
/// frame is and how fast the camera wrote it. Plain values, `Sendable`, no UI
/// — `VideoInfoFormatter` turns them into the strings on screen, which is
/// what the unit tests pin down; the analysis diagnostics (chainlink #93)
/// record them as they are.
struct VideoInfo: Sendable, Equatable {
    /// Length in seconds exactly as AVFoundation reports it.
    let duration: TimeInterval
    /// Frame size in pixels after the track's preferred transform, so a
    /// portrait phone recording reads 1080 × 1920, not 1920 × 1080.
    let width: Int
    let height: Int
    /// The track's nominal frame rate in frames per second — what the
    /// *recording* was made at (about 60 on the iPhone), against the ~30 fps
    /// the app analyses at (chainlink #93). `0` when the track reports none,
    /// which is a value rather than a guess.
    var frameRate: Double = 0
}

/// The picked video as a file: PhotosPicker hands it over through this
/// `Transferable`, and the app reads it with `AVURLAsset` (`VideoInfoReader`).
struct MovieFile: Transferable, Sendable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            // The received file only exists for the duration of this call, so
            // keep a private copy before returning.
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return MovieFile(url: copy)
        }
    }
}

/// Reading a video's duration and pixel size with AVFoundation — everything
/// the scaffold does with the file it picks.
enum VideoInfoReader {
    enum ReadError: Error, Equatable {
        /// The file has no video track: it is not (only) a movie.
        case notAVideo
    }

    static func read(url: URL) async throws -> VideoInfo {
        let asset = AVURLAsset(url: url)
        let seconds = try await asset.load(.duration).seconds
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ReadError.notAVideo
        }
        let naturalSize = try await track.load(.naturalSize)
        let preferredTransform = try await track.load(.preferredTransform)
        let nominalFrameRate = Double(try await track.load(.nominalFrameRate))
        let oriented = naturalSize.applying(preferredTransform)
        return VideoInfo(
            duration: seconds,
            width: Int(abs(oriented.width).rounded()),
            height: Int(abs(oriented.height).rounded()),
            // A track that will not say reads 0, not a made-up 30: the
            // diagnostics record what the file says.
            frameRate: nominalFrameRate.isFinite && nominalFrameRate > 0 ? nominalFrameRate : 0
        )
    }
}

/// The display strings for `VideoInfo`, kept free of SwiftUI so they can be
/// tested without a view in the way.
enum VideoInfoFormatter {
    /// Clock time: `12.4` → `"0:12"`, `65.4` → `"1:05"`, `3661` →
    /// `"1:01:01"`. Anything that is not a real number of seconds (negative,
    /// `NaN`, infinite, absurdly large) reads `"0:00"` instead of trapping.
    static func duration(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite, seconds >= 0, seconds < 1_000_000_000 else {
            return "0:00"
        }
        let total = Int(seconds.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 {
            return "\(hours):\(padded(minutes)):\(padded(secs))"
        }
        return "\(minutes):\(padded(secs))"
    }

    /// `1920 × 1080` — the multiplication sign, in the way the camera
    /// recorded the frame (width before height).
    static func pixelSize(width: Int, height: Int) -> String {
        "\(width) × \(height)"
    }

    private static func padded(_ value: Int) -> String {
        value < 10 ? "0\(value)" : "\(value)"
    }
}
