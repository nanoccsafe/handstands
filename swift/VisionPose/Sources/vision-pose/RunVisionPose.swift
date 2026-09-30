// vision-pose — Apple Vision body-pose keypoints for one clip.
//
// Decodes with AVAssetReader in **display** orientation (the track's
// `preferredTransform` applied, so a 1024x576 clip with rotation -90 is decoded
// as 576x1024, exactly what OpenCV hands the MediaPipe runner), runs
// `VNDetectHumanBodyPoseRequest` on every frame, and writes one CSV row per
// (frame, person, joint) in the schema `handstand.vision_import` turns into the
// pipeline's parquet files.
//
//     vision-pose --video <path> --rotate none|180|auto --out <file.csv>
//                 [--max-frames N] [--expect-frames N] [--clip-id ID]
//
// Two things are worth knowing about the coordinates:
//
// * Vision normalises points with the origin at the **bottom** left, so the
//   conversion to the pipeline's top-left pixels flips `y` —
//   ``CoordinateMath/normalizedToPixels(_:in:)``.
// * `--rotate 180` and `--rotate auto` hand Vision the frame turned upside down
//   (via `CGImagePropertyOrientation.down`, which costs no pixel copy) and map
//   the points back, so no rotated coordinate ever reaches the CSV.
//
// Neither happens here any more: per-frame inference, point mapping and the
// auto-rotation state live in `VisionPoseService`, and the display-orientation
// conversion in `DisplayFrames` (chainlink #82), so the runner and the iOS app
// run the same code. What is left below is the parts only a runner has: decode,
// CSV rows, the report and the manifest.

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import Vision
import VisionPoseCore
import VisionPoseKit

/// Something that went wrong while decoding or running the model.
enum RunnerError: Error, CustomStringConvertible {
    case noVideoTrack(String)
    case cannotStartReading(String)
    case readFailed(String)
    case noPixelBuffer(frame: Int)
    case noFramesDecoded(String)
    case cannotRecognizePoints(Int)

    var description: String {
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
        case .cannotRecognizePoints(let frame):
            return "could not read the body-pose points of frame \(frame)"
        }
    }
}

/// What a run did, for the summary line and the run manifest.
struct ClipReport {
    var frameCount = 0
    var detectedFrames = 0
    var rotatedFrames = 0
    /// People per frame -> number of frames, including the frames with nobody.
    var framesByPeople: [Int: Int] = [:]
    var displaySize = PixelSize(width: 0, height: 0)
    var runtimeSeconds: Double = 0
    var maxPeople = 0

    var detectedPercent: Double {
        frameCount == 0 ? 0 : 100 * Double(detectedFrames) / Double(frameCount)
    }

    var rotatedPercent: Double {
        frameCount == 0 ? 0 : 100 * Double(rotatedFrames) / Double(frameCount)
    }

    var peopleSummary: String {
        framesByPeople.keys.sorted().map { "\($0):\(framesByPeople[$0]!)" }.joined(separator: " ")
    }
}

@main
struct RunVisionPose {
    static func main() async {
        do {
            let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
            let report = try await run(options)
            summarise(options: options, report: report)
        } catch CLIError.helpRequested {
            print(Options.usage)
        } catch let error as CLIError {
            fail("\(error)\n\n\(Options.usage)")
        } catch {
            fail("\(error)")
        }
    }

    private static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("vision-pose: \(message)\n".utf8))
        exit(1)
    }

    private static func summarise(options: Options, report: ClipReport) {
        // The frame count has to match the other runner's, or a bake-off is
        // comparing frames that do not line up. OpenCV decodes every frame;
        // AVFoundation is allowed to drop some, so say so when it does.
        if let expected = options.expectFrames, expected != report.frameCount {
            FileHandle.standardError.write(
                Data(
                    """
                    vision-pose: warning: \(options.videoFileName) decoded \
                    \(report.frameCount) frames but the reference run has \
                    \(expected); the two keypoint sets do not line up frame by frame

                    """.utf8
                )
            )
        }
        print(
            """
            clip \(options.videoFileName) id=\(options.clipId ?? "-") rotate=\(options.rotate.rawValue) \
            frames=\(report.frameCount) display=\(report.displaySize) \
            detected=\(String(format: "%.1f", report.detectedPercent))% \
            rotated=\(String(format: "%.1f", report.rotatedPercent))% \
            people[\(report.peopleSummary)] maxPeople=\(report.maxPeople) \
            runtime=\(String(format: "%.1f", report.runtimeSeconds))s \
            -> \(options.outPath)
            """
        )
    }

    /// Decode, run the model over every frame, write the CSV and the manifest.
    static func run(_ options: Options) async throws -> ClipReport {
        let started = Date()
        let videoURL = URL(fileURLWithPath: options.videoPath)
        guard FileManager.default.fileExists(atPath: options.videoPath) else {
            throw CocoaError(.fileNoSuchFile)
        }
        let asset = AVURLAsset(url: videoURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw RunnerError.noVideoTrack(options.videoPath)
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
        let displaySize = displayTransform.displaySize

        // `add`/`startReading`/`copyNextSampleBuffer` are the spellings that
        // have existed since macOS 10.7; the SDK's newer
        // `outputProvider(for:)` API is macOS 27+ and would drop the macOS 15
        // support this package claims.
        let reader = try AVAssetReader(asset: asset)
        let trackOutput = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        )
        trackOutput.alwaysCopiesSampleData = false
        reader.add(trackOutput)
        guard reader.startReading() else {
            throw RunnerError.cannotStartReading(reader.error?.localizedDescription ?? "unknown")
        }

        FileManager.default.createFile(atPath: options.outPath, contents: nil)
        guard let writer = FileHandle(forWritingAtPath: options.outPath) else {
            throw CocoaError(.fileWriteUnknown)
        }
        defer { try? writer.close() }
        try writer.write(contentsOf: Data((KeypointRow.header + "\n").utf8))

        // One detector owns one Vision request for the whole clip (its
        // `revision` goes into the manifest); the service owns the mapping and
        // the `--rotate auto` state — the same class the iOS app runs.
        let detector = VisionBodyPoseDetector()
        let service = VisionPoseService(rotate: options.rotate, detector: detector)
        var report = ClipReport()
        report.displaySize = displaySize

        var frameIdx = 0
        while let sampleBuffer = trackOutput.copyNextSampleBuffer() {
            if let limit = options.maxFrames, frameIdx >= limit { break }
            guard let stored = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                throw RunnerError.noPixelBuffer(frame: frameIdx)
            }
            // `nil` from `DisplayFrames` means the frame is already upright,
            // in which case the decoder's own buffer goes to the model.
            let display = try DisplayFrames.displayBuffer(
                from: stored, transform: displayTransform
            ) ?? stored
            let tMs = Int(
                (CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)) * 1000)
                    .rounded()
            )

            // The whole per-frame pipeline — Vision, the point mapping, the
            // `--rotate auto` state — in one call to the shared backend.
            let result: PoseFrameResult
            do {
                result = try service.processAll(display, tMs: tMs)
            } catch is PoseServiceError {
                // Which frame failed is the runner's to know, not the kit's.
                throw RunnerError.cannotRecognizePoints(frameIdx)
            }
            try write(
                csvRows(
                    for: result.people, frameIdx: frameIdx, tMs: tMs,
                    rotated: result.rotated
                ),
                to: writer
            )

            let people = result.people
            report.frameCount += 1
            report.rotatedFrames += result.rotated ? 1 : 0
            report.detectedFrames += people.isEmpty ? 0 : 1
            report.maxPeople = max(report.maxPeople, people.count)
            report.framesByPeople[people.count, default: 0] += 1
            frameIdx += 1
        }

        if reader.status == .failed {
            throw RunnerError.readFailed(reader.error?.localizedDescription ?? "unknown")
        }
        guard report.frameCount > 0 else {
            throw RunnerError.noFramesDecoded(options.videoPath)
        }
        report.runtimeSeconds = Date().timeIntervalSince(started)
        try writeManifest(options: options, report: report, request: detector.request)
        return report
    }

    /// The CSV rows of one frame: one block per person, or the placeholder
    /// block when nobody was found. Joint order is `VisionJoint.allCases`,
    /// person order is Vision's — exactly the order the rows have always been
    /// written in.
    private static func csvRows(
        for people: [DisplayPerson],
        frameIdx: Int,
        tMs: Int,
        rotated: Bool
    ) -> [KeypointRow] {
        guard !people.isEmpty else {
            return KeypointRow.missingFrame(
                frameIdx: frameIdx,
                tMs: tMs,
                joints: VisionJoint.columnNames,
                rotated: rotated
            )
        }
        var rows: [KeypointRow] = []
        for (personIdx, person) in people.enumerated() {
            for joint in VisionJoint.allCases {
                guard let found = person.points[joint] else { continue }
                rows.append(
                    KeypointRow(
                        frameIdx: frameIdx,
                        tMs: tMs,
                        personIdx: personIdx,
                        joint: joint.columnName,
                        point: found.point,
                        confidence: found.confidence,
                        rotated: rotated,
                        detected: true
                    )
                )
            }
        }
        return rows
    }

    private static func write(_ rows: [KeypointRow], to writer: FileHandle) throws {
        guard !rows.isEmpty else { return }
        var text = ""
        text.reserveCapacity(rows.count * 48)
        for row in rows {
            text += row.csvLine
            text += "\n"
        }
        try writer.write(contentsOf: Data(text.utf8))
    }

    /// The run manifest, next to the CSV.
    ///
    /// The CSV is deliberately nothing but keypoints; this is where the facts
    /// only the Mac knows come from — the macOS version, the Vision revision
    /// and the runtime — and `handstand.vision_import` folds them into the
    /// parquet's sidecar.
    private static func writeManifest(
        options: Options,
        report: ClipReport,
        request: VNDetectHumanBodyPoseRequest
    ) throws {
        let osVersion = ProcessInfo.processInfo.operatingSystemVersion
        let manifest: [String: Any] = [
            "clip_id": options.clipId ?? (options.outPath as NSString).deletingPathExtension,
            "source_file": options.videoFileName,
            "model": "apple-vision-body-pose",
            "vision_revision": request.revision,
            "macos_version": "\(osVersion.majorVersion).\(osVersion.minorVersion).\(osVersion.patchVersion)",
            "rotate": options.rotate.rawValue,
            "display_width": report.displaySize.width,
            "display_height": report.displaySize.height,
            "frame_count": report.frameCount,
            "detected_frame_count": report.detectedFrames,
            "rotated_frame_count": report.rotatedFrames,
            "frames_by_people": report.framesByPeople.mapKeys { String($0) },
            "runtime_seconds": (report.runtimeSeconds * 1000).rounded() / 1000,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys]
        )
        try (data + Data("\n".utf8)).write(to: URL(fileURLWithPath: options.manifestPath))
    }
}

extension Dictionary {
    /// The same dictionary with its keys run through `transform` — used for the
    /// manifest's people histogram, whose keys are frame counts.
    func mapKeys<T: Hashable>(_ transform: (Key) -> T) -> [T: Value] {
        var out = [T: Value](minimumCapacity: count)
        for (key, value) in self {
            out[transform(key)] = value
        }
        return out
    }
}
