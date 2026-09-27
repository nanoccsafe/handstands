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

import AVFoundation
import CoreImage
import CoreMedia
import CoreVideo
import Foundation
import Vision
import VisionPoseCore

/// Something that went wrong while decoding or running the model.
enum RunnerError: Error, CustomStringConvertible {
    case noVideoTrack(String)
    case cannotStartReading(String)
    case readFailed(String)
    case noPixelBuffer(frame: Int)
    case pixelBufferAllocation(Int, Int)
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
        case .pixelBufferAllocation(let width, let height):
            return "could not allocate a \(width)x\(height) pixel buffer"
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

/// Turns decoded frames into display-orientation frames.
struct DisplayOrienter {
    private let transform: AffineTransform2D
    private let displaySize: PixelSize
    private let context: CIContext

    init(transform: DisplayTransform) {
        self.transform = transform.pixelTransform
        self.displaySize = transform.displaySize
        // Working in a known colour space keeps the render from shifting the
        // colours; sRGB is what the pixel buffers are.
        self.context = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
    }

    /// `nil` when the stored frame is already upright, in which case the
    /// decoder's own buffer is used as it is.
    func displayBuffer(from pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer? {
        if transform.a == 1, transform.b == 0, transform.c == 0, transform.d == 1,
            transform.tx == 0, transform.ty == 0
        {
            return nil
        }
        let image = CIImage(cvPixelBuffer: pixelBuffer).transformed(
            by: CGAffineTransform(
                a: transform.a, b: transform.b, c: transform.c,
                d: transform.d, tx: transform.tx, ty: transform.ty
            )
        )
        var rotated: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            displaySize.width,
            displaySize.height,
            kCVPixelFormatType_32BGRA,
            nil,
            &rotated
        )
        guard status == kCVReturnSuccess, let rotated else {
            throw RunnerError.pixelBufferAllocation(displaySize.width, displaySize.height)
        }
        context.render(image, to: rotated)
        return rotated
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
        let orienter = DisplayOrienter(transform: displayTransform)

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

        let request = VNDetectHumanBodyPoseRequest()
        var auto = AutoRotation()
        var report = ClipReport()
        report.displaySize = displaySize

        var frameIdx = 0
        while let sampleBuffer = trackOutput.copyNextSampleBuffer() {
            if let limit = options.maxFrames, frameIdx >= limit { break }
            guard let stored = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                throw RunnerError.noPixelBuffer(frame: frameIdx)
            }
            // `nil` from the orienter means the frame is already upright.
            let display = try orienter.displayBuffer(from: stored) ?? stored
            let tMs = Int(
                (CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)) * 1000)
                    .rounded()
            )
            let rotated = options.rotate.isRotated(auto.rotateNextFrame)

            // `.down` is a 180° turn, and telling Vision the orientation is
            // free where turning the pixels would not be.
            let handler = VNImageRequestHandler(
                cvPixelBuffer: display,
                orientation: rotated ? .down : .up,
                options: [:]
            )
            do {
                try handler.perform([request])
            } catch {
                throw RunnerError.cannotRecognizePoints(frameIdx)
            }

            let observations = request.results ?? []
            var people: [DisplayPose] = []
            for (personIdx, observation) in observations.enumerated() {
                let recognized: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]
                do {
                    recognized = try observation.recognizedPoints(.all)
                } catch {
                    throw RunnerError.cannotRecognizePoints(frameIdx)
                }
                var points: [VisionJoint: PixelPoint] = [:]
                var rows: [KeypointRow] = []
                for joint in VisionJoint.allCases {
                    guard let found = recognized[joint.visionJointName] else { continue }
                    var pixel = try CoordinateMath.normalizedToPixels(
                        NormalizedPoint(x: found.location.x, y: found.location.y),
                        in: displaySize
                    )
                    if rotated {
                        pixel = try CoordinateMath.mapBackHalfTurn(pixel, in: displaySize)
                    }
                    points[joint] = pixel
                    rows.append(
                        KeypointRow(
                            frameIdx: frameIdx,
                            tMs: tMs,
                            personIdx: personIdx,
                            joint: joint.columnName,
                            point: pixel,
                            confidence: Double(found.confidence),
                            rotated: rotated,
                            detected: true
                        )
                    )
                }
                people.append(DisplayPose(points: points))
                try write(rows, to: writer)
            }
            if people.isEmpty {
                try write(
                    KeypointRow.missingFrame(
                        frameIdx: frameIdx,
                        tMs: tMs,
                        joints: VisionJoint.columnNames,
                        rotated: rotated
                    ),
                    to: writer
                )
            }

            report.frameCount += 1
            report.rotatedFrames += rotated ? 1 : 0
            report.detectedFrames += people.isEmpty ? 0 : 1
            report.maxPeople = max(report.maxPeople, people.count)
            report.framesByPeople[people.count, default: 0] += 1
            if options.rotate == .auto {
                auto.update(with: people)
            }
            frameIdx += 1
        }

        if reader.status == .failed {
            throw RunnerError.readFailed(reader.error?.localizedDescription ?? "unknown")
        }
        guard report.frameCount > 0 else {
            throw RunnerError.noFramesDecoded(options.videoPath)
        }
        report.runtimeSeconds = Date().timeIntervalSince(started)
        try writeManifest(options: options, report: report, request: request)
        return report
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
