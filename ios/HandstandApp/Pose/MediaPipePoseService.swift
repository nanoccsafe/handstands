import Accelerate
import CoreVideo
import Foundation
import HandstandCore
import VisionPoseKit

// --------------------------------------------------------------------------- #
// The MediaPipe backend of the app (chainlink #45): `--rotate best`, ported
// to a clip-level extraction.
//
// The pipeline decides each frame's orientation from the *whole clip*
// (`handstand.pose_mediapipe.choose_orientations`: a margin smoothed over a
// centred 0.5 s window of clip time, an opening orientation, and hysteresis
// — see docs/bakeoff.md for why the phone must match it exactly), so one
// frame cannot be answered on its own. The two passes therefore run over the
// whole video first, and only then are the chosen pass's keypoints handed to
// the pipeline as `PostProcessInputFrame`s:
//
//     pass 1: upright landmarker  → display frame as the viewer sees it
//     pass 2: rotated landmarker   → the same frame turned 180° exactly
//     choose: OrientationChooser.chooseOrientations(tMs, scoreUp, scoreRot)
//     emit:  the chosen pass, mapped back to display pixels, 33 → 15 joints
//
// Which landmarker sees which orientation is fixed for the whole clip —
// VIDEO mode tracks between frames, and a tracker that saw both orientations
// of the same clip would carry state from one into the other (the same reason
// `pose_mediapipe.run_clip` builds two of them for `--rotate best`).
//
// The model itself is behind ``MediaPipeLandmarkDetecting``, so the tests in
// HandstandAppTests inject fake landmark sequences — no model, no video, no
// network.
// --------------------------------------------------------------------------- //

/// One landmark as MediaPipe reported it: `x`/`y`/`z` normalised to the
/// frame the model *saw* (the display frame for the upright pass, the turned
/// frame for the rotated pass), plus MediaPipe's own scores (`nil` when the
/// model reported none — Python reads those as `None` → NaN).
public struct MediaPipeLandmark: Sendable, Equatable {
    public var x: Double
    public var y: Double
    public var z: Double
    public var visibility: Double?
    public var presence: Double?

    public init(
        x: Double, y: Double, z: Double,
        visibility: Double? = nil, presence: Double? = nil
    ) {
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence
    }
}

/// One VIDEO-mode landmarker — the seam between the app and MediaPipe.
///
/// Mirrors `LandmarkerLike` in `pose_mediapipe.py`: one call per frame with
/// a millisecond timestamp, the answer either one pose's 33 landmarks or
/// `nil` for "nobody in this frame" (`numPoses = 1`, so at most one pose;
/// a frame with more than 33 landmarks is a bug, not a person, and the
/// extractor refuses it the way `_detect_pose` raises).
public protocol MediaPipeLandmarkDetecting: Sendable {
    /// One VIDEO-mode call. `frame` is whatever this landmarker's pass feeds
    /// the model — upright for the upright pass, 180°-turned for the rotated
    /// pass — and the landmarks are normalised to *that* frame.
    func detectForVideo(_ frame: CVPixelBuffer, tMs: Int) throws -> [MediaPipeLandmark]?
}

/// What the clip extractor reads: the same frames twice, once per pass.
///
/// `VideoFrameSource` conforms — its `frames()` makes a fresh reader each
/// call, so both passes see the identical display frames with the identical
/// timestamps. Tests supply a fake of canned pixel buffers.
public protocol ClipFrameSource: Sendable {
    func estimatedFrameCount() async throws -> Int
    func frames() -> AsyncThrowingStream<(buffer: CVPixelBuffer, tMs: Int), Error>
}

extension VideoFrameSource: ClipFrameSource {}

/// What the extraction can refuse before the model is even asked: a pass
/// that saw a different clip, a pose of the wrong length, a buffer shape
/// the 180° turn cannot make (or a pixel format it will not touch).
public enum MediaPipeExtractorError: Error, Equatable, CustomStringConvertible {
    /// The rotated pass did not see the same frames, in the same order,
    /// with the same timestamps as the upright pass.
    case passMismatch(String)
    /// One pose was not `OrientationChooser.landmarkCount` (33) landmarks —
    /// Python refuses this too (`expected 33 landmarks, got …`).
    case wrongLandmarkCount(Int)
    /// The two passes disagreed about the frame size, so the map-back of the
    /// rotated pass would be wrong.
    case sizeMismatch(String)
    /// The rotated pass only handles `kCVPixelFormatType_32BGRA` (what
    /// `VideoFrameSource` decodes); anything else is refused rather than
    /// silently turned wrong.
    case unsupportedPixelFormat(UInt32)
    /// The turned copy of a frame could not be allocated.
    case pixelBufferAllocation(width: Int, height: Int)

    public var description: String {
        switch self {
        case .passMismatch(let message):
            return "the two MediaPipe passes disagree: \(message)"
        case .wrongLandmarkCount(let count):
            return "expected 33 landmarks, got \(count)"
        case .sizeMismatch(let message):
            return "the two MediaPipe passes disagree: \(message)"
        case .unsupportedPixelFormat(let format):
            return "unsupported pixel format \(format); the 180° turn needs 32BGRA"
        case .pixelBufferAllocation(let width, let height):
            return "could not allocate a \(width)x\(height) pixel buffer"
        }
    }
}

/// A display frame turned 180° for the rotated pass — `apply_display_rotation(frame, 180)`
/// of `pose_mediapipe.py`, in pixels.
///
/// Two exact vImage reflects (horizontal, then vertical) rather than a
/// rotation: a pure pixel permutation, so pixel `(x, y)` lands on
/// `(width-1-x, height-1-y)` for any frame size — the same map-back the
/// keypoints take, with no interpolation to drift a coordinate by.
public enum MediaPipeFrameRotation {
    /// The frame, turned 180°. The output has the same size (a half turn
    /// never swaps the axes), which is what lets the extractor keep one
    /// display size per frame for both passes.
    public static func turn180(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA else {
            throw MediaPipeExtractorError.unsupportedPixelFormat(
                CVPixelBufferGetPixelFormatType(source))
        }
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        var destination: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &destination)
        guard status == kCVReturnSuccess, let turned = destination else {
            throw MediaPipeExtractorError.pixelBufferAllocation(
                width: width, height: height)
        }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
        CVPixelBufferLockBaseAddress(turned, [])
        defer { CVPixelBufferUnlockBaseAddress(turned, []) }
        guard let sourceAddress = CVPixelBufferGetBaseAddress(source),
            let turnedAddress = CVPixelBufferGetBaseAddress(turned)
        else {
            throw MediaPipeExtractorError.pixelBufferAllocation(
                width: width, height: height)
        }

        var sourceBuffer = vImage_Buffer(
            data: sourceAddress,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: CVPixelBufferGetBytesPerRow(source))
        var turnedBuffer = vImage_Buffer(
            data: turnedAddress,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: CVPixelBufferGetBytesPerRow(turned))
        var scratch = vImage_Buffer()
        guard
            vImageBuffer_Init(
                &scratch, vImagePixelCount(height), vImagePixelCount(width), 32,
                vImage_Flags(kvImageNoFlags)) == kvImageNoError,
            scratch.data != nil
        else {
            throw MediaPipeExtractorError.pixelBufferAllocation(
                width: width, height: height)
        }
        // `vImageBuffer_Init` allocates with malloc; free() is the pair.
        defer { free(scratch.data) }

        let horizontal = vImageHorizontalReflect_ARGB8888(
            &sourceBuffer, &scratch, vImage_Flags(kvImageNoFlags))
        let vertical = vImageVerticalReflect_ARGB8888(
            &scratch, &turnedBuffer, vImage_Flags(kvImageNoFlags))
        guard horizontal == kvImageNoError, vertical == kvImageNoError else {
            throw MediaPipeExtractorError.pixelBufferAllocation(
                width: width, height: height)
        }
        return turned
    }
}

/// The clip-level extraction `--rotate best` is on the phone (chainlink #45).
public enum MediaPipeClipExtractor {
    /// The two landmarkers of one clip — one instance per orientation,
    /// exactly as `pose_mediapipe.run_clip` calls its factory twice for
    /// `--rotate best`, so VIDEO-mode tracking never sees mixed input.
    public struct Landmarkers: Sendable {
        public let upright: any MediaPipeLandmarkDetecting
        public let rotated: any MediaPipeLandmarkDetecting

        public init(
            upright: any MediaPipeLandmarkDetecting,
            rotated: any MediaPipeLandmarkDetecting
        ) {
            self.upright = upright
            self.rotated = rotated
        }
    }

    /// What one extraction produced: the frames of the shared schema plus
    /// the per-frame orientation the choice made — the pipeline's `rotated`
    /// column, which the real-clip parity test compares against
    /// (chainlink #45). `extract` is `run(...).frames`.
    public struct Report: Sendable {
        /// One entry per frame, in frame order — the chosen pass's keypoints.
        public let frames: [PostProcessInputFrame]
        /// Per frame: `true` when the rotated pass won (its keypoints are
        /// the ones in `frames`), mirroring the parquet's `rotated` column.
        public let rotated: [Bool]

        public init(frames: [PostProcessInputFrame], rotated: [Bool]) {
            self.frames = frames
            self.rotated = rotated
        }
    }

    /// Both passes over `source`, the orientation choice, then the chosen
    /// pass's keypoints — the whole `--rotate best` run of one clip, with
    /// the choices kept beside the frames.
    ///
    /// - Parameters:
    ///   - source: the display-orientation frames, read **twice** (one pass
    ///     each; a fresh reader per pass, same frames, same timestamps).
    ///   - makeLandmarkers: builds the pair for this clip — the real model
    ///     through `run(source:modelURL:progress:)`, fakes in tests.
    ///   - progress: called from the extraction's task with 0…1 — `0` at the
    ///     start, the upright pass's frames against the estimated total in
    ///     `0…0.5`, `0.5` between the passes, the rotated pass's in
    ///     `0.5…1`, and `1` when both are done. A clip whose length cannot
    ///     be estimated gets the milestones only.
    /// - Throws: a `VideoFrameSourceError` (or the asset's own error) when
    ///   the video cannot be read, ``MediaPipeExtractorError`` when the two
    ///   passes cannot be aligned, the landmarker's own errors, and
    ///   `CancellationError` at the first frame after `cancel()`.
    public static func run(
        source: some ClipFrameSource,
        makeLandmarkers: @escaping @Sendable () throws -> Landmarkers,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> Report {
        progress(0)
        let estimated = (try? await source.estimatedFrameCount()) ?? 0
        let landmarkers = try makeLandmarkers()

        // Pass 1: the upright landmarker sees the frame as the viewer does.
        var tMs: [Int] = []
        var widths: [Int] = []
        var heights: [Int] = []
        var uprightPoses: [[MediaPipeLandmark]?] = []
        var seen = 0
        for try await frame in source.frames() {
            try Task.checkCancellation()
            let pose = try checked(
                landmarkers.upright.detectForVideo(frame.buffer, tMs: frame.tMs))
            tMs.append(frame.tMs)
            widths.append(CVPixelBufferGetWidth(frame.buffer))
            heights.append(CVPixelBufferGetHeight(frame.buffer))
            uprightPoses.append(pose)
            seen += 1
            if estimated > 0 {
                progress(Swift.min(0.5, 0.5 * Double(seen) / Double(estimated)))
            }
        }
        try Task.checkCancellation()
        progress(0.5)

        // Pass 2: the same frames, turned 180° exactly, in the same order
        // with the same timestamps — the alignment the choice depends on.
        var rotatedPoses: [[MediaPipeLandmark]?] = []
        seen = 0
        for try await frame in source.frames() {
            try Task.checkCancellation()
            guard seen < tMs.count, frame.tMs == tMs[seen] else {
                throw MediaPipeExtractorError.passMismatch(
                    "frame \(seen) of the rotated pass has tMs \(frame.tMs), "
                        + "the upright pass had \(seen < tMs.count ? tMs[seen] : -1)")
            }
            let turned = try MediaPipeFrameRotation.turn180(frame.buffer)
            guard CVPixelBufferGetWidth(turned) == widths[seen],
                CVPixelBufferGetHeight(turned) == heights[seen]
            else {
                throw MediaPipeExtractorError.sizeMismatch(
                    "the 180° turn changed frame \(seen)'s size")
            }
            let pose = try checked(
                landmarkers.rotated.detectForVideo(turned, tMs: frame.tMs))
            rotatedPoses.append(pose)
            seen += 1
            if estimated > 0 {
                progress(0.5 + Swift.min(0.5, 0.5 * Double(seen) / Double(estimated)))
            }
        }
        guard seen == tMs.count else {
            throw MediaPipeExtractorError.passMismatch(
                "the upright pass saw \(tMs.count) frames, the rotated pass \(seen)")
        }
        try Task.checkCancellation()
        progress(1.0)

        // Both passes are in: the orientation is the clip's to decide, and
        // `OrientationChooser` is the exact port of Python's
        // `choose_orientations` — the same smoothed margin, opening
        // orientation and hysteresis the pipeline's thresholds were built on.
        let choices = OrientationChooser.chooseOrientations(
            tMs: tMs.map(Double.init),
            scoreUpright: uprightPoses.map(score(of:)),
            scoreRotated: rotatedPoses.map(score(of:))
        )

        // The chosen pass's keypoints, in display pixels and the shared
        // 15-joint schema — `detected` is the chosen pass's answer, exactly
        // as Python writes the chosen pass's poses into the parquet.
        let frames = tMs.indices.map { index in
            let rotated = choices[index]
            let pose = rotated ? rotatedPoses[index] : uprightPoses[index]
            let width = widths[index]
            let height = heights[index]
            var joints: [Joint: Keypoint] = [:]
            if let pose {
                for (joint, landmarkIndex) in mediaPipeLandmarkIndex {
                    let landmark = pose[landmarkIndex]
                    guard let visibility = landmark.visibility, visibility.isFinite,
                        landmark.x.isFinite, landmark.y.isFinite
                    else { continue }
                    // Normalised → pixels of the frame the model saw, then
                    // the 180° map-back when it saw the turned frame:
                    // `x = width - 1 - x` — `_display_pixels`, exact.
                    var x = landmark.x * Double(width - 1)
                    var y = landmark.y * Double(height - 1)
                    if rotated {
                        x = Double(width - 1) - x
                        y = Double(height - 1) - y
                    }
                    joints[joint] = Keypoint(x: x, y: y, visibility: visibility)
                }
            }
            return PostProcessInputFrame(
                tMs: tMs[index],
                detected: pose != nil,
                trainerContact: false,
                joints: joints
            )
        }
        return Report(frames: frames, rotated: choices)
    }

    /// Both passes over `source`, the orientation choice, then the chosen
    /// pass's keypoints — `run(...)` with only the frames kept, which is
    /// what the analysis hands the pipeline.
    public static func extract(
        source: some ClipFrameSource,
        makeLandmarkers: @escaping @Sendable () throws -> Landmarkers,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> [PostProcessInputFrame] {
        try await run(
            source: source, makeLandmarkers: makeLandmarkers, progress: progress).frames
    }

    /// Both passes over `source` with the real `pose_landmarker_full.task`
    /// at `modelURL`, choices included — what the real-clip parity check
    /// runs (chainlink #45).
    public static func run(
        source: some ClipFrameSource,
        modelURL: URL,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> Report {
        try await run(
            source: source,
            makeLandmarkers: {
                try Landmarkers(
                    upright: MediaPipeLandmarker(modelURL: modelURL),
                    rotated: MediaPipeLandmarker(modelURL: modelURL)
                )
            },
            progress: progress
        )
    }

    /// Both passes over `source` with the real `pose_landmarker_full.task`
    /// at `modelURL` — the call the analysis makes; the tests inject the
    /// factory instead.
    public static func extract(
        source: some ClipFrameSource,
        modelURL: URL,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> [PostProcessInputFrame] {
        try await run(source: source, modelURL: modelURL, progress: progress).frames
    }

    /// `mean_main_visibility`: one pass's score for one frame — the mean
    /// MediaPipe visibility over `OrientationChooser.mainJointIndices` (the
    /// 12 main joints, no face and no fingers), NaN when the pass found
    /// nobody or reported no score at all, exactly as Python's
    /// `mean_main_visibility` skips the non-finite scores.
    static func score(of pose: [MediaPipeLandmark]?) -> Double {
        guard let pose else { return .nan }
        var sum = 0.0
        var count = 0
        for landmarkIndex in OrientationChooser.mainJointIndices {
            guard landmarkIndex < pose.count,
                let visibility = pose[landmarkIndex].visibility,
                visibility.isFinite
            else { continue }
            sum += visibility
            count += 1
        }
        return count == 0 ? .nan : sum / Double(count)
    }

    /// One pose is either nobody (`nil`) or a full 33-landmark pose —
    /// `_detect_pose` raises on any other length, so the extraction does too.
    private static func checked(_ pose: [MediaPipeLandmark]?) throws -> [MediaPipeLandmark]? {
        if let pose, pose.count != OrientationChooser.landmarkCount {
            throw MediaPipeExtractorError.wrongLandmarkCount(pose.count)
        }
        return pose
    }
}

/// The MediaPipe backend behind `PoseService`.
///
/// The protocol is per frame; this backend is per **clip** (the orientation
/// choice reads the whole thing — chainlink #45), so `process` refuses
/// honestly rather than guessing one frame at a time, and `extract` is the
/// call the analysis makes instead.
public final class MediaPipePoseService: PoseService {
    /// The bundled `pose_landmarker_full.task` this service runs.
    public let modelURL: URL

    public init(modelURL: URL) {
        self.modelURL = modelURL
    }

    public var backendName: String { "mediapipe" }

    /// Nothing to reset: each `extract` builds its own landmarker pair, so
    /// clip two starts with the same "no frames seen yet" as clip one.
    public func reset() {}

    /// The per-frame API cannot answer for this backend — the choice needs
    /// every frame of the clip first — so it says so
    /// (``PoseServiceError/clipLevelOnly(_:)``) instead of inventing a
    /// frame-by-frame rule the thresholds were not built on.
    public func process(_ displayFrame: CVPixelBuffer, tMs: Int) throws -> PostProcessInputFrame {
        throw PoseServiceError.clipLevelOnly(
            "the MediaPipe backend decides orientation over the whole clip — run extract(source:progress:)"
        )
    }

    /// The whole clip through both passes and the orientation choice; see
    /// ``MediaPipeClipExtractor/extract(source:makeLandmarkers:progress:)``.
    public func extract(
        source: some ClipFrameSource,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> [PostProcessInputFrame] {
        try await MediaPipeClipExtractor.extract(
            source: source, modelURL: modelURL, progress: progress)
    }
}
