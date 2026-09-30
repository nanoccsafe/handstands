// The Vision backend of `PoseService`: the per-frame pipeline, in the order
// the runner has always run it.
//
//     rotated?  ->  detect  ->  map to display pixels  ->  choose a person
//               ->  update the auto-rotation state  ->  PostProcessInputFrame
//
// Every step reuses `VisionPoseCore`: `RotateMode.isRotated`, `CoordinateMath`
// (the y flip, then the 180° map-back), `AutoRotation.chosenPose` /
// `AutoRotation.update`. The runner (`RunVisionPose.swift`) calls exactly this
// class, so its CSV and the app's analysis see the same numbers.

import CoreVideo
import Foundation
import HandstandCore
import VisionPoseCore

/// One person mapped into **display** pixels, with Vision's confidence kept
/// beside each point.
///
/// `DisplayPose` (in `VisionPoseCore`) carries the points only; the runner
/// writes a `confidence` column per joint and the shared schema a
/// `visibility`, so the pair travels together here.
public struct DisplayPerson: Sendable {
    /// Display pixels + confidence, by Vision's joint. A joint Vision did not
    /// report is absent — there is no zero or placeholder.
    public var points: [VisionJoint: (point: PixelPoint, confidence: Double)]

    public init(points: [VisionJoint: (point: PixelPoint, confidence: Double)]) {
        self.points = points
    }

    /// The same person as the core's `DisplayPose` (points, no confidence) —
    /// the shape `AutoRotation`'s rule is written against.
    public var pose: DisplayPose {
        DisplayPose(points: points.mapValues { $0.point })
    }
}

/// What one `processAll` call saw.
public struct PoseFrameResult: Sendable {
    /// The chosen person in the shared schema — what `process` returns.
    public let frame: PostProcessInputFrame
    /// Every person found, in Vision's order. The app assumes one athlete, so
    /// it reads `frame`; the runner writes one CSV block per person from this.
    public let people: [DisplayPerson]
    /// Was **this** frame handed to Vision turned 180°?
    public let rotated: Bool
}

extension VisionJoint {
    /// The shared-schema joint this Vision joint becomes, or `nil` for the
    /// joints the 15-joint schema does not track.
    ///
    /// ``VisionJoint/columnName`` is MediaPipe's spelling, so
    /// `Joint(rawValue:)` is exactly the mapping: the 13 joints both
    /// skeletons have resolve, `left_eye`/`right_eye`/the ears/`neck`/`root`
    /// do not exist in the schema, and Vision has no `foot_index` at all —
    /// so those two joints are simply never in a Vision frame.
    public var sharedJoint: Joint? { Joint(rawValue: columnName) }
}

/// The Apple Vision backend: built into iOS, so it needs no download and no
/// third-party code (chainlink #47 runs it after recording; MediaPipe lands
/// in #45, behind the bake-off of #16).
public final class VisionPoseService: PoseService {
    /// Which frames are handed to Vision turned 180°.
    public let rotate: RotateMode
    private let detector: any BodyPoseDetecting
    /// The `--rotate auto` state: what the *previous* frames decided about the
    /// next one. Fresh (`false`) until `reset()` or the first frame.
    private var autoRotation = AutoRotation()

    /// - Parameters:
    ///   - rotate: which frames to feed upside down (`.auto`, the default,
    ///     judges from the previous frame's lowest-wrist person).
    ///   - detector: the Vision call; a fake in tests.
    public init(rotate: RotateMode = .auto, detector: BodyPoseDetecting = VisionBodyPoseDetector()) {
        self.rotate = rotate
        self.detector = detector
    }

    public var backendName: String { "vision" }

    /// Back to "the first frame is not rotated" — call before each new clip.
    public func reset() {
        autoRotation = AutoRotation()
    }

    public func process(_ displayFrame: CVPixelBuffer, tMs: Int) throws -> PostProcessInputFrame {
        try processAll(displayFrame, tMs: tMs).frame
    }

    /// The same frame as `process`, plus what the runner needs beside it: every
    /// person (its multi-person CSV) and the rotation this frame ran with (its
    /// `rotated` column).
    ///
    /// - Throws: ``PoseServiceError`` when Vision fails; `VisionPoseError`
    ///   when the frame has no usable size.
    public func processAll(_ displayFrame: CVPixelBuffer, tMs: Int) throws -> PoseFrameResult {
        // 1. The decision comes from the *previous* frames; the first frame of
        //    a clip (and of a reset service) is never rotated.
        let rotated = rotate.isRotated(autoRotation.rotateNextFrame)

        // 2. Ask the model, telling Vision how the image is oriented.
        let detections = try detector.detect(displayFrame, rotated: rotated)

        // 3. Normalised -> display pixels: the y-origin flip first, then the
        //    180° map-back when Vision saw the frame upside down. The frame is
        //    already in display orientation, so its own size is the size every
        //    coordinate is in.
        let size = PixelSize(
            width: CVPixelBufferGetWidth(displayFrame),
            height: CVPixelBufferGetHeight(displayFrame)
        )
        var people: [DisplayPerson] = []
        for detection in detections {
            var points: [VisionJoint: (point: PixelPoint, confidence: Double)] = [:]
            for (joint, found) in detection {
                var pixel = try CoordinateMath.normalizedToPixels(found.point, in: size)
                if rotated {
                    pixel = try CoordinateMath.mapBackHalfTurn(pixel, in: size)
                }
                points[joint] = (point: pixel, confidence: found.confidence)
            }
            people.append(DisplayPerson(points: points))
        }

        // 4./5. One athlete on the device, so `chosenPose` is a guard rather
        //       than a policy: the lowest-wrist person is the one reported.
        let poses = people.map(\.pose)
        let chosen = AutoRotation.chosenPose(poses).flatMap { pose in
            people.first { $0.pose == pose }
        }

        // 6. With `.auto`, judge the *next* frame from everyone seen now. An
        //    empty frame changes nothing (that is `AutoRotation`'s own rule),
        //    and the other modes never look at the state.
        if rotate == .auto {
            autoRotation.update(with: poses)
        }

        // 7. The chosen person, in the shared schema. `trainerContact` is
        //    always false: there is no trainer logic on the device, the app
        //    assumes one athlete.
        var joints: [Joint: Keypoint] = [:]
        if let chosen {
            for (visionJoint, found) in chosen.points {
                guard let joint = visionJoint.sharedJoint else { continue }
                joints[joint] = Keypoint(
                    x: found.point.x,
                    y: found.point.y,
                    visibility: found.confidence
                )
            }
        }
        let input = PostProcessInputFrame(
            tMs: tMs,
            detected: !people.isEmpty,
            trainerContact: false,
            joints: joints
        )
        return PoseFrameResult(frame: input, people: people, rotated: rotated)
    }
}
