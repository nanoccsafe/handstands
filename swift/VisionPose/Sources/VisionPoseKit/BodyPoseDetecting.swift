// The Vision call, behind a protocol so tests never need Vision at all.
//
// Everything the runner used to do inline — build the request, tell Vision how
// the image is oriented, read the recognised points — is one method here.
// `VisionPoseService` owns the rotation decision and the mapping; this owns
// only "ask the model".

import CoreVideo
import Foundation
import Vision
import VisionPoseCore

/// A body-pose detector: people in, normalised points out.
public protocol BodyPoseDetecting {
    /// People Vision found in `frame` when told the image orientation is `.up`
    /// (rotated == false) or `.down` (rotated == true).
    ///
    /// Points are Vision's raw normalized coordinates (origin bottom-left, in
    /// the space of the image **as oriented** — a `.down` frame's points are
    /// in the upside-down frame, which is why the caller maps them back).
    /// One dictionary per person, in Vision's order; a joint Vision did not
    /// report is absent from that person's dictionary.
    func detect(_ frame: CVPixelBuffer, rotated: Bool) throws
        -> [[VisionJoint: (point: VisionPoseCore.NormalizedPoint, confidence: Double)]]
}

/// The real detector: `VNDetectHumanBodyPoseRequest` with the same settings
/// the runner has always used (`.all` joints, `.up`/`.down` orientation, which
/// costs no pixel copy where turning the pixels would).
public struct VisionBodyPoseDetector: BodyPoseDetecting {
    /// The request every `detect` call performs.
    ///
    /// One request is reused for the whole clip, exactly as the runner did
    /// before the split: the model behind it is cached by Vision, so a fresh
    /// request per frame would only re-pay that lookup. The runner reads
    /// `revision` off it for the run manifest.
    public let request = VNDetectHumanBodyPoseRequest()

    public init() {}

    public func detect(_ frame: CVPixelBuffer, rotated: Bool) throws
        -> [[VisionJoint: (point: VisionPoseCore.NormalizedPoint, confidence: Double)]]
    {
        // `.down` is a 180° turn, and telling Vision the orientation is free
        // where turning the pixels would not be.
        let handler = VNImageRequestHandler(
            cvPixelBuffer: frame,
            orientation: rotated ? .down : .up,
            options: [:]
        )
        do {
            try handler.perform([request])
        } catch {
            throw PoseServiceError.visionFailed(error.localizedDescription)
        }
        let observations = request.results ?? []
        return try observations.map { observation in
            let recognized: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]
            do {
                recognized = try observation.recognizedPoints(.all)
            } catch {
                throw PoseServiceError.visionFailed(error.localizedDescription)
            }
            var person: [VisionJoint: (point: VisionPoseCore.NormalizedPoint, confidence: Double)] = [:]
            for joint in VisionJoint.allCases {
                guard let found = recognized[joint.visionJointName] else { continue }
                person[joint] = (
                    point: VisionPoseCore.NormalizedPoint(x: found.location.x, y: found.location.y),
                    confidence: Double(found.confidence)
                )
            }
            return person
        }
    }
}
