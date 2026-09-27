// The CSV the runner writes, one row per (frame, person, joint).
//
// A frame with nobody in it is still written: one row per joint with
// `person_idx = -1`, empty `x`/`y`/`confidence` and `detected = false`. So every
// frame is represented exactly once, and `handstand.vision_import` never has to
// re-index or guess whether a frame was skipped. The column order is
//
//     frame_idx,t_ms,person_idx,joint,x,y,confidence,rotated,detected
//
// and `NO_PERSON_IDX` is the same `-1` the MediaPipe multi-person schema uses
// for its empty block.

import Foundation

/// `person_idx` of the placeholder block written for a frame with nobody in it.
public let NO_PERSON_IDX = -1

/// One CSV row. A `nil` point or confidence is written as an empty field.
public struct KeypointRow: Equatable, Sendable {
    public var frameIdx: Int
    public var tMs: Int
    public var personIdx: Int
    public var joint: String
    public var point: PixelPoint?
    public var confidence: Double?
    public var rotated: Bool
    public var detected: Bool

    public init(
        frameIdx: Int,
        tMs: Int,
        personIdx: Int,
        joint: String,
        point: PixelPoint?,
        confidence: Double?,
        rotated: Bool,
        detected: Bool
    ) {
        self.frameIdx = frameIdx
        self.tMs = tMs
        self.personIdx = personIdx
        self.joint = joint
        self.point = point
        self.confidence = confidence
        self.rotated = rotated
        self.detected = detected
    }

    /// The placeholder block for a frame nobody was detected in: one row per
    /// joint, no coordinates, `person_idx = NO_PERSON_IDX`.
    public static func missingFrame(
        frameIdx: Int,
        tMs: Int,
        joints: [String],
        rotated: Bool
    ) -> [KeypointRow] {
        joints.map { joint in
            KeypointRow(
                frameIdx: frameIdx,
                tMs: tMs,
                personIdx: NO_PERSON_IDX,
                joint: joint,
                point: nil,
                confidence: nil,
                rotated: rotated,
                detected: false
            )
        }
    }

    /// The row as a CSV line, without the trailing newline.
    public var csvLine: String {
        [
            String(frameIdx),
            String(tMs),
            String(personIdx),
            Self.csvField(joint),
            Self.formatCoordinate(point?.x),
            Self.formatCoordinate(point?.y),
            Self.formatConfidence(confidence),
            rotated ? "true" : "false",
            detected ? "true" : "false",
        ].joined(separator: ",")
    }

    /// Pixel coordinates to 4 decimals: a hundredth of a pixel is far finer
    /// than a pose model's accuracy.
    private static func formatCoordinate(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "" }
        return String(format: "%.4f", value)
    }

    /// Confidences to 6 decimals — they are the finest number in the file and
    /// a 0.4/0.8 threshold should not be sitting on the rounding.
    private static func formatConfidence(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "" }
        return String(format: "%.6f", value)
    }

    /// The header line of the CSV.
    public static let header = "frame_idx,t_ms,person_idx,joint,x,y,confidence,rotated,detected"

    /// Quote a field if it contains a comma, a quote or a newline. The joint
    /// names never do, but the reader on the other side is `csv.DictReader` and
    /// this is what keeps a future name from breaking the file.
    static func csvField(_ value: String) -> String {
        guard value.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" || $0 == "\r" }) else {
            return value
        }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }
}
