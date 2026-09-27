// The CSV the runner writes.

import XCTest

@testable import VisionPoseCore

final class KeypointRowTests: XCTestCase {
    func testTheHeader() {
        XCTAssertEqual(
            KeypointRow.header,
            "frame_idx,t_ms,person_idx,joint,x,y,confidence,rotated,detected"
        )
    }

    func testADetectedRow() {
        let row = KeypointRow(
            frameIdx: 7,
            tMs: 234,
            personIdx: 1,
            joint: "left_wrist",
            point: PixelPoint(x: 310.125, y: 596.75),
            confidence: 0.4375,
            rotated: true,
            detected: true
        )
        XCTAssertEqual(row.csvLine, "7,234,1,left_wrist,310.1250,596.7500,0.437500,true,true")
    }

    /// The empty block of a frame nobody was detected in: one row per joint,
    /// no coordinates, `person_idx = -1`, `detected = false`. Every frame is
    /// therefore represented exactly once.
    func testAMissingFrameHasOneEmptyRowPerJoint() {
        let rows = KeypointRow.missingFrame(
            frameIdx: 3, tMs: 100, joints: VisionJoint.columnNames, rotated: false
        )
        XCTAssertEqual(rows.count, 19)
        for row in rows {
            XCTAssertEqual(row.personIdx, NO_PERSON_IDX)
            XCTAssertFalse(row.detected)
            XCTAssertNil(row.point)
            XCTAssertNil(row.confidence)
        }
        XCTAssertEqual(rows[0].csvLine, "3,100,-1,nose,,,,false,false")
        XCTAssertEqual(rows[1].csvLine, "3,100,-1,left_eye,,,,false,false")
        // `rotated` still describes the frame, so a consumer can see the
        // decision that was made for a frame nothing was found in.
        let rotated = KeypointRow.missingFrame(
            frameIdx: 4, tMs: 133, joints: ["nose"], rotated: true
        )
        XCTAssertEqual(rotated[0].csvLine, "4,133,-1,nose,,,,true,false")
    }

    func testBooleansAreLowercase() {
        let row = KeypointRow(
            frameIdx: 0, tMs: 0, personIdx: 0, joint: "nose",
            point: nil, confidence: nil, rotated: false, detected: false
        )
        XCTAssertTrue(row.csvLine.hasSuffix("false,false"))
    }

    /// Coordinates at 4 decimals, confidences at 6: finer than either is
    /// measured in, and it keeps the file readable.
    func testTheNumberFormat() {
        let row = KeypointRow(
            frameIdx: 0, tMs: 0, personIdx: 0, joint: "nose",
            point: PixelPoint(x: 0.123456789, y: 575.5),
            confidence: 0.123456789,
            rotated: false, detected: true
        )
        XCTAssertEqual(row.csvLine, "0,0,0,nose,0.1235,575.5000,0.123457,false,true")
    }

    /// A non-finite value is written as an empty field rather than as `nan`,
    /// which `csv.DictReader` would hand to pandas as a string.
    func testNonFiniteValuesBecomeEmptyFields() {
        let row = KeypointRow(
            frameIdx: 0, tMs: 0, personIdx: 0, joint: "nose",
            point: PixelPoint(x: .nan, y: .infinity),
            confidence: .nan, rotated: false, detected: true
        )
        XCTAssertEqual(row.csvLine, "0,0,0,nose,,,,false,true")
    }

    func testAFieldWithASeparatorIsQuoted() {
        XCTAssertEqual(KeypointRow.csvField("left_wrist"), "left_wrist")
        XCTAssertEqual(KeypointRow.csvField("a,b"), "\"a,b\"")
        XCTAssertEqual(KeypointRow.csvField("a\"b"), "\"a\"\"b\"")
        XCTAssertEqual(KeypointRow.csvField("a\nb"), "\"a\nb\"")
    }
}
