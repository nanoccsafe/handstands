import XCTest

@testable import HandstandApp
import HandstandCore

/// `ReferenceLoader` (chainlink #47, part D): where the scoring reference
/// comes from, which is nowhere by default — the real one is #28's user
/// data, never in this repo, so a build of this public tree has no
/// `reference-line.json` to find and must load `nil` rather than fail.
final class ReferenceLoaderTests: XCTestCase {
    // MARK: - No resource

    /// No bundled file is the normal state: `nil`, for every hold, from a
    /// bundle that (like every build of this repo) carries no reference.
    func testABundleWithNoReferenceLoadsNil() {
        let bundle = Bundle(for: ReferenceLoaderTests.self)
        XCTAssertNil(
            ReferenceLoader.load(for: .line, bundle: bundle),
            "no reference-line.json in the test bundle → no reference")
        XCTAssertNil(ReferenceLoader.load(for: .tuck, bundle: bundle))
    }

    // MARK: - Valid bytes

    func testValidReferenceBytesDecode() throws {
        let reference = try XCTUnwrap(
            ReferenceLoader.load(data: ParityReference.data),
            "the parity reference is a valid reference")

        XCTAssertEqual(reference.nHolds, 6)
        XCTAssertEqual(reference.features.count, 16, "every scored feature is there")
        XCTAssertEqual(reference.features["hip_angle"]?.mean ?? 0, 173.065833, accuracy: 1e-6)
        XCTAssertEqual(reference.features["hip_angle"]?.sd ?? 0, 10.245097, accuracy: 1e-6)
        XCTAssertEqual(reference.features["hip_angle"]?.n, 6)
        XCTAssertFalse(reference.builtFrom.isEmpty, "a reference says where its numbers came from")
    }

    // MARK: - Invalid bytes

    /// Invalid input returns `nil` — never throws, never crashes, whatever
    /// is wrong with the file: not JSON at all, JSON that is not an object,
    /// and an object that is not this schema.
    func testInvalidReferenceBytesLoadNil() {
        let bad: [String] = [
            "",  // no bytes at all
            "not json at all",
            "[1, 2, 3]",  // JSON, but not an object
            "{}",  // an object without the schema
            #"{"schema":"some-other-schema"}"#,
            #"{"schema":"handstand-reference","version":1,"signs":"athlete"}"#,  // no features
        ]
        for text in bad {
            XCTAssertNil(
                ReferenceLoader.load(data: Data(text.utf8)),
                "these bytes must read as no reference: \(text.prefix(40))")
        }
    }

    /// The same invalid bytes twice are logged once, not once per screen —
    /// the loader's only side effect must stay quiet. (Checked by not
    /// crashing and by the API contract: `load` takes no logger.)
    func testLoadingIsRepeatableAndPure() {
        let invalid = Data("{}".utf8)
        XCTAssertNil(ReferenceLoader.load(data: invalid))
        XCTAssertNil(ReferenceLoader.load(data: invalid))
        // And a valid decode gives an equal value every time.
        XCTAssertEqual(
            ReferenceLoader.load(data: ParityReference.data),
            ReferenceLoader.load(data: ParityReference.data)
        )
    }
}
