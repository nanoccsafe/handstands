import Foundation

import HandstandCore

/// The golden fixture schema of chainlink #25, shared by every test that
/// reads `Fixtures/golden/*.json`: the `meta`, `input` and `expected`
/// sections exactly as `pipeline/handstand/golden.py` writes them, plus the
/// `Bundle.module` loading the test target does for all of them.
///
/// One definition, used by the smoke test (`GoldenFixtureTests`) and by the
/// ports' parity tests (`PostProcessTests`, later phases/features): a change
/// to the fixture contract must break both at once rather than let each test
/// invent its own reading of the file. The folder's `README.md` documents the
/// format from the Python side.
enum GoldenFixtures {
    /// A named fixture that is not in `Bundle.module`.
    struct FixtureMissing: Error, CustomStringConvertible {
        let name: String
        var description: String { "\(name).json is missing from Bundle.module" }
    }

    // MARK: - The fixture schema, as Swift sees it

    struct Meta: Decodable {
        struct Display: Decodable {
            let width: Int
            let height: Int
        }

        /// `meta.case` — the case's name, and the file's own name.
        let caseName: String
        let mode: String
        let generatorVersion: String
        let seed: Int?
        let display: Display?
        let frameCount: Int
        let holdCount: Int
        /// The joints, **in the order every per-joint array in the file uses**.
        let jointNames: [String]
        let featureColumns: [String]
        let tolerances: [String: Double]

        enum CodingKeys: String, CodingKey {
            case caseName = "case"
            case mode
            case generatorVersion = "generator_version"
            case seed
            case display
            case frameCount = "frame_count"
            case holdCount = "hold_count"
            case jointNames = "joint_names"
            case featureColumns = "feature_columns"
            case tolerances
        }
    }

    struct InputFrame: Decodable {
        let frameIdx: Int
        let tMs: Int
        let detected: Bool
        let trainerContact: Bool
        /// `name -> [x, y, visibility]`, all three `null` on a frame nobody was
        /// detected in: the schema's NaN, which JSON has to spell as null.
        let joints: [String: [Double?]]

        enum CodingKeys: String, CodingKey {
            case frameIdx = "frame_idx"
            case tMs = "t_ms"
            case detected
            case trainerContact = "trainer_contact"
            case joints
        }
    }

    struct PostFrame: Decodable {
        /// One boolean per joint, in `meta.joint_names` order — not keyed by
        /// name, because gating, outliers and gap fill act on a joint.
        let valid: [Bool]
        let filled: [Bool]
        /// `name -> [x, y]`, both `null` where `valid` is false.
        let joints: [String: [Double?]]
    }

    struct BodyLength: Decodable {
        let usable: Bool
        /// `null` when the clip is usable — Python's empty-string `reason`.
        let reason: String?
        let torsoPx: Double?
        let thighPx: Double?
        let shinPx: Double?
        let totalPx: Double?
        /// How many frames each part was measurable on: `torso`, `thigh`, `shin`.
        let frames: [String: Int]

        enum CodingKeys: String, CodingKey {
            case usable
            case reason
            case torsoPx = "torso_px"
            case thighPx = "thigh_px"
            case shinPx = "shin_px"
            case totalPx = "total_px"
            case frames
        }
    }

    struct PhaseRow: Decodable {
        let phase: String
        let holdId: Int

        enum CodingKeys: String, CodingKey {
            case phase
            case holdId = "hold_id"
        }
    }

    /// One value of a feature or hold-summary row: a number, a label such as
    /// `balance_zone`, or `null` where the frame could not support the column.
    enum Value: Decodable {
        case number(Double)
        case text(String)
        case missing

        init(from decoder: any Decoder) throws {
            let container = try decoder.singleValueContainer()
            if container.decodeNil() {
                self = .missing
            } else if let number = try? container.decode(Double.self) {
                self = .number(number)
            } else {
                self = .text(try container.decode(String.self))
            }
        }
    }

    /// A whole JSON document as a tree, for the sections too deep to give a
    /// `Decodable` shape to without writing that shape down twice —
    /// `expected.score`, which the scorer's parity test walks field by field
    /// against `meta.tolerances` the way Python's `tests/test_golden.py`
    /// does.
    enum JSONValue: Decodable, Equatable {
        case null
        case bool(Bool)
        case number(Double)
        case text(String)
        case array([JSONValue])
        case object([String: JSONValue])

        init(from decoder: any Decoder) throws {
            let container = try decoder.singleValueContainer()
            if container.decodeNil() {
                self = .null
            } else if let value = try? container.decode(Bool.self) {
                // Only JSON `true`/`false` decode as a Bool: `JSONSerialization`
                // (and the decoder built on it) refuses a number here, so `1`
                // falls through to the number case.
                self = .bool(value)
            } else if let value = try? container.decode(Double.self) {
                self = .number(value)
            } else if let value = try? container.decode(String.self) {
                self = .text(value)
            } else if let value = try? container.decode([JSONValue].self) {
                self = .array(value)
            } else if let value = try? container.decode([String: JSONValue].self) {
                self = .object(value)
            } else {
                throw DecodingError.dataCorruptedError(
                    in: container, debugDescription: "unsupported JSON value")
            }
        }
    }

    struct Expected: Decodable {
        struct Postprocess: Decodable {
            let bodyLength: BodyLength
            let frames: [PostFrame]

            enum CodingKeys: String, CodingKey {
                case bodyLength = "body_length"
                case frames
            }
        }

        let postprocess: Postprocess
        let phases: [PhaseRow]
        let features: [[String: Value]]
        let holdSummary: [[String: Value]]
        /// `{"holds": [...], "clip_hold_id": ...}` — the scorer's answers
        /// (#41), read as a tree because every leaf is a number, a string, a
        /// `null` or a nested container.
        let score: JSONValue

        enum CodingKeys: String, CodingKey {
            case postprocess
            case phases
            case features
            case holdSummary = "hold_summary"
            case score
        }
    }

    struct Fixture: Decodable {
        let meta: Meta
        let input: [InputFrame]
        let expected: Expected
    }

    // MARK: - Loading

    /// The one reference the fixtures' `expected.score` was scored against,
    /// beside them in `Fixtures/golden/` — `parity_reference.json`.
    ///
    /// It is a reference file (`docs/scoring.md`'s schema v1), not a fixture
    /// document, so `urls()` excludes it and the scorer's parity test loads it
    /// by name through `parityReferenceData()`.
    static let parityReferenceName = "parity_reference"

    /// Every committed **fixture**, in file-name order so a failure names a
    /// stable list. `parity_reference.json` is not one — see above.
    static func urls() -> [URL] {
        let urls =
            Bundle.module.urls(forResourcesWithExtension: "json", subdirectory: "Fixtures/golden")
            ?? []
        return urls
            .filter { $0.lastPathComponent != "\(parityReferenceName).json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// The URL of one case, e.g. `url(for: "line_hold")`.
    static func url(for name: String) throws -> URL {
        guard let url = Bundle.module.url(
            forResource: name, withExtension: "json", subdirectory: "Fixtures/golden")
        else {
            throw FixtureMissing(name: name)
        }
        return url
    }

    static func load(_ url: URL) throws -> Fixture {
        try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    /// One case by name, e.g. `load("hand_step")`.
    static func load(_ name: String) throws -> Fixture {
        try load(url(for: name))
    }

    /// The bytes of `parity_reference.json`, for `ScoreReference.decode`.
    static func parityReferenceData() throws -> Data {
        guard let url = Bundle.module.url(
            forResource: parityReferenceName,
            withExtension: "json",
            subdirectory: "Fixtures/golden")
        else {
            throw FixtureMissing(name: parityReferenceName)
        }
        return try Data(contentsOf: url)
    }

    /// The fixture's `input` as the post-process sees it — the conversion
    /// every parity test and `AnalyzerTests` start from, spelled once:
    /// `[x, y, visibility]` with all three `null` is the schema's NaN (a
    /// joint that was not seen at all), and a joint with no entry in
    /// `joints` is the same thing.
    static func inputFrames(from fixture: Fixture) -> [PostProcessInputFrame] {
        fixture.input.map { frame in
            var joints: [Joint: Keypoint] = [:]
            for (name, values) in frame.joints {
                guard let joint = Joint(rawValue: name) else { continue }
                guard values.count >= 3, let x = values[0], let y = values[1],
                    let visibility = values[2]
                else { continue }
                joints[joint] = Keypoint(x: x, y: y, visibility: visibility)
            }
            return PostProcessInputFrame(
                tMs: frame.tMs,
                detected: frame.detected,
                trainerContact: frame.trainerContact,
                joints: joints
            )
        }
    }
}
