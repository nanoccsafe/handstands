@testable import HandstandCore

/// The one comparison every golden parity test runs: a fixture's `expected`
/// against what the Swift chain answered, at `meta.tolerances`.
///
/// Chainlink #42's refactor: the sections each per-stage test used to compare
/// inline (`PostProcessTests`, `PhasesTests`, `FeaturesTests`, `ScorerTests`)
/// moved here, so the rules live in exactly one place and cannot drift apart
/// between them —
///
/// * a **category** (a boolean `valid`/`filled`, a phase label, a hold number,
///   `balance_zone`, an integer, a `null`) is compared exactly whatever the
///   tolerance says;
/// * a **number** is compared at the tolerance `meta.tolerances` declares for
///   its own dotted path, through `tolerance(for:in:)` — the scorer's per-field
///   lookup included, whose unit test stays with it in `ScorerTests`;
/// * `null` on the Python side ⇔ `nil`/`NaN` on this one: a missing value is
///   never a value, and a `NaN` never matches a number.
///
/// Each function covers one section of `expected` and returns the mismatches
/// as **strings** — case, frame or hold, key, both values and the tolerance —
/// and leaves the reporting to its caller: the per-stage tests `XCTFail` each
/// string, `AnalyzerTests` gathers all five sections of one chain run, and
/// `RealParityTests` counts them per clip before failing once. No tolerance
/// here is looser than the one the per-stage tests declared; the strings are
/// the same messages they used to build, spelled once.
enum GoldenComparison {
    // MARK: - The sections of `expected`

    /// `expected.postprocess`: the body length at
    /// `meta.tolerances["expected.postprocess.body_length"]`, `valid` and
    /// `filled` exactly (booleans are categories, not quantities), and the
    /// joint positions at `meta.tolerances["expected.postprocess.frames.joints"]`.
    static func postprocess(
        _ processed: ProcessedClip, fixture: GoldenFixtures.Fixture
    ) -> [String] {
        let name = fixture.meta.caseName
        let expected = fixture.expected.postprocess
        guard let bodyTolerance = fixture.meta.tolerances["expected.postprocess.body_length"]
        else {
            return ["\(name): meta.tolerances has no body_length entry"]
        }
        guard let jointTolerance = fixture.meta.tolerances["expected.postprocess.frames.joints"]
        else {
            return ["\(name): meta.tolerances has no frames.joints entry"]
        }
        var out: [String] = []

        // The body length.
        if processed.bodyLength.usable != expected.bodyLength.usable {
            out.append(
                mismatch(
                    "\(name): body_length.usable",
                    want: "\(expected.bodyLength.usable)",
                    got: "\(processed.bodyLength.usable)",
                    tolerance: 0))
        }
        if processed.bodyLength.reason != expected.bodyLength.reason {
            out.append(
                mismatch(
                    "\(name): body_length.reason",
                    want: quote(expected.bodyLength.reason),
                    got: quote(processed.bodyLength.reason),
                    tolerance: 0))
        }
        let parts: [(String, Double?, Double?)] = [
            ("torso_px", processed.bodyLength.torsoPx, expected.bodyLength.torsoPx),
            ("thigh_px", processed.bodyLength.thighPx, expected.bodyLength.thighPx),
            ("shin_px", processed.bodyLength.shinPx, expected.bodyLength.shinPx),
            ("total_px", processed.bodyLength.totalPx, expected.bodyLength.totalPx),
        ]
        for (label, mine, theirs) in parts {
            switch (mine, theirs) {
            case let (got?, want?):
                if differs(got, want, tolerance: bodyTolerance) {
                    out.append(
                        mismatch(
                            "\(name): body_length.\(label)",
                            want: "\(want)", got: "\(got)", tolerance: bodyTolerance))
                }
            case (nil, nil):
                continue
            default:
                out.append(
                    mismatch(
                        "\(name): body_length.\(label)",
                        want: number(theirs), got: number(mine), tolerance: bodyTolerance))
            }
        }
        if processed.bodyLength.frames != expected.bodyLength.frames {
            out.append(
                mismatch(
                    "\(name): body_length frames",
                    want: "\(expected.bodyLength.frames)",
                    got: "\(processed.bodyLength.frames)",
                    tolerance: 0))
        }

        // valid and filled exactly, then the positions at the joint tolerance.
        if processed.frames.count != expected.frames.count {
            out.append(
                "\(name): frame count: Python \(expected.frames.count), "
                    + "Swift \(processed.frames.count)")
        }
        let count = min(processed.frames.count, expected.frames.count)
        for index in 0..<count {
            let mine = processed.frames[index]
            let row = expected.frames[index]
            for (column, jointName) in fixture.meta.jointNames.enumerated() {
                guard let joint = Joint(rawValue: jointName) else {
                    out.append("\(name): frame \(index): unknown joint \(jointName)")
                    continue
                }
                let at = "\(name): frame \(index) \(jointName)"
                if mine.valid.contains(joint) != row.valid[column] {
                    out.append(
                        mismatch(
                            "\(at) valid",
                            want: "\(row.valid[column])",
                            got: "\(mine.valid.contains(joint))",
                            tolerance: 0))
                }
                if mine.filled.contains(joint) != row.filled[column] {
                    out.append(
                        mismatch(
                            "\(at) filled",
                            want: "\(row.filled[column])",
                            got: "\(mine.filled.contains(joint))",
                            tolerance: 0))
                }

                // The position, at the joint tolerance.
                guard let pair = row.joints[jointName], pair.count == 2 else {
                    out.append("\(at): expected position is not [x, y]")
                    continue
                }
                let wantX = pair[0]
                let wantY = pair[1]
                guard row.valid[column] else {
                    if let point = mine.joints[joint] {
                        out.append(
                            "\(at): invalid in Python, but Swift has "
                                + "(\(point.x), \(point.y))")
                    }
                    continue
                }
                guard let point = mine.joints[joint] else {
                    out.append(
                        "\(at): Python has (\(number(wantX)), \(number(wantY))), Swift has none")
                    continue
                }
                for (axis, want, got) in [("x", wantX, point.x), ("y", wantY, point.y)] {
                    guard let want else {
                        out.append(
                            mismatch(
                                "\(at) \(axis)", want: "null", got: "\(got)",
                                tolerance: jointTolerance))
                        continue
                    }
                    if differs(got, want, tolerance: jointTolerance) {
                        out.append(
                            mismatch(
                                "\(at) \(axis)",
                                want: "\(want)", got: "\(got)", tolerance: jointTolerance))
                    }
                }
            }
        }
        return out
    }

    /// `expected.phases`: `phase` and `hold_id` frame by frame at **zero**
    /// tolerance (both are categories), plus `holdCount` with
    /// `meta.hold_count`. A mismatch carries the frame's `t_ms` and that
    /// frame's signals, which is how a flipped threshold shows itself.
    static func phases(_ phases: ClipPhases, fixture: GoldenFixtures.Fixture) -> [String] {
        let name = fixture.meta.caseName
        let rows = fixture.expected.phases
        let phaseTolerance = fixture.meta.tolerances["expected.phases.phase"] ?? 0
        let holdTolerance = fixture.meta.tolerances["expected.phases.hold_id"] ?? 0
        var out: [String] = []

        if phases.phase.count != rows.count {
            out.append("\(name): frame count: Python \(rows.count), Swift \(phases.phase.count)")
        }
        let count = min(phases.phase.count, rows.count)
        for index in 0..<count {
            let want = rows[index]
            let gotPhase = phases.phase[index].rawValue
            let gotHold = phases.holdId[index]
            guard gotPhase != want.phase || gotHold != want.holdId else { continue }
            let tMs = index < fixture.input.count ? fixture.input[index].tMs : -1
            let answers = "Python \(want.phase)/\(want.holdId), Swift \(gotPhase)/\(gotHold)"
            let signals = describe(signals: phases.signals, at: index)
            out.append(
                "\(name): frame \(index) t_ms=\(tMs): \(answers) "
                    + "(tolerance \(phaseTolerance)/\(holdTolerance))\nsignals: \(signals)")
        }
        if phases.holdCount != fixture.meta.holdCount {
            out.append(
                mismatch(
                    "\(name): meta.hold_count",
                    want: "\(fixture.meta.holdCount)",
                    got: "\(phases.holdCount)",
                    tolerance: 0))
        }
        return out
    }

    /// `expected.features`: every column of every frame at
    /// `meta.tolerances["expected.features"]`, with `null` ⇔ `NaN` and
    /// `balance_zone` compared exactly — it is a label, not a quantity.
    static func features(_ features: ClipFeatures, fixture: GoldenFixtures.Fixture) -> [String] {
        let name = fixture.meta.caseName
        guard let tolerance = fixture.meta.tolerances["expected.features"] else {
            return ["\(name): meta.tolerances has no features entry"]
        }
        let expected = fixture.expected.features
        var out: [String] = []

        if features.frames != expected.count {
            out.append(
                "\(name): feature rows: Python \(expected.count), Swift \(features.frames)")
        }
        let count = min(features.frames, expected.count)
        for index in 0..<count {
            let row = expected[index]
            for column in fixture.meta.featureColumns {
                let want = row[column] ?? .missing
                if column == "balance_zone" {
                    let zone = features.balanceZone[index]
                    switch want {
                    case .text(let word):
                        if zone?.rawValue != word {
                            out.append(
                                mismatch(
                                    "\(name): frame \(index) column balance_zone",
                                    want: "'\(word)'",
                                    got: zone.map { "'\($0.rawValue)'" } ?? "nil",
                                    tolerance: 0))
                        }
                    case .missing:
                        if let zone {
                            out.append(
                                mismatch(
                                    "\(name): frame \(index) column balance_zone",
                                    want: "null", got: "\(zone)", tolerance: 0))
                        }
                    case .number:
                        out.append(
                            "\(name): frame \(index) column balance_zone: "
                                + "the fixture has a number where the port has a zone")
                    }
                    continue
                }
                let mine = features.value(column)
                guard index < mine.count else {
                    out.append("\(name): frame \(index): Swift has no column \(column)")
                    continue
                }
                let label = "\(name): frame \(index) column \(column)"
                switch want {
                case .number(let expectedValue):
                    if !differs(mine[index], expectedValue, tolerance: tolerance) {
                        continue
                    }
                    out.append(
                        mismatch(
                            label,
                            want: "\(expectedValue)",
                            got: mine[index].isNaN ? "NaN" : "\(mine[index])",
                            tolerance: tolerance))
                case .missing:
                    if !mine[index].isNaN {
                        out.append(
                            mismatch(
                                label, want: "null", got: "\(mine[index])", tolerance: tolerance))
                    }
                case .text(let word):
                    out.append(
                        mismatch(
                            label, want: "text '\(word)'", got: "\(mine[index])",
                            tolerance: tolerance))
                }
            }
        }
        return out
    }

    /// `expected.hold_summary` — `Features.holdRows(_:)`' answer, one row per
    /// hold: the integer fields exact, `hold_duration_s` and every stat at
    /// `meta.tolerances["expected.hold_summary"]`, `null` ⇔ `NaN`, and the
    /// keys Swift writes equal to the keys the fixture stores (so nothing is
    /// left un-compared either way).
    static func holdSummary(
        _ rows: [HoldSummaryRow], fixture: GoldenFixtures.Fixture
    ) -> [String] {
        let name = fixture.meta.caseName
        guard let tolerance = fixture.meta.tolerances["expected.hold_summary"] else {
            return ["\(name): meta.tolerances has no hold_summary entry"]
        }
        let expected = fixture.expected.holdSummary
        var out: [String] = []

        if rows.count != expected.count {
            out.append("\(name): hold rows: Python \(expected.count), Swift \(rows.count)")
        }
        let exactFields: Set<String> = [
            "hold_id", "hold_frames", "valid_frames", "hold_start_ms", "hold_end_ms",
        ]
        let nonStats: Set<String> = exactFields.union(["clip_id", "source", "hold_duration_s"])
        let count = min(rows.count, expected.count)
        for index in 0..<count {
            let want = expected[index]
            let row = rows[index]
            let label = "\(name): hold \(index) (\(row.holdId))"

            for field in exactFields.sorted() {
                guard let value = want[field] else {
                    out.append("\(label): fixture row has no \(field)")
                    continue
                }
                guard case .number(let number) = value else {
                    out.append("\(label): \(field) is not a number in the fixture")
                    continue
                }
                let mine: Int
                switch field {
                case "hold_id": mine = row.holdId
                case "hold_frames": mine = row.holdFrames
                case "valid_frames": mine = row.validFrames
                case "hold_start_ms": mine = row.holdStartMs
                default: mine = row.holdEndMs
                }
                if mine != Int(number) {
                    out.append(
                        mismatch(
                            "\(label): column \(field)",
                            want: "\(Int(number))", got: "\(mine)", tolerance: 0))
                }
            }

            if let value = want["hold_duration_s"] {
                guard case .number(let expectedValue) = value else {
                    out.append("\(label): hold_duration_s is not a number")
                    continue
                }
                if differs(row.holdDurationS, expectedValue, tolerance: tolerance) {
                    out.append(
                        mismatch(
                            "\(label): column hold_duration_s",
                            want: "\(expectedValue)",
                            got: "\(row.holdDurationS)",
                            tolerance: tolerance))
                }
            }

            // Every other key of the row is a stat, and every stat is one of
            // those keys: nothing is left un-compared either way.
            var comparedStats = Set<String>()
            for (key, value) in want where !nonStats.contains(key) {
                comparedStats.insert(key)
                guard let mine = row.stats[key] else {
                    out.append("\(label): Swift has no stat \(key)")
                    continue
                }
                switch value {
                case .number(let expectedValue):
                    if !differs(mine, expectedValue, tolerance: tolerance) {
                        continue
                    }
                    out.append(
                        mismatch(
                            "\(label): column \(key)",
                            want: "\(expectedValue)",
                            got: mine.isNaN ? "NaN" : "\(mine)",
                            tolerance: tolerance))
                case .missing:
                    if !mine.isNaN {
                        out.append(
                            mismatch(
                                "\(label): column \(key)",
                                want: "null", got: "\(mine)", tolerance: tolerance))
                    }
                case .text(let word):
                    out.append(
                        mismatch(
                            "\(label): column \(key)",
                            want: "text '\(word)'", got: "\(mine)", tolerance: tolerance))
                }
            }
            let missing = comparedStats.subtracting(Set(row.stats.keys)).sorted()
            let extra = Set(row.stats.keys).subtracting(comparedStats).sorted()
            if !missing.isEmpty || !extra.isEmpty {
                out.append(
                    "\(label): the stats Swift writes are not the fixture's columns "
                        + "(missing=\(missing), extra=\(extra))")
            }
        }
        return out
    }

    /// `expected.score`: the scorer's whole document, walked leaf by leaf at
    /// `meta.tolerances` — the per-field lookup of #41 included — with
    /// integers, strings, `missing_groups` and the top-fault *names* exact
    /// and `null` ⇔ `nil`/`NaN`. This is the walk `tests/test_golden.py`
    /// does, returning the mismatches instead of failing at the first one.
    static func score(
        _ scores: [HoldScore], clipHoldId: Int?, fixture: GoldenFixtures.Fixture
    ) -> [String] {
        compareJSON(
            stored: fixture.expected.score,
            mine: scoreJSON(scores, clipHoldId: clipHoldId),
            path: "expected.score",
            tolerances: fixture.meta.tolerances,
            label: fixture.meta.caseName
        )
    }

    // MARK: - The scorer's answer as the fixture writes it

    /// One scorer answer as the fixture's JSON shape:
    /// `{"holds": [...], "clip_hold_id": ...}`, with `NaN` written the way
    /// Python writes it — `null`.
    static func scoreJSON(_ scores: [HoldScore], clipHoldId: Int?) -> GoldenFixtures.JSONValue {
        let holds: [GoldenFixtures.JSONValue] = scores.map { hold -> GoldenFixtures.JSONValue in
            var groups: [String: GoldenFixtures.JSONValue] = [:]
            for (name, value) in hold.groups {
                groups[name] = numberJSON(value ?? .nan)
            }
            let faults: [GoldenFixtures.JSONValue] = hold.topFaults.map { fault in
                GoldenFixtures.JSONValue.array([
                    .text(fault.feature),
                    numberJSON(fault.value),
                    numberJSON(fault.mean),
                    numberJSON(fault.z),
                ])
            }
            return .object([
                "hold_id": .number(Double(hold.holdId)),
                "hold_frames": .number(Double(hold.holdFrames)),
                "valid_frames": .number(Double(hold.validFrames)),
                "hold_start_ms": .number(Double(hold.holdStartMs)),
                "hold_end_ms": .number(Double(hold.holdEndMs)),
                "hold_duration_s": numberJSON(hold.holdDurationS),
                "score": numberJSON(hold.score ?? .nan),
                "reason": .text(hold.reason),
                "deviation": numberJSON(hold.deviation),
                "penalty": numberJSON(hold.penalty),
                "groups": .object(groups),
                "values": .object(hold.values.mapValues { numberJSON($0) }),
                "z": .object(hold.z.mapValues { numberJSON($0) }),
                "top_faults": .array(faults),
                "missing_groups": .array(
                    hold.missingGroups.map { GoldenFixtures.JSONValue.text($0) }),
            ])
        }
        return .object([
            "holds": .array(holds),
            "clip_hold_id": numberJSON(clipHoldId.map { Double($0) }),
        ])
    }

    /// One number as JSON: a non-finite one — or no number at all — is
    /// Python's `null`.
    static func numberJSON(_ value: Double?) -> GoldenFixtures.JSONValue {
        guard let value, value.isFinite else { return .null }
        return .number(value)
    }

    /// Every place `mine` differs from `stored` past the field's tolerance —
    /// the recursive walk, collecting rather than reporting.
    private static func compareJSON(
        stored: GoldenFixtures.JSONValue,
        mine: GoldenFixtures.JSONValue,
        path: String,
        tolerances: [String: Double],
        label: String
    ) -> [String] {
        var out: [String] = []
        switch (stored, mine) {
        case (.null, .null):
            return out
        case (.null, _):
            return ["\(label): \(path): Python null, Swift \(describe(mine))"]
        case (_, .null):
            return ["\(label): \(path): Python \(describe(stored)), Swift null"]
        case (.text(let want), .text(let got)):
            if want != got {
                out.append("\(label): \(path): Python '\(want)', Swift '\(got)'")
            }
        case (.bool(let want), .bool(let got)):
            if want != got {
                out.append("\(label): \(path): Python \(want), Swift \(got)")
            }
        case (.number(let want), .number(let got)):
            let tolerance = tolerance(for: path, in: tolerances)
            if abs(want - got) > tolerance {
                out.append(
                    "\(label): \(path): Python \(want), Swift \(got) (tolerance \(tolerance))")
            }
        case (.object(let want), .object(let got)):
            let missing = want.keys.filter { got[$0] == nil }.sorted()
            let extra = got.keys.filter { want[$0] == nil }.sorted()
            if !missing.isEmpty || !extra.isEmpty {
                out.append(
                    "\(label): \(path): keys differ (missing=\(missing), extra=\(extra))")
            }
            for (key, value) in want {
                if let other = got[key] {
                    out.append(
                        contentsOf: compareJSON(
                            stored: value, mine: other, path: "\(path).\(key)",
                            tolerances: tolerances, label: label))
                }
            }
        case (.array(let want), .array(let got)):
            if want.count != got.count {
                out.append(
                    "\(label): \(path): \(want.count) entries stored, \(got.count) recomputed")
                return out
            }
            for (index, item) in want.enumerated() {
                out.append(
                    contentsOf: compareJSON(
                        stored: item, mine: got[index], path: "\(path).\(index)",
                        tolerances: tolerances, label: label))
            }
        default:
            out.append(
                "\(label): \(path): Python \(describe(stored)), Swift \(describe(mine))")
        }
        return out
    }

    // MARK: - The tolerance lookup

    /// The tolerance that applies to `path`.
    ///
    /// Under `expected.score` the **field** names the key of
    /// `meta.tolerances`: `values.*` is `expected.score.values`, `z.*` is
    /// `expected.score.z`, `groups.*` is `expected.score.groups`, a top
    /// fault's value and mean are values and its z is a z, `deviation`,
    /// `penalty` and `score` are their own keys, and the integer hold fields
    /// are exact — like the strings, which never ask for a tolerance at all.
    /// A plain prefix match cannot do that: the `holds.<n>` (and list-index)
    /// segments in the middle mean `expected.score.score` is no prefix of
    /// `expected.score.holds.0.score`, which is how every hold field used to
    /// fall back to `default` and the declared per-field tolerances were
    /// never applied.
    ///
    /// Anywhere else — `expected.features.12.com_u` and friends — the longest
    /// key that prefixes the path wins, list indices dropped first, as
    /// `tolerance_for` in `tests/test_golden.py` does.
    static func tolerance(for path: String, in tolerances: [String: Double]) -> Double {
        let segments = path.split(separator: ".")
        if segments.count >= 3, segments[0] == "expected", segments[1] == "score" {
            // `holds.<n>.<field>…` under the section, or a field of the
            // section itself (`clip_hold_id`).
            let rest = Array(segments.dropFirst(2))
            let field =
                rest.first == "holds" && rest.count >= 3 ? Array(rest.dropFirst(2)) : rest
            if let declared = scoreTolerance(field, in: tolerances) {
                return declared
            }
        }
        let parts = segments.filter { segment in !segment.allSatisfy { $0.isNumber } }
        var best: [Substring] = []
        var value = tolerances["default"] ?? 1e-6
        for (key, tolerance) in tolerances {
            let keyParts = key.split(separator: ".")
            guard keyParts.count >= best.count, parts.count >= keyParts.count else { continue }
            if Array(parts.prefix(keyParts.count)) == keyParts {
                best = keyParts
                value = tolerance
            }
        }
        return value
    }

    /// The tolerance `meta.tolerances` declares for one field of
    /// `expected.score`, or `nil` for a field with no key of its own — the
    /// caller then falls back to the longest prefixing key, i.e. `default`
    /// for every path this section has.
    static func scoreTolerance(_ field: [Substring], in tolerances: [String: Double]) -> Double? {
        guard let name = field.first else { return nil }
        switch name {
        case "values", "z", "groups":
            // `values.<feature>`, `z.<feature>`, `groups.<name>` — the
            // container itself is never a number.
            guard field.count >= 2 else { return nil }
            return tolerances["expected.score.\(name)"]
        case "deviation", "penalty", "score":
            return tolerances["expected.score.\(name)"]
        case "top_faults":
            // `top_faults.<fault>.<element>`: 0 is the feature name, 1 the
            // value, 2 the mean, 3 the z — the fixture's four-tuple. Value
            // and mean are measured like values, the z like a z.
            guard field.count >= 3 else { return nil }
            switch field[2] {
            case "1", "2": return tolerances["expected.score.values"]
            case "3": return tolerances["expected.score.z"]
            default: return nil
            }
        case "hold_id", "hold_frames", "valid_frames", "hold_start_ms", "hold_end_ms",
            "clip_hold_id":
            return 0.0  // integers: exact, no decimal place to be lenient about
        default:
            return nil
        }
    }

    // MARK: - The messages

    /// One mismatch as this file spells it: the place (case, frame or hold,
    /// key), then both values and the tolerance they were compared at.
    static func mismatch(
        _ label: String, want: String, got: String, tolerance: Double
    ) -> String {
        "\(label): Python \(want), Swift \(got) (tolerance \(tolerance))"
    }

    /// Two numbers apart by more than `tolerance` — or either side not a
    /// number at all, which `XCTAssertEqual(_:accuracy:)` counts as a
    /// failure too. The written side is always finite (Python writes `null`
    /// for NaN), so a `NaN` here is always Swift's.
    static func differs(_ mine: Double, _ want: Double, tolerance: Double) -> Bool {
        guard mine.isFinite, want.isFinite else { return true }
        return abs(mine - want) > tolerance
    }

    /// A double as a message prints it: `nil` for no number at all.
    static func number(_ value: Double?) -> String {
        value.map { "\($0)" } ?? "nil"
    }

    /// A string as a message prints it: `nil` for absent, quoted otherwise.
    static func quote(_ value: String?) -> String {
        value.map { "'\($0)'" } ?? "nil"
    }

    /// One JSON value in `null`/number/string form, for a failure message.
    static func describe(_ value: GoldenFixtures.JSONValue) -> String {
        switch value {
        case .null: return "null"
        case .bool(let flag): return "\(flag)"
        case .number(let number): return "\(number)"
        case .text(let text): return "'\(text)'"
        case .array: return "[…]"
        case .object: return "{…}"
        }
    }

    /// One frame's signals as a single line, for a phase mismatch — which
    /// signal crossed its threshold is the whole question there.
    static func describe(signals: FrameSignals, at index: Int) -> String {
        guard index < signals.frames else { return "no signals for this frame" }
        return """
            known=\(signals.known[index]) reason=\(signals.unknownReason[index]) \
            v_ankle=\(signals.vAnkleMid[index]) u_ankle=\(signals.uAnkleMid[index]) \
            v_hip=\(signals.vHipMid[index]) angle=\(signals.bodyAngleDeg[index]) \
            inverted=\(signals.inverted[index]) hands_low=\(signals.handsLow[index]) \
            wrist_speed=\(signals.wristSpeedLPerS[index]) wrist_step=\(signals.wristStepL[index]) \
            hands_down=\(signals.handsDown[index]) hand_step=\(signals.handStep[index]) \
            legs_velocity=\(signals.legsVelocityLPerS[index]) \
            legs_rising=\(signals.legsRising[index]) legs_falling=\(signals.legsFalling[index])
            """
    }
}
