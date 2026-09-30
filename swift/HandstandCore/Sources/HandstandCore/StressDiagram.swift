import Foundation

// --------------------------------------------------------------------------- #
// The stress diagram (#48): what one frame of the overlay draws, as data.
//
// This file is the drawing's *logic* — which joints and bones exist, how far
// each of them is off, where the stack line and the centre of mass are — and
// nothing about pixels on a screen: the app turns a `DiagramFrame` into
// strokes and circles (`OverlayGeometry` places them over the video), so the
// whole thing is testable without a view, a player or a video.
//
// Side view only for now (every clip in the dataset is one): nothing in here
// says "side", so the front/back alignment of chainlink #83 can add its own
// features and bands without renaming a type.
//
// The ghost "ideal" skeleton is deliberately absent: it needs the user's
// reference skeleton (chainlink #28), which does not exist yet.
// `DiagramFrame.ideal` is the extension point — always `nil` until #28
// provides one, and never guessed at.
//
// Severity comes from two places, and which one is used depends on whether
// the app was built with a scoring reference:
//
// * **with a reference** — the same z-score the Scorer scores with:
//   `severity = min(|z|, Z_CAP) / Z_CAP` with `z = (value - mean) /
//   max(sd, SD floor)`, athlete-signed (the five `SIGNED_BY_FACING` features
//   multiplied by the frame's `facing_sign`, and skipped when that sign is
//   NaN). The floors are `Scorer.sdFloors` — the scorer's own table, not a
//   copy of it;
// * **without a reference** — `FeatureTolerances`, the port of Python's
//   `TOLERANCES` / `fails_tolerance`: 0 when the value passes the target,
//   `min(1, 0.3 + excess / band)` when it fails, where `excess` is how far
//   past the threshold it sits and `band` is 20 for degree features, 0.2 for
//   body-length features (the same bands `docs/ios.md` quotes).
//
// Outside a hold — or on a frame the features could not measure — there is
// no severity at all: every band is `.neutral` and the skeleton is drawn
// grey, because a warm-up is not a fault.
// --------------------------------------------------------------------------- #

/// Which side of a target is the wrong side — Python's `EITHER`, `BELOW` and
/// `ABOVE` of `handstand.features`, with the same spellings as raw values.
public enum FeatureSide: String, Sendable, CaseIterable {
    /// Wrong either way round: `abs(value) > threshold`, for the two shape
    /// features where bending one way is as much a fault as the other.
    case either
    /// Wrong only below the threshold: joint angles, where straight is 180°.
    case below
    /// Wrong only above the threshold.
    case above
}

/// One row of Python's `TOLERANCES`: the feature the target is about, the
/// number the test compares against, and which side of it fails —
/// `TOLERANCES`' `(feature, description, threshold, side)` minus the
/// description, which only the Python summary prints.
public struct FeatureTolerance: Sendable, Equatable {
    /// The feature's name, a `Features.all` / `ClipFeatures.value` column.
    public var feature: String
    /// The number the target is measured against, in the feature's own unit.
    public var threshold: Double
    /// Which side of `threshold` misses the target.
    public var side: FeatureSide

    public init(feature: String, threshold: Double, side: FeatureSide) {
        self.feature = feature
        self.threshold = threshold
        self.side = side
    }
}

/// Port of the targets half of `pipeline/handstand/features.py`: the
/// `TOLERANCES` table (same rows, same order, same numbers) and
/// `fails_tolerance`.
public enum FeatureTolerances {
    /// `TOLERANCES` — every feature's target, in Python's row order. The
    /// thresholds are the module constants themselves (`Features` already
    /// pins them against Python's), so the table and the docstring that
    /// quotes them cannot drift apart.
    public static let all: [FeatureTolerance] = [
        FeatureTolerance(
            feature: "line_deviation", threshold: Features.lineToleranceL, side: .either),
        FeatureTolerance(
            feature: "body_angle", threshold: Features.bodyAngleTolDeg, side: .either),
        FeatureTolerance(
            feature: "shoulder_angle", threshold: Features.openShoulderDeg, side: .below),
        FeatureTolerance(feature: "hip_angle", threshold: Features.pikeHipDeg, side: .below),
        FeatureTolerance(
            feature: "knee_angle", threshold: Features.straightKneeDeg, side: .below),
        FeatureTolerance(
            feature: "elbow_angle", threshold: Features.bentElbowDeg, side: .below),
        FeatureTolerance(
            feature: "banana", threshold: Features.bananaToleranceL, side: .either),
        FeatureTolerance(feature: "head", threshold: Features.headFlexionDeg, side: .either),
        FeatureTolerance(feature: "leg_separation", threshold: Features.splitDeg, side: .above),
    ]

    /// The target of one feature, or `nil` for a feature nobody wrote one
    /// for (`off_hip`, `hand_width`, the CoM columns) — those are simply not
    /// judged against a fixed number.
    public static func tolerance(for feature: String) -> FeatureTolerance? {
        all.first { $0.feature == feature }
    }

    /// Is one value outside its target? — `fails_tolerance`.
    ///
    /// `side` says which way is wrong, and an unmeasured value fails nothing:
    /// a hold the model could not see is not a hold that missed a target.
    public static func fails(_ value: Double, threshold: Double, side: FeatureSide) -> Bool {
        guard value.isFinite else { return false }
        switch side {
        case .either: return Swift.abs(value) > threshold
        case .below: return value < threshold
        case .above: return value > threshold
        }
    }

    /// `fails` through one table row.
    public static func fails(_ value: Double, _ tolerance: FeatureTolerance) -> Bool {
        fails(value, threshold: tolerance.threshold, side: tolerance.side)
    }

    /// How far past the threshold a failing value sits — what the diagram's
    /// `0.3 + excess / band` severity grows with. Only meaningful for a value
    /// that `fails` already said is failing; for one that passes it is
    /// negative or zero.
    public static func excess(_ value: Double, _ tolerance: FeatureTolerance) -> Double {
        switch tolerance.side {
        case .either: return Swift.abs(value) - tolerance.threshold
        case .below: return tolerance.threshold - value
        case .above: return value - tolerance.threshold
        }
    }
}

/// How far a measurement is off, on the diagram's own 0…1 scale: 0 is on the
/// target (or, with a reference, exactly at its mean) and 1 is as bad as it
/// gets (`Z_CAP` SDs off, or a failure past the severity band).
public enum Severity {
    /// The colour a severity is drawn in — `.ok < 0.25 <= .warn < 0.5 <=
    /// .bad`. `SeverityBand.neutral` is never returned: it means "not
    /// measured", which is a fact about the frame rather than about the
    /// number.
    public static func colourBand(_ severity: Double) -> SeverityBand {
        if severity < 0.25 { return .ok }
        if severity < 0.5 { return .warn }
        return .bad
    }
}

/// The four colours the diagram speaks: the three severities plus "nobody is
/// measuring this" — a frame outside a hold, or one the features could not
/// measure. Raw values are the swatch labels' own words, so a legend, a log
/// line and a test all spell the same band the same way.
public enum SeverityBand: String, Sendable, CaseIterable {
    /// Outside a hold, or unmeasured: drawn grey, no severity behind it.
    case neutral
    /// Inside a hold and on target.
    case ok
    /// Inside a hold and off, but not badly.
    case warn
    /// Inside a hold and badly off.
    case bad
}

/// One joint of the drawn skeleton: where it is in display pixels and how
/// far off it is. `severity` is `nil` exactly when `band` is `.neutral` or
/// `.ok` without a number (a joint in a hold no feature judges, like the
/// wrists).
public struct DiagramJoint: Sendable, Equatable {
    /// Which joint this is.
    public var joint: Joint
    /// Where it sits in display pixels (`processed.frames[i].joints`).
    public var point: Point2
    /// How far off it is, 0…1; `nil` outside a hold or unmeasured.
    public var severity: Double?
    /// The colour it is drawn in.
    public var band: SeverityBand

    public init(
        joint: Joint, point: Point2, severity: Double?, band: SeverityBand
    ) {
        self.joint = joint
        self.point = point
        self.severity = severity
        self.band = band
    }
}

/// One bone: the worse of its two joints, so a line never hides a bad
/// joint behind a good one.
public struct DiagramBone: Sendable, Equatable {
    /// Where the bone starts.
    public var from: Joint
    /// Where it ends.
    public var to: Joint
    /// The colour it is drawn in — `Severity.colourBand` of `severity`.
    public var band: SeverityBand
    /// The larger of the two joints' severities; `nil` when neither has one.
    public var severity: Double?

    public init(from: Joint, to: Joint, band: SeverityBand, severity: Double?) {
        self.from = from
        self.to = to
        self.band = band
        self.severity = severity
    }
}

/// The stack line: the vertical line through the wrist midpoint, running the
/// full height of the frame. A struct rather than a `(top:bottom:)` tuple
/// because tuples of two `Point2`s do not make `DiagramFrame` `Equatable`.
public struct DiagramStackLine: Sendable, Equatable {
    /// The line's top end — `(wristMidX, 0)`.
    public var top: Point2
    /// The line's bottom end — `(wristMidX, height)`.
    public var bottom: Point2

    public init(top: Point2, bottom: Point2) {
        self.top = top
        self.bottom = bottom
    }
}

/// What one frame of the stress diagram draws — chainlink #48's model, one
/// per playback moment. Pure data: the app maps it onto the video with
/// `OverlayGeometry` and draws it.
public struct DiagramFrame: Sendable, Equatable {
    /// The frame's timestamp in milliseconds — the clip's own clock, the one
    /// `StressDiagram.frameIndex` looks a playback time up in.
    public var tMs: Int
    /// Was this frame inside a hold (`phases.phase[i] == .hold`)? Outside one
    /// every band is `.neutral`.
    public var inHold: Bool
    /// The joints to draw — only the ones `processed.frames[i].valid` has.
    public var joints: [DiagramJoint]
    /// The bones to draw — both endpoints among `joints`, plus the head line.
    public var bones: [DiagramBone]
    /// The vertical line through the wrist midpoint, full frame height;
    /// `nil` when no wrist was visible to put it through.
    public var stackLine: DiagramStackLine?
    /// The colour of the stack line — the worse of `line_deviation` and
    /// `body_angle`, `.neutral` outside a hold or unmeasured.
    public var stackBand: SeverityBand
    /// Where the centre of mass is, display pixels; `nil` when `com_u` /
    /// `com_v` were not measured on this frame (or there is no wrist
    /// midpoint or body length to place it with).
    public var com: Point2?
    /// The same point dropped onto the hand line (`v = 0`) — where the CoM
    /// *projects* onto the floor the hands are on.
    public var comFloor: Point2?
    /// Which side of the base of support the CoM is on (`features.balanceZone`).
    public var balanceZone: BalanceZone?
    /// The ghost "ideal" skeleton to compare against — **always `nil`
    /// forever in this issue**: it needs the user's reference skeleton,
    /// chainlink #28. The field exists so #28 has somewhere to put it and
    /// the drawing code never has to grow a second shape.
    public var ideal: [Joint: Point2]?

    public init(
        tMs: Int,
        inHold: Bool,
        joints: [DiagramJoint],
        bones: [DiagramBone],
        stackLine: DiagramStackLine?,
        stackBand: SeverityBand,
        com: Point2?,
        comFloor: Point2?,
        balanceZone: BalanceZone?,
        ideal: [Joint: Point2]?
    ) {
        self.tMs = tMs
        self.inHold = inHold
        self.joints = joints
        self.bones = bones
        self.stackLine = stackLine
        self.stackBand = stackBand
        self.com = com
        self.comFloor = comFloor
        self.balanceZone = balanceZone
        self.ideal = ideal
    }
}

/// The stress diagram's own logic: the skeleton it draws, the severity of
/// one frame of an `Analysis`, and the lookup from a playback time to the
/// frame to draw.
public enum StressDiagram {
    /// Every bone of the skeleton, as pairs of joints: arms (shoulder →
    /// elbow → wrist), the torso (shoulders → hips, and the shoulder line),
    /// legs (hip → knee → ankle, and the hip line) and the feet (ankle →
    /// foot index).
    ///
    /// The head line — nose to the *shoulder midpoint* — is not in here: the
    /// 15-joint schema has no midpoint joint to name, so `frame(_:…)` draws
    /// it from the nose to the nearer shoulder instead.
    public static let bones: [(Joint, Joint)] = [
        // Arms, both sides.
        (.leftShoulder, .leftElbow),
        (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow),
        (.rightElbow, .rightWrist),
        // Torso: shoulders to hips, and the shoulder line across.
        (.leftShoulder, .leftHip),
        (.rightShoulder, .rightHip),
        (.leftShoulder, .rightShoulder),
        // Legs, both sides, and the hip line across.
        (.leftHip, .leftKnee),
        (.leftKnee, .leftAnkle),
        (.rightHip, .rightKnee),
        (.rightKnee, .rightAnkle),
        (.leftHip, .rightHip),
        // Feet.
        (.leftAnkle, .leftFootIndex),
        (.rightAnkle, .rightFootIndex),
    ]

    /// Which features say how off each joint is — the diagram's one table,
    /// so a joint's colour is defined in exactly one place. The joint's
    /// severity is the **max** over the features that touch it; a joint with
    /// no feature (the wrists) is therefore `.ok` inside a hold, never a
    /// number made of nothing.
    public static let jointFeatures: [Joint: [String]] = [
        .nose: ["head"],
        .leftShoulder: ["shoulder_angle", "off_shoulder"],
        .rightShoulder: ["shoulder_angle", "off_shoulder"],
        .leftElbow: ["elbow_angle"],
        .rightElbow: ["elbow_angle"],
        // Wrists are the origin every offset is measured *from*: there is
        // nothing for them to be off.
        .leftWrist: [],
        .rightWrist: [],
        .leftHip: ["hip_angle", "off_hip", "banana"],
        .rightHip: ["hip_angle", "off_hip", "banana"],
        .leftKnee: ["knee_angle", "off_knee"],
        .rightKnee: ["knee_angle", "off_knee"],
        .leftAnkle: ["off_ankle", "leg_separation"],
        .rightAnkle: ["off_ankle", "leg_separation"],
        .leftFootIndex: ["off_ankle", "leg_separation"],
        .rightFootIndex: ["off_ankle", "leg_separation"],
    ]

    /// The frame to draw at playback time `t` milliseconds: the **last**
    /// frame whose timestamp is at or before `t` (so the overlay never
    /// shows a frame the playback has not reached), `nil` before the first
    /// one.
    ///
    /// `tMs` is the clip's own clock (`features.tMs`), variable frame rate
    /// included — no fps is assumed anywhere.
    public static func frameIndex(atMs t: Int, in tMs: [Int]) -> Int? {
        var found: Int?
        for (index, stamp) in tMs.enumerated() where stamp <= t {
            found = index
        }
        return found
    }

    /// What to draw for frame `i` of `analysis`.
    ///
    /// - Parameters:
    ///   - i: an index into the clip's frames (every stage writes one row
    ///     per input frame).
    ///   - analysis: the pipeline's answer — the joints come from
    ///     `processed`, the hold and the feature values from `phases` and
    ///     `features`.
    ///   - reference: the scoring reference to be severe against, or `nil`
    ///     to fall back to `FeatureTolerances`' built-in targets.
    ///   - height: the frame's height in display pixels — how far down the
    ///     stack line runs.
    public static func frame(
        _ i: Int,
        analysis: Analysis,
        reference: ScoreReference?,
        height: Double
    ) -> DiagramFrame {
        let processed = analysis.processed
        let features = analysis.features
        precondition(
            i >= 0 && i < processed.frames.count && i < features.tMs.count,
            "frame index \(i) out of range"
        )
        let source = processed.frames[i]
        // Measured means "inside a hold *and* a frame the features stand
        // behind": everything else is drawn, but nothing is judged.
        let inHold = i < analysis.phases.phase.count && analysis.phases.phase[i] == .hold
        let measured = inHold && i < features.valid.count && features.valid[i]

        // One feature's severity at this frame — the two modes documented at
        // the top of this file, and `nil` for a value nobody measured or a
        // feature neither source has an opinion about.
        func severity(of name: String) -> Double? {
            guard measured else { return nil }
            let column = features.value(name)
            guard i < column.count else { return nil }
            let raw = column[i]
            guard raw.isFinite else { return nil }

            if let reference {
                guard let stat = reference.features[name] else { return nil }
                var value = raw
                if Scorer.signedByFacing.contains(name) {
                    let signs = features.value("facing_sign")
                    guard i < signs.count, signs[i].isFinite else { return nil }
                    value *= signs[i]
                }
                let floor = Scorer.sdFloors[name] ?? Scorer.sdFloorL
                let z = (value - stat.mean) / Swift.max(stat.sd, floor)
                guard z.isFinite else { return nil }
                return Swift.min(Swift.abs(z), Scorer.zCap) / Scorer.zCap
            }

            guard let tolerance = FeatureTolerances.tolerance(for: name) else { return nil }
            guard FeatureTolerances.fails(raw, tolerance) else { return 0 }
            let bandSize = severityBandSize(of: name)
            return Swift.min(1.0, 0.3 + FeatureTolerances.excess(raw, tolerance) / bandSize)
        }

        /// The colour of a severity: `.neutral` when this frame is not
        /// measured, `.ok` when it is measured but nothing is wrong with it
        /// (the wrists, and every joint of a hold nobody found fault with),
        /// and the severity's own band otherwise.
        func band(of severity: Double?) -> SeverityBand {
            guard measured else { return .neutral }
            guard let severity else { return .ok }
            return Severity.colourBand(severity)
        }

        // Joints: only the ones this frame has a position for.
        var joints: [DiagramJoint] = []
        var drawn: [Joint: DiagramJoint] = [:]
        for joint in Joint.allCases where source.valid.contains(joint) {
            guard let point = source.joints[joint] else { continue }
            let jointSeverity = (StressDiagram.jointFeatures[joint] ?? [])
                .compactMap(severity(of:))
                .max()
            let entry = DiagramJoint(
                joint: joint, point: point, severity: jointSeverity, band: band(of: jointSeverity))
            joints.append(entry)
            drawn[joint] = entry
        }

        // Bones: both joints drawn, severity the worse of the two.
        var bones: [DiagramBone] = []
        for (from, to) in StressDiagram.bones {
            guard let start = drawn[from], let end = drawn[to] else { continue }
            bones.append(
                bone(
                    from: from, to: to,
                    severities: [start.severity, end.severity],
                    band: band
                )
            )
        }
        // The head line: nose to the shoulder *midpoint*, which the schema
        // has no joint for — so it is drawn to the nearer shoulder instead
        // (in a side view the two shoulders sit on top of each other and
        // either reads the same).
        if let nose = drawn[.nose] {
            let shoulders = [Joint.leftShoulder, Joint.rightShoulder].compactMap { drawn[$0] }
            let nearer = shoulders.min { first, second in
                distanceSquared(nose.point, first.point) < distanceSquared(nose.point, second.point)
            }
            if let nearer {
                bones.append(
                    bone(
                        from: .nose, to: nearer.joint,
                        severities: [nose.severity, nearer.severity],
                        band: band
                    )
                )
            }
        }

        // The stack line: the vertical through the wrist midpoint, the full
        // height of the frame. One visible wrist is enough (the same rule
        // `Features.sideMean` puts the two sides together by).
        let wristMid = midpoint(
            position(of: .leftWrist, in: source),
            position(of: .rightWrist, in: source)
        )
        let stackLine = wristMid.map {
            DiagramStackLine(
                top: Point2(x: $0.x, y: 0),
                bottom: Point2(x: $0.x, y: height)
            )
        }
        let stackSeverity = [severity(of: "line_deviation"), severity(of: "body_angle")]
            .compactMap { $0 }
            .max()

        // The centre of mass: the body-frame column placed back into display
        // pixels through the very transform the features were measured with
        // (`BodyFrame.fromBodyFrame`, the exact inverse of `toBodyFrame`).
        var com: Point2?
        var comFloor: Point2?
        if let wristMid,
            let length = processed.bodyLength.totalPx,
            length.isFinite, length > 0,
            let u = finiteValue("com_u", features, i), let v = finiteValue("com_v", features, i)
        {
            let placed = BodyFrame.fromBodyFrame(
                u: u, v: v, wristMidX: wristMid.x, wristMidY: wristMid.y, bodyLength: length)
            com = Point2(x: placed.x, y: placed.y)
            let onTheLine = BodyFrame.fromBodyFrame(
                u: u, v: 0, wristMidX: wristMid.x, wristMidY: wristMid.y, bodyLength: length)
            comFloor = Point2(x: onTheLine.x, y: onTheLine.y)
        }

        return DiagramFrame(
            tMs: features.tMs[i],
            inHold: inHold,
            joints: joints,
            bones: bones,
            stackLine: stackLine,
            stackBand: band(of: stackSeverity),
            com: com,
            comFloor: comFloor,
            balanceZone: i < features.balanceZone.count ? features.balanceZone[i] : nil,
            // Chainlink #28 owns the ideal skeleton; until it exists there
            // is nothing to put here.
            ideal: nil
        )
    }

    // MARK: - Private

    /// The severity band a *failed tolerance* is spread over: a feature is
    /// fully red once it is one band past its target — 20° for the degree
    /// features, 0.2 L for the body-length ones (the same widths the
    /// no-reference mode documents).
    private static func severityBandSize(of feature: String) -> Double {
        let unit = Features.all.first { $0.name == feature }?.unit
        return unit == "deg" ? 20.0 : 0.2
    }

    /// One bone from its two joints' severities: the worse of the two.
    private static func bone(
        from: Joint, to: Joint, severities: [Double?], band: (Double?) -> SeverityBand
    ) -> DiagramBone {
        let severity = severities.compactMap { $0 }.max()
        return DiagramBone(from: from, to: to, band: band(severity), severity: severity)
    }

    /// One point of a processed frame, `nil` when the joint was not valid
    /// (or has no position).
    private static func position(of joint: Joint, in frame: ProcessedFrame) -> Point2? {
        guard frame.valid.contains(joint) else { return nil }
        return frame.joints[joint]
    }

    /// The mean of two points: both when both exist, the one that does when
    /// only one does (a side view often hides a wrist), `nil` when neither
    /// does — `Features.sideMean`'s rule, over points instead of numbers.
    private static func midpoint(_ first: Point2?, _ second: Point2?) -> Point2? {
        guard let first, let second else { return first ?? second }
        return Point2(x: (first.x + second.x) / 2.0, y: (first.y + second.y) / 2.0)
    }

    /// One column's value at frame `i`, `nil` when it is not a finite
    /// number there.
    private static func finiteValue(_ name: String, _ features: ClipFeatures, _ i: Int) -> Double? {
        let column = features.value(name)
        guard i < column.count, column[i].isFinite else { return nil }
        return column[i]
    }

    /// Squared distance between two display-frame points — enough to pick
    /// the *nearer* shoulder without a square root.
    private static func distanceSquared(_ first: Point2, _ second: Point2) -> Double {
        let dx = first.x - second.x
        let dy = first.y - second.y
        return dx * dx + dy * dy
    }
}
