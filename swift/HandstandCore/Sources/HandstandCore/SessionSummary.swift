import Foundation

// --------------------------------------------------------------------------- #
// The summary view's answers (chainlink #49): three pure functions over an
// `Analysis`, plus the coaching-cue texts, with no view, player or video in
// any of them.
//
// * **`SessionSummary.heatStrip`** — the clip's timeline as bins, each one
//   the worst frame-severity inside it, coloured by the diagram's own
//   `SeverityBand`: how good the form was over the clip, in one strip.
// * **`SessionSummary.worstMoment`** — the worst frame of the hold (edges
//   skipped: entries and exits are not the held form), for the card that
//   seeks there.
// * **`CoachingCues.cues`** — up to three plain-language cues about the
//   clip's longest hold, the one `Scorer.clipScore` represents a clip by.
//
// Severity is *not* re-derived here: everything reads
// `StressDiagram.featureSeverities(_:analysis:reference:)` /
// `frameSeverity(_:…)` / `valueSeverity(_:feature:reference:)`, so the
// strip, the card, the overlay and the cues are one set of numbers
// (`StressDiagramTests` pins them against `frame(_:…)`).
//
// Side view only for now (every clip in the dataset is one): nothing in
// here says "side", so front/back (#83) can arrive without renaming a type.
// --------------------------------------------------------------------------- #

/// One bin of the timeline heat strip: the slice of the clip it covers and
/// how bad the form was inside it.
public struct HeatBin: Sendable, Equatable {
    /// The bin's first timestamp in milliseconds — where it starts on the
    /// strip (`[startMs, endMs)`; the last bin ends on the clip's last).
    public var startMs: Int
    /// The bin's last timestamp in milliseconds, exclusive for every bin
    /// but the last (which is exactly the clip's last `t_ms`).
    public var endMs: Int
    /// The **max** frame severity of the frames inside the bin that are in
    /// a hold; `nil` when the bin holds no such frame — the band is
    /// `.neutral` then, and `inHold` says whether that is because the clip
    /// was not being held there or because the frames could not be
    /// measured.
    public var severity: Double?
    /// The colour the bin is drawn in — `Severity.colourBand(severity)`,
    /// `.neutral` when there is no severity.
    public var band: SeverityBand
    /// Did any frame of the bin fall inside a hold?
    public var inHold: Bool

    public init(
        startMs: Int, endMs: Int, severity: Double?, band: SeverityBand, inHold: Bool
    ) {
        self.startMs = startMs
        self.endMs = endMs
        self.severity = severity
        self.band = band
        self.inHold = inHold
    }
}

/// The worst frame of the hold — the one the "worst moment" card shows and
/// seeks to.
public struct WorstMoment: Sendable, Equatable {
    /// The frame's index into the analysis's frames.
    public var frameIndex: Int
    /// The frame's timestamp in milliseconds — where the card seeks.
    public var tMs: Int
    /// The frame's severity: the max over its features (`nil` frames and
    /// severities of 0 are never the worst moment).
    public var severity: Double
    /// Which hold the frame was in.
    public var holdId: Int
    /// The features at severity ≥ 0.5 on that frame, worst first — the
    /// ones the card says in plain words.
    public var features: [String]

    public init(
        frameIndex: Int, tMs: Int, severity: Double, holdId: Int, features: [String]
    ) {
        self.frameIndex = frameIndex
        self.tMs = tMs
        self.severity = severity
        self.holdId = holdId
        self.features = features
    }
}

/// One coaching cue: the feature it is about, how bad that was (0…1, on
/// the same scale as the diagram) and the sentence the screen shows.
public struct CoachingCue: Sendable, Equatable {
    /// The feature's name — a `Features.all` spelling, `banana_pos` /
    /// `banana_neg` for the two directions of `banana`, `balance_over` /
    /// `balance_under` for the two balance cues, `"none"` for the "nothing
    /// is over the limits" cue.
    public var feature: String
    /// How bad it was: `min(|z|, Z_CAP) / Z_CAP` with a reference, the
    /// tolerance severity (#48) without one, and a flat 0.4 for the balance
    /// cues.
    public var severity: Double
    /// The plain-language text, from `CoachingCues.texts`.
    public var text: String

    public init(feature: String, severity: Double, text: String) {
        self.feature = feature
        self.severity = severity
        self.text = text
    }
}

/// The coaching cues of an analysed session: the texts, and how an
/// `Analysis` becomes up to three of them (chainlink #49).
public enum CoachingCues {
    /// How far off a reference has to be before it is worth saying out
    /// loud — `Z ≥ 1.5` standard deviations on a `HoldScore.topFaults`
    /// entry, taken as a **magnitude** (`|z|`, like the severity itself):
    /// a hip two SDs *below* the reference's mean is as much a fault as
    /// one two SDs above it.
    public static let zThreshold = 1.5

    /// How much of the hold spent outside the base of support makes the
    /// balance cue worth saying — `pct_over` / `pct_under` above this.
    public static let balancePctThreshold = 30.0

    /// The one table of cue texts.
    ///
    /// **The wording of these cues is to be reviewed by the user** — they
    /// are short and actionable on purpose (say the fault, say the fix in
    /// one clause), but they are copy, not maths, and may be reworded.
    /// One table so the tests, the screen and any future export all say
    /// exactly the same sentence.
    public static let texts: [String: String] = [
        "shoulder_angle": "Open your shoulders: push the floor away, arms by your ears.",
        "hip_angle": "Hips are piked: squeeze your glutes and bring your legs in line.",
        "knee_angle": "Straighten your knees and point your toes.",
        "elbow_angle": "Lock your elbows.",
        "banana_pos": "You're arching (banana): pull your ribs in and tuck the pelvis.",
        "banana_neg": "You're hollowing at the hips: open the hips to a straight line.",
        "head": "Keep your head neutral: eyes on the floor between your hands.",
        "leg_separation": "Keep your legs together.",
        "line_deviation": "Your line leans: stack hips over shoulders over hands.",
        "body_angle": "Your line leans: stack hips over shoulders over hands.",
        "off_shoulder": "Stack your shoulders over your hands.",
        "off_hip": "Stack your hips over your hands.",
        "off_knee": "Stack your knees over your hands.",
        "off_ankle": "Stack your ankles over your hands.",
        "balance_over": "You're tipping towards your fingers: press into your fingertips.",
        "balance_under": "You're sitting back on your palms: shift a little over your fingers.",
        "none": "Solid line: nothing over the limits in this hold.",
    ]

    /// The cues for this clip's **longest hold** — the hold
    /// `Scorer.clipScore` / `ClipFeatures.longestHoldId` represents the clip
    /// by — at most `max` of them, worst first.
    ///
    /// * **With a reference** the candidates are that hold's
    ///   `HoldScore.topFaults` with `|z| ≥ zThreshold`.
    /// * **Without one** they are the features whose hold **median** fails
    ///   `FeatureTolerances`.
    /// * **Balance**, in both modes: `pct_over > 30` is `balance_over`,
    ///   `pct_under > 30` is `balance_under` (severity 0.4).
    ///
    /// Ordered by severity (ties keep the order above — the scorer's rank
    /// with a reference, the tolerance table's row order without one), then
    /// cut to `max`. Candidates with no text (`com_forward`, `com_sway_sd`
    /// — scored, but not things a cue can be written about yet) are
    /// dropped rather than shown as a bare key.
    ///
    /// Returns `[]` for a clip with no hold and for one that could not be
    /// measured at all; a hold where nothing is over the limits gets the
    /// single `"none"` cue instead of an empty section.
    public static func cues(
        analysis: Analysis, reference: ScoreReference?, max: Int = 3
    ) -> [CoachingCue] {
        let features = analysis.features
        guard features.usable, analysis.phases.usable else { return [] }
        let longest = features.longestHoldId()
        guard longest != PhaseSegmenter.noHold else { return [] }
        guard let row = Features.holdRows(features).first(where: { $0.holdId == longest }) else {
            return []
        }

        // The candidates, worst-sorted below by severity (ties keeping
        // this order): (cue key, severity).
        var candidates: [(feature: String, severity: Double)] = []

        if reference != nil {
            // The scorer's own ranking of the hold, filtered to faults far
            // enough off to be worth a sentence. `fault.value` is the hold's
            // median, which is what picks banana's direction.
            let hold = analysis.holdScores.first { $0.holdId == longest }
            for fault in hold?.topFaults ?? [] where Swift.abs(fault.z) >= zThreshold {
                guard let feature = cueKey(for: fault.feature, median: fault.value) else {
                    continue
                }
                candidates.append(
                    (
                        feature,
                        Swift.min(Swift.abs(fault.z), Scorer.zCap) / Scorer.zCap
                    ))
            }
        } else {
            // The hold's medians against the built-in targets (#48's rules
            // through the one severity function).
            for tolerance in FeatureTolerances.all {
                let median = row.stats["\(tolerance.feature)_median"] ?? .nan
                guard FeatureTolerances.fails(median, tolerance) else { continue }
                guard let feature = cueKey(for: tolerance.feature, median: median) else {
                    continue
                }
                guard
                    let severity = StressDiagram.valueSeverity(
                        median, feature: tolerance.feature, reference: nil)
                else { continue }
                candidates.append((feature, severity))
            }
        }

        // Balance, from the same hold's row — a cue about where the CoM
        // sat, not about a feature's shape, so it speaks in both modes.
        if let pctOver = row.stats["pct_over"], pctOver > balancePctThreshold {
            candidates.append(("balance_over", 0.4))
        }
        if let pctUnder = row.stats["pct_under"], pctUnder > balancePctThreshold {
            candidates.append(("balance_under", 0.4))
        }

        // Only cues there is a sentence for.
        candidates = candidates.filter { texts[$0.feature] != nil }

        guard !candidates.isEmpty else {
            return [
                CoachingCue(
                    feature: "none", severity: 0,
                    text: texts["none"] ?? "Solid line: nothing over the limits in this hold.")
            ]
        }

        let ranked = candidates.enumerated()
            .sorted { first, second in
                if first.element.severity != second.element.severity {
                    return first.element.severity > second.element.severity
                }
                return first.offset < second.offset  // ties keep the order above
            }
            .prefix(Swift.max(max, 0))
            .map { entry in
                CoachingCue(
                    feature: entry.element.feature,
                    severity: entry.element.severity,
                    text: texts[entry.element.feature] ?? ""
                )
            }
        return Array(ranked)
    }

    /// The text-table key for one feature: `banana` picks its direction
    /// from the hold's median (banana is athlete-signed already — positive
    /// is the arch, negative the hollow), everything else is its own key.
    /// A median of exactly zero has no direction to coach, so it is no
    /// cue at all.
    static func cueKey(for feature: String, median: Double) -> String? {
        guard feature == "banana" else { return feature }
        guard median.isFinite, median != 0 else { return nil }
        return median > 0 ? "banana_pos" : "banana_neg"
    }
}

/// The summary view's answers for one analysed clip (chainlink #49): the
/// timeline heat strip, the worst moment of the hold, and — through
/// `CoachingCues` — what to work on. All pure, all over the same
/// severities the overlay draws.
public enum SessionSummary {
    /// The clip's timeline as `bins` bins, in time order.
    ///
    /// The bins split `[first tMs, last tMs]` evenly and tile it exactly:
    /// bin `b` runs from `first + span·b/bins` to `first + span·(b+1)/bins`
    /// (the last one ending on the clip's last `t_ms`), so the strip covers
    /// the whole clip with no gaps and no overlaps.
    ///
    /// A bin's severity is the **max** frame severity of the frames inside
    /// it that are in a hold, and its band is `Severity.colourBand` of that
    /// — `.neutral` with no severity. A bin is `inHold` when any of its
    /// frames was in a hold. Fewer frames than bins is fine: the bins with
    /// no frames are `.neutral` and say nothing.
    ///
    /// Returns `[]` for a clip with no frames, or `bins < 1`.
    public static func heatStrip(
        analysis: Analysis, reference: ScoreReference?, bins: Int = 120
    ) -> [HeatBin] {
        guard bins >= 1 else { return [] }
        let tMs = analysis.features.tMs
        guard let first = tMs.first, let last = tMs.last else { return [] }
        let span = last - first

        // The bins' boundaries: even in milliseconds, the last ending on
        // the clip's own last timestamp.
        let boundaries = (0..<bins).map { bin -> (start: Int, end: Int) in
            (first + span * bin / bins, first + span * (bin + 1) / bins)
        }

        var severities = [Double?](repeating: nil, count: bins)
        var inHold = [Bool](repeating: false, count: bins)

        for i in tMs.indices {
            // Which bin this frame's timestamp falls in — the first bin
            // whose end is past it, the last bin for a frame at the very
            // end of the clip. (A span of one frame or less puts every
            // frame in the first bin.)
            let bin: Int
            if span <= 0 {
                bin = 0
            } else {
                bin = boundaries.firstIndex { tMs[i] < $0.end } ?? (bins - 1)
            }
            guard i < analysis.phases.phase.count, analysis.phases.phase[i] == .hold else {
                continue
            }
            inHold[bin] = true
            guard let severity = StressDiagram.frameSeverity(
                i, analysis: analysis, reference: reference)
            else { continue }
            severities[bin] = severities[bin].map { Swift.max($0, severity) } ?? severity
        }

        return boundaries.enumerated().map { bin, range in
            let severity = severities[bin]
            return HeatBin(
                startMs: range.start,
                endMs: range.end,
                severity: severity,
                band: severity.map { Severity.colourBand($0) } ?? .neutral,
                inHold: inHold[bin]
            )
        }
    }

    /// The worst frame of the clip's holds — what the "worst moment" card
    /// shows and seeks to.
    ///
    /// Only frames inside a hold count, and the first and last `edgeS`
    /// seconds of each hold are skipped: an entry and an exit are not the
    /// held form, so the kick-up and the landing can never be the worst
    /// moment. The worst frame is the one with the highest frame severity
    /// (the max over its features), ties going to the **earlier** frame.
    ///
    /// `nil` when the clip has no hold, or when no frame qualifies with a
    /// finite severity above 0 — a clean hold has no worst moment to show.
    public static func worstMoment(
        analysis: Analysis, reference: ScoreReference?, edgeS: Double = 0.3
    ) -> WorstMoment? {
        let features = analysis.features
        let phases = analysis.phases
        let edgeMs = Int((Swift.max(edgeS, 0) * 1000.0).rounded())

        // Each hold's span, so its edges can be skipped.
        var starts: [Int: Int] = [:]
        var ends: [Int: Int] = [:]
        for i in features.tMs.indices {
            guard i < phases.phase.count, phases.phase[i] == .hold,
                i < phases.holdId.count, phases.holdId[i] >= 0
            else { continue }
            let holdId = phases.holdId[i]
            let t = features.tMs[i]
            starts[holdId] = Swift.min(starts[holdId] ?? t, t)
            ends[holdId] = Swift.max(ends[holdId] ?? t, t)
        }

        var worst: (frameIndex: Int, severity: Double, holdId: Int)?
        for i in features.tMs.indices {
            guard i < phases.phase.count, phases.phase[i] == .hold,
                i < phases.holdId.count
            else { continue }
            let holdId = phases.holdId[i]
            guard holdId >= 0, let start = starts[holdId], let end = ends[holdId] else {
                continue
            }
            let t = features.tMs[i]
            guard t >= start + edgeMs, t <= end - edgeMs else { continue }
            guard let severity = StressDiagram.frameSeverity(
                i, analysis: analysis, reference: reference), severity > 0
            else { continue }
            // Ties go to the earlier frame: only a strictly worse one wins.
            if let worst, severity <= worst.severity { continue }
            worst = (i, severity, holdId)
        }

        guard let worst else { return nil }
        let severities = StressDiagram.featureSeverities(
            worst.frameIndex, analysis: analysis, reference: reference)
        let ranked = severities
            .filter { $0.value >= 0.5 }
            .sorted { first, second in
                if first.value != second.value { return first.value > second.value }
                // Ties keep `Features.all` order, so the list is stable.
                return featureOrder(first.key) < featureOrder(second.key)
            }
            .map(\.key)
        return WorstMoment(
            frameIndex: worst.frameIndex,
            tMs: features.tMs[worst.frameIndex],
            severity: worst.severity,
            holdId: worst.holdId,
            features: ranked
        )
    }

    /// A feature's position in `Features.all` — the tie-break that keeps
    /// `worstMoment.features` in a deterministic order.
    private static func featureOrder(_ feature: String) -> Int {
        Features.featureNames.firstIndex(of: feature) ?? Int.max
    }
}
