// --------------------------------------------------------------------------- #
// The whole on-device chain, one entry point (#42): PostProcess.process ->
// PhaseSegmenter.classify -> Features.extract -> (Scorer.scoreClip + clipScore).
//
// `handstand.golden.run_pipeline` runs the first three stages in Python and the
// fixture's `expected.score` is then the scorer over that run; this file is the
// same chain in the same order, wired together with the arguments the per-stage
// parity tests (#39, #40, #81, #41) already pass. No new maths lives here: if a
// number disagrees with Python the fix belongs in the stage that computed it,
// and this entry point picks it up for free.
//
// Scoring needs a reference (`docs/scoring.md`'s schema v1). The parity
// fixtures carry the synthetic `parity_reference.json` beside them; the app
// will carry #28's real one. With no reference the analysis simply has no
// scores — `holdScores` is empty and `clipScore` nil, never a score made up
// against nothing.
// --------------------------------------------------------------------------- #

/// One clip after every stage has run on device — chainlink #42's bundle of
/// the four ports' answers, in the order the pipeline computes them.
public struct Analysis: Sendable, Equatable {
    /// The post-processed trajectory: per-frame positions with `valid` and
    /// `filled`, plus the `BodyLength` sidecar (`PostProcess.process`).
    public var processed: ProcessedClip
    /// The label and hold number of every frame (`PhaseSegmenter.classify`).
    public var phases: ClipPhases
    /// The per-frame features the scorer and the app read (`Features.extract`).
    public var features: ClipFeatures
    /// One score per hold, in `hold_id` order — empty when `reference` is `nil`.
    public var holdScores: [HoldScore]
    /// The hold the clip is represented by: `Scorer.clipScore(holdScores)`,
    /// hence `nil` whenever `holdScores` is empty or none of them scored.
    public var clipScore: HoldScore?

    public init(
        processed: ProcessedClip,
        phases: ClipPhases,
        features: ClipFeatures,
        holdScores: [HoldScore],
        clipScore: HoldScore?
    ) {
        self.processed = processed
        self.phases = phases
        self.features = features
        self.holdScores = holdScores
        self.clipScore = clipScore
    }
}

/// The on-device chain, exactly as `golden.py`'s `run_pipeline` + the scorer
/// run it in Python — one call, every stage, in order.
public enum Analyzer {
    /// Run the whole chain over one clip.
    ///
    /// The inputs are the ones the per-stage parity tests use: the raw input
    /// frames (`tMs` is the clip's clock — variable frame rate — and
    /// `trainerContact` the **raw** contact flags, because the post-process
    /// gates those frames away while the segmenter still has to say *why* a
    /// frame is unknown). Then, in pipeline order:
    ///
    /// 1. `PostProcess.process(frames)` — with `config`, the post-process
    ///    tuning (its `minVisibility` gate comes from the backend that ran,
    ///    `PoseBackend.postProcessConfig`, so a per-backend recalibration is
    ///    one number there);
    /// 2. `PhaseSegmenter.classify(tMs:processed:trainerContact:)`;
    /// 3. `Features.extract(tMs:processed:phases:trainerContact:)`;
    /// 4. with a reference, `Scorer.scoreClip(features, reference:)` and
    ///    `Scorer.clipScore(_:)` — without one, no scores at all.
    public static func analyze(
        _ frames: [PostProcessInputFrame],
        reference: ScoreReference?,
        config: PostProcessConfig = PostProcessConfig()
    ) -> Analysis {
        let tMs = frames.map(\.tMs)
        let trainerContact = frames.map(\.trainerContact)

        let processed = PostProcess.process(frames, config: config)
        let phases = PhaseSegmenter.classify(
            tMs: tMs, processed: processed, trainerContact: trainerContact)
        let features = Features.extract(
            tMs: tMs, processed: processed, phases: phases, trainerContact: trainerContact)

        let holdScores = reference.map { Scorer.scoreClip(features, reference: $0) } ?? []
        return Analysis(
            processed: processed,
            phases: phases,
            features: features,
            holdScores: holdScores,
            clipScore: Scorer.clipScore(holdScores)
        )
    }
}
