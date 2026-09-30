import XCTest

@testable import HandstandCore

/// Chainlink #42's end-to-end check: one `Analyzer.analyze` over every golden
/// fixture, compared with **all** of `expected` in a single test —
/// `postprocess`, `phases`, `features`, `hold_summary` and `score`, each at
/// `meta.tolerances`.
///
/// The per-stage tests (#39, #40, #81, #41) prove each port answers Python's
/// numbers on its own; this proves the *chain* does — the stages wired
/// together in the order `golden.py`'s `run_pipeline` runs them, with the
/// folder's own `parity_reference.json`. The comparisons are the shared
/// `GoldenComparison`'s, so "each stage is right" and "the chain is right"
/// cannot drift into two different statements.
///
/// Nothing here reads a real video: the fixtures are the five synthetic cases
/// `pipeline/handstand/golden.py` writes. The real-clip run of the same chain
/// is `RealParityTests`, which skips without its private data.
final class AnalyzerTests: XCTestCase {
    /// Every committed fixture through the whole chain with the reference its
    /// `expected.score` was scored against, then every section compared — one
    /// test, all five, so a reviewer sees the whole chain's answer at once.
    func testEveryFixtureMatchesTheWholePythonChain() throws {
        let reference = try ScoreReference.decode(GoldenFixtures.parityReferenceData())
        let urls = GoldenFixtures.urls()
        XCTAssertGreaterThanOrEqual(urls.count, 5, "the five committed cases")

        for url in urls {
            let fixture = try GoldenFixtures.load(url)
            let analysis = Analyzer.analyze(
                GoldenFixtures.inputFrames(from: fixture), reference: reference)
            for mismatch in allSections(analysis, fixture: fixture) {
                XCTFail(mismatch)
            }
        }
    }

    /// The chain with no reference scores nothing: empty `holdScores`, nil
    /// `clipScore` — never a score made up against a reference that is not
    /// there — while every stage before the scorer still ran.
    func testWithoutAReferenceThereAreNoScores() throws {
        let fixture = try GoldenFixtures.load("line_hold")
        let analysis = Analyzer.analyze(
            GoldenFixtures.inputFrames(from: fixture), reference: nil)

        XCTAssertTrue(analysis.holdScores.isEmpty, "no reference, no scores")
        XCTAssertNil(analysis.clipScore, "no scores, no clip score")
        XCTAssertEqual(
            analysis.phases.phase.count, fixture.input.count, "the stages before it still ran")
        XCTAssertEqual(analysis.features.frames, fixture.input.count)
        XCTAssertFalse(analysis.processed.frames.isEmpty)
    }

    /// `analyze` wires the stages, it does not recompute anything: every
    /// intermediate answer is the per-stage call's own result.
    ///
    /// The comparison is field by field rather than `==` because
    /// `FrameSignals` and `ClipFeatures.values` hold `NaN` (a window with no
    /// samples yet, a column nobody measured) and `NaN != NaN` would call two
    /// identical runs different — the JSON round the scorer's answers take
    /// (`scoreJSON`) maps a non-finite number to `null`, which is how the
    /// fixture comparison stays honest about the same values.
    func testTheChainIsTheStagesCalledInOrder() throws {
        let fixture = try GoldenFixtures.load("hand_step")
        let frames = GoldenFixtures.inputFrames(from: fixture)
        let tMs = frames.map(\.tMs)
        let trainerContact = frames.map(\.trainerContact)

        let processed = PostProcess.process(frames)
        let phases = PhaseSegmenter.classify(
            tMs: tMs, processed: processed, trainerContact: trainerContact)
        let features = Features.extract(
            tMs: tMs, processed: processed, phases: phases, trainerContact: trainerContact)
        let reference = try ScoreReference.decode(GoldenFixtures.parityReferenceData())
        let scores = Scorer.scoreClip(features, reference: reference)

        let analysis = Analyzer.analyze(frames, reference: reference)

        XCTAssertEqual(analysis.processed, processed)
        XCTAssertEqual(analysis.phases.tMs, phases.tMs)
        XCTAssertEqual(analysis.phases.phase, phases.phase)
        XCTAssertEqual(analysis.phases.holdId, phases.holdId)
        XCTAssertEqual(analysis.phases.runs, phases.runs)
        XCTAssertEqual(analysis.phases.usable, phases.usable)
        XCTAssertEqual(analysis.phases.unusableReason, phases.unusableReason)
        XCTAssertEqual(analysis.phases.signals.known, phases.signals.known)
        XCTAssertEqual(analysis.phases.signals.inverted, phases.signals.inverted)
        XCTAssertTrue(
            sameNumbers(
                analysis.phases.signals.wristSpeedLPerS, phases.signals.wristSpeedLPerS),
            "the wrist window is the same computation")

        XCTAssertEqual(analysis.features.tMs, features.tMs)
        XCTAssertEqual(analysis.features.phase, features.phase)
        XCTAssertEqual(analysis.features.holdId, features.holdId)
        XCTAssertEqual(analysis.features.valid, features.valid)
        XCTAssertEqual(analysis.features.comComplete, features.comComplete)
        XCTAssertEqual(analysis.features.balanceZone, features.balanceZone)
        XCTAssertEqual(analysis.features.unusableReason, features.unusableReason)
        XCTAssertEqual(Set(analysis.features.values.keys), Set(features.values.keys))
        for (column, mine) in analysis.features.values {
            XCTAssertTrue(
                sameNumbers(mine, features.values[column] ?? []),
                "column \(column) is the same computation")
        }

        XCTAssertEqual(
            GoldenComparison.scoreJSON(analysis.holdScores, clipHoldId: analysis.clipScore?.holdId),
            GoldenComparison.scoreJSON(scores, clipHoldId: Scorer.clipScore(scores)?.holdId),
            "the scores are scoreClip's, not a second scoring"
        )
        XCTAssertEqual(analysis.clipScore?.holdId, Scorer.clipScore(scores)?.holdId)
    }

    /// Two columns of doubles equal, `NaN` counted equal to `NaN` — the point
    /// is "the same computation wrote both", and `NaN` is the answer on both
    /// sides as often as not.
    private func sameNumbers(_ lhs: [Double], _ rhs: [Double]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        return zip(lhs, rhs).allSatisfy { mine, want in
            mine == want || (mine.isNaN && want.isNaN)
        }
    }

    /// All five sections of one run, in the order the fixture writes them —
    /// the same list `RealParityTests` reports per section.
    private func allSections(
        _ analysis: Analysis, fixture: GoldenFixtures.Fixture
    ) -> [String] {
        GoldenComparison.postprocess(analysis.processed, fixture: fixture)
            + GoldenComparison.phases(analysis.phases, fixture: fixture)
            + GoldenComparison.features(analysis.features, fixture: fixture)
            + GoldenComparison.holdSummary(
                Features.holdRows(analysis.features), fixture: fixture)
            + GoldenComparison.score(
                analysis.holdScores,
                clipHoldId: analysis.clipScore?.holdId,
                fixture: fixture
            )
    }
}
