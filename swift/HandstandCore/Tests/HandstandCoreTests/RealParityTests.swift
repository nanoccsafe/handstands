import Foundation
import XCTest

@testable import HandstandCore

/// Chainlink #42's real-clip parity: the fixtures of **real** clips through
/// `Analyzer.analyze`, compared with every section of `expected` at
/// `meta.tolerances` — the check the synthetic fixtures (#80's crude cases)
/// cannot make.
///
/// The real keypoints are private (the repo is public), so this test reads
/// them from wherever `HANDSTAND_REAL_GOLDEN` points — normally
/// `~/handstand-private/golden_real/` on the Mac, which
/// `tools/mac/real_parity.sh` fills from
/// `handstand.golden --real-sample N`'s output in the shared data tree and
/// copies the fixtures' `parity_reference.json` beside (the reference is
/// synthetic, but it keeps the directory self-contained). Without the
/// variable the test skips, so a plain `swift test` never needs the data.
///
/// The report is for all clips at once, not just the first failure: per clip,
/// the mismatch count of each section and the first three mismatch strings of
/// each section, then one summary line — `real parity: X/Y clips identical` —
/// which is also what the test fails on, so `tools/mac/real_parity.sh` exits
/// non-zero if **any** clip mismatches.
final class RealParityTests: XCTestCase {
    func testEveryRealClipMatchesThePythonChain() throws {
        let directory = try goldenDirectory()
        let files = fixtureFiles(in: directory)

        // The reference beside the fixtures, exactly as the synthetic
        // end-to-end test reads it — same directory, same bytes.
        let referenceURL = directory.appendingPathComponent(
            "\(GoldenFixtures.parityReferenceName).json")
        guard let referenceData = try? Data(contentsOf: referenceURL) else {
            XCTFail("no \(GoldenFixtures.parityReferenceName).json in \(directory.path) — "
                + "tools/mac/real_parity.sh copies it there beside the fixtures")
            return
        }
        let reference = try ScoreReference.decode(referenceData)

        var identical = 0
        var mismatching = 0
        for url in files {
            let fixture = try GoldenFixtures.load(url)
            let name = url.deletingPathExtension().lastPathComponent
            guard fixture.meta.mode == "real" else {
                // The mode is how a test tells the two apart without reading a
                // single coordinate: something synthetic (or worse) is sitting
                // in the private folder.
                XCTFail("\(name): meta.mode is '\(fixture.meta.mode)', expected 'real'")
                mismatching += 1
                continue
            }

            let analysis = Analyzer.analyze(
                GoldenFixtures.inputFrames(from: fixture), reference: reference)
            let sections: [(name: String, mismatches: [String])] = [
                ("postprocess", GoldenComparison.postprocess(analysis.processed, fixture: fixture)),
                ("phases", GoldenComparison.phases(analysis.phases, fixture: fixture)),
                ("features", GoldenComparison.features(analysis.features, fixture: fixture)),
                (
                    "hold_summary",
                    GoldenComparison.holdSummary(
                        Features.holdRows(analysis.features), fixture: fixture)
                ),
                (
                    "score",
                    GoldenComparison.score(
                        analysis.holdScores,
                        clipHoldId: analysis.clipScore?.holdId,
                        fixture: fixture)
                ),
            ]
            let total = sections.reduce(0) { $0 + $1.mismatches.count }
            guard total > 0 else {
                identical += 1
                continue
            }
            mismatching += 1
            let counts = sections.map { "\($0.name)=\($0.mismatches.count)" }
                .joined(separator: " ")
            print("\(name): \(total) mismatch(es) — \(counts)")
            for section in sections {
                for mismatch in section.mismatches.prefix(3) {
                    print("  [\(section.name)] \(mismatch)")
                }
            }
        }

        let summary = "real parity: \(identical)/\(files.count) clips identical"
        print(summary)
        if mismatching > 0 {
            XCTFail(summary)
        }
    }

    // MARK: - Where the private fixtures live

    /// The directory `HANDSTAND_REAL_GOLDEN` names, or the skip that keeps a
    /// plain `swift test` green without private data.
    private func goldenDirectory() throws -> URL {
        let raw = ProcessInfo.processInfo.environment["HANDSTAND_REAL_GOLDEN"] ?? ""
        let path = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else {
            throw XCTSkip(
                "HANDSTAND_REAL_GOLDEN is unset or empty: the real clips are private — "
                    + "run tools/mac/real_parity.sh to generate them and point this test "
                    + "at them")
        }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
    }

    /// Every fixture JSON in that directory, in file-name order so the report
    /// is stable. `parity_reference.json` sits beside them and is not one —
    /// it carries no `meta`, exactly as on the synthetic side.
    private func fixtureFiles(in directory: URL) -> [URL] {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        else {
            XCTFail("HANDSTAND_REAL_GOLDEN points at \(directory.path), which cannot be read")
            return []
        }
        let files = contents
            .filter {
                $0.pathExtension == "json"
                    && $0.lastPathComponent != "\(GoldenFixtures.parityReferenceName).json"
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        if files.isEmpty {
            XCTFail("no real fixtures in \(directory.path) — "
                + "run tools/mac/real_parity.sh to generate them")
        }
        return files
    }
}
