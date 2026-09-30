import AVFoundation
import CoreGraphics
import XCTest

import AnnotatedVideo
import HandstandCore

// --------------------------------------------------------------------------- #
// The export itself (chainlink #50): every source frame written, the diagram
// on the **pixel** it belongs to (upright source and quarter-turn source
// alike — the orientation check is the whole point), the strip along the
// bottom, an unusable analysis still exporting, and a cancellation that
// leaves no file behind.
//
// Everything is synthetic: mid-grey frames written here, an analysis built
// from hand-made `PostProcessInputFrame`s (`SyntheticClip`). The checks are
// about pixels, so the background being mid-grey is what makes them read:
// drawn = not grey, untouched = still grey.
// --------------------------------------------------------------------------- #

final class AnnotatedExportTests: XCTestCase {
    /// The frame the pixel checks read: the middle of the clip, inside the
    /// hold. The export draws, for source frame `midIndex`, the analysis's
    /// own frame at the same timestamp — so the diagram below *is* what is
    /// on the picture.
    private static let midIndex = 22

    // MARK: - The output file

    func testTheExportIsReadableKeepsTheDisplaySizeAndIsUpright() async throws {
        let (source, directory) = try await SyntheticClip.write()
        defer { try? FileManager.default.removeItem(at: directory) }

        let (output, progress) = try await export(
            source: source, directory: directory,
            analysis: SyntheticClip.holdingAnalysis())

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: output.path), "the movie is written")
        let info = try await SyntheticClip.info(output)
        XCTAssertEqual(
            info.naturalSize, CGSize(width: 180, height: 320),
            "the display size of the source — upright, no rotation carried over")
        XCTAssertEqual(
            info.transform, .identity, "no transform is left on the output track")

        // Readable: every frame decodes back.
        let frames = try await SyntheticClip.read(output)
        XCTAssertEqual(frames.count, SyntheticClip.frameCount)

        XCTAssertEqual(progress.values.first ?? -1, 0, "progress starts at 0")
        XCTAssertEqual(progress.values.last ?? -1, 1, "progress ends at 1")
        for (previous, next) in zip(progress.values, progress.values.dropFirst()) {
            XCTAssertLessThanOrEqual(previous, next, "progress only moves forward")
            XCTAssertGreaterThanOrEqual(previous, 0)
            XCTAssertLessThanOrEqual(next, 1)
        }
    }

    func testTheFrameCountAndDurationFollowTheSource() async throws {
        let (source, directory) = try await SyntheticClip.write()
        defer { try? FileManager.default.removeItem(at: directory) }

        let output = try await export(
            source: source, directory: directory,
            analysis: SyntheticClip.holdingAnalysis()).output

        let sourceInfo = try await SyntheticClip.info(source)
        let outputInfo = try await SyntheticClip.info(output)
        let sourceFrames = try await SyntheticClip.read(source)
        let outputFrames = try await SyntheticClip.read(output)

        XCTAssertLessThanOrEqual(
            abs(outputFrames.count - sourceFrames.count), 1,
            "every source frame is written — within the one frame writers round")
        let oneFrame = 1.0 / Double(SyntheticClip.fps)
        XCTAssertLessThanOrEqual(
            abs(outputInfo.duration.seconds - sourceInfo.duration.seconds),
            oneFrame + 0.001, "the duration is the source's, within a frame")
    }

    // MARK: - Where the diagram lands

    func testTheBoneIsDrawnOnItsDisplayPixel() async throws {
        let (source, directory) = try await SyntheticClip.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let analysis = SyntheticClip.holdingAnalysis()

        let output = try await export(
            source: source, directory: directory, analysis: analysis).output

        try await assertTheDiagramLandsOnItsPixels(output: output, analysis: analysis)
    }

    /// The same check over a source stored **sideways** with a quarter-turn
    /// `preferredTransform` — the phone-held-landscape clip. The export must
    /// show the same display frame the analysis was written in, so the bone
    /// is at the same display pixel as above: orientation included.
    func testTheBoneIsDrawnOnItsDisplayPixelWhenTheSourceIsSideways() async throws {
        let (source, directory) = try await SyntheticClip.write(
            storedWidth: SyntheticClip.displayHeight,
            storedHeight: SyntheticClip.displayWidth,
            transform: CGAffineTransform(rotationAngle: .pi / 2)
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        // If the writer refused to keep the transform there is no rotation
        // to check — say so rather than asserting the wrong size.
        let sourceInfo = try await SyntheticClip.info(source)
        guard sourceInfo.transform != .identity else {
            throw XCTSkip(
                "AVAssetWriter did not keep the quarter-turn transform; there is no "
                    + "rotated track to check the display orientation against")
        }

        let analysis = SyntheticClip.holdingAnalysis()
        let output = try await export(
            source: source, directory: directory, analysis: analysis).output

        let info = try await SyntheticClip.info(output)
        XCTAssertEqual(
            info.naturalSize, CGSize(width: 180, height: 320),
            "the display size of the sideways source, not its stored 320 × 180")
        XCTAssertEqual(info.transform, .identity, "upright, whatever the source says")

        try await assertTheDiagramLandsOnItsPixels(output: output, analysis: analysis)
    }

    // MARK: - An analysis that cannot be used

    func testAnUnusableAnalysisStillExportsWithNoSkeletonAndANeutralStrip() async throws {
        let (source, directory) = try await SyntheticClip.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let analysis = SyntheticClip.nobodyAnalysis()
        XCTAssertFalse(analysis.features.usable, "the clip nobody could be measured in")

        let output = try await export(
            source: source, directory: directory, analysis: analysis).output

        XCTAssertTrue(FileManager.default.fileExists(atPath: output.path))
        let info = try await SyntheticClip.info(output)
        XCTAssertEqual(info.naturalSize, CGSize(width: 180, height: 320))
        XCTAssertEqual(info.transform, .identity)

        let frames = try await SyntheticClip.read(output)
        XCTAssertGreaterThanOrEqual(frames.count, SyntheticClip.frameCount - 1)
        let frame = frames[Self.midIndex]

        // Where the skeleton *would* have been — the bone's midpoint, the
        // stack line's column, a joint — every one still the background.
        for (x, y) in [(92, 230), (96, 230), (100, 275), (96, 100)] {
            XCTAssertTrue(
                SyntheticClip.isBackground(frame.pixel(x: x, y: y)),
                "no bone is drawn at (\(x), \(y))")
        }
        // And the strip is neutral: its own grey (`#8E8E93` — *drawn*, not
        // the background) the whole way along the bottom.
        for x in stride(from: 5, to: SyntheticClip.displayWidth, by: 20) {
            let pixel = frame.pixel(x: x, y: SyntheticClip.displayHeight - 2)
            XCTAssertTrue(
                SyntheticClip.isNeutralGrey(pixel),
                "the neutral strip is drawn at x=\(x) — got \(pixel)")
        }
    }

    // MARK: - Cancellation

    func testCancellingDeletesThePartialFile() async throws {
        let (source, directory) = try await SyntheticClip.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("annotated.mp4")
        let box = CancelBox()
        let export = AnnotatedExport(
            source: source, analysis: SyntheticClip.holdingAnalysis(),
            reference: nil, output: output)

        box.task = Task {
            try await export.run { progress in
                box.note(progress)
            }
        }
        do {
            try await box.task?.value
            XCTFail("a cancelled export must throw CancellationError")
        } catch is CancellationError {
            // The expected answer.
        }

        XCTAssertNotNil(
            box.cancelledAt, "the run was cancelled after it had started writing")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: output.path),
            "the partial file is deleted")
    }

    // MARK: - Helpers

    /// Runs the export into `directory`'s `annotated.mp4` and hands back the
    /// output and the progress it reported.
    private func export(
        source: URL, directory: URL, analysis: Analysis
    ) async throws -> (output: URL, progress: ProgressRecorder) {
        let output = directory.appendingPathComponent("annotated.mp4")
        let progress = ProgressRecorder()
        let export = AnnotatedExport(
            source: source, analysis: analysis, reference: nil, output: output)
        try await export.run { progress.record($0) }
        return (output, progress)
    }

    /// The three pixel checks every export must pass, against the middle
    /// frame of the output:
    ///
    /// * the bone's midpoint **is** drawn (not background grey) — the
    ///   diagram is in the right place, in display coordinates;
    /// * a pixel far from the skeleton is untouched (still grey);
    /// * the strip's coloured bins are coloured along the bottom row.
    private func assertTheDiagramLandsOnItsPixels(output: URL, analysis: Analysis) async throws {
        let bone = try checkedBone(analysis: analysis)
        XCTAssertNotEqual(
            bone.band, SeverityBand.neutral,
            "the synthetic clip is a measured hold, so the bone has a colour")

        let frames = try await SyntheticClip.read(output)
        XCTAssertGreaterThan(frames.count, Self.midIndex, "the middle frame decodes")
        let frame = frames[Self.midIndex]

        let on = frame.pixel(x: Int(bone.mid.x), y: Int(bone.mid.y))
        XCTAssertFalse(
            SyntheticClip.isBackground(on),
            "the bone is drawn at its display midpoint (\(bone.mid.x), \(bone.mid.y)) — got \(on)")

        let far = frame.pixel(x: 30, y: 150)
        XCTAssertTrue(
            SyntheticClip.isBackground(far),
            "a pixel far from the skeleton is still the background — got \(far)")

        for x in try stripSamples(analysis: analysis) {
            guard x >= 0, x < SyntheticClip.displayWidth else { continue }
            let pixel = frame.pixel(x: x, y: SyntheticClip.displayHeight - 2)
            XCTAssertFalse(
                SyntheticClip.isBackground(pixel),
                "the strip's coloured bin at x=\(x) is drawn — got \(pixel)")
        }
    }

    /// The bone the pixel check reads — `left shoulder → left elbow`, a
    /// vertical line at x = 92 of the display frame — with its midpoint and
    /// band **out of the diagram the export drew** (the analysis's own
    /// processed positions, not the test's input ones).
    private func checkedBone(analysis: Analysis) throws -> (
        mid: CGPoint, band: SeverityBand
    ) {
        let diagram = StressDiagram.frame(
            Self.midIndex, analysis: analysis, reference: nil,
            height: Double(SyntheticClip.displayHeight))
        let bone = try XCTUnwrap(
            diagram.bones.first {
                ($0.from == .leftShoulder && $0.to == .leftElbow)
                    || ($0.from == .leftElbow && $0.to == .leftShoulder)
            },
            "the synthetic body has a shoulder→elbow bone")
        let positions = Dictionary(
            uniqueKeysWithValues: diagram.joints.map { ($0.joint, $0.point) })
        let start = try XCTUnwrap(positions[bone.from])
        let end = try XCTUnwrap(positions[bone.to])
        return (
            CGPoint(x: (start.x + end.x) / 2, y: (start.y + end.y) / 2),
            bone.band
        )
    }

    /// The x positions to sample the strip at: the middle of every bin the
    /// analysis coloured, through the **same** `SessionSummary.heatStrip`
    /// call (same reference, same cap) the export makes — so "the strip is
    /// drawn" is checked where it has something to say, rather than at a
    /// grey bin no frame ever fell into.
    private func stripSamples(analysis: Analysis) throws -> [Int] {
        let bins = SessionSummary.heatStrip(
            analysis: analysis, reference: nil,
            bins: Swift.min(120, Swift.max(analysis.features.tMs.count, 1)))
        let first = try XCTUnwrap(bins.first, "the strip has bins").startMs
        let last = try XCTUnwrap(bins.last, "the strip has bins").endMs
        let span = Double(last - first)
        XCTAssertGreaterThan(span, 0, "the clip has a span")
        let samples = bins.filter { $0.band != .neutral }.map { bin -> Int in
            let middle = Double(bin.startMs + bin.endMs) / 2
            return Int(((middle - Double(first)) / span * Double(SyntheticClip.displayWidth))
                .rounded())
        }
        XCTAssertGreaterThan(samples.count, 10, "the held clip has coloured bins to sample")
        return samples
    }
}

/// Cancels its export on the first **real** progress call — past 0, i.e.
/// after at least one frame was written, so there is a partial file for the
/// run to delete.
///
/// `@unchecked Sendable` because the progress closure (called from the
/// export's background task) and the test (after the `await`) both touch it:
/// every write is one word, the export serialises its own progress calls,
/// and the task is only ever read once it has been awaited.
private final class CancelBox: @unchecked Sendable {
    var task: Task<Void, Error>?
    private(set) var cancelledAt: Double?

    func note(_ value: Double) {
        guard value > 0, cancelledAt == nil else { return }
        cancelledAt = value
        task?.cancel()
    }
}
