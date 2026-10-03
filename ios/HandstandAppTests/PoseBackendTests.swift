import XCTest

@testable import HandstandApp
import HandstandCore
import VisionPoseKit

/// The pose backend (chainlink #45): MediaPipe is the default whenever its
/// model is in the bundle — the bake-off picked it (docs/bakeoff.md) — and
/// Apple Vision is the fallback that always works.
///
/// The model file is never committed, so availability is asked with an
/// injected `modelURL`: the same code path answers "present" and "absent"
/// whichever way this build was made.
final class PoseBackendTests: XCTestCase {
    /// A real, existing file stands in for a bundled model.
    private func fakeModel() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pose_landmarker_full-\(UUID().uuidString).task")
        try Data("not a model".utf8).write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    /// A path that cannot exist stands in for a build without the model.
    private func missingModel() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("no-such-model-\(UUID().uuidString).task")
    }

    // MARK: - Availability, with and without the model

    func testVisionIsAlwaysAvailableAndMakesItsService() throws {
        XCTAssertTrue(PoseBackend.vision.isAvailable(modelURL: nil))
        XCTAssertTrue(PoseBackend.vision.isAvailable(modelURL: try fakeModel()))
        XCTAssertTrue(PoseBackend.vision.isAvailable)

        let service = try XCTUnwrap(PoseBackend.vision.makeService())
        XCTAssertTrue(service is VisionPoseService)
        XCTAssertEqual(service.backendName, "vision")
    }

    func testMediaPipeIsAvailableOnlyWithTheModelFile() throws {
        XCTAssertTrue(PoseBackend.mediapipe.isAvailable(modelURL: try fakeModel()))
        XCTAssertFalse(PoseBackend.mediapipe.isAvailable(modelURL: missingModel()))
        XCTAssertFalse(PoseBackend.mediapipe.isAvailable(modelURL: nil))
    }

    func testMediaPipeMakesItsServiceOnlyWithTheModelFile() throws {
        // `makeService` reads the *bundle* (it has no injected path), so the
        // injected checks above are the deterministic ones; what holds either
        // way is the shape: vision always answers, an unavailable backend
        // answers nil rather than a broken service.
        XCTAssertNotNil(PoseBackend.vision.makeService())
        if !PoseBackend.mediapipe.isAvailable {
            XCTAssertNil(PoseBackend.mediapipe.makeService())
        }
        // With a service in hand, the identity is the backend's raw value —
        // what `PoseCache.backend` stamps into the document.
        if let service = PoseBackend.preferred.makeService() {
            XCTAssertEqual(service.backendName, PoseBackend.preferred.rawValue)
        }
    }

    // MARK: - preferred

    func testPreferredIsMediaPipeWithTheModelAndVisionWithout() throws {
        XCTAssertEqual(PoseBackend.preferred(modelURL: try fakeModel()), .mediapipe)
        XCTAssertEqual(PoseBackend.preferred(modelURL: missingModel()), .vision)
        XCTAssertEqual(PoseBackend.preferred(modelURL: nil), .vision)
    }

    func testPreferredMatchesWhatTheModelFileSays() {
        // The bundle-backed answer, whatever this build is: MediaPipe if the
        // model was fetched into LocalResources, Vision if not — never a
        // third thing, and the answer is itself available.
        let expected = PoseBackend.mediapipe.isAvailable ? PoseBackend.mediapipe : .vision
        XCTAssertEqual(PoseBackend.preferred, expected)
        XCTAssertTrue(PoseBackend.preferred.isAvailable)
    }

    // MARK: - The per-backend gate and version

    func testTheVisibilityGateIsPerBackendAndBothAreHalf() {
        // The gate lives in PoseBackend, one number per backend (a later
        // Vision recalibration changes exactly one of these).
        XCTAssertEqual(PoseBackend.vision.minVisibility, 0.5)
        XCTAssertEqual(PoseBackend.mediapipe.minVisibility, 0.5)
        // …and it is what reaches the pipeline: the config the run hands
        // `Analyzer.analyze` carries it, everything else the pipeline's own
        // default (so the golden parity of #25 still holds for either).
        XCTAssertEqual(PoseBackend.vision.postProcessConfig.minVisibility, 0.5)
        XCTAssertEqual(PoseBackend.mediapipe.postProcessConfig.minVisibility, 0.5)
        XCTAssertEqual(PoseBackend.mediapipe.postProcessConfig, PostProcessConfig())
    }

    @MainActor
    func testTheAnalysisVersionNamesTheBackendThatRan() {
        XCTAssertEqual(PoseBackend.mediapipe.analysisVersion, "mediapipe-1")
        XCTAssertEqual(PoseBackend.vision.analysisVersion, "vision-1")
        // What the app writes follows the same rule as `preferred`, so a
        // cache written by one backend is never replayed by the other — the
        // old `"vision-1"` caches recompute once MediaPipe decides frames.
        // (`AnalysisService` is main-actor isolated, hence the annotation.)
        XCTAssertEqual(AnalysisService.analysisVersion, PoseBackend.preferred.analysisVersion)
    }

    // MARK: - Names

    func testNamesAndOrder() {
        XCTAssertEqual(PoseBackend.allCases.map(\.rawValue), ["vision", "mediapipe"])
        XCTAssertEqual(PoseBackend.vision.displayName, "Apple Vision")
        XCTAssertEqual(PoseBackend.mediapipe.displayName, "MediaPipe")
    }

    // MARK: - The 33 → 15 landmark map, as the app sees it

    func testTheAppSeesTheSameLandmarkMapAsThePipeline() {
        // `mediaPipeLandmarkIndex` lives in HandstandCore (public), and it
        // is the table the extractor maps through — spelled here against
        // pose_mediapipe.JOINT_NAMES so a rename on either side fails.
        XCTAssertEqual(mediaPipeLandmarkIndex.count, 15)
        let expected: [Joint: Int] = [
            .nose: 0,
            .leftShoulder: 11, .rightShoulder: 12,
            .leftElbow: 13, .rightElbow: 14,
            .leftWrist: 15, .rightWrist: 16,
            .leftHip: 23, .rightHip: 24,
            .leftKnee: 25, .rightKnee: 26,
            .leftAnkle: 27, .rightAnkle: 28,
            .leftFootIndex: 31, .rightFootIndex: 32,
        ]
        XCTAssertEqual(mediaPipeLandmarkIndex, expected)
        // The 12 joints the per-frame score averages are the body joints of
        // that table, in schema order — no face, no fingers.
        XCTAssertEqual(OrientationChooser.mainJointIndices, [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
    }
}
