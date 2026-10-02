import AVFoundation
import CoreVideo
import Darwin
import Foundation
import XCTest

@testable import HandstandApp
import HandstandCore
import VisionPoseKit

/// The real-clip parity check (chainlink #45): `MediaPipeClipExtractor` on a
/// private video against the pipeline's own output,
/// `data/keypoints/mediapipe/best/<clip>.parquet`.
///
/// **Skipped** unless both environment variables are set — the real clips
/// and their keypoints are private (chainlink #80) and never committed, so
/// the lead points this test at a pair during review:
///
///     HANDSTAND_MP_PARITY_VIDEO     a real clip (absolute path)
///     HANDSTAND_MP_PARITY_PARQUET    its mediapipe/best parquet (absolute path)
///     HANDSTAND_MP_PARITY_PYTHON     optional: an interpreter with pandas +
///                                    pyarrow (default `python3`); point it
///                                    at the pipeline's uv venv if the
///                                    system python has no parquet reader
///
/// The parquet is read on the host by that python (pandas), everything else
/// runs in the test. Two numbers are printed and gated: orientation must
/// agree on ≥ 95 % of frames, and the median joint distance must be under
/// 0.05 × the image height.
final class MediaPipeClipParityTests: XCTestCase {
    func testTheAppAgreesWithThePipelineOnARealClip() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard
            let videoPath = environment["HANDSTAND_MP_PARITY_VIDEO"],
            let parquetPath = environment["HANDSTAND_MP_PARITY_PARQUET"],
            FileManager.default.fileExists(atPath: videoPath),
            FileManager.default.fileExists(atPath: parquetPath)
        else {
            throw XCTSkip(
                "HANDSTAND_MP_PARITY_VIDEO / HANDSTAND_MP_PARITY_PARQUET are not both set to "
                    + "existing files — the real clip and its "
                    + "data/keypoints/mediapipe/best parquet are private (chainlink #80); "
                    + "the lead runs this during review")
        }
        guard let modelURL = PoseBackend.mediapipeModelURL else {
            throw XCTSkip(
                "pose_landmarker_full.task is not in the bundle — "
                    + "run tools/ios/fetch_mediapipe_model.sh and rebuild")
        }

        // 1. The pipeline's answer, read from the parquet by python (pandas
        //    + pyarrow are the pipeline's own dependencies).
        let pipeline = try Self.readParquet(
            parquetPath,
            python: environment["HANDSTAND_MP_PARITY_PYTHON"] ?? "python3")

        // 2. The app's answer: every frame of the same video (`maxFps 0`,
        //    which is how the pipeline reads it), both passes, the choice.
        let source = VideoFrameSource(url: URL(fileURLWithPath: videoPath), maxFps: 0)
        let report = try await MediaPipeClipExtractor.run(source: source, modelURL: modelURL)
        let imageHeight = try await source.displaySize().height

        // 3. Frame alignment: both readers keep every frame of the clip in
        //    order, so index is frame identity. A disagreement there is a
        //    finding, not something to paper over.
        let appCount = report.frames.count
        let pipelineCount = pipeline.rotated.count
        guard appCount == pipelineCount, appCount > 0 else {
            return XCTFail(
                "frame counts differ: the app extracted \(appCount) frames, the pipeline "
                    + "wrote \(pipelineCount) — the two readers disagree about the clip")
        }

        // 4. The orientation: the decision itself, frame for frame.
        var orientationMatches = 0
        for (index, rotated) in report.rotated.enumerated() where rotated == pipeline.rotated[index] {
            orientationMatches += 1
        }
        let agreement = Double(orientationMatches) / Double(appCount)

        // 5. The keypoints: one distance per (frame, joint) both sides have.
        var distances: [Double] = []
        for (index, frame) in report.frames.enumerated() {
            let frameIndex = pipeline.frameIndices[index]
            guard let pipelineJoints = pipeline.joints[frameIndex] else { continue }
            for joint in Joint.allCases {
                guard let app = frame.joints[joint],
                    let expected = pipelineJoints[joint.rawValue]
                else { continue }
                let dx = app.x - expected.x
                let dy = app.y - expected.y
                distances.append((dx * dx + dy * dy).squareRoot())
            }
        }
        guard !distances.isEmpty else {
            return XCTFail(
                "no joint pairs to compare — the parquet or the extraction carried no joints")
        }
        let medianDistance = distances.sorted(by: <)[distances.count / 2]
        let distanceGate = 0.05 * imageHeight

        // The numbers the gates are read from — printed, so the review log
        // carries them even when both pass.
        print(
            String(
                format:
                    "mediapipe parity: %d frames | orientation agreement %.1f%% "
                        + "(gate 95.0%%, %d/%d) | median joint distance %.2f px "
                        + "(gate %.2f px = 0.05 × height %.0f, %d pairs)",
                appCount, agreement * 100, orientationMatches, appCount,
                medianDistance, distanceGate, imageHeight, distances.count))

        XCTAssertGreaterThanOrEqual(
            agreement, 0.95,
            "the app must choose the pipeline's orientation on ≥ 95 % of frames; "
                + "printed above for the review")
        XCTAssertLessThan(
            medianDistance, distanceGate,
            "the median joint distance must be below 0.05 × the image height; "
                + "printed above for the review")
    }

    // MARK: - The parquet, through python

    /// One frame of the pipeline's answer.
    private struct PipelineFrame {
        var rotated: Bool = false
        var joints: [String: (x: Double, y: Double)] = [:]
    }

    /// The parquet's answer, in frame order (the file is long-format: one
    /// row per (frame, joint)).
    private struct PipelineAnswer {
        var frameIndices: [Int] = []
        var rotated: [Bool] = []
        var joints: [Int: [String: (x: Double, y: Double)]] = [:]
    }

    /// Single-quote a shell argument (`'…'`, with embedded quotes closed and
    /// reopened) — the path and the script both go through `/bin/sh` this way.
    private static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// Run one shell line and capture stdout to the end — POSIX
    /// `posix_spawn` with a pipe of our own, because iOS Foundation offers
    /// neither `Process` nor `popen` to Swift ("process spawning is
    /// unavailable"; the compiler points at posix_spawn). stderr is
    /// inherited, so a failing interpreter prints its own message straight
    /// into the test log. Throws with the exit status when the command
    /// fails, and the captured bytes are returned only for status 0.
    private static func runCapturingStdout(_ command: String) throws -> Data {
        // `posix_spawn_file_actions_t` is an opaque pointer; every
        // `posix_spawn_file_actions_*` call wants the address of the optional
        // holding it, so all of them take `&created`.
        var created: posix_spawn_file_actions_t?
        guard posix_spawn_file_actions_init(&created) == 0, created != nil else {
            throw NSError(
                domain: "MediaPipeClipParityTests", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "posix_spawn_file_actions_init failed"])
        }
        defer { posix_spawn_file_actions_destroy(&created) }

        var pipeFds: [Int32] = [0, 0]
        guard pipe(&pipeFds) == 0 else {
            throw NSError(
                domain: "MediaPipeClipParityTests", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "pipe() failed with errno \(errno)"])
        }
        // The child: stdout → the pipe's write end; both pipe fds closed in
        // the child afterwards, so only fd 1 keeps the write end alive.
        posix_spawn_file_actions_adddup2(&created, pipeFds[1], STDOUT_FILENO)
        posix_spawn_file_actions_addclose(&created, pipeFds[0])
        posix_spawn_file_actions_addclose(&created, pipeFds[1])

        var arguments: [UnsafeMutablePointer<CChar>?] = ["sh", "-c", command].map {
            strdup($0)!
        }
        arguments.append(nil)
        defer {
            for pointer in arguments {
                if let pointer { free(pointer) }
            }
        }

        var pid: pid_t = 0
        let spawned = posix_spawn(&pid, "/bin/sh", &created, nil, &arguments, nil)
        close(pipeFds[1])  // the parent's write end: EOF waits only on the child now.
        guard spawned == 0 else {
            close(pipeFds[0])
            throw NSError(
                domain: "MediaPipeClipParityTests", code: Int(spawned),
                userInfo: [
                    NSLocalizedDescriptionKey: "posix_spawn(/bin/sh) failed: \(spawned)"
                ])
        }

        var output = Data()
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            let count = read(pipeFds[0], &buffer, buffer.count)
            if count > 0 {
                output.append(contentsOf: buffer[0..<count])
            } else if count == 0 {
                break  // EOF: the child exited and every write end is closed.
            } else if errno == EINTR {
                continue
            } else {
                close(pipeFds[0])
                _ = waitpid(pid, nil, 0)
                throw NSError(
                    domain: "MediaPipeClipParityTests", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "could not read the parquet dump"])
            }
        }
        close(pipeFds[0])

        var status: Int32 = 0
        waitpid(pid, &status, 0)
        let exitCode = (status >> 8) & 0xFF
        guard exitCode == 0 else {
            throw NSError(
                domain: "MediaPipeClipParityTests", code: Int(exitCode),
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "the parquet dump exited \(exitCode) — set "
                        + "HANDSTAND_MP_PARITY_PYTHON to an interpreter with pandas and "
                        + "pyarrow (the pipeline's venv); python's own message is in "
                        + "the test log above"
                ])
        }
        return output
    }

    /// Dump the parquet to JSON records with `python -c` and parse them —
    /// the test host has no parquet reader of its own.
    private static func readParquet(_ path: String, python: String) throws -> PipelineAnswer {
        // Selected columns keep the payload small; NaN x/y (an undetected
        // frame) arrives as null and is skipped, like everywhere else.
        let script = """
            import sys
            import pandas as pd
            df = pd.read_parquet(sys.argv[1])
            cols = [c for c in ("frame_idx", "joint", "x", "y", "rotated")
                    if c in df.columns]
            print(df[cols].to_json(orient="records"))
            """
        // One shell line: interpreter, script, path — each quoted so a path
        // with spaces cannot break the command. iOS Foundation has neither
        // `Process` nor `popen` (Swift marks both unavailable there), so the
        // subprocess goes through POSIX `posix_spawn` with a pipe of our own;
        // python's stderr is inherited and lands in the test log.
        let command = "\(python) -c \(shellQuote(script)) \(shellQuote(path))"
        let outData = try runCapturingStdout(command)

        let rows = try JSONSerialization.jsonObject(with: outData) as? [[String: Any]] ?? []
        var answer = PipelineAnswer()
        var byIndex: [Int: PipelineFrame] = [:]
        for row in rows {
            guard
                let frameNumber = (row["frame_idx"] as? NSNumber)?.intValue,
                let joint = row["joint"] as? String
            else { continue }
            var frame = byIndex[frameNumber, default: PipelineFrame()]
            if let rotated = row["rotated"] as? Bool {
                frame.rotated = rotated
            }
            if let x = row["x"] as? Double, let y = row["y"] as? Double,
                x.isFinite, y.isFinite
            {
                frame.joints[joint] = (x: x, y: y)
            }
            byIndex[frameNumber] = frame
        }
        answer.frameIndices = byIndex.keys.sorted()
        for index in answer.frameIndices {
            let frame = byIndex[index] ?? PipelineFrame()
            answer.rotated.append(frame.rotated)
            answer.joints[index] = frame.joints
        }
        return answer
    }
}
