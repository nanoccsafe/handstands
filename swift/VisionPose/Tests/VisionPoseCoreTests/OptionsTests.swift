// Command-line parsing.

import XCTest

@testable import VisionPoseCore

final class OptionsTests: XCTestCase {
    func testTheMinimalCommandLine() throws {
        let options = try Options.parse([
            "--video", "clip.mp4", "--out", "clip.csv",
        ])
        XCTAssertEqual(options.videoPath, "clip.mp4")
        XCTAssertEqual(options.outPath, "clip.csv")
        XCTAssertEqual(options.rotate, .none)
        XCTAssertNil(options.maxFrames)
        XCTAssertNil(options.expectFrames)
        XCTAssertNil(options.clipId)
    }

    func testEveryFlag() throws {
        let options = try Options.parse([
            "--video", "a b.mp4",
            "--rotate", "auto",
            "--out", "/tmp/x/y.csv",
            "--max-frames", "12",
            "--expect-frames", "244",
            "--clip-id", "6508f9b355bd",
        ])
        XCTAssertEqual(options.videoPath, "a b.mp4")
        XCTAssertEqual(options.rotate, .auto)
        XCTAssertEqual(options.outPath, "/tmp/x/y.csv")
        XCTAssertEqual(options.maxFrames, 12)
        XCTAssertEqual(options.expectFrames, 244)
        XCTAssertEqual(options.clipId, "6508f9b355bd")
        XCTAssertEqual(options.videoFileName, "a b.mp4")
        XCTAssertEqual(options.manifestPath, "/tmp/x/y.json")
    }

    func testThe180ModeIsSpelledWithDigits() throws {
        let options = try Options.parse(["--video", "a.mp4", "--out", "a.csv", "--rotate", "180"])
        XCTAssertEqual(options.rotate, .halfTurn)
    }

    func testRequiredFlags() {
        XCTAssertThrowsError(try Options.parse(["--out", "a.csv"])) {
            XCTAssertEqual($0 as? CLIError, .missingRequired(flag: "--video"))
        }
        XCTAssertThrowsError(try Options.parse(["--video", "a.mp4"])) {
            XCTAssertEqual($0 as? CLIError, .missingRequired(flag: "--out"))
        }
    }

    func testBadArguments() {
        XCTAssertThrowsError(try Options.parse(["--video", "a.mp4", "--out", "a.csv", "--nope"])) {
            XCTAssertEqual($0 as? CLIError, .unknownArgument("--nope"))
        }
        XCTAssertThrowsError(try Options.parse(["--video"])) {
            XCTAssertEqual($0 as? CLIError, .missingValue(flag: "--video"))
        }
        XCTAssertThrowsError(
            try Options.parse(["--video", "a.mp4", "--out", "a.csv", "--rotate", "90"])
        ) { XCTAssertEqual($0 as? CLIError, .invalidValue(flag: "--rotate", value: "90", expected: "none, 180 or auto")) }
        for bad in ["0", "-1", "x"] {
            XCTAssertThrowsError(
                try Options.parse(["--video", "a.mp4", "--out", "a.csv", "--max-frames", bad])
            ) { XCTAssertEqual($0 as? CLIError, .invalidValue(flag: "--max-frames", value: bad, expected: "a whole number >= 1")) }
        }
    }

    func testHelp() {
        XCTAssertThrowsError(try Options.parse(["--help"])) {
            XCTAssertEqual($0 as? CLIError, .helpRequested)
        }
        XCTAssertThrowsError(try Options.parse(["-h"])) {
            XCTAssertEqual($0 as? CLIError, .helpRequested)
        }
        XCTAssertTrue(Options.usage.contains("--rotate"))
    }

    func testTheManifestReplacesTheCsvExtension() throws {
        let options = try Options.parse(["--video", "a.mp4", "--out", "dir/clip.csv"])
        XCTAssertEqual(options.manifestPath, "dir/clip.json")
    }
}
