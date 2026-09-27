// Command-line parsing for `vision-pose`, kept in the core so it can be tested
// without decoding a video.
//
//     vision-pose --video <path> --rotate none|180|auto --out <file.csv>
//                 [--max-frames N] [--expect-frames N] [--clip-id ID]
//
// `--expect-frames` is the MediaPipe runner's frame count for the same clip.
// OpenCV decodes every frame and AVFoundation does not have to, so the two can
// differ; when they do the runner says so on stderr rather than letting a
// bake-off compare frames that were never decoded.

import Foundation

/// Which frames are turned 180° before inference.
public enum RotateMode: String, CaseIterable, Equatable, Sendable {
    /// Feed frames exactly as displayed.
    case none
    /// Turn every frame 180°.
    case halfTurn = "180"
    /// Turn a frame when the *previous* frame's body looked upside down.
    case auto

    /// Is a given frame rotated, given the previous frame's judgement?
    ///
    /// - Parameters:
    ///   - mode: the `--rotate` mode.
    ///   - rotateNext: the ``AutoRotation`` state after the previous frame.
    public func isRotated(_ rotateNext: Bool) -> Bool {
        switch self {
        case .none: return false
        case .halfTurn: return true
        case .auto: return rotateNext
        }
    }
}

/// A parsed command line.
public struct Options: Equatable, Sendable {
    public var videoPath: String
    public var rotate: RotateMode
    public var outPath: String
    /// Stop after this many frames; `nil` for all of them.
    public var maxFrames: Int?
    /// The frame count another runner produced for this clip, for the check
    /// above.
    public var expectFrames: Int?
    /// Recorded in the manifest next to the CSV; the pipeline's clip id.
    public var clipId: String?

    public init(
        videoPath: String,
        rotate: RotateMode,
        outPath: String,
        maxFrames: Int? = nil,
        expectFrames: Int? = nil,
        clipId: String? = nil
    ) {
        self.videoPath = videoPath
        self.rotate = rotate
        self.outPath = outPath
        self.maxFrames = maxFrames
        self.expectFrames = expectFrames
        self.clipId = clipId
    }

    /// Where the run manifest is written: the CSV path with `.json`.
    public var manifestPath: String {
        (outPath as NSString).deletingPathExtension + ".json"
    }

    /// The file name of the video, for the manifest.
    public var videoFileName: String {
        (videoPath as NSString).lastPathComponent
    }
}

/// A command line that could not be used.
public enum CLIError: Error, Equatable, CustomStringConvertible {
    case missingValue(flag: String)
    case unknownArgument(String)
    case missingRequired(flag: String)
    case invalidValue(flag: String, value: String, expected: String)
    case helpRequested

    public var description: String {
        switch self {
        case .missingValue(let flag):
            return "\(flag) needs a value"
        case .unknownArgument(let argument):
            return "unknown argument: \(argument)"
        case .missingRequired(let flag):
            return "\(flag) is required"
        case .invalidValue(let flag, let value, let expected):
            return "\(flag) got '\(value)', expected \(expected)"
        case .helpRequested:
            return Options.usage
        }
    }
}

extension Options {
    /// The `--help` text.
    public static let usage = """
        usage: vision-pose --video <path> --rotate none|180|auto --out <file.csv>
                           [--max-frames N] [--expect-frames N] [--clip-id ID]

          --video PATH          the clip to run on
          --rotate MODE         none, 180 or auto (default: none)
          --out FILE.csv        where to write the keypoints
          --max-frames N        stop after N frames (default: all)
          --expect-frames N     warn if the decoded frame count differs from N
          --clip-id ID          recorded in the run manifest
          -h, --help            this text
        """

    /// Parse a command line (without the executable name).
    public static func parse(_ arguments: [String]) throws -> Options {
        var videoPath: String?
        var rotate = RotateMode.none
        var outPath: String?
        var maxFrames: Int?
        var expectFrames: Int?
        var clipId: String?

        var index = arguments.startIndex
        while index < arguments.endIndex {
            let argument = arguments[index]
            func value() throws -> String {
                let next = arguments.index(after: index)
                guard next < arguments.endIndex else { throw CLIError.missingValue(flag: argument) }
                index = next
                return arguments[next]
            }
            switch argument {
            case "--video":
                videoPath = try value()
            case "--rotate":
                let raw = try value()
                guard let mode = RotateMode(rawValue: raw) else {
                    throw CLIError.invalidValue(flag: "--rotate", value: raw, expected: "none, 180 or auto")
                }
                rotate = mode
            case "--out":
                outPath = try value()
            case "--max-frames":
                maxFrames = try positiveInt(value(), flag: "--max-frames")
            case "--expect-frames":
                expectFrames = try positiveInt(value(), flag: "--expect-frames")
            case "--clip-id":
                clipId = try value()
            case "-h", "--help":
                throw CLIError.helpRequested
            default:
                throw CLIError.unknownArgument(argument)
            }
            index = arguments.index(after: index)
        }

        guard let videoPath else { throw CLIError.missingRequired(flag: "--video") }
        guard let outPath else { throw CLIError.missingRequired(flag: "--out") }
        return Options(
            videoPath: videoPath,
            rotate: rotate,
            outPath: outPath,
            maxFrames: maxFrames,
            expectFrames: expectFrames,
            clipId: clipId
        )
    }

    private static func positiveInt(_ value: String, flag: String) throws -> Int {
        guard let number = Int(value), number >= 1 else {
            throw CLIError.invalidValue(flag: flag, value: value, expected: "a whole number >= 1")
        }
        return number
    }
}
