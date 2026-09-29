// swift-tools-version: 6.0
import PackageDescription

/// The on-device port of the Python pipeline, so the iOS app (chainlink #44)
/// can run the same maths the workstation runs.
///
/// * `HandstandCore` — pure value types and pure functions ported from
///   `pipeline/handstand/` (joints, keypoints, rotation, body frame,
///   postprocess; the phases / features ports land in later issues). No I/O,
///   no UIKit or AppKit, so the same target builds for iOS and macOS and
///   `swift test` covers all of it.
/// * `HandstandCoreTests` — XCTest mirror of the Python tests, plus a tiny
///   JSON fixture read through `Bundle.module` to prove resources work for the
///   parity fixtures of chainlink #25/#42.
///
/// No third-party dependencies. The sibling `swift/VisionPose` package
/// (chainlink #15) is separate and untouched by this one.
let package = Package(
    name: "HandstandCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "HandstandCore", targets: ["HandstandCore"]),
    ],
    targets: [
        .target(name: "HandstandCore"),
        .testTarget(
            name: "HandstandCoreTests",
            dependencies: ["HandstandCore"],
            resources: [
                .copy("Fixtures"),
            ]
        ),
    ]
)
