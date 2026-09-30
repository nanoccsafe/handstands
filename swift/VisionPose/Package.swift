// swift-tools-version: 6.0
import PackageDescription

/// Apple Vision body-pose keypoints, written in the same schema as the
/// MediaPipe runner (`pipeline/handstand/pose_mediapipe.py`) so the two can be
/// compared frame by frame.
///
/// * ``VisionPoseCore`` — the maths: coordinate conversion, the 180° map-back,
///   the display transform, the joint-name mapping and the ``--rotate auto``
///   rule. Pure functions only, no I/O, so ``swift test`` covers all of it.
///   Builds for iOS and macOS.
/// * ``VisionPoseKit`` — the shared backend (chainlink #82): ``PoseService``,
///   the Vision call behind ``BodyPoseDetecting``, and ``DisplayFrames``. One
///   implementation of "frame in, shared schema out" for the macOS runner *and*
///   the iOS app (which gets MediaPipe as a second backend in #45).
/// * ``vision-pose`` — the executable: AVAssetReader decode, Vision inference
///   through ``VisionPoseService``, CSV out.
let package = Package(
    name: "VisionPose",
    platforms: [.macOS(.v15), .iOS(.v17)],
    products: [
        .library(name: "VisionPoseCore", targets: ["VisionPoseCore"]),
        .library(name: "VisionPoseKit", targets: ["VisionPoseKit"]),
        .executable(name: "vision-pose", targets: ["vision-pose"]),
    ],
    dependencies: [
        // The shared schema (`Joint`, `Keypoint`, `PostProcessInputFrame`) —
        // the same package the app links, resolved from this repo.
        .package(path: "../HandstandCore"),
    ],
    targets: [
        .target(name: "VisionPoseCore"),
        .target(
            name: "VisionPoseKit",
            dependencies: [
                "VisionPoseCore",
                .product(name: "HandstandCore", package: "HandstandCore"),
            ],
            linkerSettings: [
                .linkedFramework("Vision"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("CoreImage"),
            ]
        ),
        .executableTarget(
            name: "vision-pose",
            dependencies: ["VisionPoseCore", "VisionPoseKit"]
        ),
        .testTarget(
            name: "VisionPoseCoreTests",
            dependencies: ["VisionPoseCore"]
        ),
        .testTarget(
            name: "VisionPoseKitTests",
            dependencies: [
                "VisionPoseCore",
                "VisionPoseKit",
                .product(name: "HandstandCore", package: "HandstandCore"),
            ]
        ),
    ]
)
