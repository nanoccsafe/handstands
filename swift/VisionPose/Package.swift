// swift-tools-version: 6.0
import PackageDescription

/// Apple Vision body-pose keypoints, written in the same schema as the
/// MediaPipe runner (`pipeline/handstand/pose_mediapipe.py`) so the two can be
/// compared frame by frame.
///
/// * ``VisionPoseCore`` — the maths: coordinate conversion, the 180° map-back,
///   the display transform, the joint-name mapping and the ``--rotate auto``
///   rule. Pure functions only, no I/O, so ``swift test`` covers all of it.
/// * ``vision-pose`` — the executable: AVAssetReader decode, Vision inference,
///   CSV out.
let package = Package(
    name: "VisionPose",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "VisionPoseCore", targets: ["VisionPoseCore"]),
        .executable(name: "vision-pose", targets: ["vision-pose"]),
    ],
    targets: [
        .target(name: "VisionPoseCore"),
        .executableTarget(
            name: "vision-pose",
            dependencies: ["VisionPoseCore"]
        ),
        .testTarget(
            name: "VisionPoseCoreTests",
            dependencies: ["VisionPoseCore"]
        ),
    ]
)
