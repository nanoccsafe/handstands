import AVFoundation

/// Which of the device's two wide-angle cameras the Record screen uses
/// (chainlink #95): the preference behind the flip button, stored as
/// `@AppStorage("cameraPosition")` and read by `CaptureService` the same way
/// the cue settings are.
///
/// The raw values are the strings the stored key holds: `"back"` and
/// `"front"`. `CameraPosition.default` is what a fresh install shows and what
/// any unknown stored value resolves to.
enum CameraPosition: String, CaseIterable, Codable, Sendable {
    /// The rear camera — the default, and what every take before this issue
    /// used.
    case back
    /// The front (selfie) camera: the athlete can see the live feed while
    /// setting up alone, and the *preview* is mirrored (the natural selfie
    /// view) while the file and the pose frames are not.
    case front

    /// A fresh install's choice, and the fallback for an unknown stored
    /// value.
    static let `default`: CameraPosition = .back

    /// This choice's `AVCaptureDevice` position.
    var avPosition: AVCaptureDevice.Position {
        switch self {
        case .back: .back
        case .front: .front
        }
    }

    /// The other camera — what the flip button switches to.
    var opposite: CameraPosition {
        switch self {
        case .back: .front
        case .front: .back
        }
    }
}
