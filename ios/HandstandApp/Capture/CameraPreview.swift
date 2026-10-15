import AVFoundation
import SwiftUI

/// The live camera picture: an `AVCaptureVideoPreviewLayer` in a `UIView`,
/// which is the only way SwiftUI gets one.
///
/// The layer shows `CaptureEngine`'s session (unstarted when the view is
/// made, running a moment later — the standard hand-off) and turns itself
/// portrait in `layoutSubviews`, to match the buffers the same session
/// rotates into the recording and the framing guide.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer {
            // Safe by construction: `layerClass` above is the preview layer.
            layer as! AVCaptureVideoPreviewLayer
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer.frame = bounds
            previewLayer.videoGravity = .resizeAspectFill
            applyPortrait()
        }

        /// Turn the preview portrait, to match the buffers the same session
        /// rotates into the recording and the framing guide. Called from
        /// `layoutSubviews` and, after a camera flip, from `updateUIView`:
        /// swapping in a new session does not by itself schedule a layout
        /// pass, so without this the flipped camera's connection could come
        /// up sideways.
        func applyPortrait() {
            if let connection = previewLayer.connection,
               connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.previewLayer.session = session
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {
        if view.previewLayer.session !== session {
            view.previewLayer.session = session
            // The new session's connection is not rotated yet and no layout
            // pass is guaranteed, so rotate it now and ask for the frame too.
            view.applyPortrait()
            view.setNeedsLayout()
        }
    }
}
