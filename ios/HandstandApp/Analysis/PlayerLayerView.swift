import AVFoundation
import SwiftUI
import UIKit

// --------------------------------------------------------------------------- #
// The video under the diagram (#48): an `AVPlayerLayer` behind SwiftUI.
//
// `VideoPlayer` (the SwiftUI control) draws its own way and exposes nothing
// to draw *on top of* with confidence, so this is the smallest representable
// that puts the picture exactly where the overlay maths says it is: an
// `AVPlayerLayer` with `.resizeAspect`, the same fit
// `OverlayGeometry.videoRect` computes — the layer and the overlay agree by
// construction, and a portrait recording in a landscape row letterboxes in
// the middle of both.
// --------------------------------------------------------------------------- #

/// One player's picture, as a SwiftUI view.
struct PlayerLayerView: UIViewRepresentable {
    /// The player whose item this layer shows.
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerContainerView {
        let view = PlayerContainerView()
        view.backgroundColor = .black
        view.playerLayer.videoGravity = .resizeAspect
        view.playerLayer.player = player
        return view
    }

    func updateUIView(_ uiView: PlayerContainerView, context: Context) {
        // The screen keeps one player for its lifetime, so this only has to
        // survive SwiftUI handing the view back to us.
        if uiView.playerLayer.player !== player {
            uiView.playerLayer.player = player
        }
    }
}

/// The UIView whose backing layer *is* an `AVPlayerLayer` — `UIView` picks
/// its layer class once, which is exactly the hook this needs.
final class PlayerContainerView: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }

    /// The layer, already typed.
    var playerLayer: AVPlayerLayer {
        layer as! AVPlayerLayer
    }
}
