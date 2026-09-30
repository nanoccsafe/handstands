import CoreGraphics
import Foundation
import HandstandCore

// --------------------------------------------------------------------------- #
// Where the diagram goes (#48): display pixels → view points.
//
// The joints `Analyzer` measured are in *display pixels* (the frame a person
// watching the video sees), while the overlay is drawn in *view points* over
// a player whose picture is aspect-fitted into a box of whatever size the
// screen gave it. Two pure functions do that mapping, and they are the only
// place the two coordinate systems ever meet — so the drawing code can be
// unit-tested without a view, and the test can prove a portrait video in a
// landscape view lands where `AVPlayerLayer`'s `.resizeAspect` puts it.
//
// `videoRect(videoSize:in:)` is exactly `AVLayerVideoGravity.resizeAspect`:
// the largest rect of the video's aspect ratio that fits inside the view,
// centred — so the drawn picture and the played picture cannot drift apart.
// --------------------------------------------------------------------------- #

/// The geometry the stress-diagram overlay is drawn in.
enum OverlayGeometry {
    /// Where `videoSize`'s picture sits inside a view of `inSize`, with
    /// `.resizeAspect`'s rule: keep the aspect ratio, fit inside, centre.
    ///
    /// A video or view with no size (an unreadable movie, a view still
    /// laying out) answers a zero rect rather than NaN — nothing is drawn
    /// until there is somewhere real to draw it.
    static func videoRect(videoSize: CGSize, in inSize: CGSize) -> CGRect {
        guard videoSize.width > 0, videoSize.height > 0,
            inSize.width > 0, inSize.height > 0
        else {
            return CGRect.zero
        }
        let scale = min(inSize.width / videoSize.width, inSize.height / videoSize.height)
        let fitted = CGSize(
            width: videoSize.width * scale, height: videoSize.height * scale)
        return CGRect(
            x: (inSize.width - fitted.width) / 2.0,
            y: (inSize.height - fitted.height) / 2.0,
            width: fitted.width,
            height: fitted.height
        )
    }

    /// One display-pixel point as a point of the view: the video's rect
    /// scaled linearly, top-left to top-left (display `y` grows downwards,
    /// and so does the view's).
    static func viewPoint(_ point: Point2, videoSize: CGSize, rect: CGRect) -> CGPoint {
        guard videoSize.width > 0, videoSize.height > 0 else { return .zero }
        return CGPoint(
            x: rect.origin.x + point.x / videoSize.width * rect.width,
            y: rect.origin.y + point.y / videoSize.height * rect.height
        )
    }
}
