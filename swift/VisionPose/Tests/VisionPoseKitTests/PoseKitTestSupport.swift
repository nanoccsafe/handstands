// Shared scaffolding for the VisionPoseKit tests: a fake detector and blank
// pixel buffers.
//
// No test here looks at a real image of a person — the repo is public. The
// fake hands the service canned points, and the only place the real Vision
// model runs is the black-frame smoke test.

import CoreVideo
import Foundation

import VisionPoseCore
import VisionPoseKit

/// A `BodyPoseDetecting` that hands back canned detections and remembers how
/// it was called: no Vision, no model, no images.
final class FakeDetector: BodyPoseDetecting {
    /// One person, as the detector reports them: normalised points and a
    /// confidence per joint Vision reported.
    typealias Person = [VisionJoint: (point: NormalizedPoint, confidence: Double)]

    /// One entry per `detect` call; calls past the end see an empty frame,
    /// which is what a detection dropout looks like.
    private let frames: [[Person]]
    private var callIndex = 0
    /// The `rotated` flag of every call, in order.
    private(set) var rotatedCalls: [Bool] = []

    init(frames: [[Person]] = []) {
        self.frames = frames
    }

    func detect(_ frame: CVPixelBuffer, rotated: Bool) throws -> [Person] {
        rotatedCalls.append(rotated)
        defer { callIndex += 1 }
        guard callIndex < frames.count else { return [] }
        return frames[callIndex]
    }
}

/// One canned person: joints in Vision's normalised coordinates (origin
/// bottom-left) and one confidence shared by every joint.
func person(
    _ points: [VisionJoint: (x: Double, y: Double)],
    confidence: Double = 0.9
) -> FakeDetector.Person {
    points.mapValues { joint in
        (point: NormalizedPoint(x: joint.x, y: joint.y), confidence: confidence)
    }
}

/// Why a blank buffer could not be made — a test-environment failure, never a
/// code path the sources take.
enum SupportError: Error {
    case pixelBufferCreationFailed(CVReturn)
}

/// A blank `32BGRA` buffer of exactly the requested size: zeros, so Vision
/// sees a black frame and the service only has its size to read.
func makePixelBuffer(width: Int = 64, height: Int = 64) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        nil,
        &buffer
    )
    guard status == kCVReturnSuccess, let buffer else {
        throw SupportError.pixelBufferCreationFailed(status)
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    if let base = CVPixelBufferGetBaseAddress(buffer) {
        memset(base, 0, CVPixelBufferGetBytesPerRow(buffer) * CVPixelBufferGetHeight(buffer))
    }
    return buffer
}

/// Paints the buffer's left half red and its right half blue (BGRA bytes), so
/// a rotation test can tell which half ended up where.
func paintHalvesRedAndBlue(_ buffer: CVPixelBuffer) {
    guard let pixel = lockPixels(buffer, readOnly: false) else { return }
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let width = pixel.width
    for y in 0..<pixel.height {
        for x in 0..<width {
            let i = y * pixel.bytesPerRow + x * 4
            let red = x < width / 2
            pixel.bytes[i] = red ? 0 : 255  // B
            pixel.bytes[i + 1] = 0  // G
            pixel.bytes[i + 2] = red ? 255 : 0  // R
            pixel.bytes[i + 3] = 255  // A
        }
    }
}

/// The red and blue bytes of one pixel, read through a lock and unlocked
/// again. Returns `(red, blue)`.
func redAndBlue(_ buffer: CVPixelBuffer, x: Int, y: Int) -> (red: UInt8, blue: UInt8)? {
    guard let pixel = lockPixels(buffer, readOnly: true) else { return nil }
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let i = y * pixel.bytesPerRow + x * 4
    return (red: pixel.bytes[i + 2], blue: pixel.bytes[i])
}

private struct LockedPixels {
    var bytes: UnsafeMutablePointer<UInt8>
    var width: Int
    var height: Int
    var bytesPerRow: Int
}

private func lockPixels(_ buffer: CVPixelBuffer, readOnly: Bool) -> LockedPixels? {
    let flags: CVPixelBufferLockFlags = readOnly ? .readOnly : []
    guard CVPixelBufferLockBaseAddress(buffer, flags) == kCVReturnSuccess else { return nil }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
        CVPixelBufferUnlockBaseAddress(buffer, flags)
        return nil
    }
    return LockedPixels(
        bytes: base.assumingMemoryBound(to: UInt8.self),
        width: CVPixelBufferGetWidth(buffer),
        height: CVPixelBufferGetHeight(buffer),
        bytesPerRow: CVPixelBufferGetBytesPerRow(buffer)
    )
}
