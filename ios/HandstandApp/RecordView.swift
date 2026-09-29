import AVFoundation
import HandstandCore
import SwiftUI
import UIKit

/// The Record screen (chainlink #46): the camera fills it, a coloured border
/// says whether the whole body is in frame and what to do about it, and a
/// big button records an attempt to a file that stays on the phone.
///
/// Two things the screen owns beyond the picture: the framing message under
/// the border (from `HandstandCore`'s `FramingCheck`), and the screen staying
/// awake while it is visible — a phone that locks mid-attempt loses the take.
struct RecordView: View {
    @State private var model = CaptureService()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            switch model.phase {
            case .starting:
                ProgressView("Starting the camera…")
            case .unavailable:
                ContentUnavailableView {
                    Label("No camera", systemImage: "video.slash")
                } description: {
                    Text(model.unavailableMessage)
                }
            case .denied:
                CameraAccessNeededView {
                    openSettings()
                }
            case .ready:
                cameraScreen
            }
        }
        .navigationTitle("Record")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await model.start()
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            model.shutdown()
        }
        .overlay {
            if let info = model.finished {
                RecordingDoneView(
                    info: info,
                    onRecordAgain: { model.recordAgain() },
                    onDone: { dismiss() }
                )
                .background(Color(.systemBackground).ignoresSafeArea())
            }
        }
    }

    // MARK: - The live camera

    private var cameraScreen: some View {
        ZStack {
            if let session = model.session {
                CameraPreview(session: session)
                    .ignoresSafeArea()
            }

            // The framing border: green whole, amber wrong framing, red no
            // single body to frame. It does not eat the touches — the
            // record button below still gets them.
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(model.borderTone.color, lineWidth: 6)
                .padding(12)
                .allowsHitTesting(false)
                .animation(.easeInOut(duration: 0.15), value: model.borderTone)

            VStack(spacing: 14) {
                // What the border colour means, in words.
                Text(model.framingMessage)
                    .font(.headline)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.top, 28)

                Spacer()

                if let error = model.errorMessage {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }

                if model.isRecording {
                    Text(CaptureFormatter.elapsed(model.elapsed))
                        .font(.system(size: 46, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.white)
                        .shadow(radius: 3)
                        .accessibilityLabel("Elapsed time")
                } else {
                    // The one assumption the whole pipeline makes, kept
                    // visible until the take starts.
                    Text("Only you in the frame, phone on a tripod.")
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.white)
                        .shadow(radius: 2)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 8)
                        .background(.black.opacity(0.4), in: Capsule())
                }

                recordButton
                    .padding(.bottom, 8)
            }
            .padding(.vertical, 24)
        }
    }

    private var recordButton: some View {
        Button {
            model.toggleRecording()
        } label: {
            ZStack {
                Circle()
                    .strokeBorder(.white.opacity(0.9), lineWidth: 5)
                    .frame(width: 88, height: 88)
                if model.isRecording {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(.red)
                        .frame(width: 38, height: 38)
                        .transition(.scale)
                } else {
                    Circle()
                        .fill(.red)
                        .frame(width: 66, height: 66)
                        .transition(.scale)
                }
            }
        }
        .buttonStyle(.plain)
        .animation(.easeInOut(duration: 0.15), value: model.isRecording)
        .accessibilityLabel(model.isRecording ? "Stop recording" : "Start recording")
    }

    private func openSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }
}
