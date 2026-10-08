import AVFoundation
import Foundation
import HandstandCore

// --------------------------------------------------------------------------- #
// The voice of live mode (chainlink #91): one protocol the service talks to,
// one synthesizer behind it. The protocol is what makes the wiring test able to
// hear the cues in order without an audio session anywhere near it.
// --------------------------------------------------------------------------- #

/// What the app needs of a voice: say this cue.
protocol CueSpeaking: AnyObject {
    func speak(_ cue: Cue)
}

/// The real voice — `AVSpeechSynthesizer`, one utterance per cue.
///
/// Four decisions, all from the issue:
///
/// * **The audio session is `.playback` with `.duckOthers` and
///   `.mixWithOthers`** — background music keeps playing; the cues are mixed
///   in rather than taking the session over, and whatever else is audible
///   ducks while a cue is said.
/// * **An utterance is never interrupted.** Nothing here calls
///   `stopSpeaking`: the synthesizer queues the next cue behind the one
///   already saying its words, so "Line. Hold it." is always heard whole even
///   if a mark falls due while it is being said. The debounce in
///   `CueScheduler` is what keeps that queue short.
/// * **A rate slightly slower than default**, so the words stay clear over the
///   noise of a gym.
/// * **Recording may capture the speech.** The writer currently records video
///   only, so nothing is leaked into the take today; if a microphone is ever
///   added to `RecordingWriter`, the cues will be on the recording's audio —
///   documented as acceptable in `docs/ios.md`.
final class SpeechCueSpeaker: CueSpeaking {
    private let synthesizer = AVSpeechSynthesizer()
    private var prepared = false

    func speak(_ cue: Cue) {
        prepareAudio()
        let utterance = AVSpeechUtterance(string: CueText.phrase(for: cue))
        // Slightly slower than `AVSpeechUtteranceDefaultSpeechRate`
        // (~0.5 → ~0.45): clear enough to follow mid-handstand, not so slow
        // it drags.
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.9
        synthesizer.speak(utterance)
    }

    /// Once per speaker: the category the issue asks for, then active. Both
    /// are `try?` — a phone mid-phone-call refuses the session, and a cue not
    /// spoken is better than a crash over it.
    private func prepareAudio() {
        guard !prepared else { return }
        prepared = true
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(
            .playback, mode: .default, options: [.duckOthers, .mixWithOthers])
        try? session.setActive(true)
    }
}
