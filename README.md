# Quran Recitation Practice

An interactive Android application designed to help users practice Quranic recitation with live ASR (Automatic Speech Recognition) and on-device acoustic Tajweed analysis. The app supports both Hafs 'an 'Asim and Qalun 'an Nafi' transmissions, gives near-real-time feedback on wrong / skipped words and per-rule Tajweed mistakes, and offers the whole UI in either English or Arabic with a single in-app language toggle.

## Demo

A 62-second promo (16:9, silent):

https://github.com/user-attachments/assets/41553f66-f1a8-4ca5-a871-5df12fb3e53c

## Features
- **Live Recitation Tracking:** Memorize word-tracking runs on the on-device **ONNX Runtime** FastConformer CTC speech model (`SttEngine.kt`, NeMo FastConformer fine-tuned for Quranic recitation) with a 4.0 s trailing sliding window / 1.5 s hop and a final full-tail decode repair; the Android `SpeechRecognizer` remains the Tarteel-mode path. The 166 MiB graph is **downloaded once on first use** (see [Speech model download & license](#speech-model-download--license)); only the 12 KB token map ships in the APK. Words advance along the surah text as the user speaks, with green highlights for correct words, red for wrong words, orange for skipped words.
- **Eight-Rule TajweedNet (on-device):** A small convolutional + BiGRU network (`TajweedNet`, weights in `assets/model_weights.json`) scores every word against eight tajweed rules: MADD, GHUNNAH, QALQALAH, IDGHAM, IKHFA, MEEM SAKINAH, ALLAM, IZHAR. See [tajweed.md](tajweed.md) for the architecture and [calibration.md](calibration.md) for the training pipeline.
- **Memorize + Tajweed:** When the user flips the **Tajweed** switch on the Memorize tab, a single `AudioRecord` (the Sherpa pipeline) feeds BOTH the STT stream (RMS-gated buffer) and the TajweedNet scorer (an **un-gated** real-time buffer that keeps the inter-word silence the model was trained on). The scorer uses a rule-aware window matching the training clips and a bounded ±6 s confirmation search before flagging anything, plus an early full-tail **verification decode** (`verifyTailWords`) that forgives words the sliding windows merely hid — so a correct recitation never pops a false mistake. A tentative wrong/skipped candidate is *ridden over* for at most 8 s while the confirm loop verifies it (a single unproven candidate must never freeze a whole surah); after that budget the app decides and raises it. A *verified* mistake pauses at the word with a popup naming the rule and the score, **rewinds the display to that word in the same instant**, and holds until dismissed; tapping the mic then re-seats the transcript so the user re-reads from exactly that word and continues. One dialog at a time, and each `(word, rule)` verdict speaks at most once per session. See section "Mistake surfacing" in [tajweed.md](tajweed.md).
- **Memorize + Tarteel:** Default mode. Word-tracking on the platform Android `SpeechRecognizer` — no acoustic tajweed scoring, no tajweed rules or scores. It gets the *same* session experience as Memorize + Tajweed: a wrong or skipped word freezes the highlight at that word and raises the same must-dismiss popup (the walk's own freeze **is** the verdict here — there is no confirm loop), a pause shows the "Recitation Paused" summary, stopping after finishing the range shows the completion summary, and dismissing the popup lets the very next mic tap continue from the flagged word instead of summarizing again. The mic stays on across the platform recognizer's own utterance boundaries: a final result or a benign timeout automatically opens the next listening segment, so a short pause no longer ends the session - only the user's stop does. Faster and lighter than Tajweed mode.
- **Read tab:** Surah text is rendered with tajweed-aware letter colouring (Qalqalah blue, Madd red, Al-Lam solar blue-grey / lunar teal, Ikhfa/Iqlab cyan, Idgham orange, Ghunnah green, Meem Sakinah magenta, Izhar purple). Audio for each ayah streams from a CDN, looping until the user stops it.
- **Goals & Stats tab:** Daily / Weekly recitation goals scoped to (surah, start-ayah, end-ayah) — always a specific range, with Al-Fatihah selected by default. Progress reacts to the surah+range the user just recited, so a goal only advances when the recitation falls inside its scope. Per-rule tajweed-mistake breakdown, total mistakes, wrong / skipped word counts, and a celebrated "Goal Accomplished" notification on completion.
- **Tajweed Rules Coach:** An interactive tab with one card per rule (8 cards). Each card shows the rule's name, a one-line description, a full explanation, and a randomly-picked Quranic example out of 12 curated excerpts (re-picked per click of the Refresh button), coloured in the same letter-by-letter per-rule palette as the Read tab. The example is **guaranteed to still contain the rule it was picked for** — the snippet window always keeps the trigger's cross-word partner and the finished snippet is re-detected before it is shown. Expanding the card and tapping the mic starts a recorder session scored against the example, whose verdict list covers **exactly the rules the example displays** — one row per rule the colouriser paints, so the row count matches the highlighted rules the user can see (including both rules of a two-rule word, each with its own score), treated green (applied) or red (needs practice) based on `decisionThresholdForHead`.
- **Session lifecycle & summaries:** In Memorize + Tajweed the Sherpa pipeline hands itself back 2.5 s after the last syllable; in Tarteel the platform recognizer is restarted after each utterance boundary, so the mic stays on until the user taps stop. The final repair decode is capped at 30 s so a long session can never sit silently thinking. Tarteel also caps the stopped display at the last ayah that was actually heard, so pausing cannot colour or scroll into the next ayah. A **pause** raises an honest "Recitation Paused" dialog ("Recited N of M words", the mistakes so far, no praise); a genuine **finish** raises the completion summary with the mistake words, their rules and their scores. Mistake feedback lives in the popup alone — never an inline message — and a popup is never auto-dismissed.
- **Riwayah support:** The user picks Hafs or Qalun on Memorize / Read. Hafs text is rendered from the remote API; Qalun text comes from the bundled `assets/qalun_nemo_manifest.jsonl` with a clear UI fallback notice if Qalun data isn't available for the surah. The riwayah switch lives in the Memorize/Read top bar only.
- **Two languages:** A single toggle in the top bar switches the entire app between English and Arabic (RTL layout mutations handled automatically by Compose). Every visible string — tabs, buttons, dialogs, alerts, rule names, descriptions, dialogs, goal form, stats panels, error messages — has an Arabic counterpart.
- **No-internet safety:** Opening the app offline no longer surfaces a raw exception string. The three network failure sites (surah text fetch, audio play on-error, audio prepare catch) test `ConnectivityManager` state and, when truly offline, present a friendly localized popup ("No Internet Connection — you can't listen to the Quran without it") instead of technical error text.
- **Tajweed model warm-up:** The first time the user opens Tajweed tab (or flips Memorize to Tajweed mode), `preloadTajweedModel()` parses `assets/model_weights.json` on `Dispatchers.IO` and surfaces a non-dismissable "Loading Tajweed Model…" dialog that auto-dismisses the instant the singleton is warm. Subsequent entries are instant.

## Architecture
- **UI:** Jetpack Compose (Material Design 3), single-Activity + CompositionLocals.
- **Speech Recognition:** Memorize (+Tajweed) uses the on-device **ONNX Runtime** recognizer (`SherpaPipeline.kt` + `SttEngine.kt`): a single 16 kHz mono `AudioRecord` feeds an RMS-gated accumulator (silent chunks dropped, so the CTC model only sees speech), with 4.0 s trailing-window decodes every 1.5 s, a monotonic merge log, and a final full-tail decode that repairs live-window artifacts (whole-session when the audio fits the 30 s cost cap, otherwise the last 10 s). The Tarteel path and the Tajweed Rules Coach still use Android `SpeechRecognizer` / `MicRecorderController` where applicable — Tarteel keeps that recognizer's superior accuracy and takes its transcript from it for every word-tracking, popup and summary decision, with all tajweed scoring gates off.
- **Tajweed analysis:** Port of the trained PyTorch model into pure Kotlin (`TajweedNet.kt`). Conv1x3 → BiGRU(64) → per-rule Linear heads, mirroring the training-time forward pass exactly (parity-tested in `TajweedNetParityTest`). The scorer consumes the pipeline's **un-gated** real-time buffer (`snapshotRawAudio()`), slices a rule-aware window per word, silence-trims it, and applies a bounded ±6 s confirmation search before flagging a `TAJWEED_ERROR`. One acoustic pass scores at most 6 words, a verdict must be *isolated* (a neighbour failing within 2 words means the scorer's audio boundary is offset, not a mistake) and a tentative candidate is ridden over for at most 8 s while the confirm loop judges it — see [tajweed.md](tajweed.md) §4.51–§4.69.
- **Calibration:** Thresholds in `decisionThresholdForHead()` (`RecitationApp.kt`). Each rule has a per-head cut tuned against a held-out calibration fold — see [calibration.md](calibration.md).
- **PC E2E validation:** Two Robolectric harnesses replay real Alafasy audio through the REAL ONNX model with exact phone window geometry: `PcSttEndToEndTest` (18 scenarios — no false WRONG/SKIPPED) and `PcTajweedFalsePositiveTest` (15 surahs — no false TAJWEED_ERROR/WRONG/SKIPPED). See `WORK_NOTES.txt` for the run command.
- **Data Persistence:** Room Database for user goals (surah-scoped, version 3 migration wipes legacy goals on upgrade using `fallbackToDestructiveMigration`). SharedPreferences for recited-ayah counters and the UI language. Debug WAV dumps under `files/sherpa_debug` are pruned to the newest 6 before each write, so a long recitation history cannot grow without bound.
- **Networking:** Retrofit + Moshi against `https://api.alquran.cloud/v1` for surah text and audio URLs.

## Speech model download & license

The 166 MiB int8 recognizer is **not bundled in the APK**. The first time the user opens the Tajweed tab (or flips Memorize to Tajweed), `SherpaModelDownloader` fetches it from the public mirror
[`saif96/itqan-fastconformer-quran-int8`](https://huggingface.co/saif96/itqan-fastconformer-quran-int8) into app storage (`files/sherpa_model/`), verifies the exact byte count and SHA-256, and only then lets the engine open it. The transfer is resumable: leaving the screen keeps a `.part` file, and the next attempt continues from the byte it reached. The 12 KB token map ships in `assets/stt_model/`.

The model is **NPL-1.1 (Quran-Lab No-Profit License)** — free to use, share and adapt for non-commercial purposes, with share-alike terms. The license text ships in the APK at `assets/stt_model/LICENSE`, the public mirror carries the same `LICENSE` at its root, and the full third-party attribution (upstream Muno459/Quran-Lab, NVIDIA's CC-BY-4.0 base model, training datasets and ONNX Runtime) is in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

The app must remain free: NPL-1.1 forbids charging for the model, access to it, or a feature it powers (including advertising on that feature).

## Permissions
- `INTERNET` — surah text, audio CDN, and the one-time speech-model download.
- `RECORD_AUDIO` — speech recognizer + `MicRecorderController` (the AudioRecord fallback).

## Getting Started

**Prerequisites:**  [Android Studio](https://developer.android.com/studio)

1. Open Android Studio
2. Select **Open** and choose this project directory
3. Let Gradle sync
4. Run on an emulator or physical device (mic + internet needed for full functionality; the first Tajweed run downloads the 166 MiB speech model)

## Documentation
- [How Hafs & Qalun Mapping Works](mapping.md)
- [How Tajweed Scoring Works](tajweed.md)
- [Tajweed Model Calibration & Training](calibration.md)
- [Training the model yourself (Python / PyTorch)](training/README.md)

## License

The app's own source code is licensed under the **MIT License** — see [LICENSE](LICENSE).

Because the app ships and runs the NPL-1.1 speech model, the application **as a whole** is non-commercial: no sales, no paid features, no subscriptions, and no advertising. The MIT grant covers the code only and does not license the model; the model's own terms govern the combined application. Full attribution for every bundled component is in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
