# Quran Recitation Practice

An interactive Android application designed to help users practice Quranic recitation with live ASR (Automatic Speech Recognition) and on-device acoustic Tajweed analysis. The app supports both Hafs 'an 'Asim and Qalun 'an Nafi' transmissions, gives near-real-time feedback on wrong / skipped words and per-rule Tajweed mistakes, and offers the whole UI in either English or Arabic with a single in-app language toggle.

## Features
- **Live Recitation Tracking:** Uses Android's `SpeechRecognizer` with partial-result chunking to transcribe the user in real time. Words advance along the surah text as the user speaks, with green highlights for correct words, red for wrong words, orange for skipped words.
- **Eight-Rule TajweedNet (on-device):** A small convolutional + BiGRU network (`TajweedNet`, weights in `assets/model_weights.json`) scores every word against eight tajweed rules: MADD, GHUNNAH, QALQALAH, IDGHAM, IKHFA, MEEM SAKINAH, ALLAM, IZHAR. See [tajweed.md](tajweed.md) for the architecture and [calibration.md](calibration.md) for the training pipeline.
- **Memorize + Tajweed:** When the user flips the **Tajweed** switch on the Memorize tab, the mic is opened by both the SpeechRecognizer (for word-tracking) AND a fallback `AudioRecord` (for reliable acoustic analysis on devices where the recognizer's bundled audio can't be decoded, e.g. Samsung A05s). Tajweed mistakes trigger a non-pausing, near-real-time popup that names the rule, so the user keeps reading. See section "Tajweed mistake popup" below.
- **Memorize + Tarteel:** Default mode. Word-tracking only, no acoustic analysis needed — faster, less battery.
- **Read tab:** Surah text is rendered with tajweed-aware letter colouring (Qalqalah blue, Madd red, Al-Lam solar blue-grey / lunar teal, Ikhfa/Iqlab cyan, Idgham orange, Ghunnah green, Meem Sakinah magenta, Izhar purple). Audio for each ayah streams from a CDN, looping until the user stops it.
- **Goals & Stats tab:** Daily / Weekly recitation goals scoped to (surah, start-ayah, end-ayah) — always a specific range, with Al-Fatihah selected by default. Progress reacts to the surah+range the user just recited, so a goal only advances when the recitation falls inside its scope. Per-rule tajweed-mistake breakdown, total mistakes, wrong / skipped word counts, and a celebrated "Goal Accomplished" notification on completion.
- **Tajweed Rules Coach:** An interactive tab with one card per rule (8 cards). Each card shows the rule's name, a one-line description, a full explanation, a randomly-picked Quranic example out of 12 curated excerpts (re-picked per click of the Refresh button), coloured in the same letter-by-letter per-rule palette as the Read tab. Expanding the card and tapping the mic starts a recorder session scored against the example, with a per-rule verdict treated green (applied) or red (needs practice) based on `decisionThresholdForHead`.
- **Riwayah support:** The user picks Hafs or Qalun on Memorize / Read. Hafs text is rendered from the remote API; Qalun text comes from the bundled `assets/qalun_nemo_manifest.jsonl` with a clear UI fallback notice if Qalun data isn't available for the surah. The riwayah switch lives in the Memorize/Read top bar only.
- **Two languages:** A single toggle in the top bar switches the entire app between English and Arabic (RTL layout mutations handled automatically by Compose). Every visible string — tabs, buttons, dialogs, alerts, rule names, descriptions, dialogs, goal form, stats panels, error messages — has an Arabic counterpart.
- **No-internet safety:** Opening the app offline no longer surfaces a raw exception string. The three network failure sites (surah text fetch, audio play on-error, audio prepare catch) test `ConnectivityManager` state and, when truly offline, present a friendly localized popup ("No Internet Connection — you can't listen to the Quran without it") instead of technical error text.
- **Tajweed model warm-up:** The first time the user opens Tajweed tab (or flips Memorize to Tajweed mode), `preloadTajweedModel()` parses `assets/model_weights.json` on `Dispatchers.IO` and surfaces a non-dismissable "Loading Tajweed Model…" dialog that auto-dismisses the instant the singleton is warm. Subsequent entries are instant.

## Architecture
- **UI:** Jetpack Compose (Material Design 3), single-Activity + CompositionLocals.
- **Speech Recognition:** Android `SpeechRecognizer` with partial results enabled. Audio bundle is requested via `android.speech.extra.GET_AUDIO` extras, but the app also maintains its own 16 kHz mono `AudioRecord` (`MicRecorderController`) whenever Tajweed acoustic analysis is required — Samsung-family devices are notorious for returning empty `audio/AMR` bundles from the recognizer, so a parallel raw-PCM recorder is the reliable fallback.
- **Tajweed analysis:** Port of the trained PyTorch model into pure Kotlin (`TajweedNet.kt`). Conv1x3 → BiGRU(64) → per-rule Linear heads, mirroring the training-time forward pass exactly (parity-tested in `TajweedNetParityTest`).
- **Calibration:** Thresholds in `decisionThresholdForHead()` (`RecitationApp.kt`). Each rule has a per-head cut tuned against a held-out calibration fold — see [calibration.md](calibration.md).
- **Data Persistence:** Room Database for user goals (surah-scoped, version 3 migration wipes legacy goals on upgrade using `fallbackToDestructiveMigration`). SharedPreferences for recited-ayah counters and the UI language.
- **Networking:** Retrofit + Moshi against `https://api.alquran.cloud/v1` for surah text and audio URLs.

## Permissions
- `INTERNET` — surah text + audio CDN.
- `RECORD_AUDIO` — speech recognizer + `MicRecorderController` (the AudioRecord fallback).

## Getting Started

**Prerequisites:**  [Android Studio](https://developer.android.com/studio)

1. Open Android Studio
2. Select **Open** and choose this project directory
3. Let Gradle sync
4. Run on an emulator or physical device (mic + internet needed for full functionality)

## Documentation
- [How Hafs & Qalun Mapping Works](mapping.md)
- [How Tajweed Scoring Works](tajweed.md)
- [Tajweed Model Calibration & Training](calibration.md)
- [Training the model yourself (Python / PyTorch)](training/README.md)
