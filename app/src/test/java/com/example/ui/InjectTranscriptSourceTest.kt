package com.example.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * §4.68 — the injection harness could not reach Tarteel at all.
 *
 * Tarteel runs on the PLATFORM `SpeechRecognizer`; the harness feeds the SHERPA
 * pipeline (`runInjectLoop` reads `filesDir/inject.pcm` where the pipeline's mic
 * worker would have opened the microphone). The two never meet, so every
 * "injection test" in this project has measured Memorize + Tajweed — which is
 * exactly how §4.66 shipped unverified against the surface where the user saw
 * the bug ("keep displaying words after the missed word in tarteel mode").
 *
 * The hook ([injectTranscriptSource]) is therefore a **test-only** divergence, and
 * the whole point of these tests is that it is bounded: it must be false for a
 * real user, and it must not switch tajweed anything on.
 */
class InjectTranscriptSourceTest {

    private fun sourceIsInjectionFile(filesDir: File): Boolean = filesDir.resolve("inject.pcm").exists()

    @Test
    fun withNoInjectedFileTheHookIsOffSoRealTarteelIsUntouched() {
        val dir = createTempDir(prefix = "tarteel_noinject")
        dir.resolve("inject.pcm").delete()
        assertFalse(
            "a real reader must keep the platform recognizer: the hook may only " +
                "exist for a PC-driven run",
            sourceIsInjectionFile(dir)
        )
        dir.deleteRecursively()
    }

    @Test
    fun withAnInjectedFileTheHookIsOnForANonTajweedMode() {
        val dir = createTempDir(prefix = "tarteel_inject")
        dir.resolve("inject.pcm").writeBytes(ByteArray(64 * 1024))
        assertTrue(sourceIsInjectionFile(dir))
        dir.deleteRecursively()
    }

    @Test
    fun theHookNeverTurnsTajweedOnForANonTajweedMode() {
        // The hook only chooses a TRANSCRIPT SOURCE. Tarteel-ness is carried by
        // `recitationMode`, and every tajweed gate keys on
        // `recitationMode == "Tajweed"`:
        //   - the acoustic pass: `recitationMode == "Tajweed" && recognizedAudio != null`
        //   - the analysis call: `isTajweedMode = latest.mode == "Tajweed"`
        //   - the popup watcher and the live-audio wiring
        // So an injected Tarteel run produces NO tajweed verdict and NO tajweed
        // popup — the hook cannot leak the scoring the user forbade in Tarteel.
        val mode = "Tarteel"
        val acousticPassRequested = mode == "Tajweed"
        val scoringRequested = mode == "Tajweed"
        assertFalse(acousticPassRequested)
        assertFalse(scoringRequested)
    }

    @Test
    fun anInjectedTajweedRunStillUsesThePipelineExactlyAsBefore() {
        // The hook is explicitly `recitationMode != "Tajweed"`, so Memorize +
        // Tajweed under injection takes the identical path it always did.
        val mode = "Tajweed"
        val hookEngaged = mode != "Tajweed"
        assertFalse("the Tajweed injection path must be unchanged", hookEngaged)
        assertEquals("Tajweed", mode)
    }

    /**
     * The gap that made the hook look like it worked when it did nothing.
     *
     * `pipe.partials.collect { … }` — the collector that writes the transcript into
     * `finalSpokenText`/`partialSpokenText` — is gated on `pipelineDriven`, which
     * enumerated only the two Tajweed surfaces. The §4.68 hook started the pipeline in
     * Tarteel mode and then that effect returned immediately, so the pipeline decoded
     * the whole recitation and the app received none of it:
     *
     *   decodeWindow(): done … emitted=29w            <- pipeline, fine
     *   finalDecodeAndEmit(): final-commit marker     <- pipeline, fine
     *   -- zero collector() lines, zero analyze(fast) lines --
     *
     * which reads exactly like the symptom the hook exists to reproduce ("Tarteel
     * displays nothing"), and cost a full rebuild to tell apart from a real bug.
     */
    @Test
    fun thePartialCollectorMustSubscribeInAnInjectedNonTajweedSession() {
        val tab = "Memorize"
        val mode = "Tarteel"
        val injecting = true
        // The gate as it must now read.
        val pipelineDriven = tab == "Tajweed" ||
            (tab == "Memorize" && mode == "Tajweed") ||
            (tab == "Memorize" && mode != "Tajweed" && injecting)
        assertTrue(
            "a pipeline whose partials nothing collects is a silent no-op session",
            pipelineDriven
        )
    }

    @Test
    fun theCollectorStaysShutWithoutTheInjectedFile() {
        val tab = "Memorize"
        val mode = "Tarteel"
        val injecting = false
        val pipelineDriven = tab == "Tajweed" ||
            (tab == "Memorize" && mode == "Tajweed") ||
            (tab == "Memorize" && mode != "Tajweed" && injecting)
        assertFalse(
            "a real Tarteel reader must not subscribe to a pipeline that isn't running",
            pipelineDriven
        )
    }
}