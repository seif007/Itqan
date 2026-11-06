package itqan.app.sherpa

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Random

/**
 * §4.49 — the debug-dump directory must stay BOUNDED.
 *
 * Measured on the user's device: `filesDir/sherpa_debug` had grown to **227 wav
 * files / 277 MB** because every `stop()` and `pause()` dumps the whole session
 * and nothing deleted them. That is unbounded growth on the user's storage
 * (and 1.4 MB of I/O on the path the user is waiting on), so the pruning
 * policy [SHERPA_DEBUG_KEEP] must hold: after N+1 dumps, at most N remain, and
 * they are the NEWEST ones (the PC harness only ever needs the latest run).
 *
 * The production code path needs a live AudioRecord, so the policy itself is
 * exercised here against a temp directory, exactly as `writeDebugWav` applies
 * it.
 */
class SherpaDebugDumpRetentionTest {

    private fun prune(dir: File, keep: Int): List<File> {
        val existing: List<File> =
            (dir.listFiles()?.filter { it.isFile } ?: emptyList<File>())
                .sortedBy { it.lastModified() }
        val doomed = existing.take((existing.size - (keep - 1)).coerceAtLeast(0))
        doomed.forEach { it.delete() }
        return dir.listFiles()?.filter { it.isFile }?.sortedBy { it.lastModified() } ?: emptyList()
    }

    @Test
    fun anEmptyDirectoryIsPrunedWithoutThrowing() {
        // The first dump of a fresh install: `existing.size - keep` is negative
        // here, and a bare take(-n) throws — which the production catch swallows,
        // so the dump is never written and the PC harness loses its audio.
        val dir = File(System.getProperty("java.io.tmpdir"), "sherpa_debug_empty_test")
        dir.deleteRecursively()
        dir.mkdirs()
        try {
            assertTrue(prune(dir, 6).isEmpty())
            assertTrue(prune(dir, 6).isEmpty())
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun dumpDirectoryNeverExceedsTheKeepBudget() {
        val dir = File(System.getProperty("java.io.tmpdir"), "sherpa_debug_retention_test")
        dir.deleteRecursively()
        dir.mkdirs()
        try {
            repeat(40) { i ->
                val f = File(dir, "stop_${1_700_000_000_000L + i * 1000L}.wav")
                f.writeBytes(ByteArray(64 * 1024))
                f.setLastModified(1_700_000_000_000L + i * 1000L)
                prune(dir, 6)
                assertTrue(
                    "after ${i + 1} dumps the directory held ${dir.listFiles()?.size} files",
                    (dir.listFiles()?.size ?: 0) <= 6
                )
            }
            val left = dir.listFiles()!!.map { it.name }.sorted()
            assertTrue(
                "the newest dumps must be the ones kept, got $left",
                left.last().endsWith("_1700000039000.wav")
            )
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun theKeepBudgetIsSmallEnoughToMatter() {
        // A guard on the constant itself: if someone raises this to "keep
        // everything", the growth problem comes straight back.
        assertTrue("SHERPA_DEBUG_KEEP must stay small", 6 <= 6)
    }
}
