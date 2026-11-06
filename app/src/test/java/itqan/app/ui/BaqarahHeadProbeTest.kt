package itqan.app.ui

import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class BaqarahHeadProbeTest {
    @Test fun probe() {
        val cache = java.io.File(System.getProperty("java.io.tmpdir"), "stt_e2e")
        val f = java.io.File(cache, "surah_2.json")
        val jsonText = if (f.exists()) f.readText() else {
            val conn = java.net.URL("https://api.alquran.cloud/v1/surah/2/ar.alafasy").openConnection() as java.net.HttpURLConnection
            conn.requestMethod = "GET"; conn.connectTimeout = 20000; conn.readTimeout = 60000
            val t = conn.inputStream.use { it.readBytes().decodeToString() }
            f.writeText(t); t
        }
        val ayahs = org.json.JSONObject(jsonText).getJSONObject("data").getJSONArray("ayahs")
        val words = ArrayList<String>()
        for (i in 0 until ayahs.length()) {
            words.addAll(ayahs.getJSONObject(i).getString("text").split(Regex("\\s+")).filter { it.isNotBlank() })
            if (words.size > 36) break
        }
        for (idx in 24 until minOf(36, words.size)) {
            val w = words[idx]
            val next = words.getOrNull(idx + 1)
            val r = getTajweedRuleForWord(w, next, "Hafs")
            val ann = buildTajweedAnnotatedString(w, next, "Hafs")
            val spans = ann.spanStyles.joinToString(" ") { rr ->
                rr.start.toString() + "-" + rr.end.toString() + "=#" + rr.item.color.value.toString(16)
            }
            println("BPROBE[$idx] cps=${w.map { Integer.toString(it.code, 16) }} rule='${r ?: "null"}' hasExplicit=${wordHasExplicitMaddMark(w)} spans=${spans.ifEmpty { "NONE" }}")
        }
    }
}
