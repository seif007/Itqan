package itqan.app.ml

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Guards the riwayah-sensitive behaviour of [TajweedRules]:
 *
 *  - Qalun writes the alif-wasl as ALIF + harakah + U+06EC (it never uses U+0671),
 *    so the definite article must be recognised in that encoding, while a
 *    word-initial hamzat qat' written the same way minus the marker (إِلَيْكَ)
 *    must NOT be reported as the article.
 *  - Hafs labels must stay put — including the cross-word noon/meem label before a
 *    standalone pause mark, which is the trap if the diacritic set is widened too far.
 */
class TajweedRulesRiwayahTest {

    private fun rules(word: String, next: String? = null) =
        TajweedRules.detectRulesForWord(word, next)

    // اِ۬لْحَمْدُ  (Qalun: alif-wasl = ALIF + KASRA + U+06EC)
    private val qalunAlHamdu = "\u0627\u0650\u06EC\u0644\u0652\u062D\u064E\u0645\u0652\u062F\u064F"
    private val qalunLillahi = "\u0644\u0650\u0644\u0647\u0650"                            // لِلهِ
    private val qatIlaika = "\u0627\u0650\u0644\u064E\u064A\u0652\u0643\u064E"             // اِلَيْكَ (إ hamzat qat')
    private val hafsAlHamdu = "\u0671\u0644\u0652\u062D\u064E\u0645\u0652\u062F\u064F"     // ٱلْحَمْدُ (alif-wasl U+0671)
    private val min = "\u0645\u0650\u0646\u0652"                                            // مِنْ
    private val kuntum = "\u0643\u064F\u0646\u062A\u064F\u0645\u0652"                       // كُنتُمْ
    private val lakum = "\u0644\u064E\u0643\u064F\u0645\u0652"                              // لَكُمْ
    private val bihim = "\u0628\u0650\u0647\u0650\u0645\u0652"                              // بِهِمْ
    private val waqfMark = "\u06DA"                                                        // standalone pause sign

    @Test
    fun qalunArticleIsRecognised() {
        assertTrue(rules(qalunAlHamdu, qalunLillahi).contains("ALLAM"))
    }

    @Test
    fun qalunHamzatQatIsNotTheArticle() {
        assertFalse(rules(qatIlaika).contains("ALLAM"))
    }

    @Test
    fun hafsArticleStillRecognised() {
        assertTrue(rules(hafsAlHamdu).contains("ALLAM"))
    }

    @Test
    fun kafIsIkhfaAndLamIsIdghamPlain() {
        assertTrue(rules(min, kuntum).contains("IKHFA"))
        val lam = rules(min, lakum)
        assertTrue(lam.contains("IDGHAM"))
        assertFalse(lam.contains("IKHFA"))
    }

    @Test
    fun meemSakinahSurvivesBeforeStandalonePauseMark() {
        assertTrue(rules(bihim, waqfMark).contains("MEEM_SAKINAH"))
    }
}
