package itqan.app.sherpa
import itqan.app.ui.normalizeArabic
import org.junit.Assert.*
import org.junit.Test
class AllahNormCheck {
    @Test fun allahVariantsNormalizeDistinctly() {
        // ayah 1 الله (with alif) vs ayah 2 لله (lam-prefixed) must stay
        // DISTINCT after normalization, so the truncated "لل" uniquely maps to
        // لله and can never be mis-restored onto الله.
        val a1 = normalizeArabic("ٱللَّه")   // بِسْمِ ٱللَّهِ
        val a2 = normalizeArabic("لِلَّه")   // ٱلْحَمْدُ لِلَّهِ
        val trunc = normalizeArabic("لِلَّ")
        println("ٱللَّه -> '$a1' ; لِلَّه -> '$a2' ; لِلَّ -> '$trunc'")
        assertEquals("الله", a1)
        assertEquals("لله", a2)
        assertEquals("لل", trunc)
        assertTrue("trunc must prefix ayah-2 لله", a2.startsWith(trunc))
        assertFalse("trunc must NOT prefix ayah-1 الله", a1.startsWith(trunc))
    }
}
