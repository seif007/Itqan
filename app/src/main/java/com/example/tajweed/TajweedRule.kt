package com.example.tajweed

enum class TajweedRule {
    MADD,
    GHUNNAH,
    QALQALAH,
    IDGHAM,
    IKHFA,
    ALLAM,          // Al-Laam (Solar / Lunar)
    MEEM_SAKINAH,  // Meem Sakinah (Labial Rules)
    SILAH,          // Silah (Qalun vowel continuation)
    IZHAR,          // Izhar (Clarity)
    NONE;

    companion object {
        fun fromString(ruleString: String): TajweedRule {
            val lower = ruleString.lowercase()
            return when {
                lower.contains("allam") || lower.contains("al-lam") -> ALLAM
                lower.contains("meem") -> MEEM_SAKINAH
                lower.contains("silah") -> SILAH
                lower.contains("izhar") -> IZHAR
                lower.contains("madd") -> MADD
                lower.contains("ghunnah") -> GHUNNAH
                lower.contains("qalqalah") -> QALQALAH
                lower.contains("idgham") -> IDGHAM
                lower.contains("ikhfa") -> IKHFA
                else -> NONE
            }
        }
    }
}
