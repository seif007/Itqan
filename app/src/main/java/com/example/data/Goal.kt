package com.example.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Per-word mistake entry stored as a JSON array in [Goal.mistakeEntriesJson].
 *
 * Each entry is one of:
 *   {"kind":"wrong","word":"...","surah":1,"ayah":1,"wordIndex":3}
 *   {"kind":"skipped","word":"...","surah":1,"ayah":1,"wordIndex":3}
 *   {"kind":"tajweed","word":"...","rule":"Madd (Vowel Extension)","score":0.42,
 *    "surah":1,"ayah":1,"wordIndex":3}
 *
 * Stored as a string column (not a relational child table) so the goal row is
 * a single self-contained record; the per-goal mistake list is bounded in size
 * (a typical recitation has a handful of mistakes), so a JSON blob is plenty
 * and avoids the migration cost of a separate MistakeEntry table.
 */
data class MistakeEntry(
    val kind: String,         // "wrong" | "skipped" | "tajweed"
    val word: String,         // canonical target word
    val rule: String? = null, // only for kind="tajweed"; null otherwise
    val score: Float? = null, // only for kind="tajweed"; null otherwise
    val surah: Int,
    val ayah: Int,
    val wordIndex: Int,
)

@Entity(tableName = "goals")
data class Goal(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val type: String, // "Daily" or "Weekly"
    val targetAyahs: Int,
    val recitedAyahs: Int = 0,
    val mistakesCount: Int = 0,
    val wrongWordsCount: Int = 0,
    val skippedWordsCount: Int = 0,
    val tajweedMistakesCount: Int = 0,
    val tajweedMistakesBreakdown: String = "", // e.g. "Ghunnah: 1, Madd: 2"
    val createdTimestamp: Long = System.currentTimeMillis(),
    /**
     * Optional scope: which Surah + which Ayah range this goal applies to.
     *
     * When set, [com.example.ui.MainViewModel.triggerGoalRealProgress] only applies recited
     * progress to goals whose [surahNumber] AND ayah range match the recitation the user just
     * finished. When null (e.g. legacy goals created before this column existed, or any future
     * "whole-Quran" goal type), the goal accumulates progress for any recitation -- the original
     * behavior. Stored as Int with sentinel 0 meaning "not scoped" so a Kotlin Int column can
     * represent the option cleanly without a Room nullability migration headache.
     */
    val surahNumber: Int = 0,
    val rangeStart: Int = 0,
    val rangeEnd: Int = 0,
    /**
     * JSON-encoded list of [MistakeEntry] for this goal. Empty string when no
     * per-word breakdown has been recorded. Used by the Stats tab to show the
     * user WHICH specific words were wrong / skipped / had which tajweed
     * rule, in addition to the aggregate counts.
     */
    val mistakeEntriesJson: String = "",
)

