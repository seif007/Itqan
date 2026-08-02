package com.example.data

import androidx.room.Entity
import androidx.room.PrimaryKey

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
    val rangeEnd: Int = 0
)

