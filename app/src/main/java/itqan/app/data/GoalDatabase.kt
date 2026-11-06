package itqan.app.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

/**
 * v4 -> v5: an on-device database created by an earlier build is missing the
 * goal-scoping columns (surahNumber / rangeStart / rangeEnd / mistakeEntriesJson)
 * that were added to [Goal] WITHOUT a version bump. Room validated the identity
 * hash instead of migrating and threw
 * "Room cannot verify the data integrity ... Expected identity hash: <new>,
 * found: <old>" on first open, so every goal update failed silently — the session
 * summary still showed, but progress was never recorded.
 *
 * Phase 1 is ADDITIVE and idempotent: each column is added only when it is not
 * already present, so it is safe for databases that already have some or all of
 * them.
 *
 * Phase 2 — LEGACY COLUMN REBUILD — was added after measuring fatihah_c23:
 * adding the four columns was NOT enough, Room still threw
 *   "Migration didn't properly handle: goals(itqan.app.data.Goal)"
 * on every single open (twice per session: 01:15:13.179 and 01:15:13.703), so
 * goal progress was *still* never written. The real cause: the on-device table
 * predates the current entity and carries two columns Room no longer knows about
 * — `completedTimestamp` and `periodStart`. Room 2.7 validates the schema
 * EXACTLY, so extra columns fail just as hard as missing ones.
 *
 * A bare `ALTER TABLE ... DROP COLUMN` would fix it, but that needs SQLite
 * 3.35+ and we cannot assume it on the user's Android 13 device, so we do the
 * portable four-step rebuild instead (create / copy / drop / rename). Only the
 * columns the entity declares are recreated; every row is copied, so no goal
 * history is lost.
 */
private val MIGRATION_4_5 = object : Migration(4, 5) {
    /** Column -> DDL fragment, in [Goal] declaration order. */
    private val expected = linkedMapOf(
        "id" to "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL",
        "type" to "TEXT NOT NULL",
        "targetAyahs" to "INTEGER NOT NULL",
        "recitedAyahs" to "INTEGER NOT NULL",
        "mistakesCount" to "INTEGER NOT NULL",
        "wrongWordsCount" to "INTEGER NOT NULL",
        "skippedWordsCount" to "INTEGER NOT NULL",
        "tajweedMistakesCount" to "INTEGER NOT NULL",
        "tajweedMistakesBreakdown" to "TEXT NOT NULL",
        "createdTimestamp" to "INTEGER NOT NULL",
        "surahNumber" to "INTEGER NOT NULL",
        "rangeStart" to "INTEGER NOT NULL",
        "rangeEnd" to "INTEGER NOT NULL",
        "mistakeEntriesJson" to "TEXT NOT NULL",
    )

    override fun migrate(db: SupportSQLiteDatabase) {
        val existing = LinkedHashSet<String>()
        db.query("PRAGMA table_info(goals)").use { c ->
            val nameIdx = c.getColumnIndex("name")
            while (c.moveToNext()) existing.add(c.getString(nameIdx))
        }
        if (existing.isEmpty()) return // table absent: Room creates it fresh.

        // ── Phase 1: add whatever is missing (idempotent) ────────────────
        fun addIfMissing(column: String, definition: String) {
            if (column !in existing) {
                db.execSQL("ALTER TABLE goals ADD COLUMN $column $definition")
            }
        }
        addIfMissing("surahNumber", "INTEGER NOT NULL DEFAULT 0")
        addIfMissing("rangeStart", "INTEGER NOT NULL DEFAULT 0")
        addIfMissing("rangeEnd", "INTEGER NOT NULL DEFAULT 0")
        addIfMissing("mistakeEntriesJson", "TEXT NOT NULL DEFAULT ''")

        // ── Phase 2: drop whatever Room no longer declares ───────────────
        val extra = (existing + addedByPhase1).filter { it !in expected.keys }
        if (extra.isEmpty()) return

        android.util.Log.i(
            "ItqanGoals",
            "MIGRATION_4_5: rebuilding goals — dropping legacy column(s) $extra"
        )
        val cols = expected.keys.joinToString(",") { "`$it`" }
        db.execSQL("CREATE TABLE IF NOT EXISTS `goals_rebuilt` (${
            expected.entries.joinToString(",") { (n, ddl) -> "`$n` $ddl" }
        })")
        db.execSQL("INSERT INTO `goals_rebuilt` ($cols) SELECT $cols FROM `goals`")
        db.execSQL("DROP TABLE `goals`")
        db.execSQL("ALTER TABLE `goals_rebuilt` RENAME TO `goals`")
    }

    /** Columns Phase 1 may have added, so Phase 2 sees the post-phase-1 schema. */
    private val addedByPhase1 = listOf(
        "surahNumber", "rangeStart", "rangeEnd", "mistakeEntriesJson"
    )
}

@Database(entities = [Goal::class], version = 5, exportSchema = false)
abstract class GoalDatabase : RoomDatabase() {
    abstract fun goalDao(): GoalDao

    companion object {
        @Volatile
        private var INSTANCE: GoalDatabase? = null

        fun getDatabase(context: Context): GoalDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    GoalDatabase::class.java,
                    "goal_database"
                )
                .addMigrations(MIGRATION_4_5)
                .fallbackToDestructiveMigration()
                .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
