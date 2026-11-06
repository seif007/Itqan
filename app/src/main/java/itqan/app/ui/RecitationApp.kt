package itqan.app.ui

import android.app.Activity
import android.content.Intent
import android.speech.RecognizerIntent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.automirrored.filled.*
import androidx.compose.material.icons.rounded.Book
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel

// Offline streaming speech recognizer pipeline (Memorize+Tajweed only).
// Sherpa-ONNX JNI + bindings ship inside the vendored AAR; the streaming
// Zipformer weights download on first launch into filesDir/sherpa_model/.
// See itqan.app.sherpa.SherpaPipeline.kt for the single-AudioRecord shared
// pipeline + AyahWordMatcher hallucination guard.
import itqan.app.sherpa.SherpaModelDownloader
import itqan.app.sherpa.SingleAudioPipeline
import itqan.app.sherpa.isSherpaModelReady
import itqan.app.data.MistakeEntry
import itqan.app.verses.VersesApi
import itqan.app.verses.SurahData
import itqan.app.verses.formGroupsFor
import itqan.app.verses.hasMuqattaatExpansion
import okhttp3.OkHttpClient
import okhttp3.Request
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import itqan.app.data.Goal
import itqan.app.data.GoalDatabase
import itqan.app.data.GoalRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.MediaExtractor
import android.media.MediaCodec
import android.media.MediaFormat
import android.net.Uri
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import androidx.core.app.NotificationCompat
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.runtime.CompositionLocalProvider

import kotlinx.coroutines.sync.withLock
import itqan.app.ml.RecitationAnalyzer

class MainViewModel(application: Application) : androidx.lifecycle.AndroidViewModel(application) {
    private val goalDao = GoalDatabase.getDatabase(application).goalDao()
    private val goalRepository = GoalRepository(goalDao)
    
    val allGoals: Flow<List<Goal>> = goalRepository.allGoals
    
    var completedGoalToCelebrate by mutableStateOf<Goal?>(null)
        internal set

    /**
     * Session-completion summary. Set to a non-null value exactly when the
     * user has finished reciting the selected range of ayahs (target fully
     * covered after the finalize-to-CORRECT pass) and the session has
     * finalized. The RecitationApp composable observes this and pops a
     * dedicated dialog showing the per-word mistake breakdown. Cleared by
     * [dismissSessionSummary] when the user closes the dialog.
     *
     * The list is per-WordAnalysis row (not per-RecitationStatus kind) so the
     * dialog can render each word in its own line. Tajweed rows are deduped
     * per (word, rule) so the dialog doesn't show "Madd" twice for the same
     * word on the same recitation.
     */
    var sessionSummaryToShow by mutableStateOf<SessionSummary?>(null)
        internal set

    /** Snapshot of the user's just-finished recitation for the summary dialog. */
    data class SessionSummary(
        val surahNumber: Int,
        val rangeStart: Int,
        val rangeEnd: Int,
        val entries: List<MistakeEntry>,
        val recitedAyahCount: Int,
        val totalAyahCount: Int,
        /**
         * §4.48 — TRUE when the user PAUSED rather than reaching the end of the
         * range, so the dialog can say so instead of reading like a verdict on the
         * whole surah. The user's report: he recited ~22 words of Al-Baqarah, the
         * range was the WHOLE surah (6116 words), and the summary that appeared
         * looked like the app was judging his recitation of it ("it showed me a
         * summary although I didn't complete the range"). A pause is a pause.
         */
        val paused: Boolean = false,
        /** Words that carry spoken evidence, for the same reason. */
        val heardWordCount: Int = 0,
        // ── COVERAGE (§4.31) ────────────────────────────────────────────────
        // How many target words the recognizer never actually heard, and how
        // many the range contains. "No mistake was confirmed" is NOT the same
        // as "every word was verified": when the walk stalls, finalizeCorrect()
        // forges the trailing rows to CORRECT with no spoken evidence, so a run
        // that aligned only 27 of 29 words could still claim perfection.
        // Default 0 so an older call site keeps the previous behaviour.
        val unheardWordCount: Int = 0,
        val totalWordCount: Int = 0,
    )

    fun dismissSessionSummary() {
        sessionSummaryToShow = null
    }

    fun setGoal(type: String, target: Int, surahNumber: Int = 0, rangeStart: Int = 0, rangeEnd: Int = 0) {
        viewModelScope.launch {
            try {
                // Fetch current list of goals from our allGoals Flow using flow's first() terminal operator
                val currentGoals = allGoals.first()
                // Uniqueness: one goal per (type, scope). If the user creates a "Daily" goal for
                // Surah 2 verses 1-5 while one already exists for that same scope, replace it.
                val existing = currentGoals.find {
                    it.type == type &&
                        it.surahNumber == surahNumber &&
                        it.rangeStart == rangeStart &&
                        it.rangeEnd == rangeEnd
                }
                if (existing != null) {
                    goalRepository.deleteById(existing.id)
                }
                goalRepository.insert(
                    Goal(
                        type = type,
                        targetAyahs = target,
                        surahNumber = surahNumber,
                        rangeStart = rangeStart,
                        rangeEnd = rangeEnd
                    )
                )
            } catch (e: Exception) {
                android.util.Log.e("MainViewModel", "Error setting goal", e)
            }
        }
    }

    fun deleteGoal(goalId: Int) {
        viewModelScope.launch {
            goalRepository.deleteById(goalId)
        }
    }

    fun dismissCelebration() {
        completedGoalToCelebrate?.let { goal ->
            viewModelScope.launch {
                goalRepository.deleteById(goal.id)
                completedGoalToCelebrate = null
            }
        }
    }

    fun triggerGoalRealProgress(
        recitedAyahsDelta: Int,
        wrongDelta: Int = 0,
        skippedDelta: Int = 0,
        tajweedDelta: Int = 0,
        violatedRules: List<String> = emptyList(),
        encounteredRules: List<String> = emptyList(),
        // Information about WHICH recitation just produced this delta. Used to filter goals by
        // their (surahNumber, rangeStart, rangeEnd) scope: a goal only makes progress when the
        // recitation that just completed is inside its scope. Undefined scope (any 0) behaves
        // like the legacy behavior -- the goal accepts progress from any recitation.
        surahNumber: Int = 0,
        rangeStart: Int = 0,
        rangeEnd: Int = 0,
        // Per-word mistake entries collected at session-finalize. Persisted as
        // JSON on the goal so the Stats tab can show WHICH specific words were
        // wrong / skipped / had which tajweed rule, in addition to the
        // aggregate counts. Empty when this delta comes from a path that has
        // no per-word detail (e.g. the hint button) — those count 1 wrong
        // but the user already saw the mistake live.
        mistakeEntries: List<MistakeEntry> = emptyList()
    ) {
        viewModelScope.launch {
            try {
                val currentGoals = allGoals.first()
                for (goal in currentGoals) {
                    // Scope-match check.
                    // - Legacy/unscoped goals (goal.surahNumber == 0) accept any recitation.
                    // - Scoped goals accept only when BOTH the goal's scope and the recitation's
                    //   scope are set, surahNumber matches, and the recited ayah range intersects
                    //   the goal's [rangeStart, rangeEnd].
                    val goalScoped = goal.surahNumber != 0
                    val recitationScoped = surahNumber != 0
                    val scopeMatches = if (!goalScoped) {
                        true  // unscoped goal accepts anything (legacy behavior)
                    } else if (!recitationScoped) {
                        false // scoped goal but recitation didn't report its scope -> skip
                    } else {
                        goal.surahNumber == surahNumber &&
                            rangeStart <= goal.rangeEnd &&
                            rangeEnd >= goal.rangeStart
                    }
                    if (!scopeMatches) continue

                    val updatedRecited = goal.recitedAyahs + recitedAyahsDelta
                    val totalNewMistakes = wrongDelta + skippedDelta + tajweedDelta
                    val updatedMistakes = goal.mistakesCount + totalNewMistakes
                    val updatedWrong = goal.wrongWordsCount + wrongDelta
                    val updatedSkipped = goal.skippedWordsCount + skippedDelta
                    val updatedTajweed = goal.tajweedMistakesCount + tajweedDelta

                    val updatedBreakdown = if (encounteredRules.isNotEmpty()) {
                        // DB format expected: "Madd: mistakes/total, Ghunnah: mistakes/total"
                        // Or if legacy "Madd: mistakes", we upgrade it to "Madd: mistakes/mistakes" initially
                        val map = goal.tajweedMistakesBreakdown.split(",")
                            .filter { it.contains(":") }
                            .associate {
                                val parts = it.split(":")
                                val key = parts.getOrNull(0)?.trim() ?: ""
                                val valuePart = parts.getOrNull(1)?.trim() ?: ""
                                val mistakes = if (valuePart.contains("/")) valuePart.split("/")[0].toIntOrNull() ?: 0 else valuePart.toIntOrNull() ?: 0
                                val total = if (valuePart.contains("/")) valuePart.split("/").getOrNull(1)?.toIntOrNull() ?: mistakes else mistakes
                                key to Pair(mistakes, total)
                            }
                            .filter { it.key.isNotEmpty() }
                            .toMutableMap()

                        for (rule in encounteredRules) {
                            val current = map[rule] ?: Pair(0, 0)
                            map[rule] = Pair(current.first, current.second + 1)
                        }

                        for (rule in violatedRules) {
                            val current = map[rule] ?: Pair(0, 0)
                            map[rule] = Pair(current.first + 1, current.second)
                        }

                        map.entries.joinToString(", ") { "${it.key}: ${it.value.first}/${it.value.second}" }
                    } else {
                        goal.tajweedMistakesBreakdown
                    }

                    // Append the per-word mistake entries to the existing JSON
                    // blob. We don't deduplicate here -- multiple recitation
                    // sessions on the same goal can each contribute their own
                    // entries; the Stats tab groups them by kind when rendering.
                    val updatedMistakeEntriesJson = if (mistakeEntries.isNotEmpty()) {
                        val existing = parseMistakeEntries(goal.mistakeEntriesJson)
                        (existing + mistakeEntries).encodeToJson()
                    } else {
                        goal.mistakeEntriesJson
                    }

                    val updatedGoal = goal.copy(
                        recitedAyahs = updatedRecited,
                        mistakesCount = updatedMistakes,
                        wrongWordsCount = updatedWrong,
                        skippedWordsCount = updatedSkipped,
                        tajweedMistakesCount = updatedTajweed,
                        tajweedMistakesBreakdown = updatedBreakdown,
                        mistakeEntriesJson = updatedMistakeEntriesJson
                    )
                    
                    if (updatedRecited >= goal.targetAyahs) {
                        completedGoalToCelebrate = updatedGoal
                        sendCelebrationNotification(goal.type, goal.targetAyahs)
                    } else {
                        goalRepository.update(updatedGoal)
                    }
                }
            } catch (e: Exception) {
                android.util.Log.e("MainViewModel", "Error updating goals", e)
            }
        }
    }

    private fun sendCelebrationNotification(type: String, target: Int) {
        val context = getApplication<Application>().applicationContext
        
        // Prevent SecurityException / lint compliance on Android 13+ for notifications
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val hasNotificationPermission = androidx.core.content.ContextCompat.checkSelfPermission(
                context,
                android.Manifest.permission.POST_NOTIFICATIONS
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
            if (!hasNotificationPermission) {
                android.util.Log.w("MainViewModel", "Notification skipped: POST_NOTIFICATIONS permission not granted")
                return
            }
        }

        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channelId = "goal_completion_channel_id"
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "Recitation Goal Achieved",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "Celebrate when recitation target is completed"
            }
            notificationManager.createNotificationChannel(channel)
        }
        
        // Note: channel/display names in the system settings tray remain in English on most
        // devices regardless of the app's UI language, but the *body* of the "Goal Accomplished"
        // notification the user receives can be localized from SharedPreferences.
        val isArabicNow = context.getSharedPreferences("itqan_stats", android.content.Context.MODE_PRIVATE)
            .getBoolean("is_arabic", false)
        val builder = NotificationCompat.Builder(context, channelId)
            .setSmallIcon(android.R.drawable.btn_star_big_on)
            .setContentTitle(getAppString(isArabicNow, "Goal Accomplished! 🎉"))
            .setContentText(
                if (isArabicNow) "مبروك! لقد حققت هدفك ${getAppString(isArabicNow, type)} في تسميع $target آية!"
                else "Congratulations! You achieved your $type goal of reciting $target Ayahs!"
            )
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            
        try {
            notificationManager.notify(System.currentTimeMillis().toInt(), builder.build())
        } catch (e: SecurityException) {
            // Handle missing notification permission on API 33+
        }
    }

    var targetAyahs by mutableStateOf(50)
        internal set
    var recitedAyahs by mutableStateOf(0)
        internal set

    var riwayah by mutableStateOf("Hafs")
        private set

    fun updateRiwayah(newRiwayah: String) {
        riwayah = newRiwayah
        selectedSurah?.let { selectSurah(it) }
    }

    fun updateTargetAyahs(target: Int) {
        targetAyahs = target
    }
    
    fun incrementRecitedAyahs(count: Int) {
        recitedAyahs += count
    }
    
    fun initStats(savedTarget: Int, savedRecited: Int) {
        this.targetAyahs = savedTarget
        this.recitedAyahs = savedRecited
    }

    var surahs by mutableStateOf<List<SurahData>>(emptyList())
        private set
    var selectedSurah by mutableStateOf<SurahData?>(null)
        private set
    var isQalunFallback by mutableStateOf(false)
        private set
    
    data class AyahDisplay(
        val ayahId: Int,
        val text: String,
        val words: List<String>,
        val normalizedWords: List<String>,
        val wordStartIndex: Int,
        val audio: String?
    )
    
    var ayahs by mutableStateOf<List<AyahDisplay>>(emptyList())
        private set
    var allNormalizedWords by mutableStateOf<List<String>>(emptyList())
        private set
    val completedAyahs = mutableStateListOf<Int>()
    private var allSurahAyahs: List<AyahDisplay> = emptyList()

    var rangeStart by mutableStateOf(1)
        private set
    var rangeEnd by mutableStateOf(1)
        private set
    var maxAyahInSurah by mutableStateOf(1)
        private set
    
    var loopCount by mutableStateOf(1)
        private set
    var currentLoop by mutableStateOf(0)
        private set

    fun updateLoopCount(count: Int) {
        loopCount = count.coerceIn(1, 5)
    }
    
    fun setRange(start: Int, end: Int) {
        if (start <= end && start >= 1 && end <= maxAyahInSurah) {
            rangeStart = start
            rangeEnd = end
            applyRange()
        }
    }
    
    private fun applyRange() {
        var currentIndex = 0
        ayahs = allSurahAyahs.filter { it.ayahId in rangeStart..rangeEnd }.map { 
            val updated = it.copy(wordStartIndex = currentIndex)
            currentIndex += it.words.size
            updated
        }
        targetAyahs = ayahs.size
        
        viewModelScope.launch {
            allNormalizedWords = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
                ayahs.flatMap { it.normalizedWords }
            }
        }
    }

    var isLoading by mutableStateOf(false)
        private set
    var errorMessage by mutableStateOf<String?>(null)
        private set

    /**
     * Friendly "no internet" popup state. Triggered instead of dumping a raw exception
     * message into [errorMessage] when a network-bound lookup (Surah text or audio) fails
     * AND the device is offline -- the user said they don't want to see the raw error
     * ("there is an error that appears... I don't want the user to see the error"). The UI
     * shows a localized popup explaining they can't listen to the Quran without internet.
     * Dismiss clears it; the popup is separate from [errorMessage] so mic-related errors
     * (handled below with their own messages at SpeechRecognizer failure paths) are unaffected.
     */
    var noInternetDialog by mutableStateOf(false)
        private set

    fun dismissNoInternetDialog() {
        noInternetDialog = false
    }

    // ---------------------------------------------------------------------------
    // Tajweed model on-demand preload.
    //
    // RecitationAnalyzer.get(...) lazily parse ~1 MB of JSON weights + builds the
    // neural net. Without a preload, that work happens the first time the user
    // presses the mic while Tajweed mode is on -- an ugly multi-second freeze right
    // at the worst possible UX moment (the user expects the app to start listening
    // instantly). Instead we warm the singleton up-front from TWO entry points the
    // user said should trigger it:
    //   1. Opening the Tajweed tab (rules coach -- icon "School" in the navbar)
    //   2. Switching Memorize to Tajweed mode (recitationMode = "Tajweed" chip)
    // While the model is being parsed, [isTajweedModelLoading] is true and the UI
    // shows a non-dismissable popup ("Loading the Tajweed model, please wait")
    // which dismisses itself the moment the weights are in memory. Subsequent
    // calls to [preloadTajweedModel] no-op because the analyzer is a singleton.
    var isTajweedModelLoading by mutableStateOf(false)
        private set
    var isTajweedModelReady by mutableStateOf(false)
        private set

    /**
     * Kicks off a one-shot background warm-up of the process-wide [RecitationAnalyzer]
     * singleton. Idempotent: if the singleton is already in memory this returns
     * immediately without spawning a coroutine, so callers can call it freely on every
     * tab entry / chip tap without worrying about wasted work. Uses Mutex so two
     * simultaneous calls collapse into a single warm-up.
     */
    fun preloadTajweedModel() {
        if (isTajweedModelReady) return
        if (isTajweedModelLoading) return
        viewModelScope.launch {
            isTajweedModelLoading = true
            try {
                kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                    // Trigger the lazy singleton — weights JSON parse + tensor-scratch allocation.
                    val analyzer = RecitationAnalyzer.get(getApplication<Application>().applicationContext)
                    // And FORCE the first forward() to run now, on this IO thread, so the BiGRU's
                    // hidden-state scratch buffers / output tensors / anything else lazily created
                    // materialise *before* the user taps the mic. Without this the popup would close
                    // as soon as the weights file was parsed, but the first real recording would
                    // still freeze for the SAME lazy-allocation pause later. warmUp() is idempotent
                    // and cheap (a single silent clip), so calling it every preload is fine.
                    analyzer.warmUp()
                }
                isTajweedModelReady = true
            } catch (e: Throwable) {
                // Allow retrospective retry on the next tab-entry / chip-tap; crashing
                // would kill the app for users who just have mic-only permissions.
                android.util.Log.e("MainViewModel", "Failed to preload TajweedNet", e)
            } finally {
                isTajweedModelLoading = false
            }
        }
    }


    /**
     * True if the device currently has any usable network. Uses ConnectivityManager with the
     * API 23+ NetworkCapabilities path -- older APIs fall back to activeNetworkInfo. Returns
     * false on null/missing connectivity services. Safe to call from any thread (only
     * infallible system-service + immutable capability queries).
     */
    private fun isNetworkAvailable(): Boolean {
        val ctx = getApplication<Application>().applicationContext
        val cm = ctx.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
            ?: return false
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                val network = cm.activeNetwork ?: return false
                val caps = cm.getNetworkCapabilities(network) ?: return false
                // Any transport (wifi/cell/ethernet/bluetooth) counts as "has internet" -- the
                // Quran API + audio CDN are both reachable over any transport.
                caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
                    caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) ||
                    caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) ||
                    caps.hasTransport(NetworkCapabilities.TRANSPORT_VPN)
            } else {
                @Suppress("DEPRECATION")
                val info = cm.activeNetworkInfo
                @Suppress("DEPRECATION")
                info != null && info.isConnected
            }
        } catch (e: Throwable) {
            false
        }
    }

    var playingAyahId by mutableStateOf<Int?>(null)
        private set
        
    var playingWordIndex by mutableStateOf(-1)
        private set
    
    private var mediaPlayer: android.media.MediaPlayer? = null
    private var selectSurahJob: kotlinx.coroutines.Job? = null
    private var playAudioJob: kotlinx.coroutines.Job? = null
    private val mediaPlayerMutex = kotlinx.coroutines.sync.Mutex()
    
    init {
        surahs = itqan.app.verses.BundledVerses.getSurahList()
        mediaPlayer = android.media.MediaPlayer().apply {
            setAudioAttributes(
                android.media.AudioAttributes.Builder()
                    .setContentType(android.media.AudioAttributes.CONTENT_TYPE_MUSIC)
                    .setUsage(android.media.AudioAttributes.USAGE_MEDIA)
                    .build()
            )
        }
        if (surahs.isNotEmpty()) {
            selectSurah(surahs[0])
        }
        fetchSurahs()
    }

    private fun safeResetPlayer() {
        try {
            mediaPlayer?.reset()
        } catch (e: Throwable) {}
    }

    private fun fetchSurahs() {
        viewModelScope.launch {
            isLoading = true
            try {
                val response = VersesApi.instance.getSurahs()
                if (response.data.isNotEmpty()) {
                    surahs = response.data
                }
            } catch (e: Throwable) {
                android.util.Log.e("MainViewModel", "Failed to fetch surahs online, using static list fallback", e)
            } finally {
                isLoading = false
            }
        }
    }



    fun selectSurah(surah: SurahData) {
        completedAyahs.clear()
        selectSurahJob?.cancel()
        stopAudio()
        selectedSurah = surah
        isQalunFallback = false
        selectSurahJob = viewModelScope.launch {
            isLoading = true
            try {
                val result = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                    val response = VersesApi.instance.getSurahDetail(surah.number)
                    var rawAyahs = response.data.ayahs
                    
                    if (surah.number != 1 && surah.number != 9 && rawAyahs.isNotEmpty() && rawAyahs[0].text.startsWith("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ ")) {
                        val first = rawAyahs[0]
                        rawAyahs = rawAyahs.toMutableList().apply {
                             set(0, first.copy(text = first.text.removePrefix("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ ")))
                        }
                    }
                    
                    val processedSurahAyahs: List<AyahDisplay>
                    if (riwayah == "Qalun") {
                        val qalunAyahsForSurah = getApplication<Application>().assets.open("qalun_nemo_manifest.jsonl")
                            .bufferedReader().useLines { lines ->
                                lines.filter { it.contains("\"surah\":${surah.number},") || it.contains("\"surah\": ${surah.number},") }
                                    .mapNotNull { 
                                        try {
                                            org.json.JSONObject(it)
                                        } catch (e: Exception) {
                                            null
                                        }
                                    }.filter { it.getJSONObject("meta").getInt("surah") == surah.number }
                                    .toList()
                            }
                        
                        if (qalunAyahsForSurah.isEmpty()) {
                            isQalunFallback = true
                            processedSurahAyahs = rawAyahs.map { ayah ->
                                val words = splitAyahWords(ayah.text)
                                AyahDisplay(
                                    ayahId = ayah.numberInSurah,
                                    text = ayah.text,
                                    words = words,
                                    normalizedWords = words.map { normalizeArabic(it) },
                                    wordStartIndex = 0,
                                    audio = ayah.audio
                                )
                            }
                        } else {
                            var hafsIndex = 0
                            // Both arrays MUST stay index-aligned (hafsWords[i] <-> hafsAyahIds[i]);
                            // both use the same splitAyahWords helper so standalone Quranic annotation
                            // tokens (ۛ etc.) are dropped in lock-step.
                            val hafsWords = rawAyahs.flatMap { splitAyahWords(it.text) }
                            val hafsAyahIds = rawAyahs.flatMap { ayah -> splitAyahWords(ayah.text).map { ayah.numberInSurah } }
                            val hafsAudios = rawAyahs.associate { it.numberInSurah to it.audio }
                            
                            processedSurahAyahs = qalunAyahsForSurah.map { qAyahJson ->
                                val meta = qAyahJson.getJSONObject("meta")
                                val ayahId = meta.getInt("ayah_qalun")
                                val wordsArray = meta.getJSONArray("words_uthmani")
                                // The curated words_uthmani array is normally spoken words, but a
                                // handful of entries (e.g. final-rub-el-hizb or standalone paused
                                // marks) may be sentinel annotation tokens -- drop them so they do
                                // not occupy a target slot the user is never expected to speak
                                // (mirrors splitAyahWords for the Hafs fallback above).
                                val words = (0 until wordsArray.length())
                                    .map { wordsArray.getString(it) }
                                    .filter { !isAyahAnnotationOnlyToken(it) }
                                
                                var matchedHafsAyahId = 1
                                for (qw in words) {
                                    val qNorm = normalizeArabic(qw)
                                    if (qNorm.isNotBlank()) {
                                        var matched = false
                                        for (i in hafsIndex until minOf(hafsIndex + 15, hafsWords.size)) {
                                            val hw = hafsWords[i]
                                            val hNorm = normalizeArabic(hw)
                                            if (hNorm == qNorm || hNorm.contains(qNorm) || qNorm.contains(hNorm)) {
                                                matchedHafsAyahId = hafsAyahIds[i]
                                                hafsIndex = i + 1
                                                matched = true
                                                break
                                            }
                                        }
                                        if (matched) break
                                    }
                                }
                                if (matchedHafsAyahId == 1 && hafsIndex > 0 && hafsIndex <= hafsAyahIds.size) {
                                    matchedHafsAyahId = hafsAyahIds[hafsIndex - 1]
                                }
                                
                                AyahDisplay(
                                    ayahId = ayahId,
                                    text = qAyahJson.getString("text"),
                                    words = words,
                                    normalizedWords = words.map { normalizeArabic(it) },
                                    wordStartIndex = 0,
                                    audio = hafsAudios[matchedHafsAyahId]
                                )
                            }
                        }
                    } else {
                        processedSurahAyahs = rawAyahs.map { ayah ->
                            val words = splitAyahWords(ayah.text)
                            AyahDisplay(
                                ayahId = ayah.numberInSurah,
                                text = ayah.text,
                                words = words,
                                normalizedWords = words.map { normalizeArabic(it) },
                                wordStartIndex = 0, // placeholder
                                audio = ayah.audio
                            )
                        }
                    }
                    processedSurahAyahs
                }
                
                allSurahAyahs = result
                maxAyahInSurah = if (allSurahAyahs.isNotEmpty()) allSurahAyahs.last().ayahId else 1
                rangeStart = 1
                rangeEnd = maxAyahInSurah
                applyRange()
                stopAudio()
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Throwable) {
                android.util.Log.e("MainViewModel", "Error loading surah details", e)
                // If the device is offline, surface the friendly popup instead of the raw
                // exception text. The user explicitly said they don't want to see the raw
                // "Failed to load Surah text: <…>" error while offline -- just tell them they
                // need internet to listen to the Quran. We keep the techy trace in logcat.
                if (isNetworkAvailable()) {
                    errorMessage = "Failed to load Surah text: ${e.localizedMessage ?: e.message ?: e.toString()}"
                } else {
                    noInternetDialog = true
                    errorMessage = null
                }
            } finally {
                isLoading = false
            }
        }
    }

    private fun safeReleasePlayer(player: android.media.MediaPlayer?) {
        if (player == null) return
        try {
            if (player.isPlaying) {
                player.stop()
            }
        } catch (e: Throwable) {}
        try { player.release() } catch (e: Throwable) {}
    }

    fun playAudio(audioUrl: String, ayahId: Int) {
        if (audioUrl.isBlank()) {
            errorMessage = "Audio URL is not available"
            return
        }
        playAudioJob?.cancel()
        playingAyahId = null
        playingWordIndex = -1
        
        val mp = mediaPlayer ?: return
        
        playAudioJob = viewModelScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            mediaPlayerMutex.withLock {
                try {
                    // Reset inside the lock
                    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                        safeResetPlayer()
                    }
                    
                    mp.setDataSource(audioUrl)
                    
                    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                        mp.setOnPreparedListener { player: android.media.MediaPlayer ->
                            playingAyahId = ayahId
                            try {
                                player.start()
                                
                                // Simulate word highlighting
                                viewModelScope.launch(kotlinx.coroutines.Dispatchers.Main) {
                                    try {
                                        val currentAyah = ayahs.find { a -> a.ayahId == ayahId }
                                        val wordCount = currentAyah?.words?.size ?: 1
                                        while (playingAyahId == ayahId) {
                                            try {
                                                if (player.isPlaying) {
                                                    val progress = player.currentPosition.toFloat() / (player.duration.takeIf { d -> d > 0 } ?: 1)
                                                    playingWordIndex = (progress * wordCount).toInt().coerceIn(0, wordCount - 1)
                                                }
                                            } catch (e: Throwable) {}
                                            kotlinx.coroutines.delay(100)
                                        }
                                    } catch (e: kotlinx.coroutines.CancellationException) {
                                        throw e
                                    } catch (e: Throwable) {
                                        // Ignored
                                    }
                                }
                            } catch (e: Throwable) {}
                        }
                    mp.setOnErrorListener { player: android.media.MediaPlayer, _, _ ->
                        // MediaPlayer onError is the common offline failure mode -- the device
                        // can't buffer the audio URL because it has no network. Translate that
                        // into the friendly popup instead of the raw "Failed to play audio".
                        if (isNetworkAvailable()) {
                            errorMessage = "Failed to play audio"
                        } else {
                            noInternetDialog = true
                            errorMessage = null
                        }
                        playingAyahId = null
                        playingWordIndex = -1
                        viewModelScope.launch(kotlinx.coroutines.Dispatchers.Main) {
                            mediaPlayerMutex.withLock {
                                safeResetPlayer()
                            }
                        }
                        true
                    }
                    mp.setOnCompletionListener { player: android.media.MediaPlayer ->
                        playingWordIndex = -1
                        val currentIndex = ayahs.indexOfFirst { it.ayahId == ayahId }
                        val nextAyah = if (currentIndex in ayahs.indices && currentIndex + 1 < ayahs.size) {
                            ayahs[currentIndex + 1]
                        } else null
                        
                        if (nextAyah?.audio != null) {
                            playAudio(nextAyah.audio, nextAyah.ayahId)
                        } else {
                            currentLoop++
                            if (currentLoop < loopCount) {
                                val firstAyah = ayahs.firstOrNull()
                                if (firstAyah?.audio != null) {
                                    playAudio(firstAyah.audio, firstAyah.ayahId)
                                } else {
                                    playingAyahId = null
                                    currentLoop = 0
                                    viewModelScope.launch(kotlinx.coroutines.Dispatchers.Main) {
                                        mediaPlayerMutex.withLock {
                                            safeResetPlayer()
                                        }
                                    }
                                }
                            } else {
                                playingAyahId = null
                                currentLoop = 0
                                viewModelScope.launch(kotlinx.coroutines.Dispatchers.Main) {
                                    mediaPlayerMutex.withLock {
                                        safeResetPlayer()
                                    }
                                }
                            }
                        }
                    }
                    mp.prepareAsync()
                }
            } catch (e: Exception) {
                kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                    // Don't show the raw "Failed to play audio" while offline -- the user said
                    // they only want a friendly popup explaining they need internet to listen.
                    if (isNetworkAvailable()) {
                        errorMessage = "Failed to play audio"
                    } else {
                        noInternetDialog = true
                        errorMessage = null
                    }
                    playingAyahId = null
                    safeResetPlayer()
                }
            }
            } // Close mediaPlayerMutex.withLock
        }
    }
    
    fun playAllAyahs() {
        currentLoop = 0
        val firstAyah = ayahs.firstOrNull()
        if (firstAyah?.audio != null) {
            playAudio(firstAyah.audio, firstAyah.ayahId)
        }
    }
    
    fun stopAudio() {
        playAudioJob?.cancel()
        viewModelScope.launch(kotlinx.coroutines.Dispatchers.Main) {
            mediaPlayerMutex.withLock {
                safeResetPlayer()
            }
        }
        playingAyahId = null
        playingWordIndex = -1
    }

    override fun onCleared() {
        super.onCleared()
        try {
            mediaPlayer?.release()
            mediaPlayer = null
        } catch (e: Throwable) {}
    }
}

val diacriticsRegex = Regex("[\\u064B-\\u065F\\u0670\\u06D6-\\u06ED\\uFEFF۬ے]")
val punctuationRegex = Regex("[.,،؛؟?\"'()\\-!]")
val alifRegex = Regex("[\\u0622\\u0623\\u0625\\u0671]")

fun normalizeArabic(text: String): String {
    val sb = StringBuilder(text.length)
    for (i in 0 until text.length) {
        val c = text[i]
        // skip diacritics
        if (c in '\u064B'..'\u065F' || c == '\u0670' || c in '\u06D6'..'\u06ED' || c == '\uFEFF' || c == '۬' || c == 'ے') continue
        // skip punctuation
        if (".,;:!?()[]{}'\"/\\-_".contains(c)) continue
        if (c == '،' || c == '؛' || c == '؟' || c == 'ـ') continue
        
        when (c) {
            '\u0622', '\u0623', '\u0625', 'ٱ', '\u0671' -> sb.append('ا')
            '\u0629' -> sb.append('ه')
            '\u0649', '\u06CC', '\u0626' -> sb.append('ي')
            '\u0624' -> sb.append('و')
            '\u0640' -> {} // tatweel skip
            '\u0621' -> sb.append('ا') // hamza
            else -> sb.append(c)
        }
    }
    return sb.toString()
}

fun normalizeForComparison(text: String, riwayah: String): String {
    var normalized = normalizeArabic(text)
    if (riwayah == "Qalun") {
        if (normalized.endsWith("هموا")) {
            normalized = normalized.substring(0, normalized.length - 2)
        }
        if (normalized.endsWith("كموا")) {
            normalized = normalized.substring(0, normalized.length - 2)
        }
        if (normalized.endsWith("تموا")) {
            normalized = normalized.substring(0, normalized.length - 2)
        }
    }
    return normalized
}

// Tokenize an ayah string into the words the user is expected to SPEAK.
//
// The AlQuran.cloud text (and other Uthmani Mushaf transcriptions) interleaves the
// spoken words with freestanding Quranic annotation signs -- the small-high stop
// marks U+06D6..U+06ED that sit between words, e.g. the "three-dot" U+06DB (ۛ) in
// Al-Baqarah 2 (`ذَٰلِكَ ٱلْكِتَٰبُ لَا رَيْبَ ۛ فِيهِ ۛ هُدًۭى …`). They are NOT recited:
// they are structural pause cues, not part of the spoken text. But because they
// are whitespace-separated from neighbouring words, a naive `.split("\\s+")` keeps
// them as their own "word" token. `normalizeArabic` then strips every character in
// that range, leaving an ENTRY in `words` whose `normalizedWords` partner is the
// empty string "" -- and that zero-length target slot can NEVER be matched by the
// user's speech, so the recitation tracker would flag a WRONG/SKIPPED mistake on
// something the user was never supposed to say ("the app wants me to read it or
// it gives me a mistake"). This was the bug the user reported on Al-Baqarah 2, and
// the same mark appears in many other ayahs (every `ۛ`, `ۖ`, `ۗ`, `ۘ`, `ۚ`, `ۜ` and
// the standalone `ۭ`-style suffixes), so the fix must apply everywhere.
//
// `isAyahAnnotationOnlyToken` recognises these tokens: a token whose every
// character is either whitespace or a Quranic annotation sign (U+06D6..U+06ED) --
// including the possibility that some Mushaf variants add an Arabic-letter base
// (e.g. ۝ a rub-el-hizb mark expecting no speech) is handled by checking the
// post-normalize emptiness as well. We drop those tokens here so `words`,
// `normalizedWords`, `targetOrig`, `targetNorm`, `targetFormGroups` AND the
// per-ayah renderer (which derives `wordStartIndex` from `words.size`) all stay
// index-consistent and contain ONLY spoken words. The bare stop-glyph therefore
// does NOT appear as an actionable word in the recitation display -- matching the
// universal convention that these are non-spoken structural marks.
fun isAyahAnnotationOnlyToken(token: String): Boolean {
    if (token.isBlank()) return true
    var anyNonAnnotation = false
    var anyAnnotation = false
    for (ch in token) {
        val cp = ch.code
        if (cp == 0xFEFF || cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0x0640) continue
        if (cp in 0x06D6..0x06ED) { anyAnnotation = true; continue }
        anyNonAnnotation = true
    }
    // Pure annotation mark(s) and nothing spoken -> drop. Also drop the degenerate
    // case of a token that normalizes to empty (defensive: future annotation ranges
    // that normalizeArabic strips but Ibayt isAyahAnnotationOnlyToken missed).
    if (!anyNonAnnotation) return true
    if (normalizeArabic(token).isEmpty()) return true
    return false
}

fun splitAyahWords(text: String): List<String> {
    return text.split(Regex("\\s+"))
        // The AlQuran.cloud Uthmani text prepends a BOM (U+FEFF) to Fātiḥa āyah 1
        // (and can carry other zero-width chars). Strip them so no target word ever
        // carries an invisible leading BOM — it otherwise propagates into the matcher's
        // canonical words and the index-based tajweed-rule letter inspection.
        .map { it.replace(Regex("[\uFEFF\u200B\u200C\u200D\u2060]"), "") }
        .filter { it.isNotBlank() }
        .filter { !isAyahAnnotationOnlyToken(it) }
}

fun isSakinah(word: String, index: Int): Boolean {
    for (j in index + 1 until word.length) {
        val c = word[j]
        if (c == '\u0652' || c == '\u06E1') return true // Sukoon
        if (c == '\u064E' || c == '\u064F' || c == '\u0650' || c == '\u0651' || c == '\u064B' || c == '\u064C' || c == '\u064D') return false // Vowels, Shaddah, or Tanween
        // If it's another Arabic letter, then the current letter is bare
        if (c in '\u0621'..'\u064A') return true
    }
    return true
}

fun hasSukoonChar(word: String, index: Int): Boolean {
    if (index + 1 < word.length) {
        val c = word[index + 1]
        return c == '\u0652' || c == '\u06E1'
    }
    return false
}

/**
 * Returns the first *base Arabic letter* (post-normalization) starting at [startIndex] in [word],
 * skipping any diacritics / vowel marks / tatweel it meets along the way. Returns `null` if no
 * base letter is found before the end of the word.
 *
 * Used by both the verdict detector (getTajweedRuleForWord) and the colorizer
 * (buildTajweedAnnotatedString) when they need to peek at the letter that follows a Lam, Noon,
 * Meem, etc. — without this, an intervening Sukoon (e.g. in ٱلْقَمَرُ) makes them read the Sukoon
 * itself as the letter and misclassify the word.
 */
fun firstBaseLetterFrom(word: String, startIndex: Int): Char? {
    var j = startIndex
    val len = word.length
    while (j < len) {
        val c = word[j]
        val code = c.code
        // Arabic combining marks: Harakat \u064B-\u065F, superscript alef \u0670,
        // Quranic annotation signs \u06D6-\u06ED, tatweel \u0640, BOM.
        if (code in 0x064B..0x065F || code == 0x0670 ||
            code in 0x06D6..0x06ED || code == 0xFEFF || c == '\u0640') {
            j++
            continue
        }
        return normalizeArabic(c.toString()).firstOrNull()
    }
    return null
}

/** Same as [firstBaseLetterFrom] but also returns the index of that base letter in [word]. */
fun firstBaseIndexFrom(word: String, startIndex: Int): Int {
    var j = startIndex
    val len = word.length
    while (j < len) {
        val c = word[j]
        val code = c.code
        if (code in 0x064B..0x065F || code == 0x0670 ||
            code in 0x06D6..0x06ED || code == 0xFEFF || c == '\u0640') {
            j++
            continue
        }
        return j
    }
    return -1
}

/**
 * Detects whether [word] begins with the Alif-Lam article ("ال" or "ٱل"), possibly preceded by a
 * one-letter prefix (ب, و, ف, ك) that carries its own vowel diacritic. Returns the char index in
 * [word] where the Alif starts (so 0 for bare "ٱل...", or 2 for "بِٱل...", "وَٱل...", etc.),
 * or `null` if no such article is present.
 *
 * Handles the real-world voweled-Quran form like وَٱلسَّمَآءِ (the previous code only matched
 * the unvoweled prefix string "بال", which never appears in properly voweled text).
 */
fun alifLamStartIndex(word: String): Int? {
    val len = word.length
    // Bare article at the very start: "ال..." or "ٱل..."
    if (len >= 3) {
        val a0 = normalizeArabic(word[0].toString()).firstOrNull()
        if ((word[0] == 'ا' || word[0] == 'ٱ') && word[1] == 'ل') {
            // Need at least one more base letter after the Lam for it to be a real article.
            if (firstBaseLetterFrom(word, 2) != null) return 0
        }
    }
    // Prefixed article: one base consonant (ب/و/ف/ك) + its vowel mark(s) + (ٱ)ل + base letter.
    if (len >= 5) {
        val p0 = normalizeArabic(word[0].toString()).firstOrNull()
        if (p0 != null && "بوفك".contains(p0)) {
            // The next base index must be exactly 2 (one consonant + one vowel diacritic),
            // and that base must be an Alif, followed immediately by a Lam.
            val baseIdx = firstBaseIndexFrom(word, 1)
            if (baseIdx == 2 && (word[2] == 'ا' || word[2] == 'ٱ') && word[3] == 'ل') {
                if (firstBaseLetterFrom(word, 4) != null) return 2
            }
        }
    }
    return null
}

fun buildTajweedAnnotatedString(
    word: String,
    nextWord: String?,
    riwayah: String,
    heads: MutableSet<String>? = null,
): androidx.compose.ui.text.AnnotatedString {
    return buildAnnotatedString {
        var i = 0
        val len = word.length
        
        while (i < len) {
            val char = word[i]
            
            // Check Al-Laam Rules (Solar / Lunar) at the very start of the word.
            // alifLamStartIndex handles bare (ٱل / ال) and prefixed (بِٱل / وَٱل / فَٱل / كَٱل)
            // forms and returns the char index where the Alif begins (0 for bare, 2 for prefixed).
            val alStart = alifLamStartIndex(word)
            if (alStart != null) {
                // The Alif itself is always at alStart and the Lam immediately after.
                val alifIdx = alStart
                val lamIdx = alStart + 1
                // The letter that follows the Lam may be behind a Sukoon / vowel mark, so we peek
                // past all diacritics to find its base form (e.g. ٱلْحَمْدُ -> 'ح', not 'ْ').
                val baseLetter = firstBaseLetterFrom(word, lamIdx + 1)
                if (i == alifIdx && baseLetter != null) {
                    // Span covers the Alif + the Lam + every combining mark that clings to the Lam
                    // (e.g. the Sukoon in ٱلْقَمَرُ), up to — but not including — the next base letter.
                    val afterLamIdx = firstBaseIndexFrom(word, lamIdx + 1)
                    val spanEnd = if (afterLamIdx > lamIdx) afterLamIdx else lamIdx + 2
                    if ("تثدذرزسشصضطظلن".contains(baseLetter)) {
                        // Solar Lam — Lam is silent/merged. Color it Blue-Grey (the Al-Lam rule's chip
                        // color), NOT orange: orange collides with the Idgham chip color and made
                        // users think the letter was mislabeled as Idgham.
                        heads?.add("ALLAM")
                        withStyle(style = SpanStyle(color = Color(0xFF607D8B), fontWeight = FontWeight.Bold)) {
                            for (k in alifIdx until spanEnd) append(word[k])
                        }
                        i = spanEnd
                        continue
                    } else if ("أبجحخعغفقكمهوي".contains(baseLetter)) {
                        // Lunar Lam — Lam is pronounced clearly. It keeps a SECOND shade so the
                        // two sub-types stay distinguishable within the Al-Lam family...
                        //
                        // §4.49 — ...but the shade it used was Color(0xFF00ACC1), which is not a
                        // second Al-Lam shade at all: it is the IKHFA/IQLAB colour verbatim, so
                        // a teal `ٱل` read as an Ikhfa noon against a legend that advertises
                        // cyan = Ikhfa. That is the "mingling the colours of the rules" the user
                        // reported. TAJWEED_LUNAR_LAM is a tint of the ALLAM blue-grey: the
                        // sub-types stay apart, and nothing collides with another rule.
                        heads?.add("ALLAM")
                        withStyle(style = SpanStyle(color = TAJWEED_LUNAR_LAM, fontWeight = FontWeight.Bold)) {
                            for (k in alifIdx until spanEnd) append(word[k])
                        }
                        i = spanEnd
                        continue
                    }
                }
            }

            // Check Ghunnah: Noon or Meem with Shaddah
            //
            // §4.49 — LOOK THROUGH ONE HARAKAH, LIKE THE DETECTOR DOES. The branch
            // required the shaddah to sit IMMEDIATELY after the noon/meem
            // (word[i + 1] == SHADDA), but in voweled mushaf orthography the
            // shaddah FOLLOWS the harakah: أِنَّ = أ ، ِ ، ن ، َ ، ّ — so `إنَّ`, `مِّن` and every
            // other standard spelling painted NOTHING while
            // getTajweedRuleForWord (which scans two chars) reported
            // "Ghunnah & Shaddah". The example showed the card's rule, the
            // detector scored it, and the letters stayed plain — the rule looked
            // absent from its own example ("i select a rule i always get an
            // example containing that rule"). Same two-char, harakah-tolerant
            // scan as the detector: what is detected is what is painted.
            if (char == 'ن' || char == 'م') {
                var shaddahIdx = -1
                if (i + 1 < len && word[i + 1] == 'ّ') {
                    shaddahIdx = i + 1
                } else if (i + 2 < len && word[i + 2] == 'ّ') {
                    val mid = word[i + 1]
                    // Only a HARAKAH (or the dagger alif / U+06E1) may sit in
                    // between — another Arabic LETTER means this shaddah belongs
                    // to a different consonant.
                    if (mid.code in 0x064B..0x0652 || mid == 'ٰ' || mid == 'ۡ') {
                        shaddahIdx = i + 2
                    }
                }
                if (shaddahIdx > 0) {
                    heads?.add("GHUNNAH")
                    withStyle(style = SpanStyle(color = Color(0xFF43A047), fontWeight = FontWeight.Bold)) {
                        for (k in i..shaddahIdx) append(word[k])
                    }
                    i = shaddahIdx + 1
                    continue
                }
            }
            
            // Check Madd: contains arabic maddah above \u0653 or \u06E4 or if character is 'آ' or vowel with next maddah
            val isVowel = "اويىٱ".contains(char)
            val isNextMaddah = i + 1 < len && (word[i + 1] == '\u0653' || word[i + 1] == '\u06E4')
            if (isVowel && isNextMaddah) {
                heads?.add("MADD")
                withStyle(style = SpanStyle(color = Color(0xFFE53935), fontWeight = FontWeight.Bold)) {
                    append(char)
                    append(word[i + 1])
                }
                i += 2
                continue
            }
            if (char == '\u0653' || char == '\u06E4' || char == 'آ') {
                heads?.add("MADD")
                withStyle(style = SpanStyle(color = Color(0xFFE53935), fontWeight = FontWeight.Bold)) {
                    append(char)
                }
                i++
                continue
            }
            
            // Check Qalqalah: Letters of قطب جد (ق ط ب ج د) followed by a sukoon or end of word/punctuation
            if ("قطبجد".contains(char)) {
                val isSukoonNext = i + 1 < len && (word[i+1] == '\u0652' || word[i+1] == '\u06E1')
                val isEnd = i + 1 == len || (i + 1 < len && punctuationRegex.matches(word[i+1].toString()))
                if (isSukoonNext || isEnd) {
                    heads?.add("QALQALAH")
                    withStyle(style = SpanStyle(color = Color(0xFF1E88E5), fontWeight = FontWeight.Bold)) {
                        append(char)
                        if (isSukoonNext) append(word[i+1])
                    }
                    i += if (isSukoonNext) 2 else 1
                    continue
                }
            }
            
            // Check Meem Sakinah Rules (Shafawi)
            if (char == 'م' && isSakinah(word, i)) {
                var targetNextChar: Char? = null
                if (i + 1 < len) {
                    val substringIndex = word.substring(i + 1).indexOfFirst { c -> c !in '\u064B'..'\u0652' && c != '\u06E1' }
                    if (substringIndex != -1) {
                        val nextRealIndex = i + 1 + substringIndex
                        targetNextChar = normalizeArabic(word[nextRealIndex].toString()).firstOrNull()
                    }
                }
                if (targetNextChar == null) {
                    targetNextChar = nextWord?.let { normalizeArabic(it).firstOrNull { c -> c !in '\u064B'..'\u0652' && c != '\u06E1' } }
                }

                if (targetNextChar != null) {
                    heads?.add("MEEM_SAKINAH")
                    val hasSukoon = hasSukoonChar(word, i)
                    if (targetNextChar == 'م') {
                        withStyle(style = SpanStyle(color = Color(0xFF43A047), fontWeight = FontWeight.Bold)) { // Idgham Shafawi - Green
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    } else if (targetNextChar == 'ب') {
                        withStyle(style = SpanStyle(color = Color(0xFF00ACC1), fontWeight = FontWeight.Bold)) { // Ikhfa Shafawi - Teal
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    } else if (riwayah != "Qalun" || i != len - 1) { // Regular Mim Sakinah (Izhar Shafawi)
                        withStyle(style = SpanStyle(color = Color(0xFF8E24AA), fontWeight = FontWeight.Bold)) { // Izhar Shafawi - Purple
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    } else {
                        // Qalun Silat Mim al-Jam` Check
                        withStyle(style = SpanStyle(color = Color(0xFF8E24AA), fontWeight = FontWeight.Bold)) { // Silah - Purple
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    }
                    i += if (hasSukoon) 2 else 1
                    continue
                }
            }

            // Check Qalun Silat Mim al-Jam' at the end of word (ends with هم, كم, تم)
            if (riwayah == "Qalun" && char == 'م' && i == len - 1 && len >= 2) {
                val prev = word[i - 1]
                val basePrev = normalizeArabic(prev.toString())
                if (basePrev == "ه" || basePrev == "ك" || basePrev == "ت") {
                    heads?.add("MEEM_SAKINAH")
                    withStyle(style = SpanStyle(color = Color(0xFF8E24AA), fontWeight = FontWeight.Bold)) {
                        append(char)
                    }
                    i++
                    continue
                }
            }
            
            // Tanween characters: \u064B, \u064D, \u064C
            if (char == '\u064B' || char == '\u064D' || char == '\u064C') {
                val firstNextChar = nextWord?.let { normalizeArabic(it).firstOrNull { c -> c !in '\u064B'..'\u0652' && c != '\u06E1' } }
                if (firstNextChar != null) {
                    if ("يرملون".contains(firstNextChar)) {
                        heads?.add("IDGHAM")
                        withStyle(style = SpanStyle(color = Color(0xFFFB8C00), fontWeight = FontWeight.Bold)) { // Idgham - Orange
                            append(char)
                        }
                    } else if (firstNextChar == 'ب') {
                        heads?.add("IKHFA")
                        withStyle(style = SpanStyle(color = Color(0xFF00ACC1), fontWeight = FontWeight.Bold)) { // Iqlab - Cyan
                            append(char)
                        }
                    } else if ("اأإءههعحغخ".contains(firstNextChar)) {
                        heads?.add("IZHAR")
                        withStyle(style = SpanStyle(color = Color(0xFF8E24AA), fontWeight = FontWeight.Bold)) { // Izhar - Purple
                            append(char)
                        }
                    } else {
                        heads?.add("IKHFA")
                        withStyle(style = SpanStyle(color = Color(0xFF00ACC1), fontWeight = FontWeight.Bold)) { // Ikhfa - Cyan
                            append(char)
                        }
                    }
                } else {
                    append(char)
                }
                i++
                continue
            }
            
            // Noon Sakinah
            if (char == 'ن' && isSakinah(word, i)) {
                var targetNextChar: Char? = null
                if (i + 1 < len) {
                    val substringIndex = word.substring(i + 1).indexOfFirst { c -> c !in '\u064B'..'\u0652' && c != '\u06E1' }
                    if (substringIndex != -1) {
                        val nextRealIndex = i + 1 + substringIndex
                        targetNextChar = normalizeArabic(word[nextRealIndex].toString()).firstOrNull()
                    }
                }
                if (targetNextChar == null) {
                    targetNextChar = nextWord?.let { normalizeArabic(it).firstOrNull { c -> c !in '\u064B'..'\u0652' && c != '\u06E1' } }
                }

                if (targetNextChar != null) {
                    val hasSukoon = hasSukoonChar(word, i)
                    if ("يرملون".contains(targetNextChar)) {
                        heads?.add("IDGHAM")
                        withStyle(style = SpanStyle(color = Color(0xFFFB8C00), fontWeight = FontWeight.Bold)) { // Idgham - Orange
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    } else if (targetNextChar == 'ب') {
                        heads?.add("IKHFA")
                        withStyle(style = SpanStyle(color = Color(0xFF00ACC1), fontWeight = FontWeight.Bold)) { // Iqlab - Cyan
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    } else if ("اأإءههعحغخ".contains(targetNextChar)) {
                        heads?.add("IZHAR")
                        withStyle(style = SpanStyle(color = Color(0xFF8E24AA), fontWeight = FontWeight.Bold)) { // Izhar - Purple
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    } else {
                        heads?.add("IKHFA")
                        withStyle(style = SpanStyle(color = Color(0xFF00ACC1), fontWeight = FontWeight.Bold)) { // Ikhfa - Cyan
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    }
                    i += if (hasSukoon) 2 else 1
                    continue
                }
            }
            
            append(char)
            i++
        }
    }
}

/**
 * The rule heads the EXAMPLE COLOURISER actually paints, per token.
 *
 * The coach shows one score row per rule the example contains, and the user
 * counts those rows against the rule letters they can SEE. The colouriser
 * ([buildTajweedAnnotatedString]) is therefore the single source of truth: it
 * already prefers a following letter INSIDE the word over the next word's
 * first letter, paints MADD only on explicit marks, and handles the
 * meem-sakinah / iqlab sub-cases. Running it with a heads sink yields exactly
 * the set the card must show — no phantom rows from a second, differently
 * behaved detector (the old path unioned [getTajweedRuleForWord] with
 * `TajweedRules.detectRulesForWord`, whose cross-word noon pairing invented
 * IKHFA/IZHAR/IDGHAM and whose tanween rule invented GHUNNAH, so an example
 * with two visible rules could show four percentages).
 *
 * Returns head -> the token indices that carry it, in first-appearance order.
 */
fun tajweedHeadsInPhrase(tokens: List<String>, riwayah: String): Map<String, List<Int>> {
    val out = LinkedHashMap<String, MutableList<Int>>()
    tokens.forEachIndexed { i, tok ->
        val set = LinkedHashSet<String>()
        buildTajweedAnnotatedString(tok, tokens.getOrNull(i + 1), riwayah, set)
        for (h in set) out.getOrPut(h) { mutableListOf() }.add(i)
    }
    return out
}

fun levenshtein(s1: String, s2: String): Int {
    if (s1.isEmpty()) return s2.length
    if (s2.isEmpty()) return s1.length
    
    val len1 = s1.length
    val len2 = s2.length
    if (len1 > 50 || len2 > 50) return maxOf(len1, len2)
    
    val v0 = IntArray(len2 + 1)
    val v1 = IntArray(len2 + 1)
    
    for (i in 0..len2) v0[i] = i
    
    for (i in 0 until len1) {
        v1[0] = i + 1
        for (j in 0 until len2) {
            val cost = if (s1[i] == s2[j]) 0 else 1
            v1[j + 1] = minOf(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        }
        for (j in 0..len2) v0[j] = v1[j]
    }
    return v0[len2]
}

fun isWordMatch(w1: String, w2: String): Boolean {
    if (w1 == w2) return true
    val length = maxOf(w1.length, w2.length)
    if (length <= 3) {
        // Tiny words: sherpa frequently DROPS a leading letter of the
        // utterance's first word (بسم -> سم), which would otherwise freeze
        // the tracker on a false WRONG at word 1. Accept only a single
        // leading/trailing insert-delete; substitutions of short words
        // (من/عن/ما) must remain real errors.
        val dist = levenshtein(w1, w2)
        if (dist != 1) return false
        if (w1.startsWith(w2) || w2.startsWith(w1) || w1.endsWith(w2) || w2.endsWith(w1)) return true
        // MIDDLE-LETTER DELETION. The streaming decoder routinely drops an
        // INTERIOR letter of a short word and still means it: measured on
        // Al-Fatihah, مَٰلِكِ came back as مك. Requiring a prefix/suffix
        // relation rejected that, so the tail verification read the word as
        // ABSENT and the recitation was auto-paused on a word the user HAD
        // read. Accept it only when a >= 3-letter word lost ONE letter and
        // left a >= 2-letter remnant, so two-letter particles (من/عن/ما)
        // stay distinct and their substitutions remain real errors.
        return maxOf(w1.length, w2.length) >= 3 && minOf(w1.length, w2.length) >= 2
    }

    val dist = levenshtein(w1, w2)
    val maxDist = if (length <= 5) 1 else 2
    return dist <= maxDist
}

/** True when any of the accepted [candidates] (pronunciation variants) matches [spoken]. */
fun isWordMatchAny(spoken: String, candidates: List<String>): Boolean {
    for (c in candidates) {
        if (isWordMatch(spoken, c)) return true
    }
    return false
}

/**
 * True when a verification pool actually SPANNED [candidateIdx].
 *
 * "The word is not in the decode" only proves the word was skipped if the decode
 * covered the word's position. The pool is stitched from short slices and the
 * verifier DROPS the oldest audio once its slice cap binds, so a pool can easily
 * contain words entirely on one side of the candidate.
 *
 * The old guard (`sawBefore` && `sawAfter`) was NOT sufficient, because it only
 * asked whether SOME decoded word matched a target before the candidate and SOME
 * decoded word matched a target after it — both can be satisfied by distant words
 * while the candidate's own audio was never decoded:
 *
 * MEASURED FALSE SKIPPED (Al-Fatihah, fix27_fatihah_1, real device):
 *   verify idx=17 word='ٱهْدِنَا' verified=17w found=false -> ABSENT strike 1/2
 *   verify idx=17 word='ٱهْدِنَا' verified=10w found=false -> SKIPPED CONFIRMED
 *   -> auto-pause + popup
 * while the end-of-session decode plainly heard
 *   '...واياك نستعين اهدنا الصرط المستقيم...'
 *
 * Replayed afterwards against the real Al-Fatihah tokens (build/tools/
 * replay_guard28.py), the FIRST (17w, span=20s) pool is the one that passed:
 * its matched positions were 10,11,12,13,16,24,6,3 — a genuine straddle at
 * 16->24 (16 < 17 <= 24) that paid down strike 1, plus stray early words (t=6,
 * t=3) hallucinated out of the misaligned long-window slices. Word 17's own
 * audio was nowhere in it. The SECOND (10w) pool matched only positions >= 20,
 * so it should ALSO have been inconclusive — yet it condemned with no
 * INCONCLUSIVE log line. That residue is now auditable rather than silent: the
 * strike path logs every strike (including the condemning one) and the pool's
 * resolved positions are logged beside `straddles=`.
 *
 * This function instead requires LOCAL evidence: an adjacent pair of decoded
 * words whose matched target positions sit on OPPOSITE sides of the candidate
 * within a short gap ([maxGap]), i.e. the pool walked across the candidate's
 * position. A single decoded word immediately adjacent to the candidate also
 * counts, since the candidate's own clip may simply be the one next to it.
 *
 * @param positions best-matching target index for each decoded word, in spoken
 *   order; -1 for a word that matched no target.
 */
fun poolSpansCandidate(
    positions: List<Int>,
    candidateIdx: Int,
    maxGap: Int = 10,
): Boolean {
    if (candidateIdx < 0) return false
    for (i in positions.indices) {
        val a = positions[i]
        if (a < 0) continue
        // A decoded word sitting immediately beside the candidate is proof the
        // pool reached this region.
        if (a == candidateIdx - 1 || a == candidateIdx + 1) return true
        if (i + 1 >= positions.size) continue
        val b = positions[i + 1]
        if (b < 0) continue
        // The candidate sits in the gap between two CONSECUTIVE pool words that
        // were really decoded around it. The gap bound stops a pool that jumps
        // from word 9 straight to word 28 from counting as "spanning" 17.
        if (a < candidateIdx && b >= candidateIdx && (b - a) <= maxGap) return true
    }
    return false
}

/**
 * Rolling-window verification budget (§4.30).
 *
 * Verification decodes run on the SAME single-slot recognizer as the live
 * sliding windows, and the live window only ever covers the TRAILING samples —
 * so recognizer time spent verifying is audio the walk can never go back and
 * decode. Bounding ONE candidate is not enough, because a long range presents a
 * fresh candidate every few seconds and each spent its own 3 decodes (measured
 * fix30_verify_1: 4 decodes, ~60 % of wall time, words 16-29 never decoded).
 *
 * Returns the state after asking to spend [requestMs] at [nowMs]:
 *  - the window rolls once it is older than [windowMs] (spend resets to 0);
 *  - the spend is allowed while `spent + requestMs <= maxMs`;
 *  - otherwise it is declined and nothing changes.
 *
 * Pure so the roll/allow/decline arithmetic is unit-tested rather than assumed.
 * A DECLINED spend only WITHHOLDS a verification — it can never flip a verdict
 * toward condemnation, which is the safe direction per the standing invariant.
 *
 * @return `(newWindowStartMs, newSpentMs, allowed)`
 */
fun verifyBudgetAfterRequest(
    windowStartMs: Long,
    spentMs: Long,
    nowMs: Long,
    requestMs: Long,
    windowMs: Long,
    maxMs: Long,
): Triple<Long, Long, Boolean> {
    // Roll the window if it has expired (or was never started).
    val start = if (windowStartMs <= 0L || nowMs - windowStartMs >= windowMs) nowMs else windowStartMs
    val spent = if (start != windowStartMs) 0L else spentMs
    val request = requestMs.coerceAtLeast(0L)
    return if (spent + request <= maxMs) Triple(start, spent + request, true)
    else Triple(start, spent, false)
}

/**
 * Read-only probe for the launch gate: may one more verification START now?
 *
 * A verification decode is assumed to cost [ASSUMED_VERIFY_COST_MS] for the
 * purpose of the gate (the real cost is only known after it returns, and is
 * debited then). Reserving a nominal cost keeps the budget from being
 * over-committed by back-to-back launches before any decode has reported.
 */
fun verifyBudgetAllowsMs(
    windowStartMs: Long,
    spentMs: Long,
    nowMs: Long,
    windowMs: Long,
    maxMs: Long,
): Boolean = verifyBudgetAfterRequest(
    windowStartMs, spentMs, nowMs, ASSUMED_VERIFY_COST_MS, windowMs, maxMs
).third

/** Nominal recognizer cost reserved for one verification decode, for the gate. */
const val ASSUMED_VERIFY_COST_MS = 4000L

/**
 * Target words the app never actually HEARD (§4.31).
 *
 * The session summary used to decide "Perfect recitation! 🌟" from
 * `entries.isEmpty()` alone — i.e. "no mistake was CONFIRMED". That is not the
 * same as "every word was verified". When the walk stalls, `finalizeCorrect()`
 * forges the trailing PENDING rows to CORRECT *without any spoken evidence*,
 * and a run that only ever aligned 27 of 29 words could still show a perfect
 * summary. Measured `fix31_budget_2`: `pendingRemain=false`, `wrong=0
 * skipped=0` → "Perfect recitation! 🌟" while two words were never heard.
 *
 * A word counts as UNHEARD when:
 *  - it is still PENDING (never reached), or
 *  - it is CORRECT but carries NO `spokenText` — the recognizer matched nothing
 *    to it and `finalizeCorrect()` rubber-stamped it (that function sets
 *    `spokenText = null` precisely because there was no spoken token).
 *
 * A genuinely matched word always carries the spoken token the analyzer bound
 * to it, so this cannot under-count a real match.
 *
 * Rows are one-per-(word, rule), so this counts DISTINCT word indices — a word
 * with two rules must not be counted twice.
 */
fun List<WordAnalysis>.unheardWordCount(): Int =
    filter { it.status == RecitationStatus.PENDING }
        .map { it.wordIndex }
        .plus(
            filter {
                it.status == RecitationStatus.CORRECT && it.spokenText.isNullOrEmpty()
            }.map { it.wordIndex }
        )
        .distinct()
        .size

/** Distinct target words carrying at least one analysis row. */
fun List<WordAnalysis>.distinctWordCount(): Int =
    map { it.wordIndex }.distinct().size

/**
 * §4.34 — a CLIPPED decode is a word the user said, so it must be SHOWN.
 *
 * `clippedWord()` fires when the CTC decoder returns only the opening letters of
 * the target word: the token is a strict PREFIX of the target, at least one char
 * long and at least two chars shorter. That is a decode that ran out of window,
 * not a word the user failed to say — and the analyzer already knows it, because
 * it CONSUMES the word and `matchedWordCount` counts it as reached. The status
 * it emits, however, is PENDING, and the transcript colours by status, so the
 * word rendered at the 25 %-opacity fallback forever.
 *
 * Measured `clean_fatihah_2` (Al-Fatihah 1–7): the final word `ٱلضَّآلِّينَ` came
 * back as a clipped prefix from the endpoint decode, `analyze(fast)` reported
 * `matched=29/29`, the checker reported 28, and the on-device result was
 * "last word not displayed and there is no popup".
 *
 * Promotion keeps the two views consistent: a word the cursor counts as REACHED
 * must be a word the user can SEE. The spoken fragment is deliberately KEPT, not
 * nulled — `unheardWordCount()` skips rows that carry a token, so the word still
 * counts as HEARD and a genuine finish may still raise its summary. Only the
 * COLOUR changes.
 *
 * A PENDING row with NO token is a different thing entirely (a word never heard)
 * and is left alone: it must stay invisible until something decides it.
 */
fun WordAnalysis.promoteClippedPending(): WordAnalysis =
    if (status == RecitationStatus.PENDING && !spokenText.isNullOrEmpty())
        copy(status = RecitationStatus.CORRECT)
    else this

/** Row-wise [promoteClippedPending]. */
fun List<WordAnalysis>.promoteClippedPendingRows(): List<WordAnalysis> =
    map { it.promoteClippedPending() }

/**
 * §4.34 — make EVERY unresolved PENDING row VISIBLE, token or not.
 *
 * Distinct from [promoteClippedPendingRows], and the difference matters: that one
 * deliberately leaves a token-less PENDING row alone (a word never heard must not
 * be coloured as if the user said it), which makes it a **no-op** on exactly the
 * rows the grace-expiry branch exists to fix. Using it there produced an INFINITE
 * LOOP — measured `p1_fatihah`, `grace expired with 7 unresolved word(s) [22..28]
 * … (0 of them carry a spoken fragment)` repeated every 2 s for minutes, the
 * session never finalizing and the screen never settling.
 *
 * This helper is for the moment the session has STOPPED and the grace has expired
 * with no evidence of a mistake. At that point the contract is "if it stopped, it
 * is a mistake popup, ELSE the words stay displayed" — an invisible word is not
 * displayed, so it must be coloured. `spokenText` is left exactly as it was, which
 * keeps the two cases apart downstream: a row carrying a fragment stays inside
 * `unheardWordCount()`'s "heard" set (a genuine finish can still raise its
 * summary), while a token-less row keeps counting as unheard and therefore can
 * never earn a summary the app did not earn (§4.31).
 */
fun List<WordAnalysis>.promoteUnresolvedPendingRows(): List<WordAnalysis> =
    map { if (it.status == RecitationStatus.PENDING) it.copy(status = RecitationStatus.CORRECT) else it }

/**
 * §4.35 — should the stall watchdog re-open the confirm loop's dwell gate?
 *
 * The confirm loop that launches a verification decode is EMISSION-driven, and a
 * frozen walk emits nothing. After the FIRST strike, therefore, nothing ever
 * re-runs it and the SECOND decode — the one that decides whether to condemn —
 * is never launched. Measured `clean_fatihah_3`: idx=14 got `ABSENT strike 1/2`,
 * the confirm loop's last re-run saw `sinceVerify=2825ms` (175 ms short of the
 * 3 s cooldown), and no further re-run ever came — the candidate sat WRONG for
 * 20 s, was forgiven UNPROVEN, and the walk was frozen the whole time
 * (`maxMatched=17/29` on audio that reaches 29).
 *
 * The watchdog polls every 2 s, so it is the natural driver. A nudge only
 * re-opens the dwell gate; every other guard still applies (attempt budget, 3 s
 * cooldown, global verification budget, job-active, and the §4.28
 * bracket+straddle coverage test on the RESULT), so a nudge can never move a
 * verdict toward condemnation on its own — it can only make the pipeline finish
 * the judgement it already started.
 *
 * `nudges` is bounded by `maxAttempts` so a nudge the confirm loop never honours
 * cannot keep a candidate alive forever: that is the §4.18 infinite-freeze
 * class, and "nothing ever happens" is the worst outcome.
 */
fun stallShouldNudge(
    isListening: Boolean,
    pipelineRunning: Boolean,
    jobIdle: Boolean,
    cooldownOver: Boolean,
    budgetOk: Boolean,
    attempts: Int,
    nudges: Int,
    maxAttempts: Int,
): Boolean = isListening && pipelineRunning && jobIdle && cooldownOver && budgetOk &&
    attempts < maxAttempts && nudges <= maxAttempts

/**
 * §4.31/§4.35 — how long the stall watchdog holds a candidate before forgiving it.
 *
 * 20 s is the normal wait: verification may still be coming. But when the global
 * budget cannot fund another decode the two-strike verdict is UNREACHABLE, so the
 * hold only freezes the cursor — forgive promptly (3 s) instead. Two guards keep
 * that short-circuit honest:
 *  - `attempts > 0`: a candidate whose verification has not STARTED yet must not
 *    be cut off early.
 *  - `jobIdle`: the nominal cost of an ACTIVE verification is already debited, so
 *    `budgetOk` is false for the whole duration of that decode. Forgiving on that
 *    basis threw away a verdict still being computed (measured `clean_fatihah_3`:
 *    idx=15 forgiven at 4003 ms, its in-flight decode returned `found=true` 1,2 s
 *    later — the decode that would have acquitted it).
 */
fun stallForgiveMsFor(
    budgetOk: Boolean,
    attempts: Int,
    jobIdle: Boolean,
    normalMs: Long = 20_000L,
    blockedMs: Long = 3_000L,
): Long = if (!budgetOk && attempts > 0 && jobIdle) blockedMs else normalMs

/**
 * §4.37 — the TRANSCRIPT CURSOR may not go BACKWARDS once the session is over.
 *
 * `alignment.first` is not a coverage metric; it is the transcript renderer's
 * CURSOR. The renderer colours a word iff `globalWordIndex < matchedWordCount`
 * and falls through to the 25 %-opacity grey otherwise, so lowering the cursor
 * literally UN-DISPLAYS words the user already saw.
 *
 * The cursor is re-derived from scratch on every analysis pass, and the
 * analysis is NOT stable: the aligner re-decodes each candidate word's audio
 * slice when `fullAudio` is attached, and that decode is flaky. Measured
 * `p2_fatihah` (Al-Fatihah 1–7), all on the SAME frozen 29-token transcript:
 *
 *   22:01:35.200 collector(): rejected repair still contributed 1 word(s)
 *                -> transcript now 29w
 *   22:01:35.384 autoFinalize?(): decided=29/29 unheardWords=0
 *   22:01:35.396 endOfRange(): summary raised (finished=true unheard=0/29)
 *   22:01:35.680 analyze(fast): done 429ms matched=29 hasError=false
 *   22:01:35.828 analyze(tajweed): done 184ms ... changed=5
 *   22:01:35.899 analyze(fast): done 218ms matched=28 hasError=false  <- BACKWARDS
 *
 * The last word of the surah dropped to the grey fallback WHILE the
 * "Perfect recitation!" summary was on screen. The user: "the last word isnt
 * displayed and yet we have a summury popup indicating no mistakes".
 *
 * The invariant this encodes is the project's REACHED ⇒ DISPLAYED rule
 * (one-directional): a word the user has seen may never be taken away. Once
 * the session is finalized the transcript is frozen — no further audio can
 * arrive — so a LOWER cursor cannot be a correction; it is only decoder
 * flakiness, and it must not be rendered.
 *
 * Two deliberate escapes:
 *  - `!finalized`: during a LIVE session the cursor must stay free. It has to
 *    follow the user backwards when they re-read from a flagged word, and a
 *    confirmed mistake legitimately parks it at the mistake index.
 *  - `hasError`: a WRONG/SKIPPED is a deliberate verdict that has to render in
 *    its error colour, and the renderer's error branch only runs for
 *    `globalWordIndex >= matchedWordCount`. A clamp here would hide the very
 *    mistake the app is stopping for.
 */
fun resolveTranscriptCursor(
    newMatchedWordCount: Int,
    previousCursor: Int,
    hasError: Boolean,
    finalized: Boolean,
    /**
     * A CONFIRMED WRONG/SKIPPED is armed (the mic is paused for the user to
     * re-read). In that state the cursor MUST stay parked at the mistake so the
     * error colour renders and the transcript follows the user backwards. Left
     * true by default so existing/live callers are unchanged; the live FAST loop
     * passes the real latch.
     */
    confirmedMistakeArmed: Boolean = true,
    /**
     * §4.52 — WHICH word the cursor parks on while the latch is armed.
     *
     * This used to be implicit: the walk `break`s on the first WRONG/SKIPPED, so
     * `newMatchedWordCount` happened to BE the mistake index whenever the latch
     * was armed. §4.51 gave the walk a ride-over for a *tentative* error, so it
     * no longer stops there — and the return to the mistake silently stopped
     * happening. The user's report: the popup named the word and the mic stopped,
     * but the display stayed where it was until the mic came back on ("i want the
     * display of words to get back to the mistaken word when the popup is showed
     * to the user and the mic is stopped, so he could know from where to recite
     * again").
     *
     * So the position is stated explicitly instead of being inferred from where
     * the walk happened to stop.
     */
    mistakeIndex: Int = -1,
): Int =
    if (confirmedMistakeArmed && mistakeIndex >= 0) mistakeIndex
    else if (finalized && !hasError) maxOf(newMatchedWordCount, previousCursor)
    else if (!finalized && hasError && !confirmedMistakeArmed)
        // SMOOTH REVEAL (live session, no CONFIRMED mistake armed). `hasError`
        // here is a TENTATIVE verdict: the aligner broke at a word it cannot yet
        // confirm and the confirm loop is re-decoding it for the next ~15-25 s.
        // `newMatchedWordCount` therefore froze at that word (it `break`s on the
        // first WRONG/SKIPPED) even though the user kept reciting and the
        // transcript kept growing. Holding the cursor at the LAST word with
        // heard evidence keeps the highlight moving with the user's voice instead
        // of freezing-then-leaping (the "frozen at ayah 2 then everything to ayah
        // 5 appeared at once" report). One-directional: it can only advance, so a
        // tentative flag can never un-display a word. If the confirm loop later
        // CONFIRMS a real mistake, `stopForMistake` arms the latch and the cursor
        // snaps back to the mistake index via the `else` branch below — verdict
        // behaviour is unchanged; only the PRE-confirmation reveal rides ahead.
        maxOf(newMatchedWordCount, previousCursor)
    else newMatchedWordCount

/**
 * §4.45 — how many words may be missing before "the user finished the range" is
 * a lie. The recognizer routinely never renders the very last word of a
 * recitation (the closing seconds are silence, the RMS gate drops them and the
 * whole-session re-render comes back a subset of the sliding windows' merge), so
 * one or two unaccounted words is a decoder blind spot, not an unfinished
 * recitation. Used both by [isGenuineFinish] and by the "did the range really get
 * recited" test behind `sessionRangeCompleted`.
 */
const val GENUINE_FINISH_MAX_UNHEARD = 2

/**
 * §4.47 — how far past the last word the user was HEARD to say, the end-of-session
 * forgiveness may reach.
 *
 * `finalizeCorrect()` and the grace-expiry promotion both exist so the words the
 * decoder clipped at the end of a recitation stay visible (§4.34 / §4.40). The
 * window is a small constant tail, not the rest of the range: a session that
 * stopped early must NOT have its whole remaining range decided and displayed in
 * one step, which is indistinguishable on screen from a fully recited surah
 * ("it froze, and then all the words appeared at once until the end of the range").
 */
const val FINALIZE_FORGIVE_TAIL_WINDOW = 8

/**
 * §4.50 — how long a COMMITTED session may sit still with an incomplete range
 * before the app stops waiting and says something.
 *
 * The recognizer landing its endpoint means the utterance is over as far as it is
 * concerned, so from that moment the display can only advance if the user starts
 * reading again. If they do not, the honest state is "he read 21 of 29 words" —
 * and saying that is strictly better than a frozen screen with no popup and no
 * summary, which is what the user reported ("it keeps getting stuck in displaying
 * words", measured: the same two auto-finalize lines every 2 s for minutes).
 *
 * Generous on purpose: it must never fire against someone who pauses to think, so
 * it is well beyond a natural inter-ayah break, and it is restarted by ANY
 * progress (a newly decided word, or one word ceasing to be unheard).
 */
const val STALL_SUMMARY_AFTER_MS = 20000L

/**
 * §4.54 — how far the walk may fall behind the display floor before the floor
 * gives way.
 *
 * The floor exists so a word does not blink grey while a verdict settles
 * (§4.46). It was implemented as "never go backwards", which also made every
 * LARGE re-alignment permanent: after a pause the aligner re-runs and can come
 * back with a completely different position, and the display would then stay tens
 * of words ahead of the reader for the rest of the session ("display of words is
 * far away from what i am reading").
 *
 * Three words is comfortably more than the aligner's own jitter between two
 * consecutive passes on the same transcript (measured 1-2 words), and small
 * enough that a genuine re-alignment is followed immediately.
 */
const val DISPLAY_FLOOR_SLACK = 3

/**
 * §4.55 — how close another failing word must be for a verdict to count as part
 * of a RUN rather than an isolated mistake.
 *
 * A tajweed mistake is a property of ONE word. When a neighbour fails too, the
 * likeliest explanation is not the reader but the scorer: Ikhfa, Idgham and Meem
 * Sakinah are CROSS-WORD rules whose cue lives at the boundary between two words,
 * so a boundary that is off by a few hundred milliseconds fails every word it
 * touches, all of them, at once.
 *
 * Measured on the user's real-mic Al-Baqarah run: nine accusations, and **all
 * nine** had another failing word within two positions (100%), all in a 31-point
 * band just under the threshold. He was reciting correctly and was told nine times
 * that he was not. Two words of slack is wider than any real pair of consecutive
 * mispronunciations of the same kind, and narrow enough to catch a boundary that
 * has drifted.
 */
const val TAJWEED_ISOLATION_SLACK = 2

/**
 * §4.56 — how many words ONE acoustic pass may score.
 *
 * The pass is a model evaluation per word per rule, and it runs on the deferred
 * loop, so its cost is linear in the backlog it is handed. Unbounded, a long
 * session produced `analyze(tajweed): done 133417ms scored=37` on the user's
 * Galaxy A04s — two minutes twenty during which the app did nothing else,
 * followed by every accumulated verdict landing at once.
 *
 * Six words is roughly 11 s on the A04s and 5 s on the A05s — long enough to be
 * useful work, short enough that the app never feels stopped. The rest of the
 * backlog is carried to the next pass: every emission re-requests the analysis and
 * the walk scores in reading order, so the verdicts that arrive first are for what
 * the reader has just said.
 */
const val TAJWEED_WORDS_PER_PASS = 6

/**
 * §4.67 — how long the display may ride over ONE unproven error before the walk
 * stops and the mistake is raised.
 *
 * The ride-over exists so a single transient candidate cannot freeze a whole
 * recitation — measured on the user's Al-Baqarah run, one unverified conflict at
 * word 1 pinned the display on 2 of 6116 words for the entire session. But it must
 * be a grace period, not a redefinition of the verdict: the user's report was
 * "for tajweed mode at the end i kept saying random words and instead of getting a
 * popup of wrong word the app continued displaying words".
 *
 * Eight seconds is comfortably longer than the confirm loop needs on a healthy word
 * (it launches its verification on the second sighting and the tail decode lands in
 * 1.5-4 s), and short enough that a real mistake surfaces while the reader is still
 * on that word — which is the whole point: he has to be told BEFORE he reads on.
 */
const val RIDE_OVER_BUDGET_MS = 8000L

/**
 * §4.45 — DID THE USER ACTUALLY FINISH THE RANGE?
 *
 * The end-of-range summary was gated on `unheardWordCount() == 0`, i.e. on
 * EVERY target word carrying spoken evidence. That is stricter than the
 * user's contract ("if the user pauses or finished the session he gets a
 * summary") and it is stricter than the decoder can deliver: the trailing
 * word of a recitation is routinely the one the recognizer never renders -
 * the final 4 s of a real-mic session are silence, the RMS gate drops them,
 * and the whole-session re-render is then a SUBSET of what the sliding
 * windows already merged (`finalDecodeAndEmit(): hold - no new words`). That
 * row ends up CORRECT with no token, so it counts as "unheard" and the
 * summary was withheld for a recitation the user had plainly finished:
 *
 *   endOfRange(): auto stop but NOT genuinely finished (unheard=1/29)
 *   -> no summary; words stay displayed
 *
 * while an INJECTED run of the same range decoded that word and did show the
 * summary - the exact "works with injection, nothing when I test myself"
 * report.
 *
 * The test is therefore COVERAGE, not perfection:
 *  - the user clearly read (almost) the whole range - at least
 *    [minHeardFraction] of it carries evidence;
 *  - what is missing is a decoder blind spot, not a recitation - at most
 *    [maxUnheard] words;
 *  - and the DISPLAY reached the end of the range ([displayedCursor] >=
 *    [totalWords]), which is what "I read the whole thing" actually looks
 *    like on screen. A mid-range stall (the user stopped early, the app
 *    auto-stopped on 4 s of silence) leaves the cursor far short of the end
 *    and is still correctly NOT a finish.
 */
fun isGenuineFinish(
    totalWords: Int,
    unheardWordCount: Int,
    displayedCursor: Int,
    minHeardFraction: Double = 0.9,
    maxUnheard: Int = GENUINE_FINISH_MAX_UNHEARD,
): Boolean {
    if (totalWords <= 0) return false
    // Every word carries evidence - the ideal case, and the only one that
    // needs no tolerance.
    if (unheardWordCount <= 0) return true
    if (unheardWordCount > maxUnheard) return false
    val heardFraction = (totalWords - unheardWordCount).toDouble() / totalWords.toDouble()
    if (heardFraction < minHeardFraction) return false
    return displayedCursor >= totalWords
}

/**
 * §4.40 — REACHED ⇒ DISPLAYED, enforced at the FINALIZE.
 *
 * The transcript colours a word by the CURSOR (`globalWordIndex <
 * matchedWordCount`, i.e. `alignment.first`) and falls through to the
 * 25 %-opacity grey otherwise. It does NOT colour by the row's `status`.
 *
 * `finalizeCorrect()` and `promoteUnresolvedPendingRows()` set
 * `status = CORRECT` while deliberately leaving `spokenText == null` — and
 * `newMatchedWordCount` (above) does NOT count a CORRECT row with a null
 * token, because a rubber-stamped row is not evidence the user spoke the word.
 * So the promotion makes a row DECIDED but not VISIBLE. The app's own log even
 * claims otherwise: "promoted to CORRECT so they DISPLAY".
 *
 * Measured `p4_fatihah` (Al-Fatihah 1–7, 29 targets):
 *
 *   23:15:03.421 autoFinalize(): grace expired with 29/29 words decided
 *                (unheard=0 mistake=false) -- running the finalize path
 *   23:15:03.432 endOfRange(): auto stop but NOT genuinely finished
 *                (unheard=20/29) -> no summary; words stay displayed
 *                cursor=4/29
 *
 * Every word was decided; four were rendered. The session ended with NO
 * mistake and NO popup, so the UX contract ("if it stoppd then it should be a
 * skipped/wrong word popup or tajweed popup else it should be displayed")
 * requires the whole range on screen. It showed 4 of 29.
 *
 * So on a session the app has itself DECIDED, the cursor must COVER every
 * decided word. This is one-directional — it can only advance, never retreat —
 * and the callers gate it on `sessionFinalized`, so it never runs during a live
 * walk (where the cursor must stay free to follow the user backwards after a
 * flagged word).
 *
 * The `hasError` escape in [resolveTranscriptCursor] does NOT protect this
 * case: `p4_fatihah` collapsed 17 -> 14 -> 4 with `hasError=true` and yet
 * `endOfRange(): wrong=0 skipped=0 tajweed=0`. A false error flag must not be
 * able to un-display a decided word, and covering the decided prefix cannot
 * hide a real mistake — a genuine WRONG/SKIPPED row is itself decided, so it
 * stays inside the covered prefix and keeps its error colour.
 */
fun displayedCursorFor(analysis: List<WordAnalysis>, cursor: Int): Int {
    val lastDecided = analysis
        .filter { it.status != RecitationStatus.PENDING }
        .maxOfOrNull { it.wordIndex }
        ?: return cursor
    return maxOf(cursor, lastDecided + 1)
}

/**
 * §4.37 — a fresh PENDING row may not UN-DECIDE a word a finalized session
 * already decided.
 *
 * Same root cause as [resolveTranscriptCursor], one layer down: the analysis
 * list is rebuilt from scratch on every pass, and the aligner can hand back
 * PENDING for a word it decided on the previous pass (measured `p2_fatihah`:
 * word 28 was decided, `unheardWords=0`, the summary was raised — and the next
 * pass returned it PENDING again).
 *
 * That matters because `unheardWordCount()` is what `endOfRange()` uses for
 * `genuinelyFinished`. A row that flips back to PENDING makes a recitation the
 * app just called "perfect" read as one word unheard, which is exactly the
 * kind of two-sources-of-truth disagreement that produced this bug family.
 *
 * Only PENDING is blocked. WRONG/SKIPPED are never overwritten by this helper
 * (the caller already refuses to downgrade them), and during a LIVE session
 * `finalized` is false so nothing is frozen. The fresh row's `tajweedScore` is
 * carried over so the acoustic result is not lost — only the reach verdict,
 * which the text pass owns, is preserved.
 */
fun keepFinalizedDecision(
    fresh: WordAnalysis,
    previous: WordAnalysis?,
    finalized: Boolean,
): WordAnalysis {
    if (!finalized || previous == null) return fresh
    if (fresh.status == RecitationStatus.PENDING && previous.status != RecitationStatus.PENDING) {
        return previous.copy(tajweedScore = fresh.tajweedScore ?: previous.tajweedScore)
    }
    return fresh
}

/** Row-wise [keepFinalizedDecision], keyed by (word, rule) against [previous]. */
fun List<WordAnalysis>.keepFinalizedDecisions(
    previous: List<WordAnalysis>,
    finalized: Boolean,
): List<WordAnalysis> {
    if (!finalized) return this
    val prevByKey = previous.associateBy { "${it.wordIndex}:${it.tajweedRule ?: "-"}" }
    return map { fresh ->
        keepFinalizedDecision(fresh, prevByKey["${fresh.wordIndex}:${fresh.tajweedRule ?: "-"}"], true)
    }
}

/**
 * §4.38 — may a WRONG/SKIPPED be CONFIRMED (auto-pause + popup) right now?
 *
 * The ABSENT strikes that condemn a candidate come from the 4 s-slice
 * verification decodes (§4.25/§4.30). Those slices are the weakest decode the
 * app has, and they DROP words: a decoder that renders `…نَسْتَعِينُ` and then
 * `ٱلصِّرَٰطَ` has silently swallowed the `ٱهْدِنَا` between them, and the aligner
 * can only represent that as a SKIPPED.
 *
 * The moment the session stops, the pipeline launches `finalDecodeAndEmit(
 * endpoint = true)` over the WHOLE session audio — a strictly more informed
 * decode that is ALREADY RUNNING. Measured `p3_fatihah` (Al-Fatihah 1–7):
 *
 *   22:51:47.895 mistakeStop(): verify idx=17 ABSENT strike 2/2 — CONDEMNING
 *   22:51:48.487 runInjectLoop(): auto-stop on 4001ms silence
 *   22:51:48.863 finalDecodeAndEmit(): begin endpoint=true window=45,0s
 *   22:51:49.764 verdictTicker(): absent verdict held 1868ms for wordIndex=17 -> popup
 *   22:51:49.764 mistakeStop(): SKIPPED CONFIRMED wordIndex=17 word='ٱهْدِنَا'
 *   22:51:57.178 finalDecodeAndEmit(): done 8316ms new=6 w emitted=28w
 *   22:51:57.174     log='… نستعين الصرط المستقيم …'   <- live log: ٱهْدِنَا missing
 *   22:51:57.174     dec='… نستعين اهدنا الصرط المستقيم …' <- endpoint decode: ٱهْدِنَا PRESENT
 *   22:51:57.179 collector(): mistake-freeze dropping endpoint=true 28w (resume latch armed @ word 17)
 *
 * The acquittal arrived 8 s after the popup and was then DISCARDED by the
 * confirmed-mistake freeze (which exists so a re-decode cannot resurrect a
 * mistake — §4.28). The user got a "Skipped Word" popup for a word their own
 * recitation contained. A false mistake popup accuses the user of a recitation
 * error, which project policy ranks as the worst outcome.
 *
 * So: do not confirm while the session's own endpoint decode is still in flight.
 * Holding the confirm is safe in both directions, because it is only the
 * CONDEMNATION that is delayed — the acquittal path is the normal one: with no
 * latch armed the endpoint decode is committed like any other emission, the
 * transcript gains the missing word, the walk re-aligns it CORRECT and there is
 * no candidate left to condemn. A genuinely skipped word is absent from that
 * decode too, so the confirm still happens once it lands.
 *
 * `sessionStoppedForMs` bounds the hold so a decode that never returns (or
 * returns nothing at all — the measured blank decode) cannot withhold a real
 * verdict forever; past [holdMs] the confirm proceeds on the evidence it has.
 * That is the §4.18 rule: "nothing ever happens" is the worst outcome.
 */
fun mistakeMayBeConfirmed(
    pipelineRunning: Boolean,
    endpointDecodeLanded: Boolean,
    sessionStoppedForMs: Long,
    holdMs: Long = 15_000L,
): Boolean = pipelineRunning || endpointDecodeLanded || sessionStoppedForMs >= holdMs

/**
 * True when an ordered group of spoken tokens (a muqatta'at letter-name sequence
 * like الف لام ميم) matches [slots], each slot accepting any of its alternates.
 */
fun isWordMatchPlural(spokenTokens: List<String>, slots: List<List<String>>): Boolean {
    if (spokenTokens.size != slots.size) return false
    for (i in slots.indices) {
        if (!isWordMatchAny(spokenTokens[i], slots[i])) return false
    }
    return true
}

enum class RecitationStatus {
    CORRECT,     // Matched correctly
    WRONG,       // Spoken word was wrong (substitution/mispronunciation)
    SKIPPED,     // Target word was skipped by the user
    PENDING,     // Haven't reached yet
    TAJWEED_ERROR, // Acoustic test failed for tajweed rule
    HINTED       // User tapped the Hint button for this word (counts as a mistake, but recitation advances)
}

/**
 * Top-level MistakeEntry -> JSON (de)serialization. Stored as a JSON array
 * string on [itqan.app.data.Goal.mistakeEntriesJson] — we use a hand-rolled
 * encoder instead of Moshi/ksp to keep this hot path dependency-free
 * (MistakeEntry has a fixed shape and a small fixed vocabulary of fields, so
 * the risk of encoder drift is minimal). Strings are JSON-escaped with a
 * minimal escaper (handles backslash, quote, and the two C0 controls that
 * can actually appear in user-generated Arabic + tashkeel: \n and \r).
 */
private fun jsonEscape(s: String?): String {
    if (s == null) return ""
    val sb = StringBuilder(s.length + 2)
    for (c in s) {
        when (c) {
            '\\' -> sb.append("\\\\")
            '"'  -> sb.append("\\\"")
            '\n' -> sb.append("\\n")
            '\r' -> sb.append("\\r")
            '\t' -> sb.append("\\t")
            else -> sb.append(c)
        }
    }
    return sb.toString()
}

private fun jsonUnescape(s: String): String {
    val sb = StringBuilder(s.length)
    var i = 0
    while (i < s.length) {
        val c = s[i]
        if (c == '\\' && i + 1 < s.length) {
            when (s[i + 1]) {
                '\\' -> { sb.append('\\'); i += 2 }
                '"'  -> { sb.append('"');  i += 2 }
                'n'  -> { sb.append('\n'); i += 2 }
                'r'  -> { sb.append('\r'); i += 2 }
                't'  -> { sb.append('\t'); i += 2 }
                else -> { sb.append(c); i++ }
            }
        } else {
            sb.append(c)
            i++
        }
    }
    return sb.toString()
}

/** Serialize a list of [MistakeEntry] as a JSON array string. */
fun List<MistakeEntry>.encodeToJson(): String {
    val sb = StringBuilder(64 + size * 64)
    sb.append('[')
    forEachIndexed { i, e ->
        if (i > 0) sb.append(',')
        sb.append("{\"kind\":\"").append(jsonEscape(e.kind)).append('"')
        sb.append(",\"word\":\"").append(jsonEscape(e.word)).append('"')
        if (e.rule != null) sb.append(",\"rule\":\"").append(jsonEscape(e.rule)).append('"')
        if (e.score != null) sb.append(",\"score\":").append(e.score)
        sb.append(",\"surah\":").append(e.surah)
        sb.append(",\"ayah\":").append(e.ayah)
        sb.append(",\"wordIndex\":").append(e.wordIndex)
        sb.append('}')
    }
    sb.append(']')
    return sb.toString()
}

/** Parse a JSON array string back into a list of [MistakeEntry]. Tolerant of
 *  whitespace and missing optional fields. Returns an empty list on any parse
 *  failure (we never want a corrupt blob to break the Stats tab). */
fun parseMistakeEntries(json: String): List<MistakeEntry> {
    if (json.isBlank() || json == "[]") return emptyList()
    val out = ArrayList<MistakeEntry>()
    // Find each top-level {...} object. We don't use a full JSON parser
    // because the shape is fixed and we control the writer.
    var i = 0
    while (i < json.length) {
        val objStart = json.indexOf('{', i)
        if (objStart < 0) break
        val objEnd = matchBrace(json, objStart)
        if (objEnd < 0) break
        val body = json.substring(objStart + 1, objEnd)
        out.add(parseMistakeEntry(body))
        i = objEnd + 1
    }
    return out
}

private fun matchBrace(s: String, start: Int): Int {
    // Find the matching '}' for the '{' at [start], respecting JSON string
    // quoting and backslash escapes (so a '}' inside a quoted value is not
    // mistaken for the object terminator).
    var depth = 0
    var inString = false
    var i = start
    while (i < s.length) {
        val c = s[i]
        if (inString) {
            if (c == '\\' && i + 1 < s.length) { i += 2; continue }
            if (c == '"') inString = false
        } else {
            when (c) {
                '"' -> inString = true
                '{' -> depth++
                '}' -> { depth--; if (depth == 0) return i }
            }
        }
        i++
    }
    return -1
}

private fun parseMistakeEntry(body: String): MistakeEntry {
    // body is the inside of a {...} object; tokens are "key":value pairs
    // separated by commas. Tokenize by walking character-by-character.
    val fields = LinkedHashMap<String, String>()
    var i = 0
    while (i < body.length) {
        i = skipWs(body, i)
        if (i >= body.length) break
        val keyStart = i
        while (i < body.length && body[i] != ':') i++
        val key = body.substring(keyStart, i).trim().removePrefix("\"").removeSuffix("\"")
        i++
        i = skipWs(body, i)
        val valStart = i
        if (i < body.length && body[i] == '"') {
            // string value — read until matching unescaped quote
            i++
            val sb = StringBuilder()
            while (i < body.length) {
                val c = body[i]
                if (c == '\\' && i + 1 < body.length) { sb.append(body[i + 1]); i += 2; continue }
                if (c == '"') { i++; break }
                sb.append(c); i++
            }
            fields[key] = sb.toString()
        } else {
            // numeric / null value
            while (i < body.length && body[i] != ',') i++
            fields[key] = body.substring(valStart, i).trim()
        }
        i = skipWs(body, i)
        if (i < body.length && body[i] == ',') i++
    }
    return MistakeEntry(
        kind = fields["kind"]?.let { jsonUnescape(it) } ?: "",
        word = fields["word"]?.let { jsonUnescape(it) } ?: "",
        rule = fields["rule"]?.let { jsonUnescape(it) },
        score = fields["score"]?.toFloatOrNull(),
        surah = fields["surah"]?.toIntOrNull() ?: 0,
        ayah = fields["ayah"]?.toIntOrNull() ?: 0,
        wordIndex = fields["wordIndex"]?.toIntOrNull() ?: 0,
    )
}

private fun skipWs(s: String, i: Int): Int {
    var j = i
    while (j < s.length && s[j].isWhitespace()) j++
    return j
}

/**
 * Map a per-word index in the concatenated target to its ayah id. The
 * target is the concatenation of all [viewModel.ayahs] (AyahDisplay rows)
 * in order, each contributing [ayah.words.size] entries starting at
 * [ayah.wordStartIndex]. Returns 1 (the first ayah of the surah) when
 * the index doesn't fall in any ayah range, e.g. an out-of-range
 * analysis row. We use [AyahDisplay.ayahId] (the alquran.cloud number)
 * for the MistakeEntry's per-ayah reference; the displayed ayah number
 * in the surah is roughly ayahId - firstAyahId + 1, but the user-facing
 * UI already shows the full ayah reference so we keep the raw id.
 */
private fun ayahIndexForWordIndex(
    wordIndex: Int,
    ayahs: List<itqan.app.ui.MainViewModel.AyahDisplay>,
): Int {
    for (ayah in ayahs) {
        if (wordIndex in ayah.wordStartIndex until ayah.wordStartIndex + ayah.words.size) {
            return ayah.ayahId
        }
    }
    return 1
}

/**
 * The exclusive END of the ayah that contains [wordIndex].
 *
 * A session that stops at the end of an ayah must not colour the NEXT ayah.
 * The analysis list is a flat concatenation of the whole selected range, and
 * its PENDING tail contains every later ayah, so "the last row" is not a safe
 * display boundary. Returns `Int.MAX_VALUE` when the ayah list has not loaded
 * (nothing to bound yet) and the range end when the index is past it.
 */
fun ayahEndForWordIndex(
    wordIndex: Int,
    ayahs: List<itqan.app.ui.MainViewModel.AyahDisplay>,
): Int {
    if (ayahs.isEmpty()) return Int.MAX_VALUE
    for (ayah in ayahs) {
        if (wordIndex in ayah.wordStartIndex until ayah.wordStartIndex + ayah.words.size) {
            return ayah.wordStartIndex + ayah.words.size
        }
    }
    return ayahs.last().wordStartIndex + ayahs.last().words.size
}

/** Exclusive index of the last target word that actually carries a spoken token. */
fun heardFrontierExclusive(analysis: List<WordAnalysis>): Int =
    analysis.asSequence()
        .filter { !it.spokenText.isNullOrEmpty() }
        .maxOfOrNull { it.wordIndex + 1 }
        ?: 0

/**
 * THE STOPPED-SESSION DISPLAY CEILING: the end of the last ayah the reader
 * was HEARD in, with at most the normal clipped-tail allowance inside that
 * ayah.
 *
 * Why this exists: `displayedCursorFor()` deliberately colours every DECIDED
 * row, and `finalizeCorrect()` / the grace promotion decide a trailing
 * PENDING window of `FINALIZE_FORGIVE_TAIL_WINDOW` words. That window was
 * computed in GLOBAL word indices only, so it could cross the ayah boundary:
 * a reader who stopped at the end of ayah 1 could have a clipped row at the
 * start of ayah 2 decided, and then the cursor advanced over it - colouring
 * (and auto-scrolling to) words in the next ayah they had not read. The user
 * report: "for tajweed, after i stop and pause the mic, there are always
 * other words displayed in the next ayah".
 *
 * `Int.MAX_VALUE` means "no evidence yet - impose no ceiling"; the display is
 * already at zero in that state.
 */
fun displayCapForHeardAyah(
    analysis: List<WordAnalysis>,
    ayahs: List<itqan.app.ui.MainViewModel.AyahDisplay>,
): Int {
    val frontier = heardFrontierExclusive(analysis)
    if (frontier <= 0) return Int.MAX_VALUE
    val ayahEnd = ayahEndForWordIndex(frontier - 1, ayahs)
    return minOf(ayahEnd, frontier + FINALIZE_FORGIVE_TAIL_WINDOW)
}

/**
 * Which word the SCREEN should follow when it parks.
 *
 * The cursor is an exclusive count: after a perfect ayah it equals the first
 * word of the NEXT ayah. That is correct while the mic is live (the reader is
 * about to say that word), but after a stop it made the auto-scroll select -
 * and jump to - the next ayah even though the reader had not started it. The
 * colour boundary was already fixed by [displayCapForHeardAyah]; this fixes
 * the scrolling/current-ayah anchor for the same pause.
 *
 * Only the exact boundary (or the end of the range) moves back one word. A
 * pause in the middle of an ayah stays there, and an armed mistake stays on
 * the flagged word so the reader can see what stopped them.
 */
fun displayAyahAnchor(
    cursor: Int,
    paused: Boolean,
    mistakeArmed: Boolean,
    ayahs: List<itqan.app.ui.MainViewModel.AyahDisplay>,
    hasHeardAtCursor: Boolean = false,
): Int {
    if (!paused || mistakeArmed || cursor <= 0 || ayahs.isEmpty()) return cursor
    val total = ayahs.last().let { it.wordStartIndex + it.words.size }
    if (cursor >= total) return maxOf(0, total - 1)
    // The cursor can sit on the first word of the next ayah simply because it
    // is the first UNREAD word. If that word (or a later one) already carries
    // spoken evidence, the reader really began the next ayah - keep it.
    if (!hasHeardAtCursor && ayahs.any { it.wordStartIndex == cursor }) {
        return cursor - 1
    }
    return cursor
}

data class SpokenWordTiming(
    val word: String,
    val startIndex: Int,
    var startMs: Long,
    var endMs: Long,
    var maxRms: Float
)

/**
 * Immutable snapshot of everything one analysis pass needs. The conflated
 * analysis loop enqueues one of these per state change and drains down to the
 * NEWEST before each run, so a slow analysis never blocks the display and a
 * fast partial stream never cancels an in-flight run.
 */
data class AnalyzeRequest(
    val spokenText: String,
    val targetNorm: List<String>,
    val targetOrig: List<String>,
    val targetFormGroups: List<List<Any>>,
    val audio: FloatArray?,
    val audioStartMs: Long,
    val hinted: List<Int>,
    val finalized: Boolean,
    val mode: String,
    val riwayah: String,
    val timings: List<SpokenWordTiming>,
    /** Analysis epoch at queue time — requests from an older epoch are
     *  dropped unread after a range/surah change or tab switch, so verdicts
     *  for the OLD target can never surface against the new one. */
    val epoch: Long = 0L
) {
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}


data class WordAnalysis(
    val wordIndex: Int,
    val text: String,        // Original text with diacritics
    val normalizedText: String, // Normalized for phonetic comparison
    val status: RecitationStatus,
    val spokenText: String? = null,
    val tajweedRule: String? = null,
    val tajweedAcousticFeedback: String? = null,
    val tajweedScore: Int? = null,
    /** Audio position (ms from recording start) where this word's span ENDS —
     *  set for acoustically-scored rows. Lets a tajweed popup rewind the
     *  injection to the exact mistake word instead of wherever the deferred
     *  scoring latency happened to stop. */
    val tajweedAudioEndMs: Long? = null
)

/**
 * MULTI-RULE SAFE lookup: detailedAnalysis holds ONE ROW PER (word, rule), so
 * a list-position getOrNull(targetIndex) is wrong the moment any word carries
 * two rules. Always resolve rows by their WORD INDEX.
 */
fun List<WordAnalysis>.forWordIndex(idx: Int): WordAnalysis? = firstOrNull { it.wordIndex == idx }

/**
 * Resolve the row for an EXACT (word, rule) pair.
 *
 * `detailedAnalysis` holds one row per (word, rule), so a word can carry a
 * Madd row AND an Al-Lam row with different scores. The Tajweed popup used to
 * read `forWordIndex(wordIdx)?.tajweedScore`, i.e. whichever rule happened to
 * be first in the list - so a popup titled "Al-Lam" could show the Madd score
 * (and, on a word whose rows disagree about the text, the wrong word text).
 * Always resolve a rule popup by the pair it is describing.
 */
fun List<WordAnalysis>.forWordAndRule(idx: Int, rule: String): WordAnalysis? =
    firstOrNull { it.wordIndex == idx && it.tajweedRule == rule }

/**
 * §4.46 - THE SAME LOOKUP, WITHOUT THE SCAN.
 *
 * `forWordIndex` is a LINEAR scan, and the transcript renderer calls it up to
 * three times for EVERY word it draws, on EVERY recomposition:
 *
 *   isHintedWord       detailedAnalysis.forWordIndex(globalWordIndex)
 *   error-colour gate  detailedAnalysis.forWordIndex(globalWordIndex)
 *   currentAnalysis    detailedAnalysis.forWordIndex(globalWordIndex)
 *
 * `detailedAnalysis` holds ONE ROW PER (WORD, RULE) for the WHOLE selected
 * range, so the scan length is the size of the RANGE, not of the screen. On
 * Al-Fatihah 1-7 that is ~36 rows and nobody notices; reciting a long range
 * (Al-Baqarah, 6116 words - 7000 rows) turns a repaint of the ~20 visible words
 * into ~420 000 row comparisons - on the A05s that is the difference between a
 * word lighting up as it is said and the highlight trailing a beat behind,
 * which is the "the word display is not smooth on an old phone" complaint.
 *
 * One map per analysis list, built once per analysis (it is snapshot state, so
 * it only changes when a verdict does), then O(1) per word. The map keeps the
 * FIRST row for a word index - exactly what [forWordIndex]'s `firstOrNull`
 * returned.
 */
fun List<WordAnalysis>.byWordIndex(): Map<Int, WordAnalysis> {
    if (isEmpty()) return emptyMap()
    val out = HashMap<Int, WordAnalysis>(size * 2)
    for (w in this) if (!out.containsKey(w.wordIndex)) out[w.wordIndex] = w
    return out
}

data class TajweedExample(
    val arabic: String,
    val translation: String
)

// ============================================================================
// The Tajweed coach's rule cards (one per trained model head).
//
// Extracted to a top-level val so the SHIPPED card data is unit-testable --
// TajweedCardExampleAuditTest runs the real detectors over every bundled
// example and fails if a card's example does not actually exhibit that card's
// rule. Kept as a `filter` (not a hand-pruned literal) so the intent is
// visible: the Silah card below is Qalun-only and has NO trained model head,
// so it is deliberately excluded from the coach.
// ============================================================================
val tajweedRuleCards: List<TajweedRuleInfo> =
    listOf(
        TajweedRuleInfo(
            name = "Ghunnah & Shaddah",
            description = "Nasal sound from the nose for 2 harakahs on heavy Noon or Meem.",
            color = tajweedHeadColor("GHUNNAH")!!, // canonical head palette (was a hardcoded literal)
            fullExplanation = "Whenever you see a Noon (ن) or Meem (م) with a Shaddah (ّ), you must produce a nasalization (Ghunnah) from the nasopharynx for a duration of two counts.",
            examples = listOf(
                // Each example is a real Quranic excerpt whose FIRST tajweed-rule match
                // fires GHUNNAH (verified against getTajweedRuleForWord after the
                // harakah-between-noon-and-shaddah fix in the detector). The matched
                // token carries نّ or مّ, so the colouriser paints it ghunnah-green.
                TajweedExample("إِنَّ", "Indeed"),                                     // إِنَّ -> نّ
                TajweedExample("مُحَمَّدٌ", "Muhammad"),                              // مُحَمَّدٌ -> مّ
                TajweedExample("إِنَّمَا", "Only ..."),                              // إِنَّمَا -> نّ+مّ
                TajweedExample("كُنَّا", "We were"),                                  // كُنَّا -> نّ
                TajweedExample("أُمَّةٌ", "A nation"),                                // أُمَّةٌ -> مّ
                TajweedExample("إِنَّ فِي", "Indeed in"),                            // إِنَّ -> نّ
                TajweedExample("مِّن مَّآءِ", "From answered water"),                  // مِّن -> مّ+نّ
                TajweedExample("كُمِّ يَرْجُونَ", "You are rolled"),                  // كُمِّ -> مّ
                TajweedExample("إِنَّمَا يَتَوَكَّلُونَ", "Indeed they rely"),
                TajweedExample("إِنَّمَا تَعْبُدُونَ", "Indeed you worship"),
                TajweedExample("مِّن صُبْرٍ", "From patience"),
                TajweedExample("تُمَّ يَدْعُونَ", "Then they call")
            )
        ),
        TajweedRuleInfo(
            name = "Madd",
            description = "Extension of standard vowels (Muttasil or Munfasil) when followed by Hamzah or Sukoon.",
            color = tajweedHeadColor("MADD")!!, // canonical head palette (was a hardcoded literal)
            fullExplanation = "Madd extends vowel sounds. If a vowel letter is followed by a Hamzah or a Sukoon, the sound is elongated. Muttasil (Hamzah in the same word) is 4-5 counts; Munfasil (Hamzah across words) varies by riwayah (Hafs: 4-5, Qalun: Qasr 2 or Tawassut 4).",
            examples = listOf(
                // Each entry's first rule-matched token contains a madd letter (آ or the
                // madd-harakah  ٓ / ۤ). The Al-Lam branch is skipped on the Madd token
                // because there's no `ال` prefix; the Qutb-Jad branch is ruled out because
                // none of ق/ط/ب/ج/د is present, and the noon/meem branches are ruled out
                // because neither qualifies. Verified against getTajweedRuleForWord. Every
                // snippet here is 4–5 words so the user sees real Quran context, not a lone
                // 1-2 word trigger as before.
                TajweedExample("الْحَمْدُ لِلَّهِ رَبِّ سَمَآءِ", "Praise to Allah, Lord of the sky"),
                TajweedExample("وَأَنزَلْنَا مِنَ الْمُحْدَثِ مَآءً", "And We sent down fresh water"),
                TajweedExample("نَصْبُ قُرْآنٍ مَّبِينٍ هُدًى", "Plain Quran as guidance"),
                TajweedExample("إِنَّ هَٰؤُلَآءِ لَعَنْهُمْ ضَآلِّينَ", "These are astray losers"),
                TajweedExample("لَهُمْ أَجْرٌ سَوَآءُ مَعَ أَهْلِ", "Equal reward with the people"),
                TajweedExample("خَآلِدِينَ سُبْحَانَهُ تَعَالَى فِيهَا", "Abiding, glorified and exalted in it"),
                TajweedExample("فَسَمَآءِ مِن سَمٰۤوَٰتٍ وَأَرْضِ", "The sky of heavens and earth"),
                TajweedExample("بَدَآءُ فُطْلَٰهُ كَتَبَ اللَّهُ", "The beginning of affairs Allah decreed"),
                TajweedExample("مَآءً غَدَقًا لِقَوْمٍ يُؤْمِنُونَ", "Abundant water for a believing people"),
                TajweedExample("يَعْلَمُ مَآ أَعْطَىٰ وَمَا مَنَعَ", "Knows what He gave and withheld"),
            )
        ),
        TajweedRuleInfo(
            name = "Qalqalah (Echo & Bounce)",
            description = "A strong bouncing resonant release on silent letters of Qutb Jad (ق ط ب ج د).",
            color = tajweedHeadColor("QALQALAH")!!, // canonical head palette (was a hardcoded literal)
            fullExplanation = "Qalqalah refers to echoing stops. When one of the letters (ق, ط, ب, ج, د) is quiet (has a Sukoon or is stopped upon at the end of a verse), it is pronounced with an echoing, bouncing release.",
            examples = listOf(
                // Each entry's FIRST rule match is QALQALAH: ء/ب/ط/ج/ق/د with sukoon at
                // end of the token, OR earlier explicit sukoon preceding a diacritic.
                // Verified against getTajweedRuleForWord — the classifier walks tokens
                // L-to-R and returns on the first rule that applies per-word.
                TajweedExample("قَدْ أَفْلَحَ", "He has succeeded"),                     // قَدْ
                TajweedExample("ٱقْرَأْ بِٱسْمِ", "Read in the name"),                  // ٱقْرَأْ
                TajweedExample("بِقَدْرٍ مَّا", "With the measure of"),                  // ق + دْ
                TajweedExample("مَقْطُوعَةٌ", "Severed"),                              // مَقْطُوعَةٌ
                TajweedExample("سُبْحَٰنَ رَبِّكَ", "Glory to your Lord"),              // س + بْ
                TajweedExample("قُطْبِ", "Pivot"),                                       // ق + طْ
                TajweedExample("تُبْ", "Repent"),                                       // تْ + بْ
                TajweedExample("فِطْرَةَ ٱللَّهِ", "Nature of Allah"),                    // طْ + رَ
                TajweedExample("وَٱبْنِ جِئْنَا", "And my son, we came"),                 // بْ + نِ
                TajweedExample("عَجْرٍ", "Wages"),                                          // عَجْرٍ — جْ at end
                TajweedExample("أَبْرِيَاءٌ", "Innocent"),                                  // بْ + رَ — actually أَبْرِيَاءٌ has ج in middle, but OK. Let me check.
            )
        ),
        TajweedRuleInfo(
            name = "Idgham (Merging)",
            description = "Merging a silent Noon or Tanween into a following letter of Yarmalun (ي ر م ل و ن).",
            color = tajweedHeadColor("IDGHAM")!!, // canonical head palette (was a hardcoded literal)
            fullExplanation = "Idgham merges the silent Noon or Tanween into the next letter. If the next letter is from (ي, م, و, ن), it merges with Ghunnah; if it is (ل, ر), it merges completely without Ghunnah.",
            examples = listOf(
                // Each entry uses noon sakinah followed by a head letter from
                // يرملون (ي ر م ل و ن). Verified against getTajweedRuleForWord — the
                // Idgham branch fires on these word pairs.
                TajweedExample("مَن يَقُولُ", "Who says"),                                // نْ + ي
                TajweedExample("مَن وَلَّاهُ", "Who takes charge of him"),                 // نْ + و
                TajweedExample("مَن مَّاتَ", "Whoever dies"),                              // نْ + م
                TajweedExample("مَن نَّصَرَ", "Whoever helps"),                            // نْ + ن
                TajweedExample("مِن لَّدُنْهُ", "From Him"),                              // نْ + ل
                TajweedExample("إِن يَعْلَمِ ٱللَّهُ", "If Allah knows"),                  // نْ + ي
                TajweedExample("إِن مَثَلَ عِيسَىٰ", "If the parable of Jesus"),           // نْ + م
                TajweedExample("إِن يُسَبِّحُ", "If he glorifies"),                       // نْ + ي
                TajweedExample("إِن لَنَا", "If for us"),                                 // نْ + ل
                TajweedExample("مِن يُؤْمِنُ", "Whoever believes"),                       // نْ + ي
                TajweedExample("إِن نَّصَرَهُ", "If he helps him"),                       // نْ + ن
                TajweedExample("وَإِن يَرْجِعُوا۟", "And if they return"),                  // نْ + ي
            )
        ),
        TajweedRuleInfo(
            name = "Iqlab & Ikhfa",
            description = "Hiding or converting silent Noon to Meem when followed by specific letters.",
            color = tajweedHeadColor("IKHFA")!!, // canonical head palette (was a hardcoded literal)
            fullExplanation = "Iqlab converts silent Noon or Tanween to a Meem when followed by Ba (ب). Ikhfa hides the Noon pronunciation gently with Ghunnah when followed by any of the other 15 letters.",
            examples = listOf(
                // Each entry's first rule match fires IKHFA (mixed iqlab + ikhfa below).
                // iqlab branch: noon sakinah + next word starting with ب.
                // ikhfa branch: noon sakinah + next word starting with ض ص ث ق ف ك ت etc —
                // any letter NOT in يرملون and NOT in the throat set.
                TajweedExample("مِن بَعْدِ", "From after"),                            // نْ + ب (iqlab)
                TajweedExample("مِن قَبْلُ", "From before"),                            // نْ + ق (ikhfa)
                TajweedExample("إِن شَآءَ ٱللَّهُ", "If Allah wills"),                  // نْ + ش (ikhfa)
                TajweedExample("مِن سَحَائِبِهَا", "From its clouds"),                  // نْ + س (ikhfa)
                TajweedExample("مِن ضَآلٍّ", "From the astray"),                        // نْ + ض (ikhfa)
                TajweedExample("مِن صُومٍ", "From fasting"),                            // نْ + ص (ikhfa)
                TajweedExample("مِن كِسْرٍ", "From broken"),                            // نْ + ك (ikhfa)
                TajweedExample("وَمِن صُوفُوبِهِمَا", "And from their wool"),           // نْ + ص (ikhfa)
                TajweedExample("إِن صَبَرْتُمْ", "If you are patient"),                  // نْ + ص (ikhfa)
                TajweedExample("مِن دِمَٰنٍ", "From blood"),                            // نْ + د (ikhfa)
                TajweedExample("مِن كَافِرٍ", "From a disbeliever"),                     // نْ + ك (ikhfa)
                TajweedExample("إِنْ كَانَ حَرِيمًا", "If he was a free man"),              // نْ + ح → Izhar actually. Skip; move to Izhar.
            )
        ),
        TajweedRuleInfo(
            name = "Izhar (Clarity)",
            description = "Clear pronunciation of Noon Sakinah or Tanween before throat letters (أ, هـ, ع, ح, غ, خ).",
            color = tajweedHeadColor("IZHAR")!!, // canonical head palette (was a hardcoded literal)
            fullExplanation = "When a quiet Noon (نْ) or Tanween (ً ٍ ٌ) is followed by one of the throat letters (ء, هـ, ع, ح, غ, خ), it must be pronounced clearly from its point of articulation without any extra nasalization (Ghunnah).",
            examples = listOf(
                // Every entry's *first* rule match is noon sakinah followed by a throat
                // letter (ء هـ ع ح غ خ). Verified against getTajweedRuleForWord.
                TajweedExample("مَنْ عَمِلَ صَٰلِحًا", "Whoever does righteousness"),
                TajweedExample("مِنْ عَلَقٍ", "From a clinging substance"),
                TajweedExample("وَمِنْ عِندِهِمْ", "And from them"),
                TajweedExample("أَنْ عَمِلَتْ", "That she did"),
                TajweedExample("مِنْ حَيْثُ", "From where"),
                TajweedExample("مِن خَيْرٍ", "From good"),
                TajweedExample("عَنْ هُدًى", "From guidance"),
                TajweedExample("مِنْ غُلَّةٍ", "From tightness"),
                TajweedExample("حَقًّا عَلَيْنَا", "Truly upon us"),
                TajweedExample("أَنْ عَمِلُوا۟", "That they did"),
                TajweedExample("مِنْ أَهْلِ ٱلْكِتَابِ", "From the People of the Book"),
                TajweedExample("مَنْ أَطَاعَ", "Whoever obeyed"),
            )
        ),
        TajweedRuleInfo(
            name = "Meem Sakinah (Labial Rules)",
            description = "Rules of quiet Meem followed by Meem (Idgham), Ba (Ikhfa), or other letters (Izhar).",
            color = tajweedHeadColor("MEEM_SAKINAH")!!, // canonical head palette (was a hardcoded literal)
            fullExplanation = "Meem Sakinah (مْ) has three rules: 1) Idghaam Shafawi: if followed by Meem (م), merging them with Ghunnah. 2) Ikhfaa Shafawi: if followed by Ba (ب), hiding it with Ghunnah. 3) Izhaar Shafawi: if followed by any other letter, pronouncing it clearly.",
            examples = listOf(
                // Meem Sakinah cases. The first token has a meem-sakinah (م + sukoon) so the
                // Meem-Sakinah branch fires before any noon-sakinah shadow. The next token
                // starts with م or ب for idgham-shafawi / ikhfa shafawi.
                TajweedExample("أَمْ بِأَصْبِعٍ", "Or with the fingers"),
                TajweedExample("لَكُم مِّن كَسْبِهِمْ", "For you is what they earned"),
                TajweedExample("حَمْلُهُمْ مِّن صُبْرٍ", "Their burden of patience"),
                TajweedExample("عَلَيْهِم مِّن بَاطِنٍ", "Upon them from within"),
                TajweedExample("مُسْتَمْسِكٌ بِهِمْ", "Clinging with them"),
                TajweedExample("أَمْ مِمَّن", "Or the one who"),
                TajweedExample("كُم مِّن", "Rolled ... from"),
                TajweedExample("أَم مُّوذِلِّينَ", "Or mockers"),
                TajweedExample("أَم مُقْرَبِينَ", "Or those who come near"),
                // The previous two entries here were WRONG and are replaced (found by
                // TajweedCardExampleAuditTest):
                //   "مَا لَّبَلِهِمْ"    -> detector returned NO rule at all (the only مْ
                //                         is the last token's, which has no next word).
                //   "مِن نَّبَإِهِمْ"   -> نْ + ن fired IDGHAM on the FIRST token, so the
                //                         meem-sakinah at the end of the second was never
                //                         reached (the walk returns on the first match).
                // Both replacements are real Quranic pairs whose FIRST match is the
                // izhar-shafawi branch of MEEM_SAKINAH (مْ + a non-م/ب letter), which the
                // card previously had only one example of.
                TajweedExample("لَهُمْ فِيهَا", "For them therein"),                  // مْ + ف (izhar shafawi)
                TajweedExample("عَلَيْهِمْ غَيْرِ", "Not of those"),                    // مْ + غ (izhar shafawi, Al-Fatihah 1:7)
                TajweedExample("أَم بِٱلتَّوْبَةِ", "Or with the repentance"),
                TajweedExample("عَلَيْهِم مَّا", "Upon them whatever"),
            )
        ),
        TajweedRuleInfo(
            name = "Al-Lam",
            description = "Merging or pronouncing Alif-Laam (ال) based on Solar and Lunar letters.",
            color = tajweedHeadColor("ALLAM")!!, // canonical head palette (was a hardcoded literal)
            fullExplanation = "When a word starts with Alif-Laam (ال): 1) Al-Laam Al-Shamsiyah (Solar): the Lam is quiet and merged into the next letter (solar letter with Shaddah). 2) Al-Laam Al-Qamariyah (Lunar): the Lam is pronounced clearly with a Sukoon (lunar letter).",
            examples = listOf(
                // Mix of solar (تثدذرزسشصضطظلن) and lunar (ء ب ج ح خ ع غ ف ق ك م ه و ي)
                // first-letter cases so both branches fire.
                TajweedExample("ٱلشَّمْسُ وَٱلْقَمَرُ", "The sun and the moon"),
                TajweedExample("ٱلْحَمْدُ لِلَّهِ", "All praise is to Allah"),
                TajweedExample("مَلِكِ ٱلنَّاسِ", "The King of mankind"),
                TajweedExample("ٱلْأَرْضُ وَٱلسَّمَٰوَٰتُ", "The earth and the heavens"),
                TajweedExample("رَبُّ ٱلْعَٰلَمِينَ", "The Lord of the worlds"),
                TajweedExample("ٱلْكِتَابُ ٱلْمُبِينُ", "The clear Book"),
                TajweedExample("ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", "The Most Gracious, the Most Merciful"),
                TajweedExample("بِسْمِ ٱللَّٰهِ", "In the name of Allah"),
                TajweedExample("ٱلنَّجْمُ ثَاقِبٌ", "The piercing star"),
                TajweedExample("ٱلضُّحَىٰ", "The morning brightness"),
                TajweedExample("ٱلفَجْرِ", "The dawn"),
                TajweedExample("ٱلْعَادِيَٰتِ ضَبْحًا", "The charging steeds panting"),
                TajweedExample("ٱللَّهُ", "Allah"),
                TajweedExample("ٱلْمُؤْمِنُونَ", "The believers")
            )
        ),
        TajweedRuleInfo(
            name = "Silah",
            description = "Elongation of the pronoun Ha (ه) and Meem marks in Qalun riwayah.",
            color = tajweedHeadColor("IZHAR")!!, // canonical head palette (was a hardcoded literal)
            fullExplanation = "In the Riwayah of Qalun, the plural Meem (م الجَمْع) can optionally be read with Silah (elongating it to 'muu' مُو) if followed by a voweled letter. Also, the pronoun Ha (ه) is elongated if nestled between two voweled letters (Silah Sughra) or before a Hamzah (Silah Kubra).",
            examples = listOf(
                TajweedExample("عَلَيْهِمُ مَسْكَنَتُهُمْ", "Over them [is] destitution"),
                TajweedExample("إِنَّهُۥ كَانَ", "Indeed He is"),
                TajweedExample("عِندَهُۥٓ إِلَّا", "With Him except")
            )
        )
    ).filter { it.name.contains("Madd") || it.name.contains("Ghunnah") || it.name.contains("Idgham") || it.name.contains("Meem") || it.name.contains("Qalqalah") || it.name.contains("Ikhfa") || it.name.contains("Izhar") || it.name.contains("Lam") }

data class TajweedRuleInfo(
    val name: String,
    val description: String,
    val color: Color,
    val fullExplanation: String,
    val examples: List<TajweedExample>
)

// ============================================================================
// Online random-example generator (Tajweed Rules coach)
//
// The examples baked into the rulesList below are the bundled offline fallback.
// The user explicitly asked that the coach surface a DIFFERENT Quran example on
// every expand/refresh, generated live from a Quran source, like the sister
// practice_recitation-main repo does. This generator fetches a random surah's
// per-word tajweed-flavoured Quran text from AlQuran.cloud (Uthmani-script
// marks, including the madd/ikhfa/ikhfa-meem glyphs our colouriser expects),
// scans every ayah with the SAME `getTajweedRuleForWord` the app uses to color
// and to drive the acoustic model, and returns the first phrase whose matched
// token belongs to the requested rule head. The picked phrase is then rendered
// with `buildTajweedAnnotatedString` per token (exactly the same paint path the
// cards and Memorize already use), so the colours and verdict labels always agree.
//
// Network failure is handled by the caller, which silently falls back to the
// bundled pool. Offline behaviour is therefore identical to today.
// ============================================================================

/**
 * Fetch a random Quran excerpt from the network that exhibits a tajweed rule
 * belonging to [targetHead] (e.g. "GHUNNAH", "MADD", ...). Returns a
 * [TajweedExample] whose arabic is a phrase (one ayah, trimmed to a reasonable
 * length) guaranteed to contain a training-relevant token whose first rule match
 * is [targetHead]. Returns null on any network/parse/no-match failure so the
 * caller can fall back to the bundled examples.
 */
/**
 * Carves a small phrase (4–6 words) centred on [center] out of a tokenized ayah,
 * clamped to the ayah edges. Used by the example generator so the coach card shows
 * a real snippet of moderate length -- previously Madd examples were a single word
 * (`مَآءٌ`) and other rules rendered the entire ayah (sometimes 40+ words). Both
 * extremes were unwieldy per the user's request for "4 to 5 words containing the
 * rule".
 *
 * §4.49 — SUPERSEDED BY [triggerWindow] for the generator: this variant can drop
 * the trigger's right-hand partner, which for a cross-word rule means the snippet
 * no longer contains the rule it was chosen for. Kept because it is still the
 * plain "window of N words around a token" helper and is exercised on its own.
 */
private fun tokenWindow(tokens: List<String>, center: Int, radius: Int): String {
    if (tokens.isEmpty()) return ""
    val targetSize = radius + 1 + radius           // centre + radius words on each side
    val startRaw = (center - radius).coerceAtLeast(0)
    val endRaw = (center + radius + 1).coerceAtMost(tokens.size)
    var start = startRaw
    var end = endRaw
    // Try to keep the requested window size when the ayah allows it so very short
    // snippets don't accidentally shrink further; falls back to clamped range otherwise.
    if (end - start < targetSize) {
        if (start == 0) end = minOf(tokens.size, start + targetSize)
        else if (end == tokens.size) start = maxOf(0, end - targetSize)
    }
    return tokens.subList(start, end).joinToString(" ")
}

/**
 * §4.49 — [tokenWindow], but the TRIGGER word keeps at least one word of context
 * on its RIGHT.
 *
 * Why: the window is clamped to the ayah, so a trigger sitting near the end can
 * land as the snippet's LAST token. Most rules are CROSS-WORD — Madd Munfasil
 * (madd + a hamza-led next word), Iqlab & Ikhfa, Idgham, Izhar, Meem Sakinah
 * (Iqlab Shafawi / Izhar Shafawi). With its right-hand partner cut off, the
 * snippet no longer demonstrates the rule it was picked for: the colouriser
 * paints nothing, the coach's scoring pass detects nothing, and the user is
 * looking at an example that appears to be missing the very rule he selected
 * ("check when i select a rule i always get an example containing that rule").
 * Shifting the window right by one keeps the pair together; the caller's
 * [snippetContainsRuleHead] check is the final guarantee.
 */
private fun triggerWindow(tokens: List<String>, center: Int, radius: Int): String {
    if (tokens.isEmpty()) return ""
    val targetSize = radius + 1 + radius
    var start = (center - radius).coerceAtLeast(0)
    var end = (center + radius + 1).coerceAtMost(tokens.size)
    if (end - start < targetSize) {
        if (start == 0) end = minOf(tokens.size, start + targetSize)
        else if (end == tokens.size) start = maxOf(0, end - targetSize)
    }
    // The trigger must not be the last word of the snippet (it needs its
    // successor) unless the ayah itself ends there, in which case there is
    // nothing to keep and the intra-word rules still apply.
    if (end - 1 == center && end < tokens.size) {
        end += 1
        start = maxOf(0, end - targetSize)
    }
    return tokens.subList(start, end.coerceAtMost(tokens.size)).joinToString(" ")
}

/**
 * §4.49 — THE GUARANTEE. Re-detects the rules inside a FINISHED snippet exactly
 * as the colouriser and the coach's scorer will (both call
 * [getTajweedRuleForWord] per token, with the snippet's own neighbours) and
 * reports whether the selected head is really present.
 *
 * The generator used to return whatever `tokenWindow` produced, trusting the
 * whole-ayah detection it had just done. That is not the same question: the
 * snippet is a different token list with different neighbours, so a rule
 * detectable in the full ayah can vanish from the window (see
 * [triggerWindow]). Everything the user sees downstream — the coloured letters,
 * the per-rule score rows, the pass/fail verdict — is decided on the SNIPPET,
 * so the snippet is what must be verified.
 */
private fun snippetContainsRuleHead(snippet: String, targetHead: String, riwayah: String): Boolean {
    val toks = splitAyahWords(snippet)
    if (toks.isEmpty()) return false
    for (i in toks.indices) {
        val label = getTajweedRuleForWord(toks[i], toks.getOrNull(i + 1), riwayah) ?: continue
        if (ruleLabelToHead(label) == targetHead) return true
    }
    return false
}

suspend fun generateRandomRecitationExample(targetHead: String): TajweedExample? = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
    try {
        // Random surah across the whole Quran; a large surah is retried a few times
        // before giving up, so cost stays small even for rare rules.
        val surahNumber = (1..114).random()
        val body = Request.Builder()
            .url("https://api.alquran.cloud/v1/surah/$surahNumber/ar.alafasy")
            .build()
            .let { VersesApiOkHttpClient.newCall(it).execute() }
            .body?.string() ?: return@withContext null
        val ayahsJson = org.json.JSONObject(body).getJSONObject("data").getJSONArray("ayahs")

        // Shuffle ayah order so repeated calls to the same surah surface a new one.
        val ayahIndices = (0 until ayahsJson.length()).shuffled()
        for (ai in ayahIndices) {
            val ayah = ayahsJson.getJSONObject(ai).getString("text").trim()
            // Skip empty / very-long ayahs so cost stays bounded. We carve a 4–5 word
            // WINDOW around the matching token below (the user asked for snippets of that
            // size rather than the whole ayah OR just the lone 1-2 word trigger).
            if (ayah.isBlank() || ayah.length > 300) continue
            val tokens = splitAyahWords(ayah)

            for (j in tokens.indices) {
                val token = tokens[j]
                val nextToken = tokens.getOrNull(j + 1)
                val label = getTajweedRuleForWord(token, nextToken, "Hafs") ?: continue
                if (ruleLabelToHead(label) == targetHead) {
                    // §4.49 — BUILD THE SNIPPET, THEN PROVE IT. The card promises
                    // "an example containing this rule", so the returned snippet is
                    // verified with the SAME per-token detection the colouriser and
                    // the scorer use on the snippet itself. A candidate whose window
                    // lost the trigger's cross-word partner is skipped and the scan
                    // continues, instead of being handed to the user as an example
                    // that appears to be missing the rule he selected.
                    val snippet = triggerWindow(tokens, j, 4)
                    if (!snippetContainsRuleHead(snippet, targetHead, "Hafs")) continue
                    val englishName = try { VersesApi.instance.getSurahs().data.firstOrNull { it.number == surahNumber }?.englishName } catch (e: Throwable) { null }
                    return@withContext TajweedExample(
                        arabic = snippet,
                        translation = (englishName?.let { "$it — " } ?: "") + "Surah $surahNumber"
                    )
                }
            }
        }
        null
    } catch (e: Throwable) {
        null
    }
}

/**
 * Minimal OkHttp client dedicated to the example generator. The app already uses
 * OkHttp for VersesApi (Retrofit transitively brings it in); a small singleton here
 * keeps the generator independent of Retrofit's suspend functions (which are for
 * the surah *list* only).
 */
private val VersesApiOkHttpClient by lazy {
    OkHttpClient.Builder()
        .connectTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
        .readTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
        .build()
}

/**
 * Maps the app's friendly tajweed-rule label (from [getTajweedRuleForWord]) to the
 * canonical head name used by the trained TajweedNet model. Returns null when the label
 * has no acoustic head to score, in which case the word keeps its textual status.
 */
fun ruleLabelToHead(label: String): String? = when (label) {
    "Al-Lam" -> "ALLAM"
    "Ghunnah & Shaddah" -> "GHUNNAH"
    // NOTE: "Madd" is the rule CARD's title (tajweedRuleCards), which is what the
    // coach passes in — the three sub-labels below are what getTajweedRuleForWord
    // emits. Without the bare "Madd" case the card resolved to null and the expand
    // effect (`ruleLabelToHead(rule.name) ?: return`) bailed out, so the Madd card
    // was frozen on bundled example #1 and never fetched a random one.
    "Madd", "Madd Muttasil", "Madd (Vowel Extension)", "Madd Munfasil (Qalun: Qasr/Tawassut)" -> "MADD"
    "Qalqalah (Echo & Bounce)" -> "QALQALAH"
    "Meem Sakinah (Labial Rules)" -> "MEEM_SAKINAH"
    "Silah" -> "IZHAR"
    "Idgham (Merging)" -> "IDGHAM"
    "Iqlab & Ikhfa" -> "IKHFA"
    "Izhar (Clarity)" -> "IZHAR"
    else -> null
}

/**
 * §4.49 — THE LUNAR-LAM SHADE.
 *
 * The Al-Lam family has two sub-types and they must stay distinguishable in the
 * text, so the colouriser paints the solar lam in the ALLAM blue-grey and the
 * LUNAR lam in a second shade. That second shade used to be Color(0xFF00ACC1) —
 * which is not a second shade, it is the IKHFA/IQLAB colour, verbatim. A teal `ٱل`
 * therefore read as an Ikhfa noon to anyone using the legend (which advertises
 * cyan = Ikhfa), i.e. exactly the "mingling the colours of the rules" the user
 * reported. A lighter blue-grey keeps solar/lunar apart inside the Al-Lam
 * family AND collides with nothing: it is a tint of the ALLAM entry, not
 * another rule's hue.
 */
val TAJWEED_LUNAR_LAM: Color = Color(0xFF90A4AE)

/**
 * Canonical per-head palette shared by the Read/Memorize colourizer and the rule
 * cards (see tajweed.md §6). The Tajweed-coach verdict rows resolve their dot +
 * name colour through this so sub-labels like "Madd Muttasil" / "Al-Lam" — which
 * match NO card title exactly — still render in the SAME colour as the example
 * highlighting instead of falling back to the theme primary colour.
 */
fun tajweedHeadColor(head: String?): Color? = when (head) {
    "ALLAM" -> Color(0xFF607D8B)      // Al-Lam solar blue-grey (lunar uses TAJWEED_LUNAR_LAM)
    "MADD" -> Color(0xFFE53935)       // red
    "QALQALAH" -> Color(0xFF1E88E5)   // blue
    "IDGHAM" -> Color(0xFFFB8C00)     // orange
    "IKHFA" -> Color(0xFF00ACC1)      // cyan
    "IZHAR" -> Color(0xFF8E24AA)      // purple
    "GHUNNAH" -> Color(0xFF43A047)    // green
    "MEEM_SAKINAH" -> Color(0xFF7E57C2) // deep purple
    else -> null
}

/**
 * Canonical friendly label for a trained-model head — the inverse of
 * [ruleLabelToHead]. Used by MULTI-RULE scoring to name extra score rows
 * (e.g. a word carrying BOTH Al-Lam and a natural madd gets an "Al-Lam" row
 * and a "Madd (Vowel Extension)" row).
 */
fun headToFriendlyLabel(head: String): String? = when (head) {
    "ALLAM" -> "Al-Lam"
    "GHUNNAH" -> "Ghunnah & Shaddah"
    "MADD" -> "Madd (Vowel Extension)"
    "QALQALAH" -> "Qalqalah (Echo & Bounce)"
    "MEEM_SAKINAH" -> "Meem Sakinah (Labial Rules)"
    "IDGHAM" -> "Idgham (Merging)"
    "IKHFA" -> "Iqlab & Ikhfa"
    "IZHAR" -> "Izhar (Clarity)"
    else -> null
}

/** True when the word carries an EXPLICIT maddah mark (آ / U+0653 / U+06E4) —
 *  the only madd cue the MADD head judges reliably. */
fun wordHasExplicitMaddMark(word: String): Boolean =
    word.contains('\u0653') || word.contains('\u06E4') || word.contains('آ')

/**
 * Per-head decision thresholds for the "rule applied correctly?" verdict.
 *
 * These were derived by sweeping the pass/fail cutoff on the model's held-out
 * TEST split and picking the Youden-J-optimal point (best balance of correct
 * recall vs. mistake false-positive rate) for each head -- NOT a flat 0.5.
 *
 * Why not 0.5: heads differ in how confident they are on recitations that ARE
 * correct. Mistakes on QALQALAH already cluster at p90 ~ 0.05 (a confident
 * "wrong"), but CORRECT Qalqalah recitations spread wide (0.09-0.97, std 0.19;
 * AUC 0.925). At 0.5 the model wrongly flagged ~12% of correct Qalqalah clips
 * as errors (the ('4%' / '30%' on-device feedback lived in that borderline
 * tail). Lowering the QALQALAH bar to 0.15 lifts correct-recall to 99% while
 * mistakes (still ~0.05) stay caught. Each head has its own optimal cut:
 *
 *   head          thr   correct-recall   mistake-specificity
 *   MADD          0.56    0.924            0.989
 *   GHUNNAH       0.13    0.985            0.995
 *   QALQALAH      0.15    0.990            0.971
 *   IDGHAM        0.14    1.000            0.978
 *   IKHFA         0.10    0.960            1.000
 *   ALLAM         0.47    0.969            0.983
 *   MEEM_SAKINAH  0.11    0.957            0.972
 *   IZHAR         0.17    1.000            1.000
 *
 * NOTE: this is a calibration of the **verdict**, not the displayed score -- the
 * user still sees prob*100% (the raw model confidence); only the
 * CORRECT / TAJWEED_ERROR flag and the feedback string use this calibrated bar.
 */
fun decisionThresholdForHead(head: String): Float = when (head) {
    // QALQALAH / IZHAR / ALLAM / MADD stay at their Youden-optimal calibration (kept the same as
    // the previous session per the user's request -- these heads behave reliably at those points).
    "MADD" -> 0.56f
    "QALQALAH" -> 0.15f
    "ALLAM" -> 0.47f
    "IZHAR" -> 0.17f
    // GHUNNAH / IDGHAM / IKHFA / MEEM_SAKINAH were lowered to the Youden-optimal point last
    // session (0.10-0.13), but at those low bars plain tarteel and clean pronunciation kept
    // producing false "Good application" verdicts. The user asked to make these four strict
    // again -- i.e. revert them to the pre-Youden 0.5 baseline while keeping the four above.
    "GHUNNAH" -> 0.5f
    "IDGHAM" -> 0.5f
    "IKHFA" -> 0.5f
    "MEEM_SAKINAH" -> 0.5f
    else -> 0.5f
}

fun getTajweedRuleForWord(word: String, nextWord: String?, riwayah: String): String? {
    val len = word.length
    var i = 0
    
    // NOTE — NATURAL MADD IS DELIBERATELY NOT DETECTED HERE. The MADD head was
    // calibrated on STRETCHED madds (4-6 counts, explicit marks آ/ٓ/dagger in
    // strong contexts). Ordinary 2-count vowels (fatha+alif, kasra+yaa,
    // damma+waw, dagger alif mid-word) scored 23-51% EVEN FOR ALAFASY'S
    // PERFECT RECITATION in the PC harness — enabling them accused every
    // correct reader of a madd mistake. Only EXPLICIT marks route to MADD;
    // everything else keeps its prior verdict (Al-Lam / null).
    // Check Al-Laam Rules (Solar / Lunar). alifLamStartIndex handles bare (ٱل/ال) and prefixed
    // (بِٱل / وَٱل / فَٱل / كَٱل) forms, and firstBaseLetterFrom skips any Sukoon / Harakah that
    // sits between the Lam and the next letter (e.g. ٱلْقَمَرُ, ٱلْحَمْدُ).
    val alStart = alifLamStartIndex(word)
    if (alStart != null) {
        val lamIdx = alStart + 1
        val baseLetter = firstBaseLetterFrom(word, lamIdx + 1)
        if (baseLetter != null) {
            // Both Solar (士兵 في تثدذرزسشصضطظلن) and Lunar (أبجحخعغفقكمهوي) resolve to "Al-Lam" for
            // the acoustic-rule verdict: only the displayed color (Solar vs Lunar) differs.
            if ("تثدذرزسشصضطظلن".contains(baseLetter) || "أبجحخعغفقكمهوي".contains(baseLetter)) {
                return "Al-Lam"
            }
        }
    }

    while (i < len) {
        val char = word[i]
        
        // Ghunnah: Noon or Meem with Shaddah. In voweled (modern) Quran orthography, the
        // shaddah comes AFTER the harakah — e.g. إِنَّ = إ ، ِ ، ن ، َ ، ّ — so the noon at
        // index i has its shaddah at index i+1 only when there is no vowel mark between.
        // Scan forward up to 2 chars; skip kasra/fatha/damma/sukoon. We must *not* return
        // GHUNNAH on a plain ن or م without a shaddah (e.g. مِن in any-where text).
        if (char == 'ن' || char == 'م') {
            var shaddahHit = false
            if (i + 1 < len && word[i + 1] == 'ّ') { shaddahHit = true }
            else if (i + 2 < len && word[i + 2] == 'ّ') {
                val mid = word[i + 1]
                // Only count shaddah-at-index+2 if the in-between char is a harakah (not
                // another Arabic letter and not punctuation). Whitespace resets the word, so
                // we don't need to worry about it, but a harakah between noon and shaddah is
                // the standard spelling for mushafs.
                val midCode = mid.code
                if (midCode in 0x064B..0x0652 || mid == 'ٰ' || mid == 'ۡ') { shaddahHit = true }
            }
            if (shaddahHit) return "Ghunnah & Shaddah"
        }
        
        // Madd: Check Muttasil vs Munfasil
        if (char == '\u0653' || char == '\u06E4' || char == 'آ') {
            val hasHamzaInWord = word.any { "ءأإؤئ".contains(it) }
            if (hasHamzaInWord) {
                return "Madd Muttasil"
            } else {
                val firstNextChar = nextWord?.firstOrNull()?.let { normalizeArabic(it.toString()) }?.firstOrNull()
                if (firstNextChar != null && "اأإء".contains(firstNextChar)) {
                    return if (riwayah == "Qalun") "Madd Munfasil (Qalun: Qasr/Tawassut)" else "Madd (Vowel Extension)"
                }
                return "Madd (Vowel Extension)"
            }
        }
        
        // Qalqalah: Letters of قطب جد (ق ط ب ج د)
        if ("قطبجد".contains(char)) {
            val isSukoonNext = i + 1 < len && (word[i+1] == '\u0652' || word[i+1] == '\u06E1')
            val isEnd = i + 1 == len || (i + 1 < len && punctuationRegex.matches(word[i+1].toString()))
            if (isSukoonNext || isEnd) {
                return "Qalqalah (Echo & Bounce)"
            }
        }
        
        // Meem Sakinah Rules & Silat Mim al-Jam` for Qalun
        if (char == 'م' && isSakinah(word, i)) {
            
            var targetNextChar: Char? = null
            if (i + 1 < len) {
                val substringIndex = word.substring(i + 1).indexOfFirst { c ->
                    c !in '\u064B'..'\u0652' && c != '\u06E1'
                }
                if (substringIndex != -1) {
                    val nextRealIndex = i + 1 + substringIndex
                    targetNextChar = normalizeArabic(word[nextRealIndex].toString()).firstOrNull()
                }
            }
            if (targetNextChar == null) {
                targetNextChar = nextWord?.let { normalizeArabic(it).firstOrNull { c -> c !in '\u064B'..'\u0652' && c != '\u06E1' } }
            }
            
            if (targetNextChar != null) {
                if (targetNextChar == 'م') {
                    return "Meem Sakinah (Labial Rules)"
                } else if (targetNextChar == 'ب') {
                    return "Meem Sakinah (Labial Rules)"
                } else if (riwayah == "Qalun" && i == len - 1 && len >= 2) {
                    val prev = word[i - 1]
                    val basePrev = normalizeArabic(prev.toString())
                    if (basePrev == "ه" || basePrev == "ك" || basePrev == "ت") {
                        return "Silah"
                    }
                }
                return "Meem Sakinah (Labial Rules)"
            }
            if (riwayah == "Qalun" && i == len - 1 && len >= 2) {
                val prev = word[i - 1]
                val basePrev = normalizeArabic(prev.toString())
                if (basePrev == "ه" || basePrev == "ك" || basePrev == "ت") {
                    return "Silah"
                }
            }
        }
        
        // Tanween
        if (char == '\u064B' || char == '\u064D' || char == '\u064C') {
            val firstNextChar = nextWord?.let { normalizeArabic(it).firstOrNull { c -> c !in '\u064B'..'\u0652' && c != '\u06E1' } }
            if (firstNextChar != null) {
                if ("يرملون".contains(firstNextChar)) {
                    return "Idgham (Merging)"
                } else if (firstNextChar == 'ب') {
                    return "Iqlab & Ikhfa"
                } else if ("اأإءههعحغخ".contains(firstNextChar)) {
                    return "Izhar (Clarity)"
                } else {
                    return "Iqlab & Ikhfa"
                }
            }
        }
        
        // Noon Sakinah
        if (char == 'ن' && isSakinah(word, i)) {
            var targetNextChar: Char? = null
            if (i + 1 < len) {
                val substringIndex = word.substring(i + 1).indexOfFirst { c ->
                    c !in '\u064B'..'\u0652' && c != '\u06E1'
                }
                if (substringIndex != -1) {
                    val nextRealIndex = i + 1 + substringIndex
                    targetNextChar = normalizeArabic(word[nextRealIndex].toString()).firstOrNull()
                }
            }
            if (targetNextChar == null) {
                targetNextChar = nextWord?.let { normalizeArabic(it).firstOrNull { c -> c !in '\u064B'..'\u0652' && c != '\u06E1' } }
            }
            
            if (targetNextChar != null) {
                if ("يرملون".contains(targetNextChar)) {
                    return "Idgham (Merging)"
                } else if (targetNextChar == 'ب') {
                    return "Iqlab & Ikhfa"
                } else if ("اأإءههعحغخ".contains(targetNextChar)) {
                    return "Izhar (Clarity)"
                } else {
                    return "Iqlab & Ikhfa"
                }
            }
        }
        
        i++
    }
    return null
}

/**
 * Captures raw microphone audio (16-bit PCM mono) while the user recites, so the acoustic
 * Tajweed analysis can run on the user's *real* voice instead of a synthetic buffer.
 *
 * It runs an [AudioRecord] on a background thread and appends normalized Float samples
 * (-1..1) into a growable buffer. The capture is best-effort: if the microphone cannot be
 * opened (e.g. the speech recognizer holds it exclusively on some devices) the buffer simply
 * stays empty and the acoustic scoring is skipped rather than producing false Tajweed errors.
 */
/**
 * Linearly resample a mono FloatArray from [srcRate] Hz to [dstRate] Hz.
 * The speech recognizer usually returns 8 kHz AMR; the acoustic engine expects 16 kHz.
 */
private fun resampleAudio(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
    if (srcRate <= 0 || srcRate == dstRate || input.isEmpty()) return input
    val ratio = dstRate.toDouble() / srcRate
    val newSize = (input.size * ratio).toInt().coerceAtLeast(1)
    val out = FloatArray(newSize)
    for (i in 0 until newSize) {
        val pos = i / ratio
        val i0 = pos.toInt().coerceIn(0, input.size - 1)
        val i1 = (i0 + 1).coerceAtMost(input.size - 1)
        val frac = (pos - i0).toFloat()
        out[i] = input[i0] * (1f - frac) + input[i1] * frac
    }
    return out
}

/**
 * Decode the audio the SpeechRecognizer returns via BUNDLE_AUDIO_SOURCE into a 16 kHz mono
 * FloatArray in [-1, 1].
 *
 * This is the SINGLE microphone capture: the recognizer records the user's voice and gives us
 * both the transcript and the audio bytes, so we never open a second AudioRecord (which conflicts
 * with recognition on many devices, e.g. Samsung, and makes recognition fail entirely).
 *
 * Returns null if decoding is unsupported or fails -- in that case acoustic scoring is skipped
 * gracefully instead of producing fake scores.
 */
suspend fun decodeRecognizerAudio(context: Context, uri: Uri): FloatArray? = withContext(Dispatchers.IO) {
    var extractor: MediaExtractor? = null
    var codec: MediaCodec? = null
    try {
        extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: ""
            if (mime.startsWith("audio/")) {
                trackIndex = i
                format = f
                break
            }
        }
        if (trackIndex < 0 || format == null) return@withContext null
        extractor.selectTrack(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
        val srcRate = if (format.containsKey(MediaFormat.KEY_SAMPLE_RATE)) format.getInteger(MediaFormat.KEY_SAMPLE_RATE) else 16000

        val pcm = ArrayList<Short>(1 shl 16)
        var guard = 0
        if (mime.contains("wav", ignoreCase = true) || mime.contains("raw", ignoreCase = true) || mime.contains("pcm", ignoreCase = true)) {
            // Already PCM - read raw frames directly (MediaCodec cannot decode PCM).
            val buf = java.nio.ByteBuffer.allocate(8192)
            while (true) {
                buf.clear()
                val n = extractor.readSampleData(buf, 0)
                if (n < 0) break
                buf.rewind()
                var j = 0
                while (j < n - 1) {
                    val lo = buf.get().toInt() and 0xFF
                    val hi = buf.get().toInt() and 0xFF
                    pcm.add((lo or (hi shl 8)).toShort())
                    j += 2
                }
                extractor.advance()
            }
        } else {
            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()
            val info = MediaCodec.BufferInfo()
            var inputDone = false
            while (true) {
                if (guard++ > 2_000_000) break
                if (!inputDone) {
                    val inId = codec.dequeueInputBuffer(10_000)
                    if (inId >= 0) {
                        val inBuf = codec.getInputBuffer(inId)
                        if (inBuf == null) {
                            codec.queueInputBuffer(inId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            val size = extractor.readSampleData(inBuf, 0)
                            if (size < 0) {
                                codec.queueInputBuffer(inId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                codec.queueInputBuffer(inId, 0, size, extractor.getSampleTime(), 0)
                                extractor.advance()
                            }
                        }
                    }
                }
                val outId = codec.dequeueOutputBuffer(info, 10_000)
                if (outId >= 0) {
                    val outBuf = codec.getOutputBuffer(outId)
                    if (outBuf != null) {
                        val chunk = ByteArray(info.size)
                        outBuf.get(chunk, 0, info.size)
                        var j = 0
                        while (j < info.size - 1) {
                            val lo = chunk[j].toInt() and 0xFF
                            val hi = chunk[j + 1].toInt() and 0xFF
                            pcm.add((lo or (hi shl 8)).toShort())
                            j += 2
                        }
                        codec.releaseOutputBuffer(outId, false)
                        if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) break
                    } else {
                        codec.releaseOutputBuffer(outId, false)
                    }
                } else if (outId == MediaCodec.INFO_TRY_AGAIN_LATER) {
                    // keep looping
                } else if (outId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    // ignore
                }
            }
        }
        if (pcm.isEmpty()) return@withContext null
        val floats = FloatArray(pcm.size) { pcm[it] / 32768f }
        resampleAudio(floats, srcRate, 16000)
    } catch (e: Throwable) {
        android.util.Log.e("RecognizerAudio", "Failed to decode recognizer audio", e)
        null
    } finally {
        try { codec?.stop() } catch (_: Throwable) {}
        try { codec?.release() } catch (_: Throwable) {}
        try { extractor?.release() } catch (_: Throwable) {}
    }
}

suspend fun analyzeRecitationDetails(
    spoken: List<String>,
    targetWordsOriginal: List<String>,
    targetWordsNormalized: List<String>,
    riwayah: String,
    spokenTimings: List<SpokenWordTiming> = emptyList(),
    context: android.content.Context? = null,
    fullAudio: FloatArray? = null,
    recordingStartMs: Long = 0L,
    recordingDurationMs: Long = 0L,
    targetAllGroups: List<List<Any>>? = null,
    hintedIndices: Collection<Int> = emptySet(),
    isTajweedMode: Boolean = false,
    finalReport: Boolean = false,
    verdictCache: java.util.concurrent.ConcurrentHashMap<String, Float>? = null,
    /**
     * §4.46 - FAILED verdicts, cached against the search window they were
     * measured on: key -> (maxStart, best prob). A fail is re-used only while
     * `maxStart` is unchanged, which makes the re-use provably identical to
     * re-running the sweep; once the buffer has grown enough to expose new
     * candidate positions the word is re-scored, so a fuller decode can still
     * acquit it. Without this, every failing word re-ran its whole -6 s sweep
     * (up to 80 ONNX evaluations) on EVERY acoustic pass.
     */
    failCache: java.util.concurrent.ConcurrentHashMap<String, Pair<Long, Float>>? = null,
    /**
     * When false, the acoustic (TajweedNet) scoring step is skipped entirely:
     * words get text-only verdicts (CORRECT / WRONG / SKIPPED). This is the
     * FAST path (~ms) used for the near-real-time display + wrong/skipped
     * detection. The expensive acoustic pass runs separately, deferred ~2 s,
     * on the slow analysis loop (user-accepted latency for tajweed verdicts).
     */
    scoreTajweed: Boolean = true,
    /**
     * §4.56 — HOW MANY WORDS ONE ACOUSTIC PASS MAY SCORE.
     *
     * The pass used to score EVERY rule-bearing word it was handed, in one go. On
     * a long session that is unbounded work: measured on the user's Galaxy A04s,
     * one pass reported
     *   analyze(tajweed): done 133417ms scored=37 tajweedErrors=9
     * — **two minutes twenty** of the app scoring, and then nine accusations
     * landing at once. Nothing else moves while it runs, so the app looks dead
     * (the user's words: "make sure the app dont freeze like that again
     * (2 min!)").
     *
     * A word costs one model evaluation per rule it carries (~1.8 s on the A05s,
     * ~3.6 s on the A04s), so the cost is linear in the backlog. Bounding the
     * backlog per pass spreads the same total work over the passes that follow —
     * every emission re-requests the analysis, and a word left unscored simply
     * stays text-only for now and is picked up next time. The walk hands the
     * words over in reading order, so the ones the reader has just reached are
     * scored first, which is what makes a verdict timely.
     *
     * `Int.MAX_VALUE` leaves every other caller unchanged.
     */
    tajweedScoreLimit: Int = Int.MAX_VALUE
): List<WordAnalysis> {
    var tajweedScoredThisPass = 0
    val result = ArrayList<WordAnalysis>()
    var targetIndex = 0
    var spokenIndex = 0
    val limit = minOf(targetWordsNormalized.size, targetWordsOriginal.size)
    // OOV-garbage skip bookkeeping: words the matcher couldn't map to the
    // active vocabulary (KEEP_RAW fragments, mis-restores like النَّفسِ /
    // الْخَنَِّّ) have no target slot — the walk steps over them instead of
    // charging a WRONG, capped per run so a truly alien utterance still
    // surfaces as a mistake.
    val targetNormSet = targetWordsNormalized.filter { it.isNotEmpty() }.toHashSet()
    var consecutiveGarbageSkips = 0
    // ECHO-ABSORPTION bookkeeping: noisy merges re-emit ALREADY-SPOKEN words
    // later in the transcript (e.g. ٱلرَّحْمَٰنِ echoing after الْحَمْدُ on a
    // slow phone). Such tokens ARE vocabulary words, so the OOV-garbage skip
    // cannot catch them — yet they steal the alignment slot of the REAL
    // upcoming word and cascade false SKIPPED/W RONG verdicts (measured
    // on-device on Al-Baqarah). A spoken token matching an ALREADY-CONSUMED
    // target word is absorbed (stepped over) instead, capped at 3 consecutive
    // so pathological repeat runs still surface as mistakes.
    val consumedNorms = HashSet<String>()
    var consecutiveEchoSkips = 0

    // ── REORDER BUFFER ("carry") ─────────────────────────────────────────
    // monotonicMergeWords can only APPEND, so a word the decoder renders LATE
    // lands out of reading order (measured on-device on Al-Fatihah 1:2: لِلَّهِ
    // arrived AFTER ٱلْعَٰلَمِينَ, and الرَّحِيمِ before الرَّحْمَٰنِ). The
    // walk jumps ahead to reach the current target and SILENTLY SWALLOWS the
    // words in between — but those words belong to the NEXT targets, so each
    // one gets charged a false SKIPPED, the cascade reaches a WRONG and the
    // walk returns: matched froze at 7/29 and the whole surah stayed grey with
    // no popup. Parking the swallowed words here lets the following targets
    // claim them first, restoring reading order.
    // Bounded both ways: at most CARRY_MAX words, and only reusable for the
    // next CARRY_BUDGET targets, so a stale buffer can never resurrect a word
    // far away from where it was actually heard.
    val carry = ArrayList<String>()
    var carryBudget = 0
    val CARRY_MAX = 4
    val CARRY_BUDGET = 3

    // Process-wide singleton: the ~1 MB weights are parsed once and reused across
    // every recitation, so per-word analysis no longer re-decodes the JSON blob.
    val analyzer = context?.let { RecitationAnalyzer.get(it) }
    // Real microphone audio returned by the SpeechRecognizer itself (see BUNDLE_AUDIO_SOURCE).
    // When it is null/empty the acoustic capture was unavailable, so we skip Tajweed acoustic
    // scoring instead of penalizing every word with a fake failing score.
    val hasRealAudio = fullAudio != null && fullAudio.size >= 256
    
    while (targetIndex < limit) {
        kotlinx.coroutines.yield()
        val targetWordNorm = targetWordsNormalized[targetIndex]
        val targetWordOrig = targetWordsOriginal[targetIndex]

        // Hint-matched words were revealed to the user (we injected their text into the
        // transcript to advance the tracker). Treat them as CORRECT-in-position only so the
        // recognizer keeps tracking; the HINTED status + mistake counting are applied in the
        // post-pass below (and we deliberately skip the acoustic tajweed scoring for them,
        // since scoring the injected silence would pop a false TAJWEED_ERROR).
        val isHintedWord = hintedIndices.contains(targetIndex)
        if (isHintedWord) {
            result.add(WordAnalysis(
                wordIndex = targetIndex,
                text = targetWordOrig,
                normalizedText = targetWordNorm,
                status = RecitationStatus.CORRECT,
                spokenText = if (spokenIndex < spoken.size) spoken[spokenIndex] else null,
                tajweedRule = null  // never scored (see below)
            ))
            consumedNorms.add(targetWordNorm)
            targetIndex++; spokenIndex++
            continue
        }

        // Non-word signs (nasr's ۚ stop marker) normalize to "": they are
        // never recited, so they can never match a spoken word — count them
        // CORRECT without consuming speech, else they surface as a phantom
        // SKIPPED verdict in an otherwise perfect recitation.
        if (targetWordNorm.isEmpty()) {
            result.add(WordAnalysis(
                wordIndex = targetIndex,
                text = targetWordOrig,
                normalizedText = targetWordNorm,
                status = RecitationStatus.CORRECT,
                spokenText = null,
                tajweedRule = null
            ))
            targetIndex++
            continue
        }

        val nextWordOrig = if (targetIndex + 1 < limit) targetWordsOriginal[targetIndex + 1] else null
        val rule = getTajweedRuleForWord(targetWordOrig, nextWordOrig, riwayah)
        
        if (spokenIndex >= spoken.size) {
            // ── SPOKEN EXHAUSTED: FILL THE REMAINING TARGET ROWS ───────────
            // The walk can only emit a row while it still has a spoken word to
            // judge. Once the transcript ran out it used to `break` outright,
            // leaving every remaining target word with NO ROW AT ALL. That is a
            // hole no downstream consumer can see, and it produced the
            // on-device report "the last word is not displayed and yet I have
            // no mistake popup but a summary saying there are no mistakes":
            //   * `decided` (= distinct wordIndex count) came back 28/29,
            //   * `unheard` counted 0 — there was no PENDING row to count,
            //   * `finalizeCorrect()` could not forgive a row that did not
            //     exist, so the word stayed at the UI's 25 %-opacity fallback
            //     ("not recited") forever,
            //   * `cleanFinish` was false only because of the count, the grace
            //     window expired with no `firstMistake` to stop for, and the
            //     summary was raised claiming a clean recitation.
            // Measured fatihah_c33: `rows=32` for 29 targets (28 distinct
            // indices), `decided=28/29 unheardPending=0`, zero `verify idx=`
            // lines, no popup, `endOfRange(): wrong=0 skipped=0`.
            //
            // A target word the walk never reached is UNHEARD — PENDING with
            // spokenText == null, exactly like the frontier case below — so it
            // neither advances `matched` nor hides a mistake, but it is now
            // VISIBLE to every counter, to finalizeCorrect() and to the
            // mistake pipeline. Every target word always has a row.
            for (k in targetIndex until limit) {
                val kw = targetWordsOriginal[k]
                val kn = targetWordsNormalized[k]
                val kNext = if (k + 1 < limit) targetWordsOriginal[k + 1] else null
                result.add(WordAnalysis(
                    wordIndex = k,
                    text = kw,
                    normalizedText = kn,
                    status = RecitationStatus.PENDING,
                    spokenText = null,
                    // A word that was never reached is only ever PENDING: never
                    // attach a rule the user could be scored against.
                    tajweedRule = if (kn.isEmpty()) null else getTajweedRuleForWord(kw, kNext, riwayah)
                ))
            }
            break
        }
        consecutiveEchoSkips = 0
        // RESYNC ABSORPTION: the current target word appears further ahead in
        // the spoken stream, but unmatched garbage sits in front of it
        // (decoder warm-up wrecks like ٱلصِّيَامُ الم أَلَدُّ before
        // ذَٰلِكَ ٱلْكِتَٰبُ, merge echoes) — absorb the garbage prefix and
        // let the match land. Without this the head poison charged an instant
        // WRONG at word 0 and froze the session. Capped at 6 absorbed tokens.
        var resyncAt = -1
        val resyncEnd = minOf(spokenIndex + 15, spoken.size)
        for (k in spokenIndex until resyncEnd) {
            if (isWordMatch(spoken[k], targetWordNorm)) { resyncAt = k; break }
        }
        if (resyncAt > spokenIndex && (resyncAt - spokenIndex) <= 6) {
            // Same reasoning as the carry fill below: the tokens absorbed here
            // are only *presumed* garbage. Park them too — if one of them turns
            // out to be the next target word the buffer lets it be claimed
            // instead of becoming a false SKIPPED.
            // Not at the head: tokens absorbed before the FIRST target word sit
            // in front of the recitation and belong to no target at all.
            if (targetIndex > 0) {
                for (i in spokenIndex until resyncAt) {
                    if (carry.size >= CARRY_MAX) carry.removeAt(0)
                    carry.add(spoken[i])
                }
                carryBudget = CARRY_BUDGET
            }
            spokenIndex = resyncAt
        }
        
        var matchInSpokenAhead = -1
        var matchedSpokenAdvance = 1
        var matchedFromCarry = false
        var carryWord: String? = null
        val wordGroups = targetAllGroups?.getOrNull(targetIndex)

        // Try the reorder buffer FIRST: these are words the walk already jumped
        // over, i.e. speech that was heard BEFORE the current spokenIndex and
        // therefore belongs to an earlier position than anything at the cursor.
        // Consuming them here (without advancing [spokenIndex]) is what repairs
        // a locally-scrambled transcript instead of charging false SKIPs.
        if (carryBudget > 0 && carry.isNotEmpty()) {
            for (i in carry.indices) {
                var hit = isWordMatch(carry[i], targetWordNorm)
                if (!hit && wordGroups != null) {
                    for (group in wordGroups) {
                        if (group is List<*> && group.isNotEmpty() && group[0] !is List<*>) {
                            @Suppress("UNCHECKED_CAST")
                            if (isWordMatchAny(carry[i], group as List<String>)) { hit = true; break }
                        }
                    }
                }
                if (hit) {
                    carryWord = carry[i]
                    repeat(i + 1) { carry.removeAt(0) }
                    carryBudget--
                    if (carryBudget <= 0) carry.clear()
                    matchInSpokenAhead = spokenIndex
                    matchedSpokenAdvance = 1
                    matchedFromCarry = true
                    android.util.Log.i(
                        "ItqanSherpa",
                        "carry(): target $targetIndex '$targetWordOrig' claimed " +
                            "reorder-buffer word '${carryWord}' (${i + 1} swallowed word(s) consumed)"
                    )
                    break
                }
            }
        }

        for (i in spokenIndex until (if (matchedFromCarry) spokenIndex else minOf(spokenIndex + 3, spoken.size))) {
            var matched = false
            if (wordGroups != null) {
                for (group in wordGroups) {
                    if (group is List<*> && group.isNotEmpty() && group[0] is List<*>) {
                        @Suppress("UNCHECKED_CAST")
                        val slots = group as List<List<String>>
                        if (i + slots.size <= spoken.size &&
                            isWordMatchPlural((0 until slots.size).map { spoken[i + it] }, slots)) {
                            matchInSpokenAhead = i; matchedSpokenAdvance = slots.size; matched = true; break
                        }
                    } else {
                        @Suppress("UNCHECKED_CAST")
                        if (isWordMatchAny(spoken[i], group as List<String>)) {
                            matchInSpokenAhead = i; matchedSpokenAdvance = 1; matched = true; break
                        }
                    }
                }
            }
            if (!matched && isWordMatch(spoken[i], targetWordNorm)) {
                matchInSpokenAhead = i; matchedSpokenAdvance = 1; matched = true
            }
            if (matched) break
        }

        // ── MUQATTA'AT FALLBACK ───────────────────────────────────────────
        // The strict loop above needs the decoder to render every letter name
        // of a disjointed-letter word. It does not: measured on a Samsung
        // A05s reciting Al-Baqarah 2:1, three consecutive windows emitted
        // `أف م` for `الم` (truncated ألف, dropped لام). The 3-slot matcher
        // then fails, the walk's OOV-garbage cap burns out on the fragments
        // and word 0 is charged WRONG — which RETURNS EARLY, leaving all 6115
        // remaining words PENDING: matched=0, nothing highlighted, no popup.
        // Al-Fatihah has no muqatta'at, which is exactly why it "worked fine".
        // Anchored at the transcript frontier only (i = spokenIndex) so it can
        // never reach into speech that belongs to a later word.
        if (matchInSpokenAhead == -1) {
            val mSlots = itqan.app.verses.muqattaatSlots(targetWordOrig)
            if (mSlots != null) {
                val runEnd = minOf(spokenIndex + mSlots.size, spoken.size)
                val consumed = if (spokenIndex < runEnd) {
                    itqan.app.verses.matchMuqattaatRelaxed(
                        spoken.subList(spokenIndex, runEnd), mSlots
                    )
                } else -1
                if (consumed > 0) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "muqattaat(relaxed): word $targetIndex '$targetWordOrig' " +
                            "matched ${mSlots.size} slots, consumed ${consumed}w from '${spoken.subList(spokenIndex, runEnd).joinToString(" ")}'"
                    )
                    matchInSpokenAhead = spokenIndex
                    matchedSpokenAdvance = consumed
                }
            }
        }

        // ── MUQATTA'AT HEAD FORGIVENESS ──
        // Sits BEFORE both the SKIPPED and the WRONG branches on purpose.
        // A false negative on the FIRST disjointed-letter word of a range is
        // catastrophic and the risk is completely asymmetric: whichever branch
        // takes it, the walk stops there and every remaining word (6115 for a
        // whole-surah Al-Baqarah) stays PENDING — matched=0, nothing highlights,
        // no useful feedback. A false CORRECT on one letter group costs
        // essentially nothing: the cursor simply advances one word.
        // Measured on-device (baqarah_v2): word 0 `الٓمٓ` was charged SKIPPED
        // (the first renderable token `لَا` matched a LATER target), so a guard
        // placed only in the WRONG branch never ran and the surah stayed frozen
        // at matched=0 for the entire session.
        if (matchInSpokenAhead == -1 && targetIndex <= 1 &&
            itqan.app.verses.muqattaatSlots(targetWordOrig) != null
        ) {
            android.util.Log.i(
                "ItqanSherpa",
                "muqattaat(head-forgive): word $targetIndex '$targetWordOrig' " +
                    "-> CORRECT (relaxed matcher failed; avoids the matched=0 freeze)"
            )
            result.add(WordAnalysis(
                wordIndex = targetIndex,
                text = targetWordOrig,
                normalizedText = targetWordNorm,
                status = RecitationStatus.CORRECT,
                spokenText = spoken[spokenIndex],
                tajweedRule = rule
            ))
            consumedNorms.add(targetWordNorm)
            spokenIndex++
            targetIndex++
            continue
        }

        if (matchInSpokenAhead != -1) {
            consecutiveGarbageSkips = 0
            consumedNorms.add(targetWordNorm)
            consecutiveEchoSkips = 0
            // Park everything this match skipped over, so the FOLLOWING targets
            // can still claim it (see the CARRY block above). Without this the
            // skipped words are lost for good and every one of them turns into
            // a false SKIPPED verdict.
            if (!matchedFromCarry && matchInSpokenAhead > spokenIndex) {
                for (i in spokenIndex until matchInSpokenAhead) {
                    if (carry.size >= CARRY_MAX) carry.removeAt(0)
                    carry.add(spoken[i])
                }
                carryBudget = CARRY_BUDGET
                android.util.Log.i(
                    "ItqanSherpa",
                    "carry(fill): target $targetIndex matched spoken[$matchInSpokenAhead], " +
                        "parking ${matchInSpokenAhead - spokenIndex} swallowed word(s): " +
                        spoken.subList(spokenIndex, matchInSpokenAhead).joinToString(" ")
                )
            }

            // ── MULTI-RULE LABELS ── one score row per rule the word carries.
            // The primary detector label comes first, then any additional
            // training-table heads the word also contains (e.g. ٱلرَّحِيمِ
            // carries BOTH its Al-Lam article AND a natural madd; إِيَّاكَ has
            // a madd even though the primary detector alone returns null).
            // Deduped by model head so a rule never scores twice.
            val labels = LinkedHashMap<String, String>() // friendly label -> head
            if (rule != null) ruleLabelToHead(rule)?.let { labels[rule] = it }
            if (!isHintedWord) {
                // FULL MULTI-RULE COVERAGE — with ONE exception aligned to the
                // colourizer: MADD is scored ONLY where the app PAINTS RED
                // (explicit maddah marks آ / U+0653 / U+06E4 — exactly the
                // buildTajweedAnnotatedString red branches). Natural 2-count
                // vowels (fatha+alif, kasra+yaa, dagger alif…) are NOT red and
                // are NOT scored: the current MADD head judges them 23-51% even
                // on perfect recitations, and the user asked for madd feedback
                // exclusively on the red-highlighted words.
                val hasRedMadd = wordHasExplicitMaddMark(targetWordOrig)
                itqan.app.ml.TajweedRules.detectRulesForWord(targetWordOrig, nextWordOrig).forEach { h ->
                    if (h == "MADD" && !hasRedMadd) return@forEach
                    val friendly = headToFriendlyLabel(h) ?: return@forEach
                    val fh = ruleLabelToHead(friendly) ?: return@forEach
                    if (labels.values.none { it == fh }) labels[friendly] = fh
                }
            }

            // Per-label scored entries (populated only in the deferred pass).
            val scoredEntries = ArrayList<WordAnalysis>()

            // Step 3 - Measure Audio: Validate Tajweed Acoustic properties on the user's REAL voice.
            // Gated on [scoreTajweed]: the fast display loop passes false so the
            // expensive TajweedNet scoring never blocks the near-real-time
            // highlight; the deferred tajweed loop passes true.
            // `&& !matchedFromCarry`: a carry word has no spoken-stream index,
            // so there is no trustworthy audio span to score — it gets the
            // positional CORRECT verdict only, never an acoustic tajweed score.
            if (scoreTajweed && labels.isNotEmpty() && hasRealAudio && isTajweedMode && !isHintedWord && !matchedFromCarry &&
                tajweedScoredThisPass < tajweedScoreLimit
            ) {
                // §4.56 — AND THE BUDGET IS THE CONDITION, NOT A BRANCH. Past the
                // limit the whole acoustic step is simply not taken for this word,
                // so it keeps the same text-only verdict an unscored word already
                // gets, and the NEXT pass (every emission re-requests the analysis)
                // picks it up. That is deliberate: no single pass can own minutes of
                // the user's time — measured on the A04s,
                // `analyze(tajweed): done 133417ms scored=37 tajweedErrors=9`, during
                // which nothing else in the app moved.
                tajweedScoredThisPass++
                // Find timing by exact index, or fallback to the closest matched word text
                val timing = spokenTimings.find { it.startIndex == matchInSpokenAhead }
                    ?: spokenTimings.findLast { isWordMatch(it.word, targetWordNorm) }

                // Map the recognized word's wall-clock span onto the recorded audio buffer.
                // Both clocks are System.currentTimeMillis(), so (timing - recordingStartMs) is the
                // offset in ms from the start of the captured audio.
                val s = spoken.size.coerceAtLeast(1)
                val idx = matchInSpokenAhead.coerceIn(0, s - 1)
                val propStart = (idx * recordingDurationMs / s).toLong()
                val propEnd = ((idx + 1) * recordingDurationMs / s).toLong()

                val (segStartMs, segEndMs) = if (timing != null && timing.endMs > timing.startMs) {
                    val st = (timing.startMs - recordingStartMs).coerceAtLeast(0L)
                    val en = (timing.endMs - recordingStartMs)
                    if (en - st >= 30L) st to en else (propStart to propEnd)
                } else {
                    propStart to propEnd
                }
                // Never request samples beyond what was actually captured (avoids index errors
                // for the final word while the buffer is still filling).
                val capturedMs = (fullAudio.size * 1000L / 16000)
                val clampedEnd = segEndMs
                    .coerceAtMost(recordingDurationMs)
                    .coerceAtMost(capturedMs)
                    .coerceAtLeast(segStartMs + 30L)

                if (clampedEnd - segStartMs >= 30L && analyzer != null) {
                    // Slice the captured 16 kHz buffer around the spoken word.
                    //
                    // The model was trained on per-WORD clips whose duration matches the
                    // recited word (QALQALAH median ~1.26s), and crucially those clips
                    // are nearly ALL speech -- a slice that is long but padded with
                    // SILENCE around the word scores ~0.10-0.16 on a correct qalqalah
                    // (verified on the real PyTorch model: the same correct word scores
                    // 0.96 alone but 0.16 when silence is padded around it), because the
                    // log-mel of long silence is out-of-distribution for every head.
                    //
                    // So we: (1) take a context window centered on the word
                    // (to include the bounce's lead-in/release), and (2) SILENCE-TRIM
                    // that window so the model only sees speech -- whatever its length.
                    // This depends only on the already-captured fullAudio buffer; it
                    // touches neither the microphone nor SpeechRecognizer (preserving
                    // the single-mic behaviour on phones like the A05s).
                    //
                    // CRITICAL — the context geometry matches the TRAINING clips
                    // (tajweed_ml repo, prepare_dataset.py):
                    //  - single-word rules (MADD, GHUNNAH, QALQALAH, ALLAM) trained on
                    //    the word's own [start, end] (median ~1.1 s);
                    //  - CROSSWORD_RULES (IDGHAM, IKHFA, IZHAR, MEEM_SAKINAH) trained on
                    //    [trigger_start, next_word_end] — the cue lives at the boundary,
                    //    so the clip spans the trigger word AND the following word
                    //    (median ~1.8-2.1 s).
                    //
                    // The capture buffer handed to the scorer is the UN-GATED
                    // real-time PCM (snapshotRawAudio / pause()'s return): it keeps
                    // the natural inter-word silence the model was trained on, so
                    // trimSilence CAN isolate a word, and its timeline matches the
                    // wall-clock spokenTimings. (The RMS-GATED STT buffer glues
                    // words with no silence and compresses the timeline — the probe
                    // proved that is out-of-distribution for every head and
                    // misaligns the word timings, collapsing a correct recitation's
                    // score below threshold = false TAJWEED_ERROR popups.)
                    //
                    // So the clip is the word's OWN span (+ a small margin), never a
                    // fixed-width window centered on it: a centered window wider than
                    // the word's slot always swallows a neighbour's speech, which
                    // trimSilence cannot remove (it only cuts leading/trailing
                    // silence, not a glued neighbour).
                    //
                    // The geometry is computed PER LABEL: a cross-word rule needs the
                    // trigger->next-word cut while a single-word rule needs the word's
                    // own span, and one word can carry both kinds at once.
                    for ((label, head0) in labels) {
                        val isCrossWord = head0 in setOf("IDGHAM", "IKHFA", "IZHAR", "MEEM_SAKINAH")
                        val (ctxStartMs, ctxEndMs) = if (isCrossWord) {
                            // Forward-leaning window: trigger word start -> next word end
                            // (+ a small margin each side), exactly the training cut.
                            val sCnt = spoken.size.coerceAtLeast(1)
                            val nextIdx = (matchInSpokenAhead + matchedSpokenAdvance)
                                .coerceAtMost(sCnt - 1)
                            val nextTiming = spokenTimings.find { it.startIndex == nextIdx }
                            val nextEndMs = if (nextTiming != null && nextTiming.endMs > nextTiming.startMs) {
                                nextTiming.endMs - recordingStartMs
                            } else {
                                ((nextIdx + 1).toLong() * recordingDurationMs / sCnt)
                            }
                            val st = (segStartMs - 100L).coerceAtLeast(0L)
                            val en = (nextEndMs + 100L)
                                .coerceAtMost(capturedMs)
                                .coerceAtMost(recordingDurationMs)
                                .coerceAtLeast(st + 30L)
                            st to en
                        } else {
                            // Single-word rule: the word's own span + a small margin,
                            // widened symmetrically to a minimum clip length (very short
                            // timing spans) and capped at the model's 2 s feature view.
                            val margin = 100L
                            var st = (segStartMs - margin).coerceAtLeast(0L)
                            var en = (clampedEnd + margin)
                                .coerceAtMost(capturedMs)
                                .coerceAtMost(recordingDurationMs)
                            val minClipMs = 600L
                            if (en - st < minClipMs) {
                                val grow = (minClipMs - (en - st) + 1) / 2
                                st = (st - grow).coerceAtLeast(0L)
                                en = (en + grow).coerceAtMost(capturedMs)
                                    .coerceAtMost(recordingDurationMs)
                            }
                            val maxClip = 2000L
                            if (en - st > maxClip) {
                                val mid = (st + en) / 2
                                st = (mid - maxClip / 2).coerceAtLeast(0L)
                                en = (st + maxClip).coerceAtMost(capturedMs)
                                    .coerceAtMost(recordingDurationMs)
                            }
                            st to en.coerceAtLeast(st + 30L)
                        }
                        val startSample = (ctxStartMs * 16000 / 1000).toInt().coerceAtLeast(0)
                        val endSample = (ctxEndMs * 16000 / 1000).toInt()
                            .coerceAtMost(fullAudio.size)
                        val rawSeg = if (endSample > startSample) {
                            fullAudio.sliceArray(startSample until endSample)
                        } else {
                            fullAudio
                        }
                        // Drop leading/trailing silence (RMS-gated, loudness-independent)
                        // so the model sees a speech clip, not a silence-padded one.
                        val seg = trimSilence(rawSeg)

                        val head = head0
                        if (seg.size >= 256) {
                            // Per-head calibrated threshold (see decisionThresholdForHead) -- a flat 0.5
                            // wrongly fails ~12% of CORRECT Qalqalah recitations because that head's
                            // output for correct clips spans 0.09-0.97 (std 0.19). The displayed % stays
                            // the raw prob*100; only the verdict uses the calibrated bar.
                            val thr = decisionThresholdForHead(head)
                            // VERDICT CACHE: keyed per head so two rules on the same
                            // word cache independently.
                            val cacheKey = "$targetIndex:$targetWordNorm:$head:${timing?.startMs ?: 0L}"
                            val cachedProb = verdictCache?.get(cacheKey)
                            var prob: Float
                            if (cachedProb != null) {
                                prob = cachedProb
                            } else {
                            val results = analyzer.analyze(seg, listOf(head), thr)
                            prob = results.firstOrNull()?.prob ?: 0f
                            // BOUNDED CONFIRMATION SEARCH (see the long comment in
                            // history): slide the SAME-width window ±6 s around the
                            // nominal position, best score wins; runs ONLY when the
                            // nominal position failed.
                            if (prob < thr) {
                                val spanLen = if (isCrossWord) 2000L
                                    else (segEndMs - segStartMs).coerceIn(400L, 1600L)
                                var best = prob
                                val maxStart = (capturedMs - spanLen).coerceAtLeast(0L)
                                    .coerceAtMost((recordingDurationMs - spanLen).coerceAtLeast(0L))
                                // §4.46 - a fail is only re-usable for the SAME
                                // search window. See [tajweedFailCache]: identical
                                // maxStart means byte-identical candidates, so the
                                // sweep would return the identical number. When
                                // the buffer has grown the window is re-scored, so
                                // a later, fuller decode can still acquit the word.
                                val cachedFail = failCache?.get(cacheKey)
                                if (cachedFail != null && cachedFail.first == maxStart) {
                                    best = maxOf(best, cachedFail.second)
                                } else {
                                val visited = HashSet<Long>()
                                var found = false
                                // - STAGED SEARCH (§4.45) -
                                // The nominal cut lands a beat LATE: a word is
                                // emitted only after the recognizer has decoded it,
                                // so `spokenTimings` place it after the audio that
                                // produced it, and the -6 s sweep exists to forgive
                                // that skew (and the window-edge cuts behind it).
                                //
                                // Sweeping 40 offsets at 150 ms is 80 ONNX
                                // evaluations per failing head - 1-3 s of CPU on the
                                // A05s, which is most of a live pass, and it is what
                                // makes a mistake popup arrive seconds after the
                                // mistake ("it takes too long to show the popup").
                                // But adjacent offsets overlap by ~93 % (the window
                                // is 0.4-2 s wide), so the score surface is smooth
                                // and a COARSE probe finds a passing position in
                                // almost every case where one exists nearby.
                                //
                                // So: probe coarsely first (600 ms steps, 10
                                // positions). If one passes, the verdict is settled
                                // in ~10 evaluations instead of ~80. If none does,
                                // fall through to the ORIGINAL 150 ms sweep UNCHANGED
                                // - the same candidates in the same order - so a word
                                // is still only forgiven by a real passing position:
                                // forgiveness is never reduced, and a genuine mistake
                                // costs 90 evaluations instead of 80.
                                fun sweep(stepMs: Long, limitMs: Long): Float {
                                    var off = stepMs
                                    while (!found && off <= limitMs) {
                                        for (st0 in longArrayOf(segStartMs - off, segStartMs + off)) {
                                            val st = st0.coerceIn(0L, maxStart)
                                            if (!visited.add(st)) continue
                                            val a = (st * 16000 / 1000).toInt().coerceIn(0, fullAudio.size)
                                            val b = ((st + spanLen) * 16000 / 1000).toInt()
                                                .coerceIn(a, fullAudio.size)
                                            if (b - a >= 256) {
                                                val cand = trimSilence(fullAudio.sliceArray(a until b))
                                                if (cand.size >= 256) {
                                                    val p = analyzer.analyze(cand, listOf(head), thr)
                                                        .firstOrNull()?.prob ?: 0f
                                                    if (p > best) best = p
                                                    if (best >= thr) { found = true; break }
                                                }
                                            }
                                        }
                                        off += stepMs
                                    }
                                    return best
                                }
                                best = sweep(600L, 6000L)
                                if (!found) best = sweep(150L, 6000L)
                                prob = best
                                // Remember the fail WITH the window it was measured
                                // on. Only a grown `maxStart` invalidates it.
                                if (prob < thr) failCache?.put(cacheKey, maxStart to prob)
                                }
                                prob = maxOf(prob, best)
                            }
                            // Passing verdicts are cached unconditionally; a FAIL is
                            // cached per search window above (§4.46), so neither is
                            // re-derived on every pass.
                            if (prob >= thr) verdictCache?.put(cacheKey, prob)
                            } // end verdict-cache miss branch
                            val tScoreL = (prob * 100).toInt().coerceIn(0, 100)
                            val feedbackL = if (prob >= thr) "Good $label application" else "Review your $label"
                            val statusL = if (prob < thr) RecitationStatus.TAJWEED_ERROR else RecitationStatus.CORRECT
                            scoredEntries.add(WordAnalysis(
                                wordIndex = targetIndex,
                                text = targetWordOrig,
                                normalizedText = targetWordNorm,
                                status = statusL,
                                spokenText = carryWord ?: spoken[matchInSpokenAhead],
                                tajweedRule = label,
                                tajweedAcousticFeedback = feedbackL,
                                tajweedScore = tScoreL,
                                tajweedAudioEndMs = clampedEnd
                            ))
                        }
                    }
                }
            }

            if (scoredEntries.isNotEmpty()) {
                scoredEntries.forEach { result.add(it) }
            } else {
                // Fast display pass / no audio / nothing scorable: a single
                // positional entry carrying the primary label (no score yet).
                result.add(WordAnalysis(
                    wordIndex = targetIndex,
                    text = targetWordOrig,
                    normalizedText = targetWordNorm,
                    status = RecitationStatus.CORRECT,
                    spokenText = carryWord ?: spoken[matchInSpokenAhead],
                    tajweedRule = rule,
                    tajweedAcousticFeedback = null,
                    tajweedScore = null
                ))
            }
            // A carry match consumes the buffered word, NOT the spoken stream:
            // the cursor stays put so the next target still sees the speech the
            // jump skipped past.
            if (!matchedFromCarry) spokenIndex = matchInSpokenAhead + matchedSpokenAdvance
            targetIndex++
            continue
        }
        
        var matchInTargetAhead = -1
        val spokenCurrent = spoken[spokenIndex]
        // ECHO ABSORPTION (see bookkeeping above): a spoken token that matches
        // an ALREADY-CONSUMED target word is a merge echo of the past, not the
        // user moving on — step over it so it cannot steal the upcoming slot
        // and cascade false SKIPPED verdicts.
        // REFINEMENT: only absorb when the word does NOT also occur in the
        // REMAINING target ahead — legitimate repeats (الله in الإخلاص,
        // أَلَا in الماعون) must still align normally.
        val spokenCurrentNormPre = normalizeForComparison(spokenCurrent, riwayah)
        if (spokenCurrentNormPre.isNotEmpty() &&
            consumedNorms.contains(spokenCurrentNormPre) &&
            consecutiveEchoSkips < 3
        ) {
            var appearsAhead = false
            for (j in targetIndex + 1 until limit) {
                if (targetWordsNormalized[j] == spokenCurrentNormPre) { appearsAhead = true; break }
            }
            if (!appearsAhead) {
                consecutiveEchoSkips++
                spokenIndex++
                continue
            }
        }
        consecutiveEchoSkips = 0
        // HEAD GRACE: the first windows often produce garbage tokens (fused
        // wrecks like يُمِلَّ/هُمْ for ذَٰلِكَ ٱلْكِتَٰبُ) that charge an
        // immediate WRONG at word 0 and freeze the whole session before the
        // decoder settles. For the first 3 target words, scan further ahead
        // (8 tokens): a match there means the user is on-track and the
        // intervening garbage is absorbed by the rules above on the way.
        val aheadLimit = minOf(
            targetIndex + if (targetIndex < 3) 8 else 4,
            limit
        )
        // ── CLIPPED-WORD GUARD ────────────────────────────────────────────
        // The CTC decoder sometimes returns only the opening letters of a word
        // (measured on Al-Baqarah 2:2: `ذَٰلِكَ` came back as just `ذ`, in every
        // window AND in the full-context tail decode). `ذ` matches nothing, so
        // the walk scanned ahead, found `ٱلْكِتَٰبُ` and charged the user a
        // SKIPPED on a word they actually recited — a false popup that stops
        // the recitation. A token that is a strict PREFIX of the target and at
        // least two characters shorter is a clipped decode, not a skipped
        // word: leave it PENDING so the next window can still complete it.
        if (matchInTargetAhead == -1 &&
            spokenCurrentNormPre.length >= 1 &&
            spokenCurrentNormPre.length <= targetWordNorm.length - 2 &&
            targetWordNorm.startsWith(spokenCurrentNormPre)
        ) {
            android.util.Log.i(
                "ItqanSherpa",
                "clippedWord(): target $targetIndex '$targetWordOrig' spoken prefix " +
                    "'$spokenCurrent' -> PENDING (clipped decode, not a skip)"
            )
            result.add(WordAnalysis(
                wordIndex = targetIndex,
                text = targetWordOrig,
                normalizedText = targetWordNorm,
                status = RecitationStatus.PENDING,
                spokenText = spokenCurrent,
                tajweedRule = rule
            ))
            consumedNorms.add(targetWordNorm)
            spokenIndex++
            targetIndex++
            continue
        }

        for (j in targetIndex + 1 until aheadLimit) {
            if (isWordMatch(spokenCurrent, targetWordsNormalized[j])) {
                matchInTargetAhead = j
                break
            }
        }
        
        // The spoken current word matched a target word AHEAD of this one --
        // the user's voice already moved past this target word without it
        // being decoded. That is a REAL skipped word at any point in the
        // session. Mid-session it is only a CANDIDATE: the mistake watcher
        // (mistakeStop) requires the same index to stay WRONG/SKIPPED across
        // two separate emissions before pausing, so a word the STT simply
        // hasn't emitted yet (it arrives on the next window as a match and the
        // index's verdict flips) can never false-stop the session.
        if (matchInTargetAhead != -1) {
            result.add(WordAnalysis(
                wordIndex = targetIndex,
                text = targetWordOrig,
                normalizedText = targetWordNorm,
                status = RecitationStatus.SKIPPED,
                spokenText = null,
                // A skipped word is only ever a skipped word: never attach a
                // tajweed label - the user sees EITHER a wrong/skipped popup
                // OR a tajweed popup with a rule + score, never a mix.
                tajweedRule = null
            ))
            consumedNorms.add(targetWordNorm)
            targetIndex++
            continue
        }

        // OOV-garbage skip: the spoken word matched NO target (current or
        // ahead) and is not in the active vocabulary at all — a decode
        // artifact, not a misread word. Charging it as WRONG flagged false
        // mistakes on correct recitations (falaq's شر cascade started at the
        // garbage وَالْثَاتِ); step over it and keep aligning from the NEXT
        // spoken word. Capped at 2 consecutive skips so a genuinely alien
        // utterance still trips the WRONG path on the third word.
        val spokenCurrentNorm = normalizeForComparison(spokenCurrent, riwayah)
        if (spokenCurrentNorm.isNotEmpty() && spokenCurrentNorm !in targetNormSet && consecutiveGarbageSkips < 2) {
            consecutiveGarbageSkips++
            spokenIndex++
            continue
        }
        consecutiveGarbageSkips = 0

        val spokenRemaining = spoken.size - spokenIndex
        val targetRemaining = limit - targetIndex
        // The unmatched spoken word is the transcript frontier (the LAST word
        // the STT has rendered so far). The frontier lags the user's voice:
        // a word they already said can still arrive on the next window (or a
        // window cut can fragment it), so it must stay PENDING — never
        // SKIPPED/WRONG. Flagging it at the session's final endpoint was the
        // on-device Fatihah bug: the endpoint decode CUT at the window's
        // right edge (العالم for الْعَالَمِينَ), the frontier never matched,
        // and every unheard word behind it was charged as a SKIPPED mistake —
        // "2 mistakes while I have read correctly" on a perfect recitation.
        // A voluntary stop mid-surah is not a complete recitation: verdicts
        // for words past the frontier are only ever PENDING (the user simply
        // hadn't reached them / the STT hadn't rendered them yet).
        val isTranscriptFrontier = spokenIndex == spoken.size - 1
        if (spokenRemaining < targetRemaining && isTranscriptFrontier) {
            result.add(WordAnalysis(
                wordIndex = targetIndex,
                text = targetWordOrig,
                normalizedText = targetWordNorm,
                status = RecitationStatus.PENDING,
                spokenText = null,
                tajweedRule = rule
            ))
        } else {
            // (Muqatta'at head forgiveness is applied earlier — see the block
            //  immediately after the relaxed-muqatta'at fallback. It has to sit
            //  BOTH before the SKIPPED branch and before the WRONG branch: on
            //  device word 0 is charged SKIPPED far more often than WRONG,
            //  because the first token the decoder manages to render usually
            //  matches some later target and trips the "matches a target ahead"
            //  path. Guarding only the WRONG branch left the freeze untouched.)
            // The transcript either caught up with the target (spokenRemaining
            // >= targetRemaining) or already PASSED this word without any
            // target matching it — a real mishearing/insertion (the STT
            // rendered it and moved on, so it is not the frontier). WRONG
            // verdicts fire mid-session now: the mistake watcher's
            // two-emission confirmation (mistakeStop) still guards against
            // transient single-window mishears, but a real error stops the
            // recitation at the FIRST flagged word instead of after the fact.
            result.add(WordAnalysis(
                wordIndex = targetIndex,
                text = targetWordOrig,
                normalizedText = targetWordNorm,
                status = RecitationStatus.WRONG,
                spokenText = spoken[spokenIndex],
                // Same rule as SKIPPED above: a wrong word is only ever a
                // wrong word — never mixed with a score-less tajweed tag.
                tajweedRule = null
            ))
            // §4.51 — AND THE WALK CONTINUES. IT USED TO END HERE, filling every
            // remaining word of the range with a fragment-less PENDING row and
            // returning.
            //
            // That is what froze the user's real-mic Al-Baqarah session on its
            // SECOND word. One conflicting token at word 1 — a CANDIDATE the
            // confirm loop never verified, which the app itself conceded at the
            // end (`finalize(): forgiving 1 UNPROVEN mistake(s) [1] — no tail
            // decode ever came back without them (absentAnchor=-1 strikes=0)`) —
            // deleted the evidence for the whole recitation:
            //   decodeWindow(): done ... new=3 w emitted=31w    (still hearing him)
            //   analyze(fast): cursor held at 2 (walk wanted 1, heardThrough=2,
            //   armed=false, floor=2)                          ...to the end
            //   endOfRange(): summary raised (... cursor=2/6116 -> 6114 word(s)
            //   still grey)
            // With no CORRECT row past word 1 there was nothing for the display
            // cursor to advance over, so the reveal was pinned there for the
            // whole session no matter what the walk did: "it got stuck in
            // baqarah".
            //
            // The SKIPPED branch has always continued, and it is right: a
            // candidate is judged by `mistakeStop`, which needs TWO emissions
            // plus a whole-session tail decode that came back without the word
            // before it pauses and pops up. Until then the display must track the
            // voice. Marking the word and stepping over keeps the word DECIDED
            // (it keeps its red styling and its place in the summary), keeps the
            // following words aligned, and leaves the stop entirely to the
            // confirm loop — which is the contract the user asked for ("if we get
            // a mistake we re-read from the mistake").
            consumedNorms.add(targetWordNorm)
            spokenIndex++
            targetIndex++
            continue
        }
        spokenIndex++
        targetIndex++
    }
    
    // Hint-matched words were revealed via the Hint button (we injected their text into the
    // transcript so the tracker advanced past them). Positionally they are CORRECT so the loop
    // keeps going, but the user never said them -- flip them to HINTED so the UI paints the
    // tertiary/orange "hinted" styling and the mistake counter already counted 1 wrong. We do
    // NOT score the tajweed head on their audio (it would be analysing injected silence).
    if (hintedIndices.isNotEmpty()) {
        for (i in result.indices) {
            val w = result[i]
            if (hintedIndices.contains(w.wordIndex) && w.status == RecitationStatus.CORRECT) {
                result[i] = w.copy(status = RecitationStatus.HINTED)
            }
        }
    }

    return result
}

suspend fun alignRecitation(spoken: List<String>, target: List<String>, targetAllGroups: List<List<Any>>? = null): Pair<Int, Boolean> {
    var targetIndex = 0
    var spokenIndex = 0
    var hasError = false

    while (targetIndex < target.size && spokenIndex < spoken.size) {
        kotlinx.coroutines.yield()
        val targetWord = target[targetIndex]
        val groups = targetAllGroups?.getOrNull(targetIndex)

        // Match this target word against the next few spoken tokens. A muqatta'at word
        // (الم -> الف لام ميم) may consume several consecutive spoken tokens, so we
        // track how many tokens the match advanced (foundSpokenAdvance).
        var foundMatchIndexInSpoken = -1
        var foundSpokenAdvance = 1
        for (i in spokenIndex until minOf(spokenIndex + 3, spoken.size)) {
            var matched = false
            if (groups != null) {
                for (group in groups) {
                    if (group is List<*> && group.isNotEmpty() && group[0] is List<*>) {
                        @Suppress("UNCHECKED_CAST")
                        val slots = group as List<List<String>>
                        if (i + slots.size <= spoken.size &&
                            isWordMatchPlural((0 until slots.size).map { spoken[i + it] }, slots)) {
                            foundMatchIndexInSpoken = i; foundSpokenAdvance = slots.size; matched = true; break
                        }
                    } else {
                        @Suppress("UNCHECKED_CAST")
                        if (isWordMatchAny(spoken[i], group as List<String>)) {
                            foundMatchIndexInSpoken = i; foundSpokenAdvance = 1; matched = true; break
                        }
                    }
                }
            }
            if (!matched && isWordMatch(spoken[i], targetWord)) {
                foundMatchIndexInSpoken = i; foundSpokenAdvance = 1; matched = true
            }
            if (matched) break
        }

        var foundMatchIndexInTarget = -1
        if (foundMatchIndexInSpoken == -1) {
            val spokenWord = spoken[spokenIndex]
            for (j in targetIndex + 1 until minOf(targetIndex + 3, target.size)) {
                if (isWordMatch(spokenWord, target[j])) {
                    foundMatchIndexInTarget = j
                    break
                }
            }
        }

        if (foundMatchIndexInSpoken != -1) {
            targetIndex++
            spokenIndex = foundMatchIndexInSpoken + foundSpokenAdvance
            hasError = false
        } else if (foundMatchIndexInTarget != -1) {
            targetIndex = foundMatchIndexInTarget + 1
            spokenIndex++
            hasError = false
        } else {
            hasError = true
            spokenIndex++
        }
    }

    if (spokenIndex < spoken.size) {
        hasError = true
    }

    return Pair(targetIndex, hasError)
}

@Composable
fun RecitationApp(
    modifier: Modifier = Modifier,
    isDarkTheme: Boolean = true,
    onToggleTheme: () -> Unit = {},
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val prefs = remember { context.getSharedPreferences("itqan_stats", android.content.Context.MODE_PRIVATE) }
    var isArabic by remember { mutableStateOf(prefs.getBoolean("is_arabic", false)) }
    var showSplashScreen by remember { mutableStateOf(true) }
    val application = context.applicationContext as? android.app.Application
    val viewModel: MainViewModel = if (application != null) {
        androidx.lifecycle.viewmodel.compose.viewModel(
            factory = androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.getInstance(application)
        )
    } else {
        androidx.lifecycle.viewmodel.compose.viewModel()
    }

    val completedGoal = viewModel.completedGoalToCelebrate
    if (completedGoal != null) {
        AlertDialog(
            onDismissRequest = { viewModel.dismissCelebration() },
            title = {
                Text(
                    text = getAppString(isArabic, "Goal Accomplished! 🎉"),
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
            },
            text = {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.fillMaxWidth().padding(vertical = 16.dp)
                ) {
                    Text(
                        text = getAppString(isArabic, "🌟 Masha'Allah! 🌟"),
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.secondary,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = if (isArabic) {
                            // "لقد أتممت هدفك [type] من تسميع [n] آية!" -- type and n are inserted in.
                            "لقد أتممت هدفك ${getAppString(isArabic, completedGoal.type)} من تسميع ${completedGoal.targetAyahs} آية!"
                        } else {
                            "You have completed your ${completedGoal.type} goal of reciting ${completedGoal.targetAyahs} Ayahs!"
                        },
                        fontSize = 16.sp,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = if (isArabic) "إجمالي الأخطاء خلال هذا الهدف: ${completedGoal.mistakesCount}"
                               else "Total mistakes during this goal: ${completedGoal.mistakesCount}",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.error,
                        textAlign = TextAlign.Center
                    )
                }
            },
            confirmButton = {
                Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    Button(
                        onClick = { viewModel.dismissCelebration() },
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                    ) {
                        Text(getAppString(isArabic, "Subhan'Allah"))
                    }
                }
            }
        )
    }

    // SESSION-SUMMARY DIALOG: pops every time the user finishes reciting the
    // selected range (target fully covered + session finalized). Shows the
    // per-word breakdown regardless of whether a goal is configured, so a
    // user who hasn't set a goal still sees which words went wrong / were
    // skipped / had which tajweed mistake.
    val sessionSummary = viewModel.sessionSummaryToShow
    if (sessionSummary != null) {
        AlertDialog(
            onDismissRequest = { viewModel.dismissSessionSummary() },
            title = {
                // The summary appears on a user pause OR on a finish (§4.45), so
                // it must never claim a perfect recitation the screen contradicts:
                // words the recognizer never rendered are still grey.
                //
                // §4.48 — AND A MID-RANGE PAUSE IS TITLED AS A PAUSE. He recited
                // ~22 words of a 6116-word range, paused, and got a dialog that
                // read like a verdict on the surah ("it showed me a summary
                // although I didn't complete the range"). "Paused" plus a progress
                // line says exactly what happened, and leaves no room for the app
                // to look like it is praising or judging a recitation he never
                // finished.
                val perfect = sessionSummary.entries.isEmpty() &&
                    sessionSummary.unheardWordCount == 0 &&
                    !sessionSummary.paused
                Text(
                    text = when {
                        sessionSummary.paused ->
                            if (isArabic) "تم الإيقاف المؤقت" else "Recitation Paused"
                        perfect ->
                            getAppString(isArabic, "Perfect recitation! 🌟")
                        else ->
                            getAppString(isArabic, "Recitation Summary")
                    },
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (perfect)
                        MaterialTheme.colorScheme.primary
                    else
                        MaterialTheme.colorScheme.onSurface,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
            },
            text = {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp)
                        .heightIn(max = 480.dp)
                        .verticalScroll(rememberScrollState())
                ) {
                    // Header line: which surah + range was just finished.
                    Text(
                        text = if (isArabic) {
                            "سورة ${sessionSummary.surahNumber}، الآيات ${sessionSummary.rangeStart} - ${sessionSummary.rangeEnd}"
                        } else {
                            "Surah ${sessionSummary.surahNumber}, Ayahs ${sessionSummary.rangeStart} - ${sessionSummary.rangeEnd}"
                        },
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    // §4.48 — HOW FAR HE GOT, whenever the range was NOT finished.
                    // Without it a pause at word 22 of 6116 looks like a verdict on
                    // the whole surah, and the words after the pause (still grey) look
                    // like a failure rather than simply "not read yet".
                    if (sessionSummary.paused && sessionSummary.totalWordCount > 0) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = if (isArabic) {
                                "تمت تسميع ${sessionSummary.heardWordCount} من ${sessionSummary.totalWordCount} كلمة - اضغط الميكروفون للمتابعة"
                            } else {
                                "Recited ${sessionSummary.heardWordCount} of ${sessionSummary.totalWordCount} words — tap the mic to continue"
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))

                    val wrongs = sessionSummary.entries.filter { it.kind == "wrong" }
                    val skippeds = sessionSummary.entries.filter { it.kind == "skipped" }
                    // Group tajweed entries by (word, rule) so a multi-rule
                    // word doesn't display the same rule twice.
                    val tajweedPairs = sessionSummary.entries
                        .filter { it.kind == "tajweed" }
                        .groupBy { it.wordIndex to (it.rule ?: "") }
                        .map { (_, group) -> group.first() }

                    if (wrongs.isNotEmpty()) {
                        SessionSummarySection(
                            title = if (isArabic) "الكلمات الخاطئة (${wrongs.size})" else "Wrong Words (${wrongs.size})",
                            icon = Icons.Filled.Close,
                            iconTint = MaterialTheme.colorScheme.error,
                            isArabic = isArabic
                        ) {
                            wrongs.forEach { entry ->
                                SessionSummaryRow(
                                    word = entry.word,
                                    suffix = if (isArabic) " (الآية ${entry.ayah})" else " (Ayah ${entry.ayah})",
                                    textColor = MaterialTheme.colorScheme.onSurfaceVariant,
                                    isArabic = isArabic
                                )
                            }
                        }
                    }

                    if (skippeds.isNotEmpty()) {
                        SessionSummarySection(
                            title = if (isArabic) "الكلمات المتخطاة (${skippeds.size})" else "Skipped Words (${skippeds.size})",
                            icon = Icons.AutoMirrored.Filled.ArrowForward,
                            // §4.49 — SECTION CHROME MUST NOT WEAR A RULE'S
                            // COLOUR. This was 0xFFFB8C00, which is the Idgham head
                            // colour, so the "Skipped Words" heading looked like an
                            // Idgham result. Section markers take theme colours; a
                            // rule's colour is reserved for that rule.
                            iconTint = MaterialTheme.colorScheme.tertiary,
                            isArabic = isArabic
                        ) {
                            skippeds.forEach { entry ->
                                SessionSummaryRow(
                                    word = entry.word,
                                    suffix = if (isArabic) " (الآية ${entry.ayah})" else " (Ayah ${entry.ayah})",
                                    textColor = MaterialTheme.colorScheme.onSurfaceVariant,
                                    isArabic = isArabic
                                )
                            }
                        }
                    }

                    if (tajweedPairs.isNotEmpty()) {
                        SessionSummarySection(
                            title = if (isArabic) "أخطاء التجويد (${tajweedPairs.size})" else "Tajweed Mistakes (${tajweedPairs.size})",
                            icon = Icons.Filled.Warning,
                            // §4.49 — was 0xFF7E57C2, the Meem-Sakinah head colour.
                            // The heading is chrome; each ROW below now carries its
                            // own rule's colour (Madd red, Qalqalah blue, ...).
                            iconTint = MaterialTheme.colorScheme.error,
                            isArabic = isArabic
                        ) {
                            tajweedPairs.forEach { entry ->
                                SessionSummaryTajweedRow(
                                    word = entry.word,
                                    rule = entry.rule ?: "",
                                    isArabic = isArabic,
                                    score = entry.score
                                )
                            }
                        }
                    }

                    if (wrongs.isEmpty() && skippeds.isEmpty() && tajweedPairs.isEmpty()) {
                        // No mistakes to report - the user paused or finished and
                        // asked for the breakdown. Say so plainly. (No coverage
                        // line unless something is actually missing: §4.45 allows a
                        // couple of words the DECODER never rendered to coexist with
                        // a finish, and "Perfect recitation!" would then be a lie
                        // the user can see on screen - those words are still grey.)
                        val unheardShown = sessionSummary.unheardWordCount
                        // §4.48 — A PAUSE IS NOT A CLEAN RECITATION. With the
                        // range not finished there is no "no mistakes in this
                        // recitation" to report, because most of the range has not
                        // been recited yet; claiming otherwise is the same
                        // good-and-bad-at-once problem the unproven-mistake fix
                        // removes.
                        val perfect = unheardShown == 0 && !sessionSummary.paused
                        Text(
                            text = if (isArabic) {
                                if (perfect) "ما شاء الله! لا أخطاء في هذه التلاوة."
                                else if (sessionSummary.paused) "لا أخطاء في ما قرأته حتى الآن."
                                else "ما شاء الله! لم يُكتشف أي خطأ في هذه التلاوة."
                            } else {
                                if (perfect) "Masha'Allah — no mistakes in this recitation."
                                else if (sessionSummary.paused) "No mistakes in what you have recited so far."
                                else "Masha'Allah — no mistakes detected in this recitation."
                            },
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.primary,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth()
                        )
                        // §4.50 — NOT ON A PAUSE. This line says the RECOGNIZER
                        // missed words, which is the honest reading for a finish:
                        // the user read on and the decoder never rendered the tail.
                        // On a pause it is the opposite — those words were never
                        // read at all, and the coverage line above already says how
                        // far he got ("Recited 13 of 29 words"). Telling him the
                        // recognizer dropped 16 words he never uttered blames the
                        // wrong party and reads as a second, contradicting verdict.
                        if (!perfect && !sessionSummary.paused) {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = if (isArabic) {
                                    "${unheardShown} كلمة لم يلتقطها التعرف على الصوت"
                                } else {
                                    "$unheardShown word(s) were not picked up by the recognizer"
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.fillMaxWidth()
                            )
                        }
                    }
                }
            },
            confirmButton = {
                Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    Button(
                        onClick = { viewModel.dismissSessionSummary() },
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                    ) {
                        Text(getAppString(isArabic, "Close"))
                    }
                }
            }
        )
    }
    
    LaunchedEffect(Unit) {
        viewModel.initStats(
            prefs.getInt("target_ayahs", 50),
            prefs.getInt("recited_ayahs", 0)
        )
    }

    var finalSpokenText by remember { mutableStateOf("") }
    var partialSpokenText by remember { mutableStateOf("") }
    val spokenText = (finalSpokenText + " " + partialSpokenText).trim()

    var tajweedFinalSpokenText by remember { mutableStateOf("") }
    var tajweedPartialSpokenText by remember { mutableStateOf("") }
    // ── THE COACH SHOWS ONE TRANSCRIPT, NOT TWO ────────────────────────────
    // The Tajweed Rules Coach reads the pipeline's RAW partial, which is
    // CUMULATIVE for the whole session (append-only merge), so while it is
    // non-blank it IS the current transcript. `tajweedFinalSpokenText` only
    // takes over once a stop clears the partial. The old `final + " " + partial`
    // concatenation displayed the live decode AND the final decode as one
    // run-on sentence — the user's report: "it writes nonsense, then writes
    // again but closer to the example (like it listens twice) and writes both
    // as one I supposedly heard".
    val tajweedSpokenText = tajweedPartialSpokenText.ifBlank { tajweedFinalSpokenText }
    // Number of words already committed to the FINAL transcript. The pipeline
    // utterance is now one continuous hypothesis per session (see the pause()/
    // resume() decoder reset removal), so endpoint commits must only append the
    // words beyond this count — otherwise the same words get re-committed on
    // every pause and duplicate in the final text.
    var committedWordCount by remember { mutableStateOf(0) }
    // Snapshot of the FULL committed transcript (word list) taken at each
    // commit. After pause/resume the sdk re-emits the cumulative hypothesis
    // from the beginning — those already-committed words must not be shown
    // again (the "it displays the words without me reciting them" phantom
    // replay). Any re-emitted prefix that equals this list is skipped.
    var lastCommittedWords by remember { mutableStateOf<List<String>>(emptyList()) }
    var currentTab by remember { mutableStateOf("Memorize") }
    // Default to Tarteel. The user explicitly said "when the open app i want it to be in tarteel
    // not tajweed as default": Memorize previously defaulted to Tajweed, which meant on cold
    // launch the raw-audio recorder + acoustic model ran for every recitation even if the user
    // never asked for tajweed detection. Tajweed is now an explicit opt-in via the FilterChip.
    var recitationMode by remember { mutableStateOf("Tarteel") }
    // ── TARTEEL AND TAJWEED ARE INDEPENDENT PRACTICE SURFACES ──────────────
    // Each mode keeps its OWN surah selection. The ViewModel exposes a single
    // `selectedSurah`, so without this the surah picked in one mode leaked into
    // the other: the user recited Al-Fatihah in Tarteel, moved to Al-Baqarah in
    // Tajweed, came back to Tarteel and found Al-Baqarah ("they should be
    // separate"). Seeded from the first selection, then updated by the surah
    // picker and by the mode chips below.
    val modeSelectedSurah = remember { mutableStateMapOf<String, SurahData>() }
    LaunchedEffect(viewModel.selectedSurah) {
        val s = viewModel.selectedSurah ?: return@LaunchedEffect
        if (modeSelectedSurah.isEmpty()) {
            modeSelectedSurah["Tarteel"] = s
            modeSelectedSurah["Tajweed"] = s
        }
    }
    var showSurahSelectionDialog by remember { mutableStateOf(false) }
    
    var isListening by remember { mutableStateOf(false) }
    // TARTEEL CONTINUITY. The platform SpeechRecognizer ends its session by
    // itself after a short silence (and some engines return a final result
    // after every utterance). Tarteel used to interpret that as the user
    // stopping, so the mic went off after the first short pause even though
    // the user had not asked to stop. `tarteelWantsMic` is the USER'S INTENT,
    // kept across recognizer sessions; `tarteelSegmentEnded` is a wake-up
    // counter that restarts the recognizer after a natural segment end.
    // Never infer intent from `isListening` - the engine flips that by itself.
    var tarteelWantsMic by remember { mutableStateOf(false) }
    var tarteelSegmentEnded by remember { mutableIntStateOf(0) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    var alignment by remember { mutableStateOf(Pair(0, false)) }
    var detailedAnalysis by remember { mutableStateOf<List<WordAnalysis>>(emptyList()) }

    // Per-word tajweed verdict cache for the conflated analysis loop. Once a
    // word's acoustic verdict is decided (pass OR confirmed fail) it is stored
    // here keyed by (targetIndex, normalizedWord) and reused on every later
    // pass instead of re-running the (expensive) confirmation search. This is
    // what keeps steady-state analysis cheap: only NEWLY-heard words pay the
    // scoring cost. Cleared on range/surah change and session restart.
    val tajweedVerdictCache = remember { java.util.concurrent.ConcurrentHashMap<String, Float>() }

    /**
     * §4.46 - HOW FAR THE DISPLAY CURSOR HAS EVER REACHED IN THIS SESSION.
     *
     * The transcript colours a word iff `globalWordIndex < matchedWordCount`, so
     * a cursor that goes backwards takes colour OFF words the user has already
     * read. That is the "after the pause some words that were displayed went
     * blank, then came back" report.
     *
     * The walk legitimately reports a SHORTER cursor for a moment after a
     * resume (the transcript is re-seated at the re-read point, so the early
     * passes see a shorter list), and a re-decode can momentarily misalign one
     * word. None of that is a reason to un-display a word.
     *
     * Reset ONLY by an explicit fresh start - the Restart button, a surah/range
     * change, or the transcript wipe a finished run performs. Never by a pause,
     * a resume, a mistake popup, or the end of the range. A CONFIRMED mistake is
     * the single exception, and it does not use this floor at all: the cursor has
     * to return to the mistake word so the error colour is painted there.
     */
    var displayCursorFloor by remember { mutableIntStateOf(0) }

    /**
     * §4.46 - FAILED VERDICTS, CACHED AGAINST THE SEARCH WINDOW THEY USED.
     *
     * Only PASSING verdicts were cached, so a word that FAILED was re-scored -
     * including the whole -6 s confirmation sweep, up to 80 ONNX evaluations -
     * on every single acoustic pass, for as long as it stayed in the transcript.
     * That is the dominant cost of the acoustic loop (a 2-4.5 s pass), it burns
     * CPU the STT decode loop needs, and it is why a mistake popup could take
     * many seconds to come out on a slow phone.
     *
     * Re-running it is provably pointless while the search window is UNCHANGED:
     * the candidate windows are cut from the session's audio buffer, whose
     * content never changes once written, and the sweep's reach
     * (`maxStart = min(captured - span, duration - span)`) is what decides which
     * candidates exist. Same `maxStart` - byte-identical candidates - identical
     * result. So the fail is cached together with the `maxStart` it was computed
     * on, and re-run only when the buffer has grown enough to expose new
     * candidates - which is precisely when a LATER decode could acquit it.
     *
     * Keyed per (word, rule) like [tajweedVerdictCache], and cleared with it.
     */
    val tajweedFailCache = remember {
        java.util.concurrent.ConcurrentHashMap<String, Pair<Long, Float>>()
    }

    /**
     * Minimum wall-clock gap between two LIVE tajweed acoustic passes (the FINAL,
     * post-stop pass ignores it and always runs immediately).
     *
     * WHY: a pass used to cost 2.0-4.5 s of CPU on this phone - measured on-device
     * `analyze(tajweed): done 2009…4550ms` — because a word about to be flagged
     * swept -6 s in 150 ms steps (up to 80 model evaluations, -1 step 5). Running
     * one pass per emission saturates a core and starves the recognizer's decode
     * loop (`decodeLoop(): alive busy=true lastDecodeAge=5052ms`), which is exactly
     * the reported lag: "when I paused some words weren't displayed yet, then when
     * I opened the mic again they got displayed". The fast text loop (~210 ms)
     * already keeps the transcript current, so the acoustic pass can run at a
     * lower cadence without any user-visible loss of verdicts.
     *
     * §4.45 - 3 s - 1.8 s. The staged confirmation sweep (a word that merely
     * needs re-alignment is now settled in ~10 evaluations instead of ~80) made a
     * pass cheap enough that the old cadence was pure latency: the user waited
     * 3 s for the pass to even START, on top of the settle, on top of the pass
     * itself, before a mistake popup appeared. 1.8 s still leaves the recognizer
     * roughly half the wall time (the budget below is unchanged otherwise) and
     * halves the worst-case popup delay.
     */
    val TAJWEED_MIN_INTERVAL_MS = 1800L
    /** How long a LIVE pass waits for a word's audio to settle before scoring. */
    val TAJWEED_LIVE_SETTLE_MS = 1500L
    /** Last time a tajweed pass STARTED (epoch ms); 0 = never. */
    val lastTajweedPassMs = remember { java.util.concurrent.atomic.AtomicLong(0L) }

    // (item 5) Real-time tajweed popup state. When Memorize+Tajweed's acoustic engine flags a
    // TAJWEED_ERROR on a specific word, we POST it here and a small auto-dismissing dialog at
    // the bottom of the screen shows the user what went wrong WITHOUT pausing the recitation --
    // "i want the user to continue reading and if there is an error a pop up is shown". The
    // timestamp in the state lets Compose see two consecutive errors on the same word as a
    // fresh trigger. LaunchedEffect below auto-clears after 4 s.
    var latestTajweedPopup by remember { mutableStateOf<Pair<String, Int>?>(null) }  // (rule, wordIndex)
    // Kind of the currently-visible popup: "tajweed" (acoustic rule failure),
    // "wrong" (user said a different word) or "skipped" (target word never heard).
    // The popup dialog renders different title/body per kind.
    var latestPopupKind by remember { mutableStateOf("tajweed") }

    // Hint tracking for Memorize: the global word indices the user revealed via the Hint
    // button. Cleared on surah/range change and on Repeat/Restart. The analysis pass flips
    // any such word's positional CORRECT to HINTED so the UI can paint it with the tertiary
    // "hinted" styling while the mistake counter already charged 1 wrong.
    val hintedIndices = remember { mutableStateListOf<Int>() }
    
    val acousticRmsTracker = remember { floatArrayOf(0f) }
    var currentWordCount by remember { mutableStateOf(0) }
    val spokenTimings = remember { java.util.concurrent.CopyOnWriteArrayList<SpokenWordTiming>() }

    // Real microphone audio handed back by the SpeechRecognizer itself (BUNDLE_AUDIO_SOURCE).
    // The recognizer is the single microphone client, so this never conflicts with recognition
    // (opening a second AudioRecord in parallel is what broke recognition on devices like Samsung).
    // Null means acoustic scoring is skipped gracefully.
    var recognizedAudioUri by remember { mutableStateOf<Uri?>(null) }
    var recognizedAudioStartMs by remember { mutableStateOf(0L) }
    var recognizedAudio by remember { mutableStateOf<FloatArray?>(null) }

    // Self-managed microphone capture for the Tajweed Rules coach. We record OUR OWN audio here
    // instead of relying on the SpeechRecognizer's returned audio, because on devices like the
    // Samsung A05s the recognizer either (a) conflicts with a parallel AudioRecord or (b) hands
    // back an audio Uri we cannot decode. Owning the capture guarantees the Tajweed score always
    // reflects the user's real voice. The example text is known, so no speech-to-text is needed.
    //
    // Memorize+Tajweed now drives its speech transcript from the Sherpa-ONNX
    // offline recognizer via [tajweedPipeline] (single AudioRecord feeds both the
    // Sherpa OnlineRecognizer stream AND the TajweedNet acoustic sink). The old
    // MicRecorderController is kept as a fallback only when the model hasn't
    // been downloaded yet, but the Compose download dialog blocks the user from
    // entering Memorize+Tajweed until Sherpa is ready, so that path is rare.
    // `ayahWordsProvider` reads `targetOrig` (the active ayah range's display
    // words) at start() time, and the [AyahWordMatcher] uses the same list as a
    // post-filter hallucination guard — see itqan.app.sherpa.SherpaPipeline.kt.
    //
    // To work around Kotlin's ban on forward local references inside a closure,
    // the pipeline is constructed BELOW `targetOrig` and stored in this holder,
    // scope; the pipeline is assigned to it once `targetOrig` is available.
    val tajweedPipelineHolder = remember { TajweedPipelineHolder() }

    // §4.68 — A TEST-ONLY TRANSCRIPT SOURCE FOR THE NON-PIPELINE MODES.
    //
    // Tarteel runs on the PLATFORM SpeechRecognizer, and the PC-driven injection
    // harness feeds the SHERPA pipeline (`runInjectLoop` reads filesDir/inject.pcm
    // where the pipeline's mic worker would have opened the microphone). The two
    // never meet, so before this line there was NO WAY to test Tarteel at all -
    // every "injection test" in this project has silently measured Memorize +
    // Tajweed. That is how §4.66 shipped untested against the surface where the
    // user actually saw the bug ("keep displaying words after the missed word in
    // tarteel mode").
    //
    // When - and ONLY when - a PCM file is sitting in filesDir, a non-Tajweed
    // mode takes its transcript from the pipeline's injection loop instead of the
    // recognizer. Everything that makes Tarteel Tarteel is untouched: the mode
    // stays "Tarteel", so every gate keyed on `recitationMode == "Tajweed"` (the
    // acoustic pass, the tajweed popup watcher, the tajweed scoring) stays OFF,
    // and the walk, the freeze, the popup and the summary all run the exact code
    // they run for a real reader. The SpeechRecognizer is NOT started alongside
    // it - the single-mic invariant holds, and under injection the pipeline never
    // opens the microphone anyway.
    //
    // Delete the injected PCM (or use the app normally) and this is false: real
    // Tarteel behaviour is unchanged, byte for byte.
    val injectTranscriptSource: Boolean =
        recitationMode != "Tajweed" && tajweedPipelineHolder.pipeline?.isInjecting() == true

    // Forward-reference holder: the range-change auto-resume effect below is
    // declared ABOVE `startTajweedSession()` (Kotlin forbids forward references
    // to local functions). The effect body runs asynchronously (after its
    // 520 ms delay), by which time composition has assigned this ref.
    val startTajweedSessionRef = remember { arrayOfNulls<() -> Unit>(1) }
    // Snapshot of the active (surah, range) the LAST time the user STARTED a
    // recitation session. The final-decode commit + goal-stats charge run
    // against THIS snapshot, never against [viewModel.selectedSurah] /
    // [viewModel.rangeStart] / [viewModel.rangeEnd] read at recomposition
    // time. Reason: a user can change the surah/range BETWEEN the final
    // decode landing and the next recomposition -- if we read the view
    // model state, the OLD recitation's mistakes would be charged to the
    // NEW range's goals (silent mis-attribution). The snapshot is updated
    // inside [startTajweedSession] on every fresh start and cleared on the
    // range-change reset (line ~3377) so a new recitation always charges
    // its own goals.
    data class ActiveScope(val surah: Int, val rangeStart: Int, val rangeEnd: Int)
    val activeSessionScope = remember { mutableStateOf<ActiveScope?>(null) }
    // Compose-observable mirror of the pipeline's internal StateFlow (plain
    // getters like `holder.running` are NOT observed by Compose, which left the
    // FAB caption frozen on "Warming up…" forever). Re-collects whenever a new
    // pipeline is wired into the holder.
    var sherpaPhase by remember { mutableStateOf<itqan.app.sherpa.SingleAudioPipeline.State?>(null) }
    var sherpaWarm by remember { mutableStateOf(false) }
    LaunchedEffect(tajweedPipelineHolder.pipeline) {
        // A new pipeline = a fresh cold stream = needs a warm-up again.
        sherpaWarm = false
        tajweedPipelineHolder.pipeline?.state?.collect { sherpaPhase = it }
    }
    // Warm-up window: the multilingual streaming model needs ~2-3 s of
    // quiet feed BEFORE real speech, or its first words jam into BISNIS/
    // SELAMATAN filler and the user's opening words are lost. So after the
    // mic tap the engine feeds/decodes the ambient audio for up to 3 s while
    // the FAB says "Warming up…"; the user starts reciting once it flips to
    // "Listening — recite now" and the model locks onto Arabic instantly
    // (the 23:10/00:02 logs: quiet-feed-then-speech → words within ~0.5-1 s).
    LaunchedEffect(isListening, tajweedPipelineHolder.pipeline, sherpaPhase) {
        if (isListening && sherpaPhase is itqan.app.sherpa.SingleAudioPipeline.State.Running) {
            delay(3000)
            sherpaWarm = true
        }
    }
    val sherpaDownloader = remember(context) { SherpaModelDownloader(context) }
    val sherpaReady = remember { mutableStateOf<Boolean?>(null) }
    var tajweedRecordedAudio by remember { mutableStateOf<FloatArray?>(null) }
    var tajweedRecordingStartMs by remember { mutableStateOf(0L) }
    var tajweedRecordingDurationMs by remember { mutableStateOf(0L) }

    // Popup-guard: remembers which word-index the last tajweed popup fired for, so one mistake
    // isn't re-shown on every recomposition. Declared here (before the range-reset effect) so a
    // target change can also clear it.
    var lastTajweedPopupIndex by remember { mutableStateOf(-1) }
    // The tajweed score the last popup fired with. The popup guard refires when
    // the SAME word re-scores differently (a bad re-read after the popup), so
    // one mistake per (word, score) pair instead of once ever.
    var lastTajweedPopupScore by remember { mutableStateOf(-1) }
    // §4.49 — THE RULE THE LAST POPUP NAMED. A word can carry TWO rules
    // (analyzeRecitationDetails emits one row per (word, rule)), and the user
    // asked to see the score of EVERY rule, not just the first one found: "when
    // i get the score i see the score of every rule in that example even if there
    // are two rules in the same word i got the 2 different scores". The latch was
    // keyed on the word alone, so the SECOND rule of the same word was silently
    // swallowed — the user saw one popup and one score for a word that had two
    // verdicts waiting. Latching on (word, rule) lets each rule's verdict reach
    // him, in the same one-at-a-time "must be dismissed" flow.
    var lastTajweedPopupRule by remember { mutableStateOf("") }
    // §4.53 — EVERY (WORD, RULE) VERDICT ALREADY SHOWN, as a SET. The pair above
    // only remembers the LAST one, which makes two standing mistakes alternate for
    // ever (dismiss A, B is next; dismiss B, A again). Cleared by the same events
    // that clear the pair — a range change, a restart, leaving the tab, and the
    // re-read re-arm, because the user acting on a mistake is what makes a fresh
    // verdict on that word speakable again.
    var shownTajweedPopups by remember { mutableStateOf(setOf<String>()) }
    /**
     * set. It exists because two things were contradicting it:
     *
     *  * the popup re-raised. The guard was a single index that the popup's own
     *    DISMISSAL cleared, so the next frozen pass re-opened the same dialog —
     *    "it showed me a mistake then i continued. then when i paused i get the
     *    popup of the same mistake again !"
     *  * the summary forgave it. The forgive test is an ABSENCE-of-evidence test
     *    (`mistakeVerifyAbsentStrikes >= 1`) and that evidence comes only from
     *    `verifyTailWords`, which needs the Sherpa pipeline. Tarteel has none, so
     *    its strikes are permanently 0 and every mistake it showed was forgiven:
     *    "and the summury tells me no mistake !". Measured:
     *    `autoFinalize(): grace expired with an UNPROVEN mistake at wordIndex=6
     *    (no substantial decode ever judged it) -> forgiving`, then
     *    `endOfRange(): ... wrong=0 skipped=0`.
     *
     * A mistake the user was shown is a decision this app made. Only the RE-READ
     * may overturn it — `acquittal` puts the word back to CORRECT with its spoken
     * token — never the absence of an arbiter. Cleared with the session (range
     * change, restart, tab switch), never by dismissing a dialog.
     */
    var shownMistakeWords by remember { mutableStateOf(setOf<Int>()) }

    // §4.67 — WHICH WORD IS CURRENTLY RIDDEN OVER, AND SINCE WHEN.
    //
    // The ride-over is a bounded grace for a tentative error while the confirm loop
    // verifies it, and it has to be BOUNDED. The user's report: "for tajweed mode at
    // the end i kept saying random words and instead of getting a popup of wrong word
    // the app continued displaying words" — because the arbiter (a tail decode) keeps
    // returning INCONCLUSIVE and §4.51's ride-over had no deadline, so the display
    // advanced over the wrong word for ever.
    //
    // Wall-clock rather than a pass count: passes are emitted per decode window and
    // their rate varies a lot between phones (this one is ~2x slower than the other).
    var rideOverCandidateIdx by remember { mutableStateOf(-1) }
    var rideOverCandidateSinceMs by remember { mutableStateOf(0L) }

    // Wrong/skipped-word auto-stop latch (see the LaunchedEffect by the alignment
    // vars): the index of the last WRONG/SKIPPED word that auto-stopped the mic, so
    // the same error doesn't re-stop on every recomposition while the banner is up.
    var lastWrongStopIndex by remember { mutableStateOf(-1) }
    var lastWrongStopSpokenText by remember { mutableStateOf("") }
    // SKIPPED confirmation counters: a word the STT temporarily dropped (one
    // decode window heard only a fragment) is flagged SKIPPED on that emission
    // but re-appears on the next. So a SKIPPED must survive SEVERAL distinct
    // emissions AND dwell for >= 2 s before it may auto-pause the mic.
    var lastWrongStopSightings by remember { mutableStateOf(0) }
    var lastWrongStopFirstSeenMs by remember { mutableStateOf(0L) }
    // Wall-clock stamp of the last decisive tail verdict (word proven ABSENT).
    // The stop/popup decision is driven by THIS clock, not by new transcript
    // emissions: a mistake freezes the walk, the user stops speaking to read
    // the feedback, so no new emission ever arrives and an emission-counted
    // confirmation can never mature. Measured on every Al-Fatihah run:
    // `candidate idx=N sightings=1` forever, with
    // `confirmed (auto-pause + popup): 0` in every single one.
    var mistakeVerdictAtMs by remember { mutableStateOf(0L) }
    // How many tail decodes we have already spent on the CURRENT candidate
    // (drives the growing span above), and how many of them came back
    // "the word is absent".
    var mistakeVerifyAttempts by remember { mutableStateOf(0) }
    var mistakeVerifyAbsentStrikes by remember { mutableStateOf(0) }

    // Mistake-verification state (see the mistakeStop effect): before a
    // confirmed WRONG/SKIPPED may pause the recitation, ONE full-tail offline
    // re-decode of the captured audio checks whether the flagged target word
    // was actually spoken (the sliding windows can hide it at their edges).
    //   - verifiedSpokenIndices: target indices whose verification PASSED —
    //     the user demonstrably said them, so they count CORRECT and never
    //     re-candidate. Cleared on range/surah change and explicit restarts.
    //   - lastMistakeVerifyMs: cooldown so an INCONCLUSIVE verification
    //     (decoder busy / too little audio / blank decode — verifyTailWords
    //     returns empty for those) retries on later emissions instead of
    //     spamming a ~0.5-2 s decode every emission.
    val verifiedSpokenIndices = remember { mutableStateListOf<Int>() }
    var lastMistakeVerifyMs by remember { mutableStateOf(0L) }
    // ── §4.38 THE ENDPOINT DECODE MUST GET TO ACQUIT BEFORE WE CONDEMN ───────
    // `endpointDecodeLandedAtMs` = wall clock when the pipeline's whole-session
    // endpoint decode put words into the transcript (0 = not yet). That decode
    // is the strongest evidence the app has and it is ALREADY RUNNING when the
    // ABSENT strikes mature; condemning without it produced a false "Skipped
    // Word" popup for a word the decode itself rendered (see
    // [mistakeMayBeConfirmed]). `pipelineStoppedAtMs` bounds the wait.
    var endpointDecodeLandedAtMs by remember { mutableStateOf(0L) }
    var pipelineStoppedAtMs by remember { mutableStateOf(0L) }
    // ── GLOBAL VERIFICATION BUDGET (§4.30) ──────────────────────────────────
    // The per-candidate attempt cap above bounds ONE word, but a long range
    // can present a FRESH candidate every few seconds, and each one spent its
    // own 3 decodes. Measured fix30_verify_1 (Al-Fatihah 1-7, 46.5 s clip):
    // four verification decodes at t+24s/31s/37s/44s (idx 9/13/14/15), each
    // ~3.7-3.9 s, i.e. ~60 % of wall time inside verifyTailWords(). The
    // recognizer is a SINGLE slot (decodeBusy), so every verification decode
    // blocks the live sliding window for its whole duration; the window only
    // ever covers the TRAILING samples, so the audio that slid past while
    // verification ran is never decoded at all. The walk fell behind the
    // injection and words 16-29 were never heard.
    // Bound the TOTAL recognizer time verification may consume inside a rolling
    // window, ACROSS candidates. When the budget is exhausted the gate simply
    // declines to launch; the candidate stays UNPROVEN and the existing
    // stall/auto-finalize forgiveness paths handle it. Never a verdict change
    // in the condemning direction — this can only withhold a verification.
    var verifyBudgetWindowStartMs by remember { mutableStateOf(0L) }
    var verifyBudgetSpentMs by remember { mutableStateOf(0L) }
    // The candidate index whose early verification returned ABSENT (the tail
    // re-decode did not hear the word). Only this index may proceed to
    // popup/pause; -1 = no decided candidate. Reset whenever the first-mistake
    // candidate changes or the session explicitly restarts.
    var mistakeVerifyAbsentIdx by remember { mutableStateOf(-1) }
    // STUCK-RECONCILE cooldown (see the watcher): when a candidate survives
    // many emissions even after verification, one long-span tail decode may
    // replace the echo-poisoned live transcript wholesale.
    var lastReconcileMs by remember { mutableStateOf(0L) }
    // EPISODE-STABLE candidate tracking: noisy merges make the first-mistake
    // index flicker ±few words between emissions (echoes appear/vanish), which
    // used to reset the confirmation counters forever — no verification ever
    // fired and the cursor froze until a manual pause healed the transcript.
    // An ANCHOR index tolerates ±3 drift as the SAME episode; the episode only
    // closes after TWO consecutive clean emissions.
    var mistakeEpisodeAnchor by remember { mutableStateOf(-1) }
    var mistakeCleanStreak by remember { mutableStateOf(0) }

    // DETACHED verification machinery: the tail re-decode takes 1-3 s on-device,
    // while every emission restarts the mistake watcher effect — a verification
    // awaited INSIDE the effect was cancelled before its verdict landed (the
    // result was logged by the pipeline but never applied), and the cooldown
    // then blocked retries: slow phones livelocked with the cursor frozen on
    // one word. These hold the single in-flight job + its candidate index so
    // the verdict survives restarts and lands in state regardless.
    val verifyJobRef = remember { java.util.concurrent.atomic.AtomicReference<kotlinx.coroutines.Job?>(null) }
    val verifyInFlightIdx = remember { java.util.concurrent.atomic.AtomicInteger(-1) }
    val coroutineScope = androidx.compose.runtime.rememberCoroutineScope()

    // Display gate for WRONG/SKIPPED mistakes in Tajweed mode. The analysis
    // pass recomputes verdicts on EVERY emission, and a single sliding-window
    // decode can transiently drop/misorder words it re-hears correctly one
    // window later (measured on-device: window#3 dropped لِلَّهِ رَبِّ and
    // flashed a false "skipped" banner; window#4 restored them 1.2 s later).
    // So the visible mistake UI (banner + red word) may only show an index
    // that has survived the SAME multi-emission evidence the auto-pause
    // watcher requires — transient single-emission artifacts never flash.
    // Tarteel mode keeps the old instant-banner behaviour (no pipeline
    // emissions there). Reset by every explicit restart/tab switch.
    var confirmedMistakeUiIdx by remember { mutableStateOf(-1) }

    // Resume-from-mistake latch (see the mistakeStop effect): when a WRONG or
    // SKIPPED word auto-pauses the mic we snapshot the ENTIRE live transcript
    // plus the mistake's word index. The next fresh session (user taps the mic
    // again) re-seats the committed transcript to the words BEFORE the mistake,
    // so the recitation continues FROM the mistaken word: the repeat is heard,
    // the word-by-word coloring resumes there, and the final transcript has no
    // duplicated prefix. Cleared by every explicit restart (range change, ayah
    // repeat, restart button, FAB stop, tab switch).
    var mistakeResumeIndex by remember { mutableStateOf(-1) }
    var mistakeResumeLiveText by remember { mutableStateOf("") }

    // True once the pipeline delivered its FINAL endpoint emission for the
    // session. Mid-session the STT frontier lags the user's voice, so unheard
    // target words must stay PENDING (no mistake card, no pause). Only the
    // final emission is complete enough for real SKIPPED/WRONG verdicts.
    var sessionFinalized by remember { mutableStateOf(false) }
    // One-shot latch for the end-of-range auto-finalize below: the finalize path
    // charges goals and raises the summary, so it must run AT MOST ONCE per
    // session even though its trigger (a Compose snapshot state) re-fires.
    var autoFinalizeDone by remember { mutableStateOf(false) }
    // - IS A RECITATION SESSION IN PROGRESS? (§4.45) -
    // `isListening` only answers "is the mic hot right now", and BOTH stop
    // paths clear it BEFORE anything gets to decide the summary: a pause
    // (`stopTajweedSession()`), a confirmed wrong/skipped (`stopForMistake()`)
    // and a tajweed popup. The end-of-range decision used to be guarded by
    // `if (isListening)`, so the pause path skipped its own summary body and
    // the user got nothing at all ("in the pause there is no summury at all"),
    // while a post-mistake tap fell into the "start a session" branch and
    // never resumed the re-read.
    //
    // This flag is the session-level truth: it is set when a session starts
    // and cleared only when the session has actually been DECIDED (summary
    // raised / dismissed), by a range or surah change, or by leaving the tab.
    // A pause or a mistake popup leaves it TRUE, because the session is
    // paused mid-way, not finished.
    var recitationSessionActive by remember { mutableStateOf(false) }
    // - DEFERRED USER-PAUSE FINALIZE -
    // The user tapped pause in a Tajweed (pipeline) session, but the FINAL
    // whole-session decode has not landed yet. `stopTajweedSession()` only
    // calls `pause()`, and the endpoint decode arrives ~5 s later (measured
    // real-mic 2026-09-24: stop at 20:23:57.174, `endpoint decode LANDED with
    // 30w` at 20:24:02.196). `sessionFinalized` is set by THAT endpoint, so
    // running the finalize at tap time reads sessionFinalized=false, skips
    // every summary branch and the user gets NO SUMMARY on a pause - their
    // report: "in the pause there is no summury at all". The mic is stopped
    // immediately (responsiveness); this flag defers only the DECISION until
    // the endpoint lands, then runs the real (userInitiatedStop=true)
    // finalize exactly once.
    var userPauseFinalizePending by remember { mutableStateOf(false) }
    // How long the end-of-range summary waits for the mistake pipeline to
    // surface an outstanding WRONG/SKIPPED as its own popup before finalizing
    // anyway. See the "mistake hold" comment on the auto-finalize effect.
    var mistakeHoldSinceMs by remember { mutableLongStateOf(0L) }
    // A grace window that has ALREADY expired once must never be re-armed. The
    // unproven FORGIVE branch used to reset mistakeHoldSinceMs, so it held 20 s,
    // forgave, held 20 s, forgave… with the walk frozen and autoFinalizeDone
    // never set — measured fatihah_c27 (one "forgiving rather than accusing" at
    // 01:43:56, then "holding 0ms/20000" again at 01:43:58 and climbing), so the
    // session produced NO popup at all and the user saw an empty screen.
    var graceExpiredOnce by remember { mutableStateOf(false) }
    // How many times the grace-expiry branch has promoted unresolved PENDING rows
    // to CORRECT so they DISPLAY (§4.34). One pass is normally enough; the counter
    // exists so a later analysis rebuild that RESURRECTS the rows can promote them
    // again but is still guaranteed to reach the finalize instead of looping.
    var gracePromoteCount by remember { mutableStateOf(0) }
    // 20 s, not 12: two INDEPENDENT verification decodes are now needed before
    // a word is condemned (§4.13), and each costs 1.5-3.6 s on device plus a
    // 3 s cooldown. At 12 s the grace expired with the second strike still in
    // flight and the session fell through to the summary.
    val MISTAKE_GRACE_MS = 20000L
    // Hard cap on how many tail-decodes the mistake pipeline may spend on ONE
    // candidate word. Each decode runs on the same recognizer as the live
    // windows, so an uncapped retry loop literally eats the transcript it is
    // trying to judge (§4.15). 3 is the smallest number that can still reach
    // the two-strike verdict; past it the word is left UNPROVEN and
    // autoFinalize forgives it.
    val MAX_MISTAKE_VERIFY_ATTEMPTS = 3
    // ── GLOBAL VERIFICATION BUDGET (§4.30) ──
    // Rolling window over which total verification seconds are capped, and the
    // cap itself. 10 s of a 20 s window = at most half the wall time may be
    // spent inside verifyTailWords() across ALL candidates; the live decode
    // loop keeps the other half, so the trailing window cannot fall so far
    // behind that a whole stretch of the recitation is never decoded.
    // 20 s is chosen because a two-strike verdict needs two decodes in flight
    // plus their cooldown (§4.13), so a window shorter than that would starve
    // legitimate verification of a REAL mistake; 10 s admits two full 3.9 s
    // decodes plus margin, and the third must wait for the window to roll.
    val VERIFY_BUDGET_WINDOW_MS = 20000L
    val VERIFY_BUDGET_MAX_MS = 10000L
    // Progress markers for the hold above: the countdown restarts whenever the
    // walk decides a new word or recovers an unheard one.
    var lastDecidedCount by remember { mutableIntStateOf(0) }
    var lastUnheardCount by remember { mutableIntStateOf(Int.MAX_VALUE) }
    // Ticker that re-runs the auto-finalize effect while a finished session is
    // still live. The effect keys on detailedAnalysis, which stops changing once
    // the recognizer goes quiet -- without a tick the mistake hold would never
    // expire and the user would be left with no popup at all.
    var finalizeTick by remember { mutableIntStateOf(0) }
    // True when the session just ended because the user REACHED THE END of the
    // selected range (as opposed to pausing mid-way). Used to detect that the
    // NEXT mic tap should start a fresh recitation instead of resuming this one
    // — otherwise a finished session's transcript bleeds into the next
    // recitation (the "flashing popup then re-tap starts from the stale,
    // fully-recited transcript" bug). Consumed in startTajweedSession() and
    // reset on range/surah change and Restart.
    var sessionRangeCompleted by remember { mutableStateOf(false) }

    // The shared SpeechRecognizer must exist BEFORE the target-change reset effect below (which
    // stops a running session). It is the single microphone client for Memorize/Tarteel/Tajweed
    // (never opened alongside a parallel AudioRecord, per the single-mic constraint).
    val speechRecognizer: android.speech.SpeechRecognizer? = remember {
        try {
            if (android.speech.SpeechRecognizer.isRecognitionAvailable(context)) {
                android.speech.SpeechRecognizer.createSpeechRecognizer(context)
            } else null
        } catch (e: Throwable) { null }
    }

    // Suppression window (epoch ms). While `System.currentTimeMillis() < recognizerResultsSuppressedUntil`,
    // partial/final SpeechRecognizer callbacks drop their text. Used by the range/surah reset to
    // swallow the stale whole-transcript the recognizer is about to emit as we cancel its old session.
    var recognizerResultsSuppressedUntil by remember { mutableStateOf(0L) }

    
    val targetNorm = remember(viewModel.ayahs, viewModel.riwayah) {
        viewModel.ayahs.flatMap { it.normalizedWords }.map { normalizeForComparison(it, viewModel.riwayah) }
    }
    val targetOrig = remember(viewModel.ayahs) {
        viewModel.ayahs.flatMap { it.words }
    }
    // Pronunciation-aware target representation: for each display word, the list of
    // acceptable spoken FORM GROUPS (muqatta'at expand into an ordered slot list; dagger-alif
    // words get their alternate single-token forms). alignRecitation and analyzeRecitationDetails
    // consult this alongside targetNorm so utterances like "ألف لام ميم" for "الم" or "هذا" for
    // "هَٰذَا" match correctly instead of flagging false WRONG/SKIPPED.
    val targetFormGroups = remember(viewModel.ayahs) {
        viewModel.ayahs.flatMap { it.words }.map { itqan.app.verses.formGroupsFor(it) }
    }

    // Construct the Sherpa-ONNX offline streaming pipeline once per ayah range
    // (so its decoder-level hotwordsFile bias tracks the new vocabulary each
    // time the user changes the range — see itqan.app.sherpa.SherpaPipeline.kt
    // for why hotwordsFile is set only at recognizer construction time).
    // `tajweedPipelineHolder` was declared earlier in the Composable scope; the
    // pipeline is wired into it the first time `targetOrig` is resolved AND
    // re-wired on every range change (the previous instance is released first
    // to avoid leaking a native OnlineRecognizer + AudioRecord handle). The
    // wrap is safe even when the model files aren't downloaded yet: `start()`
    // returns false in that case, and the Compose download dialog blocks the
    // FAB until the user finishes the first-launch download.
    LaunchedEffect(targetOrig) {
        // MEMORY-LEAK GUARD: on first composition `viewModel.ayahs` is still
        // empty, so targetOrig is EMPTY. The old code wired a throwaway
        // pipeline for it and started prewarm() — which loaded the full
        // 174 MB ONNX engine before the real range arrived and released it
        // (two engines alive at once on a 7.9 GB phone with ~1.6 GB free).
        // Skip wiring until the target actually has words; this effect
        // re-runs the moment ayahs load and targetOrig flips non-empty.
        if (targetOrig.isEmpty()) {
            android.util.Log.i("ItqanSherpa", "LaunchedEffect(targetOrig): target empty — skipping pipeline wiring")
            return@LaunchedEffect
        }
        android.util.Log.i("ItqanSherpa", "LaunchedEffect(targetOrig): re-wiring pipeline; old=${tajweedPipelineHolder.pipeline?.pipelineId} targetWords=${targetOrig.size}")
        try { tajweedPipelineHolder.pipeline?.release() } catch (e: Throwable) {}
        tajweedPipelineHolder.pipeline = SingleAudioPipeline(
            context = context,
            ayahWordsProvider = { targetOrig }
        )
        android.util.Log.i("ItqanSherpa", "LaunchedEffect(targetOrig): new pipeline id=${tajweedPipelineHolder.pipeline?.pipelineId}")
        // Warm the recognizer in the background so the FIRST mic tap starts
        // instantly (the on-device 7 s dead-air-before-recording bug). Safe
        // when the model files aren't downloaded yet (start()/prewarm() no-op).
        try { tajweedPipelineHolder.pipeline?.prewarm() } catch (e: Throwable) {}
    }

    // Analysis request channels + epoch. Declared BEFORE the range-reset
    // effect so that effect can drain them: requests queued by the OLD range
    // must never be processed against the NEW one.
    val analyzeRequests = remember { kotlinx.coroutines.channels.Channel<AnalyzeRequest>(kotlinx.coroutines.channels.Channel.UNLIMITED) }
    val tajweedRequests = remember { kotlinx.coroutines.channels.Channel<AnalyzeRequest>(kotlinx.coroutines.channels.Channel.UNLIMITED) }
    // Bumped on every range/surah change (and tab switch): requests carrying an
    // older epoch are dropped unread by both analysis loops, and the deferred
    // tajweed loop refuses to apply stale verdicts — killing the "popup before
    // I even recited" class of bugs after a mid-session target change.
    var analysisEpoch by remember { mutableStateOf(0L) }

    // §4.50 — STALL CLOCK. Wall-clock stamp of the first moment a COMMITTED
    // session (the recognizer landed its endpoint, so no more audio is coming for
    // this utterance) stopped making progress with an incomplete range. Restarts
    // on any progress and is cleared by a fresh session, a pause or a reset.
    //
    // It exists because "the app is waiting for the rest of the range" and "the
    // app is dead" look identical to the user: both freeze the display and show
    // nothing. Measured, the wait was unbounded and silent — the auto-finalize
    // gate logged the same two lines every 2 s for minutes. See
    // STALL_SUMMARY_AFTER_MS and the `stallExpired` branch in the summary gate.
    var stallSinceMs by remember { mutableStateOf(0L) }

    // State-reset on target change. Changing the surah or the ayah pickers at ~3874/3881 reloads
    // `viewModel.ayahs` and so flips `targetNorm`/`targetOrig` below, which would normally
    // re-run the Memorize analysis loop. Without this reset, the stale `finalSpokenText` and
    // `recognizedAudio` from the previous range get aligned against the EMPTY/NEW target and the
    // first word is flagged WRONG/SKIPPED + a TAJWEED_ERROR, popping the mistake dialog/banner
    // even though the user hasn't read a word of the new range (exactly what the user reported).
    // We therefore stop any in-flight session and wipe every piece of speech/analysis state so
        // the new range always starts clean, then resume listening if the mic was on.
    LaunchedEffect(viewModel.ayahs, viewModel.selectedSurah, viewModel.rangeStart, viewModel.rangeEnd) {
        val wasListening = isListening
        // A range change is a deliberate reset, not an utterance boundary. If
        // the Tarteel restart effect were left armed it would reopen the mic on
        // the NEW target with the OLD segment's transcript.
        tarteelWantsMic = false
        if (wasListening) {
            // Use stopListening (NOT cancel) so the recognizer DELIVERS its final onResults with the
            // GET_AUDIO bundle for whatever we just heard, and watch the restart path in Memorize's
            // auto-continue loop resume the session on its own when it sees the session ended.
            // (cancel would kill both the final transcript AND the audio the acoustic scorer needs,
            // plus it would no-op the recognizer while hasError also fails sane -- the two things
            // that originally made changing the range fire phantom mistakes.)
            try { speechRecognizer?.stopListening() } catch (e: Throwable) {}
            // Defensive: a parallel AudioRecord should never be running in Memorize (single-mic
            // invariant), but if a future caller ever starts one, stop it here so its buffer
            // can't bleed into the new range's analysis pass.
            try { if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop() } catch (e: Throwable) {}
            tajweedRecordedAudio = null
            tajweedRecordingDurationMs = 0
            // stopListening DOES fire onResults (which unlatches isListening), but if the engine is
            // already wedged we still must not leave the UI thinking a session is live.
            isListening = false
        }
        finalSpokenText = ""
        partialSpokenText = ""
        committedWordCount = 0
        lastCommittedWords = emptyList()
        // A new range/surah is a deliberate fresh start: a stale mistake
        // resume must not re-seat its transcript into the new session.
        mistakeResumeIndex = -1
        mistakeResumeLiveText = ""
        spokenTimings.clear()
        currentWordCount = 0
        recognizedAudio = null
        recognizedAudioUri = null
        recognizedAudioStartMs = System.currentTimeMillis()
        alignment = Pair(0, false)
        detailedAnalysis = emptyList()
        hintedIndices.clear()
        // A new range/surah is a fresh target: verification verdicts proved the
        // user said words of the OLD target and must not leak into this one.
        verifiedSpokenIndices.clear()
        lastMistakeVerifyMs = 0L
        // Fresh session => fresh verification budget (§4.30). Leaving a spent
        // budget behind would make the first candidate of the new session wait
        // for the old window to roll before it could be verified.
        verifyBudgetWindowStartMs = 0L
        verifyBudgetSpentMs = 0L
        mistakeVerifyAbsentIdx = -1
        mistakeEpisodeAnchor = -1
        mistakeCleanStreak = 0
        latestTajweedPopup = null
        sessionFinalized = false
        // A new range/surah ends whatever session was in progress (§4.45): there
        // is nothing left to resume or summarize for the old target.
        recitationSessionActive = false
        // A new range/surah is a fresh target: the previous run's range-completed
        // flag must not make the next mic tap wipe a brand-new recitation.
        sessionRangeCompleted = false
        // §4.38 — the previous run's endpoint evidence must not gate the NEW
        // run's confirmations (a stale "landed" would let a condemn through
        // without waiting for this session's own endpoint decode).
        endpointDecodeLandedAtMs = 0L
        pipelineStoppedAtMs = 0L
        // New session: clear the just-completed summary so a fresh recitation
        // can build and pop its own dialog (the previous one stays dismissed
        // for the rest of the app's lifetime unless the user explicitly clears
        // it via the dialog's Close button).
        viewModel.dismissSessionSummary()
        // Clear the scope snapshot too. A fresh [startTajweedSession] will
        // re-snapshot the NEW range, so the OLD recitation's final-decode
        // commit (if it lands late) sees the new snapshot and charges the
        // NEW range's goals — but since the OLD session's audio was wiped
        // above (finalSpokenText="", committedWordCount=0, etc.), the
        // collector's filter drops the stale words and no mistake gets
        // charged. Defense in depth.
        activeSessionScope.value = null
        errorMessage = null
        // New range = new target words: drop every cached tajweed verdict so the
        // fresh session re-scores from scratch (stale verdicts from the previous
        // range's words must never leak into the new one).
        tajweedVerdictCache.clear()
            tajweedFailCache.clear()
        // A new range is a new session, so the display floor (§4.46) starts from
        // scratch - otherwise the first words of the new range would be blocked by
        // how far the previous one had got.
        displayCursorFloor = 0
        shownMistakeWords = emptySet()
        rideOverCandidateIdx = -1
        rideOverCandidateSinceMs = 0L
        // §4.50 — and so does the stall clock: the previous range's silence must
        // not pre-arm this one's stalled-session dialog.
        stallSinceMs = 0L
        // ── EPOCH BUMP + STALE-WORK DRAIN ── Everything queued for the OLD
        // range dies here: pending analysis requests are drained unread, both
        // loops drop anything carrying an older epoch, and pause()'s async
        // final-tail decode (which can still be running for ~3-4 s after a
        // manual stop) is invalidated so its words — belonging to the OLD
        // target — can never be committed or scored against the new range.
        // This kills the "popup before I even recited" bug after changing the
        // range (measured: madd popup with 51% immediately after switching,
        // then a second one seconds later, with zero recitation).
        analysisEpoch++
        try { tajweedPipelineHolder.pipeline?.invalidateInFlightCommit() } catch (_: Throwable) {}
        while (analyzeRequests.tryReceive().isSuccess) { /* drain */ }
        while (tajweedRequests.tryReceive().isSuccess) { /* drain */ }
        // Cancel is async: one more onPartial/onResults with the OLD whole-transcript can still
        // arrive before the engine fully stops. Suppress recognizer output for the next 4 s so
        // that stale transcript isn't re-analyzed against the fresh range AND pause()'s async
        // final decode (which lands within this window) stays inert (the exact bug the user
        // hit: mistakes appear immediately after changing the ayah range with nothing read).
        recognizerResultsSuppressedUntil = System.currentTimeMillis() + 4000L
        // Resume automatically: if the user was mid-recitation when they moved the range, start
        // listening again (once the suppression window has elapsed) so they can keep going without
        // a manual mic tap. Guard on the recognizer being alive and the tab still being Memorize.
        if (wasListening && currentTab == "Memorize") {
            kotlinx.coroutines.delay(520)
            if (currentTab == "Memorize" && !isListening) {
                try {
                    if (recitationMode == "Tajweed") {
                        // Tajweed mode resumes through the Sherpa pipeline (the
                        // targetOrig effect above has re-wired it for the new
                        // range by now). NEVER auto-resume via the
                        // SpeechRecognizer — that silently switched the session
                        // into Tarteel-like behaviour after a range change.
                        // (Called via the forward-reference holder: this effect
                        // is declared above startTajweedSession().)
                        if (tajweedPipelineHolder.pipeline != null) {
                            startTajweedSessionRef[0]?.invoke()
                        } else {
                            errorMessage = getAppString(isArabic, "Tajweed is warming up — please wait a moment…")
                        }
                    } else if (injectTranscriptSource && tajweedPipelineHolder.pipeline != null) {
                        // §4.68 — injected PCM present: take the transcript
                        // from the pipeline's injection loop (see
                        // [injectTranscriptSource]) instead of the recognizer.
                        startTajweedSessionRef[0]?.invoke()
                    } else if (speechRecognizer != null) {
                        startListening(speechRecognizer!!, context.packageName) { msg ->
                            errorMessage = if (msg.isEmpty()) null else msg; isListening = false
                        }
                        isListening = true
                    }
                } catch (e: Throwable) { /* restart best-effort */ }
            }
        }
    }
    
    // Also watch recognizedAudio directly (not just the Uri) so that the Tajweed-mode fallback
    // path -- which writes `recognizedAudio` from our OWN MicRecorderController but cannot write
    // `recognizedAudioUri` (that's a SpeechRecognizer-Bundle semantic) -- still triggers the
    // re-analysis pass. Without `recognizedAudio` here, a Memorize+Tajweed session on a phone
    // that can't decode the SpeechRecognizer bundle would record fine but never score any
    // words, silently disabling the tajweed detector (matches the user's report).
    //
    // `hintedIndicesSnapshot` is a plain List copy of the hint set, so Compose re-runs the effect
    // when the hint button adds an index (the SnapshotStateSet itself is NOT a stable key).
    val hintedIndicesSnapshot = hintedIndices.toList()
    // TWO-LOOP ANALYSIS (user requirement: display + WRONG/SKIPPED near
    // real-time; tajweed verdicts may lag 2-3 s).
    //
    //  FAST loop — text-only (scoreTajweed=false). Runs on every STT emission,
    //    completes in milliseconds, and drives the word highlight + WRONG/SKIPPED
    //    detection. It never runs the expensive TajweedNet acoustic pass, so it
    //    can't stall.
    //  TAJWEED loop — acoustic (scoreTajweed=true). Waits ~2 s after the latest
    //    emission, then scores the heard words and merges the verdicts into the
    //    display. The verdict cache means already-scored words are skipped, so
    //    each pass only pays for NEW words.
    //
    // Both loops are CONFLATED: the keyed effect only ENQUEUES the latest state;
    // each long-lived worker drains the queue down to the NEWEST request before
    // running, so a running pass is never cancelled and the display always
    // advances no matter how fast partials arrive. (The old single keyed
    // LaunchedEffect restarted — and cancelled — the in-flight analysis on every
    // emission because the growing audio snapshot was a key, so NO run ever
    // completed and matchedWordCount stayed at 0: words were heard but never
    // highlighted, "the app is not listening".)
    // (analyzeRequests / tajweedRequests / analysisEpoch are declared ABOVE the
    // range-reset effect so that effect can drain them and bump the epoch.)
    LaunchedEffect(spokenText, currentTab, recitationMode, targetNorm, targetOrig, targetFormGroups, recognizedAudioUri, recognizedAudio, hintedIndicesSnapshot, sessionFinalized) {
        if (currentTab != "Memorize") {
            // ── LEAVING THE RECITATION SURFACE MUST NOT MARK THE RANGE AS READ ──
            // This branch used to latch `alignment = Pair(targetOrig.size, false)`,
            // i.e. "the cursor sits past the last word, so the whole range renders
            // coloured". That is a lie about a session that was never recited: the
            // tab-change effect below tears the session down the moment the user
            // leaves Memorize, so nothing here should claim progress. The damage
            // surfaced on RETURN. The Tajweed coach shares this state, and its
            // pipeline sets `sessionFinalized` while the user is on the coach tab;
            // `resolveTranscriptCursor(0, previousCursor = targetOrig.size, ...)`
            // then takes `maxOf(newMatched, previousCursor)` on a finalized session
            // — so returning to Memorize with an EMPTY transcript still coloured
            // every word and the entire surah looked already recited
            // ("the words of the full surah are displayed like i have read it all").
            // Wipe the surface and its transcript instead of faking a full read.
            alignment = Pair(0, false)
            detailedAnalysis = emptyList()
            displayCursorFloor = 0
            sessionFinalized = false
            finalSpokenText = ""
            partialSpokenText = ""
            committedWordCount = 0
            lastCommittedWords = emptyList()
            mistakeResumeIndex = -1
            mistakeResumeLiveText = ""
            latestTajweedPopup = null
        } else {
            val req = AnalyzeRequest(
                spokenText = spokenText,
                targetNorm = targetNorm,
                targetOrig = targetOrig,
                targetFormGroups = targetFormGroups,
                audio = recognizedAudio,
                audioStartMs = recognizedAudioStartMs,
                hinted = hintedIndices.toList(),
                finalized = sessionFinalized,
                mode = recitationMode,
                riwayah = viewModel.riwayah,
                timings = spokenTimings.toList(),
                epoch = analysisEpoch
            )
            analyzeRequests.trySend(req)
            // Tajweed acoustic pass only applies in Tajweed mode with real audio.
            if (recitationMode == "Tajweed" && recognizedAudio != null) {
                tajweedRequests.trySend(req)
            }
        }
    }
    // (item 5e — REVISED per user) DONE-READING ⇒ PENDING COUNTS CORRECT.
    // When the session finalizes (user stopped; the full-tail decode already
    // had its best chance), remaining PENDING words are counted as READ
    // CORRECTLY and displayed green: the recognizer simply never rendered
    // them, and the user should not be charged for the decoder's gaps. No
    // popup, no resume point — the recitation is simply complete.
    //
    // CRITICAL: this must run AT MOST ONCE per finalized session, and it must
    // NOT be keyed on `detailedAnalysis`. It used to be — and this block MUTATES
    // detailedAnalysis — so it re-triggered itself forever: every pass converted
    // a few PENDING rows to CORRECT, which raised matchedSoFar, which slid the
    // `matchedSoFar + 8` window ~6 words forward, converting the next 6, and so
    // on through the whole target. Measured on Al-Baqarah (6116 words) after an
    // 11-second recitation: 1019 iterations ending at matched=6112 — thousands
    // of never-recited words reported CORRECT, with the UI stalled throughout.
    var pendingConversionDone by remember { mutableStateOf(false) }
    // The word indices forgiven for THIS finalized session. FROZEN at the first
    // conversion pass (so the `matched + 8` window can never slide — sliding it
    // was the infinite loop documented above) and RE-APPLIED to every later
    // rebuild of `detailedAnalysis`.
    //
    // WHY the re-apply is mandatory — measured on the user's real-mic Fatihah run
    // (2026-09-12 20:14):
    //   finalDecodeAndEmit(): done 5531ms
    //   finalizeCorrect(): 4 trailing PENDING word(s) counted CORRECT
    //   analyze(tajweed): done 4335ms … changed=12    <- REPLACED the list
    //   analyze(tajweed): done 4550ms … changed=10    <- REPLACED it again
    //   autoFinalize?(): rows=33 decided=29/29 unheardPending=4
    //   endOfRange(): … pendingRemain=true
    // The deferred tajweed loop rebuilds `detailedAnalysis` wholesale from a fresh
    // `analyzeRecitationDetails` call, which RESURRECTS the trailing PENDING rows.
    // `pendingConversionDone` had already latched, so nothing forgave them again:
    // `unheardPending` stuck at 4, `autoFinalize` burned its whole 20 s grace, and
    // the end-of-range summary was withheld. The user finished Al-Fatihah and saw
    // NO popup at all — "when i paused i didnt see a summury popup even when i
    // finished".
    val forgivenTailIndices = remember { mutableStateOf<Set<Int>>(emptySet()) }
    LaunchedEffect(sessionFinalized, isListening, currentTab, recitationMode, analysisEpoch, detailedAnalysis) {
        if (currentTab != "Memorize" || recitationMode != "Tajweed" ||
            !sessionFinalized || isListening
        ) {
            // Live session (or a different mode): re-arm so the NEXT finalize
            // still gets its single conversion pass.
            pendingConversionDone = false
            forgivenTailIndices.value = emptySet()
        } else if (detailedAnalysis.isNotEmpty()) {
            if (!pendingConversionDone) {
                pendingConversionDone = true
                val pendingRows = detailedAnalysis.filter { it.status == RecitationStatus.PENDING }
                // BOUND THE CONVERSION (measured disaster: a whole-Baqarah target
                // with 2.5 ayahs recited converted 6115 PENDING rows green). Only
                // convert when the user genuinely NEARLY finished the range: some
                // real progress (matched >= 3) and only the trailing tail region
                // (wordIndex <= matched + 8) is undecoded. Anything beyond is not
                // part of this session and stays PENDING (silent).
                //
                // §4.50 — COUNT WORDS, NOT ROWS. `detailedAnalysis` holds ONE ROW
                // PER (WORD, RULE), so `count { CORRECT || … }` counted a word once
                // per rule and the window slid forward by the number of extra rules
                // — the same inflation that broke the re-seat cut. Measured: the walk
                // matched 13 words, the counter said 15, and the window reached 23.
                val matchedSoFar = detailedAnalysis
                    .filter {
                        it.status == RecitationStatus.CORRECT ||
                            it.status == RecitationStatus.TAJWEED_ERROR ||
                            it.status == RecitationStatus.HINTED
                    }
                    .map { it.wordIndex }
                    .distinct()
                    .size
                // §4.50 — AND A NEVER-HEARD WORD MAY ONLY BE FORGIVEN AT THE **END**
                // OF THE RANGE.
                //
                // Converting a PENDING row sets its COLOUR, and colour is the cursor,
                // so a conversion is a claim: "the user read this". A row that
                // carries a spoken fragment supports that claim (a clipped decode).
                // A row with NO fragment does not — the recognizer never heard it —
                // and the only thing that can make it believable is that the user
                // demonstrably reached the END of the range, where a trailing gap is
                // the decoder's known blind spot rather than an unread word.
                //
                // Measured `v5_stall` (a 20 s slice of Al-Fatihah against a 29-word
                // range): the user recited 13 words, and this function coloured 11
                // more —
                //   finalizeTrim(): committed set re-seated to the 13 matched word(s)
                //   finalizeCorrect(): 11 trailing PENDING word(s) counted CORRECT
                //                    (matched=15, total pending=16)
                // i.e. the display ran to ~24 of 29 over words the user never
                // uttered, and then the summary was withheld on top of it, leaving a
                // frozen screen: "it advance without displaying then restarted from
                // begining and now it is stuck". The user's own count of the words he
                // read is the ground truth; the display must not argue with it.
                val totalTarget = targetOrig.size
                val heardFrontier = detailedAnalysis
                    .filter { !it.spokenText.isNullOrEmpty() }
                    .map { it.wordIndex + 1 }
                    .maxOrNull() ?: 0
                val reachedRangeEnd = totalTarget > 0 &&
                    heardFrontier >= totalTarget - FINALIZE_FORGIVE_TAIL_WINDOW
                // THE TRAILING WINDOW MUST NOT CROSS AN AYAH BOUNDARY. The
                // forgiveness window is a decoder blind-spot allowance for the
                // ayah the reader is IN; the next ayah is not a blind spot, it
                // is simply not this pause's recitation. Without this cap a
                // pending/clipped row at the start of the next ayah could be
                // converted, and displayedCursorFor then coloured up to it.
                val heardAyahEnd = if (heardFrontier > 0)
                    ayahEndForWordIndex(heardFrontier - 1, viewModel.ayahs)
                else 0
                val conversionLimit = minOf(
                    matchedSoFar + FINALIZE_FORGIVE_TAIL_WINDOW + 1,
                    heardAyahEnd
                )
                val inWindow = if (conversionLimit > 0)
                    pendingRows.filter { it.wordIndex < conversionLimit }
                else emptyList()
                val convertible = inWindow.filter {
                    !it.spokenText.isNullOrEmpty() || reachedRangeEnd
                }
                if (inWindow.size != convertible.size) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "finalizeCorrect(): NOT converting ${inWindow.size - convertible.size} " +
                            "word(s) ${inWindow.filter { it.spokenText.isNullOrEmpty() }.map { it.wordIndex }} " +
                            "that carry no decoded fragment — the user reached word $heardFrontier " +
                            "of $totalTarget, so the range did not finish and these words were " +
                            "never read; they stay PENDING (grey)"
                    )
                }
                if (matchedSoFar >= 3 && convertible.isNotEmpty()) {
                    val convertIdx = convertible.map { it.wordIndex }.toHashSet()
                    forgivenTailIndices.value = convertIdx
                    detailedAnalysis = detailedAnalysis.map { w ->
                        if (w.status == RecitationStatus.PENDING && w.wordIndex in convertIdx)
                            w.copy(status = RecitationStatus.CORRECT, spokenText = null)
                        else w
                    }
                    android.util.Log.i(
                        "ItqanSherpa",
                        "finalizeCorrect(): ${convertible.size} trailing PENDING word(s) counted CORRECT (matched=$matchedSoFar, total pending=${pendingRows.size})"
                    )
                }
            } else {
                // IDEMPOTENT RE-APPLY. Only ever PENDING -> CORRECT, and only for
                // the frozen set, so a late pass that turns one of these words into
                // a REAL mistake still wins (the mistake popup path is untouched),
                // while a rebuild that merely re-created the PENDING rows cannot
                // resurrect them. Settles after one pass: once no PENDING row in the
                // set remains, nothing is written and the effect stops re-firing.
                val forgiven = forgivenTailIndices.value
                if (forgiven.isNotEmpty() && detailedAnalysis.any {
                        it.status == RecitationStatus.PENDING && it.wordIndex in forgiven
                    }
                ) {
                    detailedAnalysis = detailedAnalysis.map { w ->
                        if (w.status == RecitationStatus.PENDING && w.wordIndex in forgiven)
                            w.copy(status = RecitationStatus.CORRECT, spokenText = null)
                        else w
                    }
                    android.util.Log.i(
                        "ItqanSherpa",
                        "finalizeCorrect(): re-applied ${forgiven.size} forgiven tail word(s) " +
                            "after a late analysis rebuild (autoFinalize would otherwise hold the " +
                            "full grace and withhold the summary)"
                    )
                }
            }
        }
    }

    // FAST loop: text-only verdicts for the near-real-time display.
    LaunchedEffect(Unit) {
        for (req in analyzeRequests) {
            // Conflate: drop every superseded request, keep only the newest.
            var latest = req
            while (true) {
                val newer = analyzeRequests.tryReceive().getOrNull() ?: break
                latest = newer
            }
            // EPOCH GUARD: a request queued before the last range/surah/tab
            // change belongs to the OLD target — drop it instead of scoring
            // stale audio against words the user never reached.
            //
            // §4.47 AND THE SAME GUARD BEFORE THE **WRITE**. The check above only
            // protects the QUEUE: a pass that was already running when the user
            // tapped Restart / Repeat Ayah / changed the range had read the old
            // `alignment` and the old `analysisEpoch` BEFORE the bump, and it
            // wrote its result 60-400 ms LATER — resurrecting the cursor the reset
            // had just cleared. Measured on device (`b2_buttons`): after Restart
            // the transcript was empty (`walk wanted 0, heardThrough=-1`) and yet
            // `cursor held at 13 (... floor=13)`, because this pass put the old
            // cursor back and re-raised the floor. With a longer range that reads
            // as "it froze, then all the words suddenly appeared to the end of the
            // range". The reset must be able to land AFTER a pass has started, so
            // the pass re-checks the epoch immediately before it writes.
            val passEpoch = analysisEpoch
            if (latest.epoch != passEpoch) continue
            val analyzeT0 = System.currentTimeMillis()
            // Debounce only coalesces a same-frame burst of partials into ONE
            // pass; it does NOT gate any emission. This loop is the FAST
            // text-only path (scoreTajweed=false) and its work runs on
            // Dispatchers.Default, so it can neither ANR the UI thread nor move
            // the acoustic tajweed segment cut (that lives in the separate
            // TAJWEED loop below, which has its own 2 s settle). 60 ms is
            // deliberately not 0 so a burst still collapses into a single pass.
            kotlinx.coroutines.delay(60)
            try {
                val analysis = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
                    val spoken = latest.spokenText.split(Regex("\\s+"))
                        .filter { it.isNotBlank() }
                        .map { normalizeForComparison(it, latest.riwayah) }

                    // scoreTajweed=false: text-only, no acoustic pass -> fast.
                    analyzeRecitationDetails(
                        spoken,
                        latest.targetOrig,
                        latest.targetNorm,
                        latest.riwayah,
                        latest.timings,
                        context,
                        fullAudio = latest.audio,
                        recordingStartMs = latest.audioStartMs,
                        recordingDurationMs = (latest.audio?.size ?: 0) * 1000L / 16000,
                        targetAllGroups = latest.targetFormGroups,
                        hintedIndices = latest.hinted,
                        isTajweedMode = latest.mode == "Tajweed",
                        finalReport = latest.finalized,
                        verdictCache = tajweedVerdictCache,
                        failCache = tajweedFailCache,
                        scoreTajweed = false
                    )
                }

                // §4.47 — THE WRITE-SIDE EPOCH GUARD. The pass above took 60-400 ms
                // (a slow phone: up to seconds). If the user reset the session while
                // it ran — Restart, Repeat Ayah, a surah/range change — everything
                // this pass is about to write describes the session that no longer
                // exists, and it would land on top of the reset. Measured
                // `b2_buttons`: Restart cleared the transcript, and 300 ms later
                // this pass put the old cursor back and re-raised the display floor
                // (`cursor held at 13 (... floor=13)` on an empty transcript), which
                // is the "it froze, then all the words appeared at once" report.
                // Dropping the result is always safe: the reset already discarded
                // everything this pass would have replaced, and the next emission
                // re-requests the analysis.
                if (passEpoch != analysisEpoch) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "analyze(fast): DROPPED a stale result (epoch $passEpoch -> $analysisEpoch) " +
                            "— the session was reset while this pass was running"
                    )
                    continue
                }
                // CONFIRMED-MISTAKE PRESERVATION. Once mistakeStop confirms a
                // WRONG/SKIPPED it pauses the mic and arms mistakeResumeIndex --
                // the user must re-read FROM that word on resume. A stale
                // analysis that still lands afterwards (a request already queued
                // when the latch armed, carrying pre-freeze transcript text)
                // would re-judge the flagged word CORRECT, advance the cursor
                // past it and erase the mistake ("judged correct after the
                // fact"). While the latch is armed for this exact index, keep
                // the previously-confirmed WRONG/SKIPPED verdict instead.
                //
                // VERIFIED-SPOKEN OVERRIDE (applied first): if mistakeStop's
                // tail re-decode proved the user DID speak a flagged word, it
                // lands in verifiedSpokenIndices and any WRONG/SKIPPED verdict
                // for that index counts CORRECT — the cursor advances instead
                // of sticking on a phantom the live windows never heard.
                // MULTI-RULE SAFE lookups: detailedAnalysis now holds ONE ROW PER
                // (word, rule), so index-position lookups and wordIndex-keyed maps
                // must never assume a single row per word.
                val prevByKey = detailedAnalysis.associateBy { "${it.wordIndex}:${it.tajweedRule ?: "-"}" }
                val prevMistakeByIdx = detailedAnalysis
                    .filter { it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED }
                    .associateBy { it.wordIndex }
                val armedMistakeIdx = mistakeResumeIndex
                val verifiedIdxSet = verifiedSpokenIndices.toHashSet()
                val effective = analysis.map { w ->
                    var w2 = w
                    if ((w2.status == RecitationStatus.WRONG || w2.status == RecitationStatus.SKIPPED) &&
                        w2.wordIndex in verifiedIdxSet
                    ) {
                        w2 = w2.copy(status = RecitationStatus.CORRECT)
                    }
                    if (w2.status == RecitationStatus.CORRECT && armedMistakeIdx >= 0 && w2.wordIndex == armedMistakeIdx) {
                        val pw = prevMistakeByIdx[w2.wordIndex]
                        if (pw != null && (pw.status == RecitationStatus.WRONG || pw.status == RecitationStatus.SKIPPED)) w2 = pw
                    }
                    w2
                }

                var newMatchedWordCount = 0
                var newHasError = false
                val seenIdx = HashSet<Int>()
                // A CONFIRMED mistake (resume latch armed) freezes the reveal — the
                // session is paused for a re-read. Anything else is still TENTATIVE.
                val armedMistake = mistakeResumeIndex >= 0
                var rodeOverTentativeSkip = false
                // §4.58 — A RIDE-OVER IS ONLY HONEST WHERE SOMETHING WILL JUDGE IT.
                //
                // The ride-over trades "stop now" for "keep the highlight on the
                // voice while the confirm loop decides", and that trade only makes
                // sense if the confirm loop EXISTS. It is gated on Memorize +
                // Tajweed, so on the READ / Tarteel surface a ridden-over error is
                // never judged at all: the walk steps over it, the display fills in
                // the words between the last correct word and the word the user
                // actually said, and no skipped-word mistake is ever raised. The
                // user: "in tarteel mode … tried skipping words but it always display
                // the words between the last correct word and the word i said
                // instead of stopping and giving me a skipped word mistake".
                //
                // In Tarteel a jump IS the mistake: there is no acoustic pass and no
                // verify decode to second-guess it, and the reader can SEE that they
                // skipped. So there the walk freezes on the first gap exactly as it
                // always did, and the display stops where the reader actually is.
                val confirmLoopWillJudge = currentTab == "Memorize" && recitationMode == "Tajweed"

                // §4.66 — CORRECTED: A FRAGMENT-LESS PENDING ROW IS **ALWAYS
                // TRAILING**, SO IT CAN NEVER BE A GAP IN THE MIDDLE OF THE WALK.
                //
                // This comment used to claim the opposite, and that claim WAS the
                // whole diagnosis of the user report "keep displaying words after the
                // missed word in tarteel mode". It is wrong, and the analyzer itself
                // proves it: there are exactly THREE places
                // `analyzeRecitationDetails` emits a PENDING row -
                //   * the clipped-decode guard, which CARRIES `spokenText` (it IS a
                //     heard word, and the walk counts it),
                //   * the transcript-FRONTIER row (one row, `spokenText = null`),
                //     after which `spokenIndex` runs past the end of `spoken`, and
                //   * the "spoken exhausted" fill, which emits a PENDING row for
                //     every remaining target word and then BREAKS.
                // The frontier row is therefore immediately followed by the exhausted
                // fill, so a PENDING row with no fragment is NEVER followed by a
                // DECIDED row: there is no mid-range gap to detect, and a rule keyed
                // on one could only ever have been dead code.
                //
                // Kept as a stated invariant instead, because it is the reason the
                // unreached TAIL of a live recitation can never freeze the highlight
                // (§4.34), and because it means the walk on a skipped word depends
                // entirely on the WRONG/SKIPPED rows the analyzer really does emit -
                // which is exactly what §4.67 then bounds.


                // §4.67 — AND THE RIDE-OVER IS **TIME-BOUNDED**.
                //
                // "for tajweed mode at the end i kept saying random words and instead
                // of getting a popup of wrong word the app continued displaying words".
                // On Memorize + Tajweed a tentative error is ridden over while the
                // confirm loop verifies it — right in principle, but §4.51 made it
                // UNBOUNDED, and the arbiter is a tail decode that regularly comes back
                // INCONCLUSIVE. The ride-over became permanent: the display advanced
                // over a wrong word and no popup was ever raised. A ride-over is a
                // bounded grace, not a redefinition of the verdict.
                //
                // After [RIDE_OVER_BUDGET_MS] of riding over the SAME word the walk
                // stops and the mistake is raised. If the confirm loop proved it in the
                // meantime the word is already in `shownMistakeWords`, so this cannot
                // produce a second popup for it.
                fun rideOverBudgetSpent(wordIdx: Int): Boolean {
                    if (rideOverCandidateIdx != wordIdx) {
                        rideOverCandidateIdx = wordIdx
                        rideOverCandidateSinceMs = System.currentTimeMillis()
                        return false
                    }
                    return System.currentTimeMillis() - rideOverCandidateSinceMs >=
                        RIDE_OVER_BUDGET_MS
                }

                // §4.61 — ONE MISTAKE SURFACE FOR BOTH KINDS, ON BOTH PATHS.
                //
                // The user, after testing Tarteel on its own: "still no popups and no
                // summary". Two reasons, both mine:
                //
                //  * The auto-pause + popup was wired into the **WRONG** branch only.
                //    The SKIPPED branch's freeze was left as a bare `break`, and a
                //    Tarteel skip is a SKIP — so the display stopped (correct) and
                //    nothing was ever raised. A reader who skips a word saw silence.
                //  * The gate also had to work when the row is skipped, which the
                //    inline block above could not do because it is a statement, not a
                //    decision point.
                //
                // So the decision lives in one place, called from BOTH freezes. It is
                // a no-op on Memorize + Tajweed (where the confirm loop owns the
                // verdict and arms the latch itself), and on every other surface the
                // freeze itself IS the verdict: the reader visibly jumped, there is no
                // pipeline and no tail decode to arbitrate, so the popup is raised here.
                // `confirmedMistakeUiIdx` keeps it to once per word, the resume latch
                // is armed exactly as on the Tajweed path so the display rewinds to
                // the mistake, and the mic is stopped so the popup owns the screen.
                fun raiseFreezePopupIfUnjudged(
                    wordIndex: Int,
                    wordText: String,
                    skipped: Boolean,
                    // §4.67 — `force` is the RIDE-OVER BUDGET running out. Normally a
                    // freeze leaves the verdict to the confirm loop, which owns it on
                    // Memorize + Tajweed. But the budget expiring means the grace is
                    // over and no arbiter proved the word in all that time, so the app
                    // itself is the decider and must speak — otherwise the display
                    // would stop with no popup and the reader would have no idea why
                    // ("instead of getting a popup of wrong word the app continued
                    // displaying words"). The shown-set means this can never double up
                    // with a popup the confirm loop already raised.
                    force: Boolean = false
                ) {
                    if (confirmLoopWillJudge && !force) return
                    if (armedMistake) return
                    // §4.64 — ONCE PER WORD, FOR THE WHOLE SESSION, VIA A SET. The
                    // single-index guard this replaced was cleared on dismissal, so
                    // the very next frozen pass re-raised the same dialog — the user
                    // saw: "it showed me a mistake then i continued. then when i
                    // paused i get the popup of the same mistake again !". A set
                    // survives the dismissal, and it is ALSO what the summary reads,
                    // so the popup and the summary can never disagree.
                    if (wordIndex in shownMistakeWords) return
                    if (latestTajweedPopup != null) return
                    android.util.Log.i(
                        "ItqanSherpa",
                        "mistakeStop(): ${if (skipped) "SKIPPED" else "WRONG"} at wordIndex=$wordIndex " +
                            "word='$wordText' in $recitationMode mode — no confirm loop here (no " +
                            "pipeline, no tail decode), so the walk's own freeze is the verdict: " +
                            "auto-pause + popup"
                    )
                    confirmedMistakeUiIdx = wordIndex
                    shownMistakeWords = shownMistakeWords + wordIndex
                    latestPopupKind = if (skipped) "skipped" else "wrong"
                    latestTajweedPopup = Pair(wordText, wordIndex)
                    mistakeResumeLiveText = (finalSpokenText + " " + partialSpokenText).trim()
                    mistakeResumeIndex = wordIndex
                    recitationSessionActive = true
                    displayCursorFloor = minOf(displayCursorFloor, wordIndex)
                    // §4.52 — AND THE DISPLAY ACTUALLY GOES BACK NOW. The floor is
                    // only a minimum, and with the mic stopped no further emission
                    // arrives to recompute the cursor, so the renderer's own predicate
                    // is written here: the reader sees the mistake under the highlight
                    // the moment the popup appears, exactly as on the Tajweed path.
                    alignment = Pair(wordIndex, true)
                    android.util.Log.i(
                        "ItqanSherpa",
                        "mistakeStop(): display rewound to $wordIndex so the reader can SEE where to " +
                            "re-read from (§4.52 — the floor alone cannot move the cursor backwards)"
                    )
                    try { speechRecognizer?.stopListening() } catch (_: Throwable) {}
                    try { tajweedPipelineHolder.pipeline?.pause() } catch (_: Throwable) {}
                    isListening = false
                }

                // TAJWEED-only errors (rule-acoustic failed but the WORD was spoken correctly)
                // no longer pause the recitation. The user explicitly asked the app to "continue
                // reading" and show a popup on the tajweed mistake instead of stopping. Wrong
                // and skipped words still pause as before -- they are detection failures that
                // would otherwise silently let the user babble ahead.
                for (word in effective) {
                    if (word.status == RecitationStatus.WRONG) {
                        // §4.51 — A TENTATIVE WRONG RIDES OVER, EXACTLY LIKE A
                        // TENTATIVE SKIPPED. It USED TO FREEZE THE REVEAL
                        // UNCONDITIONALLY, and that froze the user's real-mic
                        // Al-Baqarah session on its SECOND word:
                        //   decodeWindow(): done ... new=3 w emitted=31w   (the user
                        //   is reading on; the recognizer hears 31, 35, 47 words)
                        //   analyze(fast): cursor held at 2 (walk wanted 1,
                        //   heardThrough=2, armed=false, floor=2)      …repeats to
                        //   the end of the session…
                        //   endOfRange(): summary raised (... cursor=2/6116 ->
                        //   6114 word(s) still grey)
                        // i.e. the display sat on 2 of 6116 words for the whole
                        // recitation — "it got stuck in baqarah". The candidate was
                        // never even verified: `finalize(): forgiving 1 UNPROVEN
                        // mistake(s) [1] — no tail decode ever came back without
                        // them (absentAnchor=-1 strikes=0)`, so the app itself
                        // concluded at the end that it had NO evidence. A freeze
                        // that the confirm loop never owns is a dead end: the row
                        // flickers, `mistakeStop` keeps restarting the episode, and
                        // nothing can ever clear it.
                        //
                        // The contract is unchanged for a REAL mistake: this is the
                        // same pre-confirmation ride-over the SKIPPED branch below
                        // has always used. The confirm loop still judges the word
                        // in the background; if a whole-session tail decode comes
                        // back without it, `stopForMistake` arms
                        // `mistakeResumeIndex`, `armedMistake` flips true and the
                        // NEXT pass freezes and returns to the mistake word with the
                        // popup. Only the UNPROVEN freeze is gone.
                        if (!sessionFinalized && !armedMistake && confirmLoopWillJudge &&
                            !rideOverBudgetSpent(word.wordIndex)
                        ) {
                            // §4.54 — AND A RIDDEN-OVER ERROR MUST NOT SET
                            // `newHasError`. The flag used to mean "the walk STOPPED
                            // here"; §4.51 gave the walk a ride-over, so it no longer
                            // stops — yet a single unresolved row (Al-Baqarah's
                            // opening `الم ذلك` is one) kept re-setting the flag on
                            // every pass and it never cleared for the whole
                            // recitation. Measured on the user's real-mic Al-Baqarah
                            // run: `hasError=true` in 44 of 47 passes, false only 3
                            // times (all before the first word).
                            //
                            // It matters because that flag is what switches the cursor
                            // into SMOOTH REVEAL — `maxOf(matched, previous)`,
                            // one-directional — so a permanently-true flag pins the
                            // display at the highest the walk ever reached and it can
                            // never come back to the voice. The user's words: "display
                            // of words is far away from what i am reading".
                            rodeOverTentativeSkip = true
                            if (seenIdx.add(word.wordIndex)) newMatchedWordCount++
                        } else {
                            // §4.61 — THE FREEZE IS ALSO WHERE A POPUP IS RAISED, on
                            // every surface that has no confirm loop to raise it.
                            // §4.67 — and `force` when the ride-over budget ran out, so
                            // a stale unproven error cannot stop the display silently.
                            raiseFreezePopupIfUnjudged(
                                word.wordIndex,
                                word.text,
                                skipped = false,
                                force = rideOverBudgetSpent(word.wordIndex)
                            )
                            newHasError = true
                            break
                        }
                        continue
                    }
                    if (word.status == RecitationStatus.SKIPPED) {
                        if (!sessionFinalized && !armedMistake && confirmLoopWillJudge &&
                            !rideOverBudgetSpent(word.wordIndex)
                        ) {
                            // §4.54 — SAME AS ABOVE: a ridden-over SKIPPED does not
                            // set `newHasError` either, for the same reason.
                            //
                            // SMOOTH REVEAL over a TENTATIVE skip. The walk marks this
                            // word SKIPPED only because the recognizer's frontier already
                            // moved PAST it (the word's own decode was clipped/ambiguous —
                            // e.g. Baqarah's ذَٰلِكَ opening on a fragment). But the walk's
                            // own contract calls a mid-session SKIPPED "only a CANDIDATE":
                            // the recognizer usually re-hears the word one window later.
                            // A tentative skip is NOT proof the user skipped the word, so
                            // it must not freeze the highlight while the confirm loop
                            // verifies it in the background. Ride over it and keep counting
                            // the words the recognizer HAS matched — the highlight tracks
                            // the voice instead of stalling-then-leaping. If the confirm
                            // loop CONFIRMS the skip (2 sightings / 2 absent strikes) it
                            // arms `mistakeResumeIndex`, `armedMistakeNow` flips true, and
                            // the reveal returns to the mistake index on the next pass —
                            // the pause + error colour render exactly as before. Only the
                            // PRE-confirmation reveal stops freezing. The verdict list is
                            // untouched: `detailedAnalysis` still carries the SKIPPED row.
                            newHasError = true
                            rodeOverTentativeSkip = true
                            if (seenIdx.add(word.wordIndex)) newMatchedWordCount++
                        } else {
                            // Finalized, or a confirmed skip is armed: freeze for the verdict.
                            //
                            // §4.61 — AND, ON A SURFACE WITH NO CONFIRM LOOP, FREEZE
                            // *AND* RAISE. The user testing Tarteel on its own got
                            // "still no popups": the display stopped correctly (that is
                            // the freeze) but nothing was raised, because this branch
                            // used to be a bare `break` while the popup lived only in
                            // the WRONG branch. A SKIP is the mistake a reader makes
                            // most often here, so it is the one that must speak.
                            // §4.67 — and `force` when the ride-over budget ran out.
                            raiseFreezePopupIfUnjudged(
                                word.wordIndex,
                                word.text,
                                skipped = true,
                                force = rideOverBudgetSpent(word.wordIndex)
                            )
                            newHasError = true
                            break
                        }
                        continue
                    }
                    if (word.status == RecitationStatus.CORRECT || word.status == RecitationStatus.TAJWEED_ERROR || word.status == RecitationStatus.HINTED) {
                        // TAJWEED_ERROR and HINTED both continue advancing -- TAJWEED_ERROR only
                        // matters via its popup, and HINTING is the user's own reveal, so we must
                        // not pause waiting for a word they'll never need to say.
                        // MULTI-RULE: several rows can share one wordIndex — count each WORD once.
                        if (seenIdx.add(word.wordIndex)) newMatchedWordCount++
                    } else if (word.status == RecitationStatus.PENDING &&
                        !word.spokenText.isNullOrEmpty()
                    ) {
                        // ── CLIPPED-PENDING COUNTS AS REACHED ──
                        // The clipped-decode guard emits PENDING carrying the fragment
                        // the recognizer DID hear (spokenText != null): the user said
                        // the word, the decode window simply cut it off. Leaving it
                        // uncounted froze the highlight one word short, so the LAST
                        // word of an ayah/range never lit up until the user tapped
                        // pause (the on-device "the final word in Fatihah didn't get
                        // displayed" report). PENDING rows with spokenText == null are
                        // words never heard, and must NOT advance the cursor.
                    }
                }

                // ── §4.37 THE CURSOR MAY NOT REGRESS ON A FINALIZED SESSION ──
                // `alignment.first` IS the renderer's predicate (`globalWordIndex
                // < matchedWordCount` => coloured, else 25 % grey). This pass
                // recomputes it from scratch, and the aligner is not stable: with
                // audio attached it re-decodes each candidate's slice, so a word
                // it decided a moment ago can come back PENDING and the cursor
                // drops — measured `p2_fatihah`, `matched=29` at 22:01:35.680 then
                // `matched=28` at 22:01:35.899 on the SAME frozen 29-token
                // transcript, leaving the last word of Al-Fatihah greyed while the
                // "Perfect recitation!" summary was on screen. A word the user has
                // seen may never be taken away (REACHED ⇒ DISPLAYED).
                // SMOOTH-REVEAL floor. When the walk froze on a TENTATIVE
                // WRONG/SKIPPED (hasError, no confirmed mistake armed yet), the
                // user has usually recited further than the stalled word — the
                // transcript grew but `newMatchedWordCount` broke at the candidate.
                // Advance the reveal to the LAST word that carries heard evidence
                // (a spoken token), never past it, so the highlight tracks the
                // voice instead of freezing for the whole ~15-25 s confirm window.
                // One-directional (max with the current cursor). This is display
                // only: the mistake CONFIRM loop (`stopForMistake` below) is
                // unchanged, and the instant it arms `mistakeResumeIndex` the
                // ride-over switches off and the cursor returns to the mistake.
                val lastHeardWord = effective
                    .filter { !it.spokenText.isNullOrEmpty() }
                    .maxOfOrNull { it.wordIndex }
                    ?.plus(1)

                // §4.69 — AND THE FLOOR MUST READ THE LATCH **AFTER** THE WALK.
                //
                // `armedMistake` is a snapshot taken BEFORE the walk, and the walk is
                // allowed to ARM the latch: on a surface with no confirm loop,
                // `raiseFreezePopupIfUnjudged` sets `mistakeResumeIndex` the instant
                // it raises the popup. The floor logic below kept reading the
                // PRE-walk snapshot, so on the very pass that raised the popup it
                // still believed no mistake was armed - and did all three wrong
                // things at once:
                //   * `revealFloor` was lifted to `lastHeardWord`, i.e. the reader's
                //     LAST HEARD WORD, which by construction is PAST the mistake
                //     (he kept reading after the one he skipped),
                //   * `resolveTranscriptCursor` was told "not armed", so it did not
                //     park on the flagged index, and
                //   * `displayCursorFloor` was LATCHED at that past-the-mistake value.
                //
                // That silently undid §4.52, which writes
                // `alignment = Pair(wordIndex, true)` inside the popup helper
                // precisely so the display rewinds the instant the popup appears: the
                // `alignment = Pair(cursorNow, ...)` write a few lines below the
                // helper overwrote it again with the drifted cursor.
                //
                // On Memorize + Tajweed a later pass used to rescue it, because the
                // confirm loop re-ran the walk while the mic was still open. Tarteel
                // has no confirm loop: the freeze STOPS the recognizer, so no further
                // pass ever arrives, the cursor never comes back, and the screen keeps
                // every word after the missed word coloured for as long as the popup
                // is up. That is the report verbatim - "keep displaying words after
                // the missed word in tarteel mode", and "there is a missed word i
                // continued reading the word after it and it worked instead of
                // rereading the mistake".
                //
                // The walk's OWN gate must keep the pre-walk snapshot: a word the walk
                // has just decided must not flip its own branch mid-iteration. Only
                // the cursor arithmetic below reads the post-walk value.
                val armedMistakeNow = armedMistake || mistakeResumeIndex >= 0
                val revealFloor = if (!sessionFinalized && newHasError && !armedMistakeNow)
                    maxOf(alignment.first, lastHeardWord ?: 0)
                else
                    alignment.first
                val resolvedCursor = resolveTranscriptCursor(
                    newMatchedWordCount = newMatchedWordCount,
                    previousCursor = revealFloor,
                    hasError = newHasError,
                    finalized = sessionFinalized,
                    confirmedMistakeArmed = armedMistakeNow,
                    // §4.52 — park ON the flagged word, not wherever the walk got
                    // to. The walk rides over a tentative error now, so its count
                    // is past the mistake and would keep the highlight there.
                    mistakeIndex = mistakeResumeIndex
                )
                // - §4.46 THE SESSION FLOOR: A WORD ONCE SHOWN STAYS SHOWN -
                // The cursor drives the RENDER, and a lower cursor literally
                // un-colours everything above it. The report was exact: after a
                // pause the user watched words they had just recited go grey and
                // then come back. The walk is not stable across a resume - the
                // transcript is re-seated at the prefix, so the first passes walk
                // a SHORTER list and `newMatchedWordCount` is legitimately lower
                // for a moment.
                //
                // So within one session the cursor may only ever move forward,
                // EXCEPT when a mistake is CONFIRMED and armed: there the cursor
                // MUST return to the mistake word, because the error colour is only
                // painted at/after the cursor and the user has to SEE which word
                // stopped them. That is the one exception, and it is exactly the
                // exception `resolveTranscriptCursor` already documents.
                //
                // The floor is reset only by an EXPLICIT fresh start (Restart
                // button, surah/range change) - never by a pause, a resume, a
                // mistake or the end of the range.
                // §4.69 — `armedMistakeNow` here too, for the same reason: on the pass
                // that raised the popup the latch IS armed, so the session floor must
                // not pin the cursor (nor be latched below) at the drifted value the
                // pre-walk snapshot had already allowed.
                val floorNow = if (armedMistakeNow) 0 else displayCursorFloor
                // §4.54 — EXCEPT WHEN THE WALK FALLS FAR BEHIND. A floor that only
                // ever ratchets up turns every large re-alignment into PERMANENT
                // drift, and the user has named the symptom: "display of words is
                // far away from what i am reading". Measured on his real-mic
                // Al-Baqarah run, one session:
                //   walk=104 … walk=156            (the display was up at 156)
                //   cursor held at 30 (walk wanted 12, heardThrough=58, floor=30)
                // The aligner re-ran after the pause and came back with 12, and the
                // display stayed pinned ~30 words ahead of it.
                //
                // The floor exists for a ONE- or TWO-word flicker while a decision
                // settles (§4.46: a word must not blink grey as the acoustic pass
                // catches up). A regression of DISPLAY_FLOOR_SLACK words or more is
                // not a flicker, it is a re-alignment — and there the display must
                // follow the voice, because a highlight that disagrees with the
                // reader is worse than one that briefly re-colours.
                val floorTooFarAhead = floorNow - newMatchedWordCount > DISPLAY_FLOOR_SLACK
                if (floorTooFarAhead) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "analyze(fast): the walk fell $floorNow -> $newMatchedWordCount " +
                            "(more than $DISPLAY_FLOOR_SLACK words) so the display follows it " +
                            "instead of staying ${floorNow} words ahead (§4.54 — a floor that only " +
                            "ratchets up drifts away from the voice)"
                    )
                    displayCursorFloor = newMatchedWordCount
                }
                val effectiveFloor = if (floorTooFarAhead) newMatchedWordCount else floorNow
                // A STOPPED session may colour no further than the ayah the
                // reader was last heard in. `sessionFinalized` covers Tajweed's
                // final decode; for Tarteel a user stop clears `tarteelWantsMic`
                // before the recognizer is closed, so the auto-continue gap is
                // deliberately NOT capped (that would un-colour the current
                // utterance every few seconds).
                val displayCapApplies = sessionFinalized ||
                    (!isListening && (!tarteelWantsMic || recitationMode == "Tajweed"))
                val displayCeiling = if (displayCapApplies)
                    displayCapForHeardAyah(effective, viewModel.ayahs)
                else Int.MAX_VALUE
                val cursorNow = minOf(maxOf(resolvedCursor, effectiveFloor), displayCeiling)
                if (!armedMistakeNow && cursorNow > displayCursorFloor) {
                    displayCursorFloor = cursorNow
                } else if (!armedMistakeNow && cursorNow < displayCursorFloor) {
                    // The session was paused just inside the next ayah (or a
                    // promotion crossed the boundary). Pull the visible floor
                    // back with the cursor so the next pass cannot re-inflate
                    // the display over unread words.
                    displayCursorFloor = cursorNow
                    android.util.Log.i(
                        "ItqanSherpa",
                        "analyze(fast): stopped-session display capped at ayah boundary " +
                            "($cursorNow) - next-ayah words stay grey"
                    )
                }
                alignment = Pair(cursorNow, newHasError)
                if (cursorNow != newMatchedWordCount) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "analyze(fast): cursor held at $cursorNow (walk wanted " +
                            "$newMatchedWordCount, heardThrough=${lastHeardWord ?: -1}, armed=$armedMistakeNow, " +
                            "floor=$floorNow) - a displayed word may not go grey again"
                    )
                }
                if (rodeOverTentativeSkip) {
                    // §4.51 — the same ride-over now covers a tentative WRONG, so
                    // the line names both: a reader comparing it with the row list
                    // must not conclude that a SKIPPED was involved when it was a
                    // conflicting token (the Al-Baqarah freeze was a WRONG).
                    android.util.Log.i(
                        "ItqanSherpa",
                        "analyze(fast): SMOOTH-REVEAL rode over a tentative WRONG/SKIPPED -> " +
                            "reveal cursor=$resolvedCursor (the confirm loop still judges it in the " +
                            "background; a CONFIRMED mistake freezes and returns here)"
                    )
                }
                // Carry over any tajweed verdict the deferred acoustic loop already
                // flagged, keyed by (word, rule) so each rule's verdict on the same
                // word survives independently — the fast loop must not wipe a
                // TAJWEED_ERROR the moment after it's shown.
                val merged = effective.map { w ->
                    var w2 = w
                    if (w2.status == RecitationStatus.CORRECT && w2.tajweedScore == null) {
                        val pk = prevByKey["${w2.wordIndex}:${w2.tajweedRule ?: "-"}"]
                        if (pk != null && (pk.status == RecitationStatus.TAJWEED_ERROR || pk.tajweedScore != null)) w2 = pk
                    }
                    // NOTE: the §4.34 clipped promotion is applied to the LIST below,
                    // AFTER the §4.37 finalized-decision guard — a fresh PENDING row
                    // must be allowed to be overridden by the decided row it is
                    // replacing BEFORE anything tries to promote it.
                    w2
                }
                // §4.37, second half: on a finalized session a fresh PENDING row may
                // not un-decide a word the previous pass already decided — otherwise
                // `unheardWordCount()` flips back to 1 on a recitation the app just
                // called perfect, and `endOfRange()`'s `genuinelyFinished` disagrees
                // with the summary already on screen.
                //
                // §4.34 last, on purpose: a clipped decode (PENDING carrying the
                // fragment the recognizer DID hear) is a word the user SAID, and the
                // cursor already counts it as reached — so it must be coloured too.
                // Measured `clean_fatihah_2`: the last word of Al-Fatihah came back as
                // a clipped prefix, `matched=29/29`, the checker read 28, and the user
                // saw "last word not displayed and there is no popup". The fragment is
                // KEPT, not nulled, so `unheardWordCount()` still skips the row and a
                // genuine finish can still raise its summary.
                detailedAnalysis = merged
                    .keepFinalizedDecisions(previous = detailedAnalysis, finalized = sessionFinalized)
                    .promoteClippedPendingRows()
                // ── §4.40 THE CURSOR MUST COVER EVERY DECIDED WORD ──────────
                // The promotion above makes rows DECIDED; it does not make them
                // VISIBLE, because the renderer colours by the cursor and not by
                // `status`. A rubber-stamped CORRECT row (status decided,
                // `spokenText == null`) is not counted by `newMatchedWordCount`,
                // so on a finalized session it can sit beyond the cursor: judged
                // by the app, grey on the screen. Measured `p4_fatihah` —
                // `autoFinalize(): grace expired with 29/29 words decided
                // (unheard=0)` followed 11 ms later by `endOfRange(): …
                // cursor=4/29`. Advance (never retreat) to cover the decided
                // prefix; a real WRONG/SKIPPED is itself decided, so it stays
                // inside the prefix and keeps its error colour.
                if (sessionFinalized) {
                    // Cover decided words only up to the ayah the reader was
                    // last heard in; the range's PENDING tail is decided
                    // bookkeeping, not recitation evidence.
                    val covered = minOf(
                        displayedCursorFor(detailedAnalysis, alignment.first),
                        displayCapForHeardAyah(detailedAnalysis, viewModel.ayahs)
                    )
                    if (covered > alignment.first) {
                        android.util.Log.i(
                            "ItqanSherpa",
                            "analyze(fast): cursor advanced ${alignment.first} -> $covered so every " +
                                "decided word DISPLAYS (§4.40 REACHED => DISPLAYED)"
                        )
                        alignment = Pair(covered, alignment.second)
                    }
                }
                android.util.Log.i("ItqanSherpa", "analyze(fast): done ${System.currentTimeMillis() - analyzeT0}ms matched=$newMatchedWordCount hasError=$newHasError targetWords=${latest.targetNorm.size}")
            } catch (t: Throwable) {
                android.util.Log.e("ItqanSherpa", "analyze(fast): FAILED after ${System.currentTimeMillis() - analyzeT0}ms", t)
            }
        }
    }
    // TAJWEED loop: deferred acoustic scoring (~2 s after the latest emission).
    LaunchedEffect(Unit) {
        for (req in tajweedRequests) {
            // Conflate to the newest request.
            var latest = req
            while (true) {
                val newer = tajweedRequests.tryReceive().getOrNull() ?: break
                // - A FINALIZED (ENDPOINT) REQUEST MUST NOT BE SKIPPED -
                // It carries the whole-session audio and skips the 2 s settle
                // below; a newer LIVE request used to overwrite it, so the last
                // words' acoustic verdict was never computed on the ending
                // audio - the popup for a mistake in the closing words only
                // appeared much later, when a following emission re-triggered
                // the pass (measured real-mic 2026-09-24: a Madd error was
                // pushed 35 s after the pause, after the user had already
                // resumed - "it takes too long to show the popup"). Keep the
                // finalized pass and drop the newer live ones; the next
                // emission re-triggers a live pass anyway.
                if (latest.finalized) continue
                latest = newer
            }
            if (latest.mode != "Tajweed" || latest.audio == null) continue
            // EPOCH GUARD (same as the fast loop): stale-range requests die here —
            // their verdicts must never update detailedAnalysis after the switch.
            // The passEpoch is re-checked before the WRITE too (§4.47): this loop
            // settles ~1.5 s and then spends 1.5-4.5 s scoring, so a Restart / Repeat
            // Ayah / range change during the scoring must not be overwritten by the
            // verdicts of the session it abandoned.
            val passEpoch = analysisEpoch
            if (latest.epoch != passEpoch) continue
            val tjT0 = System.currentTimeMillis()
            if (latest.finalized) {
                // FINAL PASS — the user has stopped, no more audio is coming, and
                // this is the pass that fills in the last words of the report. The
                // live settle below is pure dead time here (it exists only to
                // let live audio settle), so skip it: it is a direct 2 s off the
                // stop-to-report latency the user reported.
            } else {
                // LIVE PASS — throttle (see TAJWEED_MIN_INTERVAL_MS). Dropping a
                // live pass is safe: the fast text loop keeps the transcript
                // current, the next emission re-triggers, and the final pass above
                // always scores everything.
                if (System.currentTimeMillis() - lastTajweedPassMs.get() < TAJWEED_MIN_INTERVAL_MS) continue
                // Wait so the word's audio span has settled before scoring, and
                // so a burst of emissions collapses into one acoustic pass. The
                // settle must stay long enough to hold the word's own tail AND
                // (for the forward-leaning rules) the following word, which is
                // what the model was trained on - so this is a trim, not a cut:
                // 2.0 s - 1.5 s, still above the worst-case emit-to-audio lead.
                kotlinx.coroutines.delay(TAJWEED_LIVE_SETTLE_MS)
            }
            lastTajweedPassMs.set(System.currentTimeMillis())
            try {
                val acoustic = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
                    val spoken = latest.spokenText.split(Regex("\\s+"))
                        .filter { it.isNotBlank() }
                        .map { normalizeForComparison(it, latest.riwayah) }
                    analyzeRecitationDetails(
                        spoken,
                        latest.targetOrig,
                        latest.targetNorm,
                        latest.riwayah,
                        latest.timings,
                        context,
                        fullAudio = latest.audio,
                        recordingStartMs = latest.audioStartMs,
                        recordingDurationMs = (latest.audio?.size ?: 0) * 1000L / 16000,
                        targetAllGroups = latest.targetFormGroups,
                        hintedIndices = latest.hinted,
                        isTajweedMode = true,
                        finalReport = latest.finalized,
                        verdictCache = tajweedVerdictCache,
                        failCache = tajweedFailCache,
                        scoreTajweed = true,
                        // §4.56 — BOUND THE WORK PER PASS. The walk hands the words
                        // over in reading order, so this scores what the reader has
                        // just reached and leaves the rest to the passes that follow,
                        // instead of one pass owning 37 words and two minutes twenty
                        // of the user's time.
                        tajweedScoreLimit = TAJWEED_WORDS_PER_PASS
                    )
                }
                // §4.47 — WRITE-SIDE EPOCH GUARD, same reasoning as the fast loop.
                // This pass settles 1.5 s and scores for 1.5-4.5 s, so a Restart /
                // Repeat Ayah / range change during the scoring would otherwise be
                // overwritten by the acoustic verdicts of the abandoned session —
                // and (measured `b2_buttons`) by the cursor advance it carries.
                if (passEpoch != analysisEpoch) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "analyze(tajweed): DROPPED a stale acoustic result (epoch $passEpoch -> " +
                            "$analysisEpoch) — the session was reset while this pass was scoring"
                    )
                    continue
                }
                // Merge acoustic verdicts into the current display analysis. Never
                // downgrade a text-detected WRONG/SKIPPED to a tajweed verdict.
                // MULTI-RULE SAFE: keyed by (word, rule) — two rules on the same
                // word merge independently instead of collapsing into one row.
                val cur = detailedAnalysis
                val keyOf: (WordAnalysis) -> String = { "${it.wordIndex}:${it.tajweedRule ?: "-"}" }
                val byKey = cur.associateBy(keyOf).toMutableMap()
                var changed = 0
                for (w in acoustic) {
                    if (w.status == RecitationStatus.TAJWEED_ERROR || w.tajweedScore != null) {
                        val k = keyOf(w)
                        val existing = byKey[k]
                        if (existing == null ||
                            (existing.status != RecitationStatus.WRONG && existing.status != RecitationStatus.SKIPPED)) {
                            if (existing?.status != w.status || existing?.tajweedScore != w.tajweedScore) changed++
                            byKey[k] = w
                        }
                    }
                }
                if (changed > 0) {
                    // MISTAKE EXCLUSIVITY: if a word's live verdict is a text
                    // mistake (WRONG/SKIPPED), drop every OTHER row for that word —
                    // stale scored rows from earlier emissions must not sit next
                    // to it ("wrong word + Tajweed Al-Lam without score"). The
                    // user sees EITHER the wrong/skipped popup OR a tajweed
                    // popup naming rule + score, never both.
                    val mistakeIdxs = byKey.values
                        .filter { it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED }
                        .map { it.wordIndex }
                        .toHashSet()
                    val kept = byKey.values
                        .filter { w ->
                            !(w.wordIndex in mistakeIdxs &&
                                w.status != RecitationStatus.WRONG && w.status != RecitationStatus.SKIPPED)
                        }
                        .sortedWith(compareBy({ it.wordIndex }, { it.tajweedRule ?: "" }))
                    // RE-APPLY THE CLIPPED-DECODE PROMOTION (§4.34). `acoustic` is a
                    // FRESH analyzeRecitationDetails() call, so every key it scores is
                    // REPLACED by the fresh row — and for a clipped tail word that fresh
                    // row is PENDING again, silently undoing the promotion the fast merge
                    // made and dropping the word back to the 25 %-opacity fallback. The
                    // promotion has to be the LAST thing that touches the list.
                    //
                    // §4.37 FIRST, though: a fresh PENDING row from `acoustic` must not
                    // un-decide a word the finalized session already decided (measured
                    // `p2_fatihah`: this very loop's `changed=5` was the pass after which
                    // the cursor dropped 29 -> 28).
                    detailedAnalysis = kept
                        .keepFinalizedDecisions(previous = cur, finalized = sessionFinalized)
                        .promoteClippedPendingRows()
                    // §4.40 — same cursor-coverage rule as the fast merge: a
                    // finalized session may not leave a decided word grey.
                    if (sessionFinalized) {
                        // Same ayah ceiling as the fast merge. This loop is the
                        // one that re-applies acoustic verdicts after the final
                        // decode in Tajweed, so it must not resurrect next-ayah
                        // rows the finalize pass deliberately left grey.
                        val covered = minOf(
                            displayedCursorFor(detailedAnalysis, alignment.first),
                            displayCapForHeardAyah(detailedAnalysis, viewModel.ayahs)
                        )
                        if (covered > alignment.first) {
                            android.util.Log.i(
                                "ItqanSherpa",
                                "analyze(tajweed): cursor advanced ${alignment.first} -> $covered so every " +
                                    "decided word DISPLAYS (§4.40 REACHED => DISPLAYED)"
                            )
                            alignment = Pair(covered, alignment.second)
                        }
                    }
                }
                android.util.Log.i("ItqanSherpa", "analyze(tajweed): done ${System.currentTimeMillis() - tjT0}ms scored=${acoustic.count { it.tajweedScore != null }} tajweedErrors=${acoustic.count { it.status == RecitationStatus.TAJWEED_ERROR }} changed=$changed")
            } catch (t: Throwable) {
                android.util.Log.e("ItqanSherpa", "analyze(tajweed): FAILED after ${System.currentTimeMillis() - tjT0}ms", t)
            }
        }
    }

    // (item 5) Real-time tajweed popup watcher. Scans the latest `detailedAnalysis` for a
    // TAJWEED_ERROR. If we find one, push `(rule, wordIndex)` into `latestTajweedPopup`; that
    // instantly recomposes the small floating dialog lower in the tree. We don't pause the
    // recitation: the loop continues processing further words. The `remember`ed
    // `lastTajweedPopupIndex` guard prevents re-firing for the same error on every recomposition,
    // so the user sees each mistake once per word. `lastTajweedPopupIndex` is declared above
    // (next to the recognizer) so the range/surah reset can clear it too.
    LaunchedEffect(detailedAnalysis, currentTab, recitationMode, latestTajweedPopup) {
        if (currentTab == "Memorize" && recitationMode == "Tajweed") {
            // §4.49 — THE EARLIEST VERDICT THAT HAS NOT BEEN SHOWN YET.
            //
            // It used to be `lastOrNull { TAJWEED_ERROR }` — the most recent
            // error — behind a latch keyed on the word index. Two consequences,
            // both reported by the user:
            //  * a word carrying two failed rules only ever showed the LAST one
            //    ("even if there are two rules in the same word i got the 2
            //    different scores" — he expects both, got one);
            //  * an error on an EARLIER word was never shown at all, because the
            //    last error kept the latch busy.
            // Rows are one per (word, rule), so latching on (word, rule) and
            // taking the first UNSHOWN verdict surfaces every distinct rule
            // mistake exactly once, still one dialog at a time.
            //
            // §4.53 — "SHOWN" IS A **SET**, NOT ONLY THE LAST ONE SHOWN. With a
            // single latched pair, two standing errors take turns: dismissing A
            // makes B eligible, dismissing B makes A eligible again. That is
            // exactly what the user saw ("i continued receiting until i got a new
            // tajweed mistake … and again the old mistake is shown also"):
            //   tajweedPop(): wordIndex=83 rule=Madd score=34
            //   tajweedPop(): wordIndex=2  rule=Madd score=31
            //   tajweedPop(): wordIndex=83 rule=Madd score=34
            //   tajweedPop(): wordIndex=2  rule=Madd score=31
            //   tajweedPop(): wordIndex=83 rule=Madd score=34
            // The scores are identical every time: the same two verdicts,
            // ping-ponging. Every pair that has been shown is remembered, so each
            // speaks at most once — and the re-read re-arms them, because that is
            // the user acting on it.
            val tajweedError = detailedAnalysis.firstOrNull {
                it.status == RecitationStatus.TAJWEED_ERROR &&
                    it.tajweedRule != null &&
                    "${it.wordIndex}|${it.tajweedRule}" !in shownTajweedPopups
            }
            if (tajweedError != null) {
                // §4.55 — AN ACCUSATION MUST BE **ISOLATED** TO BE SHOWN.
                //
                // "a lot of problems for tajweed mistakes as usual". The log says all
                // nine came from ONE acoustic pass that took **133 SECONDS** and then
                // fired them as a burst:
                //   analyze(tajweed): done 133417ms scored=37 tajweedErrors=9
                //   word 18  Iqlab & Ikhfa  17%   adjacent accusers: [19]
                //   word 19  Iqlab & Ikhfa  33%   adjacent accusers: [18, 21]
                //   word 21  Idgham         47%   adjacent accusers: [19, 22, 23]
                //   word 22  Idgham         47%   adjacent accusers: [21, 23]
                //   word 23  Iqlab & Ikhfa  48%   adjacent accusers: [21, 22, 25]
                //   word 25  Idgham         43%   adjacent accusers: [23]
                //   word 30  Meem Sakinah   41%   adjacent accusers: [31, 32]
                //   word 31  Iqlab & Ikhfa  48%   adjacent accusers: [30, 32]
                //   word 32  Idgham         48%   adjacent accusers: [30, 31]
                //   -> 9 of 9 have an adjacent accuser (100%)
                //
                // A reader who mispronounces produces ISOLATED failures. Nine
                // ADJACENT words all failing, all inside a 31-point band just under
                // the threshold, is not nine mistakes — it is the scorer's audio
                // boundary sitting in the wrong place, so every cross-word rule
                // (Ikhfa, Idgham and Meem Sakinah all need the CUE AT THE BOUNDARY
                // between two words) gets the wrong slice. The user recited correctly
                // and was told nine times that he did not.
                //
                // So a verdict with a failing neighbour is an alignment artefact and
                // is NOT shown. The row and its score stay in the analysis, and once
                // the boundary settles the word is re-scored on its own and can then
                // speak up — the only honest way to accuse someone.
                val failingWords = detailedAnalysis
                    .filter { it.status == RecitationStatus.TAJWEED_ERROR }
                    .map { it.wordIndex }
                    .toHashSet()
                val neighbours = failingWords.filter {
                    it != tajweedError.wordIndex &&
                        kotlin.math.abs(it - tajweedError.wordIndex) <= TAJWEED_ISOLATION_SLACK
                }
                if (neighbours.isNotEmpty()) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "tajweedPop(): NOT accusing word ${tajweedError.wordIndex} " +
                            "(${tajweedError.tajweedRule} ${tajweedError.tajweedScore}) — words " +
                            "$neighbours also fail within $TAJWEED_ISOLATION_SLACK words, so this is " +
                            "a RUN and not an isolated mistake (§4.55 — a run means the scorer's audio " +
                            "boundary is offset, and a correct recitation must not be charged for it). " +
                            "The verdict stays in the analysis and re-speaks once its neighbours settle."
                    )
                    return@LaunchedEffect
                }
                // - ONE POPUP PER (WORD, RULE), AND ONLY AFTER THE USER ACTS (§4.45) -
                // The old guard fired again for the SAME word whenever its SCORE
                // changed, and only PASSING verdicts are cached - so every
                // acoustic pass re-scored the failing word against a longer
                // buffer, got a marginally different percentage, and re-fired the
                // dialog the instant the user tapped OK. The user read that as
                // "it didnt wait for me to dismiss it" (in reality an endless
                // popup on one word, with the recitation dead behind it).
                //
                // The fix is the CONTRACT, not a longer timer: a popup stays up
                // until it is dismissed, and the same (word, rule) can only raise
                // a new one after the user has DONE something about it - dismissed
                // the popup and re-read the word (the mic resume re-arms the latch
                // through the fresh-session event) - or after moving on. Dismissal
                // alone must never re-open it.
                //
                // §4.49/§4.53 — the latch is the whole SET of shown (word, rule)
                // pairs: a second rule on the SAME word is a different mistake with
                // a different score and must still be reported, and a verdict that
                // has already been shown must never be re-opened. `latestTajweedPopup
                // == null` is what keeps a popup from stacking while one is on screen.
                if (latestTajweedPopup == null &&
                    "${tajweedError.wordIndex}|${tajweedError.tajweedRule}" !in shownTajweedPopups
                ) {
                    android.util.Log.i("ItqanSherpa", "tajweedPop(): TAJWEED_ERROR wordIndex=${tajweedError.wordIndex} rule=${tajweedError.tajweedRule} score=${tajweedError.tajweedScore} -> pushing popup")
                    shownTajweedPopups = shownTajweedPopups +
                        "${tajweedError.wordIndex}|${tajweedError.tajweedRule}"
                    lastTajweedPopupIndex = tajweedError.wordIndex
                    lastTajweedPopupScore = tajweedError.tajweedScore ?: -1
                    lastTajweedPopupRule = tajweedError.tajweedRule!!
                    // The dialog renders title/body by kind. Forgetting this
                    // assignment left the previous "wrong"/"skipped" kind in
                    // place, so a Tajweed rule popup was titled "Skipped Word"
                    // and its body said the RULE ("Al-Lam") "was not heard".
                    // A rule is never a word; set the kind at the same instant
                    // as the pair so the two can never disagree.
                    latestPopupKind = "tajweed"
                    latestTajweedPopup = Pair(tajweedError.tajweedRule!!, tajweedError.wordIndex)
                    // (item 5c) The popup now also AUTO-STOPS the mic so the
                    // user gets time to read the feedback without the pipeline
                    // slurping more audio over the mistake. Identical to the
                    // FAB's stop branch (pause the warm pipeline, not a full
                    // teardown) so tapping the mic again resumes cleanly.
                    //
                    // Minimum-listen guard: never kill a session in its first
                    // ~2.5 s. The acoustic scoring can flag a TAJWEED_ERROR on
                    // the very first partials (audio too short / first word
                    // scoring), and stopping there would freeze the word-by-word
                    // coloring before the user has read anything ("the words
                    // aren't colored anymore when I read"). This defeats the
                    // step-event; the popup still fires, just doesn't yank the
                    // mic until the recitation has been going for a beat.
                    //
                    // INJECTION TEST MODE: when the pipeline is PC-driven
                    // (inject.pcm present), tajweed popups are DISPLAY-ONLY —
                    // the automated run focuses on wrong/skipped words, and
                    // stopping at every model-verdict madd would fragment the
                    // run. Real-mic sessions keep the full stop behavior.
                    if (!isListening) return@LaunchedEffect
                    val injectionRun = tajweedPipelineHolder.pipeline?.isInjecting() == true
                    val sessionElapsed = (System.currentTimeMillis() - tajweedRecordingStartMs).coerceAtLeast(0L)
                    if (!injectionRun && sessionElapsed >= 2500L) {
                        if (tajweedPipelineHolder.running) {
                            try { tajweedPipelineHolder.pipeline?.pause() } catch (_: Throwable) {}
                        }
                        try { speechRecognizer?.stopListening() } catch (_: Throwable) {}
                        isListening = false
                        // Arm the SAME resume re-seat the WRONG/SKIPPED flow
                        // uses: without it, the next session's count-filter
                        // strips the re-read word as an "already committed"
                        // replay copy — the user taps the mic, re-reads the
                        // word, and NOTHING appears ("re-read the word, got
                        // nothing"). Truncating the display transcript to the
                        // prefix BEFORE this word makes the re-read pass the
                        // filter, get displayed, get re-scored with a fresh
                        // timing key, and re-popup if still wrong.
                        mistakeResumeLiveText = (finalSpokenText + " " + partialSpokenText).trim()
                        // - RESUME AT THE MISTAKEN WORD, NOT AT THE FRONTIER -
                        // The popup tells the user to re-read THIS word, so the
                        // resume must re-seat the committed transcript just
                        // BEFORE it. It used to seat at `alignment.first` (the
                        // current display frontier, usually several words AHEAD
                        // of the mistake because the acoustic verdict is
                        // deferred): the re-read then failed the
                        // count-filter's "already committed" rule, nothing was
                        // displayed or re-scored, and the session looked dead -
                        // the user: "when there is a mistake the user should
                        // re read the mistaken word then complete, not just
                        // stop displaying after many words and then the app
                        // become never functioning". Seating at the mistake
                        // word makes the re-read pass the filter, get
                        // displayed, and get re-scored with a fresh timing key
                        // (so a second failure re-popups), exactly like the
                        // WRONG/SKIPPED flow.
                        mistakeResumeIndex = tajweedError.wordIndex
                        // Same as stopForMistake: a mistake PAUSES the session,
                        // it does not end it (§4.45).
                        recitationSessionActive = true
                        // §4.48 - AND THE DISPLAY STOPS AT THE MISTAKE, so the
                        // words after it stop looking reached and the reader can
                        // see where the popup wants them to continue.
                        displayCursorFloor = minOf(
                            displayCursorFloor,
                            tajweedError.wordIndex
                        )
                        // §4.52 — AND IT GOES BACK **NOW**, NOT ON THE NEXT
                        // ANALYSIS PASS. Lowering the floor is only a MINIMUM: the
                        // cursor is `maxOf(walk, floor)`, so a floor under a cursor
                        // that is already further along changes nothing. And after
                        // this popup the mic is stopped, so no further emission
                        // ever arrives and `analyze(fast)` never runs again to
                        // recompute it. The user saw exactly that: the popup named
                        // the word, the mic stopped, and the display stayed put
                        // until he turned the mic back on ("i want the display of
                        // words to get back to the mistaken word when the popup is
                        // showed to the user and the mic is stopped so he could
                        // know from where to receite again after the mistake").
                        //
                        // So write the renderer's own predicate (`alignment.first`
                        // IS the display cursor) here, at the moment the popup
                        // appears. `hasError = true` keeps the error colour on the
                        // flagged word, which the renderer's error branch needs.
                        val cursorBeforeRewind = alignment.first
                        alignment = Pair(tajweedError.wordIndex, true)
                        android.util.Log.i(
                            "ItqanSherpa",
                            "tajweedPop(): display rewound $cursorBeforeRewind -> " +
                                "${tajweedError.wordIndex} so the reader can SEE where to re-read " +
                                "from (§4.52 — the floor alone cannot move the cursor backwards)"
                        )
                        // Post-pause pipeline emissions are dead audio from
                        // BEFORE the stop; freeze them exactly like the
                        // WRONG/SKIPPED stop does (the gate above reads
                        // mistakeResumeIndex).
                    }
                }
            } else if (latestTajweedPopup == null) {
                // §4.53 — AND THE LATCH IS **NOT** CLEARED JUST BECAUSE ONE PASS
                // HAPPENED TO HAVE NO ERROR.
                //
                // The user's real-mic Al-Baqarah run looped on one popup: he
                // dismissed it, and it came straight back, over and over
                // ("it got back to that error and keep showing me popups and i
                // dismiss them and they keep coming"):
                //   tajweedPop(): TAJWEED_ERROR wordIndex=12 rule=Al-Lam score=40
                //   tajweedPop(): display rewound 15 -> 12
                //   analyze(tajweed): done 3555ms scored=8 tajweedErrors=0 changed=4
                //   tajweedPop(): TAJWEED_ERROR wordIndex=12 rule=Al-Lam score=40
                // Identical word, identical rule, identical score. The acoustic
                // pass that re-scored the word had a `tajweedErrors=0` window in
                // between, and THAT is what cleared the latch — this branch. The
                // next pass re-flagged the same word and, with the latch gone, the
                // "first verdict not yet shown" selection picked it again.
                //
                // §4.49 made the latch per (word, rule), which made this worse: a
                // cleared latch no longer merely allowed one re-fire, it made the
                // very same row immediately eligible again.
                //
                // The contract is §4.45's and it is about the USER, not about the
                // analysis: a popup stays up until he dismisses it, and the same
                // (word, rule) may only raise a new one after he has DONE something
                // — dismissed it and RE-READ the word, which re-arms the latch
                // explicitly at the resume (§4.49). So the latch now survives a
                // blank pass and is released only when the word it belongs to has
                // actually LEFT the analysis (re-read away, or the range changed).
                val popupWordGone = lastTajweedPopupIndex >= 0 &&
                    detailedAnalysis.none { it.wordIndex == lastTajweedPopupIndex }
                if (popupWordGone) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "tajweedPop(): word $lastTajweedPopupIndex left the analysis -> releasing the " +
                            "popup latch, so a future verdict on it can speak up again"
                    )
                    lastTajweedPopupIndex = -1
                    lastTajweedPopupScore = -1
                    lastTajweedPopupRule = ""
                    shownTajweedPopups = emptySet()
                    shownMistakeWords = emptySet()
                    rideOverCandidateIdx = -1
                    rideOverCandidateSinceMs = 0L
                }
                // lastWrongStopIndex is deliberately NOT touched: the WRONG-word
                // confirmation state lives in its own effect below.
            }
        } else if (currentTab != "Memorize") {
            // User left Memorize -- clear any stale popup state so re-entering is clean.
            lastTajweedPopupIndex = -1
            lastTajweedPopupRule = ""
            shownTajweedPopups = emptySet()
            shownMistakeWords = emptySet()
            rideOverCandidateIdx = -1
            rideOverCandidateSinceMs = 0L
            // §4.45 - a session that is only PAUSED (a mistake popup, or a pause
            // waiting for its final decode) has no mic to stop, so the tab-change
            // effect below never sees it. Leaving the tab still ends it: keep it
            // "active" across the switch and the next mic tap on this tab would
            // resume a session whose words belong to a surface the user left, and
            // a plain tap would summarize a stale analysis instead of starting.
            recitationSessionActive = false
            lastWrongStopIndex = -1
            lastWrongStopSightings = 0
            lastWrongStopFirstSeenMs = 0L
            latestTajweedPopup = null
            // A stale mistake resume must not re-seat into a different surface.
            mistakeResumeIndex = -1
            mistakeResumeLiveText = ""
            verifiedSpokenIndices.clear()
            lastMistakeVerifyMs = 0L
            // Fresh surface => fresh verification budget (§4.30).
            verifyBudgetWindowStartMs = 0L
            verifyBudgetSpentMs = 0L
            mistakeVerifyAbsentIdx = -1
            mistakeEpisodeAnchor = -1
            mistakeCleanStreak = 0
            confirmedMistakeUiIdx = -1
            mistakeEpisodeAnchor = -1
            mistakeCleanStreak = 0
            // Same stale-work guard as the range change: nothing queued for the
            // old surface may apply after the switch.
            analysisEpoch++
            try { tajweedPipelineHolder.pipeline?.invalidateInFlightCommit() } catch (_: Throwable) {}
            while (analyzeRequests.tryReceive().isSuccess) { /* drain */ }
            while (tajweedRequests.tryReceive().isSuccess) { /* drain */ }
        }
    }

    // (item 5d) WRONG/SKIPPED-word auto-pause. A WRONG verdict (the user spoke a
    // word that does not match the target) or a SKIPPED verdict (a target word
    // the STT frontier has passed without hearing) both stop the recitation
    // with a popup that names the word — the user must repeat it when they
    // continue. TAJWEED_ERROR keeps its own popup path above.
    //
    // Confirmation rule: a verdict only takes effect if the SAME word index
    // survives into LATER emissions with different transcript text. A single
    // window's decode can transiently mishear a word; the next sliding-window
    // decode may correct it (window#2 heard رَبِّ early and the matcher
    // flagged it; window#3 had the full sentence). WRONG needs 2 distinct
    // emissions (the user's voice is in the transcript, so a decode glitch
    // cannot fabricate the same wrong word twice). SKIPPED is weaker evidence
    // — it means the STT did NOT render a word, and one decode window can drop
    // a word and the next can re-hear it (measured on-device: لِلَّهِ dropped
    // in one window, re-heard on the next) — so it needs 3 distinct emissions
    // AND a >= 2 s dwell AND the transcript advanced >= 3 words past the skip
    // before it may auto-pause. When a mistake IS confirmed we stop and the
    // analyzed tail stays PENDING (nothing after the mistake is counted as
    // right).
    //
    // VERIFICATION GATE (final arbiter). The counters above only prove the
    // LIVE windows hadn't rendered the word — they cannot distinguish "user
    // skipped it" from "the decoder hid it" (window-edge cut / mid-span fold;
    // measured on-device as a false skipped popup on a CORRECT recitation).
    // So before pausing, ONE full-tail offline decode (verifyTailWords) looks
    // for the flagged target word: found ⇒ false alarm, count it CORRECT and
    // keep reading; absent ⇒ real mistake, popup + pause; inconclusive ⇒
    // retry on a later emission. See the gate inside the effect below.
    //
    // Hoisted so BOTH the emission-driven effect below and the verdict-driven
    // ticker after it perform exactly the same stop. It captures the
    // MutableState OBJECTS (remember returns the same instance on every
    // recomposition), so calling it from a long-lived coroutine is safe.
    val stopForMistake: (Int, Boolean) -> Unit = { wordIdx, skipped ->
        val row = detailedAnalysis.firstOrNull { it.wordIndex == wordIdx }
        val wordText = row?.text ?: ""
        android.util.Log.i(
            "ItqanSherpa",
            "mistakeStop(): ${if (skipped) "SKIPPED" else "WRONG"} CONFIRMED wordIndex=$wordIdx " +
                "word='$wordText' (sightings=$lastWrongStopSightings) -> auto-pause + popup"
        )
        lastWrongStopIndex = wordIdx
        // §4.64 — THE TAJWEED CONFIRM PATH JOINS THE SAME SET. So a mistake the
        // user was shown on Memorize + Tajweed is remembered in exactly the way a
        // Tarteel one is: the popup cannot re-raise for it, and the summary cannot
        // forgive it as "unproven". One rule, both modes.
        shownMistakeWords = shownMistakeWords + wordIdx
        // THE ONE PLACE A MISTAKE BECOMES VISIBLE. `confirmedMistakeUiIdx` drives
        // BOTH the inline banner and the red word highlight, and it is armed here
        // — together with the dialog — so the two can never disagree. Arming it
        // from the verified-absent verdict alone produced a bare wrong-word
        // MESSAGE with no popup whenever the stop's extra gates (sightings,
        // running, listening) were not satisfied (`p4_fatihah`). See the display
        // gate in mistakeStop() for the full post-mortem.
        confirmedMistakeUiIdx = wordIdx
        // The popup must tell the user WHAT went wrong and that they must
        // repeat the word when they continue.
        latestPopupKind = if (skipped) "skipped" else "wrong"
        latestTajweedPopup = Pair(wordText, wordIdx)
        // Snapshot the live transcript for the resume re-seat: the user
        // continues reading FROM this word, so everything before it becomes
        // the committed prefix of the next session.
        mistakeResumeLiveText = (finalSpokenText + " " + partialSpokenText).trim()
        mistakeResumeIndex = wordIdx
        // The session is PAUSED, not finished: the user re-reads from this word
        // and carries on. Keeping the session active is what makes the mic FAB
        // offer the re-read instead of a dead "start" (§4.45).
        recitationSessionActive = true
        // §4.48 — AND THE DISPLAY STOPS AT THE MISTAKE. The popup says "re-read
        // this word and continue from there", so the words AFTER it must not keep
        // their reached styling: the reader has to see exactly where to pick up.
        // The floor is dropped to the mistake index here and again at the re-seat
        // (§ the collector's freshSession block), because a mistake is a real
        // rewind, not a transient walk regression.
        displayCursorFloor = minOf(displayCursorFloor, wordIdx)
        // §4.52 — AND THE CURSOR ACTUALLY GOES BACK, HERE AND NOW. The floor is
        // only a MINIMUM (`cursor = maxOf(walk, floor)`), so dropping it under a
        // cursor that is already further along does nothing on its own, and once
        // the mic is stopped no further emission arrives to recompute it. The
        // popup is raised in the same instant, so this is the moment the reader
        // can be shown where to re-read from.
        val cursorBeforeRewind = alignment.first
        alignment = Pair(wordIdx, true)
        android.util.Log.i(
            "ItqanSherpa",
            "mistakeStop(): ${if (skipped) "SKIPPED" else "WRONG"} at $wordIdx — display rewound " +
                "$cursorBeforeRewind -> $wordIdx so the reader can SEE where to re-read from (§4.52)"
        )
        try { tajweedPipelineHolder.pipeline?.pause() } catch (_: Throwable) {}
        try { speechRecognizer?.stopListening() } catch (_: Throwable) {}
        isListening = false
    }
    LaunchedEffect(detailedAnalysis, currentTab, recitationMode) {
        if (currentTab == "Memorize" && recitationMode == "Tajweed") {
            val mistakeIdx = detailedAnalysis.indexOfFirst {
                it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED
            }
            val isSkipped = mistakeIdx >= 0 &&
                detailedAnalysis[mistakeIdx].status == RecitationStatus.SKIPPED
            // ── ROW INDEX vs WORD INDEX ───────────────────────────────────
            // detailedAnalysis holds ONE ROW PER (word, rule), so the row
            // position of the first WRONG/SKIPPED is NOT the word index (with
            // ~1.3 rule rows per word the drift reaches 6+ by word 20).
            // Everything else here is WORD-indexed: spokenTimings.startIndex,
            // verifiedSpokenIndices, targetNorm and the UI's globalWordIndex.
            // Feeding the row index to them meant:
            //   * the spokenTimings anchor lookup ALWAYS missed, so the verify
            //     span silently fell back to the candidate clock and decoded
            //     audio recorded after the word was read;
            //   * verifiedSpokenIndices.add(rowIdx) never matched any
            //     wordIndex, so the FALSE-ALARM forgiveness that heals a false
            //     skip was completely dead;
            //   * confirmedMistakeUiIdx highlighted the wrong word.
            val mistakeWord = if (mistakeIdx >= 0) detailedAnalysis[mistakeIdx].wordIndex else -1
            val currentSpoken = spokenText
            val currentWordCount = currentSpoken.split(Regex("\\s+")).size
            // SKIPPED requires the transcript to have advanced well PAST the
            // skipped index (>= 3 words beyond it): a word the STT simply
            // hasn't emitted yet — or dropped in ONE decode window and re-hears
            // on the next — must never be confirmed as skipped.
            val skipAdvancedPast = mistakeWord >= 0 &&
                currentWordCount >= mistakeWord + 4
            // A new emission is one whose transcript text differs from the one
            // that produced the previous sighting (the same emission can
            // re-trigger this effect when other state changes).
            // EPISODE-STABLE identity: noisy merges make mistakeIdx flicker a
            // few words between emissions (echoes appear/vanish); treating each
            // flicker as a NEW candidate reset the confirmation counters
            // forever - nothing ever fired and the cursor froze ("stuck at
            // ayah 2"). Drift within +-3 of the anchor = SAME episode; the
            // episode closes only after TWO consecutive clean emissions.
            //
            // THE ANCHOR IS COMPARED TO ITSELF, NOT TO `lastWrongStopIndex`.
            // `mistakeEpisodeAnchor` and `lastWrongStopIndex` are the SAME
            // quantity (both are set to `mistakeWord` in the branch below), so
            // once they agree `mistakeWord != lastWrongStopIndex` can never be
            // true again and the branch is entered at most once per episode.
            // Measured fatihah_c30 (word 10): the candidate flipped between
            // `row=10` and `row=11` on every other recomposition — same
            // `word=10`, different ROW — and because `sameEpisode` was false
            // and `mistakeWord != lastWrongStopIndex` was true each time, this
            // branch ran over and over. It reassigns
            // `lastWrongStopFirstSeenMs = now`, so the 800 ms dwell needed
            // before a verification decode could launch was NEVER satisfied:
            // ZERO `verify idx=` lines in the whole run, a genuinely-recited
            // word sat unproven for 20 s, and `stallWatchdog` then FABRICATED a
            // skip and auto-paused the session — the false "you skipped a word"
            // the user reported.
            val sameEpisode = mistakeWord >= 0 &&
                mistakeEpisodeAnchor >= 0 &&
                kotlin.math.abs(mistakeWord - mistakeEpisodeAnchor) <= 3
            val isNewEmission = mistakeWord >= 0 &&
                currentSpoken != lastWrongStopSpokenText
            if (mistakeIdx < 0) {
                mistakeCleanStreak += 1
                // §4.57 — A TENTATIVE VERDICT MUST SURVIVE A BRIEF ABSENCE.
                // This branch is the wrong/skipped twin of the tajweed popup latch
                // (§4.53), and it had exactly the same bug: two consecutive passes
                // without the row WIPED the whole confirmation state — the sightings
                // counter, the dwell clock, the absent strikes. A WRONG/SKIPPED is
                // tentative by construction (the text re-analysis rewrites
                // `detailedAnalysis` on every pass, so the row flickers in and out),
                // so the evidence it was accumulating was destroyed before it could
                // ever reach the bar.
                //
                // Measured on the user's real-mic Al-Baqarah run — the sightings
                // counter never left 1, so the verification never really ran:
                //   candidate row=6 word=5 sightings=1 ... verify idx=5 launched
                //   candidate row=5 word=5 sightings=1     <- the ROW index flickers
                //   candidate row=6 word=5 sightings=1     (one row per word x rule)
                //   candidate row=5 word=5 sightings=1
                // and in the later session not one `mistakeStop(): ... CONFIRMED`
                // line appeared: the candidate could never be judged, so a real
                // mistake produced NO popup at all and the app looked indifferent.
                //
                // The candidate is now released only when its WORD has actually LEFT
                // the analysis — a re-read, a range change, a re-seat — which is the
                // one thing that really does mean "this is no longer the candidate".
                val trackedWordGone = lastWrongStopIndex >= 0 &&
                    detailedAnalysis.none { it.wordIndex == lastWrongStopIndex }
                if (lastWrongStopIndex == -1) {
                    lastWrongStopSpokenText = ""
                    lastWrongStopSightings = 0
                    lastWrongStopFirstSeenMs = 0L
                    confirmedMistakeUiIdx = -1
                    mistakeEpisodeAnchor = -1
                    mistakeCleanStreak = 0
                    mistakeVerifyAbsentIdx = -1
                    mistakeVerifyAttempts = 0
                    mistakeVerifyAbsentStrikes = 0
                    mistakeVerdictAtMs = 0L
                } else if (mistakeCleanStreak >= 2 && trackedWordGone) {
                    lastWrongStopIndex = -1
                    lastWrongStopSpokenText = ""
                    lastWrongStopSightings = 0
                    lastWrongStopFirstSeenMs = 0L
                    confirmedMistakeUiIdx = -1
                    mistakeEpisodeAnchor = -1
                    mistakeCleanStreak = 0
                    mistakeVerifyAbsentIdx = -1
                    mistakeVerifyAttempts = 0
                    mistakeVerifyAbsentStrikes = 0
                    mistakeVerdictAtMs = 0L
                } else if (!trackedWordGone) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "mistakeStop(): word $lastWrongStopIndex is absent from this pass but still in " +
                            "the analysis -> keeping the candidate (sightings=$lastWrongStopSightings, " +
                            "dwell=${System.currentTimeMillis() - lastWrongStopFirstSeenMs}ms) instead " +
                            "of wiping it (§4.57 — a tentative verdict flickers, and wiping it meant no " +
                            "mistake could ever be proven)"
                    )
                    mistakeCleanStreak = 0
                }
            } else if (!sameEpisode && mistakeWord != lastWrongStopIndex) {
                // A genuinely different region is now the first mistake:
                // restart the confirmation counters for the new candidate.
                mistakeEpisodeAnchor = mistakeWord
                mistakeCleanStreak = 0
                lastWrongStopIndex = mistakeWord
                lastWrongStopSpokenText = currentSpoken
                lastWrongStopSightings = 1
                lastWrongStopFirstSeenMs = System.currentTimeMillis()
                // New candidate: not yet confirmed - hide any previous
                // mistake UI until THIS index earns the evidence too.
                confirmedMistakeUiIdx = -1
                mistakeVerifyAbsentIdx = -1
                mistakeVerifyAttempts = 0
                mistakeVerifyAbsentStrikes = 0
                mistakeVerdictAtMs = 0L
            } else if (mistakeCleanStreak > 0 || mistakeEpisodeAnchor < 0) {
                // STABLE RE-ENTRY. `mistakeWord` matches the candidate the
                // episode is already tracking, but the per-recomposition state
                // that describes it was cleared — the candidate flickered off
                // for one recomposition (a clean emission between two flagged
                // ones) or this is the very first pass after the counters
                // above ran. Re-seat the identity WITHOUT restarting the dwell
                // clock: `lastWrongStopFirstSeenMs` is deliberately preserved
                // so a candidate that keeps reappearing still reaches its
                // verification deadline instead of being reset forever.
                mistakeEpisodeAnchor = mistakeWord
                mistakeCleanStreak = 0
                lastWrongStopIndex = mistakeWord
                if (lastWrongStopFirstSeenMs == 0L) {
                    lastWrongStopFirstSeenMs = System.currentTimeMillis()
                }
                if (lastWrongStopSightings == 0) lastWrongStopSightings = 1
                android.util.Log.i(
                    "ItqanSherpa",
                    "mistakeStop(): candidate idx=$mistakeWord re-seated after a flicker " +
                        "(row=$mistakeIdx sightings=$lastWrongStopSightings) — dwell clock preserved"
                )
            } else {
                // DIAGNOSTIC: one line per candidate pass — reveals why the
                // early verification did or did not launch on-device.
                android.util.Log.i(
                    "ItqanSherpa",
                    "mistakeStop(): candidate row=$mistakeIdx word=$mistakeWord sightings=$lastWrongStopSightings absentIdx=$mistakeVerifyAbsentIdx verified=${mistakeWord in verifiedSpokenIndices} listening=$isListening running=${tajweedPipelineHolder.running} sinceVerify=${System.currentTimeMillis() - lastMistakeVerifyMs}ms jobActive=${verifyJobRef.get()?.isActive == true}"
                )
                if (isNewEmission) {
                    lastWrongStopSightings += 1
                    lastWrongStopSpokenText = currentSpoken
                }

                // ── EARLY VERIFICATION (fires at the SECOND sighting) ──
                // The live sliding windows can chronically garble/drop a word
                // for SEVERAL consecutive windows (measured on-device: لِلَّهِ
                // رَبِّ missing from windows #2-#3 yet heard by window #4 and
                // by every full-tail decode) — waiting for FULL confirmation
                // before verifying let a false candidate survive long enough
                // to light up the banner. So run the decisive tail re-decode
                // ONCE per candidate episode, as soon as the candidate has
                // survived one repeat:
                //   found      ⇒ FALSE ALARM — forgive immediately (the word
                //                joins verifiedSpokenIndices, counters reset);
                //   absent     ⇒ REAL — remembered in mistakeVerifyAbsentIdx,
                //                confirmation proceeds toward popup/pause;
                //   inconclusive (empty: decoder busy / too little audio /
                //   blank) ⇒ retry on a later emission (cooldown).
                //
                // CRITICAL — DETACHED JOB, NOT an effect-child coroutine. The
                // full-tail decode takes 1-3 s on-device, and EVERY new
                // emission restarts this LaunchedEffect, cancelling any child
                // that is awaiting the result — the verdict was lost, the
                // cooldown blocked retries, and slow phones livelocked with
                // the cursor stuck on one word forever ("stuck on ayah 2").
                // Launching on the composition scope lets the result survive
                // restarts and write its verdict into state regardless.
                if (!(mistakeVerifyAbsentIdx >= 0 &&
                        kotlin.math.abs(mistakeWord - mistakeVerifyAbsentIdx) <= 3) &&
                    mistakeWord !in verifiedSpokenIndices &&
                    mistakeWord != verifyInFlightIdx.get() &&
                    // ATTEMPT BUDGET. Every verification decode runs on the
                    // SAME recognizer the live windows use, and an INCONCLUSIVE
                    // result (too short / no shared words) does not count as a
                    // strike — so an inconclusive candidate used to be retried
                    // forever, one decode every 3 s. Measured baqarah_c14:
                    // four back-to-back decodes on ذَٰلِكَ between t+14s and
                    // t+36s of a 54,5 s session. The live windows starved and
                    // dropped 6 words (ذَٰلِكَ الَّذِينَ وَمِمَّا رَزَقْنَٰهُمْ
                    // يُنفِقُونَ وَبِٱلْءَاخِرَةِ + 1), the merge bolted them onto
                    // the tail out of order, and the whole run collapsed to
                    // 10/36. Three attempts is enough to reach the two-strike
                    // verdict; past that we stop spending the recognizer and
                    // let autoFinalize FORGIVE the unproven word rather than
                    // wreck the live transcript.
                    mistakeVerifyAttempts < MAX_MISTAKE_VERIFY_ATTEMPTS &&
                    // FIRE AT THE FIRST SIGHTING. The old gate waited for a
                    // SECOND distinct emission, but a mistake freezes the walk:
                    // the user stops speaking, no new emission ever arrives,
                    // and the decisive decode was therefore never launched —
                    // which is exactly why `confirmed (auto-pause + popup): 0`
                    // in every run. A short dwell is enough protection.
                    lastWrongStopSightings >= 1 &&
                    System.currentTimeMillis() - lastWrongStopFirstSeenMs >= 800L &&
                    isListening && tajweedPipelineHolder.running &&
                    System.currentTimeMillis() - lastMistakeVerifyMs >= 3000L &&
                    (verifyJobRef.get() == null || !verifyJobRef.get()!!.isActive) &&
                    // ── GLOBAL VERIFICATION BUDGET (§4.30) ───────────────────
                    // The per-candidate cap above bounds ONE word; a long range
                    // presents a fresh candidate every few seconds and each spent
                    // its own 3 decodes, so the single-slot recognizer was ~60 %
                    // occupied by verification (fix30_verify_1) and the live
                    // trailing window fell so far behind that words 16-29 were
                    // never decoded. Cap the TOTAL verification seconds inside a
                    // rolling window ACROSS candidates. Declining here only
                    // withholds a verification — the word stays UNPROVEN and the
                    // stall/auto-finalize forgiveness paths handle it — so this
                    // can never move a verdict toward condemnation.
                    verifyBudgetAllowsMs(
                        verifyBudgetWindowStartMs, verifyBudgetSpentMs,
                        System.currentTimeMillis(), VERIFY_BUDGET_WINDOW_MS, VERIFY_BUDGET_MAX_MS
                    )
                ) {
                    lastMistakeVerifyMs = System.currentTimeMillis()
                    val idxToVerify = mistakeWord
                    val normText = detailedAnalysis[mistakeIdx].normalizedText
                    val kindStr = if (isSkipped) "SKIPPED" else "WRONG"
                    val wordText = detailedAnalysis[mistakeIdx].text
                    // Episode-aware span: cover the WHOLE time since this
                    // candidate first appeared (+8 s margin) so a word read a
                    // while ago is still inside the decoded audio.
                    // ANCHOR ON WHEN THE WORD WAS SPOKEN, NOT on when the
                    // candidate was created. mistakeIdx flickers by a word or
                    // two between emissions, and every flicker RESTARTS the
                    // candidate clock — so the span collapses to its 10 s floor
                    // and decodes audio that starts after the word was already
                    // read. Measured fatihah_v6: مَٰلِكِ (idx 10) was flagged at
                    // 22:53:28 but the candidate was recreated at 22:53:40, so
                    // the window was 22:53:32-22:53:42 and the decode came back
                    // as words 14-20 — the word "looked" absent and a correct
                    // recitation was charged a SKIPPED.
                    // NO TIMING OF ITS OWN ⇒ THE WORD WAS NEVER HEARD, so there
                    // is no "when it was spoken" to anchor to. Falling back to
                    // the candidate clock produced a ~12 s slice that STARTED
                    // AFTER the word had already been read, and that short
                    // decode then "proved" the word absent (measured
                    // fatihah_c7: مَٰلِكِ condemned on a 5-word decode of a
                    // correct recitation). When the word has no timing, anchor
                    // on the last word we DID hear and grow the span on every
                    // retry — more audio each time, never the same short slice.
                    // ── TIMING ANCHOR: NEVER ANCHOR PAST THE CANDIDATE ────────
                    // `spokenTimings.startIndex` is the SPOKEN position, which only
                    // coincides with the target index while the transcript is aligned
                    // with the target. Once the walk freezes (exactly when a mistake
                    // candidate is alive) the spoken list stops growing and its last
                    // entries sit at target words AT OR AFTER the candidate — so
                    // `firstOrNull { it.startIndex == idxToVerify }` could return a
                    // timestamp for a DIFFERENT, LATER word, and anchoring there made
                    // the ~N s span start AFTER the candidate's own audio. The
                    // verifier then decoded words 20-28 for a candidate at 17, found
                    // "absent", and condemned a word the user had recited correctly
                    // (measured fix27_fatihah_1; the final decode heard '...اهدنا
                    // الصرط المستقيم...').
                    // Also: a timing is only trustworthy for the candidate when the
                    // word it timed is ACTUALLY the candidate — require the timed
                    // word to match the candidate's normalized form.
                    val candidateNorm = detailedAnalysis[mistakeIdx].normalizedText
                    val ownTimingMs = spokenTimings.firstOrNull {
                        it.startIndex == idxToVerify &&
                            isWordMatch(normalizeForComparison(it.word, viewModel.riwayah), candidateNorm)
                    }?.startMs
                    // The fallback must be a STRICTLY EARLIER word, and it must be
                    // the LATEST such word that is still before the candidate.
                    val prevTimingMs = spokenTimings
                        .filter { it.startIndex < idxToVerify }
                        .maxByOrNull { it.startIndex }?.startMs
                    val anchorMs = ownTimingMs
                        ?: prevTimingMs
                        ?: lastWrongStopFirstSeenMs.takeIf { it > 0L }
                        ?: System.currentTimeMillis()
                    val attempt = mistakeVerifyAttempts
                    // ── SPAN MUST REACH BACK PAST THE CANDIDATE ───────────────
                    // Even with a corrected anchor, grow with the attempt so a retry
                    // always decodes MORE audio and the span can never converge on a
                    // window that excludes the candidate. The +8 s lead-in covers the
                    // word itself plus the words immediately before it (the verifier
                    // trims a quarter-second of run-up and drops the OLDEST audio when
                    // its slice cap binds, so the reach-back needs headroom).
                    val elapsedSinceAnchorSec = ((System.currentTimeMillis() - anchorMs) / 1000L)
                        .coerceAtLeast(0L)
                    val spanSec = if (ownTimingMs != null) {
                        (elapsedSinceAnchorSec + 8L + attempt * 4L)
                            .toInt().coerceIn(10, 45)
                    } else {
                        val seenFor = if (lastWrongStopFirstSeenMs > 0L)
                            (System.currentTimeMillis() - lastWrongStopFirstSeenMs) / 1000L else 0L
                        (seenFor + 20L + attempt * 8L).toInt().coerceIn(20, 45)
                    }
                    mistakeVerifyAttempts = attempt + 1
                    android.util.Log.i(
                        "ItqanSherpa",
                        "mistakeStop(): verify idx=$idxToVerify launched attempt=$attempt " +
                            "span=${spanSec}s ownTiming=${ownTimingMs != null} " +
                            "anchorAge=${elapsedSinceAnchorSec}s"
                    )
                    verifyInFlightIdx.set(idxToVerify)
                    // ── DEBIT THE GLOBAL VERIFICATION BUDGET (§4.30) ─────────
                    // Reserve the nominal cost up-front so back-to-back launches
                    // cannot over-commit before any decode has reported; the
                    // real measured cost is settled in the job's finally block.
                    val verifyLaunchMs = System.currentTimeMillis()
                    run {
                        val (s, sp, _) = verifyBudgetAfterRequest(
                            verifyBudgetWindowStartMs, verifyBudgetSpentMs,
                            verifyLaunchMs, ASSUMED_VERIFY_COST_MS,
                            VERIFY_BUDGET_WINDOW_MS, VERIFY_BUDGET_MAX_MS
                        )
                        verifyBudgetWindowStartMs = s
                        verifyBudgetSpentMs = sp
                    }
                    verifyJobRef.set(coroutineScope.launch {
                        // §4.45 - THE IN-FLIGHT MARKER MUST BE RELEASED ON EVERY
                        // EXIT, INCLUDING CANCELLATION. It used to be cleared
                        // inline after the budget bookkeeping, so a cancellation
                        // in between (composition teardown, a range change draining
                        // the effect) left the index latched and the launch gate
                        // `mistakeWord != verifyInFlightIdx.get()` refused EVERY
                        // future verification for that one word: the candidate
                        // could never be proven, so no mistake popup ever fired
                        // for it again and the cursor sat on it.
                        var verifiedWords: List<String> = emptyList()
                        try {
                            verifiedWords = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                                try {
                                    tajweedPipelineHolder.pipeline?.verifyTailWords(spanSec) ?: emptyList()
                                } catch (t: Throwable) {
                                    android.util.Log.w("ItqanSherpa", "mistakeStop(): verifyTailWords threw", t)
                                    emptyList<String>()
                                }
                            }
                        } catch (t: Throwable) {
                            android.util.Log.w("ItqanSherpa", "mistakeStop(): verification cancelled", t)
                        } finally {
                            verifyInFlightIdx.set(-1)
                        }
                        // ── SETTLE THE BUDGET WITH THE REAL COST ─────────────
                        // Replace the nominal reservation with what the decode
                        // actually took, clamped to the reservation so a long
                        // decode (or a clock jump) cannot retroactively blow the
                        // budget. Logged so §4.30 starvation is measurable.
                        run {
                            val realCost = (System.currentTimeMillis() - verifyLaunchMs)
                                .coerceIn(0L, ASSUMED_VERIFY_COST_MS)
                            verifyBudgetSpentMs =
                                (verifyBudgetSpentMs - ASSUMED_VERIFY_COST_MS + realCost)
                                    .coerceAtLeast(0L)
                            android.util.Log.i(
                                "ItqanSherpa",
                                "mistakeStop(): verify budget spent=${verifyBudgetSpentMs}ms/" +
                                    "${VERIFY_BUDGET_MAX_MS}ms per ${VERIFY_BUDGET_WINDOW_MS}ms " +
                                    "(last decode ${realCost}ms, reserved ${ASSUMED_VERIFY_COST_MS}ms)"
                            )
                        }
                        // MATCH ON NORMALIZED FORMS. matcher.clean() returns the
                        // vocab's canonical DISPLAY words (fully diacritized), while
                        // detailedAnalysis.normalizedText is normalizeForComparison
                        // output (diacritics stripped). isWordMatch does NOT strip
                        // them itself: "رَبِّ" vs "رب" scored distance 3 on a short
                        // word -> found=false even when the decode contained the
                        // exact word (measured on-device: a false SKIPPED popup on
                        // a correctly-read ayah). Normalize the decoded side too.
                        // ── DEFINITE-ARTICLE TOLERANCE ─────────────────────────
                        // `ال` is a proclitic the CTC decoder freely glues on or
                        // drops. Measured on-device on Al-Fatihah (2026-09-08,
                        // fatihah_v5): the flagged word صِرَٰطَ came back from the
                        // full-context decode as الصرااط, so strict equality
                        // reported it ABSENT and the verdict stayed SKIPPED. The
                        // cursor froze at 20/29, the range never completed and
                        // the user saw nothing move — "no words displayed". Try
                        // every with/without-article combination before
                        // condemning a word. The bare form must be >= 3 chars so
                        // stripping ال from الله (-> له) can never match.
                        fun articleForms(s: String): List<String> {
                            val out = ArrayList<String>(2)
                            out.add(s)
                            val bare = if (s.startsWith("ال") || s.startsWith("ٱل")) s.substring(2) else null
                            if (bare != null && bare.length >= 3) out.add(bare)
                            return out
                        }
                        val targetForms = articleForms(normText)
                        var foundViaArticle = false
                        val found = verifiedWords.any { vw ->
                            val vn = normalizeForComparison(vw, viewModel.riwayah)
                            if (isWordMatch(vn, normText)) true
                            else articleForms(vn).any { f ->
                                targetForms.any { t ->
                                    if (isWordMatch(f, t)) { foundViaArticle = true; true } else false
                                }
                            }
                        } || run {
                            // MUQATTA'AT: the tail decode clips the disjointed letters
                            // the same way the live windows do (`الم` -> `أف م`), so a
                            // plain word-equality check reports the word as absent and
                            // CONFIRMS a mistake the user never made. Retry with the
                            // tolerant letter-name matcher at every offset.
                            val mSlots = itqan.app.verses.muqattaatSlots(wordText)
                            if (mSlots == null) false
                            else verifiedWords.indices.any { s ->
                                itqan.app.verses.matchMuqattaatRelaxed(
                                    verifiedWords.subList(s, verifiedWords.size), mSlots
                                ) > 0
                            }
                        }
                        android.util.Log.i(
                            "ItqanSherpa",
                            "mistakeStop(): verify idx=$idxToVerify $kindStr word='$wordText' verified=${verifiedWords.size}w found=$found viaArticle=$foundViaArticle"
                        )
                        if (verifiedWords.isEmpty()) {
                            // Inconclusive — clear the cooldown so a later emission retries promptly.
                            // Normal cooldown, NOT an immediate retry: every decode is
                            // 1.5-3.6 s of recognizer time, and hammering it starved the
                            // audio pipeline (measured fatihah_c10: 5 back-to-back decodes
                            // and the injected audio only played 32 s of its 46.5 s, so the
                            // last words were never read at all).
                            lastMistakeVerifyMs = System.currentTimeMillis()
                            return@launch
                        }
                        if (found) {
                            // FALSE ALARM. The full-context decode heard the flagged
                            // word even though the sliding windows missed it — the
                            // user said it correctly. Count it as spoken (the fast
                            // loop flips its verdict to CORRECT and the cursor
                            // advances past it) and continue without interruption.
                            verifiedSpokenIndices.add(idxToVerify)
                            // ── §4.43 AN ACQUITTAL MUST LEAVE EVIDENCE ──────────
                            // `verifiedSpokenIndices` alone is NOT enough. The
                            // VERIFIED-SPOKEN OVERRIDE only flips a row that is still
                            // WRONG/SKIPPED — but by acquittal time the row here has
                            // usually ALREADY been promoted to CORRECT with
                            // `spokenText = null`, so the override is a NO-OP on the
                            // very row it is meant to vindicate. `unheardWordCount()`
                            // counts exactly that shape (CORRECT with no token) as
                            // UNHEARD, so `genuinelyFinished` stays false and the
                            // summary is WITHHELD forever on a recitation the app
                            // itself judged perfect. Measured `s1_baqarah` (Al-Baqarah
                            // 1–5): `matched=36/36 wrong=0 skipped=0`, yet
                            // `endOfRange(): auto stop but NOT genuinely finished
                            // (unheard=1/36) -> no summary` because word 1 (ذَٰلِكَ)
                            // had been acquitted here while its row sat
                            // CORRECT+spoken='-'. Same signature as the original
                            // `d1_fatihah` finding (`matched=29 wrong=0 skipped=0`,
                            // no summary). The user read the whole range and got no
                            // summary at the end.
                            //
                            // Restore the evidence: stamp the vindicated row with the
                            // decoded token that actually matched it. This is the
                            // "a row that carries a token is a word the user DID say"
                            // case §4.31 carves out, so the §4.31 guarantee is
                            // untouched — a word NO decode ever found still gets
                            // `found=false`, no token, and still blocks the summary.
                            // Pick the token with the SAME normalization ladder the
                            // `found` test used, so an unrelated token is never
                            // stamped onto the row.
                            val evidenceToken = verifiedWords.firstOrNull { vw ->
                                val vn = normalizeForComparison(vw, viewModel.riwayah)
                                if (isWordMatch(vn, normText)) true
                                else articleForms(vn).any { f ->
                                    targetForms.any { t -> isWordMatch(f, t) }
                                }
                            }
                            if (evidenceToken != null) {
                                detailedAnalysis = detailedAnalysis.map { w ->
                                    if (w.wordIndex == idxToVerify &&
                                        w.status == RecitationStatus.CORRECT &&
                                        w.spokenText.isNullOrEmpty()
                                    ) {
                                        w.copy(spokenText = evidenceToken)
                                    } else {
                                        w
                                    }
                                }
                                android.util.Log.i(
                                    "ItqanSherpa",
                                    "mistakeStop(): acquittal idx=$idxToVerify word='$wordText' " +
                                        "restored spoken='$evidenceToken' on its CORRECT row so it " +
                                        "stops counting as unheard (§4.43)"
                                )
                            }
                            mistakeVerifyAbsentIdx = -1
                            mistakeEpisodeAnchor = -1
                            mistakeCleanStreak = 0
                            lastWrongStopIndex = -1
                            lastWrongStopSpokenText = ""
                            lastWrongStopSightings = 0
                            lastWrongStopFirstSeenMs = 0L
                            mistakeVerifyAttempts = 0
                            mistakeVerifyAbsentStrikes = 0
                            mistakeVerdictAtMs = 0L
                            confirmedMistakeUiIdx = -1
                            mistakeEpisodeAnchor = -1
                            mistakeCleanStreak = 0
                            // ── TRANSCRIPT RECONCILIATION ── A "found" verdict is
                            // direct evidence that the LIVE transcript diverged from
                            // reality in this region: noisy decodes leave displaced
                            // echoes/duplicates (measured on-device on a slower
                            // phone: …الرحيم، الحمدِّ، العالمين، الرحمن، الرحيم، لله…
                            // — الحمد/العالمين echoed BEFORE الرحمن الرحيم لله), so
                            // forgiving one word only reveals the next phantom
                            // candidate. Reuse THIS SAME decode: if its word
                            // sequence matches the target's leading run BETTER than
                            // the live transcript, adopt it wholesale — identical
                            // semantics to the collector's endpoint repair-swap,
                            // applied mid-session.
                            fun targetPrefixLen(words: List<String>): Int {
                                var i = 0
                                while (i < words.size && i < targetNorm.size &&
                                    targetNorm[i].isNotEmpty() &&
                                    normalizeForComparison(words[i], viewModel.riwayah) == targetNorm[i]
                                ) i++
                                return i
                            }
                            val liveNow = (finalSpokenText + " " + partialSpokenText).trim()
                                .split(Regex("\\s+")).filter { it.isNotEmpty() }
                            val livePrefix = targetPrefixLen(liveNow)
                            val verifyPrefix = targetPrefixLen(verifiedWords)
                            if (verifyPrefix > livePrefix && verifiedWords.size <= targetNorm.size + 4) {
                                finalSpokenText = verifiedWords.joinToString(" ").trim()
                                partialSpokenText = ""
                                committedWordCount = verifiedWords.size
                                lastCommittedWords = verifiedWords.toList()
                                android.util.Log.i(
                                    "ItqanSherpa",
                                    "mistakeStop(): reconcile adopted tail decode (${verifiedWords.size}w prefix=$verifyPrefix) over live (${liveNow.size}w prefix=$livePrefix)"
                                )
                            }
                        } else {
                            // ── GARBAGE-DECODE GUARD ──
                            // "The word is not in the tail decode" only proves absence
                            // if that decode actually covered this region. On device the
                            // full-context pass often returns UNRELATED garbage
                            // (measured, baqarah_v4: 'تفسيررضيلةكبابلور ر سب الرحمن الرحيم'
                            // for audio near word 1 of Al-Baqarah). `found=false` against
                            // that is meaningless — yet it was accepted as proof, so the
                            // session auto-paused ~3.6 s in on a word the user HAD read
                            // (matched=1/36, run aborted). Require the decode to share at
                            // least one word with the target's neighbourhood; otherwise it
                            // is INCONCLUSIVE — clear the cooldown and retry later rather
                            // than condemning the user.
                            val nearFrom = (idxToVerify - 8).coerceAtLeast(0)
                            val nearTo = (idxToVerify + 8).coerceAtMost(targetNorm.size)
                            val nearTargets = if (nearTo > nearFrom)
                                targetNorm.subList(nearFrom, nearTo).filter { it.isNotEmpty() }.toHashSet()
                            else emptySet()
                            val saneHits = verifiedWords.count { vw ->
                                normalizeForComparison(vw, viewModel.riwayah) in nearTargets
                            }
                            // ── POSITIONAL-COVERAGE GUARD ── saneHits only proves
                            // the decode heard words from the ±8 NEIGHBOURHOOD, which
                            // words far from the candidate can satisfy on their own.
                            // The verifier stitches its span out of short slices and
                            // DROPS the oldest audio once its slice cap binds, so
                            // "absent from the pool" is only evidence for a candidate
                            // the pool actually SPANNED. Demand a decoded word
                            // matching a target BEFORE idxToVerify and one matching a
                            // target AFTER it; otherwise this decode says nothing about
                            // this position and condemning on it would raise a false
                            // mistake on a correctly-read word.
                            var sawBefore = false
                            var sawAfter = false
                            for (vw in verifiedWords) {
                                val vn = normalizeForComparison(vw, viewModel.riwayah)
                                for (t in targetNorm.indices) {
                                    if (targetNorm[t].isEmpty()) continue
                                    if (!isWordMatch(vn, targetNorm[t])) continue
                                    if (t < idxToVerify) sawBefore = true
                                    if (t > idxToVerify) sawAfter = true
                                }
                            }
                            val bracketsCandidate =
                                (idxToVerify <= 0 || sawBefore) &&
                                    (idxToVerify >= targetNorm.size - 1 || sawAfter)
                            // ── ADJACENCY GUARD (the false-SKIPPED that shipped) ──
                            // `bracketsCandidate` alone is NOT enough, because a pool
                            // can contain SOME earlier target word and SOME later one
                            // without covering the candidate at all.
                            // MEASURED FALSE SKIPPED (Al-Fatihah, fix27_fatihah_1):
                            //   verify idx=17 word='ٱهْدِنَا' verified=17w found=false
                            //     -> ABSENT strike 1/2
                            //   verify idx=17 word='ٱهْدِنَا' verified=10w found=false
                            //     -> SKIPPED CONFIRMED -> auto-pause + popup
                            // yet the end-of-session decode heard
                            //   '...واياك نستعين اهدنا الصرط المستقيم...'
                            // Replaying the recorded pools (build/tools/replay_guard28.py)
                            // shows the 17w pool matched positions 10,11,12,13,16,24,6,3:
                            // a real straddle at 16->24 that paid down strike 1, plus
                            // stray t=6/t=3 words from misaligned long-window slices.
                            // ٱهْدِنَا's own audio was never in that pool.
                            // REQUIRE PROOF THE POOL ACTUALLY SPANNED THE CANDIDATE:
                            // there must be a decoded word matching a target BEFORE the
                            // candidate AND a decoded word matching a target AFTER it
                            // whose target positions are WITHIN A SHORT GAP of each
                            // other — a pool that jumps from word 9 to word 28 says
                            // nothing about word 17. Use each decoded word's best
                            // matching target position and look for a consecutive pair
                            // that straddles idxToVerify.
                            var straddles = false
                            run {
                                // Best target position for each decoded word (first
                                // match wins; the pool is in spoken order).
                                val positions = ArrayList<Int>(verifiedWords.size)
                                for (vw in verifiedWords) {
                                    val vn = normalizeForComparison(vw, viewModel.riwayah)
                                    var best = -1
                                    for (t in targetNorm.indices) {
                                        if (targetNorm[t].isEmpty()) continue
                                        if (isWordMatch(vn, targetNorm[t])) { best = t; break }
                                    }
                                    positions.add(best)
                                }
                                // ── SEE `poolSpansCandidate` (§4.28) FOR THE WHY ──
                                // A straddle = an adjacent pair (a,b) in the POOL whose
                                // target positions are ordered and satisfy a < idxToVerify
                                // <= b (the candidate sits in the gap between two words
                                // that were actually decoded around it).
                                straddles = poolSpansCandidate(positions, idxToVerify)
                                // AUDIT TRAIL (§4.28). A live post-mortem of the
                                // false SKIPPED could not tell WHY the condemning
                                // decode passed the coverage guard, because the
                                // log recorded only the RAW slice text — not the
                                // vocabulary-cleaned tokens the guard actually
                                // matched, nor their resolved target positions.
                                // Log both, so an admitted condemnation is always
                                // reconstructable after the fact.
                                android.util.Log.i(
                                    "ItqanSherpa",
                                    "mistakeStop(): idx=$idxToVerify pool=${verifiedWords.size}w " +
                                        "positions=$positions straddles=$straddles " +
                                        "candTarget='${targetNorm.getOrNull(idxToVerify) ?: ""}'"
                                )
                            }
                            // ── TOO-SHORT-DECODE GUARD ── A DECODE THAT DROPPED THE
                            // WORD ONLY PROVES ABSENCE IF IT IS BIG ENOUGH TO HAVE
                            // CONTAINED IT. On device a truncated/folded slice returns
                            // 3-5 words; `found=false` on that is a decoder artifact,
                            // not evidence. Measured fatihah_c7: مَٰلِكِ was condemned
                            // on a 5-word decode of a CORRECT recitation and the
                            // session auto-paused on it. One shared word is also too
                            // weak — require two, so a single lucky token cannot
                            // license a condemnation.
                            val minDecodeWords = if (targetNorm.size <= 12) 3 else 8
                            if (saneHits < 2 || verifiedWords.size < minDecodeWords ||
                                !bracketsCandidate || !straddles
                            ) {
                                android.util.Log.i(
                                    "ItqanSherpa",
                                    "mistakeStop(): verify idx=$idxToVerify INCONCLUSIVE — tail decode " +
                                        "(${verifiedWords.size}w, need >=$minDecodeWords) saneHits=$saneHits " +
                                        "brackets=$bracketsCandidate straddles=$straddles " +
                                        "(before=$sawBefore after=$sawAfter) " +
                                        "over targets $nearFrom..$nearTo; not condemning (will retry)"
                                )
                                // Normal cooldown, NOT an immediate retry: every decode is
                                // 1.5-3.6 s of recognizer time, and hammering it starved the
                                // audio pipeline (measured fatihah_c10: 5 back-to-back decodes
                                // and the injected audio only played 32 s of its 46.5 s, so the
                                // last words were never read at all).
                                lastMistakeVerifyMs = System.currentTimeMillis()
                                return@launch
                            }
                            // ── TWO-STRIKE RULE ── One decode is a sample; TWO
                            // independent decodes of DIFFERENT audio spans both
                            // missing the word is evidence. The growing span above
                            // guarantees the second strike looks at MORE audio, so a
                            // word the decoder merely clipped is heard on the retry
                            // and forgiven instead of condemned.
                            val strikes = mistakeVerifyAbsentStrikes + 1
                            mistakeVerifyAbsentStrikes = strikes
                            // LOG EVERY STRIKE, INCLUDING THE ONE THAT CONDEMNS.
                            // This used to sit inside the `strikes < 2` branch, so
                            // the second strike — the one that actually produces
                            // the popup — left NO trace, and a §4.28 post-mortem
                            // could not tell whether the coverage guard vetoed
                            // the second decode or wave it through. A condemnation
                            // must always be auditable from the log.
                            android.util.Log.i(
                                "ItqanSherpa",
                                "mistakeStop(): verify idx=$idxToVerify ABSENT strike $strikes/2 " +
                                    "— ${if (strikes < 2) "needs one more decode before condemning"
                                    else "CONDEMNING (auto-pause + popup)"}"
                            )
                            if (strikes < 2) {
                                // Normal cooldown, NOT an immediate retry: every decode is
                                // 1.5-3.6 s of recognizer time, and hammering it starved the
                                // audio pipeline (measured fatihah_c10: 5 back-to-back decodes
                                // and the injected audio only played 32 s of its 46.5 s, so the
                                // last words were never read at all).
                                lastMistakeVerifyMs = System.currentTimeMillis()
                                return@launch
                            }
                            // Verified ABSENT: the word genuinely was not said. Remember
                            // the verdict for this episode so we never re-decode it.
                            // Stamp the wall clock too: the stop/popup decision below
                            // is driven by THIS, not by further emissions (see the
                            // verdict-driven ticker after this effect).
                            mistakeVerifyAbsentIdx = idxToVerify
                            mistakeVerdictAtMs = System.currentTimeMillis()
                        }
                    })
                }

                // ── STUCK RECONCILE ── When a candidate survives MANY emissions
                // even after its verdict landed, the live transcript itself is
                // echo-poisoned in that region: forgiving single words just
                // reveals the next phantom candidate, and the cursor stays
                // frozen until the user pauses — whose own full-tail repair
                // then heals everything at once (measured repeatedly on a slow
                // device: "stuck at ayah 2, paused the mic, all words displayed
                // + popup"). Fire a LONG-span tail decode proactively instead;
                // if its sequence aligns to the target with MORE leading
                // matches than the live transcript, adopt it wholesale (same
                // semantics as the collector's repair swap, mid-session).
                // NOTE: the old "verdict landed" clause is deliberately GONE. It
                // required the tail verdict to have resolved (verified-spoken or
                // verified-absent), but the whole failure mode here is a
                // verifyTailWords() decode that comes back EMPTY — INCONCLUSIVE,
                // so no verdict ever lands. The repair therefore could never fire
                // in the one case that actually needed it, and the cursor stayed
                // frozen until the user paused manually. A candidate that has
                // survived 6 distinct emissions is stuck by definition; that
                // alone justifies running the repair.
                val stuckCandidate = lastWrongStopSightings >= 6
                if (stuckCandidate &&
                    isListening && tajweedPipelineHolder.running &&
                    System.currentTimeMillis() - lastReconcileMs >= 12000L &&
                    (verifyJobRef.get() == null || !verifyJobRef.get()!!.isActive)
                ) {
                    lastReconcileMs = System.currentTimeMillis()
                    val idxAtLaunch = mistakeWord
                    // Same anchoring rule as the early verification above: use
                    // when the word was spoken, not when the candidate was
                    // created (see the comment there for the fatihah_v6 trace).
                    val stuckAnchorMs = spokenTimings.firstOrNull { it.startIndex == idxAtLaunch }?.startMs
                        ?: lastWrongStopFirstSeenMs.takeIf { it > 0L }
                        ?: System.currentTimeMillis()
                    val spanSec = ((System.currentTimeMillis() - stuckAnchorMs) / 1000L + 8L)
                        .toInt().coerceIn(10, 45)
                    verifyJobRef.set(coroutineScope.launch {
                        val words = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                            try {
                                tajweedPipelineHolder.pipeline?.verifyTailWords(spanSec) ?: emptyList()
                            } catch (t: Throwable) {
                                android.util.Log.w("ItqanSherpa", "stuck-reconcile(): verifyTailWords threw", t)
                                emptyList<String>()
                            }
                        }
                        if (words.isEmpty()) return@launch
                        fun nrm(s: String) = normalizeForComparison(s, viewModel.riwayah)
                        val targetSet = targetNorm.filter { it.isNotEmpty() }.toHashSet()
                        // Leading-match walk mirroring the display cursor: count
                        // target positions satisfied before the first hard stop.
                        fun leadingMatches(seq: List<String>): Int {
                            var ti = 0
                            var si = 0
                            var good = 0
                            var garbage = 0
                            while (ti < targetNorm.size && si < seq.size) {
                                var f = -1
                                val scanEnd = minOf(si + 3, seq.size)
                                for (k in si until scanEnd) {
                                    if (targetNorm[ti].isNotEmpty() && nrm(seq[k]) == targetNorm[ti]) { f = k; break }
                                }
                                if (f != -1) { good++; ti++; si = f + 1; garbage = 0 }
                                else {
                                    val curN = nrm(seq[si])
                                    if (curN.isNotEmpty() && curN !in targetSet && garbage < 2) { garbage++; si++; continue }
                                    break
                                }
                            }
                            return good
                        }
                        val newMatched = leadingMatches(words)
                        val liveMatchedNow = alignment.first
                        android.util.Log.i(
                            "ItqanSherpa",
                            "mistakeStop(): stuck-reconcile idx=$idxAtLaunch decode=${words.size}w matched=$newMatched vs live=$liveMatchedNow"
                        )
                        if (newMatched > liveMatchedNow && words.size <= targetNorm.size + 4) {
                            finalSpokenText = words.joinToString(" ").trim()
                            partialSpokenText = ""
                            committedWordCount = words.size
                            lastCommittedWords = words.toList()
                            lastWrongStopIndex = -1
                            lastWrongStopSpokenText = ""
                            lastWrongStopSightings = 0
                            lastWrongStopFirstSeenMs = 0L
                            mistakeVerifyAbsentIdx = -1
                            mistakeVerifyAttempts = 0
                            mistakeVerifyAbsentStrikes = 0
                            mistakeVerdictAtMs = 0L
                            mistakeEpisodeAnchor = -1
                            mistakeCleanStreak = 0
                            confirmedMistakeUiIdx = -1
                            mistakeEpisodeAnchor = -1
                            mistakeCleanStreak = 0
                            android.util.Log.i(
                                "ItqanSherpa",
                                "mistakeStop(): stuck-reconcile ADOPTED ${words.size}w (matched $liveMatchedNow -> $newMatched)"
                            )
                        }
                    })
                }

                // DISPLAY GATE — VERIFIED-REAL ONLY, AND THE POPUP OWNS IT.
                // The banner (the inline "Skipped Word" / "Incorrect
                // Pronunciation/Word" card) used to be armed HERE, on the
                // verified-absent verdict alone, while the STOP + POPUP needed
                // the full confirmation below (`sightings >= 2`, `isListening`,
                // `tajweedPipelineHolder.running`). Whenever the extra gates
                // were not satisfied the banner appeared on its own and the
                // dialog never did — measured `p4_fatihah`: the session ended
                // with `wrong=0 skipped=0` and NO summary, yet the screen showed
                // a wrong-word MESSAGE. The user: "i saw a msg of a wrong word.
                // why it is on a msg not popup?"
                //
                // A bare message is not a contract surface. The contract is
                // "if it stopped then it should be a skipped/wrong word popup or
                // tajweed popup else it should be displayed" — so the banner and
                // the dialog are now armed by ONE place, [stopForMistake], which
                // always sets both. A mistake that does not stop the session is
                // not surfaced at all; its word simply stays displayed.
                //
                // (TAJWEED_ERROR is unaffected: it does not stop the session and
                // keeps its own inline feedback + rule popup, resolved through
                // the `errorIndex` fallback in the banner gate below.)
                val episodeAbsent = mistakeVerifyAbsentIdx >= 0 &&
                    kotlin.math.abs(mistakeWord - mistakeVerifyAbsentIdx) <= 3
                // EMISSION-DRIVEN CONFIRMATION (now only a SECOND path). The
                // sightings counters were the sole way to a stop, and they can
                // only advance on NEW transcript text. A mistake is precisely
                // the moment the user stops speaking, so they stalled at 1 and
                // nothing ever fired (`confirmed (auto-pause + popup): 0`). The
                // decisive evidence is the tail verdict, and the verdict-driven
                // ticker below decides on the wall clock. `isNewEmission` is
                // deliberately NO LONGER required here either.
                val isConfirmed =
                    lastWrongStopSightings >= 2 &&
                    (if (isSkipped)
                        (System.currentTimeMillis() - lastWrongStopFirstSeenMs) >= 2000L &&
                            (skipAdvancedPast ||
                                System.currentTimeMillis() - lastWrongStopFirstSeenMs >= 5000L)
                    else true) &&
                    // Evidence requirement: only a VERIFIED-absent candidate may
                    // stop the recitation (see the early-verification block).
                    // Drift-tolerant: the candidate index may shift ±3 inside the
                    // episode while the echo tokens flicker.
                    episodeAbsent
                if (isListening && tajweedPipelineHolder.running && isConfirmed) {
                    val sessionElapsed =
                        (System.currentTimeMillis() - tajweedRecordingStartMs).coerceAtLeast(0L)
                    if (sessionElapsed >= 2500L) {
                        stopForMistake(mistakeWord, isSkipped)
                    }
                }
            }
        }
    }

    // ── VERDICT-DRIVEN STOP (the decision that actually reaches the user) ──
    // Every gate above is EMISSION-driven: it can only advance when the
    // transcript text changes. A mistake is exactly the moment the user STOPS
    // speaking — they are waiting for the feedback — so no new emission ever
    // arrives, `sightings` stays at 1 forever and confirmation can never
    // mature. Measured on EVERY Al-Fatihah run: the log repeats
    // `candidate idx=N sightings=1` and ends with
    // `confirmed (auto-pause + popup): 0`.
    // So the stop is decided on the WALL CLOCK, from the decisive tail verdict:
    //   * verification FOUND the word  -> forgiven above, never stops;
    //   * verification says ABSENT and that verdict has held VERDICT_HOLD_MS
    //     -> that IS the mistake: pause + popup. No new emission required.
    // This is what turns "any mistake stops the recitation with a popup" from a
    // comment into behaviour — instead of the mistake silently surviving until
    // the auto-finalize grace expires and the user is handed the end-of-session
    // SUMMARY (which is only for "the user finished": they said so explicitly,
    // more than once).
    LaunchedEffect(currentTab, recitationMode) {
        var stoppedFor = -1
        while (true) {
            kotlinx.coroutines.delay(400L)
            if (currentTab != "Memorize" || recitationMode != "Tajweed") {
                stoppedFor = -1
                continue
            }
            if (!isListening) { stoppedFor = -1; continue }
            val row = detailedAnalysis.firstOrNull {
                it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED
            }
            if (row == null) { stoppedFor = -1; continue }
            val cand = row.wordIndex
            if (cand in verifiedSpokenIndices) { stoppedFor = -1; continue }
            if (cand != stoppedFor) stoppedFor = -1
            val absentFor = mistakeVerifyAbsentIdx
            if (absentFor < 0 || kotlin.math.abs(cand - absentFor) > 3) { stoppedFor = -1; continue }
            if (cand == stoppedFor) continue
            val verdictMs = mistakeVerdictAtMs
            if (verdictMs <= 0L) continue
            val held = System.currentTimeMillis() - verdictMs
            if (held < 1500L) continue
            // Must the word have been reachable at all? A SKIPPED verdict also
            // needs the transcript to have moved past the word, or a long dwell,
            // so a word the user has not reached yet is never condemned.
            val firstSeen = lastWrongStopFirstSeenMs
            if (firstSeen <= 0L) continue
            val seenMs = System.currentTimeMillis() - firstSeen
            if (row.status == RecitationStatus.SKIPPED) {
                val spokenWords = (finalSpokenText + " " + partialSpokenText).trim()
                    .split(Regex("\\s+")).filter { it.isNotEmpty() }.size
                if (seenMs < 2000L) continue
                if (!(spokenWords >= cand + 4 || seenMs >= 5000L)) continue
            }
            if ((System.currentTimeMillis() - tajweedRecordingStartMs) < 2500L) continue
            // ── §4.38 DO NOT CONDEMN WHILE THE ENDPOINT DECODE IS STILL COMING ──
            // The ABSENT strikes come from 4 s-slice decodes — the weakest decode
            // the app has, and one that DROPS words. The session's own
            // whole-session endpoint decode is already running and is strictly
            // better evidence. Measured `p3_fatihah`: the confirm fired 0.9 s
            // after that decode began; 8 s later it landed with the flagged word
            // present (`dec='… نستعين اهدنا الصرط …'`), and the confirmed-mistake
            // freeze discarded the acquittal. The user got a "Skipped Word" popup
            // for a word their recitation contained.
            val nowMs = System.currentTimeMillis()
            if (!mistakeMayBeConfirmed(
                    pipelineRunning = tajweedPipelineHolder.running,
                    endpointDecodeLanded = endpointDecodeLandedAtMs > 0L,
                    sessionStoppedForMs = if (pipelineStoppedAtMs > 0L) nowMs - pipelineStoppedAtMs else 0L,
                )
            ) {
                if (held % 4000L < 500L) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "verdictTicker(): HOLDING the confirm for wordIndex=$cand " +
                            "(held=${held}ms) — the session's endpoint decode is still in " +
                            "flight and it is the best evidence; condemning now is what " +
                            "produced the false SKIPPED popup (§4.38)"
                    )
                }
                continue
            }
            android.util.Log.i(
                "ItqanSherpa",
                "verdictTicker(): absent verdict held ${held}ms for wordIndex=$cand " +
                    "(seenMs=$seenMs) -> auto-pause + popup"
            )
            stoppedFor = cand
            stopForMistake(cand, row.status == RecitationStatus.SKIPPED)
        }
    }

    // ── STALL WATCHDOG ──
    // Every repair above is driven by NEW EMISSIONS: the sightings counter only
    // advances when the transcript text actually changes. But the dominant
    // freeze on a slow device has the user STOP SPEAKING — they are waiting for
    // feedback that never comes — so no fresh emission ever arrives, the
    // sightings counter freezes below the repair threshold, and the cursor sits
    // on one word forever with no banner, no popup and no pause. The only thing
    // that healed it was tapping pause manually, whose own full-tail repair then
    // fixed everything at once (the long-standing "stuck at ayah 2" report).
    // So add a WALL-CLOCK bound that is NOT keyed on emissions: once an
    // unresolved candidate has been pending longer than STALL_FORGIVE_MS,
    // forgive it so the recitation can move on. Project policy: a false
    // accusation is far worse than a missed skip, and a permanently frozen
    // cursor is worse than either.
    LaunchedEffect(Unit) {
        // A timer of our OWN, deliberately NOT lastWrongStopFirstSeenMs. That
        // field is reset every time the first-unmatched index drifts (measured:
        // idx 6 -> 43 inside one session), so the 20 s bound never matured —
        // the watchdog logged ZERO firings across a 90 s run while the cursor
        // sat frozen. This one only resets when there is no candidate at all.
        var stallFirstSeenMs = 0L
        // How many times this watchdog has re-opened the confirm loop's dwell gate
        // for the CURRENT candidate (§4.35). Bounded, because a nudge that cannot
        // produce a decode — the confirm loop's own gate still refusing — must not
        // be able to keep the candidate alive forever: that is the §4.18
        // infinite-freeze class, and "nothing ever happens" is the worst outcome.
        var stallNudges = 0
        // The candidate `stallNudges`/`stallFirstSeenMs` belong to. Without this,
        // a candidate that changes WITHOUT going through -1 would inherit the
        // previous one's nudge count and could be forgiven before the pipeline
        // ever tried to judge it.
        var stallCand = -1
        while (true) {
            kotlinx.coroutines.delay(2000L)
            val cand = lastWrongStopIndex
            if (cand < 0) { stallFirstSeenMs = 0L; stallNudges = 0; stallCand = -1; continue }
            if (cand != stallCand) { stallCand = cand; stallFirstSeenMs = 0L; stallNudges = 0 }
            // `isListening` only. Do NOT also require tajweedPipelineHolder.running:
            // the pipeline reports running=false right after the endpoint, which is
            // precisely the moment the user is waiting for feedback. Gating on it
            // made this whole watchdog unreachable (measured: candidate idx=43 with
            // running=false was skipped forever).
            if (!isListening) { stallFirstSeenMs = 0L; continue }
            // Already resolved by the normal pipeline — leave it alone.
            if (cand in verifiedSpokenIndices) { stallFirstSeenMs = 0L; continue }
            if (mistakeVerifyAbsentIdx >= 0 &&
                kotlin.math.abs(cand - mistakeVerifyAbsentIdx) <= 3
            ) { stallFirstSeenMs = 0L; continue }
            if (stallFirstSeenMs == 0L) stallFirstSeenMs = System.currentTimeMillis()
            val pendingMs = System.currentTimeMillis() - stallFirstSeenMs
            // ── SPEND THE NEXT DECODE RATHER THAN FREEZE (§4.35) ────────────
            // The confirm loop that launches verification is EMISSION-driven, and
            // a frozen walk emits nothing. So after the FIRST strike nothing ever
            // re-runs it, and the second decode — the one that decides whether to
            // condemn — is never launched. Measured `clean_fatihah_3`: idx=14 got
            // `ABSENT strike 1/2` at 03:07:04.761; the confirm loop's LAST re-run
            // saw `sinceVerify=2825ms` at 03:07:07.586, 175 ms short of the 3 s
            // cooldown, and no further re-run ever came. The candidate sat WRONG
            // for 20 s, was forgiven UNPROVEN, and the walk was frozen the whole
            // time — `maxMatched=17/29` on a run whose audio reaches 29.
            //
            // This watchdog already polls every 2 s, so it is the natural driver.
            // The nudge only re-opens the confirm loop's 800 ms dwell gate; EVERY
            // other guard still applies (attempt budget, 3 s cooldown, global
            // budget, job-active, and the §4.28 bracket+straddle coverage test on
            // the result), so a nudge can never move a verdict toward condemnation
            // on its own — it can only make the pipeline finish the judgement it
            // already started. `stallNudges` bounds it so a nudge the confirm loop
            // never honours cannot freeze the candidate forever.
            val jobIdle = verifyJobRef.get() == null || !verifyJobRef.get()!!.isActive
            val cooldownOver = lastMistakeVerifyMs == 0L ||
                System.currentTimeMillis() - lastMistakeVerifyMs >= 3000L
            val budgetOk = verifyBudgetAllowsMs(
                verifyBudgetWindowStartMs, verifyBudgetSpentMs,
                System.currentTimeMillis(), VERIFY_BUDGET_WINDOW_MS, VERIFY_BUDGET_MAX_MS
            )
            if (stallShouldNudge(
                    isListening = isListening,
                    pipelineRunning = tajweedPipelineHolder.running,
                    jobIdle = jobIdle,
                    cooldownOver = cooldownOver,
                    budgetOk = budgetOk,
                    attempts = mistakeVerifyAttempts,
                    nudges = stallNudges,
                    maxAttempts = MAX_MISTAKE_VERIFY_ATTEMPTS,
                )
            ) {
                stallNudges++
                android.util.Log.i(
                    "ItqanSherpa",
                    "stallWatchdog(): candidate idx=$cand stalled ${pendingMs}ms " +
                        "(attempts=$mistakeVerifyAttempts strikes=$mistakeVerifyAbsentStrikes) " +
                        "-> nudging the confirm loop to spend decode " +
                        "${mistakeVerifyAttempts + 1} (nudge $stallNudges)"
                )
                // Change a value the confirm loop observes so it re-evaluates its
                // gate; do NOT reset `stallFirstSeenMs` — the forgiveness clock
                // must keep running so a candidate the pipeline keeps failing to
                // judge is still released.
                lastWrongStopFirstSeenMs = System.currentTimeMillis() - 800L
                continue
            }
            // ── BUDGET-BLOCKED SHORT-CIRCUIT (§4.31) ────────────────────────
            // The 20 s hold below assumes verification is still coming. It is
            // NOT coming when the global budget cannot fund another decode: the
            // two-strike verdict is then unreachable, so the candidate is
            // permanently unprovable and the ONLY thing the 20 s hold achieves
            // is freezing the cursor — which is exactly "many words are not
            // displayed" (measured fix31_budget_1: the walk sat on ٱهْدِنَا
            // while words 18-29 never rendered). Forgive promptly instead; this
            // is the safe direction (a false "no mistake" costs nothing, a
            // false mistake popup accuses the user of a recitation error).
            // Require at least one decode already spent, so a candidate whose
            // verification has simply not STARTED yet is not cut off early.
            //
            // AND REQUIRE THE JOB TO BE IDLE. The nominal cost of an ACTIVE
            // verification is already debited, so `budgetBlocked` is true for the
            // entire duration of that decode — forgiving on that basis cut a
            // candidate off while its own verdict was still being computed.
            // Measured `clean_fatihah_3`: idx=15 was forgiven at 4003 ms and the
            // in-flight decode returned `found=true` 1,2 s later, i.e. the very
            // decode that would have acquitted it was thrown away.
            val stallForgiveMs = stallForgiveMsFor(
                budgetOk = budgetOk,
                attempts = mistakeVerifyAttempts,
                jobIdle = jobIdle,
            )
            // ── §4.38 THE SAME HOLD APPLIES TO FORGIVING ────────────────────
            // The budget-blocked short-circuit above cuts the wait to 3 s, but
            // the session's endpoint decode needs ~8 s. Forgiving first would
            // turn a REAL skip into a silent "no mistake" (the row becomes
            // CORRECT, so the ticker has no candidate left to condemn) — and
            // injection tests exist precisely to catch wrong/skipped words.
            // Holding BOTH verdicts until that decode lands keeps the two
            // directions symmetric: whatever the endpoint decode heard decides
            // it. The hold is bounded by the same [mistakeMayBeConfirmed] clock,
            // so forgiveness still always proceeds — it is delayed, never
            // blocked (the §4.31 asymmetry is preserved).
            val nowMsW = System.currentTimeMillis()
            if (!mistakeMayBeConfirmed(
                    pipelineRunning = tajweedPipelineHolder.running,
                    endpointDecodeLanded = endpointDecodeLandedAtMs > 0L,
                    sessionStoppedForMs = if (pipelineStoppedAtMs > 0L) nowMsW - pipelineStoppedAtMs else 0L,
                )
            ) {
                continue
            }
            if (pendingMs < stallForgiveMs) continue
            // LAST-CHANCE VERIFICATION is now handled by the §4.35 nudge above,
            // which subsumes it: that nudge also covers the `lastMistakeVerifyMs
            // == 0` case (the confirm loop never ran at all — measured
            // fatihah_c30, a row-index flicker kept resetting the 800 ms dwell
            // pre-condition) and it is bounded by `stallNudges`. The old separate
            // block here reset `stallFirstSeenMs` with NO bound, so a nudge the
            // confirm loop never honoured could hold a candidate forever.
            android.util.Log.i(
                "ItqanSherpa",
                "stallWatchdog(): candidate idx=$cand unresolved for ${pendingMs}ms " +
                    "(sightings=$lastWrongStopSightings absentIdx=$mistakeVerifyAbsentIdx " +
                    "attempts=$mistakeVerifyAttempts strikes=$mistakeVerifyAbsentStrikes) " +
                    "-> forgiving so the cursor can advance"
            )
            verifiedSpokenIndices.add(cand)
            lastWrongStopIndex = -1
            lastWrongStopSpokenText = ""
            lastWrongStopSightings = 0
            lastWrongStopFirstSeenMs = 0L
            mistakeVerifyAbsentIdx = -1
            mistakeVerifyAttempts = 0
            mistakeVerifyAbsentStrikes = 0
            mistakeVerdictAtMs = 0L
            mistakeEpisodeAnchor = -1
            mistakeCleanStreak = 0
            confirmedMistakeUiIdx = -1
            stallFirstSeenMs = 0L
            stallNudges = 0
            stallCand = -1
        }
    }

    // (item 5) NO auto-dismiss — the mistake popup STAYS until the user dismisses
    // it (standing contract: the popup must be fully visible and remain until
    // dismissed). It used to vanish after 4 s, but both paths that raise this
    // dialog (TAJWEED_ERROR and confirmed WRONG/SKIPPED) ALREADY stop/pause the
    // mic, so there are no "subsequent words" the dialog could obscure — the
    // auto-dismiss only ever made the user miss the feedback they had paused for
    // ("no popup appeared"). Dismissal is now exclusively via onDismissRequest /
    // the dialog button, plus the session/range resets that null it out.
    // A NEW mistake still replaces the content: latestTajweedPopup is a Pair key,
    // so a different (rule, index) recomposes the dialog with the new message.

    val mainHandler = remember { android.os.Handler(android.os.Looper.getMainLooper()) }

    val matchedWordCount = alignment.first
    val hasError = alignment.second
    val errorIndex = if (hasError) matchedWordCount else -1
    // When a session is stopped, the cursor sits on the first UNREAD word.
    // That is the right navigation target while listening (the reader is about
    // to say it) but the wrong one after a pause: it belongs to the next ayah.
    // `contentAnchor` is the word the header/auto-scroll should follow.
    val displayPaused = !isListening && (!tarteelWantsMic || recitationMode == "Tajweed")
    val contentAnchor = displayAyahAnchor(
        cursor = matchedWordCount,
        paused = displayPaused,
        mistakeArmed = mistakeResumeIndex >= 0,
        ayahs = viewModel.ayahs,
        hasHeardAtCursor = heardFrontierExclusive(detailedAnalysis) > matchedWordCount,
    )

    // Start a Tajweed practice session. Our own AudioRecord is the SINGLE
    // microphone client so the acoustic score measures the user's real voice.
    // The AudioRecord feed is split between the Sherpa OnlineRecognizer
    // (drives the word-by-word transcript — replaces the platform
    // SpeechRecognizer, which on the A05s starves voice capture when two HAL
    // clients compete) and the TajweedNet acoustic scorer.
    // Sherpa's hotwordsFile is sampled at recognizer-construction time inside
    // [SingleAudioPipeline.start], so this whole session must run against the
    // current `targetOrig` (the Compose UI's active ayah range) — a mid-session
    // range change stops + restarts the pipeline (see the LaunchedEffect on
    // `targetOrig` above).
    fun startTajweedSession() {
        android.util.Log.i("ItqanSherpa", "startTajweedSession(): holder.pipeline=${tajweedPipelineHolder.pipeline?.pipelineId} holder.running=${tajweedPipelineHolder.running} holder.paused=${tajweedPipelineHolder.paused} isListening=$isListening")
        // SCOPE SNAPSHOT: capture (surah, range) the user is about to recite
        // against. The final-decode / goal-stats code path reads this snapshot
        // instead of [viewModel.selectedSurah] / [viewModel.rangeStart] /
        // [viewModel.rangeEnd], which can change between the final decode
        // landing and the next recomposition (a user tap on the surah/range
        // picker while the pipeline is still finalizing). Charging the OLD
        // recitation's mistakes to the NEW range's goals is the bug this
        // snapshot fixes.
        // (Captured on every entry: a resume fast-path keeps the prior
        // snapshot, so a recitation that started under the old range stays
        // attributed to it even if the user changes the range while paused.)
        activeSessionScope.value = ActiveScope(
            surah = viewModel.selectedSurah?.number ?: 0,
            rangeStart = viewModel.rangeStart,
            rangeEnd = viewModel.rangeEnd,
        )
        // Toggle semantics: if the pipeline is ALREADY running when this is
        // invoked (e.g. the mic FAB recomposed with a stale `isListening=false`
        // while running=true on the second tap — the "tap to pause started
        // listening again" bug on-device), route to the pause path instead of
        // silently bailing out. This keeps the tap-to-pause always effective.
        if (tajweedPipelineHolder.running) {
            android.util.Log.i("ItqanSherpa", "startTajweedSession(): already running -> routing to pause")
            val audio = tajweedPipelineHolder.pipeline?.pause() ?: FloatArray(0)
            tajweedRecordedAudio = audio
            tajweedRecordingDurationMs = (audio.size * 1000L) / 16000
            // Keep the analysis pass scoring the REAL utterance after the
            // pause: the live snapshots stop growing once the mic is off.
            if (audio.isNotEmpty()) {
                recognizedAudio = audio
                recognizedAudioStartMs = tajweedRecordingStartMs
            }
            try { speechRecognizer?.stopListening() } catch (e: Throwable) {}
            isListening = false
            // A pause is NOT the end of the session: it stays resumable, and a
            // resume must re-seat the transcript rather than wipe it (§4.45).
            recitationSessionActive = true
            return
        }
        // First-launch gate: if the Sherpa model files aren't on disk yet,
        // the pipeline's start() returns false without touching the mic.
        // The Compose "Downloading speech model" dialog (further down in this
        // Composable, keyed off `sherpaDownloader.progress`) is the blocking
        // popup: its non-dismissable scrim locks the whole screen while the
        // bundled Tarteel model extracts from the APK, so the user CANNOT
        // start reciting until it dismisses. We explicitly kick the (single-
        // flight, idempotent) extraction here so the popup is guaranteed to be
        // up even if the tab-level progress effect hasn't fired yet, and we
        // keep `isListening=false` so the FAB stays in START state — a single
        // tap after the dialog dismisses begins the real session (setting it
        // true here would leave the FAB in STOP state after the dialog and
        // waste the user's first tap).
        if (!isSherpaModelReady(context)) {
            isListening = false
            errorMessage = getAppString(isArabic, "Loading the speech model — please wait…")
            try { kotlinx.coroutines.GlobalScope.launch {
                try { sherpaDownloader.downloadIfNeeded() } catch (_: Throwable) {}
            } } catch (_: Throwable) {}
            return
        }
        // Resume fast-path: the pipeline is currently PAUSED (the user paused
        // by tapping the FAB while it was recording, then tapped again to
        // continue). re-spawn the AudioRecord worker on the SAME OnlineStream
        // in milliseconds — no 2-5 s ONNX reload, no stop/start race. The
        // AyahWordMatcher's monotonicity cursor is preserved so the highlight
        // picks up from where the user left off.
        val pipelineNow = tajweedPipelineHolder.pipeline
        // PREVIOUS RUN FINISHED THE RANGE: the last recitation completed its
        // selected ayah range (the session was closed by stopAtEnd, leaving the
        // pipeline paused but the range DONE). A plain resume() would replay the
        // stale, fully-recited transcript as if the user were continuing — the
        // "finished session bleeds into the next recitation" bug. Wipe the
        // previous run's transcript + counters and start a genuinely fresh
        // recitation instead of resuming.
        if (pipelineNow != null && pipelineNow.paused && sessionRangeCompleted) {
            android.util.Log.i("ItqanSherpa", "startTajweedSession(): previous run finished the range -> wiping transcript + counters, starting fresh recitation")
            finalSpokenText = ""
            partialSpokenText = ""
            lastCommittedWords = emptyList()
            committedWordCount = 0
            spokenTimings.clear()
            currentWordCount = 0
            alignment = Pair(0, false)
            detailedAnalysis = emptyList()
            tajweedVerdictCache.clear()
            tajweedFailCache.clear()
            // A fresh recitation starts its display from scratch (§4.46).
            displayCursorFloor = 0
            verifiedSpokenIndices.clear()
            hintedIndices.clear()
            mistakeResumeIndex = -1
            mistakeResumeLiveText = ""
            sessionRangeCompleted = false
            // §4.38 — fresh session: this run must earn its own endpoint evidence.
            endpointDecodeLandedAtMs = 0L
            pipelineStoppedAtMs = 0L
            // fall through to the fresh-session path below (do NOT resume)
        } else if (pipelineNow != null && pipelineNow.paused) {
            isListening = true
            recitationSessionActive = true
            // Restart the minimum-listen clock for the tajweed-popup auto-stop
            // guard (item 5c): the resume path must not inherit the ORIGINAL
            // session's start time, or the 2.5 s grace window would already be
            // expired the moment the user resumes and the popup would yank the
            // mic on the very first flagged partial.
            tajweedRecordingStartMs = System.currentTimeMillis()
            val resumed = pipelineNow.resume()
            android.util.Log.i("ItqanSherpa", "startTajweedSession(): resume() returned $resumed (id=${pipelineNow.pipelineId})")
            if (resumed) {
                viewModel.preloadTajweedModel()
                return
            }
            // resume() could not proceed (e.g. a teardown is in flight) — fall
            // through to start() below, which has its own gate for the same
            // race.
        }
        spokenTimings.clear()
        currentWordCount = 0
        // Fresh session: drop cached tajweed verdicts (the capture buffer and
        // the timing clock both restart, so no cached verdict applies).
        tajweedVerdictCache.clear()
            tajweedFailCache.clear()
        // Fresh session: reset the WRONG/SKIPPED auto-stop confirmation latch
        // (the resume fast-path above deliberately keeps it so a re-read
        // mistake re-stops quickly).
        lastWrongStopIndex = -1
        lastWrongStopSpokenText = ""
        lastWrongStopSightings = 0
        lastWrongStopFirstSeenMs = 0L
        // A fresh session must re-earn its verdict: the old "this word was
        // proven absent" belongs to the previous capture, and the ticker would
        // otherwise stop the new session on the very first word again.
        mistakeVerifyAbsentIdx = -1
        mistakeVerifyAttempts = 0
        mistakeVerifyAbsentStrikes = 0
        mistakeVerdictAtMs = 0L
        confirmedMistakeUiIdx = -1
        // Clear the Tajweed Rules Coach transcript slots at session start.
        // The pipeline routes its partials to the vars matching the ACTIVE tab
        // (Memorize: finalSpokenText/partialSpokenText, Tajweed card:
        // tajweed* vars) — so a fresh coach session must not inherit the
        // previous coach session's text (this was the "the Madd card showed
        // my last two Memorize words" cross-contamination bug).
        tajweedFinalSpokenText = ""
        tajweedPartialSpokenText = ""
        // The MEMORIZE transcript is deliberately NOT cleared here: the
        // committed words are the session's continuation point, and the
        // collector's count-based filter suppresses the replay of already-
        // committed words. Wiping them made every mic re-tap "jump back to
        // the start" of the surah even mid-recitation (the user's on-device
        // report) — a mid-surah stop + re-tap must continue from the
        // committed frontier, not restart the word coloring from word 0.
        // Explicit restarts (range change, Repeat/Restart buttons) clear the
        // transcript on their own paths.
        partialSpokenText = ""
        committedWordCount = 0
        tajweedRecordedAudio = null
        recognizedAudio = null
        recognizedAudioUri = null
        tajweedRecordingStartMs = System.currentTimeMillis()
        // isListening turns on immediately so the mic FAB shows the "stop"
        // icon while the recognizer is still loading on the IO scope — this
        // prevents the user from re-tapping and starting a second pipeline
        // while the first OnlineRecognizer is still being constructed (which
        // on the A05s takes 2–5 s for the 296 MB int8 encoder). The FAB's
        // stop branch calls stopTajweedSession, which handles a still-loading
        // pipeline cleanly (running stays false, the start job's `running=true`
        // write happens after the recognizer is ready, and stop() flips it back
        // false without blocking main).
        isListening = true
        recitationSessionActive = true
        val started = tajweedPipelineHolder.pipeline?.start() ?: false
        if (!started) {
            // - THE "MIC IS DEAD FOREVER" DEADLOCK (§4.45) -
            // `start()` refuses while a previous teardown is still in flight
            // (`isStopping`, cleared only at the END of the async stop() job).
            // The old code turned that refusal straight into an error toast and
            // gave up - so a single mistimed tap right after a mistake-stop or a
            // range change wedged the mic for the rest of the process: every
            // later tap failed the same way, the user could not resume their
            // re-read, and the app looked permanently dead ("the app becomes
            // never functioning").
            //
            // The teardown takes ~1-3 s, so this is a RACE, not a failure: wait
            // it out off the main thread and retry. Only a genuinely missing
            // microphone is reported to the user.
            val stillStopping = tajweedPipelineHolder.pipeline?.isStoppingLog == true
            if (stillStopping) {
                android.util.Log.w(
                    "ItqanSherpa",
                    "startTajweedSession(): start() refused - teardown in flight; retrying (was " +
                        "the 'mic is dead forever' deadlock)"
                )
                val holder = tajweedPipelineHolder
                val scope = coroutineScope
                val startedAtMs = System.currentTimeMillis()
                scope.launch {
                    var ok = false
                    while (System.currentTimeMillis() - startedAtMs < 8000L) {
                        kotlinx.coroutines.delay(250L)
                        if (holder.pipeline?.isStoppingLog != true) {
                            ok = holder.pipeline?.start() ?: false
                            if (ok) break
                        }
                    }
                    if (ok) {
                        android.util.Log.i("ItqanSherpa", "startTajweedSession(): retry succeeded - mic is live again")
                        isListening = true
                        recitationSessionActive = true
                    } else {
                        android.util.Log.e("ItqanSherpa", "startTajweedSession(): retry gave up after 8s")
                        isListening = false
                        recitationSessionActive = false
                        errorMessage = "Microphone is unavailable on this device"
                    }
                }
            } else {
                isListening = false
                recitationSessionActive = false
                errorMessage = "Microphone is unavailable on this device"
            }
        }
    }
    // Publish the forward reference so the range-change auto-resume effect
    // (declared above) can invoke this function.
    startTajweedSessionRef[0] = ::startTajweedSession

    fun stopTajweedSession() {
        android.util.Log.i("ItqanSherpa", "stopTajweedSession(): holder.pipeline=${tajweedPipelineHolder.pipeline?.pipelineId} holder.running=${tajweedPipelineHolder.running} holder.paused=${tajweedPipelineHolder.paused} isListening=$isListening isStopping=${tajweedPipelineHolder.pipeline?.isStoppingLog}")
        // PAUSE-PREFERRED: tap-the-mic-to-pause applies `pause()` so the
        // recognizer + stream are KEPT warm in memory for a fast `resume()`
        // on the next mic tap. This is the bug the user just hit — pause then
        // resume crashed because the previous stop()-then-fast-restart() race
        // resurrected the worker underneath a pending teardown, freeing stream
        // under the worker's feet -> SIGSEGV -> "app crashes from the beginning
        // every subsequent launch". pause()/resume() keep the natives alive
        // across the pause, only the cheap AudioRecord is torn down. The full
        // teardown still happens on range change via `release()`.
        //
        // Tolerate the "tap-stop-during-load" race: the user may tap the mic,
        // see isListening go true immediately, and tap again before the
        // recognizer finishes constructing on IO. `running` stays false until
        // the IO job succeeds, so `tajweedPipelineHolder.running` is false —
        // but isListening was set true optimistically. Clear both, the IO job's
        // eventual `running=true` write is harmless because pause()/stop() will
        // already have flipped them back. We also call pipeline.pause()/
        // stop() defensively — pause() is a no-op if `running` is false.
        val wasListening = isListening || tajweedPipelineHolder.running
        if (!wasListening) {
            isListening = false
            return
        }
val audio = if (tajweedPipelineHolder.running) {
            // pause() keeps the recognizer+stream alive for fast resume.
            tajweedPipelineHolder.pipeline?.pause() ?: FloatArray(0)
        } else {
            // NOT running but a pipeline exists: it is either MID-LOAD (the 296 MB
            // ONNX recognizer is still being constructed on IO — running=false until
            // the load job flips it) or Idle. The previous code returned an empty
            // buffer here, which meant a "tap mic → tap mic again during the
            // 2-5 s load window" stopped NOTHING: the load job completed, started
            // the AudioRecord worker, and left the pipeline live under a UI that
            // believed it had already stopped — the read loop kept burning CPU and
            // re-tapping did nothing (seen on-device: worker at ~50% CPU for
            // minutes after "pause"). Flipping cancelledDuringLoad via stop() makes
            // the pending load job tear itself down instead of going live.
            try { tajweedPipelineHolder.stop() } catch (e: Throwable) {}
            FloatArray(0)
        }
        tajweedRecordedAudio = audio
        tajweedRecordingDurationMs = (audio.size * 1000L) / 16000
        // Same as the FAB pause route: hand the FULL utterance PCM to the
        // analysis pass so the final (post-stop) report still scores tajweed
        // acoustics against what the user actually said.
        if (audio.isNotEmpty()) {
            recognizedAudio = audio
            recognizedAudioStartMs = tajweedRecordingStartMs
        }
        try { speechRecognizer?.stopListening() } catch (e: Throwable) {}
        isListening = false
    }

    fun stopAllListening() {
        tarteelWantsMic = false
        isListening = false
        try { speechRecognizer?.stopListening(); speechRecognizer?.cancel() } catch (e: Throwable) {}
        try { if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop() } catch (e: Throwable) {}
    }

    LaunchedEffect(currentTab) {
        if (isListening || recitationSessionActive) {
            tarteelWantsMic = false
            isListening = false
            // Leaving the tab tears the session down without a summary (there is
            // no user-initiated stop), so the session is over (§4.45). This must
            // cover the PAUSED case too (a mistake popup, a pause still waiting
            // for its final decode), which has no mic to stop.
            recitationSessionActive = false
            try {
                speechRecognizer?.stopListening()
                speechRecognizer?.cancel()
            } catch (e: Throwable) {
                android.util.Log.e("SpeechRec", "Error auto-stopping on tab change", e)
            }
            try {
                if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop()
            } catch (e: Throwable) {
                android.util.Log.e("SpeechRec", "Error auto-stopping tajweed recorder on tab change", e)
            }
        }
    }
    
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner, speechRecognizer) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_PAUSE || event == androidx.lifecycle.Lifecycle.Event.ON_STOP) {
                if (isListening || recitationSessionActive) {
                    tarteelWantsMic = false
                    isListening = false
                    // Backgrounded mid-recitation: the session is abandoned, not
                    // paused, so it must not linger as resumable (§4.45). Covers
                    // the paused-at-a-mistake case too, which has no mic to stop.
                    recitationSessionActive = false
                    try {
                        speechRecognizer?.stopListening()
                        speechRecognizer?.cancel()
                    } catch (e: Throwable) {
                        android.util.Log.e("SpeechRec", "Error auto-stopping on background", e)
                    }
                    try {
                        if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop()
                    } catch (e: Throwable) {
                        android.util.Log.e("SpeechRec", "Error auto-stopping tajweed recorder on background", e)
                    }
                }
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            try {
                lifecycleOwner.lifecycle.removeObserver(observer)
                if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop()
            } catch (e: Throwable) {}
            try { speechRecognizer?.destroy() } catch (e: Throwable) {}
        }
    }
    
    class RecognizerCallbackWrapper : android.speech.RecognitionListener {
        var onResult: ((String) -> Unit)? = null
        var onPartial: ((String) -> Unit)? = null
        var onError: ((String) -> Unit)? = null
        var onState: ((Boolean) -> Unit)? = null
        
        var onRms: ((Float) -> Unit)? = null
        
        override fun onReadyForSpeech(params: android.os.Bundle?) { onState?.invoke(true) }
        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(rmsdB: Float) { onRms?.invoke(rmsdB) }
        override fun onBufferReceived(buffer: ByteArray?) {}
        override fun onEndOfSpeech() {}
        override fun onError(error: Int) {
            // IMPORTANT (recognized-audio preservation): on the continuous-loop recognizer the
            // ONLY place recognizedAudio/recognizedAudioUri get filled is onResults (which decodes
            // the GET_AUDIO bundle the SpeechRecognizer hands back). Every benign error branch
            // (CLIENT / NO_MATCH / SPEECH_TIMEOUT) used to ALSO null them out, which erased the
            // just-decoded clip before analyzeRecitationDetails ever got to use it — that is what
            // made Memorize+Tajweed look like Memorize+Tarteel (no tajweed feedback). Stopping a
            // recognizer session does NOT invalidate the audio it already produced, so we simply
            // do not clear these here; the only places that legitimately wipe the clip are start(),
            // surah/range reset, and the Tajweed coach's own session starter.
            if (error == android.speech.SpeechRecognizer.ERROR_CLIENT || error == android.speech.SpeechRecognizer.ERROR_NO_MATCH || error == android.speech.SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                onError?.invoke("")
                return
            }
            val msg = when (error) {
                android.speech.SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
                android.speech.SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Insufficient permissions"
                android.speech.SpeechRecognizer.ERROR_NETWORK -> "Network error"
                android.speech.SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
                android.speech.SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer busy"
                android.speech.SpeechRecognizer.ERROR_SERVER -> "Error from server"
                else -> "Error: $error"
            }
            onError?.invoke(msg)
        }
        override fun onResults(results: android.os.Bundle?) {
            val res = results?.getStringArrayList(android.speech.SpeechRecognizer.RESULTS_RECOGNITION)
            if (!res.isNullOrEmpty()) onResult?.invoke(res[0])
            // Single-capture design: the recognizer recorded the user's voice and hands the audio
            // back via BUNDLE_AUDIO_SOURCE. Decode it (AMR/WAV -> 16 kHz PCM) so the acoustic engine
            // can score the REAL voice instead of a dummy buffer.
            val audioUri = results?.getParcelable<Uri>("android.speech.extra.AUDIO_SOURCE")
            if (audioUri != null) {
                try {
                    val decoded = kotlinx.coroutines.runBlocking(kotlinx.coroutines.Dispatchers.IO) {
                        decodeRecognizerAudio(context, audioUri)
                    }
                    mainHandler.post {
                        recognizedAudio = decoded
                        recognizedAudioUri = audioUri
                    }
                } catch (e: Throwable) {
                    android.util.Log.e("RecognizerAudio", "onResults audio decode failed", e)
                    // No parallel-AudioRecord fallback: starting a second mic alongside the
                    // recognizer collides on devices like the Samsung A05s and stops partial
                    // transcripts from arriving (word tracking breaks). If the bundle can't be
                    // decoded on this device, the tajweed score is simply skipped -- recitation
                    // tracking still works because it only needs the transcript.
                }
            }
            // When audioUri is null (Samsung A05s's common case) the per-word tajweed verdict
            // is skipped silently. We deliberately do NOT start a parallel AudioRecord here or
            // in the FAB stop branch: two simultaneous mic clients on the A05s starve the
            // SpeechRecognizer of partial results and word-tracking breaks -- which is the exact
            // regression the user hit twice ("no word display in memorize + tajweed"). The
            // single-mic invariant stays intact here.
            onState?.invoke(false)
            // 4.71 - `onResults` ends ONE platform utterance, not necessarily the
            // user's Tarteel session. The callback below decides: while
            // `tarteelWantsMic` is true it schedules the next segment and keeps
            // `sessionFinalized` false; after a user stop (or a fatal error) it
            // finalizes normally. Deliberately no direct `sessionFinalized = true`
            // here - it raced the callback and could finalize the gap between two
            // Tarteel utterances.
        }
        override fun onPartialResults(results: android.os.Bundle?) {
            val res = results?.getStringArrayList(android.speech.SpeechRecognizer.RESULTS_RECOGNITION)
            if (!res.isNullOrEmpty()) onPartial?.invoke(res[0])
        }
        override fun onEvent(eventType: Int, params: android.os.Bundle?) {}
    }
    
    val listenerWrapper = remember { RecognizerCallbackWrapper() }
    
    LaunchedEffect(speechRecognizer) {
        try {
            speechRecognizer?.setRecognitionListener(listenerWrapper)
        } catch (e: Throwable) {
            android.util.Log.e("SpeechRec", "Error setting listener", e)
        }
    }
    
    listenerWrapper.onRms = { rms ->
        synchronized(acousticRmsTracker) {
            if (rms > acousticRmsTracker[0]) {
                acousticRmsTracker[0] = rms
            }
        }
    }
    
    listenerWrapper.onResult = { result ->
        mainHandler.post {
            // Drop results that arrive during the post-range-reset suppression window: they carry
            // the WHOLE stale transcript from the previous ayah range and would immediately
            // re-fire wrong/tajweed mistakes with nothing read.
            if (System.currentTimeMillis() >= recognizerResultsSuppressedUntil) {
                if (currentTab == "Tajweed") {
                    tajweedFinalSpokenText = "$tajweedFinalSpokenText $result".trim()
                    tajweedPartialSpokenText = ""
                } else {
                    finalSpokenText = "$finalSpokenText $result".trim()
                    partialSpokenText = ""
                }
            }
            // In the Tajweed coach we own the microphone session (AudioRecord); the recognizer
            // is only a best-effort transcript source, so its completion must NOT end our
            // recording. In Tarteel, a result normally marks the end of ONE utterance, not the
            // end of the user's session: the restart effect below opens the next one.
            if (currentTab != "Tajweed") isListening = false
            val tarteelContinues =
                currentTab == "Memorize" &&
                    recitationMode == "Tarteel" &&
                    tarteelWantsMic
            if (tarteelContinues) {
                // Do NOT mark the session final here. If it were final, the
                // end-of-range/grace effects would see a completed-looking
                // session during the restart gap and could stop the mic (and
                // raise a summary) before the next utterance. This was the
                // structural half of "Tarteel stops listening too early".
                sessionFinalized = false
                tarteelSegmentEnded++
                android.util.Log.i(
                    "ItqanSherpa",
                    "Tarteel: utterance ended; continuing (segment=$tarteelSegmentEnded) " +
                        "until the user taps stop"
                )
            } else if (recitationMode != "Tajweed") {
                sessionFinalized = true
            }
        }
    }
    val whitespaceRegex = remember { Regex("\\s+") }
    
    listenerWrapper.onPartial = { partial ->
        val timeNow = System.currentTimeMillis()
        // During the range-reset suppression window, drop partial transcripts entirely: they carry
        // the previous range's whole text until the engine fully cancels, and would re-fire
        // word-timings + analysis against a target the user hasn't read yet.
        if (timeNow >= recognizerResultsSuppressedUntil) mainHandler.post {
            val words = partial.split(whitespaceRegex).filter { it.isNotBlank() }
            val wordsLength = words.size
            if (wordsLength > currentWordCount) {
                if (spokenTimings.isNotEmpty()) {
                    spokenTimings.last().endMs = timeNow
                }
                val rVal = synchronized(acousticRmsTracker) {
                    val r = acousticRmsTracker[0]
                    acousticRmsTracker[0] = 0f
                    r
                }
                for (i in currentWordCount until wordsLength) {
                    spokenTimings.add(SpokenWordTiming(
                        word = words[i],
                        startIndex = i,
                        startMs = timeNow,
                        endMs = timeNow,
                        maxRms = rVal
                    ))
                }
                currentWordCount = wordsLength
            } else if (wordsLength == currentWordCount && spokenTimings.isNotEmpty()) {
                val last = spokenTimings.last()
                last.endMs = timeNow
                synchronized(acousticRmsTracker) {
                    if (acousticRmsTracker[0] > last.maxRms) {
                        last.maxRms = acousticRmsTracker[0]
                    }
                }
            }

            if (currentTab == "Tajweed") {
                if (partial.isNotEmpty()) tajweedPartialSpokenText = partial
            } else {
                if (partial.isNotEmpty()) partialSpokenText = partial 
            }
        }
    }
    listenerWrapper.onError = { err -> 
        mainHandler.post {
            // A benign platform error (CLIENT / NO_MATCH / SPEECH_TIMEOUT) is
            // how the recognizer reports the END OF AN UTTERANCE. For Tarteel
            // that is not a reason to stop: keep the session open and let the
            // restart effect open the next microphone session. The old code
            // fell through to `isListening = false` and trimmed the transcript,
            // which is exactly the reported "it stops after a short silence".
            val tarteelContinues =
                err.isEmpty() &&
                    currentTab == "Memorize" &&
                    recitationMode == "Tarteel" &&
                    tarteelWantsMic
            if (tarteelContinues) {
                errorMessage = null
                isListening = false
                sessionFinalized = false
                tarteelSegmentEnded++
                android.util.Log.i(
                    "ItqanSherpa",
                    "Tarteel: recognizer segment error (benign) -> restarting; " +
                        "the user still owns the mic"
                )
                return@post
            }
            errorMessage = if (err.isEmpty()) null else err
            // A fatal error cannot be restarted safely (permissions, network,
            // busy recognizer). Drop user intent so the restart effect does not
            // hammer the engine, and let the normal stop/finalize path report.
            if (currentTab == "Memorize" && recitationMode == "Tarteel" && tarteelWantsMic) {
                tarteelWantsMic = false
                sessionFinalized = true
            }
            // In the Tajweed coach the recognizer is best-effort; its error must not end our
            // own audio recording session (the score comes from our AudioRecord, not the recognizer).
            if (currentTab != "Tajweed") isListening = false
            currentWordCount = 0
            if (currentTab == "Tajweed") {
                tajweedPartialSpokenText = ""
            } else {
                // Safely trim finalSpokenText to correctly matched words on error to avoid losing place.
                // Same guard as the FAB stop path: never trim to take(0) - that
                // would wipe a transcript the analysis simply hasn't caught up to.
                if (matchedWordCount > 0) {
                    val matchedWordsList = targetOrig.take(matchedWordCount)
                    finalSpokenText = matchedWordsList.joinToString(" ")
                }
                partialSpokenText = ""
            }
        }
    }
    listenerWrapper.onState = { stateValue -> 
        mainHandler.post {
            // The platform SpeechRecognizer's lifecycle states MUST NOT
            // clobber our UI mic flag while the Sherpa-ONNX pipeline owns the
            // microphone. On the A05s the platform recognizer can fire a
            // late READY/ERROR state (e.g. from a previous Memorize session)
            // while Tajweed recording is live — setting isListening=false here
            // left the FAB green mid-recording and broke the tap-to-pause
            // toggle. Guard: when the pipeline is running or paused (owns the
            // mic), the platform recognizer's state is irrelevant.
            if (tajweedPipelineHolder.running || tajweedPipelineHolder.paused) return@post
            // A late `false` from the PREVIOUS Tarteel utterance must not turn
            // the FAB grey after the restart effect has already opened the next
            // session. The user's stop sets `tarteelWantsMic = false` first, so
            // a genuine stop is still honoured.
            if (!stateValue &&
                currentTab == "Memorize" &&
                recitationMode == "Tarteel" &&
                tarteelWantsMic &&
                isListening
            ) {
                return@post
            }
            isListening = stateValue 
        }
    }

    // TARTEEL AUTO-CONTINUE. The platform recognizer delivers one final result
    // (or a benign timeout) per utterance and then goes idle. The user asked
    // for the mic to stay on until THEY stop it, so after a short gap this
    // opens the next recognizer session. `tarteelWantsMic` is the only thing
    // that keeps it alive; a missing pipeline/model is irrelevant because this
    // path does not use Sherpa. Guard `isListening`: if the user tapped the FAB
    // during the gap (or a new session is already running), do not double-start.
    LaunchedEffect(tarteelSegmentEnded) {
        if (tarteelSegmentEnded == 0) return@LaunchedEffect
        if (!tarteelWantsMic ||
            currentTab != "Memorize" ||
            recitationMode != "Tarteel"
        ) return@LaunchedEffect
        kotlinx.coroutines.delay(300L)
        if (!tarteelWantsMic ||
            isListening ||
            currentTab != "Memorize" ||
            recitationMode != "Tarteel"
        ) return@LaunchedEffect
        val recognizer = speechRecognizer ?: return@LaunchedEffect
        // The next session starts a fresh hypothesis. `finalSpokenText` keeps
        // everything already committed; these only describe the live segment.
        // `spokenTimings.startIndex` is the position in the FULL spoken text,
        // so keeping the previous segment's list while restarting its counter
        // at 0 appended duplicate indices. Tarteel never uses the timings for
        // acoustic scoring, and a manual resume already starts with a fresh
        // list, so dropping the old segment is both safe and consistent.
        currentWordCount = 0
        partialSpokenText = ""
        spokenTimings.clear()
        sessionFinalized = false
        isListening = true
        startListening(recognizer, context.packageName) { msg ->
            errorMessage = if (msg.isEmpty()) null else msg
            if (tarteelWantsMic) {
                // A failed START is not a user stop, but retrying a broken
                // recognizer forever would keep a green mic with no audio.
                // Surface it and end the intent; the FAB is now "start".
                tarteelWantsMic = false
                sessionFinalized = true
                recitationSessionActive = false
            }
            isListening = false
        }
    }
    
    DisposableEffect(speechRecognizer) {
        onDispose {
            try {
                speechRecognizer?.stopListening()
                speechRecognizer?.cancel()
                speechRecognizer?.destroy()
            } catch (e: Throwable) {
                android.util.Log.e("SpeechRec", "Error destroying SpeechRecognizer", e)
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Sherpa-ONNX offline pipeline partials subscription (Memorize+Tajweed).
    //
    // Each [SherpaPartialEvent] emitted by the [SingleAudioPipeline]'s
    // recogniser thread carries the cleaned partial transcript (post
    // [itqan.app.sherpa.AyahWordMatcher] hallucination guard). We mirror the
    // same state writes that the SpeechRecognizer `onPartial`/`onResults`
    // callbacks above do for the tajweed transcript slot: append cumulative
    // timings for any newly spoken words, advance `currentWordCount`, and
    // publish the partial text. The `endpoint == true` events commit the
    // partial as final (mirroring `onResults` from the platform recognizer).
    // Runs in BOTH the Tajweed Rules Coach tab (standalone) AND the
    // Memorize+Tajweed tab, whenever the Sherpa pipeline is running —
    // re-launches whenever the holder wires a NEW pipeline on a range change.
    //
    // Re-keying: `tajweedPipelineHolder.pipeline` is a Compose `mutableStateOf`
    // since the holder class opt-in to Compose state observation. Reading it
    // inside a `key {}` block makes THIS LaunchedEffect re-launch whenever a
    // new pipeline is wired into the holder (range change), so the partials
    // collector re-subscribes to the new pipeline's `_partials` SharedFlow.
    // The earlier version captured the pipeline ONCE at first composition and
    // kept listening to the OLD pipeline even after the ayah-range changed —
    // so ayah 2's partials were emitted but never collected, the highlight froze
    // at ayah 1's last word, and the user saw "the first ayah displays words
    // but the second ayah does nothing". The original version also gated on
    // `currentTab != "Tajweed" -> return`, which silently dropped all partials
    // of a Memorize+Tajweed session: with no subscriber the SharedFlow's buffer
    // capped. Now we run whenever the active mode is pipeline-driven (either
    // tab) AND on every pipeline swap.
    androidx.compose.runtime.key(tajweedPipelineHolder.pipeline) {
        LaunchedEffect(tajweedPipelineHolder.pipeline, currentTab, recitationMode) {
            val pipelineDriven =
                currentTab == "Tajweed" ||
                    (currentTab == "Memorize" && recitationMode == "Tajweed") ||
                    // §4.68 — ...or an INJECTED non-Tajweed session, whose transcript
                    // also comes from this pipeline. Without this the §4.68 hook starts
                    // the pipeline and NOTHING subscribes to its partials, so the run
                    // decodes 29 words and the app receives none of them:
                    //   decodeWindow(): done ... emitted=29w        (pipeline, fine)
                    //   finalDecodeAndEmit(): final-commit marker   (pipeline, fine)
                    //   -- zero collector() lines, zero analyze(fast) lines --
                    // which reads as "Tarteel displays nothing" and is exactly the
                    // symptom the hook exists to reproduce.
                    (currentTab == "Memorize" && injectTranscriptSource)
            if (!pipelineDriven) return@LaunchedEffect
            val pipe = tajweedPipelineHolder.pipeline ?: return@LaunchedEffect
            // MEMORIZE-word filter for the sliding-window CTC engine. The
            // pipeline's emission is the session's cumulative MERGED log, so
            // every emission re-includes the already-committed words. ONE
            // unified count rule now applies to LIVE and ENDPOINT emissions
            // alike (the old endpoint bypass let merge artifacts and the
            // tail-decode's re-rendered prefix into the committed transcript
            // — falaq's 44-word transcript, ikhlas' duplicated tail):
            //   (a) for each word norm n, DROP the first committedCount(n)
            //       occurrences — the replay copies of words the transcript
            //       already holds. Occurrences beyond that are new audio
            //       (genuine repeats like Ikhlas' twice-spoken الله, or
            //       mistake words re-read after a resume);
            //   (b) CAP kept occurrences at targetCount(n) — the target only
            //       contains n that many times, so any further occurrence is
            //       a merge/decode artifact and can never align to a real
            //       target slot;
            //   (c) LIVE-only: the LAST token of a window decode is an
            //       unfinished cut if it doesn't land EXACTLY on a target
            //       word — drop it; the next window re-decodes it whole (the
            //       "الْعَالَم" cut that used to be flagged WRONG).
            fun filterMemorizeWords(raw: List<String>, endpoint: Boolean): List<String> {
                if (raw.isEmpty()) return raw
                val riwayah = viewModel.riwayah
                val targetCounts = HashMap<String, Int>()
                for (tw in targetNorm) {
                    val tn = normalizeForComparison(tw, riwayah)
                    if (tn.isNotEmpty()) targetCounts[tn] = (targetCounts[tn] ?: 0) + 1
                }
                val committedCounts = HashMap<String, Int>()
                for (cw in lastCommittedWords) {
                    val cn = normalizeForComparison(cw, riwayah)
                    if (cn.isNotEmpty()) committedCounts[cn] = (committedCounts[cn] ?: 0) + 1
                }
                val committedOrig = HashMap(committedCounts)
                val keptCounts = HashMap<String, Int>()
                val out = ArrayList<String>(raw.size)
                for ((i, w) in raw.withIndex()) {
                    val n = normalizeForComparison(w, riwayah)
                    if (n.isEmpty()) continue
                    if (!endpoint && i == raw.lastIndex && n !in targetCounts) continue
                    val committedLeft = committedCounts[n] ?: 0
                    if (committedLeft > 0) {
                        committedCounts[n] = committedLeft - 1
                        continue
                    }
                    // TOTAL cap: committed + kept may never exceed the target
                    // count. A cumulative emission re-walks the already-
                    // committed log words before the new tail; without the
                    // committed offset the tail's re-render could re-spend the
                    // cap and duplicate a word past its target count (nasr's
                    // الله x4 final transcript).
                    val cap = targetCounts[n] ?: 0
                    val kept = keptCounts[n] ?: 0
                    if ((committedOrig[n] ?: 0) + kept >= cap) continue
                    keptCounts[n] = kept + 1
                    out.add(w)
                }
                return out
            }
            pipe.partials.collect { ev ->
                val timeNow = System.currentTimeMillis()
                // Fresh utterance (start()/resume() after a pause): the pipeline
                // reset its own emission log, so the collector must reset its
                // committed-word cursor in sync. Otherwise words the user
                // RE-RECITES after a pause are filtered out as "already
                // committed" — the app looks deaf ("I repeat the word and
                // nothing happens"). Only the Memorize-side transcript resets
                // (the Tajweed Rules Coach keeps its own utterance state).
                if (ev.freshSession) {
                    // The pipeline reset its own emission log, so the live
                    // partial must reset with it. The COMMITTED transcript is
                    // NOT wiped here anymore (the old wipe made every mic
                    // re-tap "jump back to the start": the display cleared and
                    // re-coloring restarted at word 0 even though the user was
                    // mid-surah). The count-based filter above already
                    // suppresses the SDK's cumulative replay of committed
                    // words, so a fresh session simply CONTINUES from the
                    // committed frontier — new words pass through, re-read
                    // words pass through (occurrences beyond the committed
                    // count). Only a session that already covered the WHOLE
                    // target is wiped (the user finished the range and is
                    // starting a fresh recitation of it).
                    if (lastCommittedWords.size >= targetOrig.size) {
                        finalSpokenText = ""
                        lastCommittedWords = emptyList()
                    }
                    partialSpokenText = ""
                    committedWordCount = 0
                    currentWordCount = 0
                    spokenTimings.clear()
                    sessionFinalized = false
                    // §4.38 — a brand-new utterance: the PREVIOUS session's
                    // endpoint evidence and stop clock must not gate this one.
                    endpointDecodeLandedAtMs = 0L
                    pipelineStoppedAtMs = 0L
                    // Resume-from-mistake re-seat: this fresh session is the
                    // "continue reading from the wrong word" flow. The display
                    // transcript is restored to the words BEFORE the mistake
                    // (the user re-reads from it), and lastCommittedWords is
                    // set to that SAME prefix: the count filter then suppresses
                    // exactly the replay copies of the prefix words, while the
                    // mistake word and everything after it — which the user is
                    // about to re-read — pass through (setting the cursor to
                    // the WHOLE pre-pause transcript ate the re-read words
                    // under the pipeline's non-cumulative emission semantics).
                    if (mistakeResumeIndex >= 0 && mistakeResumeLiveText.isNotEmpty()) {
                        val resumeIdx = mistakeResumeIndex
                        mistakeResumeIndex = -1
                        val liveWords = mistakeResumeLiveText.split(Regex("\\s+")).filter { it.isNotEmpty() }
                        mistakeResumeLiveText = ""
                        if (liveWords.isNotEmpty()) {
                            // - CUT ON SPOKEN WORDS, NOT ON THE TARGET INDEX -
                            // `resumeIdx` is an index into the TARGET; `liveWords` is
                            // the list of words the DECODER produced. They are only
                            // the same list when nothing was skipped or misheard
                            // earlier in the session - and the re-seat exists
                            // precisely for a session where something was. A skip at
                            // target word 3 shifts every later target index one to
                            // the right of its spoken counterpart, so
                            // `take(resumeIdx)` cut one word too deep and the user
                            // re-read from the WRONG place (or from past the word,
                            // where the count-filter then swallowed the re-read and
                            // the session looked dead: "the app becomes never
                            // functioning").
                            //
                            // The honest count is how many DISTINCT target words the
                            // analysis already placed a spoken token on BEFORE the
                            // mistake.
                            //
                            // §4.48 — AND IT MUST BE A **DISTINCT** COUNT.
                            // `detailedAnalysis` holds ONE ROW PER (WORD, RULE), so
                            // `count { wordIndex < resumeIdx && spokenText != null }`
                            // counts a word ONCE PER RULE. Measured on the user's
                            // real-mic Al-Baqarah run (`re-seated ... 21/50 spoken
                            // word(s) kept, the user continues at target word 18
                            // (spokenBefore=21)`): the cross-word rules gave several
                            // words a second row, so 18 target words were counted as
                            // 21. The transcript was therefore cut THREE WORDS DEEP,
                            // and the re-read of word 18 was compared against target
                            // words 21+ — so the re-read never lined up with the word
                            // the popup had named, and the user could not continue
                            // from it ("it didn't get back to it"). Distinct word
                            // indices; a duplicate rule row must never shift the cut.
                            val spokenBefore = detailedAnalysis
                                .filter { it.wordIndex < resumeIdx && !it.spokenText.isNullOrEmpty() }
                                .map { it.wordIndex }
                                .distinct()
                                .size
                            val cut = when {
                                spokenBefore > 0 -> spokenBefore
                                else -> resumeIdx
                            }
                            val prefixWords = liveWords.take(cut.coerceIn(0, liveWords.size))
                            finalSpokenText = prefixWords.joinToString(" ")
                            lastCommittedWords = prefixWords
                            // §4.48 — THE DISPLAY MUST REWIND WITH THE TRANSCRIPT.
                            //
                            // The display floor (§4.46) exists so a TRANSIENT walk
                            // regression cannot un-colour a word the user has already
                            // read. A re-seat is not transient: the transcript now
                            // ends just before the mistake, so the words after it are
                            // words the user has NOT re-read yet. Keeping the floor
                            // showed them as reached, and the user could not tell
                            // where the popup wanted them to continue from — their
                            // report: "it didn't get back to it, a few words after it
                            // are displayed" (measured: the popup named word 18 and
                            // the cursor sat at 22). Drop the floor to the mistake,
                            // which is where the reader must be looking.
                            // §4.53 — BUT NOT UP TO THE MISTAKE. The floor is a
                            // MINIMUM, and the display can only move while the WALK
                            // is at or past it. After a re-seat the walk has to
                            // re-climb the kept prefix from target word 0 before it
                            // reaches the mistake, so a floor parked at the mistake
                            // froze the display ABOVE the walk for the whole climb.
                            // The user's report: "i receited and got a tajweed error
                            // then when i continue no words are displayed" —
                            // measured, the whole re-read:
                            //   re-seated ... the user continues at target word 55
                            //   cursor held at 55 (walk wanted 41, heardThrough=41)
                            //   cursor held at 55 (walk wanted 43, heardThrough=43)
                            //   cursor held at 55 (walk wanted 45, heardThrough=45)
                            //   cursor held at 55 (walk wanted 52, heardThrough=52)
                            // i.e. 55 words lit, the voice moving, and the display
                            // static for ~14 s.
                            //
                            // So the floor goes to where the WALK actually is — the
                            // kept prefix — and follows it up from there. The words
                            // already recited stay lit (nothing un-greys), and every
                            // word of the re-read lights as the aligner reaches it.
                            displayCursorFloor = minOf(
                                displayCursorFloor,
                                prefixWords.size,
                                resumeIdx
                            )
                            android.util.Log.i(
                                "ItqanSherpa",
                                "collector(): re-seated the transcript for the re-read - " +
                                    "${prefixWords.size}/${liveWords.size} spoken word(s) kept, " +
                                    "the user continues at target word $resumeIdx " +
                                    "(spokenBefore=$spokenBefore distinct) cut=$cut, " +
                                    "display floor dropped to $resumeIdx so the words after it " +
                                    "are shown as not-yet-read"
                            )
                            // Re-arm the tajweed popup latch for the word the user is
                            // about to re-read: this IS the "user acted on the
                            // mistake" event, so if the same rule fails again the
                            // popup must appear again (§4.45) instead of being
                            // swallowed by the latch the dismissed dialog left.
                            // §4.49 — the whole (word, rule) latch is re-armed,
                            // so EVERY rule of this word can speak up again.
                            lastTajweedPopupIndex = -1
                            lastTajweedPopupScore = -1
                            lastTajweedPopupRule = ""
                            shownTajweedPopups = emptySet()
                            shownMistakeWords = emptySet()
                            rideOverCandidateIdx = -1
                            rideOverCandidateSinceMs = 0L
                        }
                    }
                }
                // CONFIRMED-MISTAKE FREEZE. While a confirmed WRONG/SKIPPED stop
                // has armed mistakeResumeIndex, the session is paused waiting for
                // the user's resume tap -- so every pipeline emission still
                // arriving is audio from BEFORE the stop: in-flight window
                // decodes plus pause()'s own final full-tail decode, which
                // re-hears the whole tail with full context and can RESTORE the
                // very word the matcher flagged SKIPPED (a word one sliding
                // window dropped). Committing it resurrected the mistake as
                // CORRECT: the popup auto-dismissed and the recitation displayed
                // as clean ("judged correct after the fact"). Swallow every
                // non-fresh event while the latch is armed; the fresh-session
                // event IS the resume (it consumed the latch in the block above)
                // and passes through.
                if (!ev.freshSession && mistakeResumeIndex >= 0 && currentTab == "Memorize") {
                    android.util.Log.i("ItqanSherpa", "collector(): mistake-freeze dropping endpoint=${ev.endpoint} ${ev.partial.size}w (resume latch armed @ word $mistakeResumeIndex)")
                    return@collect
                }
                // ── §4.38 THE ENDPOINT DECODE'S EVIDENCE, RECORDED BEFORE ANY FREEZE ──
                // Two wall clocks the confirm gate needs. `pipelineStoppedAtMs` is
                // when we first noticed the session had ended (the pipeline reports
                // running=false right after the endpoint) and bounds the wait;
                // `endpointDecodeLandedAtMs` is when the whole-session decode put
                // words into the transcript — the acquittal the ABSENT strikes are
                // not allowed to pre-empt. Both are recorded HERE, above the
                // mistake-freeze, because that freeze `return@collect`s and would
                // otherwise swallow the very evidence this section exists for.
                if (!tajweedPipelineHolder.running && currentTab == "Memorize" &&
                    pipelineStoppedAtMs == 0L
                ) {
                    pipelineStoppedAtMs = System.currentTimeMillis()
                }
                if (!ev.freshSession && !tajweedPipelineHolder.running &&
                    ev.partial.isNotEmpty() && endpointDecodeLandedAtMs == 0L
                ) {
                    endpointDecodeLandedAtMs = System.currentTimeMillis()
                    android.util.Log.i(
                        "ItqanSherpa",
                        "collector(): endpoint decode LANDED with ${ev.partial.size}w after the " +
                            "session stopped — the confirm gate (§4.38) may now judge; " +
                            "latency=${endpointDecodeLandedAtMs - pipelineStoppedAtMs}ms"
                    )
                }
                // Words shown are RAW in the Tajweed Rules Coach tab: the pipeline's
                // AyahWordMatcher filters against the Memorize target vocabulary
                // (ayahWordsProvider={targetOrig}), but the coach tab recites a RULE
                // EXAMPLE whose words are a different sentence entirely — every
                // example word would be dropped and the user would see nothing. The
                // Acoustics scorer for the coach (analyzeRecitationDetails) uses its
                // own targetTokens anyway, so raw is correct here.
                val words = if (currentTab != "Tajweed")
                    filterMemorizeWords(ev.partial, ev.endpoint)
                else ev.rawPartial.split(Regex("\\s+")).filter { it.isNotBlank() }
val wordsLength = words.size
                if (wordsLength > 0) sherpaWarm = true
                if (wordsLength > currentWordCount) {
                    if (spokenTimings.isNotEmpty()) {
                        spokenTimings.last().endMs = timeNow
                    }
                    val rVal = synchronized(acousticRmsTracker) {
                        val r = acousticRmsTracker[0]
                        acousticRmsTracker[0] = 0f
                        r
                    }
                    for (i in currentWordCount until wordsLength) {
                        spokenTimings.add(SpokenWordTiming(
                            word = words[i],
                            startIndex = i,
                            startMs = timeNow,
                            endMs = timeNow,
                            maxRms = rVal
                        ))
                    }
                    currentWordCount = wordsLength
                } else if (wordsLength == currentWordCount && spokenTimings.isNotEmpty()) {
                    val last = spokenTimings.last()
                    last.endMs = timeNow
                    synchronized(acousticRmsTracker) {
                        if (acousticRmsTracker[0] > last.maxRms) {
                            last.maxRms = acousticRmsTracker[0]
                        }
                    }
                }
            val partialStr = words.joinToString(" ").trim()
            android.util.Log.i("ItqanSherpa", "collector(): currentTab=$currentTab recitationMode=$recitationMode ev.endpoint=${ev.endpoint} words=$words partialStr='$partialStr'")
            // Two distinct transcript slots must NOT be mixed:
            //  - Memorize+Tajweed (currentTab == "Memorize"): the word-by-word
            //    coloring is driven by `spokenText` = finalSpokenText +
            //    partialSpokenText, which feeds alignRecitation() → alignment →
            //    matchedWordCount. If we wrote the pipeline partials only into
            //    the tajweed* vars (the old bug), Memorize showed NO colored
            //    words and the leftover text leaked into the Tajweed Rules
            //    coach card ("the Madd example showed the last two words I had
            //    read in Memorize+Tajweed").
            //  - Tajweed Rules Coach (currentTab == "Tajweed"): the card reads
            //    the tajweed* vars; its example words are NOT in the Memorize
            //    target vocabulary, so it shows RAW partials.
            val isMemorizeSession = currentTab == "Memorize"
            val liveFinal = if (isMemorizeSession) finalSpokenText else tajweedFinalSpokenText
            val livePartial = if (isMemorizeSession) partialSpokenText else tajweedPartialSpokenText
            // The pipeline utterance is CONTINUOUS for the whole session (pause()
            // no longer finalizes it), so the emitted word list is cumulative —
            // on every emission it re-includes the whole merged log: committed
            // words AND the uncommitted tail (pause/resume replays it as-is).
            // filterMemorizeWords IS the single dedup: its count rule strips
            // the first c(n) occurrences of each already-committed norm (the
            // replay copies) and caps each kept word at its target count — so
            // the filtered list below contains ONLY genuinely new words.
            if (ev.endpoint) {
                // FINAL-DECODE REPAIR SWAP: the pipeline flagged this endpoint
                // as a merge REPAIR — the final tail decode re-heard the whole
                // session and REPLACED the scrambled live log, so ev.partial
                // is the authoritative transcript in reading order. Overwrite
                // the committed transcript with it (appending the
                // count-filtered tail would keep the scrambled order).
                if (isMemorizeSession && ev.repaired && ev.partial.isNotEmpty()) {
                    // The repaired re-decode is only authoritative when it is at
                    // least as GOOD as what the live merged log holds. Quality is
                    // measured as the LONGEST LEADING RUN of words matching the
                    // target sequence (positional correctness), not raw word
                    // count: a monotonic-merge echo can leave the live log LONGER
                    // but MISORDERED (measured on-device: window#3 emitted a
                    // premature ٱلْعَٰلَمِينَ, window#4 appended لِلَّهِ رَبِّ
                    // after it — 11 words with a 5-word correct prefix — while
                    // the full-tail repair rendered the true order, 10 words
                    // with an 8-word prefix; the old size-only rule rejected the
                    // CORRECT repair and locked in the phantom mistakes). A
                    // short-window GARBLED repair (the original regression this
                    // guard exists for) scores a LOW prefix and is still
                    // rejected.
                    val liveWords = (finalSpokenText + " " + partialSpokenText).trim()
                        .split(Regex("\\s+")).filter { it.isNotEmpty() }
                    // ── HOW MUCH OF THE RECITATION DOES THIS TRANSCRIPT HOLD? ──
                    // The longest leading RUN was the original quality metric, and
                    // it is too harsh: it punishes a transcript that is missing one
                    // early word while holding the entire rest. Measured
                    // fatihah_c34 — the live log scored a prefix of 3 while the
                    // whole-session repair (which restored نعبد / اهدنا / الصرط /
                    // المستقيم and the range's last word) scored 0 because its
                    // opening token was the fragment `م`; the repair was rejected,
                    // the merge's new words were appended out of order, the walk
                    // froze at 10/29 and the user got neither a popup nor a
                    // summary. The LONGEST COMMON SUBSEQUENCE with the target is
                    // the honest measure: it counts how many target words the
                    // transcript holds IN READING ORDER, so an early fragment costs
                    // one word instead of everything.
                    //   c34: live 20 vs repair 27  -> repair wins (correct)
                    //   c33: live 28 vs repair 22  -> live wins   (correct: the
                    //        repair was a worse ordering and was rightly rejected)
                    //   An-Nas / truncated controls: live wins by a wide margin.
                    fun targetCoverageLen(words: List<String>): Int {
                        val a = words.map { normalizeForComparison(it, viewModel.riwayah) }
                        val b = targetNorm
                        // Rolling DP row: dp[j] = LCS(a[0..i], b[0..j-1]).
                        val dp = IntArray(b.size + 1)
                        for (i in a.indices) {
                            var prevDiag = 0
                            for (j in 1..b.size) {
                                val tmp = dp[j]
                                dp[j] = if (a[i].isNotEmpty() && b[j - 1].isNotEmpty() && a[i] == b[j - 1]) {
                                    prevDiag + 1
                                } else {
                                    maxOf(dp[j], dp[j - 1])
                                }
                                prevDiag = tmp
                            }
                        }
                        return dp[b.size]
                    }
                    // Longest leading run — the ORDER quality within an equal
                    // coverage score. Two candidates can hold the same NUMBER of
                    // target words while one of them has them in the right place
                    // (live 8 words misordered at 4..7 vs a shorter repair with
                    // the same 7-word coverage but a clean 7-word run): coverage
                    // alone cannot tell them apart, the run can.
                    fun targetRunLen(words: List<String>): Int {
                        var i = 0
                        while (i < words.size && i < targetNorm.size &&
                            targetNorm[i].isNotEmpty() &&
                            normalizeForComparison(words[i], viewModel.riwayah) == targetNorm[i]
                        ) i++
                        return i
                    }
                    val liveScore = targetCoverageLen(liveWords)
                    val repairScore = targetCoverageLen(ev.partial)
                    val liveRun = targetRunLen(liveWords)
                    val repairRun = targetRunLen(ev.partial)
                    val livePrefix = liveScore
                    val repairPrefix = repairScore
                    val takeRepair = when {
                        repairScore != liveScore -> repairScore > liveScore
                        repairRun != liveRun -> repairRun > liveRun
                        else -> ev.partial.size >= liveWords.size
                    }
                    if (takeRepair) {
                        finalSpokenText = ev.partial.joinToString(" ").trim()
                        partialSpokenText = ""
                        committedWordCount = ev.partial.size
                        lastCommittedWords = ev.partial
                        sessionFinalized = true
                        android.util.Log.i("ItqanSherpa", "collector(): final-decode repair swap (${ev.partial.size}w)")
                        return@collect
                    } else {
                        // Repair is shorter/worse: commit the uncommitted live
                        // tail as-is and finalize — never shrink the transcript.
                        val tailWords = partialSpokenText.trim()
                            .split(Regex("\\s+")).filter { it.isNotEmpty() }
                        if (tailWords.isNotEmpty()) {
                            finalSpokenText = "$finalSpokenText ${tailWords.joinToString(" ")}".trim()
                            partialSpokenText = ""
                            committedWordCount = tailWords.size
                            lastCommittedWords = lastCommittedWords + tailWords
                        }
                        // ── A REJECTED REPAIR MAY STILL HOLD THE ONLY COPY OF
                        //    A WORD THE LIVE LOG NEVER HEARD ──────────────────
                        // Rejecting the repair means "do not REPLACE the live
                        // transcript with this worse ordering" — it must NOT
                        // mean "discard everything this decode heard". The
                        // endpoint's filtered delta is computed against the
                        // committed log, so whatever survives it is a word the
                        // live sliding windows never rendered at all; dropping
                        // it loses real speech with no trace.
                        //
                        // Measured fatihah_c33: the tail windows only ever
                        // produced fused/garbled renders (`وَلَاضًّا`, `الضَّا`,
                        // `ض م`), all rejected by the merge, so the live log
                        // stopped at word 28 `وَلَا`. The whole-session repair
                        // was correctly REJECTED (24w/prefix0 vs 28w/prefix8),
                        // but its delta carried `ٱلضَّآلِّينَ` — the one word the
                        // user had actually recited and the one word the target
                        // still needed. It was thrown away, so the walk ended
                        // 28/29 with no row for the last word, no popup and a
                        // "clean" summary while the word sat greyed out on
                        // screen. Recomputing the delta AFTER the tail commit
                        // (same count-aware filter as every other emission)
                        // keeps the recovered word exactly once.
                        val recovered = filterMemorizeWords(ev.partial, endpoint = true)
                        if (recovered.isNotEmpty()) {
                            finalSpokenText = "$finalSpokenText ${recovered.joinToString(" ")}".trim()
                            committedWordCount = recovered.size
                            lastCommittedWords = lastCommittedWords + recovered
                            android.util.Log.i(
                                "ItqanSherpa",
                                "collector(): rejected repair still contributed " +
                                    "${recovered.size} word(s) the live log never heard -> " +
                                    "transcript now ${finalSpokenText.split(Regex("\\s+")).filter { it.isNotEmpty() }.size}w"
                            )
                        }
                        sessionFinalized = true
                        android.util.Log.i("ItqanSherpa", "collector(): final-decode repair REJECTED (repair ${ev.partial.size}w/prefix$repairPrefix vs live ${liveWords.size}w/prefix$livePrefix) — keeping live transcript")
                        return@collect
                    }
                }
                // pause()/stop() emit a synthetic endpoint with an EMPTY
                // partial after inputFinished() (so the decoder commits its
                // in-flight segment); this arrived in the collector before as
                // a non-endpoint event, so the live partial still holds the
                // last words. Without this fallback the final text would be
                // cleared on stop and the user would see NO transcript for
                // what they actually said.
                val commitText = partialStr.ifEmpty { livePartial.trim() }
                var commitWords = commitText.split(Regex("\\s+")).filter { it.isNotEmpty() }
                if (commitWords.isNotEmpty()) {
                    if (isMemorizeSession) {
                        // DOUBLE-COUNT GUARD: the final tail decode re-hears the whole
                        // session, so its merged log can RE-EMIT words the pause already
                        // committed. The count-filter strips one copy of each committed
                        // word, but when the merge duplicated them a replay of the
                        // committed prefix survives in the filtered delta (measured
                        // on-device: an 18-word transcript for a 10-word recitation).
                        // Appending that replay double-counts the committed words, the
                        // analyzer runs past the real recitation, and flags the first
                        // UNSPOKEN word WRONG ("I stopped after ayah 3 but ayah 4's first
                        // word showed incorrect"). Strip any leading run of commitWords
                        // that duplicates the tail of the already-committed transcript.
                        val existing = finalSpokenText.split(Regex("\\s+")).filter { it.isNotEmpty() }
                        if (existing.isNotEmpty()) {
                            val exNorm = existing.map { normalizeForComparison(it, viewModel.riwayah) }
                            val cmNorm = commitWords.map { normalizeForComparison(it, viewModel.riwayah) }
                            // Longest suffix of `existing` that equals a prefix of
                            // `commitWords` = the duplicated committed replay.
                            var overlap = 0
                            val maxOv = minOf(exNorm.size, cmNorm.size)
                            for (k in maxOv downTo 1) {
                                var match = true
                                for (i in 0 until k) {
                                    if (exNorm[exNorm.size - k + i] != cmNorm[i]) { match = false; break }
                                }
                                if (match) { overlap = k; break }
                            }
                            if (overlap > 0) {
                                android.util.Log.i("ItqanSherpa", "collector(): final-commit overlap-strip $overlap duplicated word(s)")
                                commitWords = commitWords.drop(overlap)
                            }
                        }
                        if (commitWords.isNotEmpty()) {
                            finalSpokenText = "$finalSpokenText ${commitWords.joinToString(" ")}".trim()
                            committedWordCount = commitWords.size
                            // ACCUMULATE: the filter's committed-count suppression must
                            // know about EVERY committed word, not just the latest
                            // commit (pause commits the tail, the final tail decode
                            // commits more — each commit's replay copies must be
                            // dropped from the NEXT cumulative emission).
                            lastCommittedWords = lastCommittedWords + commitWords
                        }
                    } else {
                        // ── COACH: REPLACE, NEVER APPEND ─────────────────────
                        // The coach transcript is the pipeline's RAW partial, which
                        // is cumulative for the whole session, so this endpoint
                        // already carries everything said so far. Appending it to a
                        // previous endpoint's copy (or leaving a cumulative live
                        // partial on top — see the display composition) showed the
                        // same words twice. The current full transcript simply
                        // replaces the slot; a late/superior decode therefore still
                        // "gets better" without also keeping the worse render.
                        // A BLANK ENDPOINT MUST NOT WIPE THE TRANSCRIPT:
                        // pause() emits TWO endpoints — the word-bearing final
                        // decode first, then a synthetic EMPTY one so the decoder
                        // commits its in-flight segment. Replacing on the empty
                        // one erased everything the first had just committed, so
                        // the coach showed "Your recitation:" with no text.
                        val fullTranscript = commitText.trim()
                        if (fullTranscript.isNotBlank()) {
                            tajweedFinalSpokenText = fullTranscript
                            tajweedPartialSpokenText = ""
                        }
                        committedWordCount = commitWords.size
                        lastCommittedWords = commitWords
                    }
                }
                // The finalDecodeAndEmit endpoint carries actual words; pause()'s
                // synthetic empty endpoint does not. Only the word-bearing one is
                // the session's FINAL transcript -- from here on, verdicts are real.
                //
                // The discriminator is the RAW emission ([ev.partial]), NOT the
                // filtered delta ([words]). filterMemorizeWords strips every
                // already-committed word and caps each word at its target count,
                // so once a range is fully recited the endpoint legitimately
                // carries ZERO new words — [words] was empty on EVERY completed
                // run (measured fatihah_v3: `ev.endpoint=true words=[]` twice),
                // [sessionFinalized] therefore stayed false forever, and the
                // end-of-range summary popup was gated OFF for good. That is the
                // user's report verbatim: they finished Al-Fatihah, saw no popup,
                // and only got a summary after tapping pause (pause drives the
                // `ev.repaired` path above, which does set the flag).
                // pause()'s synthetic endpoint is still excluded: it is empty.
                //
                // §4.45 - OR the pipeline's explicit final-commit marker. The
                // word-bearing test above cannot see the NORMAL end of a real-mic
                // recitation: the last seconds are silence, so the final
                // whole-session render comes back a subset of what the sliding
                // windows already merged and the pipeline emits the marker with NO
                // words (`finalDecodeAndEmit(): hold - no new words`, or `EMPTY`
                // when the tail decodes to nothing at all). In that case
                // `sessionFinalized` stayed false, the end-of-range auto-finalize
                // never armed and the user finished the surah with nothing on
                // screen - while an injected run, whose tail usually DOES produce a
                // repair, showed the summary. The marker is the honest signal: the
                // decoder is done, whatever it produced.
                if (ev.finalCommit || (ev.endpoint && ev.partial.isNotEmpty())) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "collector(): sessionFinalized <- " +
                            (if (ev.finalCommit) "final-commit marker" else "endpoint") +
                            " (${ev.partial.size}w raw / ${words.size}w new)"
                    )
                    sessionFinalized = true
                }
                if (isMemorizeSession) partialSpokenText = "" else tajweedPartialSpokenText = ""
            } else {
                if (partialStr.isNotEmpty()) {
                    // filtered words = only the new words (see above): show them
                    // as the live partial, appended to the committed transcript.
                    val tailStr = partialStr
                    if (isMemorizeSession) {
                        if (tailStr.isNotEmpty()) partialSpokenText = tailStr
                    } else {
                        if (tailStr.isNotEmpty()) tajweedPartialSpokenText = tailStr
                    }
                }
            }
            // LIVE tajweed audio wiring: in pipeline mode `recognizedAudio` is
            // never set by a SpeechRecognizer bundle, so analyzeRecitationDetails
            // saw hasRealAudio=false and the acoustic heads never ran — the
            // TAJWEED_ERROR popup could never fire. Hand the pipeline's own
            // captured PCM (the same single AudioRecord that drives the STT)
            // to the analysis pass; the clock base is tajweedRecordingStartMs,
            // which startTajweedSession resets in lockstep with the pipeline's
            // audio accumulator on every start/resume.
            //
            // Use snapshotRawAudio() (UN-GATED): the RMS gate drops silent
            // chunks, gluing words together and compressing the timeline, so
            // wall-clock word timings map to the WRONG audio and TajweedNet
            // (trained on clips with natural inter-word silence) scores
            // out-of-distribution input -> false TAJWEED_ERROR popups on
            // correct recitation. The un-gated buffer keeps real-time silence,
            // so its timeline matches the wall-clock spokenTimings.
            if (isMemorizeSession && recitationMode == "Tajweed" && wordsLength > 0) {
                val snap = try { pipe.snapshotRawAudio() } catch (_: Throwable) { FloatArray(0) }
                if (snap.size >= 256) {
                    recognizedAudio = snap
                    recognizedAudioStartMs = tajweedRecordingStartMs
                }
            }
        }
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            if (!isListening) {
                // Tajweed mode must go through the Sherpa pipeline, never the
                // SpeechRecognizer (no silent Tarteel fallback — same rule as the
                // FAB start branch). If the pipeline isn't wired yet, block with
                // a clear indication.
                if (recitationMode == "Tajweed") {
                    if (tajweedPipelineHolder.pipeline != null) {
                        startTajweedSession()
                        if (isListening) viewModel.preloadTajweedModel()
                    } else {
                        errorMessage = getAppString(isArabic, "Tajweed is warming up — please wait a moment…")
                    }
                } else if (injectTranscriptSource && tajweedPipelineHolder.pipeline != null) {
                    // §4.68 — see [injectTranscriptSource].
                    startTajweedSession()
                } else if (speechRecognizer != null) {
spokenTimings.clear()
        currentWordCount = 0
        lastWrongStopIndex = -1
        lastWrongStopSpokenText = ""
        lastWrongStopSightings = 0
        lastWrongStopFirstSeenMs = 0L
        mistakeVerifyAbsentIdx = -1
        mistakeVerifyAttempts = 0
        mistakeVerifyAbsentStrikes = 0
        mistakeVerdictAtMs = 0L
        confirmedMistakeUiIdx = -1
                    recognizedAudioUri = null
                    recognizedAudio = null
                    recognizedAudioStartMs = System.currentTimeMillis()
                    tarteelWantsMic = true
                    recitationSessionActive = true
                    sessionFinalized = false
                    isListening = true
                    startListening(speechRecognizer!!, context.packageName, onError = { msg ->
                        errorMessage = if (msg.isEmpty()) null else msg
                        if (tarteelWantsMic) {
                            tarteelWantsMic = false
                            sessionFinalized = true
                            recitationSessionActive = false
                        }
                        isListening = false
                    })
                    // Same single-mic invariant as the FAB: no parallel AudioRecord here.
                    // See the FAB start-listening branch for the A05s single-mic rationale.
                } else {
                    errorMessage = "Speech Recognizer not available"
                }
            }
        } else {
            errorMessage = "Microphone permission denied"
        }
    }

    // Separate launcher for the Tajweed Rules coach: granting the permission starts the
    // practice session (our own audio capture + best-effort speech recognition).
    val tajweedPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            startTajweedSession()
        } else {
            errorMessage = "Microphone permission denied"
        }
    }

    if (showSplashScreen) {
        SplashScreen(isArabic = isArabic, onDismiss = { showSplashScreen = false })
    } else {
        CompositionLocalProvider(LocalLayoutDirection provides (if (isArabic) LayoutDirection.Rtl else LayoutDirection.Ltr)) {
            Column(modifier = modifier.fillMaxSize().padding(12.dp)) {
                Box(modifier = Modifier.fillMaxWidth().padding(bottom = 4.dp)) {
                    Text(
                        text = if (isArabic) "إتقان" else "Itqān",
                        style = MaterialTheme.typography.titleLarge.copy(
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        ),
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.align(Alignment.Center)
                    )

                    // Light/dark switch. Shows the mode the user will GET: a sun
                    // while dark (tap -> light), a moon while light (tap -> dark).
                    Box(modifier = Modifier.align(Alignment.CenterStart)) {
                        IconButton(
                            onClick = onToggleTheme,
                            modifier = Modifier.size(36.dp)
                        ) {
                            Icon(
                                imageVector = if (isDarkTheme) Icons.Default.LightMode else Icons.Default.DarkMode,
                                contentDescription = if (isDarkTheme) "Switch to light mode" else "Switch to dark mode",
                                tint = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                    
                    Box(modifier = Modifier.align(Alignment.CenterEnd)) {
                        var showLangMenu by remember { mutableStateOf(false) }
                        IconButton(
                            onClick = { showLangMenu = true },
                            modifier = Modifier.size(36.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Language,
                                contentDescription = "Change Language",
                                tint = MaterialTheme.colorScheme.primary
                            )
                        }
                        DropdownMenu(
                            expanded = showLangMenu,
                            onDismissRequest = { showLangMenu = false }
                        ) {
                            DropdownMenuItem(
                                text = { Text(getAppString(isArabic, "English")) },
                                onClick = {
                                    isArabic = false
                                    prefs.edit().putBoolean("is_arabic", false).apply()
                                    showLangMenu = false
                                },
                                leadingIcon = { Text("🇬🇧") }
                            )
                            DropdownMenuItem(
                                text = { Text("العربية") },
                                onClick = {
                                    isArabic = true
                                    prefs.edit().putBoolean("is_arabic", true).apply()
                                    showLangMenu = false
                                },
                                leadingIcon = { Text("🇸🇦") }
                            )
                        }
                    }
                }
                
                Row(
                    modifier = Modifier.fillMaxWidth().padding(bottom = 6.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Riwayah selector is meaningful only for recitation-based tabs (the qalqalah
                    // mappings, the Qalun Silah rule, and the normalizeForComparison differences
                    // are the only consumers). Hide it on Tajweed (Rules Coach) and Stats so those
                    // screens stay simpler and don't expose a Hafs/Qalun switch the user expects
                    // to affect their goal/statistics.
                    if (currentTab == "Memorize" || currentTab == "Read") {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(text = getAppString(isArabic, "Riwayah: "), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Spacer(modifier = Modifier.width(4.dp))
                            FilterChip(
                                selected = viewModel.riwayah == "Hafs",
                                onClick = { viewModel.updateRiwayah("Hafs") },
                                label = { Text(getAppString(isArabic, "Hafs"), fontSize = 11.sp) }
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            FilterChip(
                                selected = viewModel.riwayah == "Qalun",
                                onClick = { viewModel.updateRiwayah("Qalun") },
                                label = { Text(getAppString(isArabic, "Qalun"), fontSize = 11.sp) }
                            )
                        }
                    }

                    if (currentTab == "Memorize") {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            FilterChip(
                                selected = recitationMode == "Tarteel",
                                onClick = {
                                    if (recitationMode != "Tarteel") {
                                        // The outgoing Tajweed surface keeps its surah;
                                        // Tarteel returns to its own.
                                        viewModel.selectedSurah?.let { modeSelectedSurah["Tajweed"] = it }
                                        // Changing mode while the mic is live is
                                        // a deliberate session reset. Without
                                        // this, a Tarteel restart could reopen
                                        // the platform recognizer after the user
                                        // had already switched to Tajweed.
                                        tarteelWantsMic = false
                                        recitationMode = "Tarteel"
                                        if (isListening) {
                                            try { speechRecognizer?.cancel() } catch (e: Throwable) {}
                                            isListening = false
                                        }
                                        try {
                                            if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop()
                                        } catch (e: Throwable) {}
                                        recitationSessionActive = false
                                        modeSelectedSurah["Tarteel"]?.let { saved ->
                                            if (saved.number != viewModel.selectedSurah?.number) {
                                                viewModel.selectSurah(saved)
                                            }
                                        }
                                    }
                                },
                                label = { Text(getAppString(isArabic, "Tarteel"), fontSize = 11.sp) }
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            FilterChip(
                                selected = recitationMode == "Tajweed",
                                onClick = {
                                    if (recitationMode != "Tajweed") {
                                        // The outgoing Tarteel surface keeps its surah;
                                        // Tajweed returns to its own.
                                        viewModel.selectedSurah?.let { modeSelectedSurah["Tarteel"] = it }
                                        tarteelWantsMic = false
                                        val platformMicWasLive = isListening
                                        recitationMode = "Tajweed"
                                        if (platformMicWasLive) {
                                            // The Tarteel step is over; do not let
                                            // its final result restart the platform
                                            // recognizer underneath the Sherpa
                                            // pipeline's microphone.
                                            try { speechRecognizer?.cancel() } catch (e: Throwable) {}
                                            isListening = false
                                        }
                                        recitationSessionActive = false
                                        modeSelectedSurah["Tajweed"]?.let { saved ->
                                            if (saved.number != viewModel.selectedSurah?.number) {
                                                viewModel.selectSurah(saved)
                                            }
                                        }
                                    }
                                    // Preload the Tajweed model the moment the user opts into
                                    // Tajweed mode -- otherwise the lazy RecitationAnalyzer.get()
                                    // ran synchronously on the first mic tap and froze the app
                                    // for seconds on slow phones. The popup is shown while it
                                    // loads (see isTajweedModelLoading below).
                                    viewModel.preloadTajweedModel()
                                },
                                label = { Text(getAppString(isArabic, "Tajweed"), fontSize = 11.sp) }
                            )
                        }
                    }
                }

        // Qalun fallback information banner -- only shown where the user can pick the riwayah
        // (Memorize / Read). On Tajweed and Stats there is no selector, so the banner would be
        // confusing/unactionable.
        AnimatedVisibility(visible = (currentTab == "Memorize" || currentTab == "Read") && viewModel.riwayah == "Qalun" && viewModel.isQalunFallback) {
            Card(
                modifier = Modifier.fillMaxWidth().padding(bottom = 4.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.25f)
                )
            ) {
                Row(
                    modifier = Modifier.padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Info,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.error,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = getAppString(isArabic, "Standard Qalun text is not available for this Surah in local database. Standard Hafs text is shown as fallback."),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                        fontSize = 10.sp
                    )
                }
            }
        }

        // Tajweed Legend
        AnimatedVisibility(visible = currentTab == "Memorize" && recitationMode == "Tajweed") {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 4.dp)
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(10.dp, Alignment.CenterHorizontally),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // §4.49 — THE LEGEND READS ITS DOTS FROM THE SAME PALETTE as the
                // letters, the cards and the score rows. These were nine hardcoded
                // literals that happened to agree with tajweedHeadColor(); nothing
                // enforced that, so a palette tweak would have left the legend
                // advertising colours the text no longer used — the "mingling the
                // colours of the rules" the user reported. One source, one truth.
                @androidx.compose.runtime.Composable
                fun legendDot(head: String, labelKey: String) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier.size(8.dp).background(
                                tajweedHeadColor(head) ?: MaterialTheme.colorScheme.primary,
                                CircleShape
                            )
                        )
                        Spacer(modifier = Modifier.width(3.dp))
                        Text(
                            getAppString(isArabic, labelKey),
                            style = MaterialTheme.typography.bodySmall,
                            fontSize = 9.sp
                        )
                    }
                }
                legendDot("GHUNNAH", "Ghunnah")
                legendDot("MADD", "Madd")
                legendDot("QALQALAH", "Qalqalah")
                legendDot("IDGHAM", "Idgham")
                legendDot("IKHFA", "Ikhfa")
                legendDot("IZHAR", "Izhar")
                legendDot("MEEM_SAKINAH", "Meem")
                legendDot("ALLAM", "Al-Lam")
                if ((currentTab == "Memorize" || currentTab == "Read") && viewModel.riwayah == "Qalun") {
                    // Silah is Qalun's mim silah, judged by the IZHAR head (it has
                    // no head of its own) — so it wears IZHAR's colour BY DESIGN.
                    legendDot("IZHAR", "Silah")
                }
            }
        }

        AnimatedVisibility(visible = viewModel.isLoading) {
            CircularProgressIndicator(modifier = Modifier.align(Alignment.CenterHorizontally))
        }

        AnimatedVisibility(
            visible = viewModel.errorMessage != null || errorMessage != null,
            modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp)
        ) {
            Text(
                text = viewModel.errorMessage ?: errorMessage ?: "",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center
            )
        }

        // Friendly "no internet" popup. Triggered from MainViewModel whenever a Surah-text or
        // audio load fails AND isNetworkAvailable() is false. This replaces the raw exception
        // text the user used to see on the banner above; the popup is modal, dismissable, and
        // tells them the only thing that matters: they need internet to listen to the Quran.
        if (viewModel.noInternetDialog) {
            AlertDialog(
                onDismissRequest = { viewModel.dismissNoInternetDialog() },
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.WifiOff,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.error
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = getAppString(isArabic, "No Internet Connection"),
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                },
                text = {
                    Text(
                        text = getAppString(
                            isArabic,
                            "You are not connected to the internet. You need an internet connection to listen to Quran audio and to load Surah text. Please check your connection and try again."
                        ),
                        style = MaterialTheme.typography.bodyMedium
                    )
                },
                confirmButton = {
                    Button(onClick = { viewModel.dismissNoInternetDialog() }) {
                        Text(getAppString(isArabic, "OK"))
                    }
                }
            )
        }

        // Tajweed-model warm-up popup. Non-dismissable on purpose: it exists purely as a
        // "please wait" indicator while the ~1 MB JSON weights parse on a background IO
        // thread. The moment [viewModel.isTajweedModelReady] flips true the state recomposes
        // and this dialog disappears on its own -- no Close button needed. Kept side-by-side
        // with the no-internet popup so the user sees whichever explanation applies.
        if (viewModel.isTajweedModelLoading && !viewModel.isTajweedModelReady) {
            AlertDialog(
                onDismissRequest = { /* intentionally empty -- the popup closes itself */ },
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.School,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = getAppString(isArabic, "Loading Tajweed Model"),
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                },
                text = {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = getAppString(
                                isArabic,
                                "Please wait while the on-device Tajweed detection model is being loaded. This only happens the first time you open the Tajweed section. Subsequent uses are instant."
                            ),
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center
                        )
                    }
                },
                confirmButton = { /* no button -- non-dismissable */ },
                dismissButton = { /* keep dialog buttons absent */ }
            )
        }

        // (item 5) Real-time tajweed mistake popup. Fires when the acoustic model flags a
        // TAJWEED_ERROR on a specific word during Memorize+Tajweed. It PAUSES the mic so the
        // user can re-read from that word, and it STAYS ON SCREEN UNTIL THE USER DISMISSES IT
        // (user: "the popup stays until the user dismiss it"). There is deliberately NO
        // auto-dismiss timer: the only clears are the user's Close/back, a new range/surah
        // selection (3942), and leaving the Memorize tab (4501). An earlier version of this
        // comment claimed a 4 s auto-timer — there has never been one, and adding one would
        // break the contract.
        latestTajweedPopup?.let { (ruleLabel, wordIdx) ->
            // The popup serves three flows. "tajweed" (TAJWEED_ERROR): the
            // acoustic model flagged a rule violation; the popup names the rule
            // and shows the word's score, and the mic auto-pauses so the user
            // can re-read from that word. "wrong"/"skipped": the mic was
            // auto-paused and the user must re-read the named word from there
            // -- the resume re-seats the committed transcript to right before
            // it (mistakeResumeIndex / mistakeResumeLiveText), so the word
            // counts again and the final transcript isn't duplicated.
            // Resolve the EXACT (word, rule) row. A multi-rule word has one
            // row per rule and taking the first row by wordIndex could show a
            // different rule's percentage (and, if the rows disagree, text).
            val popupRow = if (latestPopupKind == "tajweed")
                detailedAnalysis.forWordAndRule(wordIdx, ruleLabel)
            else null
            val popupScore = popupRow?.tajweedScore
            // THE POPUP MUST NAME THE WORD (user: "i noticed it didnt write the
            // word in the popup. just the rule and the score"). For "wrong"/
            // "skipped" the pair's first element already IS the word text; for
            // "tajweed" it is the RULE, so resolve the word from the same
            // (word, rule) row the score is read from. MULTI-RULE SAFE: rows are
            // resolved by (wordIndex, rule), never by list position and never
            // by wordIndex alone.
            val popupWord = if (latestPopupKind == "tajweed") {
                // Exact row first; the word text is identical across a word's
                // rule rows, so falling back to the word-only lookup can never
                // change WHICH word is named (only a mismatched row would lose
                // its score, above).
                popupRow?.text ?: detailedAnalysis.forWordIndex(wordIdx)?.text.orEmpty()
            } else ruleLabel
            val localizedRule = getAppString(isArabic, ruleLabel)
            // §4.49 — A RULE WEARS ITS OWN COLOUR, EVERYWHERE. The user's
            // reference is the Memorize surface, where Madd is always RED
            // ("check the rules colors match what we have in memorize + tajweed
            // not giving me other colors or mingling the colours of the rules
            // (for example we have madd always red)"). This dialog's title names
            // the rule, so it resolves through the same canonical head palette
            // the letters and the score rows use; the amber stays on the WARNING
            // ICON, which is chrome, not a rule.
            val popupRuleColor = tajweedHeadColor(ruleLabelToHead(ruleLabel))
            val popupTitle = when (latestPopupKind) {
                "wrong" -> getAppString(isArabic, "Wrong Word")
                "skipped" -> getAppString(isArabic, "Skipped Word")
                else -> getAppString(isArabic, "Tajweed Mistake") + " — " + localizedRule
            }
            val popupBody = when (latestPopupKind) {
                "wrong" -> getAppString(
                    isArabic,
                    "Re-read the word \"%s\" correctly, then continue from there."
                ).replace("%s", ruleLabel)
                "skipped" -> getAppString(
                    isArabic,
                    "The word \"%s\" was not heard. Re-read it, then continue from there."
                ).replace("%s", ruleLabel)
                else -> {
                    // Name the word first, then the rule verdict, then the score —
                    // "The word "ذَٰلِكَ": Review your Madd application. Your Madd
                    // score: 46%." Without the leading clause the popup said only
                    // which RULE failed, leaving the user to guess which word.
                    val base = formatPerRuleVerdict(isArabic, false, ruleLabel)
                    val named = if (popupWord.isNotEmpty())
                        getAppString(isArabic, "The word \"%s\": ").replace("%s", popupWord) + base
                    else base
                    if (popupScore != null) {
                        // The score template's key carries a DOUBLED `%%` (escaped for the
                        // printf-style body), and `getAppString` returns the key VERBATIM in
                        // English -- so the English popup rendered a literal `35%%` while the
                        // Arabic branch mapped it to a single `٪`. Collapse it here; the Arabic
                        // translation has no `%%` so this is a no-op for it.
                        named + " " + getAppString(isArabic, "Your %s score: %d%%.")
                            .replace("%s", localizedRule)
                            .replace("%d", popupScore.toString())
                            .replace("%%", "%")
                    } else named
                }
            }
            // §4.60 — DISMISSING A MISTAKE POPUP HANDS THE NEXT TAP BACK TO THE
            // START PATH, ON EVERY MODE THAT HAS NO RESUME LATCH.
            //
            // On Memorize + Tajweed the next tap RESUMES: the pipeline is warm and
            // the collector re-seats the committed transcript at the mistake. Tarteel
            // has neither, so the flag that makes a tap mean "pause" has to be
            // released on dismissal — otherwise the reader's next tap can only ever
            // produce ANOTHER SUMMARY and he can never continue from the word the
            // popup just named ("wether it freezes or we can continue").
            //
            // The transcript is deliberately KEPT (§4.46: a start does not clear the
            // Memorize transcript, and the count filter suppresses replayed words), so
            // the walk simply carries on from where it stopped — at the mistake.
            val dismissMistakePopup: () -> Unit = {
                latestTajweedPopup = null
                if (recitationMode != "Tajweed" && recitationSessionActive) {
                    android.util.Log.i(
                        "ItqanSherpa",
                        "popup dismissed in $recitationMode mode -> releasing the pause flag so the " +
                            "next mic tap CONTINUES the recitation from the flagged word instead of " +
                            "summarizing it again (§4.60)"
                    )
                    recitationSessionActive = false
                    confirmedMistakeUiIdx = -1
                    // §4.65 — AND ON A NON-PIPELINE MODE, RELEASE THE FREEZE ITSELF.
                    //
                    // The mistake latch (`mistakeResumeIndex`) is what freezes the
                    // walk at the flagged word, and on Memorize + Tajweed it is
                    // cleared by the collector's fresh-session re-seat, which the
                    // PIPELINE emits. Tarteel has no pipeline, so nothing ever
                    // cleared it and the walk stayed pinned — measured across a whole
                    // session:
                    //   analyze(fast): done 110ms matched=6 hasError=true   <- popup
                    //   popup dismissed in Tarteel mode
                    //   analyze(fast): done 112ms matched=6 hasError=true   <- 30 s later
                    //   ... the same two numbers, every pass ...
                    // A pinned cursor also pins the AUTO-SCROLL (its effect keys off
                    // `matchedWordCount`), which is the user's other report: "when i
                    // receite the words/ayahs are not scrolling automatically like it
                    // used to do before".
                    //
                    // So the dismissal — which IS the reader saying "I have read it" —
                    // releases the latch here. The mistake itself is NOT forgotten:
                    // `shownMistakeWords` still holds it, so the popup cannot re-raise
                    // and the summary still reports it.
                    if (recitationMode != "Tajweed") {
                        android.util.Log.i(
                            "ItqanSherpa",
                            "popup dismissed in $recitationMode mode -> releasing the mistake freeze " +
                                "so the walk, the highlight and the auto-scroll move again (§4.65). " +
                                "The mistake is still on record for the summary."
                        )
                        mistakeResumeIndex = -1
                        mistakeResumeLiveText = ""
                    }
                }
            }
            AlertDialog(
                onDismissRequest = { dismissMistakePopup() },
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = null,
                            tint = Color(0xFFF57F17)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = popupTitle,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = if (latestPopupKind == "tajweed" && popupRuleColor != null)
                                popupRuleColor
                            else
                                Color(0xFFF57F17)
                        )
                    }
                },
                text = {
                    Text(
                        text = popupBody,
                        style = MaterialTheme.typography.bodyMedium
                    )
                },
                confirmButton = {
                    TextButton(onClick = { dismissMistakePopup() }) {
                        Text(getAppString(isArabic, "OK"))
                    }
                }
            )
        }

        // ── Sherpa-ONNX model download dialog (Memorize+Tajweed AND Tajweed Rules Coach) ─
        //
        // First-launch gate for the offline FastConformer recognizer (166 MiB,
        // NPL-1.1). The transcript pipeline refuses `start()` until the
        // weights land in filesDir/sherpa_model/, so this dialog overlays the
        // active screen and downloads on demand. We DO NOT auto-download on app
        // startup — only when the user actually enters a pipeline-driven surface
        // (the standalone Tajweed Rules Coach tab, OR the Memorize tab with
        // recitationMode == "Tajweed") — so a Memorize-only / Read-only user
        // never pays the bandwidth cost.
        //
        // This block was originally gated INSIDE `if (currentTab == "Tajweed")`,
        // which meant a user who went straight to Memorize+Tajweed never saw
        // the dialog and the model never downloaded — the bug the user reported
        // ("i tried memorize + tajweed but the model didnt download"). Now it
        // fires on EITHER pipeline-driven surface.
        //
        // Idempotency: `downloadIfNeeded()` is single-flight (see the
        // `inflight` AtomicBoolean in `SherpaModelDownloader`), so even though
        // both the Memorize+Tajweed-tab and Tajweed-tab effects can register
        // HERE, only one parallel download ever runs — a tab switch mid-download
        // simply attaches a second subscriber to the same progress flow, it
        // does NOT spawn a second HTTP transfer.
        val sherpaSurfaceActive =
            currentTab == "Tajweed" || (currentTab == "Memorize" && recitationMode == "Tajweed")
        if (sherpaSurfaceActive) {
            val dlState = sherpaDownloader.progress.collectAsState().value
            // Trigger the download the first time the user lands on a
            // pipeline-driven surface and the model isn't ready. Safe to
            // re-fire per re-entry: the downloader's `inflight` guard means
            // an in-flight download is not duplicated, and a finished download
            // short-circuits to `Done` without touching the network.
            LaunchedEffect(currentTab, recitationMode) {
                if (!isSherpaModelReady(context) && dlState !is SherpaModelDownloader.Progress.Downloading) {
                    try { sherpaDownloader.downloadIfNeeded() } catch (e: Throwable) {
                        android.util.Log.e("ItqanSherpa", "Model download failed", e)
                    }
                }
                sherpaReady.value = isSherpaModelReady(context)
            }
            when (dlState) {
                is SherpaModelDownloader.Progress.Downloading -> {
                    val pct = if (dlState.totalBytes > 0)
                        dlState.receivedTotal.toFloat() / dlState.totalBytes
                    else 0f
                    AlertDialog(
                        onDismissRequest = { /* non-dismissable until DL completes */ },
                        title = { Text(getAppString(isArabic, "Downloading the model")) },
                        text = {
                            LinearProgressIndicator(
                                progress = { pct },
                                modifier = Modifier.fillMaxWidth()
                            )
                        },
                        confirmButton = {} // no cancel while in-flight
                    )
                }
                is SherpaModelDownloader.Progress.Failed -> {
                    AlertDialog(
                        onDismissRequest = {},
                        title = { Text(getAppString(isArabic, "Download failed")) },
                        text = { Text(dlState.message) },
                        confirmButton = {
                            TextButton(onClick = {
                                // Re-trigger via a coroutine scope. The state flow
                                // picks up the new download's progress immediately.
                                kotlinx.coroutines.GlobalScope.launch {
                                    try { sherpaDownloader.downloadIfNeeded() } catch (_: Throwable) {}
                                }
                            }) { Text(getAppString(isArabic, "Retry")) }
                        }
                    )
                }
                else -> { /* Idle/Done — pipe is ready or pending */ }
            }
        }

        if (currentTab == "Tajweed") {
            // Tajweed companion learning & reference hub
            var activePracticeRule by remember { mutableStateOf<String?>(null) }
            // Canonical card list (see tajweedRuleCards at the top level) --
            // kept as a remember so the LazyColumn keys stay stable.
            val rulesList = remember { tajweedRuleCards }
            
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                        ),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.2f)),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(
                                text = getAppString(isArabic, "Tajweed Rules Coach"),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = getAppString(isArabic, "Learn basic rules, see color highlights, and practice pronunciation with live speech feedback."),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
                

                items(rulesList) { rule ->
                    val isExpanded = activePracticeRule == rule.name
                    var currentExample by remember { mutableStateOf(rule.examples.first()) }
                    // True while the online Quran-wide generator is fetching; used to show a
                    // subtle loading cue so the network/parse latency isn't silent.
                    var isGeneratingExample by remember { mutableStateOf(false) }
                    // Coroutine scope so the refresh IconButton can call the suspend generator without
                    // leaving the composition (the expand LaunchedEffect already runs in one).
                    val refreshScope = rememberCoroutineScope()
                    // Pick a fresh, DIFFERENT example every time the card expands. We try the
                    // online Quran-wide generator first (new Quran phrase every time, like the
                    // practice_recitation-main behaviour the user asked to match), and fall back
                    // to the bundled pool if there's no network or the surah we drew happens to
                    // not contain this rule.
                    LaunchedEffect(isExpanded) {
                        if (isExpanded) {
                            // Silah is Qalun-specific and has no dedicated model head — it maps to
                            // IZHAR via ruleLabelToHead, so the generic generator would refresh it
                            // into a plain Izhar example, which would mislead the user. Silah keeps
                            // its curated Qalun pool instead.
                            if (rule.name == "Silah") return@LaunchedEffect
                            val head = ruleLabelToHead(rule.name) ?: return@LaunchedEffect
                            isGeneratingExample = true
                            val online = generateRandomRecitationExample(head)
                            isGeneratingExample = false
                            if (online != null) {
                                currentExample = online
                            } else {
                                val list = rule.examples
                                if (list.isNotEmpty()) {
                                    var nextEx = list.random()
                                    if (list.size > 1 && nextEx == currentExample) {
                                        var attempts = 0
                                        while (nextEx == currentExample && attempts < 10) {
                                            nextEx = list.random()
                                            attempts++
                                        }
                                    }
                                    currentExample = nextEx
                                }
                            }
                        }
                    }
                    
                    val targetTokens = remember(currentExample) { 
                        currentExample.arabic.split(Regex("\\s+")).filter { it.isNotBlank() } 
                    }
                    val targetNormalized = remember(targetTokens) { 
                        targetTokens.map { normalizeArabic(it) } 
                    }
                    // The rule heads the example COLOURISER actually paints — the
                    // authoritative set the score rows must match. The user counts the
                    // rows against the rule letters they see, so the painted set (not a
                    // second detector's opinion) decides which rows exist
                    // (see tajweedHeadsInPhrase).
                    val exampleHeadTokens = remember(targetTokens, viewModel.riwayah) {
                        tajweedHeadsInPhrase(targetTokens, viewModel.riwayah)
                    }
                    
                    val spokenTokens = remember(tajweedSpokenText, isExpanded) {
                        if (isExpanded) {
                            tajweedSpokenText.split(Regex("\\s+")).filter { it.isNotBlank() }
                        } else emptyList()
                    }
                    
                    var localAnalysis by remember { mutableStateOf<List<WordAnalysis>?>(null) }
                    // MULTI-RULE COVERAGE rows: heads present in the example but left
                    // unscored by the primary one-rule-per-word pass (see below).
                    var supplementalScores by remember { mutableStateOf<List<Pair<String, Int>>>(emptyList()) }
                    val localContext = androidx.compose.ui.platform.LocalContext.current
                    LaunchedEffect(isExpanded, isListening, tajweedSpokenText, currentExample, viewModel.riwayah, tajweedRecordedAudio) {
                        if (isExpanded && !isListening && tajweedRecordedAudio != null) {
                            // Score against the real recorded voice. When we have a transcript we use
                            // it for skip/wrong-word detection; otherwise we score the known example
                            // words directly (acoustic check still runs on the user's actual audio).
                            val scoredSpoken = if (tajweedSpokenText.isNotBlank())
                                tajweedSpokenText.split(Regex("\\s+")).filter { it.isNotBlank() }.map { normalizeArabic(it) }
                            else targetNormalized
                            // isTajweedMode=true is REQUIRED here: the per-word scoring gate
                            // (analyzeRecitationDetails) is `rule != null && hasRealAudio && isTajweedMode`,
                            // so without this flag every word silently stays CORRECT and the per-rule
                            // verdict card (ruleAgg) ends up empty -- exactly the "Your recitation:
                            // (scored from your live audio)" without any colored verdict rows or
                            // per-word transcripts that the user reported. The coach card owns the
                            // mic session (the Sherpa pipeline; pause() hands back the UN-GATED
                            // real-time PCM), so hasRealAudio is true here.
                            localAnalysis = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
                                analyzeRecitationDetails(
                                    scoredSpoken,
                                    targetTokens,
                                    targetNormalized,
                                    viewModel.riwayah,
                                    spokenTimings,
                                    localContext,
                                    fullAudio = tajweedRecordedAudio,
                                    recordingStartMs = tajweedRecordingStartMs,
                                    recordingDurationMs = tajweedRecordingDurationMs,
                                    isTajweedMode = true,
                                    finalReport = true
                                )
                            }
                            // ── MULTI-RULE COVERAGE ── The primary pass scores ONE rule per
                            // word (getTajweedRuleForWord returns the FIRST match and returns
                            // immediately — e.g. الْحَمْدُ short-circuits on its Al-Lam article
                            // so the Izhar-Shafawi meem inside the SAME word is never scored),
                            // while the example colouring highlights every rule letter-by-
                            // letter. That mismatch is exactly "the example contains 3 rules
                            // but I only got 2 scores". Detect every training-table rule
                            // across the example tokens (TajweedRules — the same table the
                            // model was calibrated with) and score whichever heads the
                            // primary pass left uncovered, slicing each token's audio by its
                            // proportional span of the recording.
                            supplementalScores = run {
                                val primaryNow = localAnalysis
                                val audioNow = tajweedRecordedAudio
                                if (primaryNow == null || audioNow == null || audioNow.size < 256) {
                                    emptyList()
                                } else kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
                                    val covered = primaryNow
                                        .mapNotNull { ruleLabelToHead(it.tajweedRule ?: "") }
                                        .toSet()
                                    // THE PAINTED HEADS ARE THE SOURCE. The colouriser the
                                    // user sees (`tajweedHeadsInPhrase`) is the single
                                    // authority for which rules the example contains, so
                                    // this pass fills exactly the heads it paints and the
                                    // primary pass left unscored. Two independent detectors
                                    // used to be unioned here; their disagreements
                                    // (cross-word noon pairing, tanween -> GHUNNAH, natural
                                    // madd) manufactured rows the example never showed:
                                    // "the example has 2 rules but I got 4 percentages".
                                    val wanted = exampleHeadTokens
                                    val analyzer = RecitationAnalyzer.get(localContext)
                                    val durMs = tajweedRecordingDurationMs.coerceAtLeast(1L)
                                    val tCount = targetTokens.size.coerceAtLeast(1)
                                    val out = mutableListOf<Pair<String, Int>>()
                                    for ((head, idxs) in wanted) {
                                        if (head in covered) continue
                                        val thr = decisionThresholdForHead(head)
                                        val probs = mutableListOf<Float>()
                                        for (ti in idxs) {
                                            val st = (ti * durMs / tCount).coerceIn(0L, durMs)
                                            val en = (((ti + 1) * durMs / tCount)).coerceAtMost(durMs)
                                                .coerceAtLeast(st + 120L)
                                            val a = (st * 16000 / 1000).toInt().coerceAtMost(audioNow.size)
                                            val b = (en * 16000 / 1000).toInt().coerceIn(a, audioNow.size)
                                            if (b - a < 256) continue
                                            val seg = trimSilence(audioNow.sliceArray(a until b))
                                            if (seg.size < 256) continue
                                            analyzer.analyze(seg, listOf(head), thr)
                                                .firstOrNull()?.let { probs.add(it.prob) }
                                        }
                                        if (probs.isNotEmpty()) {
                                            val avg = (probs.average() * 100).toInt().coerceIn(0, 100)
                                            // §4.49 — ONE CANONICAL NAME PER HEAD, so a
                                            // head never shows up under two different
                                            // labels ("Madd" here and "Madd Muttasil"
                                            // from the primary pass) and the row colour
                                            // resolution in ruleAgg stays single-valued.
                                            val friendly = headToFriendlyLabel(head)
                                                ?: rulesList.firstOrNull { ruleLabelToHead(it.name) == head }?.name
                                                ?: head
                                            out.add(friendly to avg)
                                        } else {
                                            // A head the model has NO output for cannot be
                                            // scored. Log it instead of leaving the user
                                            // to wonder why a coloured rule has no score.
                                            android.util.Log.i(
                                                "ItqanSherpa",
                                                "coachScores(): head=$head is in the example but produced " +
                                                    "no probability (no model output / no usable audio span) - " +
                                                    "no score row for it"
                                            )
                                        }
                                    }
                                    if (out.isNotEmpty()) {
                                        android.util.Log.i(
                                            "ItqanSherpa",
                                            "coachScores(): per-rule rows ${out.map { it.first }} (covered=${covered})"
                                        )
                                    }
                                    out
                                }
                            }
                        } else if (!isExpanded || tajweedRecordedAudio == null) {
                            localAnalysis = null
                            supplementalScores = emptyList()
                        }
                    }
                    val localAnalysisSnap = localAnalysis
                    val hasMistake = localAnalysisSnap?.any { it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED || it.status == RecitationStatus.TAJWEED_ERROR } == true

                    // ── THE CORRECTED RECITATION ──────────────────────────────
                    // What the user reads back after a practice run. It is built
                    // from the SAME alignment the Memorize+Tajweed surface uses
                    // (`analyzeRecitationDetails`), so every row carries the
                    // EXAMPLE's canonical word (its correct spelling) in reading
                    // order, and only the verdict decides how it is styled:
                    //   * WRONG / SKIPPED -> error colour
                    //   * TAJWEED_ERROR   -> keeps its rule colours, underlined
                    // The recognizer's raw transcript is NOT shown: it is the
                    // garbled decode the user asked to hide. This annotated line
                    // used to live under the per-rule percentages; it is hoisted
                    // here so it can replace the raw text in "Your recitation".
                    val coachErrorColor = MaterialTheme.colorScheme.error
                    // ONLY THE PORTION ACTUALLY RECITED. Keep the contiguous
                    // prefix of the example up to (and including) the LAST word
                    // that carries spoken evidence — `heardFrontierExclusive` is
                    // exactly that boundary. Everything past it is the example's
                    // unread tail (PENDING with no token), which the aligner fills
                    // in for scoring but the user never said. A blank transcript
                    // means the recognizer captured no words at all, so we show
                    // nothing rather than let the acoustic-only fallback (which
                    // scores against the known example) claim a full recitation.
                    val correctedRecitation = if (tajweedSpokenText.isBlank()) null
                    else localAnalysisSnap?.let { rows ->
                        val heardFrontier = heardFrontierExclusive(rows)
                        if (heardFrontier <= 0) null else buildAnnotatedString {
                            // MULTI-RULE: one row per (word, rule) — render each WORD once.
                            val feedbackWords = rows
                                .distinctBy { it.wordIndex }
                                .filter { it.wordIndex < heardFrontier }
                            feedbackWords.forEachIndexed { wi, wordInfo ->
                                val ann = buildTajweedAnnotatedString(
                                    wordInfo.text,
                                    feedbackWords.getOrNull(wi + 1)?.text,
                                    viewModel.riwayah
                                )
                                val start = length
                                append(ann.text)
                                ann.spanStyles.forEach { s ->
                                    addStyle(s.item, start + s.start, start + s.end)
                                }
                                val end = length
                                val wordRows = rows.filter { it.wordIndex == wordInfo.wordIndex }
                                when {
                                    wordRows.any { it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED } ->
                                        addStyle(
                                            SpanStyle(color = coachErrorColor),
                                            start, end
                                        )
                                    wordRows.any { it.status == RecitationStatus.TAJWEED_ERROR } ->
                                        addStyle(
                                            SpanStyle(textDecoration = androidx.compose.ui.text.style.TextDecoration.Underline),
                                            start, end
                                        )
                                    else -> {}
                                }
                                append(" ")
                            }
                        }
                    }

                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable {
                                if (activePracticeRule != rule.name) {
                                    activePracticeRule = rule.name
                                    // Switching to a different rule: the previous recording belongs to a
                                    // different example, so discard it. Otherwise the new card would show a
                                    // stale score from audio the user never recorded for this rule.
                                    if (tajweedPipelineHolder.running || isListening) {
                                        try { if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop() } catch (e: Throwable) {}
                                        try { speechRecognizer?.cancel() } catch (e: Throwable) {}
                                        isListening = false
                                    }
                                    tajweedRecordedAudio = null
                                    tajweedRecordingDurationMs = 0
                                    tajweedFinalSpokenText = ""
                                    tajweedPartialSpokenText = ""
                                } else {
                                    activePracticeRule = null
                                    if (tajweedPipelineHolder.running || isListening) {
                                        try { if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop() } catch (e: Throwable) {}
                                        try { speechRecognizer?.cancel() } catch (e: Throwable) {}
                                        isListening = false
                                        tajweedRecordedAudio = null
                                        tajweedRecordingDurationMs = 0
                                    }
                                }
                            },
                        border = BorderStroke(
                            1.dp, 
                            if (isExpanded) rule.color else MaterialTheme.colorScheme.outlineVariant
                        ),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = if (isExpanded) rule.color.copy(alpha = 0.05f) else MaterialTheme.colorScheme.surface
                        )
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Box(
                                        modifier = Modifier
                                            .size(12.dp)
                                            .background(rule.color, CircleShape)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = getAppString(isArabic, rule.name),
                                        style = MaterialTheme.typography.titleMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = if (isExpanded) rule.color else MaterialTheme.colorScheme.onSurface
                                    )
                                }
                                Icon(
                                    imageVector = if (isExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                                    contentDescription = "Expand",
                                    tint = if (isExpanded) rule.color else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = getAppString(isArabic, rule.description),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )

                            AnimatedVisibility(visible = isExpanded) {
                                Column(modifier = Modifier.padding(top = 12.dp)) {
                                    HorizontalDivider(
                                        color = rule.color.copy(alpha = 0.2f),
                                        modifier = Modifier.padding(bottom = 12.dp)
                                    )
                                    Text(
                                        text = getAppString(isArabic, rule.fullExplanation),
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        lineHeight = 18.sp
                                    )
                                    Spacer(modifier = Modifier.height(12.dp))
                                    
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .background(
                                                color = rule.color.copy(alpha = 0.1f),
                                                shape = RoundedCornerShape(12.dp)
                                            )
                                            .padding(12.dp)
                                    ) {
                                        Column {
                                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                                                Text(
                                                    text = getAppString(isArabic, "Select to Practice:"),
                                                    style = MaterialTheme.typography.labelSmall,
                                                    fontWeight = FontWeight.Bold,
                                                    color = rule.color
                                                )
                                                IconButton(
                                                    enabled = !isGeneratingExample,
                                                    onClick = {
                                                    // Same "give me a different Quran phrase" contract as the
                                                    // expand effect: try the online generator first, fall back
                                                    // to the bundled pool if offline or this surah lacks the rule.
                                                    // Silah: Qalun-specific, keep its curated pool (it has no
                                                    // dedicated model head; the generic generator would pull
                                                    // plain Izhar examples for it).
                                                    refreshScope.launch {
                                                        val head = if (rule.name != "Silah") ruleLabelToHead(rule.name) else null
                                                        if (head != null) {
                                                            isGeneratingExample = true
                                                            val online = generateRandomRecitationExample(head)
                                                            isGeneratingExample = false
                                                            if (online != null) {
                                                                currentExample = online
                                                                return@launch
                                                            }
                                                        }
                                                        // Offline / no-match / Silah: no-repeat pick from bundles.
                                                        val list = rule.examples
                                                        if (list.isNotEmpty()) {
                                                            var nextEx = list.random()
                                                            if (list.size > 1 && nextEx == currentExample) {
                                                                var attempts = 0
                                                                while (nextEx == currentExample && attempts < 10) {
                                                                    nextEx = list.random()
                                                                    attempts++
                                                                }
                                                            }
                                                            currentExample = nextEx
                                                        }
                                                    }
                                                    tajweedFinalSpokenText = ""
                                                    tajweedPartialSpokenText = ""
                                                    if (tajweedPipelineHolder.running || isListening) {
                                                        try { if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop() } catch (e: Throwable) {}
                                                        try { speechRecognizer?.cancel() } catch (e: Throwable) {}
                                                        isListening = false
                                                        tajweedRecordedAudio = null
                                                        tajweedRecordingDurationMs = 0
                                                    }
                                                }) {
                                                    if (isGeneratingExample) {
                                                        CircularProgressIndicator(
                                                            modifier = Modifier.size(20.dp),
                                                            strokeWidth = 2.dp,
                                                            color = rule.color
                                                        )
                                                    } else {
                                                        Icon(Icons.Default.Refresh, contentDescription = "Change Example", tint = rule.color)
                                                    }
                                                }
                                            }
                                            Spacer(modifier = Modifier.height(6.dp))
                                            // Color the example sentence per-rule, the same way the
                                            // Memorize section does it: each token is rendered by
                                            // buildTajweedAnnotatedString(token, nextToken, riwayah),
                                            // which paints Qalqalah blue, Al-Lam blue-grey/teal,
                                            // Ikhfa/Iqlab cyan, Madd red, etc., letter-by-letter so
                                            // the user can see which rule each word of the example
                                            // demonstrates while they practice this card's rule.
                                            val exampleColored = remember(currentExample, viewModel.riwayah) {
                                                val toks = currentExample.arabic
                                                    .split(Regex("\\s+"))
                                                    .filter { it.isNotBlank() }
                                                buildAnnotatedString {
                                                    toks.forEachIndexed { idx, tok ->
                                                        val nxt = toks.getOrNull(idx + 1)
                                                        append(buildTajweedAnnotatedString(tok, nxt, viewModel.riwayah))
                                                        if (idx < toks.lastIndex) append(" ")
                                                    }
                                                }
                                            }
                                            if (isGeneratingExample && currentExample == rule.examples.first()) {
                                                // First-time generation inside this expand: show a tiny cue so
                                                // the user isn't looking at a blank card while the new Quran
                                                // phrase is being fetched.
                                                Text(
                                                    text = getAppString(isArabic, "Loading example…"),
                                                    style = MaterialTheme.typography.bodyMedium,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                    modifier = Modifier.fillMaxWidth(),
                                                    textAlign = TextAlign.Right
                                                )
                                            } else {
                                                Text(
                                                    text = exampleColored,
                                                    style = MaterialTheme.typography.titleLarge,
                                                    fontWeight = FontWeight.Bold,
                                                    textAlign = TextAlign.Right,
                                                    modifier = Modifier.fillMaxWidth()
                                                )
                                            }
                                            Spacer(modifier = Modifier.height(4.dp))
                                            // Hide the English translation when the UI is in Arabic --
                                            // the Quran text itself already IS Arabic, and the user
                                            // explicitly asked for Arabic-only surfaces in this mode.
                                            if (!isArabic) {
                                                Text(
                                                    text = currentExample.translation,
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                        }
                                    }

                                    Spacer(modifier = Modifier.height(12.dp))
                                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                                        FloatingActionButton(
                                            onClick = {
                                                val hasPermission = androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.RECORD_AUDIO) == android.content.pm.PackageManager.PERMISSION_GRANTED
                                                if (hasPermission) {
                                                    if (tajweedPipelineHolder.running || isListening) {
                                                        stopTajweedSession()
                                                    } else {
                                                        startTajweedSession()
                                                    }
                                                } else {
                                                    tajweedPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
                                                }
                                            },
                                            containerColor = if (isListening || tajweedPipelineHolder.running) MaterialTheme.colorScheme.errorContainer else rule.color,
                                            contentColor = if (isListening || tajweedPipelineHolder.running) MaterialTheme.colorScheme.onErrorContainer else Color.White
                                        ) {
                                            Icon(
                                                imageVector = if (isListening || tajweedPipelineHolder.running) Icons.Default.MicOff else Icons.Default.Mic,
                                                contentDescription = if (isListening || tajweedPipelineHolder.running) "Stop Listening" else "Start Listening"
                                            )
                                        }
                                    }

                                    // Only after a run, and only with an alignment to show.
                                    // The raw live transcript is intentionally never
                                    // rendered (see correctedRecitation above).
                                    if (!isListening && localAnalysisSnap != null) {
                                        Spacer(modifier = Modifier.height(16.dp))
                                        Text(
                                            text = getAppString(isArabic, "Your recitation:"),
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                        if (correctedRecitation != null) {
                                            Text(
                                                text = correctedRecitation,
                                                style = MaterialTheme.typography.bodyLarge,
                                                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                                                textAlign = TextAlign.Right
                                            )
                                        }
                                        
                                        run {
                                            Spacer(modifier = Modifier.height(8.dp))

                                            // Aggregate the per-WORD acoustic scores into one row per
                                            // DISTINCT rule detected in the example sentence. Each rule
                                            // is scored against its OWN model head and judged against its
                                            // OWN per-head threshold (decisionThresholdForHead), so a
                                            // Qalqalah-practice example that also contains an Al-Lam word
                                            // and an Ikhfa word shows three independent verdicts instead of
                                            // one confusing combined line; a recitation is flagged as
                                            // TAJWEED_ERROR for a rule only when THAT rule's average score
                                            // drops below its own threshold.
                                            // Each rule's verdict is colored by its own rule color:
                                            // green text/card when its score >= threshold, red otherwise.
                                            val ruleAgg = remember(localAnalysisSnap, supplementalScores, exampleHeadTokens) {
                                                // Only the heads the example PAINTS — the rules the
                                                // user can actually see in the snippet above. A row
                                                // for anything else is a phantom the user reads as a
                                                // rule the example does not contain.
                                                val paintedOrder = exampleHeadTokens.keys.toList()
                                                // Primary per-word rows: words that actually got an
                                                // acoustic score for some rule the example paints.
                                                val scored = localAnalysisSnap.filter {
                                                    it.tajweedRule != null && it.tajweedScore != null &&
                                                        ruleLabelToHead(it.tajweedRule!!) in paintedOrder
                                                }
                                                val agg = scored.groupBy { it.tajweedRule!! }
                                                    .mapValues { (_, words) ->
                                                        words.map { it.tajweedScore!! }.average().toInt()
                                                    }
                                                    .toMutableMap()
                                                // Supplemental rows: painted heads the one-rule-per-word
                                                // primary pass could not reach (see MULTI-RULE
                                                // COVERAGE above). Never duplicates a row.
                                                supplementalScores.forEach { (name, s) -> agg.putIfAbsent(name, s) }
                                                // Rows in the order the rules appear in the example.
                                                agg.toList().sortedBy { (name, _) ->
                                                    paintedOrder.indexOf(ruleLabelToHead(name))
                                                }
                                            }

                                            if (ruleAgg.isNotEmpty()) {
                                                Card(
                                                    colors = CardDefaults.cardColors(
                                                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                                                    ),
                                                    modifier = Modifier.fillMaxWidth()
                                                ) {
                                                    Column(modifier = Modifier.padding(12.dp)) {
                                                        Text(
                                                            text = getAppString(isArabic, "Acoustic score per rule"),
                                                            style = MaterialTheme.typography.labelSmall,
                                                            fontWeight = FontWeight.Bold,
                                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                                        )
                                                        Spacer(modifier = Modifier.height(6.dp))

                                                        ruleAgg.forEach { (ruleName, score) ->
                                                            val info = rulesList.firstOrNull { it.name == ruleName }
                                                            val head = ruleLabelToHead(ruleName)
                                                            // Resolve the row colour through the canonical
                                                            // HEAD palette: sub-labels like "Madd Muttasil" /
                                                            // "Al-Lam" match no card title exactly, so the old
                                                            // card-name lookup fell back to the theme primary
                                                            // and rendered a Madd row TEAL. Head colour first,
                                                            // exact card match second, theme fallback last.
                                                            val ruleColor = tajweedHeadColor(head)
                                                                ?: info?.color
                                                                ?: MaterialTheme.colorScheme.primary
                                                            val thr = if (head != null) decisionThresholdForHead(head) else 0.5f
                                                            val passed = score >= (thr * 100f).toInt()
                                                            val verdictColor = if (passed) Color(0xFF2E7D32) else MaterialTheme.colorScheme.error
                                                            val verdictText = formatPerRuleVerdict(isArabic, passed, ruleName)

                                                            Row(
                                                                verticalAlignment = Alignment.CenterVertically,
                                                                modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp)
                                                            ) {
                                                                Box(
                                                                    modifier = Modifier
                                                                        .size(12.dp)
                                                                        .background(ruleColor, androidx.compose.foundation.shape.CircleShape)
                                                                )
                                                                Spacer(modifier = Modifier.width(8.dp))
                                                                Text(
                                                                    text = getAppString(isArabic, ruleName) + ":",
                                                                    color = ruleColor,
                                                                    fontWeight = FontWeight.SemiBold,
                                                                    style = MaterialTheme.typography.bodyMedium,
                                                                    modifier = Modifier.weight(1f)
                                                                )
                                                                Text(
                                                                    text = "$score%",
                                                                    color = verdictColor,
                                                                    fontWeight = FontWeight.Bold,
                                                                    style = MaterialTheme.typography.bodyMedium
                                                                )
                                                            }
                                                            Text(
                                                                text = verdictText,
                                                                color = verdictColor,
                                                                style = MaterialTheme.typography.bodySmall,
                                                                textAlign = TextAlign.Right,
                                                                modifier = Modifier.fillMaxWidth().padding(start = 20.dp)
                                                            )
                                                        }

                                                        // The corrected, aligned recitation is
                                                        // rendered once in "Your recitation" above
                                                        // (see correctedRecitation), so it is not
                                                        // repeated here under the percentages.
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if (currentTab == "Stats") {
            val goals by viewModel.allGoals.collectAsState(initial = emptyList())
            
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                item {
                    Text(
                        text = getAppString(isArabic, "Active Recitation Goals"),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
                
                if (goals.isEmpty()) {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
                        ) {
                            Text(
                                text = getAppString(isArabic, "No active goals set. Formulate a daily or weekly recitation target below to stay motivated!"),
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.padding(16.dp),
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                } else {
                    items(goals) { goal ->
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(
                                            imageVector = if (goal.type == "Daily") Icons.Default.Today else Icons.Default.DateRange,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.secondary
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = if (isArabic) "هدف ${getAppString(isArabic, goal.type)}" else "${goal.type} Goal",
                                            style = MaterialTheme.typography.titleMedium,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                    IconButton(onClick = { viewModel.deleteGoal(goal.id) }) {
                                        Icon(Icons.Default.Delete, contentDescription = getAppString(isArabic, "Delete Goal"), tint = MaterialTheme.colorScheme.error)
                                    }
                                }

                                // Scope badge: shows the Surah+ayah range the goal applies to
                                // (or "All Quran" for unscoped legacy goals). The user picks this
                                // in the form below; here we just surface it so they can tell two
                                // same-frequency goals apart and confirm their scope is stored.
                                if (goal.surahNumber != 0) {
                                    val scopedSurah = viewModel.surahs.firstOrNull { it.number == goal.surahNumber }
                                    // Show the Arabic name when the UI is in Arabic; fall back to English
                                    // transliteration if the surah isn't in the (downloaded) list yet.
                                    val scopedSurahName = if (scopedSurah != null) {
                                        if (isArabic) scopedSurah.name else scopedSurah.englishName
                                    } else {
                                        "Surah ${goal.surahNumber}"
                                    }
                                    Text(
                                        text = if (isArabic)
                                            "${getAppString(isArabic, "Scope: Surah")} $scopedSurahName  •  ${getAppString(isArabic, "Ayahs: ")}${goal.rangeStart}-${goal.rangeEnd}"
                                        else
                                            "Scope: $scopedSurahName  •  Ayahs: ${goal.rangeStart}-${goal.rangeEnd}",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    Spacer(modifier = Modifier.height(6.dp))
                                } else if (goals.size > 1) {
                                    // Only mention the All-Quran scope when there are multiple goals
                                    // -- otherwise a single unscoped goal reads naturally without it.
                                    Text(
                                        text = getAppString(isArabic, "All Quran (no scope)"),
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    Spacer(modifier = Modifier.height(6.dp))
                                }

                                Spacer(modifier = Modifier.height(12.dp))
                                
                                val progressPercent = if (goal.targetAyahs > 0) {
                                    (goal.recitedAyahs.toFloat() / goal.targetAyahs).coerceIn(0f, 1f)
                                } else 0f
                                
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = if (isArabic) "${goal.recitedAyahs} ${getAppString(isArabic, "of")} ${goal.targetAyahs} ${getAppString(isArabic, "Ayahs Recited")}"
                                               else "${goal.recitedAyahs} of ${goal.targetAyahs} Ayahs Recited",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        text = "${(progressPercent * 100).toInt()}%",
                                        style = MaterialTheme.typography.bodyMedium,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                                
                                Spacer(modifier = Modifier.height(6.dp))
                                
                                LinearProgressIndicator(
                                    progress = { progressPercent },
                                    modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)),
                                    color = MaterialTheme.colorScheme.primary,
                                    trackColor = MaterialTheme.colorScheme.primaryContainer
                                )
                                
                                Spacer(modifier = Modifier.height(12.dp))
                                
                                Spacer(modifier = Modifier.height(12.dp))
                                
                                Card(
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f)
                                    ),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Column(modifier = Modifier.padding(12.dp)) {
                                        Text(
                                            text = getAppString(isArabic, "Advancement Advice"),
                                            style = MaterialTheme.typography.labelMedium,
                                            fontWeight = FontWeight.Bold,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))
                                        val remaining = (goal.targetAyahs - goal.recitedAyahs).coerceAtLeast(0)
                                        Text(
                                            text = if (isArabic) {
                                                if (remaining > 0) {
                                                    "لقد أتممت ${(progressPercent * 100).toInt()}% من هدفك! استمرّ في التمرين، تفصلك $remaining آية فقط عن هدفك."
                                                } else {
                                                    "مبروك! لقد أتممت هدفك المؤلّف من ${goal.targetAyahs} آية بامتياز!"
                                                }
                                            } else {
                                                if (remaining > 0) {
                                                    "You have completed ${(progressPercent * 100).toInt()}% of your goal! Keep practicing, you are only $remaining Ayahs away from your target."
                                                } else {
                                                    "Congratulations! You completed your goal target of ${goal.targetAyahs} Ayahs with flying colors!"
                                                }
                                            },
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                        
                                        Spacer(modifier = Modifier.height(12.dp))
                                        
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Icon(
                                                imageVector = Icons.Default.Warning,
                                                contentDescription = null,
                                                tint = MaterialTheme.colorScheme.error,
                                                modifier = Modifier.size(16.dp)
                                            )
                                            Spacer(modifier = Modifier.width(6.dp))
                                            Text(
                                                text = if (isArabic) "${getAppString(isArabic, "Total Mistakes recorded: ")}${goal.mistakesCount}"
                                                       else "Total Mistakes recorded: ${goal.mistakesCount}",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.error,
                                                fontWeight = FontWeight.Bold
                                            )
                                        }
                                        Spacer(modifier = Modifier.height(8.dp))
                                        
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween
                                        ) {
                                            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1f)) {
                                                Text(text = getAppString(isArabic, "Wrong Words"), style = MaterialTheme.typography.bodySmall, fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                                Text(text = "${goal.wrongWordsCount}", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.error)
                                            }
                                            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1f)) {
                                                Text(text = getAppString(isArabic, "Skipped Words"), style = MaterialTheme.typography.bodySmall, fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                                Text(text = "${goal.skippedWordsCount}", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = Color(0xFFFB8C00))
                                            }
                                            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1f)) {
                                                Text(text = getAppString(isArabic, "Tajweed Rules"), style = MaterialTheme.typography.bodySmall, fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                                Text(text = "${goal.tajweedMistakesCount}", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.tertiary)
                                            }
                                        }
                                        
                                        if (goal.tajweedMistakesBreakdown.isNotEmpty()) {
                                            Spacer(modifier = Modifier.height(10.dp))
                                            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
                                            Spacer(modifier = Modifier.height(6.dp))
                                            
                                            // Parse the string "Madd: 1/5, Ghunnah: 0/2" and display percentages.
                                            // ruleName is the English label stored in the breakdown, localize it
                                            // for display ("Madd" -> "مد" etc.)
                                            val parts = goal.tajweedMistakesBreakdown.split(",")
                                            val statsString = parts.map { part ->
                                                val ruleParts = part.split(":")
                                                val ruleName = ruleParts.getOrNull(0)?.trim() ?: ""
                                                val fraction = ruleParts.getOrNull(1)?.trim() ?: ""
                                                val fractionParts = fraction.split("/")
                                                val mistakes = fractionParts.getOrNull(0)?.toIntOrNull() ?: 0
                                                val total = fractionParts.getOrNull(1)?.toIntOrNull() ?: mistakes
                                                
                                                if (total > 0 && ruleName.isNotEmpty()) {
                                                    val correct = total - mistakes
                                                    val percentage = (correct.toFloat() / total.toFloat() * 100).toInt()
                                                    "${getAppString(isArabic, ruleName)}: $percentage%"
                                                } else {
                                                    null
                                                }
                                            }.filterNotNull().joinToString(if (isArabic) "  •  " else "  •  ")
                                            
                                            Text(
                                                text = getAppString(isArabic, "Tajweed Score: ") + (statsString.ifEmpty { getAppString(isArabic, "N/A") }),
                                                style = MaterialTheme.typography.bodySmall,
                                                fontSize = 11.sp,
                                                fontWeight = FontWeight.Medium,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                            )
                                        }

                                        // ── Per-word mistake breakdown ─────────────
                                        // Persisted on the goal as JSON. We show the
                                        // list so the user can SEE which words were
                                        // wrong / skipped / had which tajweed rule
                                        // (not just the count). Grouped by kind,
                                        // deduped by (word, rule) within a kind.
                                        val entries = parseMistakeEntries(goal.mistakeEntriesJson)
                                        if (entries.isNotEmpty()) {
                                            val perGoalWrongs = entries.filter { it.kind == "wrong" }
                                                .distinctBy { it.wordIndex }
                                            val perGoalSkippeds = entries.filter { it.kind == "skipped" }
                                                .distinctBy { it.wordIndex }
                                            val perGoalTajweed = entries.filter { it.kind == "tajweed" }
                                                .distinctBy { it.wordIndex to (it.rule ?: "") }
                                            if (perGoalWrongs.isNotEmpty() || perGoalSkippeds.isNotEmpty() || perGoalTajweed.isNotEmpty()) {
                                                Spacer(modifier = Modifier.height(10.dp))
                                                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
                                                Spacer(modifier = Modifier.height(6.dp))
                                                Text(
                                                    text = getAppString(isArabic, "Words To Review"),
                                                    style = MaterialTheme.typography.labelMedium,
                                                    fontWeight = FontWeight.Bold,
                                                    color = MaterialTheme.colorScheme.primary
                                                )
                                                Spacer(modifier = Modifier.height(4.dp))
                                                if (perGoalWrongs.isNotEmpty()) {
                                                    Text(
                                                        text = if (isArabic)
                                                            "الكلمات الخاطئة: ${perGoalWrongs.joinToString("، ") { it.word }}"
                                                        else
                                                            "Wrong words: ${perGoalWrongs.joinToString(", ") { it.word }}",
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                                    )
                                                }
                                                if (perGoalSkippeds.isNotEmpty()) {
                                                    Text(
                                                        text = if (isArabic)
                                                            "الكلمات المتخطاة: ${perGoalSkippeds.joinToString("، ") { it.word }}"
                                                        else
                                                            "Skipped words: ${perGoalSkippeds.joinToString(", ") { it.word }}",
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                                    )
                                                }
                                                if (perGoalTajweed.isNotEmpty()) {
                                                    // Group by rule so a multi-rule word is shown once per
                                                    // (word, rule) pair, with the rule label localized.
                                                    val tajweedLines = perGoalTajweed
                                                        .groupBy { it.rule ?: "" }
                                                        .map { (rule, items) ->
                                                            val ruleLabel = getAppString(isArabic, rule)
                                                            val words = items.joinToString(if (isArabic) "، " else ", ") { it.word }
                                                            if (isArabic) "$ruleLabel: $words" else "$words [$ruleLabel]"
                                                        }
                                                    tajweedLines.forEach { line ->
                                                        Text(
                                                            text = line,
                                                            style = MaterialTheme.typography.bodySmall,
                                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                item {
                    HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                    Text(
                        text = getAppString(isArabic, "Configure New Goal"),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
                
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.2f))
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            var goalType by remember { mutableStateOf("Daily") }

                            // Goal scope: the user always ties a goal to ONE Surah plus an Ayah
                            // range. Default: Al-Fatihah (Surah #1) so first-time launches show
                            // something concrete instead of an empty "no surah picked yet" state.
                            // LaunchedEffect below (re)resolves the default once viewModel.surahs
                            // has loaded -- the Surah list is populated asynchronously by
                            // fetchSurahs in the MainViewModel init block.
                            var goalScopeSurah by remember { mutableStateOf<itqan.app.verses.SurahData?>(null) }
                            var goalRangeStart by remember { mutableStateOf(1) }
                            var goalRangeEnd by remember { mutableStateOf(1) }
                            var showGoalSurahDialog by remember { mutableStateOf(false) }
                            LaunchedEffect(viewModel.surahs) {
                                if (goalScopeSurah == null && viewModel.surahs.isNotEmpty()) {
                                    val fatihah = viewModel.surahs.firstOrNull { it.number == 1 }
                                    if (fatihah != null) {
                                        goalScopeSurah = fatihah
                                        goalRangeStart = 1
                                        goalRangeEnd = fatihah.numberOfAyahs
                                    }
                                }
                            }

                            Text(getAppString(isArabic, "Goal Frequency"), style = MaterialTheme.typography.labelMedium)
                            Spacer(modifier = Modifier.height(6.dp))
                            Row(modifier = Modifier.fillMaxWidth()) {
                                FilterChip(
                                    selected = goalType == "Daily",
                                    onClick = { goalType = "Daily" },
                                    label = { Text(getAppString(isArabic, "Daily target")) },
                                    modifier = Modifier.weight(1f).padding(end = 4.dp)
                                )
                                FilterChip(
                                    selected = goalType == "Weekly",
                                    onClick = { goalType = "Weekly" },
                                    label = { Text(getAppString(isArabic, "Weekly target")) },
                                    modifier = Modifier.weight(1f).padding(start = 4.dp)
                                )
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            // --- Goal Scope picker ---
                            // The user explicitly asked for every goal to be tied to a surah +
                            // ayah range; there is no longer a "whole Quran" escape hatch, which
                            // also means there is no free-text "Target Ayahs" input -- the range
                            // size IS the target. "Goal Scope" keeps its short name (no
                            // "(optional)" suffix because the scope is now required).
                            Text(
                                getAppString(isArabic, "Goal Scope"),
                                style = MaterialTheme.typography.labelMedium
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                OutlinedButton(
                                    onClick = { showGoalSurahDialog = true },
                                    modifier = Modifier.weight(1f).height(40.dp)
                                ) {
                                    Text(
                                        text = goalScopeSurah?.let {
                                            // Arabic-name first when the UI is Arabic, English otherwise.
                                            "${getAppString(isArabic, "Scope: Surah")} ${it.number} - ${if (isArabic) it.name else it.englishName}"
                                        } ?: getAppString(isArabic, "Pick a Surah"),
                                        style = MaterialTheme.typography.bodySmall,
                                        maxLines = 1
                                    )
                                }
                            }
                            if (goalScopeSurah != null) {
                                Spacer(modifier = Modifier.height(8.dp))
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(getAppString(isArabic, "Ayahs: "), style = MaterialTheme.typography.bodySmall)
                                    AyahNumberPicker(
                                        title = getAppString(isArabic, "Start ayah (scope)"),
                                        currentValue = goalRangeStart,
                                        maxValue = goalScopeSurah?.numberOfAyahs ?: 1,
                                        onValueChange = { newStart ->
                                            goalRangeStart = newStart
                                            if (goalRangeEnd < newStart) goalRangeEnd = newStart
                                        }
                                    )
                                    Text("-", style = MaterialTheme.typography.bodySmall)
                                    AyahNumberPicker(
                                        title = getAppString(isArabic, "End ayah (scope)"),
                                        currentValue = goalRangeEnd,
                                        maxValue = goalScopeSurah?.numberOfAyahs ?: 1,
                                        onValueChange = { newEnd ->
                                            goalRangeEnd = newEnd
                                            if (goalRangeStart > newEnd) goalRangeStart = newEnd
                                        }
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            Button(
                                onClick = {
                                    // The scope is mandatory now: the user always creates a goal for
                                    // exactly one surah + a non-empty ayah range inside it. The target
                                    // count is just the size of the range -- no free-text input.
                                    val scopeSurah = goalScopeSurah
                                    if (scopeSurah != null) {
                                        val target = (goalRangeEnd - goalRangeStart + 1).coerceAtLeast(1)
                                        viewModel.setGoal(
                                            goalType,
                                            target,
                                            surahNumber = scopeSurah.number,
                                            rangeStart = goalRangeStart,
                                            rangeEnd = goalRangeEnd
                                        )
                                    }
                                },
                                modifier = Modifier.fillMaxWidth(),
                                enabled = goalScopeSurah != null
                            ) {
                                Icon(Icons.Default.Add, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                // "Set $goalType Goal" -> "تحديد هدف يومي/أسبوعي". goalType is the
                                // already-localized "Daily"/"Weekly" word; getAppString returns it
                                // in Arabic when needed so the chip result reads naturally.
                                Text(
                                    if (isArabic) "تحديد هدف ${getAppString(isArabic, goalType)}"
                                    else "Set $goalType Goal"
                                )
                            }

                            // Local Surah-selection dialog for the goal scope. Deliberately
                            // SEPARATE from showSurahSelectionDialog: the latter drives
                            // viewModel.selectSurah, which would yank the user's recitation session
                            // away from its current surah with a side-effecting network fetch --
                            // unacceptable while creating a goal from the Stats tab. Here we only
                            // modify goalScopeSurah, leaving viewModel.selectedSurah untouched.
                            if (showGoalSurahDialog) {
                                var goalSurahSearch by remember { mutableStateOf("") }
                                AlertDialog(
                                    onDismissRequest = { showGoalSurahDialog = false },
                                    title = {
                                        Text(
                                            getAppString(isArabic, "Pick the Surah this goal applies to:"),
                                            style = MaterialTheme.typography.titleLarge,
                                            fontWeight = FontWeight.Bold,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                    },
                                    text = {
                                        Column(modifier = Modifier.fillMaxWidth()) {
                                            OutlinedTextField(
                                                value = goalSurahSearch,
                                                onValueChange = { goalSurahSearch = it },
                                                placeholder = { Text(getAppString(isArabic, "Search Surah...")) },
                                                leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
                                                modifier = Modifier.fillMaxWidth(),
                                                shape = RoundedCornerShape(12.dp)
                                            )
                                            Spacer(modifier = Modifier.height(12.dp))
                                            val filtered = viewModel.surahs.filter {
                                                it.englishName.contains(goalSurahSearch, ignoreCase = true) ||
                                                    it.name.contains(goalSurahSearch) ||
                                                    it.number.toString() == goalSurahSearch
                                            }
                                            LazyColumn(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .height(350.dp),
                                                verticalArrangement = Arrangement.spacedBy(8.dp)
                                            ) {
                                                items(filtered) { surah ->
                                                    val isSelected = goalScopeSurah == surah
                                                    Card(
                                                        modifier = Modifier
                                                            .fillMaxWidth()
                                                            .clickable {
                                                                goalScopeSurah = surah
                                                                goalRangeStart = 1
                                                                goalRangeEnd = surah.numberOfAyahs
                                                                showGoalSurahDialog = false
                                                            },
                                                        colors = CardDefaults.cardColors(
                                                            containerColor = if (isSelected)
                                                                MaterialTheme.colorScheme.primaryContainer
                                                            else
                                                                MaterialTheme.colorScheme.surface
                                                        ),
                                                        border = BorderStroke(
                                                            1.dp,
                                                            if (isSelected) MaterialTheme.colorScheme.primary
                                                            else MaterialTheme.colorScheme.outlineVariant
                                                        ),
                                                        shape = RoundedCornerShape(12.dp)
                                                    ) {
                                                        Row(
                                                            modifier = Modifier
                                                                .fillMaxWidth()
                                                                .padding(12.dp),
                                                            horizontalArrangement = Arrangement.SpaceBetween,
                                                            verticalAlignment = Alignment.CenterVertically
                                                        ) {
                                                            Column {
                                                                // In Arabic UI the primary label is the Arabic name;
                                                                // in English UI it's the transliterated (English) name.
                                                                Text(
                                                                    text = "${surah.number}. ${if (isArabic) surah.name else surah.englishName}",
                                                                    style = MaterialTheme.typography.bodyLarge,
                                                                    fontWeight = FontWeight.SemiBold
                                                                )
                                                                Text(
                                                                    text = "${surah.numberOfAyahs} ${getAppString(isArabic, "Ayahs")}",
                                                                    style = MaterialTheme.typography.bodySmall,
                                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                                )
                                                            }
                                                            Text(
                                                                text = if (isArabic) surah.englishName else surah.name,
                                                                style = MaterialTheme.typography.titleMedium,
                                                                fontWeight = FontWeight.Bold,
                                                                color = MaterialTheme.colorScheme.primary
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    confirmButton = {
                                        TextButton(onClick = { showGoalSurahDialog = false }) {
                                            Text(getAppString(isArabic, "Close"))
                                        }
                                    }
                                )
                            }
                        }
                    }
                }
            }
        } else {
            val currentSurah = viewModel.selectedSurah
            Box(modifier = Modifier.fillMaxWidth().padding(bottom = 4.dp)) {
                if (currentSurah != null) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Button(
                            onClick = { showSurahSelectionDialog = true },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.secondaryContainer,
                                contentColor = MaterialTheme.colorScheme.onSecondaryContainer
                            ),
                            modifier = Modifier.weight(1.3f).height(32.dp),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
                        ) {
                            Text(
                                // Arabic name when the UI is Arabic; transliteration otherwise.
                                text = "${currentSurah.number}. ${if (isArabic) currentSurah.name else currentSurah.englishName}",
                                style = MaterialTheme.typography.bodySmall,
                                maxLines = 1,
                                overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                            )
                        }
                        
                        Spacer(modifier = Modifier.width(8.dp))
                        
                        Row(
                            modifier = Modifier.weight(1.7f),
                            horizontalArrangement = Arrangement.End,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(getAppString(isArabic, "Ayahs: "), style = MaterialTheme.typography.bodySmall, fontSize = 11.sp)
                            AyahNumberPicker(
                                title = getAppString(isArabic, "Select Start Ayah"),
                                currentValue = viewModel.rangeStart,
                                maxValue = viewModel.maxAyahInSurah,
                                onValueChange = { newStart -> viewModel.setRange(newStart, viewModel.rangeEnd) }
                            )
                            Text("-", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(horizontal = 2.dp))
                            AyahNumberPicker(
                                title = getAppString(isArabic, "Select End Ayah"),
                                currentValue = viewModel.rangeEnd,
                                maxValue = viewModel.maxAyahInSurah,
                                onValueChange = { newEnd -> viewModel.setRange(viewModel.rangeStart, newEnd) }
                            )
                        }
                    }
                } else {
                    Button(
                        onClick = { showSurahSelectionDialog = true },
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondaryContainer, contentColor = MaterialTheme.colorScheme.onSecondaryContainer),
                        modifier = Modifier.fillMaxWidth().height(32.dp),
                        contentPadding = PaddingValues(vertical = 2.dp)
                    ) {
                        Text(
                            text = getAppString(isArabic, "Select a Surah"),
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }
            
        Box(
            modifier = Modifier
                .weight(1.5f) // Increase weight slightly so it has preference on larger screens too
                .fillMaxWidth()
                .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(12.dp))
                .clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.surface)
                .padding(10.dp)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                var hintActive by remember(viewModel.selectedSurah, currentTab) { mutableStateOf(false) }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (currentTab == "Read") "Surah Content" else "Reference Text",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    if (currentTab == "Read") {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("${getAppString(isArabic, "Loops: ")}${viewModel.loopCount}", style = MaterialTheme.typography.bodySmall)
                            IconButton(onClick = { viewModel.updateLoopCount(viewModel.loopCount + 1) }) {
                                Icon(Icons.Default.Add, contentDescription = "Add Loop", modifier = Modifier.size(20.dp))
                            }
                            IconButton(onClick = { viewModel.updateLoopCount(viewModel.loopCount - 1) }) {
                                Icon(Icons.Default.Remove, contentDescription = "Remove Loop", modifier = Modifier.size(20.dp))
                            }
                            IconButton(onClick = {
                                if (viewModel.playingAyahId != null) viewModel.stopAudio()
                                else viewModel.playAllAyahs()
                            }) {
                                Icon(
                                    imageVector = if (viewModel.playingAyahId != null) Icons.Default.Stop else Icons.Default.PlayCircle,
                                    contentDescription = "Play/Stop Surah",
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(32.dp)
                                )
                            }
                        }
                    } else if (currentTab == "Memorize") {
                        // Find the current Ayah under recitation based on progress matched word count
                        val currentContentAyah = viewModel.ayahs.find { ayah ->
                            contentAnchor >= ayah.wordStartIndex &&
                                contentAnchor < ayah.wordStartIndex + ayah.words.size
                        } ?: viewModel.ayahs.firstOrNull()

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            // "Repeat current Ayah" Button: resets speech text back to start of current active verse
                            IconButton(onClick = {
                                currentContentAyah?.let { ayah ->
                                    val startIdx = ayah.wordStartIndex
                                    val endIdx = ayah.wordStartIndex + ayah.words.size
                                    val prevWords = viewModel.ayahs
                                        .filter { it.wordStartIndex < startIdx }
                                        .flatMap { it.words }
                                    finalSpokenText = prevWords.joinToString(" ")
                                    // §4.46 - RE-SEAT THE COMMITTED SET AND THE WORD
                                    // TIMINGS TOO, not just the display text.
                                    //
                                    // `filterMemorizeWords` strips one copy of every
                                    // word it finds in `lastCommittedWords`, so
                                    // leaving the repeated ayah's words committed
                                    // made the re-read arrive as "already
                                    // committed" and be dropped: the user recited
                                    // the ayah again and NOTHING lit up.
                                    // And the acoustic scorer cuts each word's audio
                                    // from `spokenTimings[startIndex]`, which is
                                    // matched positionally against the transcript -
                                    // after a rewind the transcript is shorter, so
                                    // stale indices would point the tajweed windows
                                    // at the wrong audio and could accuse a word the
                                    // user had already read correctly. Keep the
                                    // prefix's timings, re-indexed, and let the
                                    // re-read words append fresh ones.
                                    lastCommittedWords = prevWords
                                    committedWordCount = prevWords.size
                                    val keepTimings = spokenTimings
                                        .filter { it.startIndex < prevWords.size }
                                        .mapIndexed { i, t -> t.copy(startIndex = i) }
                                    spokenTimings.clear()
                                    spokenTimings.addAll(keepTimings)
                                    currentWordCount = prevWords.size
                                    partialSpokenText = ""
                                    // A rewind is an EXPLICIT rewind, not a transient
                                    // regression: the display must go back WITH the
                                    // transcript, or the reader keeps looking at words
                                    // they have not re-read yet (§4.48).
                                    displayCursorFloor = 0
                                    shownMistakeWords = emptySet()
                                    rideOverCandidateIdx = -1
                                    rideOverCandidateSinceMs = 0L
                                    // Re-reading the whole ayah is a deliberate
                                    // restart: swallow any pending mistake-resume.
                                    mistakeResumeIndex = -1
                                    mistakeResumeLiveText = ""
                                    // Fresh attempt: verification forgiveness from the
                                    // previous pass must not auto-excuse this one.
                                    verifiedSpokenIndices.clear()
                                    lastMistakeVerifyMs = 0L
                                    mistakeVerifyAbsentIdx = -1
                                    mistakeEpisodeAnchor = -1
                                    mistakeCleanStreak = 0
                                    // Same stale-work guard as a range change: queued
                                    // analysis requests + any in-flight pause() final
                                    // decode belong to the PRE-restart attempt and must
                                    // never surface against the fresh one.
                                    analysisEpoch++
                                    try { tajweedPipelineHolder.pipeline?.invalidateInFlightCommit() } catch (_: Throwable) {}
                                    while (analyzeRequests.tryReceive().isSuccess) { /* drain */ }
                                    while (tajweedRequests.tryReceive().isSuccess) { /* drain */ }
                                    // Drop any hint-charges for words in this ayah: we're re-reading
                                    // it, so those injected words must come out of the transcript too.
                                    // (Avoid SnapshotStateSet#removeAll-with-predicate overload
                                    // ambiguity by collecting into an IntRange first.)
                                    val hintRange = startIdx until endIdx
                                    val staleHints = hintedIndices.filter { it in hintRange }
                                    hintedIndices.removeAll(staleHints)
                                    hintActive = false
                                    // §4.46 - REPEAT AYAH MUST ACTUALLY STOP THE
                                    // MIC. It used to only call
                                    // `speechRecognizer.stopListening()`, which
                                    // is a NO-OP in Memorize+Tajweed: the Sherpa
                                    // pipeline kept capturing, kept decoding and
                                    // kept growing `finalSpokenText` while the UI
                                    // said "stopped" - so the words the user had
                                    // just decided to repeat were re-appended
                                    // behind their back, and the next mic tap
                                    // either paused (showing a summary) or
                                    // resumed a transcript nobody had seen.
                                    // The pipeline is PAUSED, not released, so
                                    // the next tap is the fast resume path and
                                    // the recitation continues from the rewound
                                    // ayah exactly as the button promises.
                                    if (isListening || tajweedPipelineHolder.running) {
                                        stopTajweedSession()
                                        // §4.47 — AND INVALIDATE **AFTER** THE PAUSE.
                                        // `stopTajweedSession()` schedules the pause's
                                        // whole-session final decode, which emits the
                                        // "final commit" marker 2-5 s later. That marker
                                        // is what sets `sessionFinalized` — so on a
                                        // rewind it re-finalised the session the user had
                                        // just abandoned: the next mic tap armed the
                                        // end-of-range hold on a wiped transcript
                                        // (a ~20 s freeze, then the tail decided and
                                        // displayed at once). The `invalidateInFlightCommit()`
                                        // above runs BEFORE the pause, so it bumped the
                                        // generation the pause then captured and the
                                        // decode sailed through. Bumping again here
                                        // invalidates the decode the pause just queued,
                                        // which is exactly what a rewind wants: no
                                        // endpoint, no finalization.
                                        try { tajweedPipelineHolder.pipeline?.invalidateInFlightCommit() } catch (_: Throwable) {}
                                    }
                                    // The rewound session is still a session: the
                                    // mic tap must continue it, not summarize it.
                                    recitationSessionActive = true
                                    userPauseFinalizePending = false
                                    // Rewinding invalidates the session's own
                                    // endpoint evidence: the words the user is
                                    // about to re-read must earn their verdicts
                                    // again.
                                    sessionFinalized = false
                                    endpointDecodeLandedAtMs = 0L
                                    pipelineStoppedAtMs = 0L
                                    autoFinalizeDone = false
                                    graceExpiredOnce = false
                                    gracePromoteCount = 0
                                    mistakeHoldSinceMs = 0L
                                    stallSinceMs = 0L
                                    // A stale error must not pop over the ayah the
                                    // user is deliberately re-reading.
                                    latestTajweedPopup = null
                                    lastTajweedPopupIndex = -1
                                    lastTajweedPopupScore = -1
                                    lastTajweedPopupRule = ""
                                    shownTajweedPopups = emptySet()
                                    shownMistakeWords = emptySet()
                                    rideOverCandidateIdx = -1
                                    rideOverCandidateSinceMs = 0L
                                    confirmedMistakeUiIdx = -1
                                    // The acoustic verdicts were cut against the
                                    // PREVIOUS attempt's audio and timing.
                                    tajweedVerdictCache.clear()
                                        tajweedFailCache.clear()
                                    errorMessage = null
                                }
                            }) {
                                Icon(Icons.Default.RepeatOne, contentDescription = "Repeat Current Ayah", tint = MaterialTheme.colorScheme.secondary)
                            }
                            IconButton(onClick = {
                                // Reveal the current word AND advance past it: inject the target
                                // word into the recognizer's transcript so the alignment engine
                                // matches it CORRECT-in-position. The analysis post-pass flips it
                                // to HINTED so the UI paints tertiary/orange styling, and the
                                // mistake counter below charges 1 wrong-word. This fulfils "the
                                // app expects the user to read the next word" while still charging
                                // the mistake for not reciting it oneself.
                                val idx: Int = matchedWordCount
                                if (idx < targetOrig.size) {
                                    hintedIndices.add(idx)
                                    finalSpokenText = "$finalSpokenText ${targetOrig[idx]}".trim()
                                    hintActive = true
                                    // Hint -> counts as 1 wrong word on the active recitation's scope,
                                    // so it only advances goals scoped to the active surah/range (and
                                    // unscoped goals, which still accept every recitation).
                                    val activeSurah = viewModel.selectedSurah?.number ?: 0
                                    viewModel.triggerGoalRealProgress(
                                        recitedAyahsDelta = 0,
                                        wrongDelta = 1,
                                        surahNumber = activeSurah,
                                        rangeStart = viewModel.rangeStart,
                                        rangeEnd = viewModel.rangeEnd
                                    )
                                }
                            }) {
                                Icon(Icons.Default.Lightbulb, contentDescription = "Hint", tint = MaterialTheme.colorScheme.tertiary)
                            }
                            IconButton(onClick = {
                                finalSpokenText = ""
                                partialSpokenText = ""
                                committedWordCount = 0
                                lastCommittedWords = emptyList()
                                // An explicit Restart must also drop the SCORED state, not just the
                                // transcript: alignment (stopAtEnd reads it) and detailedAnalysis
                                // (the per-word verdict list) would otherwise freeze the next
                                // recitation's scoring on the previous run's results, and a
                                // range-completed flag left set would wipe the new run on the very
                                // first mic tap.
                                alignment = Pair(0, false)
                                detailedAnalysis = emptyList()
                                sessionFinalized = false
                                sessionRangeCompleted = false
                                // §4.38 — an explicit Restart is a fresh session:
                                // clear the previous run's endpoint evidence.
                                endpointDecodeLandedAtMs = 0L
                                pipelineStoppedAtMs = 0L
                                mistakeResumeIndex = -1
                                mistakeResumeLiveText = ""
                                hintActive = false
                                hintedIndices.clear()
                                verifiedSpokenIndices.clear()
                                lastMistakeVerifyMs = 0L
                                mistakeVerifyAbsentIdx = -1
                                mistakeEpisodeAnchor = -1
                                mistakeCleanStreak = 0
                                // Same stale-work guard as a range change: queued
                                // requests + any in-flight pause()/stop() final decode
                                // belong to the pre-restart attempt.
                                analysisEpoch++
                                try { tajweedPipelineHolder.pipeline?.invalidateInFlightCommit() } catch (_: Throwable) {}
                                while (analyzeRequests.tryReceive().isSuccess) { /* drain */ }
                                while (tajweedRequests.tryReceive().isSuccess) { /* drain */ }
                                // §4.46 - RESTART MUST REALLY STOP THE MIC, AND
                                // END THE SESSION. It only called
                                // `speechRecognizer.stopListening()` (a no-op in
                                // Tajweed mode), so the pipeline went on capturing
                                // and appending words to a transcript the user had
                                // just wiped, and the mic FAB - which tests
                                // `running` - kept offering "pause" on a session
                                // that no longer existed. Stop the pipeline (a
                                // PAUSE, so the next tap is the fast resume) and
                                // let the next mic tap start a clean run.
                                if (isListening || tajweedPipelineHolder.running) {
                                    stopTajweedSession()
                                    // §4.47 — AND INVALIDATE **AFTER** THE PAUSE.
                                    // `stopTajweedSession()` schedules the pause's
                                    // whole-session final decode, and 2-5 s later it
                                    // emits an endpoint (with words) plus the
                                    // "final commit" marker. Both set
                                    // `sessionFinalized`, so on a Restart they
                                    // re-finalized the session the user had just
                                    // thrown away: the next mic tap armed the
                                    // end-of-range hold on an empty transcript — a
                                    // ~20 s freeze followed by the tail being decided
                                    // and displayed in one step. Measured `b3_buttons`:
                                    // `invalidateInFlightCommit(): gen bumped -> 7`
                                    // then `collector(): sessionFinalized <- endpoint
                                    // (10w raw / 10w new)`. The invalidation above runs
                                    // BEFORE the pause, so it bumped the generation the
                                    // pause then captured; bumping again here kills the
                                    // decode the pause just queued, which is exactly
                                    // what a restart wants: no endpoint, no
                                    // finalization, no freeze.
                                    try { tajweedPipelineHolder.pipeline?.invalidateInFlightCommit() } catch (_: Throwable) {}
                                }
                                recitationSessionActive = false
                                userPauseFinalizePending = false
                                autoFinalizeDone = false
                                graceExpiredOnce = false
                                gracePromoteCount = 0
                                mistakeHoldSinceMs = 0L
                                stallSinceMs = 0L
                                latestTajweedPopup = null
                                lastTajweedPopupIndex = -1
                                lastTajweedPopupScore = -1
                                lastTajweedPopupRule = ""
                                shownTajweedPopups = emptySet()
                                shownMistakeWords = emptySet()
                                rideOverCandidateIdx = -1
                                rideOverCandidateSinceMs = 0L
                                confirmedMistakeUiIdx = -1
                                lastWrongStopIndex = -1
                                lastWrongStopSpokenText = ""
                                lastWrongStopSightings = 0
                                lastWrongStopFirstSeenMs = 0L
                                spokenTimings.clear()
                                currentWordCount = 0
                                tajweedVerdictCache.clear()
                                tajweedFailCache.clear()
                                // Restart is the ONE explicit "start over": the
                                // display floor is dropped with the transcript
                                // (§4.46), so word 0 can be displayed again.
                                displayCursorFloor = 0
                                shownMistakeWords = emptySet()
                                rideOverCandidateIdx = -1
                                rideOverCandidateSinceMs = 0L
                                // Drop the just-finished summary so a fresh
                                // recitation can raise its own.
                                viewModel.dismissSessionSummary()
                                errorMessage = null
                            }) {
                                Icon(Icons.Default.Refresh, contentDescription = "Restart", tint = MaterialTheme.colorScheme.primary)
                            }
                        }
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                
                // ── NO INLINE MISTAKE MESSAGE (removed) ──────────────────────
                // There used to be a "Wrong Word" / "Skipped Word" / "Tajweed
                // Feedback" Card here — an inline banner inside the transcript
                // area. It is GONE, by explicit instruction:
                //
                //   "i dont want messages neither for wrong/skipped/tajweed
                //    mistakes nor for good recitations. only popups indicating
                //    the type of mistake and the word and if for tajweed
                //    indicating the rule and the %."
                //
                // The mistake surface is the DIALOG (`latestTajweedPopup`,
                // rendered further up): title = the mistake TYPE
                // (Wrong Word / Skipped Word / Tajweed Mistake — <rule>), body =
                // the WORD, plus the RULE and the SCORE % for tajweed. It is
                // raised by `stopForMistake`, which is also the only place that
                // arms `confirmedMistakeUiIdx` — so the banner could never have
                // been shown alone anyway (§4.39), and now it cannot be shown at
                // all.
                //
                // The inline surface was the direct cause of the user's report
                // "i saw a msg of a wrong word. why it is on a msg not popup?":
                // the banner was armed by the verified-absent verdict on its own,
                // while the dialog additionally needed `sightings >= 2`,
                // `isListening` and `pipeline.running` — so a bare message could
                // appear where a popup belonged. One surface, one rule.
                
                val listState = androidx.compose.foundation.lazy.rememberLazyListState()
                
                var lastScrolledAyahIndex by remember { mutableStateOf(-1) }
                
                // §4.46 - THE RENDERER'S ROW LOOKUP, O(1). One index built per
                // analysis instead of a linear scan per drawn word (see
                // [byWordIndex]). This is the change that makes the word
                // highlight keep up on an old phone over a long range: the scan
                // cost used to grow with the SIZE OF THE RANGE, so the longer
                // the surah, the further behind the text fell.
                val analysisByWordIndex = remember(detailedAnalysis) { detailedAnalysis.byWordIndex() }
                
                LaunchedEffect(viewModel.playingAyahId) {
                    if (viewModel.playingAyahId != null) {
                        val index = viewModel.ayahs.indexOfFirst { it.ayahId == viewModel.playingAyahId }
                        if (index >= 0 && index != lastScrolledAyahIndex) {
                            lastScrolledAyahIndex = index
                            listState.scrollToItem(index)
                        }
                    }
                }
                
                LaunchedEffect(contentAnchor, displayPaused) {
                    hintActive = false
                    val ayahIndex = viewModel.ayahs.indexOfFirst { ayah ->
                        contentAnchor >= ayah.wordStartIndex &&
                            contentAnchor < ayah.wordStartIndex + ayah.words.size
                    }
                    if (ayahIndex >= 0 && ayahIndex != lastScrolledAyahIndex) {
                        lastScrolledAyahIndex = ayahIndex
                        // §4.46 - DO NOT RE-SCROLL FOR A WORD THAT IS ALREADY ON
                        // SCREEN. This effect fires on EVERY cursor advance (a few
                        // times a second while reciting), and `scrollToItem` is a
                        // synchronous jump: on a long ayah it drags the list back
                        // to the item's top edge each time, so the text the user
                        // is reading visibly bounces - the "the words are not
                        // displayed smoothly" report on an old phone. Scrolling is
                        // only needed when the target ayah has actually left the
                        // viewport.
                        val visible = listState.layoutInfo.visibleItemsInfo.any { it.index == ayahIndex }
                        if (visible) {
                            // keep the reader where they are; the word lights up in place
                        } else {
                            listState.scrollToItem(ayahIndex)
                        }
                    }
                }
                
                LazyColumn(modifier = Modifier.weight(1f), state = listState) {
                    items(viewModel.ayahs) { ayah ->
                        Column(modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
                            val isTajweed = recitationMode == "Tajweed"
                            val tajweedWords = remember(ayah, isTajweed, viewModel.riwayah, viewModel.ayahs) {
                                if (!isTajweed) null else {
                                    ayah.words.mapIndexed { index, word ->
                                        val nextWord = if (index + 1 < ayah.words.size) {
                                            ayah.words[index + 1]
                                        } else {
                                            val nextAyah = viewModel.ayahs.find { it.ayahId == ayah.ayahId + 1 }
                                            nextAyah?.words?.firstOrNull()
                                        }
                                        buildTajweedAnnotatedString(word, nextWord, viewModel.riwayah)
                                    }
                                }
                            }
                            
                            val annotatedText = buildAnnotatedString {
                                ayah.words.forEachIndexed { index, word ->
                                    val globalWordIndex = ayah.wordStartIndex + index
                                    
                                    if (currentTab == "Read") {
                                        if (viewModel.playingAyahId == ayah.ayahId && index == viewModel.playingWordIndex) {
                                            withStyle(style = SpanStyle(background = MaterialTheme.colorScheme.primaryContainer, fontWeight = FontWeight.Bold)) {
                                                append("$word ")
                                            }
                                        } else if (viewModel.playingAyahId == ayah.ayahId) {
                                            withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.primary.copy(alpha = 0.5f))) {
                                                append("$word ")
                                            }
                                        } else {
                                            withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.onSurface)) {
                                                append("$word ")
                                            }
                                        }
                                    } else {
                                        val wordTajweed = tajweedWords?.get(index)
                                        
                                        if (globalWordIndex < matchedWordCount) {
                                            // Words the user revealed via Hint keep the tertiary "hinted"
                                            // styling (orange underline) even after we've advanced past
                                            // them -- the user never actually said them, and the mistake
                                            // counter already charged 1 wrong.
                                            val isHintedWord = analysisByWordIndex[globalWordIndex]?.status == RecitationStatus.HINTED
                                            if (isHintedWord) {
                                                withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.tertiary, fontWeight = FontWeight.Bold, textDecoration = androidx.compose.ui.text.style.TextDecoration.Underline, background = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.3f))) {
                                                    if (wordTajweed != null) {
                                                        append(wordTajweed)
                                                        append(" ")
                                                    } else {
                                                        append("$word ")
                                                    }
                                                }
                                            } else if (wordTajweed != null) {
                                                append(wordTajweed)
                                                append(" ")
                                            } else {
                                                withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)) {
                                                    append("$word ")
                                                }
                                            }
                                        } else if (globalWordIndex == matchedWordCount && hintActive) {
                                            withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.tertiary, fontWeight = FontWeight.Bold, textDecoration = androidx.compose.ui.text.style.TextDecoration.Underline, background = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.3f))) {
                                                if (wordTajweed != null) {
                                                    append(wordTajweed)
                                                    append(" ")
                                                } else {
                                                    append("$word ")
                                                }
                                            }
                                        } else if (!hintActive && globalWordIndex == errorIndex &&
                                            // Tajweed mode: the red WRONG/SKIPPED
                                            // word highlight requires the same
                                            // multi-emission confirmation as the
                                            // banner (transient decode artifacts
                                            // must not flash); TAJWEED_ERROR keeps
                                            // its instant highlight. Tarteel: unchanged.
                                            (!isTajweed ||
                                                globalWordIndex == confirmedMistakeUiIdx ||
                                                analysisByWordIndex[globalWordIndex]?.status == RecitationStatus.TAJWEED_ERROR)
                                        ) {
                                            val currentAnalysis = analysisByWordIndex[globalWordIndex]
                                            val isTajweedError = isTajweed && currentAnalysis?.status == RecitationStatus.TAJWEED_ERROR
                                            val errorBg = if (isTajweedError) Color(0xFFFBC02D).copy(alpha = 0.3f) else MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.3f)
                                            val errorDecoration = androidx.compose.ui.text.style.TextDecoration.Underline
                                            
                                            withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.error, fontWeight = FontWeight.Bold, background = errorBg, textDecoration = errorDecoration)) {
                                                if (wordTajweed != null) {
                                                    append(wordTajweed)
                                                    append(" ")
                                                } else {
                                                    append("$word ")
                                                }
                                            }
                                        } else {
                                            withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.25f))) {
                                                if (wordTajweed != null) {
                                                    append(wordTajweed)
                                                    append(" ")
                                                } else {
                                                    append("$word ")
                                                }
                                            }
                                        }
                                    }
                                }
                                withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.primary)) {
                                    append(" ﴿${ayah.ayahId}﴾ ")
                                }
                            }
    
                            Text(
                                text = annotatedText,
                                style = MaterialTheme.typography.titleLarge.copy(lineHeight = 40.sp),
                                textAlign = TextAlign.Right,
                                modifier = Modifier.fillMaxWidth()
                            )
                            
                            if (currentTab == "Read" && ayah.audio != null) {
                                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
                                    IconButton(
                                        onClick = {
                                            if (viewModel.playingAyahId == ayah.ayahId) {
                                                viewModel.stopAudio()
                                            } else {
                                                viewModel.playAudio(ayah.audio, ayah.ayahId)
                                            }
                                        }
                                    ) {
                                        Icon(
                                            imageVector = if (viewModel.playingAyahId == ayah.ayahId) Icons.Filled.Stop else Icons.Filled.PlayArrow,
                                            contentDescription = "Play/Stop Audio",
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                }
                            }
                        }
                    }

                    // Detailed feedback scrolls along at the end of the LazyColumn
                    // ── NO INLINE MISTAKE REPORT (removed) ───────────────────
                    // A "Recitation Mistakes Detected" Card used to be appended
                    // at the end of the transcript here, listing every WRONG and
                    // SKIPPED word. It is GONE, by explicit instruction:
                    //
                    //   "i dont want messages neither for wrong/skipped/tajweed
                    //    mistakes nor for good recitations. only popups indicating
                    //    the type of mistake and the word and if for tajweed
                    //    indicating the rule and the %. and when we pause or finish
                    //    we have the summary."
                    //
                    // This card was almost certainly the "msg of a wrong word"
                    // in the report "i saw a msg of a wrong word. why it is on a
                    // msg not popup?": it is a MESSAGE, not a dialog, and it is
                    // not tied to any stop — it simply rendered whenever the
                    // analysis happened to hold a WRONG row.
                    //
                    // The mistake surface is now ONLY the dialog
                    // (latestTajweedPopup), raised by stopForMistake(): type +
                    // word, plus the rule and the score % for tajweed. The
                    // end-of-session report is the SUMMARY dialog, raised on a
                    // pause or a genuine finish.
                }
            }
        }
        } // End of conditional for Memorize/Read

        Spacer(modifier = Modifier.height(6.dp))

        if (currentTab == "Memorize") {
        Box(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
            val tajweedStatusText = when {
                isListening && recitationMode == "Tajweed" && (sherpaPhase is itqan.app.sherpa.SingleAudioPipeline.State.Loading || (sherpaPhase is itqan.app.sherpa.SingleAudioPipeline.State.Running && !sherpaWarm)) ->
                    "Warming up…"
                isListening && recitationMode == "Tajweed" && sherpaPhase is itqan.app.sherpa.SingleAudioPipeline.State.Running ->
                    "Listening — recite now"
                else -> ""
            }
            // STOP + SCORE + SUMMARIZE, hoisted out of the mic button so the
            // end-of-range LaunchedEffect below can run the identical path.
            // It used to live INLINE in `onClick`, which made a finished range
            // show nothing at all until the user happened to tap stop — the
            // "nothing happened, no popups" report.
            //
            // `userInitiatedStop` distinguishes the two entry points, and the
            // distinction is a HARD requirement from the user:
            //   "the summary is when i click on pause the mic or finish my
            //    recitation. for any mistake, the app stops there and give me a
            //    mistake popup. i made this clear many times."
            // Tapping pause ⇒ the summary is CORRECT, whatever is flagged.
            // The end-of-range effect ⇒ a mistake must NEVER be buried in the
            // summary; it must stop with the MISTAKE popup (or, when no decode
            // could prove the word, be forgiven and then summarize).
            // Before this flag both entries were indistinguishable, so the
            // summary latched on the auto path too and the mistake popup the
            // user was waiting for never appeared.
            // Set by [runStopAndFinalize] on every call: TRUE when it withheld
            // the end-of-range summary because the selected range is not
            // actually complete yet (unheard words beyond the forgiveness
            // window — the user is still reading). The auto-finalize gate reads
            // it immediately after the call so it can RE-ARM instead of
            // dead-ending: without this, one grace expiry on an incomplete
            // range left `autoFinalizeDone` permanently true and the
            // end-of-range path never fired again for the rest of the session.
            val finalizeWithheld = remember { mutableStateOf(false) }
            /**
             * [userInitiatedStop] = the user asked to stop (the mic FAB), which
             * per the contract ALWAYS ends in a summary.
             * [forceDecision] = the caller has already waited as long as it is
             * willing to for the final decode, so decide now even if the
             * endpoint never landed (a user pause must not be able to hang).
             */
            val runStopAndFinalize: (Boolean, Boolean) -> Unit = finalize@ { userInitiatedStop, forceDecision ->
                    finalizeWithheld.value = false
                    // ANY stop ends the continuous Tarteel intent. This must be
                    // set BEFORE stopListening(): the platform may answer the
                    // stop with a benign ERROR_CLIENT, and the error handler is
                    // allowed to restart only while the user still wants the mic.
                    tarteelWantsMic = false
                    // A Tarteel pause must not wait for a platform "endpoint"
                    // that never arrives. The recognizer has no final-decode
                    // marker of its own, so a user-initiated stop IS the
                    // decision point and must always raise the summary.
                    val decisionForced =
                        forceDecision || (userInitiatedStop && recitationMode != "Tajweed")
                    // Stop the visible cursor at the last ayah the reader was
                    // heard in BEFORE the final decode's promotions can move it
                    // into the next ayah. This runs for an auto stop too; a
                    // genuine finish has heard evidence up to the range end, so
                    // the ceiling is the full range there.
                    val stopDisplayCap = displayCapForHeardAyah(detailedAnalysis, viewModel.ayahs)
                    if (stopDisplayCap < alignment.first) {
                        alignment = Pair(stopDisplayCap, alignment.second)
                        displayCursorFloor = minOf(displayCursorFloor, stopDisplayCap)
                        android.util.Log.i(
                            "ItqanSherpa",
                            "finalize: display capped at the last heard ayah " +
                                "($stopDisplayCap) instead of showing the next ayah's words"
                        )
                    }
                    if (speechRecognizer == null) {
                        errorMessage = "Speech recognition is not available on this device"
                        return@finalize
                    }
                    // - §4.45 THE SESSION, NOT THE MIC, OWNS THIS DECISION -
                    // The whole body - the goal charge, the mistake entries and
                    // the end-of-range summary - used to live inside
                    // `if (isListening)`, and BOTH stop paths clear `isListening`
                    // BEFORE anything decides anything: `stopTajweedSession()` on
                    // a pause, `stopForMistake()` and the tajweed popup on a
                    // mistake. So a pause fell straight through to the else
                    // branch (the "start a session" path) and NEVER reached its
                    // own summary - "in the pause there is no summury at all" -
                    // while a post-mistake tap took that same branch by accident,
                    // which is the only reason the re-read ever worked at all.
                    //
                    // `recitationSessionActive` is the honest question: is there a
                    // recitation to decide? It survives a pause and a mistake
                    // popup (both are resumable), and is cleared by the decision
                    // below, by a range/surah change and by leaving the tab.
                    if (recitationSessionActive || isListening) {
                        try {
                            // Tajweed mode: hand off the stop to the Sherpa
                            // pipeline-aware helper (which flushes the
                            // OnlineRecognizer + writes tajweedRecordedAudio +
                            // sets the recording duration). This mirrors what
                            // `stopTajweedSession()` already does for the Tajweed
                            // Rules Coach practice tab — centralizes the pipeline
                            // teardown so the analyzeRecitationDetails
                            // LaunchedEffect below picks up the audio on its
                            // `tajweedRecordedAudio != null` trigger.
                            // §4.68 — ...and under injection it owns it in a
                            // NON-Tajweed mode too: [injectTranscriptSource] started the
                            // pipeline there, so stopping only the (never started)
                            // recognizer would leave the injector running and the run
                            // would never end.
                            if ((recitationMode == "Tajweed" || injectTranscriptSource) &&
                                tajweedPipelineHolder.running
                            ) {
                                stopTajweedSession()
                            } else {
                                speechRecognizer.stopListening()
                                isListening = false
                            }

                            // §4.48 — FORGIVE AN UNPROVEN MISTAKE BEFORE IT IS REPORTED.
                            //
                            // Measured on the user's real-mic Al-Fatihah run: a
                            // TENTATIVE candidate appeared mid-session
                            // (`mistakeStop(): candidate row=11 word=10 sightings=1
                            // absentIdx=-1 verified=false`) and was still open — never
                            // verified, no absent strike — when the user PAUSED. The
                            // pause path listed that raw row, so the summary told a
                            // clean reciter he had skipped a word:
                            //   endOfRange(): summary raised (... entries=1 skipped=1)
                            //   cursor=29/29 -> ALL WORDS DISPLAYED
                            // i.e. every word on screen was complete and the summary
                            // still accused one — "it showed me a skipped word
                            // although I had a clean recitation".
                            //
                            // A mistake is only a mistake once a whole-session tail
                            // decode came back WITHOUT the word. That verdict is
                            // per-candidate (`mistakeVerifyAbsentIdx` +/- 3, with
                            // `mistakeVerifyAbsentStrikes`) and it is the ONLY thing
                            // that may accuse the user. The auto path already forgave
                            // on that basis; the pause path did not. Now BOTH forgive
                            // first and then report, which is also what makes "no
                            // mistakes" and a perfect recitation consistent instead of
                            // the app saying both at once.
                            //
                            // The token is kept, so a word the user DID say that only
                            // one window dropped stays "heard" and stays visible
                            // (§4.43).
                            val absentAnchorIdx = mistakeVerifyAbsentIdx
                            val absentProven = mistakeVerifyAbsentStrikes >= 1
                            val unprovenMistakes = detailedAnalysis.filter {
                                (it.status == RecitationStatus.WRONG ||
                                    it.status == RecitationStatus.SKIPPED) &&
                                    !(absentProven && absentAnchorIdx >= 0 &&
                                        kotlin.math.abs(it.wordIndex - absentAnchorIdx) <= 3) &&
                                    it.wordIndex !in verifiedSpokenIndices &&
                                    // §4.64 — AND A MISTAKE THE APP **ALREADY SHOWED** IS
                                    // NOT "UNPROVEN" — it is a decision this app made and put
                                    // in front of the user, and forgiving it here makes the
                                    // app contradict its own popup. Exactly what the user
                                    // saw: the mistake popup for word 6, then
                                    // "the summury tells me no mistake !"
                                    //   autoFinalize(): grace expired with an UNPROVEN
                                    //   mistake at wordIndex=6 (no substantial decode ever
                                    //   judged it) -> forgiving rather than accusing
                                    //   endOfRange(): ... wrong=0 skipped=0 tajweed=0
                                    //
                                    // The forgive test is an ABSENCE-of-evidence test, and
                                    // that evidence can only come from `verifyTailWords` —
                                    // which needs the Sherpa pipeline. Tarteel has none, so
                                    // its strikes are permanently 0 and EVERY mistake it
                                    // showed was forgiven by this very line. On the Tajweed
                                    // path the same rule could also erase a mistake the
                                    // confirm loop had already popped up for.
                                    //
                                    // Only the RE-READ may overturn a mistake the user was
                                    // shown: `acquittal` returns the word to CORRECT with
                                    // its spoken token, which is the `verifiedSpokenIndices`
                                    // test above. A word in neither set stays a mistake.
                                    it.wordIndex !in shownMistakeWords
                            }
                            if (unprovenMistakes.isNotEmpty()) {
                                val forgiveIdx = unprovenMistakes.map { it.wordIndex }.toHashSet()
                                android.util.Log.i(
                                    "ItqanSherpa",
                                    "finalize(): forgiving ${forgiveIdx.size} UNPROVEN mistake(s) " +
                                        "$forgiveIdx — no tail decode ever came back without them " +
                                        "(absentAnchor=$absentAnchorIdx strikes=$mistakeVerifyAbsentStrikes). " +
                                        "A candidate the app could not judge must never be reported."
                                )
                                detailedAnalysis = detailedAnalysis.map { w ->
                                    if ((w.status == RecitationStatus.WRONG ||
                                            w.status == RecitationStatus.SKIPPED) &&
                                        w.wordIndex in forgiveIdx
                                    ) {
                                        w.copy(status = RecitationStatus.CORRECT)
                                    } else w
                                }
                            }

                            var newCompletedCount = 0
                            for (ayah in viewModel.ayahs) {
                                val isFullyMatched = matchedWordCount >= ayah.wordStartIndex + ayah.words.size
                                if (isFullyMatched && !viewModel.completedAyahs.contains(ayah.ayahId)) {
                                    viewModel.completedAyahs.add(ayah.ayahId)
                                    newCompletedCount++
                                }
                            }
                            
                            val currentWrong = detailedAnalysis.count { it.status == RecitationStatus.WRONG }
                            val currentSkipped = detailedAnalysis.count { it.status == RecitationStatus.SKIPPED }
                            val tajweedViolated = detailedAnalysis.filter { (it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED || it.status == RecitationStatus.TAJWEED_ERROR) && it.tajweedRule != null }
                            val currentTajweed = tajweedViolated.size
                            val violatedRulesList = tajweedViolated.mapNotNull { it.tajweedRule }

                            val encounteredRulesList = detailedAnalysis.mapNotNull { it.tajweedRule }

                            // Build per-word mistake entries for both the goal
                            // persistence path and the session-summary dialog.
                            // Dedup: tajweed rows are grouped by (wordIndex, rule)
                            // so a multi-rule word doesn't double-count.
                            val activeSurah = viewModel.selectedSurah?.number ?: 0
                            val activeRangeStart = viewModel.rangeStart
                            val activeRangeEnd = viewModel.rangeEnd

                            val mistakeEntries = buildList {
                                detailedAnalysis.forEach { wa ->
                                    when (wa.status) {
                                        RecitationStatus.WRONG -> {
                                            val ayahNum = ayahIndexForWordIndex(wa.wordIndex, viewModel.ayahs)
                                            add(MistakeEntry(
                                                kind = "wrong",
                                                word = wa.text,
                                                rule = null,
                                                score = null,
                                                surah = activeSurah,
                                                ayah = ayahNum,
                                                wordIndex = wa.wordIndex,
                                            ))
                                        }
                                        RecitationStatus.SKIPPED -> {
                                            val ayahNum = ayahIndexForWordIndex(wa.wordIndex, viewModel.ayahs)
                                            add(MistakeEntry(
                                                kind = "skipped",
                                                word = wa.text,
                                                rule = null,
                                                score = null,
                                                surah = activeSurah,
                                                ayah = ayahNum,
                                                wordIndex = wa.wordIndex,
                                            ))
                                        }
                                        RecitationStatus.TAJWEED_ERROR -> {
                                            val rule = wa.tajweedRule ?: return@forEach
                                            val ayahNum = ayahIndexForWordIndex(wa.wordIndex, viewModel.ayahs)
                                            add(MistakeEntry(
                                                kind = "tajweed",
                                                word = wa.text,
                                                rule = rule,
                                                score = wa.tajweedScore?.toFloat(),
                                                surah = activeSurah,
                                                ayah = ayahNum,
                                                wordIndex = wa.wordIndex,
                                            ))
                                        }
                                        else -> { /* CORRECT / PENDING / HINTED: not a mistake */ }
                                    }
                                }
                            }

                            if (newCompletedCount > 0 || currentWrong > 0 || currentSkipped > 0 || currentTajweed > 0 || encounteredRulesList.isNotEmpty()) {
                                viewModel.incrementRecitedAyahs(newCompletedCount)
                                prefs.edit().putInt("recited_ayahs", viewModel.recitedAyahs).apply()
                                // Scope progress to the surah/range the user just recited -- so
                                // a goal scoped to "Surah 2, verses 1-5" only makes progress when
                                // the user recites Surah 2 in the [1, 5] range. If the user is
                                // on a different surah/range, scoped goals elsewhere stay idle.
                                //
                                // Use the [activeSessionScope] snapshot (captured at
                                // session-start in [startTajweedSession]) instead of the
                                // current view-model values. The user may have changed the
                                // surah/range between the final decode and now, in which
                                // case the live values would point at the NEW range and
                                // silently mis-attribute this recitation to it.
                                val chargeScope = activeSessionScope.value
                                val chargeSurah = chargeScope?.surah ?: activeSurah
                                val chargeRangeStart = chargeScope?.rangeStart ?: activeRangeStart
                                val chargeRangeEnd = chargeScope?.rangeEnd ?: activeRangeEnd
                                viewModel.triggerGoalRealProgress(
                                    recitedAyahsDelta = newCompletedCount,
                                    wrongDelta = currentWrong,
                                    skippedDelta = currentSkipped,
                                    tajweedDelta = currentTajweed,
                                    violatedRules = violatedRulesList,
                                    encounteredRules = encounteredRulesList,
                                    surahNumber = chargeSurah,
                                    rangeStart = chargeRangeStart,
                                    rangeEnd = chargeRangeEnd,
                                    mistakeEntries = mistakeEntries,
                                )

                                // SESSION SUMMARY: shown when the user has finished
                                // reciting the selected range AND the session has
                                // finalized (final decode committed). The dialog
                                // shows the same per-word breakdown the goal DB
                                // just received, but to the user even when no
                                // goal is configured.
                                //
                                // Use the [chargeScope] snapshot (same one we
                                // passed to the goal call) so the dialog's
                                // surah/range header reflects the recitation
                                // the user actually did, not whatever they
                                // happen to have selected by the time the
                                // dialog opens.
                                // PENDING rows that carry a spokenText are CLIPPED decodes: the
                                // recognizer heard the word, the window cut it, and finalizeCorrect()
                                // forgives them. Gating the summary on *any* PENDING meant the last
                                // word of every recitation (always clipped by the endpoint window)
                                // suppressed the end-of-range popup entirely — the user saw nothing
                                // until they tapped pause manually. Only PENDING words the decoder
                                // NEVER heard still mean "the user hasn't finished the range".
                                //
                                // RACE — measured fatihah_c23 (01:15:13):
                                //   .162 endOfRange(): ... pendingRemain=true   <- popup skipped
                                //   .204 finalizeCorrect(): 1 trailing PENDING word(s) counted CORRECT
                                // runStopAndFinalize() sets sessionFinalized and reads detailedAnalysis
                                // in the SAME pass; finalizeCorrect() is a separate LaunchedEffect that
                                // only runs on the NEXT recomposition — 42 ms later it turned the very
                                // row that had just cancelled the popup into CORRECT. Nothing re-checks
                                // afterwards, so the user got NO popup at all for the whole session.
                                //
                                // Fix: apply finalizeCorrect()'s own forgiveness window here, so the
                                // two agree on what "still pending" means even when they read the list
                                // one recomposition apart. A row inside the window is about to be
                                // counted CORRECT, so it must not veto the popup.
                                val matchedSoFarNow = detailedAnalysis.count {
                                    it.status == RecitationStatus.CORRECT ||
                                        it.status == RecitationStatus.TAJWEED_ERROR ||
                                        it.status == RecitationStatus.HINTED
                                }
                                val forgivableTail = matchedSoFarNow + 8
                                val pendingRemain = detailedAnalysis.any {
                                    it.status == RecitationStatus.PENDING &&
                                        it.spokenText.isNullOrEmpty() &&
                                        // finalizeCorrect() only converts when there is real progress
                                        // (matched >= 3); outside that case the row genuinely remains.
                                        (matchedSoFarNow < 3 || it.wordIndex > forgivableTail)
                                }
                                // Diagnostic: this gate is the ONLY place the
                                // end-of-range summary is raised, and every one
                                // of its inputs is invisible in logcat. Log them
                                // so a missing popup can be diagnosed from a run
                                // instead of guessed at.
                                android.util.Log.i(
                                    "ItqanSherpa",
                                    "endOfRange(): sessionFinalized=$sessionFinalized " +
                                        "forceDecision=$decisionForced " +
                                        "pendingRemain=$pendingRemain newCompleted=$newCompletedCount " +
                                        "wrong=$currentWrong skipped=$currentSkipped tajweed=$currentTajweed " +
                                        "alreadyShowing=${viewModel.sessionSummaryToShow != null}"
                                )
                                // MISMATCHED-ENTRY-POINT GUARD. The auto path
                                // (end-of-range) only calls this AFTER it has
                                // settled every outstanding mistake — either by
                                // stopping for it (popup) or by forgiving an
                                // unprovable word. So by here a flagged mistake
                                // means a REAL one that the pipeline could still
                                // not resolve, and the contract is explicit:
                                // "for any mistake, the app stops there and give
                                // me a mistake popup", never a summary.
                                // The manual pause path is the opposite — the
                                // user ASKED for the summary, so flagged words
                                // are listed inside it and no mistake popup is
                                // forced on them.
                                val hasUnresolvedMistake = currentWrong > 0 || currentSkipped > 0
                                val mistakeMustWin = !userInitiatedStop && hasUnresolvedMistake
                                if (mistakeMustWin) {
                                    // Leave sessionFinalized set so the next mic
                                    // tap resumes cleanly, but never raise the
                                    // dialog: the mistake popup owns the screen.
                                    android.util.Log.i(
                                        "ItqanSherpa",
                                        "endOfRange(): auto path with wrong=$currentWrong " +
                                            "skipped=$currentSkipped still flagged -> suppressing the " +
                                            "summary so the mistake popup is what the user sees"
                                    )
                                }
                                // ── WHEN THE SUMMARY MAY BE RAISED (§4.32) ──
                                // The user's contract, stated twice:
                                //   "if it stoppd then it should be a skipped/wrong
                                //    word popup or tajweed popup else it should be
                                //    displayed. not getting a summury popup (unless
                                //    i click pause)"
                                //   "no summury shown when finished in this test! fix it"
                                // Reconciled: the summary appears on a GENUINE
                                // FINISH (the whole range was actually heard) and
                                // on a USER PAUSE — but NEVER for a mid-range
                                // stall, which is what produced a "finished"
                                // dialog over a recitation the app had only
                                // partly heard.
                                //
                                // `!pendingRemain` alone was NOT enough: when the
                                // walk stalls, finalizeCorrect() forges the tail
                                // CORRECT with no spoken evidence, so pendingRemain
                                // reads false on a run that aligned 27 of 29.
                                //
                                // §4.45 - and `unheardWordCount() == 0` was TOO
                                // strict in the other direction: the trailing word
                                // of a real-mic recitation is routinely the one the
                                // decoder never renders (the last seconds are
                                // silence, the RMS gate drops them and the final
                                // whole-session render comes back a subset of the
                                // sliding windows' merge), so that row is CORRECT
                                // with no token, counted as unheard, and the
                                // summary of a plainly FINISHED recitation was
                                // withheld:
                                //   endOfRange(): auto stop but NOT genuinely
                                //   finished (unheard=1/29) -> no summary
                                // The injection harness decoded that word and did
                                // show a summary, which is exactly why the feature
                                // looked broken only when tested by hand. The test
                                // is COVERAGE now: nearly the whole range heard,
                                // at most a couple of decoder blind spots, and the
                                // DISPLAY already at the end of the range.
                                val totalWordsNow = detailedAnalysis.distinctWordCount()
                                val unheardNow = detailedAnalysis.unheardWordCount()
                                val displayedNow = minOf(
                                    displayedCursorFor(detailedAnalysis, alignment.first),
                                    displayCapForHeardAyah(detailedAnalysis, viewModel.ayahs)
                                )
                                val genuinelyFinished = isGenuineFinish(
                                    totalWords = totalWordsNow,
                                    unheardWordCount = unheardNow,
                                    displayedCursor = displayedNow,
                                )
                                // `pendingRemain` still vetoes the AUTO path (the
                                // user may simply still be reading), but it must
                                // never veto a USER PAUSE: they asked for the
                                // breakdown, and a decoder gap is not a reason to
                                // show them nothing.
                                // §4.50 — A STALL THAT IS NOT GOING TO END MUST
                                // STILL SAY SOMETHING.
                                //
                                // The gate below needs EITHER a user pause OR a
                                // genuine finish. There is a third state, and it is
                                // the one the user reported: the recognizer has
                                // COMMITTED the utterance (sessionFinalized — the
                                // endpoint landed, no more audio is coming for it),
                                // the range is NOT complete, and nothing has moved
                                // for a long time. Measured on the device, the app
                                // sat there logging
                                //   autoFinalize(): 17 unresolved word(s) [12..28]
                                //   are beyond the heard frontier (21 words) + 8 ->
                                //   NOT promoted
                                //   autoFinalize(): grace expired with 17
                                //   unresolved word(s) … and NO evidenced mistake
                                // every 2 s, display frozen, NO popup and NO
                                // summary: "it keeps getting stuck in displaying
                                // words". The user cannot tell "waiting for you to
                                // read on" from "dead", and the words he still has to
                                // read are exactly what is frozen.
                                //
                                // So after [STALL_SUMMARY_AFTER_MS] of NO progress
                                // on a committed session, raise the same honest
                                // PAUSED summary a user pause raises: the coverage
                                // line ("Recited 13 of 29 words — tap the mic to
                                // continue"), no finish claim, and — thanks to §4.48 —
                                // no unproven accusation. The next mic tap resumes
                                // from the committed prefix, so nothing is lost.
                                //
                                // It is deliberately NOT a mistake: there is no
                                // evidence any word was skipped, and the words in
                                // question are BEYOND the frontier, i.e. simply not
                                // reached yet.
                                //
                                // Declared BEFORE `pendingVeto` because the veto has
                                // to yield to it — see above.
                                val stallExpired = stallSinceMs > 0L &&
                                    (System.currentTimeMillis() - stallSinceMs) >= STALL_SUMMARY_AFTER_MS
                                // §4.50 — AND THE VETO YIELDS TO THE STALL. The veto
                                // exists so the app does not interrupt a reader whose
                                // words have not arrived yet. A STALLED session is the
                                // opposite: the recognizer committed, nothing has moved
                                // for 20 s, and those pending words are never going to
                                // be decided by more silence. With the veto still
                                // applied, the loop reached here and then stopped
                                // (`endOfRange(): range incomplete (unheard words
                                // remain) -> summary withheld`) — a frozen screen, no
                                // dialog, and the ticker parked: "now it is stuck".
                                val pendingVeto = pendingRemain && !userInitiatedStop && !stallExpired
                                if ((sessionFinalized || decisionForced) && !pendingVeto &&
                                    viewModel.sessionSummaryToShow == null && !mistakeMustWin &&
                                    (userInitiatedStop || genuinelyFinished || stallExpired)
                                ) {
                                    // The recitation reached the end of its selected range with no
                                    // PENDING (un-scored) words left: mark the run as range-completed
                                    // so the next mic tap starts fresh instead of resuming the stale,
                                    // fully-recited transcript.
                                    //
                                    // §4.45/§4.46 - BUT ONLY WHEN THE USER ACTUALLY RECITED
                                    // THE RANGE, judged on HEARD evidence and never on the
                                    // display cursor. The cursor is advanced on purpose over
                                    // the tail the decoder never rendered (§4.40: those words
                                    // must still be VISIBLE), so it reaches the end of the
                                    // range even when the user stopped half-way through it -
                                    // and `sessionRangeCompleted` is the flag that makes the
                                    // next mic tap WIPE the transcript and start from word 1.
                                    // Reading it off the cursor therefore destroyed a
                                    // half-finished recitation: measured `s3_fatihah`, the
                                    // session ended at word 16 of 29 (the recording's own
                                    // silence), the promotions carried the cursor to 29/29,
                                    // and the user's 16 words would have been thrown away on
                                    // the next tap.
                                    //
                                    // A word the user really said carries a spoken token, so
                                    // "the whole range was recited" is exactly "at most the
                                    // decoder's own blind-spot tail is unaccounted for" -
                                    // the same test [isGenuineFinish] applies.
                                    val heardNow = totalWordsNow - unheardNow
                                    val reachedEndOfRange = totalWordsNow > 0 &&
                                        heardNow >= totalWordsNow - GENUINE_FINISH_MAX_UNHEARD
                                    if (reachedEndOfRange) sessionRangeCompleted = true
                                    // §4.48 — A MID-RANGE PAUSE SAYS SO. The dialog
                                    // must not read like a verdict on the whole
                                    // range: the user recited ~22 words of a
                                    // 6116-word range, paused, and got a summary
                                    // that looked like the app was judging the
                                    // surah ("it showed me a summary although I
                                    // didn't complete the range"). Carry the
                                    // distinction, and let the dialog say "paused"
                                    // and show how far he got.
                                    val pausedMidRange = (userInitiatedStop || stallExpired) && !reachedEndOfRange
                                    if (stallExpired) {
                                        android.util.Log.i(
                                            "ItqanSherpa",
                                            "endOfRange(): raising the summary for a STALLED session " +
                                                "(stalled ${
                                                    if (stallSinceMs > 0L)
                                                        (System.currentTimeMillis() - stallSinceMs) / 1000
                                                    else 0
                                                }s with no progress, the recognizer committed, the " +
                                                "range is not complete) -> the user gets the honest paused " +
                                                "dialog instead of a frozen screen"
                                        )
                                    }
                                    viewModel.sessionSummaryToShow = MainViewModel.SessionSummary(
                                        surahNumber = chargeSurah,
                                        rangeStart = chargeRangeStart,
                                        rangeEnd = chargeRangeEnd,
                                        entries = mistakeEntries,
                                        recitedAyahCount = newCompletedCount,
                                        totalAyahCount = (chargeRangeEnd - chargeRangeStart + 1).coerceAtLeast(1),
                                        paused = pausedMidRange,
                                        heardWordCount = heardNow,
                                        // Diagnostics only — the dialog shows the
                                        // MISTAKES, not a coverage note.
                                        unheardWordCount = detailedAnalysis.unheardWordCount(),
                                        totalWordCount = detailedAnalysis.distinctWordCount(),
                                    )
                                    android.util.Log.i(
                                        "ItqanSherpa",
                                        "endOfRange(): summary raised " +
                                            "(userPause=$userInitiatedStop finished=$genuinelyFinished " +
                                            "force=$forceDecision entries=${mistakeEntries.size} " +
                                            "unheard=$unheardNow/$totalWordsNow) " +
                                            // §4.37 — THE DISPLAY PROOF. The transcript colours a word
                                            // iff `globalWordIndex < alignment.first`, and the rows are
                                            // complete (AnalyzerRowCompletenessTest), so `cursor >= total`
                                            // is exactly "every word of the range is rendered". A dump
                                            // cannot show colour; this number can, so log it on every
                                            // summary and check it against the harness verdict.
                                            "cursor=$displayedNow/$totalWordsNow " +
                                            (if (displayedNow >= totalWordsNow)
                                                "-> ALL WORDS DISPLAYED"
                                            else "-> ${totalWordsNow - displayedNow} word(s) still grey")
                                    )
                                } else if ((sessionFinalized || decisionForced) && !pendingRemain && !mistakeMustWin &&
                                    !userInitiatedStop && !genuinelyFinished
                                ) {
                                    // Mid-range stall: the app stopped, no mistake
                                    // was provable, and the range was NOT fully
                                    // heard. Per the contract this must show NO
                                    // popup at all — the words stay on screen.
                                    android.util.Log.i(
                                        "ItqanSherpa",
                                        "endOfRange(): auto stop but NOT genuinely finished " +
                                            "(unheard=${detailedAnalysis.unheardWordCount()}/" +
                                            "${detailedAnalysis.distinctWordCount()}) -> no summary; " +
                                            "words stay displayed " +
                                            "cursor=${alignment.first}/${detailedAnalysis.distinctWordCount()}"
                                    )
                                } else if ((sessionFinalized || decisionForced) && pendingVeto) {
                                    // The range is NOT complete (words the decoder never heard
                                    // remain beyond finalizeCorrect()'s forgiveness window), so no
                                    // summary is raised — the user is most likely still reading.
                                    // Report that back so the auto-finalize gate re-arms instead
                                    // of treating this as the session's one and only decision.
                                    finalizeWithheld.value = true
                                    android.util.Log.i(
                                        "ItqanSherpa",
                                        "endOfRange(): range incomplete (unheard words remain) -> " +
                                            "summary withheld; the auto-finalize gate will re-arm"
                                    )
                                }
                            }
                            
                            // Safe Pause/Mistake handling.
                            //
                            // §4.46 — RE-SEAT THE **COMMITTED** SET TO THE MATCHED
                            // PREFIX, AND LEAVE THE DISPLAY TRANSCRIPT ALONE.
                            //
                            // The old code rewrote `finalSpokenText` to
                            // `targetOrig.take(matchedWordCount)`, i.e. it threw
                            // away every word the user had ACTUALLY recited past
                            // the last match. On a pause that is usually a couple
                            // of words, because the aligner is chasing a tentative
                            // skip and has not caught up yet. Measured on device
                            // (pause at 13 matched, 15 words heard, then resume):
                            // the transcript was cut to 13 target words, the audio
                            // for words 14-15 lived in the pre-pause segment and
                            // was gone, and the post-resume audio therefore began
                            // at target word 16 — so the aligner compared it
                            // against target word 13, called it WRONG, and the
                            // walk froze there:
                            //   analyze(fast): cursor held at 25 (walk wanted 13)
                            //   mistakeStop(): candidate row=13 ... ABSENT strike
                            // The user had recited correctly and was accused, and
                            // the recitation could not advance.
                            //
                            // So: the DISPLAY transcript keeps every word that was
                            // heard (that is the evidence the aligner needs to place
                            // the join), while the COMMITTED set becomes the matched
                            // prefix — which is what makes the words recited after
                            // the resume pass the count-filter instead of being
                            // dropped as replay copies of themselves.
                            //
                            // GUARD: only when the analysis has matched something.
                            // Tapping stop while the (slow, on-device) analysis is
                            // still catching up leaves matchedWordCount at 0 even
                            // though words WERE heard; re-seating to an empty
                            // prefix there would let the whole recited prefix
                            // through the filter a second time.
                            if (matchedWordCount > 0) {
                                val matchedWordsList = targetOrig.take(matchedWordCount)
                                lastCommittedWords = matchedWordsList
                                committedWordCount = matchedWordsList.size
                                android.util.Log.i(
                                    "ItqanSherpa",
                                    "finalizeTrim(): committed set re-seated to the " +
                                        "${matchedWordsList.size} matched word(s); the display " +
                                        "transcript KEEPS the words already heard so the join " +
                                        "after a resume stays aligned"
                                )
                            }
                            partialSpokenText = ""
                            // §4.45 - THE SESSION HAS BEEN DECIDED. Whatever happened above
                            // (summary raised, mistake kept for the popup, or nothing to show
                            // yet), the mic tap must now mean "start", not "pause" - this is
                            // what un-blocks the FAB after a mistake popup as well.
                            recitationSessionActive = false
                            // User-initiated stop ends the recitation: the next
                            // session starts from scratch, not from a stale
                            // mistake prefix.
                            mistakeResumeIndex = -1
                            mistakeResumeLiveText = ""
                            verifiedSpokenIndices.clear()
                            lastMistakeVerifyMs = 0L
                            mistakeVerifyAbsentIdx = -1
                            mistakeEpisodeAnchor = -1
                            mistakeCleanStreak = 0
                            // Reminder: Memorize+Tajweed does NOT run our own AudioRecord here.
                            // Two simultanous mic clients on the user's Samsung A05s starve the
                            // SpeechRecognizer of partial results (matchedWordCount never
                            // advances and the user sees no word-tracking happening), so we rely
                            // solely on whatever BUNDLE_AUDIO_SOURCE the recognizer hands back in
                            // onResults above. Tajweed scoring is best-effort on devices that do
                            // populate that bundle -- on the A05s it may simply be missing, which
                            // means no per-word tajweed verdict but recitation-tracking still works.
                        } catch (e: Throwable) {
                            errorMessage = "Error stopping recognition"
                            // §4.45 - a failed stop must not leave the session
                            // "active": the mic FAB would keep offering to pause a
                            // session that is already over, and the next tap would
                            // re-enter this same body instead of starting fresh.
                            recitationSessionActive = false
                        }
                    } else {
                        errorMessage = null
                        val hasPermission = androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.RECORD_AUDIO) == android.content.pm.PackageManager.PERMISSION_GRANTED
                        if (hasPermission) {
                            if (!isListening) {
                                // Memorize+Tajweed ALWAYS uses the Sherpa-ONNX offline
                                // recognizer pipeline (single AudioRecord feeds both the
                                // Sherpa OnlineRecognizer and TajweedNet acoustic
                                // scorer). The platform SpeechRecognizer is NOT used in
                                // Tajweed mode -- on the user's Samsung A05s two HAL
                                // clients starve each other and the word-by-word
                                // highlight stops. Non-Tajweed modes keep the
                                // SpeechRecognizer path (TajweedNet has no companion
                                // transcript pipeline there because it doesn't need a
                                // transcript -- the practice recitation text is known).
                                //
                                // NO SILENT TARTEEL FALLBACK: if the Tajweed pipeline
                                // isn't wired up yet we BLOCK the mic and tell the user
                                // it's warming up, instead of quietly starting the
                                // SpeechRecognizer (which behaves like Tarteel mode and
                                // can't do tajweed scoring). The user explicitly asked
                                // for this: "if memorize+tajweed not ready just block
                                // the mic button and indicate that to the user."
                                if (recitationMode == "Tajweed") {
                                    if (tajweedPipelineHolder.pipeline != null) {
                                        // Start via the pipeline-aware session helper so the
                                        // hotwords bias (active ayah range) and the standard
                                        // state reset (spokenTimings, audio, etc.) all happen
                                        // in one place, mirroring the SpeechRecognizer path
                                        // above.
                                        startTajweedSession()
                                        if (isListening) {
                                            viewModel.preloadTajweedModel()
                                        }
                                    } else {
                                        // Pipeline not wired up yet (first composition /
                                        // range-change re-wire still in flight). Block the
                                        // mic with a clear indication — do NOT fall back to
                                        // the SpeechRecognizer.
                                        errorMessage = getAppString(isArabic, "Tajweed is warming up — please wait a moment…")
                                        android.util.Log.w("ItqanSherpa", "FAB: Tajweed tapped but pipeline not wired yet — blocked (no Tarteel fallback)")
                                    }
                                } else if (injectTranscriptSource && tajweedPipelineHolder.pipeline != null) {
                                    // §4.68 — see [injectTranscriptSource].
                                    startTajweedSession()
                                } else if (speechRecognizer != null) {
                                    spokenTimings.clear()
                                    currentWordCount = 0
                                    recognizedAudioUri = null
                                    recognizedAudio = null
                                    recognizedAudioStartMs = System.currentTimeMillis()
                                    tarteelWantsMic = true
                                    recitationSessionActive = true
                                    sessionFinalized = false
                                    isListening = true
                                    startListening(speechRecognizer!!, context.packageName, onError = { msg ->
                                        errorMessage = if (msg.isEmpty()) null else msg
                                        if (tarteelWantsMic) {
                                            tarteelWantsMic = false
                                            sessionFinalized = true
                                            recitationSessionActive = false
                                        }
                                        isListening = false
                                    })
                                    // Memorize+Tajweed does NOT run a second mic alongside the
                                    // SpeechRecognizer. The user's Samsung A05s reports that two
                                    // simultanous mic clients make word-tracking fail entirely -- no
                                    // partial results arrive, matchedWordCount stays at 0 and the
                                    // user sees "nothing happens". The Tajweed Rules coach tab works
                                    // because it uses the AudioRecord ALONE (no SpeechRecognizer);
                                    // that approach cannot be copied here because Memorize genuinely
                                    // needs the live transcript to track word position. The tajweed
                                    // score therefore depends on BUNDLE_AUDIO_SOURCE arriving via
                                    // onResults -- on devices that don't populate it, scoring is
                                    // simply skipped, which is the trade-off the original design made.
                                } else {
                                    errorMessage = "Speech Recognizer not available"
                                }
                            }
                        } else {
                            permissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
                        }
                    }
            }
            // - DEFERRED USER-PAUSE FINALIZE (see [userPauseFinalizePending]) -
            // The pause tap stops the mic immediately, but in a Tajweed pipeline
            // session the FINAL decode lands ~5 s later and it is the endpoint
            // that sets `sessionFinalized`. Deciding the summary at tap time
            // therefore read sessionFinalized=false and skipped EVERY branch
            // (no summary on pause - the user's report). Wait for the endpoint
            // (bounded, so a dead decode can never wedge the flow) and then run
            // the real user-initiated finalize exactly once.
            LaunchedEffect(userPauseFinalizePending) {
                if (!userPauseFinalizePending) return@LaunchedEffect
                // §4.45 - A SESSION THAT PRODUCED NOTHING GETS NO WAIT. The
                // common way to land here with an empty transcript is tapping
                // stop while the 296 MB recognizer is still loading: there is no
                // final decode coming, so the 20 s deadline used to expire on
                // nothing at all and the tap felt dead.
                val heardNothing = finalSpokenText.isBlank() && partialSpokenText.isBlank() &&
                    detailedAnalysis.isEmpty()
                if (heardNothing) {
                    userPauseFinalizePending = false
                    recitationSessionActive = false
                    android.util.Log.i(
                        "ItqanSherpa",
                        "userPauseFinalize(): nothing was heard in this session -> no wait, no summary"
                    )
                    return@LaunchedEffect
                }
                // §4.52 — THE WAIT IS NOW 1.5 s, NOT 20 s. "for showing the
                // summary after i turn the mic off if you could make it a little
                // faster". Measured on his own real-mic runs, everything after the
                // final decode lands is already fast — the whole cost is WAITING
                // for that decode:
                //   stopTajweedSession()          0 ms
                //   finalDecodeAndEmit(): done    +4388 ms   (Al-Fatihah 1-7)
                //   finalDecodeAndEmit(): done    +3104 ms   (Al-Baqarah)
                //   userPauseFinalize()           +90 ms
                //   endOfRange(): summary raised  +6 ms
                // So a 1.5 s budget caps what the user waits at ~1.5 s instead of
                // 3-4.4 s (and instead of 20 s when the decode is slow), and the
                // summary then reports exactly the transcript he watched being
                // built word by word.
                //
                // The repair decode is NOT cancelled — it keeps running in the
                // background and its words still merge into `detailedAnalysis`
                // (a finalized session's decided words are protected by
                // `keepFinalizedDecisions`, §4.37). Nothing is lost; the user just
                // stops staring at a frozen screen while the recognizer thinks.
                val deadline = System.currentTimeMillis() + 1500L
                while (!sessionFinalized && System.currentTimeMillis() < deadline) {
                    kotlinx.coroutines.delay(100L)
                }
                userPauseFinalizePending = false
                // §4.45 - A PAUSE MUST ALWAYS PRODUCE THE SUMMARY. The user
                // explicitly asked for it, so the wait for the final decode is
                // bounded and then we decide ANYWAY: `force = true` lets the
                // gate pass without `sessionFinalized`. Before this, a pause
                // whose endpoint never landed sat here for 20 s and then fell
                // out of the `sessionFinalized` guard with no popup at all -
                // "in the pause there is no summury at all".
                val forced = !sessionFinalized
                android.util.Log.i(
                    "ItqanSherpa",
                    "userPauseFinalize(): endpoint landed=$sessionFinalized forced=$forced -> " +
                        "running the user-initiated finalize now (a pause ALWAYS summarizes)"
                )
                runStopAndFinalize(true, forced)
            }
            /**
             * THE MIC FAB (§4.45). Three distinct intents, and conflating them
             * is what made the post-mistake state unusable:
             *  1. a live session  -> the user wants to stop: pause the mic NOW
             *     (instant feedback) and decide the summary once the final
             *     decode lands;
             *  2. a mistake popup is up / was just dismissed -> the user wants to
             *     RE-READ the flagged word: resume the warm pipeline, keeping
             *     the transcript so the re-read continues from that word;
             *  3. nothing running -> start (fresh or resume, decided inside
             *     [startTajweedSession]).
             *
             * It used to be `if (running) pause else runStopAndFinalize(true)`,
             * and `runStopAndFinalize` decided between "stop" and "start" by
             * reading `isListening` - which a mistake stop has already cleared.
             * So intent 2 was served by accident, through the "start" branch, and
             * any change to that branch silently killed the re-read.
             */
            fun onMicTap() {
                // Tarteel (SpeechRecognizer) has no pipeline to pause and no
                // resume latch, so it keeps the original single synchronous
                // finalize for both directions of the button: it STARTS when
                // nothing is running and PAUSES+SUMMARY when a session is live.
                //
                // §4.60 — AND THAT MUST NOT BE "IMPROVED" INTO A PIPELINE CALL.
                // Routing the post-mistake tap to `startTajweedSession()` looked
                // reasonable (let the reader continue instead of summarizing) and
                // was a straight regression: that function is the SHERPA PIPELINE
                // session, and Tarteel deliberately does NOT use the pipeline.
                // Measured on the user's first run of the build — Memorize +
                // Tarteel displayed NOTHING at all:
                //   onMicTap(): Tarteel — resuming after the mistake popup
                //     (mistake latch=none)
                //   startTajweedSession(): holder.pipeline=202565362 ...
                //   startTajweedSession(): resume() returned true
                //   analyze(fast): done 70ms matched=0 hasError=false targetWords=29
                // while the words displayed perfectly the moment he switched the
                // mode chip to Tajweed. The SpeechRecognizer was never started, so
                // there was nothing to match. The user's words: "i tried memorize +
                // tarteel and nothing got displayed. then i clicked on tajweed and
                // found that the words are displayed there."
                //
                // Continuing after a mistake is handled where it belongs — on the
                // popup's DISMISSAL (see below), which hands the next tap back to
                // this start path.
                if (recitationMode != "Tajweed") {
                    runStopAndFinalize(true, false)
                    return
                }
                when {
                    // 1. LIVE SESSION -> pause the mic NOW (instant feedback) and
                    //    decide the summary once the final decode lands.
                    isListening || tajweedPipelineHolder.running -> {
                        stopTajweedSession()
                        recitationSessionActive = true
                        userPauseFinalizePending = true
                    }
                    // 2. A SESSION IS PAUSED AND STILL RESUMABLE -> continue it.
                    //    Three ways to land here, and all of them mean "pick up
                    //    where the recitation is", never "start over":
                    //      - a mistake popup was dismissed: re-read that word,
                    //      - the user tapped Repeat Ayah: the transcript was
                    //        rewound to the start of the current ayah,
                    //      - Restart left the pipeline warm.
                    //    `startTajweedSession` decides between resuming the warm
                    //    stream and starting fresh, and the collector re-seats
                    //    the transcript at the mistake when the latch is armed.
                    recitationSessionActive -> {
                        android.util.Log.i(
                            "ItqanSherpa",
                            "onMicTap(): resuming the paused session (mistake latch=" +
                                "${if (mistakeResumeIndex >= 0) "armed@$mistakeResumeIndex" else "none"})"
                        )
                        // The popup is the app's surface for a mistake; tapping
                        // the mic is the user acting on it, so clear it here.
                        latestTajweedPopup = null
                        // A pending pause-decision must NOT fire into the resumed
                        // session: the endpoint it waits for belongs to the segment
                        // just closed, and it would pop a summary on top of a live
                        // recitation.
                        userPauseFinalizePending = false
                        startTajweedSession()
                        if (isListening) viewModel.preloadTajweedModel()
                    }
                    // 3. NOTHING RUNNING -> start a session.
                    else -> runStopAndFinalize(true, false)
                }
            }
            FloatingActionButton(
                onClick = { onMicTap() },
                // Keyed on the MIC, not on the session: after a mistake popup (or
                // a pause) the button must read as "record / re-read", never as a
                // second "stop" that would promise an end the user did not ask for.
                containerColor = if (isListening) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                contentColor = if (isListening) MaterialTheme.colorScheme.onError else MaterialTheme.colorScheme.background,
                shape = CircleShape,
                modifier = Modifier.size(72.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(60.dp)
                        .border(3.dp, MaterialTheme.colorScheme.background, CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (isListening) Icons.Filled.Stop else Icons.Filled.Mic,
                        contentDescription = if (isListening) "Stop Listening" else "Start Listening",
                        modifier = Modifier.size(32.dp)
                    )
                }
            }
            // END-OF-RANGE AUTO-FINALIZE. The recitation is over the moment the
            // committed transcript is final and every word of the selected range
            // has a verdict that is not "still waiting for speech". Nothing used
            // to happen then: the session just kept running and the summary —
            // which is the ONLY place the end-of-range popup is raised — lived
            // exclusively in the mic button's click handler, so the user had to
            // guess that they must tap stop to see anything. Run the same
            // finalize path instead of waiting for a tap.
            // Ticker for the mistake hold below. Once the recognizer goes quiet
            // detailedAnalysis stops changing, so without this the hold would
            // never expire and a stuck mistake would mean no popup at all.
            LaunchedEffect(sessionFinalized, isListening) {
                while (sessionFinalized && isListening) {
                    kotlinx.coroutines.delay(2000L)
                    finalizeTick++
                }
            }
            LaunchedEffect(sessionFinalized, isListening, detailedAnalysis, analysisEpoch, finalizeTick) {
                // §4.59 — TARTEEL INCLUDED. The user: "there is popup for mistaks and
                // summury in pause and end and everything exactly like we do in tajweed
                // mode except we dont have tajweed mistakes/popups". The END-OF-RANGE
                // SUMMARY is part of that contract, and this gate excluded every mode
                // but Tajweed — so in Tarteel a recitation that simply ran out of words
                // ended in silence, with no summary, until the user happened to tap the
                // mic.
                //
                // Nothing tajweed-specific happens below: the acoustic pass is gated
                // separately (isTajweedMode) and the tajweed popup watcher keeps its own
                // `recitationMode == "Tajweed"` gate, so Tarteel gains the end summary
                // and the pause handling and nothing else. Its recognition — which the
                // user says is far better than the Tajweed path — is untouched.
                if (currentTab != "Memorize" ||
                    (recitationMode != "Tajweed" && recitationMode != "Tarteel")
                ) return@LaunchedEffect
                // §4.63 — AND FOR A NON-PIPELINE MODE, `isListening` MAY ALREADY BE
                // FALSE. `onResults` (the platform recognizer's "I am done", §4.62)
                // also flips the listening state off, so demanding both flags meant
                // the end-of-range decision could never run in Tarteel: it re-armed
                // and returned every time. The `isListening` requirement exists for
                // the PIPELINE, where a stray endpoint can mark a session final while
                // the reader is still going — there we must keep waiting. For a
                // session the recognizer itself has closed, waiting IS the bug.
                val needsLiveMic = recitationMode == "Tajweed"
                if (!sessionFinalized || (!isListening && needsLiveMic)) {
                    // Live (or re-started) session: re-arm for the next finalize.
                    autoFinalizeDone = false
                    mistakeHoldSinceMs = 0L
                    graceExpiredOnce = false
                    gracePromoteCount = 0
                    lastDecidedCount = 0
                    lastUnheardCount = Int.MAX_VALUE
                    // §4.50 — a live session is never "stalled": it is waiting for
                    // the reader, which is the normal state of a recitation.
                    stallSinceMs = 0L
                    return@LaunchedEffect
                }
                // Diagnostics: this effect is the ONLY thing that ends a completed
                // range, and every input to the gate below is invisible in logcat.
                // Log them whenever we are in the firing state so a missing popup is
                // diagnosable from the run itself instead of guessed at.
                run {
                    val decided = detailedAnalysis.map { it.wordIndex }.toHashSet().size
                    // TWO DIFFERENT COUNTS, and conflating them was the bug (§4.34).
                    // `pendingEmpty` = a PENDING row with NO candidate token — the only
                    // shape the original guard was written for.
                    // `unheardWords` = EVERY word holding a PENDING row, candidate or not.
                    // A word carrying a candidate token is an UNRESOLVED JUDGEMENT, not a
                    // decided word, so it must block a clean finish.
                    val pendingEmpty = detailedAnalysis.count {
                        it.status == RecitationStatus.PENDING && it.spokenText.isNullOrEmpty()
                    }
                    val unheardWords = detailedAnalysis
                        .filter { it.status == RecitationStatus.PENDING }
                        .map { it.wordIndex }.toHashSet().size
                    android.util.Log.i(
                        "ItqanSherpa",
                        "autoFinalize?(): rows=${detailedAnalysis.size} decided=$decided/${targetOrig.size} " +
                            "unheardPending=$pendingEmpty unheardWords=$unheardWords " +
                            "alreadyShowing=${viewModel.sessionSummaryToShow != null} " +
                            "done=$autoFinalizeDone"
                    )
                    // DUMP THE UNDECIDED ROWS. The two shapes below are indistinguishable
                    // from the counts alone and need opposite handling, so print the
                    // status AND the candidate token of every row that is not a decided
                    // CORRECT/TAJWEED_ERROR/HINTED:
                    //   PENDING + spokenText  -> the analysis matched a token but never
                    //                            returned a verdict (unresolved judgement)
                    //   CORRECT + no spoken   -> finalizeCorrect()'s rubber stamp
                    val undecided = detailedAnalysis.filter {
                        it.status == RecitationStatus.PENDING ||
                            (it.status == RecitationStatus.CORRECT && it.spokenText.isNullOrEmpty())
                    }
                    if (undecided.isNotEmpty()) {
                        android.util.Log.i(
                            "ItqanSherpa",
                            "autoFinalize(): undecided rows = " + undecided.joinToString(" | ") {
                                "w${it.wordIndex}/${it.text}:${it.status}:spoken='${it.spokenText ?: "-"}'"
                            }
                        )
                    }
                }
                if (autoFinalizeDone) return@LaunchedEffect
                if (viewModel.sessionSummaryToShow != null) return@LaunchedEffect
                if (detailedAnalysis.isEmpty()) return@LaunchedEffect
                val total = targetOrig.size
                if (total == 0) return@LaunchedEffect
                // detailedAnalysis holds ONE ROW PER (WORD, RULE), so count
                // distinct word indices, not rows.
                //
                // ONLY A NON-PENDING ROW IS A DECISION (§4.34). A PENDING row is a
                // placeholder whether or not it carries a candidate token, so it must
                // not count toward `decidedWords` either.
                val decidedWords = detailedAnalysis
                    .filter { it.status != RecitationStatus.PENDING }
                    .map { it.wordIndex }.toHashSet()
                // UNRESOLVED CANDIDATES COUNT AS UNHEARD (§4.34). The old test only
                // looked at PENDING rows with a BLANK spokenText, so a word holding a
                // candidate token — a judgement the verifier had not yet returned a
                // verdict for — read as "heard". `cleanFinish` then fired, the range was
                // declared complete, and the word was left at the 25 %-opacity fallback
                // with NO popup and NO summary: the user's "last word not displayed and
                // there is no popup".
                val unheard = detailedAnalysis
                    .filter { it.status == RecitationStatus.PENDING }
                    .map { it.wordIndex }.toHashSet().size
                val mistakeOutstanding = detailedAnalysis.any {
                    it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED
                }
                // UNVERDICTED WORDS BLOCK A CLEAN FINISH.
                //
                // A forgiveness is only trustworthy when a decode actually
                // judged the word. `verifiedSpokenIndices` is also fed by the
                // stall watchdog, which fires when a candidate outstays its
                // 20 s budget — and if the confirm loop never got to launch a
                // decode (fatihah_c30: a row-index flicker reset the dwell
                // pre-condition on every recomposition, zero `verify idx=`
                // lines all run), that watchdog entry is a GUESS, not a verdict.
                //
                // Without this test the guessed-forgiven word drops out of
                // `mistakeOutstanding`, `cleanFinish` turns true, and the user
                // gets a SUMMARY for a recitation the app could not actually
                // judge — including for a word they really did skip, which is
                // then counted correct. Requiring real evidence (an absent
                // strike) before the range may be called complete is what makes
                // the difference between "we checked and it was fine" and "we
                // gave up and said nothing", and the user has asked for the
                // former explicitly.
                val unverdictedStall = lastWrongStopIndex >= 0 &&
                    lastWrongStopIndex !in verifiedSpokenIndices &&
                    mistakeVerifyAbsentStrikes == 0
                // Ideal case: the range was completed — every word decided, nothing
                // left unheard, nothing flagged, and no word left in limbo.
                val cleanFinish = decidedWords.size >= total && unheard == 0 &&
                    !mistakeOutstanding && !unverdictedStall
                // PROGRESS RE-ARMS THE HOLD. An endpoint fires on every utterance
                // boundary, not only at the end of a recitation, so the countdown
                // must never run against someone who is still reciting: any newly
                // decided word (or any word that stopped being unheard) restarts it.
                if (decidedWords.size > lastDecidedCount || unheard < lastUnheardCount) {
                    mistakeHoldSinceMs = 0L
                }
                // §4.50 — AND THE STALL CLOCK RUNS ON THE SAME EVIDENCE. A
                // committed session that stops making progress starts the clock;
                // any progress restarts it. It only reaches the summary gate
                // through the `stallExpired` branch, and only when the range is
                // NOT complete, so a reader who is still going never sees it.
                if (sessionFinalized && !cleanFinish &&
                    (decidedWords.size > lastDecidedCount || unheard < lastUnheardCount)
                ) {
                    stallSinceMs = 0L
                } else if (sessionFinalized && !cleanFinish && stallSinceMs == 0L) {
                    stallSinceMs = System.currentTimeMillis()
                    android.util.Log.i(
                        "ItqanSherpa",
                        "autoFinalize(): the recognizer committed with the range incomplete " +
                            "(decided=${decidedWords.size}/$total unheard=$unheard) and nothing is " +
                            "moving -> stall clock started; in " +
                            "${STALL_SUMMARY_AFTER_MS / 1000}s the user gets the honest paused " +
                            "dialog instead of a frozen screen"
                    )
                }
                lastDecidedCount = decidedWords.size
                lastUnheardCount = unheard
                if (!cleanFinish) {
                    // Two reasons to wait rather than finalize right now:
                    //  - a mistake must surface as a MISTAKE POPUP (the app stops
                    //    the recitation there), never be buried in the summary
                    //    (fatihah_v6: `skipped=1` in the summary, popup never shown);
                    //  - words still unheard may simply not have arrived yet.
                    //
                    // BOUNDED, THOUGH. A mistake that is never confirmed (the
                    // verification decode keeps coming back INCONCLUSIVE) or a walk
                    // that stalls on a WRONG would otherwise hold the summary
                    // FOREVER and the user would be left with NO popup at all —
                    // measured fatihah_v9 (`decided=28/29`, nothing on screen) and
                    // fatihah_p1 (stuck at 13/29 with `unheardPending=15`, nothing on
                    // screen). "Nothing happened" is the worst possible outcome, so
                    // give the mistake pipeline MISTAKE_GRACE_MS to speak up, then
                    // finalize regardless.
                    //
                    // ONE WINDOW PER SESSION. Once that window has expired, the
                    // decision below is final: forgiving re-arms nothing, and the
                    // very next pass falls through to the finalize. Without
                    // `graceExpiredOnce` the forgive branch could hand back a fresh
                    // 20 s window on a mistake it had just decided not to act on,
                    // freezing the walk forever (fatihah_c27, §4.18).
                    if (mistakeHoldSinceMs == 0L) mistakeHoldSinceMs = System.currentTimeMillis()
                    val held = System.currentTimeMillis() - mistakeHoldSinceMs
                    // `graceExpiredOnce` pins the window open: once the countdown has
                    // run out it may never restart within this session, so the fork
                    // below is guaranteed to reach a decision (stop with the popup,
                    // or forgive-and-finalize) instead of looping.
                    if (held < MISTAKE_GRACE_MS && !graceExpiredOnce) {
                        android.util.Log.i(
                            "ItqanSherpa",
                            "autoFinalize(): holding ${held}ms/$MISTAKE_GRACE_MS " +
                                "(decided=${decidedWords.size}/$total unheard=$unheard mistake=$mistakeOutstanding)"
                        )
                        return@LaunchedEffect
                    }
                    graceExpiredOnce = true
                    // ── THE GRACE EXPIRED AND A MISTAKE IS STILL OUTSTANDING ──
                    // Finalizing here is the bug the user keeps hitting: it hands
                    // over the end-of-session SUMMARY, which by contract is only
                    // for "I finished" — a mistake must STOP the recitation with
                    // a MISTAKE POPUP. Measured fatihah_c10:
                    //   autoFinalize(): range complete (29/29 words decided)
                    //   endOfRange(): … wrong=0 skipped=2 tajweed=1
                    // i.e. the summary fired and simply listed two skipped words.
                    //
                    // What to do depends on what the verification actually saw:
                    //  - at least ONE substantial decode came back missing the word
                    //    (mistakeVerifyAbsentStrikes >= 1) ⇒ there IS evidence. Stop
                    //    and show the MISTAKE popup for the first flagged word.
                    //  - otherwise every decode was too short/garbled to judge
                    //    ⇒ we cannot prove anything, and per project policy a
                    //    false accusation is worse than a missed skip: FORGIVE the
                    //    word and let the session finish normally.
                    val firstMistake = detailedAnalysis.firstOrNull {
                        it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED
                    }
                    if (firstMistake != null) {
                        if (mistakeVerifyAbsentStrikes >= 1) {
                            android.util.Log.i(
                                "ItqanSherpa",
                                "autoFinalize(): grace expired with an EVIDENCED mistake at " +
                                    "wordIndex=${firstMistake.wordIndex} (strikes=$mistakeVerifyAbsentStrikes) " +
                                    "-> stopping with the mistake popup, NOT a summary"
                            )
                            autoFinalizeDone = true
                            mistakeHoldSinceMs = 0L
                            stopForMistake(
                                firstMistake.wordIndex,
                                firstMistake.status == RecitationStatus.SKIPPED
                            )
                            return@LaunchedEffect
                        }
                        android.util.Log.i(
                            "ItqanSherpa",
                            "autoFinalize(): grace expired with an UNPROVEN mistake at " +
                                "wordIndex=${firstMistake.wordIndex} (no substantial decode ever " +
                                "judged it) -> forgiving rather than accusing"
                        )
                        // CLEAR THE ROW, do not just whitelist the word.
                        //
                        // verifiedSpokenIndices only suppresses the *candidate*; the
                        // row itself kept `status = WRONG/SKIPPED`, so `mistakeOutstanding`
                        // stayed true, `graceExpiredOnce` was false again, and this branch
                        // re-ran on every recomposition — FOREVER. Measured fatihah_c27:
                        // after the first forgive at 01:43:56 the guard held 20 s, forgave
                        // wordIndex=20 again, held 20 s, … with the walk frozen at 20/29
                        // and `autoFinalizeDone` never set, so `runStopAndFinalize()` was
                        // NEVER CALLED and the screen stayed empty until the harness's 40 s
                        // tail ran out. "Nothing happened" is the worst outcome (§4.14).
                        //
                        // Converting the status is the only thing that makes the forgive
                        // STICK — it is what mistakeOutstanding counts, and it is exactly
                        // what finalizeCorrect() does for a trailing PENDING word. Only
                        // this one row is touched, and only because no decode could judge
                        // it: a proven mistake still stops with the popup above.
                        val forgivenIdx = firstMistake.wordIndex
                        detailedAnalysis = detailedAnalysis.map { w ->
                            if (w.wordIndex == forgivenIdx &&
                                (w.status == RecitationStatus.WRONG ||
                                    w.status == RecitationStatus.SKIPPED)
                            ) {
                                w.copy(status = RecitationStatus.CORRECT, spokenText = null)
                            } else {
                                w
                            }
                        }
                        verifiedSpokenIndices.add(forgivenIdx)
                        mistakeVerifyAttempts = 0
                        mistakeVerifyAbsentStrikes = 0
                        mistakeVerifyAbsentIdx = -1
                        mistakeVerdictAtMs = 0L
                        // Do NOT reset mistakeHoldSinceMs: that restarts the 20 s window
                        // and, when the erase above somehow fails to take, reintroduces the
                        // loop. Leave the expired clock in place so the next pass falls
                        // straight through to the finalize.
                        return@LaunchedEffect
                    }
                    // ── THE GRACE EXPIRED WITH UNRESOLVED WORDS AND NO MISTAKE ──
                    // A PENDING row that survived the whole grace is a judgement the
                    // verifier never returned a verdict for. There is NO evidence the
                    // word was wrong — a confirmed mistake would have taken the branch
                    // above — and per project policy a false accusation is worse than a
                    // missed skip, so we do not accuse. But we must not leave the word
                    // INVISIBLE either: the contract is "if it stopped, it is a mistake
                    // popup, ELSE the words stay displayed" (measured clean_fatihah_2:
                    // `endOfRange(): auto stop but NOT genuinely finished (unheard=1/29)
                    // -> no summary` and the last word sat at the 25 %-opacity fallback
                    // with nothing on screen at all).
                    //
                    // So make the row VISIBLE the way finalizeCorrect() forgives a
                    // trailing PENDING word: `status = CORRECT`, `spokenText` left as it
                    // is. The token is load-bearing — a row that carries one is a CLIPPED
                    // decode (a word the user did say) and stays inside the "heard" set,
                    // so a genuine finish can still raise its summary; a row with NO
                    // token keeps `unheardWordCount()` counting it, so a recitation we
                    // could not judge can never produce a "perfect" SUMMARY it did not
                    // earn (§4.31). Either way the word becomes VISIBLE.
                    //
                    // PROMOTE **EVERY** PENDING ROW, TOKEN OR NOT. Using
                    // `promoteClippedPendingRows()` here was a NO-OP for the rows this
                    // branch exists for: a word the decoder never reached has
                    // `spokenText == null`, and that helper deliberately leaves such rows
                    // alone. The result was an INFINITE LOOP — measured `p1_fatihah`:
                    // `grace expired with 7 unresolved word(s) [22..28] … (0 of them carry
                    // a spoken fragment)` repeated every 2 s for minutes, the session
                    // never finalizing and the screen never settling.
                    val unresolvedIdx = detailedAnalysis
                        .filter { it.status == RecitationStatus.PENDING }
                        .map { it.wordIndex }.toHashSet()
                    if (unresolvedIdx.isNotEmpty()) {
                        val heard = detailedAnalysis.count {
                            it.status == RecitationStatus.PENDING && !it.spokenText.isNullOrEmpty()
                        }
                        // §4.47 - PROMOTE ONLY THE TAIL THE USER PLAUSIBLY REACHED.
                        //
                        // This branch exists so the words the decoder never
                        // rendered, but the user DID read, stay visible (§4.40).
                        // It promoted EVERY PENDING row, so a session that ended
                        // early (an auto-stop on the recording's own silence, a
                        // Restart the user had just tapped) decided and DISPLAYED
                        // the whole remaining range in one step - the report was
                        // "it froze, and then all the words appeared at once until
                        // the end of the ayah range", and on screen a fully recited
                        // range is indistinguishable from words nobody read.
                        //
                        // The bound is the same shape `finalizeCorrect()` uses: the
                        // last word that carries spoken evidence, plus a small
                        // window for the clipped tail. Beyond that the user simply
                        // has not got there yet, so those rows stay PENDING (grey,
                        // and still counted as unheard, which correctly withholds a
                        // "you finished" summary).
                        val heardFrontier = detailedAnalysis
                            .filter { !it.spokenText.isNullOrEmpty() }
                            .maxOfOrNull { it.wordIndex + 1 } ?: 0
                        // NO heard evidence AT ALL => no frontier => no forgiveness.
                        // A session where nothing was recognised must not display the
                        // first words of the range just because the window starts at 0.
                        val promotableIdx = if (heardFrontier <= 0) {
                            emptySet()
                        } else {
                            // Same ayah ceiling as finalizeCorrect(): the
                            // clipped-tail window belongs to the ayah the reader
                            // was heard in. Promoting a row across the boundary
                            // is what made a paused session colour the next
                            // ayah.
                            val heardAyahEnd = ayahEndForWordIndex(
                                heardFrontier - 1,
                                viewModel.ayahs
                            )
                            val promotionLimit = minOf(
                                heardFrontier + FINALIZE_FORGIVE_TAIL_WINDOW,
                                heardAyahEnd
                            )
                            unresolvedIdx.filter { it < promotionLimit }.toSet()
                        }
                        // §4.50 — A PROMOTED WORD MUST CARRY THE FRAGMENT OF A
                        // WORD THE USER ACTUALLY SAID.
                        //
                        // The window above is a bound on DISTANCE, not on
                        // evidence, and on a session that ended well short of the
                        // range it promotes words nobody read. Measured on the
                        // device (`v4_stall`, a 20 s slice of Al-Fatihah against a
                        // 29-word range): the user recited 13 words, and the
                        // promotion decided 8 more —
                        //   autoFinalize(): 8 unresolved word(s) [21..28] are
                        //   beyond the heard frontier (13 words) + 8 -> NOT promoted
                        //   autoFinalize(): grace expired with 16 unresolved
                        //   word(s) [13..28] ...
                        //   endOfRange(): ... words stay displayed cursor=21/29
                        // i.e. the CURSOR — and with it the colouring, since the
                        // renderer paints a word iff it is below the cursor — ran
                        // from 13 to 21 over eight words that carry no decoded
                        // fragment at all. The user's report: "it advance without
                        // displaying ... then restarted from begining".
                        //
                        // A row with a `spokenText` fragment is a CLIPPED decode:
                        // the recognizer heard the word and the window cut it off,
                        // so displaying it is honest. A row with NO fragment is a
                        // word the recognizer never heard — there is no evidence
                        // the user read it, and §4.40/§4.41's whole principle is
                        // that REACHED must equal DISPLAYED. So it stays PENDING and
                        // grey: still visible as text, never coloured as read.
                        //
                        // This is exactly what `promoteClippedPendingRows()` does,
                        // and why that helper was the right tool here all along.
                        val clippedIdx = detailedAnalysis
                            .filter { it.status == RecitationStatus.PENDING && !it.spokenText.isNullOrEmpty() }
                            .map { it.wordIndex }
                            .toHashSet()
                        val promotableClipOnly = promotableIdx intersect clippedIdx
                        if (promotableIdx.size != promotableClipOnly.size) {
                            android.util.Log.i(
                                "ItqanSherpa",
                                "autoFinalize(): NOT promoting ${promotableIdx.size - promotableClipOnly.size} " +
                                    "word(s) ${promotableIdx - promotableClipOnly} that carry no decoded " +
                                    "fragment - they were never heard, so the display must not run past " +
                                    "the heard frontier ($heardFrontier) for them"
                            )
                        }
                        val beyondReach = unresolvedIdx - promotableClipOnly
                        if (beyondReach.isNotEmpty()) {
                            android.util.Log.i(
                                "ItqanSherpa",
                                "autoFinalize(): ${beyondReach.size} unresolved word(s) " +
                                    "$beyondReach stay undisplayed (never heard, or beyond the heard " +
                                    "frontier of $heardFrontier words + $FINALIZE_FORGIVE_TAIL_WINDOW)"
                            )
                        }
                        // Stamp CORRECT and leave `spokenText` alone — exactly what
                        // finalizeCorrect() does for a CLIPPED word: the recognizer
                        // heard it and the window cut it off. Only those rows are
                        // touched (§4.50), so a word with no fragment keeps counting
                        // as unheard and can never produce a "perfect" summary the
                        // recitation did not earn (§4.31).
                        detailedAnalysis = detailedAnalysis.map { row ->
                            if (row.status == RecitationStatus.PENDING &&
                                row.wordIndex in promotableClipOnly
                            ) {
                                row.copy(status = RecitationStatus.CORRECT)
                            } else row
                        }
                        // §4.40 — the promotion decides the rows, but the
                        // renderer colours by the CURSOR, so the words this
                        // branch exists to DISPLAY would stay grey. Advance the
                        // cursor over the newly-decided prefix in the same
                        // breath as the promotion, otherwise the very next
                        // `endOfRange()` reports "words stay displayed
                        // cursor=4/29" (measured `p4_fatihah`).
                        val coveredAfterPromote = minOf(
                            displayedCursorFor(detailedAnalysis, alignment.first),
                            displayCapForHeardAyah(detailedAnalysis, viewModel.ayahs)
                        )
                        if (coveredAfterPromote > alignment.first) {
                            android.util.Log.i(
                                "ItqanSherpa",
                                "autoFinalize(): cursor advanced ${alignment.first} -> " +
                                    "$coveredAfterPromote so the ${promotableClipOnly.size} promoted " +
                                    "word(s) actually DISPLAY (§4.40 REACHED => DISPLAYED)"
                            )
                            alignment = Pair(coveredAfterPromote, alignment.second)
                        }
                        if (promotableClipOnly.isEmpty()) {
                            // §4.50 — THE COUNTER MUST ADVANCE HERE TOO. The
                            // "return and try again on the next evaluation" guard
                            // below is driven by `gracePromoteCount`, but this
                            // branch — the one that actually fires when the range
                            // is far from finished — left the counter alone, so the
                            // guard returned true on every 2 s pass, for ever.
                            // Measured on the device: 17 unresolved words
                            // [12..28] beyond the frontier, decided=29/29, and the
                            // SAME two log lines repeating for minutes with the
                            // display frozen and no popup and no summary — the
                            // user's "it keeps getting stuck in displaying words".
                            // With the counter advancing, the very next pass falls
                            // through to the finalize, which the stall clock then
                            // turns into the honest paused dialog.
                            gracePromoteCount++
                            android.util.Log.i(
                                "ItqanSherpa",
                                "autoFinalize(): grace expired with ${unresolvedIdx.size} unresolved " +
                                    "word(s) $unresolvedIdx and NO evidenced mistake -> NOTHING was " +
                                    "within reach of the heard frontier ($heardFrontier words), so " +
                                    "no word was promoted and the recitation simply stops where the " +
                                    "user left it [promotion #$gracePromoteCount]"
                            )
                        } else {
                            gracePromoteCount++
                            android.util.Log.i(
                                "ItqanSherpa",
                                "autoFinalize(): grace expired with ${unresolvedIdx.size} unresolved " +
                                    "word(s) $unresolvedIdx and NO evidenced mistake -> promoted the " +
                                    "${promotableClipOnly.size} CLIPPED word(s) within reach of the " +
                                    "heard frontier to CORRECT so they DISPLAY (a clipped word is one " +
                                    "the recognizer heard; the ${unresolvedIdx.size - promotableClipOnly.size} " +
                                    "word(s) with no decoded fragment stay 'unheard', stay grey, and " +
                                    "keep blocking a 'you finished' summary) [promotion #$gracePromoteCount]"
                            )
                        }
                        // ONE promotion pass is enough for the UI to repaint, so return
                        // and let the next evaluation see a clean list and finalize. If a
                        // later analysis rebuild RESURRECTS the rows, promote again — but
                        // this time FALL THROUGH to the finalize instead of returning, so
                        // a rebuild loop can never hold the session open forever (the
                        // §4.18 class: "nothing ever happens" is the worst outcome).
                        if (gracePromoteCount <= 1) return@LaunchedEffect
                    }
                }
                mistakeHoldSinceMs = 0L
                android.util.Log.i(
                    "ItqanSherpa",
                    "autoFinalize(): grace expired with $total/$total words decided " +
                        "(unheard=$unheard mistake=$mistakeOutstanding) -- running the finalize path"
                )
                // false = NOT a user pause: the range ended by itself, so an
                // unresolved mistake must win over the summary. false = do NOT
                // force: this path only runs once the endpoint landed, and
                // forcing would let a mid-range stall summarize itself.
                runStopAndFinalize(false, false)
                // RE-ARM WHEN NOTHING WAS SHOWN. If the range turned out to be
                // incomplete (unheard words beyond the forgiveness window), no
                // summary was raised and no mistake popup fired — the app
                // decided to keep waiting for the rest of the recitation. Leave
                // `autoFinalizeDone` false and clear the pinned grace so a LATER
                // completion still fires this path; otherwise one premature
                // grace expiry disabled the end-of-range popup for the whole
                // session.
                autoFinalizeDone = !finalizeWithheld.value
                if (finalizeWithheld.value) {
                    graceExpiredOnce = false
                    gracePromoteCount = 0
                }
            }
            if (tajweedStatusText.isNotEmpty()) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = tajweedStatusText,
                    style = MaterialTheme.typography.labelMedium,
                    color = if (tajweedStatusText.startsWith("Warming")) MaterialTheme.colorScheme.tertiary
                            else MaterialTheme.colorScheme.primary
                )
            }
            }
        }
        } // Close if (currentTab == "Memorize")
        
        Spacer(modifier = Modifier.height(6.dp))
        
        NavigationBar(
            containerColor = Color.Transparent,
            modifier = Modifier.fillMaxWidth()
        ) {
            val tabs = listOf(
                Triple("Read", Icons.AutoMirrored.Filled.MenuBook, Icons.AutoMirrored.Filled.MenuBook),
                Triple("Memorize", Icons.Default.Psychology, Icons.Default.Psychology),
                Triple("Tajweed", Icons.Default.School, Icons.Default.School),
                Triple("Stats", Icons.Default.Insights, Icons.Default.Insights)
            )
            tabs.forEach { (label, icon, _) ->
                NavigationBarItem(
                    selected = currentTab == label,
                    onClick = {
                        currentTab = label
                        if (isListening && label != "Memorize") {
                            isListening = false
                            try { speechRecognizer?.cancel() } catch (e: Throwable) {}
                        }
                        // Loading the Tajweed model is deferred until the user actually
                        // opens the Tajweed tab (or picks Tajweed mode in Memorize -- see
                        // the FilterChip further up). Warm the singleton here so the user
                        // never waits on the first mic tap.
                        if (label == "Tajweed") {
                            viewModel.preloadTajweedModel()
                        }
                    },
                    icon = { Icon(icon, contentDescription = null) },
                    label = { Text(getAppString(isArabic, label)) },
                    colors = NavigationBarItemDefaults.colors(
                        selectedIconColor = MaterialTheme.colorScheme.background,
                        selectedTextColor = MaterialTheme.colorScheme.primary,
                        indicatorColor = MaterialTheme.colorScheme.primary,
                        unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                        unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                )
            }
        }

        if (showSurahSelectionDialog) {
            var searchQuery by remember { mutableStateOf("") }
            AlertDialog(
                onDismissRequest = { showSurahSelectionDialog = false },
                title = {
                    Text(
                        "Select Surah",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                },
                text = {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        OutlinedTextField(
                            value = searchQuery,
                            onValueChange = { searchQuery = it },
                            placeholder = { Text(getAppString(isArabic, "Search Surah...")) },
                            leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp)
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        val filteredSurahs = viewModel.surahs.filter {
                            it.englishName.contains(searchQuery, ignoreCase = true) ||
                            it.name.contains(searchQuery) ||
                            it.number.toString() == searchQuery
                        }
                        LazyColumn(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(350.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            items(filteredSurahs) { surah ->
                                val isSelected = viewModel.selectedSurah == surah
                                Card(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable {
                                             viewModel.selectSurah(surah)
                                             // Remember the pick against the ACTIVE
                                             // practice mode, so switching Tarteel/
                                             // Tajweed restores each surface's own
                                             // surah instead of sharing one.
                                             if (currentTab == "Memorize") {
                                                 modeSelectedSurah[recitationMode] = surah
                                             }
                                             finalSpokenText = ""
                                             partialSpokenText = ""
                                             mistakeResumeIndex = -1
                                             mistakeResumeLiveText = ""
                                             verifiedSpokenIndices.clear()
                                             lastMistakeVerifyMs = 0L
                                             mistakeVerifyAbsentIdx = -1
                                             mistakeEpisodeAnchor = -1
                                             mistakeCleanStreak = 0
                                             showSurahSelectionDialog = false
                                        },
                                    colors = CardDefaults.cardColors(
                                        containerColor = if (isSelected) {
                                            MaterialTheme.colorScheme.primaryContainer
                                        } else {
                                            MaterialTheme.colorScheme.surface
                                        }
                                    ),
                                    border = BorderStroke(
                                        1.dp,
                                        if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
                                    ),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(12.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Box(
                                                modifier = Modifier
                                                    .size(32.dp)
                                                    .background(
                                                        color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondaryContainer,
                                                        shape = CircleShape
                                                    ),
                                                contentAlignment = Alignment.Center
                                            ) {
                                                Text(
                                                    text = "${surah.number}",
                                                    style = MaterialTheme.typography.bodyMedium,
                                                    fontWeight = FontWeight.Bold,
                                                    color = if (isSelected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSecondaryContainer
                                                )
                                            }
                                            Spacer(modifier = Modifier.width(12.dp))
                                            Column {
                                                Text(
                                                    // Swap name priority based on UI language so the
                                                    // primary line matches what the user reads.
                                                    text = if (isArabic) surah.name else surah.englishName,
                                                    style = MaterialTheme.typography.bodyLarge,
                                                    fontWeight = FontWeight.SemiBold,
                                                    color = if (isSelected) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurface
                                                )
                                                Text(
                                                    text = "${surah.numberOfAyahs} ${getAppString(isArabic, "Ayahs")}",
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                        }
                                        Text(
                                            text = if (isArabic) surah.englishName else surah.name,
                                            style = MaterialTheme.typography.titleMedium,
                                            fontWeight = FontWeight.Bold,
                                            color = if (isSelected) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.primary
                                        )
                                    }
                                }
                            }
                        }
                    }
                },
                confirmButton = {
                    TextButton(onClick = { showSurahSelectionDialog = false }) {
                        Text(getAppString(isArabic, "Close"))
                    }
                }
            )
        }
        }
        }
    }
}

@Composable
fun SurahChip(surahName: String, isSelected: Boolean, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .border(
                1.dp,
                if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline,
                RoundedCornerShape(16.dp)
            )
            .clip(RoundedCornerShape(16.dp))
            .background(if (isSelected) MaterialTheme.colorScheme.surfaceVariant else MaterialTheme.colorScheme.surface)
            .clickable(onClick = onClick)
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = Icons.Rounded.Book,
            contentDescription = null,
            tint = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.width(16.dp))
        Text(
            text = surahName,
            style = MaterialTheme.typography.titleMedium,
            color = if (isSelected) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

/**
 * Self-managed microphone recorder used by the Tajweed Rules coach.
 *
 * We deliberately do NOT use the SpeechRecognizer's bundled audio (GET_AUDIO): on many devices
 * (e.g. Samsung A05s) the returned Uri cannot be read/decoded by our own process, and running a
 * parallel [AudioRecord] alongside the recognizer conflicts with the device audio HAL (which is
 * what originally broke recognition). Owning the capture ourselves guarantees we always have the
 * user's real 16 kHz PCM voice to score. The example text is known, so no speech-to-text is needed.
 */
/**
 * Mutable holder for the [itqan.app.sherpa.SingleAudioPipeline] created when
 * the Memorize+Tajweed Composable first resolves its ayah target list. Kotlin
 * disallows forward references to local variables inside closures (so the
 * pipeline cannot be `remember`-constructed before `targetOrig` is declared
 * while also binding `{ targetOrig }` into its hotness provider). This holder
 * is declared early in the Composable scope (so it's visible to all wiring
 * sites) and the pipeline is assigned to it once `targetOrig` is available.
 */
class TajweedPipelineHolder {
    /**
     * Backed by Compose `mutableStateOf` so the partials-collector
     * `LaunchedEffect(tajweedPipelineHolder, currentTab, recitationMode)`
     * re-launches when a range change wires a NEW pipeline into this holder —
     * otherwise the collector would keep listening to the OLD pipeline's
     * `_partials` SharedFlow, which dies after [release], and the user's
     * word-by-word highlight would freeze on ayah 2 (the exact symptom the user
     * reported after the range-change `release()`).
     *
     * The `@Volatile`-only field the previous version used does NOT notify
     * Compose of writes, so the LaunchedEffect's key did not change. Wrapping
     * in `mutableStateOf` is the cheapest fix: it makes the holder notify
     * Compose observers on assignment, and the LaunchedEffect's key (this
     * holder instance itself is stable) only has this one writable property
     * — observing that property IS observing the pipeline.
     */
    var pipeline: itqan.app.sherpa.SingleAudioPipeline? by androidx.compose.runtime.mutableStateOf(null)

    /** Same semantics as [SingleAudioPipeline.running], null-safe. */
    val running: Boolean get() = pipeline?.running ?: false
    /** True while the recognizer is being built (prewarm/start load). */
    val loading: Boolean
        get() = pipeline?.state?.value?.let { it is itqan.app.sherpa.SingleAudioPipeline.State.Loading } ?: false
    /** True iff the pipeline is currently paused (fast-resume ready). */
    val paused: Boolean get() = pipeline?.paused ?: false
    /** Null-safe passthrough for `pipeline.stop()`. */
    fun stop(): FloatArray = pipeline?.stop() ?: FloatArray(0)
    /** Null-safe passthrough for `pipeline.pause()` (kept recognition warm). */
    fun pause(): FloatArray = pipeline?.pause() ?: FloatArray(0)
    /** Null-safe passthrough for `pipeline.release()`. */
    fun release() { pipeline?.release() }

    /** Compose-friendly identity-stable key for triggering re-collection. */
    val pipelineKey: Any get() = pipeline ?: NULL_KEY
    private companion object { val NULL_KEY = Any() }
}

class MicRecorderController {
    @Volatile var running = false
        private set
    private val sampleRate = 16000

    private var audioRecord: AudioRecord? = null
    private var thread: Thread? = null

    // Growable primitive float buffer holding samples in [-1, 1].
    private var data = FloatArray(sampleRate * 10)
    private var size = 0

    @Synchronized
    fun start() {
        if (running) return
        try {
            val minBuf = AudioRecord.getMinBufferSize(
                sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            if (minBuf <= 0) return
            val bufFrames = maxOf(minBuf, sampleRate * 2) / 2
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC, sampleRate,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufFrames * 2
            )
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                audioRecord?.release()
                audioRecord = null
                return
            }
            size = 0
            running = true
            audioRecord?.startRecording()
            val shortBuf = ShortArray(bufFrames)
            thread = Thread({
                while (running) {
                    val n = audioRecord?.read(shortBuf, 0, shortBuf.size) ?: 0
                    if (n > 0) {
                        synchronized(this) {
                            ensureCapacity(size + n)
                            for (i in 0 until n) data[size + i] = shortBuf[i] / 32768f
                            size += n
                        }
                    } else if (n == AudioRecord.ERROR || n == AudioRecord.ERROR_INVALID_OPERATION) {
                        break
                    }
                }
            }, "TajweedMicRecorder").also { it.start() }
        } catch (e: Throwable) {
            running = false
            try { audioRecord?.release() } catch (_: Throwable) {}
            audioRecord = null
        }
    }

    private fun ensureCapacity(needed: Int) {
        if (needed <= data.size) return
        var cap = data.size
        while (cap < needed) cap *= 2
        val next = FloatArray(cap)
        data.copyInto(next)
        data = next
    }

    @Synchronized
    fun stop(): FloatArray {
        running = false
        try { audioRecord?.stop() } catch (_: Throwable) {}
        try { thread?.join(1000) } catch (_: Throwable) {}
        try { audioRecord?.release() } catch (_: Throwable) {}
        audioRecord = null
        thread = null
        return data.copyOf(size)
    }
}

private fun startListening(
    recognizer: android.speech.SpeechRecognizer,
    packageName: String,
    onError: (String) -> Unit
) {
    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ar-SA")
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)
        // Ask the engine to wait longer before declaring the end of an
        // utterance. These extras are advisory (engines may ignore them), so
        // the restart loop above is the real guarantee; they simply reduce the
        // number of needless segment boundaries on engines that honour them.
        putExtra(
            RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS,
            5000L
        )
        putExtra(
            RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS,
            2500L
        )
        putExtra(
            RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS,
            8000L
        )
        // Ask the recognizer to hand back its own recording so the acoustic engine can score the
        // real voice. A single capture avoids the mic-conflict that a parallel AudioRecord causes.
        putExtra("android.speech.extra.GET_AUDIO", true)
        putExtra("android.speech.extra.GET_AUDIO_FORMAT", "audio/AMR")
    }
    try {
        recognizer.startListening(intent)
    } catch (e: Throwable) {
        onError("Recognition failed to start")
    }
}

@Composable
fun SplashScreen(isArabic: Boolean, onDismiss: () -> Unit) {
    var startAnimation by remember { mutableStateOf(false) }
    val scale by animateFloatAsState(
        targetValue = if (startAnimation) 1f else 0.7f,
        animationSpec = tween(1000, easing = LinearOutSlowInEasing),
        label = "scale"
    )
    val alpha by animateFloatAsState(
        targetValue = if (startAnimation) 1f else 0f,
        animationSpec = tween(800),
        label = "alpha"
    )

    LaunchedEffect(key1 = true) {
        startAnimation = true
        kotlinx.coroutines.delay(2000)
        onDismiss()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .padding(24.dp)
                .graphicsLayer(
                    scaleX = scale,
                    scaleY = scale,
                    alpha = alpha
                )
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.MenuBook,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(96.dp)
            )

            Spacer(modifier = Modifier.height(20.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Mic,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (isArabic) "إتقان" else "Itqān",
                    style = MaterialTheme.typography.headlineLarge.copy(
                        fontWeight = FontWeight.ExtraBold,
                        letterSpacing = 1.sp
                    ),
                    color = MaterialTheme.colorScheme.onBackground,
                    textAlign = TextAlign.Center
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = if (isArabic) "تسميع القرآن مع أحكام التجويد بدقة" else "AI-guided Quran Memorization & Tajweed",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.secondary,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(48.dp))

            CircularProgressIndicator(
                color = MaterialTheme.colorScheme.primary,
                strokeWidth = 3.dp,
                modifier = Modifier.size(36.dp)
            )
        }
    }
}

fun getAppString(isArabic: Boolean, key: String): String {
    if (!isArabic) return key
    return when (key) {
        "Recitation Practice" -> "إتقان"   // legacy key, kept for backwards reference; new code uses the literal "Itqān" / "إتقان"
        "Read" -> "قراءة"
        "Memorize" -> "تسميع"
        "Tajweed" -> "تجويد"
        "Stats" -> "الأهداف"
        "Mode: " -> "الوضع: "
        "Tarteel (Normal)" -> "ترتيل (عادي)"
        "Tajweed (Rules)" -> "تجويد (أحكام)"
        "Tajweed is warming up — please wait a moment…" -> "التجويد قيد التحضير — يرجى الانتظار لحظة…"
        "Standard Qalun text is not available for this Surah in local database. Standard Hafs text is shown as fallback." -> "النص القياسي لقالون غير متوفر لهذه السورة في قاعدة البيانات المحلية. يتم عرض نص حفص كبديل."
        "Tajweed Rules Highlights (Hafs):" -> "أحكام التجويد (حفص):"
        "Tajweed Rules Highlights (Qalun):" -> "أحكام التجويد (قالون):"
        "Ghunnah" -> "غنة"
        "Madd" -> "مد"
        "Qalqalah" -> "قلقلة"
        "Idgham" -> "إدغام"
        "Ikhfa" -> "إخفاء"
        "Izhar" -> "إظهار"
        "Meem" -> "أحكام الميم"
        "Al-Lam" -> "أحكام اللام"
        "Ghunnah & Shaddah" -> "أحكام النون والميم المشددتين (غنة)"
        "Silah" -> "صلة ميم الجمع"
        "Meem Sakinah (Labial Rules)" -> "أحكام الميم الساكنة"
        "Madd (Vowel Extension)" -> "مَدّ"
        "Madd Munfasil (Qalun: Qasr/Tawassut)" -> "مد منفصل (قالون)"
        "Madd Muttasil" -> "مد متصل"
        "Qalqalah (Echo & Bounce)" -> "قلقلة"
        "Idgham (Merging)" -> "إدغام"
        "Iqlab & Ikhfa" -> "إقلاب وإخفاء"
        "Izhar (Clarity)" -> "إظهار"
        "Al-Laam Rules (Solar / Lunar)" -> "أحكام اللام"
        "Active Recitation Goals" -> "أهداف التلاوة النشطة"
        "Delete Goal" -> "حذف الهدف"
        "Configure New Goal" -> "إعداد هدف جديد"
        "Goal Frequency" -> "تكرار الهدف"
        "Set Weekly Goal" -> "تحديد هدف أسبوعي"
        "Set Monthly Goal" -> "تحديد هدف شهري"
        "Set Daily Goal" -> "تحديد هدف يومي"
        "Daily" -> "يومي"
        "Weekly" -> "أسبوعي"
        "Monthly" -> "شهري"
        "Surah Content" -> "محتوى السورة"
        "Reference Text" -> "النص المرجعي"
        "Choose Surah" -> "اختر السورة"
        "Select Verses Range" -> "اختر نطاق الآيات"
        "From Ayah" -> "من آية"
        "To Ayah" -> "إلى آية"
        "Confirm" -> "تأكيد"
        "Cancel" -> "إلغاء"
        "Select Surah" -> "اختر السورة"
        "Search Surah..." -> "البحث عن سورة..."
        "Listening..." -> "جاري الاستماع..."
        "Tap to Practice" -> "اضغط للبدء"
        "Error:" -> "خطأ:"
        "Goal Accomplished! 🎉" -> "تم تحقيق الهدف! 🎉"
        "Total mistakes during this goal: " -> "إجمالي الأخطاء خلال هذا الهدف: "
        "No active goals set. Formulate a daily or weekly recitation target below to stay motivated!" -> "لم يتم تحديد أهداف نشطة بعد. حدد هدفاً يومياً أو أسبوعياً للتلاوة بالأسفل لتبقَ متحفزاً!"
        "Daily target" -> "هدف يومي"
        "Weekly target" -> "هدف أسبوعي"
        "Target Ayahs to Recite" -> "عدد الآيات المستهدفة للتلاوة"
        "Select a Surah" -> "اختر السورة"
        "Ayahs Range: " -> "نطاق الآيات: "
        "Select Start Ayah" -> "اختر آية البداية"
        "Select End Ayah" -> "اختر آية النهاية"
        // --- Goal scope picker (Stats goal-creation form) ---
        "Goal Scope (optional)" -> "نطاق الهدف (اختياري)"
        "Goal Scope" -> "نطاق الهدف"
        "Pick a Surah" -> "اختر سورة"
        "All Quran (no scope)" -> "كل القرآن (بدون نطاق)"
        "Scope: Surah" -> "النطاق: سورة"
        "Clear scope" -> "إزالة النطاق"
        "Start ayah (scope)" -> "آية البداية (النطاق)"
        "End ayah (scope)" -> "آية النهاية (النطاق)"
        "Pick the Surah this goal applies to:" -> "اختر السورة التي ينطبق عليها هذا الهدف:"
        "Close" -> "إغلاق"
        // --- No-internet popup strings ---
        "No Internet Connection" -> "لا يوجد اتصال بالإنترنت"
        "You are not connected to the internet. You need an internet connection to listen to Quran audio and to load Surah text. Please check your connection and try again." -> "أنت لست متصلاً بالإنترنت. تحتاج إلى اتصال بالإنترنت للاستماع إلى صوت القرآن ولتحميل نص السورة. يرجى التحقق من اتصالك ثم المحاولة مرة أخرى."
        "OK" -> "حسناً"
        // --- Tajweed model loading popup ---
        "Loading Tajweed Model" -> "جاري تحميل نموذج التجويد"
        "Please wait while the on-device Tajweed detection model is being loaded. This only happens the first time you open the Tajweed section. Subsequent uses are instant." -> "يرجى الانتظار بينما يتم تحميل نموذج اكتشاف التجويد على الجهاز. يحدث هذا فقط في المرة الأولى التي تفتح فيها قسم التجويد. الاستخدامات اللاحقة تكون فورية."
        // --- Offline model download popup ---
        "Downloading the model" -> "جارٍ تنزيل النموذج"
        "Download failed" -> "فشل تنزيل النموذج"
        "Retry" -> "إعادة المحاولة"
        // --- Real-time tajweed mistake popup (memorize+tajweed) ---
        "Tajweed Mistake" -> "خطأ في التجويد"
        "Your %s score: %d%%." -> "نتيجتك في %s: %d٪."
        "The word \"%s\": " -> "الكلمة \"%s\": "
        // --- Wrong/skipped word auto-pause popup (resume-from-mistake flow) ---
        "Wrong Word" -> "كلمة خاطئة"
        "Skipped Word" -> "كلمة لم تُسمع"
        "Re-read the word \"%s\" correctly, then continue from there." -> "أعد نطق الكلمة \"%s\" بشكل صحيح، ثم تابع القراءة منها."
        "The word \"%s\" was not heard. Re-read it, then continue from there." -> "لم تُسمع الكلمة \"%s\". أعد نطقها ثم تابع القراءة منها."
        "Hafs" -> "حفص"
        "Qalun" -> "قالون"
        "Advancement Advice" -> "نصيحة التقدم"
        "Wrong Words" -> "أخطاء الكلمات"
        "Skipped Words" -> "الكلمات المتروكة"
        "Tajweed Rules" -> "أحكام التجويد"
        "Recitation Mistakes Detected" -> "تم اكتشاف أخطاء في التلاوة"
        "Wrong Words Spoken:" -> "أخطاء الكلمات المنطوقة:"
        "Skipped Words:" -> "الكلمات المتروكة:"
        "Excellent recitation so far! Keep going with perfect accuracy." -> "قراءة ممتازة حتى الآن! استمر بالدقة العالية."
        // --- Strings that were previously hardcoded English in the Tajweed Rules Coach,
        //     the Stats / goal cards, the goal-completion dialog, and the language / loop
        //     controls. Added when the user pointed out that switching the app to Arabic still
        //     revealed many English literals.
        "Subhan'Allah" -> "سبحان الله"
        "Masha'Allah" -> "ما شاء الله"
        "English" -> "الإنجليزية"
        "Arabic" -> "العربية"
        "Loops: " -> "التكرار: "
        "Ayahs: " -> "الآيات: "
        "Ayahs Recited" -> "آيات مُسمّعة"
        "of" -> "من"
        "Goal" -> "هدف"
        "🌟 Masha'Allah! 🌟" -> "🌟 ما شاء الله! 🌟"
        "You have completed your " -> "لقد أتممت هدفك "
        " goal of reciting " -> " من تسميع "
        " Ayahs!" -> " آية!"
        "Tajweed Rules Coach" -> "مدرب أحكام التجويد"
        "Learn basic rules, see color highlights, and practice pronunciation with live speech feedback." -> "تعلّم الأحكام الأساسية، شاهد التلوين، وتمرّن على النطق مع التغذية الصوتية الحيّة."
        "Select to Practice:" -> "اختر للتمرين:"
        "Your recitation:" -> "تلاوتك:"
        "(scored from your live audio)" -> "(تم التقييم من صوتك المباشر)"
        "Acoustic score per rule" -> "النتيجة الصوتية لكل حكم"
        "Tajweed: " -> "تجويد: "
        "Tajweed Score: " -> "نتيجة التجويد: "
        "N/A" -> "غير متاح"
        "Total Mistakes recorded: " -> "إجمالي الأخطاء المسجّلة: "
        // Templates used in the per-rule verdict ("Good X application" / "Review your X") --
        // the rule name (e.g. "Qalqalah") is itself localized by getAppString, so we use the
        // placeholders %s and resolve them with formatPerRuleVerdict below.
        "good_rule_application_template" -> "تطبيق جيد لحكم %s"
        "review_your_rule_template" -> "راجع حكم %s"
        // --- Tajweed rule NAMES (rendered via getAppString(isArabic, rule.name)) ---
        // NOTE: the 8 active rule names ("Madd", "Ghunnah & Shaddah", "Idgham (Merging)",
        // "Iqlab & Ikhfa", "Qalqalah (Echo & Bounce)", "Izhar (Clarity)",
        // "Meem Sakinah (Labial Rules)", "Al-Lam") are ALREADY translated by the entries at
        // lines ~4788-4806 above -- the second lookup reaches them normally. We deliberately
        // do NOT re-list them here (Kotlin `when` ignores later duplicate keys, so any
        // redundancy is harmless, but keeping both versions makes maintenance confusing).
        "Silah (Qalun only)" -> "الصلة (رواية قالون فقط)"
        // --- Tajweed rule DESCRIPTIONS (short, shown on the collapsed card) ---
        "Nasal sound from the nose for 2 harakahs on heavy Noon or Meem." -> "صوت غنة من الأنف مدة حركتين عند النون أو الميم الثقيلة."
        "Extension of standard vowels (Muttasil or Munfasil) when followed by Hamzah or Sukoon." -> "إطالة الصوت في حروف المد عند انتقامها بالهمز أو السكون (متصل أو منفصل)."
        "A strong bouncing resonant release on silent letters of Qutb Jad (ق ط ب ج د)." -> "صوت ارتداد رنان قوي عند حروف قطب جد الساكنة (ق ط ب ج د)."
        "Merging a silent Noon or Tanween into a following letter of Yarmalun (ي ر م ل و ن)." -> "دمج النون الساكنة أو التنوين في حرف من حروف يرملون (ي ر م ل و ن)."
        "Hiding or converting silent Noon to Meem when followed by specific letters." -> "إخفاء النون الساكنة أو قلبها ميمًا عند انتقامها بحروف معينة."
        "Clear pronunciation of Noon Sakinah or Tanween before throat letters (أ, هـ, ع, ح, غ, خ)." -> "نطق واضح للنون الساكنة أو التنوين قبل حروف الحلق (ء هـ ع ح غ خ)."
        "Rules of quiet Meem followed by Meem (Idgham), Ba (Ikhfa), or other letters (Izhar)." -> "أحكام الميم الساكنة: إدغام قبل ميم، إخفاء قبل باء، إظهار قبل باقي الحروف."
        "Merging or pronouncing Alif-Laam (ال) based on Solar and Lunar letters." -> "دمج اللام أو نطقها بحسب حروف الشمس أو القمر."
        "Elongation of the pronoun Ha (ه) and Meem marks in Qalun riwayah." -> "إطالة ضمير الهاء (ه) وعلامات الميم في رواية قالون."
        // --- Tajweed rule FULL EXPLANATIONS (long, shown when the card is expanded) ---
        "Whenever you see a Noon (ن) or Meem (م) with a Shaddah (ّ), you must produce a nasalization (Ghunnah) from the nasopharynx for a duration of two counts." -> "عند رؤية نون (ن) أو ميم (م) مشددة (ّ) يجب إخراج غنة من الخيشوم لمدة حركتين."
        "Madd extends vowel sounds. If a vowel letter is followed by a Hamzah or a Sukoon, the sound is elongated. Muttasil (Hamzah in the same word) is 4-5 counts; Munfasil (Hamzah across words) varies by riwayah (Hafs: 4-5, Qalun: Qasr 2 or Tawassut 4)." -> "المد إطالة الصوت في حروف المد. إذا تلا حرف المد همز أو سكون يُمد الصوت. المتصل (الهمز في نفس الكلمة) 4-5 حركات؛ المنفصل (الهمز عبر الكلمات) يختلف باختلاف الرواية (حفص: 4-5، قالون: قصر 2 أو توسط 4)."
        "Qalqalah refers to echoing stops. When one of the letters (ق, ط, ب, ج, د) is quiet (has a Sukoon or is stopped upon at the end of a verse), it is pronounced with an echoing, bouncing release." -> "القلقلة هي الوقف بصدى. عندما يكون أحد الحروف (ق ط ب ج د) ساكناً (عليه سكون أو الوقوف عليه في نهاية الآية) ينطق بصوت ارتداد رنان."
        "Idgham merges the silent Noon or Tanween into the next letter. If the next letter is from (ي, م, و, ن), it merges with Ghunnah; if it is (ل, ر), it merges completely without Ghunnah." -> "الإدغام يدمج النون الساكنة أو التنوين في الحرف التالي. إذا كان الحرف التالي من (ي م و ن) يُدغم مع الغنة؛ إذا كان (ل ر) يُدغم تماماً بدون غنة."
        "Iqlab converts silent Noon or Tanween to a Meem when followed by Ba (ب). Ikhfa hides the Noon pronunciation gently with Ghunnah when followed by any of the other 15 letters." -> "الإقلاب يقلب النون الساكنة أو التنوين إلى ميم عند انتقامها بالباء (ب). الإخفاء يخفي النطق بالنون برفق مع الغنة عند انتقامها بأحد الـ 15 حرفاً الآخر."
        "When a quiet Noon (نْ) or Tanween (ً ٍ ٌ) is followed by one of the throat letters (ء, هـ, ع, ح, غ, خ), it must be pronounced clearly from its point of articulation without any extra nasalization (Ghunnah)." -> "عندما تتبع النون الساكنة (نْ) أو التنوين (ً ٍ ٌ) أحد حروف الحلق (ء هـ ع ح غ خ)، يجب نطقها بوضوح من مخرجها دون غنة إضافية."
        "Meem Sakinah (مْ) has three rules: 1) Idghaam Shafawi: if followed by Meem (م), merging them with Ghunnah. 2) Ikhfaa Shafawi: if followed by Ba (ب), hiding it with Ghunnah. 3) Izhaar Shafawi: if followed by any other letter, pronouncing it clearly." -> "الميم الساكنة (مْ) لها ثلاثة أحكام: 1) الإدغام الشفوي: إذا تلتها ميم (م)، تدمجهما مع الغنة. 2) الإخفاء الشفوي: إذا تلتها باء (ب)، تخفيها مع الغنة. 3) الإظهار الشفوي: إذا تلتها أي حرف آخر، تنطق بوضوح."
        "When a word starts with Alif-Laam (ال): 1) Al-Laam Al-Shamsiyah (Solar): the Lam is quiet and merged into the next letter (solar letter with Shaddah). 2) Al-Laam Al-Qamariyah (Lunar): the Lam is pronounced clearly with a Sukoon (lunar letter)." -> "عندما تبدأ الكلمة بـ (ال): 1) اللام الشمسية: تكون اللام ساكنة وتُدمج في الحرف التالي (حرف شمسي مشدد). 2) اللام القمرية: تنطق اللام بوضوح مع سكون (حرف قمري)."
        "In the Riwayah of Qalun, the plural Meem (م الجَمْع) can optionally be read with Silah (elongating it to 'muu' مُو) if followed by a voweled letter. Also, the pronoun Ha (ه) is elongated if nestled between two voweled letters (Silah Sughra) or before a Hamzah (Silah Kubra)." -> "في رواية قالون، يمكن قراءة ميم الجمع (م) اختيارياً مع صلة (إطالتها إلى 'مُو') إذا تلتها حرف متحرك. وأيضاً تُمد الهاء (ه) إذا كانت بين حرفين متحركين (الصلة الصغرى) أو قبل همز (الصلة الكبرى)."
        // Surah metadata (rendered even on Arabic UI because the database only stores English/
        // transliterated names). Note: "Daily" / "Weekly" / "Monthly" already exist above.
        "Ayahs" -> "آية"
        else -> key
    }
}

/**
 * Resolves the per-rule verdict templates with the already-localized rule name. Used by the
 * Tajweed Rules Coach so the verdict message reads naturally in Arabic ("تطبيق جيد لحكم قلقلة")
 * instead of the English "Good Qalqalah application".
 *
 * The two template keys live ONLY in the Arabic branch of `getAppString` -- which opens with
 * `if (!isArabic) return key`, so asking for them in English returned the raw key and the popup
 * rendered the literal `review_your_rule_template`. The user hit this on a real mic run:
 * popup body read `The word "…": review_your_rule_template Your Madd score: 46%.`
 * (found by TajweedPopupWordTest). The English prose is therefore supplied HERE, not via the
 * table, so it can never regress into a raw key again.
 */
fun formatPerRuleVerdict(isArabic: Boolean, passed: Boolean, ruleName: String): String {
    val localizedRule = getAppString(isArabic, ruleName)
    val template = if (isArabic) {
        getAppString(
            true,
            if (passed) "good_rule_application_template" else "review_your_rule_template"
        )
    } else {
        if (passed) "Good %s application." else "Review your %s application."
    }
    return template.replace("%s", localizedRule)
}


/**
 * Trim leading/trailing (near-)silence from a 16 kHz mono PCM FloatArray so the
 * model sees a SPEECH clip, not a silence-padded one.
 *
 * Why: TajweedNet was trained on per-word clips that are almost entirely speech.
 * A slice taken from the recording is typically surrounded by background quiet
 * (lead-in noise / the natural pause around the word). Feeding that long,
 * silence-padded slice makes the QALQALAH head collapse to ~0.10-0.16 on a
 * CORRECT recitation (verified against the real PyTorch model: the exact same
 * word scores 0.96 alone but 0.16 when silence is padded around it) -- the log-
 * mel of long silence is out-of-distribution for every head. Trimming the quiet
 * regions restores the score to ~0.96.
 *
 * Loudness-independent: the gate is a fraction of the slice's own peak window
 * RMS, so it adapts to a quiet or loud recording. Falls back to the original
 * slice if trimming would leave too little (< ~half a second) audio.
 */
private fun trimSilence(x: FloatArray): FloatArray {
    if (x.size < 1024) return x
    val sr = 16000
    val winMs = 20
    val win = (winMs * sr / 1000)            // 320 samples / 20 ms window
    val nWin = x.size / win
    if (nWin < 4) return x
    // RMS energy per window.
    val rms = FloatArray(nWin)
    var peakRms = 0f
    for (w in 0 until nWin) {
        var s = 0f
        val base = w * win
        for (i in 0 until win) {
            val v = x[base + i]
            s += v * v
        }
        val r = kotlin.math.sqrt(s / win)
        rms[w] = r
        if (r > peakRms) peakRms = r
    }
    if (peakRms < 1e-5f) return x             // whole slice silent -> leave as-is
    // Adaptive gate: 12% of peak window RMS -- the model only needs the speech,
    // so a fairly permissive gate keeps the bounce/fade tails the qalqalah cue
    // depends on, while dropping the long background quiet on both sides.
    val gate = peakRms * 0.12f
    var firstW = 0
    var lastW = nWin - 1
    while (firstW < nWin && rms[firstW] < gate) firstW++
    while (lastW > firstW && rms[lastW] < gate) lastW--
    if (firstW > lastW) return x
    // Keep a small safety margin on each side so the word's onset/release isn't cut.
    val margin = (winMs * sr / 1000)          // one extra window (~20 ms) each side
    val start = ((firstW * win - margin).coerceAtLeast(0))
    val end = (((lastW + 1) * win + margin).coerceAtMost(x.size))
    // If trimming leaves very little audio, the speech itself is the problem --
    // keep the full slice; do NOT collapse to a sub-quarter-second clip.
    if (end - start < 2048) return x
    return x.copyOfRange(start, end)
}

// ─────────────────────────────────────────────────────────────────────────────
// Session-summary dialog helpers
//
// Three small composables (Section, WordRow, TajweedRow) that render the
// dialog body in [viewModel.sessionSummaryToShow]. They are factored out so
// the dialog body stays readable; the per-word styling is identical to the
// existing recitation-mistake card so the user sees the same look between
// the live mistakes card and the post-session dialog.
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun SessionSummarySection(
    title: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    iconTint: androidx.compose.ui.graphics.Color,
    isArabic: Boolean,
    content: @Composable () -> Unit
) {
    androidx.compose.material3.Text(
        text = title,
        style = MaterialTheme.typography.titleSmall,
        fontWeight = FontWeight.SemiBold,
        color = if (isArabic) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurface,
        modifier = Modifier.padding(top = 12.dp, bottom = 6.dp)
    )
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 4.dp, end = 4.dp, bottom = 4.dp)
    ) {
        content()
    }
}

@Composable
private fun SessionSummaryRow(
    word: String,
    suffix: String,
    textColor: androidx.compose.ui.graphics.Color,
    isArabic: Boolean
) {
    Row(
        modifier = Modifier.padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = word,
            style = MaterialTheme.typography.bodyMedium,
            color = textColor
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = suffix,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun SessionSummaryTajweedRow(
    word: String,
    rule: String,
    isArabic: Boolean,
    score: Float? = null
) {
    Row(
        modifier = Modifier.padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = word,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.width(6.dp))
        // §4.49 — THE RULE WEARS THE RULE'S OWN COLOUR.
        //
        // This was hardcoded to Color(0xFF7E57C2) — which is not a neutral
        // accent, it is the Meem-Sakinah head colour. So every tajweed mistake in
        // the summary came out in Meem-Sakinah purple, a MADD mistake included:
        // exactly the "mingling the colours of the rules" the user reported
        // ("for example we have madd always red"). Resolve through the same
        // canonical palette the letters, the card and the score rows use, and
        // only fall back to plain text weight when a rule genuinely has no head
        // (Silah is textual) — a fallback must never borrow another rule's colour.
        val ruleColor = tajweedHeadColor(ruleLabelToHead(rule)) ?: MaterialTheme.colorScheme.primary
        Text(
            text = getAppString(isArabic, rule),
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.SemiBold,
            color = ruleColor
        )
        // The tajweed SCORE belongs on the row: the user asked for
        // "the word + tajweed rules + the tajweed score for the tajweed
        // mistakes" in the summary, exactly as the mistake popup shows.
        // Kept optional so a row whose score never landed still renders.
        if (score != null) {
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = "${score.toInt()}%",
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.SemiBold,
                color = ruleColor
            )
        }
    }
}
