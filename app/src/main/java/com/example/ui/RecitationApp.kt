package com.example.ui

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
// See com.example.sherpa.SherpaPipeline.kt for the single-AudioRecord shared
// pipeline + AyahWordMatcher hallucination guard.
import com.example.sherpa.SherpaModelDownloader
import com.example.sherpa.SingleAudioPipeline
import com.example.sherpa.isSherpaModelReady
import com.example.quran.QuranApi
import com.example.quran.SurahData
import com.example.quran.formGroupsFor
import com.example.quran.hasMuqattaatExpansion
import okhttp3.OkHttpClient
import okhttp3.Request
import kotlinx.coroutines.launch
import com.example.data.Goal
import com.example.data.GoalDatabase
import com.example.data.GoalRepository
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
import com.example.ml.RecitationAnalyzer

class MainViewModel(application: Application) : androidx.lifecycle.AndroidViewModel(application) {
    private val goalDao = GoalDatabase.getDatabase(application).goalDao()
    private val goalRepository = GoalRepository(goalDao)
    
    val allGoals: Flow<List<Goal>> = goalRepository.allGoals
    
    var completedGoalToCelebrate by mutableStateOf<Goal?>(null)
        internal set

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
        rangeEnd: Int = 0
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
                    
                    val updatedGoal = goal.copy(
                        recitedAyahs = updatedRecited,
                        mistakesCount = updatedMistakes,
                        wrongWordsCount = updatedWrong,
                        skippedWordsCount = updatedSkipped,
                        tajweedMistakesCount = updatedTajweed,
                        tajweedMistakesBreakdown = updatedBreakdown
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
        val isArabicNow = context.getSharedPreferences("quran_stats", android.content.Context.MODE_PRIVATE)
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
        surahs = com.example.quran.StaticQuran.getSurahList()
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
                val response = QuranApi.instance.getSurahs()
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
                    val response = QuranApi.instance.getSurahDetail(surah.number)
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

fun buildTajweedAnnotatedString(word: String, nextWord: String?, riwayah: String): androidx.compose.ui.text.AnnotatedString {
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
                        withStyle(style = SpanStyle(color = Color(0xFF607D8B), fontWeight = FontWeight.Bold)) {
                            for (k in alifIdx until spanEnd) append(word[k])
                        }
                        i = spanEnd
                        continue
                    } else if ("أبجحخعغفقكمهوي".contains(baseLetter)) {
                        // Lunar Lam — Lam is pronounced clearly. Keep Teal so the two sub-types stay
                        // distinguishable within the Al-Lam family.
                        withStyle(style = SpanStyle(color = Color(0xFF00ACC1), fontWeight = FontWeight.Bold)) {
                            for (k in alifIdx until spanEnd) append(word[k])
                        }
                        i = spanEnd
                        continue
                    }
                }
            }

            // Check Ghunnah: Noon or Meem with Shaddah
            if ((char == 'ن' || char == 'م') && i + 1 < len && word[i + 1] == '\u0651') {
                withStyle(style = SpanStyle(color = Color(0xFF43A047), fontWeight = FontWeight.Bold)) {
                    append(char)
                    append(word[i + 1])
                }
                i += 2
                continue
            }
            
            // Check Madd: contains arabic maddah above \u0653 or \u06E4 or if character is 'آ' or vowel with next maddah
            val isVowel = "اويىٱ".contains(char)
            val isNextMaddah = i + 1 < len && (word[i + 1] == '\u0653' || word[i + 1] == '\u06E4')
            if (isVowel && isNextMaddah) {
                withStyle(style = SpanStyle(color = Color(0xFFE53935), fontWeight = FontWeight.Bold)) {
                    append(char)
                    append(word[i + 1])
                }
                i += 2
                continue
            }
            if (char == '\u0653' || char == '\u06E4' || char == 'آ') {
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
                        withStyle(style = SpanStyle(color = Color(0xFFFB8C00), fontWeight = FontWeight.Bold)) { // Idgham - Orange
                            append(char)
                        }
                    } else if (firstNextChar == 'ب') {
                        withStyle(style = SpanStyle(color = Color(0xFF00ACC1), fontWeight = FontWeight.Bold)) { // Iqlab - Cyan
                            append(char)
                        }
                    } else if ("اأإءههعحغخ".contains(firstNextChar)) {
                        withStyle(style = SpanStyle(color = Color(0xFF8E24AA), fontWeight = FontWeight.Bold)) { // Izhar - Purple
                            append(char)
                        }
                    } else {
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
                        withStyle(style = SpanStyle(color = Color(0xFFFB8C00), fontWeight = FontWeight.Bold)) { // Idgham - Orange
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    } else if (targetNextChar == 'ب') {
                        withStyle(style = SpanStyle(color = Color(0xFF00ACC1), fontWeight = FontWeight.Bold)) { // Iqlab - Cyan
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    } else if ("اأإءههعحغخ".contains(targetNextChar)) {
                        withStyle(style = SpanStyle(color = Color(0xFF8E24AA), fontWeight = FontWeight.Bold)) { // Izhar - Purple
                            append(char)
                            if (hasSukoon) append(word[i + 1])
                        }
                    } else {
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
    if (length <= 3) return w1 == w2

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

data class SpokenWordTiming(
    val word: String,
    val startIndex: Int,
    var startMs: Long,
    var endMs: Long,
    var maxRms: Float
)


data class WordAnalysis(
    val wordIndex: Int,
    val text: String,        // Original text with diacritics
    val normalizedText: String, // Normalized for phonetic comparison
    val status: RecitationStatus,
    val spokenText: String? = null,
    val tajweedRule: String? = null,
    val tajweedAcousticFeedback: String? = null,
    val tajweedScore: Int? = null
)

data class TajweedExample(
    val arabic: String,
    val translation: String
)

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

suspend fun generateRandomQuranExample(targetHead: String): TajweedExample? = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
    try {
        // Random surah across the whole Quran; a large surah is retried a few times
        // before giving up, so cost stays small even for rare rules.
        val surahNumber = (1..114).random()
        val body = Request.Builder()
            .url("https://api.alquran.cloud/v1/surah/$surahNumber/ar.alafasy")
            .build()
            .let { QuranApiOkHttpClient.newCall(it).execute() }
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
                    val englishName = try { QuranApi.instance.getSurahs().data.firstOrNull { it.number == surahNumber }?.englishName } catch (e: Throwable) { null }
                    // Build a 4–6 word window centred on the matched token (clamped to ayah
                    // edges). The trigger word stays near the centre so the rule-colourizer
                    // paints the right segment, but the snippet is large enough to give the
                    // user context -- previously Madd examples were just "مَآءٌ" (1 word) and
                    // other rules rendered the entire ayah, neither of which the user wanted.
                    val snippet = tokenWindow(tokens, j, 4)
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
 * OkHttp for QuranApi (Retrofit transitively brings it in); a small singleton here
 * keeps the generator independent of Retrofit's suspend functions (which are for
 * the surah *list* only).
 */
private val QuranApiOkHttpClient by lazy {
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
    "Madd Muttasil", "Madd (Vowel Extension)", "Madd Munfasil (Qalun: Qasr/Tawassut)" -> "MADD"
    "Qalqalah (Echo & Bounce)" -> "QALQALAH"
    "Meem Sakinah (Labial Rules)" -> "MEEM_SAKINAH"
    "Silah" -> "IZHAR"
    "Idgham (Merging)" -> "IDGHAM"
    "Iqlab & Ikhfa" -> "IKHFA"
    "Izhar (Clarity)" -> "IZHAR"
    else -> null
}

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
    isTajweedMode: Boolean = false
): List<WordAnalysis> {
    val result = ArrayList<WordAnalysis>()
    var targetIndex = 0
    var spokenIndex = 0
    val limit = minOf(targetWordsNormalized.size, targetWordsOriginal.size)
    
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
            targetIndex++; spokenIndex++
            continue
        }

        val nextWordOrig = if (targetIndex + 1 < limit) targetWordsOriginal[targetIndex + 1] else null
        val rule = getTajweedRuleForWord(targetWordOrig, nextWordOrig, riwayah)
        
        if (spokenIndex >= spoken.size) {
            break
        }
        
        var matchInSpokenAhead = -1
        var matchedSpokenAdvance = 1
        val wordGroups = targetAllGroups?.getOrNull(targetIndex)
        for (i in spokenIndex until minOf(spokenIndex + 3, spoken.size)) {
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
        
        if (matchInSpokenAhead != -1) {
            var finalStatus = RecitationStatus.CORRECT
            var acousticFeedback: String? = null
            var tScore: Int? = null
            
            // Step 3 - Measure Audio: Validate Tajweed Acoustic properties on the user's REAL voice.
            if (rule != null && hasRealAudio && isTajweedMode && !isHintedWord) {
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
                    // So we: (1) take a generous context window centered on the word
                    // (to include the bounce's lead-in/release), and (2) SILENCE-TRIM
                    // that window so the model only sees speech -- whatever its length.
                    // This depends only on the already-captured fullAudio buffer; it
                    // touches neither the microphone nor SpeechRecognizer (preserving
                    // the single-mic behaviour on phones like the A05s).
                    val targetCtxMs = 3500L
                    val wordMidMs = (segStartMs + clampedEnd) / 2L
                    val ctxStartMs = (wordMidMs - targetCtxMs / 2).coerceAtLeast(0L)
                    val ctxEndMs = (wordMidMs + targetCtxMs / 2)
                        .coerceAtMost(capturedMs)
                        .coerceAtMost(recordingDurationMs)
                        .coerceAtLeast(ctxStartMs + 30L)
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

                    // Map the app's friendly tajweed label to the trained model head.
                    val head = ruleLabelToHead(rule)
                    if (head != null && seg.size >= 256) {
                        // Per-head calibrated threshold (see decisionThresholdForHead) -- a flat 0.5
                        // wrongly fails ~12% of CORRECT Qalqalah recitations because that head's
                        // output for correct clips spans 0.09-0.97 (std 0.19). The displayed % stays
                        // the raw prob*100; only the verdict uses the calibrated bar.
                        val thr = decisionThresholdForHead(head)
                        val results = analyzer.analyze(seg, listOf(head), thr)
                        val prob = results.firstOrNull()?.prob ?: 0f
                        tScore = (prob * 100).toInt().coerceIn(0, 100)
                        acousticFeedback = if (prob >= thr) {
                            "Good $rule application"
                        } else {
                            "Review your $rule"
                        }
                        // The model head outputs P(correct); below the head's calibrated
                        // threshold the rule was not applied correctly, so flag a Tajweed error.
                        if (prob < thr) {
                            finalStatus = RecitationStatus.TAJWEED_ERROR
                        }
                    }
                }
            }
            
            result.add(WordAnalysis(
                wordIndex = targetIndex,
                text = targetWordOrig,
                normalizedText = targetWordNorm,
                status = finalStatus,
                spokenText = spoken[matchInSpokenAhead],
                tajweedRule = rule,
                tajweedAcousticFeedback = acousticFeedback,
                tajweedScore = tScore
            ))
            spokenIndex = matchInSpokenAhead + matchedSpokenAdvance
            targetIndex++
            continue
        }
        
        var matchInTargetAhead = -1
        val spokenCurrent = spoken[spokenIndex]
        for (j in targetIndex + 1 until minOf(targetIndex + 4, limit)) {
            if (isWordMatch(spokenCurrent, targetWordsNormalized[j])) {
                matchInTargetAhead = j
                break
            }
        }
        
        if (matchInTargetAhead != -1) {
            result.add(WordAnalysis(
                wordIndex = targetIndex,
                text = targetWordOrig,
                normalizedText = targetWordNorm,
                status = RecitationStatus.SKIPPED,
                spokenText = null,
                tajweedRule = rule
            ))
            targetIndex++
            continue
        }
        
        result.add(WordAnalysis(
            wordIndex = targetIndex,
            text = targetWordOrig,
            normalizedText = targetWordNorm,
            status = RecitationStatus.WRONG,
            spokenText = spoken[spokenIndex],
            tajweedRule = rule
        ))
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
fun RecitationApp(modifier: Modifier = Modifier) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val prefs = remember { context.getSharedPreferences("quran_stats", android.content.Context.MODE_PRIVATE) }
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
    val tajweedSpokenText = (tajweedFinalSpokenText + " " + tajweedPartialSpokenText).trim()
    var currentTab by remember { mutableStateOf("Memorize") }
    // Default to Tarteel. The user explicitly said "when the open app i want it to be in tarteel
    // not tajweed as default": Memorize previously defaulted to Tajweed, which meant on cold
    // launch the raw-audio recorder + acoustic model ran for every recitation even if the user
    // never asked for tajweed detection. Tajweed is now an explicit opt-in via the FilterChip.
    var recitationMode by remember { mutableStateOf("Tarteel") }
    var showSurahSelectionDialog by remember { mutableStateOf(false) }
    
    var isListening by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    var alignment by remember { mutableStateOf(Pair(0, false)) }
    var detailedAnalysis by remember { mutableStateOf<List<WordAnalysis>>(emptyList()) }

    // (item 5) Real-time tajweed popup state. When Memorize+Tajweed's acoustic engine flags a
    // TAJWEED_ERROR on a specific word, we POST it here and a small auto-dismissing dialog at
    // the bottom of the screen shows the user what went wrong WITHOUT pausing the recitation --
    // "i want the user to continue reading and if there is an error a pop up is shown". The
    // timestamp in the state lets Compose see two consecutive errors on the same word as a
    // fresh trigger. LaunchedEffect below auto-clears after 4 s.
    var latestTajweedPopup by remember { mutableStateOf<Pair<String, Int>?>(null) }  // (rule, wordIndex)

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
    // post-filter hallucination guard — see com.example.sherpa.SherpaPipeline.kt.
    //
    // To work around Kotlin's ban on forward local references inside a closure,
    // the pipeline is constructed BELOW `targetOrig` and stored in this holder,
    // scope; the pipeline is assigned to it once `targetOrig` is available.
    val tajweedPipelineHolder = remember { TajweedPipelineHolder() }
    val sherpaDownloader = remember(context) { SherpaModelDownloader(context) }
    val sherpaReady = remember { mutableStateOf<Boolean?>(null) }
    var tajweedRecordedAudio by remember { mutableStateOf<FloatArray?>(null) }
    var tajweedRecordingStartMs by remember { mutableStateOf(0L) }
    var tajweedRecordingDurationMs by remember { mutableStateOf(0L) }

    // Popup-guard: remembers which word-index the last tajweed popup fired for, so one mistake
    // isn't re-shown on every recomposition. Declared here (before the range-reset effect) so a
    // target change can also clear it.
    var lastTajweedPopupIndex by remember { mutableStateOf(-1) }

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
        viewModel.ayahs.flatMap { it.words }.map { com.example.quran.formGroupsFor(it) }
    }

    // Construct the Sherpa-ONNX offline streaming pipeline once per ayah range
    // (so its decoder-level hotwordsFile bias tracks the new vocabulary each
    // time the user changes the range — see com.example.sherpa.SherpaPipeline.kt
    // for why hotwordsFile is set only at recognizer construction time).
    // `tajweedPipelineHolder` was declared earlier in the Composable scope; the
    // pipeline is wired into it the first time `targetOrig` is resolved AND
    // re-wired on every range change (the previous instance is released first
    // to avoid leaking a native OnlineRecognizer + AudioRecord handle). The
    // wrap is safe even when the model files aren't downloaded yet: `start()`
    // returns false in that case, and the Compose download dialog blocks the
    // FAB until the user finishes the first-launch download.
    LaunchedEffect(targetOrig) {
        try { tajweedPipelineHolder.pipeline?.release() } catch (e: Throwable) {}
        tajweedPipelineHolder.pipeline = SingleAudioPipeline(
            context = context,
            ayahWordsProvider = { targetOrig }
        )
    }

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
        spokenTimings.clear()
        currentWordCount = 0
        recognizedAudio = null
        recognizedAudioUri = null
        recognizedAudioStartMs = System.currentTimeMillis()
        alignment = Pair(0, false)
        detailedAnalysis = emptyList()
        hintedIndices.clear()
        latestTajweedPopup = null
        errorMessage = null
        // Cancel is async: one more onPartial/onResults with the OLD whole-transcript can still
        // arrive before the engine fully stops. Suppress recognizer output for the next 500 ms so
        // that stale transcript isn't re-analyzed against the fresh range (the exact bug the user
        // hit: mistakes appear immediately after changing the ayah range with nothing read).
        recognizerResultsSuppressedUntil = System.currentTimeMillis() + 500L
        // Resume automatically: if the user was mid-recitation when they moved the range, start
        // listening again (once the suppression window has elapsed) so they can keep going without
        // a manual mic tap. Guard on the recognizer being alive and the tab still being Memorize.
        if (wasListening && speechRecognizer != null && currentTab == "Memorize") {
            kotlinx.coroutines.delay(520)
            if (currentTab == "Memorize" && !isListening) {
                try {
                    startListening(speechRecognizer!!, context.packageName) { msg ->
                        errorMessage = if (msg.isEmpty()) null else msg; isListening = false
                    }
                    isListening = true
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
    LaunchedEffect(spokenText, currentTab, recitationMode, targetNorm, targetOrig, targetFormGroups, recognizedAudioUri, recognizedAudio, hintedIndicesSnapshot) {
        if (currentTab != "Memorize") {
            alignment = Pair(targetOrig.size, false)
            detailedAnalysis = emptyList()
        } else {
            kotlinx.coroutines.delay(200) // Debounce rapid speech recognition partial updates to avoid UI thread ANRs
            val result = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
                val spoken = spokenText.split(Regex("\\s+"))
                    .filter { it.isNotBlank() }
                    .map { normalizeForComparison(it, viewModel.riwayah) }

                val align = alignRecitation(spoken, targetNorm, targetFormGroups)
                val analysis = analyzeRecitationDetails(
                    spoken,
                    targetOrig,
                    targetNorm,
                    viewModel.riwayah,
                    spokenTimings,
                    context,
                    fullAudio = recognizedAudio,
                    recordingStartMs = recognizedAudioStartMs,
                    recordingDurationMs = (recognizedAudio?.size ?: 0) * 1000L / 16000,
                    targetAllGroups = targetFormGroups,
                    hintedIndices = hintedIndices,
                    isTajweedMode = recitationMode == "Tajweed"
                )
                
                var newMatchedWordCount = 0
                var newHasError = false
                val isTajweedMode = recitationMode == "Tajweed"

                // TAJWEED-only errors (rule-acoustic failed but the WORD was spoken correctly)
                // no longer pause the recitation. The user explicitly asked the app to "continue
                // reading" and show a popup on the tajweed mistake instead of stopping. Wrong
                // and skipped words still pause as before -- they are detection failures that
                // would otherwise silently let the user babble ahead.
                for (word in analysis) {
                    if (word.status == RecitationStatus.CORRECT || word.status == RecitationStatus.TAJWEED_ERROR || word.status == RecitationStatus.HINTED) {
                        // TAJWEED_ERROR and HINTED both continue advancing -- TAJWEED_ERROR only
                        // matters via its popup, and HINTING is the user's own reveal, so we must
                        // not pause waiting for a word they'll never need to say.
                        newMatchedWordCount++
                    } else if (word.status == RecitationStatus.WRONG || word.status == RecitationStatus.SKIPPED) {
                        newHasError = true
                        break
                    }
                }

                Pair(Pair(newMatchedWordCount, newHasError), analysis)
            }
            alignment = result.first
            detailedAnalysis = result.second
        }
    }

    // (item 5) Real-time tajweed popup watcher. Scans the latest `detailedAnalysis` for a
    // TAJWEED_ERROR. If we find one, push `(rule, wordIndex)` into `latestTajweedPopup`; that
    // instantly recomposes the small floating dialog lower in the tree. We don't pause the
    // recitation: the loop continues processing further words. The `remember`ed
    // `lastTajweedPopupIndex` guard prevents re-firing for the same error on every recomposition,
    // so the user sees each mistake once per word. `lastTajweedPopupIndex` is declared above
    // (next to the recognizer) so the range/surah reset can clear it too.
    LaunchedEffect(detailedAnalysis, currentTab, recitationMode) {
        if (currentTab == "Memorize" && recitationMode == "Tajweed") {
            // Find the LATEST tajweed mistake (last in word order -- the most recent error).
            val tajweedError = detailedAnalysis.lastOrNull {
                it.status == RecitationStatus.TAJWEED_ERROR && it.tajweedRule != null
            }
            if (tajweedError != null) {
                if (tajweedError.wordIndex != lastTajweedPopupIndex) {
                    lastTajweedPopupIndex = tajweedError.wordIndex
                    latestTajweedPopup = Pair(tajweedError.tajweedRule!!, tajweedError.wordIndex)
                }
            } else if (latestTajweedPopup == null) {
                // No error currently AND no popup showing -> reset the latch so that if the user
                // re-reads the SAME word and makes the SAME mistake again after the previous popup
                // auto-dismissed, the fresh TAJWEED_ERROR re-triggers instead of being swallowed
                // by the stale index.
                lastTajweedPopupIndex = -1
            }
        } else if (currentTab != "Memorize") {
            // User left Memorize -- clear any stale popup state so re-entering is clean.
            lastTajweedPopupIndex = -1
            latestTajweedPopup = null
        }
    }

    // (item 5) Auto-dismiss the tajweed popup after 4 seconds so the user can keep reading.
    // Latency is intentional: long enough to read the message, short enough not to obscure
    // subsequent words. If a new mistake arrives while the dialog is still up, the pair-key
    // changes and this timer restarts with the new content.
    LaunchedEffect(latestTajweedPopup) {
        if (latestTajweedPopup != null) {
            kotlinx.coroutines.delay(4000)
            latestTajweedPopup = null
        }
    }

    val mainHandler = remember { android.os.Handler(android.os.Looper.getMainLooper()) }

    val matchedWordCount = alignment.first
    val hasError = alignment.second
    val errorIndex = if (hasError) matchedWordCount else -1

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
        if (tajweedPipelineHolder.running) return
        // First-launch gate: if the Sherpa model files aren't on disk yet,
        // the pipeline's start() returns false without touching the mic.
        // The Compose download dialog (further down in this Composable) is
        // the user-facing restart button — bail out silently here so the FAB
        // doesn't flap when they tap before the model has landed.
        if (!isSherpaModelReady(context)) {
            isListening = true
            errorMessage = getAppString(isArabic, "Downloading the speech recognizer model — please wait")
            return
        }
        spokenTimings.clear()
        currentWordCount = 0
        tajweedFinalSpokenText = ""
        tajweedPartialSpokenText = ""
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
        val started = tajweedPipelineHolder.pipeline?.start() ?: false
        if (!started) {
            isListening = false
            errorMessage = "Microphone is unavailable on this device"
        }
    }

    fun stopTajweedSession() {
        // Tolerate the "tap-stop-during-load" race: the user may tap the mic,
        // see isListening go true immediately, and tap again before the
        // recognizer finishes constructing on IO. `running` stays false until
        // the IO job succeeds, so `tajweedPipelineHolder.running` is false —
        // but isListening was set true optimistically. Clear both, and the IO
        // job's eventual `running=true` / state=Running write is harmless
        // because stop() will already have flipped them back. We also call
        // pipeline.stop() defensively — it's a no-op if `running` is false,
        // and we add the @Volatile guard so the warm-native path sets
        // running=false before tearing down.
        val wasListening = isListening || tajweedPipelineHolder.running
        if (!wasListening) {
            isListening = false
            return
        }
        val audio = if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop() else FloatArray(0)
        tajweedRecordedAudio = audio
        tajweedRecordingDurationMs = (audio.size * 1000L) / 16000
        try { speechRecognizer?.stopListening() } catch (e: Throwable) {}
        isListening = false
    }

    fun stopAllListening() {
        isListening = false
        try { speechRecognizer?.stopListening(); speechRecognizer?.cancel() } catch (e: Throwable) {}
        try { if (tajweedPipelineHolder.running) tajweedPipelineHolder.stop() } catch (e: Throwable) {}
    }

    LaunchedEffect(currentTab) {
        if (isListening) {
            isListening = false
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
                if (isListening) {
                    isListening = false
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
                // In the Tajweed coach we own the microphone session (AudioRecord); the recognizer
                // is only a best-effort transcript source, so its completion must NOT end our
                // recording. Otherwise this result ends the Memorize/Tarteel session.
                if (currentTab != "Tajweed") isListening = false
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
            errorMessage = if (err.isEmpty()) null else err
            // In the Tajweed coach the recognizer is best-effort; its error must not end our
            // own audio recording session (the score comes from our AudioRecord, not the recognizer).
            if (currentTab != "Tajweed") isListening = false
            currentWordCount = 0
            if (currentTab == "Tajweed") {
                tajweedPartialSpokenText = ""
            } else {
                // Safely trim finalSpokenText to correctly matched words on error to avoid losing place
                val matchedWordsList = targetOrig.take(matchedWordCount)
                finalSpokenText = matchedWordsList.joinToString(" ")
                partialSpokenText = ""
            }
        }
    }
    listenerWrapper.onState = { stateValue -> 
        mainHandler.post {
            isListening = stateValue 
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
    // [com.example.sherpa.AyahWordMatcher] hallucination guard). We mirror the
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
                currentTab == "Tajweed" || (currentTab == "Memorize" && recitationMode == "Tajweed")
            if (!pipelineDriven) return@LaunchedEffect
            val pipe = tajweedPipelineHolder.pipeline ?: return@LaunchedEffect
            pipe.partials.collect { ev ->
                val timeNow = System.currentTimeMillis()
                val words = ev.partial
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
            val partialStr = words.joinToString(" ").trim()
            if (ev.endpoint) {
                if (partialStr.isNotEmpty()) {
                    tajweedFinalSpokenText = "$tajweedFinalSpokenText $partialStr".trim()
                }
                tajweedPartialSpokenText = ""
            } else {
                if (partialStr.isNotEmpty()) tajweedPartialSpokenText = partialStr
            }
        }
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            if (!isListening) {
                if (speechRecognizer != null) {
                    spokenTimings.clear()
                    currentWordCount = 0
                    recognizedAudioUri = null
                    recognizedAudio = null
                    recognizedAudioStartMs = System.currentTimeMillis()
                    isListening = true
                    startListening(speechRecognizer!!, context.packageName, onError = { errorMessage = if(it.isEmpty()) null else it; isListening = false })
                    // Same single-mic invariant as the FAB: no parallel AudioRecord here.
                    // See the FAB start-listening branch for the A05s single-mic rationale.
                    if (recitationMode == "Tajweed") {
                        viewModel.preloadTajweedModel()
                    }
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
                                onClick = { recitationMode = "Tarteel" },
                                label = { Text(getAppString(isArabic, "Tarteel"), fontSize = 11.sp) }
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            FilterChip(
                                selected = recitationMode == "Tajweed",
                                onClick = {
                                    recitationMode = "Tajweed"
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
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(modifier = Modifier.size(8.dp).background(Color(0xFF43A047), CircleShape))
                    Spacer(modifier = Modifier.width(3.dp))
                    Text(getAppString(isArabic, "Ghunnah"), style = MaterialTheme.typography.bodySmall, fontSize = 9.sp)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(modifier = Modifier.size(8.dp).background(Color(0xFFE53935), CircleShape))
                    Spacer(modifier = Modifier.width(3.dp))
                    Text(getAppString(isArabic, "Madd"), style = MaterialTheme.typography.bodySmall, fontSize = 9.sp)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(modifier = Modifier.size(8.dp).background(Color(0xFF1E88E5), CircleShape))
                    Spacer(modifier = Modifier.width(3.dp))
                    Text(getAppString(isArabic, "Qalqalah"), style = MaterialTheme.typography.bodySmall, fontSize = 9.sp)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(modifier = Modifier.size(8.dp).background(Color(0xFFFB8C00), CircleShape))
                    Spacer(modifier = Modifier.width(3.dp))
                    Text(getAppString(isArabic, "Idgham"), style = MaterialTheme.typography.bodySmall, fontSize = 9.sp)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(modifier = Modifier.size(8.dp).background(Color(0xFF00ACC1), CircleShape))
                    Spacer(modifier = Modifier.width(3.dp))
                    Text(getAppString(isArabic, "Ikhfa"), style = MaterialTheme.typography.bodySmall, fontSize = 9.sp)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(modifier = Modifier.size(8.dp).background(Color(0xFF8E24AA), CircleShape))
                    Spacer(modifier = Modifier.width(3.dp))
                    Text(getAppString(isArabic, "Izhar"), style = MaterialTheme.typography.bodySmall, fontSize = 9.sp)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(modifier = Modifier.size(8.dp).background(Color(0xFF7E57C2), CircleShape))
                    Spacer(modifier = Modifier.width(3.dp))
                    Text(getAppString(isArabic, "Meem"), style = MaterialTheme.typography.bodySmall, fontSize = 9.sp)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(modifier = Modifier.size(8.dp).background(Color(0xFF607D8B), CircleShape))
                    Spacer(modifier = Modifier.width(3.dp))
                    Text(getAppString(isArabic, "Al-Lam"), style = MaterialTheme.typography.bodySmall, fontSize = 9.sp)
                }
                if ((currentTab == "Memorize" || currentTab == "Read") && viewModel.riwayah == "Qalun") {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(modifier = Modifier.size(8.dp).background(Color(0xFF8E24AA), CircleShape))
                        Spacer(modifier = Modifier.width(3.dp))
                        Text(getAppString(isArabic, "Silah"), style = MaterialTheme.typography.bodySmall, fontSize = 9.sp)
                    }
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
        // TAJWEED_ERROR on a specific word during Memorize+Tajweed. Deliberately does NOT
        // pause the recitation: the analysis loop keeps running, this dialog posts itself
        // on top, auto-dismisses after 4 s, and the user keeps reading. The `latestTajweedPopup`
        // state (and its 4 s auto-timer) lives next to detailedAnalysis above.
        latestTajweedPopup?.let { (ruleLabel, _) ->
            AlertDialog(
                onDismissRequest = { latestTajweedPopup = null },
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = null,
                            tint = Color(0xFFF57F17)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = getAppString(isArabic, "Tajweed Mistake"),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFFF57F17)
                        )
                    }
                },
                text = {
                    Text(
                        text = formatPerRuleVerdict(isArabic, false, ruleLabel),
                        style = MaterialTheme.typography.bodyMedium
                    )
                },
                confirmButton = {
                    TextButton(onClick = { latestTajweedPopup = null }) {
                        Text(getAppString(isArabic, "OK"))
                    }
                }
            )
        }

        // ── Sherpa-ONNX model download dialog (Memorize+Tajweed AND Tajweed Rules Coach) ─
        //
        // First-launch gate for the offline streaming Zipformer (~323 MB,
        // Apache-2.0). The transcript pipeline refuses `start()` until the
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
                        title = { Text(getAppString(isArabic, "Downloading speech model")) },
                        text = {
                            Column {
                                Text(getAppString(
                                    isArabic,
                                    "Itqān needs the offline speech recognizer (${(dlState.totalBytes / 1_048_576).toInt()} MB) for word-by-word highlight in Memorize+Tajweed."
                                ))
                                Spacer(Modifier.height(12.dp))
                                Text("${(pct * 100).toInt()}%  •  ${dlState.currentFile}")
                                Spacer(Modifier.height(8.dp))
                                LinearProgressIndicator(
                                    progress = { pct },
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }
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
            val rulesList = remember {
                listOf(
                    TajweedRuleInfo(
                        name = "Ghunnah & Shaddah",
                        description = "Nasal sound from the nose for 2 harakahs on heavy Noon or Meem.",
                        color = Color(0xFF43A047), // Green
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
                        color = Color(0xFFE53935), // Red
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
                        color = Color(0xFF1E88E5), // Blue
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
                        color = Color(0xFFFB8C00), // Orange
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
                        color = Color(0xFF00ACC1), // Cyan / Teal
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
                        color = Color(0xFF8E24AA), // Purple
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
                        color = Color(0xFF7E57C2), // Deep Purple
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
                            TajweedExample("مَا لَّبَلِهِمْ", "Whatever [they] please"),
                            TajweedExample("مِن نَّبَإِهِمْ", "From their news"),
                            TajweedExample("أَم بِٱلتَّوْبَةِ", "Or with the repentance"),
                            TajweedExample("عَلَيْهِم مَّا", "Upon them whatever"),
                        )
                    ),
                    TajweedRuleInfo(
                        name = "Al-Lam",
                        description = "Merging or pronouncing Alif-Laam (ال) based on Solar and Lunar letters.",
                        color = Color(0xFF607D8B), // Blue Grey
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
                        color = Color(0xFF8E24AA), // Purple
                        fullExplanation = "In the Riwayah of Qalun, the plural Meem (م الجَمْع) can optionally be read with Silah (elongating it to 'muu' مُو) if followed by a voweled letter. Also, the pronoun Ha (ه) is elongated if nestled between two voweled letters (Silah Sughra) or before a Hamzah (Silah Kubra).",
                        examples = listOf(
                            TajweedExample("عَلَيْهِمُ مَسْكَنَتُهُمْ", "Over them [is] destitution"),
                            TajweedExample("إِنَّهُۥ كَانَ", "Indeed He is"),
                            TajweedExample("عِندَهُۥٓ إِلَّا", "With Him except")
                        )
                    )
                ).filter { it.name.contains("Madd") || it.name.contains("Ghunnah") || it.name.contains("Idgham") || it.name.contains("Meem") || it.name.contains("Qalqalah") || it.name.contains("Ikhfa") || it.name.contains("Izhar") || it.name.contains("Lam") }
            }
            
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
                            val online = generateRandomQuranExample(head)
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
                    
                    val spokenTokens = remember(tajweedSpokenText, isExpanded) {
                        if (isExpanded) {
                            tajweedSpokenText.split(Regex("\\s+")).filter { it.isNotBlank() }
                        } else emptyList()
                    }
                    
                    var localAnalysis by remember { mutableStateOf<List<WordAnalysis>?>(null) }
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
                            // mic session (MicRecorderController), so hasRealAudio is true here.
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
                                    isTajweedMode = true
                                )
                            }
                        } else if (!isExpanded || tajweedRecordedAudio == null) {
                            localAnalysis = null
                        }
                    }
                    val localAnalysisSnap = localAnalysis
                    val hasMistake = localAnalysisSnap?.any { it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED || it.status == RecitationStatus.TAJWEED_ERROR } == true

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
                                                            val online = generateRandomQuranExample(head)
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
                                            containerColor = if (isListening) MaterialTheme.colorScheme.errorContainer else rule.color,
                                            contentColor = if (isListening) MaterialTheme.colorScheme.onErrorContainer else Color.White
                                        ) {
                                            Icon(
                                                imageVector = if (isListening) Icons.Default.MicOff else Icons.Default.Mic,
                                                contentDescription = if (isListening) "Stop Listening" else "Start Listening"
                                            )
                                        }
                                    }

                                    if (tajweedSpokenText.isNotEmpty() || localAnalysisSnap != null) {
                                        Spacer(modifier = Modifier.height(16.dp))
                                        Text(
                                            text = getAppString(isArabic, "Your recitation:"),
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                        Text(
                                            text = if (tajweedSpokenText.isNotBlank()) tajweedSpokenText else getAppString(isArabic, "(scored from your live audio)"),
                                            style = MaterialTheme.typography.bodyLarge,
                                            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                                            textAlign = TextAlign.Right
                                        )
                                        
                                        if (!isListening && localAnalysisSnap != null) {
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
                                            val ruleAgg = remember(localAnalysisSnap) {
                                                // words that actually got an acoustic score for some rule
                                                val scored = localAnalysisSnap.filter {
                                                    it.tajweedRule != null && it.tajweedScore != null
                                                }
                                                scored.groupBy { it.tajweedRule!! }
                                                    .mapValues { (_, words) ->
                                                        words.map { it.tajweedScore!! }.average().toInt()
                                                    }
                                                    .toList()
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
                                                            val ruleColor = info?.color ?: MaterialTheme.colorScheme.primary
                                                            val head = ruleLabelToHead(ruleName)
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

                                                        Spacer(modifier = Modifier.height(6.dp))
                                                        // Per-word transcript, colored by that word's per-word
                                                        // status (red on a Tajweed error) -- the per-rule verdict
                                                        // rows above are the headline; this is just a visual cue.
                                                        val feedbackText = buildAnnotatedString {
                                                            localAnalysisSnap.forEach { wordInfo ->
                                                                val color = when (wordInfo.status) {
                                                                    RecitationStatus.WRONG, RecitationStatus.SKIPPED, RecitationStatus.TAJWEED_ERROR -> MaterialTheme.colorScheme.error
                                                                    else -> MaterialTheme.colorScheme.onSurface
                                                                }
                                                                withStyle(style = SpanStyle(color = color)) {
                                                                    append("${wordInfo.text} ")
                                                                }
                                                            }
                                                        }
                                                        Text(text = feedbackText, textAlign = TextAlign.Right, modifier = Modifier.fillMaxWidth())
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
                            var goalScopeSurah by remember { mutableStateOf<com.example.quran.SurahData?>(null) }
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
                            matchedWordCount >= ayah.wordStartIndex && matchedWordCount < ayah.wordStartIndex + ayah.words.size
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
                                    partialSpokenText = ""
                                    // Drop any hint-charges for words in this ayah: we're re-reading
                                    // it, so those injected words must come out of the transcript too.
                                    // (Avoid SnapshotStateSet#removeAll-with-predicate overload
                                    // ambiguity by collecting into an IntRange first.)
                                    val hintRange = startIdx until endIdx
                                    val staleHints = hintedIndices.filter { it in hintRange }
                                    hintedIndices.removeAll(staleHints)
                                    hintActive = false
                                    if (isListening) {
                                        try { speechRecognizer?.stopListening(); isListening = false } catch (e: Throwable) {}
                                        errorMessage = null
                                    }
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
                                hintActive = false
                                hintedIndices.clear()
                                if (isListening) {
                                    try { speechRecognizer?.stopListening(); isListening = false } catch (e: Throwable) {}
                                    errorMessage = null
                                }
                            }) {
                                Icon(Icons.Default.Refresh, contentDescription = "Restart", tint = MaterialTheme.colorScheme.primary)
                            }
                        }
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                
                // Show Error Banner if applicable
                if (currentTab == "Memorize" && hasError) {
                    val currentAnalysis = detailedAnalysis.getOrNull(errorIndex)
                    if (currentAnalysis != null) {
                        val isTajweedError = currentAnalysis.status == RecitationStatus.TAJWEED_ERROR && recitationMode == "Tajweed"
                        val isWrongWord = currentAnalysis.status == RecitationStatus.WRONG
                        val isSkippedWord = currentAnalysis.status == RecitationStatus.SKIPPED
                        
                        if (isTajweedError || isWrongWord || isSkippedWord) {
                            val bannerColor = if (isTajweedError) Color(0xFFFFF9C4) else MaterialTheme.colorScheme.errorContainer
                            val borderColor = if (isTajweedError) Color(0xFFFBC02D) else MaterialTheme.colorScheme.error
                            val iconColor = if (isTajweedError) Color(0xFFF57F17) else MaterialTheme.colorScheme.error
                            
                            val title = when {
                                isTajweedError -> "Tajweed Feedback (${currentAnalysis.tajweedRule ?: "Rule"})"
                                isSkippedWord -> "Skipped Word"
                                else -> "Incorrect Pronunciation/Word"
                            }
                            
                            val message = when {
                                isTajweedError -> currentAnalysis.tajweedAcousticFeedback ?: "Review ${currentAnalysis.tajweedRule} here."
                                isSkippedWord -> "You skipped the word: ${currentAnalysis.text}"
                                else -> "Expected ${currentAnalysis.text}, but heard something else."
                            }
                            
                            Card(
                                modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
                                colors = CardDefaults.cardColors(containerColor = bannerColor),
                                border = BorderStroke(1.dp, borderColor)
                            ) {
                                Row(modifier = Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                                    Icon(Icons.Default.Warning, contentDescription = "Notice", tint = iconColor, modifier = Modifier.size(24.dp))
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Column {
                                        Text(title, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold, color = iconColor)
                                        Text(message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    }
                                }
                            }
                        }
                    }
                }
                
                val listState = androidx.compose.foundation.lazy.rememberLazyListState()
                
                var lastScrolledAyahIndex by remember { mutableStateOf(-1) }
                
                LaunchedEffect(viewModel.playingAyahId) {
                    if (viewModel.playingAyahId != null) {
                        val index = viewModel.ayahs.indexOfFirst { it.ayahId == viewModel.playingAyahId }
                        if (index >= 0 && index != lastScrolledAyahIndex) {
                            lastScrolledAyahIndex = index
                            listState.scrollToItem(index)
                        }
                    }
                }
                
                LaunchedEffect(matchedWordCount) {
                    hintActive = false
                    val ayahIndex = viewModel.ayahs.indexOfFirst { ayah -> 
                        matchedWordCount >= ayah.wordStartIndex && matchedWordCount < ayah.wordStartIndex + ayah.words.size
                    }
                    if (ayahIndex >= 0 && ayahIndex != lastScrolledAyahIndex) {
                        lastScrolledAyahIndex = ayahIndex
                        listState.scrollToItem(ayahIndex)
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
                                            val isHintedWord = detailedAnalysis.getOrNull(globalWordIndex)?.status == RecitationStatus.HINTED
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
                                        } else if (globalWordIndex == errorIndex && !hintActive) {
                                            val currentAnalysis = detailedAnalysis.getOrNull(globalWordIndex)
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
                    if (spokenText.isNotEmpty() && currentTab == "Memorize") {
                        item {
                            Spacer(modifier = Modifier.height(16.dp))
                            val wrongs = detailedAnalysis.filter { it.status == RecitationStatus.WRONG }
                            val skippeds = detailedAnalysis.filter { it.status == RecitationStatus.SKIPPED }
                            
                            if (wrongs.isNotEmpty() || skippeds.isNotEmpty()) {
                                Card(
                                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.15f)
                                    ),
                                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.3f)),
                                    shape = RoundedCornerShape(16.dp)
                                ) {
                                    Column(modifier = Modifier.padding(16.dp)) {
                                        Text(
                                            text = getAppString(isArabic, "Recitation Mistakes Detected"),
                                            style = MaterialTheme.typography.titleMedium,
                                            fontWeight = FontWeight.Bold,
                                            color = MaterialTheme.colorScheme.error
                                        )
                                        Spacer(modifier = Modifier.height(8.dp))
                                        
                                        if (wrongs.isNotEmpty()) {
                                            Text(
                                                text = getAppString(isArabic, "Wrong Words Spoken:"),
                                                style = MaterialTheme.typography.bodyMedium,
                                                fontWeight = FontWeight.SemiBold,
                                                color = MaterialTheme.colorScheme.onErrorContainer
                                            )
                                            wrongs.forEach { analysis ->
                                                Row(
                                                    modifier = Modifier.padding(vertical = 4.dp),
                                                    verticalAlignment = Alignment.CenterVertically
                                                ) {
                                                    Icon(
                                                        imageVector = Icons.Filled.Close,
                                                        contentDescription = "Wrong Word",
                                                        tint = MaterialTheme.colorScheme.error,
                                                        modifier = Modifier.size(16.dp)
                                                    )
                                                    Spacer(modifier = Modifier.width(6.dp))
                                                    Text(
                                                        text = if (isArabic) "خطأ في كلمة \"${analysis.text}\"" else "Incorrect word: \"${analysis.text}\"",
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                                    )
                                                    if (analysis.tajweedRule != null) {
                                                        Spacer(modifier = Modifier.width(6.dp))
                                                        SuggestionChip(
                                                            onClick = {},
                                                            label = { Text("${getAppString(isArabic, "Tajweed: ")}${getAppString(isArabic, analysis.tajweedRule)}", fontSize = 10.sp) },
                                                            colors = SuggestionChipDefaults.suggestionChipColors(
                                                                containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.5f),
                                                                labelColor = MaterialTheme.colorScheme.onTertiaryContainer
                                                            )
                                                        )
                                                    }
                                                }
                                            }
                                            Spacer(modifier = Modifier.height(8.dp))
                                        }
                                        
                                        if (skippeds.isNotEmpty()) {
                                            Text(
                                                text = getAppString(isArabic, "Skipped Words:"),
                                                style = MaterialTheme.typography.bodyMedium,
                                                fontWeight = FontWeight.SemiBold,
                                                color = MaterialTheme.colorScheme.onErrorContainer
                                            )
                                            skippeds.forEach { analysis ->
                                                Row(
                                                    modifier = Modifier.padding(vertical = 4.dp),
                                                    verticalAlignment = Alignment.CenterVertically
                                                ) {
                                                    Icon(
                                                        imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                                                        contentDescription = "Skipped Word",
                                                        tint = Color(0xFFFB8C00),
                                                        modifier = Modifier.size(16.dp)
                                                    )
                                                    Spacer(modifier = Modifier.width(6.dp))
                                                    Text(
                                                        text = if (isArabic) "تخطي كلمة \"${analysis.text}\"" else "Skipped word: \"${analysis.text}\"",
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                                    )
                                                    if (analysis.tajweedRule != null) {
                                                        Spacer(modifier = Modifier.width(6.dp))
                                                        SuggestionChip(
                                                            onClick = {},
                                                            label = { Text("${getAppString(isArabic, "Tajweed: ")}${getAppString(isArabic, analysis.tajweedRule)}", fontSize = 10.sp) },
                                                            colors = SuggestionChipDefaults.suggestionChipColors(
                                                                containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.5f),
                                                                labelColor = MaterialTheme.colorScheme.onTertiaryContainer
                                                            )
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else if (detailedAnalysis.any { it.status == RecitationStatus.CORRECT }) {
                                Card(
                                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.2f)
                                    ),
                                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)),
                                    shape = RoundedCornerShape(16.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.padding(16.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Icon(
                                            imageVector = Icons.Filled.CheckCircle,
                                            contentDescription = "Perfect Recitation",
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                        Spacer(modifier = Modifier.width(12.dp))
                                        Text(
                                            text = getAppString(isArabic, "Excellent recitation so far! Keep going with perfect accuracy."),
                                            style = MaterialTheme.typography.bodyMedium,
                                            fontWeight = FontWeight.Medium,
                                            color = MaterialTheme.colorScheme.onPrimaryContainer
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        } // End of conditional for Memorize/Read

        Spacer(modifier = Modifier.height(6.dp))

        if (currentTab == "Memorize") {
        Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
            FloatingActionButton(
                onClick = {
                    if (speechRecognizer == null) {
                        errorMessage = "Speech recognition is not available on this device"
                        return@FloatingActionButton
                    }
                    if (isListening) {
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
                            if (recitationMode == "Tajweed" && tajweedPipelineHolder.running) {
                                stopTajweedSession()
                            } else {
                                speechRecognizer.stopListening()
                                isListening = false
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
                            
                            if (newCompletedCount > 0 || currentWrong > 0 || currentSkipped > 0 || currentTajweed > 0 || encounteredRulesList.isNotEmpty()) {
                                viewModel.incrementRecitedAyahs(newCompletedCount)
                                prefs.edit().putInt("recited_ayahs", viewModel.recitedAyahs).apply()
                                // Scope progress to the surah/range the user just recited -- so
                                // a goal scoped to "Surah 2, verses 1-5" only makes progress when
                                // the user recites Surah 2 in the [1, 5] range. If the user is
                                // on a different surah/range, scoped goals elsewhere stay idle.
                                val activeSurah = viewModel.selectedSurah?.number ?: 0
                                val activeRangeStart = viewModel.rangeStart
                                val activeRangeEnd = viewModel.rangeEnd
                                viewModel.triggerGoalRealProgress(
                                    recitedAyahsDelta = newCompletedCount,
                                    wrongDelta = currentWrong,
                                    skippedDelta = currentSkipped,
                                    tajweedDelta = currentTajweed,
                                    violatedRules = violatedRulesList,
                                    encounteredRules = encounteredRulesList,
                                    surahNumber = activeSurah,
                                    rangeStart = activeRangeStart,
                                    rangeEnd = activeRangeEnd
                                )
                            }
                            
                            // Safe Pause/Mistake handling: Trims spoken text down to only matched words so they can continue seamlessly
                            val matchedWordsList = targetOrig.take(matchedWordCount)
                            finalSpokenText = matchedWordsList.joinToString(" ")
                            partialSpokenText = ""
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
                        }
                    } else {
                        errorMessage = null
                        val hasPermission = androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.RECORD_AUDIO) == android.content.pm.PackageManager.PERMISSION_GRANTED
                        if (hasPermission) {
                            if (!isListening) {
                                // Memorize+Tajweed now uses the Sherpa-ONNX offline
                                // recognizer pipeline (single AudioRecord feeds both the
                                // Sherpa OnlineRecognizer and TajweedNet acoustic
                                // scorer). The platform SpeechRecognizer is NOT used in
                                // Tajweed mode -- on the user's Samsung A05s two HAL
                                // clients starve each other and the word-by-word
                                // highlight stops. Non-Tajweed modes keep the
                                // SpeechRecognizer path (TajweedNet has no companion
                                // transcript pipeline there because it doesn't need a
                                // transcript -- the practice recitation text is known).
                                if (recitationMode == "Tajweed" && tajweedPipelineHolder.pipeline != null) {
                                    // Start via the pipeline-aware session helper so the
                                    // hotwords bias (active ayah range) and the standard
                                    // state reset (spokenTimings, audio, etc.) all happen
                                    // in one place, mirroring the SpeechRecognizer path
                                    // above.
                                    startTajweedSession()
                                    if (isListening) {
                                        viewModel.preloadTajweedModel()
                                    }
                                } else if (speechRecognizer != null) {
                                    spokenTimings.clear()
                                    currentWordCount = 0
                                    recognizedAudioUri = null
                                    recognizedAudio = null
                                    recognizedAudioStartMs = System.currentTimeMillis()
                                    isListening = true
                                    startListening(speechRecognizer!!, context.packageName, onError = { msg -> errorMessage = if(msg.isEmpty()) null else msg; isListening = false })
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
                                    if (recitationMode == "Tajweed") {
                                        viewModel.preloadTajweedModel()
                                    }
                                } else {
                                    errorMessage = "Speech Recognizer not available"
                                }
                            }
                        } else {
                            permissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
                        }
                    }
                },
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
                                            finalSpokenText = ""
                                            partialSpokenText = ""
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
 * Mutable holder for the [com.example.sherpa.SingleAudioPipeline] created when
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
    var pipeline: com.example.sherpa.SingleAudioPipeline? by androidx.compose.runtime.mutableStateOf(null)

    /** Same semantics as [SingleAudioPipeline.running], null-safe. */
    val running: Boolean get() = pipeline?.running ?: false
    /** Null-safe passthrough for `pipeline.stop()`. */
    fun stop(): FloatArray = pipeline?.stop() ?: FloatArray(0)
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
            .background(Color(0xFF040D0E)),
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
                tint = Color(0xFF5EEAD4),
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
                    tint = Color(0xFF99F6E4),
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (isArabic) "إتقان" else "Itqān",
                    style = MaterialTheme.typography.headlineLarge.copy(
                        fontWeight = FontWeight.ExtraBold,
                        letterSpacing = 1.sp
                    ),
                    color = Color.White,
                    textAlign = TextAlign.Center
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = if (isArabic) "تسميع القرآن مع أحكام التجويد بدقة" else "AI-guided Quran Memorization & Tajweed",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFFBEC8C8),
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(48.dp))

            CircularProgressIndicator(
                color = Color(0xFF5EEAD4),
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
        // --- Real-time tajweed mistake popup (memorize+tajweed) ---
        "Tajweed Mistake" -> "خطأ في التجويد"
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
 * Resolves the per-rule verdict templates ("good_rule_application_template" /
 * "review_your_rule_template") with the already-localized rule name. Used by the Tajweed Rules
 * Coach so the verdict message reads naturally in Arabic ("تطبيق جيد لحكم قلقلة") instead of the
 * English "Good Qalqalah application".
 */
fun formatPerRuleVerdict(isArabic: Boolean, passed: Boolean, ruleName: String): String {
    val template = getAppString(
        isArabic,
        if (passed) "good_rule_application_template" else "review_your_rule_template"
    )
    val localizedRule = getAppString(isArabic, ruleName)
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
