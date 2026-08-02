package com.example

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.unit.dp
import com.example.ml.RecitationAnalyzer
import com.example.ui.RecitationApp
import com.example.ui.theme.MyApplicationTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    companion object {
        var globalCrashMessage by androidx.compose.runtime.mutableStateOf<String?>(null)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Warm up the TajweedNet model on a background thread so the ~1 MB weights
        // are parsed once, up front, instead of on the first recitation. This does
        // NOT open the microphone or touch the speech recognizer — it only reads the
        // asset JSON — so the single-mic capture path is unaffected.
        CoroutineScope(Dispatchers.IO).launch {
            try {
                RecitationAnalyzer.get(applicationContext)
            } catch (e: Throwable) {
                android.util.Log.e("TajweedNet", "Model preload failed", e)
            }
        }

        val prefs = getSharedPreferences("crash_prefs", android.content.Context.MODE_PRIVATE)
        globalCrashMessage = prefs.getString("crash_trace", null)
        prefs.edit().remove("crash_trace").apply()
        
        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            try {
                val trace = android.util.Log.getStackTraceString(e)
                android.util.Log.e("FATAL_ERROR", trace)
                prefs.edit().putString("crash_trace", e.javaClass.name + "\n" + e.message + "\n" + trace).commit()
            } catch (ex: Throwable) {
                android.util.Log.e("FATAL_ERROR", "Failed to write crash trace to prefs", ex)
            }
            defaultHandler?.uncaughtException(t, e)
        }

        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    containerColor = MaterialTheme.colorScheme.background
                ) { innerPadding ->
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                Brush.radialGradient(
                                    colors = listOf(
                                        MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
                                        Color.Transparent
                                    ),
                                    radius = 1000f
                                )
                            )
                    ) {
                        if (globalCrashMessage != null) {
                            androidx.compose.foundation.lazy.LazyColumn(modifier = Modifier.fillMaxSize().padding(16.dp)) {
                                item {
                                    androidx.compose.material3.Text(
                                        text = "CRASH:\n$globalCrashMessage",
                                        color = Color.Red
                                    )
                                }
                            }
                        } else {
                            RecitationApp(modifier = Modifier.padding(innerPadding))
                        }
                    }
                }
            }
        }
    }
}
