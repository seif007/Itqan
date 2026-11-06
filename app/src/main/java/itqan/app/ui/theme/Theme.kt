package itqan.app.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = PrimaryColor,
    primaryContainer = PrimaryDark,
    onPrimary = Color(0xFF040D0E),
    secondary = PrimaryLight,
    background = BackgroundDark,
    surface = SurfaceDark,
    surfaceVariant = Color(0xFF11292C),
    onBackground = TextPrimary,
    onSurface = TextPrimary,
    onSurfaceVariant = TextSecondary
)

private val LightColorScheme = lightColorScheme(
    primary = PrimaryDark,
    onPrimary = Color.White,
    primaryContainer = PrimaryLight,
    onPrimaryContainer = Color(0xFF003731),
    secondary = PrimaryDark,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFCCFBF1),
    onSecondaryContainer = Color(0xFF003731),
    background = Color(0xFFF1F5F5),
    onBackground = BackgroundDark,
    surface = Color.White,
    onSurface = BackgroundDark,
    surfaceVariant = Color(0xFFE2E8E8),
    onSurfaceVariant = SurfaceDark,
    error = Color(0xFFB3261E),
    onError = Color.White,
    errorContainer = Color(0xFFF9DEDC),
    onErrorContainer = Color(0xFF410E0B),
    outline = Color(0xFF6F7979),
    outlineVariant = Color(0xFFBFC8C8)
)

@Composable
fun ItqanTheme(
    darkTheme: Boolean = true, // Default to dark theme to ensure specific background color
    dynamicColor: Boolean = false, // Disable dynamic colors to enforce the specific scheme
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(colorScheme = colorScheme, typography = Typography, content = content)
}
