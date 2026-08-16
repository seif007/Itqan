plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.compose)
  alias(libs.plugins.google.devtools.ksp)
  alias(libs.plugins.roborazzi)
  alias(libs.plugins.secrets)
}

android {
  namespace = "com.example"
  compileSdk { version = release(36) { minorApiLevel = 1 } }

  defaultConfig {
    applicationId = "com.aistudio.recitationpractice.quranstt"
    minSdk = 24
    targetSdk = 36
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
  }

  signingConfigs {
    create("release") {
      val keystorePath = System.getenv("KEYSTORE_PATH") ?: "${rootDir}/my-upload-key.jks"
      storeFile = file(keystorePath)
      storePassword = System.getenv("STORE_PASSWORD")
      keyAlias = "upload"
      keyPassword = System.getenv("KEY_PASSWORD")
    }
    create("debugConfig") {
      storeFile = file("${rootDir}/debug.keystore")
      storePassword = "android"
      keyAlias = "androiddebugkey"
      keyPassword = "android"
    }
  }

  buildTypes {
    release {
      isCrunchPngs = false
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
      signingConfig = signingConfigs.getByName("release")
    }
    debug {
      signingConfig = signingConfigs.getByName("debugConfig")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
  }
  buildFeatures {
    compose = true
    buildConfig = true
  }
  testOptions {
    unitTests {
      isIncludeAndroidResources = true
      // The PC-side harness loads the 166 MB ONNX graph (and the Android
      // artifact's own shared libs must not shadow the desktop natives).
      // Heap is 1g, not 2g/4g: the ONNX graph lives in NATIVE memory, and on
      // this 8 GB box (only ~2 GB free after Chrome + ZCode + Windows) a 2g
      // test JVM plus the Gradle daemon froze the machine. The harness keeps
      // its per-test audio buffers small (one surah at a time), so 1g heap is
      // plenty for the JVM-side objects.
      //
      // forkEvery=1 + maxParallelForks=1: each test CLASS runs in its own JVM,
      // sequentially. The ~167 MB ONNX native allocation is freed when that
      // JVM exits, so native memory never accumulates across the two heavy
      // harness classes — the accumulation is what froze the 8 GB box.
      all {
        it.maxHeapSize = "1g"
        it.systemProperty("file.encoding", "UTF-8")
        it.setForkEvery(1)
        it.maxParallelForks = 1
      }
    }
  }
}

// Configure the Secrets Gradle Plugin to use .env and .env.example files
// to match the convention used in Web projects.
secrets {
  propertiesFileName = ".env"
  defaultPropertiesFileName = ".env.example"
}

// Some unused dependencies are commented out below instead of being removed.
// This makes it easy to add them back in the future if needed.
dependencies {
  implementation(platform(libs.androidx.compose.bom))
  implementation(platform(libs.firebase.bom))
  // implementation(libs.accompanist.permissions)
  implementation(libs.androidx.activity.compose)
  // implementation(libs.androidx.camera.camera2)
  // implementation(libs.androidx.camera.core)
  // implementation(libs.androidx.camera.lifecycle)
  // implementation(libs.androidx.camera.view)
  implementation(libs.androidx.compose.material.icons.core)
  implementation(libs.androidx.compose.material.icons.extended)
  implementation(libs.androidx.compose.material3)
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.graphics)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.core.ktx)
  // implementation(libs.androidx.datastore.preferences)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  // implementation(libs.androidx.navigation.compose)
  implementation(libs.androidx.room.ktx)
  implementation(libs.androidx.room.runtime)
  // implementation(libs.coil.compose)
  implementation(libs.converter.moshi)
  // implementation(libs.firebase.ai)
  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.kotlinx.coroutines.core)
  implementation(libs.logging.interceptor)
  implementation(libs.moshi.kotlin)
  implementation(libs.okhttp)
  // implementation(libs.play.services.location)
  implementation(libs.retrofit)
  testImplementation(libs.androidx.compose.ui.test.junit4)
  testImplementation(libs.androidx.core)
  testImplementation(libs.androidx.junit)
  testImplementation(libs.junit)
  testImplementation(libs.kotlinx.coroutines.test)
  testImplementation(libs.robolectric)
  testImplementation(libs.roborazzi)
  testImplementation(libs.roborazzi.compose)
  testImplementation(libs.roborazzi.junit.rule)
  androidTestImplementation(platform(libs.androidx.compose.bom))
  androidTestImplementation(libs.androidx.compose.ui.test.junit4)
  androidTestImplementation(libs.androidx.espresso.core)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.runner)
  debugImplementation(libs.androidx.compose.ui.test.manifest)
  debugImplementation(libs.androidx.compose.ui.tooling)
  "ksp"(libs.androidx.room.compiler)
  "ksp"(libs.moshi.kotlin.codegen)
testImplementation("javazoom:jlayer:1.0.1")
  // PC-side end-to-end STT harness (PcQuranSttEndToEndTest): loads the SAME
  // quran_stt ONNX model with the DESKTOP onnxruntime so unit tests can run the
  // real recognizer on real audio without a device. The Android artifact stays
  // on the main compile classpath (identical ai.onnxruntime API); at test
  // RUNTIME the Android natives would shadow the desktop ones, so the Android
  // artifact is excluded from the test runtime classpath below.
  testImplementation("com.microsoft.onnxruntime:onnxruntime:1.17.1")

  // ONNX Runtime (Android) drives the FastConformer CTC model from
  // Saboorhsn/quran-stt-onnx bundled in assets/quran_stt/. Runs on the CPU EP.
  implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")
}

// Keep the Android ORT natives OFF the unit-test runtime so the desktop
// onnxruntime (added above as a testImplementation) loads ITS natives and can
// run the same quran_stt model on the PC. (testRuntimeClasspath is created
// lazily by AGP, hence configureEach.)
configurations.configureEach {
  if (name == "testRuntimeClasspath") {
    exclude(group = "com.microsoft.onnxruntime", module = "onnxruntime-android")
  }
}

