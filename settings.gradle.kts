pluginManagement {
  repositories {
    google {
      content {
        includeGroupByRegex("com\\.android.*")
        includeGroupByRegex("com\\.google.*")
        includeGroupByRegex("androidx.*")
      }
    }
    mavenCentral()
    gradlePluginPortal()
  }
}

plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0" }

dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
    // Local AAR repository for vendored prebuilt libraries (sherpa-onnx, etc.).
    // Prebuilt `*.aar` files dropped into `app/libs/` are resolved here so we
    // don't depend on Maven Central availability for binaries that must ship
    // inside the APK (e.g. the Sherpa-ONNX JNI bridge that exposes
    // libsherpa-onnx-jni.so for the offline streaming recognizer). Coordinates
    // `group = null, name = "<file>", ext = "aar"` resolve to
    // `app/libs/<file>.aar` here.
    flatDir {
      dirs("app/libs")
    }
  }
}

rootProject.name = "My Application"

include(":app")
