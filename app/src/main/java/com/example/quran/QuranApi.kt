package com.example.quran

import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import retrofit2.http.GET
import retrofit2.http.Path
import retrofit2.http.Query
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import com.squareup.moshi.JsonClass
import com.squareup.moshi.Json

@JsonClass(generateAdapter = true)
data class SurahListResponse(
    val code: Int,
    val status: String,
    val data: List<SurahData>
)

@JsonClass(generateAdapter = true)
data class SurahData(
    val number: Int,
    val name: String,
    val englishName: String,
    val numberOfAyahs: Int
)

@JsonClass(generateAdapter = true)
data class SurahDetailResponse(
    val code: Int,
    val status: String,
    val data: SurahDetailData
)

@JsonClass(generateAdapter = true)
data class SurahDetailData(
    val number: Int,
    val name: String,
    val ayahs: List<Ayah>
)

@JsonClass(generateAdapter = true)
data class Ayah(
    val number: Int,
    val text: String,
    val numberInSurah: Int,
    val audio: String? = null
)

// --- QuranCDN Word Timing API ---

@JsonClass(generateAdapter = true)
data class QuranCdnResponse(
    @field:Json(name = "audio_files") val audioFiles: List<AudioFile>
)

@JsonClass(generateAdapter = true)
data class AudioFile(
    @field:Json(name = "verse_timings") val verseTimings: List<VerseTiming>
)

@JsonClass(generateAdapter = true)
data class VerseTiming(
    @field:Json(name = "verse_key") val verseKey: String,
    @field:Json(name = "verse_id") val verseId: Int,
    @field:Json(name = "segments") val segments: List<List<Double>>
)

interface QuranApi {
    @GET("surah")
    suspend fun getSurahs(): SurahListResponse

    @GET("surah/{number}/ar.alafasy")
    suspend fun getSurahDetail(@Path("number") number: Int): SurahDetailResponse

    companion object {
        private val moshi = Moshi.Builder()
            .add(KotlinJsonAdapterFactory())
            .build()
            
        val instance: QuranApi by lazy {
            Retrofit.Builder()
                .baseUrl("https://api.alquran.cloud/v1/")
                .addConverterFactory(MoshiConverterFactory.create(moshi))
                .build()
                .create(QuranApi::class.java)
        }
    }
}

interface QuranCdnApi {
    @GET("audio/reciters/7/audio_files")
    suspend fun getAudioTimings(
        @Query("chapter") chapter: Int,
        @Query("segments") segments: Boolean = true
    ): QuranCdnResponse

    companion object {
        private val moshi = Moshi.Builder()
            .add(KotlinJsonAdapterFactory())
            .build()
            
        val instance: QuranCdnApi by lazy {
            Retrofit.Builder()
                .baseUrl("https://api.qurancdn.com/api/qdc/")
                .addConverterFactory(MoshiConverterFactory.create(moshi))
                .build()
                .create(QuranCdnApi::class.java)
        }
    }
}
