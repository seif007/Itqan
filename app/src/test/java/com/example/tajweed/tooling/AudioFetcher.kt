package com.example.tajweed.tooling

import javazoom.jl.decoder.Bitstream
import javazoom.jl.decoder.Decoder
import javazoom.jl.decoder.Header
import javazoom.jl.decoder.SampleBuffer
import java.io.InputStream
import java.net.URL

object AudioFetcher {

    fun downloadAndDecodeMp3(urlStr: String): FloatArray {
        val bytes = URL(urlStr).openStream().use { input ->
            input.readBytes()
        }
        return decodeMp3(bytes)
    }

    private fun decodeMp3(mp3Data: ByteArray): FloatArray {
        val inputStream = java.io.ByteArrayInputStream(mp3Data)
        val bitstream = Bitstream(inputStream)
        val decoder = Decoder()

        val pcmList = mutableListOf<Float>()
        var header = bitstream.readFrame()
        while (header != null) {
            val sampleBuffer = decoder.decodeFrame(header, bitstream) as SampleBuffer
            val samples = sampleBuffer.buffer
            val sampleCount = sampleBuffer.bufferLength
            
            for (i in 0 until sampleCount) {
                // Short to normalized Float
                val s = samples[i]
                pcmList.add(s / 32768.0f)
            }
            bitstream.closeFrame()
            header = bitstream.readFrame()
        }

        return pcmList.toFloatArray()
    }
}
