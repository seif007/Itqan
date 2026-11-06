package itqan.app.ui

import javazoom.jl.decoder.Bitstream
import javazoom.jl.decoder.Decoder
import javazoom.jl.decoder.SampleBuffer
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.File
import kotlin.math.min

/** PC-side tool: mp3 -> 16 kHz mono 16-bit LE PCM for on-device injection. */
class PcmGenTool {
    @Test
    fun generateBaqarahPcm() {
        val mp3 = File("C:/Users/seifg/AppData/Local/Temp/stt_e2e/baqarah_full.mp3")
        val out = File("C:/Users/seifg/AppData/Local/Temp/stt_e2e/baqarah_16k.pcm")
        val bitstream = Bitstream(ByteArrayInputStream(mp3.readBytes()))
        val decoder = Decoder()
        val pcm44 = ArrayList<Float>((mp3.length() / 2).toInt())
        var rate = 0
        var header = bitstream.readFrame()
        while (header != null) {
            if (rate == 0) rate = decoder.outputFrequency
            val sb = decoder.decodeFrame(header, bitstream) as SampleBuffer
            val b = sb.buffer
            if (sb.channelCount == 1) {
                for (i in 0 until sb.bufferLength) pcm44.add(b[i] / 32768f)
            } else {
                var i = 0
                while (i + 1 < sb.bufferLength) {
                    pcm44.add(((b[i] + b[i + 1]) * 0.5f) / 32768f)
                    i += 2
                }
            }
            bitstream.closeFrame()
            header = bitstream.readFrame()
        }
        println("PCMGEN decoded ${pcm44.size} samples @ $rate Hz = ${pcm44.size / rate.toFloat()}s")
        val ratio = rate.toDouble() / 16000.0
        val out16 = FloatArray((pcm44.size / ratio).toInt())
        for (i in out16.indices) {
            val pos = i * ratio
            val i0 = pos.toInt()
            val i1 = min(i0 + 1, pcm44.size - 1)
            val frac = (pos - i0).toFloat()
            out16[i] = pcm44[i0] * (1f - frac) + pcm44[i1] * frac
        }
        java.io.DataOutputStream(java.io.FileOutputStream(out)).use { dos ->
            for (v in out16) {
                val s = (v.coerceIn(-1f, 1f) * 32767f).toInt()
                dos.writeByte(s and 0xFF)
                dos.writeByte((s shr 8) and 0xFF)
            }
        }
        println("PCMGEN wrote ${out.length()} bytes = ${out.length() / 32000f}s @16k mono -> ${out.absolutePath}")
    }
}
