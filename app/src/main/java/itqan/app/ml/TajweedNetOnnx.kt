package itqan.app.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlin.math.exp

/**
 * ONNX-Runtime backed TajweedNet. Runs the SAME graph as the pure-Kotlin
 * [TajweedNet] (exported from the identical PyTorch checkpoint via
 * training/src/export_onnx.py) but on onnxruntime's optimized native kernels.
 *
 * Why: the Kotlin forward pass costs ~370 ms per word on the Samsung A05s
 * (scalar conv + GRU loops), which stalled the live tajweed analysis for tens
 * of seconds. The same graph on ORT runs in ~10-20 ms, making per-word
 * acoustic scoring cheap enough for the deferred (~2 s) tajweed loop.
 *
 * Inputs (float32, fixed batch 1):
 *   x   : (1, 40, 200)  log-mel, padded/trimmed to MAX_FRAMES
 *   dur : (1, 1)        normalized duration scalar
 *   ev  : (1, 5)        clip-level acoustic-evidence vector
 * Output:
 *   logits : (1, 8)     per-rule logits in [TajweedFeatures.RULES] order
 *
 * The model file ships at assets/tajweed_net.onnx (~196 KB). Construction
 * loads it once; [forward] is the hot path and allocates only the three input
 * tensors + one output read per call.
 */
class TajweedNetOnnx private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession
) : AutoCloseable {

    /** @param logmel [N_MELS][MAX_FRAMES], dur scalar, ev[5]. Returns rule->logit. */
    fun forward(logmel: Array<FloatArray>, dur: Float, ev: FloatArray): Map<String, Float> {
        val nMels = logmel.size
        val frames = if (nMels > 0) logmel[0].size else 0
        // Flatten (nMels, frames) -> (1, nMels, frames) row-major.
        val flat = FloatArray(nMels * frames)
        var k = 0
        for (i in 0 until nMels) {
            val row = logmel[i]
            System.arraycopy(row, 0, flat, k, frames)
            k += frames
        }
        val xShape = longArrayOf(1L, nMels.toLong(), frames.toLong())
        val durArr = floatArrayOf(dur)
        val evArr = ev.copyOf(5)

        OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(flat), xShape).use { xT ->
            OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(durArr), longArrayOf(1L, 1L)).use { durT ->
                OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(evArr), longArrayOf(1L, 5L)).use { evT ->
                    session.run(mapOf("x" to xT, "dur" to durT, "ev" to evT)).use { result ->
                        val out = result.get(0) as OnnxTensor
                        val logits = (out.value as Array<FloatArray>)[0]
                        val rules = TajweedFeatures.RULES
                        val map = HashMap<String, Float>(rules.size)
                        for (i in rules.indices) {
                            if (i < logits.size) map[rules[i]] = logits[i]
                        }
                        return map
                    }
                }
            }
        }
    }

    /** Same sigmoid mapping as [TajweedNet.sigmoidLogits]. */
    fun sigmoidLogits(logits: Map<String, Float>): Map<String, Float> =
        logits.mapValues { 1f / (1f + exp(-it.value)) }

    override fun close() {
        try { session.close() } catch (_: Throwable) {}
    }

    companion object {
        /**
         * Load the ONNX model from assets. Returns null if the asset is missing
         * or ORT fails to build the session (caller falls back to the Kotlin net).
         * [threads] caps intra-op parallelism to keep the footprint small on the
         * low-RAM A05s (matches the STT engine's 2-thread setting).
         */
        fun create(context: android.content.Context, threads: Int = 2): TajweedNetOnnx? {
            return try {
                val bytes = context.assets.open("tajweed_net.onnx").use { it.readBytes() }
                val env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(threads) }
                val session = env.createSession(bytes, opts)
                TajweedNetOnnx(env, session)
            } catch (t: Throwable) {
                android.util.Log.w("TajweedNetOnnx", "failed to load tajweed_net.onnx; falling back to Kotlin net", t)
                null
            }
        }
    }
}
