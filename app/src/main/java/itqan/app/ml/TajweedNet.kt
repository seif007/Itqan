package itqan.app.ml

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.tanh

/**
 * Native-Kotlin port of TajweedNet (src/train.py).
 * Architecture: log-mel(40,T) -> Conv1d x3 (+ReLU, MaxPool2) -> BiGRU(64)
 * -> temporal mean-pool -> per-rule Linear heads. Each head input = [pooled(128)]
 * + (dur if MADD) + (ev[5]).
 *
 * Weights loaded from assets/model_weights.json (exported by src/export_weights.py).
 * Math mirrors PyTorch exactly (see parity test in the app's unit test).
 */
class TajweedNet(private val modelJson: JSONObject) {

    private val meta = modelJson.getJSONObject("meta")
    private val W = modelJson.getJSONObject("weights")
    private val rules: List<String> = (meta.getJSONArray("rules")).let { a ->
        List(a.length()) { a.getString(it) }
    }
    private val convChannels = meta.getInt("conv_channels")
    private val gruUnits = meta.getInt("gru_units")

    // Pre-load conv weights (order conv0, conv1, conv2)
    private data class ConvW(val w: Array<FloatArray>, val b: FloatArray)
    private val convs = List(3) { i ->
        val wj = W.getJSONArray("conv${i}_W")
        val bj = W.getJSONArray("conv${i}_b")
        val outCh = bj.length()
        val inCh = wj.getJSONArray(0).length()
        val k = wj.getJSONArray(0).getJSONArray(0).length()
        val w = Array(outCh) { FloatArray(inCh * k) }
        for (oc in 0 until outCh) {
            val row = wj.getJSONArray(oc)
            for (ic in 0 until inCh) {
                val kr = row.getJSONArray(ic)
                for (ki in 0 until k) w[oc][ic * k + ki] = kr.getDouble(ki).toFloat()
            }
        }
        val b = FloatArray(outCh) { bj.getDouble(it).toFloat() }
        ConvW(w, b)
    }

    // GRU weights -- FORWARD direction
    private val gruWih = load2D("gru_Wih")   // (3*h, input=convCh)
    private val gruWhh = load2D("gru_Whh")   // (3*h, h)
    private val gruBih = load1D("gru_bih")
    private val gruBhh = load1D("gru_bhh")
    // GRU weights -- BACKWARD (reverse) direction
    private val gruWihRev = load2D("gru_Wih_rev")
    private val gruWhhRev = load2D("gru_Whh_rev")
    private val gruBihRev = load1D("gru_bih_rev")
    private val gruBhhRev = load1D("gru_bhh_rev")

    private fun load2D(key: String): Array<FloatArray> {
        val a = W.getJSONArray(key)
        return Array(a.length()) { i ->
            val row = a.getJSONArray(i)
            FloatArray(row.length()) { row.getDouble(it).toFloat() }
        }
    }
    private fun load1D(key: String): FloatArray {
        val a = W.getJSONArray(key)
        // Head weights are exported as shape (1, in) -> JSON [[...]]; flatten.
        val first = a.get(0)
        return if (first is JSONArray) {
            val inner = a.getJSONArray(0)
            FloatArray(inner.length()) { inner.getDouble(it).toFloat() }
        } else {
            FloatArray(a.length()) { a.getDouble(it).toFloat() }
        }
    }

    /** @param logmel [N_MELS][MAX_FRAMES], dur scalar, ev[5]. Returns rule->logit. */
    fun forward(logmel: Array<FloatArray>, dur: Float, ev: FloatArray): Map<String, Float> {
        // x: (n_mels, T) -> conv over "channels" = n_mels
        var c = logmel
        for (i in 0..2) {
            c = conv1dRelu(c, convs[i])
            c = maxpool2(c)
        }
        // GRU expects (T, convCh)
        val T = c[0].size
        val C = c.size
        val h = FloatArray(gruUnits) // forward hidden
        val fwd = Array(T) { FloatArray(gruUnits) }
        for (t in 0 until T) {
            val xt = FloatArray(C) { c[it][t] }
            val nf = gruStep(xt, h, gruWih, gruWhh, gruBih, gruBhh)
            fwd[t] = nf.copyOf()
            for (j in h.indices) h[j] = nf[j]
        }
        // backward direction: process the sequence in REVERSE time order,
        // using the reverse GRU weights.
        val hRev = FloatArray(gruUnits)
        val bwd = Array(T) { FloatArray(gruUnits) }
        for (t in T - 1 downTo 0) {
            val xt = FloatArray(C) { c[it][t] }
            val nb = gruStep(xt, hRev, gruWihRev, gruWhhRev, gruBihRev, gruBhhRev)
            bwd[t] = nb.copyOf()
            for (j in hRev.indices) hRev[j] = nb[j]
        }
        // temporal mean-pool of concatenated [fwd; bwd] -> (2*gruUnits)
        val pooled = FloatArray(2 * gruUnits)
        for (t in 0 until T) {
            for (j in 0 until gruUnits) pooled[j] += fwd[t][j]
            for (j in 0 until gruUnits) pooled[gruUnits + j] += bwd[t][j]
        }
        val inv = 1.0f / T
        for (j in pooled.indices) pooled[j] *= inv

        // heads
        val out = LinkedHashMap<String, Float>()
        for (r in rules) {
            val parts = ArrayList<FloatArray>()
            parts.add(pooled)
            if (r == "MADD") parts.add(floatArrayOf(dur))
            parts.add(ev)
            val feat = concat(parts)
            val hw = load1D("head_${r}_W")
            val hb = load1D("head_${r}_b")[0]
            var acc = hb
            for (i in feat.indices) acc += hw[i] * feat[i]
            out[r] = acc
        }
        return out
    }

    fun sigmoidLogits(logits: Map<String, Float>): Map<String, Float> =
        logits.mapValues { 1f / (1f + exp(-it.value)) }

    // ---------------- low-level ops ----------------
    private fun conv1dRelu(x: Array<FloatArray>, cw: ConvW): Array<FloatArray> {
        // x: (inCh, Tin) -> (outCh, Tout) with k=3,padding=1,stride=1
        val inCh = x.size
        val Tin = x[0].size
        val outCh = cw.w.size
        val Tout = Tin // same because padding=1,k=3
        val out = Array(outCh) { FloatArray(Tout) }
        for (oc in 0 until outCh) {
            for (t in 0 until Tout) {
                var s = cw.b[oc]
                // kernel indices -1,0,1
                for (ki in 0..2) {
                    val tp = t + ki - 1
                    if (tp in 0 until Tin) {
                        for (ic in 0 until inCh) s += cw.w[oc][ic * 3 + ki] * x[ic][tp]
                    }
                }
                out[oc][t] = if (s > 0f) s else 0f // ReLU
            }
        }
        return out
    }

    private fun maxpool2(x: Array<FloatArray>): Array<FloatArray> {
        val C = x.size
        val Tin = x[0].size
        val Tout = Tin / 2
        val out = Array(C) { FloatArray(Tout) }
        for (c in 0 until C) {
            for (t in 0 until Tout) {
                out[c][t] = maxOf(x[c][t * 2], x[c][t * 2 + 1])
            }
        }
        return out
    }

    private fun gruStep(x: FloatArray, h: FloatArray, Wih: Array<FloatArray>, Whh: Array<FloatArray>,
                        bih: FloatArray, bhh: FloatArray): FloatArray {
        val hDim = h.size
        val xDim = x.size
        val ih = FloatArray(3 * hDim)
        val hh = FloatArray(3 * hDim)
        for (g in 0 until 3 * hDim) {
            var a = bih[g]
            for (k in 0 until xDim) a += Wih[g][k] * x[k]
            ih[g] = a
            var b = bhh[g]
            for (k in 0 until hDim) b += Whh[g][k] * h[k]
            hh[g] = b
        }
        val r = FloatArray(hDim); val z = FloatArray(hDim); val ng = FloatArray(hDim)
        // Gate slicing: PyTorch GRU gates are (reset, update, new) at [0:h],[h:2h],[2h:3h]
        for (j in 0 until hDim) {
            r[j] = sigmoid(ih[j] + hh[j])
            z[j] = sigmoid(ih[hDim + j] + hh[hDim + j])
            val nInput = ih[2 * hDim + j] + r[j] * hh[2 * hDim + j]
            ng[j] = tanh(nInput)
        }
        val out = FloatArray(hDim)
        for (j in 0 until hDim) out[j] = (1 - z[j]) * ng[j] + z[j] * h[j]
        return out
    }

    private fun sigmoid(v: Float) = 1f / (1f + exp(-v))

    private fun concat(parts: List<FloatArray>): FloatArray {
        var n = 0
        for (p in parts) n += p.size
        val out = FloatArray(n)
        var o = 0
        for (p in parts) { p.copyInto(out, o); o += p.size }
        return out
    }
}
