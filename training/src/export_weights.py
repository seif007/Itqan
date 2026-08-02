#!/usr/bin/env python3
"""
export_weights.py - Dump TajweedNet weights (from model_best.pt) to a compact
JSON asset for the Android native-Kotlin port, and self-check numeric parity.

Why JSON (not .pt/.tflite): the Android app runs a hand-ported Kotlin replica of
TajweedNet, so it only needs the raw tensors. JSON is trivially parseable with
org.json and keeps the app dependency-free. The model is tiny (<1 MB of weights).

Outputs:
  dataset/model_weights.json   -> the asset to copy into app/src/main/assets/
  (and prints a known-answer parity check using the same forward math)

Run:  python src/export_weights.py
"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn

# ---- replicate the exact model definition from train.py (so we can load + run) ----
SR = 16000
N_MELS = 40
WIN_MS = 25
HOP_MS = 10
N_FFT = int(SR * WIN_MS / 1000)
HOP = int(SR * HOP_MS / 1000)
MAX_FRAMES = 200
CONV_CHANNELS = 32
GRU_UNITS = 64
RULES = ["MADD", "GHUNNAH", "QALQALAH", "IDGHAM", "IKHFA",
         "ALLAM", "MEEM_SAKINAH", "IZHAR"]


class TajweedNet(nn.Module):
    def __init__(self, n_mels=N_MELS, conv_ch=CONV_CHANNELS, gru=GRU_UNITS,
                 rules=RULES, dropout=0.2, madd_rule="MADD", ev_in=5):
        super().__init__()
        self.n_mels = n_mels
        self.madd_rule = madd_rule
        self.ev_in = ev_in
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_ch, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_ch, conv_ch, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_ch, conv_ch, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(conv_ch, gru, batch_first=True, bidirectional=True)
        self.head_in = gru * 2
        self.heads = nn.ModuleDict({
            r: nn.Linear(self.head_in + (1 if r == madd_rule else 0) + ev_in, 1)
            for r in rules})
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dur=None, ev=None):
        c = self.conv(x)
        c = c.permute(0, 2, 1)
        out, _ = self.gru(c)
        pooled = out.mean(dim=1)
        pooled = self.drop(pooled)
        logits = {}
        for r in self.heads:
            parts = [pooled]
            if r == self.madd_rule and dur is not None:
                parts.append(dur.view(-1, 1))
            if ev is not None:
                parts.append(ev)
            logits[r] = self.heads[r](torch.cat(parts, dim=1)).squeeze(-1)
        return logits


def to_list(t):
    return t.detach().cpu().numpy().tolist()


def conv_weights(conv_seq, prefix):
    """Extract weights for the 3 Conv1d layers in order."""
    out = {}
    ci = 0
    for layer in conv_seq:
        if isinstance(layer, nn.Conv1d):
            w = layer.weight.data  # (out_ch, in_ch, k)
            b = layer.bias.data    # (out_ch,)
            out[f"{prefix}conv{ci}_W"] = to_list(w)
            out[f"{prefix}conv{ci}_b"] = to_list(b)
            ci += 1
    return out


def gru_weights(gru, prefix):
    """PyTorch bidirectional GRU has TWO sets of weights:
       forward : weight_ih_l0 / weight_hh_l0 / bias_ih_l0 / bias_hh_l0
       backward: weight_ih_l0_reverse / weight_hh_l0_reverse / bias_ih_l0_reverse
                 / bias_hh_l0_reverse
       Each set: (3*h, input) for ih, (3*h, h) for hh, (3*h,) for biases.
       Gate order is (W_ir, W_iz, W_in) = (reset, update, new).  We export both
       sets so the Kotlin port can implement both GRU directions."""
    g = {}
    g[f"{prefix}gru_Wih"] = to_list(gru.weight_ih_l0)
    g[f"{prefix}gru_Whh"] = to_list(gru.weight_hh_l0)
    g[f"{prefix}gru_bih"] = to_list(gru.bias_ih_l0)
    g[f"{prefix}gru_bhh"] = to_list(gru.bias_hh_l0)
    g[f"{prefix}gru_Wih_rev"] = to_list(gru.weight_ih_l0_reverse)
    g[f"{prefix}gru_Whh_rev"] = to_list(gru.weight_hh_l0_reverse)
    g[f"{prefix}gru_bih_rev"] = to_list(gru.bias_ih_l0_reverse)
    g[f"{prefix}gru_bhh_rev"] = to_list(gru.bias_hh_l0_reverse)
    return g


def head_weights(heads, rules):
    out = {}
    for r in rules:
        lin = heads[r]
        out[f"head_{r}_W"] = to_list(lin.weight.data)   # (1, in)
        out[f"head_{r}_b"] = to_list(lin.bias.data)     # (1,)
    return out


def main():
    ds = os.path.join(HERE, "..", "dataset")
    best = os.path.join(ds, "model_best.pt")
    if not os.path.exists(best):
        print("ERROR: model_best.pt not found at", best)
        sys.exit(1)
    model = TajweedNet(rules=RULES, dropout=0.0)
    model.load_state_dict(torch.load(best, map_location="cpu"))
    model.eval()

    sd = model.state_dict()
    print("[export] loaded state_dict with %d tensors" % len(sd))

    weights = {}
    weights.update(conv_weights(model.conv, ""))
    weights.update(gru_weights(model.gru, ""))
    weights.update(head_weights(model.heads, RULES))

    # metadata for the Kotlin port
    meta = {
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop": HOP,
        "max_frames": MAX_FRAMES,
        "conv_channels": CONV_CHANNELS,
        "gru_units": GRU_UNITS,
        "sr": SR,
        "rules": RULES,
        "madd_rule": "MADD",
        "ev_in": 5,
        "note": "Weights exported from model_best.pt. Conv order: conv0,conv1,conv2. "
                "GRU gates (reset,update,new). Logits = sum(w_i * x_i) + b. "
                "MADD head input appended with dur; all heads appended with ev[5].",
    }

    out = {"meta": meta, "weights": weights}
    out_path = os.path.join(ds, "model_weights.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    size = os.path.getsize(out_path)
    print("[export] wrote", out_path, "(%.1f KB)" % (size / 1024))

    # ---- known-answer parity self-check (pure-python forward replica) ----
    print("[export] running parity self-check on random inputs...")
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((2, N_MELS, MAX_FRAMES)).astype(np.float32))
    dur = torch.from_numpy(rng.random((2, 1)).astype(np.float32))
    ev = torch.from_numpy(rng.standard_normal((2, 5)).astype(np.float32))

    with torch.no_grad():
        ref = model(x, dur, ev)
    ref_out = {r: float(ref[r][0]) for r in RULES}

    # independent numpy forward using the exported weights (mirror Kotlin math)
    def np_forward(xb, db, eb):
        # xb: (n_mels, T)
        def conv1d(xin, W, b, k=3, pad=1, stride=1):
            cin = xin.shape[0]; T = xin.shape[1]
            Tout = (T + 2 * pad - k) // stride + 1
            padded = np.pad(xin, ((0, 0), (pad, pad)), mode="constant")
            out = np.zeros((W.shape[0], Tout), dtype=np.float32)
            for co in range(W.shape[0]):
                for ti in range(Tout):
                    for ki in range(k):
                        for ci_ in range(cin):
                            out[co, ti] += W[co, ci_, ki] * padded[ci_, ti + ki]
                    out[co, ti] += b[co]
            return np.maximum(out, 0.0)  # ReLU
        def maxpool(xin, p=2):
            cin = xin.shape[0]; T = xin.shape[1]
            Tout = T // p
            out = np.zeros((cin, Tout), dtype=np.float32)
            for c in range(cin):
                for ti in range(Tout):
                    out[c, ti] = np.max(xin[c, ti * p:(ti + 1) * p])
            return out
        c = xb
        c = conv1d(c, np.array(weights["conv0_W"]), np.array(weights["conv0_b"]))
        c = maxpool(c)
        c = conv1d(c, np.array(weights["conv1_W"]), np.array(weights["conv1_b"]))
        c = maxpool(c)
        c = conv1d(c, np.array(weights["conv2_W"]), np.array(weights["conv2_b"]))
        c = maxpool(c)
        # GRU over time (c is (channels, time))
        C, T = c.shape
        hDim = GRU_UNITS
        Wih = np.array(weights["gru_Wih"]);  Whh = np.array(weights["gru_Whh"])
        bih = np.array(weights["gru_bih"]);  bhh = np.array(weights["gru_bhh"])
        Wir = np.array(weights["gru_Wih_rev"]);  Whr = np.array(weights["gru_Whh_rev"])
        bir = np.array(weights["gru_bih_rev"]);  bhr = np.array(weights["gru_bhh_rev"])
        def gru_dir(seq, Wih_, Whh_, bih_, bhh_):
            h = np.zeros(hDim, dtype=np.float32)
            hs = []
            for t in range(len(seq)):
                x_t = seq[t]
                ih = Wih_ @ x_t + bih_
                hh = Whh_ @ h + bhh_
                r = 1.0 / (1.0 + np.exp(-(ih[0:hDim] + hh[0:hDim])))
                z = 1.0 / (1.0 + np.exp(-(ih[hDim:2*hDim] + hh[hDim:2*hDim])))
                n = np.tanh(ih[2*hDim:3*hDim] + r * hh[2*hDim:3*hDim])
                h = (1 - z) * n + z * h
                hs.append(h)
            return np.array(hs)  # (T, hDim)
        fwd = gru_dir([c[:, t] for t in range(T)], Wih, Whh, bih, bhh)
        # backward: process the sequence in reverse time order
        rev = [c[:, t] for t in range(T - 1, -1, -1)]
        bwd_rev = gru_dir(rev, Wir, Whr, bir, bhr)
        bwd = bwd_rev[::-1]  # align back to forward time index
        pooled_f = fwd.mean(0)
        pooled_b = bwd.mean(0)
        pooled2 = np.concatenate([pooled_f, pooled_b])
        logits = {}
        for r in RULES:
            parts = [pooled2]
            if r == "MADD":
                parts.append(np.array([db]))
            parts.append(eb)
            feat = np.concatenate(parts)
            W = np.array(weights[f"head_{r}_W"]).flatten()
            b = np.array(weights[f"head_{r}_b"]).flatten()[0]
            logits[r] = float(np.dot(W, feat) + b)
        return logits

    np_out = np_forward(x[0].numpy(), dur[0].numpy()[0], ev[0].numpy())
    print("[export] parity self-check (numpy replica vs torch reference, sample 0):")
    worst = 0.0
    for r in RULES:
        d = abs(np_out[r] - ref_out[r])
        worst = max(worst, d)
        print("   %-13s torch=%.4f  np=%.4f  |d|=%.2e" % (r, ref_out[r], np_out[r], d))
    print("[export] worst |d| = %.2e" % worst)
    if worst > 1e-2:
        print("[export] WARNING: self-check worse than 1e-2 -- exported weights may mismatch.")
    else:
        print("[export] self-check PASSED (worst |d| < 1e-2)")
    print("[export] DONE. Copy dataset/model_weights.json -> app/src/main/assets/")


if __name__ == "__main__":
    main()
