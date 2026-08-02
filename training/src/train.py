#!/usr/bin/env python3
"""
train.py - Tiny on-device Tajweed mistake-detection model (training sketch).

Architecture (PLAN section 5.2):
    log-mel/MFCC features (streamed, 40 bins, 25 ms frame) -> shared Conv1D
    backbone (3 small layers) -> GRU (64 units) -> per-rule sigmoid heads.

The model is MULTI-TASK: one shared acoustic backbone feeds one binary head
per Tajweed rule.  For each training step we sample a batch from ONE rule and
back-propagate only that rule's head (the backbone is shared, so it learns
general "correct vs wrong recitation" cues).  This matches the recommended
"shared Conv1D+GRU backbone with per-rule heads" design.

RAM safety (PLAN section 5.4):
    * Audio is decoded on the fly from the absolute WAV paths in the manifests;
      nothing is copied or held in RAM between batches.
    * Features are computed per-clip inside a streaming DataLoader collate fn.
    * Reciter-disjoint train/val/test comes straight from dataset/splits.json.

Checkpointing / resume (added):
    * Latest state is saved to model_checkpoint.pt (model + optimizer + RNG
      state + global step + epoch + training history) every --save_every steps
      AND at every epoch boundary.  The best model by aggregate val AUC is
      saved to model_best.pt.
    * If a checkpoint exists, training RESUMES automatically (continues the
      epoch + step + RNG so the run is reproducible).  Use --fresh to restart.
      This makes the run robust to a power cut or a manual pause: just re-run
      the same command and it picks up where it left off.  You may pass a new
      --lr on resume to adjust the learning rate for fine-tuning.

Fit diagnostics (added):
    * Every epoch prints train/val AUC + loss per rule, the train/val AUC gap,
      and a clear verdict: UNDERFIT / OVERFIT / MARGINAL / GOOD, with concrete
      suggestions for what to change.  Optional --patience early-stops when val
      AUC stops improving.

TFLite / int8 (PLAN section 5.3):
    The model exports to TorchScript.  If onnx + (onnx2tf or TensorFlow) are
    available it is converted to an int8 TFLite model via a representative
    dataset; otherwise we save the traced model + a representative-data generator
    and print the quantization command.  The Android app is NOT touched.

Run:
    python train.py --dataset_dir ../dataset --smoke      # quick self-check
    python train.py --dataset_dir ../dataset --epochs 20  # real (CPU) training
    python train.py --dataset_dir ../dataset              # resumes if a
                                                          # model_checkpoint.pt
                                                          # exists; --fresh to restart
"""
import os
import sys
import json
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
try:
    import torchaudio  # optional; we fall back to a numpy path if missing/broken
except Exception:
    torchaudio = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    from rule_acoustics import clip_features as _clip_features
except Exception:
    _clip_features = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SR = 16000
N_MELS = 40
WIN_MS = 25
HOP_MS = 10
N_FFT = int(SR * WIN_MS / 1000)
HOP = int(SR * HOP_MS / 1000)
# fixed clip length (frames) after padding/trimming, keeps tensors uniform
MAX_FRAMES = 200          # ~2 s at 10 ms hop; word segments are usually shorter
CONV_CHANNELS = 32
GRU_UNITS = 64
RULES = ["MADD", "GHUNNAH", "QALQALAH", "IDGHAM", "IKHFA",
         "ALLAM", "MEEM_SAKINAH", "IZHAR"]

CKPT_NAME = "model_checkpoint.pt"
BEST_NAME = "model_best.pt"


def log(*a):
    print("[train]", *a, flush=True)


# ---------------------------------------------------------------------------
# Features: log-mel (numpy + torchaudio if present, else pure numpy STFT)
# ---------------------------------------------------------------------------
_mel_filters = None


def _build_mel_filters(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmax=8000):
    """Triangular mel filterbank (n_mels x (n_fft//2+1)) built from scratch."""
    def hz_to_mel(h):
        return 2595.0 * math.log10(1.0 + h / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    low, high = 0.0, fmax
    mel_pts = np.linspace(hz_to_mel(low), hz_to_mel(high), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        f0, f1, f2 = bins[m], bins[m + 1], bins[m + 2]
        if f1 <= f0 or f2 <= f1:
            continue
        for k in range(f0, f1):
            filters[m, k] = (k - f0) / (f1 - f0)
        for k in range(f1, f2):
            filters[m, k] = (f2 - k) / (f2 - f1)
    return filters


def compute_logmel(wav, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop=HOP):
    """Return a (n_mels x T) float32 log-mel matrix for a mono float waveform."""
    global _mel_filters
    if _mel_filters is None:
        _mel_filters = _build_mel_filters(n_mels, n_fft, sr)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # framing
    n = len(wav)
    n_frames = 1 + max(0, (n - n_fft) // hop)
    if n_frames <= 0:
        # too short: zero-pad to one frame
        wav = np.pad(wav, (0, n_fft - n))
        n_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        wav, shape=(n_frames, n_fft),
        strides=(wav.strides[0] * hop, wav.strides[0]))
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)) ** 2
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    mel = _mel_filters @ spec.T            # (n_mels x n_frames)
    mel = np.clip(mel, 1e-8, None)
    mel = np.nan_to_num(mel, nan=1e-8, posinf=1e-8, neginf=1e-8)
    logmel = np.log10(mel + 1e-8)
    return logmel.astype(np.float32)


def load_clip(wav_abs, start_ms, end_ms, sr=SR):
    """Decode [start_ms, end_ms] of a WAV on the fly; resample to 16 kHz mono."""
    import soundfile as sf
    info = sf.info(wav_abs)
    start_f = max(0, int(start_ms / 1000.0 * info.samplerate))
    end_f = int(end_ms / 1000.0 * info.samplerate)
    data, native_sr = sf.read(wav_abs, start=start_f, stop=end_f, dtype="float32",
                              always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if native_sr != sr:
        # lightweight resample via torchaudio if present, else linear interp
        if torchaudio is not None:
            t = torch.from_numpy(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, native_sr, sr).squeeze(0)
            data = t.numpy()
        else:
            n_new = int(round(len(data) * sr / native_sr))
            data = np.interp(np.linspace(0, len(data) - 1, n_new),
                             np.arange(len(data)), data).astype(np.float32)
    if len(data) == 0:
        data = np.zeros(sr // 50, dtype=np.float32)
    wav = np.asarray(data, dtype=np.float32)
    # guard against NaN/Inf samples (corrupt source clips) poisoning training
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav


def clip_evidence_vector(wav):
    """Clip-level acoustic evidence cues, bounded to ~[-1,1] via tanh.

    These are the SAME cues rule_acoustics uses for labeling (nasal murmur,
    lateral /l/, stop closure, release onset, spectral tilt).  Like MADD's
    duration, they are clip-level scalars that temporal mean-pooling in the
    backbone destroys -- so we feed them straight to the heads.  Because this is
    a pure function of the audio (no labels), it is identical at train and
    inference time.  Each head's Linear learns which cue matters for its rule.
    """
    if _clip_features is None:
        return np.zeros(5, dtype=np.float32)
    f = _clip_features(wav)
    if f is None:
        return np.zeros(5, dtype=np.float32)
    raw = np.array([f["nasal_ratio"], f["lam"], f["closure"],
                    f["onset"], f["tilt"]], dtype=np.float32)
    # empirical scales to bring each component near unit range before tanh
    scale = np.array([10.0, 2.0, 1.0, 1.0, 5.0], dtype=np.float32)
    return np.tanh(raw / scale).astype(np.float32)


def clip_to_tensor(wav_abs, start_ms, end_ms):
    wav = load_clip(wav_abs, start_ms, end_ms)
    feat = compute_logmel(wav)                 # (n_mels x T)
    # pad / trim to MAX_FRAMES
    T = feat.shape[1]
    if T < MAX_FRAMES:
        feat = np.pad(feat, ((0, 0), (0, MAX_FRAMES - T)), mode="constant")
    else:
        feat = feat[:, :MAX_FRAMES]
    # Normalized clip duration.  MADD's acoustic signature is vowel ELONGATION,
    # which temporal mean-pooling (below) destroys.  We feed this scalar only to
    # the MADD head so the other rules (which key off spectral shape: nasal /
    # lateral energy, stops) are not distracted by clip length.  Clipped to a
    # sane range and scaled so ~1.0 ~= 1 second.
    dur_sec = max(0.0, (end_ms - start_ms) / 1000.0)
    dur_feat = min(dur_sec, 4.0) / 4.0
    # Clip-level acoustic evidence cues (see clip_evidence_vector).
    ev = clip_evidence_vector(wav)
    return (torch.from_numpy(feat).unsqueeze(0),   # (1, n_mels, MAX_FRAMES)
            torch.tensor([dur_feat], dtype=torch.float32),
            torch.from_numpy(ev).unsqueeze(0))     # (1, 5)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TajweedNet(nn.Module):
    def __init__(self, n_mels=N_MELS, conv_ch=CONV_CHANNELS, gru=GRU_UNITS,
                 rules=RULES, dropout=0.2, madd_rule="MADD", ev_in=5):
        super().__init__()
        self.n_mels = n_mels
        self.madd_rule = madd_rule
        self.ev_in = ev_in
        # Conv1D backbone (features are (B, n_mels, T) -> treat n_mels as channels)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_ch, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_ch, conv_ch, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_ch, conv_ch, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # GRU over the time axis (after permute to (B, T, C))
        self.gru = nn.GRU(conv_ch, gru, batch_first=True, bidirectional=True)
        self.head_in = gru * 2
        # MADD head ALSO receives the duration scalar; every head ALSO receives
        # the 5-dim clip-level acoustic-evidence vector (nasal / lateral /
        # closure / onset / tilt) which temporal pooling would otherwise destroy.
        self.heads = nn.ModuleDict({
            r: nn.Linear(self.head_in + (1 if r == madd_rule else 0) + ev_in, 1)
            for r in rules})
        self.drop = nn.Dropout(dropout)
        # Hybrid device: heavy Conv backbone runs on the primary device (GPU),
        # the Bi-GRU runs on CPU.  The GTX 950M (sm_50) SIGSEGVs / hangs inside
        # the CUDA GRU kernel, so we isolate just that layer on CPU.
        self.gru_device = torch.device("cpu")
        self._main_device = None

    def _set_main_device(self, x):
        if self._main_device is None:
            self._main_device = x.device

    def forward(self, x, dur=None, ev=None):
        # x: (B, n_mels, T)
        self._set_main_device(x)
        main = self._main_device
        c = self.conv(x)                 # (B, conv_ch, T')  on main device
        c = c.permute(0, 2, 1)           # (B, T', conv_ch)
        # GRU on CPU (avoids broken CUDA GRU kernel on sm_50)
        c_cpu = c.to(self.gru_device)
        out, _ = self.gru(c_cpu)         # (B, T', 2*gru) on CPU
        out = out.to(main)               # back to main device
        pooled = out.mean(dim=1)         # (B, 2*gru) simple temporal mean
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Streaming dataset per rule (reads manifest rows, decodes on the fly)
# ---------------------------------------------------------------------------
class RuleStream(torch.utils.data.Dataset):
    def __init__(self, rows):
        # rows: list of dicts {wav_abs, start_ms, end_ms, y, split}
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x, dur, ev = clip_to_tensor(r["wav_abs"], r["start_ms"], r["end_ms"])
        y = torch.tensor(float(r["y"]), dtype=torch.float32)
        return x, dur, ev, y


def collate(batch):
    xs = torch.cat([b[0] for b in batch], dim=0)
    durs = torch.cat([b[1] for b in batch], dim=0)
    evs = torch.cat([b[2] for b in batch], dim=0)
    ys = torch.stack([b[3] for b in batch], dim=0)
    return xs, durs, evs, ys


# ---------------------------------------------------------------------------
# Load manifests (reciter-disjoint splits)
# ---------------------------------------------------------------------------
def load_rule_rows(dataset_dir, rule, split):
    path = os.path.join(dataset_dir, "manifest_%s.jsonl" % rule)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("split") != split:
                continue
            rows.append(o)
    return rows


# ---------------------------------------------------------------------------
# Evaluation helper (val AUC + loss, and train-AUC sampling for gap)
# ---------------------------------------------------------------------------
def evaluate(model, rows, rule, device, batch_size):
    """Return (roc_auc, mean_bce_loss) over `rows` for one rule. NaN if empty."""
    if not rows:
        return float("nan"), float("nan")
    was_training = model.training
    model.eval()                      # eval mode for inference (no dropout)
    try:
        ds = RuleStream(rows)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                             collate_fn=collate)
        preds, gts, losses = [], [], []
        crit = nn.functional.binary_cross_entropy_with_logits
        with torch.no_grad():
            for xs, durs, evs, ys in loader:
                xs = xs.to(device)
                durs = durs.to(device)
                evs = evs.to(device)
                ys = ys.to(device)
                logit = model(xs, durs, evs)[rule]
                losses.append(float(crit(logit, ys).item()))
                p = torch.sigmoid(logit).detach().cpu().numpy()
                preds.extend(p.tolist())
                gts.extend(ys.detach().cpu().numpy().tolist())
        return _auc(gts, preds), float(np.mean(losses))
    finally:
        if was_training:
            model.train()             # restore training mode (CuDNN GRU needs it)


def _auc(y_true, y_score):
    """ROC-AUC via rank (works for binary, no sklearn dependency)."""
    n = len(y_true)
    if n == 0 or sum(y_true) in (0, n):
        return float("nan")
    order = sorted(range(n), key=lambda i: y_score[i])
    ranks = [0] * n
    # average ranks for ties
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    n_pos = sum(y_true)
    n_neg = n - n_pos
    sum_r = sum(ranks[i] for i in range(n) if y_true[i] == 1)
    return (sum_r - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, optim, epoch, step, rng, history,
                    best_val_auc, best_epoch, args, epochs_since_improve=0,
                    scheduler_state=None):
    state = {
        "epoch": epoch,            # epoch index to RESUME INTO (next to run)
        "step": step,              # global round-robin step (restored on resume)
        "rng_state": rng.getstate(),
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "history": history,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "epochs_since_improve": epochs_since_improve,  # patience counter
        "args": vars(args),
    }
    if scheduler_state is not None:
        state["scheduler"] = scheduler_state
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)          # atomic replace, safe against power loss mid-write


def load_checkpoint(path, model, optim, rng, args):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optim.load_state_dict(state["optim"])
    # allow adjusting lr on resume (e.g. lower lr for fine-tuning after a pause)
    for g in optim.param_groups:
        g["lr"] = args.lr
    rng.setstate(state["rng_state"])
    return {
        "epoch": state.get("epoch", 0),
        "step": state.get("step", 0),
        "history": state.get("history", []),
        "best_val_auc": state.get("best_val_auc", -1.0),
        "best_epoch": state.get("best_epoch", -1),
        "epochs_since_improve": state.get("epochs_since_improve", 0),
        "scheduler_state": state.get("scheduler", None),
    }


# ---------------------------------------------------------------------------
# Fit diagnostics: underfit / overfit / marginal / good
# ---------------------------------------------------------------------------
def diagnose(history, active):
    """Print a clear per-rule + aggregate fit verdict with tuning advice."""
    if not history:
        return
    last = history[-1]
    val = {r: v for r, v in last["val_auc"].items()
           if r in active and not math.isnan(v)}
    tr = {r: v for r, v in last["train_auc"].items()
          if r in active and not math.isnan(v)}
    if not val:
        log("diagnose: no validation rows for any active rule yet.")
        return
    mean_val = float(np.mean(list(val.values())))
    mean_tr = float(np.mean(list(tr.values()))) if tr else mean_val
    gap = mean_tr - mean_val

    # trend over last 3 epochs (aggregate val AUC)
    agg_prev = None
    if len(history) >= 3:
        agg_prev = history[-3].get("agg_val_auc")
    trend = (mean_val - agg_prev) if agg_prev is not None else 0.0

    # per-rule classification
    under, over, good = [], [], []
    for r in active:
        if r not in val:
            continue
        rv = val[r]
        rg = (tr.get(r, rv) - rv)
        if rv >= 0.85 and rg < 0.10:
            good.append(r)
        elif rv < 0.65:
            under.append(r)
        elif rg > 0.12:
            over.append(r)
        else:
            pass  # marginal -> not called out explicitly

    if mean_val >= 0.85 and gap < 0.08 and trend >= -0.02:
        verdict = "GOOD FIT"
    elif gap > 0.12 and mean_tr >= 0.70:
        verdict = "OVERFIT"   # train learns but val does not generalize
    elif mean_val < 0.65:
        verdict = "UNDERFIT"  # neither train nor val is learning
    else:
        verdict = "MARGINAL"

    log("-" * 60)
    log("FIT DIAGNOSIS: %s" % verdict)
    log("  aggregate val AUC = %.3f | train AUC = %.3f | gap = %.3f | "
        "3-epoch trend = %+.3f" % (mean_val, mean_tr, gap, trend))
    if under:
        log("  UNDERFITTING rules (val AUC < 0.65): %s" % under)
    if over:
        log("  OVERFITTING rules (train-val gap > 0.12): %s" % over)
    if good:
        log("  WELL-FIT rules (val AUC >= 0.85): %s" % good)
    # flag thin rules whose tiny val sets make their AUC statistically noise
    THIN = {"IDGHAM", "IZHAR", "IKHFA"}
    thin_present = [r for r in active if r in THIN]
    if thin_present:
        log("  NOTE: thin rules with very small val sets (noisy AUC, do not "
            "trust their per-rule verdict): %s" % thin_present)

    # concrete advice
    if verdict == "UNDERFIT":
        log("  ADVICE: model is not learning the signal. Try: more epochs; "
            "higher --lr (e.g. 3e-3); bigger backbone (CONV_CHANNELS/GRU_UNITS); "
            "longer clips (MAX_FRAMES); check label quality / feature extraction.")
    elif verdict == "OVERFIT":
        log("  ADVICE: memorizing train, failing to generalize. Try: stronger "
            "Dropout; fewer epochs / --patience early stop; weight decay; more/"
            "harder augmentations; more training data per rule; smaller backbone.")
    elif verdict == "MARGINAL":
        log("  ADVICE: decent but not strong. Nudge lr or add a little dropout; "
            "give it more epochs; inspect per-rule gaps above.")
    else:
        log("  ADVICE: healthy fit. If you need more headroom, gather more "
            "balanced data for the weaker rules or fine-tune lr slightly.")
    log("-" * 60)


# ---------------------------------------------------------------------------
# Train (with checkpoint / resume / diagnostics)
# ---------------------------------------------------------------------------
def train(args):
    # ---- device selection (GPU if available, else CPU) --------------------
    if args.device == "cuda" and not torch.cuda.is_available():
        log("WARNING: --device cuda requested but torch.cuda.is_available() is "
            "False (no CUDA GPU, or a CPU-only torch build). Falling back to CPU.")
        chosen = "cpu"
    elif args.device == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        chosen = args.device
    device = torch.device(chosen)
    if device.type == "cuda":
        # The GTX 950M (Maxwell, sm_50) SIGSEGVs / hangs inside the CUDA GRU
        # kernel.  We keep the Conv backbone on GPU but route the Bi-GRU to CPU
        # (see TajweedNet.forward); cuDNN is disabled as a belt-and-braces guard.
        torch.backends.cudnn.enabled = False
        log("device: %s (hybrid: Conv on GPU, GRU on CPU)" % device)
    else:
        log("device: %s" % device)
    # filter rules if --rules was given (for thin-rule-only training)
    requested_rules = [r for r in args.rules.split(",") if r.strip()]
    active_rules = [r for r in requested_rules if r in RULES]
    if not active_rules:
        active_rules = list(RULES)
    log("training rules:", active_rules)
    model = TajweedNet(rules=active_rules, dropout=args.dropout,
                       conv_ch=args.conv_channels, gru=args.gru_units).to(device)
    # In hybrid mode the Bi-GRU must physically reside on CPU (its weights are
    # moved off the GPU) so forward/backward run on the stable CPU kernel.
    if device.type == "cuda":
        model.gru.to("cpu")
        model.gru_device = torch.device("cpu")
    log("model params: %d (%.2f MB @ fp32)" %
        (count_params(model), count_params(model) * 4 / 1e6))

    # group rows by rule+split
    data = {r: {"train": load_rule_rows(args.dataset_dir, r, "train"),
                "val": load_rule_rows(args.dataset_dir, r, "val"),
                "test": load_rule_rows(args.dataset_dir, r, "test")}
            for r in active_rules}
    active = [r for r in active_rules if data[r]["train"]]
    log("active rules (have train rows):", active)
    if not active:
        log("ERROR: no training rows found. Run prepare_dataset.py first.")
        return

    # per-rule class weight (pos_weight = neg/pos) from train rows
    pos_w = {}
    for r in active:
        pos = sum(1 for x in data[r]["train"] if x["y"] == 1)
        neg = len(data[r]["train"]) - pos
        pos_w[r] = torch.tensor(float(neg / max(pos, 1)),
                                dtype=torch.float32, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    # LR scheduler: cosine (warm restarts) or plateau (ReduceLROnPlateau)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=10, T_mult=2, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="max", factor=0.5, patience=4, threshold=1e-4,
            verbose=True, min_lr=1e-6)

    ckpt_path = os.path.join(args.dataset_dir, CKPT_NAME)
    best_path = os.path.join(args.dataset_dir, BEST_NAME)

    # ---- resume? ----------------------------------------------------------
    start_epoch = 0
    step = 0
    history = []
    best_val_auc = -1.0
    best_epoch = -1
    epochs_since_improve = 0          # patience counter (restored on resume)
    rng = random.Random(0)
    if (not args.fresh) and os.path.exists(ckpt_path):
        restored = load_checkpoint(ckpt_path, model, optim, rng, args)
        start_epoch = restored["epoch"]
        step = restored["step"]
        history = restored["history"]
        best_val_auc = restored["best_val_auc"]
        best_epoch = restored["best_epoch"]
        epochs_since_improve = restored.get("epochs_since_improve", 0)
        # restore LR scheduler state if present (args.lr sets base; plateau tracks from here)
        # scheduler state restored below after scheduler creation
        log("RESUMED from %s: epoch=%d step=%d best_val_auc=%.3f "
            "epochs_since_improve=%d (lr set to %.1e)"
            % (CKPT_NAME, start_epoch, step, best_val_auc,
               epochs_since_improve, args.lr))
    else:
        if args.fresh and os.path.exists(ckpt_path):
            log("--fresh: ignoring existing checkpoint, starting over.")
        log("starting fresh training from epoch 0.")

    # restore scheduler state if available
    sched_state = restored.get("scheduler_state") if 'restored' in locals() else None
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)

    # round-robin rule order (shuffle each epoch); rng restored on resume

    for epoch in range(start_epoch, args.epochs):
        model.train()                # ensure training mode
        rng.shuffle(active)
        # Per-rule step counts: each rule does a bounded number of batches so a
        # single huge rule (e.g. MADD=15k clips) does not force every other rule
        # to also run 475 steps.  We iterate rule-by-rule each epoch.
        per_rule_steps = {
            r: max(1, min(len(data[r]["train"]) // args.batch_size,
                          args.max_steps_per_rule))
            for r in active}
        epoch_losses = []
        # thin rules get higher LR (underfit), medium rules get lower LR (near-converged)
        THIN_RULES = {"IDGHAM", "IKHFA", "IZHAR", "MEEM_SAKINAH"}
        for r in active:
            rows = data[r]["train"]
            if not rows:
                continue
            # adjust LR per rule
            rule_lr = args.thin_lr if r in THIN_RULES else args.medium_lr
            for g in optim.param_groups:
                g["lr"] = rule_lr
            for _ in range(per_rule_steps[r]):
                step += 1
                idx = rng.sample(range(len(rows)), min(args.batch_size, len(rows)))
                batch = [rows[i] for i in idx]
                ds = RuleStream(batch)
                loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                                     collate_fn=collate)
                xs, durs, evs, ys = next(iter(loader))
                xs, durs, evs, ys = xs.to(device), durs.to(device), evs.to(device), ys.to(device)
                logits = model(xs, durs, evs)[r]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, ys, pos_weight=pos_w[r])
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_losses.append(float(loss.item()))
                # mid-epoch checkpoint (safe against power cut / pause)
                if step % args.save_every == 0:
                    save_checkpoint(ckpt_path, model, optim, epoch, step, rng,
                                    history, best_val_auc, best_epoch, args,
                                    epochs_since_improve, scheduler.state_dict())
        # ---- end of epoch: evaluate + diagnose + checkpoint ----------------
        val_auc, val_loss, train_auc = {}, {}, {}
        for r in active:
            va, vl = evaluate(model, data[r]["val"], r, device, args.batch_size)
            val_auc[r], val_loss[r] = va, vl
            # sample train rows to estimate the train/val gap (bounded sample)
            tr = data[r]["train"]
            if tr:
                samp = tr if len(tr) <= 300 else rng.sample(tr, 300)
                ta, _ = evaluate(model, samp, r, device, args.batch_size)
                train_auc[r] = ta
        agg_val = float(np.nanmean([v for v in val_auc.values()
                                    if not math.isnan(v)])) if val_auc else float("nan")
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "val_auc": val_auc,
            "val_loss": val_loss,
            "train_auc": train_auc,
            "agg_val_auc": agg_val,
        }
        history.append(entry)

        log("epoch %d | train_loss=%.4f | val-auc=%s" %
            (epoch, entry["train_loss"],
             {k: round(v, 3) for k, v in val_auc.items()}))
        log("          train-auc=%s" %
            {k: round(v, 3) for k, v in train_auc.items()})
        diagnose(history, active)

        # best-model tracking
        improved = (not math.isnan(agg_val)) and (agg_val > best_val_auc)
        if improved:
            best_val_auc = agg_val
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.state_dict(), best_path)
            log("  new best model -> %s (agg_val_auc=%.3f)" % (BEST_NAME, agg_val))
            if getattr(args, "export", False):
                export_tflite(model, args.dataset_dir, active)  # keep best export fresh
        else:
            epochs_since_improve += 1

        # always save latest checkpoint at epoch boundary
        save_checkpoint(ckpt_path, model, optim, epoch + 1, step, rng,
                        history, best_val_auc, best_epoch, args,
                        epochs_since_improve)

        # LR scheduler step
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(agg_val)

        # early stopping
        if args.patience > 0 and epochs_since_improve >= args.patience:
            log("EARLY STOP: no val-AUC improvement for %d epochs." % args.patience)
            break

    log("training loop done. last checkpoint -> %s ; best -> %s (epoch %d, "
        "agg_val_auc=%.3f)" % (CKPT_NAME, BEST_NAME, best_epoch, best_val_auc))

    # ---- final test AUC (report only; best model already exported) --------
    final_state = torch.load(best_path, map_location="cpu") if os.path.exists(best_path) else model.state_dict()
    model.load_state_dict(final_state)
    with torch.no_grad():
        for r in active:
            trows = data[r]["test"]
            if not trows:
                continue
            ta, _ = evaluate(model, trows, r, device, args.batch_size)
            log("TEST rule %-13s auc=%.3f n=%d" % (r, ta, len(trows)))

    # ensure a traced export exists for the chosen (best) model.
    # NOTE: torch.jit.trace SEGV's on the sm_50 GPU with the hybrid
    # (Conv-on-GPU / GRU-on-CPU) model, so it is opt-in via --export.
    if getattr(args, "export", False):
        export_tflite(model, args.dataset_dir, active)
    else:
        log("skipping TorchScript export (--export not set; model_best.pt is the deliverable)")
    log("done.")


# ---------------------------------------------------------------------------
# TFLite / int8 export
# ---------------------------------------------------------------------------
def export_tflite(model, dataset_dir, active_rules):
    """Export to TorchScript, then to int8 TFLite if tooling is present."""
    was_training = model.training
    model.eval()                      # eval mode for tracing
    try:
        dev0 = next(model.parameters()).device
        dummy = torch.zeros(1, N_MELS, MAX_FRAMES, device=dev0)
        dummy_dur = torch.zeros(1, 1, device=dev0)
        dummy_ev = torch.zeros(1, 5, device=dev0)
        traced = torch.jit.trace(model, (dummy, dummy_dur, dummy_ev), strict=False)
        pt_path = os.path.join(dataset_dir, "model_traced.pt")
        traced.save(pt_path)
        log("saved TorchScript model ->", pt_path)

        # representative dataset generator (stream a few real clips)
        def rep_reader():
            for r in active_rules:
                rows = load_rule_rows(dataset_dir, r, "test") or \
                       load_rule_rows(dataset_dir, r, "train")
                for o in rows[:8]:
                    x, dur, ev = clip_to_tensor(o["wav_abs"], o["start_ms"],
                                                o["end_ms"])
                    yield x.numpy(), dur.numpy(), ev.numpy()

        try:
            import onnx
            import onnx2tf  # optional; only if user installed it
            # wrap so ONNX/TFLite sees a single tensor output (B, n_rules) in rule order
            class Wrapped(nn.Module):
                def __init__(self, m, rules):
                    super().__init__()
                    self.m = m
                    self.rules = rules

                def forward(self, x, dur, ev):
                    out = self.m(x, dur, ev)
                    return torch.stack([out[r] for r in self.rules], dim=1)
            wrapped = Wrapped(model, active_rules)
            onnx_path = os.path.join(dataset_dir, "model.onnx")
            torch.onnx.export(wrapped, (dummy, dummy_dur, dummy_ev), onnx_path,
                              input_names=["logmel", "dur", "ev"],
                              output_names=["logits"])
            log("exported ONNX ->", onnx_path)
            log("Convert to int8 TFLite with e.g.:")
            log("  onnx2tf -i %s -o %s -osd" % (onnx_path,
                                                os.path.join(dataset_dir,
                                                             "tflite_int8")))
            log("(then apply int8 quantization via a representative dataset; "
                "see TRAINING.md)")
        except Exception as e:
            log("int8 TFLite tooling (onnx/onnx2tf) not available: %s" % e)
            log("Saved TorchScript model + representative-data generator. To finish")
            log("int8 export, install onnx2tf and run the command printed above, "
                "or use")
            log("TensorFlow's TFLiteConverter with a representative_dataset built "
                "from")
            log("`rep_reader()` (see TRAINING.md).")
    finally:
        if was_training:
            model.train()             # restore training mode (CuDNN GRU needs it)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",
                    default=os.environ.get("TARGET_DIR",
                                            os.path.join(HERE, "..", "dataset")))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_every", type=int, default=50,
                    help="save a resumable checkpoint every N optimizer steps "
                         "(protects against power cut / pause)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop if val AUC does not improve for N epochs "
                         "(0 = off)")
    ap.add_argument("--max_steps_per_rule", type=int, default=10000,
                    help="max training batches per rule per epoch (caps the "
                         "step count so a large rule does not slow every epoch). "
                         "Set high to disable cap.")
    ap.add_argument("--export", action="store_true",
                    help="also run TorchScript/int8 export at the end (OFF by "
                         "default: torch.jit.trace SEGV's on sm_50 GPU with the "
                         "hybrid model, so the model_best.pt checkpoint is the "
                         "deliverable)")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="dropout after the backbone pooling (regularization "
                         "against overfit)")
    ap.add_argument("--weight_decay", type=float, default=1e-3,
                    help="Adam L2 weight decay (regularization against overfit)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and train from scratch")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="compute device (auto = CUDA if available, else CPU)")
    ap.add_argument("--conv_channels", type=int, default=32,
                    help="Conv1d backbone channels (32 default; 48 overfits thin rules)")
    ap.add_argument("--gru_units", type=int, default=64,
                    help="Bi-GRU hidden units (64 default; 96 overfits thin rules)")
    ap.add_argument("--scheduler", choices=["plateau", "cosine"], default="cosine",
                    help="LR scheduler: cosine (warm restarts) or plateau")
    ap.add_argument("--rules", type=str, default=",".join(RULES),
                    help="comma-separated subset of rules to train")
    ap.add_argument("--thin_lr", type=float, default=1e-3,
                    help="LR for thin rules (higher, since they underfit)")
    ap.add_argument("--medium_lr", type=float, default=3e-4,
                    help="LR for medium rules (lower, since they are near-converged)")
    ap.add_argument("--smoke", action="store_true",
                    help="run 2 batches + export path check, then exit")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 0
        model = TajweedNet(rules=RULES)
        log("SMOKE: params=%d" % count_params(model))
        # one forward on a dummy clip
        rows = load_rule_rows(args.dataset_dir, "MADD", "train")
        if rows:
            x, dur, ev = clip_to_tensor(rows[0]["wav_abs"], rows[0]["start_ms"],
                                        rows[0]["end_ms"])
            out = model(x, dur, ev)
            log("SMOKE: MADD logit=%.3f (others present: %s)" %
                (float(out["MADD"][0]), list(out.keys())))
        else:
            log("SMOKE: no MADD train rows; skipping forward (run prepare_dataset).")
        export_tflite(model, args.dataset_dir, RULES)
        log("SMOKE OK")
        return
    train(args)


if __name__ == "__main__":
    main()
