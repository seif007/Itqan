#!/usr/bin/env python3
"""
rule_acoustics.py - Per-rule ACOUSTIC EVIDENCE for Tajweed clips.

WHY this module exists
----------------------
The previous labeling used ONE global reciter flag ("is this reciter a tajweed
reciter?") derived solely from MADD elongation, and applied it to *every* rule.
That is why only MADD learned: GHUNNAH / IKHFA / IDGHAM / MEEM_SAKINAH / IZHAR /
QALQALAH / ALLAM labels were uncorrelated with their own acoustics, so the model
saw noise.

This module instead measures, for each rule, the acoustic phenomenon that rule
actually produces, and returns a single scalar "evidence of correct
performance".  The dataset builder then labels a clip CORRECT/MISTAKE from its
*own rule's* evidence (see prepare_dataset.py), not from MADD elongation.  Each
head becomes "detect this rule's acoustic signature" - which is learnable.

It is deliberately librosa-free (numpy + soundfile only) so it matches the
training stack and stays light on RAM.

Rules and their acoustic correlates
-----------------------------------
  MADD          -> handled separately by duration ratio (see prepare_dataset).
  GHUNNAH       -> nasal murmur  (~250 Hz).  Measured as nasal-band energy ratio.
  IKHFA         -> nasalization into a non-emphatic consonant (nasal).  nasal ratio.
  IDGHAM        -> full nasal merge into ي ن م و (nasal).  nasal ratio.
  MEEM_SAKINAH  -> nasal/labial merge (nasal).  nasal ratio.
  IZHAR         -> the OPPOSITE: a clear noon before a throat letter (NO nasal).
                  Evidence = clarity = (1 - nasal_ratio).
  QALQALAH      -> a complete stop ("bounce"): a closure (silence) then a release
                  burst.  Evidence = closure depth + burst onsets.
  ALLAM         -> lam shamsiyyah/qamariyyah: presence of a lateral /l/.  Evidence
                  = high-frequency (lateral) energy ratio.  EXPERIMENTAL.

All features are NaN/inf-safe and robust to tiny or silent clips.
"""
import os
import sys
import numpy as np
import soundfile as sf

SR = 16000
N_FFT = 512                       # ~32 ms window at 16k
HOP = 160                         # 10 ms hop (matches train.py)

# rules whose acoustic event spans into the following word (noon/meem saakin
# families).  For these the clip is extended to include the next word so the
# relevant transition is actually inside the audio the model sees.
CROSSWORD_RULES = {"IDGHAM", "IKHFA", "IZHAR", "MEEM_SAKINAH"}

NASAL_LO, NASAL_HI = 200.0, 500.0     # nasal murmur band
LOW_HI = 500.0
MID_HI = 2000.0
HIGH_HI = 8000.0


# ---------------------------------------------------------------------------
# Audio loading (mirrors train.py: soundfile slice + mono + NaN guard)
# ---------------------------------------------------------------------------
def load_clip(wav_abs, start_ms, end_ms, sr=SR):
    """Load [start_ms, end_ms] of wav_abs as mono float32 at target sr."""
    start_f = max(0, int(round(start_ms / 1000.0 * sr)))
    end_f = max(start_f + 1, int(round(end_ms / 1000.0 * sr)))
    try:
        data, native_sr = sf.read(wav_abs, start=start_f, stop=end_f,
                                  dtype="float32", always_2d=False)
    except Exception:
        return np.zeros(1, dtype=np.float32), sr
    data = np.nan_to_num(np.asarray(data, dtype=np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if native_sr != sr and native_sr:
        # defend against non-16k sources with a cheap linear resample
        n = int(round(len(data) * sr / float(native_sr)))
        if n > 0:
            data = np.interp(np.linspace(0, len(data) - 1, n),
                             np.arange(len(data)), data).astype(np.float32)
    return data, sr


# ---------------------------------------------------------------------------
# Lightweight STFT magnitude (numpy only)
# ---------------------------------------------------------------------------
def _stft_mag(y, n_fft=N_FFT, hop=HOP):
    n = len(y)
    if n < n_fft:
        y = np.pad(y, (0, n_fft - n))
        n = len(y)
    n_frames = 1 + (n - n_fft) // hop
    if n_frames < 1:
        y = np.pad(y, (0, n_fft - n + hop))
        n_frames = 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.clip(idx, 0, len(y) - 1)
    frames = y[idx] * np.hanning(n_fft)
    spec = np.fft.rfft(frames, n=n_fft)
    mag = np.abs(spec)
    return np.nan_to_num(mag, nan=1e-8, posinf=1e8, neginf=1e8)


# ---------------------------------------------------------------------------
# Per-clip acoustic descriptors
# ---------------------------------------------------------------------------
def clip_features(y, sr=SR):
    """Return a dict of robust acoustic descriptors for one clip, or None."""
    y = np.asarray(y, dtype=np.float32)
    if y.size < 256:
        return None
    # peak-normalise so descriptors are loudness-independent
    peak = np.max(np.abs(y))
    if peak > 1e-4:
        y = y / peak
    mag = _stft_mag(y)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    mag = mag + 1e-8
    mean_mag = np.mean(mag, axis=0)            # (n_bins,)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return float(np.mean(mag[:, m])) if m.any() else 0.0

    total = float(np.mean(mag)) + 1e-8
    nasal = band(NASAL_LO, NASAL_HI)
    low = band(0.0, LOW_HI)
    mid = band(LOW_HI, MID_HI)
    high = band(MID_HI, HIGH_HI)

    # spectral tilt: slope of mean log-magnitude vs log-frequency
    valid = freqs > 0
    if valid.sum() > 2:
        slope = float(np.polyfit(np.log(freqs[valid]),
                                 np.log(mean_mag[valid]), 1)[0])
    else:
        slope = 0.0

    # energy envelope across frames -> closure + release burst.
    # QALQALAH's hallmark is a COMPLETE consonant closure (a silent gap) sharply
    # released by a burst ("bounce").  The old cue used the global quietest frame
    # (any word has those) and any rising edge -- too crude (rel_gap only 0.13).
    # Here closure = depth of the deepest INTERIOR silence gap (a real stop, not a
    # word boundary), and onset = the sharp energy rise immediately after it.
    env = np.mean(mag, axis=1)
    env_max = np.max(env)
    if env_max > 0:
        env = env / env_max
    n_f = len(env)
    # restrict to interior frames so clip-edge padding/silence isn't mistaken
    # for a qalqalah closure
    lo_i, hi_i = max(1, n_f // 6), max(2, (5 * n_f) // 6)
    interior = env[lo_i:hi_i] if hi_i > lo_i else env
    # windowed-min: a closure is a *sustained* quiet, not a single-frame dip
    W = max(2, int(0.03 * sr / HOP))          # ~30 ms window in frames
    if len(interior) >= W:
        win_min = np.min(
            np.lib.stride_tricks.sliding_window_view(interior, W), axis=1)
        k = int(np.argmin(win_min))
        silence_depth = float(1.0 - win_min[k])      # deeper = stronger closure
        # burst = sharp rise in the 1-2 frames right after the silent window
        after = lo_i + k + W
        if after + 1 < n_f:
            burst = float(env[after + 1] - env[after])
        elif after < n_f:
            burst = float(env[after] - env[after - 1])
        else:
            burst = 0.0
        # only count burst if it follows an actual quiet region
        burst = max(0.0, burst) if win_min[k] < 0.35 else 0.0
    else:
        silence_depth = float(1.0 - float(np.min(interior))) if len(interior) else 0.0
        burst = 0.0
    closure = silence_depth
    onset = burst

    return {
        "nasal_ratio": nasal / total,
        "low_ratio": low / total,
        "tilt": slope,
        "closure": closure,
        "onset": onset,
        "lam": high / (mid + 1e-8),
    }


# ---------------------------------------------------------------------------
# Map a clip's descriptors -> a single "evidence of correct performance" for a rule
# ---------------------------------------------------------------------------
def rule_evidence(rule, feat):
    """Higher = more evidence the rule was performed correctly."""
    if feat is None:
        return None
    if rule in ("GHUNNAH", "IKHFA", "IDGHAM", "MEEM_SAKINAH"):
        return float(feat["nasal_ratio"])
    if rule == "IZHAR":
        # IZHAR = clear noon (NOT nasalised) before a throat letter
        return float(1.0 - feat["nasal_ratio"])
    if rule == "QALQALAH":
        # QALQALAH's "bounce" = a DEEP interior silence (closure) sharply
        # released by a burst (onset), with extra low-frequency body.  A deeper
        # closure means a more fully articulated stop -> HIGHER evidence (note:
        # this is the opposite sign from a "gap = bad" intuition; for qalqalah
        # the stop is the target).  low_ratio adds the characteristic low
        # resonance of the echoed consonant.
        return float(0.5 * feat["closure"] + 0.3 * feat["onset"]
                     + 0.2 * (feat["low_ratio"] - 4.0))
    if rule == "ALLAM":
        # lateral /l/ shows up as relatively strong high-frequency energy
        return float(feat["lam"])
    return None


def clip_evidence(rule, wav_abs, start_ms, end_ms, next_end_ms=None):
    """Convenience: load the (possibly extended) clip and return rule evidence."""
    if rule in CROSSWORD_RULES and next_end_ms and next_end_ms > start_ms:
        end = next_end_ms
    else:
        end = end_ms
    y, _ = load_clip(wav_abs, start_ms, end)
    return rule_evidence(rule, clip_features(y))
