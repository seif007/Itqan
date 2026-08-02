#!/usr/bin/env python3
"""
download_tadabur.py — Download a REAL subset of the Tadabur Quran-recitation
dataset (FaisaI/tadabur, ~938 GB) for offline Tajweed calibration.

Why this implementation:
  * The dataset is 385 parquet shards; each shard is a SINGLE row-group and
    embeds the verse audio as `bytes` inside the `audio` struct column.
  * There is NO tarteel/tajweed style label anywhere. "tajweed vs tarteel"
    filtering is done acoustically later (see tadabur_adapter.py).
  * Network egress here is slow (~0.1-0.4 MB/s), so we download only a few
    whole shards ("a part").

ROOT-CAUSE NOTE (machine-freeze investigation):
  * The `audio.bytes` column is stored as ONE ~2.4 GB PLAIN page. Reading even
    a single verse with pyarrow forces the whole 2.4 GB page to be loaded into
    RAM and read from disk (~5 min on a slow disk). On an 8.5 GB machine this
    saturated RAM at NORMAL priority and made the machine appear "blocked".
  * FIX: this script (a) runs at BELOW_NORMAL CPU priority so the UI stays
    responsive, (b) reads the audio column exactly ONCE per shard (2.4 GB fits
    in the ~4.9 GB free RAM with margin), (c) writes each WAV and frees the
    blob immediately, and (d) is resumable at BOTH shard and verse granularity
    so an interrupt loses almost nothing.

Usage:
  SHARDS="0,1" SURAH_MAX=20 python download_tadabur.py
  (SHARDS may be "0,1,2" or "0-3"; default "0,1".)
  For the FULL set on a fast connection: SHARDS="0-384" SURAH_MAX=114

Output (under tools/calib/tadabur_part/):
  <reciter_id>/<surah>_<ayah>.wav        (16 kHz mono, 16-bit PCM)
  <reciter_id>/<surah>_<ayah>.json       (sidecar with word_alignments & texts)
  And a top-level tadabur_part/_shards_done.txt checkpoint.
  Progress is logged to tadabur_part/process.log.
"""
import os, io, sys, json, time, urllib.request, gc, logging

# --- Priority -----------------------------------------------------------------
# NOTE: we deliberately do NOT use BELOW_NORMAL here. Profiling showed the
# ~2.4 GB audio column is CPU-bound to decode into an Arrow buffer; at
# BELOW_NORMAL that decode crawled for 20+ minutes. The job fits in RAM
# (peak ~3-4 GB on an 8.5 GB machine, leaving headroom), so NORMAL priority
# keeps it fast AND the UI stays responsive (CPU is mostly idle during the
# disk read). We set NORMAL explicitly for predictability.
try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00000020)  # NORMAL
except Exception:
    pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "tadabur_part")
DLCACHE = os.path.join(ROOT, "dl_cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DLCACHE, exist_ok=True)

DATASET = os.environ.get("DATASET", "FaisaI/tadabur")
SURAH_MAX = int(os.environ.get("SURAH_MAX", "20"))

def parse_shards(spec):
    if not spec:
        return [0, 1]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))

SHARDS = parse_shards(os.environ.get("SHARDS", "0,1"))
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/data/train-{{:05d}}.parquet"

TARGET_SR = 16000

# --- logging -----------------------------------------------------------------
log_path = os.path.join(OUT, "process.log")
logger = logging.getLogger("tadabur")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(log_path, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(_fh)
_con = logging.StreamHandler(sys.stdout)
_con.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(_con)

def log(*a):
    logger.info(" ".join(str(x) for x in a))

def download_parquet(idx):
    """Stream-download one shard to dl_cache (resumable via HTTP Range)."""
    url = BASE_URL.format(idx)
    local = os.path.join(DLCACHE, f"train-{idx:05d}.parquet")
    # Already fully downloaded? A complete shard is ~2.4 GB; if we have a file
    # that big it is the cached full shard -> skip the (harmless) resume check.
    if os.path.exists(local) and os.path.getsize(local) > 2_300_000_000:
        log(f"shard {idx}: cached ({os.path.getsize(local)//1_000_000} MB)")
        return local
    # Already fully downloaded & valid size?
    if os.path.exists(local) and os.path.getsize(local) > 1_000_000:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tadabur-calib", "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers.get("Content-Range", "")
                total = int(cr.split("/")[-1]) if "/" in cr else 0
            if total and os.path.getsize(local) == total:
                log(f"shard {idx}: cached ({os.path.getsize(local)//1_000_000} MB)")
                return local
        except Exception:
            pass
    log(f"shard {idx}: downloading {url}")
    # determine total size
    total = 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tadabur-calib"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        log(f"shard {idx}: HEAD failed {e}")
    start = os.path.getsize(local) if os.path.exists(local) else 0
    mode = "ab" if start else "wb"
    t0 = time.time()
    got = start
    last_log = 0
    with open(local, mode) as fh:
        if start:
            log(f"  shard {idx}: resuming from {start//1_000_000} MB")
            fh.seek(start)
        while True:
            req = urllib.request.Request(
                url, headers={"User-Agent": "tadabur-calib", "Range": f"bytes={got}-"}
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
            except Exception as e:
                attempt = getattr(fh, "_attempt", 0) + 1
                fh._attempt = attempt
                if attempt >= 4:
                    log(f"  shard {idx}: download failed after 4 attempts ({e}); "
                        f"will retry from scratch on next run")
                    break
                log(f"  shard {idx}: read interrupted at {got//1_000_000} MB ({e}); "
                    f"retrying ({attempt}/4)...")
                time.sleep(15)
                continue
            now = time.time()
            if now - last_log > 15:
                dt = now - t0
                spd = ((got - start) / 1e6) / dt if dt > 0 else 0
                pct = (got / total * 100) if total else 0
                log(f"  shard {idx}: {got//1_000_000} MB {pct:.0f}% {spd:.2f} MB/s")
                last_log = now
            if total and got >= total:
                break
    log(f"shard {idx}: saved {os.path.getsize(local)//1_000_000} MB")
    return local

def resample_to_mono_16k(data, sr):
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR and sr and data.size:
        from fractions import Fraction
        from scipy.signal import resample_poly
        fr = Fraction(TARGET_SR, int(sr)).limit_denominator(1000)
        if fr.numerator != fr.denominator:
            # small rational factor -> tiny FIR filter, bounded RAM
            data = resample_poly(data.astype(np.float64), fr.numerator, fr.denominator)
    # normalize to roughly [-1, 1]
    peak = np.max(np.abs(data)) if data.size else 0
    if peak > 1.0:
        data = data / peak
    return data.astype(np.float32)

def process_shard(idx):
    local = download_parquet(idx)
    log(f"shard {idx}: extracting audio via mmap + streaming snappy (bounded RAM)...")
    # Safe extraction: reads SMALL columns via pyarrow, then mmap-streams the
    # ~2.14 GB audio dictionary page and writes each WAV immediately.  This
    # never materializes the audio column in RAM (the old pq.read_table with
    # "audio.bytes" saturated the machine's memory).
    import extract_shard
    rv = extract_shard.extract_audio(local, OUT, SURAH_MAX, log, shard_tag=f"shard{idx}")
    return rv

def main():
    log(f"Tadabur subset downloader. SHARDS={SHARDS} SURAH_MAX={SURAH_MAX}")
    log(f"Running at NORMAL priority; fits in RAM (~3-4 GB peak) so the UI stays responsive.")
    all_rec = {}
    done_path = os.path.join(OUT, "_shards_done.txt")
    done = set()
    if os.path.exists(done_path):
        done = set(line.strip() for line in open(done_path) if line.strip())
    for idx in SHARDS:
        if str(idx) in done:
            log(f"shard {idx}: already processed, skip")
            continue
        rv = process_shard(idx)
        for k, v in rv.items():
            all_rec[k] = all_rec.get(k, 0) + v
        with open(done_path, "a") as fh:
            fh.write(f"{idx}\n")
        gc.collect()
    log("=" * 50)
    log(f"TOTAL reciters={len(all_rec)} verses={sum(all_rec.values())}")
    top = sorted(all_rec.items(), key=lambda kv: -kv[1])[:15]
    log("Top reciters by verse count (surahs 1..%d): %s", SURAH_MAX, top)
    log("Output dir:", OUT)

if __name__ == "__main__":
    main()
