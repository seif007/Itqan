#!/usr/bin/env python3
"""Memory-safe extraction of verse WAVs + sidecars from a Tadabur shard.

Replaces the RAM-killing `pq.read_table(..., "audio.bytes")` formerly used in
download_tadabur.py.  The audio column is a ~2.14 GB dictionary page; we never
materialize it.  Instead (see _parquet_audio.iter_audio_bytes):

  * small columns (reciter/surah/ayah/metadata) are read via pyarrow alone;
  * the data page maps each row -> a dictionary index (bounded);
  * the dictionary page is stream-decompressed with a bounded raw-snappy
    decoder and each WAV blob is written and released immediately.

Peak RAM stays at a few tens of MB regardless of shard size.
"""
import os, io, json, sys, gc
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import _parquet_audio as P
import _snappy_stream as S

TARGET_SR = 16000


def resample_to_mono_16k(data, sr):
    """Downmix to mono and resample to TARGET_SR, normalizing to <=1.0."""
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR and sr and data.size:
        from fractions import Fraction
        from scipy.signal import resample_poly
        fr = Fraction(TARGET_SR, int(sr)).limit_denominator(1000)
        if fr.numerator != fr.denominator:
            data = resample_poly(data.astype(np.float64), fr.numerator, fr.denominator)
    peak = np.max(np.abs(data)) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    return data.astype(np.float32)


def extract_audio(parquet_path, out_dir, surah_max, log=None, shard_tag=None):
    """Extract every verse with surah_id <= surah_max into out_dir as
    <[shard_tag/]reciter>/<surah>_<ayah>_<row>.wav + .json sidecar.  Returns a
    dict {reciter_id: verse_count} for progress logging.

    `log` is an optional callable used for progress messages.  `shard_tag`
    (e.g. "shard0") namespaces the output so multiple shards never collide on
    per-shard row indices.
    """
    if log is None:
        log = lambda *a: print("[extract]", " ".join(str(x) for x in a))

    t = pq.read_table(parquet_path,
                      columns=["reciter_id", "surah_id", "ayah_id", "metadata"])
    rec = t.column("reciter_id").to_pylist()
    sur = t.column("surah_id").to_pylist()
    aya = t.column("ayah_id").to_pylist()
    meta = t.column("metadata").to_pylist()
    n = len(rec)
    log(f"shard: small columns read ({n} rows); mapping rows->audio via data page...")

    del t
    os.makedirs(out_dir, exist_ok=True)
    reciter_verses = {}
    written = 0
    skipped = 0

    def writer(rows, blob):
        nonlocal written, skipped
        if not rows:
            return
        try:
            data, sr = sf.read(io.BytesIO(blob))
        except Exception as e:
            log(f"  wav decode failed: {e}")
            return
        if data.size < 256:
            return
        pcm = resample_to_mono_16k(data, sr)
        del data
        for i in rows:
            surah = int(sur[i])
            if surah < 1 or surah > surah_max:
                continue
            reciter = int(rec[i])
            ayah = int(aya[i])
            rdir = (os.path.join(out_dir, shard_tag, str(reciter))
                    if shard_tag else os.path.join(out_dir, str(reciter)))
            base = f"{surah}_{ayah}_{i}"
            wav_name = base + ".wav"
            json_name = base + ".json"
            wav_path = os.path.join(rdir, wav_name)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                skipped += 1
                reciter_verses[reciter] = reciter_verses.get(reciter, 0) + 1
                continue
            m = meta[i]
            if isinstance(m, (bytes, bytearray)):
                m = m.decode("utf-8", "replace")
            mdict = json.loads(m) if isinstance(m, str) else (m or {})
            aligns = mdict.get("word_alignments", [])
            if not aligns:
                continue
            os.makedirs(rdir, exist_ok=True)
            sf.write(wav_path, pcm, TARGET_SR, subtype="PCM_16")
            wav_rel = (f"{shard_tag}/{reciter}/{wav_name}"
                       if shard_tag else f"{reciter}/{wav_name}")
            sidecar = {
                "reciter_id": reciter,
                "surah_id": surah,
                "ayah_id": ayah,
                "wav": wav_rel,
                "word_alignments": aligns,
                "text_ar_uthmani": mdict.get("text_ar_uthmani", ""),
                "text_ar_simple": mdict.get("text_ar_simple", ""),
                "ayah_duration_s": mdict.get("ayah_duration_s", 0.0),
            }
            with open(os.path.join(rdir, json_name), "w", encoding="utf-8") as fh:
                json.dump(sidecar, fh, ensure_ascii=False)
            reciter_verses[reciter] = reciter_verses.get(reciter, 0) + 1
            written += 1
        del pcm, blob

    P.iter_audio_bytes(parquet_path, writer)
    gc.collect()
    log(f"shard: done. reciters={len(reciter_verses)} "
        f"verses={sum(reciter_verses.values())} (wrote {written}, skipped {skipped})")
    return reciter_verses


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "tadabur_part"
    smax = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    extract_audio(path, out, smax)
