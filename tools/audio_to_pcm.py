"""
Fetch Alafasy audio for a surah/ayah range and convert to 16kHz mono PCM
(16-bit signed little-endian, the exact format `inject.pcm` expects on the
phone). Mirrors `AudioFetcher.kt` for the PC-side injection tool.

Usage:
  python audio_to_pcm.py SURAH [AYAH_START AYAH_END] -o out.pcm
  python audio_to_pcm.py SURAH 1 286 -o baqarah_full.pcm
  python audio_to_pcm.py 1 1 1 -o fatihah.pcm
  python audio_to_pcm.py 112 1 4 -o ikhlas.pcm

The Alafasy audio URLs follow the everyayah.com pattern:
  https://everyayah.com/data/Alafasy_128kbps/{surah:03d}{ayah:03d}.mp3
"""
import argparse
import os
import struct
import sys
import tempfile
import urllib.request


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'muno-injection/1.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def decode_to_pcm_int16(mp3_path: str):
    """Decode MP3 file → mono 16-bit PCM, returns (samples_list, src_sr).
    Uses librosa + soundfile so it works without ffmpeg.
    """
    import librosa
    import numpy as np
    # sr=None preserves native sample rate; mono=True gives 1 channel.
    y, sr = librosa.load(mp3_path, sr=None, mono=True)
    # y is float32 in [-1, 1]. Convert to int16.
    y_int16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    return y_int16.tolist(), sr


def resample_linear(samples, src_sr: int, dst_sr: int = 16000):
    if src_sr == dst_sr:
        return list(samples)
    n = int(len(samples) * dst_sr / src_sr)
    out = [0] * n
    ratio = src_sr / dst_sr
    for i in range(n):
        pos = i * ratio
        i0 = int(pos)
        i1 = min(i0 + 1, len(samples) - 1)
        frac = pos - i0
        out[i] = int(samples[i0] * (1 - frac) + samples[i1] * frac)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("surah", type=int)
    p.add_argument("ayah_start", type=int, nargs="?", default=1)
    p.add_argument("ayah_end", type=int, nargs="?", default=None)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Hard cap on output length (seconds); used to chunk long surahs")
    p.add_argument("--silence-between-ayahs", type=float, default=0.7,
                   help="Seconds of silence inserted between ayahs (default 0.7)")
    p.add_argument("--cache-dir", default=os.path.join(tempfile.gettempdir(), "muno_inject_cache"))
    args = p.parse_args()

    if args.ayah_end is None:
        args.ayah_end = args.ayah_start

    os.makedirs(args.cache_dir, exist_ok=True)
    all_samples = []
    src_sr = None
    for ayah in range(args.ayah_start, args.ayah_end + 1):
        cache_path = os.path.join(args.cache_dir, f"alafasy_{args.surah:03d}{ayah:03d}.mp3")
        if not os.path.exists(cache_path):
            url = f"https://everyayah.com/data/Alafasy_128kbps/{args.surah:03d}{ayah:03d}.mp3"
            print(f"  fetch {url}", file=sys.stderr)
            with open(cache_path, "wb") as f:
                f.write(fetch_bytes(url))
        samples, sr = decode_to_pcm_int16(cache_path)
        if src_sr is None:
            src_sr = sr
        elif sr != src_sr:
            print(f"  WARN: ayah {ayah} sr={sr} differs from first sr={src_sr}; using first",
                  file=sys.stderr)
        print(f"  ayah {ayah}: {len(samples)/src_sr:.2f}s @ {src_sr}Hz", file=sys.stderr)
        all_samples.extend(samples)
        all_samples.extend([0] * int(args.silence_between_ayahs * src_sr))

    if src_sr != 16000:
        print(f"  resample {src_sr}Hz -> 16000Hz", file=sys.stderr)
        all_samples = resample_linear(all_samples, src_sr, 16000)
        src_sr = 16000

    if args.max_seconds is not None:
        cap = int(args.max_seconds * 16000)
        if len(all_samples) > cap:
            all_samples = all_samples[:cap]

    with open(args.out, "wb") as f:
        # chunked write to avoid struct.pack arg-count limits on large lists
        chunk = 65536
        n = len(all_samples)
        print(f"  writing {n} samples ({n*2} bytes) to {args.out}", file=sys.stderr)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            f.write(struct.pack(f"<{j-i}h", *all_samples[i:j]))
    dur = len(all_samples) / 16000
    print(f"  wrote {args.out}: {dur:.2f}s ({len(all_samples)*2} bytes)")


if __name__ == "__main__":
    main()
