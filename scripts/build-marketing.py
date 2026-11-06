#!/usr/bin/env python3
"""Build Itqan's marketing artwork and promo video.

The source screenshots come from a phone and include Android's status and
navigation bars.  Everything in docs/marketing/store is composed from the
cropped UI-only copies in docs/marketing/screenshots, so no system chrome ends
up in a store listing.

Usage:
    python scripts/build-marketing.py crop       # clean the raw phone shots
    python scripts/build-marketing.py static     # store frames, hero, banner
    python scripts/build-marketing.py video      # vertical + 16:9 promo + web cut
    python scripts/build-marketing.py all

Raw screenshots are read from Desktop/itqan_screenshots by default (override
with --raw).  Once docs/marketing/screenshots exists, crop is optional.

The promo videos are silent on purpose.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MARKETING = ROOT / "docs" / "marketing"
CLEAN = MARKETING / "screenshots"
STORE = MARKETING / "store"
TMP = ROOT / ".tmp-marketing"

W, H = 1080, 1920
LW, LH = 1920, 1080

# The phone produces 1080x2400 images.  The status bar ends at 126 px and the
# navigation bar begins at 2266 px; the crop keeps the app UI only.
CROP_TOP = 126
CROP_BOTTOM = 2265

RAW_MAP = {
    "00_startup.png": "01-live.png",
    "tajweed_forcedpopup_words_4s.png": "02-tajweed-mode.png",
    "04_tajweed_mistake_popup.png": "03-mistake.png",
    "05_tajweed_skipped_popup.png": "04-skipped.png",
    "06_tajweed_summary_all.png": "05-summary.png",
    "02_read_word_2.png": "06-read.png",
    "07_coach_initial.png": "07-coach-rules.png",
    "07b_coach_expanded.png": "08-coach-explain.png",
    "332a8ba5-a8ba-4dd3-bf05-bf48e649dfab.jpg": "09-coach-practice.png",
    "01_stats_initial.png": "10-goals.png",
}

# This shot arrived at 922x2048 with a different system-bar layout: the status
# bar ends at 110 px and the white navigation bar starts at 1933 px.
CROP_OVERRIDES = {
    "332a8ba5-a8ba-4dd3-bf05-bf48e649dfab.jpg": (110, 1932),
}

# --------------------------------------------------------------------------- palette

BG_TOP = (6, 20, 22)
BG_BOTTOM = (3, 9, 10)
TEXT = (241, 245, 245)
MUTED = (190, 200, 200)
DIM = (120, 140, 142)
TEAL = (94, 234, 212)
TEAL_D = (20, 184, 166)
EMERALD = (52, 211, 153)
CYAN = (34, 211, 238)
AMBER = (251, 191, 36)

FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
FONT_BOLD = FONT_DIR / "segoeuib.ttf"
FONT_SEMI = FONT_DIR / "segoeuisl.ttf"
FONT_REG = FONT_DIR / "segoeui.ttf"
FONT_BLACK = FONT_BOLD


def _font_path(path: Path, fallback: Path) -> Path:
    return path if path.exists() else fallback


FONT_SEMI = _font_path(FONT_SEMI, FONT_BOLD)
FONT_REG = _font_path(FONT_REG, FONT_BOLD)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def text_width(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=f)
    return box[2] - box[0]


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start: int,
    minimum: int,
    path: Path = FONT_BOLD,
) -> ImageFont.FreeTypeFont:
    size = start
    while size > minimum:
        f = font(path, size)
        if text_width(draw, text, f) <= max_width:
            return f
        size -= 1
    return font(path, minimum)


def draw_tracked(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    f: ImageFont.FreeTypeFont,
    fill,
    tracking: int,
) -> int:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=f, fill=fill)
        x += text_width(draw, ch, f) + tracking
    return x - tracking


def tracked_width(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, tracking: int) -> int:
    if not text:
        return 0
    return sum(text_width(draw, ch, f) for ch in text) + tracking * (len(text) - 1)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def ease_out(x: float) -> float:
    return 1.0 - (1.0 - clamp01(x)) ** 3


# --------------------------------------------------------------------------- background


def make_background(accent: tuple[int, int, int], width: int = W, height: int = H) -> Image.Image:
    """A dark, quiet teal backdrop with one accent glow and a faint lattice."""
    top = np.array(BG_TOP, dtype=np.float32)
    bottom = np.array(BG_BOTTOM, dtype=np.float32)
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    arr = top[None, None, :] * (1.0 - t) + bottom[None, None, :] * t
    arr = np.repeat(arr, width, axis=1)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    for cx, cy, radius, strength, colour in (
        (width * 0.88, -height * 0.08, width * 0.74, 0.40, accent),
        (width * 0.05, height * 0.94, width * 0.62, 0.15, TEAL_D),
    ):
        d = np.sqrt(((xx - cx) / radius) ** 2 + ((yy - cy) / radius) ** 2)
        a = np.clip(1.0 - d, 0.0, 1.0) ** 2.2 * strength
        arr = arr * (1.0 - a[..., None]) + np.array(colour, dtype=np.float32) * a[..., None]

    d = np.sqrt(((xx - width / 2) / (width * 0.78)) ** 2 + ((yy - height / 2) / (height * 0.78)) ** 2)
    vig = np.clip(1.0 - 0.30 * np.clip(d - 0.45, 0.0, 1.0) / 0.55, 0.0, 1.0)
    arr *= vig[..., None]

    rng = np.random.default_rng(7)
    arr += rng.normal(0.0, 2.1, (height, width, 1))

    base = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    # A faint diagonal lattice, like woven geometric ornament.
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    step = max(72, int(width / 11))
    for x in range(-height, width + height, step):
        od.line((x, 0, x + height, height), fill=(255, 255, 255, 5), width=1)
        od.line((x, height, x + height, 0), fill=(255, 255, 255, 4), width=1)
    base = Image.alpha_composite(base, overlay)
    return base.convert("RGB")


_BG_CACHE: dict[tuple[int, int, int, int, int], Image.Image] = {}


def background(accent: tuple[int, int, int], width: int = W, height: int = H) -> Image.Image:
    key = (accent[0], accent[1], accent[2], width, height)
    if key not in _BG_CACHE:
        _BG_CACHE[key] = make_background(accent, width, height).convert("RGBA")
    return _BG_CACHE[key].copy()


# --------------------------------------------------------------------------- phone mockup


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def make_phone(
    shot_path: Path,
    width: int,
    accent: tuple[int, int, int],
    bezel: int = 9,
    radius_ratio: float = 0.058,
    glow: bool = True,
) -> Image.Image:
    """Return an RGBA mockup with drop shadow and accent halo."""
    shot = Image.open(shot_path).convert("RGB")
    height = round(width * shot.height / shot.width)
    shot = shot.resize((width, height), Image.Resampling.LANCZOS)
    radius = max(18, round(width * radius_ratio))

    pad = max(46, round(width * 0.13))
    outer_w = width + bezel * 2
    outer_h = height + bezel * 2
    canvas_w = outer_w + pad * 2
    canvas_h = outer_h + pad * 2

    if glow:
        halo = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        hd = ImageDraw.Draw(halo)
        inset = pad - max(10, width // 34)
        hd.rounded_rectangle(
            (inset, inset + 8, canvas_w - inset, canvas_h - inset + 8),
            radius=radius + max(16, width // 20),
            fill=(*accent, 130),
        )
        halo = halo.filter(ImageFilter.GaussianBlur(max(18, width // 18)))

    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        (pad, pad + max(16, width // 14), pad + outer_w, pad + outer_h + max(16, width // 14)),
        radius=radius + bezel,
        fill=(0, 0, 0, 190),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(14, width // 26)))

    body = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    bd.rounded_rectangle(
        (pad, pad, pad + outer_w, pad + outer_h),
        radius=radius + bezel,
        fill=(9, 26, 29, 255),
    )
    bd.rounded_rectangle(
        (pad + 1, pad + 1, pad + outer_w - 1, pad + outer_h - 1),
        radius=radius + bezel - 1,
        outline=(255, 255, 255, 28),
        width=2,
    )
    shot_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    shot_layer.paste(shot, (pad + bezel, pad + bezel), rounded_mask((width, height), radius))
    body = Image.alpha_composite(body, shot_layer)

    out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if glow:
        out = Image.alpha_composite(out, halo)
    out = Image.alpha_composite(out, shadow)
    out = Image.alpha_composite(out, body)
    return out


def paste_centered(canvas: Image.Image, layer: Image.Image, cx: int, y: int) -> None:
    canvas.alpha_composite(layer, (round(cx - layer.width / 2), round(y)))


def set_alpha(layer: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 0.999:
        return layer
    out = layer.copy()
    a = out.getchannel("A").point(lambda v: int(v * alpha))
    out.putalpha(a)
    return out


# --------------------------------------------------------------------------- icon


# The launcher icon is an adaptive icon: a flat #040D0E background plus the
# microphone / sound-wave / open-book foreground from the app's own vector
# drawable.  Rendering that vector keeps the video's icon the real app icon
# instead of a stand-in logo.
ICON_FOREGROUND = ROOT / "app" / "src" / "main" / "res" / "drawable" / "ic_launcher_foreground.xml"
ICON_BACKGROUND = (4, 13, 14)  # #040D0E, from ic_launcher_background.xml


def _parse_color(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    v = value.strip().lstrip("#")
    if len(v) == 8:  # #AARRGGBB
        return tuple(int(v[i : i + 2], 16) for i in (2, 4, 6, 0))  # type: ignore[return-value]
    if len(v) == 6:  # #RRGGBB
        return tuple(int(v[i : i + 2], 16) for i in (0, 2, 4)) + (255,)  # type: ignore[return-value]
    return None


def _parse_path(d: str) -> list[tuple[str, list[float]]]:
    """Split an SVG path string into (command, numbers) pairs.

    Implicit repeats are expanded; a repeated moveto becomes a lineto, as SVG
    specifies.
    """
    items = re.findall(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", d)
    arity = {
        "M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7,
        "m": 2, "l": 2, "h": 1, "v": 1, "c": 6, "s": 4, "q": 4, "t": 2, "a": 7,
    }
    out: list[tuple[str, list[float]]] = []
    i = 0
    while i < len(items):
        cmd = items[i]
        if not cmd.isalpha():
            raise ValueError(f"malformed path data: {d!r}")
        i += 1
        if cmd in "Zz":
            out.append((cmd, []))
            continue
        n = arity[cmd]
        first = True
        while i < len(items) and not items[i].isalpha():
            lead = cmd if (first or cmd not in "Mm") else ("L" if cmd == "M" else "l")
            out.append((lead, [float(x) for x in items[i : i + n]]))
            first = False
            i += n
    return out


def _cubic(p0, p1, p2, p3, n: int = 24) -> list[tuple[float, float]]:
    pts = []
    for k in range(1, n + 1):
        t = k / n
        u = 1.0 - t
        pts.append((
            u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
            u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1],
        ))
    return pts


def _quad(p0, p1, p2, n: int = 24) -> list[tuple[float, float]]:
    pts = []
    for k in range(1, n + 1):
        t = k / n
        u = 1.0 - t
        pts.append((
            u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
        ))
    return pts


def _arc(p0, rx, ry, rot, large, sweep, p1, n: int = 40) -> list[tuple[float, float]]:
    """SVG elliptical arc from p0 to p1 as sampled points (spec F.6.5)."""
    if rx == 0 or ry == 0:
        return [p1]
    phi = math.radians(rot)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2.0, (p0[1] - p1[1]) / 2.0
    xp = cp * dx + sp * dy
    yp = -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = xp * xp / (rx * rx) + yp * yp / (ry * ry)
    if lam > 1.0:
        s = math.sqrt(lam)
        rx *= s
        ry *= s
    num = rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp
    den = rx * rx * yp * yp + ry * ry * xp * xp
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large == sweep:
        coef = -coef
    cxp = coef * rx * yp / ry
    cyp = -coef * ry * xp / rx
    cx = cp * cxp - sp * cyp + (p0[0] + p1[0]) / 2.0
    cy = sp * cxp + cp * cyp + (p0[1] + p1[1]) / 2.0

    def angle(ux, uy, vx, vy):
        nrm = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / nrm)))
        return -a if ux * vy - uy * vx < 0 else a

    u = ((xp - cxp) / rx, (yp - cyp) / ry)
    v = ((-xp - cxp) / rx, (-yp - cyp) / ry)
    th0 = angle(1.0, 0.0, *u)
    dth = angle(*u, *v)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    elif sweep and dth < 0:
        dth += 2 * math.pi
    pts = []
    for k in range(1, n + 1):
        th = th0 + dth * k / n
        pts.append((
            cx + rx * math.cos(th) * cp - ry * math.sin(th) * sp,
            cy + rx * math.cos(th) * sp + ry * math.sin(th) * cp,
        ))
    return pts


def _vector_subpaths(d: str) -> list[list[tuple[float, float]]]:
    """Flatten a path string into polylines, one per subpath."""
    subs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    x = y = 0.0
    start = (0.0, 0.0)
    for cmd, a in _parse_path(d):
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = (x + a[0], y + a[1]) if rel else (a[0], a[1])
            cur = [(x, y)]
            subs.append(cur)
            start = (x, y)
        elif c == "L":
            x, y = (x + a[0], y + a[1]) if rel else (a[0], a[1])
            cur.append((x, y))
        elif c == "H":
            x = x + a[0] if rel else a[0]
            cur.append((x, y))
        elif c == "V":
            y = y + a[0] if rel else a[0]
            cur.append((x, y))
        elif c == "C":
            p1 = (x + a[0], y + a[1]) if rel else (a[0], a[1])
            p2 = (x + a[2], y + a[3]) if rel else (a[2], a[3])
            p3 = (x + a[4], y + a[5]) if rel else (a[4], a[5])
            cur.extend(_cubic((x, y), p1, p2, p3))
            x, y = p3
        elif c == "Q":
            p1 = (x + a[0], y + a[1]) if rel else (a[0], a[1])
            p2 = (x + a[2], y + a[3]) if rel else (a[2], a[3])
            cur.extend(_quad((x, y), p1, p2))
            x, y = p2
        elif c == "A":
            ex, ey = (x + a[5], y + a[6]) if rel else (a[5], a[6])
            cur.extend(_arc((x, y), a[0], a[1], a[2], int(a[3]), int(a[4]), (ex, ey)))
            x, y = ex, ey
        elif c == "Z":
            if cur:
                cur.append(start)
            x, y = start
    return subs


_ICON_PATHS: list[tuple] | None = None


def _icon_paths() -> list[tuple]:
    """(fill, stroke, stroke_width, subpaths) for each vector path, in order."""
    global _ICON_PATHS
    if _ICON_PATHS is None:
        xml = ICON_FOREGROUND.read_text(encoding="utf-8")
        out = []
        for tag in re.findall(r"<path\b[^>]*/>", xml):
            def attr(name: str) -> str | None:
                m = re.search(rf'android:{name}="([^"]*)"', tag)
                return m.group(1) if m else None

            out.append((
                _parse_color(attr("fillColor")),
                _parse_color(attr("strokeColor")),
                float(attr("strokeWidth") or 0.0),
                _vector_subpaths(attr("pathData") or ""),
            ))
        _ICON_PATHS = out
    return _ICON_PATHS


_ICON_ART: dict[int, Image.Image] = {}


def launcher_icon(size: int) -> Image.Image:
    """The app's real launcher icon, rasterised from its VectorDrawable."""
    if size not in _ICON_ART:
        supersample = 4
        dim = size * supersample
        scale = dim / 108.0  # the drawable's 108x108 viewport
        art = Image.new("RGBA", (dim, dim), (0, 0, 0, 0))
        d = ImageDraw.Draw(art)
        d.rectangle((0, 0, dim - 1, dim - 1), fill=(*ICON_BACKGROUND, 255))
        for fill, stroke, width, subs in _icon_paths():
            if fill is not None:
                for sub in subs:
                    if len(sub) >= 3:
                        d.polygon([(px * scale, py * scale) for px, py in sub], fill=fill)
            if stroke is not None:
                w = max(1, round(width * scale))
                for sub in subs:
                    pts = [(px * scale, py * scale) for px, py in sub]
                    if len(pts) >= 2:
                        d.line(pts, fill=stroke, width=w, joint="curve")
                        r = w / 2.0
                        for px, py in (pts[0], pts[-1]):  # round caps
                            d.ellipse((px - r, py - r, px + r, py + r), fill=stroke)
        _ICON_ART[size] = art.resize((size, size), Image.Resampling.LANCZOS)
    return _ICON_ART[size]


_ICON_BASE: Image.Image | None = None


def icon_layer(size: int, accent: tuple[int, int, int] = TEAL) -> Image.Image:
    """The app's real launcher icon, with the same halo and shadow as the phones."""
    global _ICON_BASE
    if _ICON_BASE is None:
        base = 420
        pad = 64
        canvas = Image.new("RGBA", (base + pad * 2, base + pad * 2), (0, 0, 0, 0))

        halo = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(halo)
        hd.rounded_rectangle(
            (pad - 18, pad - 10, pad + base + 18, pad + base + 26),
            radius=round(base * 0.26),
            fill=(*accent, 120),
        )
        halo = halo.filter(ImageFilter.GaussianBlur(34))
        canvas = Image.alpha_composite(canvas, halo)

        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rounded_rectangle(
            (pad, pad + 22, pad + base, pad + base + 22),
            radius=round(base * 0.23),
            fill=(0, 0, 0, 190),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(24))
        canvas = Image.alpha_composite(canvas, shadow)

        # Rounded square in the icon's own colours, carrying the app's real
        # foreground art (microphone, sound waves, open book).
        body = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        radius = round(base * 0.23)
        body.paste(launcher_icon(base), (pad, pad), rounded_mask((base, base), radius))
        bd = ImageDraw.Draw(body)
        bd.rounded_rectangle(
            (pad, pad, pad + base - 1, pad + base - 1),
            radius=radius,
            outline=(255, 255, 255, 34),
            width=2,
        )
        canvas = Image.alpha_composite(canvas, body)
        _ICON_BASE = canvas

    scale = size / 420
    return _ICON_BASE.resize(
        (round(_ICON_BASE.width * scale), round(_ICON_BASE.height * scale)),
        Image.Resampling.LANCZOS,
    )


def draw_wordmark(
    d: ImageDraw.ImageDraw,
    x: int,
    y: int,
    size: int,
    alpha: float = 1.0,
    center: int | None = None,
) -> tuple[int, int]:
    f = font(FONT_BOLD, size)
    a = int(255 * clamp01(alpha))
    head = "Itq"
    tail = "\u0101n"  # macron a: Itqan
    tw = text_width(d, head, f) + text_width(d, tail, f)
    if center is not None:
        x = center - tw // 2
    d.text((x, y), head, font=f, fill=(*TEXT, a))
    d.text((x + text_width(d, head, f), y), tail, font=f, fill=(*TEAL, a))
    return x + tw, y


def draw_chip(
    d: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    f: ImageFont.FreeTypeFont,
    alpha: float,
    accent: tuple[int, int, int] = TEAL,
) -> int:
    a = int(255 * clamp01(alpha))
    tw = text_width(d, text, f)
    h = round(f.size * 2.0)
    d.rounded_rectangle(
        (x, y, x + tw + round(f.size * 1.45), y + h),
        radius=h // 2,
        fill=(255, 255, 255, int(12 * clamp01(alpha))),
        outline=(*accent, int(70 * clamp01(alpha))),
        width=1,
    )
    d.text((x + round(f.size * 0.72), y + round(f.size * 0.44)), text, font=f, fill=(*TEXT, a))
    return x + tw + round(f.size * 1.45) + round(f.size * 0.6)


# --------------------------------------------------------------------------- store frames


FEATURES = [
    dict(
        n=1,
        shot="01-live.png",
        kicker="LIVE RECITATION",
        headline="Recite.\nIt follows every word.",
        sub="Word-by-word tracking as you read, on device.",
        accent=TEAL,
    ),
    dict(
        n=2,
        shot="02-tajweed-mode.png",
        kicker="TAJWEED MODE",
        headline="Eight rules,\nscored as you recite.",
        sub="Madd \u00b7 Ghunnah \u00b7 Qalqalah \u00b7 Idgham \u00b7 Ikhfa \u00b7 Meem \u00b7 Izhar \u00b7 Al-Lam.",
        accent=CYAN,
    ),
    dict(
        n=3,
        shot="03-mistake.png",
        kicker="TAJWEED MISTAKE",
        headline="Missed a rule?\nIt tells you which.",
        sub="A verified mistake pauses at the word and names the rule.",
        accent=AMBER,
    ),
    dict(
        n=4,
        shot="04-skipped.png",
        kicker="WRONG & SKIPPED",
        headline="Wrong or skipped,\ncaught instantly.",
        sub="The highlight freezes and you re-read from that word.",
        accent=AMBER,
    ),
    dict(
        n=5,
        shot="05-summary.png",
        kicker="SESSION SUMMARY",
        headline="Every session,\nsummarised honestly.",
        sub="Wrong words, skipped words and tajweed mistakes \u2014 no praise.",
        accent=TEAL,
    ),
    dict(
        n=6,
        shot="06-read.png",
        kicker="READ",
        headline="Read with full\ntajweed colouring.",
        sub="Letter-by-letter colouring with ayah audio on loop.",
        accent=EMERALD,
    ),
    dict(
        n=7,
        shot="07-coach-rules.png",
        kicker="TAJWEED COACH",
        headline="Learn each rule,\none card at a time.",
        sub="Eight rules, plain explanations and real Quranic examples.",
        accent=TEAL,
    ),
    dict(
        n=8,
        shot="08-coach-explain.png",
        kicker="RULE EXPLANATIONS",
        headline="Understand why,\nnot just what.",
        sub="Solar and lunar lam, idgham, ikhfa, izhar and more.",
        accent=CYAN,
    ),
    dict(
        n=9,
        shot="09-coach-practice.png",
        kicker="COACH PRACTICE",
        headline="Practise with\nlive feedback.",
        sub="Recite the example and get scored against its rules.",
        accent=EMERALD,
    ),
    dict(
        n=10,
        shot="10-goals.png",
        kicker="GOALS & STATS",
        headline="Set a goal.\nWatch it progress.",
        sub="Daily or weekly goals with a per-rule mistake breakdown.",
        accent=TEAL,
    ),
]


def draw_store_top(
    canvas: Image.Image,
    accent: tuple[int, int, int],
    n: int,
    total: int,
    kicker: str,
    headline: str,
    sub: str,
) -> int:
    d = ImageDraw.Draw(canvas)
    margin = 84

    brand_f = font(FONT_BOLD, 30)
    d.text((margin, 62), "Itq", font=brand_f, fill=TEXT)
    head_w = text_width(d, "Itq", brand_f)
    d.text((margin + head_w, 62), "\u0101n", font=brand_f, fill=TEAL)
    counter_f = font(FONT_SEMI, 25)
    counter = f"{n:02d} / {total:02d}"
    d.text((W - margin - text_width(d, counter, counter_f), 64), counter, font=counter_f, fill=(*accent, 230))

    kick_f = font(FONT_BOLD, 25)
    y = 138
    draw_tracked(d, (margin, y), kicker.upper(), kick_f, accent, 6)
    y += 42

    lines = headline.split("\n")
    max_w = W - margin * 2
    size = 88
    for line in lines:
        size = min(size, fit_font(d, line, max_w, size, 62).size)
    head_f = font(FONT_BLACK, size)
    line_h = round(size * 1.13)
    for line in lines:
        d.text((margin, y), line, font=head_f, fill=TEXT)
        y += line_h
    y += 14

    sub_size = 29
    while text_width(d, sub, font(FONT_REG, sub_size)) > max_w and sub_size > 20:
        sub_size -= 1
    d.text((margin, y), sub, font=font(FONT_REG, sub_size), fill=MUTED)
    y += round(sub_size * 1.55)
    return y


def build_store() -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    for f in FEATURES:
        accent = f["accent"]
        canvas = background(accent).convert("RGBA")
        y = draw_store_top(canvas, accent, f["n"], len(FEATURES), f["kicker"], f["headline"], f["sub"])

        phone = make_phone(CLEAN / f["shot"], 680, accent)
        top = max(y + 24, 470)
        if top + phone.height - 46 > H:
            top = H - phone.height + 46
        paste_centered(canvas, phone, W // 2, top)
        name = f"{f['n']:02d}-" + f["kicker"].lower().replace("&", "and").replace("'", "").replace(" ", "-") + "-1080x1920.png"
        canvas.convert("RGB").save(STORE / name, quality=95, optimize=True)
        print(f"  ok {name}")


def build_banner() -> None:
    """1024x500 Play Store feature graphic."""
    w, h = 1024, 500
    canvas = background(TEAL, w, h).convert("RGBA")
    d = ImageDraw.Draw(canvas)

    brand_f = font(FONT_BOLD, 52)
    d.text((60, 46), "Itq", font=brand_f, fill=TEXT)
    cw = text_width(d, "Itq", brand_f)
    d.text((60 + cw, 46), "\u0101n", font=brand_f, fill=TEAL)

    tag_f = font(FONT_BOLD, 31)
    d.text((60, 114), "Recite with itqan.", font=tag_f, fill=TEXT)
    body_f = font(FONT_REG, 19)
    body = "Live word tracking, on-device Tajweed scoring,\na rules coach and recitation goals."
    yy = 160
    for line in body.split("\n"):
        d.text((60, yy), line, font=body_f, fill=MUTED)
        yy += 27

    chips = ["8 tajweed rules", "Hafs & Qalun", "Offline-first"]
    chip_f = font(FONT_SEMI, 16)
    x = 60
    for chip in chips:
        tw = text_width(d, chip, chip_f)
        d.rounded_rectangle((x, 232, x + tw + 28, 268), radius=18, outline=(255, 255, 255, 38), width=1)
        d.text((x + 14, 240), chip, font=chip_f, fill=TEXT)
        x += tw + 40

    d.text((60, 435), "Android \u00b7 On-device speech recognition", font=font(FONT_SEMI, 16), fill=DIM)

    front = make_phone(CLEAN / "01-live.png", 214, TEAL, bezel=6, glow=False)
    rear = make_phone(CLEAN / "03-mistake.png", 172, AMBER, bezel=6, glow=False)
    rear = rear.rotate(-8, expand=True, resample=Image.Resampling.BICUBIC)
    front = front.rotate(4, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(rear, (640, 92))
    canvas.alpha_composite(front, (742, 54))
    canvas.convert("RGB").save(MARKETING / "feature-graphic-1024x500.png", quality=96, optimize=True)
    print("  ok feature-graphic-1024x500.png")


def build_hero() -> None:
    """1600x800 README hero."""
    w, h = 1600, 800
    canvas = background(TEAL, w, h).convert("RGBA")
    d = ImageDraw.Draw(canvas)

    brand_f = font(FONT_BOLD, 78)
    d.text((100, 82), "Itq", font=brand_f, fill=TEXT)
    cw = text_width(d, "Itq", brand_f)
    d.text((100 + cw, 82), "\u0101n", font=brand_f, fill=TEAL)

    tag_f = font(FONT_BLACK, 54)
    d.text((100, 194), "Recite with itqan.", font=tag_f, fill=TEXT)

    body_f = font(FONT_REG, 25)
    body = (
        "Live recitation tracking, on-device Tajweed scoring,\n"
        "a rules coach and recitation goals \u2014 offline-first."
    )
    yy = 274
    for line in body.split("\n"):
        d.text((100, yy), line, font=body_f, fill=MUTED)
        yy += 36

    chips = ["8 tajweed rules", "Hafs & Qalun", "English & Arabic", "Android"]
    chip_f = font(FONT_SEMI, 20)
    x = 100
    for chip in chips:
        tw = text_width(d, chip, chip_f)
        d.rounded_rectangle((x, 374, x + tw + 32, 416), radius=21, outline=(255, 255, 255, 38), width=1)
        d.text((x + 16, 384), chip, font=chip_f, fill=TEXT)
        x += tw + 46

    d.text((100, 700), "Offline-first Quran recitation practice for Android", font=font(FONT_SEMI, 20), fill=DIM)

    phones = [
        ("02-tajweed-mode.png", 280, CYAN, -8, 930),
        ("01-live.png", 310, TEAL, 0, 1110),
        ("06-read.png", 280, EMERALD, 8, 1300),
    ]
    layers = []
    for shot, width, accent, angle, x in phones:
        layer = make_phone(CLEAN / shot, width, accent, bezel=7, glow=False)
        if angle:
            layer = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        layers.append((layer, x))
    for layer, x in layers:
        canvas.alpha_composite(layer, (x, 90))
    canvas.convert("RGB").save(MARKETING / "hero-1600x800.png", quality=96, optimize=True)
    print("  ok hero-1600x800.png")


# --------------------------------------------------------------------------- video helpers


_PHONE_CACHE: dict[tuple[str, int, tuple[int, int, int], bool], Image.Image] = {}


def phone(shot: str, width: int, accent: tuple[int, int, int], glow: bool = False) -> Image.Image:
    key = (shot, width, accent, glow)
    if key not in _PHONE_CACHE:
        _PHONE_CACHE[key] = make_phone(CLEAN / shot, width, accent, glow=glow)
    return _PHONE_CACHE[key]


def draw_feature_copy(
    d: ImageDraw.ImageDraw,
    scene: dict,
    x: int,
    y: int,
    max_width: int,
    alpha: float,
    kicker_size: int = 25,
    headline_size: int = 88,
    sub_size: int = 29,
    headline_font: Path = FONT_BLACK,
) -> int:
    a = int(255 * clamp01(alpha))
    accent = scene["accent"]
    kf = font(FONT_BOLD, kicker_size)
    draw_tracked(d, (x, y), scene["kicker"].upper(), kf, (*accent, a), 6)
    y += round(kicker_size * 1.7)

    lines = scene["headline"].split("\n")
    size = headline_size
    for line in lines:
        size = min(size, fit_font(d, line, max_width, size, 60, headline_font).size)
    hf = font(headline_font, size)
    line_h = round(size * 1.13)
    for line in lines:
        d.text((x, y), line, font=hf, fill=(*TEXT, a))
        y += line_h
    y += 14

    sf = font(FONT_REG, sub_size)
    while text_width(d, scene["sub"], sf) > max_width and sf.size > 20:
        sf = font(FONT_REG, sf.size - 1)
    d.text((x, y), scene["sub"], font=sf, fill=(*MUTED, a))
    return y + round(sf.size * 1.55)


def feature_content_vertical(scene: dict, local: float, duration: float) -> Image.Image:
    content = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(content)
    accent = scene["accent"]

    word_a = smoothstep(local / 0.30)
    draw_wordmark(d, 84, 62, 30, word_a)
    cf = font(FONT_SEMI, 25)
    counter = f"{scene['n']:02d} / {len(FEATURES):02d}"
    d.text((W - 84 - text_width(d, counter, cf), 64), counter, font=cf, fill=(*accent, int(230 * word_a)))

    text_a = smoothstep(local / 0.45)
    dy = round(26 * (1.0 - ease_out(clamp01(local / 0.6))))
    draw_feature_copy(d, scene, 84, 138 + dy, W - 168, text_a)

    phone_a = smoothstep((local - 0.10) / 0.55)
    phone_dy = round(110 * (1.0 - ease_out(clamp01((local - 0.10) / 0.8))))
    if phone_a > 0.01:
        layer = phone(scene["shot"], 680, accent)
        x = W // 2 - layer.width // 2
        y = H - layer.height + 46 + phone_dy - round(7 * clamp01(local / duration))
        content.alpha_composite(set_alpha(layer, phone_a), (x, y))
    return content


def feature_content_landscape(scene: dict, local: float, duration: float, index: int = 0) -> Image.Image:
    content = Image.new("RGBA", (LW, LH), (0, 0, 0, 0))
    d = ImageDraw.Draw(content)
    accent = scene["accent"]

    word_a = smoothstep(local / 0.30)
    draw_wordmark(d, 110, 58, 34, word_a)
    cf = font(FONT_SEMI, 25)
    counter = f"{scene['n']:02d} / {len(FEATURES):02d}"
    d.text((LW - 100 - text_width(d, counter, cf), 62), counter, font=cf, fill=(*accent, int(230 * word_a)))

    text_a = smoothstep(local / 0.45)
    dy = round(24 * (1.0 - ease_out(clamp01(local / 0.6))))
    draw_feature_copy(d, scene, 110, 180 + dy, 840, text_a, kicker_size=25, headline_size=96, sub_size=30)

    prev_shot = FEATURES[(index - 1) % len(FEATURES)]["shot"] if index else "02-tajweed-mode.png"
    rear = phone(prev_shot, 300, accent, glow=False).rotate(-8, expand=True, resample=Image.Resampling.BICUBIC)
    rear = set_alpha(rear, 0.92 * smoothstep((local - 0.05) / 0.7))
    content.alpha_composite(rear, (890, 290))

    phone_a = smoothstep((local - 0.10) / 0.55)
    phone_dx = round(120 * (1.0 - ease_out(clamp01((local - 0.10) / 0.8))))
    if phone_a > 0.01:
        layer = set_alpha(phone(scene["shot"], 500, accent), phone_a)
        x = 1300 + phone_dx
        y = -16 - round(5 * clamp01(local / duration))
        content.alpha_composite(layer, (x, y))
    return content


# --------------------------------------------------------------------------- intro / outro

INTRO_LINE = "Live tracking \u00b7 Tajweed scoring \u00b7 Rules coach \u00b7 Goals"


def intro_content_vertical(local: float, duration: float) -> Image.Image:
    content = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(content)

    icon_a = smoothstep(local / 0.55)
    if icon_a > 0.01:
        size = round(300 * (0.90 + 0.10 * ease_out(clamp01(local / 0.7))))
        ic = set_alpha(icon_layer(size), icon_a)
        content.alpha_composite(ic, (W // 2 - ic.width // 2, 330 - ic.height // 2))

    word_a = smoothstep((local - 0.18) / 0.55)
    draw_wordmark(d, 0, 700, 92, word_a, center=W // 2)

    tag_a = smoothstep((local - 0.42) / 0.55)
    tf = font(FONT_BOLD, 44)
    tag = "Recite with itqan."
    d.text((W // 2 - text_width(d, tag, tf) // 2, 830), tag, font=tf, fill=(*TEAL, int(255 * tag_a)))

    line_a = smoothstep((local - 0.62) / 0.5)
    if line_a > 0.01:
        half = round(190 * ease_out(clamp01((local - 0.62) / 0.7)))
        d.rounded_rectangle((W // 2 - half, 918, W // 2 + half, 920), radius=1, fill=(*EMERALD, int(210 * line_a)))

    feat_a = smoothstep((local - 0.72) / 0.55)
    ff = font(FONT_REG, 28)
    d.text((W // 2 - text_width(d, INTRO_LINE, ff) // 2, 950), INTRO_LINE, font=ff, fill=(*MUTED, int(255 * feat_a)))

    chips = ["8 tajweed rules", "Hafs & Qalun", "English & Arabic", "Offline-first"]
    chip_f = font(FONT_SEMI, 23)
    widths = [text_width(d, c, chip_f) + round(23 * 1.45) + round(23 * 0.6) for c in chips]
    total = sum(widths) + 22 * (len(chips) - 1)
    x = W // 2 - total // 2
    chip_a = smoothstep((local - 1.0) / 0.6)
    for c in chips:
        x = draw_chip(d, x, 1020, c, chip_f, chip_a)

    bot_a = smoothstep((local - 1.35) / 0.7)
    bf = font(FONT_SEMI, 22)
    bot = "Offline-first Quran recitation for Android"
    d.text((W // 2 - text_width(d, bot, bf) // 2, 1742), bot, font=bf, fill=(*DIM, int(255 * bot_a)))
    return content


def outro_content_vertical(local: float, duration: float) -> Image.Image:
    content = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(content)

    icon_a = smoothstep(local / 0.45)
    if icon_a > 0.01:
        size = round(230 * (0.92 + 0.08 * ease_out(clamp01(local / 0.6))))
        ic = set_alpha(icon_layer(size), icon_a)
        content.alpha_composite(ic, (W // 2 - ic.width // 2, 300 - ic.height // 2))

    word_a = smoothstep((local - 0.12) / 0.5)
    draw_wordmark(d, 0, 610, 86, word_a, center=W // 2)

    tag_a = smoothstep((local - 0.36) / 0.55)
    tf = font(FONT_BOLD, 48)
    tag = "Recite. Correct. Perfect."
    d.text((W // 2 - text_width(d, tag, tf) // 2, 740), tag, font=tf, fill=(*TEXT, int(255 * tag_a)))

    line_a = smoothstep((local - 0.52) / 0.5)
    if line_a > 0.01:
        half = round(230 * ease_out(clamp01((local - 0.52) / 0.7)))
        d.rounded_rectangle((W // 2 - half, 820, W // 2 + half, 822), radius=1, fill=(*TEAL, int(220 * line_a)))

    feat_a = smoothstep((local - 0.62) / 0.55)
    ff = font(FONT_REG, 27)
    d.text((W // 2 - text_width(d, INTRO_LINE, ff) // 2, 856), INTRO_LINE, font=ff, fill=(*MUTED, int(255 * feat_a)))

    bot_a = smoothstep((local - 0.92) / 0.6)
    bf = font(FONT_SEMI, 24)
    bot = "Itq\u0101n \u00b7 Android \u00b7 Offline-first"
    d.text((W // 2 - text_width(d, bot, bf) // 2, 1742), bot, font=bf, fill=(*DIM, int(255 * bot_a)))
    return content


def intro_content_landscape(local: float, duration: float) -> Image.Image:
    content = Image.new("RGBA", (LW, LH), (0, 0, 0, 0))
    d = ImageDraw.Draw(content)

    word_a = smoothstep(local / 0.4)
    draw_wordmark(d, 110, 120, 104, word_a)

    tag_a = smoothstep((local - 0.25) / 0.5)
    tf = font(FONT_BOLD, 58)
    d.text((110, 282), "Recite with itqan.", font=tf, fill=(*TEAL, int(255 * tag_a)))

    feat_a = smoothstep((local - 0.45) / 0.5)
    ff = font(FONT_REG, 28)
    for i, line in enumerate(("Live recitation tracking, on-device Tajweed scoring,", "a rules coach and goals \u2014 offline-first.")):
        d.text((110, 390 + i * 38), line, font=ff, fill=(*MUTED, int(255 * feat_a)))

    chips = ["8 tajweed rules", "Hafs & Qalun", "English & Arabic", "Offline-first"]
    chip_f = font(FONT_SEMI, 22)
    x = 110
    ca = smoothstep((local - 0.75) / 0.55)
    for c in chips:
        x = draw_chip(d, x, 500, c, chip_f, ca)

    bot_a = smoothstep((local - 1.0) / 0.6)
    bf = font(FONT_SEMI, 21)
    d.text((110, 900), "Android \u00b7 On-device speech recognition", font=bf, fill=(*DIM, int(255 * bot_a)))

    icon_a = smoothstep((local - 0.1) / 0.6)
    if icon_a > 0.01:
        size = round(430 * (0.92 + 0.08 * ease_out(clamp01(local / 0.8))))
        ic = set_alpha(icon_layer(size), icon_a)
        content.alpha_composite(ic, (1450 - ic.width // 2, 480 - ic.height // 2))
    return content


def outro_content_landscape(local: float, duration: float) -> Image.Image:
    content = Image.new("RGBA", (LW, LH), (0, 0, 0, 0))
    d = ImageDraw.Draw(content)

    word_a = smoothstep(local / 0.4)
    draw_wordmark(d, 110, 150, 92, word_a)

    tag_a = smoothstep((local - 0.25) / 0.5)
    tf = font(FONT_BOLD, 62)
    d.text((110, 290), "Recite. Correct. Perfect.", font=tf, fill=(*TEXT, int(255 * tag_a)))

    feat_a = smoothstep((local - 0.45) / 0.5)
    ff = font(FONT_REG, 28)
    for i, line in enumerate(("Live tracking, Tajweed scoring, a rules coach", "and recitation goals \u2014 offline-first.")):
        d.text((110, 402 + i * 38), line, font=ff, fill=(*MUTED, int(255 * feat_a)))

    bot_a = smoothstep((local - 0.7) / 0.55)
    bf = font(FONT_SEMI, 22)
    d.text((110, 520), "Itq\u0101n \u00b7 Android \u00b7 Offline-first", font=bf, fill=(*DIM, int(255 * bot_a)))

    icon_a = smoothstep((local - 0.1) / 0.6)
    if icon_a > 0.01:
        size = round(410 * (0.92 + 0.08 * ease_out(clamp01(local / 0.8))))
        ic = set_alpha(icon_layer(size), icon_a)
        content.alpha_composite(ic, (1450 - ic.width // 2, 500 - ic.height // 2))
    return content


# --------------------------------------------------------------------------- timeline

INTRO_D = 6.5
FEATURE_D = 5.0
OUTRO_D = 6.0
XFADE = 0.45
FPS = 30


def build_timeline(feature_d: float = FEATURE_D, intro_d: float = INTRO_D, outro_d: float = OUTRO_D) -> list[dict]:
    tl: list[dict] = [dict(kind="intro", start=0.0, duration=intro_d, scene=None)]
    t = intro_d
    for f in FEATURES:
        tl.append(dict(kind="feature", start=t, duration=feature_d, scene=f))
        t += feature_d
    tl.append(dict(kind="outro", start=t, duration=outro_d, scene=None))
    return tl


TIMELINE = build_timeline()
TOTAL_D = TIMELINE[-1]["start"] + TIMELINE[-1]["duration"]


def segment_at(t: float, timeline: list[dict]) -> tuple[int, dict]:
    for i, seg in enumerate(timeline):
        if t < seg["start"] + seg["duration"] or i == len(timeline) - 1:
            return i, seg
    return len(timeline) - 1, timeline[-1]


def content_vertical(seg: dict, local: float, index: int) -> Image.Image:
    if seg["kind"] == "intro":
        return intro_content_vertical(local, seg["duration"])
    if seg["kind"] == "outro":
        return outro_content_vertical(local, seg["duration"])
    return feature_content_vertical(seg["scene"], local, seg["duration"])


def content_landscape(seg: dict, local: float, index: int) -> Image.Image:
    if seg["kind"] == "intro":
        return intro_content_landscape(local, seg["duration"])
    if seg["kind"] == "outro":
        return outro_content_landscape(local, seg["duration"])
    return feature_content_landscape(seg["scene"], local, seg["duration"], index)


def background_vertical(seg: dict) -> Image.Image:
    if seg["kind"] in ("intro", "outro"):
        return background(TEAL)
    return background(seg["scene"]["accent"])


def background_landscape(seg: dict) -> Image.Image:
    if seg["kind"] in ("intro", "outro"):
        return background(TEAL, LW, LH)
    return background(seg["scene"]["accent"], LW, LH)


def render_frame(t: float, timeline: list[dict], vertical: bool = True, xfade: float = XFADE) -> Image.Image:
    idx, seg = segment_at(t, timeline)
    local = t - seg["start"]
    if vertical:
        bg = background_vertical(seg).copy()
        fg = content_vertical(seg, local, idx)
    else:
        bg = background_landscape(seg).copy()
        fg = content_landscape(seg, local, idx)

    if idx > 0 and local < xfade:
        prev = timeline[idx - 1]
        a = smoothstep(local / xfade)
        if vertical:
            prev_bg = background_vertical(prev)
            prev_fg = content_vertical(prev, prev["duration"], idx - 1)
        else:
            prev_bg = background_landscape(prev)
            prev_fg = content_landscape(prev, prev["duration"], idx - 1)
        bg = Image.blend(prev_bg.convert("RGB"), bg.convert("RGB"), a).convert("RGBA")
        out = bg.copy()
        out.alpha_composite(set_alpha(prev_fg, 1.0 - a))
        out.alpha_composite(set_alpha(fg, a))
        return out.convert("RGB")
    bg.alpha_composite(fg)
    return bg.convert("RGB")


# --------------------------------------------------------------------------- encode


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise RuntimeError("ffmpeg is required. Install imageio-ffmpeg or put ffmpeg on PATH.")


def encode_video(
    out_path: Path,
    vertical: bool,
    duration: float = TOTAL_D,
    timeline: list[dict] | None = None,
    fps: int = FPS,
    crf: int = 20,
) -> None:
    """Encode a silent H.264 MP4 from rendered frames piped to ffmpeg."""
    timeline = timeline or TIMELINE
    w, h = (W, H) if vertical else (LW, LH)
    nframes = int(round(duration * fps))
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.0",
        "-g",
        str(fps * 2),
        "-movflags",
        "+faststart",
        "-an",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(nframes):
            t = i / fps
            frame = render_frame(t, timeline, vertical=vertical)
            proc.stdin.write(frame.convert("RGB").tobytes())
            if i % 120 == 0:
                print(f"    frame {i:4d}/{nframes}")
        proc.stdin.close()
    except BrokenPipeError:
        pass
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}): {stderr[-3000:]}")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  ok {out_path.name}  {w}x{h}, {duration:.1f}s, {size_mb:.1f} MB")


def build_web_cut(master: Path, out_path: Path, bitrate: str = "950k") -> None:
    """Re-encode the 16:9 master for a README: same cut, silent, ~1 Mb/s.

    The result stays under GitHub's 10 MB attachment cap, which is the only way
    a README gets an autoplaying inline player (attach the file through the web
    UI and use the user-attachments URL it returns).
    """
    TMP.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    log = TMP / "web-2pass"
    common = [
        "-i", str(master),
        "-c:v", "libx264", "-preset", "slow", "-b:v", bitrate,
        "-pix_fmt", "yuv420p", "-g", str(FPS * 2),
        "-profile:v", "high", "-level", "4.0",
    ]
    # Two passes so the file size is predictable; both passes must share the
    # exact encoder settings or libx264 refuses the second one.
    for p in ("1", "2"):
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", *common, "-pass", p, "-passlogfile", str(log), "-an"]
        cmd += ["-f", "null", "-"] if p == "1" else ["-movflags", "+faststart", str(out_path)]
        subprocess.run(cmd, check=True)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  ok {out_path.name}  {LW}x{LH}, {size_mb:.1f} MB")


def build_posters() -> None:
    t = INTRO_D + FEATURE_D * 0.5
    render_frame(t, TIMELINE, vertical=True).save(MARKETING / "promo-poster-1080x1920.png", optimize=True)
    render_frame(t, TIMELINE, vertical=False).save(MARKETING / "promo-poster-1920x1080.png", optimize=True)
    print("  ok promo posters")


def build_video() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    build_posters()
    encode_video(MARKETING / "itqan-promo.mp4", vertical=True)
    encode_video(MARKETING / "itqan-promo-16x9.mp4", vertical=False)
    build_web_cut(MARKETING / "itqan-promo-16x9.mp4", MARKETING / "itqan-promo-16x9-web.mp4")


# --------------------------------------------------------------------------- crop


def crop_screenshots(raw: Path) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    missing = []
    for raw_name, clean_name in RAW_MAP.items():
        src = raw / raw_name
        if not src.exists():
            missing.append(raw_name)
            continue
        im = Image.open(src).convert("RGB")
        top, bottom = CROP_OVERRIDES.get(raw_name, (CROP_TOP, CROP_BOTTOM))
        if im.size != (1080, 2400) and raw_name not in CROP_OVERRIDES:
            print(f"  ! {raw_name}: unexpected size {im.size}")
        im.crop((0, top, im.width, min(bottom, im.height))).save(CLEAN / clean_name, optimize=True)
        bottom = min(bottom, im.height)
        print(f"  ok {clean_name}  {im.width}x{bottom - top}")
    if missing:
        print("  missing raw screenshots:")
        for name in missing:
            print(f"    {name}")
        print("  (the existing clean copies were left in place)")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["crop", "static", "video", "all"], nargs="?", default="all")
    parser.add_argument("--raw", type=Path, default=Path.home() / "Desktop" / "itqan_screenshots")
    args = parser.parse_args()

    if args.what in ("crop", "all"):
        print("Cropping phone screenshots (status + navigation bars removed)")
        crop_screenshots(args.raw)
    if args.what in ("static", "all"):
        print("Building store frames and banners")
        build_store()
        build_banner()
        build_hero()
    if args.what in ("video", "all"):
        print("Building promo video (silent)")
        build_video()


if __name__ == "__main__":
    main()
