#!/usr/bin/env python3
"""Render a 10 second, Ghibli-style sunset scene: a snow-capped mountain, the
sun sinking behind its flank, and a bullet train crossing left to right.

Everything is drawn procedurally (numpy + Pillow) and piped into ffmpeg.

    python3 render.py                # full video -> ghibli-sunset-bullet-train.mp4
    python3 render.py --preview 0 150 299   # write PNG stills of those frames
"""
import argparse
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1080, 1920
FPS, DUR = 30, 10
N = FPS * DUR
SS = 2  # supersampling for Pillow-drawn shapes

HERE = os.path.dirname(os.path.abspath(__file__))
Xg = np.arange(W, dtype=np.float32)[None, :]
Yg = np.arange(H, dtype=np.float32)[:, None]
XS = np.arange(W, dtype=np.float32)


# --------------------------------------------------------------------------- helpers
def col(c):
    return np.array(c, np.float32) / 255.0


def pal(a, b, p):
    """Palette colour: a at golden hour (p=0), b at dusk (p=1)."""
    return col(a) * (1 - p) + col(b) * p


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def noise1d(x, seed, base=0.004, octaves=5, persist=0.5):
    r = np.random.default_rng(seed)
    out = np.zeros_like(x, dtype=np.float32)
    f, a, tot = base, 1.0, 0.0
    for _ in range(octaves):
        for _ in range(2):
            ff = f * r.uniform(0.7, 1.3)
            out += a * np.sin(x * ff * 2 * np.pi + r.uniform(0, 2 * np.pi))
            tot += a
        f *= 2.1
        a *= persist
    return out / tot * 2.0  # roughly -1..1


def noise2d(h, w, cells, seed):
    r = np.random.default_rng(seed)
    g = (r.random((max(2, h // cells), max(2, w // cells))) * 255).astype(np.uint8)
    im = Image.fromarray(g).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(cells / 4))
    return np.asarray(im, np.float32) / 255.0


def blur(arr, radius):
    """Gaussian blur of a float (h,w) or (h,w,c) array in 0..1 via Pillow."""
    if arr.ndim == 2:
        im = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
        return np.asarray(im.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255.0
    return np.stack([blur(arr[..., i], radius) for i in range(arr.shape[2])], -1)


def mask_canvas(w=W, h=H):
    im = Image.new("L", (w * SS, h * SS), 0)
    return im, ImageDraw.Draw(im)


def mask_to_np(im, w=W, h=H, radius=0.0):
    im = im.resize((w, h), Image.LANCZOS)
    if radius:
        im = im.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im, np.float32) / 255.0


def S(v):
    return v * SS


def ellipse(d, cx, cy, rx, ry, fill=255):
    d.ellipse([S(cx - rx), S(cy - ry), S(cx + rx), S(cy + ry)], fill=fill)


class Layer:
    """Premultiplied RGBA accumulation buffer."""

    def __init__(self, h=H, w=W):
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.a = np.zeros((h, w), np.float32)

    def over(self, rgb, a):
        a = np.clip(a, 0, 1)
        rgb = np.broadcast_to(rgb, self.rgb.shape)
        self.rgb = rgb * a[..., None] + self.rgb * (1 - a[..., None])
        self.a = a + self.a * (1 - a)


def below(ridge, soft=0.5):
    return np.clip((Yg - ridge[None, :]) / (2 * soft) + 0.5, 0, 1)


# --------------------------------------------------------------------------- scene constants
PEAK_X, PEAK_Y, BASE_Y, HALF_W = 470.0, 800.0, 1250.0, 720.0
RAIL_Y = 1474  # bottom of the train body
PARAPET_Y = 1462
SUN_R = 52


def sun_pos(t):
    e = t
    return 770 + 40 * e, 880 + 250 * e


def fuji_ridge():
    d = np.maximum(0, np.abs(XS - PEAK_X) - 38) / HALF_W
    ridge = PEAK_Y + (BASE_Y - PEAK_Y) * np.power(np.clip(d, 0, 1.4), 0.62)
    ridge += 2.5 * noise1d(XS, 3, base=0.05, octaves=3)  # rough crater rim / jaggies
    ridge[np.abs(XS - PEAK_X) < 38] += 4 * np.abs(np.sin((XS[np.abs(XS - PEAK_X) < 38] - PEAK_X) * 0.15))
    return ridge


FUJI_RIDGE = fuji_ridge()


# --------------------------------------------------------------------------- sky
SKY_YS = [0, 520, 900, 1100, 1240, 1920]
SKY0 = [(56, 74, 150), (106, 114, 190), (192, 146, 188), (250, 186, 158), (255, 216, 160), (255, 220, 170)]
SKY1 = [(16, 22, 62), (42, 48, 116), (118, 80, 138), (214, 108, 108), (244, 150, 100), (244, 156, 104)]


def build_sky(p):
    stops = [pal(a, b, p) for a, b in zip(SKY0, SKY1)]
    ys = np.arange(H, dtype=np.float32)
    colv = np.stack([np.interp(ys, SKY_YS, [s[i] for s in stops]) for i in range(3)], -1)
    sky = np.repeat(colv[:, None, :], W, axis=1)
    # subtle painterly unevenness
    n = noise2d(H, W, 120, 5)[..., None]
    sky *= 0.97 + 0.06 * n
    return sky.astype(np.float32)


def build_stars():
    r = np.random.default_rng(21)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for _ in range(140):
        x, y = r.uniform(0, W), r.uniform(0, 640) ** 1.0
        b = int(r.uniform(90, 255))
        if r.random() < 0.15:
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=b)
        else:
            d.point((x, y), fill=b)
    a = np.asarray(im.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0 * 2.2
    fade = smooth(300, 700, Yg)  # fewer near the glow
    return np.clip(a * (1 - fade), 0, 1)


def build_moon():
    im, d = mask_canvas()
    cx, cy, r = 470, 300, 22
    ellipse(d, cx, cy, r, r)
    d2 = ImageDraw.Draw(im)
    ellipse(d2, cx + 10, cy - 7, r, r, fill=0)
    m = mask_to_np(im)
    glow = blur(m, 10) * 0.35
    return m, glow


# --------------------------------------------------------------------------- clouds
CLOUD_PAD = 240  # clouds drift right, so the canvas extends past the left edge
CW = W + CLOUD_PAD


def build_cloud_masks():
    r = np.random.default_rng(42)
    sh, dsh = mask_canvas(CW, 1300)
    li, dli = mask_canvas(CW, 1300)
    cumulus = [  # cx, base_y, width, height
        (CLOUD_PAD + 210, 590, 560, 270),
        (CLOUD_PAD + 830, 400, 470, 210),
        (CLOUD_PAD + 600, 790, 300, 110),
        (CLOUD_PAD - 120, 300, 380, 150),
        (CLOUD_PAD + 1040, 700, 260, 90),
    ]
    for cx, by, cw, ch in cumulus:
        n = int(cw / 22)
        puffs = []
        for _ in range(n):
            u = r.uniform(-1, 1)
            env = ch * (1 - u * u) ** 0.8
            rr = r.uniform(0.28, 0.5) * max(env, ch * 0.25)
            rr = max(rr, 14)
            y = by - rr * r.uniform(0.85, 1.0) - r.uniform(0, max(1, env - rr))
            puffs.append((cx + u * cw / 2, y, rr))
        puffs.sort(key=lambda q: q[1])
        for x, y, rr in puffs:
            ellipse(dsh, x, y, rr, rr)
            ellipse(dli, x + 0.22 * rr, y + 0.3 * rr, rr * 0.78, rr * 0.78)
    shadow = mask_to_np(sh, CW, 1300, 1.2)
    light = mask_to_np(li, CW, 1300, 4) * shadow

    st, dst = mask_canvas(CW, 1300)
    streaks = [  # cx, cy, half-width, half-height
        (CLOUD_PAD + 820, 1000, 260, 9), (CLOUD_PAD + 980, 1060, 220, 7),
        (CLOUD_PAD + 640, 1110, 300, 8), (CLOUD_PAD + 900, 1150, 330, 6),
        (CLOUD_PAD + 300, 1050, 240, 7), (CLOUD_PAD + 120, 980, 200, 10),
        (CLOUD_PAD + 700, 930, 180, 6), (CLOUD_PAD + 1060, 900, 200, 8),
        (CLOUD_PAD + 420, 160, 280, 7), (CLOUD_PAD + 760, 220, 240, 5),
        (CLOUD_PAD + 140, 110, 260, 6),
    ]
    for cx, cy, hw, hh in streaks:
        for _ in range(4):
            ox, oy = r.uniform(-hw * 0.4, hw * 0.4), r.uniform(-hh, hh)
            ellipse(dst, cx + ox, cy + oy, hw * r.uniform(0.5, 0.9), hh * r.uniform(0.6, 1.1), fill=170)
    streak = mask_to_np(st, CW, 1300, 4)
    return shadow, light, streak


CLOUD_MASKS = build_cloud_masks()


def build_clouds(p):
    shadow, light, streak = CLOUD_MASKS
    ys = np.arange(1300, dtype=np.float32)[:, None, None]
    vt = np.clip(ys / 1100.0, 0, 1)
    sh_col = pal((140, 122, 182), (78, 68, 126), p) * (1 - vt) + pal((186, 140, 178), (120, 74, 118), p) * vt
    li_col = pal((255, 206, 168), (250, 132, 108), p)
    rgb = sh_col + (li_col - sh_col) * light[..., None]
    # streak clouds: lit warm from below the horizon
    st_col = pal((255, 176, 150), (238, 104, 98), p) * (1 - vt) + pal((255, 214, 170), (255, 150, 96), p) * vt
    a1 = shadow * 0.96
    a2 = streak * 0.75
    out_a = a2 + a1 * (1 - a2)
    out_rgb = st_col * a2[..., None] + rgb * a1[..., None] * (1 - a2[..., None])
    return out_rgb.astype(np.float32), out_a.astype(np.float32)  # premultiplied


# --------------------------------------------------------------------------- landscape (behind train)
def build_back(p):
    L = Layer()

    # ---- the mountain
    a = below(FUJI_RIDGE)
    dx = Xg - PEAK_X
    dy = np.maximum(Yg - PEAK_Y, 1)
    phi = np.arctan2(dx, dy + 30)
    streak = 0.5 + 0.5 * np.sin(phi * 40 + 2.6 * np.sin(phi * 11) + 1.7 * np.sin(phi * 23 + 1))
    light = np.clip(0.42 + 0.9 * np.tanh(dx / (dy + 60) * 1.8) + 0.22 * (streak - 0.5), 0, 1)
    light = blur(light, 2.5)
    fingers = 125 * np.power(0.5 + 0.5 * np.sin(phi * 34 + 2 * np.sin(phi * 7)), 3)
    snowline = PEAK_Y + 135 + 45 * noise1d(XS, 8, base=0.01)[None, :] + fingers
    snow = np.clip((snowline - Yg) / 2.5, 0, 1) * (0.55 + 0.45 * smooth(0.0, 0.45, streak + 0.25))
    snow_lit, snow_sh = pal((255, 214, 196), (250, 162, 150), p), pal((150, 150, 206), (104, 96, 160), p)
    rock_lit, rock_sh = pal((132, 112, 168), (100, 72, 126), p), pal((70, 76, 138), (46, 44, 98), p)
    haze = pal((214, 160, 178), (160, 98, 132), p)
    lt = light[..., None]
    snow_c = snow_sh + (snow_lit - snow_sh) * lt
    rock_c = rock_sh + (rock_lit - rock_sh) * lt
    v = np.clip((Yg - PEAK_Y) / (BASE_Y - PEAK_Y), 0, 1)[..., None]
    rock_c = rock_c + (haze - rock_c) * np.power(v, 1.7) * 0.85
    m = rock_c + (snow_c - rock_c) * snow[..., None]
    rim = np.exp(-np.maximum(Yg - FUJI_RIDGE[None, :], 0) / 3.0) * smooth(0, 300, dx) * 0.6
    m = m + (pal((255, 220, 190), (255, 150, 120), p) - m) * rim[..., None]
    L.over(m, a)

    # ---- haze band across the mountain base
    hn = noise1d(XS, 12, base=0.003)[None, :]
    mist = 0.65 * np.exp(-((Yg - 1215 - 8 * hn) / 34) ** 2)
    L.over(haze, mist)

    # ---- distant ridges, near to far painted back to front
    ridges = [
        # base_y, amp, seed, colour0, colour1, tree texture
        (1232, 22, 31, (160, 124, 168), (112, 76, 128), 0.0),
        (1268, 34, 32, (118, 104, 150), (76, 62, 112), 1.5),
        (1318, 30, 33, (78, 98, 108), (50, 50, 84), 3.0),
    ]
    for by, amp, seed, c0, c1, tex in ridges:
        rr = by + amp * noise1d(XS, seed, base=0.0025)
        if tex:
            rr = rr - tex * np.abs(np.sin(XS * 0.21 + noise1d(XS, seed + 50, 0.02) * 3))
        c = pal(c0, c1, p)
        g = np.clip((Yg - rr[None, :]) / 120, 0, 1)[..., None]
        cc = c * (1 - 0.18 * g)
        top = np.exp(-np.maximum(Yg - rr[None, :], 0) / 5) * 0.35
        cc = cc + (pal((240, 186, 160), (200, 118, 110), p) - cc) * top[..., None]
        L.over(cc, below(rr))
        L.over(haze, below(rr) * np.clip(1 - (Yg - rr[None, :]) / 60, 0, 1) * 0.25)

    # ---- fields / paddies down to the bottom of the frame
    fr = 1352 + 6 * noise1d(XS, 40, base=0.004)
    patches = noise2d(H, W, 36, 41)
    green = pal((112, 150, 72), (56, 72, 66), p)
    gold = pal((176, 168, 82), (98, 82, 74), p)
    f = green + (gold - green) * (0.25 + 0.4 * smooth(0.3, 0.7, patches))[..., None]
    depth = np.clip((Yg - 1352) / 400, 0, 1)[..., None]
    f = f * (0.86 + 0.24 * depth)
    u = 2600.0 / np.maximum(Yg - 1230, 1)
    rows = 0.5 + 0.5 * np.sin(u * 2 * np.pi)
    f = f * (1 - 0.07 * smooth(0.6, 1.0, rows)[..., None])
    sheen = np.exp(-((Xg - 800) / 200) ** 2) * np.exp(-np.maximum(Yg - 1352, 0) / 90) * 0.3
    f = f + (pal((255, 210, 150), (230, 130, 100), p) - f) * sheen[..., None]
    f = f + (haze - f) * (1 - depth) * 0.35
    L.over(f, below(fr))

    # ---- tree clumps and a little village behind the viaduct
    L_trees(L, p, seed=60, clusters=[(330, 1388, 160), (560, 1380, 90), (840, 1392, 210), (1040, 1384, 120)],
            scale=0.8, sh=((44, 84, 62), (28, 40, 56)), li=((132, 160, 78), (78, 92, 74)))
    village(L, p)
    L_trees(L, p, seed=61, clusters=[(250, 1446, 110), (690, 1450, 140), (960, 1452, 100)],
            scale=1.0, sh=((40, 78, 56), (24, 36, 50)), li=((126, 156, 70), (72, 86, 70)))
    return L


def L_trees(L, p, seed, clusters, scale, sh, li):
    r = np.random.default_rng(seed)
    ms, dms = mask_canvas()
    ml, dml = mask_canvas()
    for cx, by, cw in clusters:
        for _ in range(int(cw / 8)):
            u = r.uniform(-1, 1)
            rr = r.uniform(10, 22) * scale
            x = cx + u * cw / 2
            y = by - rr - r.uniform(0, 26 * scale) * (1 - u * u)
            ellipse(dms, x, y, rr, rr)
            ellipse(dml, x + rr * 0.3, y - rr * 0.25, rr * 0.7, rr * 0.7)
    a = mask_to_np(ms, radius=0.4)
    lit = mask_to_np(ml, radius=1.2) * a
    c = pal(*sh, p) + (pal(*li, p) - pal(*sh, p)) * lit[..., None]
    L.over(c, a)


def village(L, p):
    houses = [(600, 1404, 34, 18), (640, 1408, 26, 15), (720, 1402, 40, 20), (768, 1410, 24, 14), (470, 1410, 28, 15)]
    wall, roof, win = mask_canvas(), mask_canvas(), mask_canvas()
    for x, y, w, h in houses:
        wall[1].rectangle([S(x), S(y - h), S(x + w), S(y)], fill=255)
        roof[1].polygon([(S(x - 4), S(y - h)), (S(x + w / 2), S(y - h - 11)), (S(x + w + 4), S(y - h))], fill=255)
        for wx in range(int(x + 5), int(x + w - 6), 10):
            win[1].rectangle([S(wx), S(y - h + 6), S(wx + 4), S(y - h + 11)], fill=255)
    a_wall, a_roof, a_win = mask_to_np(wall[0]), mask_to_np(roof[0]), mask_to_np(win[0])
    L.over(pal((232, 214, 196), (140, 112, 130), p), a_wall)
    L.over(pal((70, 86, 128), (44, 46, 82), p), a_roof)
    L.over(pal((96, 92, 110), (255, 214, 140), p), a_win)
    glow = blur(a_win, 5) * 0.8 * p
    L.over(col((255, 200, 120)), glow * (1 - a_wall * 0.0))


# --------------------------------------------------------------------------- train
TRAIN_L, TRAIN_H = 1560, 80
NOSE = 270


def train_profile():
    xs = np.linspace(0, TRAIN_L, 600)
    top = np.full_like(xs, 6.0)
    s = np.clip((xs - (TRAIN_L - NOSE)) / NOSE, 0, 1)
    top = np.maximum(top, 6 + 54 * np.power(s, 1.7))
    s2 = np.clip((NOSE - xs) / NOSE, 0, 1)
    top = np.maximum(top, 6 + 54 * np.power(s2, 1.7))
    return xs, top


def build_train(p):
    w, h = TRAIN_L + 4, TRAIN_H
    body, db = mask_canvas(w, h)
    xs, top = train_profile()
    pts = [(S(x), S(y)) for x, y in zip(xs, top)]
    pts += [(S(TRAIN_L - 18), S(78)), (S(18), S(78))]
    db.polygon(pts, fill=255)
    a = mask_to_np(body, w, h)

    stripe, ds = mask_canvas(w, h)
    ds.rectangle([0, S(47), S(w), S(55)], fill=255)
    ds.rectangle([0, S(58), S(w), S(60.5)], fill=255)
    a_stripe = mask_to_np(stripe, w, h) * a

    skirt, dk = mask_canvas(w, h)
    dk.rectangle([0, S(66), S(w), S(80)], fill=255)
    a_skirt = mask_to_np(skirt, w, h) * a

    win, dw = mask_canvas(w, h)
    car = TRAIN_L / 5
    for c in range(5):
        x0, x1 = c * car + 34, (c + 1) * car - 34
        if c == 0:
            x0 = NOSE + 10
        if c == 4:
            x1 = TRAIN_L - NOSE - 10
        x = x0
        while x + 11 < x1:
            dw.rounded_rectangle([S(x), S(22), S(x + 11), S(34)], radius=S(2), fill=255)
            x += 18
    # cockpit windows on both noses
    for sgn in (1, -1):
        base = TRAIN_L - NOSE if sgn == 1 else NOSE
        poly = []
        for k in np.linspace(0.08, 0.42, 20):
            x = base + sgn * NOSE * k
            poly.append((S(x), S(6 + 54 * k ** 1.7 + 3)))
        for k in np.linspace(0.42, 0.08, 20):
            x = base + sgn * NOSE * k
            poly.append((S(x), S(6 + 54 * k ** 1.7 + 14 - 6 * k)))
        dw.polygon(poly, fill=255)
    a_win = mask_to_np(win, w, h) * a

    joint, dj = mask_canvas(w, h)
    for c in range(1, 5):
        dj.rectangle([S(c * car - 1), S(4), S(c * car + 1), S(80)], fill=255)
        for dxd in (-26, 22):
            dj.rectangle([S(c * car + dxd), S(18), S(c * car + dxd + 4), S(64)], outline=255, width=1)
    a_joint = mask_to_np(joint, w, h) * a

    ys = np.arange(h, dtype=np.float32)[:, None, None]
    white = pal((246, 240, 238), (196, 170, 186), p)
    shade = pal((168, 170, 200), (110, 100, 146), p)
    vt = np.clip((ys - 6) / 70, 0, 1)
    c = white + (shade - white) * np.power(vt, 1.4)
    warm = np.exp(-np.maximum(ys - 6, 0) / 4) * 0.8
    c = c + (pal((255, 214, 170), (255, 150, 120), p) - c) * warm
    out = Layer(h, w)
    out.over(c, a)
    out.over(pal((22, 70, 170), (24, 46, 120), p), a_stripe)
    out.over(pal((120, 122, 140), (70, 64, 92), p), a_skirt)
    out.over(pal((70, 70, 92), (60, 52, 80), p), a_joint * 0.6)
    out.over(pal((48, 58, 86), (255, 222, 156), p), a_win)
    # pantographs
    pg, dp = mask_canvas(w, h)
    for x in (car * 1.5, car * 3.5):
        dp.line([S(x - 10), S(6), S(x), S(0.5), S(x + 12), S(1)], fill=255, width=S(1))
    out.over(pal((70, 70, 90), (40, 36, 60), p), mask_to_np(pg, w, h))
    # motion blur along x (constant speed, so baked once)
    k = 3
    rgb = np.cumsum(np.pad(out.rgb, ((0, 0), (k, k), (0, 0)), mode="edge"), axis=1)
    al = np.cumsum(np.pad(out.a, ((0, 0), (k, k)), mode="edge"), axis=1)
    rgb = (rgb[:, 2 * k:] - rgb[:, :-2 * k]) / (2 * k)
    al = (al[:, 2 * k:] - al[:, :-2 * k]) / (2 * k)
    return rgb.astype(np.float32), al.astype(np.float32)


# --------------------------------------------------------------------------- foreground (in front of train)
def build_front(p):
    L = Layer()
    concrete_lit = pal((222, 200, 190), (142, 112, 134), p)
    concrete_sh = pal((150, 140, 162), (82, 74, 108), p)

    # pillars under the deck
    pm, dp = mask_canvas()
    for x in range(70, W + 200, 210):
        dp.polygon([(S(x - 22), S(1500)), (S(x + 22), S(1500)), (S(x + 14), S(1516)),
                    (S(x + 14), S(1700)), (S(x - 14), S(1700)), (S(x - 14), S(1516))], fill=255)
    a_p = mask_to_np(pm)
    side = (np.mod(Xg - 70 + 105, 210) - 105 > 2).astype(np.float32)
    pc = concrete_sh + (concrete_lit - concrete_sh) * (0.35 + 0.4 * side)[..., None]
    pc = pc * (1 - 0.25 * smooth(1500, 1640, Yg))[..., None]
    L.over(pc, a_p)
    # deck slab and the sound-barrier parapet
    slab = ((Yg >= PARAPET_Y) & (Yg < 1502)).astype(np.float32) * np.ones_like(Xg)
    vc = concrete_lit + (concrete_sh - concrete_lit) * smooth(PARAPET_Y, 1502, Yg)[..., None]
    vc = vc * (1 - 0.08 * (np.mod(Xg, 64) < 1.2))[..., None]
    vc = vc * (1 - 0.18 * ((Yg >= 1482) & (Yg < 1484)))[..., None]
    L.over(vc, slab)
    edge = np.exp(-np.abs(Yg - PARAPET_Y) / 1.0) * np.ones_like(Xg)
    L.over(pal((255, 234, 210), (240, 160, 140), p), edge * 0.7)
    L.over(concrete_sh * 0.6, ((Yg >= 1500) & (Yg < 1505)) * np.ones_like(Xg) * 0.6)

    # meadow
    mr = 1662 + 16 * noise1d(XS, 70, base=0.004)
    g = np.clip((Yg - 1660) / 260, 0, 1)[..., None]
    mc = pal((118, 150, 66), (60, 78, 58), p) * (1 - g) + pal((54, 92, 50), (26, 40, 42), p) * g
    mn = noise2d(H, W, 26, 71)[..., None]
    mc = mc * (0.9 + 0.2 * mn)
    rim = np.exp(-np.maximum(Yg - mr[None, :], 0) / 6)[..., None] * 0.5
    mc = mc + (pal((214, 206, 120), (128, 112, 88), p) - mc) * rim
    L.over(mc, below(mr))

    # big tree on the left edge
    r = np.random.default_rng(80)
    ms, dms = mask_canvas()
    ml, dml = mask_canvas()
    trunk, dt = mask_canvas()
    dt.polygon([(S(36), S(1920)), (S(96), S(1920)), (S(84), S(1560)), (S(118), S(1470)), (S(104), S(1462)),
                (S(72), S(1530)), (S(40), S(1420)), (S(28), S(1430)), (S(52), S(1560))], fill=255)
    for _ in range(70):
        ang = r.uniform(0, 2 * math.pi)
        rad = r.uniform(0, 1) ** 0.6
        x = 40 + math.cos(ang) * rad * 210
        y = 1360 + math.sin(ang) * rad * 170
        rr = r.uniform(28, 62)
        ellipse(dms, x, y, rr, rr)
        ellipse(dml, x + rr * 0.32, y - rr * 0.2, rr * 0.66, rr * 0.66)
    a_t = mask_to_np(trunk)
    L.over(pal((60, 46, 52), (30, 24, 36), p), a_t)
    a_c = mask_to_np(ms, radius=0.5)
    lit = mask_to_np(ml, radius=2) * a_c
    t_sh, t_li = pal((30, 66, 54), (18, 30, 44), p), pal((116, 156, 70), (86, 98, 70), p)
    tc = t_sh + (t_li - t_sh) * lit[..., None]
    L.over(tc, a_c)

    # power pole + wires on the right, as in the reference
    pole, dpo = mask_canvas()
    px = 972
    dpo.polygon([(S(px - 10), S(1920)), (S(px + 10), S(1920)), (S(px + 6), S(1060)), (S(px - 6), S(1060))], fill=255)
    for y, hw in ((1100, 70), (1150, 56)):
        dpo.rectangle([S(px - hw), S(y), S(px + hw), S(y + 7)], fill=255)
        for xi in (px - hw + 8, px - hw / 3, px + hw / 3, px + hw - 8):
            dpo.rectangle([S(xi - 3), S(y - 12), S(xi + 3), S(y)], fill=255)
    dpo.rectangle([S(px + 8), S(1240), S(px + 34), S(1290)], fill=255)  # transformer
    ends = [(px - 62, 1088), (px - 23, 1088), (px + 23, 1088), (px + 62, 1088), (px - 48, 1138), (px + 48, 1138)]
    for i, (x0, y0) in enumerate(ends):
        for x1, y1, sag in ((-40, 1010 + 22 * i, 70), (W + 40, 1120 + 14 * i, 22)):
            seg = []
            for s in np.linspace(0, 1, 60):
                seg.append((S(x0 + (x1 - x0) * s), S(y0 + (y1 - y0) * s + 4 * sag * s * (1 - s))))
            dpo.line(seg, fill=255, width=S(1) + 1)
    a_pole = mask_to_np(pole)
    L.over(pal((52, 48, 78), (24, 20, 42), p), a_pole)
    return L


# --------------------------------------------------------------------------- per-frame elements
GRASS_TOP = 1600
_gr = np.random.default_rng(90)
BLADES = []
for _ in range(420):
    x = _gr.uniform(-20, W + 20)
    by = _gr.uniform(1690, 1940)
    depth = (by - 1690) / 250
    BLADES.append((x, by, _gr.uniform(30, 70) + 140 * depth * _gr.uniform(0.4, 1.0),
                   _gr.uniform(2, 4.5) * (0.6 + depth), _gr.uniform(-0.25, 0.25), _gr.uniform(0, 6.28), depth))
BLADES.sort(key=lambda b: b[1])


def draw_grass(time, p):
    gh = H - GRASS_TOP
    im = Image.new("RGBA", (W * SS, gh * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for x, by, hgt, wid, lean, ph, depth in BLADES:
        sway = 10 * math.sin(time * 1.9 + x * 0.012 + ph * 0.3) + 5 * math.sin(time * 3.1 + ph)
        sway *= 0.5 + depth
        tx, ty = x + lean * hgt + sway, by - hgt
        mx, my = x + (lean * hgt + sway) * 0.4, by - hgt * 0.5
        base = pal((44, 86, 46), (22, 36, 40), p) * (1 - 0.3 * depth)
        tip = pal((180, 190, 96), (104, 112, 82), p)
        c = base + (tip - base) * (0.25 + 0.35 * (1 - depth))
        cc = tuple(int(v * 255) for v in c) + (255,)
        poly = [(x - wid, by), (mx - wid * 0.6, my), (tx, ty), (mx + wid * 0.6, my), (x + wid, by)]
        d.polygon([(S(px), S(py - GRASS_TOP)) for px, py in poly], fill=cc)
    im = im.resize((W, gh), Image.LANCZOS)
    arr = np.asarray(im, np.float32) / 255.0
    return arr[..., :3], arr[..., 3]


BIRDS = [(0, 0), (34, 14), (60, -8), (90, 20), (-26, 22)]


def draw_birds(time, p):
    top, bh = 560, 360
    im = Image.new("L", (W * SS, bh * SS), 0)
    d = ImageDraw.Draw(im)
    cx = 760 - 46 * time
    cy = 700 + 6 * math.sin(time * 0.8)
    for i, (ox, oy) in enumerate(BIRDS):
        x = cx + ox + 4 * math.sin(time * 1.3 + i)
        y = cy + oy + 3 * math.sin(time * 1.7 + i * 2) - top
        flap = math.sin(time * 9 + i * 1.7)
        span = 9
        d.line([(S(x - span), S(y - 4 * flap)), (S(x - 3), S(y - 1)), (S(x), S(y + 1)),
                (S(x + 3), S(y - 1)), (S(x + span), S(y - 4 * flap))], fill=255, width=S(1) + 1)
    a = np.asarray(im.resize((W, bh), Image.LANCZOS), np.float32) / 255.0
    return top, a


# --------------------------------------------------------------------------- compositing
def lerp(a, b, w):
    return a + (b - a) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, nargs="*")
    ap.add_argument("--outdir", default=HERE, help="where --preview stills go")
    ap.add_argument("--out", default=os.path.join(HERE, "ghibli-sunset-bullet-train.mp4"))
    args = ap.parse_args()

    print("building static layers...", flush=True)
    sky = [build_sky(0), build_sky(1)]
    clouds = [build_clouds(0), build_clouds(1)]
    back = [build_back(0), build_back(1)]
    front = [build_front(0), build_front(1)]
    train = [build_train(0), build_train(1)]
    stars = build_stars()
    moon, moon_glow = build_moon()
    grain = (noise2d(H, W, 3, 99) - 0.5) * 0.035 + (noise2d(H, W, 40, 98) - 0.5) * 0.03
    vig = 1 - 0.22 * (((Xg - W / 2) / (W * 0.75)) ** 2 + ((Yg - H * 0.52) / (H * 0.62)) ** 2)
    print("rendering...", flush=True)

    frames = args.preview if args.preview is not None else range(N)
    enc = None
    if args.preview is None:
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
            stdin=subprocess.PIPE)

    for fi in frames:
        t = fi / (N - 1)
        time = fi / FPS
        w = smooth(-0.15, 1.05, t)

        # sky, glow, stars, moon
        frame = lerp(sky[0], sky[1], w)
        sx, sy = sun_pos(t)
        r = np.sqrt((Xg - sx) ** 2 + (Yg - sy) ** 2)
        gcol = pal((255, 206, 150), (255, 130, 84), w)
        glow = (0.55 * np.exp(-r / 430) + 0.65 * np.exp(-r / 120)) * (1 - 0.35 * t)
        frame = 1 - (1 - frame) * (1 - np.clip(glow, 0, 1)[..., None] * gcol)
        sfade = smooth(0.45, 1.0, t) * (0.75 + 0.25 * math.sin(time * 5))
        frame = frame + stars[..., None] * sfade * 0.8 * col((240, 236, 255))
        ma = 0.35 + 0.6 * smooth(0.0, 1.0, t)
        frame = lerp(frame, col((255, 246, 222)), (moon * ma)[..., None])
        frame = frame + moon_glow[..., None] * ma * col((200, 200, 255))

        # sun disc
        sun_a = np.clip(SUN_R - r + 0.8, 0, 1)
        sun_c = pal((255, 248, 222), (255, 196, 140), w)
        frame = lerp(frame, sun_c, sun_a[..., None])

        # clouds drifting right
        off = CLOUD_PAD - 1 - 6.0 * time
        o0 = int(math.floor(off))
        fr_ = off - o0
        crgb = lerp(clouds[0][0], clouds[1][0], w)
        ca = lerp(clouds[0][1], clouds[1][1], w)
        crgb = crgb[:, o0:o0 + W] * (1 - fr_) + crgb[:, o0 + 1:o0 + 1 + W] * fr_
        ca = ca[:, o0:o0 + W] * (1 - fr_) + ca[:, o0 + 1:o0 + 1 + W] * fr_
        frame[:1300] = crgb + frame[:1300] * (1 - ca[..., None])

        # landscape behind the train
        brgb = lerp(back[0].rgb, back[1].rgb, w)
        frame = brgb + frame * (1 - back[0].a[..., None])

        # train
        tx = int(round(-TRAIN_L - 60 + (W + TRAIN_L + 140) * np.clip((time - 0.2) / 9.6, 0, 1)))
        trgb = lerp(train[0][0], train[1][0], w)
        ta = train[0][1]
        ty = RAIL_Y - 78
        x0, x1 = max(tx, 0), min(tx + trgb.shape[1], W)
        if x1 > x0:
            sl = slice(x0 - tx, x1 - tx)
            reg = frame[ty:ty + TRAIN_H, x0:x1]
            frame[ty:ty + TRAIN_H, x0:x1] = trgb[:, sl] + reg * (1 - ta[:, sl][..., None])

        # foreground
        frgb = lerp(front[0].rgb, front[1].rgb, w)
        frame = frgb + frame * (1 - front[0].a[..., None])
        g_rgb, g_a = draw_grass(time, w)
        frame[GRASS_TOP:] = lerp(frame[GRASS_TOP:], g_rgb, g_a[..., None])
        btop, ba = draw_birds(time, w)
        frame[btop:btop + ba.shape[0]] = lerp(frame[btop:btop + ba.shape[0]], pal((70, 56, 96), (34, 26, 56), w), ba[..., None])

        # bloom where the sun peeks over the mountain flank
        vis = float(np.mean(1 - back[0].a[sun_a > 0.5])) if np.any(sun_a > 0.5) else 0.0
        bloom = vis * (0.35 * np.exp(-r / 70) + 0.18 * np.exp(-r / 220))
        frame = 1 - (1 - np.clip(frame, 0, 1)) * (1 - np.clip(bloom, 0, 1)[..., None] * gcol)
        # after it is gone, a warm rim of afterglow on the ridge
        frame = frame + (1 - vis) * smooth(0.6, 0.9, t) * 0.25 * np.exp(-r / 160)[..., None] * gcol

        frame = np.clip((frame + grain[..., None]) * vig[..., None], 0, 1)
        out = (frame * 255 + 0.5).astype(np.uint8)
        if enc:
            enc.stdin.write(out.tobytes())
            if fi % 30 == 0:
                print(f"  frame {fi}/{N}", flush=True)
        else:
            path = os.path.join(args.outdir, f"preview_{fi:03d}.png")
            Image.fromarray(out).save(path)
            print("wrote", path)

    if enc:
        enc.stdin.close()
        enc.wait()
        print("wrote", args.out)


if __name__ == "__main__":
    main()
