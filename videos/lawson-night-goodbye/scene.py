#!/usr/bin/env python3
"""Render "Kal kab milna hai?" — a quiet night scene outside the Lawson.

The supplied painting (background.webp) is used as-is as the set; the only
change to it is the automatic door sliding open. Two hand-built characters
(a Kiki-inspired girl and a Howl-inspired boy) are drawn on top as vector
cel art with Pillow, the camera slowly pushes in, and Hindi dialogue lines
(voices/*.wav, see voices.py) drive the lip-sync and subtitles.

    python3 scene.py                       # silent video -> scene_silent.mp4
    python3 scene.py --preview 60 200 330  # PNG stills
    python3 audio.py                       # soundtrack + final mux
"""
import argparse
import math
import os
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920
FPS, DUR = 30, 17.0
N = int(FPS * DUR)
SS = 2
UP = 3  # background upscale factor
BASE = 1920 / 1302  # screen px per source px at zoom 1
UNIT = 0.78  # source px per character unit (boy is 100 units = 78 px tall)


# --------------------------------------------------------------------------- timing helpers
def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def kf(t, keys):
    """Piecewise eased interpolation over [(time, value-or-tuple), ...]."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            e = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1
            if isinstance(v0, tuple):
                return tuple(a + (b - a) * e for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * e
    return keys[-1][1]


def win(t, t0, t1, fade=0.15):
    return min(ease((t - t0) / fade + 1) if fade else 1, ease((t1 - t) / fade + 1) if fade else 1) if t0 - fade <= t <= t1 + fade else 0.0


# --------------------------------------------------------------------------- dialogue
LINES = [  # start, file, speaker, subtitle
    (4.7, "boy_kal_kab", "boy", "Kal kab milna hai?"),
    (6.75, "girl_same_time", "girl", "Same time, same place."),
    (11.62, "girl_ok_bye", "girl", "Ok, bye!"),
    (13.65, "boy_bye", "boy", "B... bye."),
]


def load_wav(path):
    import soundfile as sf  # noqa: lazy so the renderer needs it only here
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    return a, sr


def mouth_tracks():
    """Per-frame mouth openness for each speaker from the voice envelopes."""
    tracks = {"boy": np.zeros(N), "girl": np.zeros(N)}
    spans = []
    for start, name, who, text in LINES:
        a, sr = load_wav(os.path.join(HERE, "voices", f"{name}.wav"))
        dur = len(a) / sr
        spans.append((start, start + dur, who, text))
        hop = sr // FPS
        env = np.array([np.sqrt(np.mean(a[i:i + hop] ** 2)) if i < len(a) else 0 for i in range(0, len(a), hop)])
        env = np.clip(env / (env.max() + 1e-6) * 1.6, 0, 1)
        f0 = int(round(start * FPS))
        for i, v in enumerate(env):
            if 0 <= f0 + i < N:
                tracks[who][f0 + i] = max(tracks[who][f0 + i], v)
    for k in tracks:  # light smoothing so the mouth doesn't chatter
        tr = tracks[k]
        tracks[k] = np.maximum(tr, np.concatenate([[0], tr[:-1]]) * 0.6)
    return tracks, spans


# --------------------------------------------------------------------------- camera
CAM = [  # time, (zoom, cx, cy) in source pixels
    (0.0, (1.0, 368, 651)),
    (1.6, (1.1, 420, 780)),
    (3.6, (2.3, 436, 1010)),
    (5.2, (4.1, 434, 1036)),
    (8.6, (4.3, 434, 1038)),
    (11.3, (5.0, 430, 1046)),
    (11.65, (5.0, 430, 1046)),
    (12.7, (4.8, 446, 1046)),
    (13.4, (4.9, 430, 1046)),
    (14.8, (4.9, 428, 1046)),
    (17.0, (3.6, 428, 1030)),
]


def camera(t):
    z, cx, cy = kf(t, CAM)
    hw, hh = W / 2 / (BASE * z), H / 2 / (BASE * z)
    cx = min(max(cx, hw), 736 - hw)
    cy = min(max(cy, hh), 1302 - hh)
    return z, cx, cy


# --------------------------------------------------------------------------- background + door
SRC = Image.open(os.path.join(HERE, "background.webp")).convert("RGB")
BG3 = SRC.resize((SRC.width * UP, SRC.height * UP), Image.LANCZOS)
DOOR_Y0, DOOR_Y1 = 976, 1071
DOOR_X0, DOOR_X1 = 374, 534
PANE_L, PANE_R = (414, 455), (456, 494)
_door_cache = {}


def door_open(t):
    return kf(t, [(1.35, 0.0), (2.0, 1.0), (4.3, 1.0), (5.0, 0.0)])


def background(o):
    key = round(o, 3)
    if key in _door_cache:
        return _door_cache[key]
    if o <= 0.001:
        img = BG3
    else:
        img = BG3.copy()
        y0, y1 = DOOR_Y0 * UP, DOOR_Y1 * UP
        region = BG3.crop((DOOR_X0 * UP, y0, DOOR_X1 * UP, y1))
        interior = region.filter(ImageFilter.GaussianBlur(10 * UP / 3))
        interior = Image.eval(interior, lambda v: min(255, int(v * 1.12 + 8)))
        mid = interior.crop(((PANE_L[0] - DOOR_X0) * UP, 0, (PANE_R[1] - DOOR_X0) * UP, y1 - y0))
        img.paste(mid, (PANE_L[0] * UP, y0))
        lp = BG3.crop((PANE_L[0] * UP, y0, PANE_L[1] * UP, y1))
        rp = BG3.crop((PANE_R[0] * UP, y0, PANE_R[1] * UP, y1))
        img.paste(lp, (int(round((PANE_L[0] - o * 40) * UP)), y0))
        img.paste(rp, (int(round((PANE_R[0] + o * 38) * UP)), y0))
        # re-paste the fixed door frame posts so the sliding panes tuck behind them
        for x in (DOOR_X0 - 2, DOOR_X1 - 2):
            img.paste(BG3.crop((x * UP, y0, (x + 3) * UP, y1)), (x * UP, y0))
    if len(_door_cache) > 64:
        _door_cache.clear()
    _door_cache[key] = img
    return img


def door_gap(o):
    return PANE_L[1] - o * 40, PANE_R[0] + o * 38


# --------------------------------------------------------------------------- characters
OUT = (46, 34, 44)
SKIN = (255, 226, 202)
SKIN_SH = (236, 194, 172)
BLUSH = (246, 160, 162)

GIRL = dict(kind="girl", bs=0.93, rx=9.4, ry=10.4, hair=(52, 38, 42), hair_hi=(84, 62, 66),
            top=(50, 44, 82), top_sh=(36, 32, 62), legs=SKIN, shoe=(132, 66, 48))
BOY = dict(kind="boy", bs=1.0, rx=9.5, ry=10.8, hair=(176, 184, 198), hair_hi=(214, 220, 230),
           top=(42, 108, 156), top_sh=(30, 82, 124), legs=(48, 50, 66), shoe=(62, 48, 44))


class Pose:
    def __init__(self, **kw):
        self.yaw = 0.0          # -1 (3/4 left) .. 1 (3/4 right)
        self.phase = 0.0        # gait phase (radians)
        self.walk = 0.0         # 0 stand .. 1 walk
        self.run = 0.0          # 0 .. 1 run
        self.lean = 0.0         # degrees toward facing
        self.mouth = 0.0
        self.smile = 0.4
        self.blink = 0.0
        self.happy = 0.0        # ^ ^ eyes
        self.blush = 0.2
        self.tilt = 0.0         # head tilt, degrees (+ = toward facing)
        self.look_down = 0.0
        self.worried = 0.0
        self.wave = 0.0         # near-arm hesitant wave raise 0..1
        self.wave_osc = 0.0
        self.behind_back = 0.0  # girl clasps hands behind her back
        self.bag = False
        self.bounce = 0.0
        self.__dict__.update(kw)


def rot(p, c, deg):
    a = math.radians(deg)
    x, y = p[0] - c[0], p[1] - c[1]
    return (c[0] + x * math.cos(a) - y * math.sin(a), c[1] + x * math.sin(a) + y * math.cos(a))


def draw_char(spec, P, s):
    """Return (RGBA image, feet_x, feet_y) for one character at s px/unit."""
    cw, ch, fx, fy = 84, 132, 42, 122
    im = Image.new("RGBA", (int(cw * s * SS), int(ch * s * SS)), (0, 0, 0, 0))
    d_ = ImageDraw.Draw(im)
    lw = max(1, int(round(0.55 * s * SS)))

    def T(p):
        return ((fx + p[0]) * s * SS, (fy + p[1]) * s * SS)

    def poly(pts, fill, outline=True):
        d_.polygon([T(p) for p in pts], fill=fill, outline=OUT if outline else None, width=lw if outline else 0)

    def circle(c, r, fill, outline=True):
        x, y = T(c)
        rr = r * s * SS
        d_.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill, outline=OUT if outline else None, width=lw if outline else 0)

    def limb(pts, width, fill):
        tp = [T(p) for p in pts]
        wo = width * s * SS + 2 * lw
        wi = width * s * SS
        d_.line(tp, fill=OUT, width=int(wo), joint="curve")
        for p in (tp[0], tp[-1]):
            d_.ellipse([p[0] - wo / 2, p[1] - wo / 2, p[0] + wo / 2, p[1] + wo / 2], fill=OUT)
        d_.line(tp, fill=fill, width=int(wi), joint="curve")
        for p in tp:
            d_.ellipse([p[0] - wi / 2, p[1] - wi / 2, p[0] + wi / 2, p[1] + wi / 2], fill=fill)

    bs = spec["bs"]
    yaw = P.yaw
    d = 1 if yaw >= 0 else -1
    ay = abs(yaw)
    side = min(1.0, ay * 1.25)
    front = 1 - side

    # ---------------- legs (hip-relative), then drop the body onto the ground
    L1, L2 = 18.5 * bs, 18 * bs
    amp = math.radians(18 * P.walk + 42 * P.run)
    kb = math.radians(20 * P.walk + 70 * P.run)
    legs = []
    for off in (0.0, math.pi):
        sw = amp * math.sin(P.phase + off)
        bend = kb * (0.5 + 0.5 * math.sin(P.phase + off + 1.3)) if (P.walk or P.run) else 0
        kx, ky = L1 * math.sin(sw) * d * side, L1 * math.cos(sw)
        ax, ay2 = kx + L2 * math.sin(sw - bend) * d * side, ky + L2 * math.cos(sw - bend)
        lift = front * max(0.0, math.sin(P.phase + off)) * 3.5 * P.walk
        legs.append(((kx, ky - lift * 0.5), (ax, ay2 - lift)))
    reach = max(l[1][1] for l in legs)
    hip_y = -reach - 2.2 * bs
    hip_y -= abs(math.sin(P.phase)) * 1.2 * P.run
    hw_near = -d * 3.4 * bs * (1 - 0.35 * ay)
    hw_far = d * 3.4 * bs * (1 - 0.6 * ay)
    hip_c = (0.0, hip_y)

    def U(p):  # upper-body point (lean around hips)
        return rot(p, hip_c, P.lean * d)

    def leg_pts(hx, leg):
        (kx, ky), (ax, ay2) = leg
        return [(hx, hip_y), (hx + kx, hip_y + ky), (hx + ax, hip_y + ay2)]

    def draw_leg(hx, leg):
        pts = leg_pts(hx, leg)
        limb(pts, 5.4 * bs if spec["kind"] == "boy" else 3.8 * bs, spec["legs"])
        ax, ay2 = pts[-1]
        x, y = T((ax + d * 1.0 * (0.4 + side), ay2 + 0.4))
        rx, ry = (2.9 + 1.4 * side) * s * SS * bs, 1.8 * s * SS * bs
        d_.ellipse([x - rx, y - ry, x + rx, y + ry], fill=spec["shoe"], outline=OUT, width=lw)

    # ---------------- skeleton points
    sh_y = hip_y - 27 * bs
    sw_ = 8.4 * bs if spec["kind"] == "boy" else 7.2 * bs
    sh_near = U((-d * sw_ * (1 - 0.25 * ay), sh_y + 1))
    sh_far = U((d * sw_ * (1 - 0.5 * ay), sh_y + 1))
    neck = U((0.4 * yaw, sh_y - 1.8 * bs))
    head_c = U((0.6 * yaw, sh_y - 3.6 * bs - spec["ry"] * 0.95))
    head_c = (head_c[0], head_c[1] + P.look_down * 1.0)

    def arm(shoulder, a1, a2):
        sideA = 0.35 + 0.65 * side
        l1, l2 = 12.5 * bs, 11.5 * bs
        r1, r2 = math.radians(a1 + P.lean), math.radians(a2 + P.lean)
        e = (shoulder[0] + l1 * math.sin(r1) * d * sideA, shoulder[1] + l1 * math.cos(r1))
        h = (e[0] + l2 * math.sin(r2) * d * sideA, e[1] + l2 * math.cos(r2))
        return [shoulder, e, h]

    swing = 16 * P.walk + 55 * P.run
    near_a = (swing * math.sin(P.phase + math.pi), swing * math.sin(P.phase + math.pi) + 10 + 70 * P.run)
    far_a = (swing * math.sin(P.phase), swing * math.sin(P.phase) + 10 + 70 * P.run)
    if P.behind_back:
        bb = P.behind_back
        near_a = (near_a[0] * (1 - bb) - 12 * bb, near_a[1] * (1 - bb) + 28 * bb)
        far_a = (far_a[0] * (1 - bb) - 12 * bb, far_a[1] * (1 - bb) + 28 * bb)
    if P.wave:
        wv = P.wave
        near_a = (near_a[0] * (1 - wv) + 38 * wv, near_a[1] * (1 - wv) + (150 + P.wave_osc) * wv)
    arm_w = 4.3 * bs

    def draw_arm(pts, sleeve):
        limb(pts[:2], arm_w, sleeve)
        limb(pts[1:], arm_w * 0.92, sleeve)
        circle(pts[2], 2.1 * bs, SKIN)

    far_arm = arm(sh_far, *far_a)
    near_arm = arm(sh_near, *near_a)

    # ---------------- draw: far arm (+bag), far leg, near leg, body, near arm, head
    hands_back = P.behind_back >= 0.5
    if hands_back:  # both arms go behind the torso, which is drawn over them
        draw_arm(far_arm, spec["top_sh"])
        draw_arm(near_arm, spec["top"])
    else:
        draw_arm(far_arm, spec["top_sh"])
    if P.bag:
        hx, hy = far_arm[2]
        bag = [(hx - 2.2, hy + 0.5), (hx + 2.2, hy + 0.5), (hx + 4.6, hy + 13), (hx - 4.4, hy + 13.5)]
        poly(bag, (244, 244, 240))
        poly([(hx - 3.6, hy + 9.5), (hx + 3.9, hy + 9.3), (hx + 4.2, hy + 10.8), (hx - 3.8, hy + 11.1)], (40, 128, 204), outline=False)
        circle((hx, hy), 1.65, SKIN)
    draw_leg(hw_far, legs[1])
    draw_leg(hw_near, legs[0])

    if spec["kind"] == "boy":
        # trousers top + untucked shirt with a mandarin collar and buttons
        poly([U((-7.0 * bs, hip_y - 2)), U((7.0 * bs, hip_y - 2)), U((6.6 * bs, hip_y + 6)), U((-6.6 * bs, hip_y + 6))], spec["legs"])
        body = [sh_far, U((d * 8.4 * (1 - 0.3 * ay), hip_y + 4.0)), U((-d * 8.6 * (1 - 0.2 * ay), hip_y + 4.0)), sh_near]
        body = [neck] + body
        poly([U((sh_near[0] * 0 - d * 3.5, sh_y - 1)), sh_far, body[2], body[3], sh_near], spec["top"])
        poly([U((d * 1.6 * ay - 3.2, sh_y - 0.6)), U((d * 1.6 * ay + 3.2, sh_y - 0.6)),
              U((d * 1.6 * ay + 2.4, sh_y + 2.4)), U((d * 1.6 * ay - 2.4, sh_y + 2.4))], spec["top_sh"])
        bx = d * 1.6 * ay
        for k in range(4):
            circle(U((bx, sh_y + 6 + k * 5.5)), 0.55, (230, 230, 240), outline=False)
    else:
        sway = math.sin(P.phase) * (1.0 * P.walk + 2.0 * P.run)
        hem = hip_y + 9 * bs
        dress = [U((-d * 2.0, sh_y - 1)), sh_far, U((d * 6.4 * (1 - 0.3 * ay), hip_y - 8)),
                 U((d * 13.0 * (1 - 0.25 * ay) + sway, hem)), U((sway * 0.5, hem + 1.4)),
                 U((-d * 13.0 * (1 - 0.1 * ay) + sway, hem)), U((-d * 6.4 * (1 - 0.2 * ay), hip_y - 8)), sh_near]
        poly(dress, spec["top"])
        poly([U((d * 0.8 * ay - 2.6, sh_y - 0.8)), U((d * 0.8 * ay + 2.6, sh_y - 0.8)), U((d * 0.8 * ay, sh_y + 2.0))], SKIN, outline=True)

    if not hands_back:
        draw_arm(near_arm, spec["top"])
    limb([neck, U((0.4 * yaw, sh_y - 0.5))], 3.8 * bs, SKIN_SH)

    draw_head(d_, spec, P, head_c, s, lw, T)

    out = im.resize((int(cw * s), int(ch * s)), Image.LANCZOS)
    return out, fx * s, fy * s


def draw_head(d_, spec, P, c, s, lw, T):
    rx, ry = spec["rx"], spec["ry"]
    yaw = P.yaw
    d = 1 if yaw >= 0 else -1
    ay = abs(yaw)
    tilt = P.tilt * d

    def H(x, y):
        return T(rot((c[0] + x, c[1] + y), (c[0], c[1] + ry), tilt))

    def hpoly(pts, fill, outline=True):
        d_.polygon([H(*p) for p in pts], fill=fill, outline=OUT if outline else None, width=lw if outline else 0)

    def hell(x, y, ex, ey, fill, outline=False, width=None):
        cx, cy = H(x, y)
        d_.ellipse([cx - ex * s * SS, cy - ey * s * SS, cx + ex * s * SS, cy + ey * s * SS], fill=fill,
                   outline=OUT if outline else None, width=width or lw)

    def hline(pts, fill, width):
        d_.line([H(*p) for p in pts], fill=fill, width=max(1, int(width * s * SS)), joint="curve")

    girl = spec["kind"] == "girl"
    back = -yaw * 1.4
    bounce = P.bounce

    # back hair
    pts = []
    for i in range(25):
        a = math.pi + math.pi * i / 24
        pts.append((back + math.cos(a) * rx * 1.14, math.sin(a) * ry * 1.1 - 0.5))
    if girl:
        pts = [(back - rx * 1.16, 6.8 + bounce)] + pts + [(back + rx * 1.16, 6.8 + bounce)]
        pts += [(back + rx * 0.6, 7.6 + bounce), (back - rx * 0.6, 7.6 + bounce)]
    else:
        tail = []
        for i in range(9):
            x = back + rx * 1.12 - i * (2 * rx * 1.12) / 8
            tail.append((x, 6.5 + (1.6 if i % 2 else 0) + 0.8 * math.sin(i * 1.7)))
        pts = [(back - rx * 1.12, 2)] + pts + [(back + rx * 1.12, 2)] + tail
    hpoly(pts, spec["hair"])

    # bow (behind the bangs, on top)
    if girl:
        bx = -yaw * 2.2
        sp = 4.6 * (1 - 0.35 * ay)
        for sgn in (-1, 1):
            hell(bx + sgn * sp, -ry * 1.12 + bounce * 0.3, 4.4 * (1 - 0.3 * ay), 3.2, (214, 42, 50), outline=True)
        hell(bx, -ry * 1.04, 1.7, 1.6, (176, 30, 40), outline=True)

    # face
    face = []
    for i in range(40):
        a = 2 * math.pi * i / 40
        x, y = math.cos(a), math.sin(a)
        if y > 0:
            x *= 1 - 0.3 * y ** 2
            y *= 1.06
        face.append((x * rx * (1 - 0.06 * ay) + yaw * 1.6 * max(0.0, y), y * ry))
    hpoly(face, SKIN)
    # ear on the far-from-viewer side shows for the boy only (girl's bob covers)
    if not girl and ay > 0.3:
        hell(-d * rx * 0.86 + back * 0.3, 1.0, 1.4, 2.0, SKIN, outline=True)

    xc = yaw * 2.9
    sp = 3.3 * (1 - 0.28 * ay)
    ey = 1.2
    # blush
    b = P.blush
    bc = tuple(int(SKIN[i] + (BLUSH[i] - SKIN[i]) * b) for i in range(3))
    for sgn in (-1, 1):
        far = (sgn != d) and ay > 0.2
        hell(xc + sgn * (sp + 0.6), ey + 3.0, 1.6 * (0.6 if far else 1), 0.9, bc)

    # eyes
    for sgn in (-1, 1):
        far = (sgn != d) and ay > 0.2
        ex = xc + sgn * sp
        wx = 1.25 * (1 - 0.4 * ay) if far else 1.25
        open_ = max(0.0, 1 - P.blink) * (1 - 0.45 * P.look_down)
        if P.happy > 0.5:
            hline([(ex - wx * 1.2, ey + 0.5), (ex, ey - 0.9), (ex + wx * 1.2, ey + 0.5)], OUT, 0.55)
        elif open_ < 0.2:
            hline([(ex - wx * 1.2, ey + 0.2), (ex, ey + 0.7), (ex + wx * 1.2, ey + 0.2)], OUT, 0.5)
        else:
            hy = 1.95 * open_
            hell(ex, ey + (1 - open_) * 0.6, wx, hy, (34, 26, 32))
            hell(ex + 0.35 * wx, ey - hy * 0.4, 0.38 * wx, 0.42, (255, 255, 255))
            hline([(ex - wx * 1.25, ey - hy * 0.8 + 0.2), (ex, ey - hy * 1.02), (ex + wx * 1.2, ey - hy * 0.85)], OUT, 0.34)
        # eyebrows (lift + inner raise when worried)
        by = ey - 3.6 - 0.2 * P.worried
        inner = 0.9 * P.worried
        x0, x1 = ex - wx * 1.1, ex + wx * 1.1
        inner_x = x1 if sgn < 0 else x0
        pts = [(x0, by - (inner if inner_x == x0 else 0)), (x1, by - (inner if inner_x == x1 else 0))]
        hline(pts, spec["hair"] if not girl else OUT, 0.42)

    # nose + mouth
    nx = xc + yaw * 1.8
    if ay > 0.15:
        hline([(nx + d * 0.2, ey + 2.2), (nx + d * 0.7, ey + 3.4), (nx, ey + 3.5)], OUT, 0.32)
    else:
        hell(nx, ey + 3.2, 0.25, 0.2, OUT)
    mx, my = xc + yaw * 0.9, ey + 5.6
    if P.mouth > 0.08:
        mh = 0.35 + 1.1 * P.mouth
        hell(mx, my + mh * 0.3, 1.05, mh, (156, 52, 64), outline=True, width=max(1, lw // 2))
        hell(mx, my + mh * 0.75, 0.6, mh * 0.35, (222, 120, 120))
    else:
        cv = 0.2 + 0.6 * P.smile
        hline([(mx - 1.3, my - cv * 0.4), (mx, my + cv * 0.35), (mx + 1.3, my - cv * 0.4)], OUT, 0.38)

    # front hair: bangs + side locks
    sx = 1 - 0.12 * ay
    sh = yaw * 1.1
    pts = []
    for i in range(21):
        a = math.pi + math.pi * i / 20
        pts.append((sh + math.cos(a) * rx * 1.1 * sx, math.sin(a) * ry * 1.08 - 0.4))
    if girl:
        fringe = [(rx * 0.86, -1.0), (rx * 0.55, -2.6), (rx * 0.32, -1.6), (rx * 0.08, -2.9), (-rx * 0.2, -1.8),
                  (-rx * 0.45, -2.8), (-rx * 0.7, -1.6), (-rx * 0.88, -1.2)]
        right = [(rx * 1.16, 5.8 + bounce), (rx * 0.82, 5.2 + bounce), (rx * 0.9, 1.0)]
        left = [(-rx * 0.9, 1.0), (-rx * 0.82, 5.2 + bounce), (-rx * 1.16, 5.8 + bounce)]
    else:
        fringe = [(rx * 0.9, 0.5), (rx * 0.6, -1.2), (rx * 0.45, 0.6), (rx * 0.2, -1.6), (0.0, 0.2),
                  (-rx * 0.25, -1.8), (-rx * 0.5, 0.4), (-rx * 0.7, -1.4), (-rx * 0.9, 0.6)]
        right = [(rx * 1.18, 3.8), (rx * 0.95, 4.8), (rx * 0.92, 1.0)]
        left = [(-rx * 0.92, 1.2), (-rx * 1.0, 5.0), (-rx * 1.18, 4.0)]
    edge = [(sh + x * sx, y) for x, y in right + fringe + left]
    hpoly(pts + edge, spec["hair"], outline=False)
    hline([pts[-1]] + edge + [pts[0]], OUT, 0.5)
    # soft sheen band
    hline([(sh - rx * 0.55, -ry * 0.62), (sh - rx * 0.1, -ry * 0.74), (sh + rx * 0.4, -ry * 0.66)], spec["hair_hi"], 0.9)


# --------------------------------------------------------------------------- scripted performance
def path_and_phase(keys, dt=1 / FPS, k=0.42):
    """Precompute per-frame (x, y, scale) and a gait phase driven by distance walked."""
    pos = [kf(i * dt, keys) for i in range(N + 1)]
    phase = [0.0]
    for i in range(1, N + 1):
        dx = (pos[i][0] - pos[i - 1][0]) / UNIT
        dy = (pos[i][1] - pos[i - 1][1]) / UNIT * 1.6
        phase.append(phase[-1] + math.hypot(dx, dy) * k)
    speed = [math.hypot(pos[min(i + 1, N)][0] - pos[i][0], pos[min(i + 1, N)][1] - pos[i][1]) * FPS for i in range(N + 1)]
    return pos, phase, speed


GIRL_PATH = [(1.9, (455.0, 1066.0, 0.93)), (3.5, (452.0, 1095.0, 1.0)), (10.6, (452.0, 1095.0, 1.0)),
             (11.3, (437.0, 1095.0, 1.0)), (11.6, (437.0, 1095.0, 1.0)), (11.62, (438.0, 1095.0, 1.0)),
             (13.2, (640.0, 1097.0, 1.0))]
BOY_PATH = [(2.35, (455.0, 1066.0, 0.93)), (3.0, (450.0, 1079.0, 0.97)), (4.1, (414.0, 1094.0, 1.0)), (17.0, (414.0, 1094.0, 1.0))]
BLINKS = {"boy": [3.3, 6.4, 9.0, 10.15, 12.95, 15.5], "girl": [3.9, 8.9, 10.4, 15.0]}


def girl_pose(t, ph, speed, mouth):
    p = Pose(phase=ph, blush=0.55, smile=0.5)
    running = t > 11.6
    p.walk = min(1.0, speed / 14) if not running else 0
    p.run = min(1.0, speed / 60) if running else 0
    p.yaw = kf(t, [(3.4, 0.0), (4.0, -0.8), (11.55, -0.8), (11.7, 0.9)])
    p.lean = 12 * p.run
    p.bounce = 0.8 * abs(math.sin(ph)) * p.run
    p.mouth = mouth
    p.smile = kf(t, [(6.5, 0.45), (7.0, 0.85), (11.0, 0.85), (11.4, 1.0)])
    p.look_down = kf(t, [(8.9, 0.0), (9.3, 0.7), (10.1, 0.7), (10.5, 0.0)])
    p.tilt = kf(t, [(9.0, 0.0), (9.4, 6.0), (10.2, 6.0), (10.6, -5.0), (11.55, -5.0), (11.7, 0.0)])
    p.behind_back = kf(t, [(8.7, 0.0), (9.2, 1.0), (11.5, 1.0), (11.6, 0.0)])
    p.happy = 1.0 if 11.62 < t < 12.5 else 0.0
    p.blush = kf(t, [(10.6, 0.55), (11.3, 0.85)])
    p.blink = blink(t, BLINKS["girl"])
    return p


def boy_pose(t, ph, speed, mouth):
    p = Pose(phase=ph, bag=True, blush=0.15)
    p.walk = min(1.0, speed / 14)
    p.yaw = kf(t, [(2.9, 0.0), (3.3, -0.6), (3.95, -0.6), (4.35, 0.8)])
    p.mouth = mouth
    p.smile = kf(t, [(4.5, 0.35), (7.5, 0.35), (8.0, 0.7), (11.6, 0.6), (12.0, 0.1), (16.0, 0.25)])
    p.blush = kf(t, [(10.5, 0.15), (11.3, 0.85), (13.0, 0.6), (17.0, 0.45)])
    p.lean = kf(t, [(10.6, 0.0), (11.3, -4.0), (11.6, -4.0), (12.0, 0.0)])
    p.tilt = kf(t, [(10.6, 0.0), (11.3, -4.0), (11.7, 0.0), (12.4, 3.0)])
    p.worried = kf(t, [(11.6, 0.0), (12.2, 1.0), (16.0, 0.8)])
    p.wave = kf(t, [(13.0, 0.0), (13.6, 0.75), (15.0, 0.75), (15.9, 0.0)])
    p.wave_osc = 9 * math.sin((t - 13.6) * 7) * kf(t, [(13.6, 0.0), (13.8, 1.0), (14.8, 1.0), (15.0, 0.0)])
    p.blink = blink(t, BLINKS["boy"])
    return p


def blink(t, times):
    v = 0.0
    for b in times:
        if b <= t <= b + 0.16:
            v = max(v, math.sin(math.pi * (t - b) / 0.16))
    return v


# --------------------------------------------------------------------------- compositing
def light_character(rgba):
    """Night grade + warm rim light from the shop window behind them."""
    a = rgba[..., 3:4]
    rgb = rgba[..., :3] * np.array([0.92, 0.88, 0.95], np.float32)
    al = rgba[..., 3]
    r = max(2, int(rgba.shape[0] / 260))
    up = np.zeros_like(al)
    up[r:] = al[:-r]
    lft = np.zeros_like(al)
    lft[:, r:] = al[:, :-r]
    rgt = np.zeros_like(al)
    rgt[:, :-r] = al[:, r:]
    rim = np.clip(al - np.minimum(np.minimum(up, lft), rgt), 0, 1)[..., None]
    rgb = rgb + rim * np.array([0.55, 0.42, 0.26], np.float32)
    return np.clip(rgb, 0, 1), a


def composite(frame, img, x, y, mask_fn=None):
    arr = np.asarray(img, np.float32) / 255.0
    rgb, a = light_character(arr)
    h, w = a.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    if mask_fn is not None:
        a = a * mask_fn(x0, y0, w, h)[..., None]
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(W, x0 + w), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sl = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
    reg = frame[fy0:fy1, fx0:fx1]
    frame[fy0:fy1, fx0:fx1] = reg * (1 - a[sl]) + rgb[sl] * a[sl]


def ground_shadow(frame, sx, sy, s):
    rx, ry = 11 * s, 3.2 * s
    x0, x1 = int(max(0, sx - rx * 1.8)), int(min(W, sx + rx * 1.8))
    y0, y1 = int(max(0, sy - ry * 2)), int(min(H, sy + ry * 4))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    m = np.exp(-(((xx - sx) / rx) ** 2 + ((yy - sy - ry * 0.6) / (ry * 1.4)) ** 2) * 1.6)
    frame[y0:y1, x0:x1] *= (1 - 0.5 * m)[..., None]


FONT = os.path.join("/usr/share/fonts", "opentype", "inter", "Inter-Medium.otf")


def find_font():
    for root, _, files in os.walk("/usr/share/fonts"):
        for f in files:
            if f in ("Inter-Medium.otf", "Inter-SemiBold.otf"):
                return os.path.join(root, f)
    return None


def draw_subtitle(img, text, alpha):
    if alpha <= 0:
        return img
    fp = find_font()
    font = ImageFont.truetype(fp, 54) if fp else ImageFont.load_default()
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    bbox = dl.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x, y = (W - tw) / 2, 1660
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((x, y + 3), text, font=font, fill=(5, 8, 20, int(200 * alpha)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    dl.text((x, y), text, font=font, fill=(255, 250, 238, int(255 * alpha)))
    base = img.convert("RGBA")
    base.alpha_composite(shadow)
    base.alpha_composite(layer)
    return base.convert("RGB")


def render_frame(fi, tracks, spans, paths):
    t = fi / FPS
    z, cx, cy = camera(t)
    o = door_open(t)
    bg = background(o)
    hw, hh = W / 2 / (BASE * z), H / 2 / (BASE * z)
    box = ((cx - hw) * UP, (cy - hh) * UP, (cx + hw) * UP, (cy + hh) * UP)
    frame_img = bg.resize((W, H), Image.LANCZOS, box=box)
    blur_r = max(0.0, (z - 1.5) * 1.6)
    if blur_r > 0.3:
        frame_img = frame_img.filter(ImageFilter.GaussianBlur(blur_r))
    frame = np.asarray(frame_img, np.float32) / 255.0
    frame = frame.copy()

    def to_screen(x, y):
        return (x - cx) * BASE * z + W / 2, (y - cy) * BASE * z + H / 2

    actors = []
    for who, spec, posefn in (("girl", GIRL, girl_pose), ("boy", BOY, boy_pose)):
        pos, phase, speed = paths[who]
        start = (GIRL_PATH if who == "girl" else BOY_PATH)[0][0]
        if t < start:
            continue
        x, y, sc = pos[fi]
        p = posefn(t, phase[fi], speed[fi], float(tracks[who][fi]))
        actors.append((y, who, spec, p, x, sc))
    actors.sort(key=lambda a: a[0])

    for y, who, spec, p, x, sc in actors:
        s = UNIT * sc * BASE * z
        sx, sy = to_screen(x, y)
        inside = y < 1073
        if not inside:
            ground_shadow(frame, sx, sy, s)
        img, ox, oy = draw_char(spec, p, s)

        mask_fn = None
        if inside:
            g0, g1 = door_gap(o)

            def mask_fn(x0, y0, w, h, g0=g0, g1=g1):
                xs = (np.arange(w) + x0 - W / 2) / (BASE * z) + cx
                ys = (np.arange(h) + y0 - H / 2) / (BASE * z) + cy
                inx = ((xs >= g0) & (xs <= g1)).astype(np.float32)[None, :]
                below = (ys >= DOOR_Y1 - 0.5).astype(np.float32)[:, None]
                return np.clip(inx + below, 0, 1)
        composite(frame, img, sx - ox, sy - oy, mask_fn)

    fade = min(ease(t / 0.9), ease((DUR - t) / 1.1))
    frame *= fade
    out = Image.fromarray((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8))
    for t0, t1, who, text in spans:
        a = win(t, t0 - 0.1, t1 + 0.35, 0.18)
        if a > 0:
            out = draw_subtitle(out, text, a * fade)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=float, nargs="*", help="times in seconds")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "scene_silent.mp4"))
    args = ap.parse_args()

    tracks, spans = mouth_tracks()
    paths = {"girl": path_and_phase(GIRL_PATH), "boy": path_and_phase(BOY_PATH)}

    if args.preview is not None:
        for ts in args.preview:
            fi = min(N - 1, int(round(ts * FPS)))
            img = render_frame(fi, tracks, spans, paths)
            path = os.path.join(args.outdir, f"scene_{fi:03d}.png")
            img.save(path)
            print("wrote", path)
        return

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    for fi in range(N):
        enc.stdin.write(render_frame(fi, tracks, spans, paths).tobytes())
        if fi % 60 == 0:
            print(f"  frame {fi}/{N}", flush=True)
    enc.stdin.close()
    enc.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
