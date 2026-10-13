#!/usr/bin/env python3
"""Synthesize the soundtrack for the sunset video and mux it into the mp4.

Layers, all generated with numpy and timed to render.py:
  * a gentle piano + strings cue (Fmaj7 - G6 - Em7 - Am9) with a small reverb
  * evening ambience: wind, higurashi cicadas, crickets as dusk falls, birds
  * the bullet train: rumble + whoosh panned left -> right with a slight
    Doppler drop, plus soft rail clicks

    python3 sound.py            # writes soundtrack.wav and muxes it into the mp4
"""
import argparse
import os
import subprocess
import wave

import numpy as np

import render as R

SR = 48000
DUR = R.DUR
NS = SR * DUR
T = np.arange(NS) / SR
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)


# --------------------------------------------------------------------------- dsp helpers
def fft_filter(x, lo=None, hi=None, tilt=0.0):
    """Static band-limit (soft edges) and spectral tilt (dB/octave re 1 kHz)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR) + 1e-3
    g = np.ones_like(f)
    if lo:
        g *= 1 / np.sqrt(1 + (lo / f) ** 4)
    if hi:
        g *= 1 / np.sqrt(1 + (f / hi) ** 4)
    if tilt:
        g *= (f / 1000) ** (tilt / 6.02)
    return np.fft.irfft(X * g, len(x))


def svf_bandpass(x, fc, q):
    """State-variable band-pass with a per-sample centre frequency."""
    out = np.empty_like(x)
    lp = bp = 0.0
    k = 1.0 / q
    fs = 2 * np.sin(np.pi * np.clip(fc, 20, SR / 6) / SR)
    for i in range(len(x)):
        f = fs[i]
        hp = x[i] - lp - k * bp
        bp += f * hp
        lp += f * bp
        out[i] = bp
    return out


def pan(mono, p):
    """Equal-power pan, p in -1..1 (scalar or per-sample)."""
    a = (np.asarray(p) + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], -1)


def place(buf, sig, t0):
    i0 = int(t0 * SR)
    if i0 >= len(buf):
        return
    n = min(len(sig), len(buf) - i0)
    buf[i0:i0 + n] += sig[:n]


def reverb(st, seconds=2.2, wet=0.3, seed=9):
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = np.zeros_like(st)
    for ch in range(2):
        ir = r.standard_normal(n) * np.exp(-tt * 6.9 / seconds)
        ir = fft_filter(ir, lo=200, hi=6000)
        ir[: int(0.012 * SR)] = 0  # pre-delay
        ir /= np.sqrt(np.sum(ir ** 2))
        m = len(st[:, ch]) + n
        nfft = 1 << (m - 1).bit_length()
        y = np.fft.irfft(np.fft.rfft(st[:, ch], nfft) * np.fft.rfft(ir, nfft), nfft)[: len(st)]
        out[:, ch] = y
    return st * (1 - wet) + out * wet * 2.2


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


# --------------------------------------------------------------------------- music
BPM = 72
BEAT = 60 / BPM


def piano(f, vel, length=6.0):
    n = int(length * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    B = 0.00035
    for k in range(1, 10):
        fk = k * f * np.sqrt(1 + B * k * k)
        if fk > SR / 2.5:
            break
        decay = (0.55 + 0.55 * k) * (f / 262) ** 0.35
        amp = (1 / k ** 1.25) * (1 - 0.5 * (k == 2))
        env = np.exp(-t * decay) * 0.8 + np.exp(-t * decay * 0.25) * 0.2
        # two slightly detuned strings per partial for a warm beating tone
        sig += amp * env * (np.sin(2 * np.pi * fk * t) + 0.8 * np.sin(2 * np.pi * fk * 1.0007 * t + 1.1))
    sig *= np.minimum(1, t / 0.004)
    thump = rng.standard_normal(int(0.02 * SR)) * np.exp(-np.arange(int(0.02 * SR)) / (0.003 * SR))
    sig[: len(thump)] += fft_filter(thump, hi=1500) * 0.4
    return sig * vel / 9


def strings(f, dur, vel):
    n = int((dur + 1.5) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.003 * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6))
    ph = 2 * np.pi * f * np.cumsum(vib) / SR
    sig = sum(np.sin(k * ph + k) / k ** 1.6 for k in range(1, 6))
    env = np.clip(t / 0.9, 0, 1) ** 2 * np.clip((dur + 1.5 - t) / 1.5, 0, 1)
    return sig * env * vel


CHORDS = [  # beat, bass, voicing (midi)
    (0, 41, [57, 60, 64]),      # Fmaj7
    (3, 43, [59, 62, 64]),      # G6
    (6, 40, [55, 59, 62]),      # Em7
    (9, 45, [60, 64, 71]),      # Am(add9)
]
MELODY = [  # beat, midi, velocity
    (0.0, 76, 0.9), (1.5, 74, 0.6), (2.0, 72, 0.75),
    (3.0, 74, 0.85), (4.5, 76, 0.6), (5.0, 79, 0.8),
    (6.0, 76, 0.85), (7.0, 74, 0.6), (7.5, 71, 0.6), (8.0, 67, 0.7),
    (9.0, 69, 0.85), (10.0, 72, 0.6), (10.5, 71, 0.55), (11.0, 69, 0.8),
]


def music():
    L = np.zeros((NS, 2))
    off = 0.15
    for beat, bass, voicing in CHORDS:
        t0 = off + beat * BEAT
        place(L, pan(piano(midi(bass), 0.8), -0.25), t0)
        for i, nte in enumerate(voicing):  # gently broken chord
            place(L, pan(piano(midi(nte), 0.45), -0.1 + 0.1 * i), t0 + 0.11 * (i + 1))
        dur = 3 * BEAT if beat < 9 else DUR - t0
        for i, nte in enumerate([bass + 12] + voicing):
            place(L, pan(strings(midi(nte), dur, 0.05), -0.4 + 0.25 * i), t0)
    for beat, nte, vel in MELODY:
        place(L, pan(piano(midi(nte), vel), 0.15), off + beat * BEAT)
    return reverb(L, 2.6, 0.35)


# --------------------------------------------------------------------------- ambience
def ambience():
    A = np.zeros((NS, 2))
    # wind: soft brown-ish noise with slow gusts (tied to the grass sway)
    gust = 0.55 + 0.25 * np.sin(T * 1.9 + 0.4) + 0.2 * np.sin(T * 0.7 + 2)
    for ch, s in enumerate((11, 12)):
        n = np.random.default_rng(s).standard_normal(NS)
        A[:, ch] += fft_filter(n, lo=60, hi=700, tilt=-3) * gust * 0.05

    dusk = R.smooth(0.0, 1.0, T / DUR)

    # higurashi ("kana-kana") phrases
    def higurashi(length, f0):
        n = int(length * SR)
        t = np.arange(n) / SR
        f = f0 * (1 - 0.12 * t / length)
        ph = 2 * np.pi * np.cumsum(f) / SR
        syl = (0.5 + 0.5 * np.cos(2 * np.pi * 7.5 * t)) ** 6
        am = 0.5 + 0.5 * np.sin(2 * np.pi * 95 * t)
        shape = np.sin(np.pi * np.clip(t / length, 0, 1)) ** 0.6 * np.exp(-t / length * 1.2)
        return (np.sin(ph) + 0.25 * np.sin(2 * ph)) * syl * am * shape

    place(A, pan(higurashi(3.0, 4300), -0.7) * 0.03, 0.8)
    place(A, pan(higurashi(3.4, 4050), 0.6) * 0.022, 5.2)
    place(A, pan(higurashi(2.2, 4500), 0.1) * 0.012, 7.4)

    # crickets fading in with the dusk
    for side, rate, f, seed in ((-0.5, 2.3, 4700, 1), (0.55, 1.9, 5100, 2)):
        r = np.random.default_rng(seed)
        t0 = 3.0 + r.uniform(0, 0.4)
        while t0 < DUR:
            n = int(0.09 * SR)
            t = np.arange(n) / SR
            pulses = (np.sin(np.pi * np.clip((t % 0.03) / 0.022, 0, 1)) ** 2) * (t < 0.088)
            chirp = np.sin(2 * np.pi * f * t) * pulses
            place(A, pan(chirp * 0.02 * dusk[int(t0 * SR)], side), t0)
            t0 += 1 / rate * r.uniform(0.85, 1.15)

    # a few distant bird chirps while the flock is in the sky
    for t0, p in ((0.7, 0.3), (0.85, 0.32), (2.4, 0.1), (2.55, 0.12), (4.9, -0.2)):
        n = int(0.07 * SR)
        t = np.arange(n) / SR
        f = 2900 + 2200 * (t / 0.07) ** 0.7
        ch = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / 0.07) ** 2
        place(A, pan(ch * 0.025, p), t0)
    return reverb(A, 1.6, 0.25, seed=4)


# --------------------------------------------------------------------------- bullet train
def train_x(time):
    """Centre x (px) of the train, same motion as render.py."""
    tx = -R.TRAIN_L - 60 + (R.W + R.TRAIN_L + 140) * np.clip((time - 0.2) / 9.6, 0, 1)
    return tx + R.TRAIN_L / 2


def train():
    cx = train_x(T)
    rel = (cx - R.W / 2)
    near = np.exp(-(rel / 1150) ** 2)  # loudest as it passes the middle of the frame
    near *= np.clip((T - 0.0) / 0.6, 0, 1)
    p = np.tanh(rel / 800) * 0.9
    dop = 1 - 0.1 * np.tanh(rel / 600)  # slight pitch drop as it recedes

    n = rng.standard_normal(NS)
    rumble = fft_filter(n, lo=30, hi=220, tilt=-2) * 1.6
    whoosh = svf_bandpass(fft_filter(rng.standard_normal(NS), lo=150, hi=5000), 650 * dop, 1.2) * 1.1
    hiss = fft_filter(rng.standard_normal(NS), lo=2500, hi=7000) * 0.12
    hum = sum(np.sin(2 * np.pi * f * np.cumsum(dop) / SR) / k for k, f in enumerate((110, 220, 330), 1)) * 0.025
    mono = (rumble + whoosh + hiss + hum) * near

    # rail clicks: "ta-tan ta-tan" as bogie pairs cross joints, one pair per car
    speed = (R.W + R.TRAIN_L + 140) / 9.6
    car_t = (R.TRAIN_L / 5) / speed
    clicks = np.zeros(NS)
    k = int(0.03 * SR)
    tk = np.arange(k) / SR
    click = fft_filter(rng.standard_normal(k), lo=900, hi=4500) * np.exp(-tk / 0.004)
    click += np.sin(2 * np.pi * 2100 * tk) * np.exp(-tk / 0.012) * 0.4
    t0 = 0.4
    while t0 < DUR:
        for d, v in ((0.0, 1.0), (0.085, 0.8), (0.28, 0.9), (0.365, 0.7)):
            place(clicks, click * v, t0 + d)
        t0 += car_t
    mono += clicks * near * 0.35
    st = pan(mono, p) * 0.33
    return reverb(st, 1.2, 0.18, seed=7)


# --------------------------------------------------------------------------- mix
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(HERE, "ghibli-sunset-bullet-train.mp4"))
    ap.add_argument("--wav", default=os.path.join(HERE, "soundtrack.wav"))
    ap.add_argument("--no-mux", action="store_true")
    args = ap.parse_args()

    mix = music() * 0.85 + ambience() * 1.8 + train() * 1.0
    fade = np.clip(T / 0.4, 0, 1) * np.clip((DUR - T) / 1.2, 0, 1) ** 1.5
    mix *= fade[:, None]
    mix *= 0.89 / np.max(np.abs(mix))
    pcm = (mix * 32767).astype("<i2")
    with wave.open(args.wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", args.wav)

    if not args.no_mux:
        tmp = args.video + ".tmp.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", args.video, "-i", args.wav, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", tmp],
            check=True)
        os.replace(tmp, args.video)
        print("muxed into", args.video)


if __name__ == "__main__":
    main()
