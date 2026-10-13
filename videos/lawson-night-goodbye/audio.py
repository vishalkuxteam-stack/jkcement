#!/usr/bin/env python3
"""Soundtrack for the Lawson night scene, timed to scene.py, then muxed.

Quiet night ambience (crickets, distant traffic, the shop's fluorescent hum),
the automatic door chime and slide, footsteps and a plastic bag rustle, the
Hindi voice lines (voices/*.wav) panned to where each character stands, and
a soft piano phrase after the boy's hesitant "bye".

    python3 audio.py   # writes soundtrack.wav and lawson-night-goodbye.mp4
"""
import argparse
import math
import os
import subprocess
import wave

import numpy as np
import soundfile as sf

import scene as S

SR = 48000
DUR = S.DUR
NS = int(SR * DUR)
T = np.arange(NS) / SR
HERE = S.HERE
rng = np.random.default_rng(5)


def fft_filter(x, lo=None, hi=None, tilt=0.0):
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


def pan(mono, p):
    a = (np.clip(np.asarray(p, dtype=float), -1, 1) + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], -1)


def place(buf, sig, t0):
    i0 = int(t0 * SR)
    if i0 < 0:
        sig, i0 = sig[-i0:], 0
    n = min(len(sig), len(buf) - i0)
    if n > 0:
        buf[i0:i0 + n] += sig[:n]


def reverb(st, seconds=1.5, wet=0.25, seed=9):
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = np.zeros_like(st)
    for ch in range(2):
        ir = fft_filter(r.standard_normal(n) * np.exp(-tt * 6.9 / seconds), lo=200, hi=6000)
        ir[: int(0.015 * SR)] = 0
        ir /= np.sqrt(np.sum(ir ** 2))
        m = len(st) + n
        nfft = 1 << (m - 1).bit_length()
        out[:, ch] = np.fft.irfft(np.fft.rfft(st[:, ch], nfft) * np.fft.rfft(ir, nfft), nfft)[: len(st)]
    return st * (1 - wet) + out * wet * 2.0


def resample(a, sr_in):
    n = int(len(a) * SR / sr_in)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def piano(f, vel, length=5.0):
    n = int(length * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for k in range(1, 9):
        fk = k * f * math.sqrt(1 + 0.00035 * k * k)
        if fk > SR / 2.5:
            break
        decay = (0.5 + 0.5 * k) * (f / 262) ** 0.35
        env = np.exp(-t * decay) * 0.8 + np.exp(-t * decay * 0.25) * 0.2
        sig += (1 / k ** 1.3) * env * (np.sin(2 * np.pi * fk * t) + 0.8 * np.sin(2 * np.pi * fk * 1.0007 * t + 1.1))
    return sig * np.minimum(1, t / 0.006) * vel / 9


def screen_pan(world_x, t):
    z, cx, _ = S.camera(t)
    sx = (world_x - cx) * S.BASE * z + S.W / 2
    return float(np.clip((sx - S.W / 2) / (S.W * 0.7), -0.85, 0.85))


# --------------------------------------------------------------------------- layers
def ambience():
    A = np.zeros((NS, 2))
    for ch, sd in enumerate((21, 22)):
        n = np.random.default_rng(sd).standard_normal(NS)
        swell = 0.7 + 0.3 * np.sin(T * 0.35 + ch)
        A[:, ch] += fft_filter(n, lo=40, hi=380, tilt=-4) * 0.05 * swell  # distant traffic / night air
    # shop fluorescent + fridge hum, from behind the characters
    hum = sum(np.sin(2 * np.pi * f * T + i) / (i + 1) for i, f in enumerate((100, 200, 300, 400, 1200)))
    hum = hum * (1 + 0.15 * np.sin(2 * np.pi * 0.21 * T)) * 0.006
    A += pan(hum, 0.1)
    # crickets, sparse and far
    for side, rate, f, sd in ((-0.6, 1.6, 4600, 1), (0.65, 1.3, 5200, 2), (0.1, 0.9, 4300, 3)):
        r = np.random.default_rng(sd)
        t0 = r.uniform(0, 0.6)
        while t0 < DUR:
            n = int(0.1 * SR)
            t = np.arange(n) / SR
            pulses = np.sin(np.pi * np.clip((t % 0.032) / 0.024, 0, 1)) ** 2 * (t < 0.096)
            place(A, pan(np.sin(2 * np.pi * f * t) * pulses * 0.012, side), t0)
            t0 += 1 / rate * r.uniform(0.8, 1.25)
    return reverb(A, 1.2, 0.2, seed=3)


def door():
    A = np.zeros((NS, 2))
    p = screen_pan(454, 1.4)
    # soft two-tone entrance chime
    for t0, nte in ((1.25, 76), (1.6, 72)):
        n = int(1.6 * SR)
        t = np.arange(n) / SR
        tone = sum(np.sin(2 * np.pi * midi(nte) * k * t) * np.exp(-t * (2.2 + k)) / k ** 1.5 for k in (1, 2, 3))
        place(A, pan(tone * 0.05, p), t0)
    # sliding panes: filtered noise swell + a tiny end thud
    for t0, t1 in ((1.35, 2.0), (4.3, 5.0)):
        n = int((t1 - t0) * SR)
        env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
        whoosh = fft_filter(rng.standard_normal(n), lo=250, hi=2200) * env * 0.035
        place(A, pan(whoosh, p), t0)
        k = int(0.08 * SR)
        thud = fft_filter(rng.standard_normal(k), hi=300) * np.exp(-np.arange(k) / (0.015 * SR)) * 0.06
        place(A, pan(thud, p), t1 - 0.02)
    return reverb(A, 1.0, 0.25, seed=6)


def footstep(vol, bright=1.0):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    click = fft_filter(rng.standard_normal(n), lo=180, hi=1800 * bright) * np.exp(-t / 0.018)
    body = np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.03) * 0.5
    return (click + body) * vol


def steps():
    A = np.zeros((NS, 2))
    for who, keys in (("girl", S.GIRL_PATH), ("boy", S.BOY_PATH)):
        pos, phase, speed = S.path_and_phase(keys)
        last = None
        for i in range(S.N):
            t = i / S.FPS
            k = int(phase[i] // math.pi)
            if last is not None and k != last and speed[i] > 2:
                running = who == "girl" and t > 11.6
                vol = 0.05 if who == "boy" else 0.04
                if running:
                    vol = 0.07 * max(0.15, 1 - (t - 11.7) / 2.2)
                if pos[i][1] < 1073:
                    vol *= 0.6
                place(A, pan(footstep(vol, 1.4 if who == "girl" else 1.0), screen_pan(pos[i][0], t)), t)
                if who == "boy":  # the plastic bag swings with each step
                    n = int(0.18 * SR)
                    rustle = fft_filter(rng.standard_normal(n), lo=1500, hi=7000) * np.sin(np.linspace(0, np.pi, n)) ** 2
                    place(A, pan(rustle * 0.018, screen_pan(pos[i][0], t)), t + 0.03)
            last = k
    return reverb(A, 0.9, 0.2, seed=8)


def voices():
    A = np.zeros((NS, 2))
    gpos, _, _ = S.path_and_phase(S.GIRL_PATH)
    for start, name, who, _ in S.LINES:
        a, sr = sf.read(os.path.join(HERE, "voices", f"{name}.wav"), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        a = resample(a, sr)
        a = a / (np.max(np.abs(a)) + 1e-9) * (0.5 if who == "girl" else 0.52)
        if name == "boy_bye":
            a *= 0.8  # hesitant, quieter
        fi = int(start * S.FPS)
        x = gpos[min(fi, S.N)][0] if who == "girl" else 414
        n = len(a)
        if name == "girl_ok_bye":  # she is already running off to the right
            px = np.array([screen_pan(gpos[min(S.N, int((start + j / SR) * S.FPS))][0], start + j / SR) for j in range(0, n, 480)])
            p = np.interp(np.arange(n), np.arange(0, n, 480), px)
        else:
            p = screen_pan(x, start)
        place(A, pan(a, p * 0.7), start)
    return reverb(A, 0.8, 0.12, seed=12)


def music():
    A = np.zeros((NS, 2))
    # a few soft notes as the scene opens, then a gentle phrase after "bye"
    for t0, nte, v, p in ((0.5, 69, 0.35, -0.2), (0.5, 57, 0.3, -0.3), (1.6, 76, 0.28, 0.1),
                          (15.0, 72, 0.4, 0.0), (15.0, 53, 0.35, -0.3), (15.0, 64, 0.25, -0.1),
                          (15.55, 74, 0.32, 0.1), (16.1, 76, 0.34, 0.0), (16.1, 57, 0.3, -0.3)):
        place(A, pan(piano(midi(nte), v), p), t0)
    return reverb(A, 2.6, 0.4, seed=15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(HERE, "scene_silent.mp4"))
    ap.add_argument("--out", default=os.path.join(HERE, "lawson-night-goodbye.mp4"))
    ap.add_argument("--wav", default=os.path.join(HERE, "soundtrack.wav"))
    args = ap.parse_args()

    mix = ambience() * 1.4 + door() + steps() * 2.2 + voices() + music() * 0.7
    fade = np.clip(T / 0.9, 0, 1) * np.clip((DUR - T) / 1.2, 0, 1)
    mix *= fade[:, None]
    mix *= 0.89 / np.max(np.abs(mix))
    pcm = (mix * 32767).astype("<i2")
    with wave.open(args.wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", args.wav)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video, "-i", args.wav, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out], check=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
