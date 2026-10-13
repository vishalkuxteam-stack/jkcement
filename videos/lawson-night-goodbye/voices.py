#!/usr/bin/env python3
"""Generate the four Hindi dialogue lines with Kokoro (offline neural TTS).

The resulting WAVs are committed under voices/, so this only needs re-running
to change a line. Kokoro model files (from the kokoro-onnx GitHub releases):

    pip install kokoro-onnx soundfile
    curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
    curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
    python3 voices.py --model-dir <dir with those two files>
"""
import argparse
import os

import soundfile as sf
from kokoro_onnx import Kokoro

HERE = os.path.dirname(os.path.abspath(__file__))

LINES = [  # file, voice, speed, text
    ("boy_kal_kab", "hm_omega", 0.92, "कल कब मिलना है?"),
    ("girl_same_time", "hf_alpha", 0.9, "सेम टाइम, सेम प्लेस।"),
    ("girl_ok_bye", "hf_alpha", 1.1, "ओके, बाय!"),
    ("boy_bye", "hm_omega", 0.78, "ब... बाय।"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=".")
    args = ap.parse_args()
    k = Kokoro(os.path.join(args.model_dir, "kokoro-v1.0.onnx"), os.path.join(args.model_dir, "voices-v1.0.bin"))
    os.makedirs(os.path.join(HERE, "voices"), exist_ok=True)
    for name, voice, speed, text in LINES:
        samples, sr = k.create(text, voice=voice, speed=speed, lang="hi")
        path = os.path.join(HERE, "voices", f"{name}.wav")
        sf.write(path, samples, sr)
        print(f"{path}: {len(samples) / sr:.2f}s")


if __name__ == "__main__":
    main()
