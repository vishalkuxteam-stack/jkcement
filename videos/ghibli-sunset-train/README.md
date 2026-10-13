# Ghibli-style sunset with bullet train

`ghibli-sunset-bullet-train.mp4`: 10 s, 1080×1920 (9:16, like the reference image), 30 fps, H.264,
with a stereo AAC soundtrack.

A snow-capped mountain at golden hour. The sun sinks behind its right flank, and a
white-and-blue bullet train crosses an elevated viaduct from left to right. Clouds
drift, the grass sways and birds fly past. As dusk falls, the stars, the crescent moon,
the village windows and the train windows light up. The power pole and wires are a nod
to the reference image.

The soundtrack (`sound.py`) is synthesized as well. It has a gentle piano and strings cue
(Fmaj7, G6, Em7, Am9), with wind, evening cicadas, crickets that fade in at dusk and a few bird
chirps. The train's rumble and whoosh pan from left to right as it crosses, with a slight drop
in pitch as it moves away and soft rail clicks.

Everything is drawn procedurally with numpy and Pillow and encoded with ffmpeg:

```sh
python3 render.py                                 # writes ghibli-sunset-bullet-train.mp4 (silent)
python3 sound.py                                  # synthesizes the soundtrack and muxes it in
python3 render.py --preview 0 150 299 --outdir /tmp   # PNG stills of chosen frames
```

Requires Python 3 with `numpy` and `Pillow`, plus `ffmpeg` with libx264.
