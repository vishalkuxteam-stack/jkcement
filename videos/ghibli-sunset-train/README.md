# Ghibli-style sunset with bullet train

`ghibli-sunset-bullet-train.mp4`: 10 s, 1080×1920 (9:16, like the reference image), 30 fps, H.264.

A snow-capped mountain at golden hour. The sun sinks behind its right flank, and a
white-and-blue bullet train crosses an elevated viaduct from left to right. Clouds
drift, the grass sways and birds fly past. As dusk falls, the stars, the crescent moon,
the village windows and the train windows light up. The power pole and wires are a nod
to the reference image.

Everything is drawn procedurally with numpy and Pillow and encoded with ffmpeg:

```sh
python3 render.py                                 # writes ghibli-sunset-bullet-train.mp4
python3 render.py --preview 0 150 299 --outdir /tmp   # PNG stills of chosen frames
```

Requires Python 3 with `numpy` and `Pillow`, plus `ffmpeg` with libx264.
