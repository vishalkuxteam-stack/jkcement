# "Kal kab milna hai?": a night outside the Lawson

`lawson-night-goodbye.mp4`: 17 s, 1080×1920, 30 fps, H.264 with stereo AAC audio.

The set is the supplied painting (`background.webp`), unchanged except that its automatic
doors slide open. Two characters walk out, inspired by the reference grid: a Kiki-style girl
(dark bob, red bow, navy dress) and a Howl-style boy (silver hair, blue shirt, carrying a
convenience-store bag). They're drawn in code as simple cel-style art, so they're nods to
those characters rather than copies.

| Time | Beat |
|---|---|
| 0–2 s | Quiet establishing shot; the entrance chime plays and the doors slide open |
| 2–4 s | They step out onto the pavement as the camera slowly pushes in |
| 4.7 s | Boy: "Kal kab milna hai?" |
| 6.75 s | Girl: "Same time, same place." |
| 8.6–10.6 s | Two seconds of silence; she looks down shyly with her hands behind her back |
| 10.6–11.3 s | She steps closer and looks up at him; he blushes |
| 11.6 s | She suddenly runs off to the right: "Ok, bye!" |
| 13–15 s | He keeps looking after her, half-raises a hand: "B... bye." |
| 15–17 s | A soft piano phrase, then fade out |

## Files

- `scene.py`: draws the picture: camera moves, door, characters, lip-sync and subtitles. Writes `scene_silent.mp4`.
- `audio.py`: builds the soundtrack and adds it to the video. It has night ambience (crickets, distant
  traffic, the shop's hum), the door chime and slide, footsteps and bag rustle, the voices placed
  left or right to match where each character stands, and piano. Writes `lawson-night-goodbye.mp4`.
- `voices.py` and `voices/*.wav`: the four Hindi lines, generated offline with Kokoro TTS (`hm_omega` for
  the boy, `hf_alpha` for the girl). The WAVs are committed, so `voices.py` only needs to run again
  to change a line.

```sh
python3 scene.py && python3 audio.py
python3 scene.py --preview 5.2 11.4 --outdir /tmp   # stills at given seconds
```

Requires Python 3 with `numpy`, `Pillow` and `soundfile`, plus `ffmpeg` with libx264.
