# How to build

## 1. Download FFMPEG
Tutorial: [Youtube](https://www.youtube.com/watch?v=c3bT-VuVoUA)
## 2. Download Whisper, and SoundDevice libraries
```sh
pip install torch
pip install git+https://github.com/openai/whisper.git
pip install sounddevice
```
# How to use
- Choose the **model** to download:
  + tiny ( ~75MiB )
  + base ( ~142MiB )
  + small ( ~466MiB )
  + medium ( ~1.5GiB )
  + large ( ~2.9GiB )
- Choose which **Microphone** device to input (*Just use **0** if you don't know anything*)
- Recording creates a *recorded_audio.wav* in the same folder as the source code.
### Have fun, bye
