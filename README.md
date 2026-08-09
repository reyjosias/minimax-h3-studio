# MiniMax H3 Local AI Studio

**A standalone local creator for MiniMax H3 video, audio and images, powered by your own ComfyUI.** No cloud account, subscription or remote rendering: prompts, references, generations and the personal library stay on your machine.

> Estudio local para crear y editar video con audio, imágenes y audio usando MiniMax H3 sobre ComfyUI. Todo se ejecuta en tu PC.

<p align="center">
  <img src="assets/screenshot.png" alt="MiniMax H3 Local AI Studio" width="100%">
</p>

## Samples

| Native 25-second video + audio | Native ~2.5K video |
|---|---|
| [![Play native 25-second sample](samples/poster-25s.jpg)](samples/sample-25s-native.mp4) | [![Play native 2.5K sample](samples/poster-2.5k.jpg)](samples/sample-2.5k.mp4) |
| **[▶ Play MP4](samples/sample-25s-native.mp4)** | **[▶ Play MP4](samples/sample-2.5k.mp4)** |

The MP4 files are stored in the repository so they can be opened outside the Studio.

## What it can do

### Create

- Text-to-video and image-to-video with native synchronized stereo audio.
- First-frame and last-frame guidance.
- Reference-to-video with one or several images, video and audio references.
- Audio-only generation with a dedicated player and MP3 download.
- Image-only generation at 1K, 2K or experimental 4K capture; choose first, middle or last frame, or extract three candidates.
- Many aspect ratios: 16:9, 9:16, 1:1, 2.35:1, 21:9, 2:1, 4:3, 3:2, 2:3, 5:4 and 4:5.
- Native video lengths entered as minutes and seconds, up to the H3 node limit (long jobs are experimental and expensive).
- Native resolutions through 2K/2.5K experiments. High resolutions may exceed 24 GB VRAM depending on duration and acceleration settings.

### Audio-conditioned video

- **Original clean audio:** H3 receives the complete source soundtrack as conditioning and generates visuals synchronized to it. The source waveform is then attached unchanged, avoiding doubled or distorted sound.
- **Reference + ambience:** uses H3 Ref2VA `partially_copy` semantics. It preserves speech, singing, music, effects and timing while allowing the prompt to add requested ambience, performances or sounds.
- The source can be speech, music, singing, effects, ambience, or a combination—not voice only.

### Native H3 video editing

- Edit an existing video by text: change clothing color, lighting, time of day, objects or scene attributes while preserving continuity.
- Combine video + image references for targeted replacement or attribute transfer.
- Camera recomposition: close-up, extreme close-up, frame only the lips, crop, focus, viewpoint, angle or shot-size changes.
- Edit only a selected time interval. The Studio sends that segment to H3, then splices it back into the untouched source at the original timing.
- Preserve the original source audio during visual edits, or condition on another audio track.
- Structured `<Video 1>`, `<Picture n>` and `<Audio n>` reference prompts are assembled automatically.

### Work like a studio

- Minimal chat composer with persistent attachments, disable/enable toggles and a **New** reset.
- Video, image and audio outputs each have the correct preview, actions and download type.
- Projects with separate libraries; **Master** shows every project.
- Real progress, elapsed time, live ETA, cancel and process pause; state survives page refresh.
- Sequential queue: keep viewing a completed result while the next item continues generating.
- **Recreate with this setup** restores prompt, dimensions, duration, seed and accelerators.
- **Creative Space:** large player, continuous clip timeline, add/extend from chat and export the joined result.
- Extract a video frame or soundtrack and immediately use it as a new reference.

### Local post-processing

- FlashVSR fast/quality upscaling.
- RIFE 48 fps and 72 fps frame interpolation while retaining duration and audio.
- Image frame extraction and MP3 export.

### Optional acceleration

- Six measured quick profiles (0–5) plus fully manual advanced controls.
- Turbo LoRA with the dedicated 4–8-step sampler.
- SageAttention + EasyCache.
- Spectrum transformer forecasting.
- MiniMax H3 FirstBlockCache.
- Memory residency/offload profiles for 24 GB cards.
- Optional INT8 video VAE and VAE compile paths.
- Default quality workflow remains **20 steps**; reference/video-edit workflows use the official Ref2VA path and bypass incompatible Turbo batching automatically.

Acceleration is hardware- and shape-dependent. A configuration that helps at 640×352 can provide little gain—or run out of memory—at 2K. The controls remain optional, and the Studio reports a real backend crash instead of leaving a fake progress bar at 90%.

## Requirements

- Windows 10/11 (the launcher and process pause controls are currently Windows-oriented).
- Python 3.10+.
- A working local [ComfyUI](https://github.com/Comfy-Org/ComfyUI) installation.
- NVIDIA CUDA GPU. Developed and measured on an RTX 3090 24 GB with 128 GB RAM.
- `ffmpeg` and `ffprobe`.
- About **76 GB** for the recommended H3 INT8/Ref2VA stack and Turbo LoRA, plus space for outputs.

### Model files

The setup wizard checks these locations under your ComfyUI model root and can download missing required/recommended files after showing the size and asking permission:

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
loras/minimax_h3_turbo_4step_ckpt500.safetensors
```

Optional measured profile:

```text
vae/minimax_h3_video_vae_int8_convrot.safetensors
```

Primary model sources: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), [MiniMax H3 Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora), and the optional [Kijai experimental INT8 VAE](https://huggingface.co/Kijai/MiniMax-H3-experimental).

### Custom nodes

`setup.bat` can clone the optional node packages after confirmation:

- [ComfyUI-MiniMax-H3-Turbo](https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo)
- [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)
- [ComfyUI-MiniMaxH3-FirstBlockCache](https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache)
- [ComfyUI-FlashVSR-Ultra-Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)

Some compiled acceleration packages (especially Triton/SageAttention) vary by Python, Torch, CUDA and GPU. If their installation fails, install them from ComfyUI Manager or leave that accelerator off; the base Studio remains usable.

## Install on Windows

```powershell
git clone https://github.com/reyjosias/minimax-h3-studio.git
cd minimax-h3-studio
setup.bat
```

The wizard:

1. Detects an existing ComfyUI installation and its Python environment.
2. Detects local/shared `models`, `input` and `output` folders.
3. Finds `ffmpeg`/`ffprobe`.
4. Writes private machine paths to `studio_config.json` (git-ignored).
5. Verifies the H3 files and optionally downloads missing large weights with confirmation and resume support.
6. Optionally installs the custom nodes.

Restart ComfyUI after setup, then run:

```text
start.bat
```

Open **http://127.0.0.1:8200/new**. `reiniciar.bat` restarts only the Studio; `apagar.bat` stops it without killing ComfyUI.

### Does cloning the repository configure ComfyUI automatically?

**Not silently.** GitHub does not include the model weights or modify another application merely by cloning a repository. Run `setup.bat`: it performs the local path configuration and offers the large model/node installations explicitly. This prevents an unexpected ~76 GB download or changes to the wrong ComfyUI installation.

If you are new to ComfyUI, give this repository to an AI coding assistant such as Claude or Codex and ask:

> Install MiniMax H3 Local AI Studio from this repository. Run its setup wizard, verify ComfyUI on port 8188, put every listed model in the correct folder, install the optional nodes, restart ComfyUI, and confirm that `/new` opens without generating a test video.

## Configuration

`studio_config.json` is machine-local and never committed. Environment variables override it:

| Variable | Purpose | Default |
|---|---|---|
| `COMFY_URL` | ComfyUI API | `http://127.0.0.1:8188` |
| `PORT` | Studio web port | `8200` |
| `COMFYUI_DIR` | ComfyUI root | setup-detected |
| `COMFY_MODEL_DIR` | model root | setup-detected |
| `COMFY_INPUT_DIR` | input folder | setup-detected |
| `COMFY_OUTPUT_DIR` | output folder | setup-detected |
| `FFMPEG` / `FFPROBE` | media binaries | PATH/setup-detected |

The personal SQLite library (`studio.db`), configuration, prompts and generated media are ignored by Git.

## Support development

This project represents substantial independent development, hardware testing and model research by **Rey Josias Reinoso**. If it saves you time or helps your work, please consider supporting continued development:

<p align="center">
  <a href="https://www.paypal.com/paypalme/rrjosias"><strong>❤️ Donate with PayPal</strong></a>
  &nbsp;·&nbsp;
  <a href="https://x.com/reyreinoso"><strong>Follow/contact @reyreinoso</strong></a>
</p>

Issues, test results, suggestions and pull requests are welcome. Please include GPU, VRAM, resolution, duration, profile and the relevant ComfyUI error when reporting a failure.

## Privacy and security

- The server binds to `127.0.0.1`; it is not exposed publicly by default.
- No API key is required.
- Generated files and the SQLite library remain local.
- Model licenses and upstream custom-node licenses still apply.

## License

[MIT](LICENSE) © 2026 Rey Josias Reinoso
