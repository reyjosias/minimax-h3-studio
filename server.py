#!/usr/bin/env python3
"""
MiniMax H3 Studio — standalone web UI for creating videos with MiniMax H3,
driving a local ComfyUI instance. Standard library only.

Real progress: the SERVER opens a websocket to ComfyUI (server->server, not
subject to the browser's cross-origin rules that block ws from the page) and
tracks true per-node execution progress, exposed through /api/status. The
browser just polls status — no browser websocket. Every finished video is
recorded in SQLite with its real generation time.

Run:  python server.py         (serve :8199, ComfyUI at :8188)
"""
import base64
import json
import os
import shutil
import socket
import sqlite3
import struct
import subprocess
import threading
import time
import urllib.request
import urllib.error
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
_host_port = COMFY.split("://", 1)[-1]
COMFY_HOST = _host_port.split(":")[0]
COMFY_PORT = int(_host_port.split(":")[1]) if ":" in _host_port else 8188
PORT = int(os.environ.get("PORT", "8199"))
HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "studio.db")

FFMPEG = r"C:\pinokio\bin\ffmpeg-env\Library\bin\ffmpeg.exe"
FFPROBE = r"C:\pinokio\bin\ffmpeg-env\Library\bin\ffprobe.exe"
OUTPUT_DIR = r"C:\Users\Rey\ComfyUI-Shared\output"
INPUT_DIR = r"C:\Users\Rey\ComfyUI-Shared\input"

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
UNET_INT4 = "minimax_h3_fl2va_pruned_int4_convrot.safetensors"  # 11GB, fits VRAM = no offload = faster
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

# node id -> (band_start, band_end, label). Progress reflects the real pipeline.
STAGE_BANDS = {
    "6": (2, 14, "Cargando modelo de video…"),
    "13": (14, 30, "Cargando text encoder (32B)…"),
    "11": (30, 32, "Preparando VAE de video…"),
    "24": (32, 34, "Preparando VAE de audio…"),
    "200": (34, 35, "Cargando imagen…"), "201": (35, 36, "Cargando imagen…"),
    "104": (36, 38, "Preparando escena…"),
    "9": (38, 39, "Programando…"), "17": (39, 40, "Iniciando…"),
    "15": (40, 41, "Iniciando…"), "16": (41, 42, "Iniciando…"),
    "14": (42, 90, "Generando (denoising)…"),
    "10": (90, 95, "Decodificando video…"),
    "23": (95, 98, "Decodificando audio…"),
    "91": (98, 99, "Creando video…"),
    "300": (2, 5, "Cargando video…"),
    "301": (5, 10, "Extrayendo frames…"),
    "93": (10, 98, "Escalando calidad (FlashVSR)…"),
    "303": (10, 95, "Interpolando (RIFE)…"),
    "92": (99, 100, "Guardando…"),
}
UPSCALE_SCALE = {"2x": 2, "4x": 4}   # FlashVSR local node scale factors

JOBS = {}                 # pid -> {meta, t0, recorded}
PROGRESS = {}             # pid -> {pct, stage, band:(s,e)}
CURRENT_PID = [None]      # most recent submitted job (single-user)
# ComfyUI routes execution/progress events ONLY to the ws whose clientId matches
# the client_id used at submit time. Use one fixed id for BOTH so the server ws
# receives real progress.
CLIENT_ID = uuid.uuid4().hex
PAUSED = [False]        # ComfyUI process suspended (pause)
COMFY_PID = [None]      # cached ComfyUI backend PID
LOCK = threading.Lock()
DB_LOCK = threading.Lock()


# ── database ─────────────────────────────────────────────────────────────
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with DB_LOCK, db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS videos(
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL, mode TEXT, prompt TEXT,
            width INTEGER, height INTEGER, seconds REAL, length INTEGER,
            seed INTEGER, gen_seconds REAL, filename TEXT)""")
        # migration: each video belongs to a project (a project = its own library;
        # the Master library shows every project's videos).
        cols = [r[1] for r in conn.execute("PRAGMA table_info(videos)").fetchall()]
        if "project" not in cols:
            conn.execute("ALTER TABLE videos ADD COLUMN project TEXT DEFAULT 'General'")
        # migration: full generation setup (JSON) so "Recrear con este setup" can
        # faithfully restore prompt + resolution + duration + accelerators.
        if "settings" not in cols:
            conn.execute("ALTER TABLE videos ADD COLUMN settings TEXT")
        conn.execute("""CREATE TABLE IF NOT EXISTS projects(
            name TEXT PRIMARY KEY, created_at REAL)""")
        conn.execute("INSERT OR IGNORE INTO projects(name, created_at) VALUES('General', ?)", (time.time(),))
        conn.commit()


def project_of(basename):
    """Project a source video belongs to (so upscale/extend/interpolate outputs
    stay in the same project). Defaults to 'General'."""
    basename = os.path.basename(str(basename or ""))
    with DB_LOCK, db() as conn:
        row = conn.execute("SELECT project FROM videos WHERE filename LIKE ? ORDER BY id DESC LIMIT 1",
                           ("%" + basename,)).fetchone()
    return (row["project"] if row and row["project"] else "General")


def record_video(meta, gen_seconds, filename):
    with DB_LOCK, db() as conn:
        settings = meta.get("settings")
        if isinstance(settings, dict):
            settings = json.dumps(settings)
        conn.execute("""INSERT INTO videos(created_at,mode,prompt,width,height,seconds,length,seed,gen_seconds,filename,project,settings)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                     (time.time(), meta.get("mode"), meta.get("prompt"), meta.get("width"),
                      meta.get("height"), meta.get("seconds"), meta.get("length"),
                      meta.get("seed"), gen_seconds, filename, meta.get("project") or "General", settings))
        conn.commit()


# ── ComfyUI websocket client (server side, for real progress) ────────────
def _apply_ws(msg):
    t = msg.get("type")
    d = msg.get("data") or {}
    pid = d.get("prompt_id") or CURRENT_PID[0]
    if not pid:
        return
    now = time.time()
    with LOCK:
        st = PROGRESS.setdefault(pid, {"pct": 0.0, "stage": "Enviando…", "band": (0, 2), "eta": None})
        if t == "executing" and d.get("node"):
            b = STAGE_BANDS.get(str(d["node"]))
            if b:
                st["stage"] = b[2]; st["band"] = (b[0], b[1]); st["pct"] = max(st["pct"], b[0])
                st["stepped"] = False  # a new node started; only re-flag if it emits steps
        elif t == "execution_cached":
            for n in d.get("nodes", []):
                b = STAGE_BANDS.get(str(n))
                if b:
                    st["pct"] = max(st["pct"], b[1])
        elif t == "progress_state":
            # 0.30.x format: data.nodes[nodeId] = {value, max, state}
            for nid, info in (d.get("nodes") or {}).items():
                b = STAGE_BANDS.get(str(nid))
                if not b:
                    continue
                mx = info.get("max") or 0
                val = info.get("value") or 0
                state = info.get("state")
                if state == "finished":
                    cand = b[1]
                elif mx > 0:
                    cand = b[0] + (val / mx) * (b[1] - b[0])
                else:
                    cand = b[0]
                st["pct"] = max(st["pct"], cand)
                # a multi-step node (sampler / FlashVSR) drives the bar by REAL
                # steps — flag it so the ease loop stops racing ahead (fake 90%).
                st["stepped"] = mx > 1
                if state == "running":
                    st["stage"] = b[2]; st["band"] = (b[0], b[1])
                    # LIVE ETA: measure real seconds/step on the dominant
                    # multi-step node (sampler node 14, or FlashVSR node 93 when
                    # upscaling) and extrapolate remaining steps + a small tail.
                    if mx > 1:
                        s = st.setdefault("_samp", {"lv": 0, "lt": now, "sps": 0.0})
                        if val > s["lv"]:
                            s["sps"] = (now - s["lt"]) / (val - s["lv"])
                            s["lv"] = val; s["lt"] = now
                        if s["sps"] > 0:
                            st["eta"] = max(0.0, (mx - val) * s["sps"] + 15)
        elif t == "executed" and str(d.get("node")) == "92":
            st["pct"] = 100.0; st["eta"] = 0
        elif t == "execution_success":
            st["pct"] = max(st["pct"], 99.5)


def _ws_recv_exact(sock, n, pre):
    while len(pre) < n:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("ws closed")
        pre += chunk
    out = bytes(pre[:n]); del pre[:n]; return out


def _ws_loop():
    """Persistent websocket client to ComfyUI; never raises out."""
    while True:
        try:
            s = socket.create_connection((COMFY_HOST, COMFY_PORT), timeout=10)
            key = base64.b64encode(os.urandom(16)).decode()
            s.sendall((
                f"GET /ws?clientId={CLIENT_ID} HTTP/1.1\r\n"
                f"Host: {COMFY_HOST}:{COMFY_PORT}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                f"Origin: http://{COMFY_HOST}:{COMFY_PORT}\r\n\r\n").encode())
            pre = bytearray()
            while b"\r\n\r\n" not in pre:
                pre += s.recv(4096)
            del pre[:pre.index(b"\r\n\r\n") + 4]
            while True:
                b0 = _ws_recv_exact(s, 1, pre)[0]
                op = b0 & 0x0f
                b1 = _ws_recv_exact(s, 1, pre)[0]
                ln = b1 & 0x7f
                if ln == 126:
                    ln = struct.unpack(">H", _ws_recv_exact(s, 2, pre))[0]
                elif ln == 127:
                    ln = struct.unpack(">Q", _ws_recv_exact(s, 8, pre))[0]
                mask = _ws_recv_exact(s, 4, pre) if (b1 & 0x80) else b""
                payload = _ws_recv_exact(s, ln, pre)
                if mask:
                    payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
                if op == 0x8:  # close
                    break
                if op == 0x9:  # ping -> pong
                    s.sendall(bytes([0x8a, len(payload)]) + payload)
                    continue
                if op == 0x1:  # text
                    try:
                        _apply_ws(json.loads(payload.decode("utf-8", "ignore")))
                    except Exception:
                        pass
        except Exception:
            time.sleep(2)


def _ease_loop():
    """Nudge progress within the current band when a stage emits no steps
    (e.g. the long model load), so the bar moves — but never past the band."""
    while True:
        time.sleep(0.5)
        pid = CURRENT_PID[0]
        if not pid or PAUSED[0]:
            continue
        with LOCK:
            st = PROGRESS.get(pid)
            if not st:
                continue
            # Never nudge during a stepped stage (denoising / FlashVSR): those
            # report REAL per-step progress, so easing would fake a high % while
            # the model is barely underway. Only ease short bands with no steps.
            if st.get("stepped"):
                continue
            s, e = st.get("band", (0, 2))
            target = e - 0.4
            if st["pct"] < target:
                st["pct"] = min(target, st["pct"] + 0.22)


def frames_for_seconds(seconds: float) -> int:
    length = max(5, round(seconds * 24))
    length += (5 - (length % 17)) % 17
    # H3 node hard-caps length at 3600; largest valid 5+17k below that is 3592
    # (~149.7s ≈ 2m30s). Beyond ~15s the model extrapolates past its training.
    return min(length, 3592)


def build_graph(*, prompt, width, height, length, seed, steps=20, first_image=None, last_image=None):
    g = {
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "24": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "104": {"class_type": "MiniMaxH3ImageToVideo",
                "inputs": {"clip": ["13", 0], "vae": ["11", 0], "prompt": prompt,
                           "width": width, "height": height, "length": length}},
        "9": {"class_type": "BasicScheduler",
              "inputs": {"model": ["6", 0], "scheduler": "simple", "steps": int(steps), "denoise": 1.0}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "16": {"class_type": "BasicGuider", "inputs": {"model": ["6", 0], "conditioning": ["104", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0],
                          "sigmas": ["9", 0], "latent_image": ["104", 1]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24, "bit_depth": 8}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": "video/MiniMaxStudio",
                          "format": "auto", "codec": "auto"}},
    }
    if first_image:
        g["200"] = {"class_type": "LoadImage", "inputs": {"image": first_image}}
        g["104"]["inputs"]["first_frame"] = ["200", 0]
    if last_image:
        g["201"] = {"class_type": "LoadImage", "inputs": {"image": last_image}}
        g["104"]["inputs"]["last_frame"] = ["201", 0]
    return g


def build_ref_graph(*, prompt, width, height, length, seed, steps=20,
                    ref_images=None, ref_video=None, use_video_audio=True, ref_audios=None):
    """Reference-to-Video (Ref2VA): generate a video conditioned on reference
    images, a reference VIDEO (video-to-video, frames + optional its audio) and
    reference audios. Wiring copied from the official video_minimax_h3_r2v
    template: the node's autogrow inputs use dotted keys and it needs audio_vae.
    Files are read by LoadImage/LoadVideo/LoadAudio from ComfyUI's input dir.
    """
    ref_images = ref_images or []
    ref_audios = ref_audios or []
    g = {
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "24": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "136": {"class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"clip": ["13", 0], "vae": ["11", 0], "audio_vae": ["24", 0],
                           "prompt": prompt, "width": width, "height": height,
                           "length": length, "ref_image_size": "match"}},
        "9": {"class_type": "BasicScheduler",
              "inputs": {"model": ["6", 0], "scheduler": "simple", "steps": int(steps), "denoise": 1.0}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "16": {"class_type": "BasicGuider", "inputs": {"model": ["6", 0], "conditioning": ["136", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0],
                          "sigmas": ["9", 0], "latent_image": ["136", 1]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24, "bit_depth": 8}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": "video/MiniMaxStudio",
                          "format": "auto", "codec": "auto"}},
    }
    nid = 200
    for i, name in enumerate(ref_images[:3]):
        g[str(nid)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["136"]["inputs"][f"ref_images.ref_image_{i}"] = [str(nid), 0]
        nid += 1
    if ref_video:
        g[str(nid)] = {"class_type": "LoadVideo", "inputs": {"file": ref_video}}
        vid = str(nid); nid += 1
        g[str(nid)] = {"class_type": "GetVideoComponents", "inputs": {"video": [vid, 0]}}
        gvc = str(nid); nid += 1
        g["136"]["inputs"]["ref_videos.ref_video_0"] = [gvc, 0]
        if use_video_audio:
            g["136"]["inputs"]["ref_video_audios.ref_video_audio_0"] = [gvc, 1]
    for j, name in enumerate(ref_audios[:3]):
        g[str(nid)] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        g["136"]["inputs"][f"ref_audios.ref_audio_{j}"] = [str(nid), 0]
        nid += 1
    return g


def _parse_upscale(upscale):
    """"esrgan_2x" | "esrgan_4x" | "flashvsr_2x" | "flashvsr_4x" -> (engine, scale)."""
    if not upscale or "_" not in str(upscale):
        return None, None
    engine, s = str(upscale).split("_", 1)
    if engine not in ("esrgan", "flashvsr", "flashfast"):
        return None, None
    return engine, (4 if "4" in s else 2)


def _upscale_nodes(engine, scale, frames):
    """Nodes that turn `frames` (IMAGE) into upscaled IMAGE at node '93'.
    esrgan = RealESRGAN per-frame (FAST, seconds). flashvsr = temporal diffusion
    VSR (best quality but SLOW on 24GB; tiling ON to avoid OOM)."""
    if engine == "flashfast":
        # FlashVSR "Ultra Fast": sparse (sage) attention via the Adv node — same
        # REAL temporal VSR model as flashvsr, but much faster. sparse_ratio/
        # kv_ratio/local_range at the fast end maximise sparsity.
        return {"95": {"class_type": "FlashVSRInitPipe",
                       "inputs": {"model": "FlashVSR-v1.1", "mode": "tiny", "alt_vae": "none",
                                  "force_offload": True, "precision": "bf16", "device": "auto",
                                  "attention_mode": "sparse_sage_attention"}},
                "93": {"class_type": "FlashVSRNodeAdv",
                       "inputs": {"pipe": ["95", 0], "frames": frames, "scale": scale,
                                  "color_fix": True, "tiled_vae": True, "tiled_dit": True,
                                  "tile_size": 256, "tile_overlap": 24, "unload_dit": False,
                                  "sparse_ratio": 1.5, "kv_ratio": 1.0, "local_range": 9, "seed": 0}}}
    if engine == "flashvsr":
        return {"93": {"class_type": "FlashVSRNode",
                       "inputs": {"frames": frames, "model": "FlashVSR-v1.1", "mode": "tiny",
                                  "scale": scale, "tiled_vae": True, "tiled_dit": True,
                                  "unload_dit": False, "seed": 0}}}
    model = "RealESRGAN_x2plus.pth" if scale == 2 else "RealESRGAN_x4plus.pth"
    return {"94": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model}},
            "93": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["94", 0], "image": frames}}}


def build_upscale_graph(source_name, upscale):
    """Standalone upscale of an existing video: load it, extract frames + audio,
    upscale the frames (ESRGAN or FlashVSR), rebuild keeping fps and audio."""
    engine, sc = _parse_upscale(upscale)
    if engine is None:
        engine, sc = "esrgan", 2
    g = {
        "300": {"class_type": "LoadVideo", "inputs": {"file": source_name}},
        "301": {"class_type": "GetVideoComponents", "inputs": {"video": ["300", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["93", 0], "audio": ["301", 1], "fps": ["301", 2]}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": "video/MiniMaxStudio",
                          "format": "auto", "codec": "auto"}},
    }
    g.update(_upscale_nodes(engine, sc, ["301", 0]))
    return g


INTERP_MODEL = "rife_v4.26.safetensors"  # RIFE, native ComfyUI frame interpolation


def build_interp_graph(source_name, multiplier):
    """Frame interpolation (RIFE): raise a video's fps (smoother motion) by
    generating intermediate frames. Keeps the SAME duration — output fps = 24 ×
    multiplier, output frames = input frames × multiplier. Audio passes through."""
    m = max(2, min(8, int(multiplier)))
    return {
        "300": {"class_type": "LoadVideo", "inputs": {"file": source_name}},
        "301": {"class_type": "GetVideoComponents", "inputs": {"video": ["300", 0]}},
        "302": {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": INTERP_MODEL}},
        "303": {"class_type": "FrameInterpolate",
                "inputs": {"interp_model": ["302", 0], "images": ["301", 0], "multiplier": m}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["303", 0], "audio": ["301", 1], "fps": 24 * m}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": "video/MiniMaxStudio",
                          "format": "auto", "codec": "auto"}},
    }


TURBO_LORA = "minimax_h3_turbo_4step_ckpt500.safetensors"  # larryvrh 4-step Turbo LoRA


def apply_turbo_lora(g, on):
    """Optional 4-8 step Turbo LoRA (larryvrh, ComfyUI-MiniMax-H3-Turbo). Patches
    the H3 diffusion model with the distilled LoRA and swaps the sampler for the
    dedicated MiniMaxH3TurboSampler (standard samplers break audio at few steps).
    Big speedup on step-dominated jobs. Works on the pruned int8_convrot base."""
    if not on or "9" not in g or "17" not in g:
        return g
    src = g["9"]["inputs"].get("model", ["6", 0])  # current model feeding the scheduler
    g["61"] = {"class_type": "MiniMaxH3TurboLoRA",
               "inputs": {"model": src, "lora_name": TURBO_LORA, "strength": 1.0}}
    g["9"]["inputs"]["model"] = ["61", 0]
    if "16" in g:
        g["16"]["inputs"]["model"] = ["61", 0]
    g["17"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
    return g


def apply_spectrum(g, on, steps=6):
    """Optional Spectrum accelerator (xmarre/ComfyUI-Spectrum-MiniMax-H3).
    Chebyshev ridge forecasting of hidden features lets it SKIP transformer
    evals on the forecast steps. It's a plain MODEL patch, so it stacks ON TOP
    of the Turbo LoRA — measured ~2x on top of Turbo at a short config with no
    execution error against this ComfyUI's H3 API. Inserted after whatever model
    currently feeds the scheduler(9)/guider(16) (i.e. after the Turbo LoRA when
    that's on). `warmup_steps` real steps run first, then it forecasts; we scale
    warmup to steps//2 so at least half the steps stay REAL (protects quality —
    it's an APPROXIMATE method, diverges on fast motion if pushed too hard). The
    node checks required H3 attributes when applied and fails with an explicit
    contract error if incompatible, so it can't silently corrupt output."""
    if not on or "9" not in g:
        return g
    src = g["9"]["inputs"].get("model", ["6", 0])  # after Turbo LoRA if present
    warmup = max(2, int(steps) // 2)
    g["62"] = {"class_type": "SpectrumApplyMiniMaxH3",
               "inputs": {"model": src, "enabled": True, "blend_weight": 0.5,
                          "degree": 4, "ridge_lambda": 0.1, "window_size": 2.0,
                          "flex_window": 0.75, "warmup_steps": warmup,
                          "tail_actual_steps": 1, "max_history": 8,
                          "debug": False, "history_storage": "system_ram"}}
    g["9"]["inputs"]["model"] = ["62", 0]
    if "16" in g:
        g["16"]["inputs"]["model"] = ["62", 0]
    return g


def apply_turbo(g, turbo):
    """Optional speed accelerators (user-activatable). Patches the diffusion
    MODEL with SageAttention (via KJNodes) and optionally EasyCache, then
    rewires the scheduler/guider to the patched model. Nodes already present in
    ComfyUI. turbo: "off" | "sage" | "cache".
      - sage:  SageAttention only (~25% faster, keeps quality)
      - cache: EasyCache + SageAttention (~2-2.5x faster, slight quality loss)
    """
    if turbo not in ("sage", "cache"):
        return g
    # NOTE: torch.compile does NOT work with this model — the INT8 "convrot"
    # quantization uses custom kernels (comfy_kitchen) that torch Dynamo cannot
    # trace ("Cannot access data pointer of FakeTensor"), so any compile backend
    # fails at the sampler. Sage + EasyCache is the fastest quality-safe stack here.
    last = "6"  # UNETLoader
    if turbo == "cache":
        g["50"] = {"class_type": "EasyCache",
                   "inputs": {"model": [last, 0], "reuse_threshold": 0.30,
                              "start_percent": 0.20, "end_percent": 0.90, "verbose": False}}
        last = "50"
    g["51"] = {"class_type": "PathchSageAttentionKJ",
               "inputs": {"model": [last, 0], "sage_attention": "auto", "allow_compile": False}}
    last = "51"
    if "9" in g:
        g["9"]["inputs"]["model"] = [last, 0]
    if "16" in g:
        g["16"]["inputs"]["model"] = [last, 0]
    return g


def apply_mem(g, factor):
    """Reduce OFFLOAD (the real bottleneck on a 24GB 3090): lowering the model's
    memory-usage factor makes ComfyUI keep more of it resident in VRAM instead of
    streaming from RAM each step → faster, with NO quality change. Too low = OOM.
    Applied AFTER turbo so it wraps the raw UNET; all consumers of the raw model
    get rerouted through the override."""
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        return g
    if abs(factor - 1.0) < 1e-6 or "6" not in g:
        return g
    g["40"] = {"class_type": "ModelMemoryUsageFactorOverride",
               "inputs": {"model": ["6", 0], "memory_usage_factor": factor}}
    for nid, node in g.items():
        if nid in ("40", "6"):
            continue
        for k, v in node.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and v[0] == "6" and v[1] == 0:
                node["inputs"][k] = ["40", 0]
    return g


def apply_upscale(g, upscale):
    """Optional local upscale (user-activatable). Inserts the upscale nodes
    (ESRGAN fast or FlashVSR quality) between VAEDecode (frames, node 10) and
    CreateVideo (node 91). upscale is like "esrgan_2x" / "flashvsr_2x" / "off"."""
    engine, sc = _parse_upscale(upscale)
    if engine is None or "10" not in g or "91" not in g:
        return g
    g.update(_upscale_nodes(engine, sc, ["10", 0]))
    g["91"]["inputs"]["images"] = ["93", 0]
    return g


def apply_model(g, model):
    """Swap the diffusion checkpoint. "int4" (~11GB convrot) fits fully in the
    3090's VRAM → no offload → much faster steps; "int8" (~20GB) = max quality.
    Both use the same convrot UNETLoader (drop-in)."""
    if model == "int4" and "6" in g:
        g["6"]["inputs"]["unet_name"] = UNET_INT4
    return g


def _find_comfy_pid():
    """PID of the ComfyUI backend (the process listening on COMFY_PORT)."""
    if COMFY_PID[0]:
        return COMFY_PID[0]
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if f":{COMFY_PORT} " in line and "LISTENING" in line:
                COMFY_PID[0] = int(line.split()[-1])
                return COMFY_PID[0]
    except Exception:  # noqa
        pass
    return None


def _set_suspended(pid, suspend):
    """Suspend/resume all threads of a process (Windows). Used to pause the GPU
    work so the machine frees up, then continue where it left off."""
    if os.name != "nt" or not pid:
        return False
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x0800, False, int(pid))  # PROCESS_SUSPEND_RESUME
        if not h:
            return False
        if suspend:
            ctypes.windll.ntdll.NtSuspendProcess(h)
        else:
            ctypes.windll.ntdll.NtResumeProcess(h)
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    except Exception as e:  # noqa
        print("[Pause] error:", e)
        return False


def comfy_post(path, data_bytes, content_type):
    req = urllib.request.Request(COMFY + path, data=data_bytes, headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def comfy_get(path):
    with urllib.request.urlopen(COMFY + path, timeout=60) as r:
        return json.load(r)


def _ffprobe(path, entries, stream=True):
    sel = ["-select_streams", "v:0"] if stream else []
    try:
        r = subprocess.run([FFPROBE, "-v", "error"] + sel +
                           ["-show_entries", entries, "-of", "default=nw=1:nk=1", path],
                           capture_output=True, text=True, timeout=25)
        return r.stdout.strip().splitlines()
    except Exception:  # noqa
        return []


def _concat_extend(source_path, new_basename):
    """Concatenate the source video and the freshly generated continuation into
    one longer MP4 (re-encoded so mismatched encodes join cleanly). Returns
    (final_basename, total_frames)."""
    new_path = os.path.join(OUTPUT_DIR, "video", new_basename)
    final_basename = f"MiniMaxStudio_ext_{int(time.time())}.mp4"
    final_path = os.path.join(OUTPUT_DIR, "video", final_basename)
    cmd = [FFMPEG, "-y", "-i", source_path, "-i", new_path,
           "-filter_complex", "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]",
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", final_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not os.path.isfile(final_path):
        raise RuntimeError((r.stderr or "")[-300:])
    frames = 0
    dur = _ffprobe(final_path, "format=duration", stream=False)
    try:
        frames = int(round(float(dur[0]) * 24)) if dur else 0
    except (ValueError, IndexError):
        frames = 0
    return final_basename, frames


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _qs(self, key, default=""):
        from urllib.parse import parse_qs, urlparse
        return parse_qs(urlparse(self.path).query).get(key, [default])[0]

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(500, {"error": "index.html missing"})
        if path in ("/new", "/indexnew.html"):
            try:
                with open(os.path.join(HERE, "new design", "indexnew.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(404, {"error": "indexnew.html missing"})
        if path == "/tailwind.js":
            try:
                with open(os.path.join(HERE, "new design", "tailwind.js"), "rb") as f:
                    return self._send(200, f.read(), "application/javascript; charset=utf-8")
            except OSError:
                return self._send(404, {"error": "tailwind.js missing"})
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if path == "/api/status":
            return self._status(self._qs("id"))
        if path == "/api/active":
            return self._active()
        if path == "/api/library":
            return self._library()
        if path == "/api/projects":
            return self._projects()
        if path == "/api/video":
            return self._video(self._qs("name"))
        if path == "/api/input":
            return self._input(self._qs("name"))
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        if path == "/api/generate":
            return self._generate(raw)
        if path == "/api/extend":
            return self._extend(raw)
        if path == "/api/upscale":
            return self._upscale(raw)
        if path == "/api/projects":
            return self._create_project(raw)
        if path == "/api/interpolate":
            return self._interpolate(raw)
        if path == "/api/lastframe":
            return self._lastframe(raw)
        if path == "/api/import-video":
            return self._import_video(raw)
        if path == "/api/move-project":
            return self._move_project(raw)
        if path == "/api/timeline":
            return self._timeline(raw)
        if path == "/api/upload":
            return self._upload(raw)
        if path == "/api/cancel":
            return self._cancel()
        if path == "/api/pause":
            return self._pause()
        if path == "/api/resume":
            return self._resume()
        return self._send(404, {"error": "not found"})

    def do_DELETE(self):
        if self.path.split("?", 1)[0] == "/api/library":
            return self._delete(self._qs("id"))
        return self._send(404, {"error": "not found"})

    def _generate(self, raw):
        try:
            body = json.loads(raw or b"{}")
            prompt = str(body.get("prompt", "")).strip()
            if not prompt:
                return self._send(400, {"error": "Escribe un prompt."})
            # MiniMax H3 samples on a latent patchified by 2 over a /16 VAE, so
            # width & height MUST be divisible by 32 — otherwise the sampler
            # reshape fails ("shape ... invalid for input of size ..."), which
            # crashed image→video / reference at e.g. 848 wide. Snap to /32.
            width = max(32, int(round(int(body.get("width", 848)) / 32.0)) * 32)
            height = max(32, int(round(int(body.get("height", 480)) / 32.0)) * 32)
            seconds = float(body.get("seconds", 4))
            length = frames_for_seconds(seconds)
            steps = max(4, min(40, int(body.get("steps") or 20)))
            turbo_lora = bool(body.get("turbo_lora"))
            if turbo_lora:
                steps = max(4, min(8, steps))  # Turbo LoRA is trained for 4-8 steps
            seed = int(body.get("seed") or uuid.uuid4().int % 2147483647)
            mode = body.get("mode", "t2v")
            if mode == "ref":
                ref_images = [os.path.basename(str(x)) for x in (body.get("ref_images") or []) if x]
                ref_video = os.path.basename(str(body.get("ref_video"))) if body.get("ref_video") else None
                ref_audios = [os.path.basename(str(x)) for x in (body.get("ref_audios") or []) if x]
                if not (ref_images or ref_video or ref_audios):
                    return self._send(400, {"error": "Añade al menos una referencia (imagen, video o audio)."})
                g = build_ref_graph(prompt=prompt, width=width, height=height, length=length,
                                    seed=seed, steps=steps, ref_images=ref_images, ref_video=ref_video,
                                    use_video_audio=bool(body.get("ref_video_audio", True)), ref_audios=ref_audios)
            else:
                g = build_graph(prompt=prompt, width=width, height=height, length=length,
                                seed=seed, steps=steps, first_image=body.get("first_image") or None,
                                last_image=body.get("last_image") or None)
            # Offload reduction (user toggle): keep more model resident in VRAM.
            # Helps short clips (fit VRAM); a no-op on big offload-bound jobs.
            g = apply_turbo(g, body.get("turbo", "off"))
            g = apply_turbo_lora(g, turbo_lora)
            g = apply_spectrum(g, bool(body.get("spectrum")), steps)
            g = apply_mem(g, body.get("mem_factor") or 1.0)
            g = apply_upscale(g, body.get("upscale", "off"))
            g = apply_model(g, body.get("model", "int8"))
            resp = comfy_post("/prompt", json.dumps({"prompt": g, "client_id": CLIENT_ID}).encode(),
                              "application/json")
            pid = resp.get("prompt_id")
            with LOCK:
                JOBS[pid] = {"t0": time.time(), "recorded": False,
                             "meta": {"mode": mode, "prompt": prompt,
                                      "width": width, "height": height, "seconds": seconds,
                                      "length": length, "seed": seed,
                                      "project": body.get("project") or "General",
                                      "settings": {"steps": steps, "turbo": body.get("turbo", "off"),
                                                   "turbo_lora": turbo_lora,
                                                   "spectrum": bool(body.get("spectrum")),
                                                   "mem_factor": body.get("mem_factor"),
                                                   "model": body.get("model", "int8"),
                                                   "upscale": body.get("upscale", "off"),
                                                   "aspect": body.get("aspect"),
                                                   "width": width, "height": height,
                                                   "seconds": seconds}}}
                PROGRESS[pid] = {"pct": 0.0, "stage": "Enviando…", "band": (0, 2), "eta": None}
                CURRENT_PID[0] = pid
            return self._send(200, {"prompt_id": pid, "seed": seed, "length": length})
        except urllib.error.HTTPError as e:
            return self._send(502, {"error": "ComfyUI: " + e.read().decode()[:500]})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _extend(self, raw):
        """Extend an existing video: grab its last frame, generate a continuation
        via image-to-video, then concatenate into one longer clip."""
        try:
            body = json.loads(raw or b"{}")
            src = os.path.basename(str(body.get("source", "")))
            if not src:
                return self._send(400, {"error": "missing source"})
            src_path = os.path.join(OUTPUT_DIR, "video", src)
            if not os.path.isfile(src_path):
                return self._send(404, {"error": "Video fuente no encontrado."})
            prompt = str(body.get("prompt", "")).strip() or \
                "Continue the scene naturally from the last frame, same style, motion and audio."
            seconds = float(body.get("seconds", 4))
            length = frames_for_seconds(seconds)
            steps = max(4, min(40, int(body.get("steps") or 20)))
            dims = _ffprobe(src_path, "stream=width,height")
            w = int(dims[0]) if len(dims) >= 2 else 848
            h = int(dims[1]) if len(dims) >= 2 else 480
            w = max(32, int(round(w / 32.0)) * 32)   # /32 for H3 sampler (see _generate)
            h = max(32, int(round(h / 32.0)) * 32)
            os.makedirs(INPUT_DIR, exist_ok=True)
            # Reference-to-video continuation: use the SOURCE'S TAIL (last ~3s,
            # frames + audio) as the reference video plus the last frame as a
            # reference image — the closest H3 gets to continuing the scene.
            tail_name = f"exttail_{uuid.uuid4().hex[:8]}.mp4"
            tail_path = os.path.join(INPUT_DIR, tail_name)
            subprocess.run([FFMPEG, "-y", "-sseof", "-3", "-i", src_path, "-c", "copy", tail_path],
                           capture_output=True, timeout=60)
            if not (os.path.isfile(tail_path) and os.path.getsize(tail_path) > 0):
                shutil.copyfile(src_path, tail_path)
            frame_name = f"extframe_{uuid.uuid4().hex[:8]}.png"
            frame_path = os.path.join(INPUT_DIR, frame_name)
            ref_images = []
            for off in ("-0.15", "-0.4", "-1.0"):
                subprocess.run([FFMPEG, "-y", "-sseof", off, "-i", src_path,
                                "-update", "1", "-frames:v", "1", frame_path],
                               capture_output=True, timeout=40)
                if os.path.isfile(frame_path) and os.path.getsize(frame_path) > 0:
                    ref_images = [frame_name]
                    break
            seed = uuid.uuid4().int % 2147483647
            g = build_ref_graph(prompt=prompt, width=w, height=h, length=length, seed=seed,
                                steps=steps, ref_images=ref_images, ref_video=tail_name,
                                use_video_audio=True)
            g = apply_turbo(g, body.get("turbo", "off"))
            g = apply_upscale(g, body.get("upscale", "off"))
            g = apply_model(g, body.get("model", "int8"))
            resp = comfy_post("/prompt", json.dumps({"prompt": g, "client_id": CLIENT_ID}).encode(),
                              "application/json")
            pid = resp.get("prompt_id")
            with LOCK:
                JOBS[pid] = {"t0": time.time(), "recorded": False, "extend_source": src_path,
                             "extend_concat": bool(body.get("concat", True)),
                             "meta": {"mode": "extend", "prompt": prompt, "width": w, "height": h,
                                      "seconds": seconds, "length": length, "seed": seed,
                                      "project": project_of(src)}}
                PROGRESS[pid] = {"pct": 0.0, "stage": "Extendiendo…", "band": (0, 2), "eta": None}
                CURRENT_PID[0] = pid
            return self._send(200, {"prompt_id": pid, "seed": seed, "length": length})
        except urllib.error.HTTPError as e:
            return self._send(502, {"error": "ComfyUI: " + e.read().decode()[:400]})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _upscale(self, raw):
        """Upscale an EXISTING library video with local FlashVSR (2x/4x)."""
        try:
            body = json.loads(raw or b"{}")
            src = os.path.basename(str(body.get("source", "")))
            upscale = str(body.get("upscale", "esrgan_2x"))
            engine, sc = _parse_upscale(upscale)
            if engine is None:
                return self._send(400, {"error": "Opción de escalado inválida."})
            src_path = os.path.join(OUTPUT_DIR, "video", src)
            if not os.path.isfile(src_path):
                return self._send(404, {"error": "Video no encontrado."})
            dims = _ffprobe(src_path, "stream=width,height")
            w = int(dims[0]) if len(dims) >= 2 else 0
            h = int(dims[1]) if len(dims) >= 2 else 0
            dur = _ffprobe(src_path, "format=duration", stream=False)
            try:
                seconds = float(dur[0]) if dur else 0.0
            except (ValueError, IndexError):
                seconds = 0.0
            os.makedirs(INPUT_DIR, exist_ok=True)
            vid_name = f"upsrc_{uuid.uuid4().hex[:8]}.mp4"
            shutil.copyfile(src_path, os.path.join(INPUT_DIR, vid_name))
            g = build_upscale_graph(vid_name, upscale)
            resp = comfy_post("/prompt", json.dumps({"prompt": g, "client_id": CLIENT_ID}).encode(),
                              "application/json")
            pid = resp.get("prompt_id")
            length = int(round(seconds * 24)) or 1
            with LOCK:
                JOBS[pid] = {"t0": time.time(), "recorded": False,
                             "meta": {"mode": f"upscale {engine} {sc}x", "prompt": f"Escalado {engine} {sc}x: {src}",
                                      "width": w * sc, "height": h * sc, "seconds": round(seconds, 2),
                                      "length": length, "seed": 0, "project": project_of(src)}}
                PROGRESS[pid] = {"pct": 0.0, "stage": "Escalando…", "band": (0, 2), "eta": None}
                CURRENT_PID[0] = pid
            return self._send(200, {"prompt_id": pid, "width": w * sc, "height": h * sc, "length": length})
        except urllib.error.HTTPError as e:
            return self._send(502, {"error": "ComfyUI: " + e.read().decode()[:400]})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _interpolate(self, raw):
        """Interpolate an existing video to a higher fps with RIFE (smoother
        motion, same duration). Standalone — does not touch the generation graph."""
        try:
            body = json.loads(raw or b"{}")
            src = os.path.basename(str(body.get("source", "")))
            mult = max(2, min(8, int(body.get("multiplier", 2))))
            src_path = os.path.join(OUTPUT_DIR, "video", src)
            if not os.path.isfile(src_path):
                return self._send(404, {"error": "Video no encontrado."})
            dims = _ffprobe(src_path, "stream=width,height")
            w = int(dims[0]) if len(dims) >= 2 else 0
            h = int(dims[1]) if len(dims) >= 2 else 0
            dur = _ffprobe(src_path, "format=duration", stream=False)
            try:
                seconds = float(dur[0]) if dur else 0.0
            except (ValueError, IndexError):
                seconds = 0.0
            os.makedirs(INPUT_DIR, exist_ok=True)
            vid_name = f"interp_{uuid.uuid4().hex[:8]}.mp4"
            shutil.copyfile(src_path, os.path.join(INPUT_DIR, vid_name))
            g = build_interp_graph(vid_name, mult)
            resp = comfy_post("/prompt", json.dumps({"prompt": g, "client_id": CLIENT_ID}).encode(),
                              "application/json")
            pid = resp.get("prompt_id")
            length = int(round(seconds * 24 * mult)) or 1
            with LOCK:
                JOBS[pid] = {"t0": time.time(), "recorded": False,
                             "meta": {"mode": f"rife {mult}x", "prompt": f"Fluidez RIFE {24 * mult}fps: {src}",
                                      "width": w, "height": h, "seconds": round(seconds, 2),
                                      "length": length, "seed": 0, "project": project_of(src)}}
                PROGRESS[pid] = {"pct": 0.0, "stage": "Interpolando…", "band": (0, 2), "eta": None}
                CURRENT_PID[0] = pid
            return self._send(200, {"prompt_id": pid, "width": w, "height": h,
                                    "length": length, "fps": 24 * mult})
        except urllib.error.HTTPError as e:
            return self._send(502, {"error": "ComfyUI: " + e.read().decode()[:400]})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _lastframe(self, raw):
        """Extract the LAST frame of a video as a PNG in ComfyUI's input dir and
        return its name — used as the first_frame of the next clip so a new
        segment starts exactly where the previous one ended (visual continuity)."""
        try:
            body = json.loads(raw or b"{}")
            src = os.path.basename(str(body.get("source", "")))
            src_path = os.path.join(OUTPUT_DIR, "video", src)
            if not os.path.isfile(src_path):
                return self._send(404, {"error": "Video no encontrado."})
            os.makedirs(INPUT_DIR, exist_ok=True)
            frame_name = f"lastframe_{uuid.uuid4().hex[:8]}.png"
            frame_path = os.path.join(INPUT_DIR, frame_name)
            for off in ("-0.05", "-0.2", "-0.5", "-1.0"):
                subprocess.run([FFMPEG, "-y", "-sseof", off, "-i", src_path,
                                "-update", "1", "-frames:v", "1", frame_path],
                               capture_output=True, timeout=40)
                if os.path.isfile(frame_path) and os.path.getsize(frame_path) > 0:
                    return self._send(200, {"name": frame_name})
            return self._send(500, {"error": "No pude extraer el último frame."})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _import_video(self, raw):
        """Bring an UPLOADED video (in INPUT_DIR) into the library (OUTPUT_DIR/video),
        re-encoded to browser/concat-friendly h264+aac, so it can be extended,
        upscaled or used in the creative space like any generated clip."""
        try:
            body = json.loads(raw or b"{}")
            name = os.path.basename(str(body.get("name", "")))
            src = os.path.join(INPUT_DIR, name)
            if not os.path.isfile(src):
                return self._send(404, {"error": "Archivo subido no encontrado."})
            work = os.path.join(OUTPUT_DIR, "video"); os.makedirs(work, exist_ok=True)
            out_name = f"MiniMaxStudio_import_{int(time.time())}.mp4"
            out_path = os.path.join(work, out_name)
            r = subprocess.run([FFMPEG, "-y", "-i", src, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out_path],
                               capture_output=True, text=True, timeout=900)
            if r.returncode != 0 or not os.path.isfile(out_path):
                shutil.copyfile(src, out_path)
            dims = _ffprobe(out_path, "stream=width,height")
            w = int(dims[0]) if len(dims) >= 2 else 0
            h = int(dims[1]) if len(dims) >= 2 else 0
            dur = _ffprobe(out_path, "format=duration", stream=False)
            try:
                seconds = float(dur[0]) if dur else 0.0
            except (ValueError, IndexError):
                seconds = 0.0
            length = int(round(seconds * 24)) or 1
            try:
                record_video({"mode": "import", "prompt": body.get("label") or "Video importado",
                              "width": w, "height": h, "seconds": round(seconds, 2), "length": length,
                              "seed": 0, "project": body.get("project") or "General"}, 0, out_name)
            except Exception as e:  # noqa
                print("[import] record failed:", e)
            return self._send(200, {"filename": out_name, "width": w, "height": h,
                                    "length": length, "seconds": round(seconds, 2)})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _move_project(self, raw):
        """Reassign a video to a different project (Master library still shows it)."""
        try:
            body = json.loads(raw or b"{}")
            vid = int(body.get("id"))
            proj = str(body.get("project", "")).strip()[:60]
            if not proj:
                return self._send(400, {"error": "Proyecto inválido."})
            with DB_LOCK, db() as conn:
                conn.execute("INSERT OR IGNORE INTO projects(name, created_at) VALUES(?,?)", (proj, time.time()))
                conn.execute("UPDATE videos SET project=? WHERE id=?", (proj, vid))
                conn.commit()
            return self._send(200, {"ok": True, "project": proj})
        except (TypeError, ValueError):
            return self._send(400, {"error": "id inválido"})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _timeline(self, raw):
        """Timeline editor export: stitch an ordered list of library clips into
        one MP4. Each clip may be trimmed (start/end seconds) and muted. Clips of
        different resolutions are scaled+padded to the first clip's size; audio is
        normalised to 48k stereo so a single concat filter joins them cleanly.
        Synchronous ffmpeg (CPU) — the browser shows an 'exporting' state."""
        try:
            body = json.loads(raw or b"{}")
            clips = body.get("clips") or []
            if not clips:
                return self._send(400, {"error": "Añade al menos un clip a la línea de tiempo."})
            work = os.path.join(OUTPUT_DIR, "video")
            os.makedirs(work, exist_ok=True)

            def _src(c):
                p = os.path.join(work, os.path.basename(str(c.get("name", ""))))
                if not os.path.isfile(p):
                    raise RuntimeError(f"Clip no encontrado: {c.get('name')}")
                return p

            first = _src(clips[0])
            dims = _ffprobe(first, "stream=width,height")
            W = int(dims[0]) if len(dims) >= 2 else 1152
            H = int(dims[1]) if len(dims) >= 2 else 640
            W -= W % 2; H -= H % 2  # libx264 needs even dims

            args = [FFMPEG, "-y"]
            for c in clips:
                p = _src(c)
                start = c.get("start"); end = c.get("end")
                opt = []
                try:
                    s = float(start) if start not in (None, "") else 0.0
                except (TypeError, ValueError):
                    s = 0.0
                try:
                    e = float(end) if end not in (None, "") else 0.0
                except (TypeError, ValueError):
                    e = 0.0
                if s > 0:
                    opt += ["-ss", f"{s:.3f}"]
                if e > s:
                    opt += ["-t", f"{e - s:.3f}"]
                args += opt + ["-i", p]

            fc = []
            for i, c in enumerate(clips):
                fc.append(f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps=24,setsar=1[v{i}]")
                mute = ",volume=0.0" if c.get("mute") else ""
                fc.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo{mute}[a{i}]")
            fc.append("".join(f"[v{i}][a{i}]" for i in range(len(clips)))
                      + f"concat=n={len(clips)}:v=1:a=1[v][a]")

            out_name = f"MiniMaxStudio_timeline_{int(time.time())}.mp4"
            out_path = os.path.join(work, out_name)
            args += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
                     "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                     "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out_path]
            t0 = time.time()
            r = subprocess.run(args, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0 or not os.path.isfile(out_path):
                return self._send(500, {"error": "ffmpeg: " + (r.stderr or "")[-300:]})
            dur = _ffprobe(out_path, "format=duration", stream=False)
            try:
                seconds = float(dur[0]) if dur else 0.0
            except (ValueError, IndexError):
                seconds = 0.0
            length = int(round(seconds * 24)) or 1
            title = str(body.get("title") or "").strip() or f"Línea de tiempo · {len(clips)} clips"
            try:
                record_video({"mode": "timeline", "prompt": title, "width": W, "height": H,
                              "seconds": round(seconds, 2), "length": length, "seed": 0,
                              "project": project_of(clips[0].get("name"))},
                             time.time() - t0, out_name)
            except Exception as e:  # noqa
                print("[Timeline] record failed:", e)
            return self._send(200, {"name": out_name, "width": W, "height": H,
                                    "length": length, "seconds": round(seconds, 2)})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _elapsed(self, pid):
        """Server-side elapsed seconds for a job, so a browser refresh keeps the
        real running time instead of restarting from zero."""
        with LOCK:
            job = JOBS.get(pid)
        return round(time.time() - job["t0"], 1) if job and job.get("t0") else None

    def _status(self, pid):
        if not pid:
            return self._send(400, {"error": "missing id"})
        el = self._elapsed(pid)
        if PAUSED[0]:
            # ComfyUI is suspended — don't query it (it can't answer) and don't
            # let progress advance; just report the paused state.
            with LOCK:
                prog = dict(PROGRESS.get(pid, {"pct": 0.0, "stage": "Pausado"}))
            return self._send(200, {"status": "paused", "pct": prog.get("pct", 0),
                                    "stage": "⏸ Pausado", "eta": None, "elapsed": el})
        with LOCK:
            prog = dict(PROGRESS.get(pid, {"pct": 0.0, "stage": "Enviando…"}))
        try:
            hist = comfy_get("/history/" + pid)
        except Exception:  # noqa
            # /history failed — ComfyUI might have crashed (e.g. OOM). Verify it's
            # reachable at all; if not, report it honestly instead of a stuck bar.
            try:
                comfy_get("/system_stats")
            except Exception:  # noqa
                return self._send(200, {"status": "error", "error": "ComfyUI se cayó (probablemente sin VRAM). Reinícialo y prueba con menos resolución o duración (ej. 768p solo en clips cortos; 15s a HD 1152×640 o 480p)."})
            return self._send(200, {"status": "running", "pct": prog["pct"], "stage": prog["stage"], "eta": prog.get("eta"), "elapsed": el})
        if pid not in hist:
            return self._send(200, {"status": "running", "pct": prog["pct"], "stage": prog["stage"], "eta": prog.get("eta"), "elapsed": el})
        entry = hist[pid]
        st = entry.get("status", {})
        if st.get("status_str") == "error":
            return self._send(200, {"status": "error", "error": "La generación falló en ComfyUI."})
        name = None
        for out in entry.get("outputs", {}).values():
            for arr in out.values():
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, dict) and str(item.get("filename", "")).endswith(".mp4"):
                            name = item["filename"]
        if name is None:
            return self._send(200, {"status": "running", "pct": prog["pct"], "stage": prog["stage"], "eta": prog.get("eta"), "elapsed": el})
        gen = None
        with LOCK:
            job = JOBS.get(pid)
            if job and not job["recorded"]:
                gen = time.time() - job["t0"]
                job["recorded"] = True
            PROGRESS.pop(pid, None)
        final_name = name
        if gen is not None:
            meta = dict(job["meta"])
            if job.get("extend_source") and job.get("extend_concat", True):
                # stitch source + continuation into one longer video
                try:
                    final_name, total_frames = _concat_extend(job["extend_source"], name)
                    if total_frames:
                        meta["length"] = total_frames
                        meta["seconds"] = round(total_frames / 24.0, 2)
                    print(f"[Extend] {os.path.basename(job['extend_source'])} + {name} -> {final_name}")
                except Exception as e:  # noqa
                    print("[Extend] concat failed, keeping continuation only:", e)
                    final_name = name
                job["final_name"] = final_name
            try:
                record_video(meta, gen, final_name)
            except Exception as e:  # noqa
                print("[DB] record failed:", e)
        elif job and job.get("final_name"):
            final_name = job["final_name"]
        return self._send(200, {"status": "done", "name": final_name, "gen_seconds": gen, "pct": 100})

    def _active(self):
        """Report the current in-flight job so the UI can reconnect after a
        page refresh instead of losing the 'generando' view."""
        pid = CURRENT_PID[0]
        if not pid:
            return self._send(200, {"active": False})
        if not PAUSED[0]:
            try:
                if pid in comfy_get("/history/" + pid):  # already finished
                    return self._send(200, {"active": False})
            except Exception:  # noqa
                pass
        with LOCK:
            prog = dict(PROGRESS.get(pid, {}))
            meta = (JOBS.get(pid, {}) or {}).get("meta", {})
        if not prog:
            return self._send(200, {"active": False})
        return self._send(200, {"active": True, "id": pid,
                                "pct": prog.get("pct", 0), "stage": prog.get("stage", "Generando…"),
                                "eta": prog.get("eta"), "elapsed": self._elapsed(pid),
                                "w": meta.get("width", 0), "h": meta.get("height", 0),
                                "length": meta.get("length", 0), "seed": meta.get("seed", 0)})

    def _library(self):
        # ?project=<name> filters to that project's library; omitted or
        # "__master__" returns every project's videos (Master library).
        proj = self._qs("project")
        with DB_LOCK, db() as conn:
            if proj and proj != "__master__":
                rows = conn.execute("SELECT * FROM videos WHERE project=? ORDER BY id DESC LIMIT 400",
                                    (proj,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM videos ORDER BY id DESC LIMIT 400").fetchall()
        return self._send(200, {"items": [dict(r) for r in rows]})

    def _projects(self):
        with DB_LOCK, db() as conn:
            rows = conn.execute("""SELECT p.name, p.created_at,
                                     (SELECT COUNT(*) FROM videos v WHERE v.project=p.name) AS count
                                   FROM projects p ORDER BY p.created_at""").fetchall()
        return self._send(200, {"projects": [dict(r) for r in rows]})

    def _create_project(self, raw):
        try:
            name = str(json.loads(raw or b"{}").get("name", "")).strip()[:60]
            if not name or name == "__master__":
                return self._send(400, {"error": "Nombre de proyecto inválido."})
            with DB_LOCK, db() as conn:
                conn.execute("INSERT OR IGNORE INTO projects(name, created_at) VALUES(?,?)",
                             (name, time.time()))
                conn.commit()
            return self._send(200, {"ok": True, "name": name})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _delete(self, vid):
        try:
            vid = int(vid)
        except (TypeError, ValueError):
            return self._send(400, {"error": "bad id"})
        with DB_LOCK, db() as conn:
            row = conn.execute("SELECT filename FROM videos WHERE id=?", (vid,)).fetchone()
            conn.execute("DELETE FROM videos WHERE id=?", (vid,))
            conn.commit()
        if row and row["filename"]:
            fpath = os.path.join(OUTPUT_DIR, "video", os.path.basename(row["filename"]))
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
            except OSError:
                pass
        return self._send(200, {"ok": True})

    def _video(self, name):
        if not name:
            return self._send(400, {"error": "missing name"})
        path = os.path.join(OUTPUT_DIR, "video", os.path.basename(name))
        # Fallback: pull from ComfyUI's /view into the output dir if missing.
        if not os.path.isfile(path):
            from urllib.parse import quote
            try:
                with urllib.request.urlopen(
                        f"{COMFY}/view?filename={quote(name)}&subfolder=video&type=output", timeout=120) as r:
                    data = r.read()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(data)
            except Exception as e:  # noqa
                return self._send(404, {"error": "video no encontrado: " + str(e)})
        # Serve from disk WITH HTTP Range support — Chrome requires 206/Accept-Ranges
        # to play <video>; without it playback fails with MEDIA_ERR_SRC_NOT_SUPPORTED.
        try:
            total = os.path.getsize(path)
            rng = self.headers.get("Range", "") or ""
            start, end, partial = 0, total - 1, False
            if rng.startswith("bytes="):
                try:
                    s, e = rng.split("=", 1)[1].split("-", 1)
                    start = int(s) if s else 0
                    end = int(e) if e else total - 1
                    end = min(end, total - 1)
                    partial = 0 <= start <= end
                except (ValueError, IndexError):
                    start, end, partial = 0, total - 1, False
            length = end - start + 1
            with open(path, "rb") as f:
                f.seek(start)
                chunk = f.read(length)
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Content-Disposition", f'inline; filename="{os.path.basename(name)}"')
            self.end_headers()
            self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa
            self._send(502, {"error": str(e)})

    def _input(self, name):
        """Serve an uploaded file from the ComfyUI input dir (thumbnails for
        references, so they survive a page refresh — the UI persists the stored
        basename and re-shows it via this route)."""
        if not name:
            return self._send(400, {"error": "missing name"})
        path = os.path.join(INPUT_DIR, os.path.basename(name))
        if not os.path.isfile(path):
            return self._send(404, {"error": "input no encontrado"})
        import mimetypes
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa
            self._send(502, {"error": str(e)})

    def _upload(self, raw):
        """Write an uploaded media file (image / video / audio) straight into
        ComfyUI's input dir so LoadImage / LoadVideo / LoadAudio can read it by
        name. Returns the stored basename."""
        filename = self.headers.get("X-Filename", "upload.bin")
        ext = os.path.splitext(filename)[1].lower() or ".bin"
        if len(ext) > 6 or "/" in ext or "\\" in ext:
            ext = ".bin"
        safe = f"studio_{uuid.uuid4().hex[:10]}{ext}"
        try:
            os.makedirs(INPUT_DIR, exist_ok=True)
            with open(os.path.join(INPUT_DIR, safe), "wb") as f:
                f.write(raw)
            return self._send(200, {"name": safe})
        except Exception as e:  # noqa
            return self._send(500, {"error": str(e)})

    def _cancel(self):
        # If paused, resume first (a suspended process can't be interrupted).
        if PAUSED[0]:
            _set_suspended(_find_comfy_pid(), False)
            PAUSED[0] = False
        for path, body in (("/interrupt", b""), ("/queue", json.dumps({"clear": True}).encode())):
            try:
                comfy_post(path, body, "application/json")
            except Exception:  # noqa
                pass
        with LOCK:
            PROGRESS.clear()
            CURRENT_PID[0] = None
        return self._send(200, {"ok": True})

    def _pause(self):
        pid = _find_comfy_pid()
        if not pid:
            return self._send(500, {"error": "No encontré el proceso de ComfyUI."})
        if _set_suspended(pid, True):
            PAUSED[0] = True
            return self._send(200, {"ok": True, "paused": True})
        return self._send(500, {"error": "No pude pausar el proceso."})

    def _resume(self):
        _set_suspended(_find_comfy_pid(), False)
        PAUSED[0] = False
        return self._send(200, {"ok": True, "paused": False})


def main():
    init_db()
    threading.Thread(target=_ws_loop, daemon=True).start()
    threading.Thread(target=_ease_loop, daemon=True).start()
    try:
        comfy_get("/system_stats")
    except Exception:  # noqa
        print(f"[WARN] No pude contactar ComfyUI en {COMFY}. Ábrelo antes de generar.")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"MiniMax H3 Studio -> http://127.0.0.1:{PORT}   (ComfyUI: {COMFY})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
