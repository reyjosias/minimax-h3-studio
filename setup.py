#!/usr/bin/env python3
"""Interactive, machine-local setup for MiniMax H3 Local AI Studio."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "studio_config.json")
HOME = os.path.expanduser("~")

HF_BASE = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
MODELS = {
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": (20_970_379_616, HF_BASE + "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors": (20_970_379_616, HF_BASE + "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors": (27_141_342_152, HF_BASE + "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
    "vae/minimax_h3_video_vae_fp16.safetensors": (5_207_808_496, HF_BASE + "vae/minimax_h3_video_vae_fp16.safetensors"),
    "vae/minimax_h3_audio_vae_fp32.safetensors": (605_254_808, HF_BASE + "vae/minimax_h3_audio_vae_fp32.safetensors"),
    "loras/minimax_h3_turbo_4step_ckpt500.safetensors": (779_849_872, "https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ckpt500.safetensors"),
}
OPTIONAL_MODELS = {
    "vae/minimax_h3_video_vae_int8_convrot.safetensors": (3_171_670_912, "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors"),
}
NODES = {
    "ComfyUI-MiniMax-H3-Turbo": "https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo.git",
    "ComfyUI-Spectrum-MiniMax-H3": "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git",
    "ComfyUI-MiniMaxH3-FirstBlockCache": "https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache.git",
    "ComfyUI-FlashVSR_Ultra_Fast": "https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git",
    "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
    "ComfyUI-Frame-Interpolation": "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git",
}


def ask(label, default=""):
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip().strip('"')
    return os.path.abspath(os.path.expandvars(os.path.expanduser(value or default))) if value or default else ""


def yes(label, default=False):
    hint = "S/n" if default else "s/N"
    value = input(f"{label} [{hint}]: ").strip().lower()
    return default if not value else value in {"s", "si", "sí", "y", "yes"}


def first_existing(paths, marker=None):
    for path in paths:
        if path and os.path.isdir(path) and (not marker or os.path.isfile(os.path.join(path, marker))):
            return os.path.abspath(path)
    return ""


def discover_comfy():
    candidates = [os.environ.get("COMFYUI_DIR", "")]
    candidates += glob.glob(os.path.join(HOME, "ComfyUI-Installs", "*", "ComfyUI"))
    candidates += [
        os.path.join(HOME, "ComfyUI"),
        os.path.join(HOME, "ComfyUI_windows_portable", "ComfyUI"),
        os.path.join(HERE, "ComfyUI"),
    ]
    return first_existing(candidates, "main.py")


def discover_python(comfy):
    candidates = [
        os.path.join(comfy, ".venv", "Scripts", "python.exe"),
        os.path.join(os.path.dirname(comfy), "python_embeded", "python.exe"),
        sys.executable,
    ]
    return next((p for p in candidates if os.path.isfile(p)), sys.executable)


def discover_model_root(comfy):
    shared = os.path.join(HOME, "ComfyUI-Shared", "models")
    roots = [os.environ.get("COMFY_MODEL_DIR", ""), shared, os.path.join(comfy, "models")]
    wanted = os.path.join("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    for root in roots:
        if root and os.path.isfile(os.path.join(root, wanted)):
            return os.path.abspath(root)
    return first_existing(roots) or os.path.join(comfy, "models")


def download(url, destination, expected):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    part = destination + ".part"
    offset = os.path.getsize(part) if os.path.isfile(part) else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        append = offset > 0 and getattr(response, "status", 200) == 206
        if not append:
            offset = 0
        mode = "ab" if append else "wb"
        with open(part, mode) as fh:
            done = offset
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                print(f"\r  {os.path.basename(destination)}: {done / 1e9:.1f}/{expected / 1e9:.1f} GB", end="", flush=True)
    print()
    if os.path.getsize(part) != expected:
        raise RuntimeError(f"descarga incompleta: {os.path.getsize(part)} de {expected} bytes")
    os.replace(part, destination)


def install_nodes(comfy, python):
    root = os.path.join(comfy, "custom_nodes")
    os.makedirs(root, exist_ok=True)
    for name, url in NODES.items():
        target = os.path.join(root, name)
        if os.path.isdir(target):
            print(f"  ✓ {name}")
            continue
        print(f"  Instalando {name}...")
        subprocess.run(["git", "clone", "--depth", "1", url, target], check=True)
        requirements = os.path.join(target, "requirements.txt")
        if os.path.isfile(requirements):
            subprocess.run([python, "-m", "pip", "install", "-r", requirements], check=True)


def main():
    print("\nMiniMax H3 Local AI Studio — configuración inicial\n")
    comfy = discover_comfy()
    comfy = ask("Carpeta de ComfyUI", comfy)
    if not os.path.isfile(os.path.join(comfy, "main.py")):
        print("ERROR: esa carpeta no contiene main.py.")
        return 1

    python = ask("Python de ComfyUI", discover_python(comfy))
    model_root = ask("Carpeta raíz de modelos (contiene diffusion_models/)", discover_model_root(comfy))
    shared = os.path.join(HOME, "ComfyUI-Shared")
    input_default = os.path.join(shared, "input") if os.path.isdir(os.path.join(shared, "input")) else os.path.join(comfy, "input")
    output_default = os.path.join(shared, "output") if os.path.isdir(os.path.join(shared, "output")) else os.path.join(comfy, "output")
    input_dir = ask("Carpeta input", input_default)
    output_dir = ask("Carpeta output", output_default)
    pinokio_ffmpeg = os.path.join(HOME, "pinokio", "bin", "ffmpeg-env", "Library", "bin", "ffmpeg.exe")
    ffmpeg = shutil.which("ffmpeg") or (pinokio_ffmpeg if os.path.isfile(pinokio_ffmpeg) else "ffmpeg")
    ffmpeg = input(f"ffmpeg [{ffmpeg}]: ").strip().strip('"') or ffmpeg
    ffprobe_default = shutil.which("ffprobe") or os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    ffprobe = input(f"ffprobe [{ffprobe_default}]: ").strip().strip('"') or ffprobe_default

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    config = {
        "comfy_url": "http://127.0.0.1:8188",
        "studio_port": 8200,
        "comfy_dir": comfy,
        "python": python,
        "model_dir": model_root,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "async_offload": 2,
    }
    desktop_yaml = os.path.join(os.environ.get("APPDATA", ""), "Comfy Desktop", "shared_model_paths.yaml")
    if os.path.isfile(desktop_yaml):
        config["extra_model_paths"] = desktop_yaml
    with open(CONFIG, "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    print(f"\n✓ Configuración local guardada en {CONFIG}")

    missing = [(rel, size, url) for rel, (size, url) in MODELS.items() if not os.path.isfile(os.path.join(model_root, *rel.split("/")))]
    if missing:
        total = sum(size for _, size, _ in missing)
        print(f"\nFaltan {len(missing)} modelos requeridos/recomendados ({total / 1e9:.1f} GB):")
        for rel, _, _ in missing:
            print(" -", rel)
        if yes("¿Descargarlos ahora desde Hugging Face?", False):
            free = shutil.disk_usage(model_root).free
            if free < total + 5_000_000_000:
                print("ERROR: no hay espacio libre suficiente.")
                return 1
            for rel, size, url in missing:
                destination = os.path.join(model_root, *rel.split("/"))
                download(url, destination, size)
    else:
        print("✓ Modelos oficiales principales encontrados.")

    optional_missing = [(rel, size, url) for rel, (size, url) in OPTIONAL_MODELS.items()
                        if not os.path.isfile(os.path.join(model_root, *rel.split("/")))]
    if optional_missing and yes("¿Descargar también el VAE INT8 opcional para los perfiles 1/2?", False):
        for rel, size, url in optional_missing:
            download(url, os.path.join(model_root, *rel.split("/")), size)

    if yes("¿Instalar los nodos opcionales de Turbo/VSR/RIFE?", False):
        try:
            install_nodes(comfy, python)
        except (OSError, subprocess.CalledProcessError) as exc:
            print("AVISO: un nodo no pudo instalarse:", exc)
            print("Puedes completar los nodos desde ComfyUI Manager.")

    print("\nListo. Reinicia ComfyUI y abre start.bat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
