#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Rey Josias Reinoso
"""Portable one-click launcher for Local AI Studio."""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("STUDIO_CONFIG", os.path.join(HERE, "studio_config.json"))


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def up(url):
    try:
        urllib.request.urlopen(url, timeout=3).close()
        return True
    except Exception:
        return False


def main():
    cfg = load_config()
    if not cfg:
        print("No existe studio_config.json. Ejecuta setup.bat una vez.")
        input("Pulsa Enter para cerrar...")
        return 1

    comfy = os.environ.get("COMFY_URL", cfg.get("comfy_url", "http://127.0.0.1:8188"))
    port = str(os.environ.get("PORT", cfg.get("studio_port", 8200)))
    studio = f"http://127.0.0.1:{port}/new"
    os.environ.update({
        "STUDIO_CONFIG": CONFIG_PATH,
        "COMFY_URL": comfy,
        "PORT": port,
    })

    if up(comfy + "/system_stats"):
        print("ComfyUI ya esta abierto.")
    else:
        comfy_dir = cfg.get("comfy_dir", "")
        python = cfg.get("python", sys.executable)
        main_py = os.path.join(comfy_dir, "main.py")
        if not os.path.isfile(main_py):
            print("No encuentro ComfyUI. Ejecuta setup.bat y corrige su ruta.")
            return 1
        args = [python, main_py, "--listen", "127.0.0.1", "--port", comfy.rsplit(":", 1)[-1],
                "--input-directory", cfg["input_dir"], "--output-directory", cfg["output_dir"]]
        extra = cfg.get("extra_model_paths")
        if extra and os.path.isfile(extra):
            args += ["--extra-model-paths-config", extra]
        streams = str(cfg.get("async_offload", 2))
        if streams != "0":
            args += ["--async-offload", streams]
        print("Iniciando ComfyUI...")
        flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        subprocess.Popen(args, cwd=comfy_dir, creationflags=flags)
        for _ in range(180):
            if up(comfy + "/system_stats"):
                break
            time.sleep(2)
        else:
            print("ComfyUI no respondio. Revisa su consola antes de generar.")

    threading.Timer(2.0, lambda: webbrowser.open(studio)).start()
    import server
    server.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
