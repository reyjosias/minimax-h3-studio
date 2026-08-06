#!/usr/bin/env python3
"""
One-click launcher for MiniMax H3 Studio.

Starts the ComfyUI server (headless, pointed at the shared model folder) if it
isn't already running, opens the Studio in the browser, then runs the Studio
web server. Close this window to stop the Studio; ComfyUI keeps running.
"""
import os
import subprocess
import threading
import time
import urllib.request
import webbrowser

COMFY = "http://127.0.0.1:8188"
STUDIO = "http://127.0.0.1:8199"
COMFY_DIR = r"C:\Users\Rey\ComfyUI-Installs\Rey (1)"
VENV_PY = r"C:\Users\Rey\ComfyUI-Installs\Rey (1)\ComfyUI\.venv\Scripts\python.exe"
YAML = r"C:\Users\Rey\AppData\Roaming\Comfy Desktop\shared_model_paths.yaml"
OUT = r"C:\Users\Rey\ComfyUI-Shared\output"
INP = r"C:\Users\Rey\ComfyUI-Shared\input"


def up(url):
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:  # noqa
        return False


def main():
    if up(COMFY + "/system_stats"):
        print("ComfyUI ya estaba abierto.")
    else:
        print("Iniciando ComfyUI (headless)…")
        args = [VENV_PY, "-s", "ComfyUI\\main.py",
                "--extra-model-paths-config", YAML,
                "--input-directory", INP, "--output-directory", OUT]
        flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        try:
            subprocess.Popen(args, cwd=COMFY_DIR, creationflags=flags)
        except Exception as e:  # noqa
            print("No pude iniciar ComfyUI automáticamente:", e)
            print("Abre ComfyUI a mano y vuelve a ejecutar esto.")
        for _ in range(120):
            if up(COMFY + "/system_stats"):
                break
            time.sleep(2)
        print("ComfyUI listo." if up(COMFY + "/system_stats")
              else "Aviso: ComfyUI no respondió aún; el Studio esperará a que arranque.")

    threading.Timer(2.5, lambda: webbrowser.open(STUDIO)).start()
    import server
    server.main()


if __name__ == "__main__":
    main()
