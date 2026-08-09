# Contributing

Thank you for helping improve MiniMax H3 Local AI Studio.

Before opening a pull request:

1. Keep generated media, model weights, databases, local configuration, logs
   and machine-specific paths out of Git.
2. Explain the user-facing problem and why the change is needed.
3. Test Python with `python -m py_compile server.py launch.py setup.py`.
4. Check the inline JavaScript and include relevant ComfyUI errors for workflow
   changes.
5. Do not launch expensive generation tests without saying so.
6. Confirm acceptance of [CLA.md](CLA.md) in the pull request.

Bug reports should include GPU, VRAM, operating system, Python/Torch/CUDA
versions, output resolution, duration, selected profile and the relevant
ComfyUI terminal error. Do not include private prompts or media unless you have
permission to share them.
