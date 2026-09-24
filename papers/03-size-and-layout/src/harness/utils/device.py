"""Which device a run uses: a CUDA GPU, Apple Silicon's GPU (torch "mps"), or the CPU.

    --device auto   CUDA if there is one, else MPS if there is one, else the CPU
    --device cpu    the CPU; the only device on which a run is bit-identical to paper 02's
    --device mps    Apple Silicon's GPU
    --device cuda   an NVIDIA GPU

What runs where. The front-end rows are computed on the CPU (and cached), so
every device is driven by bit-identical input. The untrained arms are built
on the CPU from their seed, so every device runs the same kernels and natural
frequencies, then moved to the device, where they are simulated and their
statistics computed; a trained baseline is built on the CPU and trained on the
device. The readout projects on the device and solves its ridge on the CPU in
float64 (MPS has no float64). A GPU's floating-point results are close to the
CPU's but not bit-identical: its FFTs, matrix products and reductions sum in
other orders.
"""

from __future__ import annotations

import platform
import subprocess

import torch

DEVICES = ("auto", "cpu", "mps", "cuda")


def available(device: str) -> bool:
    if device == "cpu":
        return True
    if device.startswith("cuda"):
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return False


def resolve(device: str = "auto") -> str:
    """The device to run on: `auto` picks CUDA, else MPS, else the CPU; any other choice must exist."""
    if device == "auto":
        return "cuda" if available("cuda") else "mps" if available("mps") else "cpu"
    if device.split(":")[0] not in DEVICES:
        raise ValueError(f"unknown device '{device}': expected one of {DEVICES}")
    if not available(device):
        raise RuntimeError(f"device '{device}' is not available on this machine")
    return device


def describe(device: str) -> dict:
    """What the record says about the device a run used."""
    out = {"device": device}
    if device.startswith("cuda") and torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(torch.device(device))
    elif device == "mps":
        brand = ""
        if platform.system() == "Darwin":
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True).stdout.strip()
        out["gpu"] = f"{brand or 'Apple Silicon'} (MPS)"
    return out
