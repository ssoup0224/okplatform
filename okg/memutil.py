"""RAM / VRAM hygiene for a ~23 GB machine. [D39]

* free_memory(): gc.collect() + torch.cuda.empty_cache()/ipc_collect() [R52] + glibc malloc_trim(0),
  which returns freed heap pages to the OS so the notebook's resident memory actually drops [R78].
* mem_report(): process RSS and system available memory from /proc (Linux), plus CUDA allocated /
  reserved per device.
* check_ram(min_gb): warns before a stage starts if available RAM is low.
* hardware_profile(cfg): picks settings that fit the detected GPU (see config.apply_hardware_profile).
"""
from __future__ import annotations

import ctypes
import gc
import os


def _meminfo_gb(field: str) -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        return None
    return None


def rss_gb() -> float | None:
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        return None
    return None


def free_memory(verbose: bool = False) -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)                 # [R78] glibc only
    except (OSError, AttributeError):
        pass
    if verbose:
        print(mem_report())


def mem_report() -> str:
    parts = []
    r, a, t = rss_gb(), _meminfo_gb("MemAvailable"), _meminfo_gb("MemTotal")
    if r is not None:
        parts.append(f"process RSS {r:.1f} GB")
    if a is not None and t is not None:
        parts.append(f"system available {a:.1f}/{t:.1f} GB")
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            parts.append(f"cuda:{i} allocated {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB, "
                         f"reserved {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB")
    except Exception:  # noqa: BLE001
        pass
    return " | ".join(parts)


def check_ram(min_gb: float, stage: str = "") -> None:
    a = _meminfo_gb("MemAvailable")
    if a is not None and a < min_gb:
        print(f"WARNING before {stage}: only {a:.1f} GB RAM available (< {min_gb} GB). Restart the kernel "
              f"(finished stages are skipped on re-run) or close other programs.")


def gpu_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1024 ** 3
    except Exception:  # noqa: BLE001
        pass
    return 0.0
