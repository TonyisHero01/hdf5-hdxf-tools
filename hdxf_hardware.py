#!/usr/bin/env python3
"""Safe, dependency-light HDXF hardware auto-tuning.

No GPU is ever required.  RAM values measure the *system*, not merely this
Python process.  The source data determines the practical number of parallel
frame blocks.  All returned limits are advisory; they are not OS quotas.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from dataclasses import dataclass

MIB = 1024 * 1024


@dataclass(frozen=True)
class SystemSnapshot:
    logical_cpus: int
    total_mib: int
    available_mib: int
    source: str

    @property
    def used_percent(self) -> float:
        return 100.0 * (1.0 - min(self.total_mib, self.available_mib) / max(1, self.total_mib))


def snapshot() -> SystemSnapshot:
    cpus = max(1, int(os.cpu_count() or 1))
    try:
        import psutil  # optional: pip install psutil
        mem = psutil.virtual_memory()
        return SystemSnapshot(cpus, int(mem.total // MIB), int(mem.available // MIB), "psutil")
    except ImportError:
        pass
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            info = MEMORYSTATUSEX()
            info.dwLength = ctypes.sizeof(info)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
                return SystemSnapshot(cpus, int(info.ullTotalPhys // MIB),
                                      int(info.ullAvailPhys // MIB), "GlobalMemoryStatusEx")
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            values = {}
            with open("/proc/meminfo", encoding="ascii") as stream:
                for line in stream:
                    k, _, v = line.partition(":")
                    values[k] = int(v.strip().split()[0]) * 1024
            total = int(values["MemTotal"] // MIB)
            available = int(values.get("MemAvailable", values["MemFree"]) // MIB)
            return SystemSnapshot(cpus, total, available, "/proc/meminfo")
        except Exception:
            pass
    if sys.platform == "darwin":
        try:
            import resource  # no universal MemAvailable; intentionally return unknown
        except Exception:
            pass
    # Unknown memory: conservative defaults, and report that information is unknown.
    return SystemSnapshot(cpus, 0, 0, "unavailable")


def reserve_mib(total_mib: int) -> int:
    """RAM set aside for Windows, the Viewer, and other applications."""
    if total_mib <= 0:
        return 1024
    return max(512, min(2048, int(total_mib * 0.08)))


def recommend_encoder(s: SystemSnapshot, block_mib: float = 0.0) -> dict[str, int]:
    cpus = s.logical_cpus
    # Workers are a ceiling; the runtime memory guard controls active jobs.
    workers = max(1, min(4, cpus // 2))
    blosc_threads = 1 if workers >= 3 else (2 if cpus >= 8 else 1)
    if not s.total_mib:
        return {"workers": workers, "active_workers": 1,
                "blosc_threads": blosc_threads, "pipeline_mib": 256,
                "reserve_mib": 1024}
    reserve = reserve_mib(s.total_mib)
    spare = max(0, s.available_mib - reserve)
    # Common 16GB / >90% case: only 1 block, even on 12+ logical processors.
    block_mib = max(1.0, float(block_mib))
    active = min(workers, max(1, int(spare / (block_mib * 3.0))))
    pipeline = min(2048 if s.total_mib >= 24576 else 1024,
                   max(128, (int(spare * 0.50) // 64) * 64))
    return {"workers": workers, "active_workers": active,
            "blosc_threads": blosc_threads, "pipeline_mib": pipeline,
            "reserve_mib": reserve}


def active_workers(s: SystemSnapshot, *, configured_workers: int, block_mib: float) -> int:
    if not s.total_mib:
        return 1  # unknown memory must not cause aggressive parallelism
    available_for_jobs = max(0, s.available_mib - reserve_mib(s.total_mib))
    return max(1, min(configured_workers, int(available_for_jobs / max(1.0, 3 * block_mib))))


def gpu_pool_limit_mib(*, free_bytes: int, total_bytes: int, manual_limit_mib: int = 0) -> int:
    """Keep approximately half the free VRAM for Windows/other GPU workloads."""
    free = max(0, int(free_bytes // MIB))
    total = max(0, int(total_bytes // MIB))
    safe_limit = max(0, min(int(free * 0.50), int(total * 0.40)))
    if safe_limit < 256:
        return 0
    if manual_limit_mib:
        return max(0, min(safe_limit, int(manual_limit_mib)))
    return safe_limit


def nvidia_gpu_overview() -> str:
    """Read optional NVIDIA hardware details without loading CUDA/CuPy into Tk."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=2, check=True,
            **({"creationflags": 0x08000000} if sys.platform == "win32" else {}),
        )
        descriptions = []
        for row in result.stdout.strip().splitlines():
            parts = [part.strip() for part in row.split(",")]
            if len(parts) >= 4:
                name, total, free, temp = parts[:4]
                descriptions.append(f"{name} · {free}/{total} MiB free · {temp}°C")
        return "; ".join(descriptions) if descriptions else "NVIDIA GPU: not reported"
    except (OSError, subprocess.SubprocessError):
        return "NVIDIA GPU: not detected (CUDA is verified at conversion start)"


def gpu_temperature(device: int = 0) -> int | None:
    """Optional NVIDIA temperature check; never a hard dependency."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "-i", str(device), "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=2, check=True,
            **({"creationflags": 0x08000000} if sys.platform == "win32" else {}),
        )
        return int(output.stdout.strip().splitlines()[0].strip())
    except (ValueError, IndexError, OSError, subprocess.SubprocessError):
        return None


class MemoryGuard:
    """Rate-limited admission control for block-encoding tasks.

    Does not claim to enforce a hard process memory cap.  At critical free RAM,
    it refuses to launch another large read instead of provoking paging/OOM.
    """
    def __init__(self, *, max_workers: int, report=print, sample_seconds: float = 2.0):
        self.max_workers = max(1, int(max_workers))
        self.report = report
        self.sample_seconds = max(0.5, sample_seconds)
        self._last_check = -1e9
        self._sample: SystemSnapshot | None = None
        self._last_active: int | None = None

    def assess(self, block_mib: float) -> tuple[int, int, SystemSnapshot]:
        now = time.monotonic()
        if self._sample is None or now - self._last_check >= self.sample_seconds:
            self._sample = snapshot()
            self._last_check = now
        s = self._sample
        active = active_workers(s, configured_workers=self.max_workers, block_mib=block_mib)
        budget = recommend_encoder(s, block_mib)["pipeline_mib"]
        if active != self._last_active:
            self.report(f"[AUTO-TUNE] RAM {s.used_percent:.1f}% used, "
                        f"available {s.available_mib} MiB; active workers {active}/{self.max_workers}")
            self._last_active = active
        return active, budget, s

    @staticmethod
    def is_critical(s: SystemSnapshot) -> bool:
        return bool(s.total_mib and s.available_mib < max(256, int(s.total_mib * 0.02)))
