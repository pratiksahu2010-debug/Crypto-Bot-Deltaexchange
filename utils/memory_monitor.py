"""Lightweight memory monitoring for Render Free Tier (512 MB)."""
import gc
from typing import Dict, Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from utils.logger import logger
from config import settings


def get_memory_mb() -> float:
    if not HAS_PSUTIL:
        return 0.0
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def check_memory(force_gc: bool = False) -> Dict[str, Any]:
    mem = get_memory_mb()
    warning = mem > settings.memory_warning_mb
    if warning:
        logger.warning(f"Memory usage high: {mem:.1f} MB (threshold {settings.memory_warning_mb} MB)")
        if force_gc:
            gc.collect()
            mem = get_memory_mb()
            logger.info(f"After GC: {mem:.1f} MB")
    return {
        "memory_mb": round(mem, 1),
        "warning": warning,
        "threshold_mb": settings.memory_warning_mb,
    }


def safe_gc() -> None:
    gc.collect()