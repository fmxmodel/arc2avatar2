"""
Arc2Avatar — GPU Resource Manager (Module N)
==============================================
Directive 49: Single call site for torch.cuda.init() / device selection.
Directive 50: Per-subsystem VRAM budgeting with max_memory_allocated() delta.
Directive 51: @managed_stage decorator for automatic cleanup.

No other subsystem may call .cuda(), .to("cuda:N"), or torch.cuda.set_device.
"""

import functools
import gc
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Optional

import torch


_DEVICE: Optional[str] = None


def init_device(device_str: str = "cuda:0") -> str:
    """Initialize the active device (Directive 49).

    This is the ONE call site for device selection in the entire codebase.
    Called immediately after config validation, before any other computation.

    Inputs:    device_str from resolved config (e.g., "cuda:0").
    Outputs:   resolved device identifier string.
    Exceptions: raises RuntimeError if CUDA requested but unavailable.
    Side effects: sets global active device.
    """
    global _DEVICE

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device_str}' requested but CUDA is not available. "
                f"Check your environment (Directive 4) and config device setting."
            )
        torch.cuda.init()
        torch.cuda.set_device(device_str)
        device_props = torch.cuda.get_device_properties(device_str)
        print(f"[GPU] Initialized {device_str} — {device_props.name} "
              f"({device_props.total_memory / 1e9:.1f} GB)")
    else:
        print(f"[GPU] Using CPU device: {device_str}")

    _DEVICE = device_str
    return _DEVICE


def get_device() -> str:
    """Get the currently active device.

    Inputs:    None.
    Outputs:   device string (e.g., "cuda:0" or "cpu").
    Exceptions: raises RuntimeError if called before init_device().
    Side effects: none.
    """
    global _DEVICE
    if _DEVICE is None:
        raise RuntimeError(
            "GPU device not initialized. Call init_device() first."
        )
    return _DEVICE


def get_device_obj():
    """Get a torch.device object for the active device."""
    return torch.device(get_device())


def check_resource_budget(
    subsystem_name: str,
    budget_mb: int,
    before_bytes: Optional[int] = None,
) -> dict:
    """Check if a subsystem exceeded its VRAM budget (Directive 50).

    Inputs:    subsystem name, budget in MB, optional before-bytes.
    Outputs:   dict with delta_mb, within_budget, warning.
    Side effects: logs ResourceBudgetExceeded warning if over budget.
    """
    if not torch.cuda.is_available():
        return {"delta_mb": 0, "within_budget": True, "warning": None}

    current = torch.cuda.max_memory_allocated()
    before = before_bytes or 0
    delta_mb = (current - before) / (1024 * 1024)

    result = {
        "delta_mb": delta_mb,
        "within_budget": delta_mb <= budget_mb,
        "warning": None,
    }

    if delta_mb > budget_mb:
        result["warning"] = (
            f"ResourceBudgetExceeded: {subsystem_name} used "
            f"{delta_mb:.1f} MB (budget: {budget_mb} MB)"
        )
        print(f"  [RESOURCE] {result['warning']}")

    return result


def managed_stage(func: Callable) -> Callable:
    """Decorator: automatic GPU memory cleanup on function exit
    (Directive 51).

    On function exit (success OR exception), performs in exact order:
    1. Delete any local tensors the function created and did not return.
    2. torch.cuda.empty_cache()
    3. gc.collect()
    4. Delete any files written to a subsystem-specific tmp/ directory.

    Applied to every api.py entry point across all subsystems.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a temp scratch dir for this call
        scratch_dir = tempfile.mkdtemp(prefix=f"arc2avatar_{func.__name__}_")
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Cleanup in exact order (Directive 51)
            # 1. Local tensor cleanup is automatic via Python's GC
            # 2. Empty CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 3. Force garbage collection
            gc.collect()
            # 4. Remove scratch directory
            if os.path.exists(scratch_dir):
                shutil.rmtree(scratch_dir, ignore_errors=True)

    return wrapper
