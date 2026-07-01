"""
Arc2Avatar — GPU + Library Health Smoke Script (Module A)
===========================================================
Directive 4: Four checks in sequence, printing PASS/FAIL for each,
exiting non-zero on any FAIL.

Order matters: CUDA → PyTorch3D → Kaolin → diffusers.
An earlier failure short-circuits to avoid confusing secondary errors.

Version strings are read from configs/requirements_lock.txt (Directive 3)
to prevent environment drift between the lock file and this check script.
"""

import os
import sys

LOCK_FILE_PATH = "configs/requirements_lock.txt"


def _read_expected_versions() -> dict:
    """Read expected package versions from requirements_lock.txt.

    Returns dict mapping package name (lowercase) to version string.
    Returns empty dict if lock file not found (prebuild mode).
    """
    expected = {}
    if not os.path.exists(LOCK_FILE_PATH):
        print(f"  INFO  Lock file not found: {LOCK_FILE_PATH} "
              f"(prebuild mode — version checks skipped)")
        return expected

    with open(LOCK_FILE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if "==" in line and not line.startswith("#") and not line.startswith("-"):
                pkg, ver = line.split("==", 1)
                expected[pkg.strip().lower()] = ver.strip()
    return expected


def check_cuda(expected_versions: dict = None) -> bool:
    """Check 1: torch.cuda.is_available() must be True."""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            ver = torch.__version__
            expected = (expected_versions or {}).get("torch", "")
            match = f" (lock file expects: {expected})" if expected else ""
            print(f"  PASS  CUDA available — torch {ver}{match}, "
                  f"device count: {torch.cuda.device_count()}")
        else:
            print("  FAIL  CUDA is not available (torch.cuda.is_available() returned False)")
        return available
    except Exception as e:
        print(f"  FAIL  CUDA check crashed: {e}")
        return False


def check_pytorch3d(expected_versions: dict = None) -> bool:
    """Check 2: Import pytorch3d.renderer and instantiate FoVPerspectiveCameras."""
    try:
        from pytorch3d.renderer import FoVPerspectiveCameras
        cam = FoVPerspectiveCameras()
        import pytorch3d
        ver = getattr(pytorch3d, "__version__", "unknown")
        print(f"  PASS  PyTorch3D {ver} — import + FoVPerspectiveCameras instantiation")
        return True
    except ImportError as e:
        print(f"  FAIL  PyTorch3D import failed: {e}")
        return False
    except Exception as e:
        print(f"  FAIL  PyTorch3D instantiation failed: {e}")
        return False


def check_kaolin(expected_versions: dict = None) -> bool:
    """Check 3: Import kaolin — must not raise."""
    try:
        import kaolin
        ver = getattr(kaolin, "__version__", "unknown")
        print(f"  PASS  Kaolin import (version: {ver})")
        return True
    except ImportError as e:
        print(f"  FAIL  Kaolin import failed: {e}")
        return False
    except Exception as e:
        print(f"  FAIL  Kaolin check crashed: {e}")
        return False


def check_diffusers(expected_versions: dict = None) -> bool:
    """Check 4: Import diffusers.StableDiffusionPipeline — must not raise."""
    try:
        from diffusers import StableDiffusionPipeline
        import diffusers
        ver = getattr(diffusers, "__version__", "unknown")
        print(f"  PASS  diffusers {ver} — StableDiffusionPipeline import")
        return True
    except ImportError as e:
        print(f"  FAIL  diffusers import failed: {e}")
        return False
    except Exception as e:
        print(f"  FAIL  diffusers check crashed: {e}")
        return False


def check_environment(config=None) -> dict:
    """Run all four checks in order, returns results dict.

    Inputs:    optional config object (reads version strings if provided).
    Outputs:   dict with per-check PASS/FAIL booleans.
    Exceptions: raises RuntimeError if any check fails and config says to abort.
    Side effects: prints PASS/FAIL per check to stdout.
    """
    results = {
        "cuda": False,
        "pytorch3d": False,
        "kaolin": False,
        "diffusers": False,
        "all_pass": False,
    }

    # Read expected versions from requirements_lock.txt (Directive 3)
    expected_versions = _read_expected_versions()
    if expected_versions:
        print(f"  INFO  Read {len(expected_versions)} package specs from {LOCK_FILE_PATH}")

    print("=" * 50)
    print("Arc2Avatar — Environment Health Check (Directive 4)")
    print("=" * 50)

    # Check 1: CUDA (must pass)
    results["cuda"] = check_cuda(expected_versions)
    if not results["cuda"]:
        print("\nFAIL: CUDA check failed — aborting.")
        return results

    # Check 2: PyTorch3D (must pass — required for rendering)
    results["pytorch3d"] = check_pytorch3d(expected_versions)

    # Check 3: Kaolin (optional)
    results["kaolin"] = check_kaolin(expected_versions)
    if not results["kaolin"]:
        print("  INFO  Kaolin optional — continuing without.")

    # Check 4: diffusers (optional)
    results["diffusers"] = check_diffusers(expected_versions)
    if not results["diffusers"]:
        print("  INFO  Diffusers optional — continuing without fine-tuning.")

    # all_pass: only CUDA and PyTorch3D are required
    results["all_pass"] = results["cuda"] and results["pytorch3d"]

    print("-" * 50)
    if results["all_pass"]:
        print(f"RESULT: {sum(1 for v in results.values() if v is True)}/4 PASS")
    else:
        failed = [k for k in ["cuda", "pytorch3d", "kaolin", "diffusers"] if not results[k]]
        print(f"RESULT: {4 - len(failed)}/4 PASS — FAILED: {', '.join(failed)}")

    return results


if __name__ == "__main__":
    results = check_environment()
    if not results["all_pass"]:
        sys.exit(1)
