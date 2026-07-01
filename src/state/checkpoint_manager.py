"""
Arc2Avatar — Checkpoint Lifecycle Manager (Module M)
======================================================
Directive 47: Five named operations — create, validate, load, replace, archive.
Directive 48: Resume execution from last valid checkpoint.

All checkpoint writes use atomic temp-file-then-rename pattern.
"""

import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import torch

from src.contracts.schemas import load_versioned, save_versioned, SCHEMA_VERSION


ARCHIVE_DIR = "checkpoints/_archive"


def save_checkpoint(state: Any, path: str, schema_version: int = SCHEMA_VERSION) -> str:
    """Create a checkpoint with atomic write (Directive 47, operation 1).

    Writes to path.tmp first, then atomically renames to path on success.
    A crash mid-write never leaves a corrupt "real" checkpoint file.

    Returns content hash (sha256 prefix).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp." + str(os.getpid())

    try:
        data = {"schema_version": schema_version, "data": state}
        torch.save(data, tmp_path)
        os.rename(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    # Compute content hash
    import hashlib
    with open(path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    return content_hash


def validate_checkpoint(path: str, expected_version: int = SCHEMA_VERSION) -> bool:
    """Validate a checkpoint file (Directive 47, operation 2).

    Checks: file exists, schema version matches, and Invariant I2
    (vertex_id length match) for GaussianState objects.

    Returns True if valid, False otherwise.
    """
    if not os.path.exists(path):
        print(f"  [CKPT] Validation FAILED — file not found: {path}")
        return False

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)

        if not isinstance(data, dict) or "schema_version" not in data:
            print(f"  [CKPT] Validation FAILED — no schema_version in {path}")
            return False

        actual_version = data["schema_version"]
        if actual_version != expected_version:
            print(f"  [CKPT] Validation FAILED — schema version mismatch: "
                  f"expected {expected_version}, got {actual_version} in {path}")
            return False

        # Invariant I2: check vertex_id length match if GaussianState
        obj = data.get("data")
        if obj is not None and hasattr(obj, "means") and hasattr(obj, "vertex_id"):
            if obj.means.shape[0] != obj.vertex_id.shape[0]:
                print(f"  [CKPT] Validation FAILED — Invariant I2 violated: "
                      f"means {obj.means.shape[0]} != vertex_id {obj.vertex_id.shape[0]}")
                return False

        return True

    except Exception as e:
        print(f"  [CKPT] Validation FAILED — error reading {path}: {e}")
        return False


def load_checkpoint(path: str, expected_version: int = SCHEMA_VERSION) -> Any:
    """Load a checkpoint with mandatory validation (Directive 47, operation 3).

    Calls validate_checkpoint internally. Refuses to return a state from
    a checkpoint that fails validation.

    Raises ValueError on validation failure (Invariant I4).
    """
    if not validate_checkpoint(path, expected_version):
        raise ValueError(
            f"Checkpoint validation failed for {path} (Invariant I4). "
            f"Cannot load corrupted or schema-mismatched checkpoint."
        )

    return load_versioned(path, expected_version)


def replace_checkpoint(old_path: str, new_state: Any,
                       schema_version: int = SCHEMA_VERSION) -> str:
    """Replace a checkpoint, archiving the old one first (Directive 47, operation 4).

    Moves old_path to checkpoints/_archive/<timestamp>_<name> before
    writing the new state to old_path. Never deletes outright.

    Returns content hash of the new checkpoint.
    """
    # Archive old file if it exists
    if os.path.exists(old_path):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{timestamp}_{os.path.basename(old_path)}"
        archive_path = os.path.join(ARCHIVE_DIR, archive_name)
        shutil.move(old_path, archive_path)
        print(f"  [CKPT] Archived: {old_path} → {archive_path}")

    # Write new checkpoint
    return save_checkpoint(new_state, old_path, schema_version)


def prune_archive(older_than_days: int = 30) -> int:
    """Permanently delete archived checkpoints older than N days
    (Directive 47, operation 5).

    This is an explicit maintenance call — never auto-invoked during
    normal pipeline execution.

    Returns number of pruned files.
    """
    if not os.path.exists(ARCHIVE_DIR):
        return 0

    cutoff = datetime.now() - timedelta(days=older_than_days)
    pruned = 0

    for filename in os.listdir(ARCHIVE_DIR):
        filepath = os.path.join(ARCHIVE_DIR, filename)
        if os.path.isfile(filepath):
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            if mtime < cutoff:
                os.remove(filepath)
                pruned += 1

    if pruned > 0:
        print(f"  [CKPT] Pruned {pruned} archived checkpoint(s) older than {older_than_days} days")

    return pruned


# ── State Machine (Directive 46) ─────────────────────────────────────────────

VALID_TRANSITIONS = {
    "Not Started": ["Initialized"],
    "Initialized": ["Running"],
    "Running":     ["Paused", "Failed", "Completed"],
    "Paused":      ["Running"],
    "Failed":      ["Recovered"],
    "Recovered":   ["Running"],
    "Completed":   [],       # Terminal
}


class InvalidStateTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""
    pass


def get_state(state_path: str) -> str:
    """Read current FSM state from disk."""
    if not os.path.exists(state_path):
        return "Not Started"

    import json
    with open(state_path, "r") as f:
        data = json.load(f)
    return data.get("state", "Not Started")


def transition(target_state: str, state_path: str = "run_state.json",
               metadata: Optional[dict] = None) -> None:
    """Transition the pipeline state machine (Directive 46).

    Persists to disk after every transition.
    Raises InvalidStateTransitionError for illegal transitions.
    """
    current = get_state(state_path)

    allowed = VALID_TRANSITIONS.get(current, [])
    if target_state not in allowed:
        raise InvalidStateTransitionError(
            f"Illegal transition: {current} → {target_state}. "
            f"Allowed from {current}: {allowed}"
        )

    import json
    state_data = {
        "state": target_state,
        "previous_state": current,
        "timestamp": datetime.now().isoformat(),
        "last_directive_completed": metadata.get("last_directive", None) if metadata else None,
    }
    if metadata:
        state_data["metadata"] = metadata

    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state_data, f, indent=2)

    print(f"  [STATE] {current} → {target_state}")


def get_last_completed_directive(state_path: str) -> Optional[int]:
    """Get the last successfully-completed directive number from state file."""
    if not os.path.exists(state_path):
        return None

    import json
    with open(state_path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    return meta.get("last_directive", None)
