"""
Arc2Avatar — Unified Logging Framework (Module P)
===================================================
Directive 56: Single get_logger() entry point — no bare print() allowed.
Directive 57: Structured logs to logs/structured/<run_id>.jsonl.
Directive 58: Artifact archiving to logs/artifacts/<run_id>/.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_LOGGERS: Dict[str, logging.Logger] = {}
_RUN_ID: Optional[str] = None


def set_run_id(run_id: str) -> None:
    """Set the global run ID for logging."""
    global _RUN_ID
    _RUN_ID = run_id


def get_run_id() -> Optional[str]:
    return _RUN_ID


def get_logger(subsystem_name: str) -> logging.Logger:
    """Get a structured logger for a subsystem (Directive 56).

    This is the ONLY sanctioned logging entry point in the codebase.
    Bare print() calls are forbidden outside the CLI summary line.

    Inputs:    subsystem name string (e.g., "OPT", "DATA").
    Outputs:   configured Python logger instance.
    Exceptions: none.
    Side effects: configures logger on first call per subsystem.
    """
    if subsystem_name in _LOGGERS:
        return _LOGGERS[subsystem_name]

    logger = logging.getLogger(f"arc2avatar.{subsystem_name}")
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        f"[%(asctime)s] [{subsystem_name}] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    _LOGGERS[subsystem_name] = logger
    return logger


def log_structured(
    subsystem_name: str,
    directive: int,
    severity: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured log line (Directive 57).

    Writes one JSON object per line to logs/structured/<run_id>.jsonl.

    The `directive` field is MANDATORY and must resolve to a real directive
    number from Part 1 or Part 2.

    Inputs:    subsystem name, directive number, severity, message, optional extra dict.
    Outputs:   None.
    Exceptions: none.
    Side effects: appends JSON line to structured log file.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") +
                     f"{datetime.now().microsecond:06d}Z",
        "module": subsystem_name.upper(),
        "directive": directive,
        "severity": severity.upper(),
        "message": message,
    }
    if extra:
        record["extra"] = extra

    # Also log to console via standard logger
    logger = get_logger(subsystem_name)
    level = getattr(logging, severity.upper(), logging.INFO)
    logger.log(level, message)

    # Write to structured log file
    run_id = get_run_id()
    if run_id:
        log_dir = f"logs/structured/{run_id}"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "events.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def archive_artifact(path: str, category: str, run_id: Optional[str] = None) -> None:
    """Archive an artifact to logs/artifacts/<run_id>/ (Directive 58).

    Copies (never moves) the artifact into the archive with a manifest entry.
    Every render, mesh, loss curve, config snapshot, or checkpoint should
    be archived via this function.

    Inputs:    path to artifact file, category name (e.g., "renders", "checkpoints"),
               optional run ID (defaults to global run ID).
    Outputs:   None.
    Exceptions: none.
    Side effects: copies file to archive, updates manifest.json.
    """
    run_id = run_id or _RUN_ID
    if not run_id or not os.path.exists(path):
        return

    dest_dir = f"logs/artifacts/{run_id}/{category}"
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, os.path.basename(path))
    shutil.copy2(path, dest_path)

    # Update manifest
    manifest_path = f"logs/artifacts/{run_id}/manifest.json"
    manifest: Dict[str, list] = {"artifacts": []}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    import hashlib
    with open(path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    manifest["artifacts"].append({
        "source": path,
        "archive_path": dest_path,
        "category": category,
        "hash": content_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
