"""
Arc2Avatar — Standard Error Hierarchy (Module R / Directive 62)
================================================================
Every raised exception in the codebase must be an instance of one of these leaves.
No bare Exception or RuntimeError anywhere (enforced by CI lint).

Each exception carries three mandatory fields (Directive 64):
- what_failed: the specific check or step
- why: the specific condition observed, with actual values
- how_to_fix: a concrete suggested next action
"""

from typing import Any, Optional


class Arc2AvatarError(Exception):
    """Base exception for all Arc2Avatar errors.

    All custom exceptions carry three mandatory fields (Directive 64).
    """

    def __init__(
        self,
        what_failed: str,
        why: str,
        how_to_fix: str,
        message: Optional[str] = None,
    ):
        self.what_failed = what_failed
        self.why = why
        self.how_to_fix = how_to_fix
        self.message = message or f"{what_failed}: {why}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return (
            f"[{self.__class__.__name__}] {self.message}\n"
            f"  What:  {self.what_failed}\n"
            f"  Why:   {self.why}\n"
            f"  Fix:   {self.how_to_fix}"
        )

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "what_failed": self.what_failed,
            "why": self.why,
            "how_to_fix": self.how_to_fix,
        }


class ConfigurationError(Arc2AvatarError):
    """Directive 38/39 validation failures."""
    pass


class DataError(Arc2AvatarError):
    """Directive 5/6/9 missing/invalid inputs."""
    pass


class TrainingError(Arc2AvatarError):
    """Base for training-related errors."""
    pass


class DivergenceError(TrainingError):
    """Directive 23's guard tripping — optimization diverged."""
    pass


class FinetuneError(TrainingError):
    """Directive 15 fine-tuning failing to converge."""
    pass


class RenderingError(Arc2AvatarError):
    """Directive 17/18 renderer failures."""
    pass


class OptimizationError(Arc2AvatarError):
    """Directive 19/20 SDS-step failures."""
    pass


class ExportError(Arc2AvatarError):
    """Directive 28-31 export failures."""
    pass


# Central recovery policy (Directive 63)
RECOVERY_POLICY = {
    ConfigurationError: {
        "action": "abort",
        "retries": 0,
        "description": "Abort immediately — a bad config will not fix itself.",
    },
    DataError: {
        "action": "abort_with_diagnostic",
        "retries": 0,
        "description": "Abort with diagnostic naming exactly which check failed.",
    },
    DivergenceError: {
        "action": "resume_with_halved_guidance",
        "retries": 2,
        "description": "Resume from last valid checkpoint with guidance scale halved.",
    },
    FinetuneError: {
        "action": "abort",
        "retries": 0,
        "description": "Abort — fine-tuning convergence failure requires human inspection.",
    },
    RenderingError: {
        "action": "retry_once",
        "retries": 1,
        "description": "Retry once immediately (transient GPU/driver hiccups).",
    },
    OptimizationError: {
        "action": "abort",
        "retries": 0,
        "description": "Abort — SDS step failure requires investigation.",
    },
    ExportError: {
        "action": "retry_then_fallback",
        "retries": 1,
        "description": "Retry once, then export partial artifacts as Completed-with-warnings.",
    },
}


def get_recovery_policy(error: Arc2AvatarError) -> dict:
    """Look up the recovery policy for a given error instance.

    Dispatches strictly on exception TYPE — never on message-string pattern
    matching (Directive 63).
    """
    for exc_type, policy in RECOVERY_POLICY.items():
        if isinstance(error, exc_type):
            return policy

    # Default: abort
    return {
        "action": "abort",
        "retries": 0,
        "description": "Unknown error type — aborting as safe default.",
    }
