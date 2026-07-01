"""
Arc2Avatar — Training Framework Factory (Module O / Directives 52-53)
======================================================================
Single call sites for optimizer and scheduler construction.

No file outside factory.py may call torch.optim.Adam(...) or any other
optimizer constructor directly (Directive 52).
Similarly, build_scheduler is the single call site for LR schedules (Directive 53).
"""

from typing import Any, Dict, Iterable, List, Optional, Union

import torch


def build_optimizer(
    params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
    cfg: Any,
    lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """Construct optimizer — the ONLY call site for optimizer constructors.

    Inputs:    model parameters, optimizer config (from PipelineConfig),
               optional override learning rate.
    Outputs:   torch.optim.Optimizer instance.
    Exceptions: none.
    Side effects: none (pure construction).

    Supports multiple parameter groups with different LRs (needed for
    Stage 1 where position and color have separate learning rates).
    """
    if hasattr(cfg, "lr_position") and hasattr(cfg, "lr_color"):
        # Two-parameter-group case (Stage 1/2: position vs color)
        if isinstance(params, (list, tuple)) and len(params) >= 2:
            param_groups = [
                {"params": params[0], "lr": cfg.lr_position},
                {"params": params[1], "lr": cfg.lr_color},
            ]
        else:
            param_groups = [{"params": params, "lr": lr or cfg.lr_position}]
    else:
        lr_val = lr if lr is not None else getattr(cfg, "learning_rate", 1e-3)
        param_groups = [{"params": params, "lr": lr_val}] if not isinstance(params, list) else params

    return torch.optim.Adam(param_groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construct LR scheduler — the ONLY call site for scheduler constructors.

    Inputs:    optimizer, scheduler config (from PipelineConfig).
    Outputs:   LR scheduler instance.
    Exceptions: none.
    Side effects: none.

    Defaults to ConstantLR (no decay) unless cfg.lr_schedule names an alternative.
    """
    schedule_type = getattr(cfg, "lr_schedule", "constant")

    if schedule_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    elif schedule_type == "cosine":
        total_iters = getattr(cfg, "iterations", 1000)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
    elif schedule_type == "step":
        step_size = getattr(cfg, "lr_step_size", 500)
        gamma = getattr(cfg, "lr_gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown LR schedule type: {schedule_type}")
