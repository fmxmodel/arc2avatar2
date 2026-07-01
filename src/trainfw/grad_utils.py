"""
Arc2Avatar — Gradient Utilities (Module O / Directive 54)
===========================================================
Single shared implementations of gradient management utilities used
by every training loop (Directives 15, 21, 22, 26).
"""

import contextlib
from typing import Iterator, Optional, Union

import torch


def clip_gradients(
    params: Union[torch.Tensor, Iterator[torch.Tensor]],
    max_norm: float = 1.0,
) -> float:
    """Clip gradients to max_norm.

    Inputs:    parameters or parameter iterator, max gradient norm.
    Outputs:   total gradient norm before clipping.
    Exceptions: none.
    Side effects: modifies gradients in place.

    Uses torch.nn.utils.clip_grad_norm_ for consistency.
    """
    if isinstance(params, torch.Tensor):
        params = [params]

    total_norm = torch.nn.utils.clip_grad_norm_(list(params), max_norm)
    return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm


def zero_grad(optimizer: torch.optim.Optimizer) -> None:
    """Zero gradients with set_to_none=True for memory/performance.

    Inputs:    optimizer.
    Outputs:   None.
    Exceptions: none.
    Side effects: zeros all gradients.

    Uses set_to_none=True per Directive 54 — chosen once, everywhere.
    """
    optimizer.zero_grad(set_to_none=True)


@contextlib.contextmanager
def gradient_accumulation(enabled: bool = True) -> Iterator[None]:
    """Context manager for gradient accumulation.

    Use when effective batch size must exceed what fits in VRAM budget
    (Directive 50) in one forward pass.

    Example:
        with gradient_accumulation():
            loss.backward()
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                zero_grad(optimizer)
    """
    if enabled:
        # No special context needed — gradient accumulation is controlled
        # by when optimizer.step() is called outside this context.
        pass
    yield


@contextlib.contextmanager
def mixed_precision(enabled: bool = True) -> Iterator[torch.cuda.amp.autocast]:
    """Context manager for mixed-precision training.

    Wraps torch.autocast for consistent mixed-precision usage across
    all training loops.
    """
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def sds_gradient_injection(
    rendered_image: torch.Tensor,
    gradient: torch.Tensor,
) -> None:
    """Inject SDS gradient via explicit backward (Directive 19).

    Uses image.backward(gradient=grad) — the correct SDS formulation.
    Do NOT compute a scalar loss and call .backward() on it naively.

    Inputs:    rendered_image [3, H, W] (requires_grad=True),
               gradient [3, H, W] from SDS formula.
    Outputs:   None (backpropagates gradient into render parameters).
    Exceptions: none.
    Side effects: populates .grad fields on all tensors in the render graph.
    """
    rendered_image.backward(gradient=gradient)
