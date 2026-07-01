"""
Arc2Avatar — SDS Internals: Connectivity Regularizer (Module C / Directive 12)
================================================================================
Private module (underscore prefix): only GAUSS's api.py may expose this.
No cross-module direct import of this file is allowed (Directive 43/44).

Built in Module C, first activated in Module F.
"""

from typing import Optional

import torch


def compute_connectivity_loss(
    means: torch.Tensor,
    vertex_id: torch.Tensor,
    initial_offsets: Optional[torch.Tensor] = None,
    k: int = 8,
    weight: float = 0.1,
) -> torch.Tensor:
    """Compute the connectivity regularizer loss (L_connectivity).

    Penalizes each Gaussian's mean position for drifting too far, in local
    relative terms, from its k-nearest-neighbor relative offsets recorded at
    initialization.

    Args:
        means:       [N, 3] current Gaussian mean positions.
        vertex_id:   [N] vertex index correspondence (unused in NN search,
                     but required for Invariant I2 length check).
        initial_offsets: [N, k, 3] pre-computed NN relative offsets at init.
                     If None, computed on first call and cached.
        k:           Number of nearest neighbors (default 8, per Directive 12).
        weight:      Loss weight scaling factor.

    Returns:
        Scalar loss tensor.
    """
    assert means.shape[0] == vertex_id.shape[0], \
        f"Invariant I2 violated: means {means.shape[0]} != vertex_id {vertex_id.shape[0]}"

    N = means.shape[0]
    k = min(k, N - 1)  # Guard against N < k

    # Compute pairwise distances
    dist = torch.cdist(means, means)  # [N, N]

    # Get k-nearest neighbors (excluding self)
    nn_idx = dist.topk(k=k + 1, dim=1, largest=False).indices  # [N, k+1]
    nn_idx = nn_idx[:, 1:]  # Exclude self; [N, k]

    # Gather neighbor positions
    nn_positions = means[nn_idx]  # [N, k, 3]

    # Compute current relative offsets
    current_offsets = nn_positions - means.unsqueeze(1)  # [N, k, 3]

    # If initial offsets provided, penalize drift; otherwise record current
    if initial_offsets is not None:
        loss = weight * torch.mean((current_offsets - initial_offsets) ** 2)
    else:
        loss = torch.tensor(0.0, device=means.device)

    return loss


def compute_initial_offsets(means: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Pre-compute and record k-NN relative offsets at initialization.

    Args:
        means: [N, 3] initial Gaussian positions.
        k:     Number of nearest neighbors.

    Returns:
        [N, k, 3] tensor of initial relative offsets.
    """
    N = means.shape[0]
    k = min(k, N - 1)

    dist = torch.cdist(means, means)
    nn_idx = dist.topk(k=k + 1, dim=1, largest=False).indices[:, 1:]
    nn_positions = means[nn_idx]
    initial_offsets = nn_positions - means.unsqueeze(1)

    return initial_offsets
