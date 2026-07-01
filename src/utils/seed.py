"""
Arc2Avatar — Global Random Seed Propagation (Module O / Directive 55)
======================================================================
Exactly one call site for seeding: set_global_seed().
Called at ORCH startup, immediately after device init.
No other file may call random.seed, numpy.random.seed, torch.manual_seed,
or torch.cuda.manual_seed_all.
"""

import random


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility.

    Seeds, in exact order:
    1. Python's random
    2. numpy.random
    3. torch.manual_seed
    4. torch.cuda.manual_seed_all

    Inputs:    seed (int) — from resolved PipelineConfig.
    Outputs:   None.
    Exceptions: none.
    Side effects: seeds all random generators globally.

    Note: This function is called EXACTLY ONCE per run. Do not call it again.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"[SEED] Global random seed set to {seed} "
          f"(Python random → numpy → torch → torch.cuda)")
