"""Reproducibility utilities — global random seed management."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for Python's ``random``, NumPy, and PyTorch (CPU + CUDA).
    When ``deterministic=True`` also enables CUDA deterministic algorithms
    at the cost of some performance.

    Args:
        seed:          The integer seed value.
        deterministic: Enable CUDA deterministic mode (slower but fully
                       reproducible even across runs).
    """
    # TODO: Propagate seed to Hugging Face datasets shuffle operations.
    # TODO: Log the seed value via structlog for experiment traceability.

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # CUDA >= 10.2 supports the global flag for deterministic ops.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=False)
    except ImportError:
        pass  # PyTorch not installed; seed Python/NumPy only.
