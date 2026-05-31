"""Dataset loading from Hugging Face Hub and local sources.

Abstracts over ``datasets.load_dataset`` and local JSON/JSONL/Parquet loading
behind a single interface so training code is agnostic to the data source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import DatasetConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSplit:
    """Container for train / validation dataset splits.

    Attributes:
        train: The training ``Dataset``.
        validation: The validation ``Dataset`` (may be ``None``).
        num_train_samples:      Number of training examples.
        num_validation_samples: Number of validation examples (0 if none).
    """

    train: Any
    validation: Any | None
    num_train_samples: int
    num_validation_samples: int


class DatasetLoader:
    """Load and split datasets for training.

    Supports:
      - Hugging Face Hub datasets (``datasets.load_dataset``).
      - Local JSONL / JSON / Parquet / Arrow files.
      - Configurable train / validation splitting.

    Args:
        config: ``DatasetConfig`` with source and split settings.
        cache_dir: Local directory for caching downloaded datasets.
        token: HF API token for private or gated datasets.
    """

    def __init__(
        self,
        config: DatasetConfig,
        cache_dir: str | Path | None = None,
        token: str | None = None,
    ) -> None:
        self._config    = config
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._token     = token

    def load(self) -> DatasetSplit:
        """Load the dataset and create a train/validation split.

        Returns:
            A ``DatasetSplit`` with ``train`` and ``validation`` datasets.

        Raises:
            FileNotFoundError: If the dataset source cannot be resolved.
        """
        # TODO: Implement via datasets.load_dataset with the configured source.
        # TODO: Apply max_samples sub-sampling if set in config.
        # TODO: Shuffle before splitting with a fixed seed for reproducibility.
        # TODO: Log dataset sizes and column names.

        logger.info(
            "dataset_load_requested",
            name=self._config.name,
            split=self._config.split,
        )
        raise NotImplementedError("DatasetLoader.load is not yet implemented.")

    def load_local(self, path: str | Path, fmt: str = "jsonl") -> Any:
        """Load a local dataset file.

        Args:
            path: Path to the local dataset file.
            fmt:  File format — one of ``"jsonl"``, ``"json"``, ``"parquet"``, ``"arrow"``.

        Returns:
            A Hugging Face ``Dataset``.

        Raises:
            ValueError: If ``fmt`` is not a supported file format.
        """
        # TODO: Implement using datasets.load_dataset with data_files.
        _supported = {"jsonl", "json", "parquet", "arrow"}
        if fmt not in _supported:
            raise ValueError(f"Unsupported format '{fmt}'. Supported: {_supported}")
        raise NotImplementedError("DatasetLoader.load_local is not yet implemented.")
