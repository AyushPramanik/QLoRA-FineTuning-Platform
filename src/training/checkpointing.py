"""Checkpoint management — save, load, and promote best checkpoints.

Provides a ``CheckpointManager`` that tracks checkpoint history, enforces
a maximum retention limit, and supports atomic save operations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside each checkpoint.

    Attributes:
        global_step:   Training step at which the checkpoint was created.
        epoch:         Epoch (fractional) at save time.
        eval_loss:     Validation loss at save time (``None`` if no eval was run).
        wall_time_sec: Wall-clock training time in seconds up to this step.
        model_name:    Base model identifier for provenance tracking.
    """

    global_step: int
    epoch: float
    eval_loss: float | None
    wall_time_sec: float
    model_name: str


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    Args:
        root_dir:       Parent directory under which checkpoint subdirectories
                        are created (e.g., ``/data/checkpoints``).
        max_keep:       Maximum number of checkpoints to retain on disk.
                        Oldest checkpoints are deleted when the limit is exceeded.
        save_best_only: If ``True``, only promote a checkpoint if it achieves a
                        better validation loss than the current best.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        max_keep: int = 3,
        save_best_only: bool = False,
    ) -> None:
        self._root_dir       = Path(root_dir)
        self._max_keep       = max_keep
        self._save_best_only = save_best_only
        self._history: list[tuple[Path, CheckpointMetadata]] = []
        self._best_loss      = float("inf")

        # TODO: Scan root_dir on init to resume history from a previous run.

    def save(
        self,
        model: Any,
        tokenizer: Any,
        metadata: CheckpointMetadata,
    ) -> Path:
        """Save model and tokenizer to a new checkpoint directory.

        Args:
            model:     The PEFT model (or merged model) to save.
            tokenizer: The tokenizer to save alongside the model.
            metadata:  ``CheckpointMetadata`` to embed in the checkpoint.

        Returns:
            Path to the created checkpoint directory.
        """
        # TODO: Create a timestamped subdirectory under root_dir.
        # TODO: Call model.save_pretrained and tokenizer.save_pretrained.
        # TODO: Write metadata to checkpoint.json.
        # TODO: Enforce max_keep by deleting the oldest checkpoint.

        raise NotImplementedError("CheckpointManager.save is not yet implemented.")

    def load_latest(self) -> tuple[Path, CheckpointMetadata] | None:
        """Return the path and metadata of the most recent checkpoint.

        Returns:
            A tuple of ``(checkpoint_path, metadata)`` or ``None`` if no
            checkpoint has been saved yet.
        """
        # TODO: Implement by sorting history by global_step descending.
        raise NotImplementedError("CheckpointManager.load_latest is not yet implemented.")

    def load_best(self) -> tuple[Path, CheckpointMetadata] | None:
        """Return the checkpoint with the lowest validation loss.

        Returns:
            A tuple of ``(checkpoint_path, metadata)`` or ``None`` if no
            checkpoint with eval_loss is available.
        """
        # TODO: Filter history where eval_loss is not None, then argmin.
        raise NotImplementedError("CheckpointManager.load_best is not yet implemented.")

    def _write_metadata(self, directory: Path, metadata: CheckpointMetadata) -> None:
        """Serialise ``CheckpointMetadata`` to ``directory/checkpoint.json``."""
        with (directory / "checkpoint.json").open("w") as f:
            json.dump(asdict(metadata), f, indent=2)

    def _read_metadata(self, directory: Path) -> CheckpointMetadata:
        """Deserialise ``CheckpointMetadata`` from ``directory/checkpoint.json``."""
        with (directory / "checkpoint.json").open() as f:
            data = json.load(f)
        return CheckpointMetadata(**data)
