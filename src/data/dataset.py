"""Custom PyTorch Dataset wrappers for QLoRA training.

Provides ``QLoRADataset`` — a lightweight ``torch.utils.data.Dataset``
around a Hugging Face ``Dataset`` that supports on-the-fly tokenization
for debugging and evaluation workflows.
"""

from __future__ import annotations

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class QLoRADataset:
    """Thin ``torch.utils.data.Dataset`` wrapper around a HF ``Dataset``.

    For production training prefer the HF ``Trainer``-native dataset path
    (which uses Arrow-backed memory mapping).  This class is provided for
    custom training loops and unit tests.

    Args:
        hf_dataset: A pre-tokenized HF ``Dataset`` object.
    """

    def __init__(self, hf_dataset: Any) -> None:
        self._dataset = hf_dataset

        # TODO: Validate that the dataset has the required columns:
        #       input_ids, attention_mask, labels.

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        # TODO: Return len(self._dataset).
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Retrieve one tokenized example as a dict of tensors.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Dictionary with keys ``input_ids``, ``attention_mask``, ``labels``
            as PyTorch tensors.
        """
        # TODO: Implement indexing into self._dataset[idx].
        # TODO: Convert numpy arrays to torch.Tensor on access.
        raise NotImplementedError
