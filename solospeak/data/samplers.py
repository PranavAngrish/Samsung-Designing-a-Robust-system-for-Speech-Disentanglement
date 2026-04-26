"""Batch samplers for class-aware contrastive training."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class ClassAwareBatchSampler(Sampler[list[int]]):
    """Sampler contract for SupCon batches with at least two examples per class."""

    def __init__(self, labels: Sequence[int], batch_size: int, samples_per_class: int = 2) -> None:
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

    def __iter__(self) -> Iterator[list[int]]:
        raise NotImplementedError("Implement in Phase 1")

    def __len__(self) -> int:
        raise NotImplementedError("Implement in Phase 1")
