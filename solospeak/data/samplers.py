"""Class-aware batch sampler for supervised contrastive learning."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class ClassAwareBatchSampler(Sampler[list[int]]):
    """Yield batches with a fixed number of classes and examples per class."""

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        num_classes_per_batch: int = 8,
        num_samples_per_class: int = 16,
        seed: int = 42,
        samples_per_class: int | None = None,
    ) -> None:
        if samples_per_class is not None:
            num_samples_per_class = samples_per_class
            if batch_size % num_samples_per_class != 0:
                raise ValueError("batch_size must be divisible by samples_per_class")
            num_classes_per_batch = batch_size // num_samples_per_class
        if num_classes_per_batch * num_samples_per_class != batch_size:
            raise ValueError("num_classes_per_batch * num_samples_per_class must equal batch_size")
        if num_samples_per_class < 2:
            raise ValueError("num_samples_per_class must be >= 2 for SupCon positives")

        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if label < 0:
                continue
            class_to_indices[int(label)].append(idx)
        if not class_to_indices:
            raise ValueError("ClassAwareBatchSampler requires at least one non-negative class label")

        self.labels = list(labels)
        self.batch_size = batch_size
        self.num_classes_per_batch = num_classes_per_batch
        self.num_samples_per_class = num_samples_per_class
        self.seed = seed
        self.class_to_indices = dict(class_to_indices)
        self.classes = sorted(self.class_to_indices)
        self._num_batches = max(1, math.ceil(sum(len(v) for v in self.class_to_indices.values()) / batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        for _ in range(self._num_batches):
            if len(self.classes) >= self.num_classes_per_batch:
                classes = rng.sample(self.classes, self.num_classes_per_batch)
            else:
                classes = [rng.choice(self.classes) for _ in range(self.num_classes_per_batch)]
            batch: list[int] = []
            for cls in classes:
                indices = self.class_to_indices[cls]
                if len(indices) >= self.num_samples_per_class:
                    batch.extend(rng.sample(indices, self.num_samples_per_class))
                else:
                    batch.extend(rng.choices(indices, k=self.num_samples_per_class))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._num_batches
