import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import BatchSampler, Sampler
from typing import Iterable, Iterator


class StratifiedSampler(BatchSampler):
    def __init__(self, y: torch.Tensor, sampler: Sampler[int] | Iterable[int], batch_size: int, drop_last: bool):
        BatchSampler.__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.y = y
        min_class_size = torch.tensor([1, 2, 3]).unique(return_counts=True)[1].min().item()
        self.batch_size = min_class_size if self.batch_size < min_class_size else self.batch_size
        self.n_splits = len(y) // self.batch_size

    def __iter__(self) -> Iterator:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self):
        if self.drop_last:
            return len(self.y) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.y) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
