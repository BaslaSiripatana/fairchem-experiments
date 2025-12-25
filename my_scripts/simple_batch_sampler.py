from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, BatchSampler
import torch

class SimpleStatefulBatchSampler:
    def __init__(self, sampler):
        self.sampler = sampler
        self.epoch = 0
        self.start_iter = 0

    def __iter__(self):
        it = iter(self.sampler)
        for _ in range(self.start_iter):
            next(it, None)
        return it

    def __len__(self):
        return len(self.sampler)

    def set_epoch_and_start_iteration(self, epoch: int, step_start: int):
        self.epoch = epoch
        self.start_iter = step_start


def simple_batch_sampler_fn(
    dataset,
    batch_size: int,
    num_replicas: int = 1,
    rank: int = 0,
):
    # minimal: ignore num_replicas/rank but accept them
    base_sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    )
    return SimpleStatefulBatchSampler(base_sampler)

