################### for training ######################

from torch.utils.data import RandomSampler, BatchSampler

class SimpleStatefulBatchSampler:
    """
    FairChem-compatible stateful wrapper.
    Wraps a BatchSampler that yields LISTS OF INDICES.
    """
    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler
        self.epoch = 0
        self.start_iter = 0

    def __iter__(self):
        it = iter(self.batch_sampler)
        for _ in range(self.start_iter):
            next(it, None)
        return it

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch_and_start_iteration(self, epoch: int, step_start: int):
        self.epoch = epoch
        self.start_iter = step_start


def simple_batch_sampler_fn(
    dataset,
    batch_size: int,
    num_replicas: int = 1,
    rank: int = 0,
):
    # IMPORTANT:
    # - yields LIST[int]
    # - NO Data objects
    # - NO nested batching

    sampler = RandomSampler(dataset)

    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=False,
    )

    return SimpleStatefulBatchSampler(batch_sampler)

##################### for finetune ########################

import torch
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler

class SimpleStatefulBatchSamplerFinetune:
    """
    Yields List[int] batches (indices).
    Must expose set_epoch_and_start_iteration for FairChem MLIPTrainEvalUnit.
    """
    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler
        self.epoch = 0
        self.start_iter = 0

    def __iter__(self):
        it = iter(self.batch_sampler)
        for _ in range(self.start_iter):
            next(it, None)
        return it

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch_and_start_iteration(self, epoch: int, step_start: int):
        self.epoch = epoch
        self.start_iter = step_start

def make_stateful_batch_sampler(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
):
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=g)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    return SimpleStatefulBatchSamplerFinetune(batch_sampler)
