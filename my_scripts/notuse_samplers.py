import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

def single_item_batch_sampler(dataset, shuffle: bool = True, seed: int = 0, drop_last: bool = False, **kwargs):
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=g)
    else:
        sampler = SequentialSampler(dataset)

    return BatchSampler(sampler, batch_size=1, drop_last=drop_last)
