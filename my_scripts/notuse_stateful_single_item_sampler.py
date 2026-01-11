import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class StatefulSingleItemBatchSampler(BatchSampler):
    """
    BatchSampler(batch_size=1) that fully satisfies FairChem MLIP requirements.
    """

    def __init__(self, dataset, shuffle=True, seed=0, drop_last=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last

        self.epoch = 0
        self.start_iteration = 0

        sampler = self._build_sampler()
        super().__init__(sampler, batch_size=1, drop_last=drop_last)

    def _build_sampler(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            return RandomSampler(self.dataset, generator=g)
        else:
            return SequentialSampler(self.dataset)

    def set_epoch_and_start_iteration(self, epoch: int, start_iteration: int):
        self.epoch = int(epoch)
        self.start_iteration = int(start_iteration)

        # rebuild sampler deterministically
        self.sampler = self._build_sampler()

    # Optional but harmless
    def __len__(self):
        return len(self.sampler)
        

def stateful_single_item_batch_sampler(
    dataset, shuffle=True, seed=0, drop_last=False, **kwargs
):
    return StatefulSingleItemBatchSampler(
        dataset=dataset,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
