from fairchem.core.datasets.samplers.max_atom_distributed_sampler import (
    MaxAtomDistributedBatchSampler,
)

class StatefulMaxAtomDistributedBatchSampler(MaxAtomDistributedBatchSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch = 0

    # common interface 1
    def state_dict(self):
        return {"epoch": self._epoch}

    def load_state_dict(self, state):
        self._epoch = int(state.get("epoch", 0))

    # common interface 2
    def get_state(self):
        return self.state_dict()

    def set_state(self, state):
        self.load_state_dict(state)

    # common interface 3
    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        # if parent has its own set_epoch, call it
        if hasattr(super(), "set_epoch"):
            try:
                super().set_epoch(epoch)
            except TypeError:
                pass
