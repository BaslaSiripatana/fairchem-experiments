
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
import torch

QM9_TARGETS = [
    "mu", "alpha", "eps_HOMO", "eps_LUMO", "delta_eps",
    "R_2_Abs", "ZPVE", "U_0", "U", "H", "G", "c_v",
    "U_0_ATOM", "U_ATOM", "H_ATOM", "G_ATOM", "A", "B", "C",
]

QM9_TARGET_INDEX = {name: i for i, name in enumerate(QM9_TARGETS)}

def _attach_qm9_targets(batch):
    if not hasattr(batch, "y"):
        return batch

    for name, idx in QM9_TARGET_INDEX.items():
        if not hasattr(batch, name):
            batch[name] = batch.y[:, idx]

    # UMA energy head
    if not hasattr(batch, "energy"):
        batch.energy = batch.U_0

    return batch


def _ensure_dataset_name(batch, name="qm9"):
    if not hasattr(batch, "dataset"):
        batch.dataset = [name]
    if not hasattr(batch, "dataset_name"):
        batch.dataset_name = [name]
    return batch


def _ensure_natoms(batch):
    if not torch.is_tensor(batch.natoms):
        batch.natoms = torch.tensor([batch.pos.shape[0]], device=batch.pos.device)
    elif batch.natoms.dim() == 0:
        batch.natoms = batch.natoms.view(1)
    return batch


def _set_qm9_pbc(batch):
    device = batch.pos.device
    batch.pbc = torch.zeros(1, 3, dtype=torch.bool, device=device)
    batch.cell = torch.eye(3, device=device).unsqueeze(0)
    return batch


def collate_atomicdata(data_list, exclude_keys=None):
    data = data_list[0]

    if getattr(data, "batch", None) is not None:
        data = _ensure_natoms(data)
        data = _attach_qm9_targets(data)
        data = _ensure_dataset_name(data)
        data = _set_qm9_pbc(data)
        return data

    batch = atomicdata_list_to_batch(data_list, exclude_keys=exclude_keys)
    batch = _ensure_natoms(batch)
    batch = _attach_qm9_targets(batch)
    batch = _ensure_dataset_name(batch)
    batch = _set_qm9_pbc(batch)
    return batch
