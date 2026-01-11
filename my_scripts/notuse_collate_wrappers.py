# from fairchem.core.units.mlip_unit.mlip_unit import mt_collater_adapter

# try:
#     from torch_geometric.data import Batch
# except Exception:
#     Batch = None


# class EnsureDatasetThenMT:
#     def __init__(self, tasks, exclude_keys=None, dataset_name="rmd17"):
#         self.dataset_name = dataset_name
#         self.inner = mt_collater_adapter(tasks=tasks, exclude_keys=exclude_keys or [])

#     def _is_pyg_batch(self, obj) -> bool:
#         # True Batch object
#         if Batch is not None and isinstance(obj, Batch):
#             return True
#         # Sometimes batch-like objects are Data with batch/ptr fields
#         if hasattr(obj, "batch") and getattr(obj, "batch") is not None and hasattr(obj, "ptr"):
#             return True
#         # Any object that explicitly can unbatch itself
#         if hasattr(obj, "to_data_list"):
#             return True
#         return False

#     def __call__(self, data_list):
#         flat = []

#         if not isinstance(data_list, (list, tuple)):
#             data_list = [data_list]

#         for item in data_list:
#             # Unwrap [[data]] -> data
#             if isinstance(item, (list, tuple)) and len(item) == 1:
#                 item = item[0]

#             # Unbatch only when it is truly batch-like
#             if self._is_pyg_batch(item) and hasattr(item, "to_data_list"):
#                 flat.extend(item.to_data_list())
#                 continue

#             # Ensure dataset attr is a list of length 1 (NOT a string)
#             if not hasattr(item, "dataset"):
#                 item.dataset = [self.dataset_name]
#             else:
#                 if isinstance(item.dataset, str):
#                     item.dataset = [item.dataset]

#             flat.append(item)

#         return self.inner(flat)


# my_scripts/collate_wrappers.py

from fairchem.core.units.mlip_unit.mlip_unit import mt_collater_adapter
from fairchem.core.datasets.atomic_data import AtomicData


class EnsureDatasetThenMT:
    def __init__(self, tasks, exclude_keys=None, dataset_name="rmd17"):
        self.dataset_name = dataset_name
        self.inner = mt_collater_adapter(
            tasks=tasks,
            exclude_keys=exclude_keys or []
        )

    def __call__(self, data_list):
        # DataLoader sometimes gives a single object
        if not isinstance(data_list, (list, tuple)):
            data_list = [data_list]

        atomic_list = []

        for d in data_list:
            # unwrap [[d]]
            if isinstance(d, (list, tuple)) and len(d) == 1:
                d = d[0]

            # 🔑 Convert PyG Data → AtomicData
            ad = AtomicData.from_pyg_data(d)

            # 🔑 REQUIRED by MTCollater
            ad.dataset = self.dataset_name

            atomic_list.append(ad)

        return self.inner(atomic_list)
