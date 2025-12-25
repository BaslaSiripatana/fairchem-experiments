from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

def collate_atomicdata(data_list, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []
    return atomicdata_list_to_batch(data_list, exclude_keys=exclude_keys)
