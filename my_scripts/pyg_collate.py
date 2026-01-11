from torch_geometric.data import Batch

def pyg_batch_collate(data_list):
    return Batch.from_data_list(data_list)
