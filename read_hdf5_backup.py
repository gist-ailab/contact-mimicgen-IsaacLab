# read hdf5 file
import h5py
import torch
import numpy as np


def load_dataset_helper(data_group):
    data = {}
    for key in data_group:
        if isinstance(data_group[key], h5py.Group):
            data[key] = load_dataset_helper(data_group[key])
        else:
            data[key] = torch.tensor(np.array(data_group[key]))
    
    return data

# read hdf5 file
with h5py.File('datasets/annotated_dataset.hdf5', 'r') as f:
    print(f.keys())
    data = f['data']['demo_0']
    data = load_dataset_helper(data)
    print(data)
    # print(data.keys())
    # print(data['obs'].keys())
    # print(data['obs'])
    