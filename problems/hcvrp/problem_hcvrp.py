from torch.utils.data import Dataset
import torch
import os
import pickle

class HCVRP:
    NAME = 'hcvrp'
    @staticmethod
    def make_dataset(*args,**kwargs):
        return HCVRPDataset(*args,**kwargs)

def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),  # scale demand
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'capacity': torch.tensor(capacity, dtype=torch.float)
    }

class HCVRPDataset(Dataset):
    def __init__(self, filename=None, size=50, veh_num=3, num_samples=10000, offset=0, distribution=None):
        super(HCVRPDataset, self).__init__()

        # self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = []
            for i in range(data['depot'].shape[0]):
                self.data.append({
                    'depot': torch.from_numpy(data['depot'][i]).float(),
                    'loc': torch.from_numpy(data['loc'][i]).float(),
                    'demand': torch.from_numpy(data['demand'][i]).float(),
                    'capacity': torch.from_numpy(data['capacity'][i]).float(),
                    'speed': torch.from_numpy(data['speed'][i]).float()
                })
        else:
            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float(),
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    # Uniform 20 - 40, scaled by capacities
                    'capacity': (torch.FloatTensor(veh_num).uniform_(19, 40).int() + 1).float(),
                    'speed': torch.FloatTensor(veh_num).uniform_(0.5, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)  # num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data

