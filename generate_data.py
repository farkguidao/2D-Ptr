import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import torch
import pickle
import argparse

def generate_hcvrp_data(seed,dataset_size, hcvrp_size, veh_num):
    rnd = np.random.RandomState(seed)

    loc = rnd.uniform(0, 1, size=(dataset_size, hcvrp_size + 1, 2))
    depot = loc[:, -1]
    cust = loc[:, :-1]
    d = rnd.randint(1, 10, [dataset_size, hcvrp_size + 1])
    d = d[:, :-1]  # the demand of depot is 0, which do not need to generate here

    # vehicle feature
    speed = rnd.uniform(0.5, 1, size=(dataset_size, veh_num))
    cap = rnd.randint(20, 41, size=(dataset_size, veh_num))

    data = {
        'depot': depot.astype(np.float32),
        'loc': cust.astype(np.float32),
        'demand': d.astype(np.float32),
        'capacity': cap.astype(np.float32),
        'speed': speed.astype(np.float32)
    }
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--dataset_size", type=int, default=1280, help="Size of the dataset")
    parser.add_argument("--veh_num", type=int, default=3, help="number of the vehicles")
    parser.add_argument('--graph_size', type=int, default=40,
                        help="Number of customers")

    opts = parser.parse_args()
    data_dir = 'data'
    problem = 'hcvrp'
    datadir = os.path.join(data_dir, problem)
    os.makedirs(datadir, exist_ok=True)
    seed = 24610  # the last seed used for generating HCVRP data
    # np.random.seed(seed)
    print(opts.dataset_size, opts.graph_size, opts.veh_num)
    filename = os.path.join(datadir, '{}_v{}_{}_seed{}.pkl'.format(problem, opts.veh_num, opts.graph_size, seed))

    dataset = generate_hcvrp_data(seed,opts.dataset_size, opts.graph_size, opts.veh_num)
    print({k:dataset[k][0] for k in dataset})
    save_dataset(dataset, filename)



