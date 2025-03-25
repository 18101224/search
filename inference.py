from argparse import ArgumentParser
import torch, pickle
import numpy as np
from vector import compute_fft_descriptors

def get_args():
    args = ArgumentParser()
    args.add_argument('--path')
    args = args.parse_args()
    return args

def get_10(path, weight, paths):
    z = compute_fft_descriptors(path).reshape(1, 7)
    dist = torch.tensor(torch.sum(np.abs(weight - z), dim=1)).reshape(-1)
    _, idx = torch.sort(dist, dim=0)
    results = []
    for i in range(10):
        results.append(paths[idx[i]])
    return results

if __name__ == "__main__":
    weight = torch.load("weight.pt")
    paths = pickle.load(open("paths.pkl","rb"))
    path = get_args().path
    print(get_10(path,weight,paths))
