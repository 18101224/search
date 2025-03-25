from vector import compute_fft_descriptors
from glob import glob
from argparse import ArgumentParser
import pickle, torch

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    root = args.path
    paths = sorted(glob(f"{root}/*.wav"))
    results = []
    for path in paths:
        z = compute_fft_descriptors(path)
        results.append(z)
    results = torch.stack(results)
    torch.save(results,"weight.pt")
    with open('paths.pkl','wb') as f:
        pickle.dump(paths,f)
