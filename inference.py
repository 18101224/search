from vector import VanlillaDB, NNDB
from glob import glob
import shutil, os
from tqdm import tqdm
from vector.utils import check_dist
from argparse import ArgumentParser

def get_names(paths):
    results = []
    for path in paths:
        name = path.split('/')[-1].split('.')[0]
        results.append(name)
    return results

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--query_dir', type=str, help='Directory containing query wav files')
    parser.add_argument('--method', type=str, choices=['fft','mfcc','passot'])
    parser.add_argument('--weights', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    db = NNDB('./wavs',weights='200k_clean_passot')

    x = 'input_wav_path'
    wav_paths = db.get_k_sims(x)
    ids = get_names(wav_paths)
    print(ids)



#
# if __name__ == '__main__':
#     args = get_args()
#     if args.method in ['fft','mfcc']:
#         db = VanlillaDB('./wavs',weights=args.weights, method=args.method)
#     else:
#         db = NNDB('./wavs',weights=args.weights)
#