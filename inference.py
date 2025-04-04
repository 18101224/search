from vector import VanlillaDB, NNDB, hierarchical_search
from glob import glob
import shutil, os, pickle
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


    nndb = NNDB('./wavs',weights='200koneshot_passt')
    name = 'ts'
    # nndb = NNDB('./wavs', weights='20koneshot_passt')
    # spdb = VanlillaDB('./wavs', weights='20koneshot_fft')
    # result = hierarchical_search(nndb,spdb,f'lp/{name}.wav')
    result, _ = nndb.get_k_sims(f'lp/{name}.wav')
    paths = get_names(result)
    with open(f'{name}.pkl','wb') as f :
        pickle.dump(paths,f)
    print(paths)


#
# if __name__ == '__main__':
#     args = get_args()
#     if args.method in ['fft','mfcc']:
#         db = VanlillaDB('./wavs',weights=args.weights, method=args.method)
#     else:
#         db = NNDB('./wavs',weights=args.weights)
#