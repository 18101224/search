import pickle, os
from argparse import ArgumentParser
import shutil
from tqdm import tqdm

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--pkl_path',type=str)
    parser.add_argument('--legacy_path',type=str)
    parser.add_argument('--oneshot_path',type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    with open(args.pkl_path,'rb') as f:
        oneshots = pickle.load(f)

    total = 0
    if not os.path.exists(args.oneshot_path):
        os.mkdir(args.oneshot_path)

    for sample in tqdm(oneshots):
        name = sample['uuid']
        if os.path.exists(f'{args.legacy_path}/{name}.wav'):
            shutil.copy(f'{args.legacy_path}/{name}.wav',f'{args.oneshot_path}/{name}.wav')
            total +=1

    print(f'total:{total}')
