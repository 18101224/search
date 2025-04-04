import boto3
import os
import pickle
from argparse import ArgumentParser
from tqdm import tqdm

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--pkl_path', type=str)
    parser.add_argument('--local_path', type=str)
    args = parser.parse_args()
    return args

def download(pkl, local_path):
    s3 = boto3.client('s3')

    with open(pkl, 'rb') as f:
        oneshots = pickle.load(f)

    total = len(oneshots)
    fail = 0
    if not os.path.exists(local_path):
        os.mkdir(local_path)
    for sample in tqdm(oneshots):
        name = sample['uuid']
        try :
            if os.path.exists(f'{local_path}/{name}.mp3'):
                continue
            else:
                s3.download_file('soundary', f'sample-audio/{name}.mp3', f'{local_path}/{name}.mp3')
        except:
            fail += 1

    print(f'total : {total}, fail : {fail}')

if __name__ == '__main__':
    args = get_args()
    download(args.pkl_path, args.local_path)