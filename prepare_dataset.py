from glob import glob
from argparse import ArgumentParser
import os, pickle, json, boto3
from tqdm import tqdm


def get_args():
    args = ArgumentParser()
    args.add_argument('--wav_dir', type=str, help='Directory to save wav files')
    args.add_argument('--json_dir', type=str, help='Directory to save json files')
    args.add_argument('--mapping_dir',type=str)
    args = args.parse_args()
    return args

def process_json(json_path):
    return_list = []
    with open(json_path,'r') as f:
        data = json.load(f)
    data = data['samples']
    for idx, contents in enumerate(data):
        path = contents['files'][0]['path']
        return_list.append((idx,path))
    return return_list

def download_mp3(mapping_file, wav_dir):
    global bucket_name
    with open(mapping_file,'rb') as f:
        mapping_list = pickle.load(f)
    segment_name = mapping_file.split('/')[-1].split('.')[0]
    s3 = boto3.client('s3')
    for idx, path in tqdm(mapping_list, desc='downloading wav files'):
        name = path.split('/')[-1].split('.')[0]
        key = f'sample-audio/{name}.mp3'
        output_path = os.path.join(wav_dir, f"{segment_name}__{idx}.mp3")
        s3.download_file(bucket_name, key, output_path)


if __name__ == '__main__':
    args = get_args()
    bucket_name = 'soundary'
    if not os.path.exists(args.wav_dir):
        os.mkdir(args.wav_dir)
    if not os.path.exists(args.mapping_dir):
        os.mkdir(args.mapping_dir)
    # metas = glob(f'{args.json_dir}/*.txt')
    # for meta in tqdm(metas, desc='generating mapping file'):
    #     name = meta.split('/')[-1].split('.')[0]
    #     mapping_list = process_json(meta)
    #     with open(f'{args.mapping_dir}/{name}.pkl','wb') as f:
    #         pickle.dump(mapping_list,f)

    mappings = glob(f'{args.mapping_dir}/*.pkl')
    for mapping in tqdm(mappings, desc='downloading wav files'):
        download_mp3(mapping, args.wav_dir)
