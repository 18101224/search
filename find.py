import pickle
from glob import glob
import json
from tqdm import tqdm

def find_meta(names):
    global metas
    count = 0
    result = []
    db = {}
    for i, id in enumerate(names):
        db[id] = i

    for meta in tqdm(metas,desc='searching meta on ..'):
        data = meta['samples']
        for idx, sample in tqdm(enumerate(data),desc='searching sample on temp json'):
            if sample['uuid'] in names:
                name = sample['name'].split('/')[-1].split('.')[0]
                result.append((db[sample['uuid']],name))
                count+=1
                if count == 10:
                    result.sort()
                    return result

    return False

if __name__ == '__main__':
    with open('fastforward.pkl','rb') as f:
        names = pickle.load(f)

    paths  = glob('meta/*.json')
    metas = []
    for path in tqdm(paths,desc='loading json files'):
        with open(path,'r',encoding='utf-8') as f:
            metas.append(json.load(f))

    print(find_meta(names))