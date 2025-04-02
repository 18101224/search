import librosa

from .vanilla import compute_fft_descriptors
from .utils import (norm_features, convert_mp3_to_wav,
                    load_weight, save_weight, get_dist, norm_minmax)
from glob import glob
import torch, os, pickle
from tqdm import tqdm
from .MCFFvec import compute_enhanced_descriptors
from hear21passt.base import get_basic_model, get_model_passt



class VanlillaDB:
    def __init__(self, audio_dir, weights=None, method='fft'):
        self.audio_dir = audio_dir
        self.weight_path = weights
        self.func = compute_fft_descriptors if method == 'fft' else compute_enhanced_descriptors
        if weights is not None and os.path.exists(weights):
            self.paths, self.vecs, self.mean, self.std = load_weight(weights)
        else:
            self.paths, self.vecs, self.mean, self.std = self.get_features()

    def get_features(self):
        print('initializing weights')
        paths = sorted(glob(f"{self.audio_dir}/*.wav"))
        entire = len(paths)
        fail = 0
        vecs = []
        result_path = []
        for path in tqdm(paths):
            vec = self.func(path)
            if vec is False :
                fail += 1
            else:
                has_nan = vec.isnan().any()
                if has_nan:
                    fail+=1
                else:
                    vecs.append(vec)
                    result_path.append(path)
        print(f'entire : {entire} , fail : {fail}')
        vecs = torch.stack(vecs)
        vecs, mean, std = norm_minmax(vecs)
        print(mean,std)
        save_weight(self.weight_path,result_path,vecs,mean,std)
        print(f'entire : {entire} , fail : {fail}')
        return paths, vecs, mean, std

    def get_k_sims(self,x,k=10):
        x = (self.func(x) - self.std) / (self.mean - self.std)
        value, idxs = get_dist(self.vecs,x)
        paths = []
        values = []
        for i in range(k):
            paths.append(self.paths[idxs[i]])
            values.append(value[i])
        return paths, values



class NNDB:
    def __init__(self,audio_dir, weights=None):
        self.audio_dir = audio_dir
        self.model = get_basic_model(mode='embed_only')
        self.model.eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.weights = weights

        if weights is not None and os.path.exists(weights):
            self.tensor = torch.load(f'{weights}/tensor.pt',map_location=self.device)
            self.paths = pickle.load(open(f'{weights}/paths.pkl','rb'))
        else:
            if not os.path.exists(weights):
                os.mkdir(weights)
            self.tensor, self.paths = self.compute_embeddings()

    def compute_embeddings(self):
        paths = sorted(glob(f'{self.audio_dir}/*.wav'))
        results = []
        return_paths = []
        fail = 0
        for path in tqdm(paths):
            y,sr = librosa.load(path,sr=32000)
            y = torch.from_numpy(y).unsqueeze(0).to('cuda')
            try:
                with torch.no_grad():
                    embedding = self.model(y)
            except:
                fail+=1
                continue
            results.append(embedding.squeeze(0))
            return_paths.append(path)
        print(f'fail: {fail}')
        with open(self.weights+'/paths.pkl','wb') as f:
            pickle.dump(return_paths,f)
        embeddings = torch.stack(results)
        torch.save(embeddings,f'{self.weights}/tensor.pt')
        return embeddings , return_paths

    @torch.no_grad()
    def get_k_sims(self,x,k=10):
        y, sr = librosa.load(x,sr=32000,duration=5)
        x = self.model(torch.from_numpy(y).to(self.device).unsqueeze(0)).squeeze(0)
        value, idxs = get_dist(self.tensor,x)
        paths = []
        values = []
        for i in range(k):
            paths.append(self.paths[idxs[i]])
            values.append(value[i])
        return paths, values