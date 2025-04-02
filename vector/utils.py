import torch, librosa, tempfile, os
import matplotlib.pyplot as plt
from pydub import AudioSegment
import pickle
import soundfile as sf


def get_statistics(tensor):
    return tensor[0], tensor[1]


def norm_features(tensor):
    mean = torch.mean(tensor,dim=0)
    std = torch.std(tensor,dim=0)
    return (tensor-mean)/std, mean, std

def norm_minmax(tensor):
    max_val,_ = torch.max(tensor,dim=0)
    min_val,_ = torch.min(tensor,dim=0)
    return (tensor-min_val)/(max_val-min_val), max_val, min_val

def check_dist(x):
    dim = x.shape[1]
    for i in range(dim):
        feature = x[:,i]
        max_val = torch.max(feature)
        min_val = torch.min(feature)
        plt.hist(feature.numpy())
        plt.title(f'feature{i}')
        plt.savefig(f'feature{i}.png')
        plt.show()
        print(f'feature{i} max:{max_val} min:{min_val}')


def convert_mp3_to_wav(mp3_path):
    """MP3 파일을 임시 WAV 파일로 변환"""
    try :
        y, sr = librosa.load(mp3_path, sr=None)
    except :
        return False
    wav_path = mp3_path.split('/')[-1][:-4] + '.wav'
    sf.write(f'wavs/{wav_path}',y,sr)


def get_dist(db,x):
    x = x.reshape(1,-1)
    dist = torch.sum(torch.abs(db-x),dim=1).reshape(-1)
    value, idx = torch.sort(dist)
    return value, idx

def load_weight(weight_path):
    with open(f'{weight_path}/paths.pkl', 'rb') as f:
        paths = pickle.load(f)
    weight = torch.load(f'{weight_path}/weight.pt')
    return paths , weight[2:], weight[0], weight[1]

def save_weight(weight_path,paths,weight,mean,std):
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    with open(f'{weight_path}/paths.pkl', 'wb') as f:
        pickle.dump(paths, f)
    mean = mean.reshape(1,-1)
    std = std.reshape(1,-1)
    weight = torch.cat((mean,std,weight),dim=0)
    torch.save(weight, f'{weight_path}/weight.pt')