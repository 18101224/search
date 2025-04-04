import os
import librosa
import soundfile as sf
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob

def get_args():
    args = ArgumentParser()
    args.add_argument('--mp3_dir')
    args.add_argument('--wav_dir')
    args = args.parse_args()
    return args

def cvt_mp3_to_wav(mp3,wav):
    try :
        y, sr = librosa.load(mp3)
        sf.write(wav, y, sr)
        return True
    except:
        return False

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.wav_dir):
        os.mkdir(args.wav_dir)
    mp3s = glob(f'{args.mp3_dir}/*.mp3')
    total = 0
    fail = 0
    for mp3 in tqdm(mp3s, desc='converting mp3 to wav'):
        name = mp3.split('/')[-1].split('.')[0]
        wav = f'{args.wav_dir}/{name}.wav'
        if os.path.exists(wav):
            total+=1
            continue
        else:
            r = cvt_mp3_to_wav(mp3,wav)
        total += 1
        if not r :
            fail +=1
    print(f'total : {total}, fail : {fail}')

