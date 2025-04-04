from vector import VanlillaDB, NNDB
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--db_dir', type=str)
    parser.add_argument('--weights', type=str) 
    parser.add_argument('--method', choices=['all','fft','mfcc','passot'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    weights = f'{args.weights}'
    if args.method == 'all':
         VanlillaDB(args.db_dir,weights=weights)
         VanlillaDB(args.db_dir,weights=weights,method='mfcc')
         NNDB(args.db_dir,weights=weights)
    elif args.method == 'fft':
        VanlillaDB(args.db_dir,weights=weights,method='fft')
    elif args.method == 'mfcc':
        print('mfcc')
        VanlillaDB(args.db_dir,weights=weights,method='mfcc')
    elif args.method == 'passot':
        NNDB(args.db_dir,weights=weights)

