from vector import VanlillaDB, NNDB
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--db_dir', type=str)
    parser.add_argument('--weights', type=str) 
    parser.add_argument('--method', choices=['all''fft','mfcc','passot'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.method == 'all':
         VanlillaDB(args.db_dir,weights=f'{args.db_dir}_fft')
         VanlillaDB(args.db_dir,weights=f'{args.db_dir}_mfcc')
         NNDB(args.db_dir,weights=f'{args.db_dir}_passot')
    elif args.method == 'fft':
        VanlillaDB(args.db_dir,weights=f'{args.db_dir}_fft')
    elif args.method == 'mfcc':
        VanlillaDB(args.db_dir,weights=f'{args.db_dir}_mfcc')
    elif args.method == 'passot':
        NNDB(args.db_dir,weights=f'{args.db_dir}_passot')

