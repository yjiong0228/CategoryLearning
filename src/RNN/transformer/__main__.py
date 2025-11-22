from .train import train, Group
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='FAMILY')
    parser.add_argument('--split_len', type=int, default=16)
    args = parser.parse_args()
    group = Group[args.group]
    train(group, args.split_len)
