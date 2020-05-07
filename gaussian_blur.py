import argparse
import sys
import os
from os.path import join as pjoin
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("gaussian blur")
    parser.add_argument("in_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=0)
    return parser.parse_args()


def main(args):
    names = sorted(os.listdir(args.in_dir))
    names = names[:args.num]

    os.makedirs(args.out_dir, exist_ok=True)
    K = round(2 * (2 * args.sigma) + 1)
    for name in tqdm(names):
        in_fn = pjoin(args.in_dir, name)
        out_fn = pjoin(args.out_dir, name)
        im = cv2.imread(in_fn)
        im = cv2.resize(im, (args.size, args.size))
        im = cv2.GaussianBlur(im, (K, K), args.sigma)
        cv2.imwrite(out_fn, im)

    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
