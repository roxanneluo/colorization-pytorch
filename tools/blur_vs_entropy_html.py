import argparse
import os
from os.path import join as pjoin
import sys
from shutil import copyfile

import cv2
import numpy as np
from tqdm import tqdm

from utils.html import HTML


def main(args):
    names = os.listdir(args.in_dir_fmt.format(args.blur_radii[0]))
    row_args = sorted(set(n.split('_')[0] for n in names))

    data_dir = pjoin(args.out_dir, 'images')
    os.makedirs(data_dir, exist_ok=True)

    # create col args
    col_args = [(0, 'real'), (0, 'gray')]
    for r in args.blur_radii:
        sufs = [
            'blur_diff',
            'entr',
            'reg',
        ]
        col_args.extend([(r, suf) for suf in sufs])
    col_names = [f'gauss_blur-r{r}_{suf}' for r, suf in col_args]

    # copy and compute image files
    out_fmt = pjoin(data_dir, '{name}_{suf}_gauss{r:02d}.png')
    for name in tqdm(row_args):
        for r, suf in col_args:
            p_str = '1p000'
            if r == 0:
                # real or gray
                in_dir = args.sharp_dir
                in_fn = pjoin(in_dir, f'{name}_{p_str}_{suf}.png')
                out_fn = out_fmt.format(name=name, suf=suf, r=r)
                copyfile(in_fn, out_fn)
                continue

            in_dir = args.in_dir_fmt.format(r)

            if suf != 'blur_diff':
                # entropy + colorized
                in_fn = pjoin(in_dir, f'{name}_{p_str}_fake_{suf}.png')
                out_fn = out_fmt.format(name=name, suf=suf, r=r)
                copyfile(in_fn, out_fn)
                continue

            # blur_diff
            gray_fn = pjoin(in_dir, f"{name}_{p_str}_gray.png")
            sharp_gray_fn = pjoin(args.sharp_dir, f"{name}_{p_str}_gray.png")
            gray = cv2.imread(gray_fn, 0)
            sharp_gray = cv2.imread(sharp_gray_fn, 0)
            diff = gray.astype(int) - sharp_gray + 127
            diff = np.clip(diff, a_min=0, a_max=255)
            diff_fn = out_fmt.format(name=name, suf=suf, r=r)
            cv2.imwrite(diff_fn, diff)

    html = HTML(args.out_dir, 'Blur vs Entropy')
    html.add_image_table(row_args, col_args,
        lambda name, r, suf: out_fmt.format(name=name, suf=suf, r=r),
        col_names=col_names
    )
    html.save()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sharp_dir")
    parser.add_argument("in_dir_fmt")
    parser.add_argument("out_dir")
    parser.add_argument("--blur_radii", type=int, nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
