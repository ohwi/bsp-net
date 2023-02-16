import argparse
import random

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# synthetic data - hollow diamond, cross, diamond
# shape 1 - hollow diamond
shape_1_outsider = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
shape_1_insider = np.array([[0.5, 0], [0, -0.5], [-0.5, 0], [0, 0.5]])

# shape 2 - cross
shape_2_outsider = np.array(
    [[1, 1.0 / 3], [1.0 / 3, 1.0 / 3], [1.0 / 3, 1],
     [-1.0 / 3, 1], [-1.0 / 3, 1.0 / 3], [-1, 1.0 / 3],
     [-1, -1.0 / 3], [-1.0 / 3, -1.0 / 3], [-1.0 / 3, -1],
     [1.0 / 3, -1], [1.0 / 3, -1.0 / 3], [1, -1.0 / 3]]
)

# shape 3 - diamond
shape_3_outsider = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shapes", type=int, default=6400)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--name", type=str, default="train")

    args = parser.parse_args()
    return args


def get_image(x1, y1, s1, x2, y2, s2, x3, y3, s3, resolution):
    img = np.zeros([resolution, resolution, 1], np.uint8)
    cv2.fillPoly(img, [(shape_1_outsider * s1 + np.array([x1, y1])).astype(np.int32)], 1)
    cv2.fillPoly(img, [(shape_1_insider * s1 + np.array([x1, y1])).astype(np.int32)], 0)
    cv2.fillPoly(img, [(shape_2_outsider * s2 + np.array([x2, y2])).astype(np.int32)], 1)
    cv2.fillPoly(img, [(shape_3_outsider * s3 + np.array([x3, y3])).astype(np.int32)], 1)
    return img


if __name__ == "__main__":
    args = parse_args()

    hdf5_file = h5py.File(f"{args.name}.hdf5", 'w')
    hdf5_file.create_dataset("pixels", [args.num_shapes, args.resolution, args.resolution, 1], np.uint8, compression=9)

    for idx in tqdm(range(args.num_shapes), total=args.num_shapes):
        while True:
            s1 = random.randint(12, 16)
            x1 = random.randint(s1 + 1, args.resolution - s1 - 2)
            y1 = random.randint(s1 + 1, args.resolution - s1 - 2)
            s2 = random.randint(12, 16)
            x2 = random.randint(s2 + 1, args.resolution - s2 - 2)
            y2 = random.randint(s2 + 1, args.resolution - s2 - 2)
            s3 = random.randint(8, 12)
            x3 = random.randint(s3 + 1, args.resolution - s3 - 2)
            y3 = random.randint(s3 + 1, args.resolution - s3 - 2)

            if x1 > x2 + max(s1, s2) and x2 > x3 + max(s2, s3):
                break

        hdf5_file["pixels"][idx] = get_image(x1, y1, s1, x2, y2, s2, x3, y3, s3, args.resolution)

    hdf5_file.close()
