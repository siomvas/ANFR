"""Pack the FedPathology images into the single .npz file the data module reads.

FedPathology is built from two public sets of colorectal histology patches by Kather et al.:
NCT-CRC-HE-100K-NONORM (100,000 training patches) and CRC-VAL-HE-7K (7,180 validation patches),
each with one folder per tissue class. Download both from https://zenodo.org/records/1214456 and run

    python fl_bench/colon/src/build_arrays.py --train_dir NCT-CRC-HE-100K-NONORM --val_dir CRC-VAL-HE-7K \
        --out $COLON_ROOT/nct_wsi_100k_no_norm.npz

The file holds train_images, train_labels, val_images and val_labels, with images as uint8 arrays in
channel-first order and labels one-hot over the nine classes. It needs about 16 GB of memory per array.

The partitions under fl_bench/colon/splits index the training array by position. They were drawn on
the arrays of our own build, whose order followed the directory listing of our file system. This script
sorts folders and files so that a rebuild is deterministic, which may order the images differently.
"""
import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

CLASSES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
# One-hot position of each class, as in the arrays used for the paper (TUM first, ADI last).
LABELS = {name: np.eye(len(CLASSES))[len(CLASSES) - 1 - i] for i, name in enumerate(CLASSES)}


def load_folder(parent):
    images, labels = [], []
    for class_name in sorted(d.name for d in os.scandir(parent) if d.is_dir()):
        files = sorted(f.path for f in os.scandir(os.path.join(parent, class_name)) if f.is_file() and f.name.endswith(".tif"))
        for path in tqdm(files, desc=f"{os.path.basename(parent)}/{class_name}"):
            images.append(np.transpose(np.array(Image.open(path)), (2, 0, 1)))
            labels.append(LABELS[class_name])
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train_dir", required=True, help="NCT-CRC-HE-100K-NONORM")
    parser.add_argument("--val_dir", required=True, help="CRC-VAL-HE-7K")
    parser.add_argument("--out", default="nct_wsi_100k_no_norm.npz")
    args = parser.parse_args()
    train_images, train_labels = load_folder(args.train_dir)
    val_images, val_labels = load_folder(args.val_dir)
    print("train", train_images.shape, train_labels.shape, "val", val_images.shape, val_labels.shape)
    np.savez(args.out, train_images=train_images, train_labels=train_labels, val_images=val_images, val_labels=val_labels)
