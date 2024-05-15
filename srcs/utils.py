import os
import random
import cv2
import torch
import numpy as np
from config import gamma


def expand_dim(ns):
    ns = np.expand_dims(ns, axis=1)
    ns = np.expand_dims(ns, axis=2)
    return ns


def imread2(filename, dsize=None):
    img = cv2.imread(filename, flags=0)
    if img is None:
        raise FileNotFoundError("file not found" + filename)
    else:
        img = img.astype(np.float64)
        # gamma correct
        img = np.power(img, 1. / gamma)
        if dsize:
            img = cv2.resize(img, dsize)
        return img


def list_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    return files


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



