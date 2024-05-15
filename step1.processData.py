import random
import os.path
import numpy as np
from multiprocessing import Pool

from srcs.Phaser import Phaser
from srcs.hdr import hdr_merge
from srcs.utils import imread2, set_seed

from config import *


def process_data(phaser, i, j, idx):
    scene_folder = os.path.join(DIR_DATA_3d, str(i + 1))
    # 01 HDR merge
    folder_L = os.path.join(scene_folder, str(3 * j + 1))  # low exposure
    folder_M = os.path.join(scene_folder, str(3 * j + 2))  # mid exposure
    folder_H = os.path.join(scene_folder, str(3 * j + 3))  # high exposure
    scene_folder_hdr = os.path.join(DIR_DATA_hdr, str(idx))
    os.makedirs(scene_folder_hdr, exist_ok=True)
    hdr_merge(folder_L, folder_M, folder_H, scene_folder_hdr, N, n)

    # 02 12-step HDR PS [label]
    files_hdr = [os.path.join(scene_folder_hdr, str(f + 1) + IMG_END) for f in range(N + gray_num)]
    (pha_absolute_hdr_12, sin_sum_hdr_12, cos_sum_hdr_12,
     KS1_hdr_12, KS2_hdr_12, I_mean_hdr_12, B_hdr_12) = phaser.calcAbsolutePhase(files_hdr, N, n)

    # 03 3-step PS [input]
    files_3 = [os.path.join(folder_M, str(f + 1) + IMG_END) for f in [0, 4, 8]]
    imgs = np.array([imread2(file, dsize=(IMG_W, IMG_H)) for file in files_3])

    # 04 save result
    os.makedirs(DIR_DATA_Train, exist_ok=True)
    save_file_train = os.path.join(DIR_DATA_Train, "scene_" + str(idx) + ".npz")

    np.savez(save_file_train,
             idx=idx,
             folder_M=folder_M,             # origin
             folder_HDR=scene_folder_hdr,   # hdr
             # input
             imgs=imgs,
             # label
             I_mean=I_mean_hdr_12,
             sin_sum_hdr=sin_sum_hdr_12,
             cos_sum_hdr=cos_sum_hdr_12)

    print("save train file:", save_file_train)


def split_data(folder, p=0.8, seed=1):
    set_seed(seed)
    files = os.listdir(folder)
    files = [os.path.join(folder, file) for file in files]
    random.shuffle(files)
    num = len(files)
    train_num = int(num * p)

    files_train = files[0:train_num]
    files_valid = files[train_num:]
    with open(FILE_train, "w") as fw:
        for file in files_train:
            fw.write(os.path.basename(file) + "\n")
    with open(FILE_test, "w") as fw:
        for file in files_valid:
            fw.write(os.path.basename(file) + "\n")


if __name__ == '__main__':
    # 01 Process data
    scene_num = 24
    one_scene = 30
    dsize = (IMG_W, IMG_H)
    phaser = Phaser(dsize)
    po = Pool(16)
    idx = 1
    for i in range(scene_num):
        for j in range(one_scene):
            # process_data(phaser, i, j, idx) # debug
            po.apply_async(process_data, args=(phaser, i, j, idx))
            idx += 1
    po.close()
    po.join()
    print("Process is finished")

    # 02 Split dataset
    # seed = 1
    # split_data(folder=DIR_DATA_Train, seed)
