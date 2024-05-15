import os
import cv2
import numpy as np
from srcs.utils import imread2
from config import *

def hdr_merge(folder_L, folder_M, folder_H, folder_HDR, N, n):
    img0 = imread2(os.path.join(folder_M, "1" + IMG_END))
    h, w = img0.shape
    indices = np.ones_like(img0, dtype=np.uint8)
    difs = np.ones_like(img0, dtype=np.float64) * 255.
    for idx, folder in enumerate([folder_H, folder_M, folder_L]):
        imgs = []
        for i in range(N):
            img = imread2(os.path.join(folder, str(i + 1) + IMG_END))
            imgs.append(np.expand_dims(img, 0))
        imgs = np.concatenate(imgs, axis=0)
        max_value = np.max(imgs, axis=0)
        min_value = np.min(imgs, axis=0)
        dif = np.abs(127.5 - (min_value + max_value) / 2)
        MASK_HDR = max_value < HDR_MAX
        MASK_DIF = dif < difs
        MASK = np.multiply(MASK_HDR, MASK_DIF)
        difs[MASK] = dif[MASK]
        indices[MASK] = idx

    os.makedirs(folder_HDR, exist_ok=True)
    for i in range(N + n + 2 + 1):
        img_H = imread2(os.path.join(folder_H, str(i + 1) + IMG_END))
        img_M = imread2(os.path.join(folder_M, str(i + 1) + IMG_END))
        img_L = imread2(os.path.join(folder_L, str(i + 1) + IMG_END))
        img = np.zeros_like(img0, dtype=np.uint8)
        img[indices == 0] = img_H[indices == 0]
        img[indices == 1] = img_M[indices == 1]
        img[indices == 2] = img_L[indices == 2]
        cv2.imwrite(os.path.join(folder_HDR, str(i + 1) + IMG_END), img)
