import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetHDR(Dataset):
    def __init__(self, files, N):
        self.N = N
        self.files = files

    def __getitem__(self, idx):
        loader = np.load(self.files[idx], allow_pickle=True)
        # i = loader["idx"]
        I_mean = loader["I_mean"] / 255.
        # 归一化到0-1之间
        imgs = loader["imgs"] / 255.
        # 让标签跟N无关（同除以N，不会让它发生变化）
        sin_hdr = loader["sin_sum_hdr"] / (self.N * 255.)
        cos_hdr = loader["cos_sum_hdr"] / (self.N * 255.)
        return imgs, sin_hdr, cos_hdr, I_mean

    def __len__(self):
        return len(self.files)


# 组织为一批的数据
def collate_fn(batch):
    imgs = []
    s_labels = []
    c_labels = []
    I_means = []
    for img, sin_sum, cos_sum, I_mean in batch:
        imgs.append(img)
        s_labels.append(sin_sum)
        c_labels.append(cos_sum)
        I_means.append(I_mean)

    imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    I_means = torch.from_numpy(np.array(I_means)).type(torch.FloatTensor)
    s_labels = torch.from_numpy(np.array(s_labels)).type(torch.FloatTensor)
    c_labels = torch.from_numpy(np.array(c_labels)).type(torch.FloatTensor)
    return imgs, s_labels, c_labels, I_means


if __name__ == '__main__':
    pass
