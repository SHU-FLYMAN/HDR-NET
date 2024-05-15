import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import savemat
from matplotlib import pyplot as plt

from srcs.model import *
from srcs.Phaser import Phaser

from config import *


def calc_pha_cnn(net, imgs, KS1, KS2, device):
    imgs = torch.from_numpy(imgs / 255.).type(torch.FloatTensor).to(device)
    # add a dim of batch_size
    imgs.unsqueeze_(0)  # 增加维度
    if FLAG_FPP:
        cnn1_out, cnn2_out = net(imgs)
    else:
        cnn2_out = net(imgs)
    s, c = torch.split(cnn2_out, 1, dim=1)
    s = np.squeeze(s.cpu().numpy())
    c = np.squeeze(c.cpu().numpy())
    pha_wrapped = np.arctan2(s, c)
    B = np.sqrt((s ** 2) + (c ** 2)) * 144 * 2
    # 包裹相位
    #pha_wrapped = np.squeeze(torch.atan2(s, c).cpu().numpys())
    e = -0.0000000001
    pha_low_mask = pha_wrapped < e
    pha_wrapped = pha_wrapped + pha_low_mask * 2. * np.pi

    mask1 = (pha_wrapped <= np.pi / 2)
    mask2 = np.multiply(np.pi / 2 < pha_wrapped, (pha_wrapped <= 3 / 2 * np.pi))
    mask3 = pha_wrapped > 3 / 2 * np.pi
    pha = np.multiply(pha_wrapped + 2 * np.pi * KS2, mask1) \
        + np.multiply(pha_wrapped + 2 * np.pi * KS1, mask2) \
        + np.multiply(pha_wrapped + 2 * np.pi * KS2 - 2 * np.pi, mask3)
    # 归一化到0 - 2pi之间
    pha_absolute = (pha / ((2 ** n) * (2 * np.pi)))
    return pha_absolute, B


def calc_error(pha_g, pha_i, dt):
    d = np.absolute(pha_g - pha_i)
    d[d > dt] = 0
    num = np.count_nonzero(d)
    return np.sum(d) / num


def test(test_file, model_file, device, phaser):
    net = torch.load(model_file, map_location=device)
    # set to eval model
    net = net.to(device).eval()

    os.makedirs(DIR_OUTPUT_Phase, exist_ok=True)
    print("save folder:", DIR_OUTPUT_Phase)

    dt = 0.002
    d_3s, d_4s, d_6s, d_12s, d_cnns = [], [], [], [], []
    with torch.no_grad():
        with open(test_file, "r") as fr:
            for line in tqdm(fr.readlines(), "start test..."):
                file = os.path.join(DIR_DATA_Train, line.strip("\n"))
                print(file)
                loader = np.load(file, allow_pickle=True)

                # --------------- LDR part  --------------- #
                folder_M = loader["folder_M"].tolist()
                files_gray = [os.path.join(folder_M, str(i + 1) + IMG_END) for i in range(N, N + gray_num)]

                # 01 3-step
                files_3 = [os.path.join(folder_M, str(i + 1) + IMG_END) for i in [0, 4, 8]]
                (pha_absolute_3, sin_sum_3, cos_sum_3,
                 KS1_3, KS2_3, I_mean_3, B_3) = phaser.calcAbsolutePhase(files_3 + files_gray, 3, n)
                pha_absolute_3 = np.multiply(pha_absolute_3, B_3 > B_min)

                # 02 3-step CNN
                pha_cnn, B_CNN = calc_pha_cnn(net, np.array(loader["imgs"]), KS1_3, KS2_3, device)
                pha_cnn = np.multiply(pha_cnn, B_CNN > B_min)

                files_4 = [os.path.join(folder_M, str(i + 1) + IMG_END) for i in [0, 3, 6, 9]]
                (pha_absolute_4, sin_sum_4, cos_sum_4,
                 KS1_4, KS2_4, I_mean_4, B_4) = phaser.calcAbsolutePhase(files_4 + files_gray, 4, n)
                pha_absolute_4 = np.multiply(pha_absolute_4, B_4 > B_min)

                # 03 6-step
                files_6 = [os.path.join(folder_M, str(i + 1) + IMG_END) for i in [0, 2, 4, 6, 8, 10]]
                (pha_absolute_6, sin_sum_6, cos_sum_6,
                 KS1_6, KS2_6, I_mean_6, B_6) = phaser.calcAbsolutePhase(files_6 + files_gray, 6, n)
                pha_absolute_6 = np.multiply(pha_absolute_6, B_6 > B_min)

                # 04 12-step
                files_12 = [os.path.join(folder_M, str(i + 1) + IMG_END) for i in range(N)]
                (pha_absolute_12, sin_sum_12, cos_sum_12,
                 KS1_12, KS2_12, I_mean_12, B_12) = phaser.calcAbsolutePhase(files_12 + files_gray, N, n)
                pha_absolute_12 = np.multiply(pha_absolute_12, B_12 > B_min)

                # --------------- HDR part  --------------- #
                folder_HDR = loader["folder_HDR"].tolist()
                files_gray_hdr = [os.path.join(folder_HDR, str(i + 1) + IMG_END) for i in range(N, N + gray_num)]

                # 05 3-step hdr
                files_3_hdr = [os.path.join(folder_HDR, str(i + 1) + IMG_END) for i in [0, 4, 8]]
                (pha_absolute_3_hdr, sin_sum_3_hdr, cos_sum_3_hdr,
                 KS1_3_hdr, KS2_3_hdr, I_mean_3_hdr, B_3_hdr) = phaser.calcAbsolutePhase(files_3_hdr + files_gray_hdr, 3, n)
                pha_absolute_3_hdr = np.multiply(pha_absolute_3_hdr, B_3_hdr > B_min)

                files_4_hdr = [os.path.join(folder_HDR, str(i + 1) + IMG_END) for i in [0, 3, 6, 9]]
                (pha_absolute_4_hdr, sin_sum_4_hdr, cos_sum_4_hdr,
                 KS1_4_hdr, KS2_4_hdr, I_mean_4_hdr, B_4_hdr) = phaser.calcAbsolutePhase(files_4_hdr + files_gray_hdr, 4, n)
                pha_absolute_4_hdr = np.multiply(pha_absolute_4_hdr, B_4_hdr > B_min)

                # 06 6-step hdr
                files_6_hdr = [os.path.join(folder_HDR, str(i + 1) + IMG_END) for i in [0, 2, 4, 6, 8, 10]]
                (pha_absolute_6_hdr, sin_sum_6_hdr, cos_sum_6_hdr,
                 KS1_6_hdr, KS2_6_hdr, I_mean_6_hdr, B_6_hdr) = phaser.calcAbsolutePhase(files_6_hdr + files_gray_hdr, 6, n)
                pha_absolute_6_hdr = np.multiply(pha_absolute_6_hdr, B_6_hdr > B_min)

                # 07 12-step hdr
                files_12_hdr = [os.path.join(folder_HDR, str(i + 1) + IMG_END) for i in range(N)]
                (pha_absolute_12_hdr, sin_sum_12_hdr, cos_sum_12_hdr,
                 KS1_12_hdr, KS2_12_hdr, I_mean_12_hdr, B_12_hdr) = phaser.calcAbsolutePhase(files_12_hdr + files_gray_hdr, N, n)
                pha_absolute_12_hdr = np.multiply(pha_absolute_12_hdr, B_12_hdr > B_min)

                # 08 save result
                idx = loader["idx"]
                save_dict = {"idx": idx,
                             "pha_absolute_3": pha_absolute_3,
                             "pha_absolute_4": pha_absolute_4,
                             "pha_absolute_6": pha_absolute_6,
                             "pha_absolute_12": pha_absolute_12,
                             "pha_absolute_3_hdr": pha_absolute_3_hdr,
                             "pha_absolute_4_hdr": pha_absolute_4_hdr,
                             "pha_absolute_6_hdr": pha_absolute_6_hdr,
                             "pha_absolute_12_hdr": pha_absolute_12_hdr,
                             "pha_cnn": pha_cnn}

                save_file = os.path.join(DIR_OUTPUT_Phase, "phase_" + str(idx) + ".mat")

                savemat(
                    save_file,
                    save_dict)

                # 09 to calculate the phase error
                d_3 = calc_error(pha_absolute_12_hdr, pha_absolute_3, dt)
                d_4 = calc_error(pha_absolute_12_hdr, pha_absolute_4, dt)
                d_6 = calc_error(pha_absolute_12_hdr, pha_absolute_6, dt)
                d_12 = calc_error(pha_absolute_12_hdr, pha_absolute_12, dt)
                d_cnn = calc_error(pha_absolute_12_hdr, pha_cnn, dt)

                d_3s.append(d_3)
                d_4s.append(d_4)
                d_6s.append(d_6)
                d_12s.append(d_12)
                d_cnns.append(d_cnn)

    save_file = os.path.join(DIR_OUTPUT, "phase_error.mat")

    d_3s = np.array(d_3s)
    d_4s = np.array(d_4s)
    d_6s = np.array(d_6s)
    d_12s = np.array(d_12s)
    d_cnns = np.array(d_cnns)
    save_dict = {
        "d_3": d_3s,
        "d_4": d_4s,
        "d_6": d_6s,
        "d_12": d_12s,
        "d_cnn": d_cnns
    }
    savemat(save_file, save_dict)

    xs = np.arange(np.shape(d_3s)[0])
    plt.plot(xs, d_3s, label="3-step")
    plt.plot(xs, d_4s, label="4-step")
    plt.plot(xs, d_6s, label="6-step")
    plt.plot(xs, d_12s, label="12-step")
    plt.plot(xs, d_cnns, label="cnn")
    if FLAG_FPP:
        title = "FPN"
    else:
        title = "UNET"
    plt.title("error of phase" + title)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(DIR_OUTPUT, "error.jpg"), dpi=300)
    print("process has been finished!")


if __name__ == '__main__':
    model_file = os.path.join(LOG_DIR, "final.pth")
    device = "cuda:0"
    dsize = (IMG_W, IMG_H)
    phaser = Phaser(dsize)
    test(FILE_test, model_file, device, phaser)
