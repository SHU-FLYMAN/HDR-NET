import time
import random
import os.path

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from srcs.model import *
from srcs.utils import set_seed
from srcs.dataset import DatasetHDR, collate_fn
from config import *


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.02)
        nn.init.zeros_(m.bias)


def train(folder, net, epochs, lr, batch_size, device, seed=1):
    set_seed(seed)  # random seed
    # -------- 01 load data  -------- #
    files_train, files_test = [], []
    with open(FILE_train, "r") as fr:
        for line in fr.readlines():
            line = line.strip().strip("\n")
            file = os.path.join(folder, line)
            files_train.append(file)

    with open(FILE_test, "r") as fr:
        for line in fr.readlines():
            line = line.strip().strip("\n")
            file = os.path.join(folder, line)
            files_test.append(file)

    dataset_train = DatasetHDR(files_train, 12)
    dataset_valid = DatasetHDR(files_test, 12)
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=1,
        collate_fn=collate_fn, pin_memory=True, drop_last=True)
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=1,
        collate_fn=collate_fn, pin_memory=True, drop_last=True)

    train_num = len(dataset_train.files) // batch_size
    valid_num = len(dataset_valid.files) // batch_size

    # -------- 02  train strategy -------- #
    net.to(device)  # 转移到GPU端
    optimizer_sgd = torch.optim.Adam(lr=lr, params=net.parameters(), weight_decay=0.0005)

    scheduler = ReduceLROnPlateau(
        optimizer_sgd, mode="min", factor=0.5, patience=5,
        verbose=True, threshold=0.0001, threshold_mode="rel",
        cooldown=0, min_lr=0, eps=1e-08)

    input_data = torch.rand(batch_size, 3, IMG_H, IMG_W).to(device)

    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    writer.add_graph(model=net, input_to_model=input_data)

    # -------- 03  start train -------- #
    for epoch in tqdm(range(epochs), desc="epoch"):
        start_time = time.time()
        # --- train part --- #
        train_loss = 0
        net.train()
        for i, (imgs, s_labels, c_labels, I_means) in enumerate(dataloader_train):
            # 01 加载数据
            imgs, s_labels, c_labels, I_means = (
                imgs.to(device), s_labels.to(device), c_labels.to(device), I_means.to(device))
            # 02 清空梯度
            optimizer_sgd.zero_grad()
            if FLAG_FPP:
                # 03 计算结果
                cnn1_out, cnn2_out = net(imgs)
                # 04 计算损失
                l1 = cnn1_loss(cnn1_out, I_means)
                l2 = cnn2_loss(cnn2_out, s_labels, c_labels)
                l = (0.2 * l1 + l2)
            else:
                cnn_out2 = net(imgs)
                l = cnn2_loss(cnn_out2, s_labels, c_labels)
            if l == "nan":
                print("nan")
            # 05 反向传播
            l.mean().backward()
            v = l.item()
            print(i, "-loss:", v)
            # 所有损失
            train_loss += v
            # 06 更新梯度
            optimizer_sgd.step()
        train_loss /= train_num
        print("epoch:", epoch, "train loss:", train_loss)
        writer.add_scalar("train_loss", train_loss, epoch)

        # --- test part --- #
        valid_loss = 0.
        with torch.no_grad():
            net.eval()  # 设置验证模式
            # 暂时先用这个数据集验证
            for i, (imgs, s_labels, c_labels, I_means) in enumerate(dataloader_valid):
                imgs, s_labels, c_labels, I_means = (
                    imgs.to(device), s_labels.to(device), c_labels.to(device), I_means.to(device))
                if FLAG_FPP:
                    cnn1_out, cnn2_out = net(imgs)
                    l1 = cnn1_loss(cnn1_out, I_means)
                    l2 = cnn2_loss(cnn2_out, s_labels, c_labels)
                    l = 0.2 * l1 + l2
                else:
                    cnn_out2 = net(imgs)
                    l = cnn2_loss(cnn_out2, s_labels, c_labels)
                v = l.item()
                print(i, "-val_loss:", v)
                valid_loss += l.item()
        valid_loss /= valid_num
        print("epoch:", epoch, "valid loss:", valid_loss)
        end_time = time.time()
        t = (end_time - start_time) / (60 * 60)
        print("epoch_time:", t)
        writer.add_scalar("valid_loss", valid_loss, epoch)

        # add learning rate to writer
        writer.add_scalar("lr", optimizer_sgd.param_groups[0]["lr"], epoch)
        scheduler.step(valid_loss)

        # to save the result for each 5 epochs
        if epoch % 5 == 0:
            save_file = os.path.join(LOG_DIR, "epoch_{}.pth".format(epoch))
            torch.save(net, save_file)

    # to save the result in the last epoch
    torch.save(net, os.path.join(LOG_DIR, "final.pth"))
    writer.close()


if __name__ == '__main__':
    seed       = 1
    batch_size = 1
    lr         = 0.0001
    epochs     = 150
    device     = "cuda:0"
    folder     = DIR_DATA_Train
    if FLAG_FPP:
        net = ResNetFPN(3)  # 网络结构
    else:
        net = UNet(3)
    net.apply(init_normal)  # 初始化
    train(folder, net, epochs, lr, batch_size, device, seed)
