import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import savemat
from config import IMG_END
from srcs.utils import *


class GrayCode(object):
    def makePatterns(self, n, W, H, is_y=False):
        patterns = []
        # 01 Gray code
        codes = self._GrayCode(n)
        for idx in range(n):
            row = codes[idx, :]
            one_row = np.zeros([W], np.uint8)
            num = len(row)
            assert (W % num == 0)  # 必须整除
            per_col = int(W / num)
            for i in range(num):
                one_row[i * per_col: (i + 1) * per_col] = row[i]
            pattern = np.tile(one_row, (H, 1)) * 255
            if is_y:
                pattern = pattern.T
            patterns.append(pattern)

        # 02 Gray code complementary
        codes_com = self._GrayCodeCom(n)
        num2 = len(codes_com)  # 互补格雷码的数量
        one_row = np.zeros([W], np.uint8)
        if W % num2 == 0:
            per_col = int(W / num2)
            for i in range(num2):
                one_row[i * per_col: (i + 1) * per_col] = codes_com[i]
            pattern = np.tile(one_row, (H, 1)) * 255
        else:
            per_col_1 = round(W / num2)
            per_col_2 = per_col_1 + 1
            per_col = per_col_1 + per_col_2
            for i in range(num2):
                idx = round(i / 2)
                if i % 2 == 0:
                    one_row[idx * per_col: idx * per_col + per_col_1] = codes_com[i]
                else:
                    one_row[idx * per_col + per_col_1: (idx + 1) * per_col] = codes_com[i]
            pattern = np.tile(one_row, (H, 1)) * 255
            if is_y:
                pattern = pattern.T
        img_white = np.ones_like(pattern, dtype=np.uint8) * 255
        img_black = np.zeros_like(pattern, dtype=np.uint8)

        patterns.append(img_black)
        patterns.append(img_white)
        patterns.append(pattern)
        return patterns

    def calcGrayCode(self, files, n, dsize=None):
        num = len(files)
        img = imread2(files[0], dsize)

        h, w = img.shape
        Is = np.zeros((num, h, w), dtype=np.float32)
        for idx, file in enumerate(files):
            Is[idx, ...] = imread2(file, dsize).astype(np.float32) / 255.

        # 02 threshold
        Is_max = np.max(Is, axis=0)
        Is_min = np.min(Is, axis=0)
        Is_std = (Is - Is_min) / (Is_max - Is_min)
        gcs = Is_std > 0.5
        gcs = gcs.astype(np.int32)

        # 04 calculate KS1 KS2
        K1S, K2S = self._calcKS(gcs, n, h, w)
        return K1S, K2S

    def _calcKS(self, gcs, n, h, w):
        V1K1_dict = self._V1toK1(n)
        V2K2_dict = self._V2toK2(n)

        gcs1 = gcs[:n, ...]
        gcs_c = np.expand_dims(gcs[-1, ...], axis=0)

        gcs2 = np.concatenate((gcs1, gcs_c), axis=0)

        ns1 = expand_dim(np.arange(n))
        ns2 = expand_dim(np.arange(n + 1))
        VS1 = np.sum(gcs1 * np.power(2, (n - (ns1 + 1))), axis=0)
        VS2 = np.sum(gcs2 * np.power(2, (n + 1 - (ns2 + 1))), axis=0)

        KS1 = np.zeros((h, w), dtype=np.int32)
        KS2 = np.zeros((h, w), dtype=np.int32)
        for v in range(h):
            for u in range(w):
                KS1[v, u] = V1K1_dict[VS1[v, u]]
                KS2[v, u] = V2K2_dict[VS2[v, u]]
        return KS1, KS2

    def _V1toK1(self, n):
        graycode = self._GrayCode(n)
        # 03 V1 -> K1
        V1_ROW = []
        for i in range(2 ** n):
            code = graycode[:, i]
            v1 = 0
            for j in range(n):
                v1 += code[j] * 2 ** (n - (j + 1))
            V1_ROW.append(v1)
        V1K = dict()
        for idx, v1 in enumerate(V1_ROW):
            V1K[v1] = idx

        # for k, v in V1K.items():
        #     print(k, v)
        return V1K

    def _V2toK2(self, n):
        graycode = self._GrayCode(n)
        # 04 V2 -> K2
        V2_ROW = []
        graycode2 = np.repeat(graycode, 2, axis=1)
        graycode_com = self._GrayCodeCom(n)
        graycode_all = np.vstack((graycode2, graycode_com))

        for i in range(2 ** (n + 1)):
            code = graycode_all[:, i]
            v2 = 0
            for j in range(n + 1):
                v2 += code[j] * 2 ** (n + 1 - (j + 1))
            V2_ROW.append(v2)

        V2K = dict()
        for idx, v2 in enumerate(V2_ROW):
            V2K[v2] = int((idx + 1) / 2)
        return V2K

    def _GrayCode(self, n:int):
        code_temp = GrayCode.__GrayCode(n)
        codes = []
        for row in range(len(code_temp[0])):
            c = []
            for idx in range(len(code_temp)):
                c.append(int(code_temp[idx][row]))
            codes.append(c)
        return np.array(codes, np.uint8)

    @staticmethod
    def __GrayCode(n:int):
        if n < 1:
            print("the number must grater than 0")
            assert (0);
        elif n == 1:
            code = ["0", "1"]
            return code
        else:
            code = []
            code_pre = GrayCode.__GrayCode(n - 1)
            for idx in range(len(code_pre)):
                code.append("0" + code_pre[idx])
            for idx in range(len(code_pre) - 1, -1, -1):
                code.append("1" + code_pre[idx])
            return code

    def _GrayCodeCom(self, n:int):
        gcs5 = [0, 1, 1, 0]
        num = 2 ** (n + 1)
        codes_com = []
        for i in range(num):
            r = i % 4
            codes_com.append(gcs5[int(r)])
        return np.array(codes_com, dtype=np.uint8)


# 相移法实现
class Phaser(object):
    def __init__(self, dsize=None):
        self._grayCode = GrayCode()
        self.dsize = dsize

    def makePatterns(self, A, B, N, n, W, H, save_dir=None):
        """ to generate the PS patterns"""
        # X direction
        xs = np.arange(W)
        T = W / (2 ** n)
        f = W / T
        imgs_pha = []
        for k in range(N):
            I = A + B * np.cos((2. * f * xs / W - 2. * k / N) * np.pi)
            img = np.tile(I, (H, 1))
            imgs_pha.append(img)
        imgs_gray = self._grayCode.makePatterns(n, W, H)

        # Y direction
        ys = np.arange(H)
        T_y = H / (2 ** n)
        f_y = H / T_y
        imgs_pha_y = []
        for k in range(N):
            I = A + B * np.cos((2. * f_y * ys / H - 2. * k / N) * np.pi)
            img = np.tile(I, (W, 1))
            img = img.T
            imgs_pha_y.append(img)
        imgs_gray_y = self._grayCode.makePatterns(n, H, W, True)

        imgs = imgs_pha + imgs_gray + imgs_pha_y + imgs_gray_y
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imshow("img", img / 255.)
                save_file = os.path.join(save_dir, str(i + 1) + IMG_END)
                cv2.imwrite(save_file, img)
                print("save phase img file into:", save_file)
                cv2.waitKey(100)
            cv2.destroyAllWindows()
        return imgs

    def calcAbsolutePhase(self, files, N, n, save_dir=None):
        """ 计算绝对相位 """
        files_phase, files_gray = files[: N], files[N:]
        # 01 wrapped phase
        pha_wrapped, B, sin_sum, cos_sum, I_mean = self._calcWrappedPhase(files_phase, self.dsize)
        # 02 Gray code
        KS1, KS2 = self._grayCode.calcGrayCode(files_gray, n, self.dsize)
        # 03 absolute phase
        mask1 = pha_wrapped <= np.pi / 2
        mask2 = np.multiply(np.pi / 2 < pha_wrapped, pha_wrapped <= (3 / 2 * np.pi))
        mask3 = pha_wrapped > (3 / 2 * np.pi)
        pha = np.multiply(pha_wrapped + 2 * np.pi * KS2, mask1) \
            + np.multiply(pha_wrapped + 2 * np.pi * KS1, mask2) \
            + np.multiply(pha_wrapped + 2 * np.pi * KS2 - 2 * np.pi, mask3)
        # normalized into 0 - 1
        pha_absolute = (pha / ((2 ** n) * (2 * np.pi)))
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            pha_show = pha_absolute
            cv2.imshow("pha", pha_show)
            save_file = os.path.join(save_dir, "pha" + IMG_END)
            cv2.imwrite(save_file, pha_show)
            cv2.waitKey(0)
            cv2.destroyWindow("pha")
        return pha_absolute, sin_sum, cos_sum, KS1, KS2, I_mean, B

    def calcAbsolutePhase2(self, files, N, n):
        pha_absolute, sin_sum, cos_sum, KS1, KS2, I_mean, B = self.calcAbsolutePhase(files, N, n, None)
        return pha_absolute, B

    def _calcWrappedPhase(self, files, dsize=None):
        N = len(files)
        sin_sum, cos_sum = 0., 0.
        I_mean = 0
        for k in range(N):
            Ik = self._loadIk(files[k], dsize=dsize)
            pk = 2. * k / N * np.pi
            sin_sum += Ik * np.sin(pk)
            cos_sum += Ik * np.cos(pk)
            I_mean += Ik
        I_mean /= N
        pha = np.arctan2(sin_sum, cos_sum)
        B = np.sqrt(sin_sum ** 2 + cos_sum ** 2) * 2 / N
        e = -0.0000000001
        pha_low_mask = pha < e
        pha = pha + pha_low_mask * 2. * np.pi
        return pha, B, sin_sum, cos_sum, I_mean

    def _loadIk(self, file, dsize=None):
        return imread2(file, dsize)


if __name__ == '__main__':
    from config import *
    from utils import *

    # make patterns
    N = 12
    save_dir = DIR_DATA_pattern
    phaser = Phaser()
    phaser.makePatterns(A, B, N, n, PRO_W, PRO_H, save_dir)

    # # test decode
    # BT = 0
    # folder = os.path.join(DATA_DIR, "Pattern")
    # files = list_files(folder)
    # phaser.calcAbsolutePhase(files, N, n, True)

