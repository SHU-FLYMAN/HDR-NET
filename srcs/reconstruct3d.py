import cv2
import numpy as np

from tqdm import tqdm
from scipy.interpolate import interp1d
from srcs.Phaser import Phaser
from config import *


def get_world_points(pattern_size):
    world_point = np.zeros((np.prod(pattern_size), 3), np.float32)
    world_point[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    world_point *= DIST
    return world_point


def detect_circles(calib_folder):
    phaser = Phaser()
    # 圆心检测参数
    paras = cv2.SimpleBlobDetector_Params()
    paras.filterByArea = True
    paras.minArea = 20
    paras.maxArea = 10e4
    paras.minDistBetweenBlobs = 10
    detector = cv2.SimpleBlobDetector_create(paras)

    folders = [os.path.join(calib_folder, f) for f in os.listdir(calib_folder)]
    folders.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))

    img_num = N + n + 2 + 1  # 全部图片
    white_num = img_num - 1  # 全白图片
    pattern_size = (COLS, ROWS)
    world_points = get_world_points(pattern_size)
    world_points_all = []
    pixel_points_all_cam = []
    pixel_points_all_pro = []
    img_size = None
    for f in tqdm(folders, "start to detect circles..."):
        # 01 检测圆心
        file = os.path.join(f, str(white_num) + IMG_END)
        img = cv2.imread(file, 0)
        if img is None:  # 如果
            raise IOError("img is empty!")

        if img_size is None:
            img_size = img.shape
        else:
            assert img_size == img.shape
        img_detect = 255 - img
        ret, points_cam = cv2.findCirclesGrid(
            img_detect, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH, blobDetector=detector)

        # 02 计算相位
        files_X = [os.path.join(f, str(i + 1) + IMG_END) for i in range(img_num)]
        files_Y = [os.path.join(f, str(i + 1 + img_num) + IMG_END) for i in range(img_num)]
        pha_X, B_X = phaser.calcAbsolutePhase2(files_X, N, n)
        pha_Y, B_Y = phaser.calcAbsolutePhase2(files_Y, N, n)

        # 03 相位一致性 -> 转换坐标
        pha_X = pha_X * PRO_W
        pha_Y = pha_Y * PRO_H

        # 04 高斯平滑
        pha_X = cv2.GaussianBlur(pha_X, (3, 3), 0)
        pha_Y = cv2.GaussianBlur(pha_Y, (3, 3), 0)

        # 05 插值获得点
        points_pro = interpolation(pha_X, pha_Y, points_cam)

        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img, pattern_size, points_cam, ret)
            world_points_all.append(world_points)
            pixel_points_all_cam.append(points_cam)
            pixel_points_all_pro.append(points_pro)

        cv2.imshow("img", img)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    return world_points_all, pixel_points_all_cam, pixel_points_all_pro, img_size


def interpolation(pha_X, pha_Y, points_cam):
    num = np.shape(points_cam)[0]
    xs_row = np.zeros(4, dtype=np.float64)
    ys_col = np.zeros(4, dtype=np.float64)
    pha_x_row_1 = np.zeros(4, dtype=np.float64)
    pha_x_row_2 = np.zeros(4, dtype=np.float64)

    pha_y_col_1 = np.zeros(4, dtype=np.float64)
    pha_y_col_2 = np.zeros(4, dtype=np.float64)

    # 3次样条曲线插值
    method = "cubic"
    points_pro = np.zeros_like(points_cam, dtype=np.float32)

    for i in range(num):
        # 对应像素点
        p_cam = np.squeeze(points_cam[i, ...])
        x, y = p_cam[0], p_cam[1]
        # 四个点
        x11, y11 = round(x), round(y)
        x21 = x11 + 1
        y12 = y11

        x12 = x11
        y21 = y12 + 1

        x22 = x11 + 1
        y22 = y12 + 1

        for j in range(4):
            xs_row[j] = x11 + j - 1
            pha_x_row_1[j] = pha_X[int(y11), int(xs_row[j])]
            pha_x_row_2[j] = pha_X[int(y21), int(xs_row[j])]

            ys_col[j] = y11 + j - 1
            pha_y_col_1[j] = pha_Y[int(ys_col[j]), int(x11)]
            pha_y_col_2[j] = pha_Y[int(ys_col[j]), int(x21)]

        # 样条曲线插值
        F_x_row_1 = interp1d(xs_row, pha_x_row_1, method)
        F_x_row_2 = interp1d(xs_row, pha_x_row_2, method)
        pha_x_1 = F_x_row_1(x)
        pha_x_2 = F_x_row_2(x)

        # 双线性插值
        pha_x = (y - y11) * pha_x_2 + (y22 - y) * pha_x_1

        F_y_col_1 = interp1d(ys_col, pha_y_col_1, method)
        F_y_col_2 = interp1d(ys_col, pha_y_col_2, method)
        pha_y_1 = F_y_col_1(y)
        pha_y_2 = F_y_col_2(y)
        pha_y = (x - x11) * pha_y_2 + (x22 - x) * pha_y_1

        p_pro = np.array([pha_x, pha_y], dtype=np.float32)
        points_pro[i] = p_pro
    return points_pro


def calib(calib_folder, save_file):
    # 01 检测圆心
    (world_points_all, pixel_points_all_cam,
     pixel_points_all_pro, img_size) = detect_circles(calib_folder)

    # 02 标定相机
    error_cam, Mat_cam, dist_cam, r_vecs_cam, t_vecs_cam = cv2.calibrateCamera(
        world_points_all,
        pixel_points_all_cam,
        img_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST  # 不标定K3、p1、p2
    )

    # 03 标定投影仪
    pro_size = (PRO_H, PRO_W)
    error_pro, Mat_pro, dist_pro, r_vecs_pro, t_vecs_pro = cv2.calibrateCamera(
        world_points_all,
        pixel_points_all_pro,
        pro_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3  # 不标定K3
    )

    # 04 计算Ac、Ap投影矩阵
    idx = 0

    R_cam = cv2.Rodrigues(r_vecs_cam[idx])[0]
    RT_cam = cv2.hconcat([R_cam, t_vecs_cam[idx]])

    R_pro = cv2.Rodrigues(r_vecs_pro[idx])[0]
    RT_pro = cv2.hconcat([R_pro, t_vecs_pro[idx]])

    Ac = np.matmul(Mat_cam, RT_cam)
    Ap = np.matmul(Mat_pro, RT_pro)

    # 05 计算map_x, map_y，用于畸变矫正
    h, w = img_size[0], img_size[1]
    map_x, map_y = cv2.initUndistortRectifyMap(Mat_cam, dist_cam, None, Mat_cam, (w, h), cv2.CV_32FC1)

    # 06 保存标定参数
    print("误差(cam):", error_cam)
    print("误差(pro):", error_pro)
    print("Mat_cam:\n", Mat_cam)
    print("Mat_pro:\n", Mat_pro)
    print("dist_cam:\n", dist_cam)
    print("dist_pro:\n", dist_pro)
    fw = cv2.FileStorage(save_file, cv2.FILE_STORAGE_WRITE)
    fw.write("img_size", img_size)      # 图像尺寸
    fw.write("pro_size", pro_size)
    fw.write("error_cam", error_cam)    # 误差
    fw.write("error_pro", error_pro)
    fw.write("Mat_cam", Mat_cam)        # 相机内参
    fw.write("Mat_pro", Mat_pro)
    fw.write("dist_cam", dist_cam)      # 畸变系数
    fw.write("dist_pro", dist_pro)
    fw.write("Ac", Ac)                  # 投影矩阵（相机）
    fw.write("Ap", Ap)
    fw.write("map_x", map_x)            # 用于提前计算即便
    fw.write("map_y", map_y)
    for i in range(np.shape(r_vecs_cam)[0]):
        fw.write("r_cam_" + str(i + 1), r_vecs_cam[i])
        fw.write("t_cam_" + str(i + 1), t_vecs_cam[i])
        fw.write("r_pro_" + str(i + 1), r_vecs_pro[i])
        fw.write("t_pro_" + str(i + 1), t_vecs_pro[i])
    fw.release()
    print("save camera parameters:", save_file)
    return save_file


def calc_3d(pha, ac, ap, map_x, map_y):
    pha = cv2.resize(pha, (CAM_W, CAM_H))
    xps = pha * PRO_W   # to projector
    # correct distortion
    xps = cv2.remap(xps, map_x, map_y, cv2.INTER_LINEAR)

    points = np.zeros((CAM_H, CAM_W, 3), dtype=np.float64)

    A1 = np.zeros((3, 3), np.float64)
    b1 = np.zeros(3, np.float64)
    for vc in tqdm(range(CAM_H), "calculate 3d point"):
        for uc in range(CAM_W):
            up = xps[vc, uc]
            if up == 0:
                p = np.array([0, 0, 0], np.float64)
            else:
                # 先计算x, y, z
                A1[0, 0] = uc * ac[2, 0] - ac[0, 0]
                A1[1, 0] = vc * ac[2, 0] - ac[1, 0]
                A1[2, 0] = up * ap[2, 0] - ap[0, 0]

                A1[0, 1] = uc * ac[2, 1] - ac[0, 1]
                A1[1, 1] = vc * ac[2, 1] - ac[1, 1]
                A1[2, 1] = up * ap[2, 1] - ap[0, 1]

                A1[0, 2] = uc * ac[2, 2] - ac[0, 2]
                A1[1, 2] = vc * ac[2, 2] - ac[1, 2]
                A1[2, 2] = up * ap[2, 2] - ap[0, 2]

                b1[0] = ac[0, 3] - uc * ac[2, 3]
                b1[1] = ac[1, 3] - vc * ac[2, 3]
                b1[2] = ap[0, 3] - up * ap[2, 3]

                ret, A1_inv = cv2.invert(A1)
                if ret:
                    p = np.matmul(A1_inv, b1)
                else:
                    p = np.array([0, 0, 0], np.float64)
            points[vc, uc, :] = p
    return points


