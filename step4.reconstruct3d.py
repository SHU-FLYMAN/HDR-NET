import cv2
import numpy as np
import open3d as o3d
from scipy.io import loadmat
from multiprocessing import Pool

from srcs.reconstruct3d import calib, calc_3d
from config import *
import time


def save_cloud(pcd, save_file, show=False):
    if show:
        o3d.visualization.draw_geometries([pcd], window_name=save_file)
    o3d.io.write_point_cloud(save_file, pcd, write_ascii=True)
    print("save point cloud into:", save_file)


def pass_through(pcd, pass_min, pass_max, pass_axis='x'):
    points = np.array(pcd.points)
    if pass_axis == 'x':
        pass_axis = 0
    elif pass_axis == 'y':
        pass_axis = 1
    else:
        pass_axis = 2
    index = np.where((points[:, pass_axis] >= pass_min) & (points[:, pass_axis] <= pass_max))[0]
    pcd = pcd.select_by_index(index)
    return pcd


def filter_points(points):
    pts = np.reshape(points, (-1, 3))
    pts = pts.astype(np.float32)
    pcd = o3d.geometry.PointCloud()  # type:o3d.geometry.PointCloud  # 非常重要，否则没有智能提示
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 01 pass filter
    pcd = pass_through(pcd, Z_MIN, Z_MAX, "z")

    # 02 voxel filter
    num_ori = np.shape(pcd.points)[0]
    pcd = pcd.voxel_down_sample(voxel_size=LEAF_SIZE)
    num_new = np.shape(pcd.points)[0]
    print("reduce the number of points from {} to {}".format(num_ori, num_new))

    # 03 radius filter
    pcd, indexs_rad = pcd.remove_radius_outlier(NB, RADIUS)

    # 04 statistical filter
    pcd, indexs_sta = pcd.remove_statistical_outlier(NB, STDV)
    return pcd

def calc_3d_all(file, out_folder, Ac, Ap, map_x, map_y, flag_show=False):
    data = loadmat(file)
    idx = data["idx"][0][0]
    # 02 calculate 3d coordinates
    points_cnn = calc_3d(data["pha_cnn"], Ac, Ap, map_x, map_y)
    points_3 = calc_3d(data["pha_absolute_3"], Ac, Ap,  map_x, map_y)
    points_4 = calc_3d(data["pha_absolute_4"], Ac, Ap, map_x, map_y)
    points_6 = calc_3d(data["pha_absolute_6"], Ac, Ap, map_x, map_y)
    points_12 = calc_3d(data["pha_absolute_12"], Ac, Ap, map_x, map_y)

    points_12_hdr = calc_3d(data["pha_absolute_12_hdr"], Ac, Ap, map_x, map_y)

    # 03 to save point cloud
    save_folder = os.path.join(out_folder, str(idx))
    os.makedirs(save_folder, exist_ok=True)
    save_file_cnn = os.path.join(save_folder, "points_cnn.pcd")
    save_file_3 = os.path.join(save_folder, "points_3.pcd")
    save_file_4 = os.path.join(save_folder, "points_4.pcd")
    save_file_6 = os.path.join(save_folder, "points_6.pcd")
    save_file_12 = os.path.join(save_folder, "points_12.pcd")
    save_file_12_hdr = os.path.join(save_folder, "points_12_hdr.pcd")

    pcd_cnn = filter_points(points_cnn)
    save_cloud(pcd_cnn, save_file_cnn, flag_show)

    pcd_3 = filter_points(points_3)
    save_cloud(pcd_3, save_file_3, flag_show)

    pcd_4 = filter_points(points_4)
    save_cloud(pcd_4, save_file_4, flag_show)

    pcd_6 = filter_points(points_6)
    save_cloud(pcd_6, save_file_6, flag_show)

    pcd_12 = filter_points(points_12)
    save_cloud(pcd_12, save_file_12, flag_show)

    pcd_12_hdr = filter_points(points_12_hdr)
    save_cloud(pcd_12_hdr, save_file_12_hdr, flag_show)


def reconstruct(calib_file, input_folder, out_folder, flag_debug=False):
    # 01 load the calibration parameters
    fr = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    print("load file:", calib_file)
    Ac = fr.getNode("Ac").mat()
    Ap = fr.getNode("Ap").mat()
    map_x = fr.getNode("map_x").mat()
    map_y = fr.getNode("map_y").mat()
    error_cam = fr.getNode("error_cam").real()
    error_pro = fr.getNode("error_pro").real()
    Mat_cam = fr.getNode("Mat_cam").mat()
    Mat_pro = fr.getNode("Mat_pro").mat()
    dist_cam = fr.getNode("dist_cam").mat()
    dist_pro = fr.getNode("dist_pro").mat()

    print("error(cam):", error_cam)
    print("error(pro):", error_pro)
    print("Mat_cam:\n", Mat_cam)
    print("Mat_pro:\n", Mat_pro)
    print("dist_cam:\n", dist_cam)
    print("dist_pro:\n", dist_pro)

    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
    po = Pool(16)
    for i, file in enumerate(files):
        # debug
        if flag_debug:
            calc_3d_all(file, out_folder, Ac, Ap, map_x, map_y, True)
        else:
            po.apply_async(calc_3d_all, args=(file, out_folder, Ac, Ap, map_x, map_y, False))
    po.close()
    po.join()


if __name__ == '__main__':
    flag_debug = True
    if not os.path.exists(FILE_calib):
        calib(DIR_DATA_calib, FILE_calib)
    reconstruct(FILE_calib, DIR_OUTPUT_Phase, DIR_OUTPUT_3d, flag_debug)
