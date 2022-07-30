import open3d as o3d
import numpy as np
from numpy import random as rd
from copy import deepcopy


def rotatePCD(q, pcd):
    """

    :param q: 四元数，[4,]
    :param pcd: open3d点云对象
    :return:
    """
    pcd_rot = deepcopy(pcd)
    R = pcd.get_rotation_matrix_from_quaternion(q)
    pcd_rot.rotate(R, center=(0, 0, 0))
    return pcd_rot


def transPCD(t_vec, pcd):
    """

    :param t_vec: 平移向量，[3,]
    :param pcd: open3d点云对象
    :return:
    """
    pcd_trans = deepcopy(pcd)
    pcd_trans.translate(t_vec, relative=True)
    return pcd_trans


def transform(q, t_vec, pcd):
    """

    :param q: 四元数，[4,]
    :param t_vec: 平移向量，[3,]
    :param pcd: open3d点云对象
    :return:
    """
    rot_result = rotatePCD(q, pcd)
    result = transPCD(t_vec, rot_result)
    return result


def showPCD(listOfPCD):
    colors = rd.uniform(0, 1, (len(listOfPCD), 3))
    for i, _pcd in enumerate(listOfPCD):
        pcd.paint_uniform_color(colors[i])
    o3d.visualization.draw_geometries(listOfPCD,  # 点云列表
                                      window_name="PCD_transform",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度


def ndarray2o3d(ndarrayPCD):
    """
    numpy的ndarray对象转换为open3d点云对象
    :param ndarrayPCD: ndarray，[N, 3]
    :return:
    """
    ret = o3d.geometry.PointCloud()
    ret.points = o3d.utility.Vector3dVector(ndarrayPCD)
    return ret


def manualTransform(q, t_vec, pcd):
    """

    :param q: 四元数，[4,]
    :param t_vec: 平移向量, [3]
    :param pcd: 点云
    :return:
    """
    pcd_copy = deepcopy(np.array(pcd.points)).T
    R = pcd.get_rotation_matrix_from_quaternion(q)
    pcd_rot = np.matmul(R, pcd_copy).T
    pcd_trans = pcd_rot + t_vec
    ret = ndarray2o3d(pcd_trans)
    return ret


if __name__ == "__main__":
    q = rd.randn(4)
    q = q / np.sqrt(np.sum(q ** 2))
    t_vec = rd.randn(3)
    pcd_pth = r"F:\data\shapenet_data\train\1.ply"
    pcd = o3d.io.read_point_cloud(pcd_pth)
    o3d_result = transform(q, t_vec, pcd)
    manul_result = manualTransform(q, t_vec, pcd)
    showPCD([pcd, o3d_result, manul_result])