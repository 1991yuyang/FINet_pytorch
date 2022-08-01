import open3d as o3d
import numpy as np
from numpy import random as rd
from copy import deepcopy
import torch as t


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
    for i in range(len(listOfPCD)):
        pcd_tem = listOfPCD[i]
        if isinstance(pcd_tem, np.ndarray):
            pcd_rep = o3d.geometry.PointCloud()
            pcd_rep.points = o3d.utility.Vector3dVector(pcd_tem)
            listOfPCD[i] = pcd_rep
    colors = rd.uniform(0, 1, (len(listOfPCD), 3))
    for i, _pcd in enumerate(listOfPCD):
        _pcd.paint_uniform_color(colors[i])
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
    pcd_copy = deepcopy(np.array(pcd.points))
    R = pcd.get_rotation_matrix_from_quaternion(q).T
    pcd_rot = np.matmul(pcd_copy, R)
    pcd_trans = pcd_rot + t_vec
    ret = ndarray2o3d(pcd_trans)
    return ret


def batch_q2R(rot_param):
    """
    一个batch的四元数转换为对应的旋转矩阵rot_matrix，对[B, N, 3]的点云pcd进行旋转就是t.bmm(pcd, rot_matrix.permute(dims=[0, 2, 1]))或t.bmm(rot_matrix, pcd.permute(dims=[0, 2, 1])).permute(dims=[0, 2, 1])
    :param rot_param: 一个batch的四元数，[B, 1, 4]
    :return: 一个batch的旋转矩阵，[B, 3, 3]
    """
    rot_param = rot_param.squeeze(1)
    w = rot_param[..., 0:1]
    x = rot_param[..., 1:2]
    y = rot_param[..., 2:3]
    z = rot_param[..., 3:4]
    r00 = 1 - 2 * t.pow(y, 2) - 2 * t.pow(z, 2)
    r01 = 2 * x * y + 2 * w * z
    r02 = 2 * x * z - 2 * w * y
    r10 = 2 * x * y - 2 * w * z
    r11 = 1 - 2 * t.pow(x, 2) - 2 * t.pow(z, 2)
    r12 = 2 * z * y + 2 * w * x
    r20 = 2 * x * z + 2 * w * y
    r21 = 2 * z * y - 2 * w * x
    r22 = 1 - 2 * t.pow(x, 2) - 2 * t.pow(y, 2)
    concate = t.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
    rot_matrix = concate.view((rot_param.size()[0], 3, 3)).permute(dims=[0, 2, 1])
    return rot_matrix


def batch_rotation(rot_matrix, point_cloud):
    """
    将一个batch的点云进行旋转
    :param rot_matrix: 一个batch的旋转矩阵，[B, 3, 3], 由batch_q2R函数得到
    :param point_cloud: 一个batch的点云数据，[B, N, 3]
    :return: 旋转后的点云数据，[B, N, 3]
    """
    result = t.bmm(point_cloud, rot_matrix.permute(dims=[0, 2, 1]))
    return result


def batch_translate(t_vec, point_cloud):
    """
    一个batch的点云进行平移
    :param t_vec: 一个batch的平移向量，[B, 1, 3]
    :param point_cloud: 一个batch的点云数据, [B, N, 3]
    :return: 平移后的点云数据, [B, N, 3]
    """
    result = point_cloud + t_vec
    return result


def batch_transform(rot_param, t_vec, point_cloud):
    """
    一个batch的点云进行旋转和平移操作
    :param rot_param: 一个batch的四元数，[B, 1, 4]
    :param t_vec: 一个batch的平移向量，[B, 1, 3]
    :param point_cloud: 一个batch的点云数据，[B, N, 3]
    :return: 变换后的点云数据, [B, N, 3]
    """
    rot_matrix = batch_q2R(rot_param)
    rot_result = batch_rotation(rot_matrix, point_cloud)
    rot_trans_result = batch_translate(t_vec, rot_result)
    return rot_trans_result


if __name__ == "__main__":
    q = rd.randn(4)
    q = q / np.sqrt(np.sum(q ** 2))
    t_vec = rd.randn(3)
    pcd_pth = r"F:\data\shapenet_data\train\1.ply"
    pcd = o3d.io.read_point_cloud(pcd_pth)
    o3d_result = transform(q, t_vec, pcd)
    manul_result = manualTransform(q, t_vec, pcd)

    batch_trans_result = batch_transform(t.from_numpy(q).unsqueeze(0).unsqueeze(0), t.from_numpy(t_vec).unsqueeze(0).unsqueeze(0), t.from_numpy(np.asarray(pcd.points)).unsqueeze(0))
    batch_trans_result = ndarray2o3d(batch_trans_result[0].numpy())
    showPCD([pcd, o3d_result, manul_result, batch_trans_result])
    rot_param = t.randn(2, 1, 4)
    rot_param = rot_param / t.norm(rot_param, dim=2, keepdim=True)
    o3d_rot_matrix = []
    for rot_param_ in rot_param:
        R_o3d = pcd.get_rotation_matrix_from_quaternion(rot_param_[0].numpy().astype(np.float64))
        o3d_rot_matrix.append(R_o3d)
    R = batch_q2R(rot_param)
    print("手动计算旋转矩阵：\n", R)
    print("open3d计算旋转矩阵：\n", np.array(o3d_rot_matrix))