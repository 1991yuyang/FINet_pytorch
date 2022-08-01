from torch.utils import data
import open3d as o3d
import os
from utils import batch_q2R, showPCD
import torch as t
import numpy as np
from numpy import random as rd


"""
root_dir
    train
        1.ply
        2.ply
        ...
    valid
        1.ply
        2.ply
        ...
"""


class MySet(data.Dataset):

    def __init__(self, root_dir, is_training, point_count):
        if is_training:
            self.data_dir = os.path.join(root_dir, "train")
        else:
            self.data_dir = os.path.join(root_dir, "valid")
        self.file_names = os.listdir(self.data_dir)
        self.file_pths = [os.path.join(self.data_dir, name) for name in self.file_names]
        self.point_count = point_count

    def __getitem__(self, index):
        file_pth = self.file_pths[index]
        reference = o3d.io.read_point_cloud(file_pth)
        reference = t.from_numpy(np.asarray(reference.points)).type(t.FloatTensor)  # [N, 3]
        random_generate_q = t.randn(4)
        random_generate_q = (random_generate_q / t.norm(random_generate_q)).unsqueeze(0) # [1, 4]
        rot_mat = batch_q2R(random_generate_q.unsqueeze(0))[0]  # [1, 3, 3]
        random_generate_t_vec = t.randn(3).type(t.FloatTensor).unsqueeze(0)
        source = t.matmul(reference, rot_mat.permute(dims=[1, 0])) + random_generate_t_vec
        point_count_of_current_pcd = reference.size()[0]
        reference_select_point_index = rd.choice(list(range(point_count_of_current_pcd)), self.point_count)
        source_select_point_index = rd.choice(list(range(point_count_of_current_pcd)), self.point_count)
        reference = reference[reference_select_point_index, :]
        source = source[source_select_point_index, :]
        return source, reference, random_generate_q, random_generate_t_vec

    def __len__(self):
        return len(self.file_pths)


def make_loader(root_dir, is_training, point_count, batch_size, num_workers):
    loader = iter(data.DataLoader(MySet(root_dir, is_training, point_count), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    root_dir = r"F:\data\shapenet_data"
    s = MySet(root_dir, True, 2800)
    loader = make_loader(root_dir, True, 2300, 4, 4)
    for s, r, q, t_vec in loader:
        print(s.size())
        print(r.size())
        print(q.size())
        print(t_vec.size())