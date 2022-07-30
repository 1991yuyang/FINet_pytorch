import torch as t
from torch import nn
import numpy as np
from numpy import random as rd


class BaseModule(nn.Module):

    def __init__(self, in_features, out_features):
        super(BaseModule, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class PFI(nn.Module):

    def __init__(self):
        super(PFI, self).__init__()

    def forward(self, source_feature, reference_feature):
        """

        :param source_feature: 源点云特征，形状为[B, N, L]
        :param reference_feature: 参考点云特征，形状为[B, N, L]
        :return:
        """
        source_feature_pool = t.max(source_feature, dim=1)[0].unsqueeze(1)  # [B, 1, L]
        reference_feature_pool = t.max(reference_feature, dim=1)[0].unsqueeze(1)  # [B, 1, L]
        source_feature_pool_repeat = source_feature_pool.repeat(1, reference_feature.size()[1], 1)
        reference_feature_pool_repeat = reference_feature_pool.repeat(1, source_feature.size()[1], 1)
        source_feature_ret = t.cat([source_feature, reference_feature_pool_repeat], dim=2)
        reference_feature_ret = t.cat([reference_feature, source_feature_pool_repeat], dim=2)
        return source_feature_ret, reference_feature_ret


class GFI(nn.Module):

    def __init__(self, in_features, out_features):
        super(GFI, self).__init__()
        middle_features = out_features // 2
        self.block = nn.Sequential(
            BaseModule(in_features=in_features, out_features=middle_features),
            BaseModule(in_features=middle_features, out_features=middle_features),
            BaseModule(in_features=middle_features, out_features=out_features)
        )

    def forward(self, source_feature, reference_feature):
        """

        :param source_feature: 源点云特征，形状为[B, 1, L]
        :param reference_feature: 参考点云特征，形状为[B, 1, L]
        :return:
        """
        source_reference_feature = t.cat([source_feature, reference_feature], dim=2)
        reference_source_feature = t.cat([reference_feature, source_feature], dim=2)
        Hx = self.block(source_reference_feature)  # [B, 1, out_features]
        Hy = self.block(reference_source_feature)  # [B, 1, out_features]
        return Hx, Hy


class RotRegBranch(nn.Module):

    def __init__(self):
        super(RotRegBranch, self).__init__()
        self.block = nn.Sequential(
            BaseModule(in_features=256 * 4, out_features=32),
            BaseModule(in_features=32, out_features=64),
            BaseModule(in_features=64, out_features=128),
            BaseModule(in_features=128, out_features=4)
        )

    def forward(self, r_source_gfi_feature, r_reference_gfi_feature, t_source_gfi_feature, t_reference_gfi_feature):
        cat_result = t.cat([r_source_gfi_feature, t_source_gfi_feature, r_reference_gfi_feature, t_reference_gfi_feature], dim=2)
        rot_param = self.block(cat_result)  # (B, 1, 4)
        return rot_param


class TransRegBranch(nn.Module):

    def __init__(self):
        super(TransRegBranch, self).__init__()
        self.block = nn.Sequential(
            BaseModule(in_features=256 * 3, out_features=32),
            BaseModule(in_features=32, out_features=64),
            BaseModule(in_features=64, out_features=128),
            BaseModule(in_features=128, out_features=3)
        )

    def forward(self, r_source_gfi_feature, r_reference_gfi_feature, t_source_gfi_feature, t_reference_gfi_feature):
        up = t.cat([r_source_gfi_feature, t_source_gfi_feature, t_reference_gfi_feature], dim=2)  # [B, 1, 3 * 256]
        bottom = t.cat([r_reference_gfi_feature, t_reference_gfi_feature, r_reference_gfi_feature], dim=2)  # [B, 1, 3 * 256]
        cx = self.block(up)  # (B, 1, 3)
        cy = self.block(bottom)  # (B, 1, 3)
        t_vec = cy - cx  # (B, 1, 3)
        return t_vec


class FINet(nn.Module):

    def __init__(self, drop_out_ratio):
        super(FINet, self).__init__()
        self.drop_out_ratio = drop_out_ratio
        self.r_base_1 = BaseModule(in_features=3, out_features=64)
        self.r_base_2 = BaseModule(in_features=64, out_features=128)
        self.r_pfi_1 = PFI()
        self.r_base_3 = BaseModule(in_features=256, out_features=256)
        self.r_base_4 = BaseModule(in_features=256, out_features=512)
        self.r_pfi_2 = PFI()
        self.r_base_5 = BaseModule(in_features=512 * 2, out_features=512)
        self.r_gfi = GFI(in_features=(64 + 128 + 256 + 512 + 512) * 2, out_features=256)
        self.rot_reg_branch = RotRegBranch()

        self.t_base_1 = BaseModule(in_features=3, out_features=64)
        self.t_base_2 = BaseModule(in_features=64, out_features=128)
        self.t_pfi_1 = PFI()
        self.t_base_3 = BaseModule(in_features=256, out_features=256)
        self.t_base_4 = BaseModule(in_features=256, out_features=512)
        self.t_pfi_2 = PFI()
        self.t_base_5 = BaseModule(in_features=512 * 2, out_features=512)
        self.t_gfi = GFI(in_features=(64 + 128 + 256 + 512 + 512) * 2, out_features=256)
        self.trans_reg_branch = TransRegBranch()

    def forward(self, source, reference):
        if self.training:
            # 要使得训练时的dropout特征和eval时的global特征接近
            rand_mask = t.from_numpy(rd.uniform(0, 1, (source.size()[0], source.size()[1], 1))).type(t.FloatTensor).to(source.device) < self.drop_out_ratio
        else:
            rand_mask = t.from_numpy(np.ones(shape=(source.size()[0], source.size()[1], 1), dtype=np.float32)).type(t.FloatTensor).to(source.device)
        r_source_base1_result = self.r_base_1(source)
        r_reference_base1_result = self.r_base_1(reference)

        r_source_base2_result = self.r_base_2(r_source_base1_result)
        r_reference_base2_result = self.r_base_2(r_reference_base1_result)

        r_source_pfi_1_result, r_reference_pfi_1_result = self.r_pfi_1(r_source_base2_result, r_reference_base2_result)

        r_source_base3_result = self.r_base_3(r_source_pfi_1_result)
        r_reference_base3_result = self.r_base_3(r_reference_pfi_1_result)

        r_source_base4_result = self.r_base_4(r_source_base3_result)
        r_reference_base4_result = self.r_base_4(r_reference_base3_result)

        r_source_pfi_2_result, r_reference_pfi_2_result = self.r_pfi_1(r_source_base4_result, r_reference_base4_result)

        r_source_base5_result = self.r_base_5(r_source_pfi_2_result)
        r_reference_base5_result = self.r_base_5(r_reference_pfi_2_result)

        r_source_cat_result = t.cat([r_source_base1_result, r_source_base2_result, r_source_base3_result, r_source_base4_result, r_source_base5_result], dim=2)
        r_reference_cat_result = t.cat([r_reference_base1_result, r_reference_base2_result, r_reference_base3_result, r_reference_base4_result, r_reference_base5_result], dim=2)

        r_source_glob_feature = t.max(r_source_cat_result, dim=1)[0].unsqueeze(1)  # (B, 1, L)
        r_reference_glob_feature = t.max(r_reference_cat_result, dim=1)[0].unsqueeze(1)  # (B, 1, L)

        r_source_dropout_result = r_source_cat_result * rand_mask
        r_reference_dropout_result = r_reference_cat_result * rand_mask

        r_source_dropout_glob_feature = t.max(r_source_dropout_result, dim=1)[0].unsqueeze(1)  # (B, 1, L)
        r_reference_dropout_glob_feature = t.max(r_reference_dropout_result, dim=1)[0].unsqueeze(1)  # (B, 1, L)

        r_Hx, r_Hy = self.r_gfi(r_source_glob_feature, r_reference_glob_feature)

        t_source_base1_result = self.t_base_1(source)
        t_reference_base1_result = self.t_base_1(reference)

        t_source_base2_result = self.t_base_2(t_source_base1_result)
        t_reference_base2_result = self.t_base_2(t_reference_base1_result)

        t_source_pfi_1_result, t_reference_pfi_1_result = self.t_pfi_1(t_source_base2_result, t_reference_base2_result)

        t_source_base3_result = self.t_base_3(t_source_pfi_1_result)
        t_reference_base3_result = self.t_base_3(t_reference_pfi_1_result)

        t_source_base4_result = self.t_base_4(t_source_base3_result)
        t_reference_base4_result = self.t_base_4(t_reference_base3_result)

        t_source_pfi_2_result, t_reference_pfi_2_result = self.t_pfi_1(t_source_base4_result, t_reference_base4_result)

        t_source_base5_result = self.t_base_5(t_source_pfi_2_result)
        t_reference_base5_result = self.t_base_5(t_reference_pfi_2_result)

        t_source_cat_result = t.cat([t_source_base1_result, t_source_base2_result, t_source_base3_result, t_source_base4_result, t_source_base5_result], dim=2)
        t_reference_cat_result = t.cat([t_reference_base1_result, t_reference_base2_result, t_reference_base3_result, t_reference_base4_result, t_reference_base5_result], dim=2)

        t_source_glob_feature = t.max(t_source_cat_result, dim=1)[0].unsqueeze(1)  # (B, 1, L)
        t_reference_glob_feature = t.max(t_reference_cat_result, dim=1)[0].unsqueeze(1)  # (B, 1, L)

        t_source_dropout_result = t_source_cat_result * rand_mask
        t_reference_dropout_result = t_reference_cat_result * rand_mask

        t_source_dropout_glob_feature = t.max(t_source_dropout_result, dim=1)[0].unsqueeze(1)  # (B, 1, L)
        t_reference_dropout_glob_feature = t.max(t_reference_dropout_result, dim=1)[0].unsqueeze(1)  # (B, 1, L)

        t_Hx, t_Hy = self.t_gfi(t_source_glob_feature, t_reference_glob_feature)

        rot_param = self.rot_reg_branch(r_Hx, r_Hy, t_Hx, t_Hy)
        rot_param = rot_param / t.norm(rot_param, dim=2, keepdim=True)  # 四元数norm应当为1
        t_vec = self.trans_reg_branch(r_Hx, r_Hy, t_Hx, t_Hy)

        return rot_param, t_vec, r_source_glob_feature, r_reference_glob_feature, t_source_glob_feature, t_reference_glob_feature, r_source_dropout_glob_feature, r_reference_dropout_glob_feature, t_source_dropout_glob_feature, t_reference_dropout_glob_feature


if __name__ == "__main__":
    model = FINet(0.3)
    source = t.randn(2, 256, 3)
    reference = t.randn(2, 256, 3)
    rot_param, t_vec, r_source_glob_feature, r_reference_glob_feature, t_source_glob_feature, t_reference_glob_feature, r_source_dropout_glob_feature, r_reference_dropout_glob_feature, t_source_dropout_glob_feature, t_reference_dropout_glob_feature = model(source, reference)
    print(rot_param.size())
    print(t_vec.size())