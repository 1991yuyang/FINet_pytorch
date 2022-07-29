import torch as t
from torch import nn


class TransSensLoss(nn.Module):

    def __init__(self, margin):
        super(TransSensLoss, self).__init__()
        self.margin = margin

    def forward(self, r_source_glob_feature, t_source_glob_feature, r_xt, r_xr, t_xt, t_xr):
        # r_xt, r_xr, t_xt, t_xr均是在eval状态下得到的, r_source_glob_feature, t_source_glob_feature为training状态下得到的
        L_s_r = t.max(t.norm(r_source_glob_feature - r_xt) - t.norm(r_source_glob_feature - r_xr) + self.margin, t.norm(r_source_glob_feature - r_xt))
        L_s_t = t.max(t.norm(t_source_glob_feature - t_xr) - t.norm(t_source_glob_feature - t_xt) + self.margin, t.norm(t_source_glob_feature - t_xr))
        Ls = L_s_r + L_s_t
        return Ls


class DropoutLoss(nn.Module):

    def __init__(self):
        super(DropoutLoss, self).__init__()
        pass

    def forward(self, r_source_glob_feature, r_reference_glob_feature, t_source_glob_feature, t_reference_glob_feature, r_source_dropout_glob_feature, r_reference_dropout_glob_feature, t_source_dropout_glob_feature, t_reference_dropout_glob_feature):
        # r_source_glob_feature, r_reference_glob_feature, t_source_glob_feature, t_reference_glob_feature全是在eval状态下得到的，r_source_dropout_glob_feature, r_reference_dropout_glob_feature, t_source_dropout_glob_feature, t_reference_dropout_glob_feature全是在training状态下得到的
        L_d_x = t.norm(r_source_glob_feature - r_source_dropout_glob_feature) + t.norm(t_source_glob_feature - t_source_dropout_glob_feature)
        L_d_y = t.norm(r_reference_glob_feature - r_reference_dropout_glob_feature) + t.norm(t_reference_glob_feature - t_reference_dropout_glob_feature)
        Ld = L_d_x + L_d_y
        return Ld


class ParamRegLoss(nn.Module):

    def __init__(self):
        super(ParamRegLoss, self).__init__()
        pass

    def forward(self):
        pass
