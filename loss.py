import torch as t
from torch import nn


class TransSensLoss(nn.Module):

    def __init__(self, margin, beta):
        super(TransSensLoss, self).__init__()
        self.margin = margin
        self.beta = beta

    def forward(self, r_source_glob_feature, t_source_glob_feature, r_xt, r_xr, t_xt, t_xr):
        # r_xt, r_xr, t_xt, t_xr均是在eval状态下得到的, r_source_glob_feature, t_source_glob_feature为training状态下得到的
        L_s_r = t.max(t.norm(r_source_glob_feature - r_xt, dim=2) - t.norm(r_source_glob_feature - r_xr, dim=2) + self.margin, t.norm(r_source_glob_feature - r_xt, dim=2))
        L_s_t = t.max(t.norm(t_source_glob_feature - t_xr, dim=2) - t.norm(t_source_glob_feature - t_xt, dim=2) + self.margin, t.norm(t_source_glob_feature - t_xr, dim=2))
        Ls = self.beta * t.mean(L_s_r + L_s_t)
        return Ls


class DropoutLoss(nn.Module):

    def __init__(self, gamma):
        super(DropoutLoss, self).__init__()
        self.gamma = gamma

    def forward(self, r_source_glob_feature, r_reference_glob_feature, t_source_glob_feature, t_reference_glob_feature, r_source_dropout_glob_feature, r_reference_dropout_glob_feature, t_source_dropout_glob_feature, t_reference_dropout_glob_feature):
        # r_source_glob_feature, r_reference_glob_feature, t_source_glob_feature, t_reference_glob_feature全是在eval状态下得到的，r_source_dropout_glob_feature, r_reference_dropout_glob_feature, t_source_dropout_glob_feature, t_reference_dropout_glob_feature全是在training状态下得到的
        L_d_x = t.norm(r_source_glob_feature - r_source_dropout_glob_feature, dim=2) + t.norm(t_source_glob_feature - t_source_dropout_glob_feature, dim=2)
        L_d_y = t.norm(r_reference_glob_feature - r_reference_dropout_glob_feature, dim=2) + t.norm(t_reference_glob_feature - t_reference_dropout_glob_feature, dim=2)
        Ld = self.gamma * t.mean(L_d_x + L_d_y)
        return Ld


class ParamRegLoss(nn.Module):

    def __init__(self, lamda):
        super(ParamRegLoss, self).__init__()
        self.lamda = lamda

    def forward(self, q_pred, t_vec_pred, q_gt, t_vec_gt):
        Lp = t.mean(t.sum(t.abs(q_pred - q_gt), dim=2) + self.lamda * t.norm(t_vec_pred - t_vec_gt, dim=2))
        return Lp