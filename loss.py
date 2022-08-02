import torch as t
from torch import nn
from utils import batch_rotation, batch_translate, batch_transform, batch_q2R


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


def iteration_for_loss(model, source, reference, times, trans_sens_loss_obj, dropout_loss_obj, param_reg_loss_obj, rot_param_gt, t_vec_gt):
    param_reg_loss = 0
    trans_sens_loss = 0
    dropout_loss = 0
    for i in range(times):
        model.train()
        rot_param, t_vec, r_source_glob_feature, r_reference_glob_feature, t_source_glob_feature, t_reference_glob_feature, r_source_dropout_glob_feature, r_reference_dropout_glob_feature, t_source_dropout_glob_feature, t_reference_dropout_glob_feature = model(source, reference)
        param_reg_loss += param_reg_loss_obj(rot_param, t_vec, rot_param_gt, t_vec_gt)
        model.eval()
        with t.no_grad():
            rot_param_eval, t_vec_eval, r_source_glob_feature_eval, r_reference_glob_feature_eval, t_source_glob_feature_eval, t_reference_glob_feature_eval, r_source_dropout_glob_feature_eval, r_reference_dropout_glob_feature_eval, t_source_dropout_glob_feature_eval, t_reference_dropout_glob_feature_eval = model(source, reference)
        dropout_loss += dropout_loss_obj(r_source_glob_feature_eval, r_reference_glob_feature_eval, t_source_glob_feature_eval, t_reference_glob_feature_eval, r_source_dropout_glob_feature, r_reference_dropout_glob_feature, t_source_dropout_glob_feature, t_reference_dropout_glob_feature)
        only_translate_result = batch_translate(t_vec_eval, source)
        only_rotation_result = batch_rotation(batch_q2R(rot_param_eval), source)
        with t.no_grad():
            rot_param_only_rot, t_vec_only_rot, r_source_glob_feature_only_rot, r_reference_glob_feature_only_rot, t_source_glob_feature_only_rot, t_reference_glob_feature_only_rot, r_source_dropout_glob_feature_only_rot, r_reference_dropout_glob_feature_only_rot, t_source_dropout_glob_feature_only_rot, t_reference_dropout_glob_feature_only_rot = model(only_rotation_result, reference)
            rot_param_only_trans, t_vec_only_trans, r_source_glob_feature_only_trans, r_reference_glob_feature_only_trans, t_source_glob_feature_only_trans, t_reference_glob_feature_only_trans, r_source_dropout_glob_feature_only_trans, r_reference_dropout_glob_feature_only_trans, t_source_dropout_glob_feature_only_trans, t_reference_dropout_glob_feature_only_trans = model(only_translate_result, reference)
        source = batch_transform(rot_param_eval, t_vec_eval, source)
        model.train()
        rot_param_after_trans, t_vec_after_trans, r_source_glob_feature_after_trans, r_reference_glob_feature_after_trans, t_source_glob_feature_after_trans, t_reference_glob_feature_after_trans, r_source_dropout_glob_feature_after_trans, r_reference_dropout_glob_feature_after_trans, t_source_dropout_glob_feature_after_trans, t_reference_dropout_glob_feature_after_trans = model(source, reference)
        trans_sens_loss += trans_sens_loss_obj(r_source_glob_feature_after_trans, t_source_glob_feature_after_trans, r_source_glob_feature_only_trans, r_source_glob_feature_only_rot, t_source_glob_feature_only_trans, t_source_glob_feature_only_rot)
    total_loss = (param_reg_loss + trans_sens_loss + dropout_loss) / times
    return total_loss


if __name__ == "__main__":
    from model import FINet
    model = FINet(0.3)
    source = t.randn(3, 10, 3)
    reference = t.randn(3, 10, 3)
    rot_param_gt = t.randn(3, 1, 4)
    rot_param_gt = rot_param_gt / t.norm(rot_param_gt, dim=2)
    t_vec_gt = t.randn(3, 1, 3)
    trans_sens_loss_obj = TransSensLoss(margin=0.1, beta=0.1)
    dropout_loss_obj = DropoutLoss(gamma=0.1)
    param_reg_loss_obj = ParamRegLoss(lamda=0.1)
    loss = iteration_for_loss(model, source, reference, 3, trans_sens_loss_obj, dropout_loss_obj, param_reg_loss_obj, rot_param_gt, t_vec_gt)
    print(loss)