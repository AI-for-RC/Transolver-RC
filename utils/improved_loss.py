import torch


class ImprovedLoss(object):
    def __init__(self, disp_weight=1.0, stress_weight=1.0, strain_weight=1.0,
                 p=2, epsilon=1e-6, abs_rel_ratio=0.5, use_standardization=True):
        super(ImprovedLoss, self).__init__()
        # 各部分损失权重
        self.disp_weight = disp_weight
        self.stress_weight = stress_weight
        self.strain_weight = strain_weight
        # 范数参数（L1/L2）
        self.p = p
        # 防除零扰动
        self.epsilon = epsilon
        # 绝对误差与相对误差的权重比
        self.abs_rel_ratio = abs_rel_ratio
        # 归一化方式（标准化/Min-Max）
        self.use_standardization = use_standardization

    def _component_loss(self, out, target):
        """统一计算单个组件（位移/应力/应变）的损失"""
        # 归一化：标准化
        if self.use_standardization:
            # 标准化：(x - 均值) / (标准差 + 扰动)，对分布偏移更稳健
            mean = target.mean()
            std = target.std()
            out_norm = (out - mean) / (std + self.epsilon)
            target_norm = (target - mean) / (std + self.epsilon)
        else:
            # 保留Min-Max归一化作为备选
            min_val = target.min()
            max_val = target.max()
            out_norm = (out - min_val) / (max_val - min_val + self.epsilon)
            target_norm = (target - min_val) / (max_val - min_val + self.epsilon)

        # 计算误差：融合绝对误差和相对误差，平衡稳定性与精度
        diff = out_norm - target_norm
        abs_error = torch.norm(diff, self.p, dim=-1)  # 绝对误差（Lp范数）
        target_norm_val = torch.norm(target_norm, self.p, dim=-1)
        rel_error = abs_error / (target_norm_val + self.epsilon)  # 相对误差

        # 组合误差：平衡绝对误差（稳定性）和相对误差（精度）
        return torch.mean(
            self.abs_rel_ratio * abs_error +
            (1 - self.abs_rel_ratio) * rel_error
        )

    def loss(self, out, y):
        # 1. 位移损失
        out_disp = out[:, :, :3]
        y_disp = y[:, :, :3]
        loss_disp = self._component_loss(out_disp, y_disp)

        # 2. 应力损失
        out_stress = [out[:, :, 3 + i * 6: 3 + (i + 1) * 6] for i in range(8)]
        y_stress = [y[:, :, 3 + i * 6: 3 + (i + 1) * 6] for i in range(8)]
        loss_stress_list = [self._component_loss(o, t) for o, t in zip(out_stress, y_stress)]
        loss_stress = torch.mean(torch.stack(loss_stress_list))

        # 3. 应变损失
        out_strain = [out[:, :, 51 + i * 6: 51 + (i + 1) * 6] for i in range(8)]
        y_strain = [y[:, :, 51 + i * 6: 51 + (i + 1) * 6] for i in range(8)]
        loss_strain_list = [self._component_loss(o, t) for o, t in zip(out_strain, y_strain)]
        loss_strain = torch.mean(torch.stack(loss_strain_list))

        # 总损失
        total_loss = (
                self.disp_weight * loss_disp +
                self.stress_weight * loss_stress +
                self.strain_weight * loss_strain
        )
        return total_loss

    def __call__(self, out, y):
        return self.loss(out, y)