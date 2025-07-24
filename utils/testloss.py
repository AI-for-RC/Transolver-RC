import torch


class TestLoss(object):
    def __init__(self, disp_weight=1, stress_weight=1, strain_weight=1, p=2, epsilon=1e-6):
        super(TestLoss, self).__init__()
        self.disp_weight = disp_weight
        self.stress_weight = stress_weight
        self.strain_weight = strain_weight
        self.p = p
        self.epsilon = epsilon

    def loss(self, out, y):
            
        out_disp = out[:, :, :3]
        y_disp = y[:, :, :3]
        out_stress = [out[:, :, 3 + i*6 : 3 + (i+1)*6] for i in range(8)]
        y_stress = [y[:, :, 3 + i*6 : 3 + (i+1)*6] for i in range(8)]
        out_strain = [out[:, :, 51 + i*6 : 51 + (i+1)*6] for i in range(8)]
        y_strain = [y[:, :, 51 + i*6 : 51 + (i+1)*6] for i in range(8)]

        disp_min = y_disp.min()
        disp_max = y_disp.max()
        out_disp_norm = (out_disp - disp_min) / (disp_max - disp_min + self.epsilon)
        y_disp_norm = (y_disp - disp_min) / (disp_max - disp_min + self.epsilon)
        diff_disp = out_disp_norm - y_disp_norm
        diff_disp_norm = torch.norm(diff_disp, self.p, dim=-1)
        y_disp_norm_val = torch.norm(y_disp_norm, self.p, dim=-1)
        loss_disp = torch.mean(diff_disp_norm / (y_disp_norm_val + self.epsilon))

        loss_stress = []
        for i in range(8):
            current_out_stress = out_stress[i]
            current_y_stress = y_stress[i]
            stress_min = current_y_stress.min()
            stress_max = current_y_stress.max()
            current_out_stress_norm = (current_out_stress - stress_min) / (stress_max - stress_min + self.epsilon)
            current_y_stress_norm = (current_y_stress - stress_min) / (stress_max - stress_min + self.epsilon)
            diff_stress = current_out_stress_norm - current_y_stress_norm
            diff_stress_norm = torch.norm(diff_stress, self.p, dim=-1)
            y_stress_norm_val = torch.norm(current_y_stress_norm, self.p, dim=-1)
            loss_stress.append(diff_stress_norm / (y_stress_norm_val + self.epsilon))
        loss_stress = torch.mean(torch.cat(loss_stress))

        loss_strain = []
        for i in range(8):
            current_out_strain = out_strain[i]
            current_y_strain = y_strain[i]
            strain_min = current_y_strain.min()
            strain_max = current_y_strain.max()
            current_out_strain_norm = (current_out_strain - strain_min) / (strain_max - strain_min + self.epsilon)
            current_y_strain_norm = (current_y_strain - strain_min) / (strain_max - strain_min + self.epsilon)
            diff_strain = current_out_strain_norm - current_y_strain_norm
            diff_strain_norm = torch.norm(diff_strain, self.p, dim=-1)
            y_strain_norm_val = torch.norm(current_y_strain_norm, self.p, dim=-1)
            loss_strain.append(diff_strain_norm / (y_strain_norm_val + self.epsilon))
        loss_strain = torch.mean(torch.cat(loss_strain))

        loss = self.disp_weight * loss_disp + self.stress_weight * loss_stress + self.strain_weight * loss_strain

        return loss

    def __call__(self, out, y):
        return self.loss(out, y)
