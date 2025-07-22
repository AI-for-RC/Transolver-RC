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

        loss_disp = torch.mean((out_disp - y_disp)**2)
        loss_stress = []
        for i in range(8):
            current_out_stress = out_stress[i]
            current_y_stress = y_stress[i]
            diff_stress = current_out_stress - current_y_stress
            diff_stress_norm = torch.norm(diff_stress, self.p, dim=-1)
            y_stress_norm = torch.norm(current_y_stress, self.p, dim=-1)
            loss_stress.append(diff_stress_norm / (y_stress_norm + self.epsilon))
        loss_stress = torch.mean(torch.cat(loss_stress))

        loss_strain = []
        for i in range(8):
            current_out_strain = out_strain[i]
            current_y_strain = y_strain[i]
            diff_strain = current_out_strain - current_y_strain
            diff_strain_norm = torch.norm(diff_strain, self.p, dim=-1)
            y_strain_norm = torch.norm(current_y_strain, self.p, dim=-1)
            loss_strain.append(diff_strain_norm / (y_strain_norm + self.epsilon))
        loss_strain = torch.mean(torch.cat(loss_strain))

        loss = self.disp_weight * loss_disp + self.stress_weight * loss_stress + self.strain_weight * loss_strain

        return loss

    def __call__(self, out, y):
        return self.loss(out, y)
