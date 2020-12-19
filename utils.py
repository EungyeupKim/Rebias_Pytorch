import torch
import torch.nn as nn

class HSIC_measure(nn.Module):
    def __init__(self, kernel_type, dataloader, device, sample_rate, sigma_f=1, sigma_g=1):
        super(HSIC_measure, self).__init__()
        self.kernel_type = kernel_type
        self.dataloader = dataloader
        self.device = device
        self.sample_rate = sample_rate
        self.sigma_f = sigma_f
        self.sigma_g = sigma_g

    def rbf_kernel(self, X, sigma):
        X_L2 = self.cal_dist(X)

        return torch.exp(- 1 / (2 * sigma ** 2) * X_L2)

    def cal_dist(self, X): # Refer to "./criterions/hsic.py" in original repository.
        X = X.view(X.shape[0], -1)
        XX = torch.mm(X, X.transpose(0, 1))
        X_L2 = -2 * XX + torch.diag(XX).unsqueeze(0) + torch.diag(XX).unsqueeze(1)

        return X_L2

    def update_sigma(self, model_f, model_g): # Refer to "./criterions/sigma_utils.py" in original repository.
        if self.kernel_type == 'median':
            with torch.no_grad():
                f_dist_list = []
                g_dist_list = []
                for idx, batch in enumerate(self.dataloader):
                    if idx > self.sample_rate * len(self.dataloader):
                        break
                    image = batch[0].to(self.device)
                    f_feats = model_f(image)[1]
                    g_feats = model_g(image)[1]
                    f_dist_list += list(self.l2_dist(f_feats))
                    g_dist_list += list(self.l2_dist(g_feats))

                f_dist_list = torch.stack(f_dist_list, dim=0)
                g_dist_list = torch.stack(g_dist_list, dim=0)
                self.sigma_f, self.sigma_g = torch.median(f_dist_list), torch.median(g_dist_list)
        else:
            self.sigma_f, self.sigma_g = 1, 1

    def forward(self, f, g):
        m = f.shape[0]
        f_kernel = self.rbf_kernel(f, self.sigma_f)
        g_kernel = self.rbf_kernel(g, self.sigma_g)

        zero_d_f = f_kernel - torch.diag(f_kernel)
        zero_d_g = g_kernel - torch.diag(g_kernel)

        hsic = (1 / (m * (m - 3))) * (torch.trace(torch.mm(zero_d_f, zero_d_g.transpose(0, 1))) +
                                    torch.sum(zero_d_f) * torch.sum(zero_d_g) / ((m - 1) * (m - 2)) -
                                    2 / (m - 2) * torch.sum(torch.sum(zero_d_f, dim=0) * torch.sum(zero_d_g, dim=0)))

        return hsic

def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    return lr