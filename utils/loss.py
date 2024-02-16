import torch


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, alphas_part_max, alphas_org):
        size = alphas_org.shape[0]
        loss_wt = 0.0
        margin = 0.1
        for i in range(size):
            loss_wt += max(torch.Tensor([0]).cuda(), margin - (alphas_part_max[i] - alphas_org[i]))
        return loss_wt / size
