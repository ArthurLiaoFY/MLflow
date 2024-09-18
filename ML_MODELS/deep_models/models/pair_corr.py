import torch


class PairwiseCorrelation(torch.nn.Module):
    def __init__(self):
        super(PairwiseCorrelation, self).__init__()

    def forward(self, target_x: torch.Tensor, target_y: torch.Tensor):
        demean_target_x = target_x - target_x.mean(dim=1, keepdim=True)
        demean_target_y = target_y - target_y.mean(dim=1, keepdim=True)
        return (demean_target_x * demean_target_y).sum(dim=1) / (
            torch.sqrt(torch.pow(input=demean_target_x, exponent=2).sum(dim=1))
            * torch.sqrt(torch.pow(input=demean_target_y, exponent=2).sum(dim=1))
        )
