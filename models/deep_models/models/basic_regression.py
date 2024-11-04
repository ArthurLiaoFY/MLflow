import torch


class LinearLReluStack(torch.nn.Module):
    def __init__(self, in_features: int = 2, out_features: int = 1):
        super(LinearLReluStack, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=out_features),
        )

    def forward(self, x: torch.Tensor):
        x = self.linear_relu_stack(x)
        return x
