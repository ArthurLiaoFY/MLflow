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


class FutureUphPredict(torch.nn.Module):
    def __init__(self, in_features: int = 2, out_features: int = 1):
        super(FutureUphPredict, self).__init__()
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


class SoHSimulate(torch.nn.Module):
    def __init__(self, in_features: int = 2, out_features: int = 2):
        super(SoHSimulate, self).__init__()
        self.linear_relu_softmax = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=out_features),
            torch.nn.Sigmoid(),
        )

    def forward(self, input_x: torch.Tensor):
        return self.linear_relu_softmax(input_x)


class UphSimulate(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, max_uph: int):
        super(UphSimulate, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=512),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=512, out_features=512),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=out_features),
        )

    def forward(self, x: torch.Tensor):
        x = self.linear_relu_stack(x)
        return x
