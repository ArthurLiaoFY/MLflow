import torch

from models.deep_models.utils.tools import get_device


class CNNLReluStack(torch.nn.Module):
    def __init__(self, in_features: tuple = (32, 1, 28, 28), out_features: int = 10):
        super().__init__()
        self.device = get_device()
        self.D_out = out_features
        self.batch_size, self.channel, self.width, self.height = in_features
        self.cnn_relu_stack = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.channel, out_channels=32, kernel_size=3),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=512, out_features=64),
            torch.nn.Linear(in_features=64, out_features=self.D_out),
        )

    def forward(self, x: torch.Tensor):
        x = self.cnn_relu_stack(x)
        return x
