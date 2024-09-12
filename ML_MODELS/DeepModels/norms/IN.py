import torch


class InstanceNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-5, subtract_last: bool = False
    ):
        super(InstanceNorm, self).__init__()
        self.num_features = num_features
        print(num_features)
        self.eps = eps
        self.subtract_last = subtract_last

        self.center, self.scale = None, None

    def _get_statistic(self, x: torch.Tensor) -> None:
        if self.subtract_last:
            self.center = torch.unsqueeze(input=x[:, -1, :], dim=1)
        else:
            self.center = torch.mean(input=x, dim=1, keepdim=True)
        self.scale = torch.std(input=x, dim=1, keepdim=True) + self.eps

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._get_statistic(x)
        normed_x = (x - self.center) / self.scale
        return normed_x

    def _denormalize(self, normed_x: torch.Tensor) -> torch.Tensor:
        x = normed_x * (
            self.scale if self.scale is not None else torch.ones_like(normed_x)
        ) + (self.center if self.center is not None else torch.zeros_like(normed_x))
        return x

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            return self._normalize(x)

        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise NotImplementedError
