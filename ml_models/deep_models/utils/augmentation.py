import torch
from torch.nn.functional import interpolate

##### Time Series #####


def data_jitter(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # batch first time series data like:
    # [B, C, L]
    return x + torch.randn(size=x.shape) * sigma


def data_scale(
    x: torch.Tensor, scale_max: float, scale_min: float = 0.0
) -> torch.Tensor:
    # batch first time series data like:
    # [B, C, L]

    return torch.mul(
        x,
        (scale_max - scale_min)
        * torch.rand(size=(x.shape[0],)).reshape(
            shape=(x.shape[0],) + tuple(1 for _ in range(len(x.shape) - 1))
        )
        + scale_min * torch.ones_like(x),
    )


def data_permutation(
    x: torch.Tensor, n_seg: int = 4, min_seg_percent: float = 0.1
) -> torch.Tensor:
    # batch first time series data like:
    # [B, C, L]
    # Do not use it when the time series contains different stages of action.

    min_seg_len = int(min_seg_percent * x.shape[-1])

    while True:
        segments = torch.zeros(size=(n_seg + 1,), dtype=int)
        segments[1:-1] = (
            torch.randint(
                low=min_seg_len,
                high=x.shape[-1] - min_seg_len,
                size=(n_seg - 1,),
            )
            .sort()
            .values
        )

        segments[-1] = x.shape[-1]
        if torch.min(segments[1:] - segments[0:-1]) > min_seg_len:
            segments = segments.numpy()
            break

    return torch.cat(
        tensors=tuple(
            torch.cat(
                tensors=tuple(
                    x[
                        batch_idx,
                        :,
                        segments[seg_order] : segments[seg_order + 1],
                    ]
                    for seg_order in torch.randperm(n=n_seg).numpy()
                ),
                dim=-1,
            )
            for batch_idx in range(x.shape[0])
        ),
        dim=0,
    ).reshape(x.shape)


def data_interpolation(
    x: torch.Tensor,
    ratio: float = 0.85,
) -> torch.Tensor:
    # batch first time series data like:
    # [B, C, L]
    return interpolate(
        input=torch.cat(
            tensors=tuple(
                x[
                    batch_idx,
                    :,
                    torch.randperm(n=int(x.shape[-1] * ratio)).sort().values,
                ]
                for batch_idx in range(x.shape[0])
            ),
            dim=0,
        ).reshape((x.shape[0], x.shape[1], int(x.shape[-1] * ratio))),
        size=x.shape[-1],
        mode="linear",
    )
