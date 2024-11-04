from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

from models.deep_models.utils.check_device import get_device


def to_tensor(
    x: np.ndarray | pd.DataFrame | torch.Tensor, device: str | None = None
) -> torch.Tensor | None:
    device = get_device() if device is None else device
    match type(x):
        case np.ndarray:
            tensor = torch.tensor(x).to(torch.float).to(device)
        case pd.Series:
            tensor = torch.tensor(x.values).to(torch.float).to(device)
        case pd.DataFrame:
            tensor = torch.tensor(x.values).to(torch.float).to(device)
        case torch.Tensor:
            tensor = x.to(torch.float).to(device)
        case _:
            try:
                tensor = torch.tensor(x).to(torch.float).to(device)
            except:
                print(
                    "X variable must be of one of the types: np.ndarray, pd.DataFrame, torch.Tensor."
                )
                tensor = None

    return tensor


# def to_dataset(x: np.ndarray | pd.DataFrame | torch.Tensor,
#                y: np.ndarray | pd.DataFrame | torch.Tensor) -> TensorDataset:
def to_dataset(*data: np.ndarray | pd.DataFrame | torch.Tensor) -> TensorDataset:
    return TensorDataset(*(to_tensor(sub_data) for sub_data in data))


def to_dataloader(
    *data: np.ndarray | pd.DataFrame | torch.Tensor,
    batch_size: int = 32,
    sampler: Sampler | None = None,
    shuffle: bool = True,
) -> DataLoader:

    return DataLoader(
        to_dataset(*data),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        drop_last=False,
    )


def one_hot_encoding(x: np.ndarray | pd.DataFrame | torch.Tensor) -> torch.Tensor:
    match type(x):
        case np.ndarray:
            tensor = torch.tensor(x).to(torch.long)
        case pd.Series:
            tensor = torch.tensor(x.values).to(torch.long)
        case pd.DataFrame:
            tensor = torch.tensor(x.values).to(torch.long)
        case torch.Tensor:
            tensor = x.to(torch.long)

    return one_hot(tensor)
