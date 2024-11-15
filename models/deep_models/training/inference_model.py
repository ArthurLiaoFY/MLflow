import numpy as np
import torch
from torch.utils.data import DataLoader

from models.deep_models.utils.prepare_data import get_device


def inference_model(
    nn_model: torch.nn.Module,
    test_dataloader: DataLoader,
) -> np.ndarray:
    inference_result = []

    device = get_device()
    nn_model.to(device)
    nn_model.eval()

    with torch.no_grad():
        for test_x in test_dataloader:
            inference_result.append(
                nn_model(test_x[0].to(device)).squeeze(dim=1).detach().cpu().numpy()
            )
    return np.concatenate(inference_result, axis=0)
