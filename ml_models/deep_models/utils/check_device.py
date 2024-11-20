import torch


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def check_model_device(model: torch.nn.Module) -> torch.device:
    return torch.device(next(model.parameters()).device)


def check_tensor_device(x: torch.Tensor) -> torch.device:
    return x.device
