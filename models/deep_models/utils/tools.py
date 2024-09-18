import gc

import torch


def release_memories() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    return None


def get_device() -> torch.device:
    return torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_built() else "cpu"
    )


def check_model_device(model: torch.nn.Module) -> torch.device:
    return torch.device(next(model.parameters()).device)


def check_tensor_device(x: torch.Tensor) -> torch.device:
    return x.device


def log_clamp(x: torch.Tensor, min_value: float = -100.0) -> torch.Tensor:
    return torch.clamp(torch.log(x), min=min_value)


def get_norm(x: torch.Tensor, norm: int = 2) -> torch.Tensor | None:
    match norm:
        case 1:
            try:
                x_norm = torch.sum(torch.abs(x), dim=1)
            except IndexError:
                x_norm = torch.sum(torch.abs(x))
        case 2:
            try:
                x_norm = torch.sum(torch.pow(x, 2), dim=1)
            except IndexError:
                x_norm = torch.sum(torch.pow(x, 2))
        case _:
            print("norm not matched")

    return x_norm
