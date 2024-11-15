import torch


def cross_entropy_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "mean",
    weight: torch.Tensor | None = None,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    if y_pred.shape != y_true.shape:
        y_true = torch.zeros_like(y_pred).scatter_(
            index=y_true.long()[None, :],
            value=1.0,
            dim=-1,
        )

    weight = (
        torch.ones_like(y_true) * (weight / weight.sum() if weight is not None else 1.0)
    ).to(y_pred.device)

    y_pred = torch.clamp(y_pred, min=epsilon, max=1 - epsilon)
    loss = (
        -1
        * weight
        * (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    )

    return loss.sum() if reduction == "sum" else loss.mean()


def focal_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    gamma: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    # Formally, the Focal Loss adds a weight factor (1-p)**gamma to the standard cross entropy criterion.
    # Setting gamma > 0 reduces the relative loss for well-classified examples (p > 0.5),
    # putting more focus on hard, misclassified examples.

    prob = -1 * y_true * torch.nn.LogSoftmax(dim=1)(y_pred)
    if reduction == "sum":
        return torch.sum(torch.pow(1 - prob, gamma) * prob)
    else:
        return torch.mean(torch.pow(1 - prob, gamma) * prob)


def mean_absolute_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return (y_pred - y_true).abs().mean()


def mean_square_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.pow((y_pred - y_true), 2).mean()


def root_mean_square_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.pow((y_pred - y_true), 2).mean())


def root_mean_square_percentage_error(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    return torch.sqrt((torch.pow((y_pred - y_true), 2) / torch.pow(y_true, 2)).mean())


def weighted_root_mean_square_error(
    y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return torch.sqrt(weight * torch.pow((y_pred - y_true), 2).mean() / weight.sum())


def fuzzy_root_mean_square_error(
    y_pred: torch.Tensor, y_true: torch.Tensor, epsilon: float = 0.1
) -> torch.Tensor:
    # y_pred[0] - y_true < 0 then epsilon * pow
    # y_true - y_pred[2] < 0 then epslion * pow
    return (
        # torch.sqrt(torch.pow((y_pred[:, 1] - y_true), 2).mean()) +
        torch.sqrt(
            (
                torch.where(y_true - y_pred[:, 0] > 0.0, epsilon, 1.0)
                * torch.pow((y_pred[:, 0] - y_true), 2)
            ).mean()
        )
        + torch.sqrt(
            (
                torch.where(y_pred[:, 1] - y_true > 0.0, epsilon, 1.0)
                * torch.pow((y_pred[:, 1] - y_true), 2)
            ).mean()
        )
    )


def sum_of_square_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.pow((y_pred - y_true), 2).sum()


def huber_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    delta: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    dist = (y_pred - y_true).abs()
    indicator = (dist <= delta).to(torch.float)
    if reduction == "sum":
        return (
            0.5 * torch.pow(dist, 2) * indicator
            + delta * (dist - delta * 0.5) * (1.0 - indicator)
        ).sum()
    else:
        return (
            0.5 * torch.pow(dist, 2) * indicator
            + delta * (dist - delta * 0.5) * (1.0 - indicator)
        ).sum()


def var_explain_rate(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return (
        torch.pow((y_true - y_pred), 2).sum()
        / torch.pow((y_true - y_true.mean()), 2).sum()
    )


# if __name__ == '__main__':
#     from ML_MODELS.DeepModels.utils.prepare_data import one_hot_encoding
#
#     model = torch.nn.Linear(in_features=50, out_features=10)
#     x = torch.randn(1, 50)
#     target = torch.randn(1, 10)
#     output = model(x)
#     loss = huber_loss(y_pred=output, y_true=target)
#     print(loss.item())
#
#     loss = torch.nn.HuberLoss(delta=0.5, reduction='mean')(output, target)
#     print(loss.item())
#
#     model = torch.nn.Sequential(torch.nn.Linear(in_features=50, out_features=2), torch.nn.Softmax(dim=1))
#     x = torch.randn(1000, 50)
#     target = one_hot_encoding(torch.argmax(torch.randn(1000, 2), dim=1))
#
#     output = model(x)
#     print(binary_cross_entropy_loss(output, target.to(torch.float), weight=torch.tensor([0.2, 0.8])).item())
#     print(binary_cross_entropy_loss(output, target.to(torch.float), weight=torch.tensor([0.2, 0.8])).item())
#     print('----------------')
#     print(binary_cross_entropy_loss(torch.tensor([0.6, 0.4]), torch.tensor([1.0, 0.0]), weight=torch.tensor([0.2, 0.8])).item())
#     print(binary_cross_entropy_loss(torch.tensor([0.6, 0.4]), torch.tensor([1.0, 0.0]), weight=torch.tensor([0.5, 0.5])).item())
#     print(binary_cross_entropy_loss(torch.tensor([0.6, 0.4]), torch.tensor([1.0, 0.0]), weight=torch.tensor([0.8, 0.2])).item())
#     print('----------------')
#     print(binary_cross_entropy_loss(torch.tensor([0.6, 0.4]), torch.tensor([0, 1.0]), weight=torch.tensor([0.2, 0.8])).item())
#     print(binary_cross_entropy_loss(torch.tensor([0.6, 0.4]), torch.tensor([0, 1.0]), weight=torch.tensor([0.5, 0.5])).item())
#     print(binary_cross_entropy_loss(torch.tensor([0.6, 0.4]), torch.tensor([0, 1.0]), weight=torch.tensor([0.8, 0.2])).item())
#     print('----------------')
#     print(torch.nn.BCELoss(reduction='mean')(output, target.to(torch.float)).item())
