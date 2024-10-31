# %%
import torch

from models.deep_models.utils.prepare_data import get_device


# regression
class RSquare:
    def __init__(self):
        self.device = get_device()
        self.initialize()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.RSE += torch.pow((y_true - y_pred), 2).sum()
        self.SSE += torch.pow((y_true - y_true.mean()), 2).sum()

    def initialize(self):
        self.RSE = torch.tensor([0.0]).to(self.device)
        self.SSE = torch.tensor([0.0]).to(self.device)

    def finish(self) -> torch.Tensor:
        result = 1.0 - self.RSE / self.SSE
        self.initialize()
        return result


# classify
class Accuracy:
    def __init__(self):
        self.device = get_device()
        self.initialize()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_pred_idx = torch.cat((self.y_pred_idx, torch.argmax(y_pred, dim=-1)))
        if len(self.y_true_idx.shape) == 1:
            self.y_true_idx = torch.cat((self.y_true_idx, y_true))
        else:
            self.y_true_idx = torch.cat((self.y_true_idx, torch.argmax(y_true, dim=-1)))

    def initialize(self):
        self.y_pred_idx = torch.tensor([]).to(self.device)
        self.y_true_idx = torch.tensor([]).to(self.device)

    def finish(self) -> torch.Tensor:
        accuracy = (self.y_pred_idx == self.y_true_idx).to(torch.float).mean()
        self.initialize()
        return accuracy


class Recall:
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self.device = get_device()
        self.initialize()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_pred_array = torch.cat(
            (
                self.y_pred_array,
                torch.zeros_like(y_pred).scatter_(
                    index=torch.argmax(y_pred, dim=-1), value=1.0, dim=-1
                ),
            )
        )
        if len(y_true.shape) == 1:
            self.y_true_array = torch.cat(
                (
                    self.y_true_array,
                    torch.zeros_like(y_pred).scatter_(index=y_true, value=1.0, dim=-1),
                )
            )
        else:
            self.y_true_array = torch.cat((self.y_true_array, y_true))

    def initialize(self):
        self.y_pred_array = torch.tensor([]).to(self.device)
        self.y_true_array = torch.tensor([]).to(self.device)

    def finish(self) -> torch.Tensor:
        recall = (
            (self.y_pred_array * self.y_true_array).sum(dim=0)
            / (self.y_true_array.sum(dim=0) + self.epsilon)
        ).mean()  # Epsilon to avoid division by zero
        self.initialize()
        return recall


class Precision:
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self.device = get_device()
        self.initialize()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_pred_array = torch.cat(
            (
                self.y_pred_array,
                torch.zeros_like(y_pred).scatter_(
                    index=torch.argmax(y_pred, dim=-1), value=1.0, dim=-1
                ),
            )
        )
        if len(y_true.shape) == 1:
            self.y_true_array = torch.cat(
                (
                    self.y_true_array,
                    torch.zeros_like(y_pred).scatter_(index=y_true, value=1.0, dim=-1),
                )
            )
        else:
            self.y_true_array = torch.cat((self.y_true_array, y_true))

    def initialize(self):
        self.y_pred_array = torch.tensor([]).to(self.device)
        self.y_true_array = torch.tensor([]).to(self.device)

    def finish(self) -> torch.Tensor:
        precision = (
            (self.y_pred_array * self.y_true_array).sum(dim=0)
            / (self.y_pred_array.sum(dim=0) + self.epsilon)
        ).mean()  # Epsilon to avoid division by zero
        self.initialize()
        return precision


class AreaUnderCurve:
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self.device = get_device()
        self.initialize()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_pred_prob = torch.cat((self.y_pred_prob, y_pred[:, 1]))
        if len(self.y_true_idx.shape) == 1:
            self.y_true_idx = torch.cat((self.y_true_idx, y_true))
        else:
            self.y_true_idx = torch.cat((self.y_true_idx, y_true[:, 1]))

    def initialize(self):
        self.y_pred_prob = torch.tensor([]).to(self.device)
        self.y_true_idx = torch.tensor([]).to(self.device)

    def finish(self) -> torch.Tensor:
        sorted_indices = torch.argsort(self.y_pred_prob, descending=True)
        sorted_true_idx = self.y_true_idx[sorted_indices]

        tpr_list = []
        fpr_list = []
        pos_count = (sorted_true_idx == 1).sum().item()
        neg_count = (sorted_true_idx == 0).sum().item()

        tp, fp = 0, 0

        for is_true in sorted_true_idx:
            if is_true == 1:
                tp += 1
            else:
                fp += 1

            tpr_list.append(tp / pos_count)
            fpr_list.append(fp / neg_count)

        tpr = torch.tensor(tpr_list)
        fpr = torch.tensor(fpr_list)
        auc = torch.trapz(tpr, fpr)

        self.initialize()
        return auc
