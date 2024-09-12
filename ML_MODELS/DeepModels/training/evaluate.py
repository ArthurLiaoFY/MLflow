import torch

from ML_MODELS.DeepModels.utils.prepare_data import get_device


class RSquare:
    def __init__(self):
        self.device = get_device()
        self.RSE, self.SSE = torch.tensor([0.0]).to(self.device), torch.tensor(
            [0.0]
        ).to(self.device)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.RSE += torch.pow((y_true - y_pred), 2).sum()
        self.SSE += torch.pow((y_true - y_true.mean()), 2).sum()

    def initialize(self):
        self.RSE, self.SSE = torch.tensor([0.0]).to(self.device), torch.tensor(
            [0.0]
        ).to(self.device)

    def finish(self) -> torch.Tensor:
        result = 1.0 - self.RSE / self.SSE
        self.initialize()
        return result


class Accuracy:
    def __init__(self):
        self.device = get_device()
        self.y_pred_idx, self.y_true_idx = torch.tensor([]).to(
            self.device
        ), torch.tensor([]).to(self.device)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_pred_idx = torch.cat((self.y_pred_idx, torch.argmax(y_pred, dim=1)))
        self.y_true_idx = torch.cat((self.y_true_idx, torch.argmax(y_true, dim=1)))

    def initialize(self):
        self.y_pred_idx, self.y_true_idx = torch.tensor([]).to(
            self.device
        ), torch.tensor([]).to(self.device)

    def finish(self) -> torch.Tensor:
        accuracy = (self.y_pred_idx == self.y_true_idx).to(torch.float).mean()
        self.initialize()
        return accuracy


class Recall:
    def __init__(self):
        self.device = get_device()
        self.y_pred_idx, self.y_true_idx = torch.tensor([]).to(
            self.device
        ), torch.tensor([]).to(self.device)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_pred_idx = torch.cat((self.y_pred_idx, torch.argmax(y_pred, dim=1)))
        self.y_true_idx = torch.cat((self.y_true_idx, torch.argmax(y_true, dim=1)))

    def initialize(self):
        self.y_pred_idx, self.y_true_idx = torch.tensor([]).to(
            self.device
        ), torch.tensor([]).to(self.device)

    def finish(self) -> torch.Tensor:
        recall = (
            self.y_pred_idx[self.y_true_idx == 0]
            == self.y_true_idx[self.y_true_idx == 0]
        ).sum() / (self.y_true_idx == 0).sum()
        self.initialize()
        return recall


class Precision:
    def __init__(self):
        self.device = get_device()
        self.y_pred_idx, self.y_true_idx = torch.tensor([]).to(
            self.device
        ), torch.tensor([]).to(self.device)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_pred_idx = torch.cat((self.y_pred_idx, torch.argmax(y_pred, dim=1)))
        self.y_true_idx = torch.cat((self.y_true_idx, torch.argmax(y_true, dim=1)))

    def initialize(self):
        self.y_pred_idx, self.y_true_idx = torch.tensor([]).to(
            self.device
        ), torch.tensor([]).to(self.device)

    def finish(self) -> torch.Tensor:
        precision = (
            self.y_pred_idx[self.y_true_idx == 0]
            == self.y_true_idx[self.y_true_idx == 0]
        ).sum() / (self.y_pred_idx == 0).sum()
        self.initialize()
        return precision


#
#
# # %%
# from ML_MODELS.DeepModels.utils.prepare_data import one_hot_encoding
#
# model = torch.nn.Sequential(
#     torch.nn.Linear(in_features=50, out_features=10),
#     torch.nn.Softmax(),
# )
# x = torch.randn(100, 50)
# target = one_hot_encoding(torch.argmax(torch.randn(100, 10), dim=1))
# # %%
# output = model(x)
# # %%
# output
# # %%
# acc = Accuracy()
# acc.update(output, target)
# print(acc.finish().item())
# # %%
