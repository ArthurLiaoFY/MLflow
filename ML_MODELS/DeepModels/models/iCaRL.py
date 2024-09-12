# %%
import numpy as np
import pandas as pd
import torch

from ML_MODELS.DeepModels.utils.prepare_data import to_tensor
from ML_MODELS.DeepModels.utils.tools import get_norm


class RepresentativeSample:
    def __init__(
        self, max_datasize: int = 2000, seed: int | None = 1122, method: str = "mean"
    ):
        self.max_datasize = max_datasize
        self.seed = seed
        self.method = method

    def construct_sample_set(
        self,
        x: np.ndarray | pd.DataFrame | torch.Tensor,
        y: np.ndarray | pd.DataFrame | torch.Tensor,
        nn_model: torch.nn.Module,
        norm: int = 2,
    ) -> dict:
        result = {}
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        samples_per_class = round(self.max_datasize / y.shape[1])

        nn_model.eval()
        y_pred = nn_model(to_tensor(x)).squeeze()

        # calculate mean per class
        for col_idx in range(y.shape[1]):
            print("class: ", col_idx)
            result[col_idx] = {}

            # select class == col_idx samples
            bool_mask = y[:, col_idx] == 1.0
            y_pred_masked = y_pred[bool_mask, :]

            # save class mean
            result[col_idx]["class_center"] = (
                y_pred_masked.mean(dim=0)
                if self.method == "mean"
                else y_pred_masked.median(dim=0)
            )

            for k in range(min(samples_per_class, y_pred_masked.shape[0])):
                if k == 0:
                    distance = get_norm(
                        x=y_pred_masked - result[col_idx]["class_center"], norm=norm
                    )
                    result[col_idx]["representative_sample"] = y_pred_masked[
                        torch.argmin(distance)
                    ][None, :]
                    y_pred_masked = y_pred_masked[
                        ~torch.all(
                            input=y_pred_masked
                            == y_pred_masked[torch.argmin(distance)],
                            dim=1,
                        ),
                        :,
                    ]

                else:
                    distance = get_norm(
                        x=y_pred_masked - result[col_idx]["class_center"], norm=norm
                    )
                    result[col_idx]["representative_sample"] = torch.cat(
                        tensors=(
                            result[col_idx]["representative_sample"],
                            y_pred_masked[torch.argmin(distance)][None, :],
                        ),
                        dim=0,
                    )
                    y_pred_masked = y_pred_masked[
                        ~torch.all(
                            input=y_pred_masked
                            == y_pred_masked[torch.argmin(distance)],
                            dim=1,
                        ),
                        :,
                    ]

        return result

    def update_sample_set(self, samples_per_class: int):
        pass


# %%
from ML_MODELS.DeepModels.utils.prepare_data import one_hot_encoding

model = torch.nn.Linear(in_features=50, out_features=10)
x_ = torch.randn(1000, 50)
target = one_hot_encoding(torch.argmax(torch.randn(1000, 10), dim=1))
output = model(x_)
rs = RepresentativeSample()
r = rs.construct_sample_set(x=x_, y=target, nn_model=model, norm=1)
r[1]
