import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import mlflow
from ML_MODELS.DeepModels.models.regression import UphSimulate
from ML_MODELS.DeepModels.training.early_stopping import EarlyStopping
from ML_MODELS.DeepModels.training.evaluate import RSquare
from ML_MODELS.DeepModels.training.loss import root_mean_square_error
from ML_MODELS.DeepModels.training.train_model import train_L3_model
from ML_MODELS.DeepModels.utils.prepare_data import to_dataloader

cudnn.benchmark = True


class UphTrainSimulationModel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __fit(
        self, masked_simulate_array: np.ndarray, masked_uph_array: np.ndarray
    ) -> torch.nn.Module | XGBRegressor:
        train_x, test_x, train_y, test_y = train_test_split(
            masked_simulate_array,
            masked_uph_array,
            test_size=self.L3_validation_size,
            shuffle=True,
            random_state=self.seed,
        )
        mlflow.set_tag("model_type", self.model_type)

        match self.model_type:
            case "NN":

                model = UphSimulate(
                    in_features=train_x.shape[1],
                    out_features=1,
                    max_uph=masked_uph_array.max(),
                )
                tuned_model = train_L3_model(
                    nn_model=model,
                    train_dataloader=to_dataloader(train_x, train_y, shuffle=True),
                    valid_dataloader=to_dataloader(test_x, test_y, shuffle=False),
                    loss_fn=root_mean_square_error,
                    evaluate_fns={"R-square": RSquare()},
                    optimizer=torch.optim.Adam(
                        model.parameters(), lr=self.L3_learning_rate
                    ),
                    early_stopping=EarlyStopping(
                        patience=self.L3_early_stopping_patience
                    ),
                    epochs=self.L3_epoch,
                )
            case "XGB":
                tuned_model = XGBRegressor(seed=self.seed)
                tuned_model.fit(
                    train_x,
                    train_y,
                )

                pred_train_y = tuned_model.predict(train_x)
                pred_test_y = tuned_model.predict(test_x)

                mlflow.log_metric(
                    key="training loss",
                    value=f"{np.sqrt(np.pow((pred_train_y - train_y), 2).mean()):4f}",
                    step=0,
                )
                mlflow.log_metric(
                    key="validation loss",
                    value=f"{np.sqrt(np.pow((pred_test_y - test_y), 2).mean()):4f}",
                    step=0,
                )
                mlflow.log_metric(
                    key="validation R-square",
                    value=f"{1.0 - sum(np.pow(test_y - pred_test_y, 2))/sum(np.pow(test_y - test_y.mean(), 2)):4f}",
                    step=0,
                )

            case _:
                raise NotImplementedError

        return tuned_model

    def train_model(
        self,
        masked_uph_dict: dict,
        masked_simulate_dict: dict,
    ) -> None:
        tuned_model = self.__fit(
            masked_simulate_array=np.array(masked_simulate_dict.get("data")),
            masked_uph_array=np.array(masked_uph_dict.get("data")),
        )

        # model_file_name = f"L3_model_{self.db_address}_{self.db_name}_{self.region_id}_{self.factory_id}_{self.owner_id}_{self.line_id}_{self.sector_id}"
        # model_infos_file_name = f"L3_model_infos_{self.db_address}_{self.db_name}_{self.region_id}_{self.factory_id}_{self.owner_id}_{self.line_id}_{self.sector_id}.pkl"

        with open(f"{self.model_file_path}/{self.run_id}_infos.pkl", "wb") as f:
            pickle.dump(
                {
                    "in_features": masked_simulate_dict.get("columns"),
                    "out_features": 1,
                    "model_type": self.model_type,
                    "max_uph": np.array(masked_uph_dict.get("data")).max(),
                },
                f,
            )

        match self.model_type:
            case "NN":
                torch.save(
                    tuned_model, f"{self.model_file_path}/{self.run_id}_model.pt"
                )
                # torch.jit.script(tuned_model).save(
                #     f"{self.model_file_path}/{run_id}_model.pt"
                # )
            case "XGB":
                with open(f"{self.model_file_path}/{self.run_id}_model.pkl", "wb") as f:
                    pickle.dump(tuned_model, f)
            case _:
                raise NotImplementedError

        return None
