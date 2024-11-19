import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ml_models.deep_models.models.basic_regression import LinearLReluStack
from ml_models.deep_models.training.early_stopping import EarlyStopping
from ml_models.deep_models.training.evaluate import RSquare
from ml_models.deep_models.training.inference_model import inference_model
from ml_models.deep_models.training.loss import root_mean_square_error
from ml_models.deep_models.training.train_model import train_model
from ml_models.deep_models.utils.prepare_data import to_dataloader, to_tensor


class EstimateSurface:
    def __init__(self, run_id, in_feature, **kwargs) -> None:
        self.run_id = run_id
        self.in_feature = in_feature
        self.__dict__.update(kwargs)
        self.model = LinearLReluStack(
            in_features=int(self.in_feature), out_features=int(self.out_feature)
        )
        self.load_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.lr),
        )

    def load_model(self):
        try:
            self.model.load_state_dict(
                torch.load(
                    f"{self.model_file_path}/{self.run_id}_model.pt", weights_only=True
                )
            )
        except FileNotFoundError:
            pass

    def fit_surface(self, X: np.ndarray, y: np.ndarray) -> None:
        train_x, test_x, train_y, test_y = train_test_split(
            X,
            y,
            test_size=float(self.validation_size),
            shuffle=True,
            random_state=int(self.seed),
        )

        self.model = train_model(
            run_id=self.run_id,
            nn_model=self.model,
            train_dataloader=to_dataloader(
                train_x,
                train_y,
                batch_size=int(self.batch_size),
                shuffle=True,
            ),
            valid_dataloader=to_dataloader(
                test_x,
                test_y,
                batch_size=int(self.batch_size),
                shuffle=False,
            ),
            loss_fn=root_mean_square_error,
            evaluate_fns={
                "R-Square": RSquare(),
            },
            optimizer=self.optimizer,
            early_stopping=EarlyStopping(
                patience=int(self.early_stopping_patience),
            ),
            log_file_path=self.log_file_path,
            epochs=int(self.epoch),
            mlflow_tracking=False,
        )
        torch.save(
            self.model.state_dict(), f"{self.model_file_path}/{self.run_id}_model.pt"
        )

    def pred_surface(self, valid_X: np.ndarray) -> np.ndarray:
        return inference_model(
            nn_model=self.model, test_dataloader=to_dataloader(valid_X, shuffle=False)
        )
