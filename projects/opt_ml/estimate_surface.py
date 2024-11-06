import numpy as np
import torch
from sklearn.model_selection import train_test_split

from models.deep_models.models.basic_regression import LinearLReluStack
from models.deep_models.training.early_stopping import EarlyStopping
from models.deep_models.training.evaluate import RSquare
from models.deep_models.training.loss import root_mean_square_error
from models.deep_models.training.train_model import train_model
from models.deep_models.utils.prepare_data import to_dataloader, to_tensor


class EstimateSurface:
    def __init__(self, run_id, **kwargs) -> None:
        self.run_id = run_id
        self.__dict__.update(kwargs)
        self.model = LinearLReluStack(
            in_features=int(self.in_feature), out_features=int(self.out_feature)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.lr),
        )

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
        torch.save(self.model, f"{self.model_file_path}/{self.run_id}_model.pt")

    def pred_surface(self, valid_X: np.ndarray) -> np.ndarray:
        return self.model(to_tensor(valid_X)).detach().cpu().numpy()
