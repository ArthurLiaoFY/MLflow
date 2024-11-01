import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split

from models.deep_models.models.conv_gru_att import ConvolutionalGRUAttention
from models.deep_models.models.tfc import TimeFrequencyConsistency
from models.deep_models.training.early_stopping import EarlyStopping
from models.deep_models.training.evaluate import (
    Accuracy,
    AreaUnderCurve,
    Precision,
    Recall,
)
from models.deep_models.training.loss import cross_entropy_loss
from models.deep_models.training.train_model import train_model
from models.deep_models.utils.prepare_data import to_dataloader
from models.deep_models.utils.tools import get_device

cudnn.benchmark = True


class TFCPretrain:
    def __init__(self, run_id, **kwargs):
        self.__dict__.update(kwargs)
        self.device = get_device()
        self.run_id = run_id
        self.loss_traj = []

        self.model = TimeFrequencyConsistency().to(device=self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.lr),
            betas=(
                float(self.beta1),
                float(self.beta2),
            ),
            weight_decay=self.weight_decay,
        )
        self.early_stopping = EarlyStopping(
            patience=int(self.early_stopping_patience),
        )

    def train_model(self, curve_array: np.ndarray, label_array: np.ndarray) -> None:

        train_x, test_x, train_y, test_y = train_test_split(
            curve_array[:, np.newaxis, :],
            label_array,
            test_size=float(self.validation_size),
            shuffle=True,
            random_state=int(self.seed),
            stratify=label_array,
        )

        # train_x, train_y = up_sampling(
        #     curve_array=train_x,
        #     label_array=train_y,
        #     seed=int(self.seed),
        # )

        tuned_model = train_model(
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
            loss_fn=cross_entropy_loss,
            evaluate_fns={
                "Accuracy": Accuracy(),
                "Precision": Precision(),
                "Recall": Recall(),
                # "AUC": AreaUnderCurve(),
            },
            optimizer=self.optimizer,
            early_stopping=EarlyStopping(
                patience=int(self.early_stopping_patience),
            ),
            log_file_path=self.log_file_path,
            epochs=int(self.epoch),
        )
        torch.save(tuned_model, f"{self.model_file_path}/{self.run_id}_model.pt")
        return None
