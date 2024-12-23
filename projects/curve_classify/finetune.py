import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split

from ml_models.deep_models.models.conv_gru_att import ConvolutionalGRUAttention
from ml_models.deep_models.models.tfc import target_classifier
from ml_models.deep_models.training.early_stopping import EarlyStopping
from ml_models.deep_models.training.evaluate import (
    Accuracy,
    AreaUnderCurve,
    Precision,
    Recall,
)
from ml_models.deep_models.training.loss import cross_entropy_loss
from ml_models.deep_models.training.train_model import train_model
from ml_models.deep_models.utils.check_device import get_device
from ml_models.deep_models.utils.prepare_data import to_dataloader

cudnn.benchmark = True


class TFCFinetune:
    def __init__(self, run_id, **kwargs):
        self.__dict__.update(kwargs)
        self.device = get_device()
        self.run_id = run_id
        self.loss_traj = []

        self.model = target_classifier(
            num_classes_target=int(self.num_classes_target),
        ).to(device=self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.lr),
            betas=(
                float(self.beta1),
                float(self.beta2),
            ),
            weight_decay=float(self.weight_decay),
        )
        self.early_stopping = EarlyStopping(
            patience=int(self.early_stopping_patience),
        )

    def finetune_model(self) -> None:
        pass
