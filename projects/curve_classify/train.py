import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split

from models.deep_models.models.conv_gru_att import ConvolutionalGRUAttention
from models.deep_models.training.early_stopping import EarlyStopping
from models.deep_models.training.evaluate import Accuracy, Precision, Recall
from models.deep_models.training.loss import binary_cross_entropy_loss
from models.deep_models.training.train_model import train_model
from models.deep_models.utils.prepare_data import to_dataloader
from projects.curve_classify.balance_data import up_sampling

cudnn.benchmark = True


class CurveClassify:
    def __init__(self, run_id, **kwargs):
        self.run_id = run_id
        self.__dict__.update(kwargs)

    def train_model(self, curve_array: np.ndarray, label_array: np.ndarray) -> None:
        train_x, test_x, train_y, test_y = train_test_split(
            curve_array[:, np.newaxis, :],
            np.array([[1.0, 0.0] if res == -1 else [0.0, 1.0] for res in label_array]),
            test_size=float(self.validation_size),
            shuffle=True,
            random_state=int(self.seed),
            stratify=label_array,
        )

        dup_curve_array, dup_label_array = up_sampling(
            curve_array=train_x,
            label_array=train_y,
            seed=int(self.seed),
        )
        model = ConvolutionalGRUAttention(
            conv_in_channels=int(self.conv_in_channels),  # C in shape : (B, C, H)
            conv_out_channels=int(self.conv_out_channels),  # C' in shape : (B, C', H)
            gru_input_size=curve_array.shape[1],
            gru_hidden_size=int(self.gru_hidden_size),
            gru_layer_amount=int(self.gru_layer_amount),
            attention_num_of_head=int(self.attention_num_of_head),
            out_feature_size=int(self.out_feature_size),
        )

        tuned_model = train_model(
            run_id=self.run_id,
            nn_model=model,
            train_dataloader=to_dataloader(
                dup_curve_array,
                dup_label_array,
                batch_size=int(self.batch_size),
                shuffle=True,
            ),
            valid_dataloader=to_dataloader(
                test_x,
                test_y,
                batch_size=int(self.batch_size),
                shuffle=False,
            ),
            loss_fn=binary_cross_entropy_loss,
            evaluate_fns={
                "Accuracy": Accuracy(),
                "Precision": Precision(),
                "Recall": Recall(),
            },
            optimizer=torch.optim.Adam(
                model.parameters(), lr=float(self.learning_rate)
            ),
            early_stopping=EarlyStopping(
                patience=int(self.early_stopping_patience),
            ),
            log_file_path=self.log_file_path,
            epochs=int(self.epoch),
        )
        torch.save(tuned_model, f"{self.model_file_path}/{self.run_id}_model.pt")
        return None
