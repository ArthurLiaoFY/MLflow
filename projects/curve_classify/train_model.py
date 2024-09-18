import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split

from models.deep_models.models.conv_gru_att import ConvolutionalGRUAttention
from models.deep_models.training.early_stopping import EarlyStopping
from models.deep_models.training.evaluate import RSquare
from models.deep_models.training.loss import root_mean_square_error
from models.deep_models.training.train_model import train_model
from models.deep_models.utils.prepare_data import to_dataloader

cudnn.benchmark = True


class CurveClassify:
    def __init__(self, run_id, **kwargs):
        self.run_id = run_id
        self.__dict__.update(kwargs)

    def train_model(
        self, run_id: str, curve_array: np.ndarray, label_array: np.ndarray
    ) -> None:
        train_x, test_x, train_y, test_y = train_test_split(
            curve_array,
            label_array,
            test_size=float(self.validation_size),
            shuffle=True,
            random_state=int(self.seed),
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
            run_id=run_id,
            nn_model=model,
            train_dataloader=to_dataloader(train_x, train_y, shuffle=True),
            valid_dataloader=to_dataloader(test_x, test_y, shuffle=False),
            loss_fn=root_mean_square_error,
            evaluate_fns={"R-square": RSquare()},
            optimizer=torch.optim.Adam(model.parameters(), lr=self.learning_rate),
            early_stopping=EarlyStopping(patience=self.early_stopping_patience),
            epochs=self.epoch,
        )
        torch.save(tuned_model, f"{self.model_file_path}/{self.run_id}_model.pt")
        return None
