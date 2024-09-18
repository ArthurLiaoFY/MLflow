import io

import numpy as np
import torch

import mlflow


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int = 7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
        """
        self.patience = patience
        self.losses = []
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = 0.0
        self.counter = 0

    def __call__(self, log, val_loss: float, model_state_dict: dict):
        """
        early stopping mechanism
        counter += 1 if validation loss was improved, else reset the counter
        """

        if len(self.losses) == 0:
            self.losses.append(val_loss)
            self.__update_model_state(
                log=log, val_loss=val_loss, model_state_dict=model_state_dict
            )
        else:
            if val_loss < self.val_loss_min:
                self.losses.append(val_loss)
                self.__update_model_state(
                    log=log, val_loss=val_loss, model_state_dict=model_state_dict
                )
                if self.counter > 0:
                    self.counter = 0
                    log.write("[Early Stopping] Counter has been reset. \n")
                self.delta = np.std(self.losses[-20:])

            elif self.val_loss_min <= val_loss < self.val_loss_min + self.delta:
                self.losses.append(val_loss)
                log.write(
                    f"[Early Stopping] Validation loss {val_loss:.4f} between confidence interval: "
                    f"[{(self.val_loss_min + self.delta):.4f}, {(self.val_loss_min - self.delta if self.val_loss_min > self.delta else 0):.4f}]\n"
                )
                self.delta = np.std(self.losses[-20:])

            else:
                self.counter += 1
                log.write(
                    f"[Early Stopping] EarlyStopping counter: {self.counter} out of {self.patience} \n",
                )
                if self.counter >= self.patience:
                    self.early_stop = True

    def __update_model_state(
        self, log, val_loss: float, model_state_dict: dict[str, any]
    ):
        """
        Saves model when validation loss decrease.
        """
        log.write(
            f"[Early Stopping] Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). \n",
        )
        self.best_model_state = io.BytesIO()
        torch.save(obj=model_state_dict, f=self.best_model_state)
        self.best_model_state.seek(0)
        self.val_loss_min = val_loss
