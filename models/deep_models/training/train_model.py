from typing import Callable, Mapping

import torch
from torch.utils.data import DataLoader

import mlflow
from models.deep_models.training.early_stopping import EarlyStopping
from models.deep_models.training.evaluate import Accuracy, Precision, Recall, RSquare
from models.deep_models.utils.prepare_data import get_device


def train_model(
    nn_model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    loss_fn: Callable,
    evaluate_fns: Mapping[str, Accuracy | Recall | Precision | RSquare],
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    epochs: int = 100,
    seed: int | None = 1122,
) -> torch.nn.Module:
    device = get_device()
    mlflow.log_text(
        text=f"currently using device: {device}",
        artifact_file="log_file.txt",
    )
    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    for epoch in range(epochs):
        # training
        nn_model.train()
        training_loss = 0.0

        for train_x, train_y in train_dataloader:
            # Forward propagation
            train_y_pred = nn_model(train_x.to(device)).squeeze(
                dim=1
            )  # Make prediction by passing X to our model
            if train_y_pred.shape != train_y.shape:
                mlflow.log_text(
                    text=f"the shape of model prediction {tuple(train_y_pred.shape)} and y value {tuple(train_y.shape)}  is different ! "
                    "this might influence the performance of the model",
                    artifact_file="log_file.txt",
                )
                # check train_y_pred shape and train_y shape
            loss = loss_fn(train_y_pred, train_y.to(device))  # Calculate loss
            training_loss += loss.item()  # Add loss to running loss

            # Backward propagation
            optimizer.zero_grad()  # Empty the gradient (look up this function)
            loss.backward()  # Do backward propagation and calculate the gradient of loss w.r.t every parameters
            optimizer.step()  # Adjust parameters to minimize loss

        training_loss /= len(train_dataloader)
        # Append train loss

        # validation
        nn_model.eval()
        validation_loss = 0.0
        validation_eval = {fn_name: 0.0 for fn_name in evaluate_fns.keys()}

        with torch.no_grad():
            for valid_x, valid_y in valid_dataloader:
                valid_y_pred = nn_model(valid_x.to(device)).squeeze(
                    dim=1
                )  # Make prediction by passing X to our model
                loss = loss_fn(valid_y_pred, valid_y.to(device))  # Calculate loss
                validation_loss += loss.item()  # Add loss to running loss
                for evaluate_fn in evaluate_fns.values():
                    evaluate_fn.update(valid_y_pred, valid_y.to(device))

            for fn_name, evaluate_fn in evaluate_fns.items():
                validation_eval[fn_name] = evaluate_fn.finish().item()

            validation_loss /= len(valid_dataloader)
        mlflow.log_metric(key="training loss", value=f"{training_loss:4f}", step=epoch)
        mlflow.log_metric(
            key="validation loss", value=f"{validation_loss:4f}", step=epoch
        )
        for fn_name, evaluate_value in validation_eval.items():
            mlflow.log_metric(
                key=f"validation {fn_name}", value=f"{evaluate_value:4f}", step=epoch
            )

        early_stopping(val_loss=validation_loss, model_state_dict=nn_model.state_dict())
        if early_stopping.early_stop:
            mlflow.log_text(
                text="*" * 4
                + " Due to the early stopping mechanism, the training process is halted "
                + "*" * 4,
                artifact_file="log_file.txt",
            )

            break
    nn_model.load_state_dict(
        torch.load(early_stopping.best_model_state, weights_only=True),
    )
    return nn_model