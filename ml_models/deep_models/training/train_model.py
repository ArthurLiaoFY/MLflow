from datetime import datetime
from typing import Callable, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow
from ml_models.deep_models.training.early_stopping import EarlyStopping
from ml_models.deep_models.training.evaluate import (
    Accuracy,
    AreaUnderCurve,
    Precision,
    Recall,
    RSquare,
)
from ml_models.deep_models.utils.prepare_data import get_device


def train_model(
    run_id: str,
    nn_model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    loss_fn: Callable,
    evaluate_fns: Mapping[
        str, Accuracy | Recall | Precision | RSquare | AreaUnderCurve
    ],
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    log_file_path: str,
    epochs: int = 100,
    seed: int | None = 1122,
    mlflow_tracking: bool = True,
) -> torch.nn.Module:

    device = get_device()
    log_file_path = f"{log_file_path}/{run_id}_log_file.txt"

    log = open(log_file_path, "w")
    log.write(f"currently using device: {device}\n")

    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    for epoch in range(epochs):
        start = datetime.now()
        # training
        nn_model.train()
        training_loss = 0.0

        for train_x, train_y in train_dataloader:
            # Forward propagation
            train_y_pred = nn_model(train_x.to(device)).squeeze(
                dim=1
            )  # Make prediction by passing X to our model

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
                    evaluate_fn.update(y_pred=valid_y_pred, y_true=valid_y.to(device))

            for fn_name, evaluate_fn in evaluate_fns.items():
                validation_eval[fn_name] = evaluate_fn.finish().item()

            validation_loss /= len(valid_dataloader)

        cost_time = datetime.now() - start
        log.write("-" * 80 + "\n")
        validate_massage = (
            f"Epoch: {epoch}, Use Time: {cost_time.seconds}.{cost_time.microseconds} Second. \n"
            + f"Train loss: {round(training_loss, 4)}; "
            + f"Valid loss: {round(validation_loss, 4)}; "
            + "; ".join(
                [
                    f"Valid {fn_name}: {round(evaluate_value, 4)}"
                    for fn_name, evaluate_value in validation_eval.items()
                ]
            )
            + "\n"
        )
        print(validate_massage)
        log.write(validate_massage)
        if mlflow_tracking:
            mlflow.log_metric(
                key="training loss",
                value=f"{training_loss:4f}",
                step=epoch,
                run_id=run_id,
            )
            mlflow.log_metric(
                key="validation loss",
                value=f"{validation_loss:4f}",
                step=epoch,
                run_id=run_id,
            )
            for fn_name, evaluate_value in validation_eval.items():
                mlflow.log_metric(
                    key=f"validation {fn_name}",
                    value=f"{evaluate_value:4f}",
                    step=epoch,
                    run_id=run_id,
                )

        early_stopping(
            log=log,
            val_loss=validation_loss,
            model_state_dict=nn_model.state_dict(),
        )
        if early_stopping.early_stop:
            log.write(
                "*" * 4
                + " Due to the early stopping mechanism, the training process is halted "
                + "*" * 4
                + "\n"
            )

            break
    nn_model.load_state_dict(
        torch.load(early_stopping.best_model_state, weights_only=True),
    )
    # mlflow.log_artifact(
    #     local_path=log_file_path,
    #     run_id=run_id,
    # )
    return nn_model


def finetune_llm_model(
    run_id: str,
    llm_model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    loss_fn: Callable,
    evaluate_fns: Mapping[
        str, Accuracy | Recall | Precision | RSquare | AreaUnderCurve
    ],
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    log_file_path: str,
    epochs: int = 100,
    seed: int | None = 1122,
    mlflow_tracking: bool = True,
) -> torch.nn.Module:

    device = get_device()
    log_file_path = f"{log_file_path}/{run_id}_log_file.txt"

    log = open(log_file_path, "w")
    log.write(f"currently using device: {device}\n")

    llm_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    for epoch in range(epochs):
        start = datetime.now()
        # training
        llm_model.train()
        training_loss = 0.0

        for train_batch in tqdm(train_dataloader, desc="Training"):
            # Forward propagation
            train_y = train_batch["labels"].to(device)
            train_y_pred = llm_model(
                input_ids=train_batch["input_ids"].to(device),
                attention_mask=train_batch["attention_mask"].to(device),
            )  # Make prediction by passing X to our model
            # check train_y_pred shape and train_y shape
            loss = loss_fn(train_y_pred.logits, train_y.to(device))  # Calculate loss
            training_loss += loss.item()  # Add loss to running loss

            # Backward propagation
            optimizer.zero_grad()  # Empty the gradient (look up this function)
            loss.backward()  # Do backward propagation and calculate the gradient of loss w.r.t every parameters
            optimizer.step()  # Adjust parameters to minimize loss

        training_loss /= len(train_dataloader)
        # Append train loss

        # validation
        llm_model.eval()
        validation_loss = 0.0
        validation_eval = {fn_name: 0.0 for fn_name in evaluate_fns.keys()}

        with torch.no_grad():
            for valid_batch in tqdm(valid_dataloader, desc="Validating"):
                valid_y = valid_batch["labels"].to(device)
                valid_y_pred = llm_model(
                    input_ids=valid_batch["input_ids"].to(device),
                    attention_mask=valid_batch["attention_mask"].to(device),
                )  # Make prediction by passing X to our model
                loss = loss_fn(
                    valid_y_pred.logits, valid_y.to(device)
                )  # Calculate loss
                validation_loss += loss.item()  # Add loss to running loss
                for evaluate_fn in evaluate_fns.values():
                    evaluate_fn.update(
                        y_pred=valid_y_pred.logits, y_true=valid_y.to(device)
                    )

            for fn_name, evaluate_fn in evaluate_fns.items():
                validation_eval[fn_name] = evaluate_fn.finish().item()

            validation_loss /= len(valid_dataloader)

        cost_time = datetime.now() - start
        log.write("-" * 80 + "\n")
        validate_massage = (
            f"Epoch: {epoch}, Use Time: {cost_time.seconds}.{cost_time.microseconds} Second. \n"
            + f"Train loss: {round(training_loss, 4)}; "
            + f"Valid loss: {round(validation_loss, 4)}; "
            + "; ".join(
                [
                    f"Valid {fn_name}: {round(evaluate_value, 4)}"
                    for fn_name, evaluate_value in validation_eval.items()
                ]
            )
            + "\n"
        )
        print(validate_massage)
        log.write(validate_massage)
        if mlflow_tracking:
            mlflow.log_metric(
                key="training loss",
                value=f"{training_loss:4f}",
                step=epoch,
                run_id=run_id,
            )
            mlflow.log_metric(
                key="validation loss",
                value=f"{validation_loss:4f}",
                step=epoch,
                run_id=run_id,
            )
            for fn_name, evaluate_value in validation_eval.items():
                mlflow.log_metric(
                    key=f"validation {fn_name}",
                    value=f"{evaluate_value:4f}",
                    step=epoch,
                    run_id=run_id,
                )

        early_stopping(
            log=log,
            val_loss=validation_loss,
            model_state_dict=llm_model.state_dict(),
        )
        if early_stopping.early_stop:
            log.write(
                "*" * 4
                + " Due to the early stopping mechanism, the training process is halted "
                + "*" * 4
                + "\n"
            )

            break
    llm_model.load_state_dict(
        torch.load(early_stopping.best_model_state, weights_only=True),
    )
    # mlflow.log_artifact(
    #     local_path=log_file_path,
    #     run_id=run_id,
    # )
    return llm_model
