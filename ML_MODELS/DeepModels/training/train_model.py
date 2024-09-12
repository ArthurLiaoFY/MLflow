from typing import Callable, Mapping

import torch
from torch.utils.data import DataLoader

import mlflow
from ML_MODELS.DeepModels.training.early_stopping import EarlyStopping
from ML_MODELS.DeepModels.training.evaluate import Accuracy, Precision, Recall, RSquare
from ML_MODELS.DeepModels.utils.prepare_data import get_device


def train_model(
    nn_model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    loss_fn: Callable,
    evaluate_fns: Mapping[str, Accuracy | Recall | Precision | RSquare] | None,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    epochs: int = 100,
    seed: int | None = 1122,
) -> torch.nn.Module:
    device = get_device()
    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for epoch in range(epochs):
        # training
        nn_model.train()
        training_loss = 0.0

        for train_x, train_y in train_dataloader:
            # Forward propagation
            train_y_pred = nn_model(train_x).squeeze(
                dim=1
            )  # Make prediction by passing X to our model
            if train_y_pred.shape != train_y.shape:
                print(
                    "the shape of model prediction and y value is different ! "
                    "this might influence the performance of the model"
                )
                # check train_y_pred shape and train_y shape
            loss = loss_fn(train_y_pred, train_y)  # Calculate loss
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
        validation_eval = (
            None
            if evaluate_fn is None
            else {fn_name: 0.0 for fn_name in evaluate_fns.keys()}
        )

        with torch.no_grad():
            for valid_x, valid_y in valid_dataloader:
                valid_y_pred = nn_model(valid_x).squeeze(
                    dim=1
                )  # Make prediction by passing X to our model
                loss = loss_fn(valid_y_pred, valid_y)  # Calculate loss
                validation_loss += loss.item()  # Add loss to running loss
                for evaluate_fn in evaluate_fns.values():
                    evaluate_fn.update(valid_y_pred, valid_y)
            if evaluate_fn is not None:
                for fn_name, evaluate_fn in evaluate_fns.items():
                    validation_eval[fn_name] = evaluate_fn.finish().item()

            validation_loss /= len(valid_dataloader)
        print("-" * 80)
        print(
            f"Epoch: {epoch} \n"
            + f"Train loss: {round(training_loss, 4)}; "
            + f"Valid loss: {round(validation_loss, 4)}; "
            + ""
            if evaluate_fn is None
            else "; ".join(
                [
                    f"Valid {fn_name}: {round(evaluate_value, 4)}"
                    for fn_name, evaluate_value in validation_eval.items()
                ]
            )
        )

        early_stopping(val_loss=validation_loss, model_state_dict=nn_model.state_dict())
        if early_stopping.early_stop:
            print(
                "\n"
                + "*" * 4
                + " Due to the early stopping mechanism, the training process is halted "
                + "*" * 4
            )
            break
    nn_model.load_state_dict(torch.load(early_stopping.best_model_state))
    return nn_model


def train_cluster_model(
    nn_model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    loss_fn: Callable,
    evaluate_fns: Mapping[str, Accuracy | Recall | Precision | RSquare] | None,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    epochs: int = 100,
    seed: int | None = 1122,
) -> torch.nn.Module:
    device = get_device()
    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for epoch in range(epochs):
        # training
        nn_model.train()
        training_loss = 0.0

        for train_distribution in train_dataloader:
            # Forward propagation
            train_distribution_pred = nn_model(train_distribution[0]).squeeze(
                dim=1
            )  # Make prediction by passing X to our model
            train_p_distribution = torch.pow(
                train_distribution_pred, 2
            ) / train_distribution_pred.sum(dim=0, keepdim=True)
            train_p_distribution /= train_p_distribution.sum(dim=1, keepdim=True)

            loss = loss_fn(
                train_distribution_pred, train_p_distribution
            )  # Calculate loss
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
        validation_eval = (
            None
            if evaluate_fns is None
            else {fn_name: 0.0 for fn_name in evaluate_fns.keys()}
        )

        with torch.no_grad():
            for valid_distribution in valid_dataloader:
                valid_distribution_pred = nn_model(valid_distribution[0]).squeeze(
                    dim=1
                )  # Make prediction by passing X to our model
                valid_p_distribution = torch.pow(
                    valid_distribution_pred, 2
                ) / valid_distribution_pred.sum(dim=0, keepdim=True)
                valid_p_distribution /= valid_p_distribution.sum(dim=1, keepdim=True)

                loss = loss_fn(
                    valid_distribution_pred, valid_p_distribution
                )  # Calculate loss
                validation_loss += loss.item()  # Add loss to running loss
                if evaluate_fns is not None:
                    for evaluate_fn in evaluate_fns.values():
                        evaluate_fn.update(
                            valid_distribution_pred, valid_p_distribution
                        )
            if evaluate_fns is not None:
                for fn_name, evaluate_fn in evaluate_fns.items():
                    validation_eval[fn_name] = evaluate_fn.finish().item()

            validation_loss /= len(valid_dataloader)
        print("-" * 80)
        print(
            f"Epoch: {epoch} \n"
            + f"Train loss: {round(training_loss, 4)}; "
            + f"Valid loss: {round(validation_loss, 4)}; "
            + ""
            if evaluate_fns is None
            else "; ".join(
                [
                    f"Valid {fn_name}: {round(evaluate_value, 4)}"
                    for fn_name, evaluate_value in validation_eval.items()
                ]
            )
        )

        early_stopping(val_loss=validation_loss, model_state_dict=nn_model.state_dict())
        if early_stopping.early_stop:
            print(
                "\n"
                + "*" * 4
                + " Due to the early stopping mechanism, the training process is halted "
                + "*" * 4
            )
            break
    nn_model.load_state_dict(torch.load(early_stopping.best_model_state))
    return nn_model


def train_autoencoder_model(
    nn_model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    loss_fn: Callable,
    evaluate_fns: Mapping[str, Accuracy | Recall | Precision | RSquare] | None,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    epochs: int = 100,
    seed: int | None = 1122,
) -> torch.nn.Module:
    device = get_device()
    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for epoch in range(epochs):
        # training
        nn_model.train()
        training_loss = 0.0

        for train_distribution in train_dataloader:
            # Forward propagation
            train_distribution_pred = nn_model(train_distribution[0]).squeeze(
                dim=1
            )  # Make prediction by passing X to our model
            # train_distribution = torch.pow(train_distribution_pred, 2)/train_distribution_pred.sum(dim=0)
            # train_distribution /= train_distribution.sum(dim=1)

            loss = loss_fn(
                train_distribution_pred, train_distribution[0]
            )  # Calculate loss
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
        validation_eval = (
            None
            if evaluate_fns is None
            else {fn_name: 0.0 for fn_name in evaluate_fns.keys()}
        )

        with torch.no_grad():
            for valid_distribution in valid_dataloader:
                valid_distribution_pred = nn_model(valid_distribution[0]).squeeze(
                    dim=1
                )  # Make prediction by passing X to our model
                # valid_distribution = torch.pow(valid_distribution_pred, 2)/valid_distribution_pred.sum(dim=0)
                # valid_distribution /= valid_distribution.sum(dim=1)

                loss = loss_fn(
                    valid_distribution_pred, valid_distribution[0]
                )  # Calculate loss
                validation_loss += loss.item()  # Add loss to running loss
                if evaluate_fns is not None:
                    for evaluate_fn in evaluate_fns.values():
                        evaluate_fn.update(
                            valid_distribution_pred, valid_distribution[0]
                        )
            if evaluate_fns is not None:
                for fn_name, evaluate_fn in evaluate_fns.items():
                    validation_eval[fn_name] = evaluate_fn.finish().item()

            validation_loss /= len(valid_dataloader)
        print("-" * 80)
        print(
            f"Epoch: {epoch} \n"
            + f"Train loss: {round(training_loss, 4)}; "
            + f"Valid loss: {round(validation_loss, 4)}; "
            + ""
            if evaluate_fns is None
            else "; ".join(
                [
                    f"Valid {fn_name}: {round(evaluate_value, 4)}"
                    for fn_name, evaluate_value in validation_eval.items()
                ]
            )
        )

        early_stopping(val_loss=validation_loss, model_state_dict=nn_model.state_dict())
        if early_stopping.early_stop:
            print(
                "\n"
                + "*" * 4
                + " Due to the early stopping mechanism, the training process is halted "
                + "*" * 4
            )
            break
    nn_model.load_state_dict(torch.load(early_stopping.best_model_state))
    return nn_model


def train_L5_model(
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
    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for epoch in range(epochs):
        # training
        nn_model.train()
        training_loss = 0.0

        for (
            train_current_additional_info,
            train_lagged_eqp_status_block,
            train_current_sn_cnt,
        ) in train_dataloader:
            # Forward propagation
            train_y_pred = nn_model(
                train_current_additional_info.to(device),
                train_lagged_eqp_status_block.to(device),
            ).squeeze(
                dim=1
            )  # Make prediction by passing X to our model
            if train_y_pred.shape != train_current_sn_cnt.shape:
                print(
                    "the shape of model prediction and y value is different ! "
                    "this might influence the performance of the model"
                )
                # check train_y_pred shape and train_current_sn_cnt shape
            loss = loss_fn(
                train_y_pred, train_current_sn_cnt.to(device)
            )  # Calculate loss
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
            for (
                valid_current_additional_info,
                valid_lagged_eqp_status_block,
                valid_current_cn_cnt,
            ) in valid_dataloader:
                valid_y_pred = nn_model(
                    valid_current_additional_info.to(device),
                    valid_lagged_eqp_status_block.to(device),
                ).squeeze(
                    dim=1
                )  # Make prediction by passing X to our model
                loss = loss_fn(
                    valid_y_pred, valid_current_cn_cnt.to(device)
                )  # Calculate loss
                validation_loss += loss.item()  # Add loss to running loss
                for evaluate_fn in evaluate_fns.values():
                    evaluate_fn.update(valid_y_pred, valid_current_cn_cnt.to(device))

            for fn_name, evaluate_fn in evaluate_fns.items():
                validation_eval[fn_name] = evaluate_fn.finish().item()

            validation_loss /= len(valid_dataloader)
        print("-" * 80)
        print(
            f"Epoch: {epoch} \n"
            + f"Train loss: {round(training_loss, 4)}; "
            + f"Valid loss: {round(validation_loss, 4)}; "
            + "; ".join(
                [
                    f"Valid {fn_name}: {round(evaluate_value, 4)}"
                    for fn_name, evaluate_value in validation_eval.items()
                ]
            )
        )

        early_stopping(val_loss=validation_loss, model_state_dict=nn_model.state_dict())
        if early_stopping.early_stop:
            print(
                "\n"
                + "*" * 4
                + " Due to the early stopping mechanism, the training process is halted "
                + "*" * 4
            )
            break
    nn_model.load_state_dict(torch.load(early_stopping.best_model_state))
    return nn_model


def train_L5_model_dev(
    nn_model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    loss_fn: Callable,
    evaluate_fns: Mapping[str, Accuracy | Recall | Precision | RSquare],
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping | None = None,
    epochs: int = 100,
    seed: int | None = 1122,
) -> torch.nn.Module:
    device = get_device()
    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for epoch in range(epochs):
        # training
        nn_model.train()
        training_loss = 0.0

        for (
            train_lagged_sn_cnt,
            train_additional_info,
            train_lagged_eqp_status_block,
            train_current_sn_cnt,
        ) in train_dataloader:
            # Forward propagation
            train_y_pred = nn_model(
                train_lagged_sn_cnt.to(device),
                train_additional_info.to(device),
                train_lagged_eqp_status_block.to(device),
            ).squeeze(
                dim=1
            )  # Make prediction by passing X to our model
            if train_y_pred.shape != train_current_sn_cnt.shape:
                print(
                    f"the shape of model prediction {tuple(train_y_pred.shape)} and y value {tuple(train_current_sn_cnt.shape)} is different ! "
                    "this might influence the performance of the model"
                )
                # check train_y_pred shape and train_current_sn_cnt shape
            loss = loss_fn(
                train_y_pred, train_current_sn_cnt.to(device)
            )  # Calculate loss
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
            for (
                valid_lagged_cn_cnt,
                valid_additional_info,
                valid_lagged_eqp_status_block,
                valid_current_cn_cnt,
            ) in valid_dataloader:
                valid_y_pred = nn_model(
                    valid_lagged_cn_cnt.to(device),
                    valid_additional_info.to(device),
                    valid_lagged_eqp_status_block.to(device),
                ).squeeze(
                    dim=1
                )  # Make prediction by passing X to our model
                loss = loss_fn(
                    valid_y_pred, valid_current_cn_cnt.to(device)
                )  # Calculate loss
                validation_loss += loss.item()  # Add loss to running loss
                for evaluate_fn in evaluate_fns.values():
                    evaluate_fn.update(valid_y_pred, valid_current_cn_cnt.to(device))

            for fn_name, evaluate_fn in evaluate_fns.items():
                validation_eval[fn_name] = evaluate_fn.finish().item()

            validation_loss /= len(valid_dataloader)
        print("-" * 80)
        print(
            f"Epoch: {epoch} \n"
            + f"Train loss: {round(training_loss, 4)}; "
            + f"Valid loss: {round(validation_loss, 4)}; "
            + "; ".join(
                [
                    f"Valid {fn_name}: {round(evaluate_value, 4)}"
                    for fn_name, evaluate_value in validation_eval.items()
                ]
            )
        )
        if early_stopping is not None:
            early_stopping(
                val_loss=validation_loss, model_state_dict=nn_model.state_dict()
            )
            if early_stopping.early_stop:
                print(
                    "\n"
                    + "*" * 4
                    + " Due to the early stopping mechanism, the training process is halted "
                    + "*" * 4
                )
                break
    if early_stopping is not None:
        nn_model.load_state_dict(torch.load(early_stopping.best_model_state))
    return nn_model


def train_L5_model_dev2(
    nn_model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    loss_fn: Callable,
    evaluate_fns: Mapping[str, Accuracy | Recall | Precision | RSquare] | None,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping | None = None,
    epochs: int = 100,
    seed: int | None = 1122,
) -> torch.nn.Module:
    device = get_device()
    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for epoch in range(epochs):
        # training
        nn_model.train()
        training_loss = 0.0

        for (
            train_lagged_sn_cnt,
            train_additional_info,
            train_lagged_eqp_status_block,
            train_current_sn_cnt,
        ) in train_dataloader:
            # Forward propagation
            train_y_pred = nn_model(
                train_lagged_sn_cnt.to(device),
                train_additional_info.to(device),
                train_lagged_eqp_status_block.to(device),
            ).squeeze(
                dim=1
            )  # Make prediction by passing X to our model
            # if train_y_pred.shape != train_current_sn_cnt.shape:
            #     print(
            #         f'the shape of model prediction {tuple(train_y_pred.shape)} and y value {tuple(train_current_sn_cnt.shape)} is different ! '
            #         'this might influence the performance of the model'
            #     )
            #     # check train_y_pred shape and train_current_sn_cnt shape
            loss = loss_fn(
                train_y_pred, train_current_sn_cnt.to(device)
            )  # Calculate loss
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
        validation_eval = (
            None
            if evaluate_fns is None
            else {fn_name: 0.0 for fn_name in evaluate_fns.keys()}
        )

        with torch.no_grad():
            for (
                valid_lagged_cn_cnt,
                valid_additional_info,
                valid_lagged_eqp_status_block,
                valid_current_cn_cnt,
            ) in valid_dataloader:
                valid_y_pred = nn_model(
                    valid_lagged_cn_cnt.to(device),
                    valid_additional_info.to(device),
                    valid_lagged_eqp_status_block.to(device),
                ).squeeze(
                    dim=1
                )  # Make prediction by passing X to our model
                loss = loss_fn(
                    valid_y_pred, valid_current_cn_cnt.to(device)
                )  # Calculate loss
                validation_loss += loss.item()  # Add loss to running loss
                if evaluate_fns is not None:
                    for evaluate_fn in evaluate_fns.values():
                        evaluate_fn.update(
                            valid_y_pred[:, 1], valid_current_cn_cnt.to(device)
                        )
            if evaluate_fns is not None:
                for fn_name, evaluate_fn in evaluate_fns.items():
                    validation_eval[fn_name] = evaluate_fn.finish().item()
            validation_loss /= len(valid_dataloader)
        print("-" * 80)
        print(
            f"Epoch: {epoch} \n"
            + f"Train loss: {round(training_loss, 4)}; "
            + f"Valid loss: {round(validation_loss, 4)}; "
            + ""
            if evaluate_fns is None
            else "; ".join(
                [
                    f"Valid {fn_name}: {round(evaluate_value, 4)}"
                    for fn_name, evaluate_value in validation_eval.items()
                ]
            )
        )
        if early_stopping is not None:
            early_stopping(
                val_loss=validation_loss, model_state_dict=nn_model.state_dict()
            )
            if early_stopping.early_stop:
                print(
                    "\n"
                    + "*" * 4
                    + " Due to the early stopping mechanism, the training process is halted "
                    + "*" * 4
                )
                break
    if early_stopping is not None:
        nn_model.load_state_dict(torch.load(early_stopping.best_model_state))
    return nn_model


def train_SoH_model(
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
    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for epoch in range(epochs):
        # training
        nn_model.train()
        training_loss = 0.0

        for train_x, train_y in train_dataloader:
            # Forward propagation
            train_y_pred = nn_model(train_x).squeeze(
                dim=1
            )  # Make prediction by passing X to our model
            if train_y_pred.shape != train_y.shape:
                print(
                    "the shape of model prediction and y value is different ! "
                    "this might influence the performance of the model"
                )
                # check train_y_pred shape and train_y shape
            # loss = loss_fn(train_y_pred, train_y, loss_weight)  # Calculate loss
            loss = loss_fn(train_y_pred, train_y)  # Calculate loss
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
                valid_y_pred = nn_model(valid_x).squeeze(
                    dim=1
                )  # Make prediction by passing X to our model
                # loss = loss_fn(valid_y_pred, valid_y, loss_weight)  # Calculate loss
                loss = loss_fn(valid_y_pred, valid_y)  # Calculate loss
                validation_loss += loss.item()  # Add loss to running loss
                for evaluate_fn in evaluate_fns.values():
                    evaluate_fn.update(valid_y_pred, valid_y)

            for fn_name, evaluate_fn in evaluate_fns.items():
                validation_eval[fn_name] = evaluate_fn.finish().item()

            validation_loss /= len(valid_dataloader)
        print("-" * 80)
        print(
            f"Epoch: {epoch} \n"
            + f"Train loss: {round(training_loss, 4)}; "
            + f"Valid loss: {round(validation_loss, 4)}; "
            + "; ".join(
                [
                    f"Valid {fn_name}: {round(evaluate_value, 4)}"
                    for fn_name, evaluate_value in validation_eval.items()
                ]
            )
        )

        early_stopping(val_loss=validation_loss, model_state_dict=nn_model.state_dict())
        if early_stopping.early_stop:
            print(
                "\n"
                + "*" * 4
                + " Due to the early stopping mechanism, the training process is halted "
                + "*" * 4
            )
            break
    nn_model.load_state_dict(torch.load(early_stopping.best_model_state))
    return nn_model


def train_L3_model(
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
        text="currently using device: {device}".format(device=device),
        artifact_file="log_file.txt",
    )
    nn_model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
                    text="the shape of model prediction and y value is different ! "
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
        # print('-' * 80)
        # print(
        #     f'Epoch: {epoch} \n' +
        #     f'Train loss: {round(training_loss, 4)}; ' +
        #     f'Valid loss: {round(validation_loss, 4)}; ' +
        #     '; '.join(
        #         [f'Valid {fn_name}: {round(evaluate_value, 4)}' for fn_name, evaluate_value in validation_eval.items()])
        # )
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
