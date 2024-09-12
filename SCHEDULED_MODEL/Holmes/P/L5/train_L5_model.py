import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

from ML_MODELS.DeepModels.models.Conv2dAtt import Convolutional2DAttention
from ML_MODELS.DeepModels.training.early_stopping import EarlyStopping
from ML_MODELS.DeepModels.training.evaluate import RSquare
from ML_MODELS.DeepModels.training.loss import root_mean_square_error
from ML_MODELS.DeepModels.training.train_model import train_L5_model_dev
from ML_MODELS.DeepModels.utils.prepare_data import to_tensor

cudnn.benchmark = True
TIME_ZONE = "Asia/Taipei"


def raw_dict_to_df(raw_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(
        index=pd.DatetimeIndex(data=raw_dict["index"], tz=TIME_ZONE),
        columns=pd.MultiIndex.from_tuples(
            tuples=raw_dict["columns"], names=raw_dict["columns_names"]
        ),
        data=raw_dict["data"],
    )


class SequenceEquipmentStatusBlock(Dataset):
    def __init__(
        self,
        lagged_sn_cnt: np.ndarray,
        additional_info: np.ndarray,
        lagged_eqp_status_block: np.ndarray,
        current_sn_cnt: np.ndarray,
        sequence_length: int,
    ):
        self.lagged_sn_cnt = lagged_sn_cnt
        self.additional_info = additional_info
        self.lagged_eqp_status_block = lagged_eqp_status_block
        self.current_sn_cnt = current_sn_cnt
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.additional_info) - self.sequence_length

    def __getitem__(self, idx):
        return (
            to_tensor(
                self.lagged_sn_cnt[idx : idx + self.sequence_length][:, np.newaxis]
            ),
            to_tensor(self.additional_info[idx + self.sequence_length, :]),
            to_tensor(
                self.lagged_eqp_status_block[idx : idx + self.sequence_length, :, :]
            ),
            to_tensor(self.current_sn_cnt[idx + self.sequence_length]),
        )


class UphTrainPredictModel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.model_name = None
        self.mo_number = None
        self.shift = 60
        self.sequence_length = 60
        self.have_plan = []
        self.sn_trunk_name = "UPH"
        self.lagged_sn_trunk_name = "lagged_" + self.sn_trunk_name
        self.xgb_params = {"seed": self.seed}

    def __get_lagged_df(
        self, agg_status_dict: dict, prod_capacity_dict: dict, sn_dict: dict
    ) -> tuple[pd.Series, pd.DataFrame, np.ndarray, pd.Series]:
        sn_df = raw_dict_to_df(sn_dict)
        agg_status_df = raw_dict_to_df(agg_status_dict)

        melt_agg_status_df = agg_status_df.copy().shift(self.shift).dropna()
        melt_agg_status_df = melt_agg_status_df.stack(
            level=1, future_stack=True
        ).fillna(0)

        try:
            prod_capacity_df = pd.Series(
                index=pd.DatetimeIndex(data=prod_capacity_dict["index"], tz=TIME_ZONE),
                name=prod_capacity_dict["columns"],
                data=prod_capacity_dict["data"],
            )
            lagged_df = pd.concat(
                (
                    sn_df.sum(axis=1).rename(self.sn_trunk_name),
                    sn_df.sum(axis=1)
                    .shift(self.shift)
                    .rename(self.lagged_sn_trunk_name),
                    prod_capacity_df.rename("plan_QTY"),
                    pd.Series(index=sn_df.index, name="hour", data=sn_df.index.hour),
                ),
                axis=1,
            ).dropna()

            lagged_eqp_status_block = np.array(
                [melt_agg_status_df.loc[(str(dt),), :].T for dt in lagged_df.index],
                dtype=float,
            )

        except Exception:
            raise
        return (
            lagged_df.get(self.lagged_sn_trunk_name),
            lagged_df.drop(columns=[self.sn_trunk_name]),
            lagged_eqp_status_block,
            lagged_df.get(self.sn_trunk_name),
        )

    def __get_train_valid_dataloader(
        self,
        lagged_sn_cnt: pd.Series,
        additional_info: pd.DataFrame,
        lagged_eqp_status_block: np.ndarray,
        current_sn_cnt: pd.Series,
    ) -> tuple[DataLoader, DataLoader]:
        split_idx = -3 * 1440
        (
            train_lagged_sn_cnt,
            valid_lagged_sn_cnt,
            train_additional_info,
            valid_additional_info,
            train_eqp_status_block,
            valid_eqp_status_block,
            train_current_sn_cnt,
            valid_current_sn_cnt,
        ) = (
            lagged_sn_cnt[:split_idx].to_numpy(),
            lagged_sn_cnt[split_idx - self.sequence_length :].to_numpy(),
            additional_info.iloc[self.sequence_length : split_idx, :].to_numpy(),
            additional_info.iloc[split_idx:, :].to_numpy(),
            lagged_eqp_status_block[:split_idx, :, :],
            lagged_eqp_status_block[split_idx - self.sequence_length :, :, :],
            current_sn_cnt.iloc[self.sequence_length : split_idx].to_numpy(),
            current_sn_cnt.iloc[split_idx:].to_numpy(),
        )

        train_dataset = SequenceEquipmentStatusBlock(
            lagged_sn_cnt=train_lagged_sn_cnt,
            additional_info=train_additional_info,
            lagged_eqp_status_block=train_eqp_status_block,
            current_sn_cnt=train_current_sn_cnt,
            sequence_length=self.sequence_length,
        )
        valid_dataset = SequenceEquipmentStatusBlock(
            lagged_sn_cnt=valid_lagged_sn_cnt,
            additional_info=valid_additional_info,
            lagged_eqp_status_block=valid_eqp_status_block,
            current_sn_cnt=valid_current_sn_cnt,
            sequence_length=self.sequence_length,
        )

        return (
            DataLoader(dataset=train_dataset, batch_size=64, shuffle=True),
            DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False),
        )

    def __nn_fit(
        self,
        num_of_eqp: int,
        num_of_status: int,
        additional_infos: int,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> torch.nn.Module:
        model = Convolutional2DAttention(
            num_of_eqp=num_of_eqp,
            num_of_status=num_of_status,
            additional_infos=additional_infos,
            cluster_num=16,
        )
        tuned_model = train_L5_model_dev(
            nn_model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            loss_fn=root_mean_square_error,
            evaluate_fns={"R-square": RSquare()},
            optimizer=torch.optim.SGD(model.parameters(), lr=self.L5_learning_rate),
            early_stopping=EarlyStopping(patience=self.L5_early_stopping_patience),
            epochs=self.L5_epoch,
        )

        return tuned_model

    def train_model(
        self,
        agg_status_dict: dict,
        prod_capacity_dict: dict,
        sn_dict: dict,
    ) -> None:
        lagged_sn_cnt, additional_info, lagged_eqp_status_block, current_sn_cnt = (
            self.__get_lagged_df(
                agg_status_dict=agg_status_dict.copy(),
                prod_capacity_dict=prod_capacity_dict.copy(),
                sn_dict=sn_dict.copy(),
            )
        )

        _, additional_infos = additional_info.shape
        _, num_of_eqp, num_of_status = lagged_eqp_status_block.shape
        train_dataloader, valid_dataloader = self.__get_train_valid_dataloader(
            lagged_sn_cnt=lagged_sn_cnt,
            additional_info=additional_info,
            lagged_eqp_status_block=lagged_eqp_status_block,
            current_sn_cnt=current_sn_cnt,
        )
        model = self.__nn_fit(
            num_of_eqp=num_of_eqp,
            num_of_status=num_of_status,
            additional_infos=additional_infos,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
        )
        suffix = f"_{self.db_address}_{self.db_name}_{self.owner_id}_{self.line_id}_{self.sector_id}"
        model_file_name = f"{self.model_file_path}/L5_model_{suffix}.pt"
        historical_eqp_status_block_file_name = f"{self.model_file_path}/{self.run_id}_historical_eqp_status_block_{suffix}.npy"
        future_eqp_status_block_file_name = (
            f"{self.model_file_path}/{self.run_id}_future_eqp_status_block_{suffix}.npy"
        )

        torch.jit.script(model).save(model_file_name)
        np.save(
            file=historical_eqp_status_block_file_name,
            arr=lagged_eqp_status_block[: -self.shift, :, :],
        )
        np.save(
            file=future_eqp_status_block_file_name,
            arr=lagged_eqp_status_block[self.sequence_length + self.shift :, :, :],
        )

        print(f"@*@{model_file_name}@*@")
        print(f"@*@{historical_eqp_status_block_file_name}@*@")
        print(f"@*@{future_eqp_status_block_file_name}@*@")

        return None
