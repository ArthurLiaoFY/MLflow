from math import floor

import torch
from torch.nn.utils import spectral_norm

from ml_models.deep_models.utils.prepare_data import get_device


class Convolutional2DAttention(torch.nn.Module):
    def __init__(
        self,
        num_of_eqp: int,
        num_of_status: int,
        additional_infos: int,
        cluster_num: int,
        in_channel_size: int = 60,
        out_channel_size: int = 60,
        max_pool_k_size: int = 2,
        out_feature_size: int = 1,
    ):
        super(Convolutional2DAttention, self).__init__()
        self.num_of_eqp = num_of_eqp
        self.num_of_status = num_of_status
        self.additional_infos = additional_infos
        self.cluster_num = cluster_num
        self.in_channel_size = in_channel_size
        self.out_channel_size = out_channel_size
        self.max_pool_k_size = max_pool_k_size
        self.attention_embed_dim = num_of_eqp * num_of_status
        self.out_feature_size = out_feature_size

        self.eqp_conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.in_channel_size,
                out_channels=self.in_channel_size,
                kernel_size=(num_of_eqp, 3),
                groups=self.in_channel_size,
                padding="same",
            ),
            torch.nn.Conv2d(
                in_channels=self.in_channel_size,
                out_channels=self.out_channel_size,
                kernel_size=1,
                padding="same",
            ),
            # out shape : (batch_size, out_channels, width, height)
            torch.nn.BatchNorm2d(num_features=out_channel_size),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(),
        )

        self.q_conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.out_channel_size,
                out_channels=self.out_channel_size,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm1d(num_features=self.out_channel_size),
            torch.nn.Dropout(0.4),
        )
        self.k_conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.out_channel_size,
                out_channels=self.out_channel_size,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm1d(num_features=self.out_channel_size),
            torch.nn.Dropout(0.4),
        )
        self.v_conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.out_channel_size,
                out_channels=self.out_channel_size,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm1d(num_features=self.out_channel_size),
            torch.nn.Dropout(0.4),
        )
        self.multihead_attention_block = torch.nn.MultiheadAttention(
            embed_dim=self.attention_embed_dim, num_heads=num_of_eqp, batch_first=True
        )
        self.linear = torch.nn.Linear(
            in_features=self.attention_embed_dim, out_features=self.cluster_num
        )

        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.additional_infos
                + self.out_channel_size * self.cluster_num,
                out_features=512,
            ),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=out_feature_size),
            torch.nn.ReLU(),
        )
        # self.linear = torch.nn.Linear(in_features=self.out_channel_size, out_features=self.cluster_num)

        # self.final_layer = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=self.additional_infos+self.attention_embed_dim * self.cluster_num, out_features=512),
        #     torch.nn.Dropout(0.3),
        #     torch.nn.LeakyReLU(negative_slope=0.01),
        #     torch.nn.Linear(in_features=512, out_features=256),
        #     torch.nn.Dropout(0.3),
        #     torch.nn.LeakyReLU(negative_slope=0.01),
        #     torch.nn.Linear(in_features=256, out_features=out_feature_size),
        #     torch.nn.ReLU(),
        # )

        self.flatten = torch.nn.Flatten()

    def forward(
        self,
        lagged_sn_count: torch.Tensor,
        lagged_infos: torch.Tensor,
        lagged_eqp_status_block: torch.Tensor,
    ):
        conv_result = self.eqp_conv_block(lagged_eqp_status_block)
        flatten_conv_result = torch.cat(
            tensors=[conv_result[:, :, idx, :] for idx in range(conv_result.size(2))],
            dim=-1,
        )

        q, k, v = (
            self.q_conv_layer(flatten_conv_result),
            self.k_conv_layer(flatten_conv_result),
            self.v_conv_layer(flatten_conv_result),
        )
        attn_output, _ = self.multihead_attention_block(q, k, v)
        # self.linear(attn_output+flatten_conv_result)  * cluster index
        # attn_output = torch.cat(tensors=(self.flatten(self.linear(torch.transpose(input=attn_output+flatten_conv_result, dim0=1, dim1=2))), lagged_infos), dim=-1)
        attn_output = torch.cat(
            tensors=(
                self.flatten(self.linear(attn_output + flatten_conv_result)),
                lagged_infos,
            ),
            dim=-1,
        )

        return self.final_layer(attn_output)
