import torch

# from ml_models.deep_models.utils.prepare_data import get_device


class ConvolutionalAttention(torch.nn.Module):
    def __init__(
        self,
        num_of_eqp: int,
        num_of_status: int,
        in_channel_size: int,
        out_channel_size: int,
        out_feature_size: int = 1,
    ):
        super(ConvolutionalAttention, self).__init__()

        self.eqp_conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channel_size,
                out_channels=out_channel_size,
                kernel_size=3,
                padding="same",
            ),
            # out shape : (batch_size, out_channels, width, height)
            torch.nn.BatchNorm2d(num_features=out_channel_size),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(),
        )

        # self.eqp_conv_block = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels=in_channel_size, out_channels=in_channel_size, kernel_size=(num_of_eqp, 3), groups=in_channel_size, padding='same'),
        #     torch.nn.Conv2d(in_channels=in_channel_size, out_channels=out_channel_size, kernel_size=1, padding='same'),
        #     # out shape : (batch_size, out_channels, width, height)
        #     torch.nn.Dropout(0.4),
        #     torch.nn.BatchNorm2d(num_features=out_channel_size),
        #     torch.nn.LeakyReLU(),
        # )

        self.q_conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=num_of_eqp,
                out_channels=num_of_eqp,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm1d(num_features=num_of_eqp),
            torch.nn.Dropout(0.4),
        )
        self.k_conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=num_of_eqp,
                out_channels=num_of_eqp,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm1d(num_features=num_of_eqp),
            torch.nn.Dropout(0.4),
        )
        self.v_conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=num_of_eqp,
                out_channels=num_of_eqp,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm1d(num_features=num_of_eqp),
            torch.nn.Dropout(0.4),
        )
        self.multihead_attention_block = torch.nn.MultiheadAttention(
            embed_dim=out_channel_size * num_of_status,
            num_heads=out_channel_size,
            batch_first=True,
        )
        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=3 + num_of_eqp * num_of_status * out_channel_size,
                out_features=256,
            ),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=64),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=64, out_features=out_feature_size),
            torch.nn.ReLU(),
        )
        self.flatten = torch.nn.Flatten()

    def forward(self, lagged_infos: torch.Tensor, eqp_status_block: torch.Tensor):
        conv_result = self.eqp_conv_block(eqp_status_block)
        flatten_conv_result = torch.cat(
            tensors=[conv_result[:, idx, :, :] for idx in range(conv_result.size(1))],
            dim=-1,
        )
        q, k, v = (
            self.q_conv_layer(flatten_conv_result),
            self.k_conv_layer(flatten_conv_result),
            self.v_conv_layer(flatten_conv_result),
        )
        self.multihead_attention_block.to(eqp_status_block.device)
        attn_output, _ = self.multihead_attention_block(q, k, v)
        attn_output = torch.cat(
            tensors=(self.flatten(attn_output + flatten_conv_result), lagged_infos),
            dim=-1,
        )

        return self.final_layer(attn_output)
