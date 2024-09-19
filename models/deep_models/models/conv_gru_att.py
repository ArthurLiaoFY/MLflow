import torch

# from models.deep_models.utils.prepare_data import get_device


class ConvolutionalGRUAttention(torch.nn.Module):
    def __init__(
        self,
        conv_in_channels: int,
        conv_out_channels: int,
        gru_input_size: int,
        gru_hidden_size: int,
        gru_layer_amount: int = 2,
        attention_num_of_head: int = 8,
        out_feature_size: int = 2,
    ):
        super(ConvolutionalGRUAttention, self).__init__()
        # self.device = get_device()
        self.gru_hidden_size = gru_hidden_size
        self.gru_layer_amount = gru_layer_amount
        self.conv_1d_block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv_in_channels,
                out_channels=conv_out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm1d(num_features=conv_out_channels),
            torch.nn.LeakyReLU(),
        )

        # conv_1d_layer in size : (batch size, in_channels, 1d input size)
        # conv_1d_layer out size : (batch size, out_channels, 1d input size)

        self.gru_layer = torch.nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_layer_amount,
            bias=True,
            batch_first=True,
            dropout=0.4,
            bidirectional=False,
        )

        # gru_layer in size : (batch size, sequence len, gru_input_size(1d input size))
        # gru_layer out size : (batch size, sequence len, gru_hidden_size)

        self.q_conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv_out_channels,
                out_channels=conv_out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.Dropout(0.4),
        )
        self.k_conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv_out_channels,
                out_channels=conv_out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.Dropout(0.4),
        )
        self.v_conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=conv_out_channels,
                out_channels=conv_out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.Dropout(0.4),
        )
        # q, k, v in size : (batch size, conv_out_channels, gru_hidden_size)
        # q, k, v out size : (batch size, conv_out_channels, gru_hidden_size)

        # if embedding into higher dimension, will the result be better
        self.multihead_attention_block = torch.nn.MultiheadAttention(
            embed_dim=gru_hidden_size,
            num_heads=attention_num_of_head,
            batch_first=True,
        )
        # multihead_attention_block in size : (batch size, conv_out_channels, gru_hidden_size)
        # multihead_attention_block out size : (batch size, conv_out_channels, gru_hidden_size)
        self.flatten = torch.nn.Flatten()
        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=conv_out_channels * gru_hidden_size,
                out_features=64,
            ),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=32, out_features=out_feature_size),
            torch.nn.Sigmoid(),
        )
        self.flatten = torch.nn.Flatten()

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.size(0)
        conv_result = self.conv_1d_block(input_tensor)
        init_hidden_state = torch.zeros(
            size=(self.gru_layer_amount, batch_size, self.gru_hidden_size)
        ).to(input_tensor.device)
        gru_result, _ = self.gru_layer(conv_result, init_hidden_state)
        q, k, v = (
            self.q_conv_layer(gru_result),
            self.k_conv_layer(gru_result),
            self.v_conv_layer(gru_result),
        )
        self.multihead_attention_block.to(input_tensor.device)
        attn_output, _ = self.multihead_attention_block(q, k, v)
        attn_output = self.flatten(attn_output + gru_result)

        return self.final_layer(attn_output)
