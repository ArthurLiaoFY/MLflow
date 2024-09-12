import torch

from ML_MODELS.DeepModels.norms.IN import InstanceNorm


class PatchMixerModel(torch.nn.Module):
    def __init__(
        self,
        train_seq_len: int,
        test_seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
    ):
        super(PatchMixerModel, self).__init__()
        self.train_seq_len = train_seq_len
        self.test_seq_len = test_seq_len

        self.instance_norm = InstanceNorm(num_features=train_seq_len)

        self.patch_len = patch_len
        self.d_model = self.patch_len**2
        self.stride = stride
        self.padding_layer = torch.nn.ReplicationPad1d((0, self.stride))
        self.patch_num = (
            int((self.train_seq_len - self.patch_len) / self.stride + 1) + 1
        )

        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.patch_len, out_features=self.d_model),
            torch.nn.Dropout(0.3),
        )

        self.linear_flatten_head = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.patch_num * self.d_model,
                out_features=self.test_seq_len,
            ),
            torch.nn.Dropout(0.3),
        )

        self.residual_conn_layer = torch.nn.Linear(self.d_model, self.patch_len)
        self.depthwise_conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.patch_num,
                self.patch_num,
                kernel_size=self.patch_len,
                stride=self.patch_len,
                groups=self.patch_num,
            ),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(self.patch_num),
        )

        self.pointwise_conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.patch_num,
                self.patch_num,
                kernel_size=1,
                stride=1,
                groups=self.patch_num,
            ),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(self.patch_num),
        )
        self.mlp_flatten_head = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.patch_num * self.patch_len, out_features=256
            ),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=256, out_features=self.test_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence_len, feature_amount = x.shape

        # instance norm
        z = self.instance_norm(x, mode="norm")
        # padding
        z = self.padding_layer(z.permute(0, 2, 1))
        # patching
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # channel independence
        z = z.reshape(
            batch * feature_amount, self.patch_num, self.patch_len, 1
        ).squeeze(-1)
        # embedding
        z = self.embedding(z)

        # linear head (Dual forcasting head)
        z_res = self.linear_flatten_head(z.reshape(batch, feature_amount, -1))

        # 3.1
        z_depth = self.depthwise_conv_block(z) + self.residual_conn_layer(z)
        # 3.2
        z_point = self.pointwise_conv_block(z_depth)
        z_point = z_point.reshape(batch, feature_amount, -1)

        # 4
        z_mlp = self.mlp_flatten_head(z_point)
        d_z = self.instance_norm((z_res + z_mlp).permute(0, 2, 1), mode="denorm")
        return d_z


pm = PatchMixerModel(train_seq_len=336, test_seq_len=60)
a = pm(torch.randn(64, 336, 6))
print(a)
print(a.shape)
