import torch


class AutoEncoder(torch.nn.Module):
    def __init__(self, in_features: int, compress_size: int):
        super(AutoEncoder, self).__init__()
        self.in_features = in_features
        self.compress_size = compress_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=512),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=128, out_features=self.compress_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.compress_size, out_features=128),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=128, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=256, out_features=512),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(in_features=512, out_features=self.in_features),
        )

    def encoding(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_tensor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(input_tensor))


class UnsupervisedDeepEmbeddingClustering(torch.nn.Module):
    def __init__(self, compress_size: int, n_clusters: int, alpha: float = 1.0):
        super(UnsupervisedDeepEmbeddingClustering, self).__init__()
        self.compress_size = compress_size
        self.n_clusters = n_clusters
        self.alpha = alpha

        self.cluster_centers = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.zeros(
                    self.n_clusters,
                    self.compress_size,
                    dtype=torch.float32,
                    requires_grad=True,
                ).cuda()
            )
        )

    def forward(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        sub_component = (
            1
            + torch.pow(input=encoded_tensor - self.cluster_centers, exponent=2)
            / self.alpha
        )
        t_distance = torch.pow(
            input=sub_component, exponent=-0.5 * (self.alpha + 1)
        ) / torch.pow(input=sub_component, exponent=-0.5 * (self.alpha + 1)).sum(
            dim=-1, keepdim=True
        )
        return t_distance
