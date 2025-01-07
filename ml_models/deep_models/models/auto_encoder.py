import torch


class ConvAutoencoder(torch.nn.Module):
    def __init__(self, in_features: tuple = (1, 28, 28)):
        super(ConvAutoencoder, self).__init__()
        self.in_features = in_features
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.in_features[0],
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=self.in_features[0],
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, in_features: int = 2):
        super(VariationalAutoEncoder, self).__init__()
        self.in_features = in_features
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
        )

        self.mu_head = torch.nn.Linear(32, 32)
        self.std_head = torch.nn.Linear(32, 32)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, self.in_features),
            torch.nn.Sigmoid(),
        )

    def sampling(self, sample_mean, sample_std):
        sample_std = torch.exp(sample_std)
        error = torch.randn_like(sample_std)
        return sample_mean + error * sample_std

    def forward(self, x):
        # x shape [batch size, features]
        z = self.encoder(x.reshape(x.shape[0], -1))
        sample_mean = self.mu_head(z)
        sample_std = self.std_head(z)

        z = self.sampling(sample_mean=sample_mean, sample_std=sample_std)
        return self.decoder(z), sample_mean, sample_std


class VariationalConvAutoEncoder(torch.nn.Module):
    def __init__(self, in_features: tuple = (1, 28, 28)):
        super(VariationalConvAutoEncoder, self).__init__()
        self.in_features = in_features
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.in_features[0],
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.Sigmoid(),
        )

        self.mu_head = torch.nn.Conv2d(
            in_channels=self.in_features[0],
            out_channels=self.in_features[0],
            kernel_size=3,
            padding=1,
        )
        self.std_head = torch.nn.Conv2d(
            in_channels=self.in_features[0],
            out_channels=self.in_features[0],
            kernel_size=3,
            padding=1,
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=self.in_features[0],
                kernel_size=3,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
        )

    def sampling(self, sample_mean, sample_std):
        sample_std = torch.exp(sample_std)
        error = torch.randn_like(sample_std)
        return sample_mean + error * sample_std

    def forward(self, x):
        # x shape [batch size, features]
        encoded = self.encoder(x)
        sample_mean = self.mu_head(encoded)
        sample_std = self.std_head(encoded)

        z = self.sampling(
            sample_mean=sample_mean,
            sample_std=sample_std,
        )
        return self.decoder(z)
        # sample_mean,
        # sample_std,
