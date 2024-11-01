import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TimeFrequencyConsistency(torch.nn.Module):
    def __init__(self) -> None:
        super(TimeFrequencyConsistency, self).__init__()

        # time domain encoder
        encoder_layers_t = TransformerEncoderLayer(
            d_model=5120,
            nhead=2,
            dim_feedforward=2 * 5120,
        )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = torch.nn.Sequential(
            torch.nn.Linear(5120, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )

        # frequency domain encoder
        encoder_layers_f = TransformerEncoderLayer(
            d_model=5120,
            nhead=2,
            dim_feedforward=2 * 5120,
        )
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = torch.nn.Sequential(
            torch.nn.Linear(5120, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )

    def forward(
        self, x_in_t: torch.Tensor, x_in_f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


class target_classifier(torch.nn.Module):
    def __init__(self, num_classes_target: int) -> None:
        super(target_classifier, self).__init__()
        self.logits = torch.nn.Linear(2 * 128, 64)
        self.logits_simple = torch.nn.Linear(64, num_classes_target)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
