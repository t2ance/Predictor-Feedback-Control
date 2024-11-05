import torch
from neuralop.models import FNO1d


class FNOProjection(torch.nn.Module):
    def __init__(self, n_input: int, n_state: int, seq_len: int, n_modes_height: int,
                 hidden_channels: int, n_layers: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_input = n_input
        self.n_state = n_state
        self.seq_len = seq_len

        self.n_input_channel = n_input + n_state
        self.n_output_channel = n_state
        self.mse_loss = torch.nn.MSELoss()
        self.n_modes_height = n_modes_height

        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=self.n_input_channel, out_channels=self.n_output_channel)

    def forward(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor, label: torch.Tensor = None, **kwargs):
        repeated_z = z.tile(u.shape[1]).reshape(z.shape[0], -1, z.shape[1])

        x = torch.concatenate([u, repeated_z], -1)
        x = self.fno(x.transpose(1, 2))
        outs = x.transpose(1, 2)[:, -1, :]
        out = outs
        if label is not None:
            return out, self.mse_loss(outs, label)
        else:
            return out

    def name(self):
        return 'FNO'


if __name__ == '__main__':
    ...
