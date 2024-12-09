from abc import abstractmethod

import torch
from neuralop.models import FNO1d
from torch import nn


class NeuralPredictor(torch.nn.Module):
    def __init__(self, n_input: int, n_state: int, seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_input = n_input
        self.n_state = n_state
        self.seq_len = seq_len

        self.n_input_channel = n_input + n_state
        self.n_output_channel = n_state
        self.mse_loss = torch.nn.MSELoss()

    def compute(self, x: torch.Tensor):
        raise NotImplementedError()

    def forward(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor, label: torch.Tensor = None, **kwargs):
        repeated_z = z.tile(u.shape[1]).reshape(z.shape[0], -1, z.shape[1])
        x = torch.concatenate([u, repeated_z], -1)
        outs = self.compute(x)
        out = outs[:, -1, :]
        if label is not None:
            return out, self.mse_loss(outs, label)
        else:
            return out

    @abstractmethod
    def name(self):
        raise NotImplementedError()


class GRUNet(NeuralPredictor):
    def __init__(self, hidden_size, num_layers, **kwargs):
        super(GRUNet, self).__init__(**kwargs)
        self.rnn = nn.GRU(self.n_input_channel, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, self.n_output_channel)

    def compute(self, x: torch.Tensor):
        output, (_) = self.rnn(x)
        x = self.projection(output)
        return x

    def name(self):
        return 'GRU'


class LSTMNet(NeuralPredictor):
    def __init__(self, hidden_size, num_layers, **kwargs):
        super(LSTMNet, self).__init__(**kwargs)
        self.rnn = nn.LSTM(self.n_input_channel, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, self.n_output_channel)

    def compute(self, x: torch.Tensor):
        output, (_) = self.rnn(x)
        x = self.projection(output)
        return x

    def name(self):
        return 'LSTM'


class FNOProjection(NeuralPredictor):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_layers: int, n_input_channel=None,
                 n_output_channel=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_modes_height = n_modes_height
        if n_input_channel is not None:
            self.n_input_channel = n_input_channel
        if n_output_channel is not None:
            self.n_output_channel = n_output_channel
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=self.n_input_channel, out_channels=self.n_output_channel)

    def compute(self, x: torch.Tensor):
        x = self.fno(x.transpose(1, 2))
        return x.transpose(1, 2)

    def name(self):
        return 'FNO'


class DeepONet(NeuralPredictor):
    def __init__(self, hidden_size, n_layer, n_input_channel=None, n_output_channel=None, *args, **kwargs):
        super(DeepONet, self).__init__(*args, **kwargs)
        if n_input_channel is not None:
            self.n_input_channel = n_input_channel
        if n_output_channel is not None:
            self.n_output_channel = n_output_channel
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.branch_net = BranchNet(n_input_channel=self.n_input_channel, seq_len=self.seq_len,
                                    hidden_size=hidden_size, n_layer=n_layer)
        self.trunk_net = TrunkNet(y_dim=1, hidden_size=hidden_size, n_layer=n_layer)
        self.output_layer = nn.Linear(self.seq_len * self.hidden_size, self.seq_len * self.n_output_channel)

    def compute(self, x: torch.Tensor):
        """
        x: (batch_size, seq_len, n_input_channel)
        y: (seq_len, y_dim)
        """
        y = torch.linspace(0, 1, steps=self.seq_len, device=x.device).unsqueeze(1)
        branch_out = self.branch_net(x)  # (batch_size, branch_size)
        trunk_out = self.trunk_net(y)  # (seq_len, trunk_size)

        # branch_size == trunk_size == hidden_size
        branch_out = branch_out.unsqueeze(1)  # (batch_size, 1, branch_size)
        trunk_out = trunk_out.unsqueeze(0)  # (1, seq_len, trunk_size)

        combined = branch_out * trunk_out  # (batch_size, seq_len, branch_size)

        combined = combined.view(combined.size(0), -1)  # (batch_size, seq_len * branch_size)

        output = self.output_layer(combined)  # (batch_size, n_output_channel * seq_len)

        output = output.view(x.shape[0], self.n_output_channel, self.seq_len)  # (batch_size, n_output_channel, seq_len)

        return output.transpose(1, 2)

    def name(self):
        return 'DeepONet'


class BranchNet(nn.Module):
    def __init__(self, n_input_channel, seq_len, hidden_size, n_layer):
        super(BranchNet, self).__init__()
        self.flatten = nn.Flatten()

        layers = [
            nn.Linear(n_input_channel * seq_len, hidden_size)
        ]

        for _ in range(n_layer):
            layers += [
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_input_channel)
        x = self.flatten(x)  # (batch_size, n_input_channel * seq_len)
        out = self.fc(x)  # (batch_size, hidden_size)
        return out


class TrunkNet(nn.Module):
    def __init__(self, y_dim, hidden_size, n_layer):
        super(TrunkNet, self).__init__()
        layers = [nn.Linear(y_dim, hidden_size)]

        for _ in range(n_layer):
            layers += [
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, y):
        # y shape: (seq_len, y_dim)
        out = self.fc(y)  # (seq_len, hidden_size)
        return out


class TimeAwareNeuralOperator(NeuralPredictor):
    def __init__(self, ffn: str, rnn: str, params, **kwargs):
        super().__init__(**kwargs)
        self.ffn_name = ffn
        self.rnn_name = rnn

        if ffn == 'FNO':
            self.ffn = FNOProjection(n_modes_height=params['fno_n_modes_height'], n_layers=params['fno_n_layers'],
                                     hidden_channels=params['fno_hidden_channels'], **kwargs)
        elif ffn == 'DeepONet':
            self.ffn = DeepONet(hidden_size=params['deeponet_hidden_size'], n_layer=params['deeponet_n_layer'],
                                **kwargs)
        else:
            raise NotImplementedError()

        if rnn == 'GRU':
            self.rnn = nn.GRU(self.n_output_channel, params['gru_hidden_size'], params['gru_n_layers'],
                              batch_first=True)
            self.projection = nn.Linear(params['gru_hidden_size'], self.n_output_channel)
        elif rnn == 'LSTM':
            self.rnn = nn.LSTM(self.n_output_channel, params['lstm_hidden_size'], params['lstm_n_layers'],
                               batch_first=True)
            self.projection = nn.Linear(params['lstm_hidden_size'], self.n_output_channel)
        else:
            raise NotImplementedError()

    def compute(self, x: torch.Tensor):
        ffn_out = self.ffn.compute(x)
        output, _ = self.rnn(ffn_out)
        rnn_out = self.projection(output)
        return rnn_out + ffn_out

    def name(self):
        return self.ffn_name + '+' + self.rnn_name


if __name__ == '__main__':
    ...
