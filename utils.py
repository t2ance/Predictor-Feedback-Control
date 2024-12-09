import os
import random
from dataclasses import dataclass
from typing import List, Any

import numpy as np
from matplotlib import pyplot as plt
from torch.nn import init

from model import *


@dataclass
class SimulationResult:
    Z0: Any = None
    U: np.ndarray = None
    Z: np.ndarray = None
    D_no: np.ndarray = None
    D_numerical: np.ndarray = None
    P_no: np.ndarray = None
    P_numerical: np.ndarray = None
    runtime: float = None
    avg_prediction_time: float = None
    P_numerical_n_iters: np.ndarray = None
    l2: float = None
    rl2: float = None
    success: bool = None
    n_parameter: int = None


@dataclass
class TestResult:
    runtime: float = None
    l2: float = None
    rl2: float = None
    n_success: int = None
    results: List = None


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def print_args(args):
    print('=' * 100)
    print(args.__class__.__name__)
    for k, v in args.__dict__.items():
        print(f'        - {k} : {v}')
    print('=' * 100)


def load_model(train_config, model_config, dataset_config):
    n_state = dataset_config.system.n_state
    n_input = dataset_config.system.n_input
    seq_len = dataset_config.n_point_delay
    model_name = model_config.model_name
    if model_name == 'DeepONet':
        model = DeepONet(hidden_size=model_config.deeponet_hidden_size, n_layer=model_config.deeponet_n_layer,
                         n_input=n_input, n_state=n_state, seq_len=seq_len)
    elif model_name == 'FNO':
        n_modes_height = model_config.fno_n_modes_height
        hidden_channels = model_config.fno_hidden_channels
        model = FNOProjection(n_modes_height=n_modes_height, hidden_channels=hidden_channels,
                              n_layers=model_config.fno_n_layer, n_input=n_input, n_state=n_state, seq_len=seq_len)
    elif model_name == 'GRU':
        model = GRUNet(hidden_size=model_config.gru_hidden_size, num_layers=model_config.gru_n_layer, n_input=n_input,
                       n_state=n_state, seq_len=seq_len)
    elif model_name == 'LSTM':
        model = LSTMNet(hidden_size=model_config.lstm_hidden_size, num_layers=model_config.lstm_n_layer,
                        n_input=n_input, n_state=n_state, seq_len=seq_len)
    elif model_name == 'FNO+GRU':
        model = TimeAwareNeuralOperator(
            ffn='FNO', rnn='GRU', n_input=n_input, n_state=n_state, seq_len=seq_len,
            params={
                'fno_n_modes_height': model_config.fno_n_modes_height,
                'fno_hidden_channels': model_config.fno_hidden_channels,
                'fno_n_layers': model_config.fno_n_layer,
                'gru_n_layers': model_config.gru_n_layer,
                'gru_hidden_size': model_config.gru_hidden_size
            })
    elif model_name == 'DeepONet+GRU':
        model = TimeAwareNeuralOperator(
            ffn='DeepONet', rnn='GRU', n_input=n_input, n_state=n_state, seq_len=seq_len,
            params={
                'deeponet_hidden_size': model_config.deeponet_hidden_size,
                'deeponet_n_layer': model_config.deeponet_n_layer,
                'gru_n_layers': model_config.gru_n_layer,
                'gru_hidden_size': model_config.gru_hidden_size
            })
    else:
        raise NotImplementedError()
    initialize_weights(model)
    print('Number of parameters', count_params(model))
    return model.to(train_config.device)


def load_optimizer(parameters, train_config):
    return torch.optim.AdamW(parameters, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)


def load_lr_scheduler(optimizer: torch.optim.Optimizer, config):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size,
                                           gamma=config.scheduler_gamma)


def l2s(P, Z, n_point_delay):
    P = P[n_point_delay:-n_point_delay]
    Z = Z[2 * n_point_delay:]
    N = Z.shape[0]
    P = np.atleast_2d(P)
    Z = np.atleast_2d(Z)
    l2 = np.sum(np.linalg.norm(P - Z, axis=1)) / N
    rl2 = np.sum(np.linalg.norm(P - Z, axis=1) / np.linalg.norm(Z, axis=1)) / N
    return l2, rl2


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def count_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def pad_zeros(segment, length, leading: bool = True):
    assert len(segment) <= length

    if len(segment) < length:
        padding_length = length - len(segment)
        if isinstance(segment, np.ndarray):
            if segment.ndim == 2:
                padding = np.zeros((padding_length, segment.shape[1]))
            else:
                padding = np.zeros(padding_length)
            if leading:
                segment = np.concatenate((padding, segment))
            else:
                segment = np.concatenate((segment, padding))
        elif isinstance(segment, torch.Tensor):
            if segment.ndim == 2:
                padding = torch.zeros((padding_length, segment.shape[1]))
            else:
                padding = torch.zeros(padding_length)
            if leading:
                segment = torch.concatenate((padding, segment))
            else:
                segment = torch.concatenate((segment, padding))
        else:
            raise NotImplementedError()

    return segment


def set_everything(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        # "font.family": "times",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6
    }

    plt.rcParams.update(tex_fonts)


if __name__ == '__main__':
    ...
